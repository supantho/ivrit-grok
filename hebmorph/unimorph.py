"""UniMorph Hebrew parsing.

Files (unimorph/heb):
  heb_voc         vocalized triples  lemma<TAB>form<TAB>tag   (UniMorph 2/3 tags)
  heb             unvocalized (plene, ktiv male) triples
  heb.args        heb_voc in UniMorph 4 notation, plus '-' rows for cells marked absent
  heb_unvoc.args  heb in UniMorph 4 notation
  from_UD_unvoc   unvocalized forms converted from UD Hebrew (corpus attestation)

Paradigms are not given explicit IDs by UniMorph.  A paradigm = a maximal run
of consecutive verb lines sharing one lemma string.  Every line keeps its file
name and 1-based line number for provenance.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .hebrew import extract_consonantal_skeleton, normalize_nfc, predict_plene, strip_niqqud
from .morphology import TagError, parse_unimorph_tag


@dataclass
class UMLine:
    file: str
    line_no: int
    lemma_raw: str
    form_raw: str
    tag_raw: str

    @property
    def lemma(self):
        return normalize_nfc(self.lemma_raw)

    @property
    def form(self):
        return normalize_nfc(self.form_raw)


@dataclass
class UMParadigm:
    paradigm_key: str          # file:first_line
    file: str
    lemma_raw: str
    lines: list[UMLine] = field(default_factory=list)
    cells: dict[str, UMLine] = field(default_factory=dict)   # canonical cell -> line
    duplicate_cells: list[tuple[str, UMLine]] = field(default_factory=list)
    tag_errors: list[tuple[UMLine, str]] = field(default_factory=list)

    @property
    def lemma(self):
        return normalize_nfc(self.lemma_raw)

    def signature(self) -> tuple:
        return tuple(sorted((c, ln.form) for c, ln in self.cells.items()))


def read_triples(path: str, file_label: str) -> list[UMLine]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{file_label}:{i}: expected 3 tab-separated fields, got {len(parts)}")
            out.append(UMLine(file_label, i, parts[0], parts[1], parts[2]))
    return out


def is_verb_tag(tag: str) -> bool:
    return tag == "V.MSDR" or tag.startswith("V;")


def segment_paradigms(lines: list[UMLine]) -> list[UMParadigm]:
    paradigms: list[UMParadigm] = []
    cur = None
    for ln in lines:
        if not is_verb_tag(ln.tag_raw):
            cur = None
            continue
        if cur is None or ln.lemma_raw != cur.lemma_raw:
            cur = UMParadigm(f"{ln.file}:{ln.line_no}", ln.file, ln.lemma_raw)
            paradigms.append(cur)
        cur.lines.append(ln)
        try:
            feats = parse_unimorph_tag(ln.tag_raw)
        except TagError as e:
            cur.tag_errors.append((ln, str(e)))
            continue
        cell = feats["canonical_cell"]
        if cell in cur.cells:
            cur.duplicate_cells.append((cell, ln))
        else:
            cur.cells[cell] = ln
    return paradigms


def plene_key(form: str) -> str:
    """Key on which a vocalized form and its plene spelling agree."""
    return extract_consonantal_skeleton(strip_niqqud(form), drop_vav_yod=True)


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def pair_voc_unvoc(voc: list[UMParadigm], unvoc: list[UMParadigm], min_agree: float = 0.75,
                   min_margin: float = 0.15):
    """Pair each vocalized paradigm with the unvocalized paradigm describing
    the same lexeme.

    Primary score = fraction of cells whose plene_key (niqqud stripped, all
    ו/י deleted) equals the unvocalized form's key in the same cell.
    Tie-break among equal-score candidates = total edit distance between the
    approximate predicted plene spelling of the vocalized forms
    (hebrew.predict_plene) and the unvocalized forms.  Candidates that
    remain tied are either content-identical (assignment immaterial; lowest
    index chosen and flagged) or reported AMBIGUOUS.

    Status: PAIRED (score 1), PAIRED_PARTIAL (min_agree <= score < 1 with a
    clear margin; the disagreeing cells are flagged downstream),
    PAIRED_TIE_IDENTICAL, AMBIGUOUS, UNPAIRED."""
    index = defaultdict(set)  # (cell, key) -> unvoc paradigm idx
    ukeys = []
    for j, up in enumerate(unvoc):
        k = {c: extract_consonantal_skeleton(ln.form, drop_vav_yod=True) for c, ln in up.cells.items()}
        ukeys.append(k)
        for c, key in k.items():
            index[(c, key)].add(j)
    results = []
    for i, vp in enumerate(voc):
        vk = {c: plene_key(ln.form) for c, ln in vp.cells.items()}
        votes = Counter()
        for c, key in vk.items():
            for j in index.get((c, key), ()):
                votes[j] += 1
        scored = []
        for j, n in votes.most_common(10):
            denom = max(len(vk), len(ukeys[j]))
            scored.append((n / denom, j))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if not scored or scored[0][0] < min_agree:
            results.append(dict(voc_index=i, unvoc_index=scored[0][1] if scored else None,
                                score=scored[0][0] if scored else 0.0,
                                runner_up_score=scored[1][0] if len(scored) > 1 else 0.0,
                                status="UNPAIRED", tied=[], edit_distance=None))
            continue
        top = scored[0][0]
        tied = [j for s, j in scored if abs(s - top) < 1e-9]
        rest = [s for s, j in scored if abs(s - top) >= 1e-9]
        second = rest[0] if rest else 0.0

        def dist(j):
            up = unvoc[j]
            return sum(levenshtein(predict_plene(vp.cells[c].form), up.cells[c].form)
                       for c in vp.cells if c in up.cells)
        dists = {j: dist(j) for j in tied}
        dmin = min(dists.values())
        best_js = sorted(j for j in tied if dists[j] == dmin)
        if len(best_js) == 1:
            status = "PAIRED" if top == 1.0 else "PAIRED_PARTIAL"
            if top < 1.0 and top - second < min_margin:
                status = "AMBIGUOUS"
        elif len({unvoc[j].signature() for j in best_js}) == 1:
            status = "PAIRED_TIE_IDENTICAL"
        else:
            status = "AMBIGUOUS"
        results.append(dict(voc_index=i, unvoc_index=best_js[0] if status != "AMBIGUOUS" else None,
                            score=top, runner_up_score=second if len(tied) == 1 else top,
                            status=status, tied=tied, edit_distance=dmin))
    return results


def args_gap_cells(lines: list[UMLine]) -> dict[str, set[str]]:
    """From a .args file: lemma -> set of UniMorph-4 tags whose form is a
    dash (the source explicitly marks the cell as non-existent)."""
    gaps = defaultdict(set)
    for ln in lines:
        if ln.form_raw.strip() in {"-", "–", "—"} and ln.tag_raw.startswith("V"):
            gaps[ln.lemma].add(ln.tag_raw)
    return gaps


def args_tag_to_v3(tag4: str) -> str | None:
    """Convert the UniMorph-4 verb tags used in heb.args to the v3 tag of heb_voc,
    for cross-checking the two renderings.  e.g. 'V;PST;NOM(3,SG,MASC)' ->
    canonical cell 'PST.3MSG'."""
    import re
    if tag4 == "V.MSDR":
        return "MSDR"
    if "(" not in tag4:  # plain v3-style tag (heb.args renders verbs this way)
        try:
            return parse_unimorph_tag(tag4)["canonical_cell"]
        except TagError:
            return None
    parts = tag4.split(";")
    if parts[0] != "V":
        return None
    tense = mood = None
    per = num = gen = ""
    nfin = False
    for p in parts[1:]:
        m = re.fullmatch(r"NOM\(([^)]*)\)", p)
        if m:
            for x in m.group(1).split(","):
                if x in "123":
                    per = x
                elif x in ("SG", "PL"):
                    num = x
                elif x == "MASC":
                    gen = "M"
                elif x == "FEM":
                    gen = "F"
        elif p in ("PST", "PRS", "FUT"):
            tense = p
        elif p == "IMP":
            mood = "IMP"
        elif p == "NFIN":
            nfin = True
        else:
            return None
    if nfin:
        return "NFIN"
    head = "IMP" if mood else tense
    return f"{head}.{per}{gen}{num}"


def lexeme_id_for(voc_lemma: str) -> str:
    return "LX-" + hashlib.sha1(("unimorph|" + normalize_nfc(voc_lemma)).encode()).hexdigest()[:10]
