"""INTERIM -> CANONICAL.  Builds roots / lexemes / forms / provenance /
validation plus conflict, duplicate and missing-metadata reports.

Conflicting information is never silently resolved: the affected field is
left NULL (root/binyan) or kept with variants, a flag is set, the record's
tier is QUESTIONABLE, and a row goes to the relevant report.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import pandas as pd

from . import PARSER_VERSIONS, paths
from .canonical_utils import (CELL_RANK, add_past_completeness, add_syncretism, join_flags,
                              root_rows_from_lexemes, worst_tier)
from .hebrew import (audit_characters, extract_consonantal_skeleton, loose_vocalized_key,
                     normalize_nfc, plene_consistent, strip_niqqud)
from .matching import (WiktionaryIndex, decide, entry_record_id, evidence_json, full_record_id,
                       root_bare_of, table_record_id)
from .morphology import (PAST_CELLS, binyan_from_paradigm_shape, feature_violations,
                         parse_unimorph_tag)
from .roots import (is_subsequence, orthographic, resolve_shin_sin, root_id_for, split_root_units,
                    stable_radicals)
from .unimorph import plene_key

# Flags that force a record to QUESTIONABLE.
LEXEME_HARD_FLAGS = {
    "ROOT_MISSING", "BINYAN_MISSING", "ROOT_AMBIGUOUS", "BINYAN_AMBIGUOUS", "MATCH_UNVOC_ONLY",
    "BINYAN_SHAPE_DISAGREES", "ROOT_FORM_MISMATCH", "SHIN_SIN_MISMATCH", "DUPLICATE_ROOT_BINYAN",
    "LEMMA_CITATION_MISMATCH", "WIKT_TABLE_ASSERTS_ROOT_BINYAN_ABSENT", "VOC_UNVOC_UNPAIRED",
    "VOC_UNVOC_PAIRING_AMBIGUOUS", "SAME_LEMMA_DIFFERENT_PARADIGMS", "SHIN_SIN_CONFLICT_IN_SOURCES",
}
FORM_HARD_FLAGS = {
    "CELL_CONFLICT_VOCALIZED", "CELL_CONFLICT_UNVOCALIZED", "VOC_UNVOC_INCONSISTENT",
    "WIKT_FORM_MISMATCH", "FEATURE_VIOLATION", "TAG_UNPARSEABLE", "MALFORMED_ORPHAN_MARK",
    "NON_HEBREW_CHARACTER", "MISSING_VOCALIZATION", "CONTAINS_SPACE", "MALFORMED_MARK_ON_INVALID_BASE",
    "MALFORMED_REPEATED_MARK",
}
# Flags that prevent GOLD but not SILVER.
LEXEME_NOGOLD_FLAGS = {"VOC_UNVOC_PAIRED_PARTIAL", "WIKT_CELL_CONFLICTS_IN_PARADIGM",
                       "BINYAN_SHAPE_UNDETERMINED", "SINGLE_EVIDENCE_KIND_ROOT",
                       "SINGLE_EVIDENCE_KIND_BINYAN", "EVIDENCE_TABLE_ROOT_FROM_TITLE",
                       "SHIN_SIN_UNRESOLVED", "WIKTIONARY_ONLY_LEXEME",
                       "UNVOC_SPELLING_VARIANTS_IN_PARADIGM"}
FORM_NOGOLD_FLAGS = {"MISSING_UNVOCALIZED", "UNVOC_SPELLING_VARIANTS", "NONSTANDARD_COMBINING_MARK",
                     "VOC_VARIANTS"}
INFO_FLAGS = {"RAW_NOT_NFC", "DUPLICATE_SOURCE_PARADIGM_MERGED", "ROOT_HOMONYM_INDEX",
              "UNVOC_ORPHAN_LINE_ATTACHED", "VOC_UNVOC_PAIRED_TIE_IDENTICAL", "WIKT_CORROBORATED",
              "WIKT_ORTHOGRAPHIC_VARIANT", "MULTIPLE_WIKT_SECTIONS", "SHIN_SIN_FROM_FORM",
              "ROOT_SHARES_ORTHOGRAPHY_WITH_OTHER_ROOT", "TRADITIONAL_CLASS_NOT_STATED"}


def _retrieval_dates():
    log = json.loads((paths.RAW / "retrieval_log.json").read_text())
    return log["unimorph_heb"]["retrieved"], log["hewiktionary_dump"]["retrieved"]


class Collector:
    def __init__(self):
        self.prov, self.val, self.conflicts, self.dups, self.missing = [], [], [], [], []
        self.um_date, self.wk_date = _retrieval_dates()

    # --- provenance -------------------------------------------------------
    def prov_unimorph(self, rec_id, rtype, field, file, line, raw):
        self.prov.append(dict(
            canonical_record_id=rec_id, record_type=rtype, field_supported=field,
            source_name="unimorph_heb", source_url=paths.unimorph_url(file, line), source_file=file,
            source_version_or_commit=paths.UNIMORPH_COMMIT, source_record_identifier=f"{file}:{line}",
            source_line_if_available=line, retrieval_date=self.um_date,
            parser_version=PARSER_VERSIONS["unimorph_parser"], raw_source_value=raw))

    def prov_wikt(self, rec_id, rtype, field, record_id, title, rev_id, raw):
        self.prov.append(dict(
            canonical_record_id=rec_id, record_type=rtype, field_supported=field,
            source_name="hewiktionary", source_url=paths.wiktionary_url(title, rev_id),
            source_file=paths.WIKTIONARY_DUMP.name, source_version_or_commit=paths.WIKTIONARY_DUMP_DATE,
            source_record_identifier=record_id, source_line_if_available=None,
            retrieval_date=self.wk_date, parser_version=PARSER_VERSIONS["wiktionary_parser"],
            raw_source_value=raw))

    # --- issues -----------------------------------------------------------
    def flag_rows(self, rec_id, rtype, flags, details=None):
        for f in flags:
            sev = ("ERROR" if f in LEXEME_HARD_FLAGS | FORM_HARD_FLAGS else
                   "INFO" if f in INFO_FLAGS else "WARNING")
            self.val.append(dict(record_id=rec_id, record_type=rtype, check_name=f, severity=sev,
                                 details=(details or {}).get(f)))

    def conflict(self, ctype, rtype, rec_id, lemma, details, candidates=None, sources=None):
        self.conflicts.append(dict(conflict_type=ctype, record_type=rtype, record_id=rec_id, lemma=lemma,
                                   details=details,
                                   candidates=json.dumps(candidates, ensure_ascii=False) if candidates is not None else None,
                                   sources=sources))


# ==========================================================================
def load_interim():
    I = paths.INTERIM
    return dict(
        lines=pd.read_parquet(I / "unimorph" / "verb_lines.parquet"),
        paradigms=pd.read_parquet(I / "unimorph" / "paradigms.parquet"),
        pairing=pd.read_parquet(I / "unimorph" / "voc_unvoc_pairing.parquet"),
        gaps=pd.read_parquet(I / "unimorph" / "args_explicit_gaps.parquet"),
        ud=pd.read_parquet(I / "unimorph" / "ud_verb_forms.parquet"),
        entries=pd.read_parquet(I / "wiktionary" / "verb_entries.parquet"),
        tables=pd.read_parquet(I / "wiktionary" / "conj_tables.parquet"),
        full=pd.read_parquet(I / "wiktionary" / "full_tables.parquet"),
    )


def _lexeme_id(source: str, key: str) -> str:
    return "LX-" + hashlib.sha1(f"{source}|{key}".encode()).hexdigest()[:10]


def _wikt_cells_for(idx: WiktionaryIndex, root_bare, binyan, lemma):
    """Wiktionary table cells for (root, binyan) from tables whose PST.3MSG
    matches the lemma.  Returns cell -> list of (form, record_id, row)."""
    out = defaultdict(list)
    if not root_bare or not binyan:
        return out
    lk = loose_vocalized_key(lemma)
    groups = defaultdict(list)
    for r in idx.tables_by_rb.get((root_bare, binyan), []):
        groups[("conj", r.page_id, r.table_index)].append(r)
    for r in idx.full_by_rb.get((root_bare, binyan), []):
        groups[("full", r.page_id, r.table_index)].append(r)
    for (kind, _, _), rows in groups.items():
        past = [f for r in rows if r.cell == "PST.3MSG" for f in (r.forms or "").split("|") if f]
        if not any(f == lemma or loose_vocalized_key(f) == lk for f in past):
            continue
        for r in rows:
            rid = table_record_id(r) if kind == "conj" else full_record_id(r)
            for f in (r.forms or "").split("|"):
                if f:
                    out[r.cell].append((f, rid, r))
    return out


def _ortho_key(s: str) -> str:
    """Neutralise holam male/haser and shuruk/qubuts spelling variation in
    vocalized text (וֹ -> ֹ on previous letter is approximated by deleting the
    vav+holam and vav+dagesh(shuruk) mater and the holam/qubuts points)."""
    k = loose_vocalized_key(s)
    k = k.replace("וֹ", "").replace("ֹ", "").replace("ֻ", "")
    k = k.replace("וּ", "")
    return k


def build_real(verbose=True):
    d = load_interim()
    C = Collector()
    idx = WiktionaryIndex(d["entries"], d["tables"], d["full"])
    lines = d["lines"]
    voc_lines = lines[lines.file == "heb_voc"]
    unv_lines = lines[lines.file == "heb"]
    par = d["paradigms"].set_index("paradigm_key")
    pairing = d["pairing"].set_index("voc_paradigm_key")
    ud_forms = set(d["ud"].form_nfc)

    def cells_of(sub):
        out = {}
        for r in sub.itertuples(index=False):
            out.setdefault(r.canonical_cell, []).append(r)
        return out
    voc_by_par = {k: cells_of(g) for k, g in voc_lines.groupby("paradigm_key", sort=False)}
    unv_by_par = {k: cells_of(g) for k, g in unv_lines.groupby("paradigm_key", sort=False)}

    # ---- group voc paradigms into lexeme candidates ----------------------
    groups = defaultdict(list)  # (lemma, signature) -> [paradigm_key]
    order = []
    for k in voc_by_par:
        key = (par.loc[k, "lemma_nfc"], par.loc[k, "signature"])
        if key not in groups:
            order.append(key)
        groups[key].append(k)
    lemma_count = defaultdict(int)
    candidates = []
    for key in order:
        lemma, _ = key
        lemma_count[lemma] += 1
        k = lemma_count[lemma]
        candidates.append(dict(lemma=lemma, members=groups[key],
                               lexeme_id=_lexeme_id("unimorph", lemma + ("" if k == 1 else f"|{k}"))))
    same_lemma_multi = {l for l, n in lemma_count.items() if n > 1}

    # ---- unvoc members: paired paradigms + orphan unvoc paradigms ---------
    paired_unvoc = set(pairing.unvoc_paradigm_key.dropna())
    orphan_unvoc = [k for k in unv_by_par if k not in paired_unvoc]
    orphan_attach = defaultdict(list)
    for ok in orphan_unvoc:
        ocells = unv_by_par[ok]
        hits = []
        for c in candidates:
            vc = voc_by_par[c["members"][0]]
            if all(cell in vc and extract_consonantal_skeleton(rows[0].form_nfc, drop_vav_yod=True)
                   == plene_key(vc[cell][0].form_nfc) for cell, rows in ocells.items()):
                hits.append(c["lexeme_id"])
        lemma_hits = [h for h in hits if any(
            pairing.loc[m, "unvoc_lemma"] == par.loc[ok, "lemma_nfc"]
            for c in candidates if c["lexeme_id"] == h for m in c["members"])]
        # prefer the lexeme whose paired unvocalized lemma equals the orphan's lemma
        chosen = lemma_hits if len(lemma_hits) == 1 else (hits if len(hits) == 1 else [])
        if chosen:
            orphan_attach[chosen[0]].append(ok)
        else:
            C.conflict("ORPHAN_UNVOCALIZED_PARADIGM", "source", ok, par.loc[ok, "lemma_nfc"],
                       f"unvocalized paradigm with {len(ocells)} cells could not be attached to a unique lexeme",
                       candidates=hits, sources=ok)

    lex_rows, form_rows = [], []
    lexeme_evidence = {}
    used_entry_ids = set()

    for cand in candidates:
        lid, lemma, members = cand["lexeme_id"], cand["lemma"], cand["members"]
        lflags, ldetails = [], {}
        m0 = members[0]
        vcells = {c: [r for m in members for r in voc_by_par[m].get(c, [])]
                  for c in sorted({c for m in members for c in voc_by_par[m]}, key=lambda c: CELL_RANK.get(c, 999))}
        if len(members) > 1:
            lflags.append("DUPLICATE_SOURCE_PARADIGM_MERGED")
            C.dups.append(dict(duplicate_type="IDENTICAL_SOURCE_PARADIGMS", record_ids=lid,
                               lemma=lemma, details="merged identical heb_voc paradigms: " + ", ".join(members)))
        if lemma in same_lemma_multi:
            lflags.append("SAME_LEMMA_DIFFERENT_PARADIGMS")
        # pairing
        pstat = [pairing.loc[m, "status"] for m in members]
        pscore = [float(pairing.loc[m, "score"]) for m in members]
        unv_members = [pairing.loc[m, "unvoc_paradigm_key"] for m in members
                       if isinstance(pairing.loc[m, "unvoc_paradigm_key"], str)]
        unv_members = list(dict.fromkeys(unv_members))
        orphan_members = orphan_attach.get(lid, [])
        if orphan_members:
            lflags.append("UNVOC_ORPHAN_LINE_ATTACHED")
        ucells = defaultdict(list)
        for m in unv_members + orphan_members:
            for c, rows in unv_by_par[m].items():
                ucells[c] += rows
        status = pstat[0]
        if any(s == "UNPAIRED" for s in pstat):
            lflags.append("VOC_UNVOC_UNPAIRED")
        elif any(s == "AMBIGUOUS" for s in pstat):
            lflags.append("VOC_UNVOC_PAIRING_AMBIGUOUS")
        elif any(s == "PAIRED_PARTIAL" for s in pstat):
            lflags.append("VOC_UNVOC_PAIRED_PARTIAL")
            status = "PAIRED_PARTIAL"
        elif any(s == "PAIRED_TIE_IDENTICAL" for s in pstat):
            lflags.append("VOC_UNVOC_PAIRED_TIE_IDENTICAL")

        def first(c, src):
            rows = src.get(c) or []
            return rows[0].form_nfc if rows else None
        past3 = first("PST.3MSG", vcells)
        inf = first("NFIN", vcells)
        lemma_unvoc = (pairing.loc[m0, "unvoc_lemma"] if isinstance(pairing.loc[m0, "unvoc_lemma"], str) else None)
        if past3 and past3 != lemma:
            lflags.append("LEMMA_CITATION_MISMATCH")
            C.conflict("LEMMA_CITATION_MISMATCH", "lexeme", lid, lemma,
                       f"UniMorph lemma {lemma} != PST.3MSG cell {past3}", sources=m0)

        # ---- matching --------------------------------------------------------
        ev = idx.evidence_for(lemma, past3, inf, lemma_unvoc)
        dec = decide(ev)
        lexeme_evidence[lid] = (ev, dec)
        for e in ev:
            if e.kind == "entry":
                used_entry_ids.add(e.record_id)
        root_bare, binyan = dec["root_bare"], dec["binyan"]
        if dec["status"] == "NO_MATCH":
            lflags += ["ROOT_MISSING", "BINYAN_MISSING"]
        if dec["status"] in ("ROOT_AMBIGUOUS", "ROOT_AND_BINYAN_AMBIGUOUS"):
            lflags.append("ROOT_AMBIGUOUS")
            C.conflict("ROOT_AMBIGUOUS", "lexeme", lid, lemma,
                       "evidence supports several roots: " + ", ".join(dec["roots_supported"]),
                       candidates=[dict(method=e.method, root=e.root, binyan=e.binyan, record=e.record_id,
                                        page=e.page_title) for e in dec["used"]],
                       sources="|".join(sorted({e.record_id for e in dec["used"]})))
        if dec["status"] in ("BINYAN_AMBIGUOUS", "ROOT_AND_BINYAN_AMBIGUOUS"):
            lflags.append("BINYAN_AMBIGUOUS")
            C.conflict("BINYAN_AMBIGUOUS", "lexeme", lid, lemma,
                       "evidence supports several binyanim: " + ", ".join(dec["binyans_supported"]),
                       candidates=[dict(method=e.method, root=e.root, binyan=e.binyan, binyan_raw=e.binyan_raw,
                                        record=e.record_id, page=e.page_title) for e in dec["used"]],
                       sources="|".join(sorted({e.record_id for e in dec["used"]})))
        if dec["status"] == "MATCHED_UNVOC_ONLY":
            lflags.append("MATCH_UNVOC_ONLY")
        if dec["status"] == "MATCHED" or dec["status"] == "MATCHED_UNVOC_ONLY":
            if root_bare is None:
                lflags.append("ROOT_MISSING")
            if binyan is None:
                lflags.append("BINYAN_MISSING")
        if root_bare and len(dec["root_kinds"]) < 2:
            lflags.append("SINGLE_EVIDENCE_KIND_ROOT")
        if binyan and len(dec["binyan_kinds"]) < 2:
            lflags.append("SINGLE_EVIDENCE_KIND_BINYAN")
        for e in dec["agree_root"]:
            if e.kind == "conj_table":
                rows = [r for r in idx.tables_by_rb.get((e.root_bare, e.binyan), []) if table_record_id(r) == e.record_id]
                if rows and rows[0].flags and "TABLE_ROOT_" in rows[0].flags:
                    lflags.append("EVIDENCE_TABLE_ROOT_FROM_TITLE")

        # ---- checks against forms --------------------------------------------
        vfirst = {c: rows[0].form_nfc for c, rows in vcells.items()}
        shape_b, shape_rule = binyan_from_paradigm_shape(vfirst)
        if binyan:
            if shape_b is None:
                lflags.append("BINYAN_SHAPE_UNDETERMINED")
            elif shape_b != binyan:
                lflags.append("BINYAN_SHAPE_DISAGREES")
                C.conflict("BINYAN_SHAPE_DISAGREES", "lexeme", lid, lemma,
                           f"source binyan {binyan} but paradigm shape suggests {shape_b} ({shape_rule}); "
                           f"PRS.MSG={vfirst.get('PRS.MSG')} NFIN={vfirst.get('NFIN')}",
                           sources="|".join(e.record_id for e in dec["agree_bin"]))
        radicals_dotted = dec["root_dotted_variants"][0].split("-") if dec["root_dotted_variants"] else None
        root_norm, ss_method = None, None
        if root_bare:
            st = stable_radicals(list(root_bare))
            skel = extract_consonantal_skeleton(past3 or lemma)
            if not is_subsequence(st, skel):
                lflags.append("ROOT_FORM_MISMATCH")
                C.conflict("ROOT_FORM_MISMATCH", "lexeme", lid, lemma,
                           f"stable radicals {'-'.join(st)} of root {root_bare} not found in order in {past3 or lemma}",
                           sources="|".join(e.record_id for e in dec["agree_root"]))
            rads, ss_method, ss_flags = resolve_shin_sin(root_bare, dec["root_dotted_variants"], lemma)
            root_norm = "".join(rads)
            lflags += ss_flags
            for fl in ss_flags:
                if fl in ("SHIN_SIN_MISMATCH", "SHIN_SIN_CONFLICT_IN_SOURCES"):
                    C.conflict(fl, "lexeme", lid, lemma,
                               f"shin/sin: source radicals {dec['root_dotted_variants']} vs lemma {lemma}",
                               sources="|".join(e.record_id for e in dec["agree_root"]))
        if root_bare and binyan and idx.absent_by_rb.get((root_bare, binyan)):
            lflags.append("WIKT_TABLE_ASSERTS_ROOT_BINYAN_ABSENT")
            rs = idx.absent_by_rb[(root_bare, binyan)]
            C.conflict("WIKT_TABLE_ASSERTS_ROOT_BINYAN_ABSENT", "lexeme", lid, lemma,
                       f"a Wiktionary conjugation table marks {root_bare} x {binyan} as non-existent (-אין-)",
                       sources="|".join(table_record_id(r) for r in rs))
        if len(dec["agree_root"]) and len({e.extra.get("homonym") for e in dec["agree_root"] if e.kind == "entry"}) > 1:
            lflags.append("MULTIPLE_WIKT_SECTIONS")

        # ---- Wiktionary cells for corroboration --------------------------------
        wcells = _wikt_cells_for(idx, root_bare, binyan, lemma) if dec["status"] == "MATCHED" else {}

        # ---- forms ---------------------------------------------------------------
        cell_list = list(vcells) + [c for c in ucells if c not in vcells]
        any_wikt_conflict = False
        any_unvoc_variants = False
        for cell in cell_list:
            fflags, fdet = [], {}
            vrows = vcells.get(cell, [])
            urows = ucells.get(cell, [])
            vforms = list(dict.fromkeys(r.form_nfc for r in vrows))
            uforms = list(dict.fromkeys(r.form_nfc for r in urows))
            # duplicate cell lines inside a single paradigm = incompatible surface forms
            if len(vforms) > 1:
                fflags.append("CELL_CONFLICT_VOCALIZED")
                C.conflict("CELL_CONFLICT_VOCALIZED", "form", f"{lid}.{cell}", lemma,
                           "several vocalized forms for one lexeme x cell: " + " | ".join(vforms),
                           sources="|".join(f"{r.file}:{r.line_no}" for r in vrows))
            if len(uforms) > 1:
                fflags.append("UNVOC_SPELLING_VARIANTS")
                any_unvoc_variants = True
                C.conflict("CELL_CONFLICT_UNVOCALIZED", "form", f"{lid}.{cell}", lemma,
                           "several plene spellings for one lexeme x cell: " + " | ".join(uforms),
                           sources="|".join(f"{r.file}:{r.line_no}" for r in urows))
            prim = vrows[0] if vrows else urows[0]
            fvoc = vforms[0] if vforms else None
            funv = uforms[0] if uforms else None
            if not vrows:
                fflags.append("MISSING_VOCALIZATION")
            if not urows:
                fflags.append("MISSING_UNVOCALIZED")
            try:
                feats = parse_unimorph_tag(prim.tag_raw)
                viol = feature_violations(feats)
                if viol:
                    fflags.append("FEATURE_VIOLATION")
                    fdet["FEATURE_VIOLATION"] = ",".join(viol)
            except Exception as e:  # noqa: BLE001
                fflags.append("TAG_UNPARSEABLE")
                fdet["TAG_UNPARSEABLE"] = str(e)
                feats = dict(pos="V", tense="NA", aspect="NA", mood="NA", person="NA", number="NA",
                             gender="NA", voice="NA", finiteness="FIN", canonical_cell=cell)
            aud = audit_characters(prim.form_raw)
            for f in aud.flags:
                fflags.append(f)
            if aud.nonstandard_marks:
                fdet["NONSTANDARD_COMBINING_MARK"] = ",".join(aud.nonstandard_marks)
            if urows:
                for f in audit_characters(urows[0].form_raw).flags:
                    if f != "RAW_NOT_NFC":
                        fflags.append(f)
            pc = plene_consistent(fvoc, funv)
            if pc is False:
                fflags.append("VOC_UNVOC_INCONSISTENT")
                C.conflict("VOC_UNVOC_INCONSISTENT", "form", f"{lid}.{cell}", lemma,
                           f"vocalized {fvoc} (stripped {strip_niqqud(fvoc)}) vs plene {funv}",
                           sources="|".join([f"{r.file}:{r.line_no}" for r in vrows[:1] + urows[:1]]))
            # Wiktionary corroboration
            wc = None
            if fvoc and cell in wcells:
                wvals = wcells[cell]
                if any(f == fvoc or loose_vocalized_key(f) == loose_vocalized_key(fvoc) for f, _, _ in wvals):
                    wc = True
                    fflags.append("WIKT_CORROBORATED")
                elif any(_ortho_key(f) == _ortho_key(fvoc) for f, _, _ in wvals):
                    wc = True
                    fflags.append("WIKT_ORTHOGRAPHIC_VARIANT")
                else:
                    wc = False
                    any_wikt_conflict = True
                    fflags.append("WIKT_FORM_MISMATCH")
                    C.conflict("WIKT_FORM_MISMATCH", "form", f"{lid}.{cell}", lemma,
                               f"UniMorph {fvoc} vs Wiktionary {' | '.join(sorted({f for f, _, _ in wvals}))}",
                               sources="|".join(sorted({rid for _, rid, _ in wvals})))
            fid = f"{lid}.{cell}"
            form_rows.append(dict(
                form_id=fid, lexeme_id=lid, form_raw=prim.form_raw, form_nfc=normalize_nfc(prim.form_raw),
                form_vocalized=fvoc, form_unvocalized=funv, form_stripped=strip_niqqud(fvoc) if fvoc else None,
                form_vocalized_variants="|".join(vforms[1:]) or None,
                form_unvocalized_variants="|".join(uforms[1:]) or None,
                pos=feats["pos"], finiteness=feats["finiteness"], tense=feats["tense"], aspect=feats["aspect"],
                mood=feats["mood"], person=feats["person"], number=feats["number"], gender=feats["gender"],
                voice=feats["voice"], canonical_cell=cell, is_past_cell=cell in PAST_CELLS,
                unimorph_tag_raw=prim.tag_raw, plene_consistent=pc, wiktionary_corroborated=wc,
                ud_attested=bool(funv and funv in ud_forms), is_attested=True, is_generated=False,
                source_primary="unimorph_heb",
                source_ids="|".join([f"{r.file}:{r.line_no}" for r in vrows] + [f"{r.file}:{r.line_no}" for r in urows]),
                _flags=fflags, _details=fdet,
            ))
            for r in vrows:
                C.prov_unimorph(fid, "form", "form_vocalized", r.file, r.line_no,
                                f"{r.lemma_raw}\t{r.form_raw}\t{r.tag_raw}")
            for r in urows:
                C.prov_unimorph(fid, "form", "form_unvocalized", r.file, r.line_no,
                                f"{r.lemma_raw}\t{r.form_raw}\t{r.tag_raw}")
            if wc is not None:
                for f, rid, r in wcells[cell]:
                    C.prov_wikt(fid, "form", "form_vocalized_corroboration", rid, r.page_title, r.rev_id, r.value_raw)
        if any_wikt_conflict:
            lflags.append("WIKT_CELL_CONFLICTS_IN_PARADIGM")
        if any_unvoc_variants:
            lflags.append("UNVOC_SPELLING_VARIANTS_IN_PARADIGM")

        # ---- lexeme row ----------------------------------------------------------
        gl = [e.extra.get("gloss_en") for e in dec["agree_root"] + dec["agree_bin"] if e.kind == "entry" and e.extra.get("gloss_en")]
        homs = sorted({e.extra.get("heading") for e in dec["agree_root"] + dec["agree_bin"] if e.kind == "entry" and "משני" in (e.extra.get("heading") or "")})
        giz = sorted({x for e in dec["agree_root"] if e.kind == "entry" and e.extra.get("gizra_labels") for x in e.extra["gizra_labels"].split("+")})
        regs = sorted({x for e in dec["agree_root"] + dec["agree_bin"] if e.kind == "entry" and e.extra.get("registers") for x in e.extra["registers"].split("|")})
        braws = sorted({e.binyan_raw.strip() for e in dec["agree_bin"] if e.binyan_raw})
        root_id = root_id_for(root_norm) if root_norm else None
        lex_rows.append(dict(
            lexeme_id=lid, language="heb", root_id=root_id,
            root_radicals_dotted="-".join(radicals_dotted) if radicals_dotted else None,
            root_homonym_index="|".join(sorted({e.extra.get("homonym") for e in dec["agree_root"] if e.extra.get("homonym")})) or None,
            binyan=binyan, binyan_raw="|".join(braws) or None,
            lemma_vocalized=lemma, lemma_unvocalized=lemma_unvoc,
            infinitive_vocalized=inf, infinitive_unvocalized=first("NFIN", ucells),
            citation_past_3ms_vocalized=past3, citation_past_3ms_unvocalized=first("PST.3MSG", ucells),
            gloss_en="; ".join(dict.fromkeys(gl)) or None,
            sense_id="|".join(homs) or None,
            modern_hebrew=True, attested=True, has_unimorph_paradigm=True,
            source_primary="unimorph_heb",
            source_ids="|".join(members + unv_members + orphan_members + sorted({e.record_id for e in dec["agree_root"] + dec["agree_bin"]})),
            match_status=dec["status"], match_method="|".join(dec["methods"]) or None,
            match_score=dec["match_score"], match_evidence=evidence_json(ev) if ev else None,
            match_ambiguous=dec["ambiguous"], binyan_shape_heuristic=shape_b, binyan_shape_rule=shape_rule,
            voc_unvoc_pairing_status=status, voc_unvoc_pairing_score=min(pscore),
            gizra_labels="+".join(giz) or None, wiktionary_registers="|".join(regs) or None,
            ud_attested=False,  # filled below
            shin_sin_resolution=ss_method,
            _root_bare=root_bare, _root_norm=root_norm, _dec=dec, _flags=lflags,
        ))
        for m in members:
            C.prov_unimorph(lid, "lexeme", "paradigm_vocalized", "heb_voc", int(m.split(":")[1]),
                            f"paradigm of {par.loc[m, 'n_lines']} lines, lemma {par.loc[m, 'lemma_raw']}")
        for m in unv_members + orphan_members:
            C.prov_unimorph(lid, "lexeme", "paradigm_unvocalized", "heb", int(m.split(":")[1]),
                            f"paradigm of {par.loc[m, 'n_lines']} lines, lemma {par.loc[m, 'lemma_raw']}")
        for e in {e.record_id: e for e in dec["agree_root"] + dec["agree_bin"]}.values():
            fld = "+".join(x for x, ok in [("root", e in dec["agree_root"]), ("binyan", e in dec["agree_bin"])] if ok)
            C.prov_wikt(lid, "lexeme", fld, e.record_id, e.page_title, e.rev_id,
                        e.extra.get("value_raw") or e.extra.get("root_value_raw") or e.matched_string)

    # ======================================================================
    # Wiktionary-only lexemes (entries with root+binyan, no UniMorph paradigm)
    # ======================================================================
    um_rb = {(r["_root_norm"], r["binyan"]) for r in lex_rows if r["_root_norm"] and r["binyan"]}
    e_df = d["entries"]
    wk_groups = defaultdict(list)
    for r in e_df.itertuples(index=False):
        rid = entry_record_id(r)
        if rid in used_entry_ids:
            continue
        from .morphology import normalize_binyan
        b, bfl = normalize_binyan(r.binyan_raw)
        rb = root_bare_of(r.root_radicals)
        if not (r.headword and rb and b):
            C.missing.append(dict(record_id=rid, record_type="wiktionary_entry", lemma=r.headword,
                                  missing_fields="|".join(x for x, ok in [("headword", r.headword), ("root", rb), ("binyan", b)] if not ok),
                                  details=f"not ingested as a lexeme; binyan_raw={r.binyan_raw!r} root_raw={r.root_value_raw!r}"))
            continue
        rn = "".join(resolve_shin_sin(rb, [r.root_radicals], r.headword)[0])
        if (rn, b) in um_rb:
            C.dups.append(dict(duplicate_type="WIKT_ENTRY_FOR_EXISTING_ROOT_BINYAN", record_ids=rid, lemma=r.headword,
                               details=f"{rb} x {b} already represented by a UniMorph lexeme; entry not added"))
            continue
        wk_groups[(r.headword, rb, b)].append(r)
    for (hw, rb, b), rs in wk_groups.items():
        lid = _lexeme_id("hewiktionary", f"{hw}|{rb}|{b}")
        lflags = ["WIKTIONARY_ONLY_LEXEME"]
        ev = idx.evidence_for(hw, hw, None, None)
        ev = [e for e in ev if e.root_bare == rb or e.binyan == b]
        dec = decide(ev)
        if dec["root_bare"] != rb or dec["binyan"] != b:
            # other records disagree (e.g. a table with another binyan for this form)
            lflags.append("ROOT_AMBIGUOUS" if dec["root_bare"] != rb else "BINYAN_AMBIGUOUS")
            C.conflict("WIKT_ONLY_EVIDENCE_DISAGREES", "lexeme", lid, hw,
                       f"entry says {rb} x {b}; other records support roots {dec['roots_supported']} binyanim {dec['binyans_supported']}",
                       sources="|".join(e.record_id for e in ev))
        if len(dec["root_kinds"]) < 2:
            lflags.append("SINGLE_EVIDENCE_KIND_ROOT")
        if len(dec["binyan_kinds"]) < 2:
            lflags.append("SINGLE_EVIDENCE_KIND_BINYAN")
        wcells = _wikt_cells_for(idx, rb, b, hw)
        cells = {"PST.3MSG": [(hw, entry_record_id(rs[0]), None)]}
        for c, vals in wcells.items():
            if c != "PST.3MSG":
                cells[c] = vals
        vf = {c: v[0][0] for c, v in cells.items()}
        shape_b, shape_rule = binyan_from_paradigm_shape(vf)
        if shape_b and shape_b != b:
            lflags.append("BINYAN_SHAPE_DISAGREES")
            C.conflict("BINYAN_SHAPE_DISAGREES", "lexeme", lid, hw, f"source {b} vs shape {shape_b} ({shape_rule})",
                       sources=entry_record_id(rs[0]))
        elif shape_b is None:
            lflags.append("BINYAN_SHAPE_UNDETERMINED")
        st = stable_radicals(list(rb))
        if not is_subsequence(st, extract_consonantal_skeleton(hw)):
            lflags.append("ROOT_FORM_MISMATCH")
            C.conflict("ROOT_FORM_MISMATCH", "lexeme", lid, hw, f"root {rb} vs headword {hw}", sources=entry_record_id(rs[0]))
        if idx.absent_by_rb.get((rb, b)):
            lflags.append("WIKT_TABLE_ASSERTS_ROOT_BINYAN_ABSENT")
            C.conflict("WIKT_TABLE_ASSERTS_ROOT_BINYAN_ABSENT", "lexeme", lid, hw,
                       f"entry exists but a conjugation table marks {rb} x {b} as non-existent",
                       sources="|".join(table_record_id(r) for r in idx.absent_by_rb[(rb, b)]))
        rads, ss_method, ss_flags = resolve_shin_sin(rb, sorted({r.root_radicals for r in rs if r.root_radicals}), hw)
        rn = "".join(rads)
        lflags += ss_flags
        for fl in ss_flags:
            if fl in ("SHIN_SIN_MISMATCH", "SHIN_SIN_CONFLICT_IN_SOURCES"):
                C.conflict(fl, "lexeme", lid, hw, f"shin/sin: {sorted({r.root_radicals for r in rs})} vs {hw}",
                           sources=entry_record_id(rs[0]))
        ktiv = next((r.ktiv_male for r in rs if r.ktiv_male), None)
        regs = sorted({x for r in rs if r.registers for x in r.registers.split("|")})
        for c, vals in cells.items():
            forms = list(dict.fromkeys(v[0] for v in vals))
            fflags = []
            if len(forms) > 1:
                fflags.append("VOC_VARIANTS")
            funv = ktiv if c == "PST.3MSG" else None
            if not funv:
                fflags.append("MISSING_UNVOCALIZED")
            pc = plene_consistent(forms[0], funv)
            if pc is False:
                fflags.append("VOC_UNVOC_INCONSISTENT")
                C.conflict("VOC_UNVOC_INCONSISTENT", "form", f"{lid}.{c}", hw, f"{forms[0]} vs ktiv male {funv}",
                           sources=entry_record_id(rs[0]))
            fflags += [f for f in audit_characters(forms[0]).flags]
            from .morphology import parse_unimorph_tag as _p
            tagmap = {"PST.3MSG": "V;3;SG;PST;MASC", "PRS.MSG": "V;SG;PRS;MASC", "FUT.3MSG": "V;3;SG;FUT;MASC",
                      "IMP.2MSG": "V;2;SG;IMP;MASC", "NFIN": "V;NFIN"}
            import re as _re
            if c in tagmap:
                feats = _p(tagmap[c])
            else:
                m = _re.fullmatch(r"(PST|PRS|FUT|IMP)\.([123]?)([MF]?)(SG|PL)", c)
                t = "V;" + ";".join(x for x in [m.group(2), m.group(4), m.group(1),
                                                {"M": "MASC", "F": "FEM", "": ""}[m.group(3)]] if x)
                feats = _p(t)
            fid = f"{lid}.{c}"
            form_rows.append(dict(
                form_id=fid, lexeme_id=lid, form_raw=forms[0], form_nfc=normalize_nfc(forms[0]),
                form_vocalized=forms[0], form_unvocalized=funv, form_stripped=strip_niqqud(forms[0]),
                form_vocalized_variants="|".join(forms[1:]) or None, form_unvocalized_variants=None,
                pos=feats["pos"], finiteness=feats["finiteness"], tense=feats["tense"], aspect="NA",
                mood=feats["mood"], person=feats["person"], number=feats["number"], gender=feats["gender"],
                voice="NA", canonical_cell=c, is_past_cell=c in PAST_CELLS, unimorph_tag_raw=None,
                plene_consistent=pc, wiktionary_corroborated=None, ud_attested=bool(funv and funv in ud_forms),
                is_attested=True, is_generated=False, source_primary="hewiktionary",
                source_ids="|".join(dict.fromkeys(v[1] for v in vals)), _flags=fflags, _details={}))
            for f, rid, row in vals:
                if row is None:
                    C.prov_wikt(fid, "form", "form_vocalized", rid, rs[0].page_title, rs[0].rev_id,
                                rs[0].section_heading_raw)
                else:
                    C.prov_wikt(fid, "form", "form_vocalized", rid, row.page_title, row.rev_id, row.value_raw)
            if funv:
                C.prov_wikt(fid, "form", "form_unvocalized", entry_record_id(rs[0]), rs[0].page_title,
                            rs[0].rev_id, f"כתיב מלא={ktiv}")
        braws = sorted({r.binyan_raw.strip() for r in rs if r.binyan_raw})
        gl = [r.gloss_en for r in rs if r.gloss_en]
        giz = sorted({x for r in rs if r.gizra_labels for x in r.gizra_labels.split("+")})
        dotted = sorted({r.root_radicals for r in rs if r.root_radicals})
        lex_rows.append(dict(
            lexeme_id=lid, language="heb", root_id=root_id_for(rn), root_radicals_dotted=dotted[0] if dotted else None,
            root_homonym_index="|".join(sorted({r.root_homonym_index for r in rs if r.root_homonym_index})) or None,
            binyan=b, binyan_raw="|".join(braws) or None, lemma_vocalized=hw, lemma_unvocalized=ktiv,
            infinitive_vocalized=vf.get("NFIN"), infinitive_unvocalized=None,
            citation_past_3ms_vocalized=hw, citation_past_3ms_unvocalized=ktiv,
            gloss_en="; ".join(dict.fromkeys(gl)) or None,
            sense_id="|".join(sorted({r.section_heading_raw for r in rs if "משני" in r.section_heading_raw})) or None,
            modern_hebrew=True if any("עברית חדשה" in x for x in regs) else None,
            attested=True, has_unimorph_paradigm=False, source_primary="hewiktionary",
            source_ids="|".join(sorted({entry_record_id(r) for r in rs} | {e.record_id for e in dec["agree_root"] + dec["agree_bin"]})),
            match_status="WIKTIONARY_ONLY", match_method="|".join(dec["methods"]) or None,
            match_score=dec["match_score"], match_evidence=evidence_json(ev) if ev else None,
            match_ambiguous=dec["ambiguous"], binyan_shape_heuristic=shape_b, binyan_shape_rule=shape_rule,
            voc_unvoc_pairing_status=None, voc_unvoc_pairing_score=None,
            gizra_labels="+".join(giz) or None, wiktionary_registers="|".join(regs) or None, ud_attested=False,
            shin_sin_resolution=ss_method, _root_bare=rb, _root_norm=rn, _dec=dec, _flags=lflags))
        for r in rs:
            C.prov_wikt(lid, "lexeme", "root+binyan+lemma", entry_record_id(r), r.page_title, r.rev_id, r.template_raw)
        for e in {e.record_id: e for e in dec["agree_root"] + dec["agree_bin"]}.values():
            if e.kind != "entry":
                C.prov_wikt(lid, "lexeme", "root+binyan", e.record_id, e.page_title, e.rev_id,
                            e.extra.get("value_raw") or e.matched_string)

    # ======================================================================
    # duplicate root x binyan detection
    # ======================================================================
    rb_map = defaultdict(list)
    for r in lex_rows:
        if r["_root_norm"] and r["binyan"]:
            rb_map[(r["_root_norm"], r["binyan"])].append(r)
    for (rb, b), rs in rb_map.items():
        if len(rs) > 1:
            for r in rs:
                r["_flags"].append("DUPLICATE_ROOT_BINYAN")
            C.dups.append(dict(duplicate_type="DUPLICATE_ROOT_BINYAN", record_ids="|".join(r["lexeme_id"] for r in rs),
                               lemma=" / ".join(r["lemma_vocalized"] for r in rs),
                               details=f"{len(rs)} lexeme records for root {rb} x {b}"))
    # same form string, incompatible analyses (different lexemes)
    fdf = pd.DataFrame(form_rows)

    # ======================================================================
    # tiers
    # ======================================================================
    lex_tier = {}
    for r in lex_rows:
        fl = set(r["_flags"])
        dec = r["_dec"]
        if fl & LEXEME_HARD_FLAGS or not (r["root_id"] and r["binyan"]):
            tier = "QUESTIONABLE"
        elif (fl & LEXEME_NOGOLD_FLAGS or r["match_status"] != "MATCHED" or
              r["voc_unvoc_pairing_status"] not in ("PAIRED", "PAIRED_TIE_IDENTICAL") or
              r["binyan_shape_heuristic"] != r["binyan"]):
            tier = "SILVER"
        else:
            tier = "GOLD"
        cr = dec["conf_root"] * (0.5 if "ROOT_FORM_MISMATCH" in fl else 1.0) if r["root_id"] else 0.0
        cb = dec["conf_binyan"] * (0.5 if "BINYAN_SHAPE_DISAGREES" in fl else 1.0) if r["binyan"] else 0.0
        r["confidence_root"] = round(cr, 4)
        r["confidence_binyan"] = round(cb, 4)
        r["confidence_lexeme"] = round(min(cr, cb) * (1.0 if r["voc_unvoc_pairing_status"] in ("PAIRED", "PAIRED_TIE_IDENTICAL") else 0.8), 4)
        r["quality_tier"] = tier
        lex_tier[r["lexeme_id"]] = (tier, r["confidence_lexeme"], r["root_id"], r["binyan"])
    for f in form_rows:
        lt, lc, rid, b = lex_tier[f["lexeme_id"]]
        fl = set(f["_flags"])
        if fl & FORM_HARD_FLAGS:
            ft = "QUESTIONABLE"
        elif fl & FORM_NOGOLD_FLAGS or f["plene_consistent"] is not True or f["wiktionary_corroborated"] is False:
            ft = "SILVER"
        else:
            ft = "GOLD"
        f["quality_tier"] = worst_tier(lt, ft)
        f["confidence"] = round(lc * (0.5 if fl & FORM_HARD_FLAGS else 1.0), 4)
        f["root_id"] = rid
        f["binyan"] = b

    # ======================================================================
    # assemble DataFrames
    # ======================================================================
    lex_ud = defaultdict(bool)
    for f in form_rows:
        if f["ud_attested"]:
            lex_ud[f["lexeme_id"]] = True
    for r in lex_rows:
        r["ud_attested"] = lex_ud[r["lexeme_id"]]
        C.flag_rows(r["lexeme_id"], "lexeme", sorted(set(r["_flags"])))
        r["quality_flags"] = join_flags(r["_flags"])
    for f in form_rows:
        C.flag_rows(f["form_id"], "form", sorted(set(f["_flags"])), f["_details"])
        f["quality_flags"] = join_flags(f["_flags"])
    lexemes = pd.DataFrame(lex_rows)
    forms = pd.DataFrame(form_rows).drop(columns=["_flags", "_details"])
    forms = add_syncretism(forms)
    lexemes = add_past_completeness(lexemes, forms)

    # same surface form -> incompatible analyses (across lexemes); reported, not an error
    amb = (forms[forms.form_vocalized.notna()].groupby("form_vocalized")
           .agg(n_lex=("lexeme_id", "nunique"), cells=("canonical_cell", lambda x: "|".join(sorted(set(x)))),
                lexemes=("lexeme_id", lambda x: "|".join(sorted(set(x))))).reset_index())
    amb = amb[amb.n_lex > 1]
    for r in amb.itertuples(index=False):
        C.dups.append(dict(duplicate_type="SAME_VOCALIZED_FORM_DIFFERENT_LEXEMES", record_ids=r.lexemes,
                           lemma=r.form_vocalized, details=f"cells {r.cells}"))

    # ---- roots --------------------------------------------------------------
    root_info = {}
    for r in lex_rows:
        rn = r["_root_norm"]
        if not rn or not r["root_id"]:
            continue
        dec = r["_dec"]
        info = root_info.setdefault(rn, dict(root_id=r["root_id"], radicals=split_root_units(rn), root_raw=None,
                                             source_primary="hewiktionary", _src=set(), _giz=set(),
                                             _gizraw=set(), _gizsets=set(), _ss=set()))
        info["_ss"].add(r["shin_sin_resolution"])
        for e in dec["agree_root"]:
            info["_src"].add(e.record_id)
            if info["root_raw"] is None:
                info["root_raw"] = e.extra.get("root_value_raw") or e.root
            if e.kind == "entry" and e.extra.get("gizra_labels"):
                info["_giz"] |= set(e.extra["gizra_labels"].split("+"))
                info["_gizraw"].add(e.extra.get("gizra_raw"))
                info["_gizsets"].add(e.extra["gizra_labels"])
        if info["root_raw"] is None:
            info["root_raw"] = r["root_radicals_dotted"] or rn
    ortho_count = defaultdict(set)
    for rn in root_info:
        ortho_count[orthographic(rn)].add(rn)
    for rn, info in root_info.items():
        if not info["_src"]:
            lx = lexemes[lexemes.root_id == info["root_id"]]
            info["_src"] = set(x for s_ in lx.source_ids for x in s_.split("|") if x.startswith("wikt:"))
        info["source_ids"] = "|".join(sorted(info["_src"])) or "unknown"
        giz = sorted(x for x in info["_giz"] if x != "UNMAPPED")
        info["traditional_class"] = "+".join(giz) or None
        info["traditional_class_raw"] = " ; ".join(sorted(x for x in info["_gizraw"] if x)) or None
        fl = []
        if len(ortho_count[orthographic(rn)]) > 1:
            fl.append("ROOT_SHARES_ORTHOGRAPHY_WITH_OTHER_ROOT")
        if "FORM" in info["_ss"] or "MIXED" in info["_ss"]:
            fl.append("SHIN_SIN_FROM_FORM")
        if "UNRESOLVED" in info["_ss"]:
            fl.append("SHIN_SIN_UNRESOLVED")
        if len(info["_gizsets"]) > 1:
            fl.append("TRADITIONAL_CLASS_VARIES_ACROSS_ENTRIES")
        if "UNMAPPED" in info["_giz"]:
            fl.append("TRADITIONAL_CLASS_UNMAPPED_LABEL")
        if not giz:
            fl.append("TRADITIONAL_CLASS_NOT_STATED")
        if len(info["radicals"]) not in (3, 4):
            fl.append("UNEXPECTED_ROOT_LENGTH")
        info["quality_flags"] = join_flags(fl)
        C.flag_rows(info["root_id"], "root", fl)
        for sid in sorted(info["_src"]):
            C.prov.append(dict(canonical_record_id=info["root_id"], record_type="root", field_supported="root",
                               source_name="hewiktionary", source_url=_wikt_url_from_record(sid, d),
                               source_file=paths.WIKTIONARY_DUMP.name,
                               source_version_or_commit=paths.WIKTIONARY_DUMP_DATE, source_record_identifier=sid,
                               source_line_if_available=None, retrieval_date=C.wk_date,
                               parser_version=PARSER_VERSIONS["wiktionary_parser"],
                               raw_source_value=info["root_raw"]))
    roots = root_rows_from_lexemes(root_info, lexemes)

    lexemes = lexemes.drop(columns=[c for c in lexemes.columns if c.startswith("_")])
    prov = pd.DataFrame(C.prov)
    prov.insert(0, "provenance_id", [f"PV{i:07d}" for i in range(len(prov))])
    val = pd.DataFrame(C.val, columns=["record_id", "record_type", "check_name", "severity", "details"])
    val.insert(0, "check_id", [f"CK{i:07d}" for i in range(len(val))])
    return dict(roots=roots, lexemes=lexemes, forms=forms, provenance=prov, validation=val,
                conflicts=pd.DataFrame(C.conflicts), duplicates=pd.DataFrame(C.dups),
                missing=pd.DataFrame(C.missing), gaps=d["gaps"], interim=d)


def _wikt_url_from_record(sid: str, d) -> str:
    import re
    m = re.match(r"wikt:\w+:p(\d+):r(\d+)", sid)
    if not m:
        return "https://he.wiktionary.org/"
    return f"https://he.wiktionary.org/w/index.php?curid={m.group(1)}&oldid={m.group(2)}"
