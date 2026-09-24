"""Hebrew Wiktionary dump parsing (structured templates only, no rendered HTML).

Structured information used:
  * {{ניתוח דקדוקי לפועל}} on entry pages: כתיב מלא (plene spelling),
    שורש וגזרה (root via {{שרש3|..}}/{{שרש4|..}}/{{שרש|..}} + [[גזרת ...]] links),
    בניין.  The level-2 heading of the section is the vocalized headword.
  * {{שורש|...}} root-link templates in the same entry section.
  * {{ת|אנגלית|...}} translation templates (English gloss).
  * {{נטיות פעלים}} on root pages ("X (שורש)"): per binyan 3MS past,
    present, future, imperative and infinitive (vocalized).
  * {{נטיות פעל בבנין}} full per-binyan conjugation tables (appendix pages).
"""
from __future__ import annotations

import bz2
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass

from .hebrew import LETTERS, SHIN_DOT, SIN_DOT, normalize_finals, normalize_nfc

HEB_WORD = re.compile(r"[א-ת][א-ת֑-ׇ]*")


# --------------------------------------------------------------------------
# Dump iteration
# --------------------------------------------------------------------------
@dataclass
class Page:
    title: str
    ns: int
    page_id: int
    rev_id: int
    timestamp: str
    text: str


def iter_dump(path: str):
    """Stream pages from a MediaWiki pages-articles XML dump (.bz2 or plain)."""
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rb") as fh:
        ctx = ET.iterparse(fh, events=("end",))
        for _, elem in ctx:
            if not elem.tag.endswith("}page"):
                continue
            ns_uri = elem.tag[: -len("page")]
            title = elem.findtext(f"{ns_uri}title")
            ns = int(elem.findtext(f"{ns_uri}ns"))
            pid = int(elem.findtext(f"{ns_uri}id"))
            rev = elem.find(f"{ns_uri}revision")
            rid = int(rev.findtext(f"{ns_uri}id"))
            ts = rev.findtext(f"{ns_uri}timestamp")
            text = rev.findtext(f"{ns_uri}text") or ""
            yield Page(title, ns, pid, rid, ts, text)
            elem.clear()


# --------------------------------------------------------------------------
# Minimal balanced template parser
# --------------------------------------------------------------------------
@dataclass
class Template:
    name: str
    params: dict      # name -> raw value (positional params keyed "1", "2", ...)
    raw: str
    start: int
    end: int


def _split_top_level(s: str, sep: str = "|") -> list[str]:
    parts, buf, depth_t, depth_l, i = [], [], 0, 0, 0
    while i < len(s):
        two = s[i:i + 2]
        if two == "{{":
            depth_t += 1; buf.append(two); i += 2; continue
        if two == "}}" and depth_t:
            depth_t -= 1; buf.append(two); i += 2; continue
        if two == "[[":
            depth_l += 1; buf.append(two); i += 2; continue
        if two == "]]" and depth_l:
            depth_l -= 1; buf.append(two); i += 2; continue
        if s[i] == sep and depth_t == 0 and depth_l == 0:
            parts.append("".join(buf)); buf = []; i += 1; continue
        buf.append(s[i]); i += 1
    parts.append("".join(buf))
    return parts


def iter_templates(text: str):
    """Yield every template (including nested ones) with parsed params."""
    stack = []
    i = 0
    n = len(text)
    while i < n - 1:
        two = text[i:i + 2]
        if two == "{{":
            stack.append(i); i += 2; continue
        if two == "}}" and stack:
            start = stack.pop()
            inner = text[start + 2:i]
            parts = _split_top_level(inner)
            name = parts[0].strip()
            params, pos = {}, 1
            for p in parts[1:]:
                k, eq, v = p.partition("=")
                if eq and "{{" not in k and "[[" not in k:
                    params[k.strip()] = v.strip()
                else:
                    params[str(pos)] = p.strip(); pos += 1
            yield Template(name, params, text[start:i + 2], start, i + 2)
            i += 2; continue
        i += 1


def find_templates(text: str, names: set[str]) -> list[Template]:
    return [t for t in iter_templates(text) if t.name in names]


# --------------------------------------------------------------------------
# Value cleaning
# --------------------------------------------------------------------------
def strip_markup(value: str) -> str:
    """[[a#b|c]] -> c ; [[a]] -> a ; remove comments, <ref>, bold/italic."""
    v = re.sub(r"<!--.*?-->", "", value, flags=re.S)
    v = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", v, flags=re.S)
    v = re.sub(r"<[^>]+>", " ", v)
    v = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", v)
    v = v.replace("'''", "").replace("''", "")
    return v.strip()


def hebrew_words(value: str) -> list[str]:
    return [normalize_nfc(w) for w in HEB_WORD.findall(strip_markup(value))]


def table_forms(value: str) -> tuple[list[str], list[str]]:
    """Vocalized word forms in a conjugation-table cell, plus annotation flags.
    Templates ({{הבהרה}}, {{מקור}}), <small> notes and unvocalized words
    (editorial remarks such as 'או', 'ארכאי') are not forms."""
    from .hebrew import is_hebrew_mark
    flags = []
    v = value
    from .hebrew import strip_niqqud
    if re.fullmatch(r"\W*אין\W*", strip_niqqud(strip_markup(re.sub(r"\{\{.*?\}\}", "", v, flags=re.S))).strip()):
        return [], ["TABLE_ASSERTS_ABSENT"]
    if re.search(r"\{\{\s*(הבהרה|מקור|דרוש מקור)", v) or "?" in strip_markup(v):
        flags.append("TABLE_VALUE_DOUBTED")
    if re.search(r"ארכאי|במקרא|מקראי|נדיר", v):
        flags.append("TABLE_VALUE_ARCHAIC_OR_RARE_NOTE")
    v = re.sub(r"<small>.*?</small>", " ", v, flags=re.S)
    v = re.sub(r"<!--.*?-->", " ", v, flags=re.S)
    prev = None
    while prev != v:  # remove innermost templates repeatedly
        prev = v
        v = re.sub(r"\{\{[^{}]*\}\}", " ", v)
    words = hebrew_words(v)
    forms = [w for w in words if any(is_hebrew_mark(c) for c in w)]
    if len(words) == 1 and not forms and words[0] not in ("אין", "או"):
        flags.append("TABLE_VALUE_UNVOCALIZED")
    if len(forms) > 1:
        flags.append("TABLE_VALUE_MULTIPLE_FORMS")
    return forms, flags


def normalize_radical(tok: str) -> tuple[str | None, str | None, list[str]]:
    """'שׁ' -> ('שׁ', None) ; 'ח ג' -> ('ח', 'ג') (homonym index) ; 'ך' -> 'כ'.
    Only shin/sin dots are kept on radicals; other marks are dropped + flagged."""
    flags = []
    tok = normalize_nfc(strip_markup(tok)).strip()
    parts = tok.split()
    if not parts:
        return None, None, ["RADICAL_EMPTY"]
    head, homonym = parts[0], (" ".join(parts[1:]) or None)
    base = [c for c in head if c in LETTERS]
    if len(base) != 1:
        return None, homonym, ["RADICAL_MALFORMED"]
    marks = [c for c in head if c not in LETTERS]
    letter = normalize_finals(base[0])
    kept = ""
    for m in marks:
        if letter == "ש" and m in (SHIN_DOT, SIN_DOT):
            kept = m
        else:
            flags.append("RADICAL_HAD_OTHER_MARK")
    return normalize_nfc(letter + kept), homonym, flags


def split_radicals(root: str | None) -> list[str]:
    return root.split("-") if root else []


def parse_root_value(value: str) -> tuple[list[str] | None, str, list[str], str | None]:
    """Extract radicals from a 'שורש וגזרה' / 'שורש' value.

    Returns (radicals or None, method, flags, homonym_index).  Radicals are
    final-normalized letters; shin/sin dots are kept when the source gives them."""
    tpls = [t for t in iter_templates(value) if re.fullmatch(r"שרש\d?", t.name)]
    cands = []
    for t in tpls:
        if t.name in ("שרש3", "שרש4", "שרש5"):
            n = int(t.name[-1])
            toks = [t.params.get(str(k), "") for k in range(1, n + 1)]
            extra = t.params.get(str(n + 1))
            cands.append(("tpl_" + t.name, toks, extra.strip() if extra else None))
        elif sum(k.isdigit() for k in t.params) >= 3 and all(
                len(_letter_units(t.params[k])) == 1 for k in t.params if k.isdigit()):
            # {{שרש|ס|כ|ך}}: one letter per positional parameter
            cands.append(("tpl_שרש_letters", [t.params[k] for k in sorted((k for k in t.params if k.isdigit()), key=int)], None))
        else:  # {{שרש|סקל|ס־ק־ל}}
            src = t.params.get("2") or t.params.get("1") or ""
            parts = [x for x in re.split(r"[־\-–]+", normalize_nfc(strip_markup(src))) if x.strip()]
            if len(parts) == 1:
                parts = [u[0] + u[1] for u in _letter_units(parts[0])]
            cands.append(("tpl_שרש", parts, None))
    if not cands:
        txt = strip_markup(re.sub(r"\[\[גזרת[^\]]*\]\]", "", value))
        m = re.search(r"([\u05D0-\u05EA][\u05C1\u05C2]?(?:\s*[־\-–]\s*[\u05D0-\u05EA][\u05C1\u05C2]?){2,4})", txt)
        if m:
            cands.append(("text_dashed", re.split(r"\s*[־\-–]\s*", m.group(1)), None))
    if not cands:
        return None, "none", ["ROOT_UNPARSEABLE" if value.strip() else "ROOT_MISSING"], None
    parsed = []
    for method, toks, homonym in cands:
        rads, flags = [], []
        for tok in toks:
            r, h, f = normalize_radical(tok)
            flags += f
            rads.append(r)
            homonym = homonym or h
        parsed.append((method, rads, flags, homonym))
    distinct = {tuple(strip_shin_sin(r) for r in p[1]) for p in parsed}
    if len(distinct) > 1:
        return None, "conflict", ["ROOT_MULTIPLE_IN_ENTRY"], None
    method, rads, flags, homonym = parsed[0]
    if any(r is None for r in rads):
        return None, method, sorted(set(flags)) or ["ROOT_MALFORMED"], homonym
    if not 3 <= len(rads) <= 5:
        return None, method, sorted(set(flags + ["ROOT_UNEXPECTED_LENGTH"])), homonym
    if homonym:
        flags.append("ROOT_HOMONYM_INDEX")
    return rads, method, sorted(set(flags)), homonym


def parse_dashed_root(root_raw: str) -> list[str] | None:
    """'א־ה־ב' (maqaf/hyphen separated) -> ['א','ה','ב'] ; None if malformed."""
    toks = [x for x in re.split(r"\s*[־\-–]\s*", normalize_nfc(strip_markup(root_raw))) if x.strip()]
    rads = [normalize_radical(t)[0] for t in toks]
    if not 3 <= len(rads) <= 5 or any(r is None for r in rads):
        return None
    return rads


def _letter_units(s: str) -> list[tuple[str, str]]:
    out = []
    for ch in normalize_nfc(s):
        if ch in LETTERS:
            out.append((ch, ""))
        elif ch in (SHIN_DOT, SIN_DOT) and out:
            out[-1] = (out[-1][0], ch)
    return out


def strip_shin_sin(r: str | None) -> str | None:
    return None if r is None else r.replace(SHIN_DOT, "").replace(SIN_DOT, "")


GIZRA_MAP = [
    # (regex over the normalized gizra text, label).  Order matters.
    (r"שלמים", "SHLEMIM"),
    (r"חפי\"?צ", "PE_YOD_TSADI"),
    (r"חפ\"?[נן]|חסרי פ\"?נ", "PE_NUN"),
    (r"נ(?:חי )?פ\"?[יו]\"?[/\\]?[יו]?|נפי\"?ו|נחי פי\"?ו|נחי פ\"?י", "PE_YOD_WAW"),
    (r"נפ\"?א", "PE_ALEF"),
    (r"נ(?:חי )?ע\"?[ו]\"?[/\\]?[יו]?|נעי\"?ו|נחי ע\"?ו", "AYIN_WAW_YOD"),
    (r"כפולים|ע\"?ע", "AYIN_AYIN"),
    (r"נל\"?[יה]\"?[/\\]?[יה]?|נחי ל\"?ה", "LAMED_YOD_HE"),
    (r"נל\"?א", "LAMED_ALEF"),
    (r"מרו?בעים", "QUADRILITERAL"),
    (r"פ' גרונית|פ\"?ג", "PE_GUTTURAL"),
    (r"ירי", "YERI"),
    (r"מחומשים", "QUINQUILITERAL"),
    (r"משושים", "SEXTILITERAL"),
    (r"ל\"?ג", "LAMED_GUTTURAL"),
    (r"ע\"?ג", "AYIN_GUTTURAL"),
    (r"ע\"?ו", "AYIN_WAW_YOD"),
]


def parse_gizra(value: str) -> tuple[list[str], list[str]]:
    """Return (raw gizra strings, normalized labels).  Labels are only what the
    source states; nothing is inferred from the radicals here."""
    v = normalize_nfc(value).replace("&quot;", '"').replace("״", '"')
    raws = []
    for m in re.finditer(r"\[\[(גזרת[^\]|#]*)", v):
        raws.append(re.sub(r"\s+", " ", m.group(1)).strip())
    if not raws:
        # plain-text mentions after the root template, e.g. "..., שלמים"
        rest = re.sub(r"\{\{[^}]*\}\}", "", v)
        rest = strip_markup(rest).strip(" ,;")
        if rest:
            raws.append(rest)
    labels = []
    for r in raws:
        body = r.replace("גזרת", "").strip()
        for pat, lab in GIZRA_MAP:
            if re.search(pat, body):
                labels.append(lab)
                break
        else:
            labels.append("UNMAPPED")
    return raws, labels


def english_glosses(section_text: str) -> list[str]:
    out = []
    for t in find_templates(section_text, {"ת"}):
        if t.params.get("1", "").strip() == "אנגלית":
            for k in sorted((k for k in t.params if k.isdigit() and k != "1"), key=int):
                g = strip_markup(t.params[k]).strip()
                if g:
                    out.append(g)
    return out


def split_level2_sections(text: str) -> list[tuple[str, str, int]]:
    """[(heading, body, offset)] for level-2 '==X==' sections."""
    out = []
    ms = list(re.finditer(r"^==(?!=)(.+?)(?<!=)==\s*$", text, flags=re.M))
    for k, m in enumerate(ms):
        end = ms[k + 1].start() if k + 1 < len(ms) else len(text)
        out.append((m.group(1).strip(), text[m.end():end], m.start()))
    return out


def clean_headword(heading: str) -> tuple[str | None, list[str]]:
    words = hebrew_words(heading)
    if not words:
        return None, ["HEADWORD_UNPARSEABLE"]
    flags = [] if len(words) == 1 and strip_markup(heading).strip() == words[0] else ["HEADWORD_HAS_EXTRA_TEXT"]
    return words[0], flags


def _ktiv_flags(raw: str) -> list[str]:
    words = hebrew_words(raw)
    f = []
    if len(words) > 1:
        f.append("KTIV_MALE_MULTIPLE_WORDS")
    if re.search(r"[A-Za-z]", raw or ""):
        f.append("KTIV_MALE_NON_HEBREW")
    if words and any(is_mark(c) for c in words[0]):
        f.append("KTIV_MALE_VOCALIZED")
    return f


def is_mark(c: str) -> bool:
    return 0x0591 <= ord(c) <= 0x05C7


def extract_verb_entries(page: Page) -> list[dict]:
    rows = []
    for heading, body, offset in split_level2_sections(page.text):
        tpls = find_templates(body, {"ניתוח דקדוקי לפועל"})
        if not tpls:
            continue
        hw, hflags = clean_headword(heading)
        alts = [w for w in hebrew_words(re.sub(r"\{\{[^}]*\}\}", " ", heading))[1:] if w not in ("גם", "או")]
        for k, t in enumerate(tpls):
            p = {normalize_nfc(key): val for key, val in t.params.items()}
            root_val = p.get("שורש וגזרה", p.get("שורש", ""))
            radicals, rmethod, rflags, homonym = parse_root_value(root_val)
            giz_raw, giz = parse_gizra(root_val)
            if "גזרה" in p:
                r2, g2 = parse_gizra(p["גזרה"])
                giz_raw += r2; giz += g2
            root_links = []
            for rt in find_templates(body, {"שורש"}):
                val = rt.params.get("1", "")
                letters = [c for c in normalize_finals(normalize_nfc(strip_markup(val))) if c in LETTERS]
                if letters:
                    root_links.append("".join(letters))
            rows.append(dict(
                page_title=normalize_nfc(page.title), page_id=page.page_id,
                rev_id=page.rev_id, rev_timestamp=page.timestamp,
                section_heading_raw=heading, section_offset=offset, template_index=k,
                headword=hw,
                headword_alternates="|".join(alts) or None,
                ktiv_male=(hebrew_words(p.get("כתיב מלא", "")) or [None])[0],
                ktiv_male_raw=p.get("כתיב מלא"),
                binyan_raw=p.get("בניין"),
                root_value_raw=root_val,
                root_radicals="-".join(radicals) if radicals else None,
                root_homonym_index=homonym,
                root_parse_method=rmethod,
                gizra_raw=" ; ".join(giz_raw) if giz_raw else None,
                gizra_labels="+".join(sorted(set(giz))) if giz else None,
                root_links="|".join(sorted(set(root_links))) if root_links else None,
                gloss_en="; ".join(english_glosses(body)) or None,
                registers="|".join(sorted({strip_markup(r.params.get("1", "")) for r in find_templates(body, {"רובד"})})) or None,
                template_raw=t.raw,
                flags="|".join(hflags + rflags + _ktiv_flags(p.get("כתיב מלא", ""))) or None,
            ))
    return rows


def extract_conj_tables(page: Page) -> list[dict]:
    rows = []
    from .morphology import TABLE_BINYAN_PREFIX, TABLE_SLOT_TO_CELL
    for k, t in enumerate(find_templates(page.text, {"נטיות פעלים"})):
        root_raw = t.params.get("שורש", "")
        letters = parse_dashed_root(root_raw)
        ok = letters is not None
        for key, val in t.params.items():
            key = normalize_nfc(key).strip()
            for pref, b in TABLE_BINYAN_PREFIX.items():
                if key.startswith(pref + " "):
                    slot = key[len(pref) + 1:]
                    if slot not in TABLE_SLOT_TO_CELL:
                        continue  # e.g. 'קל הווה פעול' (passive participle)
                    forms, vflags = table_forms(val)
                    rows.append(dict(
                        page_title=normalize_nfc(page.title), page_id=page.page_id,
                        rev_id=page.rev_id, rev_timestamp=page.timestamp, table_index=k,
                        root_raw=root_raw, root_radicals="-".join(letters) if ok else None,
                        binyan=b, binyan_raw=pref, slot_raw=slot,
                        cell=TABLE_SLOT_TO_CELL[slot], value_raw=val,
                        forms="|".join(forms) if forms else None,
                        n_forms=len(forms), flags="|".join(vflags) or None,
                    ))
    return validate_table_roots(rows, page.title)


def title_root(title: str) -> list[str] | None:
    """'סגר (שורש)' / 'ספר א (שורש)' -> ['ס','ג','ר'] (root-page naming convention)."""
    m = re.fullmatch(r"(.+?)\s*\(שורש\)", normalize_nfc(title).strip())
    if not m:
        return None
    word = m.group(1).split()[0]
    rads = [u[0] + u[1] for u in _letter_units(word)]
    rads = [normalize_finals(r[0]) + r[1:] for r in rads]
    return rads if 3 <= len(rads) <= 5 else None


def validate_table_roots(rows: list[dict], page_title: str) -> list[dict]:
    """Check each table's root parameter against its own PST.3MSG forms (a root
    is consistent if its stable radicals are a subsequence of >= 50% of the
    forms' letters) and the page-title root.  Sets root_radicals to the parameter root if consistent;
    else to the title root if *it* is consistent (flag
    TABLE_ROOT_PARAM_REPLACED_BY_TITLE); else None (TABLE_ROOT_INCONSISTENT)."""
    from .hebrew import extract_consonantal_skeleton
    from .roots import is_subsequence, stable_radicals
    tr = title_root(page_title)
    by_table = defaultdict(list)
    for r in rows:
        by_table[r["table_index"]].append(r)

    def consistent(rads, forms):
        if not rads or not forms:
            return None
        st = stable_radicals(rads)
        ok = sum(is_subsequence(st, extract_consonantal_skeleton(f)) for f in forms)
        return ok / len(forms) >= 0.5  # a few legitimately irregular forms are tolerated
    for k, rs in by_table.items():
        forms = [f for r in rs if r["cell"] == "PST.3MSG" and r["forms"] for f in r["forms"].split("|")]
        param = rs[0]["root_radicals"].split("-") if rs[0]["root_radicals"] else None
        ok_param = consistent(param, forms)
        ok_title = consistent(tr, forms)
        new, flag = rs[0]["root_radicals"], None
        if ok_param is False:
            if ok_title:
                new, flag = "-".join(tr), "TABLE_ROOT_PARAM_REPLACED_BY_TITLE"
            else:
                new, flag = None, "TABLE_ROOT_INCONSISTENT"
        elif param is None and ok_title:
            new, flag = "-".join(tr), "TABLE_ROOT_FROM_TITLE"
        for r in rs:
            r["root_param_radicals"] = r["root_radicals"]
            r["root_radicals"] = new
            r["title_root"] = "-".join(tr) if tr else None
            if flag:
                r["flags"] = "|".join(x for x in [r.get("flags"), flag] if x)
    return rows


def extract_full_tables(page: Page) -> list[dict]:
    from .morphology import FULL_TABLE_PARAM_TO_CELL, normalize_binyan
    rows = []
    for k, t in enumerate(find_templates(page.text, {"נטיות פעל בבנין"})):
        root_raw = t.params.get("שרש", t.params.get("שורש", ""))
        letters = parse_dashed_root(root_raw)
        ok = letters is not None
        braw = t.params.get("בנין", t.params.get("בניין", ""))
        b, bflags = normalize_binyan(braw)
        for key, val in t.params.items():
            cell = FULL_TABLE_PARAM_TO_CELL.get(normalize_nfc(key).strip())
            if not cell:
                continue
            forms, vflags = table_forms(val)
            bflags = bflags + vflags
            rows.append(dict(
                page_title=normalize_nfc(page.title), page_id=page.page_id, rev_id=page.rev_id,
                rev_timestamp=page.timestamp, table_index=k, root_raw=root_raw,
                root_radicals="-".join(letters) if ok else None, binyan=b, binyan_raw=braw,
                param=key, cell=cell, value_raw=val,
                forms="|".join(forms) if forms else None, n_forms=len(forms),
                flags="|".join(bflags) or None,
            ))
    return rows
