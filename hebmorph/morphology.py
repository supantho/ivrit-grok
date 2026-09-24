"""Binyan normalization, UniMorph feature normalization, paradigm cells."""
from __future__ import annotations

import re

from .hebrew import normalize_nfc, strip_niqqud

BINYANIM = ["PAAL", "NIFAL", "PIEL", "PUAL", "HIFIL", "HUFAL", "HITPAEL"]

PAST_CELLS = ["PST.1SG", "PST.2MSG", "PST.2FSG", "PST.3MSG", "PST.3FSG",
              "PST.1PL", "PST.2MPL", "PST.2FPL", "PST.3PL"]
CITATION_CELL = "PST.3MSG"

# --------------------------------------------------------------------------
# Binyan labels
# --------------------------------------------------------------------------
# Keys are niqqud-stripped labels unless the vocalization disambiguates.
_VOCALIZED_BINYAN = {
    "פָּעַל": "PAAL", "פָעַל": "PAAL", "קַל": "PAAL",
    "נִפְעַל": "NIFAL",
    "פִּעֵל": "PIEL", "פִעֵל": "PIEL",
    "פֻּעַל": "PUAL", "פֻעַל": "PUAL", "פּוּעַל": "PUAL",
    "הִפְעִיל": "HIFIL",
    "הֻפְעַל": "HUFAL", "הָפְעַל": "HUFAL", "הוּפְעַל": "HUFAL",
    "הִתְפַּעֵל": "HITPAEL", "הִתְפַעֵל": "HITPAEL",
}
_UNVOCALIZED_BINYAN = {
    "קל": "PAAL", "פעל קל": "PAAL",
    "נפעל": "NIFAL",
    "פיעל": "PIEL",
    "פועל": "PUAL", "פעל סביל": "PUAL",
    "הפעיל": "HIFIL",
    "הופעל": "HUFAL", "הפעל": "HUFAL",
    "התפעל": "HITPAEL",
}
# Unvocalized labels that are genuinely ambiguous between binyanim.
_AMBIGUOUS_UNVOCALIZED = {"פעל": ("PAAL", "PUAL", "PIEL")}


def _label_tokens(raw: str) -> list[str]:
    """Pull candidate label strings out of a wikitext value such as
    '[[פעל#פֻּעַל|פֻּעַל]]', 'פָּעַל (קל)', '[[קל]]'."""
    s = normalize_nfc(raw)
    s = re.sub(r"<!--.*?-->", " ", s, flags=re.S)
    toks = []
    for m in re.finditer(r"\[\[([^\]]*)\]\]", s):
        inner = m.group(1)
        for part in re.split(r"[|#]", inner):
            toks.append(part)
    s2 = re.sub(r"\[\[[^\]]*\]\]", " ", s)
    toks += re.split(r"[()\[\],/;]| או ", s2)
    out = []
    for t in toks:
        t = t.strip().strip("'\"").strip()
        if t:
            out.append(t)
    return out


def normalize_binyan(raw: str | None) -> tuple[str | None, list[str]]:
    """Return (normalized binyan or None, flags).  Never guesses: labels that
    map to more than one binyan, or to none, return None with a flag."""
    if raw is None or not raw.strip():
        return None, ["BINYAN_MISSING"]
    found = set()
    ambiguous = False
    for tok in _label_tokens(raw):
        if tok in _VOCALIZED_BINYAN:
            found.add(_VOCALIZED_BINYAN[tok])
            continue
        bare = strip_niqqud(tok).strip()
        if tok == bare:  # unvocalized token
            if bare in _UNVOCALIZED_BINYAN:
                found.add(_UNVOCALIZED_BINYAN[bare])
            elif bare in _AMBIGUOUS_UNVOCALIZED:
                ambiguous = True
        else:  # vocalized but not in table: try the stripped form, if unambiguous
            if bare in _UNVOCALIZED_BINYAN and bare not in _AMBIGUOUS_UNVOCALIZED:
                found.add(_UNVOCALIZED_BINYAN[bare])
    if len(found) == 1:
        return found.pop(), []
    if len(found) > 1:
        return None, ["BINYAN_LABEL_CONFLICT"]
    if ambiguous:
        return None, ["BINYAN_LABEL_AMBIGUOUS"]
    return None, ["BINYAN_LABEL_UNRECOGNIZED"]


# Wiktionary conjugation-table prefixes ({{נטיות פעלים}}).
TABLE_BINYAN_PREFIX = {"קל": "PAAL", "נפעל": "NIFAL", "פיעל": "PIEL", "פועל": "PUAL",
                       "הפעיל": "HIFIL", "הופעל": "HUFAL", "התפעל": "HITPAEL"}
TABLE_SLOT_TO_CELL = {"עבר": "PST.3MSG", "הווה": "PRS.MSG", "עתיד": "FUT.3MSG",
                      "ציווי": "IMP.2MSG", "שם הפועל": "NFIN"}

# {{נטיות פעל בבנין}} full-paradigm parameters -> canonical cells
FULL_TABLE_PARAM_TO_CELL = {
    "מדבר_עבר": "PST.1SG", "נוכח_עבר": "PST.2MSG", "נוכחת_עבר": "PST.2FSG",
    "נסתר_עבר": "PST.3MSG", "נסתרת_עבר": "PST.3FSG", "מדברים_עבר": "PST.1PL",
    "נוכחים_עבר": "PST.2MPL", "נוכחות_עבר": "PST.2FPL", "נסתרים_עבר": "PST.3PL",
    "מדבר_הוה": "PRS.MSG", "מדברת_הוה": "PRS.FSG", "מדברים_הוה": "PRS.MPL",
    "מדברות_הוה": "PRS.FPL",
    "מדבר_עתיד": "FUT.1SG", "נוכח_עתיד": "FUT.2MSG", "נוכחת_עתיד": "FUT.2FSG",
    "נסתר_עתיד": "FUT.3MSG", "נסתרת_עתיד": "FUT.3FSG", "מדברים_עתיד": "FUT.1PL",
    "נוכחים_עתיד": "FUT.2MPL", "נוכחות_עתיד": "FUT.2FPL", "נסתרים_עתיד": "FUT.3MPL",
    "נוכח_צווי": "IMP.2MSG", "נוכחת_צווי": "IMP.2FSG", "נוכחים_צווי": "IMP.2MPL",
    "נוכחות_צווי": "IMP.2FPL", "מקור": "NFIN",
}

# --------------------------------------------------------------------------
# UniMorph features
# --------------------------------------------------------------------------
_TENSES = {"PST", "PRS", "FUT"}
_MOODS = {"IMP"}
_PERSONS = {"1", "2", "3"}
_NUMBERS = {"SG", "PL"}
_GENDERS = {"MASC": "M", "FEM": "F"}
NA = "NA"


class TagError(ValueError):
    pass


def parse_unimorph_tag(tag: str) -> dict:
    """Parse a UniMorph (v2/3 style, as in unimorph/heb ``heb``/``heb_voc``)
    verb tag into normalized features and a canonical cell name.

    Features the source does not mark are ``NA`` -- nothing is invented
    (e.g. PST.1SG has gender NA; present participles have person NA)."""
    tag = tag.strip()
    if tag == "V.MSDR":
        return dict(pos="V.MSDR", tense=NA, aspect=NA, mood=NA, person=NA, number=NA,
                    gender=NA, voice=NA, finiteness="NFIN", canonical_cell="MSDR")
    parts = tag.split(";")
    if parts[0] != "V":
        raise TagError(f"not a verb tag: {tag}")
    feats = dict(pos="V", tense=NA, aspect=NA, mood=NA, person=NA, number=NA,
                 gender=NA, voice=NA, finiteness="FIN")
    seen = set()
    for p in parts[1:]:
        if p in seen:
            raise TagError(f"repeated feature {p} in {tag}")
        seen.add(p)
        if p in _TENSES:
            feats["tense"] = p
        elif p in _MOODS:
            feats["mood"] = p
        elif p in _PERSONS:
            feats["person"] = p
        elif p in _NUMBERS:
            feats["number"] = p
        elif p in _GENDERS:
            feats["gender"] = p
        elif p == "NFIN":
            feats["finiteness"] = "NFIN"
        else:
            raise TagError(f"unknown feature {p} in {tag}")
    if feats["tense"] != NA and feats["mood"] == NA:
        feats["mood"] = "IND"
    feats["canonical_cell"] = cell_name(feats)
    return feats


def cell_name(f: dict) -> str:
    if f["pos"] == "V.MSDR":
        return "MSDR"
    if f["finiteness"] == "NFIN":
        return "NFIN"
    head = f["tense"] if f["mood"] != "IMP" else "IMP"
    p = "" if f["person"] == NA else f["person"]
    g = _GENDERS.get(f["gender"], "")
    n = "" if f["number"] == NA else f["number"]
    return f"{head}.{p}{g}{n}"


def feature_violations(f: dict) -> list[str]:
    """Impossible or unexpected feature combinations for Modern Hebrew verbs."""
    v = []
    if f["pos"] == "V.MSDR" or f["finiteness"] == "NFIN":
        if any(f[k] != NA for k in ("tense", "person", "number", "gender")):
            v.append("NONFINITE_WITH_AGREEMENT")
        return v
    if f["mood"] == "IMP":
        if f["tense"] != NA:
            v.append("IMP_WITH_TENSE")
        if f["person"] != "2":
            v.append("IMP_NOT_2ND_PERSON")
    elif f["tense"] == NA:
        v.append("FINITE_WITHOUT_TENSE")
    if f["tense"] == "PRS" and f["person"] != NA:
        v.append("PRS_WITH_PERSON")
    if f["tense"] in ("PST", "FUT") and f["person"] == NA:
        v.append("PERSON_MISSING")
    if f["number"] == NA:
        v.append("NUMBER_MISSING")
    if f["person"] == "1" and f["gender"] != NA:
        v.append("1ST_PERSON_WITH_GENDER")
    if f["tense"] == "PST" and f["person"] == "3" and f["number"] == "PL" and f["gender"] != NA:
        v.append("PST_3PL_WITH_GENDER")
    if f["person"] in ("2",) and f["gender"] == NA:
        v.append("2ND_PERSON_WITHOUT_GENDER")
    if f["tense"] == "PRS" and f["gender"] == NA:
        v.append("PRS_WITHOUT_GENDER")
    return v


def target_feature_string(cell: str) -> str:
    """'PST.2MSG' -> 'PST,2,SG,MASC' ; 'PST.1SG' -> 'PST,1,SG' ; 'PST.3PL' -> 'PST,3,PL'."""
    head, _, rest = cell.partition(".")
    if not rest:
        return head
    m = re.fullmatch(r"([123]?)([MF]?)(SG|PL)", rest)
    if not m:
        raise ValueError(cell)
    out = [head]
    if m.group(1):
        out.append(m.group(1))
    out.append(m.group(3))
    if m.group(2):
        out.append({"M": "MASC", "F": "FEM"}[m.group(2)])
    return ",".join(out)


# --------------------------------------------------------------------------
# Binyan from the vocalized paradigm shape (VALIDATION SIGNAL ONLY)
# --------------------------------------------------------------------------
def _first_letter_and_points(s: str) -> tuple[str, str, str]:
    """(first letter, its points, remainder) of an NFC vocalized string."""
    s = normalize_nfc(s)
    i = 1
    while i < len(s) and 0x0591 <= ord(s[i]) <= 0x05C7:
        i += 1
    return s[0], s[1:i], s[i:]


def binyan_from_paradigm_shape(cells: dict[str, str]) -> tuple[str | None, str]:
    """Infer a binyan from vocalized PRS.MSG, NFIN and PST.3MSG forms using
    transparent prefix/vowel rules.  Returns (binyan or None, rule).

    This is a heuristic used only to *check* source annotations; it is never
    written into the binyan column."""
    prs = cells.get("PRS.MSG")
    inf = cells.get("NFIN")
    pst = cells.get("PST.3MSG")
    if not prs or not pst:
        return None, "missing_prs_or_pst"
    l1, p1, rest = _first_letter_and_points(prs)
    # Participles with a מ-prefix
    if l1 == "מ":
        if "ִ" in p1:  # hiriq: מִתְ-, מִשְׁתַּ-, מִסְתַּ-, מִזְדַּ-, מִצְטַ-, מִדַּ-, מִטַּ-
            return "HITPAEL", "prs_mi"
        if p1 and set(p1) & {"ַ", "ֵ", "ֶ"} and not pst.startswith("מ"):
            return "HIFIL", "prs_ma_me"
        if "ֻ" in p1 or rest.startswith("וּ") or ("ָ" in p1 and inf is None):
            return "HUFAL", "prs_mu"
        if p1 and set(p1) & {"ְ", "ֲ", "ֱ"}:
            l2, p2, _ = _first_letter_and_points(rest)
            if l2 == "ו" and p2 in ("ּ",):
                return "PUAL", "prs_me_u"
            if set(p2) & {"ֻ", "ֹ"}:
                return "PUAL" if inf is None else "PIEL", "prs_me_u_vowel"
            if inf is None:
                return "PUAL", "prs_me_no_inf"
            return "PIEL", "prs_me"
    if inf is not None:
        il, ip, irest = _first_letter_and_points(inf)
        if il == "ל" and "ְ" in ip and irest.startswith("ה"):
            hl, hp, hrest = _first_letter_and_points(irest)
            if "ַ" in hp:
                return "HIFIL", "inf_leha"
            if l1 == "נ":
                return "NIFAL", "inf_lehi_prs_n"
    if l1 == "נ" and pst.startswith("נ") and "ֹ" not in (prs[1:3]) and not prs.startswith("נוֹ"):
        if inf is not None and inf.startswith("לְה"):
            return "NIFAL", "prs_n_inf_leh"
        return None, "prs_n_ambiguous"
    if inf is not None and inf.startswith("לְ") and not inf.startswith("לְה"):
        return None, "inf_le_without_prs_me"
    if l1 != "מ" or pst.startswith("מ"):
        if inf is None or not inf.startswith("לְה"):
            return "PAAL", "prs_no_prefix"
    return None, "no_rule"
