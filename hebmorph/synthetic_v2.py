"""Synthetic v2: Hebrew-like verb paradigms with RULE-GOVERNED WEAK ROOTS.

Purpose: test whether grokking appears once outputs stop being additive. In v1 every output
letter depended on the root alone or on (template, cell) alone. In v2, weak roots trigger
rewrite rules whose effect depends jointly on the root's letters x the template x the cell.

Representation: a template is a sequence of UNITS (base, marks); base is a slot '1','2','3'
(a radical) or a literal Hebrew letter; marks are vowel points / dagesh. Weak rules rewrite the
unit list before radicals are substituted. Rules (Hebrew-inspired, simplified, deterministic):

  PE_NUN   r1=נ : a vowelless (shva) non-initial slot 1 assimilates: deleted, dagesh on slot 2
  PE_YOD   r1=י : a vowelless non-initial slot 1 is deleted; after a binyan prefix (NIFAL, HIFIL,
                  HUFAL) the prefix vowel becomes holam-waw (וֹ), otherwise the prefix vowel -> tsere
  LAMED_HE r3=ה : word-final   -> ה, slot-2 vowel qamats (past) / segol (other tenses)
                  before consonant suffix -> י, slot-2 vowel hiriq (active past) / tsere (other
                                     past) / segol (future, imperative)
                  before vowel suffix -> deleted; its vowel moves to slot 2
                                     (past 3FSG exception: -> ת, slot-2 vowel shva)
  HOLLOW   r2=ו : slot 2 deleted; its full vowel (if any) replaces slot 1's vowel
  GEMINATE r2=r3: slot 2 deleted as for HOLLOW; slot 3 takes dagesh when a suffix follows

Not modelled: begadkefat spirantization, gutturals, stem-vowel lexical classes.
"""
from __future__ import annotations

from .hebrew import apply_final_forms, normalize_nfc

SHEVA, HATAF = "ְ", {"ֱ", "ֲ", "ֳ"}
FULL_VOWELS = {"ִ", "ֵ", "ֶ", "ַ", "ָ", "ֹ", "ֻ", "ׇ"}
DAGESH, HIRIQ, TSERE, SEGOL, QAMATS, HOLAM = "ּ", "ִ", "ֵ", "ֶ", "ָ", "ֹ"
MARKS = FULL_VOWELS | HATAF | {SHEVA, DAGESH, "ׁ", "ׂ"}

BINYANIM = ["PAAL", "NIFAL", "PIEL", "PUAL", "HIFIL", "HUFAL", "HITPAEL"]
PASSIVES = {"PUAL", "HUFAL"}
PREFIXED = {"NIFAL", "HIFIL", "HUFAL"}  # binyanim whose own prefix hosts the PE_YOD holam-waw

# --------------------------------------------------------------------------- strong templates
_PAST_SFX = {"PST.1SG": "ְתִּי", "PST.2MSG": "ְתָּ", "PST.2FSG": "ְתְּ", "PST.1PL": "ְנוּ",
             "PST.2MPL": "ְתֶּם", "PST.2FPL": "ְתֶּן"}


def _past(pre, v1, gem, v2_3ms, v2_cons, v2_vow, pre_pl2=None, v1_pl2=None):
    t = {"PST.3MSG": f"{pre}1{v1}2{gem}{v2_3ms}3", "PST.3FSG": f"{pre}1{v1}2{gem}{v2_vow}3ָה",
         "PST.3PL": f"{pre}1{v1}2{gem}{v2_vow}3וּ"}
    for c, s in _PAST_SFX.items():
        p, v = (pre_pl2, v1_pl2) if c in ("PST.2MPL", "PST.2FPL") and pre_pl2 is not None else (pre, v1)
        t[c] = f"{p}1{v}2{gem}{v2_cons}3{s}"
    return t


PAST = {
    "PAAL": _past("", "ָ", "", "ַ", "ַ", "ְ", pre_pl2="", v1_pl2="ְ"),
    "NIFAL": _past("נִ", "ְ", "", "ַ", "ַ", "ְ"),
    "PIEL": _past("", "ִ", "ּ", "ֵ", "ַ", "ְ"),
    "PUAL": _past("", "ֻ", "ּ", "ַ", "ַ", "ְ"),
    "HIFIL": _past("הִ", "ְ", "", "ִי", "ַ", "ִי"),
    "HUFAL": _past("הֻ", "ְ", "", "ַ", "ַ", "ְ"),
    "HITPAEL": _past("הִתְ", "ַ", "ּ", "ֵ", "ַ", "ְ"),
}
PRESENT = {  # MSG, FSG, MPL, FPL
    "PAAL": ["1וֹ2ֵ3", "1וֹ2ֶ3ֶת", "1וֹ2ְ3ִים", "1וֹ2ְ3וֹת"],
    "NIFAL": ["נִ1ְ2ָ3", "נִ1ְ2ֶ3ֶת", "נִ1ְ2ָ3ִים", "נִ1ְ2ָ3וֹת"],
    "PIEL": ["מְ1ַ2ֵּ3", "מְ1ַ2ֶּ3ֶת", "מְ1ַ2ְּ3ִים", "מְ1ַ2ְּ3וֹת"],
    "PUAL": ["מְ1ֻ2ָּ3", "מְ1ֻ2ֶּ3ֶת", "מְ1ֻ2ָּ3ִים", "מְ1ֻ2ָּ3וֹת"],
    "HIFIL": ["מַ1ְ2ִי3", "מַ1ְ2ִי3ָה", "מַ1ְ2ִי3ִים", "מַ1ְ2ִי3וֹת"],
    "HUFAL": ["מֻ1ְ2ָ3", "מֻ1ְ2ֶ3ֶת", "מֻ1ְ2ָ3ִים", "מֻ1ְ2ָ3וֹת"],
    "HITPAEL": ["מִתְ1ַ2ֵּ3", "מִתְ1ַ2ֶּ3ֶת", "מִתְ1ַ2ְּ3ִים", "מִתְ1ַ2ְּ3וֹת"],
}
PRS_CELLS = ["PRS.MSG", "PRS.FSG", "PRS.MPL", "PRS.FPL"]
# future: (prefix vowel, 1SG-prefix vowel, bare stem, stem before vowel suffix, stem before -נָה)
FUTURE_STEMS = {
    "PAAL": ("ִ", "ֶ", "1ְ2ֹ3", "1ְ2ְ3", "1ְ2ֹ3ְ"),
    "NIFAL": ("ִ", "ֵ", "1ָּ2ֵ3", "1ָּ2ְ3", "1ָּ2ַ3ְ"),
    "PIEL": ("ְ", "ֲ", "1ַ2ֵּ3", "1ַ2ְּ3", "1ַ2ֵּ3ְ"),
    "PUAL": ("ְ", "ֲ", "1ֻ2ַּ3", "1ֻ2ְּ3", "1ֻ2ַּ3ְ"),
    "HIFIL": ("ַ", "ַ", "1ְ2ִי3", "1ְ2ִי3", "1ְ2ֵ3ְ"),
    "HUFAL": ("ֻ", "ֻ", "1ְ2ַ3", "1ְ2ְ3", "1ְ2ַ3ְ"),
    "HITPAEL": ("ִ", "ֶ", "תְ1ַ2ֵּ3", "תְ1ַ2ְּ3", "תְ1ַ2ֵּ3ְ"),
}
FUT_CELLS = {  # cell -> (prefix consonant, stem kind, suffix)
    "FUT.1SG": ("א", "bare", ""), "FUT.2MSG": ("תּ", "bare", ""), "FUT.2FSG": ("תּ", "vow", "ִי"),
    "FUT.3MSG": ("י", "bare", ""), "FUT.3FSG": ("תּ", "bare", ""), "FUT.1PL": ("נ", "bare", ""),
    "FUT.2MPL": ("תּ", "vow", "וּ"), "FUT.2FPL": ("תּ", "nah", "נָה"),
    "FUT.3MPL": ("י", "vow", "וּ"), "FUT.3FPL": ("תּ", "nah", "נָה"),
}
IMPERATIVE = {  # 2MSG, 2FSG, 2MPL, 2FPL (active binyanim only)
    "PAAL": ["1ְ2ֹ3", "1ִ2ְ3ִי", "1ִ2ְ3וּ", "1ְ2ֹ3ְנָה"],
    "NIFAL": ["הִ1ָּ2ֵ3", "הִ1ָּ2ְ3ִי", "הִ1ָּ2ְ3וּ", "הִ1ָּ2ַ3ְנָה"],
    "PIEL": ["1ַ2ֵּ3", "1ַ2ְּ3ִי", "1ַ2ְּ3וּ", "1ַ2ֵּ3ְנָה"],
    "HIFIL": ["הַ1ְ2ֵ3", "הַ1ְ2ִי3ִי", "הַ1ְ2ִי3וּ", "הַ1ְ2ֵ3ְנָה"],
    "HITPAEL": ["הִתְ1ַ2ֵּ3", "הִתְ1ַ2ְּ3ִי", "הִתְ1ַ2ְּ3וּ", "הִתְ1ַ2ֵּ3ְנָה"],
}
IMP_CELLS = ["IMP.2MSG", "IMP.2FSG", "IMP.2MPL", "IMP.2FPL"]
TENSES = {"PST": list(PAST["PAAL"]), "PRS": PRS_CELLS, "FUT": list(FUT_CELLS), "IMP": IMP_CELLS}


def templates(binyan: str, tenses=("PST", "PRS", "FUT", "IMP")) -> dict[str, str]:
    """cell -> template string for one binyan."""
    out = {}
    if "PST" in tenses:
        out.update(PAST[binyan])
    if "PRS" in tenses:
        out.update(dict(zip(PRS_CELLS, PRESENT[binyan])))
    if "FUT" in tenses:
        pv, pv1, bare, vow, nah = FUTURE_STEMS[binyan]
        for cell, (pc, kind, sfx) in FUT_CELLS.items():
            v = pv1 if cell == "FUT.1SG" else pv
            stem = {"bare": bare, "vow": vow, "nah": nah}[kind]
            out[cell] = f"{pc}{v}{stem}{sfx}"
    if "IMP" in tenses and binyan in IMPERATIVE:
        out.update(dict(zip(IMP_CELLS, IMPERATIVE[binyan])))
    return out


# --------------------------------------------------------------------------- units & rules
def parse_units(t: str) -> list[list[str]]:
    units: list[list[str]] = []
    for ch in normalize_nfc(t):
        if ch in MARKS and units:
            units[-1][1] += ch
        else:
            units.append([ch, ""])
    return units


def _vowel(marks: str) -> str:
    return next((m for m in marks if m in FULL_VOWELS), "")


def _set_vowel(marks: str, v: str) -> str:
    kept = "".join(m for m in marks if m not in FULL_VOWELS and m != SHEVA and m not in HATAF)
    return kept + v


def _slot(units, s):
    return next((i for i, u in enumerate(units) if u[0] == s), None)


def classify(radicals: list[str]) -> str:
    r1, r2, r3 = [r[0] for r in radicals]
    if r2 == r3:
        return "GEMINATE"
    if r2 in ("ו", "י"):
        return "HOLLOW"
    if r3 == "ה":
        return "LAMED_HE"
    if r1 == "נ":
        return "PE_NUN"
    if r1 == "י":
        return "PE_YOD"
    return "STRONG"


def _suffix_context(units, i3) -> str:
    """'final' | 'consonant' | 'vowel' for what follows slot 3."""
    if i3 == len(units) - 1:
        return "final"
    nxt = units[i3 + 1]
    if _vowel(units[i3][1]) or (nxt[0] == "ו" and (DAGESH in nxt[1] or HOLAM in nxt[1])):
        return "vowel"
    return "consonant"


def apply_rules(units: list[list[str]], cls: str, binyan: str, cell: str) -> list[list[str]]:
    u = [list(x) for x in units]
    tense = cell.split(".")[0]
    i1, i2, i3 = _slot(u, "1"), _slot(u, "2"), _slot(u, "3")
    if cls == "PE_NUN" and i1 not in (None, 0) and (SHEVA in u[i1][1] or not _vowel(u[i1][1])):
        u[i2][1] = DAGESH + u[i2][1].replace(DAGESH, "")   # assimilated nun -> doubling of r2
        del u[i1]
    elif cls == "PE_YOD" and i1 not in (None, 0) and (SHEVA in u[i1][1] or not _vowel(u[i1][1])):
        prev = i1 - 1
        if binyan in PREFIXED and prev == 0:
            u[prev][1] = ""
            u.insert(prev + 1, ["ו", HOLAM])
            del u[i1 + 1]
        else:
            u[prev][1] = _set_vowel(u[prev][1], TSERE)
            del u[i1]
    elif cls == "LAMED_HE" and i3 is not None:
        if i3 - i2 == 2 and u[i2 + 1] == ["י", ""]:   # hiriq-yod mater (HIFIL) merges into the weak ending
            del u[i2 + 1]
            i3 -= 1
        ctx = _suffix_context(u, i3)
        if ctx == "final":
            u[i3] = ["ה", ""]
            u[i2][1] = _set_vowel(u[i2][1], QAMATS if tense == "PST" else SEGOL)
        elif ctx == "consonant":
            u[i3] = ["י", ""]
            u[i3 + 1][1] = u[i3 + 1][1].replace(DAGESH, "")   # no dagesh after a vowel letter
            v = (HIRIQ if binyan not in PASSIVES | {"NIFAL"} else TSERE) if tense == "PST" else SEGOL
            u[i2][1] = _set_vowel(u[i2][1], v)
        else:
            if cell == "PST.3FSG":
                u[i3] = ["ת", _vowel(u[i3][1])]
                u[i2][1] = _set_vowel(u[i2][1], SHEVA)
            else:
                v3 = _vowel(u[i3][1])
                u[i2][1] = _set_vowel(u[i2][1], v3)
                del u[i3]
    elif cls in ("HOLLOW", "GEMINATE") and i2 is not None:
        v2 = _vowel(u[i2][1])
        if i2 - i1 == 2 and u[i1 + 1][0] == "ו":      # holam-waw mater (PAAL present) collapses: קָם
            del u[i1 + 1]
            i2 -= 1
            v2 = QAMATS
        if v2:
            u[i1][1] = _set_vowel(u[i1][1], v2)
        del u[i2]
        if cls == "GEMINATE":
            i3 = _slot(u, "3")
            if i3 is not None and i3 < len(u) - 1:
                u[i3][1] = DAGESH + u[i3][1].replace(DAGESH, "")
    return u


def realize(template: str, radicals: list[str], binyan: str, cell: str, final_forms=True) -> str:
    cls = classify(radicals)
    u = apply_rules(parse_units(template), cls, binyan, cell)
    out = []
    for base, marks in u:
        if base in "123":
            r = radicals[int(base) - 1]
            out.append(r + marks)
        else:
            out.append(base + marks)
    s = normalize_nfc("".join(out))
    return apply_final_forms(s) if final_forms else s
