"""Transparent structural root features, computed ONLY from the radicals.

These are deliberately not traditional גזרה labels: e.g. 'hollow_candidate'
means r2 in {ו,י}, which includes roots that inflect as strong verbs.
"""
from __future__ import annotations

import hashlib

from .hebrew import GUTTURALS, normalize_finals, normalize_nfc

_SYM = {"נ": "N", "י": "Y", "ו": "W", "ה": "H", "א": "A", "ח": "X", "ע": "E", "ר": "R"}

STRUCTURAL_CLASSES = ["quadriliteral", "geminate", "hollow_candidate", "final_he", "initial_nun",
                      "initial_yod_waw", "final_alef", "initial_alef", "guttural_containing", "strong"]


def root_id_for(root_normalized: str, prefix: str = "RT-") -> str:
    return prefix + hashlib.sha1(("root|" + root_normalized).encode()).hexdigest()[:10]


def bare_radicals(radicals: list[str]) -> list[str]:
    """Drop shin/sin dots, normalize finals: ['שׂ','ח','ק'] -> ['ש','ח','ק']."""
    return [normalize_finals(normalize_nfc(r).replace("ׁ", "").replace("ׂ", "")) for r in radicals]


def structural_features(radicals: list[str]) -> dict:
    r = bare_radicals(radicals)
    n = len(r)
    if n < 3:
        raise ValueError(f"root too short: {radicals}")
    g = lambda i: r[i] if i < n else None
    r1, r2, r3 = r[0], r[1], r[2]
    tri = n == 3
    gem = tri and r2 == r3
    redup = n == 4 and r[0] == r[2] and r[1] == r[3]
    f = dict(
        radical_1=r1, radical_2=r2, radical_3=r3, radical_4=g(3), radical_5=g(4), root_length=n,
        r1_is_nun=r1 == "נ", r1_is_yod=r1 == "י", r1_is_waw=r1 == "ו", r1_is_alef=r1 == "א",
        r2_is_yod=r2 == "י", r2_is_waw=r2 == "ו",
        r3_is_he=r3 == "ה", r3_is_yod=r3 == "י", r3_is_alef=r3 == "א",
        rfinal_is_he=r[-1] == "ה",
        r1_is_guttural=r1 in GUTTURALS, r2_is_guttural=r2 in GUTTURALS, r3_is_guttural=r3 in GUTTURALS,
        has_resh="ר" in r, is_geminate=gem, is_reduplicated_quadriliteral=redup,
    )
    f["has_weak_radical"] = (r1 in {"נ", "י", "ו", "א"} or r2 in {"ו", "י"} or r3 in {"ה", "י", "א"} or gem)
    f["structural_signature"] = ".".join(_SYM.get(x, "C") for x in r) + ("+GEM" if gem else "")
    anygutt = any(x in GUTTURALS for x in r)
    sc = dict(
        sc_quadriliteral=n >= 4,
        sc_geminate=gem,
        sc_hollow_candidate=tri and r2 in {"ו", "י"},
        sc_final_he=tri and r3 == "ה",
        sc_initial_nun=r1 == "נ",
        sc_initial_yod_waw=r1 in {"י", "ו"},
        sc_guttural_containing=anygutt,
    )
    sc["sc_strong"] = tri and not f["has_weak_radical"] and not anygutt
    f.update(sc)
    order = [("quadriliteral", sc["sc_quadriliteral"]), ("geminate", gem),
             ("hollow_candidate", sc["sc_hollow_candidate"]), ("final_he", sc["sc_final_he"]),
             ("initial_nun", sc["sc_initial_nun"]), ("initial_yod_waw", sc["sc_initial_yod_waw"]),
             ("final_alef", tri and r3 == "א"), ("initial_alef", r1 == "א"),
             ("guttural_containing", anygutt)]
    f["structural_class"] = next((k for k, v in order if v), "strong")
    return f


def stable_radicals(radicals: list[str]) -> list[str]:
    """Radicals expected to surface in the written PST.3MSG consonantal skeleton
    of every binyan: all except r1 in {נ,י,ו}, r2 in {ו,י}, r3 in {ה,י},
    and the 2nd copy of a geminate."""
    r = bare_radicals(radicals)
    keep = []
    for i, x in enumerate(r):
        if i == 0 and x in {"נ", "י", "ו"}:
            continue
        if i == 1 and len(r) == 3 and x in {"ו", "י"}:
            continue
        if i == len(r) - 1 and x in {"ה", "י"}:
            continue
        if i == 2 and len(r) == 3 and r[1] == r[2]:
            continue
        keep.append(x)
    return keep


def is_subsequence(needle: list[str], hay: str) -> bool:
    it = iter(hay)
    return all(ch in it for ch in needle)


def split_root_units(root: str) -> list[str]:
    """'שׂכר' -> ['שׂ','כ','ר'] (a radical = letter + optional shin/sin dot)."""
    out = []
    for ch in normalize_nfc(root):
        if ch in ("ׁ", "ׂ") and out:
            out[-1] += ch
        else:
            out.append(ch)
    return out


def orthographic(root: str) -> str:
    return "".join(bare_radicals(split_root_units(root)))


def shin_sin_dots(text: str) -> set[str]:
    """Set of dots (U+05C1 / U+05C2) carried by ש letters in a vocalized string.
    NFC places the dot after vowel points and dagesh, so all marks up to the
    next letter are inspected."""
    dots, cur = set(), None
    for ch in normalize_nfc(text or ""):
        if 0x05D0 <= ord(ch) <= 0x05EA:
            cur = ch
        elif cur == "ש" and ch in ("\u05C1", "\u05C2"):
            dots.add(ch)
    return dots


def resolve_shin_sin(root_bare: str, dotted_variants: list[str], lemma: str) -> tuple[list[str], str, list[str]]:
    """Decide שׁ vs שׂ for every ש radical.

    1. If sources give dots for a position, use them (conflicting dots ->
       SHIN_SIN_CONFLICT_IN_SOURCES, left bare).
    2. Otherwise read the dot off the lexeme's own vocalized lemma when all ש
       letters in it carry the same dot (flag SHIN_SIN_FROM_FORM); else leave
       bare (SHIN_SIN_UNRESOLVED).
    3. A source dot that contradicts the lemma -> SHIN_SIN_MISMATCH.
    Returns (radicals, method in {NA, SOURCE, FORM, MIXED, UNRESOLVED}, flags)."""
    rads = list(root_bare)
    if "ש" not in rads:
        return rads, "NA", []
    flags = []
    lem = normalize_nfc(lemma or "")
    lemma_dots = shin_sin_dots(lem)
    lemma_dot = next(iter(lemma_dots)) if len(lemma_dots) == 1 else None
    methods = set()
    for pos, r in enumerate(rads):
        if r != "ש":
            continue
        given = set()
        for v in dotted_variants:
            units = v.split("-")
            if pos < len(units) and len(units[pos]) > 1:
                given.add(units[pos][1])
        if len(given) > 1:
            flags.append("SHIN_SIN_CONFLICT_IN_SOURCES")
            methods.add("UNRESOLVED")
        elif len(given) == 1:
            dot = next(iter(given))
            rads[pos] = "ש" + dot
            methods.add("SOURCE")
            if lemma_dots and dot not in lemma_dots:
                flags.append("SHIN_SIN_MISMATCH")
        elif lemma_dot:
            rads[pos] = "ש" + lemma_dot
            methods.add("FORM")
            flags.append("SHIN_SIN_FROM_FORM")
        else:
            methods.add("UNRESOLVED")
            flags.append("SHIN_SIN_UNRESOLVED")
    method = next(iter(methods)) if len(methods) == 1 else "MIXED"
    return [normalize_nfc(x) for x in rads], method, sorted(set(flags))
