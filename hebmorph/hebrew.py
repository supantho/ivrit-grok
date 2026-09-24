"""Hebrew Unicode utilities.

All comparisons of Hebrew strings in this project go through ``normalize_nfc``
first.  Stripping never silently drops unexpected characters: callers can ask
for an audit (``audit_characters``) and the pipeline logs every nonstandard
combining mark it meets.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field

LETTERS = {chr(c) for c in range(0x05D0, 0x05EB)}  # alef .. tav incl. finals
FINAL_TO_REGULAR = {"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"}
REGULAR_TO_FINAL = {v: k for k, v in FINAL_TO_REGULAR.items()}

# Standard Tiberian vowel points and diacritics used in modern vocalized text.
SHEVA, HATAF_SEGOL, HATAF_PATAH, HATAF_QAMATS = "ְ", "ֱ", "ֲ", "ֳ"
HIRIQ, TSERE, SEGOL, PATAH, QAMATS = "ִ", "ֵ", "ֶ", "ַ", "ָ"
HOLAM, HOLAM_HASER_FOR_VAV, QUBUTS = "ֹ", "ֺ", "ֻ"
DAGESH, METEG, RAFE = "ּ", "ֽ", "ֿ"
SHIN_DOT, SIN_DOT, QAMATS_QATAN = "ׁ", "ׂ", "ׇ"

STANDARD_POINTS = {
    SHEVA, HATAF_SEGOL, HATAF_PATAH, HATAF_QAMATS, HIRIQ, TSERE, SEGOL, PATAH,
    QAMATS, HOLAM, HOLAM_HASER_FOR_VAV, QUBUTS, DAGESH, SHIN_DOT, SIN_DOT,
    QAMATS_QATAN,
}
# Marks we accept but consider nonstandard for modern vocalized verb forms.
NONSTANDARD_HEBREW_MARKS = {METEG, RAFE, "ׄ", "ׅ"} | {
    chr(c) for c in range(0x0591, 0x05B0)  # cantillation (te'amim)
}
HEBREW_PUNCT = {"־": "MAQAF", "׀": "PASEQ", "׃": "SOF_PASUQ",
                "׆": "NUN_HAFUKHA", "׳": "GERESH", "״": "GERSHAYIM"}
GUTTURALS = {"א", "ה", "ח", "ע"}
VOWEL_POINTS = {SHEVA, HATAF_SEGOL, HATAF_PATAH, HATAF_QAMATS, HIRIQ, TSERE,
                SEGOL, PATAH, QAMATS, HOLAM, HOLAM_HASER_FOR_VAV, QUBUTS,
                QAMATS_QATAN}


def normalize_nfc(text: str | None) -> str | None:
    """Unicode NFC.  Also decomposes Hebrew presentation forms (U+FB1D-FB4F),
    which are composition-excluded, and puts marks in canonical order."""
    if text is None:
        return None
    return unicodedata.normalize("NFC", text)


def is_hebrew_mark(ch: str) -> bool:
    """Any combining mark in the Hebrew block (points + cantillation)."""
    return 0x0591 <= ord(ch) <= 0x05C7 and unicodedata.category(ch) == "Mn"


def strip_niqqud(text: str | None) -> str | None:
    """Remove every Hebrew combining mark (vowel points, dagesh, shin/sin dots,
    meteg, cantillation).  Letters, punctuation and spaces are kept, so the
    result is the *defective* (ktiv haser) spelling of a vocalized form, not
    the plene (ktiv male) spelling a modern writer would use."""
    if text is None:
        return None
    return "".join(ch for ch in normalize_nfc(text) if not is_hebrew_mark(ch))


def normalize_finals(text: str) -> str:
    """Map final letter forms to their regular forms (for root comparison)."""
    return "".join(FINAL_TO_REGULAR.get(ch, ch) for ch in text)


def apply_final_forms(text: str) -> str:
    """Use final letter forms at word ends, regular forms elsewhere.
    Works on vocalized strings (marks after the last letter are kept)."""
    chars = list(normalize_finals(text))
    last = None
    for i in range(len(chars) - 1, -1, -1):
        if chars[i] in LETTERS:
            last = i
            break
    if last is not None and chars[last] in REGULAR_TO_FINAL:
        chars[last] = REGULAR_TO_FINAL[chars[last]]
    return "".join(chars)


def extract_consonantal_skeleton(text: str | None, *, normalize_final_letters: bool = True,
                                 drop_vav_yod: bool = False) -> str | None:
    """Sequence of Hebrew letters only (marks, spaces and punctuation removed).

    ``normalize_final_letters`` maps ך->כ etc.  ``drop_vav_yod`` removes every
    ו and י: a crude but transparent key under which a defective and a plene
    spelling of the same word coincide (matres lectionis are ו/י)."""
    if text is None:
        return None
    s = "".join(ch for ch in normalize_nfc(text) if ch in LETTERS)
    if normalize_final_letters:
        s = normalize_finals(s)
    if drop_vav_yod:
        s = s.replace("ו", "").replace("י", "")
    return s


def loose_vocalized_key(text: str | None) -> str | None:
    """A vocalized comparison key that neutralises orthographic variants which
    do not change the intended vocalization: meteg and cantillation dropped,
    qamats qatan -> qamats, holam-haser-for-vav -> holam, final letters
    normalized.  Used only as a *weaker* match signal."""
    if text is None:
        return None
    s = normalize_nfc(text)
    out = []
    for ch in s:
        if ch == METEG or ch == RAFE or (0x0591 <= ord(ch) <= 0x05AF):
            continue
        if ch == QAMATS_QATAN:
            ch = QAMATS
        elif ch == HOLAM_HASER_FOR_VAV:
            ch = HOLAM
        out.append(ch)
    return normalize_nfc(normalize_finals("".join(out)))


def has_niqqud(text: str | None) -> bool:
    return bool(text) and any(ch in VOWEL_POINTS for ch in normalize_nfc(text))


@dataclass
class CharAudit:
    """Result of auditing a string's characters."""
    nonstandard_marks: list[str] = field(default_factory=list)
    non_hebrew_chars: list[str] = field(default_factory=list)
    hebrew_punct: list[str] = field(default_factory=list)
    has_space: bool = False
    presentation_forms: list[str] = field(default_factory=list)
    was_not_nfc: bool = False
    orphan_marks: bool = False  # a combining mark not preceded by a letter
    invalid_base: list[str] = field(default_factory=list)  # e.g. shin dot on a non-shin letter
    repeated_marks: list[str] = field(default_factory=list)  # same mark twice on one letter

    @property
    def flags(self) -> list[str]:
        f = []
        if self.nonstandard_marks:
            f.append("NONSTANDARD_COMBINING_MARK")
        if self.non_hebrew_chars:
            f.append("NON_HEBREW_CHARACTER")
        if self.hebrew_punct:
            f.append("HEBREW_PUNCTUATION")
        if self.has_space:
            f.append("CONTAINS_SPACE")
        if self.presentation_forms:
            f.append("PRESENTATION_FORM_IN_RAW")
        if self.was_not_nfc:
            f.append("RAW_NOT_NFC")
        if self.orphan_marks:
            f.append("MALFORMED_ORPHAN_MARK")
        if self.invalid_base:
            f.append("MALFORMED_MARK_ON_INVALID_BASE")
        if self.repeated_marks:
            f.append("MALFORMED_REPEATED_MARK")
        return f


def audit_characters(raw: str) -> CharAudit:
    a = CharAudit()
    a.was_not_nfc = unicodedata.normalize("NFC", raw) != raw
    a.presentation_forms = [f"U+{ord(c):04X}" for c in raw if 0xFB1D <= ord(c) <= 0xFB4F]
    s = normalize_nfc(raw)
    prev_base = None
    marks_here: set = set()
    for ch in s:
        if ch in LETTERS:
            prev_base = ch
            marks_here = set()
        elif is_hebrew_mark(ch):
            if prev_base is None:
                a.orphan_marks = True
            if ch not in STANDARD_POINTS:
                a.nonstandard_marks.append(f"U+{ord(ch):04X}")
            if ch in marks_here:
                a.repeated_marks.append(f"U+{ord(ch):04X} on {prev_base}")
            marks_here.add(ch)
            if (ch in (SHIN_DOT, SIN_DOT) and prev_base != "ש") or (ch == HOLAM_HASER_FOR_VAV and prev_base != "ו"):
                a.invalid_base.append(f"U+{ord(ch):04X} on {prev_base}")
        elif ch == " ":
            a.has_space = True
            prev_base = None
        elif ch in HEBREW_PUNCT:
            a.hebrew_punct.append(HEBREW_PUNCT[ch])
            prev_base = None
        elif unicodedata.category(ch) == "Mn":
            a.nonstandard_marks.append(f"U+{ord(ch):04X}")
        else:
            a.non_hebrew_chars.append(f"U+{ord(ch):04X}")
            prev_base = None
    return a


def mark_inventory(strings) -> Counter:
    """Count every combining mark over an iterable of strings (for logs)."""
    c = Counter()
    for s in strings:
        for ch in normalize_nfc(s or ""):
            if unicodedata.category(ch) == "Mn":
                c[f"U+{ord(ch):04X} {unicodedata.name(ch, '?')}"] += 1
    return c


def _units(text: str) -> list[tuple[str, str]]:
    """Split NFC text into (letter, marks) units; non-letters become units too."""
    units = []
    for ch in normalize_nfc(text):
        if is_hebrew_mark(ch) and units:
            units[-1] = (units[-1][0], units[-1][1] + ch)
        else:
            units.append((ch, ""))
    return units


def predict_plene(vocalized: str) -> str:
    """APPROXIMATE ktiv male from a vocalized form (Academy-style rules):
    holam/qubuts on a non-vav letter -> add ו; hiriq in an open syllable
    (next letter without sheva, and not already י) -> add י;
    word-internal consonantal ו -> וו.  Used only as a tie-breaker and a
    diagnostic, never to create data."""
    u = _units(vocalized)
    out = []
    for i, (ch, marks) in enumerate(u):
        nxt = u[i + 1] if i + 1 < len(u) else None
        if ch == "ו" and marks and HOLAM not in marks and not (marks == DAGESH) and 0 < i < len(u) - 1:
            out.append("וו")
            continue
        out.append(ch)
        if ch not in LETTERS or ch == "ו":
            continue
        if (HOLAM in marks or QUBUTS in marks) and not (nxt and nxt[0] == "ו"):
            out.append("ו")
        elif HIRIQ in marks and nxt and nxt[0] in LETTERS and nxt[0] != "י":
            if SHEVA not in nxt[1] and i + 2 < len(u):
                out.append("י")
    return "".join(out)


def plene_consistent(vocalized: str | None, unvocalized: str | None) -> bool | None:
    """Is a plene (ktiv male) spelling consistent with a vocalized form?

    They are consistent if, after stripping niqqud from the vocalized form and
    deleting all ו/י (possible matres lectionis) from both, the letter
    sequences are equal.  Returns None if either side is missing."""
    if not vocalized or not unvocalized:
        return None
    return (extract_consonantal_skeleton(strip_niqqud(vocalized), drop_vav_yod=True)
            == extract_consonantal_skeleton(unvocalized, drop_vav_yod=True))
