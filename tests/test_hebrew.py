import unicodedata

from hebmorph.hebrew import (apply_final_forms, audit_characters, extract_consonantal_skeleton, has_niqqud,
                             loose_vocalized_key, normalize_nfc, plene_consistent, predict_plene, strip_niqqud)

# כָּתַבְתִּי written with marks in a NON-canonical order (dagesh before qamats/hiriq)
KATAVTI_WEIRD = "\u05DB\u05BC\u05B8\u05EA\u05B7\u05D1\u05B0\u05EA\u05BC\u05B4\u05D9"
KATAVTI = "כָּתַבְתִּי"


def test_nfc_reorders_combining_marks():
    assert unicodedata.normalize("NFC", KATAVTI_WEIRD) != KATAVTI_WEIRD
    assert normalize_nfc(KATAVTI_WEIRD) == normalize_nfc(KATAVTI)
    assert normalize_nfc(None) is None


def test_nfc_decomposes_presentation_forms():
    # U+FB2A = SHIN WITH SHIN DOT (presentation form) -> ש + U+05C1
    assert normalize_nfc("שׁ") == "שׁ"


def test_strip_niqqud():
    assert strip_niqqud(KATAVTI) == "כתבתי"
    assert strip_niqqud(KATAVTI_WEIRD) == "כתבתי"
    assert strip_niqqud("שָׂכַר") == "שכר"          # sin dot removed
    assert strip_niqqud("הִשְׁתַּמֵּשׁ") == "השתמש"
    assert strip_niqqud("לִהְיוֹת יָכוֹל") == "להיות יכול"  # space kept
    assert strip_niqqud("כתב") == "כתב"             # idempotent on bare text
    assert not has_niqqud(strip_niqqud(KATAVTI))


def test_consonantal_skeleton():
    assert extract_consonantal_skeleton("כָּתַבְתְּ") == "כתבת"
    assert extract_consonantal_skeleton("הָלַךְ") == "הלכ"            # final kaf normalized
    assert extract_consonantal_skeleton("הָלַךְ", normalize_final_letters=False) == "הלך"
    assert extract_consonantal_skeleton("כּוֹתְבִים", drop_vav_yod=True) == "כתבמ"


def test_plene_consistency():
    assert plene_consistent("דִּבֵּר", "דיבר")
    assert plene_consistent("כָּתַבְתִּי", "כתבתי")
    assert plene_consistent("כָּרַתִּי", "כרתתי") is False   # real UniMorph discrepancy
    assert plene_consistent(None, "x") is None


def test_predict_plene():
    assert predict_plene("דִּבֵּר") == "דיבר"
    assert predict_plene("קָטֹנְתִּי") == "קטונתי"
    assert predict_plene("הִתְוַכֵּחַ") == "התווכח"


def test_loose_key_neutralises_orthographic_variants():
    assert loose_vocalized_key("כָּתַבֽ") == loose_vocalized_key("כָּתַב")   # meteg
    assert loose_vocalized_key("עָוֹן") == loose_vocalized_key("עָוֺן")    # holam haser for vav


def test_final_forms():
    assert apply_final_forms("כתבכ") == "כתבך"
    assert apply_final_forms("ךלמ") == "כלם"


def test_audit_logs_nonstandard_and_malformed():
    a = audit_characters("כָּתַב֑")                 # cantillation mark
    assert "NONSTANDARD_COMBINING_MARK" in a.flags
    assert "MALFORMED_MARK_ON_INVALID_BASE" in audit_characters("גְּנׁב").flags   # shin dot on nun
    assert "MALFORMED_REPEATED_MARK" in audit_characters("גָּּדַד").flags
    assert "NON_HEBREW_CHARACTER" in audit_characters("כתבx").flags
    assert "MALFORMED_ORPHAN_MARK" in audit_characters("ָכ").flags
    assert "RAW_NOT_NFC" in audit_characters(KATAVTI_WEIRD).flags
    assert audit_characters(KATAVTI).flags == []
