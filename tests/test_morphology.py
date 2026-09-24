import pytest

from hebmorph.morphology import (PAST_CELLS, TagError, binyan_from_paradigm_shape, feature_violations,
                                 normalize_binyan, parse_unimorph_tag, target_feature_string)
from hebmorph.roots import resolve_shin_sin, stable_radicals, structural_features


@pytest.mark.parametrize("raw,expected", [
    ("קל", "PAAL"), (" קל ", "PAAL"), ("[[קל]]", "PAAL"), ("פָּעַל (קל)", "PAAL"), ("פָּעַל", "PAAL"),
    ("נפעל", "NIFAL"), ("נִפְעַל", "NIFAL"), ("פיעל", "PIEL"), ("פִּעֵל", "PIEL"),
    ("פועל", "PUAL"), ("פֻּעַל", "PUAL"), ("[[פעל#פֻּעַל|פֻּעַל]]", "PUAL"),
    ("הפעיל", "HIFIL"), ("הופעל", "HUFAL"), ("התפעל", "HITPAEL"), ("[[התפעל]]", "HITPAEL"),
])
def test_binyan_normalization(raw, expected):
    assert normalize_binyan(raw) == (expected, [])


@pytest.mark.parametrize("raw,flag", [
    ("", "BINYAN_MISSING"), (None, "BINYAN_MISSING"), ("פעל", "BINYAN_LABEL_AMBIGUOUS"),
    ("קל ופיעל", "BINYAN_LABEL_UNRECOGNIZED"), ("פָּעֵל", "BINYAN_LABEL_UNRECOGNIZED"),
])
def test_binyan_never_guessed(raw, flag):
    b, flags = normalize_binyan(raw)
    assert b is None and flag in flags


def test_feature_normalization_and_cells():
    f = parse_unimorph_tag("V;1;SG;PST")
    assert f["canonical_cell"] == "PST.1SG" and f["gender"] == "NA" and f["person"] == "1"
    assert parse_unimorph_tag("V;2;SG;PST;MASC")["canonical_cell"] == "PST.2MSG"
    assert parse_unimorph_tag("V;2;SG;PST;FEM")["canonical_cell"] == "PST.2FSG"
    assert parse_unimorph_tag("V;3;PL;PST")["canonical_cell"] == "PST.3PL"
    assert parse_unimorph_tag("V;SG;PRS;FEM")["canonical_cell"] == "PRS.FSG"
    assert parse_unimorph_tag("V;2;PL;IMP;FEM")["canonical_cell"] == "IMP.2FPL"
    assert parse_unimorph_tag("V;NFIN")["canonical_cell"] == "NFIN"
    assert parse_unimorph_tag("V.MSDR")["canonical_cell"] == "MSDR"
    with pytest.raises(TagError):
        parse_unimorph_tag("V;PST;XYZ")
    with pytest.raises(TagError):
        parse_unimorph_tag("N;SG")


def test_canonical_past_paradigm_has_nine_cells_and_no_invented_gender():
    tags = ["V;1;SG;PST", "V;2;SG;PST;MASC", "V;2;SG;PST;FEM", "V;3;SG;PST;MASC", "V;3;SG;PST;FEM",
            "V;1;PL;PST", "V;2;PL;PST;MASC", "V;2;PL;PST;FEM", "V;3;PL;PST"]
    cells = [parse_unimorph_tag(t) for t in tags]
    assert sorted(c["canonical_cell"] for c in cells) == sorted(PAST_CELLS)
    for c in cells:
        assert feature_violations(c) == []
        if c["person"] == "1" or c["canonical_cell"] == "PST.3PL":
            assert c["gender"] == "NA"


def test_impossible_feature_combinations_detected():
    f = parse_unimorph_tag("V;1;SG;PST;MASC")
    assert "1ST_PERSON_WITH_GENDER" in feature_violations(f)
    assert "IMP_NOT_2ND_PERSON" in feature_violations(parse_unimorph_tag("V;3;SG;IMP;MASC"))


def test_target_features():
    assert target_feature_string("PST.1SG") == "PST,1,SG"
    assert target_feature_string("PST.2MSG") == "PST,2,SG,MASC"
    assert target_feature_string("PST.3PL") == "PST,3,PL"


def test_root_structural_features():
    f = structural_features(list("כתב"))
    assert f["structural_class"] == "strong" and f["structural_signature"] == "C.C.C" and f["sc_strong"]
    f = structural_features(list("נפל"))
    assert f["r1_is_nun"] and f["sc_initial_nun"] and f["has_weak_radical"]
    f = structural_features(list("קום"))
    assert f["r2_is_waw"] and f["sc_hollow_candidate"] and f["structural_class"] == "hollow_candidate"
    f = structural_features(list("בנה"))
    assert f["r3_is_he"] and f["sc_final_he"]
    f = structural_features(list("סבב"))
    assert f["is_geminate"] and f["structural_signature"].endswith("+GEM")
    f = structural_features(list("שאל"))
    assert f["r2_is_guttural"] and f["sc_guttural_containing"] and not f["sc_strong"]
    f = structural_features(["שׂ", "ח", "ק"])          # dotted radical handled
    assert f["radical_1"] == "ש"
    f = structural_features(list("גלגל"))
    assert f["sc_quadriliteral"] and f["is_reduplicated_quadriliteral"] and f["radical_4"] == "ל"


def test_stable_radicals():
    assert stable_radicals(list("נפל")) == ["פ", "ל"]
    assert stable_radicals(list("קום")) == ["ק", "מ"]
    assert stable_radicals(list("סבב")) == ["ס", "ב"]


def test_shin_sin_resolution():
    rads, m, fl = resolve_shin_sin("שכר", ["שׂ-כ-ר"], "שָׂכַר")
    assert rads[0] == "שׂ" and m == "SOURCE" and fl == []
    rads, m, fl = resolve_shin_sin("שכר", [], "שָׁכַר")        # dot read off the lemma (NFC puts it after qamats)
    assert rads[0] == "שׁ" and m == "FORM" and "SHIN_SIN_FROM_FORM" in fl
    _, m, fl = resolve_shin_sin("שכר", ["שׂ-כ-ר", "שׁ-כ-ר"], "שָׂכַר")
    assert m == "UNRESOLVED" and "SHIN_SIN_CONFLICT_IN_SOURCES" in fl
    _, _, fl = resolve_shin_sin("שכר", ["שׂ-כ-ר"], "שָׁכַר")
    assert "SHIN_SIN_MISMATCH" in fl


def test_binyan_shape_heuristic():
    assert binyan_from_paradigm_shape({"PST.3MSG": "כָּתַב", "PRS.MSG": "כּוֹתֵב", "NFIN": "לִכְתֹּב"})[0] == "PAAL"
    assert binyan_from_paradigm_shape({"PST.3MSG": "כִּתֵּב", "PRS.MSG": "מְכַתֵּב", "NFIN": "לְכַתֵּב"})[0] == "PIEL"
    assert binyan_from_paradigm_shape({"PST.3MSG": "הִכְתִּיב", "PRS.MSG": "מַכְתִּיב", "NFIN": "לְהַכְתִּיב"})[0] == "HIFIL"
    assert binyan_from_paradigm_shape({"PST.3MSG": "הִתְכַּתֵּב", "PRS.MSG": "מִתְכַּתֵּב", "NFIN": "לְהִתְכַּתֵּב"})[0] == "HITPAEL"
    assert binyan_from_paradigm_shape({"PST.3MSG": "נִכְתַּב", "PRS.MSG": "נִכְתָּב", "NFIN": "לְהִכָּתֵב"})[0] == "NIFAL"
    assert binyan_from_paradigm_shape({"PST.3MSG": "כֻּתַּב", "PRS.MSG": "מְכֻתָּב"})[0] == "PUAL"
    assert binyan_from_paradigm_shape({"PST.3MSG": "הֻכְתַּב", "PRS.MSG": "מֻכְתָּב"})[0] == "HUFAL"
