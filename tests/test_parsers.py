"""UniMorph and Wiktionary parsing on small hand-written fixtures."""
from hebmorph.unimorph import UMLine, args_tag_to_v3, pair_voc_unvoc, segment_paradigms
from hebmorph.wiktionary import (Page, extract_conj_tables, extract_verb_entries, parse_gizra, parse_root_value,
                                 table_forms)


def _lines(file, triples, start=1):
    return [UMLine(file, i, *t) for i, t in enumerate(triples, start)]


def test_segmentation_keeps_duplicates_and_non_verbs_out():
    ls = _lines("heb_voc", [("אֵם", "אִמִּי", "N;SG;PSSD;PSS1S"),
                            ("כָּתַב", "כָּתַב", "V;3;SG;PST;MASC"),
                            ("כָּתַב", "כָּתַבְתִּי", "V;1;SG;PST"),
                            ("כָּתַב", "כָּתַבְתִּי", "V;1;SG;PST"),      # duplicate cell line
                            ("לָמַד", "לָמַד", "V;3;SG;PST;MASC")])
    ps = segment_paradigms(ls)
    assert [p.lemma for p in ps] == ["כָּתַב", "לָמַד"]
    assert set(ps[0].cells) == {"PST.3MSG", "PST.1SG"}
    assert [c for c, _ in ps[0].duplicate_cells] == ["PST.1SG"]
    assert ps[0].cells["PST.1SG"].line_no == 3            # provenance line kept


def test_voc_unvoc_pairing_uses_plene_and_breaks_ties():
    voc = segment_paradigms(_lines("heb_voc", [
        ("קָטַן", "קָטֹן", "V;3;SG;PST;MASC"), ("קָטַן", "קָטַנְתִּי", "V;1;SG;PST"), ("קָטַן", "קָטַנּוּ", "V;1;PL;PST"),
        ("קָטֹן", "קָטֹן", "V;3;SG;PST;MASC"), ("קָטֹן", "קָטֹנְתִּי", "V;1;SG;PST"), ("קָטֹן", "קָטֹנּוּ", "V;1;PL;PST")]))
    unv = segment_paradigms(_lines("heb", [
        ("קטן", "קטן", "V;3;SG;PST;MASC"), ("קטן", "קטנתי", "V;1;SG;PST"), ("קטן", "קטנו", "V;1;PL;PST"),
        ("קטון", "קטון", "V;3;SG;PST;MASC"), ("קטון", "קטונתי", "V;1;SG;PST"), ("קטון", "קטונו", "V;1;PL;PST")]))
    r = pair_voc_unvoc(voc, unv)
    assert [x["unvoc_index"] for x in r] == [0, 1]
    assert all(x["status"] == "PAIRED" for x in r)


def test_args_tag_conversion():
    assert args_tag_to_v3("V;PST;NOM(3,SG,MASC)") == "PST.3MSG"
    assert args_tag_to_v3("V;3;SG;PST;MASC") == "PST.3MSG"
    assert args_tag_to_v3("V;NFIN") == "NFIN"


WIKI = """==כָּתַב==
{{ניתוח דקדוקי לפועל|
|כתיב מלא=כתב
|שורש וגזרה={{שרש3|כ|ת|ב}}, [[גזרת השלמים]]
|בניין=קל}}
# write
===תרגום===
*אנגלית: {{ת|אנגלית|write}}
{{שורש|כתב}}
==שָׂכַר {{משני|א}}==
{{ניתוח דקדוקי לפועל|
|כתיב מלא=שכר
|שורש וגזרה={{שרש3|שׂ|כ|ר|א}}
|בניין=[[קל]]}}
"""


def test_wiktionary_entry_extraction():
    rows = extract_verb_entries(Page("כתב", 0, 1, 99, "t", WIKI))
    assert len(rows) == 2
    a, b = rows
    assert a["headword"] == "כָּתַב" and a["root_radicals"] == "כ-ת-ב" and a["gizra_labels"] == "SHLEMIM"
    assert a["binyan_raw"] == "קל" and a["gloss_en"] == "write" and a["ktiv_male"] == "כתב"
    assert b["headword"] == "שָׂכַר" and b["root_radicals"] == "שׂ-כ-ר" and b["root_homonym_index"] == "א"


def test_root_parsing_variants_and_conflicts():
    assert parse_root_value("{{שרש|ס|כ|ך}}")[0] == ["ס", "כ", "כ"]
    assert parse_root_value("{{שרש|סקל|ס־ק־ל}}, שלמים")[0] == ["ס", "ק", "ל"]
    r, method, flags, _ = parse_root_value("{{שרש3|ז|י|ד}} או {{שרש3|ז|ו|ד}}")
    assert r is None and "ROOT_MULTIPLE_IN_ENTRY" in flags            # never guessed
    assert parse_root_value("")[2] == ["ROOT_MISSING"]


def test_gizra_labels():
    assert parse_gizra('[[גזרת נל"י/ה]]')[1] == ["LAMED_YOD_HE"]
    assert parse_gizra('[[גזרת חפ"נ]]')[1] == ["PE_NUN"]
    assert parse_gizra("[[גזרת הכפולים]]")[1] == ["AYIN_AYIN"]


def test_conjugation_tables_absence_and_placeholder_root():
    text = """{{נטיות פעלים|שורש=פ־ע־ל
|קל עבר=[[סגר|סָגַר]]
|קל שם הפועל=לִסְגֹּר
|פיעל עבר=סִגֵּר
|הופעל עבר=-אין-
}}"""
    rows = extract_conj_tables(Page("סגר (שורש)", 0, 5, 7, "t", text))
    by = {(r["binyan"], r["cell"]): r for r in rows}
    assert by[("PAAL", "PST.3MSG")]["forms"] == "סָגַר"
    assert by[("PAAL", "NFIN")]["forms"] == "לִסְגֹּר"
    assert "TABLE_ASSERTS_ABSENT" in by[("HUFAL", "PST.3MSG")]["flags"]
    # template placeholder root פ־ע־ל contradicted by the forms -> page-title root used, flagged
    assert by[("PAAL", "PST.3MSG")]["root_radicals"] == "ס-ג-ר"
    assert "TABLE_ROOT_PARAM_REPLACED_BY_TITLE" in by[("PAAL", "PST.3MSG")]["flags"]


def test_table_forms_ignore_annotations():
    f, fl = table_forms("יֹאהַב או יֶאֱהַב")
    assert f == ["יֹאהַב", "יֶאֱהַב"] and "TABLE_VALUE_MULTIPLE_FORMS" in fl
    f, fl = table_forms("[[אדם|אִדֵּם]]{{הבהרה|היכן קיים}}")
    assert f == ["אִדֵּם"] and "TABLE_VALUE_DOUBTED" in fl
    assert table_forms("-אַיִן-")[1] == ["TABLE_ASSERTS_ABSENT"]
