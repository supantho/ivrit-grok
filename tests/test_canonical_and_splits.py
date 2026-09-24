"""Integration tests over the BUILT dataset (run the pipeline first).
Covers: schema, provenance completeness, duplicate handling, syncretism,
paradigm cells, derived-view isolation, and split leakage for every manifest."""
import json

import pandas as pd
import pytest

from hebmorph import paths
from hebmorph.canonical_utils import add_syncretism
from hebmorph.loader import VISIBLE_FIELDS, load_split
from hebmorph.morphology import PAST_CELLS
from hebmorph.schema import check_foreign_keys, enforce
from hebmorph.splits import LeakageError, verify

C = paths.CANONICAL
pytestmark = pytest.mark.skipif(not (C / "forms.parquet").exists(), reason="dataset not built")
A = "analysis__"


@pytest.fixture(scope="module")
def tables():
    return {t: pd.read_parquet(C / f"{t}.parquet") for t in ["roots", "lexemes", "forms", "provenance", "validation"]}


@pytest.fixture(scope="module")
def pr_view():
    return pd.read_parquet(paths.DERIVED / "past_reinflection" / "past_reinflection.parquet")


def test_schema_and_foreign_keys(tables):
    for t, df in tables.items():
        enforce(df, t)
    check_foreign_keys(tables["roots"], tables["lexemes"], tables["forms"], tables["provenance"])


def test_every_record_has_provenance(tables):
    covered = set(tables["provenance"].canonical_record_id)
    for t, key in [("roots", "root_id"), ("lexemes", "lexeme_id"), ("forms", "form_id")]:
        assert set(tables[t][key]) <= covered, t


def test_unimorph_forms_have_line_provenance(tables):
    p = tables["provenance"]
    um = p[p.source_name == "unimorph_heb"]
    assert um.source_line_if_available.notna().all()
    assert um.source_url.str.contains(paths.UNIMORPH_COMMIT).all()


def test_one_row_per_lexeme_cell(tables):
    f = tables["forms"]
    assert not f.duplicated(["lexeme_id", "canonical_cell"]).any()
    assert (f.form_id == f.lexeme_id + "." + f.canonical_cell).all()


def test_2msg_2fsg_kept_separate_even_when_identical(tables):
    f = tables["forms"]
    p = f[f.canonical_cell.isin(["PST.2MSG", "PST.2FSG"]) & f.form_unvocalized.notna()]
    both = p.groupby("lexeme_id").form_unvocalized.agg(["nunique", "size"])
    assert ((both["size"] == 2) & (both["nunique"] == 1)).sum() > 500   # identical strings, two records


def test_no_invented_gender(tables):
    f = tables["forms"]
    assert (f[f.person == "1"].gender == "NA").all()
    assert (f[f.canonical_cell == "PST.3PL"].gender == "NA").all()


def test_binyan_labels_normalized(tables):
    assert set(tables["lexemes"].binyan.dropna()) <= {"PAAL", "NIFAL", "PIEL", "PUAL", "HIFIL", "HUFAL", "HITPAEL"}


def test_questionable_when_root_or_binyan_missing(tables):
    l = tables["lexemes"]
    assert (l[l.root_id.isna() | l.binyan.isna()].quality_tier == "QUESTIONABLE").all()


def test_ambiguous_matches_not_guessed(tables):
    l = tables["lexemes"]
    amb = l[l.match_status.isin(["ROOT_AMBIGUOUS", "ROOT_AND_BINYAN_AMBIGUOUS"])]
    assert amb.root_id.isna().all()
    amb = l[l.match_status.isin(["BINYAN_AMBIGUOUS", "ROOT_AND_BINYAN_AMBIGUOUS"])]
    assert amb.binyan.isna().all()


def test_form_tier_never_better_than_lexeme(tables):
    rank = {"GOLD": 0, "SILVER": 1, "QUESTIONABLE": 2}
    f = tables["forms"].merge(tables["lexemes"][["lexeme_id", "quality_tier"]], on="lexeme_id", suffixes=("", "_lex"))
    assert (f.quality_tier.map(rank) >= f.quality_tier_lex.map(rank)).all()


def test_niqqud_stripping_consistent(tables):
    from hebmorph.hebrew import strip_niqqud
    f = tables["forms"].dropna(subset=["form_vocalized"])
    assert (f.form_vocalized.map(strip_niqqud) == f.form_stripped).all()


def test_duplicates_reported_not_dropped(tables):
    d = pd.read_csv(paths.REPORTS / "duplicates.tsv", sep="\t")
    ids = set(x for s in d[d.duplicate_type == "DUPLICATE_ROOT_BINYAN"].record_ids for x in s.split("|"))
    l = tables["lexemes"]
    assert ids <= set(l.lexeme_id)                       # both records still present
    assert l[l.lexeme_id.isin(ids)].quality_flags.str.contains("DUPLICATE_ROOT_BINYAN").all()


def test_syncretism_detection_unit():
    f = pd.DataFrame(dict(lexeme_id=["L"] * 3, canonical_cell=["PST.2MSG", "PST.2FSG", "PST.1SG"],
                          form_vocalized=["כָּתַבְתָּ", "כָּתַבְתְּ", "כָּתַבְתִּי"],
                          form_unvocalized=["כתבת", "כתבת", "כתבתי"]))
    s = add_syncretism(f).set_index("canonical_cell")
    assert not s.is_syncretic.any()                     # distinct when vocalized
    assert s.loc["PST.2MSG", "syncretism_group_unvocalized"] == s.loc["PST.2FSG", "syncretism_group_unvocalized"]
    assert not s.loc["PST.1SG", "is_syncretic_unvocalized"]


def test_syncretism_in_built_data(tables):
    f = tables["forms"]
    g = f[f.syncretism_group_unvocalized.notna()].groupby("syncretism_group_unvocalized").form_unvocalized.nunique()
    assert (g == 1).all()


# ---------------- derived views ----------------
def test_view_citation_source_and_eight_targets(pr_view):
    assert (pr_view.source_cell == "PST.3MSG").all()
    assert set(pr_view.target_cell) == set(PAST_CELLS) - {"PST.3MSG"}
    assert not pr_view.example_id.duplicated().any()


def test_loader_exposes_only_visible_fields():
    ex = load_split("past_reinflection", "root_holdout_seed0", "train", "voc")
    assert ex and set(ex[0].__dataclass_fields__) == set(VISIBLE_FIELDS)
    assert not any(hasattr(ex[0], k) for k in ["root", "binyan", "lexeme_id", "root_id", "analysis"])


# ---------------- split leakage ----------------
def _manifests(view):
    return sorted((paths.SPLITS / view).glob("*/manifest.json"))


@pytest.mark.parametrize("mpath", _manifests("past_reinflection") + _manifests("full_paradigm"),
                         ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_split_constraints(mpath):
    m = json.loads(mpath.read_text())
    view = pd.read_parquet(paths.DERIVED / mpath.parent.parent.name / f"{mpath.parent.parent.name}.parquet")
    view = view[view.condition == "voc"]
    if m["strategy"] == "class_ood" and m["params"]["property"].startswith(A + "trad_"):
        g = m["params"]["property"][len(A + "trad_"):]
        view[m["params"]["property"]] = view[A + "traditional_class"].fillna("").str.split("+").map(lambda xs: g in xs)
    verify(m, view)
    P = {p: view[view.pair_id.isin(set(v["pair_ids"]))] for p, v in m["partitions"].items()}
    tr, te = P["train"], P["test"]
    if m["strategy"] == "root_holdout":
        assert set(tr[A + "root_id"]).isdisjoint(set(te[A + "root_id"]))
        # all binyanim of a root on the same side
        assert set(tr[A + "orthographic_root_group_id"]).isdisjoint(set(te[A + "orthographic_root_group_id"]))
    if m["strategy"] == "root_binyan_holdout":
        assert set(te[A + "root_id"]) <= set(tr[A + "root_id"])
        assert set(te[A + "binyan"]) <= set(tr[A + "binyan"])
        assert set(te[A + "root_binyan_key"]).isdisjoint(set(tr[A + "root_binyan_key"]))
    if m["strategy"] == "lexeme_holdout":
        assert set(te.lexeme_id).isdisjoint(set(tr.lexeme_id))
    if m["strategy"] == "cell_k":   # same lexeme on both sides BY DESIGN, cells disjoint
        ex = m["exposed_cells"]
        assert all(r.target_cell not in ex[r.lexeme_id] for r in te.itertuples())


def test_verifier_catches_injected_leak(pr_view):
    m = json.loads((paths.SPLITS / "past_reinflection" / "root_holdout_seed0" / "manifest.json").read_text())
    view = pr_view[pr_view.condition == "voc"]
    # move one test example of a train root into test -> must fail
    tr_ids = m["partitions"]["train"]["pair_ids"]
    m["partitions"]["test"]["pair_ids"] = m["partitions"]["test"]["pair_ids"] + [tr_ids[0]]
    m["partitions"]["train"]["pair_ids"] = tr_ids[1:]   # its root still has other examples in train
    with pytest.raises(LeakageError):
        verify(m, view)
