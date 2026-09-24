"""QC reports over the canonical tables (TSV, human-readable)."""
from __future__ import annotations

import json

import pandas as pd

from . import paths
from .morphology import PAST_CELLS


def _tsv(df: pd.DataFrame, name: str):
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.REPORTS / name, sep="\t", index=False)


def lexeme_view(roots, lexemes):
    rcols = ["root_id", "root_normalized", "structural_class", "structural_signature", "traditional_class",
             "root_length"] + [c for c in roots.columns if c.startswith("sc_")]
    return lexemes.merge(roots[rcols], on="root_id", how="left")


def source_coverage(lexemes, forms, interim) -> pd.DataFrame:
    rows = []
    um = lexemes[lexemes.source_primary == "unimorph_heb"]
    s = json.loads((paths.INTERIM / "unimorph" / "ingest_summary.json").read_text())
    w = json.loads((paths.INTERIM / "wiktionary" / "ingest_summary.json").read_text())
    add = lambda src, metric, v: rows.append(dict(source=src, metric=metric, value=v))
    add("unimorph_heb", "verb lines heb_voc", s["voc_verb_lines"])
    add("unimorph_heb", "verb lines heb", s["unvoc_verb_lines"])
    add("unimorph_heb", "paradigms heb_voc", s["voc_paradigms"])
    add("unimorph_heb", "paradigms heb", s["unvoc_paradigms"])
    for k, v in s["pairing_status"].items():
        add("unimorph_heb", f"voc->unvoc pairing {k}", v)
    for k, v in s["args_check"].items():
        add("unimorph_heb", f"heb.args cross-check {k}", v)
    add("unimorph_heb", "UD-derived verb lines (attestation only)", s["ud_verb_lines"])
    add("unimorph_heb", "lexemes (after merging identical duplicate paradigms)", len(um))
    for k, v in um.match_status.value_counts().items():
        add("unimorph_heb", f"lexemes match_status={k}", int(v))
    add("unimorph_heb", "lexemes with root assigned", int(um.root_id.notna().sum()))
    add("unimorph_heb", "lexemes with binyan assigned", int(um.binyan.notna().sum()))
    add("unimorph_heb", "lexemes ud_attested", int(um.ud_attested.sum()))
    for k, v in w.items():
        if k != "parser_version":
            add("hewiktionary", k, v)
    wk = lexemes[lexemes.source_primary == "hewiktionary"]
    add("hewiktionary", "Wiktionary-only lexemes added", len(wk))
    fu = forms[forms.source_primary == "unimorph_heb"]
    add("unimorph_heb", "forms", len(fu))
    add("unimorph_heb", "forms with vocalized", int(fu.form_vocalized.notna().sum()))
    add("unimorph_heb", "forms with plene unvocalized", int(fu.form_unvocalized.notna().sum()))
    add("hewiktionary", "UniMorph forms corroborated by a Wiktionary table", int((fu.wiktionary_corroborated == True).sum()))  # noqa: E712
    add("hewiktionary", "UniMorph forms contradicted by a Wiktionary table", int((fu.wiktionary_corroborated == False).sum()))  # noqa: E712
    return pd.DataFrame(rows)


def paradigm_coverage(roots, lexemes) -> pd.DataFrame:
    v = lexeme_view(roots, lexemes)
    v["binyan"] = v.binyan.fillna("UNKNOWN")
    v["structural_class"] = v.structural_class.fillna("UNKNOWN_ROOT")
    out = []
    for dim in ["binyan", "structural_class", "quality_tier", "source_primary"]:
        g = v.groupby(dim).agg(lexemes=("lexeme_id", "size"),
                               mean_past_completeness=("past_completeness", "mean"),
                               complete_9cell=("past_complete", "sum"),
                               observed_past_cells=("past_observed_cells", "sum")).reset_index()
        g.insert(0, "dimension", dim)
        g = g.rename(columns={dim: "value"})
        out.append(g)
    t = dict(dimension="ALL", value="ALL", lexemes=len(v), mean_past_completeness=v.past_completeness.mean(),
             complete_9cell=int(v.past_complete.sum()), observed_past_cells=int(v.past_observed_cells.sum()))
    df = pd.concat([pd.DataFrame([t])] + out, ignore_index=True)
    df["mean_past_completeness"] = df.mean_past_completeness.round(4)
    return df


def per_lexeme_completeness(lexemes, forms) -> pd.DataFrame:
    past = forms[forms.canonical_cell.isin(PAST_CELLS) & forms.form_vocalized.notna()]
    have = past.groupby("lexeme_id").canonical_cell.apply(set)
    rows = []
    for r in lexemes.itertuples(index=False):
        h = have.get(r.lexeme_id, set())
        rows.append(dict(lexeme_id=r.lexeme_id, lemma_vocalized=r.lemma_vocalized, binyan=r.binyan,
                         source_primary=r.source_primary, quality_tier=r.quality_tier,
                         past_observed_cells=len(h), past_completeness=round(len(h) / 9, 4),
                         missing_past_cells="|".join(c for c in PAST_CELLS if c not in h) or None))
    return pd.DataFrame(rows)


def syncretism_report(forms) -> pd.DataFrame:
    rows = []
    for level, gcol, fcol in [("vocalized", "syncretism_group", "form_vocalized"),
                              ("unvocalized", "syncretism_group_unvocalized", "form_unvocalized")]:
        g = forms[forms[gcol].notna()].groupby(gcol)
        for gid, sub in g:
            cells = sub.canonical_cell.tolist()
            rows.append(dict(level=level, syncretism_group=gid, lexeme_id=sub.lexeme_id.iloc[0],
                             form=sub[fcol].iloc[0], n_cells=len(cells), cells="|".join(cells),
                             past_cells="|".join(c for c in cells if c in PAST_CELLS) or None,
                             cross_tense=len({c.split(".")[0] for c in cells}) > 1,
                             binyan=sub.binyan.iloc[0]))
    return pd.DataFrame(rows)


def syncretism_patterns(forms) -> pd.DataFrame:
    s = syncretism_report(forms)
    if not len(s):
        return s
    return (s.groupby(["level", "cells"]).size().rename("n_lexemes").reset_index()
            .sort_values(["level", "n_lexemes"], ascending=[True, False]))


def binyan_distribution(lexemes, forms) -> pd.DataFrame:
    l = lexemes.assign(binyan=lexemes.binyan.fillna("UNKNOWN"))
    f = forms.assign(binyan=forms.binyan.fillna("UNKNOWN"))
    a = pd.crosstab(l.binyan, l.quality_tier).add_prefix("lexemes_")
    a["lexemes_total"] = a.sum(axis=1)
    a["lexemes_unimorph"] = l[l.source_primary == "unimorph_heb"].groupby("binyan").size()
    a["lexemes_complete_past"] = l[l.past_complete].groupby("binyan").size()
    a["forms_total"] = f.groupby("binyan").size()
    a["past_forms"] = f[f.is_past_cell].groupby("binyan").size()
    return a.fillna(0).astype(int).reset_index()


def root_class_distribution(roots, lexemes) -> pd.DataFrame:
    rows = []
    for c in ["structural_class", "structural_signature", "traditional_class"]:
        vc = roots[c].fillna("NOT_STATED").value_counts()
        for k, v in vc.items():
            rows.append(dict(dimension=c, value=k, roots=int(v)))
    for c in [c for c in roots.columns if c.startswith("sc_")] + ["has_weak_radical", "is_geminate"]:
        rows.append(dict(dimension="flag", value=c, roots=int(roots[c].sum())))
    df = pd.DataFrame(rows)
    # structural vs traditional cross-tab (long format) -- the two views are kept separate
    x = roots.assign(traditional_class=roots.traditional_class.fillna("NOT_STATED"))
    ct = x.groupby(["structural_class", "traditional_class"]).size().rename("roots").reset_index()
    ct.insert(0, "dimension", "structural_x_traditional")
    ct["value"] = ct.structural_class + " x " + ct.traditional_class
    df = pd.concat([df, ct[["dimension", "value", "roots"]]], ignore_index=True)
    return df


def missing_metadata(roots, lexemes, forms, wikt_missing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in lexemes.itertuples(index=False):
        miss = [c for c in ["root_id", "binyan", "lemma_unvocalized", "infinitive_vocalized",
                            "citation_past_3ms_vocalized", "gloss_en"] if pd.isna(getattr(r, c))]
        if miss:
            rows.append(dict(record_id=r.lexeme_id, record_type="lexeme", lemma=r.lemma_vocalized,
                             source_primary=r.source_primary, quality_tier=r.quality_tier,
                             missing_fields="|".join(miss), details=r.match_status))
    for r in forms[forms.form_vocalized.isna() | forms.form_unvocalized.isna()].itertuples(index=False):
        miss = [c for c in ["form_vocalized", "form_unvocalized"] if pd.isna(getattr(r, c))]
        rows.append(dict(record_id=r.form_id, record_type="form", lemma=None, source_primary=r.source_primary,
                         quality_tier=r.quality_tier, missing_fields="|".join(miss), details=r.canonical_cell))
    for r in roots[roots.traditional_class.isna()].itertuples(index=False):
        rows.append(dict(record_id=r.root_id, record_type="root", lemma=r.root_normalized,
                         source_primary=r.source_primary, quality_tier=r.quality_tier,
                         missing_fields="traditional_class", details="no גזרה stated by any matched entry"))
    df = pd.DataFrame(rows)
    if len(wikt_missing):
        w = wikt_missing.assign(source_primary="hewiktionary", quality_tier=None)
        df = pd.concat([df, w[df.columns]], ignore_index=True)
    return df


def write_canonical_reports(roots, lexemes, forms, validation, wikt_missing, interim=None):
    _tsv(source_coverage(lexemes, forms, interim), "source_coverage.tsv")
    _tsv(paradigm_coverage(roots, lexemes), "paradigm_coverage.tsv")
    _tsv(per_lexeme_completeness(lexemes, forms), "paradigm_completeness_by_lexeme.tsv")
    _tsv(syncretism_report(forms), "syncretism.tsv")
    _tsv(syncretism_patterns(forms), "syncretism_patterns.tsv")
    _tsv(binyan_distribution(lexemes, forms), "binyan_distribution.tsv")
    _tsv(root_class_distribution(roots, lexemes), "root_class_distribution.tsv")
    _tsv(missing_metadata(roots, lexemes, forms, wikt_missing), "missing_metadata.tsv")
    v = validation.groupby(["record_type", "check_name", "severity"]).size().rename("n").reset_index()
    _tsv(v.sort_values(["severity", "n"], ascending=[True, False]), "validation_summary.tsv")
