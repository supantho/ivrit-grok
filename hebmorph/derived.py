"""CANONICAL -> EXPERIMENTAL VIEWS.  Canonical tables are only read.

A derived record has three kinds of columns:
  * identity (never shown to a model): example_id, pair_id, condition, lexeme_id
  * MODEL-VISIBLE: source_form, target_features, target_form
    (+ source_cell/target_cell, which are equivalent to features)
  * ANALYSIS-ONLY: every column prefixed 'analysis__' (root, binyan, root
    class, tiers, provenance IDs, syncretism ...).  See hebmorph.loader for
    the whitelist that the neural dataloader applies.
"""
from __future__ import annotations

import json

import pandas as pd

from .canonical_utils import TIER_RANK
from .morphology import CITATION_CELL, PAST_CELLS, target_feature_string

CONDITIONS = {"voc": "form_vocalized", "unv": "form_unvocalized"}
MODEL_VISIBLE = ["source_form", "target_features", "target_form"]
IDENTITY = ["example_id", "pair_id", "condition", "lexeme_id", "source_cell", "target_cell"]


ROOT_COLS = ["root_id", "root_normalized", "root_orthographic", "orthographic_root_group_id", "root_length",
             "structural_class", "structural_signature", "traditional_class", "has_weak_radical", "is_geminate"]
TIER_WORST = {v: k for k, v in TIER_RANK.items()}


def build_pairs(roots, lexemes, forms, mode: str, view_name: str) -> pd.DataFrame:
    """Vectorized builder.  mode='citation': PST.3MSG -> every other past cell;
    mode='all': every ordered pair of distinct past cells."""
    rcols = ROOT_COLS + [c for c in roots.columns if c.startswith("sc_")]
    lx = lexemes[["lexeme_id", "root_id", "binyan", "lemma_vocalized", "quality_tier", "source_primary"]].rename(
        columns={"quality_tier": "lexeme_quality_tier", "source_primary": "lexeme_source_primary"})
    lx = lx.merge(roots[rcols], on="root_id", how="left").rename(columns={"root_normalized": "root"})
    fcols = ["lexeme_id", "canonical_cell", "form_id", "quality_tier", "source_ids", "form_vocalized",
             "form_unvocalized", "is_syncretic", "syncretism_group", "is_syncretic_unvocalized",
             "syncretism_group_unvocalized"]
    f = forms.loc[forms.canonical_cell.isin(PAST_CELLS), fcols]
    out = []
    for cond, col in CONDITIONS.items():
        fc = f[f[col].notna()].rename(columns={col: "form"})
        # which past cells share this form within the lexeme (syncretism, per condition)
        same = fc.groupby(["lexeme_id", "form"]).canonical_cell.apply(
            lambda x: "|".join(sorted(x, key=PAST_CELLS.index))).rename("target_same_form_past_cells")
        S = fc[["lexeme_id", "canonical_cell", "form", "form_id", "quality_tier", "source_ids"]].add_prefix("s_")
        T = fc.add_prefix("t_")
        if mode == "citation":
            S = S[S.s_canonical_cell == CITATION_CELL]
        m = S.merge(T, left_on="s_lexeme_id", right_on="t_lexeme_id")
        m = m[m.s_canonical_cell != m.t_canonical_cell]
        m = m.merge(same, left_on=["t_lexeme_id", "t_form"], right_index=True, how="left")
        m = m.merge(lx, left_on="s_lexeme_id", right_on="lexeme_id", how="left")
        sync = "t_is_syncretic" if cond == "voc" else "t_is_syncretic_unvocalized"
        sgrp = "t_syncretism_group" if cond == "voc" else "t_syncretism_group_unvocalized"
        worst = pd.concat([m.lexeme_quality_tier.map(TIER_RANK), m.s_quality_tier.map(TIER_RANK),
                           m.t_quality_tier.map(TIER_RANK)], axis=1).max(axis=1).map(TIER_WORST)
        has_rb = m.root_id.notna() & m.binyan.notna()
        d = pd.DataFrame({
            "example_id": view_name + ":" + cond + ":" + m.lexeme_id + ":" + m.s_canonical_cell + ">" + m.t_canonical_cell,
            "pair_id": m.lexeme_id + ":" + m.s_canonical_cell + ">" + m.t_canonical_cell,
            "condition": cond, "lexeme_id": m.lexeme_id, "source_form": m.s_form,
            "source_cell": m.s_canonical_cell, "target_cell": m.t_canonical_cell,
            "target_features": m.t_canonical_cell.map(target_feature_string), "target_form": m.t_form,
        })
        a = {
            "root_id": m.root_id, "root": m.root, "root_orthographic": m.root_orthographic,
            "orthographic_root_group_id": m.orthographic_root_group_id, "root_length": m.root_length,
            "binyan": m.binyan, "root_binyan_key": (m.root_id.astype("string") + "|" + m.binyan.astype("string")).where(has_rb),
            "structural_class": m.structural_class, "structural_signature": m.structural_signature,
            "traditional_class": m.traditional_class,
            **{c: m[c] for c in rcols if c.startswith("sc_")},
            "lemma_vocalized": m.lemma_vocalized, "lexeme_quality_tier": m.lexeme_quality_tier,
            "source_form_quality_tier": m.s_quality_tier, "target_form_quality_tier": m.t_quality_tier,
            "example_quality_tier": worst,
            "eligible_default": (worst != "QUESTIONABLE") & has_rb,
            "source_form_id": m.s_form_id, "target_form_id": m.t_form_id,
            "source_ids": m.s_source_ids.fillna("") + "|" + m.t_source_ids.fillna(""),
            "lexeme_source_primary": m.lexeme_source_primary,
            "target_is_syncretic": m[sync].astype(bool), "target_syncretism_group": m[sgrp],
            "target_same_form_past_cells": m.target_same_form_past_cells,
            "source_equals_target": m.s_form == m.t_form, "target_form_length": m.t_form.str.len(),
        }
        for k, v in a.items():
            d["analysis__" + k] = v.values if hasattr(v, "values") else v
        out.append(d)
    df = pd.concat(out, ignore_index=True)
    rank = {c: i for i, c in enumerate(PAST_CELLS)}
    df = df.sort_values(["lexeme_id", "condition"], kind="stable").reset_index(drop=True)
    df["analysis__eligible_default"] = df["analysis__eligible_default"].astype(bool)
    # plain object columns: pandas 'string' dtype is very slow to iterate
    for c in df.columns:
        if str(df[c].dtype) == "string":
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df


def past_reinflection(roots, lexemes, forms) -> pd.DataFrame:
    return build_pairs(roots, lexemes, forms, "citation", "PR")


def full_paradigm(roots, lexemes, forms) -> pd.DataFrame:
    """All ordered (source, target) pairs of distinct past cells."""
    return build_pairs(roots, lexemes, forms, "all", "FP")


def write_view(df: pd.DataFrame, outdir, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outdir / f"{name}.parquet", index=False)
    df.to_csv(outdir / f"{name}.tsv", sep="\t", index=False)
    acols = [c for c in df.columns if c.startswith("analysis__")]
    with open(outdir / f"{name}.jsonl", "w", encoding="utf-8") as fh:
        for rec in df.to_dict("records"):
            out = {k: rec[k] for k in IDENTITY + MODEL_VISIBLE}
            out["analysis"] = {k[len("analysis__"):]: (None if (isinstance(rec[k], float) and pd.isna(rec[k])) else rec[k])
                               for k in acols}
            fh.write(json.dumps(out, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    if o is pd.NA:
        return None
    return str(o)
