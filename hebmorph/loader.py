"""Model-facing dataloader.  Exposes ONLY source_form, target_features and
target_form.  Everything else (root, binyan, root class, lexeme IDs, tiers,
provenance) stays behind this interface.

    from hebmorph.loader import load_split
    train = load_split("past_reinflection", "root_holdout_seed0", "train", condition="voc")
    for ex in train: ex.source_form, ex.target_features, ex.target_form
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import paths

VISIBLE_FIELDS = ("source_form", "target_features", "target_form")


@dataclass(frozen=True)
class Example:
    source_form: str
    target_features: str
    target_form: str


def _view_path(view: str, base: Path | None) -> Path:
    base = base or paths.DERIVED
    return base / view / f"{view}.parquet"


def load_split(view: str, split: str, partition: str, condition: str = "voc",
               derived_base: Path | None = None, splits_base: Path | None = None) -> list[Example]:
    splits_base = splits_base or paths.SPLITS
    manifest = json.loads((splits_base / view / split / "manifest.json").read_text())
    if partition not in manifest["partitions"]:
        raise KeyError(f"{partition} not in {list(manifest['partitions'])}")
    ids = set(manifest["partitions"][partition]["pair_ids"])
    df = pd.read_parquet(_view_path(view, derived_base), columns=["pair_id", "condition", *VISIBLE_FIELDS])
    df = df[(df.condition == condition) & df.pair_id.isin(ids)]
    df = df[list(VISIBLE_FIELDS)]  # hard whitelist
    return [Example(*row) for row in df.itertuples(index=False, name=None)]


def load_split_analysis(view: str, split: str, partition: str, condition: str = "voc",
                        derived_base: Path | None = None, splits_base: Path | None = None) -> pd.DataFrame:
    """ANALYSIS ONLY -- never feed this to a model.  Same rows, in the same order, as
    load_split(), plus identity and analysis__* metadata (root, binyan, cell, ...),
    for probing / information-theoretic analysis of trained models."""
    splits_base = splits_base or paths.SPLITS
    manifest = json.loads((splits_base / view / split / "manifest.json").read_text())
    ids = set(manifest["partitions"][partition]["pair_ids"])
    df = pd.read_parquet(_view_path(view, derived_base))
    df = df[(df.condition == condition) & df.pair_id.isin(ids)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Table-completion task (Power et al.-style): opaque root/template/cell symbols
# ---------------------------------------------------------------------------
TABLE_VISIBLE = ("root_symbol", "template_symbol", "cell_symbol", "target_form")


@dataclass(frozen=True)
class TableExample:
    root_symbol: str       # opaque, e.g. 'R017' -- carries no letters
    template_symbol: str   # opaque, e.g. 'T3'
    cell_symbol: str       # opaque, e.g. 'C4'
    target_form: str


def table_dir(name: str) -> Path:
    return paths.DATA / "experiments" / "table_completion" / name


def load_table_split(name: str, split: str, partition: str, condition: str = "voc") -> list[TableExample]:
    d = table_dir(name)
    m = json.loads((d / "splits" / split / "manifest.json").read_text())
    ids = set(m["partitions"][partition]["pair_ids"])
    col = {"voc": "target_voc", "unv": "target_unv"}[condition]
    df = pd.read_parquet(d / "table.parquet", columns=["eq_id", "root_symbol", "template_symbol", "cell_symbol", col])
    df = df[df.eq_id.isin(ids)]
    return [TableExample(*r) for r in df[["root_symbol", "template_symbol", "cell_symbol", col]].itertuples(index=False, name=None)]


def load_table_split_analysis(name: str, split: str, partition: str, condition: str = "voc") -> pd.DataFrame:
    """ANALYSIS ONLY: same rows/order as load_table_split plus analysis__* columns."""
    d = table_dir(name)
    m = json.loads((d / "splits" / split / "manifest.json").read_text())
    ids = set(m["partitions"][partition]["pair_ids"])
    df = pd.read_parquet(d / "table.parquet")
    df = df[df.eq_id.isin(ids)].reset_index(drop=True)
    df["target_form"] = df["target_voc" if condition == "voc" else "target_unv"]
    return df
