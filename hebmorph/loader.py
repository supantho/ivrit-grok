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
