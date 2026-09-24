"""Table I/O: Parquet (canonical) + UTF-8 TSV (human-readable)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, stem: Path, tsv: bool = True) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(stem.with_suffix(".parquet"), index=False)
    if tsv:
        # TSV: tabs/newlines inside values would break the format -> escape them visibly
        safe = df.copy()
        for c in safe.columns:
            if safe[c].dtype == object or str(safe[c].dtype) == "string":
                safe[c] = safe[c].astype("string").str.replace("\t", "\\t", regex=False).str.replace("\n", "\\n", regex=False)
        safe.to_csv(stem.with_suffix(".tsv"), sep="\t", index=False, encoding="utf-8")
