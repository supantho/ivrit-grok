"""CANONICAL -> derived views (past_reinflection, full_paradigm) + verified split manifests."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from hebmorph import paths  # noqa: E402
from hebmorph.run_views import build_views_and_splits  # noqa: E402

if __name__ == "__main__":
    C = paths.CANONICAL
    roots, lexemes, forms = (pd.read_parquet(C / f"{t}.parquet") for t in ["roots", "lexemes", "forms"])
    for d in [paths.SPLITS / "past_reinflection", paths.SPLITS / "full_paradigm"]:
        if d.exists():
            shutil.rmtree(d)  # manifests are fully regenerated from canonical data
    summary, views = build_views_and_splits(roots, lexemes, forms, paths.DERIVED, paths.SPLITS)
    summary.to_csv(paths.REPORTS / "split_summary.tsv", sep="\t", index=False)
    for name, v in views.items():
        print(name, len(v), "rows;", int(v.analysis__eligible_default.sum()), "eligible")
    print(summary.groupby("view").split.nunique())
