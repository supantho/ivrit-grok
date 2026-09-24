"""Synthetic dataset v1: canonical tables (same schema) + derived views + verified splits.
Usage: python scripts/40_synthetic.py [config.json]   (default config if omitted)"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hebmorph import paths  # noqa: E402
from hebmorph.io import write_table  # noqa: E402
from hebmorph.run_views import build_views_and_splits  # noqa: E402
from hebmorph.schema import check_foreign_keys, enforce  # noqa: E402
from hebmorph.synthetic import SyntheticConfig, generate  # noqa: E402

if __name__ == "__main__":
    cfg = SyntheticConfig(**json.loads(Path(sys.argv[1]).read_text())) if len(sys.argv) > 1 else SyntheticConfig()
    out = paths.DERIVED / "synthetic" / cfg.version
    if out.exists():
        shutil.rmtree(out)
    (out / "canonical").mkdir(parents=True)
    (out / "synthetic_config.json").write_text(cfg.to_json())
    t = generate(cfg)
    relax = frozenset({"binyan"})
    tables = {k: enforce(t[k], k, relax_enums=relax) for k in ["roots", "lexemes", "forms", "provenance", "validation"]}
    check_foreign_keys(tables["roots"], tables["lexemes"], tables["forms"], tables["provenance"])
    for k, df in tables.items():
        write_table(df, out / "canonical" / k)
    summary, views = build_views_and_splits(tables["roots"], tables["lexemes"], tables["forms"], out, out / "splits",
                                            seeds=[0, 1, 2], with_trad=False)
    summary.to_csv(out / "split_summary.tsv", sep="\t", index=False)
    for k, df in tables.items():
        print(f"{k:12s} {len(df):8d}")
    for k, v in views.items():
        print(k, len(v))
    print(summary.groupby("view").split.nunique())
