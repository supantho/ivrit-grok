"""INTERIM -> CANONICAL (+ conflict/duplicate/missing reports).  Fails loudly on schema violations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hebmorph import paths  # noqa: E402
from hebmorph.build import build_real  # noqa: E402
from hebmorph.io import write_table  # noqa: E402
from hebmorph.schema import check_foreign_keys, enforce, schema_json  # noqa: E402

if __name__ == "__main__":
    out = build_real()
    tables = {t: enforce(out[t], t) for t in ["roots", "lexemes", "forms", "provenance", "validation"]}
    check_foreign_keys(tables["roots"], tables["lexemes"], tables["forms"], tables["provenance"])
    paths.CANONICAL.mkdir(parents=True, exist_ok=True)
    for t, df in tables.items():
        write_table(df, paths.CANONICAL / t, tsv=True)
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    for name in ["conflicts", "duplicates"]:
        out[name].to_csv(paths.REPORTS / f"{name}.tsv", sep="\t", index=False)
    from hebmorph.reports import write_canonical_reports
    write_canonical_reports(tables["roots"], tables["lexemes"], tables["forms"], tables["validation"], out["missing"])
    (paths.METADATA / "schema.json").write_text(schema_json())
    for t, df in tables.items():
        print(f"{t:12s} {len(df):7d} rows")
    print("conflicts", len(out["conflicts"]), "duplicates", len(out["duplicates"]))
