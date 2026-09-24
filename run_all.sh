#!/usr/bin/env bash
# Rebuild the entire dataset from raw sources. Every step is deterministic.
set -euo pipefail
cd "$(dirname "$0")"
bash   scripts/00_fetch_sources.sh        # raw sources at pinned versions (verified)
python scripts/10_ingest.py               # RAW -> INTERIM
python scripts/20_build_canonical.py      # INTERIM -> CANONICAL (+ conflict/QC reports); fails on schema violations
python scripts/30_derive_and_split.py     # CANONICAL -> derived views + verified split manifests
python scripts/40_synthetic.py            # synthetic v1 (separate) + its views and splits
python scripts/50_summary_and_manifest.py # corpus_summary.md, SCHEMA.md, dataset_manifest.json
python scripts/60_inspect_paradigms.py 24 # reports/paradigm_inspection.md (manual review sheet)
python -m pytest -q                       # unit + integration + split-leakage tests
