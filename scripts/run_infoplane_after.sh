#!/usr/bin/env bash
# Wait for the training chain to finish, then run the information-plane analysis on every run
# of the given sweeps (P analyses in parallel on the local GPU).
# Usage: run_infoplane_after.sh <chain_log> P sweep1 sweep2 ...
cd "$(dirname "$0")/.."
LOG=$1; P=$2; shift 2
until grep -q "CHAIN DONE" "$LOG" 2>/dev/null; do sleep 60; done
PY=${PYTHON:-/home/sr2982/.conda/envs/gpt-env/bin/python}
for s in "$@"; do
  ls -d runs/"$s"/*/ | grep -v logs | xargs -P "$P" -I{} bash -c \
    "$PY -m training.infoplane {} --n 3000 > {}/infoplane.log 2>&1; echo \"\$(date +%T) infoplane done {}\""
done
echo "INFOPLANE DONE"
