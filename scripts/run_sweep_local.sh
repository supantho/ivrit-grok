#!/usr/bin/env bash
# Run every config of a sweep on the local GPU, P runs concurrently (tiny models share one GPU).
# Usage: nohup setsid bash scripts/run_sweep_local.sh grok_v2 6 > runs/grok_v2_local.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
SWEEP=$1; P=${2:-6}
PY=${PYTHON:-/home/sr2982/.conda/envs/gpt-env/bin/python}
mkdir -p "runs/$SWEEP/logs"
# one runner per sweep: a second launch would write into the same run directories
exec 9>"runs/$SWEEP/.lock"
if ! flock -n 9; then echo "ERROR: sweep $SWEEP is already running (lock held)"; exit 1; fi
ls configs/sweeps/"$SWEEP"/*.json | xargs -P "$P" -I{} bash -c \
  'f={}; n=$(basename $f .json); echo "$(date +%T) start $n"; '"$PY"' -m training.train --config $f > runs/'"$SWEEP"'/logs/$n.log 2>&1; echo "$(date +%T) done $n exit=$?"'
echo "ALL DONE"
