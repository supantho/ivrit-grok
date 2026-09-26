#!/usr/bin/env bash
# Wait until <log> contains "CHAIN DONE", then run a chain of sweeps.
# Usage: run_after.sh <wait_log> P sweep1 [sweep2 ...]
cd "$(dirname "$0")/.."
LOG=$1; shift
until grep -q "CHAIN DONE" "$LOG" 2>/dev/null; do sleep 60; done
bash scripts/run_chain.sh "$@"
