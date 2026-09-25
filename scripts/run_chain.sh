#!/usr/bin/env bash
# Run several sweeps back to back on the local GPU. Usage: run_chain.sh P sweep1 sweep2 ...
cd "$(dirname "$0")/.."
P=$1; shift
for s in "$@"; do bash scripts/run_sweep_local.sh "$s" "$P"; done
echo "CHAIN DONE"
