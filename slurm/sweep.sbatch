#!/bin/bash
#SBATCH --job-name=ivrit-sweep
#SBATCH --account=adele
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm-%x-%A_%a.out
# Usage: sbatch --array=0-N%8 slurm/sweep.sbatch <sweep_name>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
PY=${PYTHON:-/home/sr2982/.conda/envs/gpt-env/bin/python}
CFG=$(printf "configs/sweeps/%s/%03d.json" "$1" "$SLURM_ARRAY_TASK_ID")
echo "config: $CFG"; cat "$CFG"
$PY -m training.train --config "$CFG"
