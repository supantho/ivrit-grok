#!/bin/bash
#SBATCH --job-name=ivrit-grok
#SBATCH --account=adele
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm-%x-%j.out
# Usage: sbatch slurm/train.sbatch configs/syn_root_holdout.json [key=value ...]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p runs
PY=${PYTHON:-/home/sr2982/.conda/envs/gpt-env/bin/python}
CFG=$1; shift
nvidia-smi --query-gpu=name,memory.total --format=csv || true
$PY -m training.train --config "$CFG" ${@:+--set "$@"}
