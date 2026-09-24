"""Generate a grokking sweep (one JSON config per run) + a Slurm array to run it.

Regime follows Power et al. (2022): 2-layer / d=128 / 4-head decoder-only transformer,
AdamW lr=1e-3, betas (0.9, 0.98), weight decay 1.0, full-batch, long training, small
training fraction of the table; evaluate exact-match on held-out cells.

    python scripts/make_sweep.py grok_v1
    sbatch --array=0-$(( $(ls configs/sweeps/grok_v1/*.json | wc -l) - 1 ))%8 slurm/sweep.sbatch grok_v1
"""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1] if len(sys.argv) > 1 else "grok_v1"
out = ROOT / "configs" / "sweeps" / name
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.json"):
    f.unlink()

BASE = dict(d_model=128, n_layers=2, n_heads=4, d_mlp=512, lr=1e-3, betas=[0.9, 0.98], weight_decay=1.0,
            decay_norm_and_bias=True, use_layernorm=True, warmup_steps=10, batch_size=-1, steps=50000, eval_every=500, eval_max_examples=1024,
            n_checkpoints=40, max_len=64, seed=0)
GRID = []
# synthetic: small training fractions of the 7,000-lexeme x 8-cell table
for split, cond, n in itertools.product(["iid_seed0", "root_holdout_seed0"], ["voc", "unv"], [250, 500, 1000, 2000]):
    GRID.append(dict(dataset="synthetic_v1", split=split, condition=cond, train_subsample=n))
# real Hebrew (frozen v0.1): all eligible training data vs a 1,000-example subsample
for split, cond, n in itertools.product(["iid_seed0", "root_holdout_seed0"], ["voc", "unv"], [None, 1000]):
    GRID.append(dict(dataset="real", split=split, condition=cond, train_subsample=n))

for i, g in enumerate(GRID):
    ds = "syn" if g["dataset"] == "synthetic_v1" else "real"
    n = g["train_subsample"] or "all"
    run = f"{name}/{ds}_{g['split'].replace('_seed0', '')}_{g['condition']}_n{n}"
    cfg = dict(BASE, **g, run_name=run)
    (out / f"{i:03d}.json").write_text(json.dumps(cfg, indent=1))
print(f"{len(GRID)} configs in {out}")
