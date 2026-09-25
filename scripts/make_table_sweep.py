"""Table-completion grokking sweep (Power et al. 2022 protocol, Omnigrok init lever).
    python scripts/make_table_sweep.py <sweep_name> <arch> [table=syn_r150]
Grid: train fraction x init scale alpha x seed.  One JSON config per run."""
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
name, arch = sys.argv[1], sys.argv[2]
table = sys.argv[3] if len(sys.argv) > 3 else "syn_r150"
out = ROOT / "configs" / "sweeps" / name
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.json"):
    f.unlink()
ARCH = {  # ~1-layer members of the model family
    "transformer": dict(arch="transformer", n_layers=1, d_model=128, n_heads=4, d_mlp=512, init="torch_default"),
    "lstm": dict(arch="lstm", n_layers=1, d_emb=128, d_model=128),
    "mlp": dict(arch="mlp", n_layers=1, d_emb=32, d_mlp=512),
}[arch]
BASE = dict(table=table, condition="voc", max_len=32,
            # Power et al. (2022) A.1.2, except lr lowered 1e-3 -> 3e-4 to suppress loss-spike collapses
            lr=3e-4, betas=[0.9, 0.98], weight_decay=1.0, decay_norm_and_bias=True, warmup_steps=10,
            batch_rule="power", steps=100000, grad_clip=1.0,
            eval_every=2000, eval_log_points=60, eval_max_examples=None, exact_match_method="teacher_forcing",
            n_checkpoints=40, ckpt_every=2000, deterministic=False)
i = 0
for frac, alpha, seed in itertools.product([0.2, 0.3, 0.5, 0.7], [1.0, 3.0, 6.0], [0, 1]):
    cfg = dict(BASE, **ARCH, split=f"frac{int(frac * 100):02d}_seed{seed}", init_scale=alpha, seed=seed,
               run_name=f"{name}/{arch}_f{int(frac * 100):02d}_a{alpha:g}_s{seed}")
    (out / f"{i:03d}.json").write_text(json.dumps(cfg, indent=1))
    i += 1
print(f"{i} configs -> {out}")
