"""v2 sweep: weak-root share x training fraction x init scale x seed (1-layer transformer,
the settings that grokked on mod-97: lr 3e-4, wd 1, Power batch rule, torch-default init)."""
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
name = "v2_transformer"
out = ROOT / "configs" / "sweeps" / name
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.json"):
    f.unlink()
BASE = dict(condition="voc", max_len=40, arch="transformer", n_layers=1, d_model=128, n_heads=4, d_mlp=512,
            init="torch_default", lr=3e-4, betas=[0.9, 0.98], weight_decay=1.0, decay_norm_and_bias=True,
            warmup_steps=10, batch_rule="power", steps=100000, eval_every=2000, eval_log_points=60,
            eval_max_examples=None, exact_match_method="teacher_forcing", n_checkpoints=40, ckpt_every=2000)
i = 0
for weak, frac, alpha, seed in itertools.product(["000", "030", "060", "100"], [0.2, 0.3, 0.5], [1.0, 3.0], [0, 1]):
    c = dict(BASE, table=f"syn2_w{weak}", split=f"frac{int(frac * 100):02d}_seed{seed}", init_scale=alpha, seed=seed,
             run_name=f"{name}/w{weak}_f{int(frac * 100):02d}_a{alpha:g}_s{seed}")
    (out / f"{i:03d}.json").write_text(json.dumps(c, indent=1))
    i += 1
print(i, "configs ->", out)
