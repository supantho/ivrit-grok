"""v3: large-init (Omnigrok alpha) x split type (random-cell vs block-holdout) x weak share x data fraction.
1-layer transformer, same optimisation as v2 (the setting that grokked on mod-97)."""
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
name = "v3_alpha"
out = ROOT / "configs" / "sweeps" / name
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.json"):
    f.unlink()
BASE = json.loads((ROOT / "configs" / "sweeps" / "v2_transformer" / "000.json").read_text())
i = 0
for weak, split, frac, alpha, seed in itertools.product(["000", "100"], ["frac", "block"], [20, 50], [1.0, 3.0, 10.0, 30.0], [0, 1]):
    c = dict(BASE, table=f"syn2_w{weak}", split=f"{split}{frac:02d}_seed{seed}", init_scale=alpha, seed=seed,
             run_name=f"{name}/w{weak}_{split}{frac:02d}_a{alpha:g}_s{seed}")
    (out / f"{i:03d}.json").write_text(json.dumps(c, indent=1))
    i += 1
print(i, "configs")
