"""Positive control: modular arithmetic in the exact format of Power et al. (2022).

Equation:  <bos> <a> <op> <b> = <c>   with c = (a op b) mod p, p = 97.
Stored in the table-completion format so the SAME trainer is used as for Hebrew:
  root_symbol = a ('N017'), template_symbol = op ('OP+'), cell_symbol = b ('N042').
a and b share one token vocabulary (same symbol for a residue on either side), as in the paper.
The answer c is ONE token: it is encoded as a single Unicode code point (U+4E00 + c), so the
character-level trainer sees exactly one answer token (+ <eos>), like the paper's single c.

    python scripts/71_make_modular_task.py [op=add] [p=97]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hebmorph import paths  # noqa: E402

OP = sys.argv[1] if len(sys.argv) > 1 else "add"
P = int(sys.argv[2]) if len(sys.argv) > 2 else 97
FRACTIONS = [0.3, 0.4, 0.5, 0.7]
SEEDS = [0, 1, 2]
fn = {"add": lambda a, b: (a + b) % P, "sub": lambda a, b: (a - b) % P,
      "div": lambda a, b: (a * pow(b, P - 2, P)) % P}[OP]
name = f"mod{P}_{OP}"
out = paths.DATA / "experiments" / "table_completion" / name
(out / "splits").mkdir(parents=True, exist_ok=True)

rows = []
for a in range(P):
    for b in range(P):
        if OP == "div" and b == 0:
            continue
        c = fn(a, b)
        rows.append(dict(eq_id=f"N{a:03d}|OP{OP}|N{b:03d}", root_symbol=f"N{a:03d}", template_symbol=f"OP{OP}",
                         cell_symbol=f"N{b:03d}", target_voc=chr(0x4E00 + c), target_unv=chr(0x4E00 + c),
                         # analysis columns (same names as the Hebrew table so the analysis code runs unchanged)
                         analysis__root=str(a), analysis__binyan=OP, analysis__cell=str(b),
                         analysis__r1=str(c), analysis__r2=str(c), analysis__r3=str(c), analysis__c=c))
tab = pd.DataFrame(rows)
tab.to_parquet(out / "table.parquet", index=False)
for frac in FRACTIONS:
    for seed in SEEDS:
        r = np.random.default_rng(10_000 * seed + int(round(frac * 1000)))
        perm = r.permutation(len(tab))
        n_tr = int(round(frac * len(tab)))
        tr, rest = tab.eq_id.values[np.sort(perm[:n_tr])], tab.eq_id.values[np.sort(perm[n_tr:])]
        # Power et al.: all non-training equations are validation. We keep a small dev slice for
        # consistency with the Hebrew splits; 'test' = the rest.
        n_dev = len(rest) // 10
        rr = np.random.default_rng(seed + 7).permutation(len(rest))
        m = dict(split_name=f"frac{int(frac * 100):02d}_seed{seed}", strategy="table_fraction", seed=seed,
                 params=dict(train_fraction=frac, table=name, n_equations=len(tab)),
                 partitions={"train": dict(pair_ids=sorted(tr.tolist())),
                             "dev": dict(pair_ids=sorted(rest[np.sort(rr[:n_dev])].tolist())),
                             "test": dict(pair_ids=sorted(rest[np.sort(rr[n_dev:])].tolist()))})
        d = out / "splits" / m["split_name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(m))
(out / "meta.json").write_text(json.dumps(dict(name=name, p=P, op=OP, n_equations=len(tab),
                                               fractions=FRACTIONS, seeds=SEEDS), indent=1))
print(name, len(tab), "equations")
