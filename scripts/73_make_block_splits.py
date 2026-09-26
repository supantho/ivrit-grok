"""Block-holdout splits for v2 tables: the split unit is a whole (root, template, tense) block.

Why: with random-cell splits, most unseen weak forms can be copied from a sibling form of the
same root, template and tense (seeing יִפֹּל gives away תִּפֹּל), so the model generalizes early
and gradually instead of having to discover the weak-root RULE. Holding out whole blocks
removes that shortcut: a test form's stem must come from (the root's letters, learned from its
other blocks) x (the class rule for that template x tense, learned from other roots).

Guarantee (asserted): every root, every template, and every (template, cell) appear in training.
    python scripts/73_make_block_splits.py syn2_w000 syn2_w060 syn2_w100
Writes splits/block<F>_seed<S>/manifest.json next to the existing random-cell splits.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hebmorph.loader import table_dir  # noqa: E402

FRACTIONS = [0.2, 0.3, 0.5]
SEEDS = [0, 1]

for name in sys.argv[1:]:
    d = table_dir(name)
    tab = pd.read_parquet(d / "table.parquet")
    tab["block"] = tab.root_symbol + "|" + tab.template_symbol + "|" + tab.analysis__tense
    blocks = np.array(sorted(tab.block.unique()))
    for frac in FRACTIONS:
        for seed in SEEDS:
            for attempt in range(1000):   # resample until every root/template/(template,cell) is in train
                r = np.random.default_rng(100_000 + 10_000 * seed + int(frac * 1000) + attempt * 7919)
                perm = r.permutation(len(blocks))
                trb = set(blocks[perm[: int(round(frac * len(blocks)))]])
                tr = tab[tab.block.isin(trb)]
                if (tr.root_symbol.nunique() == tab.root_symbol.nunique()
                        and tr.template_symbol.nunique() == tab.template_symbol.nunique()
                        and len(tr.groupby(["template_symbol", "cell_symbol"])) ==
                        len(tab.groupby(["template_symbol", "cell_symbol"]))):
                    break
            else:
                raise RuntimeError(f"{name} frac={frac} seed={seed}: could not satisfy coverage")
            rest = tab[~tab.block.isin(trb)]
            rest_blocks = np.array(sorted(rest.block.unique()))
            rr = np.random.default_rng(seed + 11).permutation(len(rest_blocks))
            dev_b = set(rest_blocks[rr[: len(rest_blocks) // 10]])
            dev = rest[rest.block.isin(dev_b)]
            te = rest[~rest.block.isin(dev_b)]
            # leakage checks: no block on two sides
            assert not (set(tr.block) & set(te.block)) and not (set(tr.block) & set(dev.block))
            m = dict(split_name=f"block{int(frac * 100):02d}_seed{seed}", strategy="table_block_holdout", seed=seed,
                     params=dict(block_fraction=frac, block_unit="root x template x tense", table=name,
                                 n_blocks=len(blocks), resample_attempts=attempt + 1),
                     partitions={"train": dict(pair_ids=sorted(tr.eq_id)), "dev": dict(pair_ids=sorted(dev.eq_id)),
                                 "test": dict(pair_ids=sorted(te.eq_id))})
            out = d / "splits" / m["split_name"]
            out.mkdir(parents=True, exist_ok=True)
            (out / "manifest.json").write_text(json.dumps(m))
            print(name, m["split_name"], {k: len(v["pair_ids"]) for k, v in m["partitions"].items()},
                  "attempts", attempt + 1)
