"""Table-completion task (the direct analogue of Power et al. 2022's binary-operation tables).

Each equation is  <ROOT_k> <TMPL_j> <CELL> = <target form characters>,
where ROOT_k and TMPL_j are OPAQUE symbols (randomly permuted indices, carrying no
letters), exactly like the residue symbols a, b in "a ∘ b = c". To produce the form the
model must infer each root's radicals and each template's pattern from the other cells
of the table in which they appear.

Table: n_roots roots x all templates x 9 past cells, drawn deterministically from the
FROZEN synthetic v1 canonical tables (150 x 7 x 9 = 9,450 ≈ 97^2 = 9,409 equations).
Splits: a random fraction of all equations is training, the rest validation/test
(Power et al. "chose a fraction of all available equations at random").
Hidden from models: radicals, template (binyan) names, root ids -- analysis columns only.

    python scripts/70_make_table_task.py [n_roots=150] [name=syn_r150]
Writes data/experiments/table_completion/<name>/{table.parquet, table.tsv, meta.json,
splits/frac<F>_seed<S>/manifest.json}.  Does not modify the frozen v0.1 release.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hebmorph import paths  # noqa: E402
from hebmorph.morphology import PAST_CELLS  # noqa: E402
from hebmorph.roots import split_root_units  # noqa: E402

N_ROOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
NAME = sys.argv[2] if len(sys.argv) > 2 else f"syn_r{N_ROOTS}"
FRACTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SEEDS = [0, 1, 2]
SELECT_SEED = 2201  # fixed: which roots form the table (arXiv id of Power et al.)

syn = paths.DERIVED / "synthetic" / "v1" / "canonical"
forms = pd.read_parquet(syn / "forms.parquet")
roots = pd.read_parquet(syn / "roots.parquet")
out = paths.DATA / "experiments" / "table_completion" / NAME
(out / "splits").mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SELECT_SEED)
root_ids = sorted(roots.root_id)
chosen = sorted(rng.choice(root_ids, size=N_ROOTS, replace=False))
templates = sorted(forms.binyan.unique())
# opaque symbols: random permutations so symbol index carries no information
root_sym = {r: f"R{i:03d}" for i, r in enumerate(rng.permutation(chosen))}
tmpl_sym = {t: f"T{i}" for i, t in enumerate(rng.permutation(templates))}
cell_sym = {c: f"C{i}" for i, c in enumerate(PAST_CELLS)}   # cells are the operation's 3rd operand

f = forms[forms.root_id.isin(chosen) & forms.is_past_cell].merge(
    roots[["root_id", "root_normalized"]], on="root_id")
tab = pd.DataFrame({
    "eq_id": [f"{root_sym[r]}|{tmpl_sym[b]}|{cell_sym[c]}" for r, b, c in zip(f.root_id, f.binyan, f.canonical_cell)],
    # ---- MODEL-VISIBLE ----
    "root_symbol": f.root_id.map(root_sym), "template_symbol": f.binyan.map(tmpl_sym),
    "cell_symbol": f.canonical_cell.map(cell_sym),
    "target_voc": f.form_vocalized, "target_unv": f.form_unvocalized,
    # ---- ANALYSIS ONLY ----
    "analysis__root_id": f.root_id, "analysis__root": f.root_normalized,
    "analysis__binyan": f.binyan, "analysis__cell": f.canonical_cell, "analysis__form_id": f.form_id,
})
rad = tab.analysis__root.map(split_root_units)
for k in range(3):
    tab[f"analysis__r{k + 1}"] = rad.map(lambda x, k=k: x[k])
tab = tab.sort_values("eq_id").reset_index(drop=True)
assert len(tab) == N_ROOTS * len(templates) * len(PAST_CELLS) and not tab.eq_id.duplicated().any()
tab.to_parquet(out / "table.parquet", index=False)
tab.to_csv(out / "table.tsv", sep="\t", index=False)

for frac in FRACTIONS:
    for seed in SEEDS:
        r = np.random.default_rng(10_000 * seed + int(round(frac * 1000)))
        perm = r.permutation(len(tab))
        n_tr = int(round(frac * len(tab)))
        tr = tab.eq_id.values[np.sort(perm[:n_tr])]
        rest = tab.eq_id.values[np.sort(perm[n_tr:])]
        # held-out equations split into dev (10% of table, for model selection) and test
        n_dev = min(len(rest) // 5, int(round(0.1 * len(tab))))
        rr = np.random.default_rng(seed + 7).permutation(len(rest))
        dev, te = rest[np.sort(rr[:n_dev])], rest[np.sort(rr[n_dev:])]
        seen = tab[tab.eq_id.isin(set(tr))]
        m = dict(split_name=f"frac{int(frac * 100):02d}_seed{seed}", strategy="table_fraction", seed=seed,
                 params=dict(train_fraction=frac, table=NAME, n_equations=len(tab)),
                 grouping_level="equation (root x template x cell)",
                 partitions={p: dict(pair_ids=sorted(ids.tolist())) for p, ids in
                             [("train", tr), ("dev", dev), ("test", te)]},
                 coverage=dict(roots_in_train=int(seen.root_symbol.nunique()),
                               templates_in_train=int(seen.template_symbol.nunique()),
                               cells_in_train=int(seen.cell_symbol.nunique())))
        assert not (set(tr) & set(te)) and not (set(tr) & set(dev)) and not (set(dev) & set(te))
        d = out / "splits" / m["split_name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(m, indent=1))

meta = dict(name=NAME, n_roots=N_ROOTS, templates=len(templates), cells=len(PAST_CELLS), n_equations=len(tab),
            select_seed=SELECT_SEED, source="data/derived/synthetic/v1/canonical (frozen HEB_MORPH_GROK_v0.1)",
            fractions=FRACTIONS, seeds=SEEDS,
            table_sha256=hashlib.sha256((out / "table.parquet").read_bytes()).hexdigest())
(out / "meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta, indent=1))
print(tab.head(3).T)
