"""Synthetic v2 table-completion task: strong + rule-governed weak roots, all tenses.

    python scripts/72_make_synthetic_v2_table.py --name syn2_w60 --n_roots 60 --weak_fraction 0.6 \
        [--tenses PST PRS FUT IMP] [--classes PE_NUN PE_YOD LAMED_HE HOLLOW GEMINATE] [--seed 0]

Equation: <ROOT_k> <TMPL_j> <CELL_c> = form   (opaque symbols; ROOT carries no letters).
Weak roots are split evenly across the chosen classes; the rest are strong.  Table size =
n_roots x sum over binyanim of the cells they have (PUAL/HUFAL have no imperative).
Analysis-only columns: root, radicals, binyan, cell, tense, root class.
Writes data/experiments/table_completion/<name>/ (table.parquet, meta.json, splits/).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hebmorph import paths  # noqa: E402
from hebmorph.hebrew import strip_niqqud  # noqa: E402
from hebmorph.synthetic_v2 import BINYANIM, classify, realize, templates  # noqa: E402

BASE = ["ב", "ג", "ד", "ז", "ט", "כ", "ל", "מ", "ס", "פ", "צ", "ק", "ר", "שׁ", "ת"]
CLASSES = ["PE_NUN", "PE_YOD", "LAMED_HE", "HOLLOW", "GEMINATE"]


def sample_root(cls, rng):
    a, b, c = (BASE[i] for i in rng.choice(len(BASE), 3, replace=False))
    return {"STRONG": [a, b, c], "PE_NUN": ["נ", b, c], "PE_YOD": ["י", b, c], "LAMED_HE": [a, b, "ה"],
            "HOLLOW": [a, "ו", c], "GEMINATE": [a, b, b]}[cls]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n_roots", type=int, default=60)
    ap.add_argument("--weak_fraction", type=float, default=0.6)
    ap.add_argument("--tenses", nargs="*", default=["PST", "PRS", "FUT", "IMP"])
    ap.add_argument("--classes", nargs="*", default=CLASSES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fractions", nargs="*", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.7])
    ap.add_argument("--split_seeds", nargs="*", type=int, default=[0, 1, 2])
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n_weak = int(round(a.weak_fraction * a.n_roots))
    plan = ["STRONG"] * (a.n_roots - n_weak) + [a.classes[i % len(a.classes)] for i in range(n_weak)]
    roots, seen = [], set()
    for cls in plan:
        while True:
            r = sample_root(cls, rng)
            if "".join(r) not in seen and classify(r) == cls:
                seen.add("".join(r))
                roots.append(r)
                break
    order = rng.permutation(len(roots))
    tmpl_perm = rng.permutation(len(BINYANIM))
    rows = []
    for k, ri in enumerate(order):
        rad = roots[ri]
        for j, bi in enumerate(tmpl_perm):
            b = BINYANIM[bi]
            for cell, t in templates(b, tuple(a.tenses)).items():
                form = realize(t, rad, b, cell)
                rows.append(dict(root_symbol=f"R{k:03d}", template_symbol=f"T{j}", cell=cell, target_voc=form,
                                 target_unv=strip_niqqud(form), analysis__root="".join(rad), analysis__binyan=b,
                                 analysis__cell=cell, analysis__tense=cell.split(".")[0],
                                 analysis__class=classify(rad),
                                 analysis__r1=rad[0], analysis__r2=rad[1], analysis__r3=rad[2]))
    tab = pd.DataFrame(rows)
    cells = sorted(tab.cell.unique())
    cperm = rng.permutation(len(cells))
    csym = {c: f"C{cperm[i]:02d}" for i, c in enumerate(cells)}
    tab["cell_symbol"] = tab.cell.map(csym)
    tab["eq_id"] = tab.root_symbol + "|" + tab.template_symbol + "|" + tab.cell_symbol
    tab = tab.drop(columns="cell").sort_values("eq_id").reset_index(drop=True)
    assert not tab.eq_id.duplicated().any()
    out = paths.DATA / "experiments" / "table_completion" / a.name
    (out / "splits").mkdir(parents=True, exist_ok=True)
    tab.to_parquet(out / "table.parquet", index=False)
    tab.to_csv(out / "table.tsv", sep="\t", index=False)
    for frac in a.fractions:
        for seed in a.split_seeds:
            r = np.random.default_rng(10_000 * seed + int(round(frac * 1000)))
            perm = r.permutation(len(tab))
            n_tr = int(round(frac * len(tab)))
            tr, rest = tab.eq_id.values[np.sort(perm[:n_tr])], tab.eq_id.values[np.sort(perm[n_tr:])]
            n_dev = len(rest) // 10
            rr = np.random.default_rng(seed + 7).permutation(len(rest))
            m = dict(split_name=f"frac{int(frac * 100):02d}_seed{seed}", strategy="table_fraction", seed=seed,
                     params=dict(train_fraction=frac, table=a.name, n_equations=len(tab)),
                     partitions={"train": dict(pair_ids=sorted(tr.tolist())),
                                 "dev": dict(pair_ids=sorted(rest[np.sort(rr[:n_dev])].tolist())),
                                 "test": dict(pair_ids=sorted(rest[np.sort(rr[n_dev:])].tolist()))})
            d = out / "splits" / m["split_name"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "manifest.json").write_text(json.dumps(m))
    meta = dict(vars(a), n_equations=len(tab), n_cells=len(cells),
                class_counts=tab.drop_duplicates("root_symbol").analysis__class.value_counts().to_dict(),
                generator="hebmorph/synthetic_v2.py",
                table_sha256=hashlib.sha256((out / "table.parquet").read_bytes()).hexdigest())
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps({k: meta[k] for k in ["name", "n_roots", "weak_fraction", "n_equations", "n_cells", "class_counts"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
