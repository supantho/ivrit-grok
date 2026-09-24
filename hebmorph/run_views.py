"""Build derived views + split manifests for one dataset (real or synthetic)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import splits as S
from .derived import full_paradigm, past_reinflection, write_view

SEEDS = [0, 1, 2, 3, 4]
KS = [2, 4, 6, 8]
STRUCT_PROPS = ["sc_strong", "sc_final_he", "sc_hollow_candidate", "sc_initial_nun", "sc_geminate",
                "sc_guttural_containing", "sc_initial_yod_waw", "sc_quadriliteral"]
TRAD_CLASSES = ["SHLEMIM", "PE_NUN", "PE_YOD_WAW", "PE_ALEF", "AYIN_WAW_YOD", "AYIN_AYIN",
                "LAMED_YOD_HE", "LAMED_ALEF", "QUADRILITERAL"]
MIN_TEST_PAIRS = 20


def _write_manifest(m, view, outdir: Path, split_rows: list):
    d = outdir / m["split_name"]
    d.mkdir(parents=True, exist_ok=True)
    S.verify(m, view[view.condition == "voc"])
    st = S.split_stats(m, view)
    m = dict(m, stats=st, verified=True)
    (d / "manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str))
    rows = [(pid, p) for p, v in m["partitions"].items() for pid in v["pair_ids"]]
    pd.DataFrame(rows, columns=["pair_id", "partition"]).to_csv(d / "assignments.tsv", sep="\t", index=False)
    for p, s in st.items():
        split_rows.append(dict(view=outdir.name, split=m["split_name"], strategy=m["strategy"], seed=m["seed"],
                               partition=p, examples_voc=s["examples_voc"], examples_unv=s["examples_unv"],
                               pair_ids=s["pair_ids"], forms=s["forms"], lexemes=s["lexemes"], roots=s["roots"],
                               orthographic_root_groups=s["orthographic_root_groups"],
                               root_binyan_pairs=s["root_binyan_pairs"],
                               **{k: v for k, v in s.items() if k.startswith("targets_identical")}))


def build_views_and_splits(roots, lexemes, forms, derived_dir: Path, splits_dir: Path,
                           seeds=SEEDS, ks=KS, with_trad=True, struct_props=STRUCT_PROPS) -> pd.DataFrame:
    views = {"past_reinflection": past_reinflection(roots, lexemes, forms),
             "full_paradigm": full_paradigm(roots, lexemes, forms)}
    for name, df in views.items():
        write_view(df, derived_dir / name, name)
    rows, skipped = [], []
    # ---- past_reinflection: all strategies ----
    view = views["past_reinflection"]
    pairs = S.eligible_pairs(view)
    out = splits_dir / "past_reinflection"
    for seed in seeds:
        for fn in (S.split_iid, S.split_lexeme_holdout, S.split_root_holdout, S.split_root_binyan_holdout):
            _write_manifest(fn(pairs, seed), view, out, rows)
        for k in ks:
            _write_manifest(S.split_cell_k(pairs, k, seed, "past_reinflection"), view, out, rows)
    for seed in seeds[:3]:
        for prop in struct_props:
            col = S.A + prop
            flag = pairs[col].astype("boolean").fillna(False).astype(bool) if col in pairs else None
            if flag is None or flag.sum() < MIN_TEST_PAIRS or (~flag).sum() < MIN_TEST_PAIRS:
                skipped.append(f"rootclass_{prop} (too few examples on one side)")
                continue
            _write_manifest(S.split_class_ood(pairs, col, f"rootclass_{prop[3:]}", seed), view, out, rows)
        if with_trad:
            tc = pairs[S.A + "traditional_class"]
            for g in TRAD_CLASSES:
                col = S.A + "trad_" + g
                pairs[col] = tc.fillna("").str.split("+").map(lambda xs, g=g: g in xs)
                view[col] = view[S.A + "traditional_class"].fillna("").str.split("+").map(lambda xs, g=g: g in xs)
                if pairs[col].sum() < MIN_TEST_PAIRS:
                    skipped.append(f"tradclass_{g} (only {int(pairs[col].sum())} eligible pairs)")
                    continue
                _write_manifest(S.split_class_ood(pairs, col, f"tradclass_{g}", seed, unknown_mask=tc.isna()),
                                view, out, rows)
    # ---- full_paradigm: cell-completion splits ----
    fview = views["full_paradigm"]
    fpairs = S.eligible_pairs(fview)
    for seed in seeds:
        for k in ks:
            _write_manifest(S.split_cell_k(fpairs, k, seed, "full_paradigm"), fview, splits_dir / "full_paradigm", rows)
    summary = pd.DataFrame(rows)
    (splits_dir / "SKIPPED.txt").write_text("\n".join(sorted(set(skipped))) + "\n")
    return summary, views
