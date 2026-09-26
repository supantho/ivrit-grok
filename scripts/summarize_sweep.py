"""Summarize a sweep: per run, first step at which train / test exact match >= threshold,
the grokking gap (test_step - train_step), and final accuracies.
    python scripts/summarize_sweep.py grok_v1 [threshold]"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1] if len(sys.argv) > 1 else "grok_v1"
thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.99
rows = []
for d in sorted((ROOT / "runs" / name).glob("*")):
    mf = d / "metrics.jsonl"
    if not mf.exists():
        continue
    L = []
    for l in mf.read_text().splitlines():
        try:
            L.append(json.loads(l))
        except json.JSONDecodeError:   # line still being written by a running job
            pass
    try:
        cfg = json.loads((d / "config.json").read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        continue
    if not L:
        continue
    first = lambda p: next((x["step"] for x in L if x.get(f"{p}/exact_match", 0) >= thr), None)
    tr, te = first("train"), first("test")
    last = L[-1]
    rows.append(dict(run=d.name, n_train=cfg["partition_sizes"]["train"], steps_done=last["step"],
                     planned=cfg["config"]["steps"], train_step=tr, test_step=te,
                     gap=(te - tr) if (tr and te) else None,
                     final_train=round(last["train/exact_match"], 3), final_test=round(last["test/exact_match"], 3),
                     best_test=round(max(x["test/exact_match"] for x in L), 3), weight_norm=round(last["weight_norm"], 1)))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.to_string(index=False) if len(df) else "no runs yet")
if len(df):
    df.to_csv(ROOT / "runs" / name / "summary.tsv", sep="\t", index=False)
