"""Per-class test accuracy across checkpoints (analysis only; uses analysis__class labels).
    python scripts/class_curves.py runs/<sweep>/<run> [...]   -> <run>/class_curves.json + png"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.data import Vocab, encode  # noqa: E402
from training.infoplane import analysis_frames, to_examples  # noqa: E402
from training.train import TrainConfig, build_model  # noqa: E402


@torch.no_grad()
def run(d: Path):
    meta = json.loads((d / "config.json").read_text())
    cfg = TrainConfig(**{k: v for k, v in meta["config"].items() if k in TrainConfig.__dataclass_fields__})
    vocab = Vocab.from_json(json.dumps(meta["vocab"], ensure_ascii=False))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    frames = analysis_frames(cfg, ["train", "test"], 10**9)
    model = build_model(cfg.arch, len(vocab), cfg).to(dev).eval()
    out = {}
    for p, df in frames.items():
        enc = encode(to_examples(cfg, df), vocab, cfg.max_len)
        cls = df["analysis__class"].values
        curves = {c: [] for c in sorted(set(cls))}
        steps = []
        for ck in sorted((d / "ckpt").glob("step_*.pt")):
            st = torch.load(ck, map_location=dev)
            model.load_state_dict(st["model"])
            ex = []
            for i in range(0, len(df), 4096):
                tok, m = enc.tokens[i:i + 4096].to(dev), enc.loss_mask[i:i + 4096].to(dev)
                ok = (model(tok)[:, :-1].argmax(-1) == tok[:, 1:]) | ~m[:, :-1]
                ex.append(ok.all(1).cpu().numpy())
            ex = np.concatenate(ex)
            steps.append(st["step"])
            for c in curves:
                curves[c].append(float(ex[cls == c].mean()))
        out[p] = dict(steps=steps, acc=curves)
    (d / "class_curves.json").write_text(json.dumps(out))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for c, v in out["test"]["acc"].items():
        ax.plot(out["test"]["steps"], v, lw=3 if c == "STRONG" else 1.8, label=f"test {c}")
    ax.plot(out["train"]["steps"], [np.mean(x) for x in zip(*out["train"]["acc"].values())], "k--", label="train (all)")
    ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.set_xlabel("step"); ax.set_ylabel("exact match")
    ax.set_title(d.name); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(d / "class_curves.png", dpi=110)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        run(Path(a))
        print("done", a)
