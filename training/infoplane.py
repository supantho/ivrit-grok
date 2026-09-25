"""Information-plane analysis across training checkpoints, using the SOFT ENTROPY estimator
of Conklin (2025) [arXiv:2505.23960], as in Conklin et al., "Learning is Forgetting: LLM
Training as Lossy Compression" (ICLR 2026), and in soft_entropy==0.1.1 / impossible_lms
src/estimators.py.

Estimator (per layer, per checkpoint, fixed analysis set):
  1. z_bar = z / ||z||                       project each representation onto S^{d-1}
  2. W ~ Uniform(S^{d-1}), n anchors          fixed seed -> identical coordinates for every
                                              checkpoint / layer / model of dimension d
  3. A_b = softmax(<z_bar_b, W> / eps*),      eps* = 1 / sqrt(2 d log n)   (Conklin eq. 7)
  4. P(Z) = mean_b A_b ;  H(Z) = -sum P log P
  5. H(Z | L=l) from the mean of A_b over examples with label l
     I(L; Z) = H(Z) - sum_l p(l) H(Z | L=l)
     Complexity w.r.t. individual inputs: I(X; Z) = H(Z) - mean_b H(A_b)   (X = example identity)
All values in nats and efficiency-normalised (divided by log n), as in the package.

Two views of the network:
  * QUERY state: residual stream / hidden state at the last prompt position ('=' / <sep>),
    i.e. just before the answer is produced.  Labels: example (complexity), root, radicals
    r1 r2 r3, template (binyan), cell, and the full answer.
  * ANSWER states: every position that predicts an answer token (Conklin's LM framing).
    X = current input token, Y = next (target) token:
       complexity  I(X; Z)      expressivity  I(Y; Z)      optimality  I(Y;Z)/I(X;Z)
    plus the variational bound on the output layer: I(Y; Z) >= H(Y) - CE.
Note: with (almost) unique answers, conditioning the QUERY state on 'answer' is the same as
conditioning on the example, so the answer-level plane is the meaningful expressivity axis.

    python -m training.infoplane runs/<run> [--n 3000] [--parts train test] [--anchors 256]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hebmorph.loader import Example, TableExample, load_split_analysis, load_table_split_analysis

from .data import EOS, Vocab, encode
from .model import forward_hidden
from .train import TrainConfig, build_model, data_bases

ANCHOR_SEED = 42


# --------------------------------------------------------------------------
# Soft entropy (Conklin 2025)
# --------------------------------------------------------------------------
_ANCHORS: dict = {}


def anchors(n: int, d: int, device) -> torch.Tensor:
    key = (n, d, str(device))
    if key not in _ANCHORS:
        g = torch.Generator().manual_seed(ANCHOR_SEED)
        _ANCHORS[key] = F.normalize(torch.randn(n, d, generator=g), dim=-1).to(device)
    return _ANCHORS[key]


def soft_assign(z: torch.Tensor, n: int) -> torch.Tensor:
    d = z.shape[-1]
    eps = 1.0 / math.sqrt(2 * d * math.log(n))
    return F.softmax(F.normalize(z.float(), dim=-1) @ anchors(n, d, z.device).T / eps, dim=-1)


def _H(p: torch.Tensor) -> float:
    p = p / p.sum()
    return float(-(p * p.clamp(min=1e-12).log()).sum())


def soft_measures(A: torch.Tensor, labels: dict[str, np.ndarray]) -> dict:
    """A: (N, n) soft assignments.  Returns nats; *_eff = divided by log n."""
    n = A.shape[1]
    out = {}
    HZ = _H(A.mean(0))
    out["H_Z"] = HZ
    Hps = -(A * A.clamp(min=1e-12).log()).sum(1)
    out["I_example_Z"] = HZ - float(Hps.mean())
    for name, y in labels.items():
        codes, inv = np.unique(np.asarray(y), return_inverse=True)
        inv_t = torch.as_tensor(inv, device=A.device)
        sums = torch.zeros(len(codes), n, device=A.device, dtype=A.dtype).index_add_(0, inv_t, A)
        cnt = torch.bincount(inv_t, minlength=len(codes)).to(A.dtype)
        cond = sum(float(cnt[k] / cnt.sum()) * _H(sums[k]) for k in range(len(codes)))
        out[f"I_{name}_Z"] = HZ - cond
        out[f"H_{name}"] = float(-(cnt / cnt.sum() * (cnt / cnt.sum()).log()).sum())
    for k in list(out):
        if k.startswith("I_") or k == "H_Z":
            out[k + "_eff"] = out[k] / math.log(n)
    return out


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def analysis_frames(cfg: TrainConfig, parts, n_max: int, seed: int = 0) -> dict:
    """Examples + analysis metadata in exactly the order the model loader uses; the training
    partition reproduces the run's train_subsample."""
    out = {}
    for p in parts:
        if cfg.table:
            df = load_table_split_analysis(cfg.table, cfg.split, p, cfg.condition)
            df["root_lbl"], df["binyan_lbl"], df["cell_lbl"] = df.analysis__root, df.analysis__binyan, df.analysis__cell
            for k in (1, 2, 3):
                df[f"r{k}_lbl"] = df[f"analysis__r{k}"]
        else:
            d, s = data_bases(cfg.dataset)
            df = load_split_analysis(cfg.view, cfg.split, p, cfg.condition, d, s)
            df["root_lbl"] = df["analysis__root"].fillna("?")
            df["binyan_lbl"] = df["analysis__binyan"].fillna("?")
            df["cell_lbl"] = df["target_cell"]
            rad = df["analysis__root"].fillna("???").map(lambda r: list(r) + ["?"] * 3)
            for k in (1, 2, 3):
                df[f"r{k}_lbl"] = rad.map(lambda x, k=k: x[k - 1])
            if p == "train" and cfg.train_subsample:
                rng = np.random.default_rng(cfg.seed)
                idx = rng.choice(len(df), size=min(cfg.train_subsample, len(df)), replace=False)
                df = df.iloc[sorted(idx)].reset_index(drop=True)
        if len(df) > n_max:
            df = df.sample(n=n_max, random_state=seed).reset_index(drop=True)
        out[p] = df
    return out


def to_examples(cfg, df):
    if cfg.table:
        return [TableExample(r.root_symbol, r.template_symbol, r.cell_symbol, r.target_form) for r in df.itertuples()]
    return [Example(r.source_form, r.target_features, r.target_form) for r in df.itertuples()]


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
@torch.no_grad()
def collect(model, enc, device, batch=2048):
    """Per layer: query states (N, d); answer states (M, d) with current/next token ids;
    and output-layer CE per answer token."""
    q_layers, a_layers, x_tok, y_tok, ce = None, None, [], [], []
    for i in range(0, enc.tokens.shape[0], batch):
        tok = enc.tokens[i:i + batch].to(device)
        m = enc.loss_mask[i:i + batch].to(device)
        logits, hs = forward_hidden(model, tok)
        rows = torch.arange(tok.shape[0], device=device)
        pos = (enc.prompt_len[i:i + batch] - 1).to(device)
        mm = m.clone()
        mm[:, -1] = False  # positions whose NEXT token is an answer token
        q = [h[rows, pos].float() for h in hs]
        a = [h[mm].float() for h in hs]
        x_tok.append(tok[mm].cpu())
        y_tok.append(tok[:, 1:][mm[:, :-1]].cpu())
        ce.append(F.cross_entropy(logits[:, :-1][mm[:, :-1]].float(), tok[:, 1:][mm[:, :-1]], reduction="none").cpu())
        q_layers = q if q_layers is None else [torch.cat([u, v]) for u, v in zip(q_layers, q)]
        a_layers = a if a_layers is None else [torch.cat([u, v]) for u, v in zip(a_layers, a)]
    return q_layers, a_layers, torch.cat(x_tok).numpy(), torch.cat(y_tok).numpy(), torch.cat(ce)


def run(run_dir: Path, parts=("train", "test"), n_max=3000, n_anchors=256, device=None, every: int = 1):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "config.json").read_text())
    cfg = TrainConfig(**{k: v for k, v in meta["config"].items() if k in TrainConfig.__dataclass_fields__})
    vocab = Vocab.from_json(json.dumps(meta["vocab"], ensure_ascii=False))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    frames = analysis_frames(cfg, parts, n_max)
    encs = {p: encode(to_examples(cfg, df), vocab, cfg.max_len) for p, df in frames.items()}
    model = build_model(cfg.arch, len(vocab), cfg).to(device).eval()
    ckpts = sorted((run_dir / "ckpt").glob("step_*.pt"))[::every]
    metrics = {}
    if (run_dir / "metrics.jsonl").exists():
        for line in (run_dir / "metrics.jsonl").read_text().splitlines():
            r = json.loads(line)
            metrics[r["step"]] = r
    rows = []
    for ck in ckpts:
        state = torch.load(ck, map_location=device)
        model.load_state_dict(state["model"])
        step = state["step"]
        for p, enc in encs.items():
            df = frames[p]
            q_layers, a_layers, x_tok, y_tok, ce = collect(model, enc, device)
            qlab = {k: df[f"{k}_lbl"].values for k in ["root", "r1", "r2", "r3", "binyan", "cell"]}
            qlab["answer"] = df.target_form.values
            ycounts = np.bincount(y_tok)
            py = ycounts[ycounts > 0] / ycounts.sum()
            H_Y = float(-(py * np.log(py)).sum())
            for li, (q, a) in enumerate(zip(q_layers, a_layers)):
                rq = soft_measures(soft_assign(q, n_anchors), qlab)
                ra = soft_measures(soft_assign(a, n_anchors), {"x": x_tok, "y": y_tok})
                rec = dict(step=step, partition=p, layer=li, n_query=len(q), n_answer_tokens=len(a),
                           n_anchors=n_anchors, **{f"query/{k}": v for k, v in rq.items()},
                           **{f"answer/{k}": v for k, v in ra.items()})
                rec["answer/complexity_eff"] = ra["I_x_Z_eff"]
                rec["answer/expressivity_eff"] = ra["I_y_Z_eff"]
                rec["answer/optimality"] = ra["I_y_Z"] / ra["I_x_Z"] if ra["I_x_Z"] > 1e-9 else float("nan")
                rec["answer/H_Y"] = H_Y
                rec["answer/I_Y_Z_variational"] = max(0.0, H_Y - float(ce.mean()))  # output layer, nats
                m = metrics.get(step)
                if m:
                    rec["train_exact"] = m.get("train/exact_match")
                    rec["test_exact"] = m.get("test/exact_match")
                    rec["weight_norm"] = m.get("weight_norm")
                rows.append(rec)
        print(f"{run_dir.name} step {step}", flush=True)
    with open(run_dir / "infoplane.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    plot(run_dir, rows, metrics)
    return rows


def plot(run_dir: Path, rows, metrics, partition="test"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = [r for r in rows if r["partition"] == partition] or rows
    if not R:
        return
    layers = sorted({r["layer"] for r in R})
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
    norm = plt.Normalize(0, math.log10(max(r["step"] for r in R)))
    for li in layers:
        pts = sorted((r for r in R if r["layer"] == li), key=lambda r: r["step"])
        x = [r["answer/complexity_eff"] for r in pts]
        y = [r["answer/expressivity_eff"] for r in pts]
        ax[0].plot(x, y, "-", color="0.8", lw=0.8, zorder=1)
        sc = ax[0].scatter(x, y, c=[math.log10(r["step"]) for r in pts], cmap="viridis", norm=norm, s=16,
                           marker="osD^v<>"[li % 7], zorder=2, label=f"layer {li}")
    fig.colorbar(sc, ax=ax[0], label="log10 step")
    ax[0].set_xlabel("complexity  I(X_token; Z) / log n")
    ax[0].set_ylabel("expressivity  I(Y_next; Z) / log n")
    ax[0].set_title(f"information plane, answer positions ({partition})")
    ax[0].legend(fontsize=7)
    top = max(layers)
    pts = sorted((r for r in R if r["layer"] == top), key=lambda r: r["step"])
    st = [r["step"] for r in pts]
    for k, c in [("root", "C0"), ("r1", "C0"), ("binyan", "C1"), ("cell", "C2"), ("example", "k")]:
        ls = ":" if k == "r1" else "-"
        ax[1].plot(st, [r[f"query/I_{k}_Z_eff"] for r in pts], ls, color=c, label=f"I({k}; Z)")
    ax[1].set_xscale("log")
    ax[1].set_title(f"query state (layer {top}): latent variables")
    ax[1].legend(fontsize=7)
    ax[2].plot(st, [r["answer/optimality"] for r in pts], color="C4", label="optimality I(Y;Z)/I(X;Z)")
    ax[2].plot(st, [r["answer/I_Y_Z_variational"] / max(r["answer/H_Y"], 1e-9) for r in pts], color="C5",
               label="(H(Y) - CE) / H(Y)")
    ax[2].set_xscale("log")
    ax[2].legend(fontsize=7)
    ax[2].set_title("optimality / variational expressivity")
    if metrics:
        ms = sorted(metrics)
        for p, c in [("train", "k"), ("test", "C3")]:
            if f"{p}/exact_match" in metrics[ms[0]]:
                ax[3].plot(ms, [metrics[s][f"{p}/exact_match"] for s in ms], color=c, label=f"{p} exact match")
        wmax = max(metrics[s]["weight_norm"] for s in ms)
        ax[3].plot(ms, [metrics[s]["weight_norm"] / wmax for s in ms], "--", color="0.5", label="weight norm (rel.)")
        ax[3].set_xscale("log")
        ax[3].legend(fontsize=7)
        ax[3].set_title("accuracy / weight norm")
    for a in ax[1:]:
        a.set_xlabel("step")
    fig.suptitle(run_dir.name)
    fig.tight_layout()
    fig.savefig(run_dir / "infoplane.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--parts", nargs="*", default=["train", "test"])
    ap.add_argument("--anchors", type=int, default=256)
    ap.add_argument("--every", type=int, default=1, help="use every k-th checkpoint")
    a = ap.parse_args(argv)
    for d in a.run_dirs:
        run(d, tuple(a.parts), a.n, a.anchors, every=a.every)


if __name__ == "__main__":
    main()
