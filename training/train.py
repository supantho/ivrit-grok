"""Minimal training loop for reinflection models (synthetic data first).

    python -m training.train --config configs/syn_root_holdout.json [--set steps=2000 lr=3e-4]

Writes runs/<run_name>/:
    config.json      resolved config + dataset/split identity + git commit + vocab
    metrics.jsonl    one line per evaluation: step, train/dev/test loss, token acc, exact-match acc,
                     per-target-cell exact match (test)
    ckpt/step_XXXXXXX.pt   log-spaced model checkpoints (for grokking / interpretability analyses)
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from hebmorph import paths

from .data import BOS, EOS, PAD, SEP, Encoded, Vocab, encode, load_partitions  # noqa: F401
from .model import TinyGPT, build_model, greedy_decode, greedy_decode_bucketed, lm_loss

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    run_name: str = "debug"
    # data
    dataset: str = "synthetic_v1"          # 'synthetic_v1' | 'real'
    view: str = "past_reinflection"
    split: str = "root_holdout_seed0"
    condition: str = "voc"                 # 'voc' | 'unv'
    train_subsample: int | None = None     # use only N training examples (data-limited / grokking regimes)
    table: str | None = None               # table-completion task name (data/experiments/table_completion/<table>)
    # model
    arch: str = "transformer"              # 'transformer' | 'lstm' | 'mlp'
    d_emb: int = 64                        # token embedding size (lstm, mlp)
    d_model: int = 128                     # transformer width / LSTM hidden size
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 512
    max_len: int = 48
    # optimisation (grokking-style defaults: AdamW, strong weight decay, constant LR after warmup)
    steps: int = 20000
    batch_size: int = 512                  # -1 = full batch
    batch_rule: str = "fixed"              # 'fixed' | 'power' (= min(512, N_train // 2), Power et al. 2022)
    lr: float = 1e-3
    weight_decay: float = 1.0
    # Power et al. (2022) used plain AdamW, i.e. weight decay on ALL parameters. Exempting LayerNorm
    # gains/biases (GPT convention) lets the model inflate the final LN gain to scale logits and
    # escape the decay entirely (observed in sweep grok_v1), so the default here decays everything.
    decay_norm_and_bias: bool = True
    use_layernorm: bool = True             # False = LayerNorm-free variant (as in Nanda et al., 2023)
    init: str = "gpt"                      # 'gpt' N(0,0.02) | 'torch_default' (Power et al. code used PyTorch defaults)
    init_scale: float = 1.0                # Omnigrok alpha: scale all initial weights
    betas: tuple = (0.9, 0.98)
    warmup_steps: int = 200
    grad_clip: float = 1.0
    seed: int = 0
    # logging
    eval_every: int = 250
    eval_max_examples: int | None = 4096   # per partition, fixed random subset (seeded); None = full partition
    eval_log_points: int = 0               # extra log-spaced evaluation steps (grokking curves on a log axis)
    exact_match_method: str = "greedy"     # 'greedy' | 'teacher_forcing' (equivalent; teacher forcing is 1 pass)
    n_checkpoints: int = 30                # log-spaced
    ckpt_every: int | None = None          # additional evenly spaced checkpoints (information-plane resolution)
    deterministic: bool = False            # torch.use_deterministic_algorithms + fixed cuBLAS workspace
    device: str = "auto"
    extra: dict = field(default_factory=dict)


def data_bases(dataset: str):
    if dataset == "synthetic_v1":
        base = paths.DERIVED / "synthetic" / "v1"
        return base, base / "splits"
    if dataset == "real":
        return paths.DERIVED, paths.SPLITS
    raise ValueError(dataset)


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def log_spaced(n: int, total: int) -> set[int]:
    return {int(round(x)) for x in np.logspace(0, math.log10(max(total, 2)), n)} | {total}


@torch.no_grad()
def evaluate(model, enc: Encoded, vocab: Vocab, device, batch=1024, per_cell=False, method="greedy"):
    """Exact match = the whole generated answer (incl. <eos>) equals the target.
    method='teacher_forcing' computes it in one pass: greedy decoding reproduces the
    target iff the argmax is correct at every answer position under teacher forcing
    (by induction over positions), so the two methods are identical."""
    model.eval()
    n = enc.tokens.shape[0]
    tot_loss, tot_tok, tot_correct_tok, exact = 0.0, 0, 0, 0
    cell_hits, cell_n = defaultdict(int), defaultdict(int)
    for i in range(0, n, batch):
        tok = enc.tokens[i:i + batch].to(device)
        m = enc.loss_mask[i:i + batch].to(device)
        logits = model(tok)
        pred, gold, mm = logits[:, :-1], tok[:, 1:], m[:, :-1]
        tot_loss += torch.nn.functional.cross_entropy(pred[mm], gold[mm], reduction="sum").item()
        tot_tok += int(mm.sum())
        tot_correct_tok += int((pred.argmax(-1)[mm] == gold[mm]).sum())
        if method == "teacher_forcing":
            ok = (pred.argmax(-1) == gold) | ~mm
            hits = ok.all(1).cpu().tolist()
            exact += sum(hits)
            if per_cell:
                for j, h in enumerate(hits):
                    cell_hits[enc.features[i + j]] += h
                    cell_n[enc.features[i + j]] += 1
            continue
        prompts = [tok[j, : int(enc.prompt_len[i + j])] for j in range(tok.shape[0])]
        max_new = model.cfg.max_len - int(enc.prompt_len[i:i + batch].max())
        if isinstance(model, TinyGPT):
            out = greedy_decode(model, prompts, max_new=max_new, eos_id=EOS, pad_id=PAD).cpu()
        else:
            out = greedy_decode_bucketed(model, prompts, max_new=max_new, eos_id=EOS)
        for j in range(tok.shape[0]):
            hit = vocab.dec_chars(out[j].tolist()) == enc.targets[i + j]
            exact += hit
            if per_cell:
                cell_hits[enc.features[i + j]] += hit
                cell_n[enc.features[i + j]] += 1
    model.train()
    r = dict(loss=tot_loss / max(tot_tok, 1), token_acc=tot_correct_tok / max(tot_tok, 1), exact_match=exact / max(n, 1), n=n)
    if per_cell:
        r["exact_match_by_features"] = {k: cell_hits[k] / cell_n[k] for k in sorted(cell_n)}
    return r


def _subset(enc: Encoded, idx) -> Encoded:
    idx = torch.as_tensor(idx)
    return Encoded(enc.tokens[idx], enc.loss_mask[idx], enc.prompt_len[idx],
                   [enc.targets[i] for i in idx.tolist()], [enc.features[i] for i in idx.tolist()])


def run(cfg: TrainConfig) -> Path:
    if cfg.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    set_seed(cfg.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    derived_base, splits_base = data_bases(cfg.dataset)
    parts = load_partitions(cfg.view, cfg.split, cfg.condition, derived_base, splits_base, table=cfg.table)
    if "train" not in parts or not parts["train"]:
        raise RuntimeError(f"no training data for {cfg.split}")
    rng = np.random.default_rng(cfg.seed)
    if cfg.train_subsample:
        idx = rng.choice(len(parts["train"]), size=min(cfg.train_subsample, len(parts["train"])), replace=False)
        parts["train"] = [parts["train"][i] for i in sorted(idx)]
    # Vocabulary from the training partition only; unseen characters elsewhere map to <unk> (counted).
    vocab = Vocab.build(parts["train"])
    enc = {p: encode(ex, vocab, cfg.max_len) for p, ex in parts.items()}
    unk = {p: int((e.tokens == 4).sum()) for p, e in enc.items()}
    eval_sets = {}
    for p, e in enc.items():
        n = e.tokens.shape[0]
        sel = np.sort(rng.choice(n, size=min(n, cfg.eval_max_examples or n), replace=False))
        eval_sets[p] = _subset(e, sel)

    model = build_model(cfg.arch, len(vocab), cfg).to(device)
    mcfg = model.cfg
    if cfg.decay_norm_and_bias:
        decay, no_decay = list(model.parameters()), []
    else:
        decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
        no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, betas=tuple(cfg.betas))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps)))

    out = ROOT / "runs" / cfg.run_name
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    if cfg.table:
        from hebmorph.loader import table_dir
        split_manifest = table_dir(cfg.table) / "splits" / cfg.split / "manifest.json"
    else:
        split_manifest = splits_base / cfg.view / cfg.split / "manifest.json"
    meta = dict(config=dataclasses.asdict(cfg), arch=cfg.arch, model=dataclasses.asdict(mcfg), n_params=model.n_params(),
                device=device, git_commit=git_commit(),
                dataset_manifest=json.loads((paths.METADATA / "dataset_manifest.json").read_text()).get("dataset_version"),
                split_manifest_sha256=hashlib.sha256(split_manifest.read_bytes()).hexdigest(),
                partition_sizes={p: len(x) for p, x in parts.items()}, unk_tokens=unk, vocab=vocab.itos,
                visible_fields="source_form,target_features,target_form (via hebmorph.loader)")
    (out / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[{cfg.run_name}] {model.n_params():,} params on {device}; sizes {meta['partition_sizes']}; unk {unk}")

    train = enc["train"]
    N = train.tokens.shape[0]
    if cfg.batch_rule == "power":
        bs = min(512, max(1, N // 2))
    else:
        bs = N if cfg.batch_size == -1 else min(cfg.batch_size, N)
    eval_steps = log_spaced(cfg.eval_log_points, cfg.steps) if cfg.eval_log_points else set()
    ckpts = log_spaced(cfg.n_checkpoints, cfg.steps)
    if cfg.ckpt_every:
        ckpts |= set(range(cfg.ckpt_every, cfg.steps + 1, cfg.ckpt_every))
    g = torch.Generator().manual_seed(cfg.seed)
    tok_all, mask_all = train.tokens.to(device), train.loss_mask.to(device)
    t0 = time.time()
    with open(out / "metrics.jsonl", "w") as mf:
        for step in range(1, cfg.steps + 1):
            idx = torch.randperm(N, generator=g)[:bs].to(device) if bs < N else slice(None)
            tok, m = tok_all[idx], mask_all[idx]
            loss = lm_loss(model(tok), tok, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            if step in ckpts:
                torch.save(dict(step=step, arch=cfg.arch, model=model.state_dict(), model_config=dataclasses.asdict(mcfg)),
                           out / "ckpt" / f"step_{step:07d}.pt")
            if step % cfg.eval_every == 0 or step == 1 or step == cfg.steps or step in eval_steps:
                rec = dict(step=step, batch_size=bs, train_batch_loss=loss.item(), lr=sched.get_last_lr()[0],
                           weight_norm=float(sum((p.detach() ** 2).sum() for p in model.parameters()) ** 0.5),
                           norm_params_norm=float(sum((p.detach() ** 2).sum() for n, p in model.named_parameters()
                                                      if p.dim() < 2) ** 0.5),
                           elapsed_s=round(time.time() - t0, 1))
                for p, e in eval_sets.items():
                    r = evaluate(model, e, vocab, device, per_cell=(p != "train"), method=cfg.exact_match_method)
                    for k, v in r.items():
                        rec[f"{p}/{k}"] = v
                mf.write(json.dumps(rec) + "\n")
                mf.flush()
                accs = " ".join(f"{p}={rec[f'{p}/exact_match']:.3f}" for p in eval_sets)
                print(f"step {step:6d} loss {loss.item():.4f} exact[{accs}] |w|={rec['weight_norm']:.1f} {rec['elapsed_s']}s", flush=True)
    return out


def _parse_overrides(pairs):
    out = {}
    for kv in pairs or []:
        k, _, v = kv.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--set", nargs="*", help="overrides key=value (JSON values)")
    a = ap.parse_args(argv)
    d = json.loads(a.config.read_text()) if a.config else {}
    d.update(_parse_overrides(a.set))
    cfg = TrainConfig(**d)
    run(cfg)


if __name__ == "__main__":
    main()
