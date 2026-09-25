"""Minimal decoder-only transformer, written out explicitly (no fused library
blocks) so attention patterns, residual streams and MLP activations are easy to
hook for mechanistic interpretability.  Pre-LayerNorm, learned positions, no dropout."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    max_len: int = 48
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_mlp: int = 512
    use_layernorm: bool = True
    init: str = "gpt"          # 'gpt' (N(0,0.02)) | 'torch_default' (PyTorch defaults, as in Power et al.'s code)
    init_scale: float = 1.0    # multiply all initial weights by alpha (Omnigrok, Liu et al. 2022)


class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        assert c.d_model % c.n_heads == 0
        self.h, self.dh = c.n_heads, c.d_model // c.n_heads
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.out = nn.Linear(c.d_model, c.d_model)
        self.register_buffer("causal", torch.tril(torch.ones(c.max_len, c.max_len, dtype=torch.bool)), persistent=False)
        self.last_pattern = None  # set when store_patterns is True (interpretability)
        self.store_patterns = False

    def forward(self, x, pad_mask):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        allowed = self.causal[:T, :T][None, None] & pad_mask[:, None, None, :]
        # every position may attend to itself: avoids all -inf rows (NaN) at pad positions,
        # which would otherwise leak into real positions via 0 * NaN in att @ v
        allowed = allowed | torch.eye(T, dtype=torch.bool, device=x.device)[None, None]
        att = att.masked_fill(~allowed, float("-inf")).softmax(-1)
        if self.store_patterns:
            self.last_pattern = att.detach()
        return self.out((att @ v).transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        norm = (lambda: nn.LayerNorm(c.d_model)) if c.use_layernorm else nn.Identity
        self.ln1, self.attn = norm(), Attention(c)
        self.ln2 = norm()
        self.mlp = nn.Sequential(nn.Linear(c.d_model, c.d_mlp), nn.GELU(), nn.Linear(c.d_mlp, c.d_model))

    def forward(self, x, pad_mask):
        x = x + self.attn(self.ln1(x), pad_mask)
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.cfg = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.ln_f = nn.LayerNorm(c.d_model) if c.use_layernorm else nn.Identity()
        self.unembed = nn.Linear(c.d_model, c.vocab_size, bias=False)
        if c.init == "gpt":
            self.apply(self._init)
        elif c.init != "torch_default":
            raise ValueError(c.init)
        if c.init_scale != 1.0:
            with torch.no_grad():
                for p in self.parameters():
                    p.mul_(c.init_scale)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens, pad_id=0):
        B, T = tokens.shape
        pad_mask = tokens != pad_id
        x = self.tok(tokens) + self.pos(torch.arange(T, device=tokens.device))[None]
        for b in self.blocks:
            x = b(x, pad_mask)
        return self.unembed(self.ln_f(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def lm_loss(logits, tokens, loss_mask):
    """Next-token cross-entropy on target positions only."""
    pred, gold = logits[:, :-1], tokens[:, 1:]
    m = loss_mask[:, :-1]
    return F.cross_entropy(pred[m], gold[m])


@torch.no_grad()
def greedy_decode(model, prompts: list[torch.Tensor], max_new: int, eos_id: int, pad_id: int = 0):
    """Batched greedy decoding.  prompts: list of 1-D tensors (<bos>..<sep>)."""
    dev = next(model.parameters()).device
    B = len(prompts)
    P = max(len(p) for p in prompts)
    # left-pad so every prompt ends at the same position
    seq = torch.full((B, P), pad_id, dtype=torch.long, device=dev)
    for i, p in enumerate(prompts):
        seq[i, P - len(p):] = p.to(dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    out = []
    for _ in range(max_new):
        if seq.shape[1] > model.cfg.max_len:
            break
        logits = _forward_left_padded(model, seq, pad_id)
        nxt = logits[:, -1].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
        out.append(nxt)
        done |= nxt == eos_id
        seq = torch.cat([seq, nxt[:, None]], 1)
        if done.all():
            break
    return torch.stack(out, 1) if out else torch.empty(B, 0, dtype=torch.long)


def _forward_left_padded(model, seq, pad_id):
    """Positions are counted from the first real token so that left padding
    does not shift the learned position embeddings seen in training."""
    B, T = seq.shape
    pad_mask = seq != pad_id
    pos = (pad_mask.long().cumsum(1) - 1).clamp(min=0)
    x = model.tok(seq) + model.pos(pos)
    for b in model.blocks:
        x = b(x, pad_mask)
    return model.unembed(model.ln_f(x))


# ==========================================================================
# Model family: every model is an autoregressive LM over the SAME sequence
# (<bos> source <sep> features <sep> target <eos>) with the same answer-only
# loss, so architectures are compared on an identical task.
# forward(tokens, return_hidden=True) -> (logits, [per-layer hidden states])
# gives the representations used by the information-plane analysis.
# ==========================================================================
@dataclass
class LSTMConfig:
    vocab_size: int
    max_len: int = 64
    d_emb: int = 64
    d_hidden: int = 128
    n_layers: int = 1


class LSTMLM(nn.Module):
    """Stacked single-layer LSTMs (kept separate so every layer's states are observable)."""
    def __init__(self, c: LSTMConfig):
        super().__init__()
        self.cfg = c
        self.tok = nn.Embedding(c.vocab_size, c.d_emb)
        dims = [c.d_emb] + [c.d_hidden] * c.n_layers
        self.layers = nn.ModuleList(nn.LSTM(dims[i], dims[i + 1], batch_first=True) for i in range(c.n_layers))
        self.unembed = nn.Linear(c.d_hidden, c.vocab_size)

    def forward(self, tokens, pad_id=0, return_hidden=False):
        x = self.tok(tokens)
        hs = [x]
        for l in self.layers:
            x, _ = l(x)
            hs.append(x)
        logits = self.unembed(x)
        return (logits, hs) if return_hidden else logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


@dataclass
class MLPConfig:
    vocab_size: int
    max_len: int = 64
    d_emb: int = 16
    d_hidden: int = 256
    n_layers: int = 2
    window: int = 64       # causal context window (= max_len: the MLP sees the whole prefix)


class CausalMLPLM(nn.Module):
    """Position t sees the concatenated embeddings of tokens t-W+1..t (zero-padded on the left);
    window slots act as positions.  No recurrence, no attention: a pure feed-forward baseline."""
    def __init__(self, c: MLPConfig):
        super().__init__()
        self.cfg = c
        self.tok = nn.Embedding(c.vocab_size, c.d_emb)
        dims = [c.window * c.d_emb] + [c.d_hidden] * c.n_layers
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(c.n_layers))
        self.unembed = nn.Linear(c.d_hidden, c.vocab_size)

    def forward(self, tokens, pad_id=0, return_hidden=False):
        B, T = tokens.shape
        W = self.cfg.window
        e = self.tok(tokens) * (tokens != pad_id)[..., None]
        e = F.pad(e, (0, 0, W - 1, 0))                     # (B, T+W-1, E)
        x = e.unfold(1, W, 1).permute(0, 1, 3, 2).reshape(B, T, W * self.cfg.d_emb)
        hs = []
        for l in self.layers:
            x = F.gelu(l(x))
            hs.append(x)
        logits = self.unembed(x)
        return (logits, hs) if return_hidden else logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def transformer_hidden(model: TinyGPT, tokens, pad_id=0):
    """Residual stream after embedding, after each block, and after the final norm."""
    B, T = tokens.shape
    pad_mask = tokens != pad_id
    x = model.tok(tokens) + model.pos(torch.arange(T, device=tokens.device))[None]
    hs = [x]
    for b in model.blocks:
        x = b(x, pad_mask)
        hs.append(x)
    hs.append(model.ln_f(x))
    return model.unembed(hs[-1]), hs


def forward_hidden(model, tokens, pad_id=0):
    if isinstance(model, TinyGPT):
        return transformer_hidden(model, tokens, pad_id)
    return model(tokens, pad_id=pad_id, return_hidden=True)


def _scale_init(m: nn.Module, alpha: float) -> nn.Module:
    if alpha != 1.0:
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(alpha)
    return m


def build_model(arch: str, vocab_size: int, cfg) -> nn.Module:
    """LSTM and MLP always use PyTorch default init; init_scale (Omnigrok alpha) applies to all archs."""
    alpha = getattr(cfg, "init_scale", 1.0)
    if arch == "lstm":
        return _scale_init(LSTMLM(LSTMConfig(vocab_size=vocab_size, max_len=cfg.max_len, d_emb=cfg.d_emb,
                                             d_hidden=cfg.d_model, n_layers=cfg.n_layers)), alpha)
    if arch == "mlp":
        return _scale_init(CausalMLPLM(MLPConfig(vocab_size=vocab_size, max_len=cfg.max_len, d_emb=cfg.d_emb,
                                                 d_hidden=cfg.d_mlp, n_layers=cfg.n_layers, window=cfg.max_len)), alpha)
    if arch == "transformer":
        return TinyGPT(ModelConfig(vocab_size=vocab_size, max_len=cfg.max_len, d_model=cfg.d_model,
                                   n_layers=cfg.n_layers, n_heads=cfg.n_heads, d_mlp=cfg.d_mlp,
                                   use_layernorm=cfg.use_layernorm, init=getattr(cfg, "init", "gpt"),
                                   init_scale=getattr(cfg, "init_scale", 1.0)))
    if arch == "lstm":
        return LSTMLM(LSTMConfig(vocab_size=vocab_size, max_len=cfg.max_len, d_emb=cfg.d_emb,
                                 d_hidden=cfg.d_model, n_layers=cfg.n_layers))
    if arch == "mlp":
        return CausalMLPLM(MLPConfig(vocab_size=vocab_size, max_len=cfg.max_len, d_emb=cfg.d_emb,
                                     d_hidden=cfg.d_mlp, n_layers=cfg.n_layers, window=cfg.max_len))
    raise ValueError(arch)


@torch.no_grad()
def greedy_decode_bucketed(model, prompts: list[torch.Tensor], max_new: int, eos_id: int):
    """Architecture-agnostic greedy decoding: prompts are grouped by length so no
    padding ever enters the model (needed for LSTM/MLP; exact for all models)."""
    dev = next(model.parameters()).device
    by_len = {}
    for i, p in enumerate(prompts):
        by_len.setdefault(len(p), []).append(i)
    out = [None] * len(prompts)
    for L, idx in by_len.items():
        seq = torch.stack([prompts[i] for i in idx]).to(dev)
        done = torch.zeros(len(idx), dtype=torch.bool, device=dev)
        gen = []
        for _ in range(max_new):
            if seq.shape[1] >= model.cfg.max_len:
                break
            nxt = model(seq)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
            gen.append(nxt)
            done |= nxt == eos_id
            seq = torch.cat([seq, nxt[:, None]], 1)
            if done.all():
                break
        g = torch.stack(gen, 1).cpu() if gen else torch.empty(len(idx), 0, dtype=torch.long)
        for j, i in enumerate(idx):
            out[i] = g[j]
    return out
