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
        self.ln1, self.attn = nn.LayerNorm(c.d_model), Attention(c)
        self.ln2 = nn.LayerNorm(c.d_model)
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
        self.ln_f = nn.LayerNorm(c.d_model)
        self.unembed = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.apply(self._init)

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
