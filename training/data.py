"""Tokenization and batching.  Data enters ONLY through hebmorph.loader, which
exposes (source_form, target_features, target_form) and nothing else.

Sequence layout (one example):
    <bos> s_1 .. s_n <sep> F:PST F:2 F:SG F:FEM <sep> t_1 .. t_m <eos>
Characters are NFC code points, so niqqud marks are separate tokens.
Loss is computed on the target span (t_1 .. <eos>) only.
"""
from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass

import torch

from hebmorph.loader import Example, TableExample, load_split, load_table_split

SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>", "<unk>"]
PAD, BOS, SEP, EOS, UNK = range(5)


class Vocab:
    def __init__(self, chars: list[str], features: list[str]):
        self.itos = SPECIALS + sorted(set(chars)) + sorted({"F:" + f for f in features})
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @classmethod
    def build(cls, examples) -> "Vocab":
        chars, feats = set(), set()
        for e in examples:
            if isinstance(e, TableExample):
                chars |= set(e.target_form)
                feats |= {"S:" + e.root_symbol, "S:" + e.template_symbol, "S:" + e.cell_symbol}
            else:
                chars |= set(e.source_form) | set(e.target_form)
                feats |= set(e.target_features.split(","))
        return cls(sorted(chars), sorted(feats))

    def __len__(self):
        return len(self.itos)

    def enc_chars(self, s: str) -> list[int]:
        return [self.stoi.get(c, UNK) for c in unicodedata.normalize("NFC", s)]

    def enc_feats(self, f: str) -> list[int]:
        return [self.stoi.get("F:" + x, UNK) for x in f.split(",")]

    def dec_chars(self, ids) -> str:
        out = []
        for i in ids:
            if i == EOS:
                break
            if i >= len(SPECIALS) and len(self.itos[i]) == 1:   # characters are single code points
                out.append(self.itos[i])
        return "".join(out)

    def to_json(self) -> str:
        return json.dumps(self.itos, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Vocab":
        v = cls.__new__(cls)
        v.itos = json.loads(s)
        v.stoi = {x: i for i, x in enumerate(v.itos)}
        return v


@dataclass
class Encoded:
    tokens: torch.Tensor     # (N, L) full sequences, PAD-padded
    loss_mask: torch.Tensor  # (N, L) True where token i+1 is a target token to predict from position i
    prompt_len: torch.Tensor  # (N,) length of <bos> .. second <sep>
    targets: list[str]
    features: list[str]


def encode(examples: list[Example], vocab: Vocab, max_len: int) -> Encoded:
    seqs, plens = [], []
    for e in examples:
        if isinstance(e, TableExample):   # <bos> <R..> <T..> <C..> = target
            prompt = [BOS] + [vocab.stoi.get("F:S:" + x, UNK) for x in (e.root_symbol, e.template_symbol, e.cell_symbol)] + [SEP]
        else:
            prompt = [BOS] + vocab.enc_chars(e.source_form) + [SEP] + vocab.enc_feats(e.target_features) + [SEP]
        seq = prompt + vocab.enc_chars(e.target_form) + [EOS]
        if len(seq) > max_len:
            raise ValueError(f"sequence of length {len(seq)} > max_len {max_len}: {e}")
        seqs.append(seq)
        plens.append(len(prompt))
    L = max(len(s) for s in seqs)
    tok = torch.full((len(seqs), L), PAD, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, (s, p) in enumerate(zip(seqs, plens)):
        tok[i, : len(s)] = torch.tensor(s)
        mask[i, p - 1: len(s) - 1] = True   # predict tokens p .. len-1 (target chars + <eos>)
    return Encoded(tok, mask, torch.tensor(plens), [e.target_form for e in examples],
                   [e.cell_symbol if isinstance(e, TableExample) else e.target_features for e in examples])


def load_partitions(view: str, split: str, condition: str, derived_base, splits_base,
                    partitions=("train", "dev", "test"), table: str | None = None) -> dict:
    out = {}
    for p in partitions:
        try:
            if table:
                out[p] = load_table_split(table, split, p, condition)
            else:
                out[p] = load_split(view, split, p, condition, derived_base=derived_base, splits_base=splits_base)
        except KeyError:
            continue  # e.g. OOD splits have no 'test' in dev-only configs
    return out
