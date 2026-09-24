"""Reusable split manifests (never one permanent split).

Units: a split assigns *pair_ids* (condition-agnostic example identities,
'<lexeme>:<src>><tgt>') to partitions; the same assignment applies to the
vocalized and unvocalized conditions.  Only examples with
analysis__eligible_default == True are assigned.

Strategies (grouping level in brackets):
  iid                  [example]   random 80/10/10 over pair_ids (lexemes shared by design)
  lexeme_holdout       [root x binyan]  novel-lexeme generalization; roots may be shared
  root_holdout         [orthographic root group]  roots(train) ∩ roots(test) = ∅, all binyanim of a root together
  root_binyan_holdout  [root x binyan]  test pairs unseen, but their root and binyan both seen in train
  cell_k{K}            [lexeme x cell]  paradigm-cell completion: each lexeme exposes K past cells
  rootclass_{P}        [orthographic root group]  OOD by transparent structural property P
  tradclass_{G}        [orthographic root group]  OOD by traditional גזרה G stated in Wiktionary
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from . import PARSER_VERSIONS
from .morphology import CITATION_CELL, PAST_CELLS

A = "analysis__"
FRACS = (0.8, 0.1, 0.1)


class LeakageError(AssertionError):
    pass


def _rng(seed: int, name: str) -> np.random.Generator:
    h = int(hashlib.sha256(f"{name}|{seed}".encode()).hexdigest()[:8], 16)
    return np.random.default_rng(h)


def _partition_groups(groups: list, seed: int, name: str, fracs=FRACS) -> dict:
    groups = sorted(groups)
    rng = _rng(seed, name)
    order = list(rng.permutation(len(groups)))
    n = len(groups)
    n_tr = int(round(fracs[0] * n))
    n_te = int(round(fracs[2] * n)) if fracs[2] > 0 else 0
    n_dev = n - n_tr - n_te  # exact: no group can fall through to an unintended partition
    if n_dev < 0 or (fracs[1] == 0 and n_dev > 0):
        n_dev, n_te = max(0, n_dev), n - n_tr - max(0, n_dev)
    out = {}
    for rank, i in enumerate(order):
        out[groups[i]] = "train" if rank < n_tr else "dev" if rank < n_tr + n_dev else "test"
    return out


def eligible_pairs(view: pd.DataFrame) -> pd.DataFrame:
    """One row per eligible pair_id (condition-agnostic).  A pair is eligible if
    its vocalized-condition example is eligible (the unvocalized example may be
    missing or ineligible; the loader filters per condition)."""
    v = view[(view.condition == "voc") & view[A + "eligible_default"]]
    return v.drop_duplicates("pair_id").reset_index(drop=True)


def _manifest(name, strategy, seed, params, grouping, assign: dict, pairs: pd.DataFrame, extra=None):
    parts = {}
    for p in ["train", "dev", "test", "excluded"]:
        ids = sorted(k for k, v in assign.items() if v == p)
        if not ids and p == "excluded":
            continue
        sub = pairs[pairs.pair_id.isin(ids)]
        parts[p] = dict(
            pair_ids=ids,
            lexeme_ids=sorted(sub.lexeme_id.unique().tolist()),
            root_ids=sorted(sub[A + "root_id"].dropna().unique().tolist()),
            orthographic_root_group_ids=sorted(sub[A + "orthographic_root_group_id"].dropna().unique().tolist()),
            root_binyan_keys=sorted(sub[A + "root_binyan_key"].dropna().unique().tolist()),
        )
    return dict(split_name=name, strategy=strategy, seed=seed, params=params, grouping_level=grouping,
                eligibility="analysis__eligible_default == True (lexeme, source form and target form not QUESTIONABLE; root and binyan assigned)",
                splitter_version=PARSER_VERSIONS["splitter"], partitions=parts, **(extra or {}))


# --------------------------------------------------------------------------
def split_iid(pairs, seed):
    a = _partition_groups(pairs.pair_id.tolist(), seed, "iid")
    return _manifest(f"iid_seed{seed}", "iid", seed, {"fractions": FRACS}, "example", a, pairs)


def split_lexeme_holdout(pairs, seed):
    g = _partition_groups(pairs[A + "root_binyan_key"].unique().tolist(), seed, "lexeme")
    a = {p: g[k] for p, k in zip(pairs.pair_id, pairs[A + "root_binyan_key"])}
    return _manifest(f"lexeme_holdout_seed{seed}", "lexeme_holdout", seed, {"fractions": FRACS},
                     "root_binyan_key (all lexeme records of one root x binyan together)", a, pairs)


def split_root_holdout(pairs, seed):
    g = _partition_groups(pairs[A + "orthographic_root_group_id"].unique().tolist(), seed, "root")
    a = {p: g[k] for p, k in zip(pairs.pair_id, pairs[A + "orthographic_root_group_id"])}
    return _manifest(f"root_holdout_seed{seed}", "root_holdout", seed, {"fractions": FRACS},
                     "orthographic_root_group_id", a, pairs)


def split_root_binyan_holdout(pairs, seed, frac_dev=0.1, frac_test=0.1):
    """Hold out whole (root, binyan) lexical pairs such that each held-out
    pair's root (orthographic group) and binyan still occur in train."""
    rb = pairs.groupby(A + "root_binyan_key").agg(root=(A + "root_id", "first"),
                                                  binyan=(A + "binyan", "first")).reset_index()
    by_root = defaultdict(set)
    for r in rb.itertuples(index=False):
        by_root[r.root].add(r[0])
    cand = sorted(k for k, root in zip(rb[A + "root_binyan_key"], rb.root) if len(by_root[root]) >= 2)
    rng = _rng(seed, "root_binyan")
    order = [cand[i] for i in rng.permutation(len(cand))]
    n_total = len(rb)
    quota = {"test": int(round(frac_test * n_total)), "dev": int(round(frac_dev * n_total))}
    assign = {k: "train" for k in rb[A + "root_binyan_key"]}
    remaining = {root: set(ks) for root, ks in by_root.items()}
    binyan_train = Counter(rb.binyan)
    info = dict(zip(rb[A + "root_binyan_key"], zip(rb.root, rb.binyan)))
    for part in ["test", "dev"]:
        for k in order:
            if quota[part] <= 0:
                break
            if assign[k] != "train":
                continue
            root, b = info[k]
            if len(remaining[root]) <= 1 or binyan_train[b] <= 1:
                continue  # would remove the root's (or binyan's) last training pair
            assign[k] = part
            remaining[root].discard(k)
            binyan_train[b] -= 1
            quota[part] -= 1
    a = {p: assign[k] for p, k in zip(pairs.pair_id, pairs[A + "root_binyan_key"])}
    return _manifest(f"root_binyan_holdout_seed{seed}", "root_binyan_holdout", seed,
                     {"frac_dev": frac_dev, "frac_test": frac_test,
                      "candidate_pairs": len(cand), "total_pairs": n_total},
                     "root_binyan_key; 'root still in train' enforced at root_id level", a, pairs)


def split_cell_k(pairs_view: pd.DataFrame, k: int, seed: int, view_kind: str, frac_dev_lex=0.1):
    """Paradigm-cell completion.  Eligible lexemes: all 9 past cells present
    and eligible (vocalized condition).  For each lexeme, PST.3MSG plus k-1
    random other cells are EXPOSED; the others are HELD OUT.
      past_reinflection view: train = 3MSG->exposed ; dev/test = 3MSG->held-out
      full_paradigm view:     train = exposed->exposed ; dev/test = exposed->held-out
      (pairs with a held-out SOURCE are 'excluded').
    dev vs test: 10% of lexemes provide dev, the rest test (dev/test lexeme-disjoint).
    The same lexeme is intentionally on both sides of train/test."""
    if not 2 <= k <= 8:
        raise ValueError(k)
    need = 8 if view_kind == "past_reinflection" else 72  # all eligible pairs over the 9 past cells
    cnt = pairs_view.groupby("lexeme_id").size()
    full = sorted(cnt[cnt == need].index)
    rng = _rng(seed, f"cell_k{k}")
    others = [c for c in PAST_CELLS if c != CITATION_CELL]
    exposed = {}
    for l in full:
        pick = [others[i] for i in rng.permutation(len(others))[: k - 1]]
        exposed[l] = sorted([CITATION_CELL] + pick, key=PAST_CELLS.index)
    lex_part = _partition_groups(full, seed, f"cell_k{k}_devtest", fracs=(0.0, frac_dev_lex, 1 - frac_dev_lex))
    exp = {(l, c) for l, cs in exposed.items() for c in cs}
    sub = pairs_view[pairs_view.lexeme_id.isin(set(full))]
    assign = {}
    for pid, l, sc, tc in zip(sub.pair_id, sub.lexeme_id, sub.source_cell, sub.target_cell):
        if (l, sc) not in exp:
            assign[pid] = "excluded"
        elif (l, tc) in exp:
            assign[pid] = "train"
        else:
            assign[pid] = "dev" if lex_part[l] == "dev" else "test"
    m = _manifest(f"cell_k{k}_seed{seed}", "cell_k", seed, {"k": k, "view": view_kind}, "lexeme x cell",
                  assign, sub, extra=dict(exposed_cells=exposed,
                                          held_out_cells={l: [c for c in PAST_CELLS if c not in ex] for l, ex in exposed.items()}))
    return m


def split_class_ood(pairs, flag_col: str, name: str, seed: int, unknown_mask=None):
    """Test = every root group with property; train/dev = the rest (90/10 by group).
    unknown_mask: rows whose class membership is unknown -> 'excluded'."""
    has = pairs[flag_col].astype("boolean").fillna(False).astype(bool)
    grp = pairs[A + "orthographic_root_group_id"]
    test_groups = set(grp[has])
    rest = sorted(set(grp) - test_groups)
    g = _partition_groups(rest, seed, f"ood_{name}", fracs=(0.9, 0.1, 0.0))
    a = {}
    for i, (p, gr) in enumerate(zip(pairs.pair_id, grp)):
        if unknown_mask is not None and unknown_mask.iloc[i]:
            a[p] = "excluded"
        elif gr in test_groups:
            a[p] = "test"
        else:
            a[p] = g[gr]
    return _manifest(f"{name}_seed{seed}", "class_ood", seed, {"property": flag_col}, "orthographic_root_group_id",
                     a, pairs)


# --------------------------------------------------------------------------
# Verification (also used by tests).  Raises LeakageError.
# --------------------------------------------------------------------------
def _part_df(m, pairs, part):
    ids = set(m["partitions"].get(part, {}).get("pair_ids", []))
    return pairs[pairs.pair_id.isin(ids)]


def verify(m: dict, pairs_all: pd.DataFrame) -> dict:
    """pairs_all: the view (voc rows) with analysis columns for ALL pair_ids."""
    pairs = pairs_all.drop_duplicates("pair_id")
    P = {p: _part_df(m, pairs, p) for p in ["train", "dev", "test"]}
    ids = [set(m["partitions"].get(p, {}).get("pair_ids", [])) for p in m["partitions"]]
    allids = [x for s in ids for x in s]
    if len(allids) != len(set(allids)):
        raise LeakageError(f"{m['split_name']}: a pair_id is in two partitions")
    if not all(pairs.set_index("pair_id").loc[list(set(allids) - set(m['partitions'].get('excluded', {}).get('pair_ids', [])))][A + "eligible_default"]):
        raise LeakageError(f"{m['split_name']}: ineligible example assigned")
    tr, dv, te = P["train"], P["dev"], P["test"]
    s = m["strategy"]
    col_root = A + "orthographic_root_group_id"
    if s in ("root_holdout", "class_ood"):
        for other, nm in [(te, "test"), (dv, "dev")]:
            if not set(tr[col_root]).isdisjoint(set(other[col_root])):
                raise LeakageError(f"{m['split_name']}: roots shared between train and {nm}")
            if not set(tr[A + "root_id"]).isdisjoint(set(other[A + "root_id"])):
                raise LeakageError(f"{m['split_name']}: root_ids shared between train and {nm}")
        if s == "root_holdout" and not set(dv[col_root]).isdisjoint(set(te[col_root])):
            raise LeakageError(f"{m['split_name']}: roots shared between dev and test")
        if s == "class_ood":
            f = m["params"]["property"]
            if not te[f].astype("boolean").fillna(False).astype(bool).all():
                raise LeakageError(f"{m['split_name']}: test contains roots without {f}")
            if tr[f].astype("boolean").fillna(False).astype(bool).any() or dv[f].astype("boolean").fillna(False).astype(bool).any():
                raise LeakageError(f"{m['split_name']}: train/dev contain roots with {f}")
    if s == "lexeme_holdout":
        k = A + "root_binyan_key"
        for other in (dv, te):
            if not set(tr[k]).isdisjoint(set(other[k])) or not set(tr.lexeme_id).isdisjoint(set(other.lexeme_id)):
                raise LeakageError(f"{m['split_name']}: lexemes shared across partitions")
    if s == "root_binyan_holdout":
        k = A + "root_binyan_key"
        train_roots, train_b, train_pairs = set(tr[A + "root_id"]), set(tr[A + "binyan"]), set(tr[k])
        for other, nm in [(te, "test"), (dv, "dev")]:
            if not set(other[A + "root_id"]) <= train_roots:
                raise LeakageError(f"{m['split_name']}: {nm} root missing from train")
            if not set(other[A + "binyan"]) <= train_b:
                raise LeakageError(f"{m['split_name']}: {nm} binyan missing from train")
            if not set(other[k]).isdisjoint(train_pairs):
                raise LeakageError(f"{m['split_name']}: {nm} (root,binyan) pair occurs in train")
            if not set(other.lexeme_id).isdisjoint(set(tr.lexeme_id)):
                raise LeakageError(f"{m['split_name']}: {nm} lexeme occurs in train")
    if s == "cell_k":
        ex = m["exposed_cells"]
        exp = {(l, c) for l, cs in ex.items() for c in cs}

        def is_exp(df, col):
            return pd.Series([(l, c) in exp for l, c in zip(df.lexeme_id, df[col])], index=df.index, dtype=bool)
        if len(tr) and not (is_exp(tr, "source_cell") & is_exp(tr, "target_cell")).all():
            raise LeakageError(f"{m['split_name']}: train example uses a held-out cell")
        for other in (dv, te):
            if len(other) and (is_exp(other, "target_cell").any() or not is_exp(other, "source_cell").all()):
                raise LeakageError(f"{m['split_name']}: eval example target exposed or source held out")
        if any(len(v) != m["params"]["k"] for v in ex.values()):
            raise LeakageError(f"{m['split_name']}: wrong number of exposed cells")
        if not set(dv.lexeme_id).isdisjoint(set(te.lexeme_id)):
            raise LeakageError(f"{m['split_name']}: dev/test lexemes overlap")
    return dict(ok=True)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def split_stats(m: dict, view: pd.DataFrame) -> dict:
    out = {}
    for p, d in m["partitions"].items():
        ids = set(d["pair_ids"])
        sub = view[view.pair_id.isin(ids)]
        voc = sub[sub.condition == "voc"]
        unv = sub[(sub.condition == "unv") & sub[A + "eligible_default"]]
        letters = Counter(ch for f in voc.target_form for ch in f if "א" <= ch <= "ת")
        tl = voc[A + "target_form_length"]
        out[p] = dict(
            examples_voc=len(voc), examples_unv=len(unv), pair_ids=len(ids),
            forms=int(voc[A + "target_form_id"].nunique()),
            lexemes=int(voc.lexeme_id.nunique()), roots=int(voc[A + "root_id"].nunique()),
            orthographic_root_groups=int(voc[A + "orthographic_root_group_id"].nunique()),
            root_binyan_pairs=int(voc[A + "root_binyan_key"].nunique()),
            binyan=voc.drop_duplicates("lexeme_id")[A + "binyan"].value_counts().to_dict(),
            structural_class=voc.drop_duplicates("lexeme_id")[A + "structural_class"].value_counts().to_dict(),
            target_cell=voc.target_cell.value_counts().to_dict(),
            target_form_length=dict(mean=round(float(tl.mean()), 3) if len(tl) else None,
                                    min=int(tl.min()) if len(tl) else None, max=int(tl.max()) if len(tl) else None,
                                    hist={int(k): int(v) for k, v in tl.value_counts().sort_index().items()}),
            hebrew_letter_distribution=dict(letters.most_common()),
            examples_per_root=dict(mean=round(float(voc.groupby(A + "root_id").size().mean()), 3) if len(voc) else None,
                                   max=int(voc.groupby(A + "root_id").size().max()) if len(voc) else None),
            syncretic_target_fraction_voc=round(float(voc[A + "target_is_syncretic"].mean()), 4) if len(voc) else None,
        )
    if m["strategy"] == "cell_k":
        # syncretism leak: eval targets whose form equals an exposed cell's form of the same lexeme
        ex = m["exposed_cells"]
        for cond in ["voc", "unv"]:
            v = view[view.condition == cond]
            forms_lc = pd.concat([v[["lexeme_id", "target_cell", "target_form"]].set_axis(["l", "c", "f"], axis=1),
                                  v[["lexeme_id", "source_cell", "source_form"]].set_axis(["l", "c", "f"], axis=1)]
                                 ).drop_duplicates(["l", "c"])
            exposed_forms = {(l, f) for l, c, f in forms_lc.itertuples(index=False) if c in set(ex.get(l, ()))}
            for p in ["dev", "test"]:
                if p not in m["partitions"]:
                    continue
                sub = v[v.pair_id.isin(set(m["partitions"][p]["pair_ids"]))]
                out[p][f"targets_identical_to_an_exposed_form_{cond}"] = int(sum(
                    (l, f) in exposed_forms for l, f in zip(sub.lexeme_id, sub.target_form)))
    return out
