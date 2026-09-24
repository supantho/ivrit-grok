"""Transparent UniMorph <-> Wiktionary matching.

For every UniMorph lexeme candidate we collect ALL Wiktionary evidence that
matches one of its strings, then decide:
  * root  := the unique orthographic root supported by vocalized evidence
  * binyan:= the unique binyan supported by vocalized evidence
If more than one root (or binyan) is supported, nothing is assigned and the
case goes to reports/conflicts.tsv.  String similarity is never used to
assign a root; only exact (or orthographic-variant-neutral 'loose') equality
of whole vocalized word forms.

Evidence weights (used for match_score / confidences, noisy-OR):
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass

import pandas as pd

from .hebrew import loose_vocalized_key, normalize_nfc
from .morphology import normalize_binyan
from .roots import bare_radicals

WEIGHTS = {
    "entry_headword_exact": 0.8,
    "entry_headword_loose": 0.6,
    "entry_headword_alternate": 0.5,
    "conj_table_past_exact": 0.8,
    "conj_table_past_loose": 0.6,
    "conj_table_infinitive_exact": 0.7,
    "full_table_past_exact": 0.8,
    "entry_ktiv_male_unvocalized": 0.3,   # weak: used only if no vocalized evidence
}
KIND = {  # evidence 'kind' = independent Wiktionary record type
    "entry_headword_exact": "entry", "entry_headword_loose": "entry", "entry_headword_alternate": "entry",
    "conj_table_past_exact": "conj_table", "conj_table_past_loose": "conj_table",
    "conj_table_infinitive_exact": "conj_table", "full_table_past_exact": "full_table",
    "entry_ktiv_male_unvocalized": "entry",
}
__doc__ += "\n".join(f"  {k:32s} {v}" for k, v in WEIGHTS.items())


@dataclass
class Evidence:
    method: str
    kind: str
    weight: float
    record_id: str
    matched_string: str
    root: str | None        # dotted radicals '-'-joined, as given
    root_bare: str | None   # orthographic root key (letters, no dots)
    binyan: str | None
    binyan_raw: str | None
    page_title: str
    page_id: int
    rev_id: int
    extra: dict


def entry_record_id(r) -> str:
    return f"wikt:entry:p{r.page_id}:r{r.rev_id}:o{r.section_offset}:t{r.template_index}"


def table_record_id(r) -> str:
    return f"wikt:conj:p{r.page_id}:r{r.rev_id}:t{r.table_index}:{r.binyan_raw} {r.slot_raw}"


def full_record_id(r) -> str:
    return f"wikt:full:p{r.page_id}:r{r.rev_id}:t{r.table_index}:{r.param}"


def root_bare_of(dotted: str | None) -> str | None:
    if not dotted:
        return None
    return "".join(bare_radicals(dotted.split("-")))


class WiktionaryIndex:
    def __init__(self, entries: pd.DataFrame, tables: pd.DataFrame, full: pd.DataFrame):
        self.entries, self.tables, self.full = entries, tables, full
        self.by_headword = defaultdict(list)
        self.by_headword_loose = defaultdict(list)
        self.by_alt = defaultdict(list)
        self.by_ktiv = defaultdict(list)
        for r in entries.itertuples(index=False):
            if r.headword:
                self.by_headword[r.headword].append(r)
                self.by_headword_loose[loose_vocalized_key(r.headword)].append(r)
            for a in (r.headword_alternates or "").split("|"):
                if a:
                    self.by_alt[a].append(r)
            if r.ktiv_male:
                self.by_ktiv[r.ktiv_male].append(r)
        self.table_by_form = defaultdict(list)      # (cell, form) -> rows
        self.table_by_form_loose = defaultdict(list)
        for r in tables.itertuples(index=False):
            for f in (r.forms or "").split("|"):
                if f:
                    self.table_by_form[(r.cell, f)].append(r)
                    self.table_by_form_loose[(r.cell, loose_vocalized_key(f))].append(r)
        self.full_by_form = defaultdict(list)
        for r in full.itertuples(index=False):
            for f in (r.forms or "").split("|"):
                if f:
                    self.full_by_form[(r.cell, f)].append(r)
        # (root_bare, binyan) -> table rows / full rows  (for corroboration)
        self.tables_by_rb = defaultdict(list)
        self.absent_by_rb = defaultdict(list)
        for r in tables.itertuples(index=False):
            rb = (root_bare_of(r.root_radicals), r.binyan)
            self.tables_by_rb[rb].append(r)
            if r.flags and "TABLE_ASSERTS_ABSENT" in r.flags and r.cell == "PST.3MSG":
                self.absent_by_rb[rb].append(r)
        # An absence assertion only counts if NO table for the same root x binyan
        # gives a real 3MS past form (homonymous root pages, e.g. 'דבר א' vs
        # 'דבר ב', legitimately differ).
        for rb in list(self.absent_by_rb):
            if any(r.cell == "PST.3MSG" and r.forms for r in self.tables_by_rb.get(rb, [])):
                del self.absent_by_rb[rb]
        self.full_by_rb = defaultdict(list)
        for r in full.itertuples(index=False):
            self.full_by_rb[(root_bare_of(r.root_radicals), r.binyan)].append(r)

    # ------------------------------------------------------------------
    def _entry_ev(self, r, method, s) -> Evidence:
        b, _ = normalize_binyan(r.binyan_raw)
        return Evidence(method, KIND[method], WEIGHTS[method], entry_record_id(r), s, r.root_radicals,
                        root_bare_of(r.root_radicals), b, r.binyan_raw, r.page_title, r.page_id, r.rev_id,
                        dict(gizra_labels=r.gizra_labels, gizra_raw=r.gizra_raw, gloss_en=r.gloss_en,
                             registers=r.registers, homonym=r.root_homonym_index,
                             heading=r.section_heading_raw, root_value_raw=r.root_value_raw,
                             ktiv_male=r.ktiv_male))

    def _table_ev(self, r, method, s) -> Evidence:
        return Evidence(method, KIND[method], WEIGHTS[method], table_record_id(r), s, r.root_radicals,
                        root_bare_of(r.root_radicals), r.binyan, r.binyan_raw, r.page_title, r.page_id,
                        r.rev_id, dict(root_value_raw=r.root_raw, value_raw=r.value_raw))

    def _full_ev(self, r, method, s) -> Evidence:
        return Evidence(method, KIND[method], WEIGHTS[method], full_record_id(r), s, r.root_radicals,
                        root_bare_of(r.root_radicals), r.binyan, r.binyan_raw, r.page_title, r.page_id,
                        r.rev_id, dict(root_value_raw=r.root_raw, value_raw=r.value_raw))

    def evidence_for(self, lemma_voc: str, past3ms_voc: str | None, inf_voc: str | None,
                     lemma_unvoc: str | None) -> list[Evidence]:
        ev, seen = [], set()

        def add(e):
            key = (e.record_id, e.method)
            if key not in seen:
                seen.add(key)
                ev.append(e)
        cites = [x for x in dict.fromkeys([lemma_voc, past3ms_voc]) if x]
        for s in cites:
            for r in self.by_headword.get(s, []):
                add(self._entry_ev(r, "entry_headword_exact", s))
            for r in self.by_alt.get(s, []):
                add(self._entry_ev(r, "entry_headword_alternate", s))
            for r in self.table_by_form.get(("PST.3MSG", s), []):
                add(self._table_ev(r, "conj_table_past_exact", s))
            for r in self.full_by_form.get(("PST.3MSG", s), []):
                add(self._full_ev(r, "full_table_past_exact", s))
        exact_ids = {e.record_id for e in ev}
        for s in cites:
            k = loose_vocalized_key(s)
            for r in self.by_headword_loose.get(k, []):
                if entry_record_id(r) not in exact_ids:
                    add(self._entry_ev(r, "entry_headword_loose", s))
            for r in self.table_by_form_loose.get(("PST.3MSG", k), []):
                if table_record_id(r) not in exact_ids:
                    add(self._table_ev(r, "conj_table_past_loose", s))
        if inf_voc:
            for r in self.table_by_form.get(("NFIN", inf_voc), []):
                add(self._table_ev(r, "conj_table_infinitive_exact", inf_voc))
        if lemma_unvoc:
            for r in self.by_ktiv.get(lemma_unvoc, []):
                add(self._entry_ev(r, "entry_ktiv_male_unvocalized", lemma_unvoc))
        return ev


def noisy_or(ws) -> float:
    p = 1.0
    for w in ws:
        p *= (1 - w)
    return round(1 - p, 4)


def decide(ev: list[Evidence]) -> dict:
    """Decide root/binyan from evidence, never guessing."""
    voc = [e for e in ev if e.method != "entry_ktiv_male_unvocalized"]
    used = voc if voc else [e for e in ev if e.method == "entry_ktiv_male_unvocalized"]
    unvoc_only = not voc and bool(used)
    roots = sorted({e.root_bare for e in used if e.root_bare})
    binyans = sorted({e.binyan for e in used if e.binyan})
    root = roots[0] if len(roots) == 1 else None
    binyan = binyans[0] if len(binyans) == 1 else None
    if unvoc_only:
        # unvocalized spelling alone cannot distinguish homographs: require that
        # every supporting entry agrees on BOTH root and binyan
        if not (root and binyan):
            root = binyan = None
    agree_root = [e for e in used if root and e.root_bare == root]
    agree_bin = [e for e in used if binyan and e.binyan == binyan]
    if not used:
        status = "NO_MATCH"
    elif len(roots) > 1 and len(binyans) > 1:
        status = "ROOT_AND_BINYAN_AMBIGUOUS"
    elif len(roots) > 1:
        status = "ROOT_AMBIGUOUS"
    elif len(binyans) > 1:
        status = "BINYAN_AMBIGUOUS"
    elif unvoc_only:
        status = "MATCHED_UNVOC_ONLY" if (root and binyan) else "NO_MATCH"
    else:
        status = "MATCHED"
    dotted = sorted({e.root for e in agree_root if e.root})
    methods = sorted({e.method for e in agree_root} | {e.method for e in agree_bin})
    return dict(
        status=status, root_bare=root, binyan=binyan, roots_supported=roots, binyans_supported=binyans,
        root_dotted_variants=dotted,
        root_kinds=sorted({e.kind for e in agree_root}), binyan_kinds=sorted({e.kind for e in agree_bin}),
        conf_root=noisy_or(e.weight for e in agree_root) if root else 0.0,
        conf_binyan=noisy_or(e.weight for e in agree_bin) if binyan else 0.0,
        match_score=noisy_or({e.record_id: e.weight for e in agree_root + agree_bin}.values()) if (root or binyan) else 0.0,
        methods=methods, agree_root=agree_root, agree_bin=agree_bin, used=used,
        ambiguous=len(roots) > 1 or len(binyans) > 1,
    )


def evidence_json(ev: list[Evidence]) -> str:
    return json.dumps([{k: v for k, v in asdict(e).items() if k != "extra"} | {"extra": {k: v for k, v in e.extra.items() if v}}
                       for e in ev], ensure_ascii=False, default=str)
