"""Helpers shared by the real-data builder and the synthetic generator, so that
both produce tables with identical schema and semantics."""
from __future__ import annotations

import pandas as pd

from .morphology import PAST_CELLS

CELL_ORDER = (PAST_CELLS +
              ["PRS.MSG", "PRS.FSG", "PRS.MPL", "PRS.FPL"] +
              ["FUT.1SG", "FUT.2MSG", "FUT.2FSG", "FUT.3MSG", "FUT.3FSG", "FUT.1PL",
               "FUT.2MPL", "FUT.2FPL", "FUT.3MPL", "FUT.3FPL"] +
              ["IMP.2MSG", "IMP.2FSG", "IMP.2MPL", "IMP.2FPL", "NFIN", "MSDR"])
CELL_RANK = {c: i for i, c in enumerate(CELL_ORDER)}

TIER_RANK = {"GOLD": 0, "SILVER": 1, "QUESTIONABLE": 2}


def worst_tier(*tiers):
    return max(tiers, key=lambda t: TIER_RANK[t])


def best_tier(tiers):
    return min(tiers, key=lambda t: TIER_RANK[t])


def join_flags(flags) -> str | None:
    fs = sorted({f for f in flags if f})
    return "|".join(fs) if fs else None


def add_syncretism(forms: pd.DataFrame) -> pd.DataFrame:
    """Within each lexeme, group cells with identical vocalized forms (and,
    separately, identical plene unvocalized forms).  Groups are numbered in
    canonical cell order.  Nothing is removed."""
    forms = forms.copy()
    forms["_rank"] = forms.canonical_cell.map(lambda c: CELL_RANK.get(c, 999))
    forms = forms.sort_values(["lexeme_id", "_rank"]).drop(columns="_rank")
    for col, gcol, flagcol, tag in [("form_vocalized", "syncretism_group", "is_syncretic", "V"),
                                    ("form_unvocalized", "syncretism_group_unvocalized",
                                     "is_syncretic_unvocalized", "U")]:
        groups, flags = [], []
        for lex, sub in forms.groupby("lexeme_id", sort=False):
            vals = sub[col].tolist()
            counts = pd.Series([v for v in vals if isinstance(v, str)]).value_counts()
            numbering = {}
            for v in vals:
                if isinstance(v, str) and counts.get(v, 0) > 1 and v not in numbering:
                    numbering[v] = len(numbering) + 1
            for v in vals:
                if isinstance(v, str) and v in numbering:
                    groups.append(f"{lex}:{tag}{numbering[v]}")
                    flags.append(True)
                else:
                    groups.append(None)
                    flags.append(False)
        forms[gcol] = groups
        forms[flagcol] = flags
    return forms


def add_past_completeness(lexemes: pd.DataFrame, forms: pd.DataFrame) -> pd.DataFrame:
    lexemes = lexemes.copy()
    past = forms[forms.canonical_cell.isin(PAST_CELLS) & forms.form_vocalized.notna()]
    obs = past.groupby("lexeme_id").canonical_cell.nunique()
    ncell = forms.groupby("lexeme_id").canonical_cell.nunique()
    lexemes["past_expected_cells"] = len(PAST_CELLS)
    lexemes["past_observed_cells"] = lexemes.lexeme_id.map(obs).fillna(0).astype(int)
    lexemes["past_completeness"] = lexemes.past_observed_cells / len(PAST_CELLS)
    lexemes["past_complete"] = lexemes.past_observed_cells == len(PAST_CELLS)
    lexemes["n_cells"] = lexemes.lexeme_id.map(ncell).fillna(0).astype(int)
    return lexemes


def root_rows_from_lexemes(root_info: dict, lexemes: pd.DataFrame) -> pd.DataFrame:
    """root_info: root_normalized -> dict(root_id, root_raw, radicals, traditional_class,
    traditional_class_raw, source_primary, source_ids, quality_flags)."""
    from .roots import orthographic, root_id_for, structural_features
    rows = []
    for rn, info in root_info.items():
        lx = lexemes[lexemes.root_id == info["root_id"]]
        f = structural_features(info["radicals"])
        row = dict(root_id=info["root_id"], root_raw=info["root_raw"], root_normalized=rn, **f,
                   traditional_class=info.get("traditional_class"),
                   traditional_class_raw=info.get("traditional_class_raw"),
                   root_orthographic=orthographic(rn),
                   orthographic_root_group_id=root_id_for(orthographic(rn), prefix=info.get("group_prefix", "RG-")),
                   n_lexemes=len(lx),
                   binyanim="|".join(sorted(lx.binyan.fillna("?").unique())),
                   source_primary=info["source_primary"], source_ids=info["source_ids"],
                   confidence=float(lx.confidence_root.max()) if len(lx) else 0.0,
                   quality_tier=best_tier(lx.quality_tier.tolist()) if len(lx) else "QUESTIONABLE",
                   quality_flags=info.get("quality_flags"))
        rows.append(row)
    return pd.DataFrame(rows)
