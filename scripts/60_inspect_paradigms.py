"""Print N random complete past paradigms (stratified over binyan x structural class) for manual review.
Writes reports/paradigm_inspection.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from hebmorph import paths  # noqa: E402
from hebmorph.morphology import PAST_CELLS  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
C = paths.CANONICAL
roots, lex, forms = (pd.read_parquet(C / f"{t}.parquet") for t in ["roots", "lexemes", "forms"])
v = lex[lex.past_complete & lex.root_id.notna() & lex.binyan.notna()].merge(
    roots[["root_id", "root_normalized", "structural_class", "traditional_class"]], on="root_id")
# stratify: round-robin over (binyan, structural_class) strata, random within each
strata = v.groupby(["binyan", "structural_class"])
picked = []
order = strata.size().sample(frac=1, random_state=7).index.tolist()
pools = {k: g.sample(frac=1, random_state=7) for k, g in strata}
i = 0
while len(picked) < N and any(len(p) for p in pools.values()):
    k = order[i % len(order)]
    if len(pools[k]):
        picked.append(pools[k].iloc[0]); pools[k] = pools[k].iloc[1:]
    i += 1
out = [f"# Manual inspection: {len(picked)} random complete past paradigms\n",
       "Stratified round-robin over binyan x structural root class (random_state=7). "
       "Columns: vocalized | plene unvocalized | tier. Syncretic (unvocalized) cells marked with *.\n"]
for L in picked:
    f = forms[(forms.lexeme_id == L.lexeme_id) & forms.is_past_cell].set_index("canonical_cell")
    out.append(f"## {L.lemma_vocalized}  root {L.root_normalized}  {L.binyan}  "
               f"[{L.structural_class}; גזרה={L.traditional_class}]  lexeme tier {L.quality_tier}  ({L.lexeme_id})\n")
    out.append("| cell | vocalized | unvocalized | tier |\n|---|---|---|---|")
    for c in PAST_CELLS:
        r = f.loc[c]
        star = "*" if r.is_syncretic_unvocalized else ""
        out.append(f"| {c} | {r.form_vocalized} | {r.form_unvocalized}{star} | {r.quality_tier} |")
    out.append("")
text = "\n".join(out)
(paths.REPORTS / "paradigm_inspection.md").write_text(text, encoding="utf-8")
print(text)
