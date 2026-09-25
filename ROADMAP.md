# Roadmap / parked ideas

## Parked (registered 2026-09-25): full-paradigm table (past + present + future + imperative)

**Idea (from Supantho):** extend table completion from the 9 past cells to all tenses:
`<ROOT_k> <TMPL_j> <CELL> = form`, where CELL covers 9 past + 4 present + 10 future + 4 imperative (+ infinitive).

**Feasibility:**
* **Real data:** all 29 UniMorph cells already exist for every UniMorph lexeme in `data/canonical/forms.parquet`, so no new data collection is needed.
* **Synthetic data:** needs present, future and imperative templates added to `hebmorph/synthetic.py`. This is straightforward, since future and present are templatic too.

**Why it might help:**
* The future is *prefix + stem + suffix*: person, number and gender are split across a prefix (א/ת/י/נ) and a suffix (-י/-ו/-נה), so the agreement features act jointly on both ends of the word.
* The future stem vowel in PAAL is partly lexical (yiktov vs. yilmad), so it's a root × template × tense interaction.
* Weak roots behave differently across tenses (initial nun drops in the future: yipol; hollow roots; final-he roots).
* There is built-in syncretism: FUT.2MSG = FUT.3FSG, FUT.2FPL = FUT.3FPL, PRS vs PST.3.

**Caveat:** regular (strong-root) synthetic future is still *factorized*: each output letter depends on the root OR on (template, cell), so on its own it probably won't produce a sharp grokking jump. It is most promising combined with root × template interactions (see next item).

## Parked: synthetic v2 with root x template interactions (to test the "non-decomposability" hypothesis)

The hypothesis is that grokking needs outputs that depend *jointly* on several inputs (as in modular arithmetic), not additively. Candidate interactions, from mild to strong:
* Weak-root rules conditioned on the root class × template (assimilation, deletion, vowel changes).
* Metathesis or radical reordering selected by the template.
* An arithmetic "cipher" control, e.g. the vowel pattern chosen by (root_index + template_index) mod k. This is the fully non-decomposable limit.
