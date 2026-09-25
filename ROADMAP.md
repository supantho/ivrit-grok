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

## Proposed next experiment plan (2026-09-25; pending the user's go after table_v1 finishes)

Evidence: share of real forms whose root letters all survive in order. Strong and guttural roots: 99-100% in every tense. Final-he: 22% past / 65% future. Hollow: 21% past / 62% future. Initial-nun: 74% past / 60% future. Initial yod/waw: 28-41%. So the non-additive root x template x tense interactions are concentrated in the weak classes and are tense-specific.

1. **Positive control.** Run modular addition mod 97 through the exact same trainer (symbol table task). This must reproduce Power et al. before any claim about Hebrew.
2. **Main experiment: synthetic v2.** All four tenses; a mix of strong roots and rule-based weak roots (final-he, hollow, initial-nun, initial yod/waw, geminate). Dial: % weak roots in {0, 30, 60, 100}; train fraction in {20, 30, 50}%; alpha in {1, 3}. Measure per-class time-to-generalize and whether weak cells show a sharp jump. The strong roots serve as the within-model control.
3. **Real Hebrew table**, all 29 cells (already in canonical/forms), with per-class curves.
4. **Information plane** on all of the above: does root-letter / rule-class information in the query state rise when weak cells generalize?

Not recommended as the primary design:
* weak verbs only: additive within a single class;
* future only: drops the tense x class interaction; can be sliced post hoc from (2).
