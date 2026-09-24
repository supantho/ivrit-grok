# Dataset card — HEB_MORPH_GROK_v0.1

## Scientific purpose

A controlled testbed for **neural systematic generalization, grokking and mechanistic interpretability**.
The question is whether small networks trained on Hebrew verb inflection spontaneously form reusable internal
representations of (i) the consonantal **root**, (ii) the **binyan** (templatic verb class), and (iii) the
**inflectional features** (person, number, gender, tense).

Hebrew is well suited because these three factors are orthogonal and interleaved non-concatenatively in the surface
form. The dataset keeps **ROOT, BINYAN, PARADIGM CELL, SURFACE FORM and PROVENANCE as cleanly separable variables**,
so that generalization can be tested along each axis independently (novel roots, novel root×binyan combinations,
novel paradigm cells, novel root classes).

This release contains data only. No model has been trained.

## Contents

| component | location | notes |
|---|---|---|
| canonical DB | `data/canonical/{roots,lexemes,forms,provenance,validation}.{parquet,tsv}` | master tables; see `data/metadata/SCHEMA.md` |
| past-reinflection view | `data/derived/past_reinflection/` | PST.3MSG + target features → target form; vocalized and unvocalized |
| full-paradigm view | `data/derived/full_paradigm/` | every ordered pair of distinct past cells (for cell-completion studies) |
| synthetic v1 | `data/derived/synthetic/v1/` | 1000 roots × 7 templates × 9 past cells; own canonical tables, views, splits |
| split manifests | `data/splits/<view>/<split>/manifest.json` | IDs per partition + stats; verified at generation |
| QC reports | `data/reports/` | conflicts, duplicates, syncretism, coverage, completeness, distributions, manual-inspection sheet |
| manifest | `data/metadata/dataset_manifest.json` | versions, seeds, counts, sha256 checksums |

Exact counts are in `data/reports/corpus_summary.md` (generated).

## Linguistic phenomena included

* All seven binyanim (PAAL, NIFAL, PIEL, PUAL, HIFIL, HUFAL, HITPAEL), with normalized labels; source labels are kept in `binyan_raw`.
* Strong and weak roots: initial-nun (assimilation, e.g. נִתַּנּוּ), initial yod/waw (נוֹשַׁב), hollow (הוּקַם, הִתְכּוֹנֵן), final-he (הִרְבֵּיתִי), final-alef (בִּטֵּאתִי), geminate (הוּסַב), guttural-containing (hataf vowels, compensatory lengthening), quadriliteral (הִתְחַשְׁמֵל), and hitpael sibilant metathesis/assimilation.
* Full UniMorph paradigms: 9 past, 4 present (participle), 10 future, 4 imperative cells, infinitive, and verbal noun (MSDR). Past is the primary target; nothing else was dropped.
* Both writing systems: **vocalized** (niqqud) and **plene unvocalized** (ktiv male, as written in modern texts), plus the deterministic defective form (`form_stripped`).
* **Syncretism**, kept and labelled rather than cleaned away:
  * unvocalized PST.2MSG = PST.2FSG (כתבת) in almost every lexeme;
  * PST.3PL = IMP.2MPL;
  * PST.3MSG = PRS.MSG in some binyanim;
  * PST.1PL = PST.3PL under nun assimilation;
  * and more (`reports/syncretism_patterns.tsv`).

## Sources and licences

UniMorph Hebrew (CC BY-SA 3.0, commit `b2bff12`) is the source of the inflected forms. Hebrew Wiktionary (CC BY-SA 4.0,
dump 2026-09-01) supplies roots, binyanim, stated גזרות, glosses and form corroboration. Full details are in
`data/metadata/SOURCES.md`. The dataset is distributed under **CC BY-SA 4.0** with attribution.

## Preprocessing

1. **Normalization:** every string is Unicode **NFC** before any comparison. The raw string is kept (`form_raw`) next to `form_nfc`, the vocalized form, the plene form and the stripped form. Nonstandard combining marks, marks on impossible base letters (e.g. a shin dot on נ), repeated marks and non-Hebrew characters are logged in `validation`.
2. **UniMorph:**
   * Paradigms are segmented as runs of lines sharing a lemma. Two identical duplicate paradigms were merged (both sources kept in provenance).
   * Vocalized and plene paradigms are paired by letters-modulo-ו/י, with ties broken by distance to a predicted plene spelling. All 1,042 are paired: 1,028 fully, 14 partially.
   * The partial pairs are genuine disagreements between the two files, e.g. כָּרַתִּי vs כרתתי. They are kept and flagged, not reconciled.
3. **Wiktionary:**
   * Only structured templates are parsed.
   * Conjugation tables whose root parameter contradicts their own forms (e.g. the template placeholder פ־ע־ל on "סגר (שורש)") are checked against the page-title root; when the title root is used instead, this is flagged.
   * `-אין-` cells are recorded as explicit absence claims.
4. **Matching:**
   * Evidence is collected for each UniMorph paradigm from:
     * exact vocalized headword matches;
     * orthographic-variant-neutral ("loose") matches;
     * conjugation-table past and infinitive matches;
     * full-table matches;
     * plene-spelling matches (weak evidence, used only if nothing vocalized matched).
   * Root and binyan are assigned only if all evidence agrees. Otherwise they are left NULL and reported. String similarity is never used.
   * Every evidence record is stored in `lexemes.match_evidence` (JSON).
5. **Checks on each lexeme:**
   * Binyan vs a paradigm-shape heuristic: 833 of 834 agree. The single disagreement, רִצֵּד, is labelled PAAL by Wiktionary but conjugates as PIEL.
   * Root radicals vs the citation form.
   * Shin/sin dot vs the vocalized lemma.
   * Lemma vs the PST.3MSG cell.
   * Wiktionary tables vs UniMorph forms, cell by cell.
6. **Shin/sin:** roots keep שׁ/שׂ, taken from the source radicals or read off the lexeme's own vocalized lemma (flagged). Roots that differ only in the dot share an `orthographic_root_group_id`. Root-holdout splits group by this ID, so identical unvocalized strings cannot leak across a split.

## Known limitations

* **Coverage:** 1,042 UniMorph paradigms. 178 of them have **no Hebrew Wiktionary entry or root page**, so they have no root or binyan and are unusable for root- or binyan-based splits. Adding English Wiktionary is the most promising fix.
* **Wiktionary-only lexemes:** 1,903 lexemes come only from Wiktionary. Most have just the PST.3MSG citation form, plus a few table cells, and no plene forms beyond the citation. Their `modern_hebrew` is unknown unless tagged.
* **Source independence:** UniMorph Hebrew was built from Wiktionary, so agreement between the two is not independent confirmation.
* **Normative vocalization:** the niqqud follows Wiktionary/Academy conventions. Spoken Modern Hebrew often differs, e.g. unvocalized כרתתי (karátti) vs the normative כָּרַתִּי.
* **Traditional גזרה:** stated for only about 35% of roots. It is never inferred.
* **Glosses:** English glosses are section-level and not sense-disambiguated.
* **Heuristic checks:** the binyan-shape heuristic and the predicted-plene tie-breaker are rule-based. They are used only to check and pair data, never to create annotations.
* **Synthetic data** is deliberately idealized: strong roots only, no phonological processes, and defective unvocalized spelling.

## Ambiguity and syncretism

* **Ambiguity is never resolved silently.** `reports/conflicts.tsv` lists every root, binyan, shin/sin, lemma, form and absence conflict along with its evidence.
* `reports/duplicates.tsv` lists duplicate root×binyan records (variant paradigms such as קָטַן/קָטֹן and דַּל/דָּלַל), merged identical source paradigms, and vocalized forms shared across lexemes.
* **Syncretism** is a property of the data and is kept as is: `forms.is_syncretic*`, `forms.syncretism_group*`, `reports/syncretism.tsv`. In the derived views, each example carries `analysis__target_is_syncretic` and `analysis__target_same_form_past_cells`. Cell-completion split stats also count evaluation targets whose string equals an exposed cell's string (trivial copy solutions).

## Quality tiers

Coarse tiers plus exact `quality_flags` on every record; the flag vocabulary is in `reports/validation_summary.tsv`.

* **GOLD**, lexeme level. All of the following:
  * root and binyan each supported by ≥2 *different kinds* of Wiktionary record (entry page + conjugation/appendix table), with no disagreeing evidence;
  * the paradigm-shape binyan check agrees;
  * the root radicals are found in the citation form;
  * vocalized and plene paradigms pair completely;
  * no conflict flags.

  A form is GOLD if its lexeme is GOLD, both spellings are present and consistent, and no Wiktionary table contradicts it.
* **SILVER:** root and binyan from at least one structured record, and no failed checks or conflicts.
* **QUESTIONABLE:** missing or ambiguous root or binyan, a source conflict, a failed check, a duplicate root×binyan, a malformed string, or root/binyan inferred only from unvocalized spelling.

Form tiers are never better than their lexeme's tier. Default experimental eligibility (`analysis__eligible_default`) excludes QUESTIONABLE lexemes and forms, and requires root and binyan.

## Split definitions (`data/splits/`)

Splits assign **pair_ids** (`<lexeme>:<source cell>><target cell>`). The same assignment applies to the vocalized and
unvocalized conditions. Seeds 0–4 (class-OOD: 0–2). Each manifest is checked by `hebmorph.splits.verify` and by `tests/`.

| strategy | grouping | guarantee |
|---|---|---|
| `iid_seed*` | example | random 80/10/10 baseline; lexemes shared by design |
| `lexeme_holdout_seed*` | root×binyan | test lexemes unseen (variant paradigms of one root×binyan kept together); roots may be shared |
| `root_holdout_seed*` | orthographic root group | roots(train) ∩ roots(test) = ∅; all binyanim of a root on one side |
| `root_binyan_holdout_seed*` | root×binyan | every test root ∈ train roots, every test binyan ∈ train binyanim, (root, binyan) ∉ train |
| `cell_k{2,4,6,8}_seed*` | lexeme × cell | each eligible lexeme exposes PST.3MSG + k−1 random cells; train = exposed targets, dev/test = held-out targets; same lexeme on both sides **by design**; dev/test lexeme-disjoint |
| `rootclass_<P>_seed*` | orthographic root group | test = all roots with structural property P (strong, final_he, hollow_candidate, initial_nun, geminate, guttural_containing, initial_yod_waw, quadriliteral); none in train/dev |
| `tradclass_<G>_seed*` | orthographic root group | same, by the traditional גזרה G *as stated by Wiktionary*; roots with no stated גזרה are `excluded` |

`full_paradigm` has `cell_k` splits in which train contains only exposed→exposed pairs and test contains only
exposed→held-out pairs. Per-split distributions (root, binyan, root class, cell, form length, letters) are in
each `manifest.json["stats"]`; counts at every level are in `reports/split_summary.tsv`.

## Information intentionally hidden from neural models

A model sees only `source_form`, `target_features` (e.g. `PST,2,SG,FEM`) and `target_form`, as returned by
`hebmorph.loader.load_split`. Deliberately hidden:

* root, radicals, root ID and orthographic group;
* binyan and template;
* structural and traditional root class;
* lexeme ID and lemma;
* quality tiers and flags;
* syncretism annotations;
* provenance and source IDs.

These are kept in `analysis__*` columns for probing and evaluation only. Test `test_loader_exposes_only_visible_fields` enforces this.
