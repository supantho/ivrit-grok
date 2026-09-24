# Sources

All raw files live under `data/raw/` exactly as downloaded (never edited). `scripts/00_fetch_sources.sh`
re-acquires them at the pinned versions and verifies them. Retrieval dates are recorded in
`data/raw/retrieval_log.json`.

| source | URL | retrieved | version / commit | license | redistribution | how used |
|---|---|---|---|---|---|---|
| **UniMorph Hebrew** (`heb_voc`, `heb`, `heb.args`, `heb_unvoc.args`) | https://github.com/unimorph/heb | 2026-09-24 | git commit `b2bff12338caa1922e6aeea128c617730f93d334` (2023-01-09) | CC BY-SA 3.0 (repo README) | Yes, with attribution and share-alike | **Primary source of inflected forms.** `heb_voc` gives the vocalized paradigms, `heb` the plene (ktiv male) unvocalized paradigms. `.args` files are cross-checked (29,484/29,484 verb triples identical); their `-` rows are recorded as explicitly absent cells. |
| UniMorph Hebrew `from_UD_unvoc` | same repo, same commit | 2026-09-24 | same | CC BY-SA 3.0; derived from UD Hebrew-HTB (CC BY-NC-SA 4.0 upstream) | Only used as a boolean signal; no forms are copied from it into our tables | **Attestation signal only** (`ud_attested`): does a plene form occur as a verb in the UD-derived list? |
| **Hebrew Wiktionary** dump `pages-articles` | https://dumps.wikimedia.org/hewiktionary/20260901/hewiktionary-20260901-pages-articles.xml.bz2 | 2026-09-24 | dump 2026-09-01, sha1 `eb96787cc3c45a938259aeede923ccc7cdc143a7` (verified against the dump's `sha1sums.txt`) | CC BY-SA 4.0 (text; also GFDL) | Yes, with attribution and share-alike; each record keeps its page and revision ID | **Secondary structured source**: roots, binyanim, stated גזרה, vocalized citation forms, plene spelling, infinitives, English glosses, and corroboration of forms. Only structured templates are parsed: `{{ניתוח דקדוקי לפועל}}`, `{{שרש3/4/5}}`, `{{שרש}}`, `{{נטיות פעלים}}`, `{{נטיות פעל בבנין}}`, `{{ת|אנגלית}}`, `{{רובד}}`. No rendered HTML is used. Every page used is cached verbatim in `data/interim/wiktionary/relevant_pages_wikitext.parquet`. |

## Not used

* **YAP or other analyzers:** not used. Their lexicons have unclear redistribution terms and were not needed for validation. Validation is done with transparent in-house checks instead: paradigm-shape binyan check, root/form consistency, and plene/niqqud consistency.
* **Proprietary conjugation sites** (e.g. Pealim) were **not** scraped.
* **English Wiktionary** was considered as an extra independent root/binyan source but is not used in v0.1.

## Attribution / licence of this dataset

Because UniMorph (CC BY-SA 3.0) and Wiktionary (CC BY-SA 4.0) content is included, the canonical and derived
tables are distributed under **CC BY-SA 4.0**, with attribution to the UniMorph Hebrew contributors
(annotator: Omer Goldman) and to Hebrew Wiktionary contributors. The synthetic dataset contains no third-party
content.

## Independence caveat

UniMorph Hebrew was itself built from Wiktionary conjugation tables (see its README). Agreement between
UniMorph and Hebrew Wiktionary is therefore **not** fully independent evidence. The GOLD tier requires agreement
between *separately edited* Wiktionary records (an entry page plus a root-page or appendix conjugation table)
plus algorithmic checks, and should be read with this in mind.
