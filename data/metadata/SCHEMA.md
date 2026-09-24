# Schema (generated from hebmorph/schema.py — do not edit by hand)

schema_version 0.1.0. Machine-readable: data/metadata/schema.json.

## roots  (primary key: `root_id`)

| column | type | nullable | description |
|---|---|---|---|
| `root_id` | string | no | Stable ID: 'RT-' + sha1('root\|' + root_normalized)[:10]. Synthetic roots use 'SYN-RT-...'. |
| `root_raw` | string | no | Root exactly as first written by the primary source record (e.g. template text 'שׁ־מ־ר' or '{{שרש3\|..}}'). |
| `root_normalized` | string | no | Root identity: radicals joined without separator, final forms normalized (ך->כ); ש carries its shin/sin dot (שׁ / שׂ) when resolved (from source radicals, else from the lexeme's vocalized lemma -- see lexemes.shin_sin_resolution), bare ש if unresolved. |
| `radical_1` | string | no | First radical (bare letter, shin/sin dot removed; see root_normalized for the dotted form). |
| `radical_2` | string | no | Second radical. |
| `radical_3` | string | no | Third radical. |
| `radical_4` | string | yes | Fourth radical (quadriliteral+ roots), else null. |
| `radical_5` | string | yes | Fifth radical (quinqueliteral roots, Wiktionary-only), else null. |
| `root_length` | Int64 | no | Number of radicals. |
| `traditional_class` | string | yes | Normalized traditional גזרה labels EXPLICITLY given by Wiktionary for this root's lexemes ('+'-joined, sorted), e.g. 'PE_NUN+LAMED_YOD_HE'. Null if no source states one. Never inferred. |
| `traditional_class_raw` | string | yes | The raw גזרה strings from the sources (' ; '-joined). |
| `r1_is_nun` | boolean | no | radical_1 == נ (computed from radicals). |
| `r1_is_yod` | boolean | no | radical_1 == י. |
| `r1_is_waw` | boolean | no | radical_1 == ו. |
| `r1_is_alef` | boolean | no | radical_1 == א. |
| `r2_is_yod` | boolean | no | radical_2 == י. |
| `r2_is_waw` | boolean | no | radical_2 == ו. |
| `r3_is_he` | boolean | no | radical_3 == ה (literal position 3, also for quadriliterals). |
| `r3_is_yod` | boolean | no | radical_3 == י. |
| `r3_is_alef` | boolean | no | radical_3 == א. |
| `rfinal_is_he` | boolean | no | Last radical == ה. |
| `r1_is_guttural` | boolean | no | radical_1 in {א,ה,ח,ע} (ר is NOT counted as guttural; see has_resh). |
| `r2_is_guttural` | boolean | no | radical_2 in {א,ה,ח,ע}. |
| `r3_is_guttural` | boolean | no | radical_3 in {א,ה,ח,ע}. |
| `has_resh` | boolean | no | Any radical == ר. |
| `is_geminate` | boolean | no | Triliteral root with radical_2 == radical_3. |
| `is_reduplicated_quadriliteral` | boolean | no | Quadriliteral with r1==r3 and r2==r4 (e.g. גלגל). |
| `has_weak_radical` | boolean | no | r1 in {נ,י,ו,א} or r2 in {ו,י} or r3 in {ה,י,א} or geminate (transparent definition; see README). |
| `structural_signature` | string | no | Per-radical symbols joined by '.': N=נ Y=י W=ו H=ה A=א X=ח E=ע R=ר C=other; '+GEM' suffix if geminate. e.g. כתב->C.C.C, נפל->N.C.C, קום->C.W.C. |
| `structural_class` | string | no | Single primary structural class by fixed priority (quadriliteral > geminate > hollow_candidate > final_he > initial_nun > initial_yod_waw > final_alef > initial_alef > guttural_containing > strong). Computed, not a traditional label. |
| `sc_strong` | boolean | no | No weak radical, no guttural (א ה ח ע), triliteral, not geminate. |
| `sc_final_he` | boolean | no | Triliteral, radical_3 == ה. |
| `sc_hollow_candidate` | boolean | no | Triliteral, radical_2 in {ו,י}. 'Candidate' because some such roots inflect as strong. |
| `sc_initial_nun` | boolean | no | radical_1 == נ. |
| `sc_initial_yod_waw` | boolean | no | radical_1 in {י,ו}. |
| `sc_geminate` | boolean | no | = is_geminate. |
| `sc_guttural_containing` | boolean | no | Any radical in {א,ה,ח,ע}. |
| `sc_quadriliteral` | boolean | no | root_length >= 4. |
| `root_orthographic` | string | no | root_normalized with shin/sin dots removed: what the root looks like in unvocalized text. |
| `orthographic_root_group_id` | string | no | 'RG-' + sha1(root_orthographic): shared by roots that differ only in shin/sin (e.g. שׂכר / שׁכר). Root-holdout splits group by this ID so that unvocalized strings cannot leak across the split. |
| `n_lexemes` | Int64 | no | Number of lexemes (root x binyan) with this root in lexemes table. |
| `binyanim` | string | no | Sorted '\|'-joined binyanim of this root's lexemes (unknown binyan -> '?'). |
| `source_primary` | string | no | Primary source asserting the root: 'hewiktionary' or 'synthetic_v1'. |
| `source_ids` | string | no | '\|'-joined source record identifiers supporting the root (see provenance). |
| `confidence` | Float64 | no | Max confidence_root over this root's lexemes. |
| `quality_tier` | string | no | Best quality tier among this root's lexemes. |
| `quality_flags` | string | yes | '\|'-joined flags; null if none. |

## lexemes  (primary key: `lexeme_id`)

| column | type | nullable | description |
|---|---|---|---|
| `lexeme_id` | string | no | Stable ID: 'LX-' + sha1(source + '\|' + NFC vocalized lemma [+ '\|k' for k-th distinct paradigm])[:10]. |
| `language` | string | no | ISO 639-3 'heb' (synthetic: 'x-syn-heb'). |
| `root_id` | string | yes | FK to roots. NULL if the root is missing or ambiguous (never guessed). |
| `root_radicals_dotted` | string | yes | Radicals as given by the source incl. shin/sin dots, '-'-joined (e.g. 'שׂ-ח-ק'); null if not given. |
| `shin_sin_resolution` | string | yes | How ש radicals got their dot: NA (no ש) \| SOURCE (root template) \| FORM (read off vocalized lemma) \| MIXED \| UNRESOLVED; null if no root. |
| `root_homonym_index` | string | yes | Wiktionary homonym index of the root (e.g. 'א','ב') if given. |
| `binyan` | string | yes | One of PAAL NIFAL PIEL PUAL HIFIL HUFAL HITPAEL; NULL if missing/ambiguous (never guessed or taken from the shape heuristic). |
| `binyan_raw` | string | yes | Raw binyan label(s) from sources ('\|'-joined). |
| `lemma_vocalized` | string | no | NFC vocalized lemma as given by the primary source (UniMorph heb_voc lemma column / Wiktionary headword). |
| `lemma_unvocalized` | string | yes | Plene (ktiv male) unvocalized lemma: UniMorph 'heb' lemma of the paired paradigm, or Wiktionary 'כתיב מלא'. |
| `infinitive_vocalized` | string | yes | NFIN cell, vocalized. |
| `infinitive_unvocalized` | string | yes | NFIN cell, plene unvocalized. |
| `citation_past_3ms_vocalized` | string | yes | PST.3MSG cell, vocalized. |
| `citation_past_3ms_unvocalized` | string | yes | PST.3MSG cell, plene unvocalized. |
| `gloss_en` | string | yes | English translations from {{ת\|אנגלית\|..}} in the matched Wiktionary section(s); null if none. Section-level, not sense-disambiguated. |
| `sense_id` | string | yes | Wiktionary homonym markers of the matched section(s) when several sections share root x binyan (e.g. 'משני:א\|משני:ב'); else null. |
| `modern_hebrew` | boolean | yes | True for UniMorph lexemes (source inventory = Modern Hebrew conjugation tables); True for Wiktionary-only lexemes only if a 'עברית חדשה' register tag is present; else null (unknown). |
| `attested` | boolean | no | Lexeme appears as an entry/paradigm in a lexicographic source (all real records True; synthetic False). |
| `has_unimorph_paradigm` | boolean | no | Lexeme has a UniMorph paradigm (else Wiktionary-only). |
| `source_primary` | string | no | 'unimorph_heb' \| 'hewiktionary' \| 'synthetic_v1'. |
| `source_ids` | string | no | '\|'-joined source record identifiers (see provenance). |
| `match_status` | string | no | MATCHED \| MATCHED_UNVOC_ONLY \| ROOT_AMBIGUOUS \| BINYAN_AMBIGUOUS \| ROOT_AND_BINYAN_AMBIGUOUS \| NO_MATCH \| WIKTIONARY_ONLY \| SYNTHETIC. |
| `match_method` | string | yes | '\|'-joined methods of the evidence that agrees with the assigned root/binyan. |
| `match_score` | Float64 | no | Noisy-OR of agreeing evidence weights (0 if nothing assigned). |
| `match_evidence` | string | yes | JSON list of all evidence records considered (incl. disagreeing ones). |
| `match_ambiguous` | boolean | no | True if >1 root or >1 binyan was supported by evidence. |
| `binyan_shape_heuristic` | string | yes | Binyan predicted from the vocalized paradigm shape (VALIDATION ONLY). |
| `binyan_shape_rule` | string | yes | Rule that produced binyan_shape_heuristic. |
| `voc_unvoc_pairing_status` | string | yes | PAIRED \| PAIRED_PARTIAL \| PAIRED_TIE_IDENTICAL \| AMBIGUOUS \| UNPAIRED (UniMorph lexemes). |
| `voc_unvoc_pairing_score` | Float64 | yes | Fraction of cells whose vocalized and plene forms agree on letters modulo ו/י. |
| `gizra_labels` | string | yes | Traditional גזרה labels stated by the matched Wiktionary entries of this lexeme. |
| `wiktionary_registers` | string | yes | {{רובד}} register tags in matched Wiktionary sections. |
| `n_cells` | Int64 | no | Number of paradigm cells in forms table. |
| `past_expected_cells` | Int64 | no | Always 9 (canonical past paradigm). |
| `past_observed_cells` | Int64 | no | Number of the 9 canonical past cells with a vocalized form. |
| `past_completeness` | Float64 | no | past_observed_cells / 9. |
| `past_complete` | boolean | no | past_observed_cells == 9. |
| `ud_attested` | boolean | no | Some plene form of this lexeme occurs as a verb form in UniMorph from_UD_unvoc (surface match only; may be a homograph). |
| `confidence_root` | Float64 | no | Noisy-OR of agreeing root evidence weights; 0 if missing/ambiguous; halved if ROOT_FORM_MISMATCH. |
| `confidence_binyan` | Float64 | no | Noisy-OR of agreeing binyan evidence weights; 0 if missing/ambiguous; halved if shape heuristic disagrees. |
| `confidence_lexeme` | Float64 | no | min(confidence_root, confidence_binyan) x (1 if voc/unvoc PAIRED else 0.8). |
| `quality_tier` | string | no | GOLD \| SILVER \| QUESTIONABLE (see README). |
| `quality_flags` | string | yes | '\|'-joined flags; null if none. |

## forms  (primary key: `form_id`)

| column | type | nullable | description |
|---|---|---|---|
| `form_id` | string | no | '<lexeme_id>.<canonical_cell>' (unique: one row per lexeme x cell). |
| `lexeme_id` | string | no | FK to lexemes. |
| `root_id` | string | yes | Copy of lexemes.root_id (analysis convenience). |
| `form_raw` | string | no | Surface string exactly as in the primary source (vocalized source when available). |
| `form_nfc` | string | no | NFC(form_raw). |
| `form_vocalized` | string | yes | NFC vocalized form (null if the only source is unvocalized). |
| `form_unvocalized` | string | yes | NFC plene (ktiv male) form from the unvocalized source; null if unavailable. NOT derived by stripping. |
| `form_stripped` | string | yes | strip_niqqud(form_vocalized): defective (ktiv haser) spelling, deterministic. |
| `form_vocalized_variants` | string | yes | '\|'-joined other vocalized forms given for the same lexeme x cell (e.g. duplicate source paradigms, Wiktionary variants). |
| `form_unvocalized_variants` | string | yes | '\|'-joined other plene spellings given for the same lexeme x cell. |
| `pos` | string | no | 'V' or 'V.MSDR' (verbal noun). |
| `finiteness` | string | no | FIN \| NFIN. |
| `tense` | string | no | PST \| PRS \| FUT \| NA. |
| `aspect` | string | no | Always NA (not morphologically marked / not annotated). |
| `mood` | string | no | IND \| IMP \| NA. |
| `person` | string | no | 1 \| 2 \| 3 \| NA (present participles and non-finite forms: NA). |
| `number` | string | no | SG \| PL \| NA. |
| `gender` | string | no | MASC \| FEM \| NA (1st person and PST.3PL: NA -- not expressed). |
| `voice` | string | no | Always NA: voice is not annotated by the sources; the binyan column carries voice-related information. |
| `binyan` | string | yes | Copy of lexemes.binyan. |
| `canonical_cell` | string | no | e.g. PST.1SG PST.2MSG PST.3PL PRS.FSG FUT.3MPL IMP.2FSG NFIN MSDR. |
| `is_past_cell` | boolean | no | canonical_cell in the 9 canonical past cells. |
| `unimorph_tag_raw` | string | yes | Original UniMorph tag (null for Wiktionary-only forms). |
| `is_syncretic` | boolean | no | Another cell of the same lexeme has an identical VOCALIZED form. |
| `syncretism_group` | string | yes | '<lexeme_id>:V<k>' shared by cells with identical vocalized forms; null if unique. |
| `is_syncretic_unvocalized` | boolean | no | Another cell of the same lexeme has an identical PLENE UNVOCALIZED form. |
| `syncretism_group_unvocalized` | string | yes | '<lexeme_id>:U<k>' shared by cells with identical plene forms; null if unique. |
| `plene_consistent` | boolean | yes | Vocalized and plene forms agree on letters modulo ו/י (null if either missing). |
| `wiktionary_corroborated` | boolean | yes | True: a Wiktionary table gives the same vocalized form in this cell; False: it gives a different one (conflict); null: no Wiktionary table value. |
| `ud_attested` | boolean | no | form_unvocalized occurs among UD-derived verb forms (surface match only). |
| `is_attested` | boolean | no | Form is taken from a source (not generated). |
| `is_generated` | boolean | no | Form was generated by a program (synthetic data only). |
| `source_primary` | string | no | 'unimorph_heb' \| 'hewiktionary' \| 'synthetic_v1'. |
| `source_ids` | string | no | '\|'-joined source record identifiers. |
| `confidence` | Float64 | no | confidence_lexeme x form-level factor (1; 0.5 if a form-level check failed). |
| `quality_tier` | string | no | GOLD \| SILVER \| QUESTIONABLE; never better than the lexeme's tier. |
| `quality_flags` | string | yes | '\|'-joined flags; null if none. |

## provenance  (primary key: `provenance_id`)

| column | type | nullable | description |
|---|---|---|---|
| `provenance_id` | string | no | Unique row ID. |
| `canonical_record_id` | string | no | root_id / lexeme_id / form_id. |
| `record_type` | string | no | root \| lexeme \| form. |
| `field_supported` | string | no | What this source record supports (e.g. form_vocalized, form_unvocalized, root, binyan, paradigm). |
| `source_name` | string | no | unimorph_heb \| hewiktionary \| synthetic_v1. |
| `source_url` | string | no | URL (UniMorph: blob URL at pinned commit with #L<line>; Wiktionary: page at pinned revision). |
| `source_file` | string | no | File name within the source (UniMorph file / dump file name). |
| `source_version_or_commit` | string | no | Git commit or dump date. |
| `source_record_identifier` | string | no | e.g. 'heb_voc:1234' or 'wikt:page=<id>:rev=<id>:sec=<offset>:tpl=<k>'. |
| `source_line_if_available` | Int64 | yes | 1-based line number in source file. |
| `retrieval_date` | string | no | Date the raw source was retrieved (ISO). |
| `parser_version` | string | no | Version of the parser that read this record. |
| `raw_source_value` | string | no | The raw source text supporting the field (line / template / cell value). |

## validation  (primary key: `check_id`)

| column | type | nullable | description |
|---|---|---|---|
| `check_id` | string | no | Unique row ID. |
| `record_id` | string | no | Affected record (or '*' for table-level checks). |
| `record_type` | string | no | root \| lexeme \| form \| table \| source. |
| `check_name` | string | no | Name of the check (see README). |
| `severity` | string | no | ERROR \| WARNING \| INFO. |
| `details` | string | yes | Human-readable details / values involved. |
