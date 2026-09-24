# Corpus summary — HEB_MORPH_GROK_v0.1

_Generated 2026-09-24 20:05 UTC by scripts/50_summary_and_manifest.py. All numbers are computed from data/canonical and data/derived._

## Headline counts

| quantity | value |
|---|---|
| roots | 1685 |
| triliteral roots | 1500 |
| quadriliteral roots | 174 |
| roots with 5+ radicals | 11 |
| lexemes (root × binyan records) | 2945 |
|   of which with a UniMorph paradigm | 1042 |
|   of which Wiktionary-only (citation forms, few cells) | 1903 |
| forms (lexeme × cell rows) | 33802 |
| past-tense forms | 11353 |
| lexemes with complete 9-cell past paradigm | 1051 |
|   UniMorph lexemes with complete past paradigm | 1042 |
| average past completeness (all lexemes) | 0.428 |
| average past completeness (UniMorph lexemes) | 1.000 |
| forms with vocalized string | 33802 |
| forms with plene unvocalized string | 31202 |
| syncretic cells (vocalized) | 5274 (15.6%) |
| syncretic cells (unvocalized) | 9075 (29.1%) |
| syncretic PAST cells (unvocalized) | 3363 (30.4%) |
| source conflicts (rows in conflicts.tsv) | 259 |
| ambiguous matches (lexemes with >1 root or binyan supported) | 49 |
| UniMorph lexemes with no Wiktionary match (no root/binyan) | 178 |
| QUESTIONABLE lexemes | 315 |
| QUESTIONABLE forms | 6432 |
| past-reinflection examples (both conditions) | 16744 |
| usable past-reinflection examples (eligible_default), vocalized | 6594 |
| usable past-reinflection examples (eligible_default), unvocalized | 6530 |

## Quality tiers

| tier | roots | lexemes | UniMorph lexemes | forms |
|---|---|---|---|---|
| GOLD | 199 | 259 | 259 | 7444 |
| SILVER | 1456 | 2371 | 565 | 19926 |
| QUESTIONABLE | 30 | 315 | 218 | 6432 |

## By binyan

| binyan | lexemes | UniMorph lexemes | complete past | GOLD | SILVER | QUESTIONABLE | forms |
|---|---|---|---|---|---|---|---|
| PAAL | 998 | 346 | 346 | 109 | 848 | 41 | 11293 |
| NIFAL | 186 | 47 | 49 | 13 | 162 | 11 | 1697 |
| PIEL | 831 | 194 | 198 | 59 | 732 | 40 | 6960 |
| PUAL | 121 | 27 | 28 | 6 | 113 | 2 | 790 |
| HIFIL | 331 | 122 | 124 | 48 | 269 | 14 | 4165 |
| HUFAL | 47 | 26 | 26 | 2 | 39 | 6 | 647 |
| HITPAEL | 246 | 95 | 95 | 22 | 208 | 16 | 3143 |
| UNKNOWN | 185 | 185 | 185 | 0 | 0 | 185 | 5107 |

## By root structural class (computed from radicals; not a traditional label)

| structural_class | roots | lexemes | UniMorph lexemes | mean past completeness |
|---|---|---|---|---|
| strong | 428 | 707 | 244 | 0.420 |
| guttural_containing | 383 | 606 | 190 | 0.390 |
| quadriliteral | 185 | 216 | 35 | 0.259 |
| hollow_candidate | 181 | 308 | 84 | 0.354 |
| final_he | 145 | 251 | 87 | 0.419 |
| geminate | 132 | 239 | 43 | 0.271 |
| initial_nun | 89 | 175 | 61 | 0.426 |
| initial_yod_waw | 63 | 111 | 39 | 0.463 |
| initial_alef | 54 | 91 | 38 | 0.482 |
| final_alef | 25 | 48 | 28 | 0.630 |

## Traditional גזרה (only where Wiktionary states it)

| traditional_class | roots |
|---|---|
| NOT_STATED | 1099 |
| SHLEMIM | 270 |
| AYIN_WAW_YOD | 60 |
| QUADRILITERAL | 52 |
| AYIN_AYIN | 45 |
| LAMED_YOD_HE | 36 |
| PE_NUN | 30 |
| LAMED_ALEF | 16 |
| PE_YOD_WAW | 16 |
| QUINQUILITERAL | 8 |
| AYIN_AYIN+SHLEMIM | 5 |
| PE_ALEF | 5 |
| PE_GUTTURAL | 4 |
| PE_NUN+SHLEMIM | 4 |
| LAMED_YOD_HE+PE_NUN | 4 |
| PE_YOD_TSADI | 4 |
| AYIN_GUTTURAL | 4 |
| LAMED_YOD_HE+PE_YOD_WAW | 4 |
| LAMED_GUTTURAL | 3 |
| AYIN_WAW_YOD+SHLEMIM | 3 |

## Conflicts and duplicates

| type | n |
|---|---|
| conflict: WIKT_FORM_MISMATCH | 121 |
| conflict: VOC_UNVOC_INCONSISTENT | 65 |
| conflict: WIKT_ONLY_EVIDENCE_DISAGREES | 31 |
| conflict: WIKT_TABLE_ASSERTS_ROOT_BINYAN_ABSENT | 12 |
| conflict: ROOT_AMBIGUOUS | 12 |
| conflict: BINYAN_AMBIGUOUS | 6 |
| conflict: ROOT_FORM_MISMATCH | 5 |
| conflict: BINYAN_SHAPE_DISAGREES | 3 |
| conflict: SHIN_SIN_MISMATCH | 2 |
| conflict: CELL_CONFLICT_UNVOCALIZED | 1 |
| conflict: LEMMA_CITATION_MISMATCH | 1 |
| duplicate: SAME_VOCALIZED_FORM_DIFFERENT_LEXEMES | 175 |
| duplicate: DUPLICATE_ROOT_BINYAN | 28 |
| duplicate: WIKT_ENTRY_FOR_EXISTING_ROOT_BINYAN | 6 |

## Splits (real data, seed 0; counts per partition)

Levels: pair_ids = examples per condition before per-condition filtering; forms = distinct target forms; root×binyan = distinct lexical pairs.

| view | split | partition | examples voc | examples unv | forms | lexemes | roots | root×binyan |
|---|---|---|---|---|---|---|---|---|
| past_reinflection | iid_seed0 | train | 5275 | 5224 | 5275 | 832 | 599 | 832 |
| past_reinflection | iid_seed0 | dev | 660 | 655 | 660 | 467 | 384 | 467 |
| past_reinflection | iid_seed0 | test | 659 | 651 | 659 | 491 | 399 | 491 |
| past_reinflection | lexeme_holdout_seed0 | train | 5267 | 5203 | 5267 | 666 | 505 | 666 |
| past_reinflection | lexeme_holdout_seed0 | dev | 663 | 663 | 663 | 83 | 80 | 83 |
| past_reinflection | lexeme_holdout_seed0 | test | 664 | 664 | 664 | 83 | 80 | 83 |
| past_reinflection | root_holdout_seed0 | train | 5245 | 5189 | 5245 | 662 | 480 | 662 |
| past_reinflection | root_holdout_seed0 | dev | 685 | 677 | 685 | 87 | 59 | 87 |
| past_reinflection | root_holdout_seed0 | test | 664 | 664 | 664 | 83 | 60 | 83 |
| past_reinflection | root_binyan_holdout_seed0 | train | 5277 | 5229 | 5277 | 666 | 599 | 666 |
| past_reinflection | root_binyan_holdout_seed0 | dev | 659 | 659 | 659 | 83 | 74 | 83 |
| past_reinflection | root_binyan_holdout_seed0 | test | 658 | 642 | 658 | 83 | 69 | 83 |
| past_reinflection | cell_k2_seed0 | train | 818 | 810 | 818 | 818 | 589 | 818 |
| past_reinflection | cell_k2_seed0 | dev | 574 | 574 | 574 | 82 | 78 | 82 |
| past_reinflection | cell_k2_seed0 | test | 5152 | 5096 | 5152 | 736 | 544 | 736 |
| past_reinflection | cell_k4_seed0 | train | 2454 | 2430 | 2454 | 818 | 589 | 818 |
| past_reinflection | cell_k4_seed0 | dev | 410 | 410 | 410 | 82 | 81 | 82 |
| past_reinflection | cell_k4_seed0 | test | 3680 | 3640 | 3680 | 736 | 537 | 736 |
| past_reinflection | cell_k6_seed0 | train | 4090 | 4050 | 4090 | 818 | 589 | 818 |
| past_reinflection | cell_k6_seed0 | dev | 246 | 246 | 246 | 82 | 82 | 82 |
| past_reinflection | cell_k6_seed0 | test | 2208 | 2184 | 2208 | 736 | 544 | 736 |
| past_reinflection | cell_k8_seed0 | train | 5726 | 5670 | 5726 | 818 | 589 | 818 |
| past_reinflection | cell_k8_seed0 | dev | 82 | 81 | 82 | 82 | 80 | 82 |
| past_reinflection | cell_k8_seed0 | test | 736 | 729 | 736 | 736 | 539 | 736 |
| past_reinflection | rootclass_strong_seed0 | train | 4275 | 4243 | 4275 | 539 | 392 | 539 |
| past_reinflection | rootclass_strong_seed0 | dev | 432 | 416 | 432 | 54 | 43 | 54 |
| past_reinflection | rootclass_strong_seed0 | test | 1887 | 1871 | 1887 | 239 | 164 | 239 |
| past_reinflection | rootclass_final_he_seed0 | train | 5276 | 5228 | 5276 | 667 | 480 | 667 |
| past_reinflection | rootclass_final_he_seed0 | dev | 590 | 574 | 590 | 74 | 53 | 74 |
| past_reinflection | rootclass_final_he_seed0 | test | 728 | 728 | 728 | 91 | 66 | 91 |
| past_reinflection | rootclass_hollow_candidate_seed0 | train | 5430 | 5366 | 5430 | 684 | 483 | 684 |
| past_reinflection | rootclass_hollow_candidate_seed0 | dev | 531 | 531 | 531 | 67 | 55 | 67 |
| past_reinflection | rootclass_hollow_candidate_seed0 | test | 633 | 633 | 633 | 81 | 61 | 81 |
| past_reinflection | rootclass_initial_nun_seed0 | train | 5296 | 5256 | 5296 | 669 | 497 | 669 |
| past_reinflection | rootclass_initial_nun_seed0 | dev | 731 | 715 | 731 | 92 | 55 | 92 |
| past_reinflection | rootclass_initial_nun_seed0 | test | 567 | 559 | 567 | 71 | 47 | 71 |
| past_reinflection | rootclass_geminate_seed0 | train | 5639 | 5591 | 5639 | 712 | 510 | 712 |
| past_reinflection | rootclass_geminate_seed0 | dev | 632 | 616 | 632 | 79 | 57 | 79 |
| past_reinflection | rootclass_geminate_seed0 | test | 323 | 323 | 323 | 41 | 32 | 41 |
| past_reinflection | rootclass_guttural_containing_seed0 | train | 2953 | 2889 | 2953 | 375 | 253 | 375 |
| past_reinflection | rootclass_guttural_containing_seed0 | dev | 283 | 283 | 283 | 36 | 28 | 36 |
| past_reinflection | rootclass_guttural_containing_seed0 | test | 3358 | 3358 | 3358 | 421 | 318 | 421 |
| past_reinflection | rootclass_initial_yod_waw_seed0 | train | 5631 | 5607 | 5631 | 709 | 516 | 709 |
| past_reinflection | rootclass_initial_yod_waw_seed0 | dev | 601 | 601 | 601 | 77 | 57 | 77 |
| past_reinflection | rootclass_initial_yod_waw_seed0 | test | 362 | 322 | 362 | 46 | 26 | 46 |
| past_reinflection | rootclass_quadriliteral_seed0 | train | 5728 | 5664 | 5728 | 720 | 515 | 720 |
| past_reinflection | rootclass_quadriliteral_seed0 | dev | 607 | 607 | 607 | 79 | 57 | 79 |
| past_reinflection | rootclass_quadriliteral_seed0 | test | 259 | 259 | 259 | 33 | 27 | 33 |
| past_reinflection | tradclass_SHLEMIM_seed0 | train | 1452 | 1420 | 1452 | 184 | 112 | 184 |
| past_reinflection | tradclass_SHLEMIM_seed0 | dev | 160 | 160 | 160 | 20 | 12 | 20 |
| past_reinflection | tradclass_SHLEMIM_seed0 | test | 1641 | 1625 | 1641 | 207 | 127 | 207 |
| past_reinflection | tradclass_SHLEMIM_seed0 | excluded | 3341 | 3325 | 3341 | 421 | 348 | 421 |
| past_reinflection | tradclass_PE_NUN_seed0 | train | 2669 | 2621 | 2669 | 338 | 206 | 338 |
| past_reinflection | tradclass_PE_NUN_seed0 | dev | 280 | 280 | 280 | 35 | 24 | 35 |
| past_reinflection | tradclass_PE_NUN_seed0 | test | 304 | 304 | 304 | 38 | 21 | 38 |
| past_reinflection | tradclass_PE_NUN_seed0 | excluded | 3341 | 3325 | 3341 | 421 | 348 | 421 |
| past_reinflection | tradclass_PE_YOD_WAW_seed0 | train | 2802 | 2770 | 2802 | 354 | 215 | 354 |
| past_reinflection | tradclass_PE_YOD_WAW_seed0 | dev | 235 | 235 | 235 | 30 | 22 | 30 |
| past_reinflection | tradclass_PE_YOD_WAW_seed0 | test | 216 | 200 | 216 | 27 | 14 | 27 |
| past_reinflection | tradclass_PE_YOD_WAW_seed0 | excluded | 3341 | 3325 | 3341 | 421 | 348 | 421 |
| past_reinflection | tradclass_PE_ALEF_seed0 | train | 2925 | 2893 | 2925 | 370 | 223 | 370 |
| past_reinflection | tradclass_PE_ALEF_seed0 | dev | 224 | 208 | 224 | 28 | 22 | 28 |
| past_reinflection | tradclass_PE_ALEF_seed0 | test | 104 | 104 | 104 | 13 | 6 | 13 |
| past_reinflection | tradclass_PE_ALEF_seed0 | excluded | 3341 | 3325 | 3341 | 421 | 348 | 421 |
| past_reinflection | tradclass_AYIN_WAW_YOD_seed0 | train | 2692 | 2644 | 2692 | 339 | 198 | 339 |
| past_reinflection | tradclass_AYIN_WAW_YOD_seed0 | dev | 192 | 192 | 192 | 24 | 19 | 24 |
| past_reinflection | tradclass_AYIN_WAW_YOD_seed0 | test | 369 | 369 | 369 | 48 | 34 | 48 |
| past_reinflection | tradclass_AYIN_WAW_YOD_seed0 | excluded | 3341 | 3325 | 3341 | 421 | 348 | 421 |
| past_reinflection | tradclass_AYIN_AYIN_seed0 | train | 2717 | 2669 | 2717 | 344 | 213 | 344 |
| past_reinflection | tradclass_AYIN_AYIN_seed0 | dev | 384 | 384 | 384 | 48 | 22 | 48 |
| past_reinflection | tradclass_AYIN_AYIN_seed0 | test | 152 | 152 | 152 | 19 | 16 | 19 |
| past_reinflection | tradclass_AYIN_AYIN_seed0 | excluded | 3341 | 3325 | 3341 | 421 | 348 | 421 |
| past_reinflection | tradclass_LAMED_YOD_HE_seed0 | train | 2503 | 2455 | 2503 | 316 | 198 | 316 |
| past_reinflection | tradclass_LAMED_YOD_HE_seed0 | dev | 406 | 406 | 406 | 52 | 28 | 52 |
| past_reinflection | tradclass_LAMED_YOD_HE_seed0 | test | 344 | 344 | 344 | 43 | 25 | 43 |
| past_reinflection | tradclass_LAMED_YOD_HE_seed0 | excluded | 3341 | 3325 | 3341 | 421 | 348 | 421 |
| past_reinflection | tradclass_LAMED_ALEF_seed0 | train | 2709 | 2677 | 2709 | 343 | 216 | 343 |
| past_reinflection | tradclass_LAMED_ALEF_seed0 | dev | 336 | 320 | 336 | 42 | 24 | 42 |
| past_reinflection | tradclass_LAMED_ALEF_seed0 | test | 208 | 208 | 208 | 26 | 11 | 26 |
| past_reinflection | tradclass_LAMED_ALEF_seed0 | excluded | 3341 | 3325 | 3341 | 421 | 348 | 421 |
| past_reinflection | tradclass_QUADRILITERAL_seed0 | train | 2885 | 2837 | 2885 | 365 | 219 | 365 |
| past_reinflection | tradclass_QUADRILITERAL_seed0 | dev | 312 | 312 | 312 | 39 | 26 | 39 |
| past_reinflection | tradclass_QUADRILITERAL_seed0 | test | 56 | 56 | 56 | 7 | 6 | 7 |
| past_reinflection | tradclass_QUADRILITERAL_seed0 | excluded | 3341 | 3325 | 3341 | 421 | 348 | 421 |
| full_paradigm | cell_k2_seed0 | train | 1636 | 1620 | 1636 | 818 | 589 | 818 |
| full_paradigm | cell_k2_seed0 | dev | 1148 | 1148 | 574 | 82 | 78 | 82 |
| full_paradigm | cell_k2_seed0 | test | 10304 | 10192 | 5152 | 736 | 544 | 736 |
| full_paradigm | cell_k2_seed0 | excluded | 45808 | 45360 | 7362 | 818 | 589 | 818 |
| full_paradigm | cell_k4_seed0 | train | 9816 | 9720 | 3272 | 818 | 589 | 818 |
| full_paradigm | cell_k4_seed0 | dev | 1640 | 1640 | 410 | 82 | 81 | 82 |
| full_paradigm | cell_k4_seed0 | test | 14720 | 14560 | 3680 | 736 | 537 | 736 |
| full_paradigm | cell_k4_seed0 | excluded | 32720 | 32400 | 7362 | 818 | 589 | 818 |
| full_paradigm | cell_k6_seed0 | train | 24540 | 24300 | 4908 | 818 | 589 | 818 |
| full_paradigm | cell_k6_seed0 | dev | 1476 | 1476 | 246 | 82 | 82 | 82 |
| full_paradigm | cell_k6_seed0 | test | 13248 | 13104 | 2208 | 736 | 544 | 736 |
| full_paradigm | cell_k6_seed0 | excluded | 19632 | 19440 | 7362 | 818 | 589 | 818 |
| full_paradigm | cell_k8_seed0 | train | 45808 | 45360 | 6544 | 818 | 589 | 818 |
| full_paradigm | cell_k8_seed0 | dev | 656 | 648 | 82 | 82 | 80 | 82 |
| full_paradigm | cell_k8_seed0 | test | 5888 | 5832 | 736 | 736 | 539 | 736 |
| full_paradigm | cell_k8_seed0 | excluded | 6544 | 6480 | 6544 | 818 | 589 | 818 |

All 91 split manifests (all seeds) are listed in reports/split_summary.tsv; every manifest was verified by hebmorph.splits.verify at generation time.

## Synthetic v1 (separate; never mixed with real data)

1000 roots × 7 templates = 7000 lexemes, 63000 past forms (config: derived/synthetic/v1/synthetic_config.json).

