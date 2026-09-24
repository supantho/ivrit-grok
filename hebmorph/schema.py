"""Canonical table schemas.  The pipeline FAILS LOUDLY on any violation.

Each column: (name, dtype, nullable, description).  dtypes are pandas
extension dtypes: string, Int64, Float64, boolean.
"""
from __future__ import annotations

import json

import pandas as pd

from . import SCHEMA_VERSION
from .morphology import BINYANIM

TIERS = ["GOLD", "SILVER", "QUESTIONABLE"]
S, I, F, B = "string", "Int64", "Float64", "boolean"

ROOTS = [
    ("root_id", S, False, "Stable ID: 'RT-' + sha1('root|' + root_normalized)[:10]. Synthetic roots use 'SYN-RT-...'."),
    ("root_raw", S, False, "Root exactly as first written by the primary source record (e.g. template text 'שׁ־מ־ר' or '{{שרש3|..}}')."),
    ("root_normalized", S, False, "Root identity: radicals joined without separator, final forms normalized (ך->כ); ש carries its shin/sin dot (שׁ / שׂ) when resolved (from source radicals, else from the lexeme's vocalized lemma -- see lexemes.shin_sin_resolution), bare ש if unresolved."),
    ("radical_1", S, False, "First radical (bare letter, shin/sin dot removed; see root_normalized for the dotted form)."),
    ("radical_2", S, False, "Second radical."),
    ("radical_3", S, False, "Third radical."),
    ("radical_4", S, True, "Fourth radical (quadriliteral+ roots), else null."),
    ("radical_5", S, True, "Fifth radical (quinqueliteral roots, Wiktionary-only), else null."),
    ("root_length", I, False, "Number of radicals."),
    ("traditional_class", S, True, "Normalized traditional גזרה labels EXPLICITLY given by Wiktionary for this root's lexemes ('+'-joined, sorted), e.g. 'PE_NUN+LAMED_YOD_HE'. Null if no source states one. Never inferred."),
    ("traditional_class_raw", S, True, "The raw גזרה strings from the sources (' ; '-joined)."),
    ("r1_is_nun", B, False, "radical_1 == נ (computed from radicals)."),
    ("r1_is_yod", B, False, "radical_1 == י."),
    ("r1_is_waw", B, False, "radical_1 == ו."),
    ("r1_is_alef", B, False, "radical_1 == א."),
    ("r2_is_yod", B, False, "radical_2 == י."),
    ("r2_is_waw", B, False, "radical_2 == ו."),
    ("r3_is_he", B, False, "radical_3 == ה (literal position 3, also for quadriliterals)."),
    ("r3_is_yod", B, False, "radical_3 == י."),
    ("r3_is_alef", B, False, "radical_3 == א."),
    ("rfinal_is_he", B, False, "Last radical == ה."),
    ("r1_is_guttural", B, False, "radical_1 in {א,ה,ח,ע} (ר is NOT counted as guttural; see has_resh)."),
    ("r2_is_guttural", B, False, "radical_2 in {א,ה,ח,ע}."),
    ("r3_is_guttural", B, False, "radical_3 in {א,ה,ח,ע}."),
    ("has_resh", B, False, "Any radical == ר."),
    ("is_geminate", B, False, "Triliteral root with radical_2 == radical_3."),
    ("is_reduplicated_quadriliteral", B, False, "Quadriliteral with r1==r3 and r2==r4 (e.g. גלגל)."),
    ("has_weak_radical", B, False, "r1 in {נ,י,ו,א} or r2 in {ו,י} or r3 in {ה,י,א} or geminate (transparent definition; see README)."),
    ("structural_signature", S, False, "Per-radical symbols joined by '.': N=נ Y=י W=ו H=ה A=א X=ח E=ע R=ר C=other; '+GEM' suffix if geminate. e.g. כתב->C.C.C, נפל->N.C.C, קום->C.W.C."),
    ("structural_class", S, False, "Single primary structural class by fixed priority (quadriliteral > geminate > hollow_candidate > final_he > initial_nun > initial_yod_waw > final_alef > initial_alef > guttural_containing > strong). Computed, not a traditional label."),
    ("sc_strong", B, False, "No weak radical, no guttural (א ה ח ע), triliteral, not geminate."),
    ("sc_final_he", B, False, "Triliteral, radical_3 == ה."),
    ("sc_hollow_candidate", B, False, "Triliteral, radical_2 in {ו,י}. 'Candidate' because some such roots inflect as strong."),
    ("sc_initial_nun", B, False, "radical_1 == נ."),
    ("sc_initial_yod_waw", B, False, "radical_1 in {י,ו}."),
    ("sc_geminate", B, False, "= is_geminate."),
    ("sc_guttural_containing", B, False, "Any radical in {א,ה,ח,ע}."),
    ("sc_quadriliteral", B, False, "root_length >= 4."),
    ("root_orthographic", S, False, "root_normalized with shin/sin dots removed: what the root looks like in unvocalized text."),
    ("orthographic_root_group_id", S, False, "'RG-' + sha1(root_orthographic): shared by roots that differ only in shin/sin (e.g. שׂכר / שׁכר). Root-holdout splits group by this ID so that unvocalized strings cannot leak across the split."),
    ("n_lexemes", I, False, "Number of lexemes (root x binyan) with this root in lexemes table."),
    ("binyanim", S, False, "Sorted '|'-joined binyanim of this root's lexemes (unknown binyan -> '?')."),
    ("source_primary", S, False, "Primary source asserting the root: 'hewiktionary' or 'synthetic_v1'."),
    ("source_ids", S, False, "'|'-joined source record identifiers supporting the root (see provenance)."),
    ("confidence", F, False, "Max confidence_root over this root's lexemes."),
    ("quality_tier", S, False, "Best quality tier among this root's lexemes."),
    ("quality_flags", S, True, "'|'-joined flags; null if none."),
]

LEXEMES = [
    ("lexeme_id", S, False, "Stable ID: 'LX-' + sha1(source + '|' + NFC vocalized lemma [+ '|k' for k-th distinct paradigm])[:10]."),
    ("language", S, False, "ISO 639-3 'heb' (synthetic: 'x-syn-heb')."),
    ("root_id", S, True, "FK to roots. NULL if the root is missing or ambiguous (never guessed)."),
    ("root_radicals_dotted", S, True, "Radicals as given by the source incl. shin/sin dots, '-'-joined (e.g. 'שׂ-ח-ק'); null if not given."),
    ("shin_sin_resolution", S, True, "How ש radicals got their dot: NA (no ש) | SOURCE (root template) | FORM (read off vocalized lemma) | MIXED | UNRESOLVED; null if no root."),
    ("root_homonym_index", S, True, "Wiktionary homonym index of the root (e.g. 'א','ב') if given."),
    ("binyan", S, True, "One of PAAL NIFAL PIEL PUAL HIFIL HUFAL HITPAEL; NULL if missing/ambiguous (never guessed or taken from the shape heuristic)."),
    ("binyan_raw", S, True, "Raw binyan label(s) from sources ('|'-joined)."),
    ("lemma_vocalized", S, False, "NFC vocalized lemma as given by the primary source (UniMorph heb_voc lemma column / Wiktionary headword)."),
    ("lemma_unvocalized", S, True, "Plene (ktiv male) unvocalized lemma: UniMorph 'heb' lemma of the paired paradigm, or Wiktionary 'כתיב מלא'."),
    ("infinitive_vocalized", S, True, "NFIN cell, vocalized."),
    ("infinitive_unvocalized", S, True, "NFIN cell, plene unvocalized."),
    ("citation_past_3ms_vocalized", S, True, "PST.3MSG cell, vocalized."),
    ("citation_past_3ms_unvocalized", S, True, "PST.3MSG cell, plene unvocalized."),
    ("gloss_en", S, True, "English translations from {{ת|אנגלית|..}} in the matched Wiktionary section(s); null if none. Section-level, not sense-disambiguated."),
    ("sense_id", S, True, "Wiktionary homonym markers of the matched section(s) when several sections share root x binyan (e.g. 'משני:א|משני:ב'); else null."),
    ("modern_hebrew", B, True, "True for UniMorph lexemes (source inventory = Modern Hebrew conjugation tables); True for Wiktionary-only lexemes only if a 'עברית חדשה' register tag is present; else null (unknown)."),
    ("attested", B, False, "Lexeme appears as an entry/paradigm in a lexicographic source (all real records True; synthetic False)."),
    ("has_unimorph_paradigm", B, False, "Lexeme has a UniMorph paradigm (else Wiktionary-only)."),
    ("source_primary", S, False, "'unimorph_heb' | 'hewiktionary' | 'synthetic_v1'."),
    ("source_ids", S, False, "'|'-joined source record identifiers (see provenance)."),
    ("match_status", S, False, "MATCHED | MATCHED_UNVOC_ONLY | ROOT_AMBIGUOUS | BINYAN_AMBIGUOUS | ROOT_AND_BINYAN_AMBIGUOUS | NO_MATCH | WIKTIONARY_ONLY | SYNTHETIC."),
    ("match_method", S, True, "'|'-joined methods of the evidence that agrees with the assigned root/binyan."),
    ("match_score", F, False, "Noisy-OR of agreeing evidence weights (0 if nothing assigned)."),
    ("match_evidence", S, True, "JSON list of all evidence records considered (incl. disagreeing ones)."),
    ("match_ambiguous", B, False, "True if >1 root or >1 binyan was supported by evidence."),
    ("binyan_shape_heuristic", S, True, "Binyan predicted from the vocalized paradigm shape (VALIDATION ONLY)."),
    ("binyan_shape_rule", S, True, "Rule that produced binyan_shape_heuristic."),
    ("voc_unvoc_pairing_status", S, True, "PAIRED | PAIRED_PARTIAL | PAIRED_TIE_IDENTICAL | AMBIGUOUS | UNPAIRED (UniMorph lexemes)."),
    ("voc_unvoc_pairing_score", F, True, "Fraction of cells whose vocalized and plene forms agree on letters modulo ו/י."),
    ("gizra_labels", S, True, "Traditional גזרה labels stated by the matched Wiktionary entries of this lexeme."),
    ("wiktionary_registers", S, True, "{{רובד}} register tags in matched Wiktionary sections."),
    ("n_cells", I, False, "Number of paradigm cells in forms table."),
    ("past_expected_cells", I, False, "Always 9 (canonical past paradigm)."),
    ("past_observed_cells", I, False, "Number of the 9 canonical past cells with a vocalized form."),
    ("past_completeness", F, False, "past_observed_cells / 9."),
    ("past_complete", B, False, "past_observed_cells == 9."),
    ("ud_attested", B, False, "Some plene form of this lexeme occurs as a verb form in UniMorph from_UD_unvoc (surface match only; may be a homograph)."),
    ("confidence_root", F, False, "Noisy-OR of agreeing root evidence weights; 0 if missing/ambiguous; halved if ROOT_FORM_MISMATCH."),
    ("confidence_binyan", F, False, "Noisy-OR of agreeing binyan evidence weights; 0 if missing/ambiguous; halved if shape heuristic disagrees."),
    ("confidence_lexeme", F, False, "min(confidence_root, confidence_binyan) x (1 if voc/unvoc PAIRED else 0.8)."),
    ("quality_tier", S, False, "GOLD | SILVER | QUESTIONABLE (see README)."),
    ("quality_flags", S, True, "'|'-joined flags; null if none."),
]

FORMS = [
    ("form_id", S, False, "'<lexeme_id>.<canonical_cell>' (unique: one row per lexeme x cell)."),
    ("lexeme_id", S, False, "FK to lexemes."),
    ("root_id", S, True, "Copy of lexemes.root_id (analysis convenience)."),
    ("form_raw", S, False, "Surface string exactly as in the primary source (vocalized source when available)."),
    ("form_nfc", S, False, "NFC(form_raw)."),
    ("form_vocalized", S, True, "NFC vocalized form (null if the only source is unvocalized)."),
    ("form_unvocalized", S, True, "NFC plene (ktiv male) form from the unvocalized source; null if unavailable. NOT derived by stripping."),
    ("form_stripped", S, True, "strip_niqqud(form_vocalized): defective (ktiv haser) spelling, deterministic."),
    ("form_vocalized_variants", S, True, "'|'-joined other vocalized forms given for the same lexeme x cell (e.g. duplicate source paradigms, Wiktionary variants)."),
    ("form_unvocalized_variants", S, True, "'|'-joined other plene spellings given for the same lexeme x cell."),
    ("pos", S, False, "'V' or 'V.MSDR' (verbal noun)."),
    ("finiteness", S, False, "FIN | NFIN."),
    ("tense", S, False, "PST | PRS | FUT | NA."),
    ("aspect", S, False, "Always NA (not morphologically marked / not annotated)."),
    ("mood", S, False, "IND | IMP | NA."),
    ("person", S, False, "1 | 2 | 3 | NA (present participles and non-finite forms: NA)."),
    ("number", S, False, "SG | PL | NA."),
    ("gender", S, False, "MASC | FEM | NA (1st person and PST.3PL: NA -- not expressed)."),
    ("voice", S, False, "Always NA: voice is not annotated by the sources; the binyan column carries voice-related information."),
    ("binyan", S, True, "Copy of lexemes.binyan."),
    ("canonical_cell", S, False, "e.g. PST.1SG PST.2MSG PST.3PL PRS.FSG FUT.3MPL IMP.2FSG NFIN MSDR."),
    ("is_past_cell", B, False, "canonical_cell in the 9 canonical past cells."),
    ("unimorph_tag_raw", S, True, "Original UniMorph tag (null for Wiktionary-only forms)."),
    ("is_syncretic", B, False, "Another cell of the same lexeme has an identical VOCALIZED form."),
    ("syncretism_group", S, True, "'<lexeme_id>:V<k>' shared by cells with identical vocalized forms; null if unique."),
    ("is_syncretic_unvocalized", B, False, "Another cell of the same lexeme has an identical PLENE UNVOCALIZED form."),
    ("syncretism_group_unvocalized", S, True, "'<lexeme_id>:U<k>' shared by cells with identical plene forms; null if unique."),
    ("plene_consistent", B, True, "Vocalized and plene forms agree on letters modulo ו/י (null if either missing)."),
    ("wiktionary_corroborated", B, True, "True: a Wiktionary table gives the same vocalized form in this cell; False: it gives a different one (conflict); null: no Wiktionary table value."),
    ("ud_attested", B, False, "form_unvocalized occurs among UD-derived verb forms (surface match only)."),
    ("is_attested", B, False, "Form is taken from a source (not generated)."),
    ("is_generated", B, False, "Form was generated by a program (synthetic data only)."),
    ("source_primary", S, False, "'unimorph_heb' | 'hewiktionary' | 'synthetic_v1'."),
    ("source_ids", S, False, "'|'-joined source record identifiers."),
    ("confidence", F, False, "confidence_lexeme x form-level factor (1; 0.5 if a form-level check failed)."),
    ("quality_tier", S, False, "GOLD | SILVER | QUESTIONABLE; never better than the lexeme's tier."),
    ("quality_flags", S, True, "'|'-joined flags; null if none."),
]

PROVENANCE = [
    ("provenance_id", S, False, "Unique row ID."),
    ("canonical_record_id", S, False, "root_id / lexeme_id / form_id."),
    ("record_type", S, False, "root | lexeme | form."),
    ("field_supported", S, False, "What this source record supports (e.g. form_vocalized, form_unvocalized, root, binyan, paradigm)."),
    ("source_name", S, False, "unimorph_heb | hewiktionary | synthetic_v1."),
    ("source_url", S, False, "URL (UniMorph: blob URL at pinned commit with #L<line>; Wiktionary: page at pinned revision)."),
    ("source_file", S, False, "File name within the source (UniMorph file / dump file name)."),
    ("source_version_or_commit", S, False, "Git commit or dump date."),
    ("source_record_identifier", S, False, "e.g. 'heb_voc:1234' or 'wikt:page=<id>:rev=<id>:sec=<offset>:tpl=<k>'."),
    ("source_line_if_available", I, True, "1-based line number in source file."),
    ("retrieval_date", S, False, "Date the raw source was retrieved (ISO)."),
    ("parser_version", S, False, "Version of the parser that read this record."),
    ("raw_source_value", S, False, "The raw source text supporting the field (line / template / cell value)."),
]

VALIDATION = [
    ("check_id", S, False, "Unique row ID."),
    ("record_id", S, False, "Affected record (or '*' for table-level checks)."),
    ("record_type", S, False, "root | lexeme | form | table | source."),
    ("check_name", S, False, "Name of the check (see README)."),
    ("severity", S, False, "ERROR | WARNING | INFO."),
    ("details", S, True, "Human-readable details / values involved."),
]

TABLES = {"roots": ROOTS, "lexemes": LEXEMES, "forms": FORMS, "provenance": PROVENANCE,
          "validation": VALIDATION}
PRIMARY_KEYS = {"roots": "root_id", "lexemes": "lexeme_id", "forms": "form_id",
                "provenance": "provenance_id", "validation": "check_id"}
ENUMS = {
    ("lexemes", "binyan"): set(BINYANIM), ("forms", "binyan"): set(BINYANIM),
    ("roots", "quality_tier"): set(TIERS), ("lexemes", "quality_tier"): set(TIERS),
    ("forms", "quality_tier"): set(TIERS),
    ("forms", "tense"): {"PST", "PRS", "FUT", "NA"}, ("forms", "mood"): {"IND", "IMP", "NA"},
    ("forms", "person"): {"1", "2", "3", "NA"}, ("forms", "number"): {"SG", "PL", "NA"},
    ("forms", "gender"): {"MASC", "FEM", "NA"}, ("forms", "aspect"): {"NA"}, ("forms", "voice"): {"NA"},
    ("provenance", "record_type"): {"root", "lexeme", "form"},
    ("validation", "severity"): {"ERROR", "WARNING", "INFO"},
}


class SchemaError(Exception):
    pass


def enforce(df: pd.DataFrame, table: str, relax_enums: frozenset = frozenset()) -> pd.DataFrame:
    """Cast to schema dtypes, order columns, and validate.  Raises SchemaError.
    relax_enums: column names whose enum check is skipped (synthetic data with
    user-defined template names in 'binyan')."""
    spec = TABLES[table]
    cols = [c for c, *_ in spec]
    missing = [c for c in cols if c not in df.columns]
    extra = [c for c in df.columns if c not in cols]
    if missing or extra:
        raise SchemaError(f"{table}: missing={missing} extra={extra}")
    df = df[cols].copy()
    for c, dt, nullable, _ in spec:
        try:
            df[c] = df[c].astype(dt)
        except Exception as e:  # pragma: no cover
            raise SchemaError(f"{table}.{c}: cannot cast to {dt}: {e}")
        if not nullable and df[c].isna().any():
            bad = df.loc[df[c].isna(), cols[0]].head(5).tolist()
            raise SchemaError(f"{table}.{c}: {int(df[c].isna().sum())} nulls in non-nullable column, e.g. {bad}")
    pk = PRIMARY_KEYS[table]
    if df[pk].duplicated().any():
        raise SchemaError(f"{table}: duplicate primary keys {df.loc[df[pk].duplicated(), pk].head(5).tolist()}")
    for (t, c), allowed in ENUMS.items():
        if t == table and c not in relax_enums:
            vals = set(df[c].dropna().unique())
            if not vals <= allowed:
                raise SchemaError(f"{table}.{c}: values outside enum: {vals - allowed}")
    for c, dt, _, _ in spec:
        # raw columns preserve the source bytes exactly (possibly non-NFC) by design
        if dt == S and not (c.endswith("_raw") or c == "raw_source_value"):
            s = df[c].dropna()
            import unicodedata
            bad = s[s.map(lambda x: unicodedata.normalize("NFC", x) != x)]
            if len(bad):
                raise SchemaError(f"{table}.{c}: {len(bad)} non-NFC strings, e.g. {bad.head(3).tolist()}")
    return df


def check_foreign_keys(roots, lexemes, forms, provenance):
    errs = []
    if not set(lexemes.root_id.dropna()) <= set(roots.root_id):
        errs.append("lexemes.root_id not in roots")
    if not set(forms.lexeme_id) <= set(lexemes.lexeme_id):
        errs.append("forms.lexeme_id not in lexemes")
    lr = dict(zip(lexemes.lexeme_id, lexemes.root_id))
    if any(lr[l] is not pd.NA and lr[l] != r and not (pd.isna(lr[l]) and pd.isna(r))
           for l, r in zip(forms.lexeme_id, forms.root_id)):
        errs.append("forms.root_id inconsistent with lexemes.root_id")
    ids = set(roots.root_id) | set(lexemes.lexeme_id) | set(forms.form_id)
    covered = set(provenance.canonical_record_id)
    missing = ids - covered
    if missing:
        errs.append(f"{len(missing)} canonical records without provenance, e.g. {sorted(missing)[:5]}")
    orphan = covered - ids
    if orphan:
        errs.append(f"{len(orphan)} provenance rows point to unknown records, e.g. {sorted(orphan)[:5]}")
    if errs:
        raise SchemaError("; ".join(errs))


def schema_json() -> str:
    return json.dumps({
        "schema_version": SCHEMA_VERSION,
        "tables": {t: {"primary_key": PRIMARY_KEYS[t],
                       "columns": [dict(name=c, dtype=d, nullable=n, description=desc) for c, d, n, desc in spec]}
                   for t, spec in TABLES.items()},
        "enums": {f"{t}.{c}": sorted(v) for (t, c), v in ENUMS.items()},
    }, ensure_ascii=False, indent=2)
