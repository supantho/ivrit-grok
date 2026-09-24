"""Synthetic Hebrew-like verbal morphology with 100% known ground truth.

Kept completely separate from real data (own directory, 'SYN-' IDs,
source_primary='synthetic_v1', language='x-syn-heb').  Produces canonical
tables with EXACTLY the real-data schema, so the same derived-view and split
code runs on both.

Idealizations (documented; intentional): only strong triconsonantal roots
from a configurable inventory; no phonological processes (no begadkefat
spirantization / dagesh lene, no guttural effects, no assimilation or
metathesis, no weak-root behaviour); final letter forms applied at word end.
The unvocalized form is the defective spelling (niqqud stripped).
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import PARSER_VERSIONS
from .canonical_utils import add_past_completeness, add_syncretism, root_rows_from_lexemes
from .hebrew import apply_final_forms, normalize_nfc, strip_niqqud
from .morphology import PAST_CELLS, parse_unimorph_tag

# Placeholders 1, 2, 3 = radicals.  Strong-verb past patterns of the 7 binyanim.
_SFX = {"PST.1SG": "ְתִּי", "PST.2MSG": "ְתָּ", "PST.2FSG": "ְתְּ", "PST.1PL": "ְנוּ",
        "PST.2MPL": "ְתֶּם", "PST.2FPL": "ְתֶּן"}


def _tpl(pre: str, v1: str, gem: str, v2_3ms: str, v2_cons: str, v2_vow: str, pre_pl2: str | None = None,
         v1_pl2: str | None = None) -> dict:
    """Build a 9-cell past template from stem pieces.
    pre   : prefix before C1 ('', 'נִ', 'הִ', 'הִתְ', 'הֻ')
    v1    : vowel on C1
    gem   : '' or dagesh on C2
    v2_3ms: vowel on C2 in 3MSG ; v2_cons: vowel on C2 before consonant-initial suffixes
    v2_vow: vowel on C2 before vowel-initial suffixes (3FSG/3PL)"""
    t = {"PST.3MSG": f"{pre}1{v1}2{gem}{v2_3ms}3",
         "PST.3FSG": f"{pre}1{v1}2{gem}{v2_vow}3ָה",
         "PST.3PL": f"{pre}1{v1}2{gem}{v2_vow}3וּ"}
    for c, sfx in _SFX.items():
        p, v = pre, v1
        if c in ("PST.2MPL", "PST.2FPL") and pre_pl2 is not None:
            p, v = pre_pl2, v1_pl2
        t[c] = f"{p}1{v}2{gem}{v2_cons}3{sfx}"
    return t


DEFAULT_TEMPLATES = {
    "PAAL": _tpl("", "ָ", "", "ַ", "ַ", "ְ", pre_pl2="", v1_pl2="ְ"),
    "NIFAL": _tpl("נִ", "ְ", "", "ַ", "ַ", "ְ"),
    "PIEL": _tpl("", "ִ", "ּ", "ֵ", "ַ", "ְ"),
    "PUAL": _tpl("", "ֻ", "ּ", "ַ", "ַ", "ְ"),
    "HIFIL": {**_tpl("הִ", "ְ", "", "ִי", "ַ", "ִי"), },
    "HUFAL": _tpl("הֻ", "ְ", "", "ַ", "ַ", "ְ"),
    "HITPAEL": _tpl("הִתְ", "ַ", "ּ", "ֵ", "ַ", "ְ"),
}
# Hif'il 3FSG/3PL keep the long hiriq-yod: הִכְתִּיבָה / הִכְתִּיבוּ (handled by v2_vow='ִי').

DEFAULT_CONSONANTS = ["ב", "ג", "ד", "ז", "ט", "כ", "ל", "מ", "נ", "ס", "פ", "צ", "ק", "ר", "שׁ", "ת"]


@dataclass
class SyntheticConfig:
    version: str = "v1"
    seed: int = 12345
    n_roots: int = 1000
    consonants: list = field(default_factory=lambda: list(DEFAULT_CONSONANTS))
    allow_repeated_radicals: bool = False
    templates: dict = field(default_factory=lambda: dict(DEFAULT_TEMPLATES))
    template_names: list | None = None      # subset of templates to use (None = all)
    lexeme_fraction: float = 1.0            # <1 -> random root x template gaps
    apply_final_forms: bool = True

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)


def realize(pattern: str, radicals: list[str], final_forms: bool = True) -> str:
    out = pattern
    for i, r in enumerate(radicals, start=1):
        out = out.replace(str(i), r)
    out = normalize_nfc(out)
    return apply_final_forms(out) if final_forms else out


def _sid(prefix, s):
    return prefix + hashlib.sha1(s.encode()).hexdigest()[:10]


def generate(cfg: SyntheticConfig):
    rng = np.random.default_rng(cfg.seed)
    cons = cfg.consonants
    if cfg.allow_repeated_radicals:
        space = list(itertools.product(cons, repeat=3))
    else:
        space = list(itertools.permutations(cons, 3))
    if cfg.n_roots > len(space):
        raise ValueError(f"n_roots={cfg.n_roots} > {len(space)} possible roots")
    pick = sorted(rng.choice(len(space), size=cfg.n_roots, replace=False))
    roots_list = [list(space[i]) for i in pick]
    tnames = cfg.template_names or list(cfg.templates)
    for t in tnames:
        missing = set(PAST_CELLS) - set(cfg.templates[t])
        if missing:
            raise ValueError(f"template {t} lacks cells {missing}")
    gen_src = f"generated://hebmorph/synthetic_{cfg.version}?seed={cfg.seed}"
    lex_rows, form_rows, prov = [], [], []
    root_info = {}
    for rads in roots_list:
        rn = "".join(rads)
        rid = _sid("SYN-RT-", f"{cfg.version}|{rn}")
        root_info[rn] = dict(root_id=rid, radicals=rads, root_raw="-".join(rads), traditional_class=None,
                             traditional_class_raw=None, source_primary=f"synthetic_{cfg.version}",
                             source_ids=f"synthetic_{cfg.version}:root:{rn}", quality_flags="SYNTHETIC",
                             group_prefix="SYN-RG-")
        prov.append(_prov(rid, "root", "root", gen_src, cfg, f"root:{rn}", "-".join(rads)))
        for t in tnames:
            if cfg.lexeme_fraction < 1.0 and rng.random() > cfg.lexeme_fraction:
                continue
            lid = _sid("SYN-LX-", f"{cfg.version}|{rn}|{t}")
            tpl = cfg.templates[t]
            forms = {c: realize(tpl[c], rads, cfg.apply_final_forms) for c in PAST_CELLS}
            for c in PAST_CELLS:
                feats = parse_unimorph_tag(_cell_to_tag(c))
                fid = f"{lid}.{c}"
                fv = forms[c]
                form_rows.append(dict(
                    form_id=fid, lexeme_id=lid, root_id=rid, form_raw=fv, form_nfc=fv, form_vocalized=fv,
                    form_unvocalized=strip_niqqud(fv), form_stripped=strip_niqqud(fv),
                    form_vocalized_variants=None, form_unvocalized_variants=None,
                    pos=feats["pos"], finiteness=feats["finiteness"], tense=feats["tense"], aspect="NA",
                    mood=feats["mood"], person=feats["person"], number=feats["number"], gender=feats["gender"],
                    voice="NA", binyan=t, canonical_cell=c, is_past_cell=True, unimorph_tag_raw=None,
                    plene_consistent=True, wiktionary_corroborated=None, ud_attested=False,
                    is_attested=False, is_generated=True, source_primary=f"synthetic_{cfg.version}",
                    source_ids=f"synthetic_{cfg.version}:template:{t}:{c}", confidence=1.0,
                    quality_tier="GOLD", quality_flags="SYNTHETIC"))
                prov.append(_prov(fid, "form", "form_vocalized", gen_src, cfg, f"template:{t}:{c}", tpl[c]))
            lex_rows.append(dict(
                lexeme_id=lid, language="x-syn-heb", root_id=rid, root_radicals_dotted="-".join(rads),
                shin_sin_resolution="SOURCE" if any(r.startswith("ש") for r in rads) else "NA",
                root_homonym_index=None, binyan=t, binyan_raw=f"synthetic_template:{t}",
                lemma_vocalized=forms["PST.3MSG"], lemma_unvocalized=strip_niqqud(forms["PST.3MSG"]),
                infinitive_vocalized=None, infinitive_unvocalized=None,
                citation_past_3ms_vocalized=forms["PST.3MSG"],
                citation_past_3ms_unvocalized=strip_niqqud(forms["PST.3MSG"]),
                gloss_en=None, sense_id=None, modern_hebrew=False, attested=False, has_unimorph_paradigm=False,
                source_primary=f"synthetic_{cfg.version}", source_ids=f"synthetic_{cfg.version}:lexeme:{rn}:{t}",
                match_status="SYNTHETIC", match_method=None, match_score=1.0, match_evidence=None,
                match_ambiguous=False, binyan_shape_heuristic=None, binyan_shape_rule=None,
                voc_unvoc_pairing_status=None, voc_unvoc_pairing_score=None, gizra_labels=None,
                wiktionary_registers=None, ud_attested=False, confidence_root=1.0, confidence_binyan=1.0,
                confidence_lexeme=1.0, quality_tier="GOLD", quality_flags="SYNTHETIC"))
            prov.append(_prov(lid, "lexeme", "root+template", gen_src, cfg, f"lexeme:{rn}:{t}", json.dumps(tpl, ensure_ascii=False)))
    lexemes = pd.DataFrame(lex_rows)
    forms = add_syncretism(pd.DataFrame(form_rows))
    lexemes = add_past_completeness(lexemes, forms)
    roots = root_rows_from_lexemes(root_info, lexemes)
    roots["source_primary"] = f"synthetic_{cfg.version}"
    provenance = pd.DataFrame(prov)
    provenance.insert(0, "provenance_id", [f"SYN-PV{i:07d}" for i in range(len(provenance))])
    validation = pd.DataFrame(columns=["check_id", "record_id", "record_type", "check_name", "severity", "details"])
    return dict(roots=roots, lexemes=lexemes, forms=forms, provenance=provenance, validation=validation)


def _cell_to_tag(c: str) -> str:
    import re
    m = re.fullmatch(r"PST\.([123])([MF]?)(SG|PL)", c)
    g = {"M": ";MASC", "F": ";FEM", "": ""}[m.group(2)]
    return f"V;{m.group(1)};{m.group(3)};PST{g}"


def _prov(rec, rtype, field_, url, cfg, ident, raw):
    return dict(canonical_record_id=rec, record_type=rtype, field_supported=field_,
                source_name=f"synthetic_{cfg.version}", source_url=url, source_file="synthetic_config.json",
                source_version_or_commit=PARSER_VERSIONS["synthetic_generator"],
                source_record_identifier=ident, source_line_if_available=None,
                retrieval_date="generated (deterministic from config + seed)",
                parser_version=PARSER_VERSIONS["synthetic_generator"], raw_source_value=raw)
