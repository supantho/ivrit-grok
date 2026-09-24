"""RAW -> INTERIM.  Parses sources into flat, provenance-carrying tables.
Nothing here resolves conflicts; it only records what each source says."""
from __future__ import annotations

import json
import logging

import pandas as pd

from . import PARSER_VERSIONS, paths
from .hebrew import audit_characters, mark_inventory, normalize_nfc
from .unimorph import (args_gap_cells, args_tag_to_v3, is_verb_tag, pair_voc_unvoc,
                       read_triples, segment_paradigms)
from .morphology import TagError, parse_unimorph_tag
from .wiktionary import extract_conj_tables, extract_full_tables, extract_verb_entries, iter_dump

log = logging.getLogger(__name__)


def ingest_unimorph() -> dict:
    out = paths.INTERIM / "unimorph"
    out.mkdir(parents=True, exist_ok=True)
    d = paths.UNIMORPH_DIR
    voc_lines = read_triples(d / "heb_voc", "heb_voc")
    unv_lines = read_triples(d / "heb", "heb")
    voc = segment_paradigms(voc_lines)
    unv = segment_paradigms(unv_lines)

    # ---- line-level table (verbs only), with character audit ----
    rows = []
    for plist in (voc, unv):
        for pi, p in enumerate(plist):
            for ln in p.lines:
                try:
                    f = parse_unimorph_tag(ln.tag_raw)
                    cell, terr = f["canonical_cell"], None
                except TagError as e:
                    cell, terr = None, str(e)
                a = audit_characters(ln.form_raw)
                al = audit_characters(ln.lemma_raw)
                rows.append(dict(
                    file=ln.file, line_no=ln.line_no, paradigm_key=p.paradigm_key,
                    paradigm_index=pi, lemma_raw=ln.lemma_raw, lemma_nfc=ln.lemma,
                    form_raw=ln.form_raw, form_nfc=ln.form, tag_raw=ln.tag_raw,
                    canonical_cell=cell, tag_error=terr,
                    char_flags="|".join(sorted(set(a.flags))) or None,
                    lemma_char_flags="|".join(sorted(set(al.flags))) or None,
                    nonstandard_marks="|".join(a.nonstandard_marks) or None,
                    non_hebrew_chars="|".join(a.non_hebrew_chars) or None,
                ))
    lines_df = pd.DataFrame(rows)
    lines_df.to_parquet(out / "verb_lines.parquet", index=False)

    # ---- paradigm table ----
    prow = []
    for plist in (voc, unv):
        for pi, p in enumerate(plist):
            prow.append(dict(paradigm_key=p.paradigm_key, file=p.file, paradigm_index=pi,
                             lemma_raw=p.lemma_raw, lemma_nfc=p.lemma, n_lines=len(p.lines),
                             n_cells=len(p.cells),
                             duplicate_cells="|".join(c for c, _ in p.duplicate_cells) or None,
                             duplicate_cell_lines="|".join(str(l.line_no) for _, l in p.duplicate_cells) or None,
                             tag_errors=len(p.tag_errors),
                             signature=json.dumps(p.signature(), ensure_ascii=False)))
    par_df = pd.DataFrame(prow)
    par_df.to_parquet(out / "paradigms.parquet", index=False)

    # ---- voc <-> unvoc pairing ----
    pairing = pair_voc_unvoc(voc, unv)
    pr = []
    for r in pairing:
        vp = voc[r["voc_index"]]
        up = unv[r["unvoc_index"]] if r["unvoc_index"] is not None else None
        pr.append(dict(voc_paradigm_key=vp.paradigm_key, voc_lemma=vp.lemma,
                       unvoc_paradigm_key=up.paradigm_key if up else None,
                       unvoc_lemma=up.lemma if up else None, score=r["score"],
                       runner_up_score=r["runner_up_score"], status=r["status"],
                       tied_unvoc_keys="|".join(unv[j].paradigm_key for j in r["tied"]) or None,
                       edit_distance=r["edit_distance"]))
    pair_df = pd.DataFrame(pr)
    pair_df.to_parquet(out / "voc_unvoc_pairing.parquet", index=False)

    # ---- .args cross-check and explicit gaps ----
    args_lines = read_triples(d / "heb.args", "heb.args")
    gaps = args_gap_cells(args_lines)
    gap_rows = [dict(lemma_nfc=l, tag4=t, cell=args_tag_to_v3(t)) for l, ts in gaps.items() for t in sorted(ts)]
    pd.DataFrame(gap_rows, columns=["lemma_nfc", "tag4", "cell"]).to_parquet(out / "args_explicit_gaps.parquet", index=False)
    voc_set = {(normalize_nfc(l.lemma_raw), normalize_nfc(l.form_raw), parse_unimorph_tag(l.tag_raw)["canonical_cell"])
               for l in voc_lines if is_verb_tag(l.tag_raw)}
    args_set = set()
    unconvertible = 0
    for l in args_lines:
        if not l.tag_raw.startswith("V") or l.form_raw.strip() in {"-", "–", "—"}:
            continue
        c = args_tag_to_v3(l.tag_raw)
        if c is None:
            unconvertible += 1
            continue
        args_set.add((l.lemma, l.form, c))
    args_check = dict(voc_triples=len(voc_set), args_triples=len(args_set),
                      in_voc_not_args=len(voc_set - args_set), in_args_not_voc=len(args_set - voc_set),
                      args_unconvertible_tags=unconvertible, args_explicit_gap_rows=len(gap_rows))

    # ---- UD-derived unvocalized verb forms (attestation signal only) ----
    ud = read_triples(d / "from_UD_unvoc", "from_UD_unvoc")
    ud_rows = [dict(line_no=l.line_no, lemma_nfc=l.lemma, form_nfc=l.form, tag_raw=l.tag_raw)
               for l in ud if l.tag_raw.startswith("V")]
    pd.DataFrame(ud_rows).to_parquet(out / "ud_verb_forms.parquet", index=False)

    marks = mark_inventory(l.form_raw for l in voc_lines if is_verb_tag(l.tag_raw))
    summary = dict(
        parser_version=PARSER_VERSIONS["unimorph_parser"],
        voc_verb_lines=sum(len(p.lines) for p in voc), unvoc_verb_lines=sum(len(p.lines) for p in unv),
        voc_paradigms=len(voc), unvoc_paradigms=len(unv),
        voc_distinct_lemmas=len({p.lemma for p in voc}), unvoc_distinct_lemmas=len({p.lemma for p in unv}),
        pairing_status=pair_df.status.value_counts().to_dict(),
        args_check=args_check, ud_verb_lines=len(ud_rows),
        combining_mark_inventory=dict(marks.most_common()),
        lines_with_char_flags=lines_df.char_flags.value_counts().to_dict(),
    )
    (out / "ingest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def ingest_wiktionary() -> dict:
    out = paths.INTERIM / "wiktionary"
    out.mkdir(parents=True, exist_ok=True)
    entries, tables, full = [], [], []
    relevant = []
    n_pages = 0
    for page in iter_dump(str(paths.WIKTIONARY_DUMP)):
        n_pages += 1
        if page.ns == 10:  # templates (documentation examples) are not lexical data
            continue
        hit = False
        if page.ns == 0 and "ניתוח דקדוקי לפועל" in page.text:
            e = extract_verb_entries(page)
            entries += e
            hit = hit or bool(e)
        if "נטיות פעלים" in page.text:
            t = extract_conj_tables(page)
            tables += t
            hit = hit or bool(t)
        if "נטיות פעל בבנין" in page.text:
            f = extract_full_tables(page)
            full += f
            hit = hit or bool(f)
        if hit:
            relevant.append(dict(title=page.title, ns=page.ns, page_id=page.page_id,
                                 rev_id=page.rev_id, timestamp=page.timestamp, text=page.text))
    # Cache of the raw wikitext of every page we used (verbatim from the dump).
    pd.DataFrame(relevant).to_parquet(out / "relevant_pages_wikitext.parquet", index=False)
    e_df = pd.DataFrame(entries)
    t_df = pd.DataFrame(tables)
    f_df = pd.DataFrame(full)
    e_df.to_parquet(out / "verb_entries.parquet", index=False)
    t_df.to_parquet(out / "conj_tables.parquet", index=False)
    f_df.to_parquet(out / "full_tables.parquet", index=False)
    summary = dict(parser_version=PARSER_VERSIONS["wiktionary_parser"], pages_scanned=n_pages,
                   relevant_pages=len(relevant), verb_entry_templates=len(e_df),
                   verb_entries_with_root=int(e_df.root_radicals.notna().sum()),
                   conj_table_cells=len(t_df), conj_table_root_pages=int(t_df.page_id.nunique()) if len(t_df) else 0,
                   full_table_cells=len(f_df))
    (out / "ingest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary
