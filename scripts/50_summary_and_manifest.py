"""corpus_summary.md + dataset_manifest.json (checksums, versions, seeds, counts).

Refuses to overwrite a manifest whose "frozen" flag is true unless the
checksums are identical: released versions are never silently mutated.
To change a frozen release, bump DATASET_VERSION in hebmorph/__init__.py."""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from hebmorph import DATASET_NAME, DATASET_VERSION, PARSER_VERSIONS, SCHEMA_VERSION, paths  # noqa: E402
from hebmorph.morphology import BINYANIM  # noqa: E402

R = paths.REPORTS


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def dataset_checksums() -> dict:
    C = paths.CANONICAL
    files = sorted([*C.glob("*.parquet"), *C.glob("*.tsv"),
                    *(paths.DERIVED / "past_reinflection").glob("*"), *(paths.DERIVED / "full_paradigm").glob("*"),
                    *(paths.DERIVED / "synthetic" / "v1" / "canonical").glob("*")])
    return {str(p.relative_to(paths.DATA)): sha256(p) for p in files}


def main(freeze: bool = False):
    """freeze=True: do not regenerate summaries; write the manifest with frozen=true,
    pinned to the current HEAD commit (which must contain the dataset, with a clean
    dataset worktree)."""
    ms = paths.METADATA / "dataset_manifest.json"
    if ms.exists():
        old = json.loads(ms.read_text())
        if old.get("frozen") and old.get("dataset_version") == f"{DATASET_NAME}_{DATASET_VERSION}":
            if old.get("checksums") != dataset_checksums():
                sys.exit(f"REFUSING: {DATASET_VERSION} is frozen and the data changed. Bump DATASET_VERSION.")
            print(f"{DATASET_VERSION} is frozen and the data are byte-identical; nothing regenerated.")
            return
    C = paths.CANONICAL
    roots, lex, forms = (pd.read_parquet(C / f"{t}.parquet") for t in ["roots", "lexemes", "forms"])
    conflicts = pd.read_csv(R / "conflicts.tsv", sep="\t")
    dups = pd.read_csv(R / "duplicates.tsv", sep="\t")
    pr = pd.read_parquet(paths.DERIVED / "past_reinflection" / "past_reinflection.parquet")
    splits = pd.read_csv(R / "split_summary.tsv", sep="\t")
    um = lex[lex.source_primary == "unimorph_heb"]
    past = forms[forms.is_past_cell]
    L = []
    w = L.append
    w(f"# Corpus summary — {DATASET_NAME}_{DATASET_VERSION}\n")
    w(f"_Generated {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by scripts/50_summary_and_manifest.py. "
      "All numbers are computed from data/canonical and data/derived._\n")
    w("## Headline counts\n")
    w("| quantity | value |\n|---|---|")
    rows = [
        ("roots", len(roots)),
        ("triliteral roots", int((roots.root_length == 3).sum())),
        ("quadriliteral roots", int((roots.root_length == 4).sum())),
        ("roots with 5+ radicals", int((roots.root_length >= 5).sum())),
        ("lexemes (root × binyan records)", len(lex)),
        ("  of which with a UniMorph paradigm", len(um)),
        ("  of which Wiktionary-only (citation forms, few cells)", int((lex.source_primary == "hewiktionary").sum())),
        ("forms (lexeme × cell rows)", len(forms)),
        ("past-tense forms", len(past)),
        ("lexemes with complete 9-cell past paradigm", int(lex.past_complete.sum())),
        ("  UniMorph lexemes with complete past paradigm", int(um.past_complete.sum())),
        ("average past completeness (all lexemes)", f"{lex.past_completeness.mean():.3f}"),
        ("average past completeness (UniMorph lexemes)", f"{um.past_completeness.mean():.3f}"),
        ("forms with vocalized string", int(forms.form_vocalized.notna().sum())),
        ("forms with plene unvocalized string", int(forms.form_unvocalized.notna().sum())),
        ("syncretic cells (vocalized)", f"{int(forms.is_syncretic.sum())} ({pct(forms.is_syncretic.sum(), forms.form_vocalized.notna().sum())})"),
        ("syncretic cells (unvocalized)", f"{int(forms.is_syncretic_unvocalized.sum())} ({pct(forms.is_syncretic_unvocalized.sum(), forms.form_unvocalized.notna().sum())})"),
        ("syncretic PAST cells (unvocalized)", f"{int(past.is_syncretic_unvocalized.sum())} ({pct(past.is_syncretic_unvocalized.sum(), past.form_unvocalized.notna().sum())})"),
        ("source conflicts (rows in conflicts.tsv)", len(conflicts)),
        ("ambiguous matches (lexemes with >1 root or binyan supported)", int(lex.match_ambiguous.sum())),
        ("UniMorph lexemes with no Wiktionary match (no root/binyan)", int((um.match_status == "NO_MATCH").sum())),
        ("QUESTIONABLE lexemes", int((lex.quality_tier == "QUESTIONABLE").sum())),
        ("QUESTIONABLE forms", int((forms.quality_tier == "QUESTIONABLE").sum())),
        ("past-reinflection examples (both conditions)", len(pr)),
        ("usable past-reinflection examples (eligible_default), vocalized", int((pr.analysis__eligible_default & (pr.condition == "voc")).sum())),
        ("usable past-reinflection examples (eligible_default), unvocalized", int((pr.analysis__eligible_default & (pr.condition == "unv")).sum())),
    ]
    for k, v in rows:
        w(f"| {k} | {v} |")
    w("\n## Quality tiers\n")
    w("| tier | roots | lexemes | UniMorph lexemes | forms |\n|---|---|---|---|---|")
    for t in ["GOLD", "SILVER", "QUESTIONABLE"]:
        w(f"| {t} | {(roots.quality_tier == t).sum()} | {(lex.quality_tier == t).sum()} | {(um.quality_tier == t).sum()} | {(forms.quality_tier == t).sum()} |")
    w("\n## By binyan\n")
    w("| binyan | lexemes | UniMorph lexemes | complete past | GOLD | SILVER | QUESTIONABLE | forms |\n|---|---|---|---|---|---|---|---|")
    for b in BINYANIM + [None]:
        s = lex[lex.binyan.isna()] if b is None else lex[lex.binyan == b]
        f = forms[forms.binyan.isna()] if b is None else forms[forms.binyan == b]
        w(f"| {b or 'UNKNOWN'} | {len(s)} | {(s.source_primary == 'unimorph_heb').sum()} | {s.past_complete.sum()} | "
          f"{(s.quality_tier == 'GOLD').sum()} | {(s.quality_tier == 'SILVER').sum()} | {(s.quality_tier == 'QUESTIONABLE').sum()} | {len(f)} |")
    w("\n## By root structural class (computed from radicals; not a traditional label)\n")
    v = lex.merge(roots[["root_id", "structural_class"]], on="root_id", how="left")
    w("| structural_class | roots | lexemes | UniMorph lexemes | mean past completeness |\n|---|---|---|---|---|")
    for c, n in roots.structural_class.value_counts().items():
        s = v[v.structural_class == c]
        w(f"| {c} | {n} | {len(s)} | {(s.source_primary == 'unimorph_heb').sum()} | {s.past_completeness.mean():.3f} |")
    w("\n## Traditional גזרה (only where Wiktionary states it)\n")
    w("| traditional_class | roots |\n|---|---|")
    for c, n in roots.traditional_class.fillna("NOT_STATED").value_counts().head(20).items():
        w(f"| {c} | {n} |")
    w("\n## Conflicts and duplicates\n")
    w("| type | n |\n|---|---|")
    for c, n in conflicts.conflict_type.value_counts().items():
        w(f"| conflict: {c} | {n} |")
    for c, n in dups.duplicate_type.value_counts().items():
        w(f"| duplicate: {c} | {n} |")
    w("\n## Splits (real data, seed 0; counts per partition)\n")
    w("Levels: pair_ids = examples per condition before per-condition filtering; forms = distinct target forms; "
      "root×binyan = distinct lexical pairs.\n")
    w("| view | split | partition | examples voc | examples unv | forms | lexemes | roots | root×binyan |\n|---|---|---|---|---|---|---|---|---|")
    for r in splits[splits.seed == 0].itertuples(index=False):
        w(f"| {r.view} | {r.split} | {r.partition} | {r.examples_voc} | {r.examples_unv} | {r.forms} | {r.lexemes} | {r.roots} | {r.root_binyan_pairs} |")
    w(f"\nAll {splits.split.nunique()} split manifests (all seeds) are listed in reports/split_summary.tsv; "
      "every manifest was verified by hebmorph.splits.verify at generation time.\n")
    syn = paths.DERIVED / "synthetic" / "v1"
    if (syn / "canonical" / "forms.parquet").exists():
        sl = pd.read_parquet(syn / "canonical" / "lexemes.parquet")
        sf = pd.read_parquet(syn / "canonical" / "forms.parquet")
        w("## Synthetic v1 (separate; never mixed with real data)\n")
        w(f"{sl.root_id.nunique()} roots × {sl.binyan.nunique()} templates = {len(sl)} lexemes, {len(sf)} past forms "
          f"(config: derived/synthetic/v1/synthetic_config.json).\n")
    if not freeze:
        (R / "corpus_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---------------- schema documentation (generated from hebmorph/schema.py) ----------------
    from hebmorph.schema import TABLES, PRIMARY_KEYS
    S = ["# Schema (generated from hebmorph/schema.py — do not edit by hand)\n",
         f"schema_version {SCHEMA_VERSION}. Machine-readable: data/metadata/schema.json.\n"]
    for t, spec in TABLES.items():
        S.append(f"## {t}  (primary key: `{PRIMARY_KEYS[t]}`)\n")
        S.append("| column | type | nullable | description |\n|---|---|---|---|")
        for c, d, n, desc in spec:
            S.append(f"| `{c}` | {d} | {'yes' if n else 'no'} | {desc.replace('|', '\\|')} |")
        S.append("")
    if not freeze:
        (paths.METADATA / "SCHEMA.md").write_text("\n".join(S), encoding="utf-8")

    # ---------------- manifest ----------------
    checks = dataset_checksums()
    split_digest = hashlib.sha256()
    for p in sorted(paths.SPLITS.rglob("manifest.json")):
        split_digest.update(p.read_bytes())
    try:
        commit = subprocess.check_output(["git", "-C", str(paths.ROOT), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        # dirtiness of the DATASET (code + data that define it); unrelated files are ignored
        dirty = bool(subprocess.check_output(["git", "-C", str(paths.ROOT), "status", "--porcelain", "--",
                                              "data", "hebmorph", "scripts", "run_all.sh"]).decode().strip())
    except Exception:
        commit, dirty = None, None
    retrieval = json.loads((paths.RAW / "retrieval_log.json").read_text())
    if freeze and (commit is None or dirty):
        sys.exit("cannot freeze: dataset must be committed and its worktree clean")
    manifest = dict(
        dataset_name=DATASET_NAME, dataset_version=f"{DATASET_NAME}_{DATASET_VERSION}", frozen=freeze,
        frozen_note=("git_commit is the commit containing this exact dataset; this manifest itself is added in "
                     "the following commit, which carries the release tag") if freeze else None,
        generation_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        git_commit=commit or "no commit yet (repository initialised, nothing committed)", git_worktree_dirty=dirty,
        source_versions={"unimorph_heb": dict(url=paths.UNIMORPH_REPO_URL, commit=paths.UNIMORPH_COMMIT,
                                              retrieved=retrieval["unimorph_heb"]["retrieved"], license=paths.UNIMORPH_LICENSE),
                         "hewiktionary": dict(url=paths.WIKTIONARY_DUMP_URL, dump_date=paths.WIKTIONARY_DUMP_DATE,
                                              sha1=paths.WIKTIONARY_DUMP_SHA1,
                                              retrieved=retrieval["hewiktionary_dump"]["retrieved"], license=paths.WIKTIONARY_LICENSE)},
        source_commits={"unimorph_heb": paths.UNIMORPH_COMMIT, "hewiktionary_dump": paths.WIKTIONARY_DUMP_DATE},
        schema_version=SCHEMA_VERSION, parser_versions=PARSER_VERSIONS,
        random_seeds=dict(splits=[0, 1, 2, 3, 4], class_ood_splits=[0, 1, 2], cell_k=[2, 4, 6, 8],
                          synthetic_v1=json.loads((paths.DERIVED / "synthetic" / "v1" / "synthetic_config.json").read_text())["seed"]
                          if (paths.DERIVED / "synthetic" / "v1" / "synthetic_config.json").exists() else None,
                          synthetic_splits=[0, 1, 2]),
        counts=dict(roots=len(roots), lexemes=len(lex), forms=len(forms), unimorph_lexemes=len(um),
                    complete_past_paradigms=int(lex.past_complete.sum()),
                    past_reinflection_examples=len(pr), past_reinflection_eligible=int(pr.analysis__eligible_default.sum()),
                    split_manifests=int(len(list(paths.SPLITS.rglob("manifest.json")))),
                    tiers={t: int((lex.quality_tier == t).sum()) for t in ["GOLD", "SILVER", "QUESTIONABLE"]}),
        checksums=checks, split_manifests_sha256=split_digest.hexdigest(),
    )
    ms.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    if not freeze:
        print((R / "corpus_summary.md").read_text()[:3000])
    else:
        print(json.dumps({k: manifest[k] for k in ["dataset_version", "frozen", "git_commit", "counts"]}, indent=1))


if __name__ == "__main__":
    main(freeze="--freeze" in sys.argv)
