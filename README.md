# HEB_MORPH_GROK v0.1 — Modern Hebrew verbal morphology database

A provenance-aware master database of Modern Hebrew verb paradigms, plus experimental views and
reusable split manifests. It is built for studying whether small networks discover roots, binyanim and
inflectional features. **No models are trained here.**

* Dataset card (purpose, phenomena, limitations, tiers, splits): [`DATASET_CARD.md`](DATASET_CARD.md)
* Sources and licences: [`data/metadata/SOURCES.md`](data/metadata/SOURCES.md)
* Every column of every canonical table: [`data/metadata/SCHEMA.md`](data/metadata/SCHEMA.md) (generated from `hebmorph/schema.py`)
* Numbers: [`data/reports/corpus_summary.md`](data/reports/corpus_summary.md)

## Rebuild

```bash
bash run_all.sh        # fetch -> ingest -> canonical -> views/splits -> synthetic -> summary -> tests
```
Requirements: Python ≥3.10, pandas, pyarrow, numpy, pytest. Everything is UTF-8 and NFC.

## Pipeline

```
RAW SOURCES            data/raw/            never edited (UniMorph git clone @ pinned commit; Wiktionary dump, sha1-verified)
  → NORMALIZATION      data/interim/        flat parsed tables + verbatim cache of every Wiktionary page used
  → CANONICAL DB       data/canonical/      roots / lexemes / forms / provenance / validation  (Parquet + TSV)
  → VALIDATION / QC    data/reports/        conflicts, duplicates, syncretism, coverage, completeness, distributions
  → EXPERIMENTAL VIEWS data/derived/        past_reinflection/, full_paradigm/, synthetic/v1/
  → SPLIT MANIFESTS    data/splits/         <view>/<split>/manifest.json + assignments.tsv (verified, with stats)
```

| code | role |
|---|---|
| `hebmorph/hebrew.py` | `normalize_nfc`, `strip_niqqud`, `extract_consonantal_skeleton`, character audit (nonstandard/malformed marks are logged, never silently dropped) |
| `hebmorph/unimorph.py` | UniMorph parsing, paradigm segmentation, vocalized↔plene paradigm pairing |
| `hebmorph/wiktionary.py` | dump streaming, balanced template parser, entry/table extraction, table-root validation |
| `hebmorph/morphology.py` | binyan normalization, UniMorph feature normalization, canonical cells, paradigm-shape check |
| `hebmorph/roots.py` | structural root features, shin/sin resolution |
| `hebmorph/matching.py` | transparent UniMorph↔Wiktionary evidence collection and decision |
| `hebmorph/build.py` | canonical builder, tiers, provenance, conflict reports |
| `hebmorph/schema.py` | table contracts; the pipeline fails loudly on violations |
| `hebmorph/derived.py`, `splits.py`, `loader.py` | views, split strategies + leakage verifier, model-facing loader |
| `hebmorph/synthetic.py` | synthetic generator with the same schema |

## The canonical tables in one example

The root כ־ת־ב (`roots`) → 5 lexemes, one per attested binyan (`lexemes`: כָּתַב PAAL, נִכְתַּב NIFAL, כִּתֵּב PIEL,
כֻּתַּב PUAL, הִכְתִיב HIFIL). Each lexeme has one `forms` row per paradigm cell:

| form_id | canonical_cell | form_vocalized | form_unvocalized | person | number | gender | syncretism_group_unvocalized |
|---|---|---|---|---|---|---|---|
| LX-006a00a0ca.PST.1SG | PST.1SG | כָּתַבְתִּי | כתבתי | 1 | SG | NA | |
| LX-006a00a0ca.PST.2MSG | PST.2MSG | כָּתַבְתָּ | כתבת | 2 | SG | MASC | LX-006a00a0ca:U1 |
| LX-006a00a0ca.PST.2FSG | PST.2FSG | כָּתַבְתְּ | כתבת | 2 | SG | FEM | LX-006a00a0ca:U1 |

`provenance` links `LX-006a00a0ca.PST.1SG` to `heb_voc` line 7513 and `heb` line 7533 at the pinned commit, and to the
Wiktionary appendix table at revision 508927 that corroborates it.

## Model-facing data

```python
from hebmorph.loader import load_split
train = load_split("past_reinflection", "root_holdout_seed0", "train", condition="voc")
train[0]   # Example(source_form='…', target_features='PST,1,SG', target_form='…')
```
The loader returns **only** `source_form`, `target_features` and `target_form`. Root, binyan, root class, lexeme
IDs, tiers and provenance stay in the `analysis__*` columns of the derived files, which models must not read.

## Key design decisions

* A **lexeme = root × binyan**. Lexemes are created only from attested source records, never by combining a root with a binyan.
* **One form row = one lexeme × one cell.** Rows are never merged because their strings coincide. Syncretism is recorded in `syncretism_group(_unvocalized)`.
* `form_unvocalized` is the **plene** spelling from UniMorph `heb`. `form_stripped` is the deterministic defective spelling (niqqud removed). They differ (דיבר vs דבר), and both are kept.
* Root and binyan are **never guessed**. If the evidence supports more than one, the field is NULL, the record is QUESTIONABLE and the case is listed in `reports/conflicts.tsv`.
* Structural root classes (computed from the radicals) and traditional גזרות (only as stated by Wiktionary) are **separate columns**.

## Training (minimal synthetic-model pipeline)

`training/` trains small decoder-only transformers on any view/split. It reads data **only** through
`hebmorph.loader`, so a model structurally cannot see root, binyan or other analysis fields.

```bash
# local CPU smoke test (~15 s)
python -m training.train --config configs/syn_smoke.json
# GPU jobs on della (account adele); key=value overrides are optional
sbatch slurm/train.sbatch configs/syn_root_holdout.json
sbatch slurm/train.sbatch configs/syn_root_holdout_grok.json seed=1 run_name='"grok_seed1"'
```

* **Input sequence:** `<bos> source chars <sep> F:PST F:2 F:SG F:FEM <sep> target chars <eos>`. Characters are NFC code points, so niqqud marks are separate tokens. Loss is on the target span only.
* **Model** (`training/model.py`): pre-LN transformer written out explicitly for interpretability. Default: 2 layers, d=128, 4 heads. Attention patterns can be stored via `attn.store_patterns = True`.
* **Optimization:** AdamW, weight decay 1.0 (grokking-style), warmup then constant LR. `train_subsample` sets up data-limited regimes, and `batch_size=-1` gives full-batch training.
* **Outputs** in `runs/<run_name>/`:
  * `config.json`: config, git commit, dataset version, split-manifest sha256, vocabulary.
  * `metrics.jsonl`: loss, token accuracy and exact-match accuracy for train/dev/test, test exact match per target feature bundle, and weight norm.
  * `ckpt/`: log-spaced checkpoints.
* **Configs:** `configs/syn_*.json`. The synthetic data must be generated first (`python scripts/40_synthetic.py`), because synthetic splits are not committed.
* **Environment:** PyTorch ≥2.1. On della: `/home/sr2982/.conda/envs/gpt-env`.
