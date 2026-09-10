# Pretrain Stage

The Pretrain stage prepares code, PDF/OCR, and general text records for pretraining. It implements portable, record-level logic; archive discovery, OCR extraction, sharding, workers, tokenization, and indexed-dataset writing are supplied by the caller.

In the AI-native governance model, the shared Pretrain pipeline provides source-family guardrails while AI can create an independent profile for each dataset. `gharchive.py`, `finepdfs.py`, and `olmocr.py` show three dataset-specific profiles.

## Files and scripts

| File | Description |
|---|---|
| [`cli.py`](cli.py) | Stage CLI `zgcm-pretrain-clean`; reads JSONL and writes cleaned records, a summary, and an optional manifest. |
| [`pipeline.py`](pipeline.py) | Combines the selected dataset profile, source-family steps, and shared steps; exposes `run_jsonl()`. |
| [`source_cleaning.py`](source_cleaning.py) | Shared source-family normalization for code, PDF/OCR, and general text. |
| [`datasets/gharchive.py`](datasets/gharchive.py) | Filters paths, extensions, binaries, vendor/build/generated/minified files; normalizes notebooks and adds a stable blob hash. |
| [`datasets/finepdfs.py`](datasets/finepdfs.py) | Applies language-score, MinHash-overlap, primary-quality, and provenance dedup rules. |
| [`datasets/olmocr.py`](datasets/olmocr.py) | Applies compression-ratio and optional education-score filters. |
| [`datasets/__init__.py`](datasets/__init__.py) | Registers profiles and their allowed source families. |

## Source families

| `--source` | Typical content | Main processing |
|---|---|---|
| `code` | Repository source files | Maps `text`/`path`/`language`; rejects vendor, build, lockfile, binary, fixture, generated, and minified files. |
| `pdf_ocr` | PDF extraction and OCR text | Maps `text`, `content`, `ocr_text`, or `document`; short text and low OCR scores are normally marked for review. |
| `general_text` | Other text collections | Maps `text`, `content`, `document`, `body`, or `raw_text`; missing text is dropped. |

Shared steps then perform schema/text normalization, integrity, provenance/license, sensitive-content, exact/near deduplication, token estimation, length bucketing, and final decisions.

## Dataset profiles

| `--dataset-profile` | Required source | Rules |
|---|---|---|
| `gharchive` | `code` | Filters non-source assets, classifies language, keeps only notebook source cells, and adds `metadata.blob_id` plus `gharchive_bucket=core`. |
| `finepdfs` | `pdf_ocr` | Defaults to `lid >= 0.85`, `minhash <= 2`, and `primary >= 0.8`; missing metrics are marked for review. |
| `olmocr` | `pdf_ocr` | Defaults to compression ratio `>= 0.08`; when `edu_score` is present, the default minimum is `0.6`. |

Profile thresholds can be overridden in JSON configuration, including `gharchive_max_file_bytes`, `finepdfs_lid_threshold`, `finepdfs_minhash_max`, `finepdfs_primary_threshold`, `olmocr_min_compression_ratio`, and `olmocr_min_edu_score`.

## CLI example

Run from the `data-process/` directory:

```bash
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
mkdir -p "${OUTPUT_DIR}"
PYTHONPATH=src python3 -m zgcm_data_pipeline.stages.pretrain.cli \
  --input examples/datasets/gharchive.jsonl \
  --output "${OUTPUT_DIR}/gharchive.cleaned.jsonl" \
  --dataset gharchive \
  --source code \
  --dataset-profile gharchive \
  --manifest "${OUTPUT_DIR}/gharchive.manifest.json"
```

Without a profile, select only a source family:

```bash
INPUT_JSONL="${INPUT_JSONL:?set INPUT_JSONL}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
mkdir -p "${OUTPUT_DIR}"
PYTHONPATH=src python3 -m zgcm_data_pipeline.stages.pretrain.cli \
  --input "${INPUT_JSONL}" --output "${OUTPUT_DIR}/output.jsonl" \
  --dataset my_text --source general_text
```

Common options include `--id-field`, `--text-fields`, `--config`, `--require-license`, `--manifest`, `--summary`, and `--keep-dropped`.

## Examples in this stage

[`examples/datasets/gharchive.jsonl`](../../../../examples/datasets/gharchive.jsonl) is the only Pretrain-specific input example. It contains `src/example.py`, which is retained, and `vendor/example.py`, which is dropped by the GHArchive profile. FinePDFs and OLMOCR currently have profile implementations but no dedicated JSONL examples. `quality-policies.json` is shared configuration, and `instruction.jsonl` is a generic Midtrain example. See [`examples/README.md`](../../../../examples/README.md) for the complete list.

## Output and limits

Outputs follow the package contract with `decision`, `reasons`, `flags`, `buckets`, `metrics`, and `record_sha256`. The manifest is rebuilt from serialized output. Length estimates are approximate without a real tokenizer; production training should inject the target tokenizer and validate sharding, checksums, indexed writer/loader compatibility, and corpus distributions.
