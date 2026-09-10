# ZGCM-1 supervised fine-tuning

This directory contains the validated ZGCM-1 Megatron-LM stack for supervised
fine-tuning (SFT). It includes the GLM5.1 tokenizer/template contract, packed
indexed SFT dataset support, FA3 variable-length attention, the Muon optimizer,
distributed checkpoint compatibility, and a single production launcher.

The checked-in candidate configuration is based on the 256K ThinkMix V1 +
Locate run. Runtime paths and experiment settings are deliberately separate so
that a new cluster or dataset does not require editing the launcher.

## Layout

| Path | Purpose |
| --- | --- |
| `training_launchers/stable_candidate/runtime.env` | Machine paths, Python/FA3/TE environment, NCCL/Gloo defaults |
| `training_launchers/stable_candidate/zgcm.conf` | Model, data identity, parallelism, optimizer, schedule, and epoch settings |
| `training_launchers/stable_candidate/launch.sh` | Validates metadata, derives iteration counts, and starts training |
| `training_launchers/stable_candidate/run_sft_thinkmix64k_fa3_tpcomm_formal_20260715.sh` | Generic, sequence-length-agnostic Megatron invocation |
| `docs/sft/README.md` | Implementation scope and validation evidence |
| `docs/sft/SOURCE_PROVENANCE.md` | Megatron-LM source revision |

## Requirements

- Linux workers with eight CUDA GPUs per worker.
- Python 3.12 environment containing the compatible PyTorch, Transformer
  Engine, FA3 overlay, and repository dependencies.
- A Megatron distributed checkpoint compatible with the configured model.
- A static GLM5.1 tokenizer directory.
- Packed indexed SFT data with aligned `tokens`, `targets`, and `cu_seqlens`
  `.bin/.idx` files.
- A per-split data-args JSON and a metadata/ready JSON containing positive
  `train_tokens` and, preferably, `train_records`.
- NCCL/RDMA connectivity for multi-node runs.

Concrete cluster paths, credentials, cookies, and proxy addresses must remain
outside the repository.

## Runtime environment

Export the five required runtime roots before launching. Values below are
environment-specific placeholders; do not add cluster paths to the repository.

```bash
export ENV_ROOT="${ENV_ROOT:?path to the Python runtime}"
export PROJECT_ROOT="${PROJECT_ROOT:?shared output root}"
export SOURCE_CKPT_ROOT="${SOURCE_CKPT_ROOT:?source checkpoint root}"
export DATA_ROOT="${DATA_ROOT:?packed indexed SFT data root}"
export TOKENIZER="${TOKENIZER:?tokenizer directory}"
```

`runtime.env` derives the common paths. Override them when the storage naming
scheme differs:

```bash
export CKPT_LOAD_ITER=8000
export SOURCE_CKPT_ITER="${SOURCE_CKPT_ROOT}/iter_0008000"
export SOURCE_DATA_ARGS="$DATA_ROOT/meta/indexed_per_split_data_args.json"
export INDEXED_DONE="$DATA_ROOT/DATA_READY"
export DATASET_METADATA="$INDEXED_DONE"
```

The metadata path is authoritative for iteration accounting. Expected token and
record counts in `zgcm.conf` are identity guards, not iteration inputs.

## Validate configuration

Validate dataset metadata, identity, and derived iteration counts without
creating a run or starting distributed workers:

```bash
VALIDATE_CONFIG_ONLY=1 \
  bash training_launchers/stable_candidate/launch.sh
```

The launcher computes:

```text
EPOCH_ITERS  = floor(TRAIN_TOKENS / (SEQ_LENGTH * GLOBAL_BATCH_SIZE))
TRAIN_ITERS  = floor(TRAIN_TOKENS * EPOCHS / (SEQ_LENGTH * GLOBAL_BATCH_SIZE))
SAVE_INTERVAL = EPOCH_ITERS * SAVE_EVERY_EPOCHS
```

It rejects a plan whose effective epoch count differs from the requested epoch
count by `0.01` or more.

## Single-node smoke test

Use all eight local GPUs with TP=8. DP overlap is automatically disabled when
the derived DP size is one; TP communication overlap remains enabled.

```bash
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29731

SFT_WORKER_NUM_OVERRIDE=1 \
SFT_RUN_NAME_OVERRIDE=zgcm-sft-smoke \
SMOKE_TRAIN_ITERS=2 \
  bash training_launchers/stable_candidate/launch.sh
```

Smoke mode disables checkpoint saving. It preserves the metadata-derived full
LR decay horizon, so limiting active iterations does not collapse the learning
rate schedule.

## Multi-node launch

Use the scheduler's PyTorch-DDP mode with one worker per eight-GPU node and RDMA
enabled. The scheduler must provide a shared `MASTER_ADDR` and a unique node
rank. If its `WORLD_SIZE` and `RANK` denote node count and node rank, use:

```bash
export NNODES="${WORLD_SIZE}"
export NODE_RANK="${RANK}"
export NPROC_PER_NODE=8

bash training_launchers/stable_candidate/launch.sh
```

For the validated 64-GPU layout, eight workers with TP=8, PP=1, CP=1 produce
DP=8. With GBS=48 and MBS=1, each optimizer step contains six microbatches per
data-parallel rank.

The scheduler entrypoint should contain only runtime exports and the call to
`launch.sh`; model and optimizer arguments belong in `zgcm.conf`.

## Full training

After configuration validation and a smoke test, launch without
`SMOKE_TRAIN_ITERS`:

```bash
bash training_launchers/stable_candidate/launch.sh
```

The checked candidate uses:

- 256K sequence length and maximum positions, RoPE base 10M.
- TP8, PP1, CP1, MBS1, GBS48.
- BF16 with Transformer Engine FP8 hybrid delayed scaling.
- FA3 variable-length flash attention and full activation recomputation.
- Muon with Adam scalar optimizer.
- Cosine LR from `1e-4` to `1e-6`, no warmup, weight decay `0.01`.
- Ten token-based epochs with one checkpoint per epoch.

To run another experiment without copying the launcher:

Set `SFT_ENV_FILE` and `SFT_CONFIG_FILE` in the scheduler or private shell,
then run `bash training_launchers/stable_candidate/launch.sh` from `sft/`.

## Outputs

Each run is written below:

```text
$PROJECT_ROOT/megatron_runs/$RUN_NAME/
├── checkpoints/
├── configs/run_manifest.json
├── data/per_split_data_args.json
├── data-cache/
├── logs/
├── status/train.done
└── tensorboard/
```

Rank-local logs include the resolved checkpoint, data inputs, parallelism,
optimizer, overlap settings, and GPU telemetry. `run_manifest.json` records the
effective launch configuration.

## Validated configurations

- Single node, 8 H100: TP8/DP1, 256K, FA3, Muon, full recompute, one optimizer
  step followed by validation/test; completed successfully.
- Eight nodes, 64 H100: TP8/DP8, 256K, FA3, Muon, DP/TP overlap, two optimizer
  steps followed by 20 validation and 20 test iterations; completed without
  NaN, OOM, NCCL error, traceback, or skipped step.
- The 64-GPU steady step took approximately 33.32 seconds at GBS48, equivalent
  to about 377.6K tokens/s for the cluster. Active GPU utilization averaged
  98.05%, and peak device memory was about 69.75 GiB per GPU.

These short runs validate runtime behavior and distributed communication. They
are not convergence or model-quality evaluations.

## Operational notes

- A TP/PP mismatch warning is expected when a distributed checkpoint is
  resharded into a different TP/PP layout; parameter loading must still report
  success.
- The generic flash-attention version warning may not recognize the FA3 overlay.
  Verify that `flash_attn_interface.py` exports `flash_attn_varlen_func` and that
  the run manifest identifies the FA3 path.
- DP grad-reduce and parameter-gather overlap require DP greater than one.
- Iteration counts are derived from dataset metadata, sequence length, global
  batch size, and epoch count by `launch.sh`.
- A smoke run does not validate checkpoint save/resume because saving is
  intentionally disabled.

## Upstream

This directory is based on NVIDIA Megatron-LM and Megatron Core. Upstream
documentation remains under `docs/`; ZGCM-1-specific SFT behavior is documented
under `docs/sft/`.
