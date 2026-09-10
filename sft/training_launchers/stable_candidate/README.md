# Stable SFT launcher

This directory keeps one source of truth for runtime paths and one source of
truth for model/training settings:

- `runtime.env`: cluster paths, Python/FA3/TE environment, and network runtime defaults.
- `zgcm.conf`: model architecture, dataset identity guards, parallelism, optimizer, and training settings for the 256K ThinkMix V1 + Locate run.
- `launch.sh`: loads both files, derives iteration counts, validates effective epochs, and starts the runner.
- `run_sft_thinkmix64k_fa3_tpcomm_formal_20260715.sh`: generic launch implementation.
- `pretrain_gpt_rank_cache_mcore_wrapper.sh`: rank-local cache isolation.
- `pretrain_gpt_force_mcore_save.py`: MCore checkpoint save compatibility wrapper.

Run from the `sft/` directory after supplying the required external assets:

```bash
export ENV_ROOT="${ENV_ROOT:?path to the Python runtime}"
export PROJECT_ROOT="${PROJECT_ROOT:?shared output root}"
export SOURCE_CKPT_ROOT="${SOURCE_CKPT_ROOT:?source checkpoint root}"
export DATA_ROOT="${DATA_ROOT:?packed indexed SFT data root}"
export TOKENIZER="${TOKENIZER:?tokenizer directory}"

VALIDATE_CONFIG_ONLY=1 \
bash training_launchers/stable_candidate/launch.sh
```

Use another machine environment or training configuration without editing the
launcher by exporting `SFT_ENV_FILE` and `SFT_CONFIG_FILE` before invocation:

```bash
export SFT_ENV_FILE="$PWD/training_launchers/stable_candidate/runtime.env"
export SFT_CONFIG_FILE="$PWD/training_launchers/stable_candidate/zgcm.conf"
bash training_launchers/stable_candidate/launch.sh
```

Validate dataset identity and all derived iteration values without creating a
run directory or starting distributed training:

Remove `VALIDATE_CONFIG_ONLY=1` only after validation succeeds and distributed
runtime settings are present.

For a bounded smoke test, `SMOKE_TRAIN_ITERS` limits only the active run after
the complete metadata-derived plan has passed validation and automatically
disables checkpoint saving. The LR decay horizon remains the complete planned
run, so smoke mode does not compress a cosine/linear schedule into a few steps.
`SFT_RUN_NAME_OVERRIDE`,
`SFT_GLOBAL_BATCH_SIZE_OVERRIDE`, and `SFT_WORKER_NUM_OVERRIDE` are explicit
runtime-only smoke controls; they do not change `zgcm.conf`.

When the derived data-parallel size is one, the generic runner automatically
disables DP gradient-reduce and parameter-gather overlap because there is no DP
peer. Tensor-parallel communication overlap remains enabled.

Before launch, provide `ENV_ROOT`, `PROJECT_ROOT`, `SOURCE_CKPT_ROOT`,
`DATA_ROOT`, and `TOKENIZER` through the scheduler or a private shell setup.
`runtime.env` derives the remaining paths and must not contain credentials or
checked-in cluster mount paths. Scheduler-provided values such as `RANK`,
`WORLD_SIZE`, `MASTER_ADDR`, and `NODE_RANK` remain runtime inputs.

`launch.sh` reads the actual train token/record counts from `DATASET_METADATA`,
checks them against the optional `EXPECTED_TRAIN_*` identity guards in
`zgcm.conf`, and then derives:

- `EPOCH_ITERS = floor(TRAIN_TOKENS / (SEQ_LENGTH * GLOBAL_BATCH_SIZE))`
- `TRAIN_ITERS = floor(TRAIN_TOKENS * EPOCHS / (SEQ_LENGTH * GLOBAL_BATCH_SIZE))`
- `SAVE_INTERVAL = EPOCH_ITERS * SAVE_EVERY_EPOCHS`

The current configuration uses a 256K training sequence,
`MAX_POSITION_EMBEDDINGS=262144`, and `ROTARY_BASE=10000000`, matching the
256K + Locate training run. Iteration counts are derived from dataset metadata.
