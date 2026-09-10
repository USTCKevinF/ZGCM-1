# ZGCM-1 Pretrain

ZGCM-1 general pretraining uses approximately **4.19T tokens** in two data
stages: **0.99T** curriculum pretraining followed by **3.20T** on the full
mixture. Both use a 16K context. The recipe combines hybrid attention, Muon,
and FP8 delayed scaling.

This directory contains the ZGCM-1 training implementation, stage configurations,
and a multi-node launcher, with Megatron-LM based on commit
`eba2eaf71d34274c1ca0a59ac5e57a6bf53eb732`. See
[General Pretraining](docs/PRETRAIN_PHASES.md) for the data curriculum and
training settings from the technical report.

## Stages

| Stage | Configuration | Data order | Initialization |
| --- | --- | --- | --- |
| 1T curriculum | `configs/stages/01_pretrain_1t_sequential.env` | Sequential | New training run |
| 3.2T full mixture | `configs/stages/02_pretrain_3p2t_global_shuffle.env` | Globally shuffled offline, consumed sequentially | Final stage-1 checkpoint |

## Quick start

```bash
# Run from pretrain/ on every worker.
export SHARED_STORAGE_ROOT="${SHARED_STORAGE_ROOT:?set SHARED_STORAGE_ROOT}"
export MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR}"
export NODE_RANK="${NODE_RANK:-${RANK:?set NODE_RANK or RANK}}"

# Stage 1: 1T curriculum data.
bash scripts/launch_train.sh configs/stages/01_pretrain_1t_sequential.env

# Stage 2: set the stage-1 checkpoint if it is outside the default run root.
export PRETRAIN_1T_FINAL_CHECKPOINT="${PRETRAIN_1T_FINAL_CHECKPOINT:?set PRETRAIN_1T_FINAL_CHECKPOINT}"
bash scripts/launch_train.sh configs/stages/02_pretrain_3p2t_global_shuffle.env
```

The [stage configuration directory](configs/stages/) contains runtime,
architecture, optimizer, parallelism, data, and checkpoint settings. The
launcher loads the selected environment file and initializes its repository
root before starting distributed training.

Data, tokenizer files, checkpoints, caches, and logs are supplied through the
stage environment. Override `PRETRAIN_1T_DATA_PREFIX`,
`PRETRAIN_3P2T_DATA_PREFIX`, `TOKENIZER_PATH`, `MEGATRON_VENV`, and output
roots when the assets do not use the default layout below `SHARED_STORAGE_ROOT`.
