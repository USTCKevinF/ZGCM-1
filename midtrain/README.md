# ZGCM-1 Midtrain

ZGCM-1 mid-training progressively extends context through **16K → 64K → 256K**
over **600.51B tokens**. The three stages use 180B, 240B, and 180.51B tokens,
respectively, while retaining shorter examples in the longer-context stages.
Interaction traces are reformulated as MDP state-action transitions, alongside
code, mathematics, knowledge, reasoning, and instruction data.

This directory contains Megatron-LM based on commit
`eba2eaf71d34274c1ca0a59ac5e57a6bf53eb732` and the ZGCM-1 long-context training
implementation. See [Mid-Training Curriculum](docs/MIDTRAIN_STAGES.md) for the
data mixture, optimization settings, and long-context qualification experiments
from the technical report.

## Launch interface

| Stage | Configuration | Token budget | Initialization |
| --- | --- | ---: | --- |
| 16K | `configs/stages/01_seq16k.env` | 180B | Final pretrain checkpoint |
| 64K | `configs/stages/02_seq64k.env` | 240B | Final 16K checkpoint |
| 256K | `configs/stages/03_seq256k.env` | 180.51B | Final 64K checkpoint |

```bash
# Run from midtrain/ on every worker.
export SHARED_STORAGE_ROOT="${SHARED_STORAGE_ROOT:?set SHARED_STORAGE_ROOT}"
export MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR}"
export NODE_RANK="${NODE_RANK:-${RANK:?set NODE_RANK or RANK}}"
export ZGCM_REPO_ROOT="$(pwd)"

# Select one stage. Later stages require the preceding stage checkpoint.
STAGE_CONFIG=configs/stages/01_seq16k.env
source "$STAGE_CONFIG"
scripts/build_midtrain_data_args.sh "$DATA_ARGS_PATH" \
  "$MIDTRAIN_LONG64K_ROOT" "$MIDTRAIN_SHORT16K_ROOT"
bash scripts/launch_train.sh "$STAGE_CONFIG"
```

For the later stages, set `STAGE_CONFIG` to `configs/stages/02_seq64k.env` or
`configs/stages/03_seq256k.env` and export `MIDTRAIN_SEQ16K_CHECKPOINT` or
`MIDTRAIN_SEQ64K_CHECKPOINT`, respectively, when the preceding output is
outside the default `TRAIN_ROOT` layout.

The [stage configurations](configs/stages/) declare the model, optimizer,
parallelism, and runtime inputs. Data, tokenizer files, generated data manifests,
and checkpoints are supplied through the environment.
