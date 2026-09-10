# ZGCM-1 supervised fine-tuning implementation

This directory provides the ZGCM-1 Megatron-LM stack for supervised fine-tuning
with packed indexed data, GLM5.1 tokenization, variable-length FlashAttention
3, Muon optimization, and distributed checkpoint loading.

## Training components

- `megatron/training/datasets/sft_dataset.py` reads aligned tokens, targets,
  packed-sequence metadata, labels, and loss masks.
- `megatron/core/tokenizers/text/libraries/sft_tokenizer.py` defines the static
  GLM5.1 tokenizer and conversation-template contract.
- `megatron/training/utils.py` propagates packed-sequence fields across tensor
  parallel ranks.
- `megatron/training/datasets/data_samplers.py` provides cyclic epoch-aware
  sample ordering.
- `pretrain_gpt.py` connects indexed SFT data to the GPT training loop.
- `megatron/core/transformer/attention.py` provides variable-length FA3 and
  layer-selective attention output gating.
- `megatron/core/optimizer/emerging_optimizers.py` provides TP-aware Muon
  handling for gated and ungated layers.
- `megatron/core/dist_checkpointing/strategies/torch.py` provides distributed
  checkpoint loading across the configured topology.

## Runtime and experiment configuration

`training_launchers/stable_candidate/runtime.env` contains the machine-facing
runtime contract. The scheduler or private shell supplies:

- `ENV_ROOT`
- `PROJECT_ROOT`
- `SOURCE_CKPT_ROOT`
- `DATA_ROOT`
- `TOKENIZER`

`training_launchers/stable_candidate/zgcm.conf` contains model architecture,
dataset identity guards, parallelism, optimizer, schedule, precision,
recompute, evaluation, and checkpoint settings.

`training_launchers/stable_candidate/launch.sh` reads dataset metadata,
validates dataset identity, computes iteration intervals, and invokes the
Megatron runner. `SFT_ENV_FILE` and `SFT_CONFIG_FILE` select alternate runtime
and experiment files without changing the launcher.

## Reference configuration

The checked configuration uses:

- 256K sequence length and maximum positions;
- RoPE base 10M;
- TP8, PP1, CP1, MBS1, and GBS48;
- BF16 with Transformer Engine FP8 hybrid delayed scaling;
- variable-length FA3 and full activation recomputation;
- Muon with Adam scalar optimization;
- cosine learning rate from `1e-4` to `1e-6`;
- ten token-based epochs with one checkpoint interval per epoch.

Actual train tokens and records come from `DATASET_METADATA`. Expected counts
in `zgcm.conf` are dataset identity guards. The launcher computes:

```text
EPOCH_ITERS   = floor(TRAIN_TOKENS / (SEQ_LENGTH * GLOBAL_BATCH_SIZE))
TRAIN_ITERS   = floor(TRAIN_TOKENS * EPOCHS / (SEQ_LENGTH * GLOBAL_BATCH_SIZE))
SAVE_INTERVAL = EPOCH_ITERS * SAVE_EVERY_EPOCHS
```

## Validation evidence

- Shell syntax and Python syntax checks pass for the release training path.
- Dataset metadata contains 76,325 train records and 19,462,049,882 train
  tokens.
- The reference plan computes 1,546 iterations per epoch and 15,467 total
  iterations, with an effective epoch count of 9.999969226469.
- An 8-H100 TP8/PP1/CP1/DP1 run completed one optimizer step plus validation
  and test iterations with finite loss and zero skipped or NaN iterations.
- A 64-H100 TP8/PP1/CP1/DP8 run completed two optimizer steps plus 20
  validation and 20 test iterations with DP/TP overlap enabled.
- The 64-H100 steady step took approximately 33.32 seconds at GBS48, about
  377.6K tokens per second cluster-wide, with 98.05% average active GPU
  utilization and about 69.75 GiB peak device memory per GPU.

These checks establish runtime and distributed-training compatibility; they do
not measure convergence or model quality.

## Repository boundaries

Datasets, checkpoints, tokenizer assets, Python environments, credentials,
generated run directories, and scheduler-specific paths are external inputs.
The checked runtime files contain only environment-variable contracts and
portable defaults.
