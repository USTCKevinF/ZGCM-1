# Experiment Protocol

Section 4.2 of the ZGCM-1 technical report describes the following mixed-RL
procedure, and `train/length_shaped_math_grpo.py` implements the final math
GRPO stage end to end:

1. Estimate prompt difficulty using rollouts from the initial policy and remove
   problems that the model already solves frequently.
2. Sample response groups of 8 responses for 384 prompts per training step
   (temperature 1.0, top-p 1.0).
3. Score responses with the binary mathematics verifier in `rewards/`, then
   apply the correct-only linear length penalty (16,384 → 65,536 tokens, max
   0.05); truncated responses receive reward 0.
4. Filter groups whose raw binary correctness has zero variance (dynamic
   sampling), plus groups with out-of-range/non-finite rewards or trajectories
   longer than the 69,632-token learner cap.
5. Construct group-relative advantages (group reward normalization, batch
   advantage normalization) and optimize the policy with GRPO at an actor
   learning rate of `1e-6` (`eps_clip = 0.2`, no reference policy, `kl_ctl = 0`).

The response and total-context budgets are 65,536 and 98,304 tokens,
respectively, with prompts capped at 4,096 tokens.

## Launching the released configuration

Training data is not redistributed with this repository; prepare the prompt
JSONL files first (schema below), then:

```bash
export ZGCM_MODEL_PATH=/path/to/zgcm1_checkpoint   # HF checkpoint with tokenizer
export ZGCM_TRAIN_JSONL=/path/to/train.jsonl
export ZGCM_VALID_JSONL=/path/to/valid.jsonl
export AREAL_ADMIN_API_KEY=local-key               # any non-empty string locally
export ZGCM_RL_STATUS_ROOT=./status                # eval markers land here

cd rl
PYTHONPATH=. python train/length_shaped_math_grpo.py configs/math_grpo_length_shaped.yaml
```

Optional gates: `ZGCM_EXPECTED_TRAIN_ROWS` / `ZGCM_EXPECTED_VALID_ROWS` enforce
exact row counts; `ZGCM_N_NODES` / `ZGCM_N_GPUS_PER_NODE` size the cluster.

### Prompt JSONL schema

One JSON object per line:

```json
{
  "sample_id": "math-000155d909841838fb9e67c7",
  "domain": "math",
  "task_type": "math",
  "messages": [{"role": "user", "content": "..."}],
  "answers": ["28"],
  "benchmark": "aime24"
}
```

`sample_id`, `domain`, `task_type`, `messages`, `answers` are required
(`domain` must be `math`, `answers` non-empty); `benchmark` is optional and
only splits evaluation statistics. Prompts must render to at most 4,096 tokens.

### Evaluation and recovery

Evaluation runs every 10 steps (and at step 0 and 177) with 8 samples per
prompt and raw correctness scoring; results are written as version-anchored
markers under `$ZGCM_RL_STATUS_ROOT/evaluations/version_XXX.json`, which also
makes periodic evaluation idempotent across restarts. Checkpoints are saved
every 10 steps and recovery state every step (`recover.mode: auto`).
