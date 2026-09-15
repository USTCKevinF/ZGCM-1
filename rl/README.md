# ZGCM-1 Reinforcement Learning

ZGCM-1 explores mixed reinforcement learning on mathematics, code, and
general-capability tasks using **Group Relative Policy Optimization (GRPO)**.
The recipe follows Section 4.2 of the technical report, and this directory
contains the exact configuration, reward implementation, and training entry
used for the final math GRPO stage of the released model (run after the DPO
stage). The training framework is [AReaL](https://github.com/areal-project/AReaL)
v2.0.0; see [Environment](ENVIRONMENT.md).

## Data and rewards

Prompt difficulty is estimated using rollouts from the initial policy. Problems
that are already solved frequently are removed to concentrate training on
useful learning signals.

- Mathematics uses binary answer-correctness rewards.
- Code uses the fraction of executable tests passed.
- General tasks use correctness or instruction-following criteria.
- Invalid or truncated responses receive no positive reward.

The released `rewards/` package implements the mathematics reward: the visible
answer is extracted after a closed `</think>` section (`rewards/completion.py`)
and compared against the gold labels with a math-verify worker
(`precision=6`, `timeout=8s`, bidirectional equivalence), returning a strictly
binary `r_raw ∈ {0, 1}`.

### Correct-only length penalty

Long correct answers are shaped with a linear penalty on the number of
generated tokens (implemented in `train/length_shaped_math_grpo.py`):

```
penalty = 0.05 * clip((n_out - 16384) / (65536 - 16384), 0, 1)
r       = r_raw * (1 - penalty)
```

- The penalty ramps linearly from 16,384 generated tokens to the 65,536 cap
  and only applies when `r_raw = 1`: a fully correct answer at the cap
  receives 0.95 instead of 1.0, a wrong answer always receives 0.
- A response truncated by the length cap (`stop_reason = length`) receives
  reward 0 outright.
- Evaluation runs disable the penalty (`apply_length_penalty = False`) and
  score raw correctness only.

### Dynamic sampling

Response groups are filtered by **raw binary correctness**, not by the shaped
reward (`accept_math_group`): a group of 8 responses is discarded when it is
all-correct or all-wrong (zero correctness variance), when any reward falls
outside [0, 1] or is non-finite, or when the longest trajectory exceeds the
69,632-token learner-side cap.

## Training settings

Final math GRPO stage (`configs/math_grpo_length_shaped.yaml`):

| Setting | Value |
| --- | --- |
| Optimizer objective | GRPO (no critic, no reference policy, `kl_ctl = 0`) |
| Actor optimizer | Adam, lr `1e-6` constant (3% warmup), weight decay 0.01, grad clip 1.0 |
| Clipping | `eps_clip = 0.2`, token-level importance sampling |
| Reward normalization | group mean/std over 8 responses (unbiased) |
| Advantage normalization | batch level |
| Minibatches per update | 12, sequence packing up to 69,632 tokens (FFD) |
| Rejection mask | token-level ratio upper bound 5.0 |
| Sampling per step | 384 prompts × 8 responses = 3,072 trajectories |
| Sampling parameters | temperature 1.0, top-p 1.0 |
| Maximum generated response | 65,536 tokens |
| Total prompt–response context | 98,304 tokens |
| Maximum prompt length | 4,096 tokens |
| Steps / epochs / seed | 177 / 3 / 20260723 |
| Evaluation | every 10 steps, 8 samples per prompt, raw correctness |

Earlier mixed-RL experiments described in the technical report also explored a
larger response-group scale at actor lr `2e-6` and a reference-policy KL
variant; the final released stage uses the values in the table above.

## Repository layout

```
rl/
├── configs/math_grpo_length_shaped.yaml   # run config (env-parameterized)
├── rewards/                               # binary math verifier reward
│   ├── completion.py                      # visible-answer extraction after </think>
│   ├── math_reward.py                     # math-verify equivalence reward
│   └── rewards.py                         # async reward entry used by the workflow
└── train/
    └── length_shaped_math_grpo.py         # training entry: workflow + group filter
                                             # + version-anchored evaluation
```

See [Environment](ENVIRONMENT.md) for the experiment inputs and dependency
pins, and [Experiment Protocol](RUNNING.md) for how to launch a run.
