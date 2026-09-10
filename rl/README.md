# ZGCM-1 Reinforcement Learning

ZGCM-1 explores mixed reinforcement learning on mathematics, code, and
general-capability tasks using **Group Relative Policy Optimization (GRPO)**.
The recipe follows Section 4.2 of the technical report.

## Data and rewards

Prompt difficulty is estimated using rollouts from the initial policy. Problems
that are already solved frequently are removed to concentrate training on
useful learning signals.

- Mathematics uses binary answer-correctness rewards.
- Code uses the fraction of executable tests passed.
- General tasks use correctness or instruction-following criteria.
- Invalid or truncated responses receive no positive reward.

The experiments also explore reference-policy KL regularization and a mild
length penalty for long responses. Dynamic sampling replaces response groups
with zero reward variance.

## Training settings

| Setting | Value |
| --- | --- |
| Optimizer objective | GRPO |
| Actor learning rate | `2e-6` |
| Sampling scale 1 | 24 prompts × 16 responses = 384 trajectories |
| Sampling scale 2 | 384 prompts × 8 responses = up to 3,072 trajectories |
| Maximum generated response | 65,536 tokens |
| Total prompt–response context | 98,304 tokens |

See [Environment](ENVIRONMENT.md) for experiment inputs and
[Experiment Protocol](RUNNING.md) for the training procedure.
