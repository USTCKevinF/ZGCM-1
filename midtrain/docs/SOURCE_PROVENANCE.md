# Source provenance and runtime

The bundled training source uses NVIDIA Megatron-LM commit
`eba2eaf71d34274c1ca0a59ac5e57a6bf53eb732` (2026-04-07).

The ZGCM-1 training source provides:

- ZGCM-1 attention output gating on sliding-window layers;
- Muon QKV splitting for a stack containing both gated and ungated layers;
- deterministic sequential GPT dataset order used by early Pretrain;
- consumed-sample offset for externally ordered indexed data;
- dataloader prefetch control;
- skipping validation/test dataloaders when `eval_iters=0`.

The reference runtime is Python 3.12.3 with PyTorch
2.7.0a0+79aa17489c.nv25.04 (CUDA 12.9), Transformer Engine
2.15.0.dev0+5f9550ff, FlashAttention 2.7.3 and FlashAttention-3 3.0.0.
