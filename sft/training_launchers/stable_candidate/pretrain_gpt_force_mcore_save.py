#!/usr/bin/env python3
import os
import importlib.util
import runpy
import sys
from pathlib import Path

from megatron.core.dist_checkpointing.strategies import torch as dcp_torch

repo_root = Path(os.environ.get("REPO", Path.cwd()))


def _force_mcore_sync_save(self, sharded_state_dict, checkpoint_dir):
    strategy = os.environ.get("MEGATRON_DCP_SAVE_ASYNC_STRATEGY", "mcore")
    async_request = self.async_save(
        sharded_state_dict, checkpoint_dir, async_strategy=strategy
    )
    async_request.execute_sync()
    del async_request


if os.environ.get("MEGATRON_DCP_FORCE_MCORE_SAVE", "1") == "1":
    dcp_torch.TorchDistSaveShardedStrategy.save = _force_mcore_sync_save
    print(
        "[mcore-save-patch] TorchDistSaveShardedStrategy.save forced to mcore sync path",
        flush=True,
    )

if os.environ.get("MEGATRON_ENABLE_GLM51_SFT_PATCH", "0") == "1":
    tokenizer_file = Path(
        os.environ.get(
            "MEGATRON_GLM51_SFT_TOKENIZER_FILE",
            str(repo_root / "megatron/core/tokenizers/text/libraries/sft_tokenizer.py"),
        )
    )
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"missing GLM51 SFT tokenizer patch file: {tokenizer_file}")
    spec = importlib.util.spec_from_file_location("glm51_sft_tokenizer_patch", tokenizer_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    import megatron.core.tokenizers.text.libraries as tokenizer_libraries
    import megatron.core.tokenizers.text.libraries.sft_tokenizer as sft_tokenizer_module

    tokenizer_libraries.SFTTokenizer = module.SFTTokenizer
    sft_tokenizer_module.SFTTokenizer = module.SFTTokenizer
    print(
        f"[glm51-sft-patch] SFTTokenizer patched from {tokenizer_file}",
        flush=True,
    )

if os.environ.get("MEGATRON_ENABLE_INDEXED_SFT_DATASET_PATCH", "0") == "1":
    dataset_file = Path(
        os.environ.get(
            "MEGATRON_GLM51_SFT_DATASET_FILE",
            str(repo_root / "megatron/training/datasets/sft_dataset.py"),
        )
    )
    if not dataset_file.exists():
        raise FileNotFoundError(f"missing indexed SFT dataset patch file: {dataset_file}")
    spec = importlib.util.spec_from_file_location("glm51_sft_dataset_patch", dataset_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    import megatron.training.datasets.sft_dataset as sft_dataset_module

    sft_dataset_module.PrepackedSFTLowLevelDataset = module.PrepackedSFTLowLevelDataset
    sft_dataset_module.IndexedPrepackedSFTLowLevelDataset = (
        module.IndexedPrepackedSFTLowLevelDataset
    )
    sft_dataset_module.SFTDataset = module.SFTDataset
    sft_dataset_module._is_indexed_prepacked_sft_prefix = (
        module._is_indexed_prepacked_sft_prefix
    )
    sft_dataset_module._is_prepacked_sft_file = module._is_prepacked_sft_file
    print(
        f"[indexed-sft-dataset-patch] SFTDataset patched from {dataset_file}",
        flush=True,
    )

sys.argv = ["pretrain_gpt.py", *sys.argv[1:]]
runpy.run_path(str(repo_root / "pretrain_gpt.py"), run_name="__main__")
