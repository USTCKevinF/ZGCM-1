from __future__ import annotations

import json
import unittest

from zgcm_data_pipeline.core import PipelineContext
from zgcm_data_pipeline.models import DataCategory, Decision, Sample, Stage
from zgcm_data_pipeline.stages.midtrain.pipeline import build_pipeline as build_midtrain
from zgcm_data_pipeline.stages.posttrain.pipeline import build_pipeline as build_posttrain
from zgcm_data_pipeline.stages.pretrain.pipeline import build_pipeline as build_pretrain


def run_one(sample: Sample, pipeline, config: dict[str, object] | None = None) -> Sample:
    return next(pipeline.run([sample], PipelineContext(config=config or {})))


class PretrainDatasetProfileTests(unittest.TestCase):
    def test_gharchive_keeps_source_and_rejects_vendor(self) -> None:
        kept = Sample(
            dataset="gharchive",
            source_id="one",
            stage=Stage.PRETRAIN,
            category=DataCategory.CODE,
            data={"text": "def answer(value):\n    return value + 1\n", "path": "src/answer.py", "source": "gharchive"},
        )
        kept = run_one(kept, build_pretrain("code", "gharchive"))
        self.assertEqual(kept.decision, Decision.KEEP)
        self.assertEqual(kept.buckets["language"], "Python")
        self.assertTrue(kept.data["metadata"]["blob_id"].startswith("sha256:"))

        vendor = Sample(
            dataset="gharchive",
            source_id="two",
            stage=Stage.PRETRAIN,
            category=DataCategory.CODE,
            data={"text": "def vendored():\n    return True\n", "path": "vendor/pkg/code.py", "source": "gharchive"},
        )
        vendor = run_one(vendor, build_pretrain("code", "gharchive"))
        self.assertEqual(vendor.decision, Decision.DROP)
        self.assertIn("gharchive_vendor_or_build_path", vendor.reasons)

    def test_finepdfs_policy_is_executable(self) -> None:
        sample = Sample(
            dataset="finepdfs",
            source_id="pdf-1",
            stage=Stage.PRETRAIN,
            category=DataCategory.PDF_OCR,
            data={
                "id": "pdf-1",
                "source": "finepdfs",
                "text": "A sufficiently long educational PDF passage for model pretraining.",
                "lid_score": 0.93,
                "minhash_count": 1,
                "primary_score": 0.88,
            },
        )
        sample = run_one(sample, build_pretrain("pdf_ocr", "finepdfs"))
        self.assertEqual(sample.decision, Decision.KEEP)
        self.assertEqual(sample.buckets["finepdfs_policy"], "relaxed_lid085_mh2_p08")

    def test_olmocr_drops_compression_artifact(self) -> None:
        sample = Sample(
            dataset="olmocr",
            source_id="pdf-2",
            stage=Stage.PRETRAIN,
            category=DataCategory.PDF_OCR,
            data={"source": "olmocr", "text": "repeated " * 2_000, "edu_score": 0.8},
        )
        sample = run_one(sample, build_pretrain("pdf_ocr", "olmocr"))
        self.assertEqual(sample.decision, Decision.DROP)
        self.assertIn("olmocr_low_compression_ratio", sample.reasons)


class MidtrainDatasetProfileTests(unittest.TestCase):
    def test_web_knowledge_bands_quality_and_length(self) -> None:
        sample = Sample(
            dataset="knowledge",
            source_id="web-1",
            stage=Stage.MIDTRAIN,
            category=DataCategory.WEB,
            data={"source": "knowledge", "text": "Useful grounded knowledge text with enough content.", "quality_mean": 4.2, "token_count": 20_000},
        )
        sample = run_one(sample, build_midtrain("web", "web_knowledge"))
        self.assertEqual(sample.decision, Decision.KEEP)
        self.assertEqual(sample.buckets["classifier_quality"], "ge4")
        self.assertEqual(sample.buckets["source_length"], "gt_16k_le_64k")

    def test_nemotron_math_drops_strict_hard_error(self) -> None:
        sample = Sample(
            dataset="nemotron_math",
            source_id="math-1",
            stage=Stage.MIDTRAIN,
            category=DataCategory.MATH,
            data={
                "source": "nemotron_math",
                "question": "What is 2 + 2?",
                "solution": "We add the two integers. Therefore the final answer is 4.",
                "hard_error_score": 4,
            },
        )
        sample = run_one(sample, build_midtrain("math", "nemotron_math"))
        self.assertEqual(sample.decision, Decision.DROP)
        self.assertIn("nemotron_math_strict_hard_error", sample.reasons)

    def test_agent_coding_accepts_pre_rendered_trace(self) -> None:
        sample = Sample(
            dataset="agent_coding",
            source_id="agent-1",
            stage=Stage.MIDTRAIN,
            category=DataCategory.AGENTIC,
            data={
                "source": "agent_coding",
                "sample_type": "agent_coding_same_repo_multiissue_pack",
                "text": "user: fix the bug\nassistant: inspect the code\ntool: tests passed\nassistant: task completed",
                "metadata": {"source_ids": ["issue-a", "issue-b"]},
            },
        )
        sample = run_one(sample, build_midtrain("agentic", "agent_coding"))
        self.assertEqual(sample.decision, Decision.KEEP)
        self.assertEqual(sample.data["dedup_priority"], 0)
        self.assertEqual(sample.data["source_trace_keys"], ["issue-a", "issue-b"])


class PosttrainDatasetProfileTests(unittest.TestCase):
    def test_dolci_think_splits_reasoning(self) -> None:
        sample = Sample(
            dataset="dolci_think",
            source_id="think-1",
            stage=Stage.POSTTRAIN,
            category=DataCategory.INSTRUCTION,
            data={
                "dataset_source": "saumyamalik/OpenThoughts3-full-filtered-math-decontam-v2",
                "messages": [
                    {"role": "user", "content": "Solve this problem carefully."},
                    {"role": "assistant", "content": "<think>First reason through the arithmetic.</think>The answer is 4."},
                ],
            },
        )
        sample = run_one(sample, build_posttrain("instruction", "dolci_think"))
        self.assertEqual(sample.decision, Decision.KEEP)
        self.assertEqual(sample.data["messages"][1]["reasoning_content"], "First reason through the arithmetic.")
        self.assertEqual(sample.data["messages"][1]["content"], "The answer is 4.")

    def test_dolci_tooluse_converts_legacy_calls(self) -> None:
        sample = Sample(
            dataset="dolci_tooluse",
            source_id="tool-1",
            stage=Stage.POSTTRAIN,
            category=DataCategory.AGENTIC,
            data={
                "source": "dolci_tooluse",
                "domain": "Tool Use",
                "messages": [
                    {"role": "system", "content": "", "functions": json.dumps([{"name": "weather", "parameters": {"type": "object"}}])},
                    {"role": "user", "content": "Check the weather."},
                    {"role": "assistant", "content": None, "function_calls": "weather(city='Beijing')"},
                    {"role": "environment", "content": "sunny"},
                    {"role": "assistant", "content": "It is sunny."},
                ],
            },
        )
        sample = run_one(sample, build_posttrain("agentic", "dolci_tooluse"))
        self.assertEqual(sample.decision, Decision.KEEP)
        self.assertEqual(sample.data["tools"][0]["function"]["name"], "weather")
        self.assertEqual(sample.data["messages"][1]["tool_calls"][0]["function"]["arguments"], {"city": "Beijing"})
        self.assertEqual(sample.data["messages"][2]["role"], "tool")

    def test_ultradata_repairs_visible_think_leak(self) -> None:
        sample = Sample(
            dataset="ultradata",
            source_id="ultra-1",
            stage=Stage.POSTTRAIN,
            category=DataCategory.INSTRUCTION,
            data={
                "source": "ultradata",
                "messages": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "Give a concise answer."},
                    {"role": "assistant", "content": "<think>hidden text</think>Visible answer."},
                ],
            },
        )
        sample = run_one(sample, build_posttrain("instruction", "ultradata"))
        self.assertEqual(sample.decision, Decision.KEEP)
        self.assertEqual([message["role"] for message in sample.data["messages"]], ["user", "assistant"])
        self.assertEqual(sample.data["messages"][1]["content"], "Visible answer.")


if __name__ == "__main__":
    unittest.main()
