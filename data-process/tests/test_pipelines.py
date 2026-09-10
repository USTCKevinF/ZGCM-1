from __future__ import annotations

import unittest

from zgcm_data_pipeline import DataCategory, PipelineContext, Sample, Stage, build_pipeline
from zgcm_data_pipeline.models import Decision
from zgcm_data_pipeline.training import (
    IGNORE_INDEX,
    TokenizedSample,
    WorkUnit,
    allocate_mix_quotas,
    balance_work_units,
    build_loss_mask,
    pack_tokenized_samples,
    plan_row_microshards,
    select_mixed_sample_ids,
    stable_partition,
)


def sample(category: DataCategory, data: dict, fields: tuple[str, ...]) -> Sample:
    return Sample("demo", "1", Stage.MIDTRAIN, category, data, fields)


class PipelineTests(unittest.TestCase):
    def run_one(self, category: DataCategory, data: dict, fields: tuple[str, ...]) -> Sample:
        return list(build_pipeline(category).run([sample(category, data, fields)]))[0]

    def test_code_pipeline(self) -> None:
        result = self.run_one(
            DataCategory.CODE,
            {"text": "def add(a, b):\n    return a + b\n", "language": "Python", "source": "repo"},
            ("text",),
        )
        self.assertEqual(result.decision, Decision.KEEP)
        self.assertEqual(result.buckets["language"], "Python")
        self.assertIn("length", result.buckets)

    def test_code_content_is_mapped_before_integrity_check(self) -> None:
        result = self.run_one(
            DataCategory.CODE,
            {"content": "def multiply(a, b):\n    return a * b\n", "source": "repo"},
            ("text",),
        )
        self.assertEqual(result.decision, Decision.KEEP)
        self.assertIn("text", result.data)

    def test_web_qa_unknown_is_dropped(self) -> None:
        result = self.run_one(
            DataCategory.WEB,
            {"question": "What is the answer?", "answer": "Unknown", "source": "web"},
            ("question", "answer"),
        )
        self.assertEqual(result.decision, Decision.DROP)

    def test_agentic_pairs(self) -> None:
        result = self.run_one(
            DataCategory.AGENTIC,
            {
                "text": "Task completed successfully after tool execution.",
                "messages": [
                    {"role": "user", "content": "inspect"},
                    {"role": "assistant", "tool_calls": [{"name": "shell"}]},
                    {"role": "tool", "content": "ok"},
                ],
                "source": "agent",
            },
            ("text",),
        )
        self.assertEqual(result.buckets["outcome"], "verified")

    def test_agentic_messages_form_canonical_text(self) -> None:
        result = self.run_one(
            DataCategory.AGENTIC,
            {
                "messages": [
                    {"role": "user", "content": "Inspect the repository and report the result."},
                    {"role": "assistant", "content": "I inspected the repository and task completed successfully."},
                ],
                "source": "agent",
            },
            ("text",),
        )
        self.assertNotEqual(result.decision, Decision.DROP)
        self.assertIn("user: Inspect", result.data["text"])

    def test_instruction_incomplete_is_dropped(self) -> None:
        result = self.run_one(
            DataCategory.INSTRUCTION,
            {"instruction": "Explain TCP", "output": "", "source": "instruction"},
            ("instruction", "output"),
        )
        self.assertEqual(result.decision, Decision.DROP)

    def test_math_missing_image_context_is_review(self) -> None:
        result = self.run_one(
            DataCategory.MATH,
            {"problem": "See the figure and solve x.", "response": "Therefore x = 1.", "source": "math"},
            ("problem", "response"),
        )
        self.assertEqual(result.decision, Decision.REVIEW)

    def test_reasoning_lineage_is_preserved(self) -> None:
        result = self.run_one(
            DataCategory.REASONING,
            {
                "prompt": "Why?",
                "response": "First, we inspect the premise. Therefore the claim follows from the stated assumption.",
                "final_answer": "Yes",
                "source_row_hash": "upstream-1",
                "source": "reasoning",
            },
            ("prompt", "response", "final_answer"),
        )
        self.assertEqual(result.data["lineage_hash"], "upstream-1")

    def test_exact_dedup_is_shared(self) -> None:
        context = PipelineContext()
        first = sample(DataCategory.CODE, {"text": "def f():\n    return 1", "source": "a"}, ("text",))
        second = sample(DataCategory.CODE, {"text": "def f():\n    return 1", "source": "b"}, ("text",))
        results = list(build_pipeline(DataCategory.CODE).run([first, second], context))
        self.assertEqual(results[0].decision, Decision.KEEP)
        self.assertEqual(results[1].decision, Decision.DROP)

    def test_optional_near_dedup_is_shared(self) -> None:
        context = PipelineContext(config={"near_dedup": True})
        base = "The quick brown fox jumps over the lazy dog near the river bank. " * 20
        first = sample(
            DataCategory.WEB,
            {"text": base, "source": "a"},
            ("text",),
        )
        second = sample(
            DataCategory.WEB,
            {"text": base.replace("river bank.", "river banks.", 1), "source": "b"},
            ("text",),
        )
        results = list(build_pipeline(DataCategory.WEB).run([first, second], context))
        self.assertNotEqual(results[0].decision, Decision.DROP)
        self.assertEqual(results[1].decision, Decision.DROP)
        self.assertIn("near_duplicate", results[1].reasons)

    def test_source_quality_policy(self) -> None:
        context = PipelineContext(
            config={
                "quality_policies": {
                    "web_a": {"score_field": "quality_mean", "keep_min": 4, "review_min": 3}
                }
            }
        )
        row = sample(
            DataCategory.WEB,
            {"text": "A sufficiently long web document for quality filtering.", "source": "web_a", "quality_mean": 2.5},
            ("text",),
        )
        result = list(build_pipeline(DataCategory.WEB).run([row], context))[0]
        self.assertEqual(result.decision, Decision.DROP)
        self.assertIn("quality_score_below_drop_threshold", result.reasons)


class TrainingPreparationTests(unittest.TestCase):
    def test_sft_stage_is_supported(self) -> None:
        self.assertEqual(Stage("sft"), Stage.SFT)

    def test_stable_partition_is_deterministic(self) -> None:
        self.assertEqual(stable_partition("row-1", 32, seed="v1"), stable_partition("row-1", 32, seed="v1"))

    def test_mix_quotas_sum_to_requested_budget(self) -> None:
        quotas = allocate_mix_quotas({"code": 0.5, "web": 0.3, "math": 0.2}, 11)
        self.assertEqual(quotas, {"code": 6, "web": 3, "math": 2})
        self.assertEqual(sum(quotas.values()), 11)

    def test_mixed_sample_selection_is_deterministic(self) -> None:
        sources = {"code": ["c1", "c2", "c3"], "web": ["w1", "w2"]}
        quotas = {"code": 2, "web": 1}
        first = select_mixed_sample_ids(sources, quotas, seed="mix-v1")
        second = select_mixed_sample_ids(sources, quotas, seed="mix-v1")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_lpt_balances_work(self) -> None:
        groups = balance_work_units([WorkUnit("a", 10), WorkUnit("b", 7), WorkUnit("c", 3)], 2)
        loads = sorted(sum(unit.weight for unit in group) for group in groups)
        self.assertEqual(loads, [10, 10])

    def test_microshards_preserve_row_ranges(self) -> None:
        plans = plan_row_microshards("source", [4, 4, 4, 4], target_tokens=10)
        self.assertEqual([(p.row_start, p.row_end, p.tokens) for p in plans], [(0, 2, 8), (2, 4, 8)])

    def test_sft_packing_preserves_targets_and_boundaries(self) -> None:
        rows = [
            TokenizedSample((1, 2, 3), "a", (IGNORE_INDEX, 2, 3)),
            TokenizedSample((4, 5), "b", (IGNORE_INDEX, 5)),
        ]
        packed, overlength = pack_tokenized_samples(rows, sequence_length=5)
        self.assertFalse(overlength)
        self.assertEqual(len(packed), 1)
        self.assertEqual(packed[0].cu_seqlens, [0, 3, 5])
        self.assertEqual(build_loss_mask(packed[0].targets or []), (0, 1, 1, 0, 1))


if __name__ == "__main__":
    unittest.main()
