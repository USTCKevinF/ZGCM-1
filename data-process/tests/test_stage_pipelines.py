from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from zgcm_data_pipeline.models import Decision
from zgcm_data_pipeline.stages.midtrain.buckets import assign_length_bucket
from zgcm_data_pipeline.stages.midtrain.mix import prefix_sample_mix
from zgcm_data_pipeline.stages.posttrain.format import split_think_content
from zgcm_data_pipeline.stages.posttrain.pipeline import build_pipeline as build_posttrain
from zgcm_data_pipeline.stages.pretrain.pipeline import build_pipeline as build_pretrain


class StagePipelineTests(unittest.TestCase):
    def test_pretrain_source_pipelines_have_no_environment_paths(self) -> None:
        self.assertIn("normalize_pdf_ocr", build_pretrain("pdf_ocr").step_names())
        self.assertIn("normalize_general_text", build_pretrain("general_text").step_names())
        self.assertIn("filter_non_source_assets", build_pretrain("code").step_names())

    def test_pretrain_pdf_cleaning_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.jsonl"
            output = root / "output.jsonl"
            source.write_text(json.dumps({"id": "p1", "content": "A sufficiently long extracted document."}) + "\n", encoding="utf-8")
            from zgcm_data_pipeline.stages.pretrain.pipeline import run_jsonl

            summary = run_jsonl(source, output, dataset="demo", source="pdf_ocr")
            self.assertEqual(summary["written"], 1)
            record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["stage"], "pretrain")

    def test_prefix_mix_is_disjoint_and_prefix_eligible(self) -> None:
        buckets = {"B16": ["a", "b", "c"], "B64": ["d", "e"], "B256": ["f", "g"]}
        result = prefix_sample_mix(buckets, {"mix16": 1, "mix64": 2, "mix256": 2}, seed="test")
        flattened = [item for values in result.values() for item in values]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(sum(len(values) for values in result.values()), 5)

    def test_length_bucket_labels(self) -> None:
        self.assertEqual(assign_length_bucket(16_384), "B16")
        self.assertEqual(assign_length_bucket(65_536), "B64")
        self.assertEqual(assign_length_bucket(262_144), "B256")

    def test_posttrain_think_split_and_pipeline(self) -> None:
        reasoning, visible = split_think_content("<think>reasoning</think>answer")
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(visible, "answer")
        self.assertIn("normalize_messages", build_posttrain("instruction").step_names())


if __name__ == "__main__":
    unittest.main()
