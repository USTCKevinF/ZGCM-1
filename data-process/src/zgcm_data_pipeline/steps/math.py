from __future__ import annotations

import re

from ..core import PipelineContext, Step
from ..models import Decision, Sample


_IMAGE_DEP = re.compile(r"(?i)(?:<img|\.(?:png|jpg|jpeg|gif|svg)\b|\\includegraphics|see (?:the )?figure)")
_ANSWER_SIGNAL = re.compile(r"(?i)(?:final answer|answer\s*:|therefore|thus|\\boxed\{|答案|所以)")


def normalize_math_schema(sample: Sample, context: PipelineContext) -> Sample:
    if "question" not in sample.data and "problem" in sample.data:
        sample.data["question"] = sample.data["problem"]
    if "solution" not in sample.data and "response" in sample.data:
        sample.data["solution"] = sample.data["response"]
    if isinstance(sample.data.get("question"), str) and isinstance(sample.data.get("solution"), str):
        sample.text_fields = ("question", "solution")
    return sample


def validate_problem_solution(sample: Sample, context: PipelineContext) -> Sample:
    problem = str(sample.data.get("question") or "")
    solution = str(sample.data.get("solution") or "")
    if not problem or not solution:
        sample.mark(Decision.DROP, "missing_problem_or_solution")
    elif _IMAGE_DEP.search(problem + "\n" + solution) and not sample.data.get("image_text"):
        sample.mark(Decision.REVIEW, "missing_multimodal_context")
    return sample


def validate_math_answer_signal(sample: Sample, context: PipelineContext) -> Sample:
    solution = str(sample.data.get("solution") or "")
    if len(solution) < 32:
        sample.mark(Decision.REVIEW, "solution_too_short")
    if not _ANSWER_SIGNAL.search(solution) and not sample.data.get("answer"):
        sample.mark(Decision.REVIEW, "missing_final_answer_signal")
    verified = sample.data.get("verified")
    sample.buckets["verified"] = "verified" if verified is True else "failed" if verified is False else "unknown"
    if verified is False:
        sample.mark(Decision.REVIEW, "verifier_failed")
    return sample


def bucket_math(sample: Sample, context: PipelineContext) -> Sample:
    sample.buckets["form"] = str(sample.data.get("form") or ("textbook" if sample.data.get("chapter") else "qa"))
    sample.buckets["difficulty"] = str(sample.data.get("difficulty") or "unknown")
    sample.buckets["domain"] = str(sample.data.get("domain") or sample.data.get("subject") or "math")
    return sample


MATH_STEPS = (
    Step("normalize_math_schema", normalize_math_schema, "统一 problem、question、solution 和 answer"),
    Step("validate_problem_solution", validate_problem_solution, "检查题解完整性和缺失多模态上下文"),
    Step("validate_math_answer_signal", validate_math_answer_signal, "检查解题过程、最终答案和 verifier 信号"),
    Step("bucket_math", bucket_math, "按 QA/textbook、学科、难度和可验证性分桶"),
)

