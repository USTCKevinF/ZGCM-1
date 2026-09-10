from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

from .models import Decision, Sample


StepFn = Callable[[Sample, "PipelineContext"], Sample]


@dataclass(slots=True)
class PipelineContext:
    """Mutable state shared by streaming steps in one pipeline run."""

    config: dict[str, object] = field(default_factory=dict)
    counters: Counter[str] = field(default_factory=Counter)
    seen_hashes: set[str] = field(default_factory=set)
    near_dedup_state: object | None = None

    def count(self, name: str, amount: int = 1) -> None:
        self.counters[name] += amount


@dataclass(frozen=True, slots=True)
class Step:
    name: str
    fn: StepFn
    description: str

    def __call__(self, sample: Sample, context: PipelineContext) -> Sample:
        before = sample.decision
        sample = self.fn(sample, context)
        context.count(f"step.{self.name}.seen")
        if sample.decision != before:
            context.count(f"step.{self.name}.{sample.decision.value}")
        return sample


@dataclass(slots=True)
class Pipeline:
    name: str
    steps: tuple[Step, ...]

    def run(
        self,
        samples: Iterable[Sample],
        context: PipelineContext | None = None,
        *,
        include_dropped: bool = True,
    ) -> Iterator[Sample]:
        context = context or PipelineContext()
        for sample in samples:
            context.count("input")
            for step in self.steps:
                sample = step(sample, context)
                if sample.decision == Decision.DROP:
                    break
            context.count(f"output.{sample.decision.value}")
            if include_dropped or sample.decision != Decision.DROP:
                yield sample

    def step_names(self) -> list[str]:
        return [step.name for step in self.steps]
