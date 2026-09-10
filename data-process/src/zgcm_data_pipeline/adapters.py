from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .models import DataCategory, Sample, Stage


Hook = Callable[[Sample], Sample]


@dataclass(frozen=True, slots=True)
class DatasetAdapter:
    """Maps one dataset schema to the canonical sample model.

    Dataset-specific repairs belong in ``hooks``. They run before the shared
    category pipeline and are deliberately excluded from category diagrams.
    """

    dataset: str
    stage: Stage
    category: DataCategory
    id_field: str = "id"
    text_fields: tuple[str, ...] = ("text",)
    field_map: Mapping[str, str] | None = None
    hooks: tuple[Hook, ...] = ()

    def adapt(self, raw: Mapping[str, Any], index: int = 0) -> Sample:
        mapped = dict(raw)
        if self.field_map:
            for source_name, canonical_name in self.field_map.items():
                if source_name in mapped and canonical_name not in mapped:
                    mapped[canonical_name] = mapped[source_name]
        source_id = str(mapped.get(self.id_field) or f"{self.dataset}:{index}")
        sample = Sample(
            dataset=self.dataset,
            source_id=source_id,
            stage=self.stage,
            category=self.category,
            data=mapped,
            text_fields=self.text_fields,
        )
        for hook in self.hooks:
            sample = hook(sample)
        return sample

    def adapt_many(self, rows: Iterable[Mapping[str, Any]]) -> Iterable[Sample]:
        for index, row in enumerate(rows):
            yield self.adapt(row, index=index)

