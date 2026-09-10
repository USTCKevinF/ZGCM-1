from __future__ import annotations

from .core import Pipeline
from .models import DataCategory
from .steps.agentic import AGENTIC_STEPS
from .steps.code import CODE_STEPS
from .steps.common import COMMON_PREFIX, COMMON_SUFFIX
from .steps.instruction import INSTRUCTION_STEPS
from .steps.math import MATH_STEPS
from .steps.reasoning import REASONING_STEPS
from .steps.web import WEB_STEPS


CATEGORY_STEPS = {
    DataCategory.CODE: CODE_STEPS,
    DataCategory.WEB: WEB_STEPS,
    DataCategory.AGENTIC: AGENTIC_STEPS,
    DataCategory.INSTRUCTION: INSTRUCTION_STEPS,
    DataCategory.MATH: MATH_STEPS,
    DataCategory.REASONING: REASONING_STEPS,
    DataCategory.PDF_OCR: (),
    DataCategory.GENERAL_TEXT: (),
}


def build_pipeline(category: DataCategory | str) -> Pipeline:
    category = DataCategory(category)
    category_steps = CATEGORY_STEPS[category]
    return Pipeline(
        name=f"{category.value}_cleaning_pipeline",
        # First map raw dataset fields into the category's canonical schema.
        # Common text/integrity checks can then safely operate on canonical fields.
        steps=category_steps[:1] + COMMON_PREFIX + category_steps[1:] + COMMON_SUFFIX,
    )
