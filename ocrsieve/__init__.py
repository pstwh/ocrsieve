from .classes import (
    BLANK,
    DESCRIPTIONS,
    DIGITAL_TEXT,
    HANDWRITING_LEVEL,
    HANDWRITTEN,
    MODEL_CLASSES,
    NEEDS_OCR,
    NEEDS_STRONG_OCR,
    TYPED_ANNOTATED,
    TYPED_CLEAN,
    TYPED_SIGNED,
)
from .classifier import VisualClassifier, available_providers, load_classifier
from .pipeline import (
    DocumentResult,
    PageClassifier,
    PageResult,
    PageSignalsView,
    classify,
    inspect,
)
from .sources import open_source
from .stage1 import analyze_page, page_signals

__all__ = [
    "DIGITAL_TEXT",
    "BLANK",
    "TYPED_CLEAN",
    "TYPED_SIGNED",
    "TYPED_ANNOTATED",
    "HANDWRITTEN",
    "MODEL_CLASSES",
    "NEEDS_OCR",
    "NEEDS_STRONG_OCR",
    "HANDWRITING_LEVEL",
    "DESCRIPTIONS",
    "PageClassifier",
    "PageResult",
    "PageSignalsView",
    "DocumentResult",
    "VisualClassifier",
    "available_providers",
    "load_classifier",
    "open_source",
    "classify",
    "inspect",
    "analyze_page",
    "page_signals",
]
__version__ = "1.0.1"
