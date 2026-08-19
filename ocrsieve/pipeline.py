from dataclasses import asdict, dataclass, field

from .classes import (
    BLANK,
    DIGITAL_TEXT,
    NEEDS_OCR,
    NEEDS_STRONG_OCR,
    TYPED_SIGNED,
)
from .classifier import load_classifier
from .rendering import THUMB_WIDTH, render_to_pil
from .sources import IMAGE, PDF, collect_paths, open_source
from .stage1 import (
    DEFAULT_MAX_IMAGE_COVER,
    DIGITAL,
    EMPTY,
    analyze_page,
)

DECIDED_BY_STAGE1 = "stage1"
DECIDED_BY_MODEL = "model"
DECIDED_BY_FALLBACK = "fallback"


@dataclass
class PageSignalsView:
    page: int
    chars: int
    image_count: int
    max_image_cover: float
    has_native_text: bool
    has_invisible_text_layer: bool
    stage1: str
    reason: str

    def to_dict(self):
        return asdict(self)


@dataclass
class PageResult:
    page: int
    label: str
    reason: str
    needs_ocr: bool = False
    needs_strong_ocr: bool = False
    decided_by: str = DECIDED_BY_STAGE1
    confidence: float = None
    score: float = None
    probabilities: dict = None
    signals: dict = field(default_factory=dict)

    @property
    def from_model(self):
        return self.decided_by == DECIDED_BY_MODEL

    def to_dict(self):
        return asdict(self)


@dataclass
class DocumentResult:
    file: str
    pages: list
    kind: str = PDF

    @property
    def mixed(self):
        return len({p.label for p in self.pages}) > 1

    @property
    def labels(self):
        return [p.label for p in self.pages]

    def pages_needing_ocr(self):
        return [p for p in self.pages if p.needs_ocr]

    def pages_needing_strong_ocr(self):
        return [p for p in self.pages if p.needs_strong_ocr]

    def to_dict(self):
        return {
            "file": self.file,
            "kind": self.kind,
            "mixed": self.mixed,
            "pages": [p.to_dict() for p in self.pages],
        }


class PageClassifier:
    def __init__(
        self,
        model_path=None,
        use_model=True,
        max_image_cover=DEFAULT_MAX_IMAGE_COVER,
        render_width=THUMB_WIDTH,
        threads=None,
        providers=None,
    ):
        self.model = (
            load_classifier(
                model_path, threads, tuple(providers) if providers else None
            )
            if use_model
            else None
        )
        self.max_image_cover = max_image_cover
        self.render_width = render_width

    @property
    def has_model(self):
        return self.model is not None

    def predict_images(self, images):
        if self.model is None:
            raise RuntimeError(
                "no visual model loaded; build the classifier with "
                "use_model=True"
            )
        return self.model.predict(images)

    def predict_image(self, image):
        return self.predict_images([image])[0]

    def classify_image(self, image, number=1):
        prediction = self.predict_image(image)
        return self._from_prediction(
            number,
            prediction,
            {
                "source": IMAGE,
                "width": image.width,
                "height": image.height,
            },
        )

    def classify_page(self, page, number=1):
        stage1, signals = analyze_page(page, self.max_image_cover)
        raw = {
            "chars": signals.chars,
            "image_count": signals.image_count,
            "max_image_cover": signals.max_image_cover,
            "has_native_text": signals.has_text and not signals.invisible_layer,
            "has_invisible_text_layer": signals.invisible_layer,
            "stage1": stage1,
        }

        if stage1 == DIGITAL:
            return self._result(number, DIGITAL_TEXT, signals.reason, raw)
        if stage1 == EMPTY:
            return self._result(number, BLANK, signals.reason, raw)
        if self.model is None:
            return self._result(
                number,
                TYPED_SIGNED,
                "no visual model loaded, routed to OCR to be safe",
                raw,
                decided_by=DECIDED_BY_FALLBACK,
                score=1.0,
            )

        prediction = self.predict_image(
            render_to_pil(page, self.render_width)
        )
        return self._from_prediction(number, prediction, raw)

    def inspect_page(self, page, number=1):
        stage1, signals = analyze_page(page, self.max_image_cover)
        return PageSignalsView(
            page=number,
            chars=signals.chars,
            image_count=signals.image_count,
            max_image_cover=signals.max_image_cover,
            has_native_text=signals.has_text and not signals.invisible_layer,
            has_invisible_text_layer=signals.invisible_layer,
            stage1=stage1,
            reason=signals.reason,
        )

    def classify(self, source, name=None):
        opened = open_source(source, name)
        try:
            if opened.kind == PDF:
                pages = [
                    self.classify_page(page, i + 1)
                    for i, page in enumerate(opened.pages())
                ]
            else:
                pages = [
                    self.classify_image(image, i + 1)
                    for i, image in enumerate(opened.pages())
                ]
        finally:
            opened.close()
        return DocumentResult(opened.name, pages, opened.kind)

    def inspect(self, source, name=None):
        opened = open_source(source, name)
        if opened.kind != PDF:
            raise TypeError(
                "inspect reads the PDF text layer; images have no signals "
                "to read"
            )
        try:
            return [
                self.inspect_page(page, i + 1)
                for i, page in enumerate(opened.pages())
            ]
        finally:
            opened.close()

    def classify_dir(self, root):
        return [self.classify(path) for path in collect_paths(root)]

    def _from_prediction(self, number, prediction, signals):
        signals = dict(signals, probabilities=prediction["probabilities"])
        return self._result(
            number,
            prediction["label"],
            f"model: {prediction['label']} (score {prediction['score']})",
            signals,
            decided_by=DECIDED_BY_MODEL,
            confidence=prediction["confidence"],
            score=prediction["score"],
            probabilities=prediction["probabilities"],
            needs_strong_ocr=prediction["needs_strong_ocr"],
        )

    def _result(
        self,
        number,
        label,
        reason,
        signals,
        decided_by=DECIDED_BY_STAGE1,
        confidence=None,
        score=None,
        probabilities=None,
        needs_strong_ocr=None,
    ):
        if needs_strong_ocr is None:
            needs_strong_ocr = label in NEEDS_STRONG_OCR
        return PageResult(
            page=number,
            label=label,
            reason=reason,
            needs_ocr=label in NEEDS_OCR,
            needs_strong_ocr=needs_strong_ocr,
            decided_by=decided_by,
            confidence=confidence,
            score=score,
            probabilities=probabilities,
            signals=signals,
        )


def classify(source, model_path=None, **kwargs):
    return PageClassifier(model_path, **kwargs).classify(source)


def inspect(source, **kwargs):
    return PageClassifier(use_model=False, **kwargs).inspect(source)
