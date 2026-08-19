from dataclasses import dataclass, field

import pymupdf

OCR_FONT_KEYWORDS = ("glyphless", "invisible", "ocr")

MIN_CHARS_FOR_TEXT = 50
DEFAULT_MAX_IMAGE_COVER = 0.03
SCAN_MIN_IMAGE_COVER = 0.5
OVERLAY_MIN_IMAGE_COVER = 0.7
EMBEDDED_INVISIBLE_RATIO = 5

DIGITAL = "digital_text"
SCAN = "scan"
TEXT_WITH_IMAGE = "text_with_image"
EMPTY = "blank"
AMBIGUOUS = "ambiguous"


@dataclass
class PageSignals:
    chars: int = 0
    invisible_chars: int = 0
    visible_chars: int = 0
    ocr_font: bool = False
    max_image_cover: float = 0.0
    image_count: int = 0
    reason: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def has_text(self):
        return self.chars >= MIN_CHARS_FOR_TEXT

    @property
    def invisible_layer(self):
        if self.ocr_font:
            return True
        return self.invisible_chars > 0 and (
            self.invisible_chars
            >= EMBEDDED_INVISIBLE_RATIO * max(self.visible_chars, 1)
        )


def page_signals(page):
    text = page.get_text().strip()
    area = abs(page.rect)
    signals = PageSignals(chars=len(text))
    try:
        for span in page.get_texttrace():
            n = len(span.get("chars", []))
            if span.get("type") == 3 or span.get("opacity") == 0:
                signals.invisible_chars += n
            else:
                signals.visible_chars += n
            font = (span.get("font") or "").lower()
            if any(k in font for k in OCR_FONT_KEYWORDS):
                signals.ocr_font = True
    except Exception:
        pass
    try:
        for info in page.get_image_info():
            signals.image_count += 1
            bbox = pymupdf.Rect(info["bbox"]) & page.rect
            cover = abs(bbox) / area if area else 0.0
            signals.max_image_cover = max(signals.max_image_cover, cover)
    except Exception:
        pass
    signals.max_image_cover = round(signals.max_image_cover, 3)
    return signals


def classify_signals(signals, max_image_cover=DEFAULT_MAX_IMAGE_COVER):
    if signals.has_text and signals.invisible_layer:
        if signals.max_image_cover >= SCAN_MIN_IMAGE_COVER:
            return SCAN, "embedded OCR layer over a full-page image"
        return AMBIGUOUS, "invisible text layer without a large image"
    if signals.has_text:
        if signals.max_image_cover <= max_image_cover:
            return DIGITAL, "visible text and little image coverage"
        if signals.max_image_cover >= OVERLAY_MIN_IMAGE_COVER:
            return SCAN, "text overlaid on a full-page scan"
        return (
            TEXT_WITH_IMAGE,
            f"native text with an image region ({signals.max_image_cover})",
        )
    if signals.chars == 0:
        if signals.max_image_cover >= SCAN_MIN_IMAGE_COVER:
            return SCAN, "no text, image covering the page"
        if signals.image_count == 0:
            return EMPTY, "no text and no image"
        return AMBIGUOUS, "no text, small image"
    return AMBIGUOUS, "too little text to decide"


def analyze_page(page, max_image_cover=DEFAULT_MAX_IMAGE_COVER):
    signals = page_signals(page)
    outcome, reason = classify_signals(signals, max_image_cover)
    signals.reason = reason
    return outcome, signals
