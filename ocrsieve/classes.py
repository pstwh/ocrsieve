DIGITAL_TEXT = "digital_text"
BLANK = "blank"
TYPED_CLEAN = "typed_clean"
TYPED_SIGNED = "typed_signed"
TYPED_ANNOTATED = "typed_annotated"
HANDWRITTEN = "handwritten"

MODEL_CLASSES = (
    BLANK,
    TYPED_CLEAN,
    TYPED_SIGNED,
    TYPED_ANNOTATED,
    HANDWRITTEN,
)

NEEDS_OCR = frozenset(
    {TYPED_CLEAN, TYPED_SIGNED, TYPED_ANNOTATED, HANDWRITTEN}
)
NEEDS_STRONG_OCR = frozenset({TYPED_ANNOTATED, HANDWRITTEN})

HANDWRITING_LEVEL = {
    BLANK: 0,
    DIGITAL_TEXT: 0,
    TYPED_CLEAN: 0,
    TYPED_SIGNED: 1,
    TYPED_ANNOTATED: 2,
    HANDWRITTEN: 3,
}

DESCRIPTIONS = {
    DIGITAL_TEXT: "native digital text, extractable from the PDF without OCR",
    BLANK: "page with no content",
    TYPED_CLEAN: "printed or typewritten page with no pen strokes",
    TYPED_SIGNED: (
        "printed page carrying only signatures, initials or stamps; the "
        "handwriting adds no information, so cheap OCR is enough"
    ),
    TYPED_ANNOTATED: (
        "printed page with handwriting that carries information (a filled "
        "date, a value, a name, an amendment); needs OCR that reads "
        "handwriting"
    ),
    HANDWRITTEN: "the body of the text is handwritten, needs a strong model",
}
