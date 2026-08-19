import json

import pytest

from ocrsieve import (
    DIGITAL_TEXT,
    HANDWRITTEN,
    NEEDS_OCR,
    TYPED_SIGNED,
    TYPED_ANNOTATED,
    PageClassifier,
)
from ocrsieve.pipeline import DECIDED_BY_FALLBACK, DECIDED_BY_STAGE1


@pytest.fixture(scope="module")
def stage1_only():
    return PageClassifier(use_model=False)


def test_routing_table_is_coherent():
    assert DIGITAL_TEXT not in NEEDS_OCR
    for label in (TYPED_SIGNED, TYPED_ANNOTATED, HANDWRITTEN):
        assert label in NEEDS_OCR


def test_small_image_stays_on_the_free_path(synthetic_pdf, stage1_only):
    page = stage1_only.classify(synthetic_pdf(0.01)).pages[0]
    assert page.label == DIGITAL_TEXT
    assert not page.needs_ocr
    assert page.decided_by == DECIDED_BY_STAGE1
    assert page.confidence is None


def test_content_image_leaves_the_free_path(synthetic_pdf, stage1_only):
    page = stage1_only.classify(synthetic_pdf(0.20)).pages[0]
    assert page.label != DIGITAL_TEXT
    assert page.needs_ocr, (
        "a page with native text and a large image must leave the zero-cost "
        "path, otherwise the content inside the image is lost silently"
    )
    assert page.decided_by == DECIDED_BY_FALLBACK


def test_empty_page_is_blank(synthetic_pdf, stage1_only):
    page = stage1_only.classify(synthetic_pdf(text=False)).pages[0]
    assert not page.needs_ocr


def test_result_is_json_serializable(synthetic_pdf, stage1_only):
    payload = json.loads(
        json.dumps(stage1_only.classify(synthetic_pdf(0.01)).to_dict())
    )
    assert payload["pages"][0]["page"] == 1
    assert payload["kind"] == "pdf"
    assert "decided_by" in payload["pages"][0]


def test_inspect_reports_signals_without_a_model(synthetic_pdf, stage1_only):
    views = stage1_only.inspect(synthetic_pdf(0.20))
    assert views
    for view in views:
        assert view.stage1
        assert view.max_image_cover >= 0


def test_predict_requires_a_model(stage1_only):
    with pytest.raises(RuntimeError):
        stage1_only.predict_images([])


def test_digital_text_pages_respect_the_image_threshold(
    synthetic_pdf, stage1_only
):
    result = stage1_only.classify(synthetic_pdf(0.01))
    for page in result.pages:
        if page.label == DIGITAL_TEXT:
            assert (
                page.signals["max_image_cover"] <= stage1_only.max_image_cover
            )
