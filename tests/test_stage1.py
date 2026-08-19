from ocrsieve.stage1 import (
    AMBIGUOUS,
    DIGITAL,
    EMPTY,
    SCAN,
    TEXT_WITH_IMAGE,
    PageSignals,
    classify_signals,
)


def signals(chars, cover, image_count=1, invisible=0):
    return PageSignals(
        chars=chars,
        visible_chars=chars,
        invisible_chars=invisible,
        max_image_cover=cover,
        image_count=image_count,
    )


def test_text_without_image_is_free():
    assert classify_signals(signals(500, 0.0, 0), 0.03)[0] == DIGITAL


def test_image_threshold_separates_qr_code_from_content():
    assert classify_signals(signals(500, 0.024), 0.03)[0] == DIGITAL
    assert classify_signals(signals(500, 0.180), 0.03)[0] == TEXT_WITH_IMAGE


def test_zero_threshold_sends_any_image_to_the_model():
    assert classify_signals(signals(500, 0.0, 0), 0.0)[0] == DIGITAL
    assert classify_signals(signals(500, 0.001), 0.0)[0] == TEXT_WITH_IMAGE


def test_full_page_image_without_text_is_a_scan():
    assert classify_signals(signals(0, 0.95, 1), 0.03)[0] == SCAN


def test_empty_page():
    assert classify_signals(signals(0, 0.0, 0), 0.03)[0] == EMPTY


def test_text_over_a_full_page_scan_is_a_scan():
    assert classify_signals(signals(500, 0.85), 0.03)[0] == SCAN


def test_little_text_is_ambiguous():
    assert classify_signals(signals(10, 0.0, 0), 0.03)[0] == AMBIGUOUS
