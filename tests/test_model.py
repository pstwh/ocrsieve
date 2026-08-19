import json

import pytest
from PIL import Image

from ocrsieve import MODEL_CLASSES, PageClassifier, VisualClassifier
from ocrsieve.classifier import DEFAULT_MODEL, ort
from ocrsieve.cli import main

pytestmark = pytest.mark.skipif(
    ort is None or not DEFAULT_MODEL.exists(),
    reason="onnxruntime and the bundled model are required",
)


@pytest.fixture(scope="module")
def classifier():
    return PageClassifier(threads=1)


def test_bundled_model_metadata():
    model = VisualClassifier(threads=1)
    assert set(model.classes) <= set(MODEL_CLASSES)
    assert 0.0 < model.decision_threshold <= 1.0


def test_predict_returns_labels_and_scores():
    model = VisualClassifier(threads=1)
    images = [Image.new("L", (500, 707), v) for v in (255, 0)]
    for result in model.predict(images):
        assert result["label"] in model.classes
        assert 0.0 <= result["confidence"] <= 1.0
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-3
        assert result["needs_strong_ocr"] == (
            result["score"] >= model.decision_threshold
        )


def test_classify_image_file(classifier, example_image):
    result = classifier.classify(example_image)
    assert result.kind == "image"
    page = result.pages[0]
    assert page.label in MODEL_CLASSES
    assert page.from_model
    assert page.confidence is not None


def test_classify_pil_image(classifier):
    page = classifier.classify(Image.new("L", (500, 707), 255)).pages[0]
    assert page.label in MODEL_CLASSES


def test_classify_pdf_uses_stage1(classifier, example_pdf):
    page = classifier.classify(example_pdf).pages[0]
    assert page.label == "digital_text"
    assert page.decided_by == "stage1"


def test_cli_on_an_image(capsys, example_image):
    assert main([str(example_image), "--json", "--quiet"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["kind"] == "image"
    assert payload[0]["pages"][0]["label"] in MODEL_CLASSES


def test_cli_writes_a_report(tmp_path, example_pdf):
    out = tmp_path / "manifest.json"
    assert main([str(example_pdf), "-o", str(out), "--quiet"]) == 0
    assert json.loads(out.read_text())[0]["pages"]


def test_cli_inspect(capsys, example_pdf):
    assert main([str(example_pdf), "--inspect", "--no-model"]) == 0
    assert "digital_text" in capsys.readouterr().out


def test_cli_without_input(capsys, tmp_path):
    assert main([str(tmp_path)]) == 1
