from pathlib import Path

import pymupdf
import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def build_pdf(path, image_fraction=0.0, text=True):
    document = pymupdf.open()
    page = document.new_page()
    if text:
        page.insert_text((60, 90), "lorem ipsum dolor sit amet " * 12, fontsize=9)
    if image_fraction:
        side = (abs(page.rect) * image_fraction) ** 0.5
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
        pixmap.set_rect(pixmap.irect, (30, 30, 30))
        page.insert_image(
            pymupdf.Rect(60, 300, 60 + side, 300 + side), pixmap=pixmap
        )
    document.save(path)
    document.close()
    return path


@pytest.fixture
def synthetic_pdf(tmp_path):
    def build(image_fraction=0.0, text=True):
        name = f"synthetic_{image_fraction}_{text}.pdf"
        return build_pdf(tmp_path / name, image_fraction, text)

    return build


@pytest.fixture(scope="session")
def example_pdf():
    return EXAMPLES / "1.pdf"


@pytest.fixture(scope="session")
def example_image():
    return EXAMPLES / "1.jpg"
