import pymupdf
import pytest
from PIL import Image

from ocrsieve.sources import IMAGE, PDF, collect_paths, open_source


def test_pdf_path(example_pdf):
    source = open_source(example_pdf)
    assert source.kind == PDF
    assert len(list(source.pages())) == 1
    source.close()


def test_image_path(example_image):
    source = open_source(example_image)
    assert source.kind == IMAGE
    assert len(list(source.pages())) == 1


def test_pdf_bytes(example_pdf):
    source = open_source(example_pdf.read_bytes(), name="memory.pdf")
    assert source.kind == PDF
    assert source.name == "memory.pdf"
    source.close()


def test_image_bytes(example_image):
    assert open_source(example_image.read_bytes()).kind == IMAGE


def test_file_object(example_pdf):
    with open(example_pdf, "rb") as handle:
        source = open_source(handle)
    assert source.kind == PDF
    source.close()


def test_pil_image():
    source = open_source(Image.new("RGB", (32, 32), "white"))
    assert source.kind == IMAGE
    assert len(list(source.pages())) == 1


def test_pymupdf_document(example_pdf):
    document = pymupdf.open(example_pdf)
    source = open_source(document)
    assert source.kind == PDF
    source.close()
    assert not document.is_closed
    document.close()


def test_multipage_tiff(tmp_path):
    path = tmp_path / "pages.tiff"
    frames = [Image.new("L", (16, 16), v) for v in (0, 128, 255)]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    assert len(list(open_source(path).pages())) == 3


def test_suffixless_file_is_sniffed(tmp_path, example_pdf):
    path = tmp_path / "document"
    path.write_bytes(example_pdf.read_bytes())
    source = open_source(path)
    assert source.kind == PDF
    source.close()


def test_unsupported_source():
    with pytest.raises(TypeError):
        open_source(42)


def test_missing_path():
    with pytest.raises(FileNotFoundError):
        open_source("/nonexistent/file.pdf")


def test_collect_paths_walks_a_directory(tmp_path, example_pdf, example_image):
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.pdf").write_bytes(example_pdf.read_bytes())
    (tmp_path / "nested" / "b.JPG").write_bytes(example_image.read_bytes())
    (tmp_path / "notes.txt").write_text("ignore me")
    found = [p.name for p in collect_paths(tmp_path)]
    assert found == ["a.pdf", "b.JPG"]
