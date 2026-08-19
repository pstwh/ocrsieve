import os
from io import BytesIO
from pathlib import Path

import pymupdf
from PIL import Image, ImageSequence

PDF_SUFFIXES = frozenset({".pdf"})
IMAGE_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".jpe",
        ".jp2",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
        ".gif",
        ".ppm",
        ".pgm",
        ".pnm",
    }
)
SUPPORTED_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES

PDF_MAGIC = b"%PDF"

PDF = "pdf"
IMAGE = "image"


class PdfSource:
    kind = PDF

    def __init__(self, document, name, owned):
        self.document = document
        self.name = name
        self.owned = owned

    def pages(self):
        return iter(self.document)

    def close(self):
        if self.owned:
            self.document.close()


class ImageSource:
    kind = IMAGE

    def __init__(self, images, name):
        self.images = images
        self.name = name

    def pages(self):
        return iter(self.images)

    def close(self):
        pass


def _frames(image):
    return [frame.copy() for frame in ImageSequence.Iterator(image)]


def _from_bytes(payload, name):
    if payload[:4] == PDF_MAGIC:
        return PdfSource(
            pymupdf.open(stream=payload, filetype="pdf"), name, owned=True
        )
    return ImageSource(_frames(Image.open(BytesIO(payload))), name)


def _from_path(path):
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return PdfSource(pymupdf.open(path), str(path), owned=True)
    if suffix in IMAGE_SUFFIXES:
        return ImageSource(_frames(Image.open(path)), str(path))
    return _from_bytes(path.read_bytes(), str(path))


def open_source(source, name=None):
    if isinstance(source, (PdfSource, ImageSource)):
        return source
    if isinstance(source, pymupdf.Document):
        return PdfSource(source, name or source.name or "<document>", False)
    if isinstance(source, Image.Image):
        return ImageSource([source], name or getattr(source, "filename", None) or "<image>")
    if isinstance(source, (bytes, bytearray, memoryview)):
        return _from_bytes(bytes(source), name or "<bytes>")
    if hasattr(source, "read"):
        return _from_bytes(
            source.read(), name or getattr(source, "name", None) or "<stream>"
        )
    if isinstance(source, (str, Path, os.PathLike)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(str(path))
        return _from_path(path)
    raise TypeError(
        "unsupported source "
        f"{type(source).__name__}: pass a path, bytes, a file object, "
        "a PIL.Image or a pymupdf.Document"
    )


def collect_paths(root, suffixes=SUPPORTED_SUFFIXES):
    path = Path(root)
    if path.is_file():
        return [path]
    return sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    )
