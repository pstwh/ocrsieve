from io import BytesIO

import pymupdf
from PIL import Image

THUMB_WIDTH = 500


def render_pixmap(page, width=THUMB_WIDTH, gray=False):
    zoom = width / page.rect.width
    return page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom),
        colorspace=pymupdf.csGRAY if gray else pymupdf.csRGB,
    )


def render_to_bytes(page, width=THUMB_WIDTH, gray=False):
    return render_pixmap(page, width, gray).tobytes("jpg", jpg_quality=85)


def render_to_pil(page, width=THUMB_WIDTH, gray=False):
    return Image.open(BytesIO(render_to_bytes(page, width, gray)))
