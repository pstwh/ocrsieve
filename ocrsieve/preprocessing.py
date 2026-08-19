import numpy as np
from PIL import Image

MEAN = 0.5
STD = 0.25


def preprocess(image, image_size):
    if isinstance(image_size, int):
        target = (image_size, image_size)
    else:
        height, width = image_size
        target = (width, height)
    if image.mode != "L":
        image = image.convert("L")
    if image.size != target:
        image = image.resize(target, Image.BILINEAR)
    plane = np.asarray(image, dtype=np.float32) / 255.0
    plane = (plane - MEAN) / STD
    return np.repeat(plane[None], 3, axis=0)


def build_batch(images, image_size):
    return np.stack([preprocess(i, image_size) for i in images]).astype(
        np.float32
    )


def softmax(logits):
    stable = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(stable)
    return exp / exp.sum(axis=1, keepdims=True)
