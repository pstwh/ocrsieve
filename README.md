# ocrsieve

A sieve for document pages. For every page it answers which extraction path the
page needs: what comes out for free, what needs OCR, and what needs a strong
model.

| Class | Meaning | Extraction |
|---|---|---|
| `digital_text` | native digital text in the PDF layer | direct read, zero cost, no OCR |
| `blank` | no content | skip |
| `typed_clean` | printed or typewritten, no pen strokes at all | cheap OCR |
| `typed_signed` | printed plus signatures, initials or stamps only | cheap OCR |
| `typed_annotated` | printed plus handwriting that carries information | OCR that reads handwriting |
| `handwritten` | the body of the text is handwritten | strong model |

The five model classes sit on one axis - how much handwritten information the
page carries - and the routing decision is a single number read off it:

```
score = p(typed_annotated) + p(handwritten)
```

Compare it with a threshold and you have the decision. Uncertainty that does
not change the decision (clean versus signed) is marginalised away, and
uncertainty that does change it pushes the score up, which is the safe side.

`typed_clean` and `typed_signed` route to the same place on purpose. Keeping
them apart splits one contradictory training signal - "a pen stroke is
irrelevant here, decisive next door" - into two learnable tasks: detecting that
a stroke exists, and judging whether it carries content.

`digital_text` is never a prediction: it is read from the file.

## Install

```bash
pip install ocrsieve[cpu]      # inference on CPU (onnxruntime)
pip install ocrsieve[gpu]      # inference on CUDA (onnxruntime-gpu)
```

The base package pulls only `pymupdf`, `pillow` and `numpy`, and the extra
picks the ONNX Runtime build - the two runtimes conflict, so exactly one of
them has to be chosen. Without either, the deterministic stage still works and
the visual model raises a clear `ImportError`.

The model ships inside the wheel, at `ocrsieve/artifacts/model.onnx` (16 MB).
There is nothing to download at first run.

## CLI

```bash
ocrsieve examples/1.jpg
ocrsieve examples/1.pdf
ocrsieve ./folder -o manifest.json       # walks the folder, writes JSON
ocrsieve a.pdf b.png ./folder --json     # JSON to stdout
ocrsieve ./folder --no-model             # deterministic stage only
ocrsieve ./folder --inspect              # per-page PDF signals, no model
ocrsieve ./folder --max-image-cover 0    # any image goes to the model
ocrsieve ./folder --threads 8
```

```
$ ocrsieve examples/1.jpg
examples/1.jpg
     1  typed_clean      0.55  model     model: typed_clean (score 0.0429)

1 pages in 1 files (0 mixed)
  typed_clean                   1 100.0%
  decided by: {'model': 1}
```

## Python

`classify` takes anything: a path to a PDF or an image, a directory entry, raw
bytes, an open file, a `PIL.Image` or a `pymupdf.Document`. Multi-page TIFFs
and PDFs come back as several pages.

```python
from ocrsieve import PageClassifier

sieve = PageClassifier()

result = sieve.classify("document.pdf")
result = sieve.classify("scan.jpg")
result = sieve.classify(pdf_bytes, name="upload.pdf")
result = sieve.classify(Image.open("page.png"))
result = sieve.classify(open("document.pdf", "rb"))

for page in result.pages:
    print(page.page, page.label, page.needs_ocr, page.decided_by, page.confidence)

result.mixed                      # pages of different classes in one file
result.pages_needing_ocr()
result.pages_needing_strong_ocr()
result.to_dict()                  # JSON-serializable
```

`confidence`, `score` and `probabilities` are filled in whenever the model
decided, and `decided_by` says who decided: `stage1`, `model` or `fallback`.

One-shot helpers, for when the classifier does not need to be kept around:

```python
from ocrsieve import classify, inspect

classify("document.pdf")
inspect("document.pdf")           # deterministic stage only, no model loaded
```

**Deterministic stage only** - per-page characteristics, no ONNX call:

```python
from ocrsieve import inspect

for view in inspect("document.pdf"):
    print(view.page, view.stage1, view.chars, view.max_image_cover,
          view.has_native_text, view.has_invisible_text_layer, view.reason)
```

**Model only** - classify images directly, bypassing the PDF logic:

```python
from ocrsieve import VisualClassifier

model = VisualClassifier()
model.predict([image])
# [{'label': 'typed_clean', 'confidence': 0.91, 'score': 0.04,
#   'needs_strong_ocr': False, 'probabilities': {...}}]
```

Options:

```python
PageClassifier(
    model_path=None,        # a different .onnx
    use_model=True,         # False = deterministic stage only
    max_image_cover=0.03,   # 0 sends any page containing an image to the model
    render_width=500,       # must match how the model was trained
    threads=None,           # ONNX Runtime cores
    providers=None,         # ONNX Runtime providers, CUDA first when available
)
```

`ocrsieve.available_providers()` reports what the installed runtime offers, so
a `[gpu]` install can be checked in one line.

## Tests

```bash
pip install -e ".[dev]"
pytest
```