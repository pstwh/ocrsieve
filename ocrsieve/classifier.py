import json
import os
from functools import lru_cache
from pathlib import Path

from .classes import NEEDS_STRONG_OCR
from .preprocessing import build_batch, softmax

try:
    import onnxruntime as ort
except ImportError:
    ort = None

DEFAULT_THRESHOLD = 0.5
DEFAULT_TEMPERATURE = 1.0
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DEFAULT_MODEL = ARTIFACTS_DIR / "model.onnx"
THREADS_ENV_VAR = "OCRSIEVE_THREADS"
MODEL_ENV_VAR = "OCRSIEVE_MODEL"

MISSING_RUNTIME = (
    "onnxruntime is not installed: the visual model cannot run. "
    "Install ocrsieve[cpu] for CPU inference or ocrsieve[gpu] for CUDA."
)


def available_providers():
    return list(ort.get_available_providers()) if ort else []


def _build_session(model_path, threads, providers):
    if ort is None:
        raise ImportError(MISSING_RUNTIME)
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if threads:
        options.intra_op_num_threads = threads
    if providers is None:
        available = ort.get_available_providers()
        providers = [
            p
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in available
        ]
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=providers
    )


class VisualClassifier:
    def __init__(self, model_path=None, threads=None, providers=None):
        self.model_path = Path(model_path or default_model_path())
        if not self.model_path.exists():
            raise FileNotFoundError(f"no model at {self.model_path}")
        self.session = _build_session(self.model_path, threads, providers)
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.classes = json.loads(metadata["classes"])
        if "input_height" in metadata:
            self.image_size = (
                int(metadata["input_height"]),
                int(metadata["input_width"]),
            )
        else:
            self.image_size = int(metadata.get("image_size", 384))
        self.arch = metadata.get("arch", "efficientnet_b0")
        self.temperature = float(
            metadata.get("temperature", DEFAULT_TEMPERATURE)
        )
        self.decision_threshold = float(
            metadata.get("decision_threshold", DEFAULT_THRESHOLD)
        )
        self.input_name = self.session.get_inputs()[0].name
        self.strong_indices = [
            i for i, name in enumerate(self.classes) if name in NEEDS_STRONG_OCR
        ]

    @property
    def providers(self):
        return self.session.get_providers()

    def predict(self, images, threshold=None):
        if not isinstance(images, (list, tuple)):
            images = [images]
        if not images:
            return []
        cut = self.decision_threshold if threshold is None else threshold
        batch = build_batch(images, self.image_size)
        logits = self.session.run(None, {self.input_name: batch})[0]
        probabilities = softmax(logits / self.temperature)
        results = []
        for row in probabilities:
            index = int(row.argmax())
            score = float(sum(row[i] for i in self.strong_indices))
            results.append(
                {
                    "label": self.classes[index],
                    "confidence": round(float(row[index]), 4),
                    "score": round(score, 4),
                    "needs_strong_ocr": score >= cut,
                    "probabilities": {
                        name: round(float(value), 4)
                        for name, value in zip(self.classes, row)
                    },
                }
            )
        return results


def default_model_path():
    return Path(os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL)


@lru_cache(maxsize=4)
def load_classifier(model_path=None, threads=None, providers=None):
    if threads is None and os.environ.get(THREADS_ENV_VAR):
        threads = int(os.environ[THREADS_ENV_VAR])
    return VisualClassifier(model_path, threads, providers)
