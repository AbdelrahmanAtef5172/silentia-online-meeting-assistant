"""
Gender detection component — pipeline stages and utilities.
"""

import importlib

from .schemas import (
    GenderLabel,
    BoundingBox,
    RawFrame,
    QualifiedFrame,
    DetectedFace,
    AlignedFace,
    RawPrediction,
    GenderResult,
)
from .device import get_device
from .transforms import get_inference_transforms, get_training_transforms

_LAZY = {
    "GenderDetectionComponent": ".component",
    "FrameGate": ".frame_gate",
    "FaceDetector": ".face_detector",
    "FaceAligner": ".face_aligner",
    "GenderInference": ".gender_classifier",
    "ResultSmoother": ".smoother",
}

def __getattr__(name):
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name], __package__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
