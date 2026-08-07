"""
engine/schemas.py
─────────────────
Single source of truth for all data contracts in the pipeline.
Every inter-stage transfer uses a type defined here.
No raw dicts or bare numpy arrays cross stage boundaries.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import numpy as np

class GenderLabel(str, Enum):
    FEMALE  = "female"
    MALE    = "male"
    NO_FACE = "no_face"

@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def width(self) -> float:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0, self.y2 - self.y1)

@dataclass
class RawFrame:
    data:      np.ndarray   # HxWx3, BGR, uint8
    frame_idx: int
    timestamp: Optional[float] = None  # seconds since stream start

@dataclass
class QualifiedFrame:
    data:      np.ndarray   # HxWx3, BGR, uint8
    frame_idx: int
    phash:     Optional[np.ndarray] = None

@dataclass
class DetectedFace:
    bbox:            BoundingBox
    detection_score: float
    face_crop:       Optional[np.ndarray] = None  # set by FaceDetector after detection

@dataclass
class AlignedFace:
    image:            np.ndarray       # 224x224x3, RGB, float32, [0,1]
    shape:            Tuple[int,...]   # (224, 224, 3)
    source_bbox:      Optional[BoundingBox] = None
    transform_matrix: Optional[np.ndarray] = None  # 2x3 affine transform

@dataclass
class RawPrediction:
    label:            GenderLabel
    confidence:       float
    logits:           np.ndarray      # shape (2,)
    probabilities:    np.ndarray      # shape (2,) — [P(female), P(male)]
    is_low_confidence: bool = False
    aligned_face:     Optional[AlignedFace] = None # Added for smoother access

@dataclass
class GenderResult:
    """
    Public output type of GenderDetectionComponent.
    This is the only type that external components should consume.
    """
    label:      GenderLabel             # "female" | "male" | "no_face"
    confidence: float                   # [0.0, 1.0]
    face_bbox:  Optional[BoundingBox]   # None if no face detected
    frame_idx:  int
    source:     str                     # "inference" | "cache" | "skipped" | "no_face"
    is_smoothed: bool = False
    detection_score: Optional[float] = None  # face detection confidence from OpenCV DNN

    def to_dict(self) -> dict:
        """Serializable dict for message bus / logging."""
        return {
            "label":       self.label.value,
            "confidence":  round(self.confidence, 4),
            "face_bbox":   {
                "x1": self.face_bbox.x1, "y1": self.face_bbox.y1,
                "x2": self.face_bbox.x2, "y2": self.face_bbox.y2,
            } if self.face_bbox else None,
            "frame_idx":   self.frame_idx,
            "source":      self.source,
            "is_smoothed": self.is_smoothed,
            "detection_score": round(self.detection_score, 4) if self.detection_score else None,
        }
