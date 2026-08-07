import cv2
import numpy as np
from typing import Optional

from engine.schemas import DetectedFace, AlignedFace
import logging

logger = logging.getLogger(__name__)

_OUTPUT_SIZE = (224, 224)

class FaceAligner:
    """
    Aligns a detected face to 224x224 for the gender classifier.

    Uses the face_crop from the detector and resizes it to 224x224
    while preserving aspect ratio (padding as needed).
    Falls back to similarity transform alignment if face_crop is unavailable.
    """

    def __init__(self):
        pass

    def align(
        self,
        detected_face: DetectedFace,
        original_frame_bgr: np.ndarray,
    ) -> Optional[AlignedFace]:
        if detected_face.face_crop is not None and detected_face.face_crop.size > 0:
            aligned_bgr = self._resize_with_padding(detected_face.face_crop)
        else:
            aligned_bgr = self._crop_from_original(detected_face, original_frame_bgr)

        if aligned_bgr is None:
            return None

        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        return AlignedFace(
            image=aligned_rgb,
            shape=(224, 224, 3),
            source_bbox=detected_face.bbox,
            transform_matrix=None,
        )

    @staticmethod
    def _resize_with_padding(face_crop: np.ndarray) -> Optional[np.ndarray]:
        h, w = face_crop.shape[:2]
        if h <= 0 or w <= 0:
            return None
        size = max(h, w)
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left
        squared = cv2.copyMakeBorder(face_crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return cv2.resize(squared, _OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _crop_from_original(
        detected_face: DetectedFace,
        original_frame_bgr: np.ndarray,
    ) -> Optional[np.ndarray]:
        x1 = max(0, int(detected_face.bbox.x1))
        y1 = max(0, int(detected_face.bbox.y1))
        x2 = min(original_frame_bgr.shape[1], int(detected_face.bbox.x2))
        y2 = min(original_frame_bgr.shape[0], int(detected_face.bbox.y2))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return None
        crop = original_frame_bgr[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        size = max(h, w)
        top = (size - h) // 2
        bottom = size - h - top
        left = (size - w) // 2
        right = size - w - left
        squared = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return cv2.resize(squared, _OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
