import numpy as np
import cv2
import os
from typing import List

from engine.schemas import DetectedFace, BoundingBox
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Single-responsibility wrapper around OpenCV DNN face detector.
    Uses the SSD-based Caffe model (ResNet-10 backbone) from OpenCV.
    returns a list of DetectedFace objects with bounding boxes and detection scores.
    """

    # Initialize the face detector with a confidence threshold and load the Caffe model.
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
        prototxt = os.path.join(model_dir, "deploy.prototxt")
        caffemodel = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

        if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
            raise FileNotFoundError(
                f"Face detection model files not found in {model_dir}/. "
                f"Expected: deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel"
            )

        self.model = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("FaceDetector initialized (OpenCV DNN, CPU)")

    # Detect faces in a BGR image and return a list of DetectedFace objects.
    def detect(self, frame_bgr: np.ndarray) -> List[DetectedFace]:
        h, w = frame_bgr.shape[:2]

        blob = cv2.dnn.blobFromImage(frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.model.setInput(blob)
        detections = self.model.forward()

        result = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence_threshold:
                continue

            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue

            bbox = BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))

            result.append(DetectedFace(
                bbox=bbox,
                detection_score=float(confidence),
                face_crop=frame_bgr[y1:y2, x1:x2].copy(),
            ))

        result.sort(key=lambda f: f.detection_score, reverse=True)
        return result
