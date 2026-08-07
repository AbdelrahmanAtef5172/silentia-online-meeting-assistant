"""
engine/component.py
───────────────────
The single public entry point for the Gender Detection Component.

Usage:
    from engine.component import GenderDetectionComponent

    component = GenderDetectionComponent.from_config("configs/config.yaml")
    result = component.process_frame(frame_bgr=frame, frame_idx=idx)
    print(result.label.value)   # "male" | "female" | "no_face"
"""

import os
from dataclasses import replace
import numpy as np
from typing import List, Optional

from engine.schemas import RawFrame, GenderResult, GenderLabel
from engine.config import load_config
from engine.frame_gate import FrameGate, FrameDisposition
from engine.face_detector import FaceDetector
from engine.face_aligner import FaceAligner
from engine.gender_classifier import GenderInference
from engine.smoother import ResultSmoother
import logging

logger = logging.getLogger(__name__)

_BASE_NO_FACE = GenderResult(
    label=GenderLabel.NO_FACE, confidence=0.0,
    face_bbox=None, frame_idx=-1, source="no_face",
)

class GenderDetectionComponent:
    """
    Top-level orchestrator for the gender detection pipeline.
    initializes and manages the frame gate, face detector, face aligner, gender classifier, and result smoother.
    """

    def __init__(self, config: dict):
        self._cfg = config
        dev = config.get("device", {}).get("preferred", "auto")

        # Adjust weight paths to be absolute or relative to the component root
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        gc_weights = os.path.join(root, "weights", config["model_paths"]["gender_classifier_weights"])

        # STEP1: initialize frame gate component
        self._gate = FrameGate(
            stride=config["frame_gate"]["stride"],
            phash_size=config["frame_gate"]["phash_size"],
            cache_hamming_threshold=config["frame_gate"]["cache_hamming_threshold"],
            cache_resize_to=tuple(config["frame_gate"]["cache_resize_to"]),
        )
        # STEP2: initialize face detector
        self._detector = FaceDetector(
            confidence_threshold=config["face_detection"]["confidence_threshold"],
        )
        # STEP3: initialize face aligner
        self._aligner = FaceAligner()
        # STEP4: initialize gender classifier 
        self._classifier = GenderInference(
            weights_path=gc_weights,
            device=dev,
            use_fp16=config["inference"].get("use_fp16", False),
        )
        # STEP5: initialize result smoother
        self._smoother = ResultSmoother(
            window_size=config["result_smoother"]["window_size"],
            min_confidence_to_include=config["result_smoother"]["min_confidence_to_include"],
        )
        # If configured, perform a warmup pass to initialize the model and cache.
        if config["inference"].get("warmup_on_init", True):
            self._warmup()

        logger.info("GenderDetectionComponent ready")

    @classmethod
    def from_config(
        cls,
        path: Optional[str] = None,  
        env: Optional[str] = None,
    ) -> "GenderDetectionComponent":
        """Load and merge config for the given environment, then instantiate."""
        if path is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            path = os.path.join(root, "configs", "config.yaml")
        config = load_config(path=path, env=env)
        return cls(config)

    # Process a single video frame through the entire pipeline 
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp: Optional[float] = None,
    ) -> GenderResult:
        """
        Process a single video frame.
        """
        raw_frame = RawFrame(data=frame_bgr, frame_idx=frame_idx, timestamp=timestamp)

        # Stage 0: Frame gate
        decision = self._gate.gate(raw_frame)  # returns a GateDecision object indicating whether to proceed with inference or use cached result
        if decision.disposition != FrameDisposition.QUALIFIED: # if the frame is not qualified for inference, return the cached result or a no-face result
            cached = decision.cached_result
            if cached is None:
                return replace(_BASE_NO_FACE, frame_idx=frame_idx)
            source = "cache" if decision.disposition == FrameDisposition.CACHE_HIT else "skipped"
            return GenderResult(
                label=cached.label, confidence=cached.confidence,
                face_bbox=cached.face_bbox, frame_idx=frame_idx,
                source=source, is_smoothed=cached.is_smoothed,
                detection_score=cached.detection_score,
            )

        qualified = decision.frame

        # Stage 1: Face detection
        faces = self._detector.detect(qualified.data) # returns a list of DetectedFace objects with bounding boxes and detection scores
        if not faces: # if no faces are detected, return a no-face result and update the cache 
            result = replace(_BASE_NO_FACE, frame_idx=frame_idx)
            self._gate.update_cached_result(result)
            return result

        face = self._select_face(faces)

        # Stage 2: Face alignment
        aligned = self._aligner.align(face, qualified.data) # returns a cropped and aligned face image (numpy array) or None if alignment fails
        if aligned is None: # if alignment fails, return a no-face result and update the cache
            result = replace(_BASE_NO_FACE, frame_idx=frame_idx)
            self._gate.update_cached_result(result)
            return result

        # Stage 3: ViT classification
        try:
            raw_pred = self._classifier.predict(aligned) # returns a GenderResult object with label and confidence
        except RuntimeError as e: # if a runtime error occurs (e.g., GPU out of memory), log the error, return a no-face result, and update the cache
            if "out of memory" in str(e).lower():
                logger.error(f"GPU OOM on frame {frame_idx}")
                result = replace(_BASE_NO_FACE, frame_idx=frame_idx)
                self._gate.update_cached_result(result)
                return result
            raise

        # Stage 4: Temporal smoothing
        final_result = self._smoother.smooth(raw_pred, frame_idx)
        final_result.face_bbox = face.bbox
        final_result.detection_score = face.detection_score
        final_result.source = "inference"

        # Stage 5: Update the cache with the final result after smoothing
        self._gate.update_cached_result(final_result)
        return final_result

    # Process a batch of frames sequentially, returning a list of GenderResult objects.
    def process_batch(
        self,
        frames: List[np.ndarray],
    ) -> List[GenderResult]:
        """Process a list of frames sequentially."""
        return [self.process_frame(f, i) for i, f in enumerate(frames)]
    
    # Reset the internal state of the component, including the frame gate and result smoother.
    def reset(self):
        """Reset all stateful components."""
        self._smoother.reset()
        self._gate.reset()

    # Get statistics from the frame gate for monitoring and debugging.
    def get_stats(self) -> dict:
        return self._gate.get_stats()

    # produce a dict representation of the GenderResult for serialization
    def to_dict(self, result: GenderResult) -> dict:
        """Serialize a GenderResult to a plain dict."""
        return {
            "component": "gender_detection",
            "version":   self._cfg["component"]["version"],
            "payload":   result.to_dict(),
        }

    # Select a single face from a list of detected faces based on the configured strategy.
    def _select_face(self, faces):
        strategy = self._cfg.get("face_detection", {}).get("multi_face_strategy", "largest")
        if strategy == "largest":
            return max(faces, key=lambda f: f.bbox.area)
        if strategy == "highest_confidence":
            return max(faces, key=lambda f: f.detection_score)
        return faces[0]

    # Perform a warmup pass to initialize the model when the component is first created.
    # This can help reduce latency on the first real inference.
    def _warmup(self):  
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.process_frame(dummy, frame_idx=0)
        self.reset()
        logger.info("Warmup pass complete")
