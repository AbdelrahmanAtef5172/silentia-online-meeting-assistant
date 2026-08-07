"""
engine/smoother.py
──────────────────
Temporal majority-vote smoother. Reduces single-frame prediction noise
by averaging over a sliding window of recent predictions.

Input:  RawPrediction
Output: GenderResult (smoothed)
"""

from collections import deque
from typing import Deque
import numpy as np

from engine.schemas import RawPrediction, GenderResult, GenderLabel

class ResultSmoother:
    """
    Sliding window majority-vote smoother.
    """
    
    # Initialize the smoother with a specified window size and minimum confidence threshold.
    def __init__(self, window_size: int = 5, min_confidence_to_include: float = 0.55):
        self.window_size = window_size                   # Size of the sliding window for smoothing predictions.
        self.min_confidence = min_confidence_to_include  # Minimum confidence threshold to include a prediction in the smoothing buffer.
        self._buffer: Deque[RawPrediction] = deque(maxlen=window_size) # Internal buffer to store recent predictions for smoothing.

    # Smooth a raw prediction by adding it to the buffer and returning a smoothed result.
    def smooth(self, raw: RawPrediction, frame_idx: int) -> GenderResult:
        """
        Add raw prediction to buffer and return smoothed result.
        """
        if raw.confidence >= self.min_confidence:
            self._buffer.append(raw)
        elif len(self._buffer) == 0:
            # Buffer is empty and we got a low-confidence prediction — include it anyway
            self._buffer.append(raw)
        if len(self._buffer) == 0:
            # Completely cold start with low confidence — pass through
            return GenderResult(
                label=raw.label,
                confidence=raw.confidence,
                face_bbox=None,
                frame_idx=frame_idx,
                source="inference_cold",
                is_smoothed=False,
            )

        # Majority vote
        labels = [p.label for p in self._buffer]
        n_female = labels.count(GenderLabel.FEMALE)
        n_male = labels.count(GenderLabel.MALE)
        voted_label = GenderLabel.FEMALE if n_female >= n_male else GenderLabel.MALE

        # Average probability for the voted label across buffer
        label_idx = 1 if voted_label == GenderLabel.FEMALE else 0
        avg_confidence = float(np.mean([p.probabilities[label_idx] for p in self._buffer]))

        return GenderResult(
            label=voted_label,   # The label determined by majority vote.
            confidence=avg_confidence, # The average confidence for the voted label across the buffer.
            # If the raw prediction has an aligned face, use its bounding box; otherwise, set to None.
            face_bbox=raw.aligned_face.source_bbox if hasattr(raw, 'aligned_face') and raw.aligned_face else None,
            frame_idx=frame_idx,
            source="inference_smoothed",
            is_smoothed=True,
        )
    
    # Reset the internal buffer of the smoother, clearing all stored predictions.
    def reset(self):
        """Clear the smoothing buffer. Call when subject changes."""
        self._buffer.clear()
