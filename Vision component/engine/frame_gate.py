"""
engine/frame_gate.py
────────────────────
Stage 0 of the pipeline. Two-level frame filter:
  Level 1 — Stride filter: skips frames not on the stride boundary
  Level 2 — Perceptual cache: skips visually identical frames (even at stride boundary)

Input:  RawFrame(data=np.ndarray, frame_idx=int)
Output: QualifiedFrame | CacheHitFrame | SkippedFrame
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

from engine.schemas import RawFrame, QualifiedFrame, GenderResult
import logging

logger = logging.getLogger(__name__)

class FrameDisposition(Enum):
    QUALIFIED   = "qualified"   # Proceed to full pipeline
    CACHE_HIT   = "cache_hit"   # Visually same as last frame; reuse cached result
    STRIDE_SKIP = "stride_skip" # Stride filter; reuse last result

@dataclass
class GateDecision:
    disposition: FrameDisposition
    frame: Optional[QualifiedFrame] = None          # set if QUALIFIED
    cached_result: Optional[GenderResult] = None    # set if CACHE_HIT or STRIDE_SKIP

class FrameGate:
    """
    Two-level frame filter with stride-based skipping and perceptual caching.
    Config parameters:
        stride: int = 5
        phash_size: int = 16   (16x16 pHash = 256-bit hash)
        cache_hamming_threshold: int = 10  (≤10 differing bits = cache hit)
        cache_resize_to: tuple = (64, 64)  (resize before hashing for speed)
    """

    def __init__(
        self,
        stride: int = 5,
        phash_size: int = 16,
        cache_hamming_threshold: int = 10,
        cache_resize_to: Tuple[int, int] = (64, 64),
    ):
        self.stride = stride
        self.phash_size = phash_size
        self.hamming_threshold = cache_hamming_threshold
        self.cache_resize_to = cache_resize_to

        self._last_phash: Optional[np.ndarray] = None
        self._last_result: Optional[GenderResult] = None

        # Stats for monitoring
        self._n_qualified = 0
        self._n_cache_hits = 0
        self._n_stride_skips = 0

    def gate(self, raw_frame: RawFrame) -> GateDecision:
        """
        Args:
            raw_frame: RawFrame with .data (BGR numpy array) and .frame_idx (int)
        Returns:
            GateDecision indicating whether to proceed with inference or use cached result
        """
        # ── Level 1: Stride filter ────────────────────────────────────────────
        if raw_frame.frame_idx % self.stride != 0:
            self._n_stride_skips += 1
            return GateDecision(
                disposition=FrameDisposition.STRIDE_SKIP, # Skip this frame due to stride filter
                cached_result=self._last_result,  # Return the last cached result (may be None if no previous inference)
            )

        # ── Level 2: Perceptual cache ─────────────────────────────────────────
        current_hash = compute_phash(
            raw_frame.data, # BGR numpy array of the current frame
            hash_size=self.phash_size,
            resize_to=self.cache_resize_to, 
        )
        # Check if the current frame's pHash is similar to the last cached pHash
        if self._last_phash is not None and self._last_result is not None:
            dist = hamming_distance(current_hash, self._last_phash)
            if dist <= self.hamming_threshold: 
                self._n_cache_hits += 1  # Increment cache hit counter
                logger.debug(f"Frame {raw_frame.frame_idx}: pHash hit (dist={dist})") 
                return GateDecision(  # Return the cached result
                    disposition=FrameDisposition.CACHE_HIT,
                    cached_result=self._last_result,
                )

        # Frame passed both gates → proceed to full pipeline
        self._last_phash = current_hash
        self._n_qualified += 1

        return GateDecision(
            disposition=FrameDisposition.QUALIFIED,
            frame=QualifiedFrame(
                data=raw_frame.data,
                frame_idx=raw_frame.frame_idx,
                phash=current_hash,
            ),
        )
    
    
    def reset(self):
        """Clear the gate state (phash cache and cached result)."""
        self._last_phash = None
        self._last_result = None
    
    # update the cached result after a full inference has been performed on a qualified frame
    def update_cached_result(self, result: GenderResult):
        """Called by the pipeline after a full inference to update the cache."""
        self._last_result = result

    # Stats for monitoring
    def get_stats(self) -> dict: # for command-line logging and monitoring
        total = self._n_qualified + self._n_cache_hits + self._n_stride_skips
        return {
            "total_frames_seen": total,
            "qualified": self._n_qualified,
            "cache_hits": self._n_cache_hits,
            "stride_skips": self._n_stride_skips,
            "cache_hit_rate": self._n_cache_hits / max(1, total),
            "effective_skip_rate": (self._n_cache_hits + self._n_stride_skips) / max(1, total),
        }

def compute_phash(
    frame_bgr: np.ndarray,
    hash_size: int = 16,
    resize_to: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """
    Compute a perceptual hash (DCT-based pHash) of a frame.
    """
    # Resize
    small = cv2.resize(frame_bgr, resize_to, interpolation=cv2.INTER_AREA)
    # Grayscale
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 2D DCT
    dct = cv2.dct(gray)
    # Top-left block (low-frequency components)
    dct_low = dct[:hash_size, :hash_size]
    # Exclude DC component (dct_low[0,0]) from median to avoid brightness dominance
    flat = dct_low.flatten()
    median = np.median(flat[1:])
    # Binarize
    return flat > median

# using xor to compute the Hamming distance between two boolean hash arrays
# number of differing bits = number of 1s in the xor result is the Hamming distance  
def hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    """
    Compute Hamming distance between two boolean hash arrays.
    """
    return int(np.count_nonzero(hash_a != hash_b))
