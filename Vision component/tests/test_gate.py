"""
tests/test_gate.py
──────────────────
Unit tests for the Frame Gate logic:
  - Stride skipping
  - Perceptual cache hits
"""

import numpy as np
import pytest
from engine.frame_gate import FrameGate, FrameDisposition
from engine.schemas import RawFrame, GenderResult, GenderLabel

def test_stride_filter():
    """Verify that frames are skipped according to the stride value."""
    gate = FrameGate(stride=5)
    
    # Frame 0: Should be QUALIFIED (start of stream)
    frame0 = RawFrame(data=np.zeros((100, 100, 3), dtype=np.uint8), frame_idx=0)
    decision0 = gate.gate(frame0)
    assert decision0.disposition == FrameDisposition.QUALIFIED
    
    # Frame 1: Should be STRIDE_SKIP
    frame1 = RawFrame(data=np.zeros((100, 100, 3), dtype=np.uint8), frame_idx=1)
    decision1 = gate.gate(frame1)
    assert decision1.disposition == FrameDisposition.STRIDE_SKIP
    
    # Frame 5: Should be QUALIFIED (on the stride boundary)
    # Note: Frame 5 will be QUALIFIED if it's visually different or if it's the first boundary hit
    frame5 = RawFrame(data=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8), frame_idx=5)
    decision5 = gate.gate(frame5)
    assert decision5.disposition == FrameDisposition.QUALIFIED

def test_perceptual_cache():
    """Verify that visually identical frames at stride boundaries are cached."""
    gate = FrameGate(stride=1, cache_hamming_threshold=10)
    
    # Frame 0: Qualified
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    frame0 = RawFrame(data=img.copy(), frame_idx=0)
    decision0 = gate.gate(frame0)
    assert decision0.disposition == FrameDisposition.QUALIFIED
    
    # Update cache with a mock result
    mock_result = GenderResult(
        label=GenderLabel.MALE, confidence=0.99, 
        face_bbox=None, frame_idx=0, source="inference"
    )
    gate.update_cached_result(mock_result)
    
    # Frame 1: Identical image -> should be CACHE_HIT
    frame1 = RawFrame(data=img.copy(), frame_idx=1)
    decision1 = gate.gate(frame1)
    assert decision1.disposition == FrameDisposition.CACHE_HIT
    assert decision1.cached_result.label == GenderLabel.MALE

def test_stats_tracking():
    """Verify that internal counters are incremented correctly."""
    gate = FrameGate(stride=2)
    
    # Frame 0: Qualified
    gate.gate(RawFrame(data=np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8), frame_idx=0))
    # Frame 1: Stride skip
    gate.gate(RawFrame(data=np.zeros((64, 64, 3), dtype=np.uint8), frame_idx=1))
    
    stats = gate.get_stats()
    assert stats["total_frames_seen"] == 2
    assert stats["qualified"] == 1
    assert stats["stride_skips"] == 1
