"""
tests/test_pipeline.py
──────────────────────
Integration tests for the full Gender Detection Pipeline.
Verifies that all stages (Gate -> Detect -> Align -> Classify -> Smooth)
communicate correctly.
"""

import numpy as np
import pytest
from engine.component import GenderDetectionComponent
from engine.schemas import GenderLabel, GenderResult

def test_pipeline_initialization():
    """Verify that the component can be initialized from the default config."""
    # Assuming tests are run from the 'Vision component' directory
    # or that the paths in component.py are handled relative to the root.
    try:
        component = GenderDetectionComponent.from_config()
        assert component is not None
        assert component._detector is not None
        assert component._classifier is not None
    except Exception as e:
        pytest.fail(f"Component failed to initialize: {e}")

def test_full_pipeline_run():
    """Run a black frame through the pipeline and ensure it returns a GenderResult."""
    component = GenderDetectionComponent.from_config()
    
    # Create a dummy BGR frame (HD resolution)
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    # Process frame
    result = component.process_frame(dummy_frame, frame_idx=0)
    
    # Assertions
    assert isinstance(result, GenderResult)
    assert result.frame_idx == 0
    # Since it's a black frame, we expect NO_FACE
    assert result.label == GenderLabel.NO_FACE
    assert result.source == "no_face"

def test_pipeline_reset():
    """Verify that resetting the component clears its internal state."""
    component = GenderDetectionComponent.from_config()
    
    # Process a frame to fill buffers/cache
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    component.process_frame(dummy_frame, frame_idx=0)
    
    # Reset
    component.reset()
    
    # Check if gate cache is empty
    assert component._gate._last_phash is None
    assert component._gate._last_result is None
