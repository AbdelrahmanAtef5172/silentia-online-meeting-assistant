"""
examples/pipeline_usage.py
───────────────────────────
How the pipeline orchestrator calls the LLM component.

This demonstrates how the Vision, SLR, and LLM components
interact in the full multi-modal pipeline.
"""

from engine.service import CorrectionService
from engine.schemas import CorrectionContext
from engine.pipeline_adapter import PipelineAdapter, SLRResult, VisionResult


service = CorrectionService.from_config()

adapter = PipelineAdapter(service)

slr_result = SLRResult(
    text="she go store yesterday buy apple",
    confidence=0.72,
    session_id="session_001",
)

correction = adapter.process_slr_output(slr_result)

print(f"SLR output:   {slr_result.text}")
print(f"Corrected:    {correction.text_for_tts}")
print(f"Status:       {correction.status.value}")
# → Corrected: She went to the store yesterday to buy apples.

# Full pipeline: Vision → SLR → LLM → TTS
vision_result = VisionResult(label="female")
slr_result = SLRResult(
    text="patient show symptom high fever three day",
    session_id="session_002",
)

correction = adapter.process_slr_output(slr_result, vision_result)

print(f"\nWith gender context ({vision_result.label}):")
print(f"  Corrected: {correction.text_for_tts}")
# → The patient has been showing symptoms of high fever for three days.

# TTS consumes the result
tts_input = correction.text_for_tts
print(f"\nTTS input: {tts_input}")

# Orchestrator can log the full result
print(f"Log: {correction.to_dict()}")
