"""
integration/pipeline_adapter.py
────────────────────────────────
Thin adapter between the SLR component's output format and
CorrectionService's input format.

SLR component outputs: SLRResult(text: str, confidence: float, session_id: str)
LLM component expects: (raw_text: str, context: CorrectionContext)

This adapter is the ONLY place that knows about both schemas.
If the SLR output format changes, only this file changes.
"""

from dataclasses import dataclass
from typing import Optional

from engine.schemas import CorrectionContext, CorrectionResult
from engine.service import CorrectionService


@dataclass
class SLRResult:
    text: str
    confidence: float = 0.0
    session_id: Optional[str] = None


@dataclass
class VisionResult:
    label: Optional[str] = None


class PipelineAdapter:
    def __init__(self, service: CorrectionService):
        self._service = service

    def process_slr_output(
        self,
        slr_result: SLRResult,
        vision_result: Optional[VisionResult] = None,
    ) -> CorrectionResult:
        """
        Args:
            slr_result:    SLRResult from the SLR component
            vision_result: Optional VisionResult from Vision component
                           (contains speaker_gender, if available)

        Returns:
            CorrectionResult ready for TTS component
        """
        context = CorrectionContext(
            session_id=getattr(slr_result, "session_id", None),
            speaker_gender=getattr(vision_result, "label", None) if vision_result else None,
        )

        return self._service.correct(
            raw_text=slr_result.text,
            context=context,
        )
