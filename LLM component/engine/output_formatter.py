"""
core/output_formatter.py
─────────────────────────
Packages the processed correction into a CorrectionResult dataclass
with all metadata required by the TTS component and the pipeline orchestrator.
"""

import time
from typing import Optional

from engine.schemas import (
    CorrectionResult,
    CorrectionStatus,
    CorrectionContext,
    InputCategory,
    ProcessedResponse,
)


class OutputFormatter:
    """
    Builds the final CorrectionResult ready for TTS consumption.
    """

    _QUALITY_TO_STATUS = {
        "good": CorrectionStatus.SUCCESS,
        "acceptable": CorrectionStatus.SUCCESS,
        "fallback": CorrectionStatus.FALLBACK,
    }

    def format(
        self,
        processed: ProcessedResponse,
        original_input: str,
        context: Optional[CorrectionContext],
        provider_used: Optional[str],
        latency_ms: float,
        from_cache: bool,
        input_category: Optional[InputCategory] = None,
        warning: Optional[str] = None,
    ) -> CorrectionResult:
        """
        Build a CorrectionResult from processed response and metadata.

        Args:
            processed:       ProcessedResponse from ResponseProcessor
            original_input:  Raw SLR input text
            context:         Optional CorrectionContext
            provider_used:   Name of the provider that handled the request
            latency_ms:      Total latency in milliseconds
            from_cache:      Whether the result came from cache
            input_category:  Category assigned by InputGuard
            warning:         Optional warning message

        Returns:
            CorrectionResult ready for TTS consumption
        """
        status = self._determine_status(processed, from_cache)

        corrected_text = processed.corrected_text if status != CorrectionStatus.PASSTHROUGH else original_input

        if status == CorrectionStatus.FAILED:
            corrected_text = original_input

        return CorrectionResult(
            corrected_text=corrected_text,
            original_text=original_input,
            status=status,
            provider_used=provider_used,
            latency_ms=round(latency_ms, 2),
            input_category=input_category,
            from_cache=from_cache,
            warning=warning,
        )

    def make_passthrough(
        self,
        text: str,
        input_category: InputCategory,
        warning: Optional[str] = None,
    ) -> CorrectionResult:
        """
        Build a passthrough result for inputs that don't need LLM correction.
        """
        return CorrectionResult(
            corrected_text=text,
            original_text=text,
            status=CorrectionStatus.PASSTHROUGH,
            input_category=input_category,
            latency_ms=0.0,
            from_cache=False,
            warning=warning,
        )

    def make_failed(
        self,
        original_text: str,
        warning: str,
    ) -> CorrectionResult:
        """Build a failed result when all providers are unreachable."""
        return CorrectionResult(
            corrected_text=original_text,
            original_text=original_text,
            status=CorrectionStatus.FAILED,
            latency_ms=0.0,
            from_cache=False,
            warning=warning,
        )

    def _determine_status(
        self,
        processed: ProcessedResponse,
        from_cache: bool,
    ) -> CorrectionStatus:
        if from_cache:
            return CorrectionStatus.CACHED
        return self._QUALITY_TO_STATUS.get(processed.quality, CorrectionStatus.FAILED)
