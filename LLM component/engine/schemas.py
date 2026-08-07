"""
utils/schemas.py
────────────────
Single source of truth for all data types crossing component boundaries.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from enum import Enum


class InputCategory(str, Enum):
    EMPTY            = "empty"
    PASSTHROUGH      = "passthrough"
    PUNCTUATION_ONLY = "punctuation_only"
    NUMBER_HEAVY     = "number_heavy"
    TOO_LONG         = "too_long"
    NORMAL           = "normal"


class CorrectionStatus(str, Enum):
    SUCCESS     = "success"
    CACHED      = "cached"
    PASSTHROUGH = "passthrough"
    FALLBACK    = "fallback"
    FAILED      = "failed"


@dataclass
class CorrectionContext:
    topic_domain:    Optional[str]  = None
    speaker_gender:  Optional[str]  = None
    session_id:      Optional[str]  = None
    custom_glossary: Optional[dict] = None


@dataclass
class CorrectionResult:
    corrected_text:  str
    original_text:   str
    status:          CorrectionStatus
    provider_used:   Optional[str]       = None
    latency_ms:      float               = 0.0
    input_category:  Optional[InputCategory] = None
    from_cache:      bool                = False
    warning:         Optional[str]       = None

    def to_dict(self) -> dict:
        return {
            "corrected_text": self.corrected_text,
            "original_text":  self.original_text,
            "status":         self.status.value,
            "provider_used":  self.provider_used,
            "latency_ms":     round(self.latency_ms, 2),
            "from_cache":     self.from_cache,
            "warning":        self.warning,
        }

    @property
    def text_for_tts(self) -> str:
        return self.corrected_text


@dataclass
class BuiltPrompt:
    system: str
    user:   str


@dataclass
class GuardDecision:
    category:         InputCategory
    action:           Literal["pass", "correct", "truncate", "reject"]
    transformed_text: str
    warning:          Optional[str] = None


@dataclass
class ProcessedResponse:
    corrected_text:    str
    quality:           Literal["good", "acceptable", "fallback"]
    extraction_method: str
