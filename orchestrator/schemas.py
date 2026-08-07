from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Literal


class ProcessingMode(Enum):
    FILE = "file"
    WEBCAM = "webcam"


class SLRMode(Enum):
    SINGLE = "single"
    CONTINUOUS = "continuous"


@dataclass
class PipelineInput:
    video_path: Optional[str] = None
    webcam_id: int = 0
    mode: ProcessingMode = ProcessingMode.FILE
    slr_mode: SLRMode = SLRMode.SINGLE
    output_dir: str = "output"
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    tts_output_mode: Literal["file", "play"] = "file"
    tts_speed: float = 1.0
    webcam_duration_sec: float = 10.0


@dataclass
class StageResult:
    status: Literal["pending", "running", "completed", "skipped", "failed"]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class PipelineResult:
    success: bool
    vision: StageResult
    slr: StageResult
    llm: StageResult
    tts: StageResult
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "vision": {k: v for k, v in self.vision.__dict__.items() if v is not None},
            "slr": {k: v for k, v in self.slr.__dict__.items() if v is not None},
            "llm": {k: v for k, v in self.llm.__dict__.items() if v is not None},
            "tts": {k: v for k, v in self.tts.__dict__.items() if v is not None},
            "error": self.error,
        }
