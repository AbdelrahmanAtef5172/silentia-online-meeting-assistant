from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class TTSRequest:
    text: str
    gender: str = "unknown"
    output_mode: Literal["file", "play", "stream"] = "play"
    output_path: Optional[str] = None
    session_id: Optional[str] = None
    speed: float = 1.0


@dataclass
class SynthesisResult:
    status: Literal["success", "skipped", "fallback", "failed"]
    audio_path: Optional[str]
    played: bool
    provider_used: str
    voice_id: str
    gender_used: str
    latency_ms: float
    duration_ms: float
    warning: Optional[str] = None
    audio_bytes: Optional[bytes] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "audio_path": self.audio_path,
            "played": self.played,
            "provider_used": self.provider_used,
            "voice_id": self.voice_id,
            "gender_used": self.gender_used,
            "latency_ms": round(self.latency_ms, 2),
            "duration_ms": round(self.duration_ms, 2),
            "warning": self.warning,
        }
