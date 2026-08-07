import time
import os
import uuid
import logging
from typing import Optional

from lib.schemas import TTSRequest, SynthesisResult
from lib.voice_selector import VoiceSelector
from lib.audio_output import AudioOutput
from lib.audio_utils import trim_silence
from engines.coqui_provider import CoquiProvider
from engines.edge_provider import EdgeProvider


logger = logging.getLogger(__name__)


class TTSService:

    def __init__(self, config: dict):
        self._cfg = config
        self._voice_selector = VoiceSelector(config)
        self._audio_output = AudioOutput(config)
        self._providers = self._build_provider_stack(config)
        logger.info("TTSService initialized | providers=%s",
                    [p.name for p in self._providers])

    @classmethod
    def from_config(cls, env: str = None, force_provider: str = None) -> "TTSService":
        from lib.config_loader import load_config
        config = load_config(env=env)
        if force_provider:
            config.setdefault("providers", {})["force_provider"] = force_provider
        log_level = config.get("component", {}).get("log_level", "INFO")
        level = getattr(logging, log_level.upper(), logging.INFO)
        logging.getLogger("lib").setLevel(level)
        logging.getLogger("engines").setLevel(level)
        return cls(config)

    def synthesize(
        self,
        text: str,
        gender: str = "unknown",
        output_mode: str = "play",
        output_path: Optional[str] = None,
        session_id: Optional[str] = None,
        speed: float = 1.0,
    ) -> SynthesisResult:
        speed = max(0.5, min(2.0, speed))
        request = TTSRequest(
            text=text,
            gender=gender,
            output_mode=output_mode,
            output_path=output_path,
            session_id=session_id,
            speed=speed,
        )
        return self.synthesize_request(request)

    def synthesize_request(self, request: TTSRequest) -> SynthesisResult:
        t_start = time.perf_counter()

        if not request.text or not request.text.strip():
            logger.debug("Empty text received — skipping synthesis")
            return SynthesisResult(
                status="skipped", audio_path=None, played=False,
                provider_used="none", voice_id="none",
                gender_used=request.gender, latency_ms=0.0, duration_ms=0.0,
                warning="Empty text input — no audio produced",
            )

        fallback_used = False
        self._warn_if_long_text(request.text)
        output_path = self._resolve_output_path(request)

        for provider in self._providers:
            voice_id, gender_used, warning = self._voice_selector.select(
                gender=request.gender,
                provider_name=provider.name,
            )

            try:
                audio_path = provider.synthesize(
                    text=request.text,
                    voice_id=voice_id,
                    speed=request.speed,
                    output_path=output_path,
                )

                if self._cfg.get("audio", {}).get("auto_trim_silence", False):
                    audio_path = trim_silence(audio_path)

                audio_bytes = None
                played = False
                if request.output_mode == "play":
                    self._audio_output.play(audio_path)
                    played = True
                elif request.output_mode == "stream":
                    with open(audio_path, "rb") as f:
                        audio_bytes = f.read()

                latency_ms = (time.perf_counter() - t_start) * 1000
                duration_ms = self._audio_output.get_duration_ms(audio_path)

                logger.info(
                    "Synthesis complete | provider=%s voice=%s gender=%s latency=%.0fms",
                    provider.name, voice_id, gender_used, latency_ms
                )

                status = "fallback" if fallback_used else "success"
                return SynthesisResult(
                    status=status,
                    audio_path=audio_path,
                    played=played,
                    provider_used=provider.name,
                    voice_id=voice_id,
                    gender_used=gender_used,
                    latency_ms=latency_ms,
                    duration_ms=duration_ms,
                    warning=warning,
                    audio_bytes=audio_bytes,
                )

            except Exception as exc:
                fallback_used = True
                logger.warning("Provider %s failed: %s — trying next", provider.name, exc)
                continue

        latency_ms = (time.perf_counter() - t_start) * 1000
        logger.error("All TTS providers failed for input: %.40s...", request.text)
        return SynthesisResult(
            status="failed", audio_path=None, played=False,
            provider_used="none", voice_id="none",
            gender_used=request.gender, latency_ms=latency_ms, duration_ms=0.0,
            warning="All providers failed — no audio produced",
        )

    def _build_provider_stack(self, config: dict):
        force = config.get("providers", {}).get("force_provider")
        order = [force] if force else config["providers"]["priority_order"]

        registry = {
            "coqui": lambda: CoquiProvider(config["providers"]["coqui"]),
            "edge": lambda: EdgeProvider(config["providers"]["edge"]),

        }
        providers = []
        for name in order:
            if name in registry:
                try:
                    provider = registry[name]()
                    if not provider.is_available():
                        logger.warning("Provider %s is not available — skipping", name)
                        continue
                    providers.append(provider)
                except Exception as e:
                    logger.warning("Provider %s failed to initialize: %s", name, e)
        return providers

    def _resolve_output_path(self, request: TTSRequest) -> str:
        if request.output_path:
            return request.output_path
        out_dir = self._cfg.get("audio", {}).get("output_dir", "output")
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{request.session_id or uuid.uuid4().hex[:8]}.wav"
        return os.path.join(out_dir, fname)

    @staticmethod
    def _warn_if_long_text(text: str, max_chars: int = 1000) -> None:
        if len(text) > max_chars:
            logger.warning(
                "Text length (%d chars) exceeds %d chars. "
                "Some TTS engines may truncate or fail.",
                len(text), max_chars,
            )
