import os
import subprocess
import logging
import platform

logger = logging.getLogger(__name__)


class AudioOutput:

    def __init__(self, config: dict):
        self._volume = config.get("audio", {}).get("volume", 1.0)

    def play(self, audio_path: str) -> None:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            logger.debug("sounddevice/soundfile not available — using subprocess playback")
        else:
            try:
                data, samplerate = sf.read(audio_path)
                sd.play(data * self._volume, samplerate)
                sd.wait()
                return
            except Exception as e:
                logger.warning("sounddevice playback failed: %s — trying subprocess", e)

        self._subprocess_play(audio_path)

    def get_duration_ms(self, audio_path: str) -> float:
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            return (info.frames / info.samplerate) * 1000
        except Exception:
            return 0.0

    def _subprocess_play(self, audio_path: str) -> None:
        system = platform.system()
        try:
            if system == "Linux":
                subprocess.run(["aplay", audio_path], check=True, capture_output=True)
            elif system == "Darwin":
                subprocess.run(["afplay", audio_path], check=True, capture_output=True)
            elif system == "Windows":
                subprocess.run(
                    ["powershell", "-c",
                     f"(New-Object Media.SoundPlayer '{audio_path}').PlaySync()"],
                    check=True, capture_output=True
                )
            else:
                logger.error("Cannot play audio: unknown platform '%s'", system)
        except subprocess.CalledProcessError as e:
            logger.error("Subprocess audio playback failed: %s", e)
