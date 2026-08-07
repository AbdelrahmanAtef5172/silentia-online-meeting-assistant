import os
import asyncio
import tempfile
import logging
import concurrent.futures
from engines.base import BaseProvider

try:
    import soundfile as sf
except ImportError:
    sf = None

logger = logging.getLogger(__name__)


class EdgeProvider(BaseProvider):

    def __init__(self, config: dict):
        self._config = config
        self._available = None
        logger.info("EdgeProvider configured")

    @property
    def name(self) -> str:
        return "edge"

    @staticmethod
    def _run_async(coro):
        """Safely run an async coroutine from a sync context.
        
        Works both outside and inside an already-running event loop
        by spawning a separate thread with its own loop when needed.
        """
        try:
            asyncio.get_running_loop()
            in_async = True
        except RuntimeError:
            in_async = False

        if not in_async:
            return asyncio.run(coro)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()

    def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float,
        output_path: str,
    ) -> str:
        try:
            import edge_tts
        except ImportError:
            raise RuntimeError("edge-tts not installed. Run: pip install edge-tts")

        if sf is None:
            raise RuntimeError("soundfile not installed. Run: pip install soundfile")

        rate_str = self._speed_to_rate(speed)

        tmp_mp3 = None
        try:
            tmp_mp3 = tempfile.mktemp(suffix=".mp3")

            async def _run():
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=voice_id,
                    rate=rate_str,
                )
                await communicate.save(tmp_mp3)

            self._run_async(_run())

            if not os.path.exists(tmp_mp3):
                raise RuntimeError("edge-tts produced no output")

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            data, sr = sf.read(tmp_mp3)
            sf.write(output_path, data, sr)

        finally:
            if tmp_mp3 and os.path.exists(tmp_mp3):
                os.unlink(tmp_mp3)

        if not os.path.exists(output_path):
            raise RuntimeError(f"Failed to write WAV at {output_path}")

        return os.path.abspath(output_path)

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import edge_tts
            import httpx
            httpx.get("https://speech.platform.bing.com", timeout=2.0)
            self._available = True
        except Exception:
            self._available = False
        return self._available

    def list_voices(self) -> dict:
        return {
            "male": [
                "en-US-GuyNeural",
                "en-US-ChristopherNeural",
                "en-GB-RyanNeural",
                "en-AU-WilliamNeural",
            ],
            "female": [
                "en-US-JennyNeural",
                "en-US-AriaNeural",
                "en-GB-SoniaNeural",
                "en-AU-NatashaNeural",
            ],
        }

    @staticmethod
    def _speed_to_rate(speed: float) -> str:
        pct = round((speed - 1.0) * 100)
        if pct >= 0:
            return f"+{pct}%"
        return f"{pct}%"
