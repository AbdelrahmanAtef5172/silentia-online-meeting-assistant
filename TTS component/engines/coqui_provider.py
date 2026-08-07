import os
import logging
from engines.base import BaseProvider

logger = logging.getLogger(__name__)

MODEL_NAME = "tts_models/en/vctk/vits"


class CoquiProvider(BaseProvider):

    def __init__(self, config: dict):
        self._config = config
        self._model = None
        self._gpu = config.get("use_gpu", False)
        logger.info("CoquiProvider configured | model=%s | gpu=%s", MODEL_NAME, self._gpu)

    @property
    def name(self) -> str:
        return "coqui"

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from TTS.api import TTS
            logger.info("Loading Coqui TTS model: %s", MODEL_NAME)
            self._model = TTS(MODEL_NAME, gpu=self._gpu)
            logger.info("Coqui TTS model loaded successfully")
        except ImportError:
            raise RuntimeError("Coqui TTS not installed. Run: pip install TTS")
        except Exception as e:
            raise RuntimeError(f"Failed to load Coqui model '{MODEL_NAME}': {e}")

    def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float,
        output_path: str,
    ) -> str:
        self._load_model()

        if speed != 1.0:
            logger.warning(
                "Coqui VITS does not natively support speed adjustment. "
                "Requested speed=%.1f will be ignored.", speed
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        self._model.tts_to_file(
            text=text,
            speaker=voice_id,
            file_path=output_path,
        )

        if not os.path.exists(output_path):
            raise RuntimeError(f"Coqui TTS wrote no output file at {output_path}")

        return os.path.abspath(output_path)

    def is_available(self) -> bool:
        try:
            import TTS
        except ImportError:
            return False
        import os
        cache = self._config.get("model_cache_dir")
        if cache:
            model_path = os.path.join(cache, "tts_models--en--vctk--vits")
        else:
            model_path = os.path.join(
                os.path.expanduser("~"), ".local", "share", "tts",
                "tts_models--en--vctk--vits",
            )
        return os.path.isdir(model_path)

    def list_voices(self) -> dict:
        return {
            "male": [
                "p226", "p227", "p232", "p243", "p254",
                "p256", "p258", "p259", "p270", "p271",
                "p274", "p275", "p278", "p279", "p281",
            ],
            "female": [
                "p225", "p228", "p229", "p230", "p231",
                "p233", "p234", "p236", "p237", "p238",
                "p239", "p240", "p241", "p243", "p244",
            ],
        }
