import os
import logging

logger = logging.getLogger(__name__)


def trim_silence(audio_path: str, threshold_db: float = -40.0) -> str:
    import soundfile as sf
    import numpy as np
    import tempfile
    import shutil

    data, sr = sf.read(audio_path)
    threshold = 10 ** (threshold_db / 20)
    mask = np.abs(data) > threshold
    if not mask.any():
        return audio_path
    start = np.argmax(mask)
    end = len(mask) - np.argmax(mask[::-1])
    trimmed = data[start:end]
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        sf.write(tmp, trimmed, sr)
        shutil.move(tmp, audio_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    logger.debug("Trimmed silence from %s (threshold=%ddB)", audio_path, threshold_db)
    return audio_path


def normalize_to_wav(input_path: str, output_path: str = None) -> str:
    import soundfile as sf

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".wav"

    data, sr = sf.read(input_path)
    sf.write(output_path, data, sr)
    return os.path.abspath(output_path)
