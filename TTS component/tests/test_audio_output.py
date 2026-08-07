import os
import pytest
import tempfile
from lib.audio_output import AudioOutput


@pytest.fixture
def audio_output():
    return AudioOutput({"audio": {"volume": 1.0}})


def test_get_duration_ms_nonexistent_file(audio_output):
    duration = audio_output.get_duration_ms("/nonexistent/file.wav")
    assert duration == 0.0


def test_play_nonexistent_file_raises(audio_output):
    with pytest.raises(FileNotFoundError):
        audio_output.play("/nonexistent/file.wav")


def test_get_duration_ms_valid_file(audio_output):
    import soundfile as sf
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    try:
        sr = 22050
        data = np.zeros(sr * 2, dtype=np.float32)
        sf.write(path, data, sr)

        duration = audio_output.get_duration_ms(path)
        assert duration == pytest.approx(2000.0, abs=50)
    finally:
        if os.path.exists(path):
            os.unlink(path)
