import os
import json
import pytest
from lib.service import TTSService

SAMPLE_TEXTS = {
    "short":  "Hello.",
    "medium": "She went to the store to buy groceries.",
    "long":   "The patient has been showing symptoms of a high fever for the past three days and needs immediate medical attention.",
}


def make_config(tmp_path):
    return {
        "providers": {
            "priority_order": ["edge", "coqui"],
            "force_provider": None,
            "coqui": {"use_gpu": False},
            "edge": {"timeout_seconds": 10},
        },
        "voices": {
            "coqui": {"male": ["p226"], "female": ["p225"]},
            "edge":  {"male": ["en-US-GuyNeural"], "female": ["en-US-JennyNeural"]},
        },
        "audio": {
            "output_dir": str(tmp_path),
            "unknown_gender_default": "male",
            "auto_trim_silence": False,
            "volume": 1.0,
        },
    }


@pytest.fixture
def service(tmp_path):
    return TTSService(make_config(tmp_path))


@pytest.mark.integration
@pytest.mark.parametrize("gender", ["male", "female"])
def test_synthesize_to_file_both_genders(service, tmp_path, gender):
    out = str(tmp_path / f"test_{gender}.wav")
    result = service.synthesize(
        text=SAMPLE_TEXTS["medium"],
        gender=gender,
        output_mode="file",
        output_path=out,
    )
    assert result.status in ("success", "fallback")
    assert result.audio_path is not None
    assert os.path.exists(result.audio_path)
    assert os.path.getsize(result.audio_path) > 1000
    assert result.gender_used == gender


@pytest.mark.integration
def test_unknown_gender_produces_audio(service, tmp_path):
    out = str(tmp_path / "test_unknown.wav")
    result = service.synthesize(
        text=SAMPLE_TEXTS["short"],
        gender="unknown",
        output_mode="file",
        output_path=out,
    )
    assert result.status in ("success", "fallback")
    assert result.warning is not None
    assert os.path.exists(result.audio_path)


@pytest.mark.integration
def test_empty_text_skipped(service):
    result = service.synthesize(text="", gender="male")
    assert result.status == "skipped"
    assert result.audio_path is None


@pytest.mark.integration
def test_long_text_produces_audio(service, tmp_path):
    out = str(tmp_path / "test_long.wav")
    result = service.synthesize(
        text=SAMPLE_TEXTS["long"],
        gender="female",
        output_mode="file",
        output_path=out,
    )
    assert result.status in ("success", "fallback")
    assert result.duration_ms > 0


@pytest.mark.integration
def test_result_to_dict_is_json_serializable(service, tmp_path):
    out = str(tmp_path / "test_serial.wav")
    result = service.synthesize(
        text="Hello.",
        gender="male",
        output_mode="file",
        output_path=out,
    )
    serialized = json.dumps(result.to_dict())
    assert "status" in serialized
