import os
import pytest
from unittest.mock import patch, MagicMock


def test_coqui_is_available_when_tts_installed():
    with patch.dict("sys.modules", {"TTS": MagicMock(), "TTS.api": MagicMock()}):
        from engines.coqui_provider import CoquiProvider
        p = CoquiProvider({"use_gpu": False})
        assert p.is_available() is True


def test_coqui_raises_on_import_failure():
    with patch.dict("sys.modules", {"TTS": None, "TTS.api": None}):
        from engines.coqui_provider import CoquiProvider
        p = CoquiProvider({"use_gpu": False})
        with pytest.raises(RuntimeError, match="not installed"):
            p._load_model()


def test_edge_speed_to_rate_positive():
    from engines.edge_provider import EdgeProvider
    assert EdgeProvider._speed_to_rate(1.5) == "+50%"


def test_edge_speed_to_rate_negative():
    from engines.edge_provider import EdgeProvider
    assert EdgeProvider._speed_to_rate(0.8) == "-20%"


def test_edge_speed_to_rate_normal():
    from engines.edge_provider import EdgeProvider
    assert EdgeProvider._speed_to_rate(1.0) == "+0%"


def test_service_skips_empty_text():
    from lib.service import TTSService
    config = _minimal_config()
    service = TTSService(config)
    result = service.synthesize(text="", gender="female")
    assert result.status == "skipped"
    assert result.played is False


def test_service_skips_whitespace_text():
    from lib.service import TTSService
    config = _minimal_config()
    service = TTSService(config)
    result = service.synthesize(text="   \n  ", gender="male")
    assert result.status == "skipped"


def _minimal_config():
    return {
        "providers": {
            "priority_order": [],
            "force_provider": None,
            "coqui": {"use_gpu": False},
            "edge": {"timeout_seconds": 5},
        },
        "voices": {
            "coqui": {"male": ["p226"], "female": ["p225"]},
            "edge":  {"male": ["en-US-GuyNeural"], "female": ["en-US-JennyNeural"]},
        },
        "audio": {
            "output_dir": "/tmp/tts_test",
            "unknown_gender_default": "male",
            "auto_trim_silence": False,
            "volume": 1.0,
        },
    }
