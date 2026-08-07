import pytest
from lib.voice_selector import VoiceSelector

MOCK_CONFIG = {
    "voices": {
        "coqui": {
            "male":   ["p226", "p227"],
            "female": ["p225", "p228"],
        },
        "edge": {
            "male":   ["en-US-GuyNeural"],
            "female": ["en-US-JennyNeural"],
        },

    },
    "audio": {"unknown_gender_default": "male"},
}


@pytest.fixture
def selector():
    return VoiceSelector(MOCK_CONFIG)


def test_male_coqui(selector):
    voice_id, gender_used, warning = selector.select("male", "coqui")
    assert voice_id == "p226"
    assert gender_used == "male"
    assert warning is None


def test_female_coqui(selector):
    voice_id, gender_used, warning = selector.select("female", "coqui")
    assert voice_id == "p225"
    assert gender_used == "female"
    assert warning is None


def test_unknown_gender_defaults_to_male(selector):
    voice_id, gender_used, warning = selector.select("unknown", "coqui")
    assert gender_used == "male"
    assert voice_id == "p226"
    assert warning is not None
    assert "unknown" in warning.lower()


def test_invalid_gender_treated_as_unknown(selector):
    voice_id, gender_used, warning = selector.select("no_face", "coqui")
    assert gender_used == "male"
    assert warning is not None


def test_edge_male(selector):
    voice_id, _, _ = selector.select("male", "edge")
    assert voice_id == "en-US-GuyNeural"


def test_edge_female(selector):
    voice_id, _, _ = selector.select("female", "edge")
    assert voice_id == "en-US-JennyNeural"


def test_list_all_returns_dict(selector):
    voices = selector.list_all("coqui")
    assert "male" in voices
    assert "female" in voices
    assert len(voices["male"]) > 0
