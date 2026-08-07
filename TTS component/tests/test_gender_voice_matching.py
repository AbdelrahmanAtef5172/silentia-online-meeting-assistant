import os
import json
import pytest
import tempfile
from lib.service import TTSService
from lib.voice_selector import VoiceSelector
from engines.coqui_provider import CoquiProvider
from engines.edge_provider import EdgeProvider

GENDERS = ["male", "female", "unknown"]
PROVIDERS = ["coqui", "edge"]

SAMPLE_TEXT = "She went to the store to buy groceries."


def make_config(output_dir, auto_trim_silence=False, force_provider=None):
    cfg = {
        "providers": {
            "priority_order": PROVIDERS[:],
            "force_provider": force_provider,
            "coqui": {"use_gpu": False},
            "edge": {"timeout_seconds": 10},
        },
        "voices": {
            "coqui": {"male": ["p226"], "female": ["p225"]},
            "edge": {"male": ["en-US-GuyNeural"], "female": ["en-US-JennyNeural"]},
        },
        "audio": {
            "output_dir": str(output_dir),
            "unknown_gender_default": "male",
            "auto_trim_silence": auto_trim_silence,
            "volume": 1.0,
        },
    }
    return cfg


# ── 1. VoiceSelector Unit Tests ─────────────────────────────────────────────

class TestVoiceSelector:

    @pytest.fixture
    def selector(self):
        cfg = make_config("/tmp")
        return VoiceSelector(cfg)

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_male_returns_male_voice(self, selector, provider):
        v_id, gender, warning = selector.select("male", provider)
        assert gender == "male"
        assert v_id is not None and len(v_id) > 0
        assert warning is None

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_female_returns_female_voice(self, selector, provider):
        v_id, gender, warning = selector.select("female", provider)
        assert gender == "female"
        assert v_id is not None and len(v_id) > 0
        assert warning is None

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_unknown_defaults_to_male(self, selector, provider):
        v_id, gender, warning = selector.select("unknown", provider)
        assert gender == "male"
        assert warning is not None
        assert "unknown" in warning.lower()

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_invalid_gender_treated_as_unknown(self, selector, provider):
        v_id, gender, warning = selector.select("no_face", provider)
        assert gender == "male"
        assert warning is not None

    @pytest.mark.parametrize("provider", PROVIDERS)
    def test_list_all_returns_voices(self, selector, provider):
        voices = selector.list_all(provider)
        assert "male" in voices
        assert "female" in voices

    def test_warning_accumulated_when_unknown_and_no_male_voices(self):
        cfg = {
            "voices": {"coqui": {"male": [], "female": ["p225"]}},
            "audio": {"unknown_gender_default": "male"},
        }
        sel = VoiceSelector(cfg)
        v_id, gender, warning = sel.select("unknown", "coqui")
        assert gender == "female"
        assert "unknown" in warning.lower()
        assert "female" in warning.lower()


# ── 2. Provider-Level Unit Tests ─────────────────────────────────────────────

class TestProviderCapabilities:

    def test_coqui_known_voices(self):
        voices = CoquiProvider({"use_gpu": False}).list_voices()
        assert len(voices["male"]) > 0
        assert len(voices["female"]) > 0
        assert "p226" in voices["male"]
        assert "p225" in voices["female"]

    def test_edge_known_voices(self):
        voices = EdgeProvider({"timeout_seconds": 5}).list_voices()
        assert len(voices["male"]) > 0
        assert len(voices["female"]) > 0
        assert "en-US-GuyNeural" in voices["male"]
        assert "en-US-JennyNeural" in voices["female"]

    def test_edge_speed_to_rate_normal(self):
        assert EdgeProvider._speed_to_rate(1.0) == "+0%"

    def test_edge_speed_to_rate_faster(self):
        assert EdgeProvider._speed_to_rate(1.5) == "+50%"

    def test_edge_speed_to_rate_slower(self):
        assert EdgeProvider._speed_to_rate(0.8) == "-20%"

    def test_edge_speed_to_rate_rounding(self):
        assert EdgeProvider._speed_to_rate(1.37) == "+37%"


# ── 3. Per-Provider × Per-Gender Integration Tests ───────────────────────────

class TestProviderGenderMatching:

    @pytest.fixture
    def tmp_output(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.mark.parametrize("provider", ["coqui", "edge"])
    def test_male_voice_selected(self, provider, tmp_output):
        service = TTSService(make_config(tmp_output, force_provider=provider))
        result = service.synthesize(text=SAMPLE_TEXT, gender="male", output_mode="file")
        assert result.status in ("success", "fallback")
        assert result.gender_used == "male"
        assert result.provider_used == provider
        assert os.path.exists(result.audio_path)
        assert os.path.getsize(result.audio_path) > 1000

    @pytest.mark.parametrize("provider", ["coqui", "edge"])
    def test_female_voice_selected(self, provider, tmp_output):
        service = TTSService(make_config(tmp_output, force_provider=provider))
        result = service.synthesize(text=SAMPLE_TEXT, gender="female", output_mode="file")
        assert result.status in ("success", "fallback")
        assert result.gender_used == "female"
        assert result.provider_used == provider
        assert os.path.exists(result.audio_path)
        assert os.path.getsize(result.audio_path) > 1000

    @pytest.mark.parametrize("provider", ["coqui", "edge"])
    def test_unknown_gender_defaults_with_warning(self, provider, tmp_output):
        service = TTSService(make_config(tmp_output, force_provider=provider))
        result = service.synthesize(text="Hello.", gender="unknown", output_mode="file")
        assert result.status in ("success", "fallback")
        assert result.gender_used == "male"
        assert result.warning is not None
        assert "unknown" in result.warning.lower()

    @pytest.mark.parametrize("provider", ["coqui", "edge"])
    def test_invalid_gender_treated_as_unknown(self, provider, tmp_output):
        service = TTSService(make_config(tmp_output, force_provider=provider))
        result = service.synthesize(text="Hello.", gender="no_face", output_mode="file")
        assert result.status in ("success", "fallback")
        assert result.gender_used == "male"
        assert result.warning is not None

    @pytest.mark.parametrize("provider", ["coqui", "edge"])
    def test_audio_file_has_content(self, provider, tmp_output):
        service = TTSService(make_config(tmp_output, force_provider=provider))
        result = service.synthesize(text=SAMPLE_TEXT, gender="female", output_mode="file")
        assert result.status in ("success", "fallback")
        size = os.path.getsize(result.audio_path)
        assert size > 5000, f"Audio file too small: {size} bytes"

    @pytest.mark.parametrize("provider", ["coqui", "edge"])
    def test_audio_latency_reported(self, provider, tmp_output):
        service = TTSService(make_config(tmp_output, force_provider=provider))
        result = service.synthesize(text="Hello.", gender="female", output_mode="file")
        assert result.status in ("success", "fallback")
        assert result.latency_ms > 0

    @pytest.mark.parametrize("provider", ["coqui", "edge"])
    def test_audio_duration_reported(self, provider, tmp_output):
        service = TTSService(make_config(tmp_output, force_provider=provider))
        result = service.synthesize(text=SAMPLE_TEXT, gender="female", output_mode="file")
        assert result.status in ("success", "fallback")
        assert result.duration_ms > 500


# ── 4. Output Mode Tests ─────────────────────────────────────────────────────

class TestOutputModes:

    @pytest.fixture
    def tmp_output(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_stream_mode_returns_audio_bytes(self, tmp_output):
        service = TTSService(make_config(tmp_output, force_provider="coqui"))
        result = service.synthesize(text="Hello.", gender="female", output_mode="stream")
        assert result.status in ("success", "fallback")
        assert result.audio_bytes is not None
        assert len(result.audio_bytes) > 1000

    def test_play_mode_does_not_crash(self, tmp_output):
        service = TTSService(make_config(tmp_output, force_provider="coqui"))
        result = service.synthesize(text="Hello.", gender="female", output_mode="play")
        assert result.status in ("success", "fallback")
        assert result.played is True

    def test_file_mode_writes_to_specified_path(self, tmp_output):
        out = os.path.join(tmp_output, "custom.wav")
        service = TTSService(make_config(tmp_output, force_provider="coqui"))
        result = service.synthesize(text="Hello.", gender="female", output_mode="file", output_path=out)
        assert result.status in ("success", "fallback")
        assert result.audio_path == os.path.abspath(out)
        assert os.path.exists(out)

    def test_session_id_in_filename(self, tmp_output):
        service = TTSService(make_config(tmp_output, force_provider="coqui"))
        result = service.synthesize(text="Hello.", gender="male", output_mode="file", session_id="test-session-1")
        assert result.status in ("success", "fallback")
        assert "test-session-1" in result.audio_path

    def test_result_is_json_serializable(self, tmp_output):
        service = TTSService(make_config(tmp_output, force_provider="coqui"))
        result = service.synthesize(text="Hello.", gender="male", output_mode="file")
        serialized = json.dumps(result.to_dict())
        assert "status" in serialized


# ── 5. Edge Case Tests ───────────────────────────────────────────────────────

class TestEdgeCases:

    @pytest.fixture
    def tmp_output(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_empty_text_skipped(self, tmp_output):
        service = TTSService(make_config(tmp_output))
        result = service.synthesize(text="", gender="female")
        assert result.status == "skipped"
        assert result.audio_path is None

    def test_whitespace_text_skipped(self, tmp_output):
        service = TTSService(make_config(tmp_output))
        result = service.synthesize(text="   \n  ", gender="male")
        assert result.status == "skipped"

    def test_speed_clamped_low(self, tmp_output):
        service = TTSService(make_config(tmp_output, force_provider="coqui"))
        result = service.synthesize(text="Hello.", gender="female", speed=-1.0, output_mode="file")
        assert result.status in ("success", "fallback")

    def test_speed_clamped_high(self, tmp_output):
        service = TTSService(make_config(tmp_output, force_provider="coqui"))
        result = service.synthesize(text="Hello.", gender="female", speed=5.0, output_mode="file")
        assert result.status in ("success", "fallback")

    def test_all_providers_fail_gracefully(self, tmp_output):
        cfg = make_config(tmp_output)
        cfg["providers"]["priority_order"] = ["nonexistent_provider"]
        service = TTSService(cfg)
        result = service.synthesize(text="Hello.", gender="female")
        assert result.status == "failed"
        assert result.warning == "All providers failed — no audio produced"

    def test_auto_trim_silence_does_not_crash(self, tmp_output):
        cfg = make_config(tmp_output, auto_trim_silence=True, force_provider="coqui")
        service = TTSService(cfg)
        result = service.synthesize(text="Hello.", gender="female", output_mode="file")
        assert result.status in ("success", "fallback")
        assert os.path.exists(result.audio_path)

    def test_long_text_produces_audio(self, tmp_output):
        long_text = "The patient has been showing symptoms of a high fever " * 10
        cfg = make_config(tmp_output, force_provider="coqui")
        service = TTSService(cfg)
        result = service.synthesize(text=long_text, gender="female", output_mode="file")
        assert result.status in ("success", "fallback")
        assert result.duration_ms > 1000


# ── 6. PipelineAdapter Tests ─────────────────────────────────────────────────

class TestPipelineAdapter:

    @pytest.fixture
    def tmp_output(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_adapter_produces_audio(self, tmp_output):
        from lib.pipeline_adapter import PipelineAdapter
        service = TTSService(make_config(tmp_output, force_provider="coqui"))

        class MockLLMResult:
            text_for_tts = "She went to the store."

        class MockVisionResult:
            class Label:
                value = "female"
            label = Label()

        adapter = PipelineAdapter(service)
        result = adapter.process(
            llm_result=MockLLMResult(),
            vision_result=MockVisionResult(),
            output_mode="file",
            output_path=os.path.join(tmp_output, "adapter_test.wav"),
        )
        assert result.status in ("success", "fallback")
        assert result.gender_used == "female"
        assert os.path.exists(result.audio_path)

    def test_adapter_no_face_unknown(self, tmp_output):
        from lib.pipeline_adapter import PipelineAdapter
        service = TTSService(make_config(tmp_output, force_provider="coqui"))

        class MockLLMResult:
            corrected_text = "Hello."

        class MockVisionResult:
            label = "no_face"

        adapter = PipelineAdapter(service)
        result = adapter.process(
            llm_result=MockLLMResult(),
            vision_result=MockVisionResult(),
            output_mode="file",
        )
        assert result.status in ("success", "fallback")
        assert result.gender_used == "male"
        assert result.warning is not None
