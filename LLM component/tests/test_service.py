"""
Integration tests for CorrectionService.
Mocks at the HTTP layer; tests full flow including cache, guard, processor.
Marked integration — require API key to run against real providers.
"""

import os
import pytest
import respx
import httpx
from engine.service import CorrectionService
from engine.schemas import CorrectionStatus

REAL_KEY_AVAILABLE = (
    os.environ.get("OPENROUTER_API_KEY")
    and "test-key" not in os.environ["OPENROUTER_API_KEY"]
)

MOCK_RESPONSE = '{"choices": [{"message": {"content": "She went to the store."}}]}'


@pytest.fixture(scope="module")
def service():
    os.environ.setdefault("OPENROUTER_API_KEY", "test-key-for-mocking")
    os.environ.setdefault("GROQ_API_KEY", "test-key-for-mocking")
    os.environ.setdefault("HF_API_TOKEN", "test-key-for-mocking")
    return CorrectionService.from_config(env="development")


@respx.mock
def test_correct_returns_corrected_text(service):
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=MOCK_RESPONSE)
    )
    result = service.correct("she go store")
    assert result.corrected_text == "She went to the store."
    assert result.status == CorrectionStatus.SUCCESS


def test_empty_input_returns_passthrough(service):
    result = service.correct("")
    assert result.status == CorrectionStatus.PASSTHROUGH
    assert result.corrected_text == ""


def test_none_input_returns_passthrough(service):
    result = service.correct(None)
    assert result.status == CorrectionStatus.PASSTHROUGH


@respx.mock
def test_cache_hit_skips_api():
    config = {
        "component": {"mode": "system"},
        "cache": {"enabled": True, "max_size": 512, "ttl_seconds": 3600},
        "prompts": {
            "system": "test",
            "domain_instructions": {},
            "glossary_instruction_template": "",
        },
        "input_guard": {
            "max_input_tokens": 400,
            "min_alpha_ratio": 0.3,
            "passthrough_max_words": 3,
        },
        "output": {
            "max_output_length_ratio": 3.0,
            "strip_markdown": True,
            "strip_quotes": True,
        },
    }
    from engine.service import CorrectionService
    svc = CorrectionService(config)
    mock = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=MOCK_RESPONSE)
    )
    svc.correct("she go store")
    svc.correct("she go store")
    assert mock.call_count == 1


@respx.mock
def test_all_providers_fail_returns_original(service):
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("timeout")
    )
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("timeout")
    )
    respx.post("https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3").mock(
        side_effect=httpx.TimeoutException("timeout")
    )
    result = service.correct("she go store")
    assert result.status == CorrectionStatus.FAILED
    assert result.corrected_text == "she go store"


@respx.mock
def test_correct_with_context(service):
    from engine.schemas import CorrectionContext
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=MOCK_RESPONSE)
    )
    context = CorrectionContext(topic_domain="medical", speaker_gender="female")
    result = service.correct("patient show symptom fever", context=context)
    assert result.status == CorrectionStatus.SUCCESS


@respx.mock
def test_correct_batch_preserves_order(service):
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=MOCK_RESPONSE)
    )
    texts = ["she go store", "i want eat", "he play ball"]
    results = service.correct_batch(texts)
    assert len(results) == 3
    assert results[0].original_text == "she go store"
    assert results[1].original_text == "i want eat"
    assert results[2].original_text == "he play ball"


@pytest.mark.skipif(not REAL_KEY_AVAILABLE, reason="requires a real OPENROUTER_API_KEY")
@pytest.mark.integration
def test_real_provider_call():
    """Requires OPENROUTER_API_KEY in environment."""
    service = CorrectionService.from_config(env="development")
    result = service.correct("i want eat pizza my friend")
    assert result.status in (CorrectionStatus.SUCCESS, CorrectionStatus.CACHED)
    assert len(result.corrected_text) > 0


def test_correct_result_has_metadata(service):
    result = service.correct("")
    assert hasattr(result, "original_text")
    assert hasattr(result, "corrected_text")
    assert hasattr(result, "status")
    assert hasattr(result, "latency_ms")
    assert hasattr(result, "to_dict")


def test_result_to_dict_includes_all_fields(service):
    result = service.correct("")
    d = result.to_dict()
    assert "corrected_text" in d
    assert "original_text" in d
    assert "status" in d
    assert "latency_ms" in d


def test_punctuation_only_returns_passthrough(service):
    result = service.correct("!!! ???")
    assert result.status == CorrectionStatus.PASSTHROUGH


def test_number_heavy_returns_passthrough(service):
    result = service.correct("1234 5678 9.99")
    assert result.status == CorrectionStatus.PASSTHROUGH
