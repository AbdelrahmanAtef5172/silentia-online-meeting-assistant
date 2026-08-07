"""
Unit tests for ProviderRouter fallback logic.
Uses respx to mock httpx calls — no real API calls made.
"""

import pytest
import respx
import httpx
from engine.provider_router import ProviderRouter, AllProvidersFailedError
from engine.schemas import BuiltPrompt

MOCK_PROMPT = BuiltPrompt(system="You are a corrector.", user="Input: she go store")
MOCK_RESPONSE = '{"choices": [{"message": {"content": "She went to the store."}}]}'
MOCK_HF_RESPONSE = '{"generated_text": "She went to the store."}'


def _make_mock_router():
    """Construct router with minimal mock providers."""
    from model_providers.openrouter import OpenRouterProvider
    from model_providers.groq import GroqProvider
    from model_providers.huggingface import HuggingFaceProvider

    providers = [
        OpenRouterProvider(
            api_key="test-key",
            model="openrouter/free",
            config={"temperature": 0.1, "max_tokens": 512},
        ),
        GroqProvider(
            api_key="test-key",
            model="llama-3.1-8b-instant",
            config={"temperature": 0.1, "max_tokens": 512},
        ),
        HuggingFaceProvider(
            api_key="test-key",
            model="mistralai/Mistral-7B-Instruct-v0.3",
            config={"temperature": 0.1, "max_tokens": 512},
        ),
    ]
    return ProviderRouter(providers, {
        "max_retries": 2,
        "timeout_seconds": 5.0,
    })


@respx.mock
def test_primary_provider_succeeds():
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=MOCK_RESPONSE)
    )
    router = _make_mock_router()
    result = router.route(MOCK_PROMPT)
    assert "store" in result
    assert router.last_provider_used == "openrouter"


@respx.mock
def test_falls_back_to_secondary_on_primary_timeout():
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("timeout")
    )
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=MOCK_RESPONSE)
    )
    router = _make_mock_router()
    result = router.route(MOCK_PROMPT)
    assert "store" in result
    assert router.last_provider_used == "groq"


@respx.mock
def test_falls_back_to_tertiary_on_secondary_failure():
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("timeout")
    )
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("timeout")
    )
    respx.post("https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3").mock(
        return_value=httpx.Response(200, text=MOCK_HF_RESPONSE)
    )
    router = _make_mock_router()
    result = router.route(MOCK_PROMPT)
    assert "store" in result
    assert router.last_provider_used == "huggingface"


@respx.mock
def test_raises_when_all_providers_fail():
    for url in [
        "https://openrouter.ai/api/v1/chat/completions",
        "https://api.groq.com/openai/v1/chat/completions",
        "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
    ]:
        respx.post(url).mock(side_effect=httpx.TimeoutException("timeout"))

    router = _make_mock_router()
    with pytest.raises(AllProvidersFailedError):
        router.route(MOCK_PROMPT)


@respx.mock
def test_stops_on_auth_error():
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(401, text='{"error": "unauthorized"}')
    )
    router = _make_mock_router()
    with pytest.raises(Exception):
        router.route(MOCK_PROMPT)


@respx.mock
def test_retries_on_rate_limit_then_breaks():
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(429, text='{"error": "rate limited"}')
    )
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=MOCK_RESPONSE)
    )
    router = _make_mock_router()
    result = router.route(MOCK_PROMPT)
    assert "store" in result
    assert router.last_provider_used == "groq"


@respx.mock
def test_retry_on_server_error():
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(503, text='{"error": "service unavailable"}')
    )
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=MOCK_RESPONSE)
    )
    router = _make_mock_router()
    result = router.route(MOCK_PROMPT)
    assert "store" in result


@respx.mock
def test_primary_succeeds_on_retry_after_failure():
    """Primary fails once on timeout, succeeds on retry."""
    mock_route = respx.post("https://openrouter.ai/api/v1/chat/completions")
    mock_route.mock(
        side_effect=[
            httpx.TimeoutException("timeout"),
            httpx.Response(200, text=MOCK_RESPONSE),
        ]
    )
    router = _make_mock_router()
    result = router.route(MOCK_PROMPT)
    assert "store" in result
    assert router.last_provider_used == "openrouter"
