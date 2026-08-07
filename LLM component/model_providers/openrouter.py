"""
providers/openrouter.py
────────────────────────
OpenRouter provider implementation.

API base URL:  https://openrouter.ai/api/v1
Endpoint:      POST /chat/completions
Auth:          Bearer {OPENROUTER_API_KEY}
Compatibility: Full OpenAI chat completions schema

Model is configured via configs/config.yaml (secondary/tertiary slots).
OpenRouter-specific headers (recommended for free tier priority):
  HTTP-Referer: https://github.com/your-org/llm_component
  X-Title: LLM Grammar Correction Component
"""

import httpx

from model_providers.base import BaseProvider, AuthError, RateLimitError, ProviderError
from engine.schemas import BuiltPrompt
from engine.retry import with_retry


class OpenRouterProvider(BaseProvider):

    BASE_URL = "https://openrouter.ai/api/v1"
    ENDPOINT = "/chat/completions"

    def __init__(self, api_key: str, model: str, config: dict):
        self._api_key = api_key
        self._model = model
        self._temperature = config.get("temperature", 0.1)
        self._max_tokens = config.get("max_tokens", 512)
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": config.get("http_referer", ""),
            "X-Title": "LLM Grammar Correction Component",
        }

    @property
    def name(self) -> str:
        return "openrouter"

    def call(self, prompt: BuiltPrompt, timeout: float = 10.0) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.BASE_URL}{self.ENDPOINT}",
                headers=self._headers,
                json=payload,
            )
        self._raise_for_status(response)
        return response.json()["choices"][0]["message"]["content"]

    async def call_async(self, prompt: BuiltPrompt, timeout: float = 10.0) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.BASE_URL}{self.ENDPOINT}",
                headers=self._headers,
                json=payload,
            )
        self._raise_for_status(response)
        return response.json()["choices"][0]["message"]["content"]

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(
                    f"{self.BASE_URL}/models",
                    headers=self._headers,
                )
            return r.status_code == 200
        except Exception:
            return False

    def _raise_for_status(self, response: httpx.Response):
        if response.status_code == 401:
            raise AuthError("OpenRouter: invalid API key")
        if response.status_code == 429:
            raise RateLimitError("OpenRouter: rate limit hit")
        if response.status_code >= 500:
            raise ProviderError(f"OpenRouter server error: {response.status_code}")
        response.raise_for_status()
