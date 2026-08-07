"""
providers/huggingface.py
─────────────────────────
Hugging Face Inference API provider.

API base URL:  https://api-inference.huggingface.co/models
Auth:          Bearer {HF_API_TOKEN}

Notes:
  - HF Inference API uses a different request schema than OpenAI.
  - For instruct models, format input as a single string with
    [INST] ... [/INST] tags (Mistral format).
  - Response is {"generated_text": "..."} or [{"generated_text": "..."}].
  - Cold start is possible (503 while model is loading).
"""

import httpx

from model_providers.base import (
    BaseProvider,
    AuthError,
    RateLimitError,
    ProviderError,
)
from engine.schemas import BuiltPrompt


class HuggingFaceProvider(BaseProvider):

    BASE_URL  = "https://router.huggingface.co/v1"

    def __init__(self, api_key: str, model: str, config: dict):
        self._api_key = api_key
        self._model = model
        self._temperature = config.get("temperature", 0.1)
        self._max_tokens = config.get("max_tokens", 512)

        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @property
    def name(self) -> str:
        return "huggingface"

    def call(self, prompt: BuiltPrompt, timeout: float = 25.0) -> str:

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": prompt.system,
                },
                {
                    "role": "user",
                    "content": prompt.user,
                },
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._headers,
                json=payload,
            )

        self._raise_for_status(response)

        data = response.json()

        return data["choices"][0]["message"]["content"]

    async def call_async(self, prompt: BuiltPrompt, timeout: float = 25.0) -> str:

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": prompt.system,
                },
                {
                    "role": "user",
                    "content": prompt.user,
                },
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._headers,
                json=payload,
            )

        self._raise_for_status(response)

        data = response.json()

        return data["choices"][0]["message"]["content"]

    def health_check(self) -> bool:
        try:
            payload = {
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ],
                "max_tokens": 1,
            }

            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self._headers,
                    json=payload,
                )

            return response.status_code == 200

        except Exception as e:
            print(e)
            return False

    def _raise_for_status(self, response: httpx.Response):
        if response.status_code == 401:
            raise AuthError("HuggingFace: invalid API key")

        if response.status_code == 429:
            raise RateLimitError("HuggingFace: rate limit hit")

        if response.status_code == 503:
            raise ProviderError("HuggingFace: model is loading (cold start)")

        if response.status_code >= 500:
            raise ProviderError(
                f"HuggingFace server error: {response.status_code}"
            )

        response.raise_for_status()