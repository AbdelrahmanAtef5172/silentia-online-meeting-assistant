from abc import ABC, abstractmethod

from engine.schemas import BuiltPrompt


class AuthError(Exception):
    """Invalid API key (401/403)."""


class RateLimitError(Exception):
    """Provider rate limit hit (429)."""


class ProviderError(Exception):
    """Any other provider-side error (5xx)."""


class BaseProvider(ABC):
    """
    All providers implement this interface.
    One file per provider in providers/.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider name for logging (e.g., 'openrouter', 'groq')."""

    @abstractmethod
    def call(self, prompt: BuiltPrompt, timeout: float) -> str:
        """
        Synchronous API call.

        Args:
            prompt:   BuiltPrompt with .system and .user fields
            timeout:  Seconds before TimeoutError is raised

        Returns:
            Raw response string from the model

        Raises:
            TimeoutError:    Request exceeded timeout
            RateLimitError:  Provider rate limit hit (429)
            AuthError:       Invalid API key (401/403)
            ProviderError:   Any other provider-side error (5xx)
        """

    @abstractmethod
    async def call_async(self, prompt: BuiltPrompt, timeout: float) -> str:
        """Async variant for batch processing."""

    @abstractmethod
    def health_check(self) -> bool:
        """Returns True if the provider is reachable and authenticated."""
