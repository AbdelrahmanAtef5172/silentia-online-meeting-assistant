"""
core/provider_router.py
────────────────────────
Routes correction requests to providers in priority order.
Handles retry, timeout, and fallback transparently.
"""

import asyncio
import concurrent.futures
import logging
from typing import Optional, Callable, Awaitable, Union

from model_providers.base import BaseProvider, AuthError, RateLimitError
from engine.schemas import BuiltPrompt

logger = logging.getLogger(__name__)


class AllProvidersFailedError(Exception):
    """All providers failed to produce a valid response."""


class ProviderRouter:
    """
    Tries providers in order: primary → secondary → tertiary.
    Each provider gets `max_retries` attempts before the next is tried.
    If all providers fail, raises AllProvidersFailedError.
    """

    def __init__(self, providers: list, config: dict):
        self._providers = providers
        self._max_retries = config.get("max_retries", 2)
        self._timeout = config.get("timeout_seconds", 10.0)
        self._last_provider_used: Optional[str] = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    @property
    def last_provider_used(self) -> Optional[str]:
        return self._last_provider_used

    def route(self, prompt: BuiltPrompt) -> str:
        """
        Synchronous routing with fallback.

        Tries providers in priority order. Each provider gets max_retries
        attempts. On RateLimitError the provider is skipped immediately.
        On any other error (timeout, server error) it retries.

        A ThreadPoolExecutor timeout protects against providers that
        ignore the timeout parameter.

        AuthError is fatal — raised immediately.

        Raises:
            AllProvidersFailedError if no provider succeeds
        """
        last_exception: Optional[Exception] = None

        for provider in self._providers:
            for attempt in range(self._max_retries):
                try:
                    response = self._call_sync(provider, prompt)
                    if response and response.strip():
                        self._last_provider_used = provider.name
                        return response
                except AuthError:
                    raise
                except RateLimitError:
                    logger.warning(
                        "%s rate limited, skipping to next provider",
                        provider.name,
                    )
                    break
                except Exception as e:
                    logger.warning(
                        "%s error: %s (attempt %d/%d)",
                        provider.name,
                        e,
                        attempt + 1,
                        self._max_retries,
                    )
                    last_exception = e
                    continue

        raise AllProvidersFailedError(
            f"All {len(self._providers)} providers failed. Last error: {last_exception}"
        )

    async def route_async(self, prompt: BuiltPrompt) -> str:
        """
        Async routing with fallback (for batch processing).
        Uses asyncio.wait_for to enforce per-call timeouts.
        """
        last_exception: Optional[Exception] = None

        for provider in self._providers:
            for attempt in range(self._max_retries):
                try:
                    response = await asyncio.wait_for(
                        provider.call_async(prompt, timeout=self._timeout),
                        timeout=self._timeout + 5,
                    )
                    if response and response.strip():
                        self._last_provider_used = provider.name
                        return response
                except AuthError:
                    raise
                except RateLimitError:
                    logger.warning(
                        "%s rate limited, skipping to next provider",
                        provider.name,
                    )
                    break
                except asyncio.TimeoutError:
                    logger.warning(
                        "%s timed out (attempt %d/%d)",
                        provider.name,
                        attempt + 1,
                        self._max_retries,
                    )
                    last_exception = TimeoutError(f"{provider.name} timed out")
                    continue
                except Exception as e:
                    logger.warning(
                        "%s error: %s (attempt %d/%d)",
                        provider.name,
                        e,
                        attempt + 1,
                        self._max_retries,
                    )
                    last_exception = e
                    continue

        raise AllProvidersFailedError(
            f"All {len(self._providers)} providers failed. Last error: {last_exception}"
        )

    def _call_sync(self, provider: BaseProvider, prompt: BuiltPrompt) -> str:
        """
        Call a provider synchronously with a hard timeout via ThreadPoolExecutor.
        This guards against providers that ignore the timeout parameter.
        """
        future = self._executor.submit(provider.call, prompt, timeout=self._timeout)
        try:
            return future.result(timeout=self._timeout + 5)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"{provider.name} call timed out after {self._timeout}s")
