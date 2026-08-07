"""
core/service.py
────────────────
The single entry point for all correction requests.
Orchestrates the pipeline of guard → cache → prompt → route → process → format.
"""

import asyncio
import time
import logging
import os
from typing import Optional, List

from dotenv import load_dotenv

from engine.schemas import (
    CorrectionResult,
    CorrectionStatus,
    CorrectionContext,
    InputCategory,
)
from engine.config_loader import load_config
from engine.input_guard import InputGuard
from engine.response_cache import ResponseCache
from engine.prompt_builder import PromptBuilder
from engine.provider_router import ProviderRouter, AllProvidersFailedError
from engine.response_processor import ResponseProcessor
from engine.output_formatter import OutputFormatter
from model_providers.openrouter import OpenRouterProvider
from model_providers.groq import GroqProvider
from model_providers.huggingface import HuggingFaceProvider

logger = logging.getLogger(__name__)


class CorrectionService:
    """
    The single public API for grammar and vocabulary correction.

    Orchestrates the full pipeline:
      InputGuard → ResponseCache → PromptBuilder → ProviderRouter
      → ResponseProcessor → OutputFormatter

    Never raises — returns a degraded CorrectionResult on failure.
    """

    def __init__(self, config: dict):
        self._config = config

        self._guard = InputGuard(config.get("input_guard", {}))
        self._cache_enabled = config.get("cache", {}).get("enabled", True)
        self._cache = ResponseCache(
            max_size=config.get("cache", {}).get("max_size", 512),
            ttl_seconds=config.get("cache", {}).get("ttl_seconds", 3600),
        )
        self._prompt_builder = PromptBuilder(config)
        self._router = self._build_router(config)
        self._processor = ResponseProcessor(config.get("output", {}))
        self._formatter = OutputFormatter()

    def correct(
        self,
        raw_text: str,
        context: Optional[CorrectionContext] = None,
    ) -> CorrectionResult:
        """
        Correct grammar and vocabulary in raw_text.

        Args:
            raw_text:  Noisy text from SLR output
            context:   Optional context (topic domain, speaker info from Vision)

        Returns:
            CorrectionResult with corrected text, confidence, and metadata.
            Never raises — returns a degraded result on failure.
        """
        start_time = time.time()

        decision = self._guard.inspect(raw_text)

        if decision.action in ("reject", "pass"):
            return self._formatter.make_passthrough(
                text=decision.transformed_text,
                input_category=decision.category,
                warning=decision.warning,
            )

        if self._cache_enabled:
            cached = self._cache.get(raw_text)
            if cached is not None:
                # Mark as cache hit
                cached.from_cache = True
                cached.latency_ms = (time.time() - start_time) * 1000
                return cached

        context = context or CorrectionContext()
        prompt = self._prompt_builder.build(decision, context)

        try:
            raw_response = self._router.route(prompt)
            latency_ms = (time.time() - start_time) * 1000
        except AllProvidersFailedError as e:
            logger.error("All providers failed: %s", e)
            return self._formatter.make_failed(
                original_text=raw_text,
                warning=str(e),
            )
        except Exception as e:
            logger.error("Unexpected routing error: %s", e)
            return self._formatter.make_failed(
                original_text=raw_text,
                warning=f"Unexpected error: {e}",
            )

        processed = self._processor.process(raw_response, raw_text)

        result = self._formatter.format(
            processed=processed,
            original_input=raw_text,
            context=context,
            provider_used=self._router.last_provider_used,
            latency_ms=latency_ms,
            from_cache=False,
            input_category=decision.category,
            warning=decision.warning,
        )

        if self._cache_enabled and result.status != CorrectionStatus.FAILED:
            result.from_cache = False
            self._cache.set(raw_text, result)

        return result

    async def correct_async(
        self,
        raw_text: str,
        context: Optional[CorrectionContext] = None,
    ) -> CorrectionResult:
        """
        Async version of correct(). Uses the provider's async pathway.
        Never raises — returns a degraded result on failure.
        """
        start_time = time.time()

        decision = self._guard.inspect(raw_text)

        if decision.action in ("reject", "pass"):
            return self._formatter.make_passthrough(
                text=decision.transformed_text,
                input_category=decision.category,
                warning=decision.warning,
            )

        if self._cache_enabled:
            cached = self._cache.get(raw_text)
            if cached is not None:
                cached.from_cache = True
                cached.latency_ms = (time.time() - start_time) * 1000
                return cached

        context = context or CorrectionContext()
        prompt = self._prompt_builder.build(decision, context)

        try:
            raw_response = await self._router.route_async(prompt)
            latency_ms = (time.time() - start_time) * 1000
        except AllProvidersFailedError as e:
            logger.error("All providers failed: %s", e)
            return self._formatter.make_failed(
                original_text=raw_text,
                warning=str(e),
            )
        except Exception as e:
            logger.error("Unexpected routing error: %s", e)
            return self._formatter.make_failed(
                original_text=raw_text,
                warning=f"Unexpected error: {e}",
            )

        processed = self._processor.process(raw_response, raw_text)

        result = self._formatter.format(
            processed=processed,
            original_input=raw_text,
            context=context,
            provider_used=self._router.last_provider_used,
            latency_ms=latency_ms,
            from_cache=False,
            input_category=decision.category,
            warning=decision.warning,
        )

        if self._cache_enabled and result.status != CorrectionStatus.FAILED:
            result.from_cache = False
            self._cache.set(raw_text, result)

        return result

    def correct_batch(
        self,
        texts: List[str],
        contexts: Optional[List[Optional[CorrectionContext]]] = None,
        max_concurrency: int = 4,
    ) -> List[CorrectionResult]:
        """
        Correct multiple texts. Uses asyncio for concurrent provider calls.
        Order of results matches order of input.

        If called from a running event loop, falls back to sequential
        processing to avoid nested event loop errors.
        """
        if contexts is None:
            contexts = [None] * len(texts)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            logger.warning(
                "correct_batch called from async context; running sequentially. "
                "Use correct_batch_async() for true concurrency."
            )
            return [self.correct(t, c) for t, c in zip(texts, contexts)]

        return asyncio.run(
            self.correct_batch_async(texts, contexts, max_concurrency)
        )

    async def correct_batch_async(
        self,
        texts: List[str],
        contexts: Optional[List[Optional[CorrectionContext]]] = None,
        max_concurrency: int = 4,
    ) -> List[CorrectionResult]:
        """
        Async batch correction. Uses asyncio.to_thread to avoid blocking
        the event loop during synchronous provider calls.

        Order of results matches order of input.
        """
        if contexts is None:
            contexts = [None] * len(texts)

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _correct_one(
            text: str,
            context: Optional[CorrectionContext],
        ) -> CorrectionResult:
            async with semaphore:
                return await asyncio.to_thread(self.correct, text, context)

        tasks = [_correct_one(t, c) for t, c in zip(texts, contexts)]
        return await asyncio.gather(*tasks)

    def _build_router(self, config: dict) -> ProviderRouter:
        """Build ProviderRouter from config, instantiating only configured providers."""
        providers_config = config.get("providers", {})
        providers = []

        # 1. Primary: Groq
        primary_cfg = providers_config.get("primary", {})
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            providers.append(GroqProvider(
                api_key=groq_key,
                model=primary_cfg.get("model", "llama-3.1-8b-instant"),
                config=primary_cfg,
            ))

        # 2. Secondary: OpenRouter (pinned model)
        or_key = os.environ.get("OPENROUTER_API_KEY", "")
        secondary_cfg = providers_config.get("secondary", {})
        if or_key:
            providers.append(OpenRouterProvider(
                api_key=or_key,
                model=secondary_cfg.get("model", "openai/gpt-oss-120b:free"),
                config=secondary_cfg,
            ))

        # 3. Tertiary: OpenRouter fallback (auto-routing)
        tertiary_cfg = providers_config.get("tertiary", {})
        if or_key and tertiary_cfg.get("model"):
            providers.append(OpenRouterProvider(
                api_key=or_key,
                model=tertiary_cfg["model"],
                config=tertiary_cfg,
            ))

        # 4. Quaternary: HuggingFace
        quaternary_cfg = providers_config.get("quaternary", {})
        hf_key = os.environ.get("HF_API_TOKEN", "")
        if hf_key:
            providers.append(HuggingFaceProvider(
                api_key=hf_key,
                model=quaternary_cfg.get("model", "mistralai/Mistral-7B-Instruct-v0.3"),
                config=quaternary_cfg,
            ))

        router_config = {
            "max_retries": primary_cfg.get("max_retries", 2),
            "timeout_seconds": primary_cfg.get("timeout_seconds", 8.0),
        }

        return ProviderRouter(providers, router_config)

    @classmethod
    def from_config(cls, env: Optional[str] = None) -> "CorrectionService":
        """
        Load config from configs/config.yaml and instantiate.

        Args:
            env: Environment name used to select config overrides.
                 e.g. "dev", "staging", "production". Pass None for defaults.

        Returns:
            Configured CorrectionService instance.
        """
        load_dotenv()
        config = load_config(env=env)
        return cls(config)
