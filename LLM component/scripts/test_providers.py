"""
scripts/test_providers.py
─────────────────────────
Manually verify all provider connections.

Usage:
    python scripts/test_providers.py

Runs health checks and a simple correction on each configured provider
to verify API keys are valid and endpoints are reachable.
"""

import os
import sys

from dotenv import load_dotenv
from engine.config_loader import load_config
from model_providers.openrouter import OpenRouterProvider
from model_providers.groq import GroqProvider
from model_providers.huggingface import HuggingFaceProvider


def main():
    load_dotenv()
    env = os.environ.get("ENV", "development")
    config = load_config(env=env)
    providers_config = config.get("providers", {})

    results = []

    primary_cfg = providers_config.get("primary", {})
    api_key = os.environ.get("GROQ_API_KEY", "")
    if api_key:
        provider = GroqProvider(
            api_key=api_key,
            model=primary_cfg.get("model", "llama-3.1-8b-instant"),
            config=primary_cfg,
        )
        results.append(_check_provider(provider, "Groq"))
    else:
        results.append(("Groq", False, "GROQ_API_KEY not set (primary)"))

    secondary_cfg = providers_config.get("secondary", {})
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if api_key:
        provider = OpenRouterProvider(
            api_key=api_key,
            model=secondary_cfg.get("model", "openai/gpt-oss-120b:free"),
            config=secondary_cfg,
        )
        results.append(_check_provider(provider, "OpenRouter"))
    else:
        results.append(("OpenRouter", False, "OPENROUTER_API_KEY not set"))

    tertiary_cfg = providers_config.get("tertiary", {})
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if api_key and tertiary_cfg.get("model"):
        provider = OpenRouterProvider(
            api_key=api_key,
            model=tertiary_cfg["model"],
            config=tertiary_cfg,
        )
        results.append(_check_provider(provider, "OpenRouter free"))

    quaternary_cfg = providers_config.get("quaternary", {})
    api_key = os.environ.get("HF_API_TOKEN", "")
    if api_key:
        provider = HuggingFaceProvider(
            api_key=api_key,
            model=quaternary_cfg.get("model", "mistralai/Mistral-7B-Instruct-v0.3"),
            config=quaternary_cfg,
        )
        results.append(_check_provider(provider, "HuggingFace"))
    else:
        results.append(("HuggingFace", False, "HF_API_TOKEN not set (fallback disabled)"))

    for name, ok, msg in results:
        icon = "✓" if ok else "✗"
        print(f"{icon} {name:15s} — {msg}")

    all_ok = all(ok for _, ok, _ in results)
    sys.exit(0 if all_ok else 1)


def _check_provider(provider, name: str):
    try:
        ok = provider.health_check()
        if ok:
            return (name, True, "reachable, authenticated")
        else:
            return (name, False, "health check failed")
    except Exception as e:
        return (name, False, str(e))


if __name__ == "__main__":
    main()
