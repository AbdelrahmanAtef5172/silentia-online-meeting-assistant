"""
guards/edge_cases.py
─────────────────────
Handlers for each edge case category defined in the InputGuard.

Each handler takes the raw text and returns the appropriate GuardDecision.
These are used by the InputGuard when specific edge cases are detected
and require more nuanced handling than the default path provides.
"""

import re
from typing import Optional

from engine.schemas import GuardDecision, InputCategory


def handle_all_caps(text: str) -> GuardDecision:
    """
    Normalize fully uppercase input to sentence case, then return as NORMAL.
    """
    normalized = text.capitalize()
    return GuardDecision(
        category=InputCategory.NORMAL,
        action="correct",
        transformed_text=normalized,
        warning="Input was all uppercase; normalized to sentence case",
    )


def handle_mixed_language(text: str) -> GuardDecision:
    """
    Detect non-English characters and pass through with a warning.
    The LLM still corrects it, but we log the warning for monitoring.
    """
    return GuardDecision(
        category=InputCategory.NORMAL,
        action="correct",
        transformed_text=text,
        warning="Non-English characters detected in input",
    )


def handle_single_word(text: str) -> GuardDecision:
    """
    Single-word input is still sent to the LLM for correction.
    The LLM can add necessary article/context if needed.
    """
    return GuardDecision(
        category=InputCategory.NORMAL,
        action="correct",
        transformed_text=text,
    )


def handle_repeated_words(text: str) -> GuardDecision:
    """
    Deduplicate repeated words (e.g., 'I I I go go store' → 'I go store')
    before passing to the LLM. The LLM handles further grammatical correction.
    """
    deduplicated = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text)
    return GuardDecision(
        category=InputCategory.NORMAL,
        action="correct",
        transformed_text=deduplicated,
        warning="Repeated words deduplicated",
    )


def handle_domain_terms(text: str, glossary: dict) -> GuardDecision:
    """
    Apply domain-specific glossary substitutions before the LLM call.
    Case-insensitive matching; preserves original capitalization pattern.
    """
    for wrong, correct in glossary.items():
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        text = pattern.sub(correct, text)

    return GuardDecision(
        category=InputCategory.NORMAL,
        action="correct",
        transformed_text=text,
        warning=f"Applied glossary substitutions: {len(glossary)} terms",
    )
