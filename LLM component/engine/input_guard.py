"""
guards/input_guard.py
──────────────────────
Validates and classifies all incoming text before it reaches the prompt
builder or cache. Returns a GuardDecision that tells the service how to
handle the input.
"""

import re
from typing import Optional

from engine.schemas import InputCategory, GuardDecision


class InputGuard:
    """
    Validates and classifies all incoming text.
    Returns a GuardDecision — never raises.
    """

    def __init__(self, config: dict):
        self._max_input_tokens = config.get("max_input_tokens", 400)
        self._min_alpha_ratio = config.get("min_alpha_ratio", 0.3)
        self._passthrough_max_words = config.get("passthrough_max_words", 3)

    def inspect(self, raw_text: Optional[str]) -> GuardDecision:
        """
        Classification order (first match wins):
          1. EMPTY           → input is None, empty string, or only whitespace
          2. PASSTHROUGH     → input is already correct (heuristic quality check)
          3. PUNCTUATION_ONLY → input contains no alphabetic characters
          4. NUMBER_HEAVY    → >60% of tokens are numeric/special
          5. TOO_LONG        → input exceeds max_input_tokens threshold
          6. NORMAL          → standard correction path

        Returns:
            GuardDecision with category, action, transformed_text, and optional warning.
        """
        if raw_text is None or not raw_text.strip():
            return GuardDecision(
                category=InputCategory.EMPTY,
                action="reject",
                transformed_text=raw_text or "",
            )

        text = raw_text.strip()

        if self._is_punctuation_only(text):
            return GuardDecision(
                category=InputCategory.PUNCTUATION_ONLY,
                action="pass",
                transformed_text=text,
            )

        if self._is_number_heavy(text):
            return GuardDecision(
                category=InputCategory.NUMBER_HEAVY,
                action="pass",
                transformed_text=text,
            )

        if self._is_too_long(text):
            truncated = self._truncate(text, self._max_input_tokens)
            return GuardDecision(
                category=InputCategory.TOO_LONG,
                action="truncate",
                transformed_text=truncated,
                warning=f"Input truncated from {len(text.split())} to {self._max_input_tokens} tokens",
            )

        if self._is_likely_correct(text):
            return GuardDecision(
                category=InputCategory.PASSTHROUGH,
                action="pass",
                transformed_text=text,
            )

        return GuardDecision(
            category=InputCategory.NORMAL,
            action="correct",
            transformed_text=text,
        )

    def _is_punctuation_only(self, text: str) -> bool:
        stripped = text.replace(" ", "").replace("\n", "").replace("\t", "")
        if not stripped:
            return True
        return not any(c.isalpha() for c in stripped) and not any(c.isdigit() for c in stripped)

    def _is_number_heavy(self, text: str) -> bool:
        tokens = text.split()
        if not tokens:
            return False
        if not any(c.isdigit() for c in text):
            return False
        alpha_count = sum(1 for t in tokens if any(c.isalpha() for c in t))
        ratio = alpha_count / len(tokens)
        return ratio < self._min_alpha_ratio

    def _is_too_long(self, text: str) -> bool:
        return len(text.split()) > self._max_input_tokens

    def _truncate(self, text: str, max_tokens: int) -> str:
        tokens = text.split()
        return " ".join(tokens[:max_tokens])

    def _is_likely_correct(self, text: str) -> bool:
        """
        Lightweight check — not a grammar checker.
        Returns True if the text looks like it doesn't need correction.
        This is an optimization heuristic, not a guarantee.
        """
        words = text.split()

        if len(words) <= 2:
            return False

        if not text[0].isupper():
            return False

        if text[-1] not in ".!?":
            return False

        function_words = {
            "the", "a", "an", "is", "are", "was", "were",
            "have", "has", "had", "will", "would", "can",
            "could", "should", "to", "of", "in", "on", "at",
        }
        word_set = {w.lower().strip(".,!?") for w in words}
        if len(word_set & function_words) == 0:
            return False

        return True

    
