"""
core/response_processor.py
───────────────────────────
Takes the raw string response from a provider and extracts the corrected text.
Defends against malformed, verbose, or off-topic LLM responses.
"""

import re
from typing import Optional

from engine.schemas import ProcessedResponse


class ResponseProcessor:
    """
    Responsibilities:
    1. Extraction: Strip any preamble the LLM added
    2. Validation: Confirm the result is non-empty and appears to be natural language
    3. Sanitization: Remove markdown, code blocks, quotes the LLM may have wrapped output in
    4. Length check: If output > 3x input length, flag it as suspicious
    5. Fallback: If extraction fails, return original input with quality="fallback"
    """

    _DEFAULT_MERGES = {
        "willgo": "will go", "willcome": "will come",
        "grandparentstomorrow": "grandparents tomorrow",
        "thecontract": "the contract", "ofhigh": "of high",
    }

    def __init__(self, config: dict):
        self._max_ratio = config.get("max_output_length_ratio", 3.0)
        self._strip_markdown = config.get("strip_markdown", True)
        self._strip_quotes = config.get("strip_quotes", True)
        self._known_merges = config.get("known_merges", self._DEFAULT_MERGES)

    def process(self, raw_response: str, original_input: str) -> ProcessedResponse:
        """
        Process a raw LLM response into a cleaned correction.

        Args:
            raw_response:   Raw string returned by the LLM provider
            original_input: Original SLR input (used for fallback and length check)

        Returns:
            ProcessedResponse with corrected_text, quality, and extraction_method
        """
        if not raw_response or not raw_response.strip():
            return ProcessedResponse(
                corrected_text=original_input,
                quality="fallback",
                extraction_method="empty_response",
            )

        text = raw_response.strip()

        text, method = self._extract(text)

        text = self._sanitize(text)

        if not text or not text.strip():
            return ProcessedResponse(
                corrected_text=original_input,
                quality="fallback",
                extraction_method=f"{method}_resulted_in_empty",
            )

        text = text.strip()
        text = self._fix_merged_words(text)

        quality = self._assess_quality(text, original_input)

        return ProcessedResponse(
            corrected_text=text,
            quality=quality,
            extraction_method=method,
        )

    def _extract(self, text: str) -> tuple[str, str]:
        """
        Strip preamble and extract the actual corrected sentence.

        Returns:
            (extracted_text, method_name)
        """
        patterns = [
            (r"^Here\s+is\s+the\s+corrected\s+(sentence|text|version)[:\s]*\n?", "strip_preamble"),
            (r"^Corrected\s+(sentence|text|version)[:\s]*\n?", "strip_preamble"),
            (r"^The\s+corrected\s+(sentence|text|version)\s+is[:\s]*\n?", "strip_preamble"),
            (r"^Correction[:\s]*\n?", "strip_preamble"),
        ]

        for pattern, method in patterns:
            stripped = re.sub(pattern, "", text, flags=re.IGNORECASE)
            if stripped != text:
                return stripped.strip(), method

        return text, "direct"

    def _sanitize(self, text: str) -> str:
        """Remove markdown, code blocks, quotes, and other artifacts."""
        text = re.sub(r"```(\w*)\n?([\s\S]*?)```", r"\2", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)

        if self._strip_markdown:
            text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
            text = re.sub(r"\*([^*]+)\*", r"\1", text)
            text = re.sub(r"__([^_]+)__", r"\1", text)
            text = re.sub(r"_([^_]+)_", r"\1", text)

        if self._strip_quotes:
            text = re.sub(r'^["\']|["\']$', "", text)
            text = re.sub(r'\u201c|\u201d', "", text)
            text = re.sub(r'\u2018|\u2019', "", text)

        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _assess_quality(self, corrected: str, original: str) -> str:
        """
        Assess the quality of the correction.

        Returns:
            "good"       — correction looks valid
            "acceptable" — minor issues but usable
            "fallback"   — unusable, should return original
        """
        if not corrected:
            return "fallback"

        original_words = len(original.split()) if original.strip() else 1
        corrected_words = len(corrected.split())

        if corrected_words > original_words * self._max_ratio:
            return "fallback"

        if corrected.lower() == original.lower():
            return "acceptable"

        if corrected_words < 1:
            return "fallback"

        return "good"

    def _fix_merged_words(self, text: str) -> str:
        """Fix common word-merging errors from ASR output."""
        if not self._known_merges:
            return text

        pattern = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in self._known_merges) + r")\b",
            re.IGNORECASE,
        )

        def _replace(m: re.Match) -> str:
            key = m.group(1).lower()
            return self._known_merges.get(key, m.group(1))

        text = pattern.sub(_replace, text)
        text = re.sub(r"\b(did|do|does|can|will)nt\b", r"\1 not", text, flags=re.IGNORECASE)
        return text.strip()
