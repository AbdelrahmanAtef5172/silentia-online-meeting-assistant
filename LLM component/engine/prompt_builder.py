"""
core/prompt_builder.py
───────────────────────
Constructs the system prompt and user message sent to the LLM.
This is the most impactful file for correction quality — treat it with care.
"""

import re
from typing import Optional

from engine.schemas import BuiltPrompt, GuardDecision, CorrectionContext


class PromptBuilder:
    """
    Builds system + user prompts from a GuardDecision and optional CorrectionContext.

    System prompt is stored in config (not hardcoded) so it can be tuned
    without code changes. Context injection is additive — if context is None,
    the prompt works without it.
    """

    _REQUIRED_PLACEHOLDERS = ("{domain_instruction}", "{glossary_instruction}")

    def __init__(self, config: dict):
        prompts = config.get("prompts", {})
        self._system_template = prompts.get("system", "")
        self._domain_instructions = prompts.get("domain_instructions", {})
        self._glossary_template = prompts.get(
            "glossary_instruction_template",
            "The following word substitutions are domain-specific and must be applied:\n{glossary_pairs}",
        )

        self._validate_template()

    def _validate_template(self) -> None:
        """Warn if the system template is missing expected placeholders."""
        missing = [
            p for p in self._REQUIRED_PLACEHOLDERS
            if p not in self._system_template
        ]
        if missing:
            import logging
            logging.getLogger(__name__).warning(
                "System prompt template is missing placeholders: %s", missing
            )

    def build(
        self,
        decision: GuardDecision,
        context: Optional[CorrectionContext] = None,
    ) -> BuiltPrompt:
        """
        Build a complete prompt from a guard decision and optional context.

        Args:
            decision: GuardDecision from InputGuard (contains transformed_text)
            context:  Optional CorrectionContext with domain, glossary, etc.

        Returns:
            BuiltPrompt with system and user fields ready for the LLM.
        """
        text = decision.transformed_text
        context = context or CorrectionContext()

        text = self._apply_glossary(text, context.custom_glossary)

        domain_instruction = self._resolve_domain_instruction(context.topic_domain)
        glossary_instruction = self._build_glossary_instruction(context.custom_glossary)

        system = self._system_template
        system = system.replace("{domain_instruction}", domain_instruction)
        system = system.replace("{glossary_instruction}", glossary_instruction)

        user = f"Input: {text}"

        return BuiltPrompt(system=system, user=user)

    def _resolve_domain_instruction(self, domain: Optional[str]) -> str:
        if not domain:
            return ""
        return self._domain_instructions.get(domain, self._domain_instructions.get("default", ""))

    def _build_glossary_instruction(self, glossary: Optional[dict]) -> str:
        if not glossary:
            return ""
        pairs = "\n".join(f'- "{k}" → "{v}"' for k, v in glossary.items())
        return self._glossary_template.replace("{glossary_pairs}", pairs)

    def _apply_glossary(self, text: str, glossary: Optional[dict]) -> str:
        """
        Apply pre-LLM word substitutions from the custom glossary.
        Case-insensitive matching; preserves original capitalization pattern.
        """
        if not glossary:
            return text
        for wrong, correct in glossary.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            text = pattern.sub(correct, text)
        return text
