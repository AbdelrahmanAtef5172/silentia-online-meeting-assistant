"""Unit tests for PromptBuilder — prompt construction and context injection."""

import pytest
from engine.prompt_builder import PromptBuilder
from engine.schemas import CorrectionContext, GuardDecision, InputCategory


@pytest.fixture
def prompt_builder():
    config = {
        "prompts": {
            "system": (
                "You are a grammar correction assistant.\n"
                "Rules:\n"
                "- Output ONLY the corrected sentence.\n"
                "- Preserve the original meaning exactly.\n"
                "{domain_instruction}"
                "{glossary_instruction}"
            ),
            "domain_instructions": {
                "medical": "Use standard medical terminology.",
                "legal": "Use precise legal terminology.",
                "casual": "",
                "default": "",
            },
            "glossary_instruction_template": (
                "Apply these substitutions:\n{glossary_pairs}"
            ),
        }
    }
    return PromptBuilder(config)


def test_basic_prompt_building(prompt_builder):
    decision = GuardDecision(
        category=InputCategory.NORMAL,
        action="correct",
        transformed_text="she go store yesterday",
    )
    prompt = prompt_builder.build(decision)
    assert "she go store yesterday" in prompt.user
    assert "grammar correction assistant" in prompt.system
    assert prompt.system.count("{") == 0


def test_prompt_with_domain_context(prompt_builder):
    decision = GuardDecision(
        category=InputCategory.NORMAL,
        action="correct",
        transformed_text="patient show symptom fever",
    )
    context = CorrectionContext(topic_domain="medical")
    prompt = prompt_builder.build(decision, context=context)
    assert "Use standard medical terminology" in prompt.system
    assert "patient show symptom fever" in prompt.user


def test_prompt_with_glossary(prompt_builder):
    decision = GuardDecision(
        category=InputCategory.NORMAL,
        action="correct",
        transformed_text="DOCTOR prescribe medicine",
    )
    context = CorrectionContext(custom_glossary={"DOCTOR": "physician"})
    prompt = prompt_builder.build(decision, context=context)
    assert "physician" in prompt.user
    assert "Apply these substitutions" in prompt.system


def test_prompt_with_legal_domain(prompt_builder):
    decision = GuardDecision(
        category=InputCategory.NORMAL,
        action="correct",
        transformed_text="lawyer sign contract",
    )
    context = CorrectionContext(topic_domain="legal")
    prompt = prompt_builder.build(decision, context=context)
    assert "Use precise legal terminology" in prompt.system
    assert "lawyer sign contract" in prompt.user


def test_prompt_without_context(prompt_builder):
    decision = GuardDecision(
        category=InputCategory.NORMAL,
        action="correct",
        transformed_text="she go store",
    )
    prompt = prompt_builder.build(decision, context=None)
    assert "she go store" in prompt.user
    assert prompt.system.count("{") == 0


def test_prompt_input_prefix_format(prompt_builder):
    decision = GuardDecision(
        category=InputCategory.NORMAL,
        action="correct",
        transformed_text="hello world",
    )
    prompt = prompt_builder.build(decision)
    assert prompt.user == "Input: hello world"


def test_prompt_empty_domain_instruction(prompt_builder):
    decision = GuardDecision(
        category=InputCategory.NORMAL,
        action="correct",
        transformed_text="she go store",
    )
    context = CorrectionContext(topic_domain="casual")
    prompt = prompt_builder.build(decision, context=context)
    assert "{domain_instruction}" not in prompt.system


def test_prompt_unknown_domain_uses_default(prompt_builder):
    decision = GuardDecision(
        category=InputCategory.NORMAL,
        action="correct",
        transformed_text="she go store",
    )
    context = CorrectionContext(topic_domain="unknown_domain")
    prompt = prompt_builder.build(decision, context=context)
    assert "{domain_instruction}" not in prompt.system


def test_prompt_passthrough_action(prompt_builder):
    decision = GuardDecision(
        category=InputCategory.PASSTHROUGH,
        action="pass",
        transformed_text="She went to the store.",
    )
    prompt = prompt_builder.build(decision)
    assert prompt.user == "Input: She went to the store."
