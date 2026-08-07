"""Unit tests for InputGuard — no API calls, no mocks needed."""
import pytest
from engine.input_guard import InputGuard
from engine.schemas import InputCategory


@pytest.fixture
def guard():
    return InputGuard(config={
        "max_input_tokens": 400,
        "min_alpha_ratio": 0.3,
        "passthrough_max_words": 3,
    })


def test_empty_string(guard):
    decision = guard.inspect("")
    assert decision.category == InputCategory.EMPTY
    assert decision.action == "reject"


def test_none_input(guard):
    decision = guard.inspect(None)
    assert decision.category == InputCategory.EMPTY


def test_whitespace_only(guard):
    decision = guard.inspect("   \n\t  ")
    assert decision.category == InputCategory.EMPTY


def test_punctuation_only(guard):
    decision = guard.inspect("... !!! ???")
    assert decision.category == InputCategory.PUNCTUATION_ONLY
    assert decision.action == "pass"


def test_number_heavy(guard):
    decision = guard.inspect("1234 5678 9.99 $100")
    assert decision.category == InputCategory.NUMBER_HEAVY


def test_normal_slr_input(guard):
    decision = guard.inspect("she go store yesterday buy apple")
    assert decision.category == InputCategory.NORMAL
    assert decision.action == "correct"


def test_too_long_input(guard):
    long_text = "she go store " * 150
    decision = guard.inspect(long_text)
    assert decision.category == InputCategory.TOO_LONG
    assert decision.action == "truncate"
    assert len(decision.transformed_text.split()) <= 400


def test_already_correct_heuristic(guard):
    decision = guard.inspect("She went to the store yesterday.")
    assert decision.category == InputCategory.PASSTHROUGH


def test_already_correct_no_capital(guard):
    decision = guard.inspect("she went to the store yesterday.")
    assert decision.category != InputCategory.PASSTHROUGH


def test_already_correct_no_period(guard):
    decision = guard.inspect("She went to the store yesterday")
    assert decision.category != InputCategory.PASSTHROUGH


def test_already_correct_short_input(guard):
    decision = guard.inspect("Go store.")
    assert decision.category != InputCategory.PASSTHROUGH


def test_already_correct_no_function_words(guard):
    decision = guard.inspect("Beautiful sunny day.")
    assert decision.category != InputCategory.PASSTHROUGH


def test_alpha_characters_in_mixed_input(guard):
    decision = guard.inspect("test123")
    assert decision.category != InputCategory.PUNCTUATION_ONLY


def test_empty_after_strip(guard):
    decision = guard.inspect("   ")
    assert decision.category == InputCategory.EMPTY


def test_min_alpha_ratio_threshold():
    guard = InputGuard(config={
        "max_input_tokens": 400,
        "min_alpha_ratio": 0.6,
        "passthrough_max_words": 3,
    })
    decision = guard.inspect("hello 123 world 456")
    assert decision.category == InputCategory.NUMBER_HEAVY
