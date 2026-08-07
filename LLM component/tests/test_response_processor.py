"""Unit tests for ResponseProcessor — response parsing, validation, sanitization."""

import pytest
from engine.response_processor import ResponseProcessor


@pytest.fixture
def processor():
    return ResponseProcessor({
        "max_output_length_ratio": 3.0,
        "strip_markdown": True,
        "strip_quotes": True,
    })


def test_strips_preamble(processor):
    raw = 'Here is the corrected sentence: She went to the store.'
    result = processor.process(raw, "she go store")
    assert result.corrected_text == "She went to the store."


def test_strips_markdown_code_blocks(processor):
    raw = '```\nShe went to the store.\n```'
    result = processor.process(raw, "she go store")
    assert result.corrected_text == "She went to the store."


def test_strips_quotes(processor):
    raw = '"She went to the store."'
    result = processor.process(raw, "she go store")
    assert result.corrected_text == "She went to the store."


def test_strips_markdown_bold(processor):
    raw = '**She went to the store.**'
    result = processor.process(raw, "she go store")
    assert result.corrected_text == "She went to the store."


def test_strips_markdown_italic(processor):
    raw = '*She went to the store.*'
    result = processor.process(raw, "she go store")
    assert result.corrected_text == "She went to the store."


def test_empty_response_falls_back(processor):
    result = processor.process("", "she go store")
    assert result.corrected_text == "she go store"
    assert result.quality == "fallback"


def test_whitespace_response_falls_back(processor):
    result = processor.process("   \n   ", "she go store")
    assert result.quality == "fallback"


def test_response_too_long_flagged(processor):
    raw = "word " * 50
    result = processor.process(raw, "short input")
    assert result.quality == "fallback"


def test_short_response_within_limit(processor):
    raw = "She went to the store."
    result = processor.process(raw, "she go store")
    assert result.quality == "good"


def test_no_preamble_response(processor):
    raw = "She went to the store."
    result = processor.process(raw, "she go store")
    assert result.corrected_text == "She went to the store."
    assert result.extraction_method == "direct"


def test_multiline_preamble(processor):
    raw = "Here is the corrected version:\n\nShe went to the store."
    result = processor.process(raw, "she go store")
    assert result.corrected_text == "She went to the store."


def test_response_with_extra_newlines(processor):
    raw = "She went to the\nstore."
    result = processor.process(raw, "she go store")
    assert result.corrected_text == "She went to the store."


def test_response_with_leading_newlines(processor):
    raw = "\n\nShe went to the store."
    result = processor.process(raw, "she go store")
    assert result.corrected_text == "She went to the store."


def test_extraction_method_documented(processor):
    raw = "She went to the store."
    result = processor.process(raw, "she go store")
    assert "strip_" in result.extraction_method or result.extraction_method == "direct"


def test_quality_acceptable_with_minor_artifacts(processor):
    raw = "She went to the store. "
    result = processor.process(raw, "she go store")
    assert result.quality in ("good", "acceptable")
