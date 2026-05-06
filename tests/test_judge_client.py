"""Tests for the external-judge fallback chain selector."""
from unittest.mock import patch

import pytest

from stem_agent.llm import judge_client as jc


@patch.dict("os.environ", {"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "y"})
def test_resolve_prefers_anthropic():
    assert jc.resolve_provider() == "anthropic"


@patch.dict("os.environ", {"OPENAI_API_KEY": "y"}, clear=True)
def test_resolve_falls_back_to_openai():
    assert jc.resolve_provider() == "openai"


@patch.dict("os.environ", {}, clear=True)
def test_resolve_falls_back_to_local():
    assert jc.resolve_provider() == "local"


def test_format_pairwise_prompt_contains_both_orders_marker():
    prompt = jc.format_pairwise_prompt("question?", "answer A", "answer B", rubric="be brief")
    assert "ANSWER_A" in prompt
    assert "ANSWER_B" in prompt
    assert "be brief" in prompt
