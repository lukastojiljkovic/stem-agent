"""Tests for the LM Studio client wrapper.

Live calls are mocked. A separate manual smoke test (run-only) lives in
plan 09 for end-to-end validation against a running LM Studio server.
"""
from unittest.mock import MagicMock, patch

import pytest

from stem_agent.llm.lm_client import LMClient, ChatMessage, ToolCall


def _mock_response(content="hello", tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return resp


@patch("stem_agent.llm.lm_client.OpenAI")
def test_chat_basic(mock_openai_cls):
    inst = mock_openai_cls.return_value
    inst.chat.completions.create.return_value = _mock_response(content="ok")
    c = LMClient()
    out = c.chat([ChatMessage(role="user", content="hi")])
    assert out.text == "ok"
    inst.chat.completions.create.assert_called_once()


@patch("stem_agent.llm.lm_client.OpenAI")
def test_chat_passes_temperature_and_top_k(mock_openai_cls):
    inst = mock_openai_cls.return_value
    inst.chat.completions.create.return_value = _mock_response()
    c = LMClient()
    c.chat([ChatMessage(role="user", content="hi")], temperature=0.3, top_p=0.9)
    kwargs = inst.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.3
    assert kwargs["top_p"] == 0.9


@patch("stem_agent.llm.lm_client.OpenAI")
def test_thinking_mode_prepends_token(mock_openai_cls):
    inst = mock_openai_cls.return_value
    inst.chat.completions.create.return_value = _mock_response()
    c = LMClient()
    c.chat(
        [ChatMessage(role="system", content="You are helpful."), ChatMessage(role="user", content="hi")],
        thinking=True,
    )
    sent = inst.chat.completions.create.call_args.kwargs["messages"]
    assert sent[0]["role"] == "system"
    assert sent[0]["content"].startswith("<|think|>")


@patch("stem_agent.llm.lm_client.OpenAI")
def test_tool_calls_returned(mock_openai_cls):
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "web_search"
    tc.function.arguments = '{"query":"hi"}'
    inst = mock_openai_cls.return_value
    inst.chat.completions.create.return_value = _mock_response(content="", tool_calls=[tc])
    c = LMClient()
    out = c.chat([ChatMessage(role="user", content="x")])
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "web_search"
    assert out.tool_calls[0].arguments == {"query": "hi"}


@patch("stem_agent.llm.lm_client.OpenAI")
def test_max_tokens_capped_below_lm_studio_silent_cap(mock_openai_cls):
    inst = mock_openai_cls.return_value
    inst.chat.completions.create.return_value = _mock_response()
    c = LMClient()
    c.chat([ChatMessage(role="user", content="x")], max_tokens=20000)
    kwargs = inst.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] <= 8192
