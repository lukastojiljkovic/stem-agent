"""Thin wrapper around the openai SDK pointed at LM Studio.

Defaults to Gemma 4 E4B sampling parameters. Provides:
  - structured ChatMessage / ToolCall types
  - a `chat()` method returning ChatResult (text + tool_calls + usage)
  - JSON-schema constrained output via `response_format`
  - safe max_tokens cap (8192) below LM Studio's documented silent cap
  - retry with exponential backoff on transient errors
  - optional `<|think|>` thinking-mode prefix on the system message
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import LM


@dataclass
class ChatMessage:
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ChatResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: Usage = field(default_factory=Usage)
    raw: Any = None


_MAX_TOKENS_HARD_CAP = 8192


class LMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float | None = None):
        self._client = OpenAI(
            base_url=base_url or LM.base_url,
            api_key=api_key or LM.api_key,
            timeout=timeout if timeout is not None else LM.request_timeout_s,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def chat(
        self,
        messages: Iterable[ChatMessage],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        thinking: bool = False,
        model: str | None = None,
    ) -> ChatResult:
        msgs_list = list(messages)
        msgs = [self._format_message(m, thinking and i == 0) for i, m in enumerate(msgs_list)]
        if thinking and msgs and msgs[0].get("role") != "system":
            msgs.insert(0, {"role": "system", "content": "<|think|>"})

        body: dict[str, Any] = {
            "model": model or LM.model,
            "messages": msgs,
            "temperature": LM.temperature_default if temperature is None else temperature,
            "top_p": LM.top_p_default if top_p is None else top_p,
            "max_tokens": min(max_tokens or LM.max_tokens_default, _MAX_TOKENS_HARD_CAP),
        }
        if top_k is not None or LM.top_k_default is not None:
            body["extra_body"] = {"top_k": top_k if top_k is not None else LM.top_k_default}
        if response_format is not None:
            body["response_format"] = response_format
        if tools:
            body["tools"] = tools

        resp = self._client.chat.completions.create(**body)
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = Usage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(resp.usage, "total_tokens", 0) or 0,
        )

        return ChatResult(
            text=msg.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            usage=usage,
            raw=resp,
        )

    @staticmethod
    def _format_message(m: ChatMessage, prepend_think_token: bool) -> dict[str, Any]:
        content = m.content
        if prepend_think_token and m.role == "system" and not content.startswith("<|think|>"):
            content = "<|think|>\n" + content
        d: dict[str, Any] = {"role": m.role, "content": content}
        if m.name:
            d["name"] = m.name
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        return d
