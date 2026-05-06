"""Reasoning primitives that wrap LLM calls into typed tools."""
from __future__ import annotations

import json
import re
from typing import Any

from ..llm.lm_client import LMClient, ChatMessage
from ..types import TypeName
from ..ui.console import log_tool
from .base import tool, ToolKind

_LM = None
def _lm() -> LMClient:
    global _LM
    if _LM is None: _LM = LMClient()
    return _LM


@tool(
    name="chain_of_thought",
    description="Produce a structured reasoning trace for the given input.",
    input_type=TypeName.TEXT,
    output_type=TypeName.REASONING_TRACE,
    kind=ToolKind.PRIMITIVE,
    cost=0.20,
)
def chain_of_thought(prompt: str, hint: str | None = None) -> dict[str, Any]:
    log_tool("chain_of_thought")
    sys = ("You are a careful reasoner. Decompose the problem into 3-7 numbered steps, "
           "then state a final answer in 1-3 sentences. Output JSON only with keys 'steps' (list of strings) "
           "and 'final' (string).")
    user = f"PROBLEM:\n{str(prompt)[:6000]}\n\n"
    if hint: user += f"HINT:\n{str(hint)[:1500]}\n"
    out = _lm().chat(
        [ChatMessage(role="system", content=sys),
         ChatMessage(role="user", content=user)],
        temperature=0.4, top_p=0.9, max_tokens=900, thinking=True,
    )
    return _safe_json(out.text, default={"steps": [], "final": out.text.strip()[:500]})


@tool(
    name="compare",
    description="Compare two texts on given criteria. Returns {summary, diffs:[{criterion,a_advantage,b_advantage,note}]}.",
    input_type=TypeName.TEXT,
    output_type=TypeName.COMPARISON,
    kind=ToolKind.PRIMITIVE,
    cost=0.20,
)
def compare(a: str, b: str, criteria: list[str] | None = None) -> dict[str, Any]:
    log_tool("compare")
    crit = criteria or ["correctness", "completeness", "specificity"]
    sys = "You compare two candidate texts. Output JSON only with keys 'summary' and 'diffs' (list)."
    user = (f"CRITERIA: {crit}\n\nTEXT_A:\n{str(a)[:3000]}\n\nTEXT_B:\n{str(b)[:3000]}\n\n"
            "For each criterion produce {criterion, a_advantage, b_advantage, note}. Then give a summary line.")
    out = _lm().chat(
        [ChatMessage(role="system", content=sys),
         ChatMessage(role="user", content=user)],
        temperature=0.3, top_p=0.9, max_tokens=700,
    )
    return _safe_json(out.text, default={"summary": out.text[:200], "diffs": []})


@tool(
    name="detect_inconsistencies",
    description="List logical/factual inconsistencies in a text as a list of issue strings.",
    input_type=TypeName.TEXT,
    output_type=TypeName.ISSUES,
    kind=ToolKind.PRIMITIVE,
    cost=0.15,
)
def detect_inconsistencies(text: str) -> list[str]:
    log_tool("detect_inconsistencies")
    if not text or not str(text).strip(): return []
    sys = "Identify inconsistencies (contradictions, conflicting numbers, mismatched entities). JSON only."
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "Issues",
            "schema": {
                "type": "object",
                "properties": {"issues": {"type": "array", "items": {"type": "string"}}},
                "required": ["issues"],
                "additionalProperties": False,
            },
        },
    }
    out = _lm().chat(
        [ChatMessage(role="system", content=sys),
         ChatMessage(role="user", content=str(text)[:6000])],
        temperature=0.2, top_p=0.9, max_tokens=600,
        response_format=schema,
    )
    obj = _safe_json(out.text, default={"issues": []})
    return list(obj.get("issues") or [])


def _safe_json(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
        return default
