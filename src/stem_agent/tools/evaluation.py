"""Deterministic eval primitives the agent can use as inline self-checks.

These are NOT the LLM judges. They are programmatic scorers that compute
F1 / coverage / consistency without calling an LLM. The agent can chain
these into pipelines (e.g., to validate a structured output).
"""
from __future__ import annotations

import re
from typing import Any

from ..types import TypeName
from ..ui.console import log_eval
from .base import tool, ToolKind


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").lower()).strip()


@tool(
    name="score_accuracy",
    description="Token-overlap F1 against a reference string (or list of acceptable references).",
    input_type=TypeName.TEXT,
    output_type=TypeName.SCORE,
    kind=ToolKind.PRIMITIVE,
    capability_tag="score",
    cost=0.01,
)
def score_accuracy(output: str, reference: str | list[str]) -> float:
    log_eval("score_accuracy")
    refs = reference if isinstance(reference, list) else [reference]
    out_tokens = set(_normalize(output).split())
    best = 0.0
    for r in refs:
        rt = set(_normalize(r).split())
        if not out_tokens or not rt: continue
        tp = len(out_tokens & rt)
        if tp == 0: continue
        prec = tp / len(out_tokens); rec = tp / len(rt)
        f1 = 2*prec*rec/(prec+rec)
        if f1 > best: best = f1
    return best


@tool(
    name="consistency_check",
    description="Heuristic 0..1 consistency score: penalizes contradictions, conflicting numbers, abrupt topic changes.",
    input_type=TypeName.TEXT,
    output_type=TypeName.SCORE,
    kind=ToolKind.PRIMITIVE,
    capability_tag="consistency",
    cost=0.01,
)
def consistency_check(text: str) -> float:
    log_eval("consistency_check")
    text = str(text or "")
    if not text.strip(): return 0.0
    score = 1.0
    contradictions = re.findall(r"\b(but|however|on the other hand|in contrast)\b", text, re.IGNORECASE)
    score -= min(0.2, 0.04 * len(contradictions))
    nums = re.findall(r"\b\d{3,}\b", text)
    if nums:
        unique_count = len(set(nums))
        if unique_count / len(nums) > 0.85:
            score -= 0.1
    bullets = text.count("\n-") + text.count("\n*") + text.count("\n1.")
    paragraphs = text.count("\n\n")
    if bullets > 0 and paragraphs > 6:
        score -= 0.05
    return max(0.0, min(1.0, score))


@tool(
    name="completeness_check",
    description="0..1 score = fraction of `requirements` keywords found in output.",
    input_type=TypeName.STRUCTURED_DATA,
    output_type=TypeName.SCORE,
    kind=ToolKind.PRIMITIVE,
    capability_tag="completeness",
    cost=0.01,
)
def completeness_check(output: str, requirements: list[str]) -> float:
    log_eval("completeness_check")
    if not requirements: return 1.0
    nrm = _normalize(output)
    hits = sum(1 for r in requirements if _normalize(r) in nrm)
    return hits / len(requirements)
