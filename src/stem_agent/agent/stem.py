"""StemAgent: the L0 base agent. Knows how to:

1. Generate a domain brief by issuing 3-5 web/wiki/arxiv queries derived from a
   small set of question templates and summarizing the responses.
2. Propose an initial seed pipeline for a task using the registry, by asking
   the LLM (constrained by JSON schema) for a list of tool names + params.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..llm.lm_client import LMClient, ChatMessage
from ..llm.prompts import render_prompt
from ..tools.registry import ToolLibrary
from ..ui.console import log_decision, log_retrieve
from ..types import TypeName
from .pipeline import Pipeline, PipelineStep, validate


@dataclass
class DomainBrief:
    domain: str
    paragraphs: list[str]
    sources: list[dict[str, str]]


_DOMAIN_QUESTIONS: dict[str, list[str]] = {
    "legal": [
        "How do legal experts approach contract analysis?",
        "What are the canonical clause categories in commercial contracts (CUAD taxonomy)?",
        "Common pitfalls in extracting obligations from contract text",
        "GDPR Article 5 principles in plain English",
    ],
    "economics": [
        "How do equity analysts read a 10-K filing?",
        "Standard financial ratios used to assess corporate distress (Altman Z, Piotroski F)",
        "Yield curve regimes and recession signals",
        "FRED time series commonly used in macro analysis",
    ],
    "contract_analysis": [
        "Best practices for redlining commercial contracts",
        "What clause types deserve special attention in software vendor agreements",
        "Common indemnification language and its implications",
    ],
}


def bootstrap_domain_brief(domain: str, lm: LMClient,
                           library: ToolLibrary, max_queries: int = 4) -> DomainBrief:
    """Run a small bootstrapping loop: search -> summarize -> store as brief."""
    questions = _DOMAIN_QUESTIONS.get(domain, [])[:max_queries]
    if not questions:
        return DomainBrief(domain=domain, paragraphs=[], sources=[])

    web = library.get("web_search").run if "web_search" in library._tools else None
    wiki = library.get("wikipedia_lookup").run if "wikipedia_lookup" in library._tools else None
    paras: list[str] = []
    sources: list[dict[str, str]] = []
    for q in questions:
        log_retrieve(f"[bootstrap/{domain}] {q}")
        snippets: list[str] = []
        try:
            if web:
                docs = web(query=q, k=3) or []
                for d in docs[:3]:
                    snippets.append(d.get("snippet","")[:600])
                    if d.get("url"):
                        sources.append({"title": d.get("title",""), "url": d.get("url","")})
        except Exception: pass
        try:
            if wiki and len(snippets) < 2:
                d = wiki(title=q.split("?")[0].split(" in ")[-1])
                if d.get("text"):
                    snippets.append(d["text"][:1000])
                    if d.get("url"):
                        sources.append({"title": d.get("title",""), "url": d.get("url","")})
        except Exception: pass
        if not snippets:
            continue
        try:
            out = lm.chat(
                [ChatMessage(role="system", content="You are a careful research assistant. Be terse."),
                 ChatMessage(role="user", content=(f"Question: {q}\n\nSources:\n" + "\n---\n".join(snippets)
                                                   + "\n\nWrite a single paragraph (<=120 words) summarizing what these sources say."))],
                temperature=0.3, top_p=0.9, max_tokens=300,
            )
            para = (out.text or "").strip()
            if para:
                paras.append(f"{q}\n{para}")
        except Exception:
            pass
    return DomainBrief(domain=domain, paragraphs=paras, sources=sources)


def render_brief(brief: DomainBrief) -> str:
    if not brief.paragraphs:
        return "(no domain brief available)"
    out = list(brief.paragraphs)
    if brief.sources:
        out.append("\nSources:")
        for s in brief.sources[:8]:
            out.append(f"- {s.get('title','(untitled)')} - {s.get('url','')}")
    return "\n\n".join(out)


def _tool_catalogue(library: ToolLibrary, domain: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in library.evolution_candidates(domain=domain):
        out.append({
            "name": t.name,
            "description": t.description,
            "input_type": t.input_type.value,
            "output_type": t.output_type.value,
            "domain": t.domain,
            "params": list((t.parameters_schema or {}).keys()),
        })
    return out


def propose_seed_pipeline(
    *,
    task_question: str,
    task_input_type: TypeName,
    library: ToolLibrary,
    lm: LMClient,
    domain: str | None = None,
    domain_brief: str = "",
    composites_summary: str = "",
    layer: str = "stem",
) -> Pipeline:
    """Ask the LLM for a 1-5 step pipeline; fall back to a hand-coded baseline on invalid output."""
    sys_name = {
        "stem": "stem_system",
        "legal": "l1_legal_system",
        "economics": "l1_econ_system",
        "contract_analysis": "l2_contract_system",
    }.get(layer, "stem_system")
    sys_prompt = render_prompt(sys_name,
                               domain_brief=domain_brief,
                               subdomain_brief=domain_brief,
                               composites_summary=composites_summary)

    catalogue = _tool_catalogue(library, domain)
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "PipelineProposal",
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {"type":"array","items":{
                        "type":"object",
                        "properties":{
                            "tool":{"type":"string"},
                            "params":{"type":"object"}
                        },
                        "required":["tool","params"],
                        "additionalProperties": False,
                    }, "minItems": 1, "maxItems": 5},
                    "rationale":{"type":"string"}
                },
                "required":["steps","rationale"],
                "additionalProperties": False,
            }
        }
    }
    user = (
        f"TYPES: {[t.value for t in TypeName]}\n\n"
        f"TOOLS:\n{json.dumps(catalogue, ensure_ascii=False)[:6000]}\n\n"
        f"TASK:\nInput type: {task_input_type.value}\n"
        f"Question/objective: {task_question}\n\n"
        "Propose a typed pipeline of 1-5 steps. Output JSON only."
    )
    try:
        out = lm.chat(
            [ChatMessage(role="system", content=sys_prompt),
             ChatMessage(role="user", content=user)],
            temperature=0.4, top_p=0.9, max_tokens=900, response_format=schema, thinking=True,
        )
        proposal = _safe_json(out.text, default={"steps":[]})
    except Exception:
        proposal = {"steps": []}

    p = Pipeline([PipelineStep(s.get("tool",""), dict(s.get("params") or {})) for s in proposal.get("steps") or []])
    registry = library._tools
    ok, msg = validate(p, registry, task_input_type)
    if not ok:
        log_decision(f"seed proposal invalid ({msg}); using fallback")
        return _fallback(layer, library, task_input_type)
    log_decision(f"seed pipeline accepted ({len(p)} steps)")
    return p


def _fallback(layer: str, library: ToolLibrary, input_type: TypeName) -> Pipeline:
    """Hand-coded minimum pipelines, gated by input_type so the result is valid."""
    if input_type == TypeName.QUERY:
        if layer == "economics":
            return Pipeline([PipelineStep("edgar_fetch"), PipelineStep("financial_ratios")])
        return Pipeline([PipelineStep("web_search"), PipelineStep("summarize")])
    if input_type in (TypeName.TEXT, TypeName.DOCUMENT):
        if layer in ("legal", "contract_analysis"):
            return Pipeline([PipelineStep("clause_extraction"), PipelineStep("summarize")])
        return Pipeline([PipelineStep("summarize")])
    if input_type == TypeName.FILING:
        return Pipeline([PipelineStep("financial_ratios")])
    return Pipeline([PipelineStep("summarize")])


def _safe_json(text: str, default: Any) -> Any:
    try: return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
        return default
