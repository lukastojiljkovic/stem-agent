"""StemAgent: the L0 base agent. Knows how to:

1. Generate a domain brief by issuing 3-5 web/wiki/arxiv queries derived from a
   small set of question templates and summarizing the responses.
2. Propose an initial seed pipeline for a task using the registry, by asking
   the LLM (constrained by JSON schema) for a list of tool names + params.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..llm.lm_client import LMClient, ChatMessage, clean_llm_query, safe_json_loads
from ..llm.prompts import render_prompt
from ..tools.registry import ToolLibrary
from ..ui.console import log_decision, log_retrieve, log_warn
from ..types import TypeName
from .pipeline import Pipeline, PipelineStep, validate

_BOOTSTRAP_MAX_RETRIES = 2


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


def _adaptive_questions(domain: str, lm: LMClient, n: int = 4) -> list[str]:
    """Ask the LLM to generate domain-grounding questions on the fly.
    Falls back to the hand-curated list on failure or empty response."""
    static = _DOMAIN_QUESTIONS.get(domain, [])
    sys = ("You generate research questions to ground a specialist agent in a "
           "domain. Output ONLY a numbered list of short questions, one per line, "
           "no preamble.")
    user = (f"DOMAIN: {domain}\n\n"
            f"Produce {n} concrete questions whose answers (gathered from Wikipedia / "
            f"Semantic Scholar / arXiv) would help an agent become competent in this "
            f"domain. Each question must be self-contained and search-engine-friendly.")
    try:
        out = lm.chat(
            [ChatMessage(role="system", content=sys),
             ChatMessage(role="user", content=user)],
            temperature=0.5, top_p=0.9, max_tokens=400,
        )
        lines = [ln.strip() for ln in (out.text or "").splitlines() if ln.strip()]
        # Strip leading enumeration like "1. " or "- ".
        cleaned: list[str] = []
        import re as _re
        for ln in lines:
            ln = _re.sub(r"^\s*(?:\d+[.)]\s*|[-*]\s*)", "", ln).strip()
            if len(ln) > 8:
                cleaned.append(ln)
        if cleaned:
            return cleaned[:n] + static[: max(0, n - len(cleaned))]
    except Exception:
        pass
    return static[:n]


def bootstrap_domain_brief(domain: str, lm: LMClient,
                           library: ToolLibrary, max_queries: int = 4,
                           adaptive: bool = True) -> DomainBrief:
    """Run a small bootstrapping loop: search -> summarize -> store as brief.

    For each domain question we try multiple key-free backends in turn (Wikipedia
    search, Semantic Scholar, OpenAlex, arXiv, plain Wikipedia lookup, web
    search). If none of them return anything for a question, we ask the LLM to
    rewrite the query and retry (up to `_BOOTSTRAP_MAX_RETRIES`).

    The retrieval phase across questions runs concurrently in a thread pool
    (independent network I/O); LLM summarization stays serial because the
    LM Studio backend queues requests anyway and we want stable log ordering.

    If `adaptive=True` and no static questions exist for the domain (or as a
    supplement), the LLM is asked to generate fresh ones. The point is to keep
    the "stem-cell environmental signal" mechanism working even when the agent
    has no API keys and DDG is throttled."""
    questions = _DOMAIN_QUESTIONS.get(domain, [])[:max_queries]
    if adaptive and (not questions or len(questions) < max_queries):
        try:
            questions = _adaptive_questions(domain, lm, n=max_queries)
        except Exception:
            pass
    if not questions:
        return DomainBrief(domain=domain, paragraphs=[], sources=[])

    # Phase 1: parallel retrieval (with rewrite retries) per question.
    from concurrent.futures import ThreadPoolExecutor

    def gather_with_retries(q: str) -> tuple[str, list[str], list[dict[str, str]]]:
        log_retrieve(f"[bootstrap/{domain}] {q}")
        snips: list[str] = []
        srcs: list[dict[str, str]] = []
        attempt_query = q
        for attempt in range(_BOOTSTRAP_MAX_RETRIES + 1):
            new_snips, new_srcs = _gather_snippets_for_query(library, attempt_query)
            snips.extend(new_snips)
            srcs.extend(new_srcs)
            if snips:
                break
            if attempt < _BOOTSTRAP_MAX_RETRIES:
                try:
                    rewritten = _rewrite_query(lm, q, attempt_query, domain)
                except Exception:
                    rewritten = ""
                if not rewritten or rewritten.strip().lower() == attempt_query.strip().lower():
                    log_retrieve(f"[bootstrap/{domain}] no rewrite for {q!r}; giving up")
                    break
                log_retrieve(f"[bootstrap/{domain}] rewriting -> {rewritten!r}")
                attempt_query = rewritten
        return q, snips, srcs

    # max_workers=4 keeps polite rate to public APIs (SS, OpenAlex, Wikipedia).
    with ThreadPoolExecutor(max_workers=min(4, len(questions))) as pool:
        gathered = list(pool.map(gather_with_retries, questions))

    # Phase 2: serial summarization (LM Studio queues per-model anyway).
    sources: list[dict[str, str]] = []
    paras: list[str] = []
    for q, snippets, q_sources in gathered:
        sources.extend(q_sources)
        if not snippets:
            continue
        try:
            # max_tokens generous enough that Gemma 4's analysis channel doesn't
            # eat the entire budget before the user-visible final channel emits.
            out = lm.chat(
                [ChatMessage(role="system",
                             content="You are a careful research assistant. Output only the final paragraph - no preamble, no analysis."),
                 ChatMessage(role="user", content=(f"Question: {q}\n\nSources:\n" + "\n---\n".join(snippets)
                                                   + "\n\nWrite a single paragraph (<=120 words) summarizing what these sources say."))],
                temperature=0.3, top_p=0.9, max_tokens=1500,
            )
            para = (out.text or "").strip()
            if para:
                paras.append(f"{q}\n{para}")
            else:
                log_warn(f"[bootstrap] LM returned empty text for question: {q!r}")
        except Exception as e:
            log_warn(f"[bootstrap] LM summarize failed for question {q!r}: {e}")

    return DomainBrief(domain=domain, paragraphs=paras, sources=sources)


def _list_backend_results(res: Any) -> tuple[list[str], list[dict[str, str]]]:
    """Adapt a backend's return value to (snippets, sources). Wikipedia
    article lookup returns a single dict with `text`/`title`/`url`; every
    other backend returns a list of {title,url,snippet|abstract}."""
    snips: list[str] = []
    srcs: list[dict[str, str]] = []
    if isinstance(res, dict) and res.get("text"):
        snips.append(res["text"][:1000])
        if res.get("url"):
            srcs.append({"title": res.get("title", ""), "url": res.get("url", "")})
    elif isinstance(res, list):
        for d in res[:3]:
            snippet = (d.get("snippet") or d.get("abstract") or "")[:800]
            if snippet:
                snips.append(snippet)
            if d.get("url"):
                srcs.append({"title": d.get("title", ""), "url": d.get("url", "")})
    return snips, srcs


def _gather_snippets_for_query(library: ToolLibrary, query: str
                                ) -> tuple[list[str], list[dict[str, str]]]:
    """Try several key-free retrieval primitives in turn for one query.
    Returns (snippets, sources). Each primitive has its own cache so retries
    are cheap once a backend has answered."""
    def get_runner(name: str) -> Callable[..., Any] | None:
        try:
            return library.get(name).run
        except KeyError:
            return None

    # Order: Wikipedia search (full-text, fast) -> Semantic Scholar -> OpenAlex
    # -> arxiv -> Wikipedia lookup -> web (Tavily/DDG, may be empty without keys).
    backends: list[tuple[Callable[..., Any] | None, dict[str, Any]]] = [
        (get_runner("wikipedia_search"), {"query": query, "k": 3}),
        (get_runner("semantic_scholar_search"), {"query": query, "k": 3}),
        (get_runner("openalex_search"), {"query": query, "k": 3}),
        (get_runner("arxiv_search"), {"query": query, "k": 2}),
        (get_runner("wikipedia_lookup"), {"title": query.rstrip("?").split(" in ")[-1]}),
        (get_runner("web_search"), {"query": query, "k": 3}),
    ]
    snippets: list[str] = []
    sources: list[dict[str, str]] = []
    for fn, kwargs in backends:
        if fn is None:
            continue
        try:
            res = fn(**kwargs)
        except Exception:
            continue
        new_snips, new_srcs = _list_backend_results(res)
        snippets.extend(new_snips)
        sources.extend(new_srcs)
        if snippets:
            # First backend that yielded text is good enough; stop early.
            break
    return snippets, sources


def _rewrite_query(lm: LMClient, original_question: str, last_query: str, domain: str) -> str:
    """Ask the LLM to produce a more search-engine-friendly query.
    Called only after all backends returned empty for `last_query`."""
    sys = ("You rewrite research questions into short, high-recall search-engine "
           "queries. Output ONLY the rewritten query, no quotes, no explanation, "
           "<= 12 words.")
    user = (
        f"DOMAIN: {domain}\n"
        f"ORIGINAL QUESTION: {original_question}\n"
        f"LAST TRIED QUERY: {last_query}\n"
        "All prior backends returned nothing for the query above; produce a "
        "different phrasing that's more likely to retrieve results."
    )
    out = lm.chat(
        [ChatMessage(role="system", content=sys),
         ChatMessage(role="user", content=user)],
        temperature=0.5, top_p=0.9, max_tokens=64,
    )
    return clean_llm_query(out.text, max_words=12)


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
    capability_tag: str | None = None,
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
        proposal = safe_json_loads(out.text, default={"steps":[]})
    except Exception:
        proposal = {"steps": []}

    p = Pipeline([PipelineStep(s.get("tool",""), dict(s.get("params") or {})) for s in proposal.get("steps") or []])
    registry = library._tools
    ok, msg = validate(p, registry, task_input_type)
    if not ok:
        log_decision(f"seed proposal invalid ({msg}); using fallback")
        return _fallback(layer, library, task_input_type, capability_tag)
    log_decision(f"seed pipeline accepted ({len(p)} steps)")
    return p


def _fallback(layer: str, library: ToolLibrary, input_type: TypeName,
              capability_tag: str | None = None) -> Pipeline:
    """Hand-coded minimum pipelines, gated by (capability_tag, input_type, layer) so the
    result is type-valid AND uses the appropriate primitive for the task family.
    L1 and L2 layers diverge on parameter regime so they evolve toward different
    composites (rather than both converging on the same default).
    """
    cap = capability_tag or ""
    # Capability-driven: dispatches first on what the task is asking for.
    if cap == "legal_qa":
        return Pipeline([PipelineStep("classify", {"labels": ["Yes", "No"]})])
    if cap == "legal_qa_grounded":
        # Variant fallback that exercises the extract_search_query bridge so
        # evolution can discover whether grounding helps the classifier.
        return Pipeline([
            PipelineStep("extract_search_query", {"max_words": 8}),
            PipelineStep("wikipedia_search", {"k": 2}),
            PipelineStep("summarize", {"max_words": 80}),
            PipelineStep("classify", {"labels": ["Yes", "No"]}),
        ])
    if cap == "obligation":
        return Pipeline([PipelineStep("obligation_detection")])
    if cap == "financial_ratios":
        return Pipeline([PipelineStep("edgar_fetch"), PipelineStep("financial_ratios")])
    if cap == "financial_qa":
        return Pipeline([PipelineStep("edgar_fetch"), PipelineStep("summarize")])
    if cap == "clause_extraction":
        # L2 (subdomain=contract_analysis) uses a stricter parameter regime so it
        # explores a different region of the search space than L1: higher
        # confidence threshold + focused 4-category subset matched to our eval.
        if layer == "contract_analysis":
            return Pipeline([PipelineStep("clause_extraction", {
                "taxonomy": "CUAD41",
                "min_conf": 0.6,
                "categories": ["Governing Law", "Termination for Convenience",
                               "Cap on Liability", "Anti-Assignment"],
            })])
        return Pipeline([PipelineStep("clause_extraction"), PipelineStep("summarize")])

    # Capability unknown: fall through to type/layer heuristics.
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
