"""Processing primitives: PDF/HTML extraction, summarize, extract_entities, classify, normalize."""
from __future__ import annotations

import json
import re
from typing import Any

from ..llm.lm_client import LMClient, ChatMessage
from ..types import TypeName
from ..ui.console import log_tool, log_warn
from .base import tool, ToolKind

_LM = None


def _lm() -> LMClient:
    global _LM
    if _LM is None:
        _LM = LMClient()
    return _LM


@tool(
    name="pdf_extract",
    description="Extract plaintext from a PDF (path or bytes). Pymupdf primary, pdfplumber fallback.",
    input_type=TypeName.DOCUMENT,
    output_type=TypeName.TEXT,
    kind=ToolKind.PRIMITIVE,
    cost=0.05,
)
def pdf_extract(path: str | None = None, bytes_b64: str | None = None,
                max_chars: int = 30000) -> str:
    log_tool(f"pdf_extract path={path}")
    if not path and not bytes_b64:
        return ""
    text = ""
    try:
        import fitz
        if path:
            doc = fitz.open(path)
        else:
            import base64
            doc = fitz.open(stream=base64.b64decode(bytes_b64), filetype="pdf")
        chunks = []
        for page in doc:
            chunks.append(page.get_text("text"))
            if sum(len(c) for c in chunks) > max_chars:
                break
        text = "\n".join(chunks)[:max_chars]
        doc.close()
    except Exception as e:
        log_warn(f"pymupdf failed: {e}; trying pdfplumber")
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                pieces = []
                for p in pdf.pages:
                    pieces.append(p.extract_text() or "")
                    if sum(len(x) for x in pieces) > max_chars:
                        break
                text = "\n".join(pieces)[:max_chars]
        except Exception as e2:
            log_warn(f"pdfplumber also failed: {e2}")
            text = ""
    return text


@tool(
    name="html_extract",
    description="Fetch a URL and return main-content text via trafilatura.",
    input_type=TypeName.QUERY,
    output_type=TypeName.TEXT,
    kind=ToolKind.PRIMITIVE,
    cost=0.05,
)
def html_extract(url: str, max_chars: int = 12000) -> str:
    log_tool(f"html_extract {url}")
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if not downloaded:
            return ""
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True) or ""
        return text[:max_chars]
    except Exception as e:
        log_warn(f"trafilatura failed: {e}")
        return ""


@tool(
    name="summarize",
    description="LLM summarize text into bullet points. Returns plain text.",
    input_type=TypeName.TEXT,
    output_type=TypeName.TEXT,
    kind=ToolKind.PRIMITIVE,
    cost=0.15,
)
def summarize(text: str, max_words: int | None = 200) -> str:
    if max_words is None or not isinstance(max_words, int):
        max_words = 200
    log_tool(f"summarize max_words={max_words}")
    if not text or not str(text).strip():
        return ""
    out = _lm().chat(
        [
            ChatMessage(role="system", content="You are a careful summarizer. Be terse, factual, no padding."),
            ChatMessage(role="user", content=f"Summarize the following in <= {max_words} words as bullet points.\n\n---\n{str(text)[:8000]}"),
        ],
        temperature=0.3, top_p=0.9, max_tokens=400,
    )
    return out.text.strip()


@tool(
    name="extract_entities",
    description="Extract named entities (parties, dates, money, locations, laws). Returns dict of lists.",
    input_type=TypeName.TEXT,
    output_type=TypeName.ENTITIES,
    kind=ToolKind.PRIMITIVE,
    cost=0.15,
)
def extract_entities(text: str) -> dict[str, list[str]]:
    log_tool(f"extract_entities chars={len(text or '')}")
    if not text or not str(text).strip():
        return {"parties": [], "dates": [], "money": [], "locations": [], "laws": []}
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "EntityBag",
            "schema": {
                "type": "object",
                "properties": {
                    "parties": {"type": "array", "items": {"type": "string"}},
                    "dates": {"type": "array", "items": {"type": "string"}},
                    "money": {"type": "array", "items": {"type": "string"}},
                    "locations": {"type": "array", "items": {"type": "string"}},
                    "laws": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["parties","dates","money","locations","laws"],
                "additionalProperties": False,
            },
        },
    }
    out = _lm().chat(
        [
            ChatMessage(role="system", content="Extract entities. Respond with JSON only."),
            ChatMessage(role="user", content=f"TEXT:\n{str(text)[:6000]}"),
        ],
        temperature=0.2, top_p=0.9, max_tokens=600,
        response_format=schema,
    )
    try:
        return json.loads(out.text)
    except Exception:
        m = re.search(r"\{.*\}", out.text, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
        return {"parties": [], "dates": [], "money": [], "locations": [], "laws": []}


@tool(
    name="classify",
    description="Multi-label classify text against a provided label list. Returns the chosen label string.",
    input_type=TypeName.TEXT,
    output_type=TypeName.LABEL,
    kind=ToolKind.PRIMITIVE,
    cost=0.10,
)
def classify(text: str, labels: list[str] | None = None) -> str:
    if not labels:
        labels = ["yes","no","unclear"]
    log_tool(f"classify labels={labels}")
    out = _lm().chat(
        [
            ChatMessage(role="system", content="You are a strict classifier. Output ONLY one of the provided labels."),
            ChatMessage(role="user",
                        content=f"LABELS: {labels}\n\nTEXT:\n{str(text)[:4000]}\n\nOutput exactly one label."),
        ],
        temperature=0.2, top_p=0.9, max_tokens=20,
    )
    raw = (out.text or "").strip()
    if not raw:
        return labels[0]
    pick = raw.splitlines()[0].strip(' "\'.,:;')
    if pick not in labels:
        for L in labels:
            if L.lower() in pick.lower():
                return L
        return labels[0]
    return pick


@tool(
    name="normalize_data",
    description="Normalize a dict against a hint schema. Drops extra keys, fills missing with None, coerces obvious types.",
    input_type=TypeName.STRUCTURED_DATA,
    output_type=TypeName.STRUCTURED_DATA,
    kind=ToolKind.PRIMITIVE,
    cost=0.02,
)
def normalize_data(data: dict[str, Any], schema: dict[str, str] | None = None) -> dict[str, Any]:
    if not isinstance(data, dict): return {}
    if not schema: return data
    out: dict[str, Any] = {}
    for key, expected in schema.items():
        v = data.get(key)
        if v is None: out[key] = None; continue
        try:
            if expected == "float": out[key] = float(v)
            elif expected == "int": out[key] = int(v)
            elif expected == "str": out[key] = str(v)
            elif expected == "list": out[key] = list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else [v]
            else: out[key] = v
        except Exception:
            out[key] = None
    return out
