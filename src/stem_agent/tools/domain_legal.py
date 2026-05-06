"""Domain-legal primitives. Wrap the LLM with strict JSON schemas + rule packs."""
from __future__ import annotations

import json
import re
from typing import Any

from ..config import PATHS
from ..llm.lm_client import LMClient, ChatMessage
from ..types import TypeName
from ..ui.console import log_tool
from .base import tool, ToolKind

_LM = None
def _lm() -> LMClient:
    global _LM
    if _LM is None: _LM = LMClient()
    return _LM


def _load_rule_pack(name: str) -> dict[str, Any]:
    p = PATHS.rule_packs / f"{name}.json"
    if not p.exists(): return {}
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return {}


@tool(
    name="clause_extraction",
    description="Extract clauses by category (default CUAD-41 taxonomy). Returns {clauses:[{category,text,confidence}]}.",
    input_type=TypeName.TEXT,
    output_type=TypeName.CLAUSES,
    kind=ToolKind.PRIMITIVE,
    domain="legal",
    subdomain="contract_analysis",
    capability_tag="clause_extraction",
    cost=0.30,
)
def clause_extraction(text: str, taxonomy: str | None = "CUAD41",
                      min_conf: float | None = 0.4, categories: list[str] | None = None) -> dict[str, Any]:
    # Defensive: callers (LLM seed proposer) sometimes pass None for params with defaults.
    if taxonomy is None: taxonomy = "CUAD41"
    if min_conf is None: min_conf = 0.4
    log_tool(f"clause_extraction taxonomy={taxonomy} min_conf={min_conf}")
    if not text or not str(text).strip():
        return {"clauses": []}

    if categories is None:
        if str(taxonomy).upper() == "CUAD41":
            pack = _load_rule_pack("cuad_taxonomy")
            categories = list(pack.get("categories") or [])[:20]
        if not categories:
            categories = ["Governing Law","Termination for Convenience","Cap on Liability",
                          "Anti-Assignment","Confidentiality","Indemnification"]

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "Clauses",
            "schema": {
                "type": "object",
                "properties": {"clauses": {"type": "array", "items": {
                    "type":"object",
                    "properties":{
                        "category":{"type":"string"},
                        "text":{"type":"string"},
                        "confidence":{"type":"number"}
                    },
                    "required":["category","text","confidence"],
                    "additionalProperties": False
                }}},
                "required":["clauses"],
                "additionalProperties": False,
            },
        },
    }
    sys = ("Extract verbatim clauses from a contract for the given categories. "
           "Provide the exact substring (no paraphrase). If a category is absent, omit it.")
    user = f"CATEGORIES: {categories}\n\nCONTRACT:\n{str(text)[:9000]}"
    out = _lm().chat(
        [ChatMessage(role="system", content=sys), ChatMessage(role="user", content=user)],
        temperature=0.2, top_p=0.9, max_tokens=1200, response_format=schema,
    )
    obj = _safe_json(out.text, default={"clauses": []})
    obj["clauses"] = [c for c in obj.get("clauses",[]) if float(c.get("confidence",0)) >= min_conf]
    return obj


@tool(
    name="obligation_detection",
    description="Detect obligations as {party,obligation,trigger,deadline}. Returns list.",
    input_type=TypeName.TEXT,
    output_type=TypeName.OBLIGATION_LIST,
    kind=ToolKind.PRIMITIVE,
    domain="legal",
    capability_tag="obligation",
    cost=0.30,
)
def obligation_detection(text: str) -> list[dict[str, Any]]:
    log_tool("obligation_detection")
    if not text or not str(text).strip(): return []
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "Oblig",
            "schema": {
                "type":"object",
                "properties":{"obligations":{"type":"array","items":{
                    "type":"object",
                    "properties":{
                        "party":{"type":"string"},
                        "obligation":{"type":"string"},
                        "trigger":{"type":"string"},
                        "deadline":{"type":"string"}
                    },
                    "required":["party","obligation","trigger","deadline"],
                    "additionalProperties": False
                }}},
                "required":["obligations"],
                "additionalProperties": False
            }
        }
    }
    sys = "Extract all obligations. Each obligation has a party, what they must do, trigger, deadline (or 'n/a')."
    out = _lm().chat(
        [ChatMessage(role="system", content=sys), ChatMessage(role="user", content=str(text)[:8000])],
        temperature=0.2, top_p=0.9, max_tokens=1000, response_format=schema,
    )
    obj = _safe_json(out.text, default={"obligations": []})
    return list(obj.get("obligations") or [])


@tool(
    name="rule_matching",
    description="Match text against a rule pack (default gdpr_art5). Returns hits as {rule_id,principle,evidence,severity}.",
    input_type=TypeName.TEXT,
    output_type=TypeName.RULE_HITS,
    kind=ToolKind.PRIMITIVE,
    domain="legal",
    capability_tag="rule_match",
    cost=0.20,
)
def rule_matching(text: str, rule_pack: str = "gdpr_art5") -> list[dict[str, Any]]:
    log_tool(f"rule_matching pack={rule_pack}")
    pack = _load_rule_pack(rule_pack)
    rules = pack.get("rules") or []
    if not text or not str(text).strip() or not rules: return []
    sys = "Identify rule violations or matches. JSON only."
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name":"Hits",
            "schema":{
                "type":"object",
                "properties":{"hits":{"type":"array","items":{
                    "type":"object",
                    "properties":{
                        "rule_id":{"type":"string"},
                        "principle":{"type":"string"},
                        "evidence":{"type":"string"},
                        "severity":{"type":"string"}
                    },
                    "required":["rule_id","principle","evidence","severity"],
                    "additionalProperties": False
                }}},
                "required":["hits"],
                "additionalProperties": False
            }
        }
    }
    user = (f"RULE PACK ({rule_pack}):\n{json.dumps(rules)[:6000]}\n\n"
            f"FACT PATTERN:\n{str(text)[:5000]}\n\n"
            "For each rule, decide whether it applies. Output only rules that match (severity in {low,med,high}).")
    out = _lm().chat(
        [ChatMessage(role="system", content=sys), ChatMessage(role="user", content=user)],
        temperature=0.2, top_p=0.9, max_tokens=900, response_format=schema,
    )
    obj = _safe_json(out.text, default={"hits": []})
    return list(obj.get("hits") or [])


def _safe_json(text: str, default: Any) -> Any:
    try: return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
        return default
