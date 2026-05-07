"""Promotion gate: convert a high-scoring pipeline into a Composite tool entry."""
from __future__ import annotations

import datetime as dt
from typing import Any

from ..tools.archive import MAPElitesArchive, PromotionDecision
from ..tools.base import Tool
from ..tools.registry import ToolLibrary
from ..ui.console import log_decision
from .pipeline import Pipeline


def composite_from_pipeline(
    *,
    pipeline: Pipeline,
    registry_tools: dict[str, Tool],
    domain: str | None,
    subdomain: str | None,
    capability_tag: str | None,
    score: float,
    parent_ids: list[str],
    session_id: str,
    description: str | None = None,
) -> dict[str, Any]:
    if not pipeline.steps:
        raise ValueError("empty pipeline")
    first = registry_tools[pipeline.steps[0].tool_name]
    last = registry_tools[pipeline.steps[-1].tool_name]
    base_id = f"comp_{(domain or 'general')[:4]}_{(capability_tag or 'gen')[:8]}"
    cid = f"{base_id}_{session_id[-6:]}"
    return {
        "id": cid,
        "kind": "composite",
        "domain": domain,
        "subdomain": subdomain,
        "capability_tag": capability_tag,
        "input_type": first.input_type.value,
        "output_type": last.output_type.value,
        "steps": [{"tool": s.tool_name, "params": s.params} for s in pipeline.steps],
        "lineage_parent_ids": parent_ids,
        "lineage_depth": 1 + max([0] + [0 for _ in parent_ids]),
        "metrics_history": [{"score": score, "session": session_id,
                              "ts": dt.datetime.utcnow().isoformat()}],
        "embedding": None,
        "description": description or f"Composite for {capability_tag or 'general'} in {domain or 'general'}",
        "born_at_session": session_id,
        "version": 1,
    }


def consider_promotion(
    *,
    library: ToolLibrary,
    archive: MAPElitesArchive,
    pipeline: Pipeline,
    score: float,
    parent_score: float,
    domain: str | None,
    subdomain: str | None,
    capability_tag: str | None,
    parent_ids: list[str],
    session_id: str,
    improvement_min: float,
) -> tuple[PromotionDecision, dict[str, Any] | None]:
    comp = composite_from_pipeline(
        pipeline=pipeline,
        registry_tools=library._tools,
        domain=domain, subdomain=subdomain, capability_tag=capability_tag,
        score=score, parent_ids=parent_ids, session_id=session_id,
    )
    decision = archive.evaluate_for_promotion(
        comp, parent_score=parent_score, score=score, improvement_min=improvement_min,
    )
    # Promotion gate is a 3-condition rule: (1) strict improvement over parent,
    # (2) no regression on regression suite [enforced upstream of this call],
    # (3) MAP-Elites novelty-or-domination. Log which condition fired so the
    # operator can see at a glance why a particular candidate was/wasn't promoted.
    cell = archive.cell_key(comp)
    occupant = archive.cells().get(cell)
    if decision.accepted:
        if decision.reason == "novel_cell":
            condition = "[gate-3 NOVEL] cell empty"
        elif decision.reason == "dominator":
            condition = (f"[gate-3 DOMINATE] beat occupant {decision.replaced} "
                         f"(prior score {occupant[1]:.3f} when checked)" if occupant else
                         "[gate-3 DOMINATE] replaced prior occupant")
        else:
            condition = f"[accepted: {decision.reason}]"
        log_decision(f"PROMOTE: {comp['id']} cell={cell} score={score:.3f} parent={parent_score:.3f}  {condition}")
        if decision.replaced and decision.replaced in library.composites:
            del library.composites[decision.replaced]
        library.register_composite(comp)
        return decision, comp
    if decision.reason == "no_improvement":
        condition = (f"[gate-1 NO-IMPROVEMENT] score {score:.3f} did not exceed "
                     f"parent {parent_score:.3f} by required +{improvement_min:.2f}")
    elif decision.reason == "dominated":
        condition = (f"[gate-3 DOMINATED] cell already occupied by {occupant[0]} "
                     f"with score {occupant[1]:.3f}" if occupant else
                     f"[gate-3 DOMINATED] cell occupant has higher score")
    else:
        condition = f"[reject: {decision.reason}]"
    log_decision(f"reject promote: {comp['id']} cell={cell} score={score:.3f} parent={parent_score:.3f}  {condition}")
    return decision, None
