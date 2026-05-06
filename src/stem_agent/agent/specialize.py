"""End-to-end orchestration: bootstrap -> seed -> beam-search -> promote.

Phases:
    PHASE 0 - load library
    PHASE 1 - domain bootstrapping + L1 evolution + promotion
    PHASE 2 - subdomain bootstrapping (deep track only) + L2 evolution + promotion
    PHASE 3 - frozen evaluation handled by eval/runner.py
"""
from __future__ import annotations

import datetime as dt
import random
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from ..config import EVO
from ..llm.lm_client import LMClient
from ..llm.judge_client import JudgeClient
from ..tools.archive import MAPElitesArchive
from ..tools.registry import ToolLibrary
from ..types import TypeName
from ..ui.console import log_info
from .beam_search import beam_search
from .pipeline import Pipeline, execute, validate
from .promote import consider_promotion
from .step_eval import evaluate_step, final_pipeline_score, should_terminate_early
from .stem import propose_seed_pipeline


@dataclass
class TaskSpec:
    name: str
    question: str
    input_type: TypeName
    initial_input: Any
    domain: str
    subdomain: str | None = None
    capability_tag: str | None = None
    reference: Any = None
    rubric: str | None = None


def make_session_id() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]


def _composites_summary(library: ToolLibrary, domain: str | None) -> str:
    rel = [c for c in library.composites.values()
           if (domain is None or c.get("domain") in (domain, None, "general"))]
    if not rel: return "(none)"
    lines = []
    for c in rel[:8]:
        s = (c.get("metrics_history") or [{}])[-1].get("score", 0.0)
        lines.append(f"- {c['id']}: {c.get('description','')} (score~{s:.2f})")
    return "\n".join(lines)


def make_fitness_fn(
    *,
    task: TaskSpec,
    library: ToolLibrary,
    judge: JudgeClient,
    deterministic_score: Callable[[Any], float] | None = None,
):
    """Return a fitness function for beam_search."""
    registry = library._tools

    def fit(pipeline: Pipeline) -> tuple[float, dict[str, Any]]:
        ok, msg = validate(pipeline, registry, task.input_type)
        if not ok:
            return -1.0, {"error": msg}
        result = execute(pipeline, registry, task.initial_input)
        step_scores: list[float] = []
        consistency_scores: list[float] = []
        for rec in result.step_outputs:
            if rec.error:
                step_scores.append(0.0); consistency_scores.append(0.0)
                continue
            q, c, d = evaluate_step(
                index=rec.index, tool_name=rec.tool_name,
                output=rec.output, domain=task.domain,
                judge=judge, task_question=task.question, rubric=task.rubric,
            )
            step_scores.append(0.5*q + 0.3*c + 0.2*d)
            consistency_scores.append(c)
        if should_terminate_early(step_scores):
            avg = sum(step_scores)/len(step_scores) if step_scores else 0.0
            return -0.5 + avg, {"early_term": True, "step_scores": step_scores}
        det = deterministic_score(result.final) if (deterministic_score and result.success) else None
        final_out_score = det if det is not None else (step_scores[-1] if step_scores else 0.0)
        avg_step = sum(step_scores)/len(step_scores) if step_scores else 0.0
        cons_across = sum(consistency_scores)/len(consistency_scores) if consistency_scores else 0.0
        complexity = float(len(pipeline.steps)) + sum(
            (registry[s.tool_name].cost if s.tool_name in registry else 0.0)
            for s in pipeline.steps
        )
        score = final_pipeline_score(
            final_output_score=final_out_score, avg_step_score=avg_step,
            consistency_across_steps=cons_across, complexity=complexity,
        )
        return score, {"final_out": final_out_score, "avg_step": avg_step,
                       "cons_across": cons_across, "complexity": complexity,
                       "deterministic": det}
    return fit


def run_phase_for_task(
    *,
    task: TaskSpec,
    library: ToolLibrary,
    archive: MAPElitesArchive,
    lm: LMClient,
    judge: JudgeClient,
    layer: str,
    domain_brief_text: str,
    session_id: str,
    deterministic_score: Callable[[Any], float] | None = None,
    parent_score: float = 0.0,
    parent_ids: list[str] | None = None,
) -> dict[str, Any]:
    log_info(f"=== Phase[{layer}] task={task.name} ===")
    seed = propose_seed_pipeline(
        task_question=task.question, task_input_type=task.input_type,
        library=library, lm=lm, domain=task.domain,
        domain_brief=domain_brief_text,
        composites_summary=_composites_summary(library, task.domain),
        layer=layer,
        capability_tag=task.capability_tag,
    )
    fit = make_fitness_fn(task=task, library=library, judge=judge,
                          deterministic_score=deterministic_score)
    rng = random.Random(hash(task.name) & 0xFFFFFFFF)
    result = beam_search(
        seed=seed, registry=library._tools, input_type=task.input_type,
        fitness_fn=fit, domain=task.domain, k=EVO.beam_k,
        iterations=EVO.max_iterations, max_steps=EVO.max_steps, rng=rng,
    )
    if result.best_pipeline is None:
        return {"task": task.name, "best_score": 0.0, "promoted": False}

    decision, comp = consider_promotion(
        library=library, archive=archive,
        pipeline=result.best_pipeline, score=result.best_score, parent_score=parent_score,
        domain=task.domain, subdomain=task.subdomain, capability_tag=task.capability_tag,
        parent_ids=parent_ids or [], session_id=session_id,
        improvement_min=EVO.promotion_min_improvement,
    )
    return {
        "task": task.name, "best_score": result.best_score,
        "best_pipeline": result.best_pipeline.to_dict(),
        "iterations_run": result.iterations_run,
        "promoted": bool(comp), "composite_id": (comp or {}).get("id"),
    }
