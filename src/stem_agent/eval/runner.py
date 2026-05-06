"""Top-level run orchestration: baseline / L1 / L2 + cross-session library carry-forward."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..config import EVO, PATHS
from ..llm.lm_client import LMClient
from ..llm.judge_client import JudgeClient
from ..tools.archive import MAPElitesArchive
from ..tools.lineage import write_dot
from ..tools.registration import register_all
from ..tools.registry import ToolLibrary
from ..ui.console import log_decision, log_info, log_report
from ..ui.events import session_log
from ..agent.pipeline import Pipeline, PipelineStep, execute, validate
from ..agent.specialize import TaskSpec, make_session_id, run_phase_for_task
from ..agent.stem import bootstrap_domain_brief, render_brief
from .scorers import (
    score_clauses_f1, score_obligations_overlap,
    score_ratios_within_tolerance, score_qa_answer_match, score_classification_accuracy,
)


def _scorer_for(task: TaskSpec) -> Callable[[Any], float] | None:
    cap = task.capability_tag or ""
    ref = task.reference or {}
    if cap == "clause_extraction":
        return lambda out: score_clauses_f1(out, ref or {})
    if cap == "obligation":
        return lambda out: score_obligations_overlap(out if isinstance(out, list) else [], ref or [])
    if cap == "financial_ratios":
        return lambda out: score_ratios_within_tolerance(out or {}, (ref or {}).get("ratios") or {})
    if cap == "financial_qa":
        return lambda out: score_qa_answer_match(_textify(out), (ref or {}).get("answer",""))
    if cap == "legal_qa":
        gold = (ref or {}).get("label","")
        return lambda out: score_classification_accuracy(_textify(out), gold)
    return None


def _textify(x: Any) -> str:
    if isinstance(x, str): return x
    if isinstance(x, dict):
        for k in ("answer","summary","text","final"):
            if k in x and isinstance(x[k], str): return x[k]
    return str(x)


def _baseline_pipeline(task: TaskSpec) -> Pipeline:
    cap = task.capability_tag or ""
    if cap == "clause_extraction":
        return Pipeline([PipelineStep("clause_extraction"), PipelineStep("summarize")])
    if cap == "obligation":
        return Pipeline([PipelineStep("obligation_detection")])
    if cap == "financial_ratios":
        return Pipeline([PipelineStep("edgar_fetch"), PipelineStep("financial_ratios")])
    if cap == "financial_qa":
        return Pipeline([PipelineStep("edgar_fetch"), PipelineStep("summarize")])
    if cap == "legal_qa":
        return Pipeline([PipelineStep("classify", {"labels": ["yes","no","unclear"]})])
    return Pipeline([PipelineStep("web_search"), PipelineStep("summarize")])


def _eval_one(*, pipeline: Pipeline, task: TaskSpec, library: ToolLibrary) -> dict[str, Any]:
    ok, msg = validate(pipeline, library._tools, task.input_type)
    if not ok: return {"task": task.name, "score": 0.0, "error": msg}
    res = execute(pipeline, library._tools, task.initial_input)
    sc = 0.0
    fn = _scorer_for(task)
    if fn and res.success:
        try: sc = fn(res.final)
        except Exception: sc = 0.0
    return {"task": task.name, "score": sc,
            "elapsed_s": res.total_elapsed_s,
            "success": res.success, "error": res.error}


def _load_tasks(domain: str, subdomain: str | None, track: str) -> list[TaskSpec]:
    from .cuad import load_cuad_tasks
    from .legalbench import load_legalbench_tasks
    from .financebench import load_financebench_tasks
    from .edgar_eval import load_ratio_tasks
    tasks: list[TaskSpec] = []
    if domain == "legal":
        tasks += load_cuad_tasks()
        # LegalBench currently dropped from the eval set: many items have empty
        # gold labels that make the baseline match by accident, so the metric
        # is uninformative. Re-enable when we have a robust label normalizer.
    if domain == "economics":
        tasks += load_ratio_tasks()
        tasks += load_financebench_tasks()
    return tasks


def _split_tasks(tasks: list[TaskSpec]) -> tuple[list[TaskSpec], list[TaskSpec]]:
    by_cap: dict[str, list[TaskSpec]] = {}
    for t in tasks: by_cap.setdefault(t.capability_tag or "", []).append(t)
    train: list[TaskSpec] = []; evals: list[TaskSpec] = []
    for cap, group in by_cap.items():
        n = len(group); k = max(1, min(5, n // 2))
        train += group[:k]; evals += group[k:]
    return train, evals


def run_full(*, domain: str, subdomain: str | None, track: str, seed: int,
             do_bootstrap: bool) -> int:
    session_id = make_session_id()
    runs_dir = PATHS.runs / session_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    log = session_log(session_id, PATHS.runs)
    log.emit("session.start", session_id=session_id, domain=domain, subdomain=subdomain, track=track)

    library = ToolLibrary(PATHS.tool_library)
    register_all(library)
    library.load()
    archive = MAPElitesArchive()
    lm = LMClient()
    judge = JudgeClient()

    tasks = _load_tasks(domain, subdomain, track)
    if not tasks:
        log_info("No tasks found. Did you create fixtures (plan 09)?")
        return 1
    train, evals = _split_tasks(tasks)
    log_info(f"loaded {len(tasks)} tasks: {len(train)} train + {len(evals)} eval")

    brief_text = ""
    if do_bootstrap:
        brief = bootstrap_domain_brief(domain, lm, library)
        brief_text = render_brief(brief)
        (runs_dir / "domain_brief.md").write_text(brief_text, encoding="utf-8")
        log.emit("bootstrap.brief", domain=domain, sources=brief.sources, paragraphs=brief.paragraphs)

    cap_per_phase = EVO.max_train_tasks_per_phase
    log_decision("=== PHASE 1: L1 specialization ===")
    l1_layer = domain
    l1_train = [t for t in train if not (t.subdomain and track == "deep" and l1_layer == "legal")]
    for t in l1_train[:cap_per_phase]:
        run_phase_for_task(
            task=t, library=library, archive=archive, lm=lm, judge=judge,
            layer=l1_layer, domain_brief_text=brief_text,
            session_id=session_id,
            deterministic_score=_scorer_for(t),
        )

    if track == "deep" and domain == "legal" and (subdomain == "contract_analysis" or subdomain is None):
        log_decision("=== PHASE 2: L2 (contract_analysis) ===")
        l2_train = [x for x in train if x.subdomain == "contract_analysis"]
        for t in l2_train[:cap_per_phase]:
            run_phase_for_task(
                task=t, library=library, archive=archive, lm=lm, judge=judge,
                layer="contract_analysis",
                domain_brief_text=brief_text,
                session_id=session_id,
                deterministic_score=_scorer_for(t),
            )

    log_decision("=== PHASE 3: frozen evaluation ===")
    metrics = {"baseline": [], "L1": [], "L2": []}
    for t in evals:
        bp = _baseline_pipeline(t)
        bm = _eval_one(pipeline=bp, task=t, library=library)
        metrics["baseline"].append(bm); log.emit("eval", level="baseline", **bm)

    l1_pipes = _best_composite_pipelines_by_cap(library, prefer_subdomain=None)
    for t in evals:
        p = l1_pipes.get(t.capability_tag or "") or _baseline_pipeline(t)
        m = _eval_one(pipeline=p, task=t, library=library)
        metrics["L1"].append(m); log.emit("eval", level="L1", **m)

    if track == "deep" and domain == "legal":
        l2_pipes = _best_composite_pipelines_by_cap(library, prefer_subdomain="contract_analysis")
        for t in evals:
            p = l2_pipes.get(t.capability_tag or "") or _baseline_pipeline(t)
            m = _eval_one(pipeline=p, task=t, library=library)
            metrics["L2"].append(m); log.emit("eval", level="L2", **m)

    library.save()
    library.snapshot(session_id)
    write_dot(library.composites.values(), PATHS.tool_library / "lineage.dot")
    (runs_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.emit("session.end", metrics_summary={k: _summary(v) for k, v in metrics.items()})

    _render_agent_answer(runs_dir, session_id, domain, metrics)

    log_report(f"session done: {session_id}; metrics in {runs_dir/'metrics.json'}")
    log.close()
    return 0


def _summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows: return {"n":0,"mean":0.0,"min":0.0,"max":0.0}
    vals = [r.get("score",0.0) for r in rows]
    return {"n": len(vals), "mean": sum(vals)/len(vals),
            "min": min(vals), "max": max(vals)}


def _best_composite_pipelines_by_cap(
    library: ToolLibrary,
    *,
    prefer_subdomain: str | None,
) -> dict[str, Pipeline]:
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for c in library.composites.values():
        cap = c.get("capability_tag") or "generic"
        if prefer_subdomain and c.get("subdomain") and c.get("subdomain") != prefer_subdomain:
            continue
        s = max((m.get("score",0.0) for m in (c.get("metrics_history") or [])), default=0.0)
        cur = best.get(cap)
        if cur is None or s > cur[0]:
            best[cap] = (s, c)
    out: dict[str, Pipeline] = {}
    for cap, (_, c) in best.items():
        out[cap] = Pipeline([PipelineStep(s["tool"], dict(s.get("params") or {})) for s in c.get("steps", [])])
    return out


def _render_agent_answer(runs_dir: Path, session_id: str, domain: str,
                         metrics: dict[str, list[dict[str, Any]]]) -> None:
    from ..tools.universal.report_finalize import report_finalize

    rows: list[list[str]] = [["Level","N","Mean","Min","Max"]]
    for level in ("baseline","L1","L2"):
        s = _summary(metrics.get(level, []))
        rows.append([level, str(s["n"]), f"{s['mean']:.3f}", f"{s['min']:.3f}", f"{s['max']:.3f}"])

    md = (
        f"# Stem Agent — answer for session `{session_id}` ({domain})\n\n"
        f"This document is the agent's final answer, produced by composing the FROZEN "
        f"LaTeX/grammar/PDF tools at the end of the run. It summarizes the agent's "
        f"performance across the three evaluation levels.\n\n"
        f"## Summary\n\n"
        f"The persisted tool library was used to build domain- and subdomain-specialized "
        f"pipelines, then evaluated against a held-out task set. See the metrics table.\n"
    )
    out_pdf = runs_dir / "reports" / "agent_answer.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    archive_dir = runs_dir / "tex_archive"
    try:
        report_finalize({
            "title": f"Stem Agent — Session {session_id[:13]}",
            "answer_md": md,
            "metrics_table": rows,
            "out_path": str(out_pdf),
            "archive_tex_dir": str(archive_dir),
            "figures": [{
                "kind": "bar",
                "data": {
                    "labels": [lv for lv in ("baseline","L1","L2") if metrics.get(lv)],
                    "values": [_summary(metrics.get(lv, []))["mean"]
                               for lv in ("baseline","L1","L2") if metrics.get(lv)],
                    "title": "Mean score by level",
                    "ylabel": "score",
                },
                "caption": "Mean score: baseline vs L1 vs L2",
                "out_dir": str(runs_dir / "figures"),
            }],
        })
    except Exception as e:
        log_report(f"agent answer rendering failed: {e}")


def run_baseline_only(*, domain: str) -> int:
    library = ToolLibrary(PATHS.tool_library)
    register_all(library); library.load()
    tasks = _load_tasks(domain, None, "shallow")
    rows = []
    for t in tasks:
        bp = _baseline_pipeline(t)
        m = _eval_one(pipeline=bp, task=t, library=library)
        rows.append(m)
    log_report(f"baseline n={len(rows)} mean={sum(r.get('score',0) for r in rows)/max(1,len(rows)):.3f}")
    return 0


def show_library(*, lineage: bool) -> int:
    library = ToolLibrary(PATHS.tool_library)
    register_all(library); library.load()
    log_info(f"primitives: {len(library.all_primitives())}")
    log_info(f"composites: {len(library.composites)}")
    for c in library.composites.values():
        s = (c.get('metrics_history') or [{}])[-1].get('score',0)
        log_info(f"  {c['id']:<48} {c.get('domain','-'):<10} {c.get('capability_tag','-'):<22} score~{s:.2f}")
    if lineage:
        write_dot(library.composites.values(), PATHS.tool_library / "lineage.dot")
        log_info(f"lineage written to {PATHS.tool_library / 'lineage.dot'}")
    return 0
