"""Step-wise evaluation and cumulative scoring.

Per-step score = 0.5*quality + 0.3*consistency + 0.2*domain_relev.
Cumulative recurrence: c_k = alpha*c_{k-1} + beta*s_k.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from ..config import EVO
from ..llm.judge_client import JudgeClient
from ..llm.prompts import load_prompt
from ..tools.evaluation import consistency_check
from ..ui.console import log_eval


@dataclass
class StepScore:
    index: int
    tool_name: str
    quality: float
    consistency: float
    domain_relevance: float
    cumulative: float

    @property
    def step_score(self) -> float:
        return 0.5 * self.quality + 0.3 * self.consistency + 0.2 * self.domain_relevance

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self); d["step_score"] = self.step_score; return d


def evaluate_step(
    *,
    index: int,
    tool_name: str,
    output: Any,
    domain: str | None,
    rubric: str | None = None,
    judge: JudgeClient | None = None,
    task_question: str = "",
) -> tuple[float, float, float]:
    """Returns (quality, consistency, domain_relevance) each in 0..1."""
    text = _stringify(output)
    cons = consistency_check(text)
    judge = judge or JudgeClient()
    rubric = rubric or load_prompt("judge_rubric")

    q = 0.5
    if text.strip():
        try:
            q = judge.score_with_rubric(
                question=task_question or f"Step {index} output check",
                answer=text[:4000], rubric=rubric,
            ).score
        except Exception:
            q = 0.5

    dr = 0.5
    if domain and text.strip():
        try:
            dr = judge.score_with_rubric(
                question=f"Is this output relevant to the {domain.upper()} domain (i.e., uses correct vocabulary, structure, and primary sources)? 0=irrelevant, 1=fully on-domain.",
                answer=text[:3000],
                rubric="Score domain relevance only. Ignore correctness; just judge whether the content is in-domain.",
            ).score
        except Exception:
            dr = 0.5

    log_eval(f"step {index} ({tool_name}): q={q:.2f} c={cons:.2f} d={dr:.2f}")
    return q, cons, dr


def _stringify(value: Any) -> str:
    if value is None: return ""
    if isinstance(value, str): return value
    if isinstance(value, (int, float, bool)): return str(value)
    try:
        import json
        return json.dumps(value, default=str, ensure_ascii=False)[:6000]
    except Exception:
        return str(value)[:6000]


def cumulative_scores(step_scores: list[float], alpha: float = EVO.cumulative_alpha,
                      beta: float = EVO.cumulative_beta) -> list[float]:
    out: list[float] = []
    prev = 0.0
    for s in step_scores:
        cur = alpha * prev + beta * s
        out.append(cur)
        prev = cur
    return out


def should_terminate_early(step_scores: list[float], threshold: float = EVO.early_term_step_min) -> bool:
    return any(s < threshold for s in step_scores)


def final_pipeline_score(
    *,
    final_output_score: float,
    avg_step_score: float,
    consistency_across_steps: float,
    complexity: float,
    lam: float = EVO.complexity_lambda,
    w_output: float = EVO.final_w_output,
    w_step_avg: float = EVO.final_w_step_avg,
    w_consistency: float = EVO.final_w_consistency,
) -> float:
    return (w_output * final_output_score
            + w_step_avg * avg_step_score
            + w_consistency * consistency_across_steps
            - lam * complexity)
