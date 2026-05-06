"""Beam search over typed tool pipelines using mutation operators.

`fitness_fn(p) -> (score, info)` is provided by the caller. The beam keeps
`k` best valid pipelines per iteration, mutating each with several proposals.
Rollback: if best-of-iteration score declines for `rollback_after_n_declines`,
revert to previous best snapshot before continuing.
"""
from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import EVO
from ..tools.base import Tool
from ..types import TypeName
from ..ui.console import log_decision, log_eval
from .mutations import (
    add_step, remove_step, replace_step, reorder_steps, inject_domain, parametric_mutate
)
from .pipeline import Pipeline, validate


FitnessFn = Callable[[Pipeline], tuple[float, dict[str, Any]]]


@dataclass
class BeamResult:
    best_pipeline: Pipeline | None
    best_score: float
    history: list[dict[str, Any]] = field(default_factory=list)
    iterations_run: int = 0


def beam_search(
    *,
    seed: Pipeline,
    registry: dict[str, Tool],
    input_type: TypeName,
    fitness_fn: FitnessFn,
    domain: str | None = None,
    k: int = EVO.beam_k,
    iterations: int = EVO.max_iterations,
    max_steps: int = EVO.max_steps,
    rng: random.Random | None = None,
    score_threshold: float = EVO.score_threshold_termination,
    epsilon: float = EVO.epsilon_termination,
    n_iters_for_epsilon: int = EVO.n_iters_for_epsilon,
    rollback_after_n_declines: int = EVO.rollback_after_n_declines,
) -> BeamResult:
    rng = rng or random.Random()
    ok, msg = validate(seed, registry, input_type, max_steps=max_steps)
    if not ok:
        return BeamResult(best_pipeline=None, best_score=0.0,
                          history=[{"reason": f"invalid seed: {msg}"}])

    score, info = fitness_fn(seed)
    beam: list[tuple[float, Pipeline]] = [(score, seed)]
    best_score = score
    best_pipe = seed
    history: list[dict[str, Any]] = [{"iter":0, "best": best_score, "info": info}]
    declines = 0
    snapshot_best: tuple[float, Pipeline] = (best_score, deepcopy(best_pipe))
    recent_bests: list[float] = [best_score]

    log_decision(f"beam_search start k={k} iters={iterations} seed_score={best_score:.3f}")

    for it in range(1, iterations + 1):
        proposals: list[Pipeline] = []
        for sc, p in beam:
            for op in (_op_add, _op_remove, _op_replace, _op_reorder, _op_inject, _op_param):
                cand = op(p, registry, input_type, rng, domain=domain, max_steps=max_steps)
                if cand is not None:
                    proposals.append(cand)
        scored: list[tuple[float, Pipeline, dict[str, Any]]] = []
        for p in proposals:
            s, inf = fitness_fn(p)
            scored.append((s, p, inf))
        all_candidates = [(s, p) for s, p, _ in scored] + beam
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beam = all_candidates[:k]
        cur_best_score, cur_best_pipe = beam[0]
        log_eval(f"iter={it} best_in_beam={cur_best_score:.3f} candidates={len(scored)}")

        if cur_best_score > best_score + epsilon:
            best_score = cur_best_score; best_pipe = cur_best_pipe
            snapshot_best = (best_score, deepcopy(best_pipe))
            declines = 0
        else:
            declines += 1
            if declines >= rollback_after_n_declines:
                log_decision(f"rollback @ iter={it} -> previous best ({snapshot_best[0]:.3f})")
                best_score, best_pipe = snapshot_best
                beam = [(best_score, deepcopy(best_pipe))]
                declines = 0

        history.append({"iter": it, "best": best_score})
        recent_bests.append(best_score)
        if best_score >= score_threshold:
            log_decision(f"threshold reached @ iter={it} score={best_score:.3f}")
            return BeamResult(best_pipeline=best_pipe, best_score=best_score,
                              history=history, iterations_run=it)
        if len(recent_bests) > n_iters_for_epsilon:
            window = recent_bests[-n_iters_for_epsilon-1:]
            if max(window) - min(window) < epsilon:
                log_decision(f"epsilon-stop @ iter={it} (Delta < {epsilon})")
                return BeamResult(best_pipeline=best_pipe, best_score=best_score,
                                  history=history, iterations_run=it)

    return BeamResult(best_pipeline=best_pipe, best_score=best_score,
                      history=history, iterations_run=iterations)


def _op_add(p, reg, it, rng, domain, max_steps):    return add_step(p, reg, it, rng, max_steps=max_steps)
def _op_remove(p, reg, it, rng, domain, max_steps): return remove_step(p, reg, it, rng)
def _op_replace(p, reg, it, rng, domain, max_steps): return replace_step(p, reg, it, rng)
def _op_reorder(p, reg, it, rng, domain, max_steps): return reorder_steps(p, reg, it, rng)
def _op_inject(p, reg, it, rng, domain, max_steps):
    return inject_domain(p, reg, it, rng, domain=domain or "general", max_steps=max_steps) if domain else None
def _op_param(p, reg, it, rng, domain, max_steps):  return parametric_mutate(p, reg, it, rng)
