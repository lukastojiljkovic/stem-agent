"""Mutation operators on Pipelines. Each operator returns a *new* Pipeline
that is type-valid, or None if no valid mutation could be constructed.
"""
from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..tools.base import Tool, ToolKind
from ..types import TypeName, is_compatible
from ..ui.console import log_mutation
from .pipeline import Pipeline, PipelineStep, validate


class MutationKind(str, Enum):
    ADD = "add_step"
    REMOVE = "remove_step"
    REPLACE = "replace_step"
    REORDER = "reorder_steps"
    INJECT_DOMAIN = "inject_domain"
    PARAMETRIC = "parametric"


@dataclass
class Mutation:
    kind: MutationKind
    rationale: str
    pipeline: Pipeline


def _types_along(p: Pipeline, registry: dict[str, Tool], input_type: TypeName) -> list[TypeName]:
    types = [input_type]
    cur = input_type
    for s in p.steps:
        t = registry.get(s.tool_name)
        if t is None: break
        cur = t.output_type
        types.append(cur)
    return types


def _candidate_tools(registry: dict[str, Tool], producer: TypeName,
                     domain: str | None = None,
                     exclude_universal: bool = True) -> list[Tool]:
    out: list[Tool] = []
    for t in registry.values():
        if exclude_universal and t.kind == ToolKind.UNIVERSAL:
            continue
        if not is_compatible(producer, t.input_type):
            continue
        if domain and t.domain not in (None, "general", domain):
            continue
        out.append(t)
    return out


def add_step(p: Pipeline, registry: dict[str, Tool], input_type: TypeName,
             rng: random.Random, max_steps: int = 5) -> Pipeline | None:
    if len(p) >= max_steps: return None
    types = _types_along(p, registry, input_type)
    insert_positions = list(range(len(p) + 1))
    rng.shuffle(insert_positions)
    for pos in insert_positions:
        producer = types[pos]
        cands = _candidate_tools(registry, producer)
        rng.shuffle(cands)
        for t in cands:
            new_p = Pipeline(deepcopy(p.steps))
            new_p.steps.insert(pos, PipelineStep(t.name, _default_params(t)))
            ok, _ = validate(new_p, registry, input_type)
            if ok:
                log_mutation(f"add_step pos={pos} tool={t.name}")
                return new_p
    return None


def remove_step(p: Pipeline, registry: dict[str, Tool], input_type: TypeName,
                rng: random.Random) -> Pipeline | None:
    if len(p) <= 1: return None
    indices = list(range(len(p)))
    rng.shuffle(indices)
    for i in indices:
        new_p = Pipeline([s for j, s in enumerate(p.steps) if j != i])
        ok, _ = validate(new_p, registry, input_type)
        if ok:
            log_mutation(f"remove_step idx={i}")
            return new_p
    return None


def replace_step(p: Pipeline, registry: dict[str, Tool], input_type: TypeName,
                 rng: random.Random, target_index: int | None = None) -> Pipeline | None:
    if len(p) == 0: return None
    types = _types_along(p, registry, input_type)
    indices = [target_index] if target_index is not None else list(range(len(p)))
    rng.shuffle(indices)
    for i in indices:
        producer = types[i]
        consumer = types[i + 1] if i + 1 < len(types) else None
        cands = [t for t in _candidate_tools(registry, producer)
                 if t.name != p.steps[i].tool_name]
        rng.shuffle(cands)
        for t in cands:
            if consumer is not None and not is_compatible(t.output_type, consumer):
                continue
            new_p = Pipeline(deepcopy(p.steps))
            new_p.steps[i] = PipelineStep(t.name, _default_params(t))
            ok, _ = validate(new_p, registry, input_type)
            if ok:
                log_mutation(f"replace_step idx={i} tool={t.name}")
                return new_p
    return None


def reorder_steps(p: Pipeline, registry: dict[str, Tool], input_type: TypeName,
                  rng: random.Random) -> Pipeline | None:
    if len(p) < 2: return None
    pairs = [(i, i+1) for i in range(len(p)-1)]
    rng.shuffle(pairs)
    for i, j in pairs:
        new_p = Pipeline(deepcopy(p.steps))
        new_p.steps[i], new_p.steps[j] = new_p.steps[j], new_p.steps[i]
        ok, _ = validate(new_p, registry, input_type)
        if ok:
            log_mutation(f"reorder_steps swap=({i},{j})")
            return new_p
    return None


def inject_domain(p: Pipeline, registry: dict[str, Tool], input_type: TypeName,
                  rng: random.Random, domain: str, max_steps: int = 5) -> Pipeline | None:
    if len(p) >= max_steps: return None
    types = _types_along(p, registry, input_type)
    positions = list(range(len(p) + 1))
    rng.shuffle(positions)
    for pos in positions:
        producer = types[pos]
        cands = [t for t in _candidate_tools(registry, producer, domain=domain)
                 if t.domain == domain]
        rng.shuffle(cands)
        for t in cands:
            new_p = Pipeline(deepcopy(p.steps))
            new_p.steps.insert(pos, PipelineStep(t.name, _default_params(t)))
            ok, _ = validate(new_p, registry, input_type)
            if ok:
                log_mutation(f"inject_domain pos={pos} tool={t.name} domain={domain}")
                return new_p
    return None


def parametric_mutate(p: Pipeline, registry: dict[str, Tool], input_type: TypeName,
                      rng: random.Random) -> Pipeline | None:
    if len(p) == 0: return None
    indices = list(range(len(p)))
    rng.shuffle(indices)
    for i in indices:
        step = p.steps[i]
        t = registry.get(step.tool_name)
        if t is None or not t.parameters_schema:
            continue
        candidate_keys = [k for k, info in t.parameters_schema.items()
                          if k not in {"text","query","doc","document","documents","time_series",
                                       "filing","data","output","project","series_dict","payload","a","b"}]
        rng.shuffle(candidate_keys)
        for key in candidate_keys:
            new_val = _propose_param_value(step.params.get(key), t.parameters_schema[key], rng)
            if new_val is None:
                continue
            new_p = Pipeline(deepcopy(p.steps))
            new_p.steps[i].params[key] = new_val
            ok, _ = validate(new_p, registry, input_type)
            if ok:
                log_mutation(f"parametric idx={i} key={key} new={new_val!r}")
                return new_p
    return None


def _default_params(t: Tool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, info in (t.parameters_schema or {}).items():
        if not info.get("required", False):
            continue
        if k in {"text","query","doc","document","documents","time_series","filing",
                "data","output","project","series_dict","payload","a","b"}:
            continue
        v = info.get("default")
        if v is not None:
            out[k] = v
    return out


def _propose_param_value(current: Any, info: dict[str, Any], rng: random.Random) -> Any:
    typ = info.get("type", "Any")
    default = info.get("default")
    if typ in ("int", "float") and isinstance(current, (int, float)):
        delta = current * (rng.uniform(-0.5, 0.5)) if current else rng.uniform(-1.0, 1.0)
        new = current + delta
        return int(new) if typ == "int" else float(new)
    if typ == "str":
        choices = ["upper","lower","CUAD41","CUAD20","gdpr_art5"]
        return rng.choice([c for c in choices if c != current])
    if typ == "int" and current is None:
        return rng.choice([3, 5, 8])
    if typ == "float" and current is None:
        return rng.choice([0.2, 0.4, 0.6])
    return default if default != current else None
