"""Financial-ratio extraction eval."""
from __future__ import annotations

import json

from ..config import PATHS
from ..agent.specialize import TaskSpec
from ..types import TypeName


def load_ratio_tasks() -> list[TaskSpec]:
    p = PATHS.fixtures / "tasks_economics.jsonl"
    if not p.exists(): return []
    out: list[TaskSpec] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        obj = json.loads(line)
        if obj.get("kind") != "ratios": continue
        out.append(TaskSpec(
            name=obj["id"],
            question=f"Compute standard financial ratios + Altman Z + Piotroski F for {obj['ticker']} {obj['form']} {obj['year']}.",
            input_type=TypeName.QUERY,
            initial_input=obj["ticker"],
            domain="economics",
            subdomain="financial_reporting",
            capability_tag="financial_ratios",
            reference={"ratios": obj["gold_ratios"]},
        ))
    return out
