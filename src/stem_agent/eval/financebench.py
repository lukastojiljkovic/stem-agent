"""FinanceBench-style adapter using a hand-curated micro-slice."""
from __future__ import annotations

import json

from ..config import PATHS
from ..agent.specialize import TaskSpec
from ..types import TypeName


def load_financebench_tasks() -> list[TaskSpec]:
    p = PATHS.fixtures / "tasks_economics.jsonl"
    if not p.exists(): return []
    out: list[TaskSpec] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        obj = json.loads(line)
        if obj.get("kind") != "fin_qa": continue
        out.append(TaskSpec(
            name=obj["id"],
            question=obj["question"],
            input_type=TypeName.QUERY,
            initial_input=f"{obj['ticker']} {obj['form']} {obj['year']}",
            domain="economics",
            subdomain=None,
            capability_tag="financial_qa",
            reference={"answer": obj["gold_answer"]},
        ))
    return out
