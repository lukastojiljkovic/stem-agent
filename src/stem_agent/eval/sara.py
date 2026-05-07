"""SARA-style statutory-reasoning loader.

Reads `data/fixtures/tasks_legal.jsonl` rows where kind=='sara_like'.
Each row carries a question (statute + fact pattern) and a Yes/No gold answer.
"""
from __future__ import annotations

import json

from ..config import PATHS
from ..agent.specialize import TaskSpec
from ..types import TypeName


def load_sara_like_tasks() -> list[TaskSpec]:
    p = PATHS.fixtures / "tasks_legal.jsonl"
    if not p.exists():
        return []
    out: list[TaskSpec] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("kind") != "sara_like":
            continue
        out.append(TaskSpec(
            name=obj["id"],
            question=obj["question"],
            input_type=TypeName.TEXT,
            initial_input=obj["question"],
            domain="legal",
            subdomain=None,
            capability_tag="legal_qa",
            reference={"label": obj["gold_answer"]},
        ))
    return out
