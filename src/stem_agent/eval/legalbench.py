"""LegalBench micro-slice loader."""
from __future__ import annotations

import os
from pathlib import Path

from ..agent.specialize import TaskSpec
from ..types import TypeName


_TASKS_DEFAULT = ("hearsay","proa","contract_qa")


def load_legalbench_tasks(tasks: tuple[str, ...] = _TASKS_DEFAULT,
                          n_per_task: int = 30) -> list[TaskSpec]:
    out: list[TaskSpec] = []
    try:
        from datasets import load_dataset
    except Exception:
        return out
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(".cache") / "hf_datasets"))
    for tname in tasks:
        try:
            ds = load_dataset("nguha/legalbench", tname, split="test")
        except Exception:
            continue
        for i, row in enumerate(ds):
            if i >= n_per_task: break
            text = row.get("text") or row.get("question") or row.get("input") or ""
            label = str(row.get("label","") or row.get("answer","") or "")
            out.append(TaskSpec(
                name=f"lb_{tname}_{i:02d}",
                question=f"Classify or answer the following LegalBench[{tname}] item.",
                input_type=TypeName.TEXT,
                initial_input=text,
                domain="legal",
                subdomain=None,
                capability_tag="legal_qa",
                reference={"label": label},
            ))
    return out
