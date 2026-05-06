"""CUAD adapter — loads contracts + gold spans from data/fixtures/."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import PATHS
from ..agent.specialize import TaskSpec
from ..types import TypeName


def load_cuad_tasks() -> list[TaskSpec]:
    p = PATHS.fixtures / "tasks_legal.jsonl"
    if not p.exists(): return []
    out: list[TaskSpec] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        obj = json.loads(line)
        if obj.get("kind") != "cuad_clause": continue
        contract_path = PATHS.fixtures / "contracts" / obj["contract"]
        text = _load_contract_text(contract_path)
        out.append(TaskSpec(
            name=obj["id"],
            question=obj.get("question",
                "Extract the listed clauses (verbatim) from this contract."),
            input_type=TypeName.TEXT,
            initial_input=text,
            domain="legal",
            subdomain="contract_analysis",
            capability_tag="clause_extraction",
            reference=obj["gold"],
        ))
    return out


def _load_contract_text(path: Path) -> str:
    if not path.exists(): return ""
    if path.suffix.lower() == ".pdf":
        try:
            import fitz
            doc = fitz.open(str(path))
            text = "\n".join(p.get_text("text") for p in doc)
            doc.close()
            return text[:30000]
        except Exception:
            return ""
    return path.read_text(encoding="utf-8", errors="ignore")[:30000]
