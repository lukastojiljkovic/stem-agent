"""MAP-Elites archive over (domain, capability_tag) cells.

Used to gate promotion of candidate composite pipelines. Acceptance rule:

  1. If the cell is empty -> NOVEL: accept.
  2. Else if candidate score > current cell occupant score by >= EPS -> DOMINATE: accept and replace.
  3. Else -> DOMINATED: reject.

This guarantees structural diversity in the persistent library and mitigates
mode collapse on a single mega-pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EPS = 1e-9


@dataclass
class PromotionDecision:
    accepted: bool
    reason: str
    replaced: str | None = None


class MAPElitesArchive:
    def __init__(self) -> None:
        self._cells: dict[tuple[str, str], tuple[str, float]] = {}

    def cell_key(self, comp: dict[str, Any]) -> tuple[str, str]:
        return (comp.get("domain") or "general", comp.get("capability_tag") or "generic")

    def occupant(self, comp: dict[str, Any]) -> tuple[str, float] | None:
        return self._cells.get(self.cell_key(comp))

    def evaluate_for_promotion(
        self,
        comp: dict[str, Any],
        *,
        parent_score: float,
        score: float,
        improvement_min: float = 0.02,
    ) -> PromotionDecision:
        key = self.cell_key(comp)
        cur = self._cells.get(key)
        if score < parent_score + improvement_min:
            return PromotionDecision(accepted=False, reason="no_improvement")

        if cur is None:
            self._cells[key] = (comp["id"], score)
            return PromotionDecision(accepted=True, reason="novel_cell")

        cur_id, cur_score = cur
        if score > cur_score + EPS:
            self._cells[key] = (comp["id"], score)
            return PromotionDecision(accepted=True, reason="dominator", replaced=cur_id)

        return PromotionDecision(accepted=False, reason="dominated")

    def cells(self) -> dict[tuple[str, str], tuple[str, float]]:
        return dict(self._cells)
