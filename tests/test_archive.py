"""Tests for the MAP-Elites archive used to gate composite-tool promotion."""
import pytest

from stem_agent.tools.archive import MAPElitesArchive


def _comp(cid: str, domain: str, cap: str, score: float, parent: str | None = None):
    return {
        "id": cid,
        "domain": domain,
        "capability_tag": cap,
        "metrics_history": [{"f1": score, "step_avg": score}],
        "lineage_parent_ids": [parent] if parent else [],
    }


def test_first_in_cell_is_accepted_as_novel():
    arc = MAPElitesArchive()
    c = _comp("a1", "legal", "clause_extraction", 0.5)
    decision = arc.evaluate_for_promotion(c, parent_score=0.0, score=0.5)
    assert decision.accepted is True
    assert decision.reason == "novel_cell"


def test_dominated_candidate_is_rejected():
    arc = MAPElitesArchive()
    arc.evaluate_for_promotion(
        _comp("a1", "legal", "clause_extraction", 0.7), parent_score=0.0, score=0.7
    )
    decision = arc.evaluate_for_promotion(
        _comp("a2", "legal", "clause_extraction", 0.5), parent_score=0.7, score=0.5,
    )
    assert decision.accepted is False
    assert decision.reason in ("dominated", "no_improvement")


def test_strict_pareto_dominator_replaces_cell_occupant():
    arc = MAPElitesArchive()
    arc.evaluate_for_promotion(
        _comp("a1", "legal", "clause_extraction", 0.5), parent_score=0.0, score=0.5
    )
    decision = arc.evaluate_for_promotion(
        _comp("a2", "legal", "clause_extraction", 0.7, parent="a1"),
        parent_score=0.5, score=0.7
    )
    assert decision.accepted is True
    assert decision.replaced == "a1"


def test_from_composites_seeds_cell_occupancy():
    """A loaded library must reconstitute the archive: cells are occupied by
    the highest-scoring prior composite per (domain, capability)."""
    composites = [
        {**_comp("legacy_a", "legal", "clause_extraction", 0.0), "metrics_history": [{"score": 0.50}]},
        {**_comp("legacy_b", "legal", "clause_extraction", 0.0), "metrics_history": [{"score": 0.70}]},
        {**_comp("legacy_c", "legal", "obligation",        0.0), "metrics_history": [{"score": 0.40}]},
    ]
    arc = MAPElitesArchive.from_composites(composites)
    cells = arc.cells()
    assert cells[("legal", "clause_extraction")] == ("legacy_b", 0.70)
    assert cells[("legal", "obligation")] == ("legacy_c", 0.40)


def test_from_composites_then_dominator_must_beat_seeded_score():
    composites = [
        {**_comp("legacy", "legal", "clause_extraction", 0.0), "metrics_history": [{"score": 0.70}]},
    ]
    arc = MAPElitesArchive.from_composites(composites)
    # A new candidate at 0.65 should be DOMINATED by the seeded occupant.
    decision = arc.evaluate_for_promotion(
        _comp("new_low", "legal", "clause_extraction", 0.65),
        parent_score=0.0, score=0.65,
    )
    assert decision.accepted is False
    assert decision.reason == "dominated"
