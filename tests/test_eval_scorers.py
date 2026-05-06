from stem_agent.eval.scorers import (
    score_clauses_f1, score_obligations_overlap,
    score_ratios_within_tolerance, score_qa_answer_match,
    score_classification_accuracy,
)


def test_clauses_f1_perfect():
    pred = {"clauses":[{"category":"Governing Law","text":"governed by Delaware law"}]}
    gold = {"Governing Law": ["governed by Delaware law"]}
    assert score_clauses_f1(pred, gold) == 1.0


def test_clauses_f1_partial():
    pred = {"clauses":[{"category":"Governing Law","text":"Delaware law applies"}]}
    gold = {"Governing Law": ["governed by Delaware law"]}
    s = score_clauses_f1(pred, gold)
    assert 0.0 < s < 1.0


def test_obligations_overlap_perfect():
    pred = [{"party":"X","obligation":"deliver Y","trigger":"start","deadline":"30d"}]
    gold = [{"party":"X","obligation":"deliver Y"}]
    assert abs(score_obligations_overlap(pred, gold) - 1.0) < 1e-9


def test_ratios_tolerance():
    pred = {"ratios":{"current_ratio": 2.05, "roa": 0.052}}
    gold = {"current_ratio": 2.0, "roa": 0.05}
    assert abs(score_ratios_within_tolerance(pred, gold, tol=0.1) - 1.0) < 1e-9


def test_ratios_tolerance_partial():
    pred = {"ratios":{"current_ratio": 2.05, "roa": 0.07}}
    gold = {"current_ratio": 2.0, "roa": 0.05}
    s = score_ratios_within_tolerance(pred, gold, tol=0.1)
    assert abs(s - 0.5) < 1e-9


def test_classification_accuracy():
    assert score_classification_accuracy("YES", "yes") == 1.0
    assert score_classification_accuracy("a", "b") == 0.0


def test_qa_answer_match():
    assert score_qa_answer_match("the revenue was 100", "revenue 100") > 0.5
