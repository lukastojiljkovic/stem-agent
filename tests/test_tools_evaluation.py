from stem_agent.tools.evaluation import score_accuracy, consistency_check, completeness_check


def test_score_accuracy_perfect():
    assert score_accuracy("the quick brown fox", "the quick brown fox") == 1.0


def test_score_accuracy_partial():
    s = score_accuracy("the quick fox", "the quick brown fox")
    assert 0.5 < s < 1.0


def test_score_accuracy_disjoint():
    assert score_accuracy("abc xyz", "qrs tuv") == 0.0


def test_consistency_clean_text():
    assert consistency_check("Apple's revenue was $383B in FY2024. Net income was $97B.") > 0.85


def test_completeness_full_hit():
    assert completeness_check("revenue $300, net income $80, ROA 5%", ["revenue","net income","ROA"]) == 1.0


def test_completeness_partial():
    s = completeness_check("revenue is X, ROA is Y", ["revenue","net income","ROA"])
    assert abs(s - 2/3) < 1e-9
