from stem_agent.tools.domain_legal import clause_extraction, obligation_detection, rule_matching


def test_clause_extraction_handles_empty():
    out = clause_extraction("")
    assert out == {"clauses": []}


def test_obligation_detection_handles_empty():
    assert obligation_detection("") == []


def test_rule_matching_handles_empty():
    assert rule_matching("") == []
