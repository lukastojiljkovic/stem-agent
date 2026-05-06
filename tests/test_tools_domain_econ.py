from stem_agent.tools.domain_econ import compute_indicators, trend_analysis, correlation_analysis, financial_ratios


def test_compute_indicators_basic():
    out = compute_indicators({"series_id":"S","values":[1,2,3,4,5]})
    assert out["n"] == 5
    assert out["mean"] == 3
    assert out["last"] == 5


def test_trend_up():
    out = trend_analysis({"values": list(range(20))})
    assert out["direction"] == "up"
    assert out["r2"] > 0.99


def test_trend_flat():
    out = trend_analysis({"values": [5]*20})
    assert out["direction"] == "flat"


def test_correlation_two_series():
    out = correlation_analysis({"a":[1,2,3,4,5], "b":[2,4,6,8,10]})
    assert abs(out["matrix"][0][1] - 1.0) < 1e-9


def test_financial_ratios_with_full_facts():
    filing = {"ticker":"TEST","xbrl_facts":{
        "revenue":1000,"net_income":100,"total_assets":2000,
        "current_assets":600,"current_liabilities":300,
        "total_liabilities":800,"stockholders_equity":1200,
        "operating_income":150,"cash_from_ops":120
    }}
    out = financial_ratios(filing)
    assert abs(out["ratios"]["current_ratio"] - 2.0) < 1e-9
    assert abs(out["ratios"]["roa"] - 0.05) < 1e-9
    assert out["ratios"]["altman_z"] is not None
