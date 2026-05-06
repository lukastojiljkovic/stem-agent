"""Tests for processing primitives that don't require LM Studio."""
from stem_agent.tools.processing import normalize_data, html_extract


def test_normalize_data_coerces_and_drops_extras():
    out = normalize_data({"a": "12", "b": "x", "c": "ignored"}, {"a": "int", "b": "str"})
    assert out == {"a": 12, "b": "x"}


def test_normalize_data_handles_missing():
    out = normalize_data({}, {"a": "float"})
    assert out == {"a": None}


def test_html_extract_empty_on_unreachable():
    out = html_extract("http://localhost:9/this-port-is-closed")
    assert isinstance(out, str)
