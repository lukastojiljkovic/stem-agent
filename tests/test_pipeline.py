"""Tests for Pipeline construction, validation, and execution."""
import pytest

from stem_agent.agent.pipeline import Pipeline, PipelineStep, validate, execute
from stem_agent.tools.base import tool
from stem_agent.types import TypeName


@tool(name="t_query_to_docs", description="", input_type=TypeName.QUERY, output_type=TypeName.DOCUMENTS)
def t_query_to_docs(query: str): return [{"title":"x","url":"","snippet":query}]

@tool(name="t_docs_to_text", description="", input_type=TypeName.DOCUMENTS, output_type=TypeName.TEXT)
def t_docs_to_text(documents): return " ".join(d.get("snippet","") for d in documents)

@tool(name="t_text_to_text_loud", description="", input_type=TypeName.TEXT, output_type=TypeName.TEXT)
def t_text_to_text_loud(text): return text.upper()

@tool(name="t_text_to_score", description="", input_type=TypeName.TEXT, output_type=TypeName.SCORE)
def t_text_to_score(text): return min(1.0, len(text) / 100.0)


def _registry():
    return {
        "t_query_to_docs": t_query_to_docs.tool,
        "t_docs_to_text": t_docs_to_text.tool,
        "t_text_to_text_loud": t_text_to_text_loud.tool,
        "t_text_to_score": t_text_to_score.tool,
    }


def test_validate_accepts_correct_chain():
    p = Pipeline([PipelineStep("t_query_to_docs", {}),
                  PipelineStep("t_docs_to_text", {}),
                  PipelineStep("t_text_to_text_loud", {})])
    ok, _ = validate(p, _registry(), input_type=TypeName.QUERY)
    assert ok


def test_validate_rejects_typed_mismatch():
    p_bad = Pipeline([PipelineStep("t_text_to_score", {}),
                      PipelineStep("t_query_to_docs", {})])
    ok, msg = validate(p_bad, _registry(), input_type=TypeName.TEXT)
    assert not ok
    assert "incompatible" in msg.lower() or "type" in msg.lower()


def test_validate_rejects_too_long():
    p = Pipeline([PipelineStep("t_text_to_text_loud", {})] * 6)
    ok, msg = validate(p, _registry(), input_type=TypeName.TEXT)
    assert not ok


def test_validate_rejects_unknown_tool():
    p = Pipeline([PipelineStep("does_not_exist", {})])
    ok, msg = validate(p, _registry(), input_type=TypeName.QUERY)
    assert not ok


def test_execute_returns_per_step_outputs_and_final():
    p = Pipeline([PipelineStep("t_query_to_docs", {}),
                  PipelineStep("t_docs_to_text", {}),
                  PipelineStep("t_text_to_text_loud", {})])
    res = execute(p, _registry(), initial_input="hello")
    assert res.success
    assert len(res.step_outputs) == 3
    assert isinstance(res.final, str)
    assert res.final == res.final.upper()


def test_execute_step_failure_recorded():
    @tool(name="t_explode", description="", input_type=TypeName.TEXT, output_type=TypeName.TEXT)
    def t_explode(text): raise RuntimeError("boom")
    reg = _registry(); reg["t_explode"] = t_explode.tool
    p = Pipeline([PipelineStep("t_query_to_docs", {}),
                  PipelineStep("t_docs_to_text", {}),
                  PipelineStep("t_explode", {})])
    res = execute(p, reg, initial_input="hi")
    assert not res.success
    assert res.failed_step_index == 2
    assert "boom" in (res.error or "")
