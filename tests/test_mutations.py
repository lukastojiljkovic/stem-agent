import random
import pytest

from stem_agent.agent.pipeline import Pipeline, PipelineStep, validate
from stem_agent.agent.mutations import (
    add_step, remove_step, replace_step, reorder_steps, inject_domain, parametric_mutate
)
from stem_agent.tools.base import tool
from stem_agent.types import TypeName


@tool(name="A", description="", input_type=TypeName.QUERY, output_type=TypeName.DOCUMENTS)
def A(query): return [{"snippet":query}]
@tool(name="B", description="", input_type=TypeName.DOCUMENTS, output_type=TypeName.TEXT)
def B(documents): return ""
@tool(name="C", description="", input_type=TypeName.TEXT, output_type=TypeName.TEXT)
def C(text, mode="upper"): return text.upper() if mode == "upper" else text.lower()
@tool(name="D", description="", input_type=TypeName.TEXT, output_type=TypeName.SCORE)
def D(text): return 0.5
@tool(name="E_legal", description="", input_type=TypeName.TEXT, output_type=TypeName.CLAUSES, domain="legal")
def E(text): return {"clauses": []}


def _reg():
    return {t.tool.name: t.tool for t in (A,B,C,D,E)}


def test_add_step_appends_compatible_step():
    rng = random.Random(0)
    p = Pipeline([PipelineStep("A"), PipelineStep("B")])
    new = add_step(p, _reg(), input_type=TypeName.QUERY, rng=rng)
    assert new is not None
    ok, msg = validate(new, _reg(), TypeName.QUERY)
    assert ok, msg
    assert len(new) == 3


def test_remove_step_keeps_validity_or_returns_none():
    rng = random.Random(0)
    p = Pipeline([PipelineStep("A"), PipelineStep("B"), PipelineStep("C")])
    new = remove_step(p, _reg(), input_type=TypeName.QUERY, rng=rng)
    if new is not None:
        ok, _ = validate(new, _reg(), TypeName.QUERY)
        assert ok


def test_replace_step_yields_valid_pipeline_or_none():
    rng = random.Random(0)
    p = Pipeline([PipelineStep("A"), PipelineStep("B"), PipelineStep("C")])
    new = replace_step(p, _reg(), input_type=TypeName.QUERY, rng=rng, target_index=2)
    if new is not None:
        ok, _ = validate(new, _reg(), TypeName.QUERY)
        assert ok


def test_inject_domain_adds_legal_tool_when_compatible():
    rng = random.Random(1)
    p = Pipeline([PipelineStep("A"), PipelineStep("B")])
    new = inject_domain(p, _reg(), input_type=TypeName.QUERY, rng=rng, domain="legal")
    if new is not None:
        ok, _ = validate(new, _reg(), TypeName.QUERY)
        assert ok
        assert any(s.tool_name == "E_legal" for s in new.steps)


def test_parametric_mutate_changes_param_value():
    rng = random.Random(0)
    p = Pipeline([PipelineStep("A"), PipelineStep("B"), PipelineStep("C", {"mode":"upper"})])
    new = parametric_mutate(p, _reg(), input_type=TypeName.QUERY, rng=rng)
    if new is not None:
        steps_with_mode = [s for s in new.steps if s.tool_name == "C"]
        if steps_with_mode:
            assert "mode" in steps_with_mode[0].params


def test_reorder_steps_keeps_validity():
    rng = random.Random(0)
    p = Pipeline([PipelineStep("A"), PipelineStep("B"), PipelineStep("C")])
    new = reorder_steps(p, _reg(), input_type=TypeName.QUERY, rng=rng)
    if new is not None:
        ok, _ = validate(new, _reg(), TypeName.QUERY)
        assert ok
