import random
import pytest

from stem_agent.agent.pipeline import Pipeline, PipelineStep
from stem_agent.agent.beam_search import beam_search
from stem_agent.tools.base import tool
from stem_agent.types import TypeName


@tool(name="t_q_to_d", description="", input_type=TypeName.QUERY, output_type=TypeName.DOCUMENTS)
def q_to_d(query): return [{"x":query}]
@tool(name="t_d_to_t", description="", input_type=TypeName.DOCUMENTS, output_type=TypeName.TEXT)
def d_to_t(documents): return " ".join(d.get("x","") for d in documents)
@tool(name="t_t_loud", description="", input_type=TypeName.TEXT, output_type=TypeName.TEXT)
def t_loud(text): return text.upper()
@tool(name="t_t_to_s", description="", input_type=TypeName.TEXT, output_type=TypeName.SCORE)
def t_score(text): return min(1.0, len(text)/100.0)


def _reg():
    return {t.tool.name: t.tool for t in (q_to_d, d_to_t, t_loud, t_score)}


def test_beam_search_returns_best_pipeline():
    seed = Pipeline([PipelineStep("t_q_to_d"), PipelineStep("t_d_to_t")])
    fitness_calls = {"n": 0}
    def fit(p):
        fitness_calls["n"] += 1
        score = 0.1 * len(p)
        if p.steps and p.steps[-1].tool_name == "t_t_to_s":
            score += 0.5
        return score, {"detail":"ok"}

    result = beam_search(
        seed=seed, registry=_reg(), input_type=TypeName.QUERY,
        fitness_fn=fit, k=3, iterations=4, max_steps=4,
        rng=random.Random(0),
    )
    assert result.best_pipeline is not None
    assert result.best_score >= 0.2
    assert fitness_calls["n"] >= 1


def test_beam_search_terminates_on_threshold():
    seed = Pipeline([PipelineStep("t_q_to_d"), PipelineStep("t_d_to_t")])
    def fit(p): return (0.99, {})
    result = beam_search(
        seed=seed, registry=_reg(), input_type=TypeName.QUERY,
        fitness_fn=fit, k=3, iterations=10, max_steps=4, score_threshold=0.85,
        rng=random.Random(0),
    )
    assert result.iterations_run <= 2
