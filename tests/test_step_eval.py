import pytest

from stem_agent.agent.step_eval import (
    cumulative_scores, final_pipeline_score, should_terminate_early
)


def test_cumulative_recurrence():
    scores = [0.5, 0.7, 0.9]
    cums = cumulative_scores(scores, alpha=0.7, beta=0.3)
    # c0 = 0.7*0 + 0.3*0.5 = 0.15
    assert abs(cums[0] - 0.15) < 1e-9
    # c1 = 0.7*0.15 + 0.3*0.7 = 0.105 + 0.21 = 0.315
    assert abs(cums[1] - 0.315) < 1e-9
    # c2 = 0.7*0.315 + 0.3*0.9 = 0.2205 + 0.27 = 0.4905
    assert abs(cums[2] - 0.4905) < 1e-9


def test_should_terminate_early_when_below_threshold():
    assert should_terminate_early([0.5, 0.2], threshold=0.25) is True


def test_should_not_terminate_when_all_above():
    assert should_terminate_early([0.5, 0.6], threshold=0.25) is False


def test_final_pipeline_score_combines_signals_with_complexity_penalty():
    s = final_pipeline_score(
        final_output_score=0.8,
        avg_step_score=0.7,
        consistency_across_steps=0.9,
        complexity=4.0, lam=0.05,
        w_output=0.5, w_step_avg=0.3, w_consistency=0.2,
    )
    # 0.5*0.8 + 0.3*0.7 + 0.2*0.9 - 0.05*4 = 0.4 + 0.21 + 0.18 - 0.20 = 0.59
    assert abs(s - 0.59) < 1e-9
