"""Centralized configuration: paths, environment variables, model defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent.parent


@dataclass(frozen=True)
class Paths:
    root: Path = field(default_factory=_project_root)

    @property
    def data(self) -> Path: return self.root / "data"
    @property
    def fixtures(self) -> Path: return self.data / "fixtures"
    @property
    def rule_packs(self) -> Path: return self.data / "rule_packs"
    @property
    def tool_library(self) -> Path: return self.root / "tool_library"
    @property
    def runs(self) -> Path: return self.root / "runs"
    @property
    def report(self) -> Path: return self.root / "report"
    @property
    def docs(self) -> Path: return self.root / "docs"


@dataclass(frozen=True)
class LMStudioConfig:
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    model: str = "gemma-4-e4b-it"
    temperature_default: float = 1.0
    top_p_default: float = 0.95
    top_k_default: int = 64
    temperature_structured: float = 0.3
    top_p_structured: float = 0.9
    max_tokens_default: int = 2048
    request_timeout_s: int = 180


@dataclass(frozen=True)
class JudgeConfig:
    anthropic_model: str = "claude-haiku-4-5"
    openai_model: str = "gpt-4o-mini"
    request_timeout_s: int = 60


@dataclass(frozen=True)
class EvolutionConfig:
    beam_k: int = 4
    max_steps: int = 5
    max_iterations: int = 8
    max_llm_calls_per_candidate: int = 30
    max_wall_time_s_per_candidate: int = 300
    cumulative_alpha: float = 0.7
    cumulative_beta: float = 0.3
    early_term_step_min: float = 0.25
    final_w_output: float = 0.5
    final_w_step_avg: float = 0.3
    final_w_consistency: float = 0.2
    complexity_lambda: float = 0.05
    rollback_after_n_declines: int = 2
    epsilon_termination: float = 0.01
    n_iters_for_epsilon: int = 3
    score_threshold_termination: float = 0.85
    promotion_min_improvement: float = 0.02


def has_anthropic_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def has_tavily_key() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY"))


def has_fred_key() -> bool:
    return bool(os.environ.get("FRED_API_KEY"))


def has_courtlistener_key() -> bool:
    return bool(os.environ.get("COURTLISTENER_TOKEN"))


PATHS = Paths()
LM = LMStudioConfig()
JUDGE = JudgeConfig()
EVO = EvolutionConfig()
