# Stem Agent

A pluripotent LLM agent that *differentiates* into a domain-specialized agent by composing, mutating, and accreting typed tools. Successful pipelines persist across sessions in a versioned skill library, so the system improves between runs as well as within them.

The framing is biological: stem cells read environmental signals, then commit to a specific cell fate. The agent here does the same — at L0 it has only minimal primitives; given a domain signal (e.g., "Legal" or "Economics") it bootstraps a domain prior from the open web, then evolves typed tool pipelines whose graduates are stored as named **composite tools** for future sessions to inherit.

## Highlights

- **Typed pipelines.** Every primitive declares an input/output type; the validator enforces compatibility (with a small set of safe coercions). Pipelines that don't type-validate cannot run.
- **Step-wise evaluation.** Each step in a pipeline gets a quality / consistency / domain-relevance score; cumulative recurrence rewards consistent improvement; pipelines whose intermediate score drops below a threshold are early-terminated.
- **Six mutation operators.** `add`, `remove`, `replace`, `reorder`, `inject_domain`, **`parametric_mutate`**. Parametric mutation (changing tool parameters without changing pipeline structure) was the unsung hero in our experiments.
- **Persistent skill library.** Promoted composites live in `tool_library/composites.json` with full lineage parents, MAP-Elites archive cells keyed on `(domain, capability_tag)` to enforce diversity, and an auto-rendered Graphviz lineage diagram.
- **Frozen universal tools.** A small set (LaTeX builder, grammar check, PDF compile) is reserved for final report rendering and is *excluded* from evolution candidate sets — the agent cannot mutate them.
- **Graceful judge fallback.** External judge defaults to Anthropic if `ANTHROPIC_API_KEY` is set, then OpenAI, then a local Gemma fallback (banner in report when no key).

## Quickstart

### Prerequisites

- **Python 3.12 or newer** (`pyproject.toml` enforces `requires-python = ">=3.12"`; the install will fail on 3.11 or earlier with a clear error). On Windows we recommend Windows Terminal + PowerShell 7 for proper Rich console rendering.
- [LM Studio](https://lmstudio.ai/) with **Gemma 4 E4B** (Q4_K_M GGUF) loaded; server running on `http://localhost:1234`.
- A LaTeX install (MiKTeX on Windows, TeX Live on Linux/macOS) for `pdf_compile`.
- **Optional helpers** (system runs without them):
  - Java runtime — enables the offline LanguageTool grammar checker (otherwise the public API is used).
  - Graphviz `dot` — turns `tool_library/lineage.dot` into a PNG.
  - `pip install -e ".[dashboard]"` — adds Streamlit for the optional read-only dashboard.
- **Optional API keys** (any subset; the system has key-free fallbacks for every retrieval and judge path):
  - `TAVILY_API_KEY` — Tavily web search; without it `web_search` falls back to DDG (rate-limited). Wikipedia / Semantic Scholar / OpenAlex / arXiv all work *without* a key.
  - `FRED_API_KEY` — FRED macroeconomic time series.
  - `COURTLISTENER_TOKEN` — US case-law search.
  - `EDGAR_USER_AGENT` — SEC requires a "Name email" string for filings; default is `Stem Agent contact@example.com` which SEC accepts. Set to your real contact for high-volume use.
  - `ANTHROPIC_API_KEY` — promotes Claude Haiku 4.5 to the final-eval pairwise judge (used for credibility numbers in the write-up).
  - `OPENAI_API_KEY` — fallback judge if no Anthropic key.

### Install

```bash
git clone https://github.com/lukastojiljkovic/stem-agent.git
cd stem-agent
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dashboard,dev]"
python scripts/fetch_fixtures.py
```

### Run

```bash
# Deep track: Legal -> Contract Analysis (CUAD micro-subset)
python -m stem_agent run --domain legal --subdomain contract_analysis --track deep

# Shallow transfer track: Economics -> Financial Reporting
python -m stem_agent run --domain economics --track shallow

# Just baselines (sanity check)
python -m stem_agent baseline --domain legal

# Show the persisted library
python -m stem_agent library --lineage
```

### Cross-session demo (3 back-to-back sessions)

```bash
python scripts/cross_session.py
```

### Optional Streamlit dashboard

```bash
streamlit run src/stem_agent/ui/dashboard/app.py
```

## What the agent actually does at runtime

1. **Phase 0 — Library load.** Read all primitives + persisted composites. Load MAP-Elites archive.
2. **Phase 1 — Bootstrap.** L0 issues 3-5 web/wiki/arxiv queries about the domain, summarises the responses into `domain_brief.md`, and embeds the brief into the L1 system prompt.
3. **Phase 1 — Evolve.** Beam search (k=4, max 5 steps, max 8 iters) over typed pipelines. Each iteration applies all six mutation operators to every beam member; per-step LLM rubric + consistency heuristic + domain-relevance form the fitness; deterministic checks (CUAD F1, ratio tolerance) anchor it where available. Promotion gate uses MAP-Elites + improvement-over-parent + regression-suite check.
4. **Phase 2 — L2 evolution (deep track only).** Same machinery, biased toward subdomain-tagged tools, with prior-session composites visible as candidates.
5. **Phase 3 — Frozen evaluation.** Baseline / L1 / L2 are all evaluated on the same held-out task split.
6. **Phase 4 — Report.** Universal frozen tools (latex_init / section / table / chart / grammar_check / pdf_compile) compose the agent's final answer PDF.
7. **Phase 5 — Snapshot.** Library snapshot, lineage diagram, and metrics.json are written to `runs/<session-id>/`.

All phases emit Rich-styled console output and a JSONL event log; the optional Streamlit dashboard reads that log.

## Repository layout

```text
src/stem_agent/         # the package
data/                   # rule packs + eval fixtures
tool_library/           # persisted composites (committed!)
runs/                   # per-session artifacts (gitignored)
report/                 # OUR 4-page write-up + report.pdf
scripts/                # fetch_fixtures + cross_session
tests/                  # unit tests
```

## Evaluation

- **Legal (deep).** CUAD micro-subset (5 contracts × 4 clause categories) + LegalBench micro-slice (3 tasks × 30 examples).
- **Economics (shallow).** EDGAR ratio extraction (Apple/Microsoft FY2024) + FinanceBench-style QA.
- **Three-tier comparison.** Hand-coded baseline (per spec §12) vs L1 vs L2.
- **Cross-session.** Three sequential runs; library carries forward; F1 trajectory plotted.

See `report/report.pdf` for numbers, framing, and what failed.

## Acknowledgements & related work

This is a synthesis. The methodological lineage:

- **Voyager** (Wang et al. 2023) — lifelong skill library + embedding retrieval over docstrings.
- **AFlow** (Zhang et al. ICLR 2025) — typed pipeline MCTS.
- **AlphaEvolve / FunSearch** (DeepMind 2023, 2025) — LLM-guided program evolution.
- **Darwin Gödel Machine** (Zhang et al. 2025) — open-ended self-modifying agents.
- **Reflexion** (Shinn et al. NeurIPS 2023), **DSPy** (Khattab et al. ICLR 2024), **TextGrad** (Yuksekgonul et al. Nature 2024).
- **Quality-Diversity through AI Feedback** (Bradley et al. ICLR 2024) — MAP-Elites with LLM eval.
- **AgentPRM** (2025) — process reward models for agents.
- Developmental framing: **Growing Neural Cellular Automata** (Mordvintsev et al. Distill 2020); *Engineering morphogenesis with differentiable programming* (Nat. Comp. Sci. 2025).

The "stem agent" framing of LLM-agent specialization as developmental differentiation appears underused in the literature; we hope it's useful.

## License

Apache 2.0. See `LICENSE`. The CUAD subset is CC-BY 4.0 (Atticus Project).

## Engineering process

This project was written in tight collaboration with Claude (Opus 4.7) via Claude Code: human engineer doing the design and verification, AI doing the heavy editing and mechanical synthesis. See `JOURNAL.md` for the running diary — including the bumps.
