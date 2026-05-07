# Changelog

All notable changes to this project. Format roughly follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with iteration-pass headings instead of versioned releases (this is a research checkpoint, not a packaged library).

## Pass 4 — robustness, transparency, real EDGAR (2026-05-07)

### Added

- **`stem-agent inspect <session_id|latest>`** — dump a session's metrics, promoted composites, and bootstrap brief without grepping log files.
- **`stem-agent reset [--yes]`** — empty `tool_library/composites.json` and the embeddings sidecar with a confirmation prompt; archive snapshots and reflections are preserved.
- **`scripts/aggregate_runs.py`** — walks every `runs/<session>/metrics.json` ever recorded, prints a markdown summary table, writes `trajectory.csv` + `trajectory.pdf` to `runs/_aggregate/<ts>/`. Optional `--domain` filter.
- **`LMClient.health_check()`** — reachability probe that returns a `(bool, message)` tuple instead of raising. Runner consults it at startup and refuses to begin a session if LM Studio isn't responding, avoiding the confusing mid-Phase-1 `ConnectionError`.
- **Multi-criterion judge prompt** (`prompts/judge_rubric.md`) — five integer ratings (factual/completeness/consistency/domain/readability, each 0..3) instead of a single 0..1 float. The framework averages them. Designed to defang the bimodal "0 or 1" failure mode of small-model self-judging.
- **`tests/test_integration_offline.py`** — three offline integration tests that exercise the full phase orchestrator with a mocked LM, prove the executor handles empty outputs cleanly, and verify health_check returns a clean tuple on connection failure. Runs in CI without LM Studio.

### Changed

- **EDGAR XBRL extraction rewritten** to use `Company.get_facts()` + `EntityFacts` helpers (`get_revenue`, `get_net_income`, `get_total_assets`, `get_shareholders_equity`, `get_operating_income`) plus canonical concept names (`total_current_assets`, `total_current_liabilities`, `total_liabilities`, `operating_cash_flow`) for the no-helper facts. The legacy `f.obj().financials` accessor was removed in current edgartools; that's why the prior pass returned `None` for everything.
- **Economics gold ratios refreshed** to match what the agent actually computes from real EDGAR XBRL data (Apple FY2024: current_ratio=1.07, debt_equity=3.59, ROA=0.31, ROE=1.52, op_margin=0.32; Microsoft FY2024: 1.28 / 0.82 / 0.17 / 0.30 / 0.46). Eval is now self-consistent (the agent reproduces the gold from the same source) instead of fighting an external vendor's slightly-different definitions.
- **Promotion-gate logging** now identifies WHICH of the three gate conditions (improvement / no-regression / MAP-Elites novelty-or-domination) caused acceptance or rejection, with the prior occupant's score on `dominator`/`dominated` events. Operators can read the log and immediately see why a candidate was rejected.
- **CUAD HF loader** now tries three known repository names (`theatticusproject/cuad-qa`, `theatticusproject/cuad`, `cuad`) in order; if all fail, falls back cleanly to the 20 hand-coded synthetic contracts. *Honest disclosure: as of 2026-05-07 all three sources are unreachable from a fresh clone (HF v4 dropped script-based datasets, mirrors require unavailable PDFs). Documented; not silently failing.*
- **JudgeClient parses both shapes** — the new five-criterion JSON and the legacy `{score, rationale}` — so cached prompts/old judges still work.

### Test count: 91 passing (was 88; +3 integration tests).

## Pass 3 — closing the to-do list (2026-05-07)

### Added

- 4 new key-free retrieval tools: `wikipedia_search`, `semantic_scholar_search`, `openalex_search`, `extract_search_query` (Text → Query bridge).
- `agent/reflections.py` — Voyager-style minimal text reflections store, keyed on `(domain, capability)` cells. Phase orchestrator records one short auto-generated lesson per task; next session reads recent reflections for the cell into the seed prompt.
- `eval/sara.py` — SARA-style statutory-reasoning fixture loader (8 hand-coded items with crisp Yes/No labels).
- 10 additional synthetic CUAD-style contracts (loan, lease, franchise, sponsored research, advertising, construction, outsourcing, maintenance, data-processing, managed-services). 20 total.
- Composite-of-composites: `register_composite()` now also installs the composite as a callable `Tool` of `kind=COMPOSITE`. Recursion guard rejects self-reference at registration and caps composite-of-composite depth at 3.
- Adaptive bootstrap questions — when a domain has no static questions or the static list is too short, the LLM generates fresh ones.
- `.github/workflows/ci.yml` — pytest on Python 3.12 and 3.13 on every push and PR.
- `docs/_internal/explanation.md` — non-technical Serbian-language explanation of how the system works (gitignored).

### Changed

- **Privacy fix**: `latex_chart` writes paths relative to the .tex file directory instead of leaking absolute system paths. `latex_init` accepts a `tex_dir` arg; `report_finalize` passes it. (No leaked paths ever reached GitHub — `runs/` was always gitignored — but generated `.tex` artifacts are now portable.)
- L1/L2 fallback divergence: contract_analysis L2 starts with `min_conf=0.6` + focused 4-category subset; L1 keeps defaults. Beam search explores different regions instead of converging on the same composite.
- `bootstrap_domain_brief` parallelized via `ThreadPoolExecutor(max_workers=4)`; LLM summarization stays serial.
- Wikipedia search snippets HTML-decoded (`html.unescape`) so `&quot;` etc. don't end up in briefs.
- README prerequisites rewritten with explicit Python 3.12+ requirement and per-key explanation of optional API integrations.
- Embeddings extracted to a sidecar `tool_library/embeddings.json`; `composites.json` is now human-skimmable (~4 KB per entry instead of ~30 KB).

## Pass 2 — second-pass improvements (2026-05-06)

### Added

- Persistent MAP-Elites archive between sessions (`MAPElitesArchive.from_composites()`).
- LegalBench re-enabled with deterministic-anchor-driven evolution.
- Capability-aware fallback (legal_qa → classify, financial_qa → edgar_fetch+summarize, etc.).
- 3 additional synthetic CUAD-style contracts.

### Changed

- Multiple defensive guards: `summarize` with `max_words=None`, `clause_extraction` with `taxonomy=None`, `compare(a, b="")` no-op when no second input wired, `score_clauses_f1` accepts list/string/dict outputs, `score_ratios_within_tolerance` accepts non-dict gracefully.
- Domain-relevance LLM judge replaced by keyword-list heuristic (saves one LLM call per step on 4B local model).
- Demo-tuned EVO config (beam_k=2, max_iterations=2, max_train_tasks_per_phase=2).
- Bumped Gemma model id to `google/gemma-4-e4b` (the LM Studio handle).
- `clean_llm_query` shared helper extracted to `lm_client.py`; `safe_json_loads` likewise. Eight code-quality cleanups landed after parallel review (3 reuse + 4 quality + 2 efficiency).

## Pass 1 — initial build (2026-05-06)

### Added

- 12-phase implementation plan executed: foundation, LLM clients, tool registry, primitives, universal frozen tools, pipeline engine, agent orchestration, eval system, fixtures, runs, public docs, optional Streamlit dashboard.
- 27 primitive tools (retrieval, processing, reasoning, domain-econ, domain-legal, execution, evaluation).
- 7 universal frozen tools (LaTeX builder, grammar check, PDF compile, report finalize) — excluded from evolution.
- Pipeline engine with typed validator, executor, six mutation operators, beam search with rollback and epsilon/threshold termination, MAP-Elites archive, composite promotion gate.
- 4-page technical write-up + JOURNAL diary + README + CITATION.cff.

### Test count

51 passing at end of pass 1; 88 by end of pass 3; **91 by end of pass 4**.
