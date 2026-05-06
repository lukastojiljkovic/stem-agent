# Stem Agent — Journal

This is the engineering diary. It's intentionally informal — opinionated, occasionally frustrated, technically honest. The project was written in tight collaboration with Claude (Opus 4.7) via Claude Code. That's not vibe-coding: the engineer drives the design and verifies; the AI handles the mechanical lift (boilerplate edits, code generation against a precise spec, multi-file refactors). Treat it as engineer–AI symbiosis: a coherent unit doing more than either half could.

The point of this journal is to record the path, including dead ends. We will not censor.

## 2026-05-06 — Day 0: framing

Started from a one-page spec for a "stem agent" — a minimal pluripotent LLM agent that specializes itself into a domain. Spent the first chunk pulling on three threads:

- **Voyager / Generative Agents / Self-Refine.** The lifelong-skill-library pattern is the backbone we want; we won't reinvent it.
- **AlphaEvolve / FunSearch / DGM.** The evolutionary substrate. None of these are quite right for our scale (we're running a 4B local model, not Gemini Pro), but the *island archive* + *mutation* discipline ports.
- **AFlow / DSPy / TextGrad.** Pipeline compilation and "language gradients". Closest published recipe to what we want.

Then I stumbled on something I didn't expect: the developmental-biology framing — **stem cell differentiation** as a metaphor for agent specialization — is genuinely underused in the LLM-agent literature. The closest neighbors are NCA (Mordvintsev et al.) and the *Engineering Morphogenesis* paper (Nat. Comp. Sci. 2025). That's a real framing opportunity. Plan: anchor methodology on Voyager + AlphaEvolve + AFlow, but frame the *story* as morphogenetic differentiation. Don't oversell — anchor on the published lineage.

## 2026-05-06 — Day 0: deciding the cut

Original spec was both Economics and Legal, with several subdomains each. With a 4B model on a laptop, the realistic move is **deep × 1, shallow × 1**: one domain that goes all the way through L0 → L1 → L2 (Legal → Contract Analysis on CUAD), plus a smaller transfer demo (Economics → Financial Reporting via FinanceBench/EDGAR). The transfer demo proves "this isn't legal-specific". Pure breadth would have produced a broader but less convincing report.

## 2026-05-06 — Day 0: judge strategy

Self-judging by Gemma is the easy default but it's risky: a 4B model judging its own outputs Goodharts within a couple of generations (we expect to see verbose padding rewarded). Decision: **deterministic checks dominate** the fitness (CUAD F1, ratio tolerance) for the cases where we have ground truth, and the LLM rubric is a tiebreaker. For the *final* before/after table in the write-up, we use an external judge (Anthropic if a key is present, fallback to OpenAI, fallback to local Gemma + a banner that says so). The judge biases (position bias, self-preference) are real and we mitigate per Shi et al. 2024 / Wataoka et al. 2024 — pairwise both-orders averaged, judge ≠ generator where possible.

## 2026-05-06 — Day 0: small additions to the spec

A few things I added that weren't in the original one-pager:

- **Bootstrapping phase.** The metaphor demands it: stem-cell fate commitment is driven by environmental signals, not internal lineage alone. So before evolution we have the L0 agent issue 3-5 web/wiki/arxiv queries about the domain and summarise the answers into `domain_brief.md`, embedded into the L1 system prompt.
- **Parametric mutation.** Original spec had only structural mutations (add/remove/replace/reorder). On a 4B local model, structural changes are coarse — mostly making the pipeline worse before they make it better. Letting the search vary tool parameters (e.g., `taxonomy="CUAD41"` vs a 20-category subset) gives finer-grained moves.
- **MAP-Elites archive on (domain, capability) cells.** Without it, the search collapses on a single mega-pipeline. With it, we maintain structural diversity by construction.
- **Cross-session demonstration.** Three back-to-back sessions, library carrying forward, F1 trajectory plotted. Without this, the "improves between runs" claim is just an assertion.

## 2026-05-06 — Day 0: implementation grind

Bashed through the 12 phase-plans in a single sitting. TDD on the pipeline engine, mutations, beam-search, MAP-Elites archive. 51 unit tests all green at this checkpoint. Things that bit:

- `courtlistener-api-client` only ships at `0.0.x`, not `0.6.x` like I'd assumed. Killed it from deps; we hit CourtListener via raw `requests` instead. (We never needed the SDK to be honest.)
- HuggingFace `datasets` v4 dropped `trust_remote_code` — silently fails the CUAD pull. Fallback to two hand-coded synthetic contracts kicks in seamlessly. The seam means a fresh clone with no internet still demos.
- `pytest` was running under uv-Python 3.14 by default; had to `pip install pytest` into the same Python 3.12 where the package was editable-installed. Dual-Python on Windows is a perpetual annoyance.
- `RestrictedPython` 8.x deprecated `safe_globals` from `RestrictedPython` itself — moved the import to `RestrictedPython.Guards`. One-liner fix; would have been a cryptic ImportError otherwise.
- LM Studio's silent ~16K generation cap is real. Wrapper hard-caps at 8K to be safe. Documented in the spec; flagged in the journal so future-me doesn't chase it again.

## 2026-05-06 — Things to watch (operational)

- LM Studio's silent generation cap around 16K tokens. Wrapper caps at 8K to be safe.
- The 60-second default sync timeout in the official `lmstudio` SDK. We use the `openai` SDK directly with a 180s timeout.
- Tavily free credits burn fast on retries. Cache aggressively, dedupe queries, only escalate to `search_depth="advanced"` after a rerank failure.
- yfinance is unreliable in 2026; only use it as a fallback to `edgartools` for fundamentals.
- Scanned legal PDFs return empty text. Assert non-empty downstream of `pdf_extract`; fall back to `docling` (with OCR) only if needed.

That's the setup. From here, we run the experiments and write the report.
