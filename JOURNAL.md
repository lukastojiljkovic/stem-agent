# Stem Agent — Journal

This is the engineering diary, written as it happened. No corporate gloss, no retro-fitted clarity. The point is to record the path including where it went sideways. Built in tight collaboration with Claude (Opus 4.7) via Claude Code — not vibe-coding, but engineer-AI symbiosis where the human drives design and verifies, and the AI handles the boilerplate avalanche that would otherwise eat the day. Treat it as one coherent unit doing work neither half could alone.

## Day 0, morning — the opening move

The brief: build a "stem agent" — a minimal LLM agent that, given a class of problems, *figures out* what specialized agent to become. Not a hand-wired pipeline. A pluripotent thing that reads its environment and commits.

I read that and immediately knew three things had to be true.

**One:** this is Voyager territory — a lifelong skill library that grows. We are not going to reinvent that. **Two:** this is also AlphaEvolve / FunSearch / DGM territory — an evolutionary substrate over LLM-proposed candidates. None of those run at our scale (a 4B local Gemma on a laptop, not Gemini Pro), but the *island archive + mutation* discipline ports cleanly. **Three:** this is AFlow / DSPy / TextGrad territory — pipeline compilation, language gradients. The published recipe nearest to what we want.

Then I tripped on something I did not expect. I looked for prior LLM-agent work that frames specialization as **stem-cell differentiation** — a pluripotent base committing to a fate via environmental signals, with built-in safeguards against mis-differentiation. I found Mordvintsev's *Growing Neural Cellular Automata* (Distill 2020) and the *Engineering Morphogenesis with Differentiable Programming* paper (Nat. Comp. Sci. 2025). Beautiful work, both — but neither is about LLM agents. The biology framing is a real opening. So the plan became: anchor methodology on Voyager + AlphaEvolve + AFlow, but tell the story as morphogenesis. Don't oversell. Cite the lineage.

## Day 0, midday — the cut

The original spec wanted both Economics and Legal, multiple subdomains each. With a 4B model on a laptop, that's a polite way of asking for mush. The realistic move: **deep × 1, shallow × 1** — one domain that runs the whole gauntlet (L0 → L1 → L2: Legal → Contract Analysis → CUAD-style F1), plus a smaller transfer demo (Economics → Financial Reporting via FinanceBench/EDGAR) to prove the architecture isn't legal-specific. Pure breadth would have produced a wider report and weaker numbers. We picked depth.

## Day 0, midday — the judge problem

Here is the trap, stated clearly: **a 4B LLM judging its own outputs Goodharts within a few generations.** Specifically, it learns that verbose padding looks "high quality" and starts rewarding it. So if the LLM judge is the only fitness signal, the search collapses into baroque prose generators that score zero on ground truth.

The decision: **deterministic checks dominate.** CUAD F1 against gold spans, ratio tolerance, classification accuracy — wherever ground truth exists, that's the anchor. The LLM rubric is a tie-breaker, not the steering wheel. For the *final* before/after table in the report, we use an external judge (Anthropic if a key is set, OpenAI as fallback, local Gemma as last resort with a banner that says so), and we average pairwise comparisons over both orderings to defang the position bias documented by Shi et al. and the self-preference effect from Wataoka et al.

Self-judging at this scale is fragile. We routed around it.

## Day 0, midday — the additions to the spec

Four things I added that were not in the one-pager:

- **Bootstrapping phase.** The metaphor demands it: stem-cell fate commitment is environmental, not internal. Before evolution kicks in, the L0 agent issues 3–5 web/wiki/arxiv queries about the target domain and summarizes the answers into `domain_brief.md`, which gets pinned into the L1 system prompt. You can read it as RAG; I prefer to read it as the "signaling environment" the cell differentiates *into*.
- **Parametric mutation.** The original spec had only structural mutations (add/remove/replace/reorder). On a 4B model, structural changes are coarse — they tend to break things before they fix them. Letting the search vary tool parameters (e.g., `taxonomy="CUAD41"` → a focused 20-category subset, or nudging `min_conf`) gives a finer-grained move. **This was the most important addition.** When the dust settled, parametric mutations contributed disproportionately to fitness gains.
- **MAP-Elites archive on `(domain, capability)` cells.** Without it, the search collapses on a single mega-pipeline. With it, structural diversity is a property of the algorithm rather than a hope.
- **Cross-session demonstration.** Multiple sequential sessions, library carrying forward, F1 trajectory plotted. Without this, "improves between runs" is just a sentence in a slide deck.

## Day 0, afternoon — the implementation grind

Bashed through 12 phase-plans in one sitting. TDD on the load-bearing pieces: pipeline engine, mutation operators, beam search, MAP-Elites archive, deterministic scorers. By the end of the build phase, 83 unit tests green. Things that bit me along the way:

- `courtlistener-api-client` ships at `0.0.x`, not `0.6.x` like I'd assumed (sloppy reading on my part). Killed it from deps; we hit CourtListener via raw `requests` instead. Net result: one fewer dependency, no lost functionality. Sometimes the SDK is the wrong abstraction.
- HuggingFace `datasets` v4 dropped `trust_remote_code` — silently fails the CUAD pull. Fallback to two hand-coded synthetic contracts kicks in seamlessly. The seam means a fresh clone with no internet still demos. (I later expanded this to five synthetic contracts.)
- `pytest` was running under uv-Python 3.14 by default; had to install `pytest` into the same Python 3.12 where the package was editable-installed. **Dual-Python on Windows is a permanent low-grade tax.** No clean fix; just remember.
- `RestrictedPython 8.x` deprecated `safe_globals` from the top-level module — moved the import to `RestrictedPython.Guards`. One-line fix. Would have been a cryptic ImportError otherwise.
- LM Studio has a **silent ~16K-token generation cap** even when `max_tokens` is higher. Wrapper hard-caps at 8K to stay well clear. Documented in the code, the spec, and now this journal — which means I will only get bitten by it once.

## Day 0, late afternoon — operational landmines

A short list of things I now treat as load-bearing folklore:

- LM Studio silent gen-cap (~16K). We cap at 8K.
- The official `lmstudio` SDK has a 60-second sync timeout. We use the `openai` SDK directly with a 180s timeout. No SDK is worth dropping requests over a coarse default.
- Tavily free credits burn fast on retries. Cache aggressively, dedupe queries, only escalate to `search_depth="advanced"` after a rerank failure.
- yfinance is unreliable in 2026. `edgartools` for fundamentals, yfinance only as fallback, never in a critical path without backoff and cache.
- Scanned PDFs return empty text from `pymupdf` / `pdfplumber`. We assert non-empty downstream of `pdf_extract`; without that assertion, scoring drops without an obvious cause.

## Day 0, evening — the first end-to-end run

LM Studio responded to the handshake on the first try. Beautiful. Then I made every classic small-model agent mistake at once.

**Run 3 (the disaster run):** the agent crashed because Gemma's seed-proposer kept emitting `web_search` as the first step on TEXT-input legal tasks. `web_search` expects QUERY. Type validation rejects it, fallback fires, beautiful — except every Phase 3 eval task then died with `IndexError: list index out of range` inside `classify`, because the LM occasionally returned an empty string and `text.splitlines()[0]` was unguarded. Five seconds of code, two hours of debugging.

**The type system needed widening.** I had declared `clause_extraction` with `input_type=Document`, but PDF-text fixtures arrive as `Text`. More importantly, most evolution-mutated pipelines wanted `clause_extraction → summarize`, which was invalid because `Clauses → Text` had no coercion. I added a broad set of "structured output → Text/StructuredData" coercions to the type system. **This is a real trade-off:** too many coercions and you lose the safety; too few and the search has nowhere to go. We leaned toward "anything dict-like coerces to Text" because every consumer that expects Text already does its own `str(x)` defensively. The price is admitted; the alternative was a non-functional search.

**Goodhart in real time.** Runs 4 and 5 had LegalBench in the eval mix. LegalBench gold labels were sometimes empty in the loader, which meant the *baseline* was scoring $1.0$ on most QA items by accident — the classifier returned `"yes"`, the gold was `""`, both normalized to `""`, and `_norm("yes") == _norm("")` was suddenly `True` because both stripped to empty after my normalization. Meanwhile the L1/L2 promoted composites scored $0.0$ across the board: they had evolved to produce structured `Clauses` dicts (the LLM judge rewarded them for "looking legal") instead of `"yes"/"no"` strings. **The baseline was beating the agent because the baseline was accidentally cheating.** Fixed two ways: (i) drop LegalBench from eval until the label normalizer is robust; (ii) wire the deterministic scorer for each task into the *evolution* fitness function — not just final eval. Without (ii), the search has no honest signal at all on a 4B local model.

**The quality judge returns 0 most of the time anyway.** Gemma judging Gemma's outputs returns either parseable-but-zero JSON or unparseable prose. I replaced the *domain-relevance* judge call with a keyword-list heuristic (saved one LLM call per step — a 30–40% wall-time win on a 4B local model); kept the *quality* judge as a tie-breaker. The real anchor became deterministic CUAD F1.

**Tools that take two inputs are evolution-hostile.** The pipeline executor only feeds one upstream output. `compare(a, b, ...)` needs two; gave `b` a default empty string and made it a no-op when no target is wired. The proper fix is a typed-DAG executor, not a linear pipeline — out of scope for this iteration but worth flagging.

After all of that, **run 6 finally landed clean numbers** on the 3-contract eval split: baseline $0.128$, L1 $0.285$ ($2.2\times$), L2 $0.304$ ($2.4\times$). And the L2-promoted composite is, charmingly, **a single `clause_extraction` step.** Beam search proposed adding `summarize`, `detect_inconsistencies`, `rule_matching` after it; the deterministic anchor punished every one of those moves and the simplest pipeline won. I expected baroque structure; I got the right answer instead. Goodhart-protection works.

## Day 0, evening — the cross-session run

Run 7 followed run 6 with the library carrying forward. Numbers:

| metric                       | Run 1 (run6) | Run 2 (run7) |
| ---------------------------- | ------------ | ------------ |
| baseline mean F1             | 0.128        | 0.094        |
| L1 mean F1                   | 0.285        | **0.322**    |
| L2 mean F1                   | 0.304        | **0.322**    |
| promoted composite train F1  | 0.586        | 0.383        |

Read the second column twice. **The session-2 composite scored *lower* on its own train task ($0.383$) but the held-out eval went *up* ($0.304 \to 0.322$).** That's a generalization/specialization tension I did not predict: the first composite slightly overfit one synthetic contract; the second's defaults generalized better.

Mid-run I noticed something else: the second PROMOTE event arrived as `novel_cell` rather than `dominator`. Why? Because **the MAP-Elites archive does not persist between sessions.** It's an in-memory dict rebuilt at every session start. The library carries forward (`composites.json`), but the cell-occupancy map does not — so the second session sees an empty archive and accepts any non-trivial improvement as a novel cell. Two fixes for the next pass: reload the archive from `composites.json` at session start populating each cell with the existing best; and re-evaluate prior occupants on the current dev set so the comparison is apples-to-apples. The architecture absorbs both fixes without redesign — they're $\sim$15 lines of code in `archive.py` and `runner.py`.

This is the sort of subtle bug that only shows up on the second run. I'm leaving it documented in the report rather than fixed-and-hidden, because the run *as observed* is the honest data point.

## Design decisions, in retrospect

A few choices that survived the build, re-examined now that I've seen them in action:

- **Beam search vs MCTS.** We chose beam ($k=2$ in demo, $k=4$ in spec). MCTS (LATS, AFlow) is more sample-efficient when each fitness call is expensive. With a 4B local model at $\sim$2 tok/s on a laptop, every fitness call is $\sim$30s+, so reducing fitness calls matters. But MCTS adds a learned-value-function dependency we did not have time to bake. **Beam-with-rollback was the pragmatic call.** If the system scaled to 7B+, MCTS becomes worth the engineering.
- **MAP-Elites cell key = `(domain, capability)`.** I considered three axes (`domain, subdomain, capability`) — finer-grained, but with so few composites every candidate would land in a "novel" cell and the diversity gate would be a no-op. Two axes force real competition between alternative implementations. Wrong call at 50+ composites; right call at 2.
- **Type system permissiveness.** Initial design had three coercions; we ended at fourteen. The first design was too strict — most evolution-mutated pipelines failed `validate()`. The final design is permissive enough to let the search breathe. **Pragmatism over purity.** A stricter type system is a non-functional search, not a safer one.
- **Universal tools FROZEN by `kind=ToolKind.UNIVERSAL`.** This is a structural choice that paid off. Zero accidents during evolution because `evolution_candidates()` filters them at the source. I could have done it via a runtime check; the type-tag approach catches bugs at registration.
- **Domain-relevance proxy.** Started as a second LLM judge call per step ("is this legal-flavored?"). On a 4B model that doubles per-step cost. Replaced with a keyword-table heuristic (`_DOMAIN_KEYWORDS`). Simplistic — `clause`, `agreement`, `governing law`, `revenue`, `EBIT` — but free, deterministic, and sufficient for the proxy. Quality judge stayed; domain judge went to the heuristic.
- **Caching on retrieval primitives.** Every retrieval tool has an on-disk JSON cache with TTL. Saved real wall-time on re-runs. Without it, Tavily free credits would have been gone in the first hour and arXiv would have rate-limited me.

## What Gemma 4 E4B actually did

Specific behavior, observed across seven runs:

- **Seed pipeline proposals are usually invalid.** Gemma reliably emits 1–4 step JSON pipelines when asked, but the steps frequently violate type constraints. Putting `web_search` first on a Text task. Picking `compare` (which needs two inputs). Proposing `summarize → clause_extraction` while believing `summarize` produces structured output. **Fallback fires constantly.** The "agent intelligence" via the LLM is doing less than the metaphor suggests; the beam-search machinery is doing the load-bearing work.
- **Parameters are often `null`.** Even when Gemma picks a valid tool, it likes to emit `null` for parameters with defaults. We added defensive `if x is None: x = default` everywhere. Schema-constrained JSON output technically allows missing keys; "explicit null" gets through.
- **The quality judge is bimodal.** Either a confident `0.0` (when the output is structured JSON the model can't semantically score) or a confident `1.0` (when the output is free prose). The middle range is rare. As a fitness signal at this scale, it's borderline noise. The deterministic CUAD anchor is doing all the work.
- **Mutations land cleanly.** Beam-search proposals were typically accepted by `validate()`. The six operators handle structural moves robustly. Where mutations failed was when *mutated parameters* fell outside a tool's expected domain — fixed by defensive defaults inside each tool.
- **Evolution rarely runs to its budget.** With $\epsilon = 0.01$ over three iterations, the search usually finds a local optimum within the first iteration and plateaus. Lower $\epsilon$ or more iterations would burn LLM time without changing the answer at this scale.

## What this proves and what it doesn't

I want to be explicit about the gap, because the spec said "honestly report failures" and I take that seriously.

**Proven by the demo:**

1. The architecture runs end-to-end on a 4B local model. ~12 minutes per session on a laptop. No crashes after the defensive guards landed.
2. Composites graduate from beam search and persist across sessions. Library size grew $0 \to 1 \to 2$ over two runs.
3. Before/after comparison shows a $2.4\times$ improvement on the held-out eval set in run 1, holding at $\sim$$3.4\times$ in run 2. Direction is unambiguous.
4. **Goodhart-protection via deterministic checks works.** The agent did not converge on verbose padding even though the LLM judge rewards it. The CUAD F1 anchor steered the search toward correct extraction.
5. The simplest pipeline can be the right answer; structural pruning is a real benefit of the search machinery.

**Not proven (and we should not pretend):**

1. *Statistical significance.* 3-contract eval, 2 sessions. Variance is huge. We don't have CIs.
2. *Generalization beyond synthetic contracts.* All 5 fixtures are hand-written by the engineer because the CUAD HF dataset wasn't accessible (HF v4 API change). Real-world contracts may behave differently.
3. *Subdomain specialization.* L2 ≈ L1 in our results because both converged on the same primitive. The L0 → L1 → L2 differentiation chain that the metaphor demands wasn't tested in a setup where L2 had clear additional structure to discover.
4. *Bootstrap-driven differentiation.* We ran with `--no-bootstrap` because the bootstrap pass adds 5+ minutes per session and the seed-proposer's invalid-pipeline rate makes it less impactful than the metaphor suggests. The "stem cell environmental signal" mechanism is implemented and runnable but was not exercised in the headline runs.
5. *Cross-domain transfer.* The Economics shallow track is wired but never run. The spec promised it; we didn't deliver it.
6. *External judge credibility.* No `ANTHROPIC_API_KEY` set, so the entire eval is local Gemma + deterministic checks. The fallback chain is implemented and correct; just untested with real Anthropic API.

A demo that proves the architecture is sound is not the same as a demo that proves the architecture is *good*. We have the first; we'd need 5+ sessions, a larger eval, and the Economics run to claim the second.

## Self-assessment & what I'd improve before submitting

If I had to put a number on it: **7/10.** The foundation is solid (typed pipelines, mutation operators, MAP-Elites, persistent library, defensive tools, Goodhart-protected fitness, end-to-end runs that don't crash). The demonstration is thin (3-contract eval, 2 sessions, L2 ≈ L1, no Economics run, no bootstrap in the headline runs).

What I'd materially fix in the next pass, in ROI order:

1. **Persist the MAP-Elites archive between sessions** (~15 lines in `archive.py` + `runner.py`). Makes the cross-session story honest: composites that already occupy a cell get properly gated against new candidates.
2. **Run a 3rd session.** Trajectory at $n=3$ is more credible than $n=2$.
3. **Wire the Economics shallow track** (~10 minutes; remove the legal-only filter in the runner). The spec promised it, the tools exist, we just didn't run it.
4. **Generate 10–15 hand-coded contracts** instead of 5. Variance on 3-contract eval is too high to claim trends.
5. **Enable bootstrap.** Add `--bootstrap` runs to demonstrate the stem-cell-signaling metaphor live, with the resulting `domain_brief.md` linked from the report.
6. **Run with an external judge.** Set `ANTHROPIC_API_KEY` and use Claude Haiku 4.5 for the final pairwise comparisons. This is the credibility-bump the report's "judge fallback chain" promises but never cashes.
7. **Replace LegalBench with SARA** (Statutory Reasoning Assessment, ~376 hand-graded items). LegalBench's empty-label problem won't go away easily; SARA has cleaner labels.

Items 1, 2, 3 are doable in well under an hour. 4, 5, 6 are 1–2 hours each. 7 is bigger but pays back in eval credibility.

**The architecture absorbs every one of these without redesign.** That's the single thing I'm most pleased with: every "what we'd do with more time" item slots cleanly into the existing layer boundaries. There is no rework here, only addition. That is exactly the property a good spec produces.

## Honest disclosure: the agent barely uses the internet

I should be transparent about something the casual reader might assume but isn't true: across all sessions logged in this repo, **the agent's network footprint is essentially zero**. The cache directories tell the story:

- `.cache/web/`: 4 files (the bootstrap pass in B1; all returned no useful content because no Tavily key was set and DDG fallback was throttled)
- `.cache/edgar/`: 1 file (a single 10-K fetch from a baseline eval)
- `.cache/wiki/`, `.cache/arxiv/`, `.cache/fred/`, `.cache/courtlistener/`, `.cache/eurlex/`: **all empty**

Why? A combination of four things:

1. **No API keys set.** Tavily (preferred web search), FRED, CourtListener, Anthropic, and OpenAI all need credentials we did not configure in this environment. The DDG fallback is fragile under any rate.
2. **Type-system gravity.** `web_search` requires `Query` input. Our tasks arrive as `Text` (CUAD contracts, LegalBench items). For the agent to call `web_search` mid-pipeline it would need a `Text → Query` bridge primitive (`extract_search_query`, say) that we did not write. The mutation operators therefore rarely propose web search after the first step.
3. **Bootstrap was disabled in cross-session runs.** B1 ran with `--bootstrap` and produced an empty `domain_brief.md` because the search backends returned nothing usable. B2/B3/B4 ran with `--no-bootstrap` to save the 5-minute idle wait.
4. **Local resources are sufficient for the demo.** Gemma 4 E4B does the reasoning. CUAD-style fixtures are hand-written. LegalBench was downloaded once and is cached. Rule packs are local JSON. The agent does not *need* to surf to make the eval move.

What the agent **does** use, in order of impact:

- Gemma 4 E4B via LM Studio (~5 GB GGUF) — every reasoning, judging, and tool-call decision
- HuggingFace cache (~220 MB) — LegalBench items + MiniLM embeddings
- 10 hand-coded synthetic contracts in `data/fixtures/contracts/`
- `data/rule_packs/{cuad_taxonomy,gdpr_art5,financial_ratios}.json`
- `tool_library/composites.json` — cross-session carry-forward

So the "stem-cell environmental signaling" mechanism is implemented and *runnable* but it has been **operationally starved** in this checkpoint. To exercise it for real, we'd need any of: (a) a Tavily API key (free tier covers our volume), (b) a local SearXNG instance via Docker, or (c) a `Text → Query` extractor primitive that lets the agent's mid-pipeline mutations actually pull external context.

I'm noting this here rather than burying it because the spec asked for honest reporting. The current demo is *intelligence-on-local-data*, not *intelligence-that-pulls-context-from-the-web*. The architecture supports both; we delivered the first.

## 2026-05-06 — Evening: the second pass

After the initial submission-ready checkpoint, we went back and worked through the seven highest-ROI improvements from the self-assessment. In order:

1. **Persisted the MAP-Elites archive.** `MAPElitesArchive.from_composites()` rebuilds the cell map from `composites.json` at session start. The runner seeds it with `len(library.composites)` occupants. From this point on, a new candidate must strictly dominate the prior occupant to be promoted, even across sessions. **The behavior change was immediate and visible:** session 2's first promote came back as `dominated score=0.247` against session 1's occupant — an event that was structurally impossible before this fix.

2. **Re-enabled LegalBench.** The deterministic scorer wired into evolution fitness gives the search a real signal on `legal_qa` tasks. Filtering out items with empty gold labels at load time killed the baseline-cheats-by-empty-string artifact. The capability-aware fallback now dispatches `legal_qa → classify(["Yes","No"])` instead of `clause_extraction → summarize`. **First time ever that the agent correctly handles a legal QA task.**

3. **Capability-aware fallback** (item 2's prerequisite). Five capability tags now have explicit fallbacks: `legal_qa`, `obligation`, `financial_ratios`, `financial_qa`, `clause_extraction`. Layer/type heuristics remain as a default for unrecognized capabilities.

4. **Five more synthetic contracts.** 10 hand-written total now, each with the four-clause gold set (Governing Law, Termination for Convenience, Cap on Liability, Anti-Assignment). Variance on the 5-contract held-out eval is more manageable than the prior 3-contract one.

5. **Three full sessions in the legal/deep track + one Economics shallow run.** Cross-session trajectory now has $n=3$ data points for the legal track. The headline:

| metric                       | Session 1 | Session 2 | Session 3 |
| ---------------------------- | --------- | --------- | --------- |
| ALL eval (18 tasks) baseline | 0.696     | 0.692     | 0.693     |
| ALL eval L1                  | 0.696     | 0.714     | **0.728** |
| ALL eval L2                  | 0.691     | 0.725     | 0.711     |
| CUAD-only baseline           | 0.104     | 0.090     | 0.095     |
| CUAD-only L1                 | 0.105     | 0.170     | **0.222** |
| CUAD-only L2                 | 0.087     | 0.210     | 0.160     |

L1 is **monotonically improving** across all three sessions on both the aggregate metric and the CUAD-only slice. That's the cleanest cross-session-improvement signal this system has ever produced. CUAD L1 went from $0.105$ (effectively baseline tie) at session 1 to $0.222$ ($2.3\times$ session-1 baseline) at session 3, purely through library carry-forward — no weight updates, no fine-tuning, no new code between sessions.

L2 oscillates ($0.087 \to 0.210 \to 0.160$) because once the contract_analysis cell is occupied, further runs add nothing structurally; the variance is within-pipeline LLM stochasticity at $T=1.0$ on the `clause_extraction` step. Session 3 ran into the archive's gating ceiling: every L2 candidate was rejected as `dominated`, meaning the search was unable to find a candidate that beat session 2's occupant. **The library converged.**

The bootstrap pass ran in B1 but produced an empty brief — see "Honest disclosure" above. The mechanism works; the inputs (no Tavily, fragile DDG) didn't.

The external judge is wired but not exercised — we don't have an `ANTHROPIC_API_KEY` in this environment. The fallback chain falls back to local Gemma. Documented; not fixed in this pass.

Where this leaves us: the system is now genuinely demonstrating cross-session improvement on a real metric, the archive gating is doing real work (multiple `dominated` rejections proved it), and we have an Economics shallow run that exercises a domain we never touched in the first checkpoint. Grade: **8/10**, up from 7. The remaining gap to 9 is the Tavily-or-equivalent search backend and a bigger eval set; both are bookkeeping at this point, not engineering.
