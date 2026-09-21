# Phase prompts — how to run the frontier programme session by session

The charter (`docs/frontier/CHARTER.md`) is the constitution; these are the copy-paste prompts
that drive it. One phase = one working session (resumable across days). Don't paste the charter
into chats — every prompt below tells the session to read it from disk, which keeps one canonical
copy and saves context.

Ordering and rough machine cost (LLM cost is minor by comparison; the machine is the rate
limiter):

| # | phase | sessions | machine time |
|---|-------|----------|--------------|
| 0 | Audit + residual repair + CORRECTED baseline | 1 builder | hours (audit cells) |
| 0.5 | Sealed banks + governance scaffolding | same session | minutes |
| 1a | Decision-capture + per-decision Stage-2 harness | 1 builder | hours |
| 1b | KV external-validity battery | 1 builder | ~1 overnight |
| 2a | Compute-scaling study (T3) | 1 builder | 2–4 overnights |
| 2b | Weighted exact DP (T4) | 1 builder (worktree ok during 2a batteries) | hours + shared battery slots |
| 3 | ONE of T5/T6 (chosen on phase-2 evidence) + red team v2 in parallel | 1 builder + 1 red-team session | 2–4 overnights |
| 4 | Composite, freeze, sealed final eval, report | 1 builder + reviewer sign-off | ~1 battery weekend |

Standing, any time: literature session (read-only), reviewer session (before every confirmatory
battery and before the freeze).

Rules that apply to every session are in the charter §8. The two worth repeating in every prompt:
single-writer on the engine, and batteries verified by expected cell counts or process exit —
never by an artifact file existing.

---

## Phase 0 — baseline audit and residual repair

> Read docs/frontier/CHARTER.md fully, then docs/RELEASE-v1.0.md and
> paper/sections_v07/12-discussion.tex (§Engineering limitations, §Future work). You are the
> builder for Phase 0 of the frontier programme.
>
> Task 1 — BASELINE-AUDIT.md. Map the deployed decision pipeline to code: exact inference
> (engine/src/blockdp.hpp), deployed Sinkhorn belief (belief.hpp), ask/declaration policy and
> tie-break (v05.hpp/v06.hpp), search + LCB guard + rollout (v06.hpp, v06_rollout.hpp), stall rule
> (v07_stall.hpp), seat-isolation audit (v07_side.hpp), harness (arena.hpp, main.cpp modes).
> Record for each: code location, runtime cost, invariants, tests covering it, and any
> paper-vs-source mismatch. Note without executing them: profiling hooks and bench modes for the
> Phase 2 scaling study.
>
> Task 2 — the cross-deal residual. The paper records it (1 deal in ~800, carrier = rollout
> engine, agents constructed once per worker thread and reused across deals, field never
> identified). Find the field: audit every member of the searching agent and rollout engine that
> survives the inter-deal reset; build a minimal two-deal reproduction where deal B's play differs
> depending on deal A. Repair it behind an explicit switch so a default build still reproduces
> HISTORICAL bit-for-bit (verify: the release-notes match command and the 34-field agreement).
> Rerun the seat-isolation audit in deployed mode with the switch on — the residual must go to
> zero across all 16 cell types. Then measure HISTORICAL vs CORRECTED play delta (incidence of
> changed deals, and a development-bank duplicate battery big enough to bound the strength change
> at ~1 pp). Register everything in EXPERIMENT-REGISTRY.md; the strength delta is a result.
>
> Do not touch published artifacts, banks, or numbers. Stop after updating BASELINE-AUDIT.md,
> the registry, and COMPUTE-LEDGER.md; end with the repair verified and the delta measured.

## Phase 0.5 — sealed material (same session, after 0 passes)

> Using the engine's existing bank/digest machinery, construct new disjoint deal banks:
> development (2), validation (2), adversarial (1), final holdout (2). Commit seeds and digests
> unplayed; record the partition rule and the digest verification in the registry. Final-holdout
> banks are not to be played until the Phase 4 freeze is registered.

## Phase 1a — Stage-2 instrument

> Read the charter, then paper/sections_v07/09-development.tex (the per-decision estimator:
> 1-pp-equivalent effects resolve in 284–560 games on the declaration channel vs 9,604 on the
> scoreboard; confirmed as estimator, rejected as fitting objective). Build the standard Stage-2
> harness this programme will use: decision capture on development banks (existing capture mode),
> plus a driver that, for any candidate mechanism, reports decision flips vs CORRECTED, the
> per-decision channel deltas with deal-clustered CIs, and cost per decision. Validate it by
> reproducing a known result cheaply (e.g., the contestation weight's effect direction). Document
> usage in BASELINE-AUDIT.md; this becomes the gate every mechanism passes before scoreboard time.

## Phase 1b — KV battery

> Read the charter §T2, engine/src/kv.hpp and kv6.hpp headers, docs/BOT_PACKAGE.md. Run the
> first external-validity battery: SESTINA-CORRECTED vs kv (ported SearchAgent) and vs kv6
> (bridge) on development banks — paired duplicate blocks, home/away, deal-clustered CIs,
> registered in advance with power stated. Add mixed-team cells (one KV teammate) if the bridge
> permits. Also capture decisions from KV games and report where its particle belief diverges
> from the exact posterior (feeds T4/T5). Verify battery completion against expected cell counts.
> Write results to RESULTS/ and the registry; negative or awkward results go in as-is.

## Phase 2a — compute-scaling study (T3)

> Read the charter §T3 and the search internals mapped in BASELINE-AUDIT.md. Registered sweep,
> one axis at a time from CORRECTED (det 12→24→48→96; cand 4→8→16; depth; entry thresholds;
> rollout policy quality via existing switches). Stage 2 first on captured states: which axes
> change decisions at all, and in what direction. Then exploratory Stage-3 batteries on
> development banks for the axes that move decisions; then joint sweeps of winners; powered
> Stage-4 cells on validation banks only for the surviving configurations, overnight, budgeted in
> the ledger. Deliverable: strength = f(compute) at matched wall-clock accounting, the plateau
> located, and the binding constraint attributed (Monte Carlo variance vs rollout-model bias —
> use paired worlds and the winner's-curse instrumentation to separate them). Update HYPOTHESES.md
> with what this implies for T5 vs T6.

## Phase 2b — weighted exact DP (T4)

> Read the charter §T4, paper/sections_v07/05-inference.tex, engine/src/blockdp.hpp, belief.hpp,
> oracle.hpp. Implement the weighted block tables and forced-card tables exactly as specified in
> the charter, in log-space, additive to blockdp (no change to the count path). Correctness
> before any evaluation: w≡1 reduction to the count DP at numerical precision; oracle.hpp
> brute-force agreement on small states; sampler agreement with weighted marginals; true owner
> never excluded. Then Stage 2: predictive metrics (log loss, calibration, allocation accuracy)
> vs the deployed Sinkhorn belief and vs the uniform exact DP, on captured states; runtime per
> rebuild vs the Sinkhorn path at all six seats. Only if it wins as a predictor at acceptable
> cost: wire as deployed/rollout belief and ride the next scheduled battery slots. Register every
> step; a null here is a publishable answer to Q1.

## Phase 3 — chosen track + red team v2

Builder prompt (T5 or T6 — decide from Phase 2 evidence and record the decision in HYPOTHESES.md):

> Read the charter (§T5 or §T6) and the phase-2 results in RESULTS/. [T5: implement teammate-side
> action-likelihood weighting first — the teammate policy is known and deterministic — with
> softened likelihoods for opponents; evaluate through the Stage-2 harness, then batteries; team
> up with the red-team session to measure the exploitability cost of sharper inversion. |
> T6: implement successive-elimination allocation over paired candidate−blueprint deltas on
> shared worlds; matched-compute comparison vs fixed budget; tied-set changes are presumed
> worthless until they beat the decorrelated-policy control.]

Red-team prompt (separate chat, own branch/worktree; runs concurrently):

> Read docs/frontier/CHARTER.md §T7, paper/sections_v07/08-threat-model.tex, D-panel.tex, and
> engine/src/v07_invert.hpp, v07_adapt.hpp, adversaries_v07.sh. You are the red team: your goal
> is to defeat SESTINA-CORRECTED and every frontier candidate, not to improve them. White-box
> read access to frozen artifacts; you never edit candidates or the shared engine core (additive
> adversary code only). Priorities: (1) scale the transcript inverter well beyond 37 coordinates
> × 8 generations × pop 12 and target the actual CORRECTED agent, not a v0.6 surrogate; (2)
> attempt the ex-ante correlated regime; (3) extend the C6 adaptive class. Calibrate every
> adversary family on planted vulnerabilities before interpreting any null. Report per charter:
> class, seats, regime, budget, calibration floor; a failed search is reported as a failed
> search. Write to docs/frontier/RESULTS/redteam/ and the registry.

## Phase 4 — freeze and sealed evaluation

> Read the charter §7 and the full registry. Assemble the candidate set (small; include the
> phase-2 composite of any compounding mechanisms per §T8 if evidence supports it). Freeze code,
> configs, panel, protocol, sample sizes, and non-confirmation criteria; register the freeze.
> Then request reviewer sign-off (below) BEFORE opening final holdout. Run the sealed battery
> overnight/weekend from the frozen binary; verify completion by cell counts; generate results
> through the existing table pipeline; write FINAL-REPORT.md and RECOMMENDED-FRONTIER.md
> separating the frontiers and stating limitations. If a defect surfaces after opening: stop,
> record it, and plan a new cycle with new sealed material.

## Standing — reviewer (fresh session, every confirmatory battery + the freeze)

> Read docs/frontier/CHARTER.md §§4–5 and the registry entry/entries named below. You are the
> methods reviewer with veto power; you have no stake in any result. Check: pairing and
> orientation; clustering unit = deal; power adequate for the claimed effect; seeds and material
> disjoint from anything the candidate saw; no leakage from development into validation/holdout;
> stopping rule as registered; negative-results completeness (nothing quietly rerun). Verify one
> cell's argv reproduces its recorded row. Write PASS/FAIL with reasons into the registry entry;
> FAIL blocks the battery or freeze.

## Standing — literature (read-only session or web-enabled agent)

> Read docs/frontier/CHARTER.md, then build docs/frontier/LITERATURE-MAP.md for: imperfect-info
> card-game search (Skat, Bridge, Skat-style inference), ISMCTS/POMCP/public-belief methods and
> their failure modes (strategy fusion, non-locality, alpha-mu, SPARTA), policy-aware/inverse
> planning and Bayesian action decoding, weighted model counting and IPF/Sinkhorn theory,
> best-arm identification and OCBA for search allocation, winner's/optimizer's curse corrections,
> team-maxmin and correlated team equilibria, exploitability estimation and local best response.
> For every entry: what problem it was designed for, its assumptions, whether six-player Fish
> satisfies them, what would falsify its usefulness here, and which charter track consumes it.
> No decorative citations. Do not edit anything but LITERATURE-MAP.md.
