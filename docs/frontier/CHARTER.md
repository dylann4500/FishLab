# FRONTIER CHARTER — the successor programme to SESTINA v1.0

Standing document. Every working session on this programme reads this file first and treats it as
binding. Adapted from an external draft (GPT, 2026-09-01) and corrected against this repository;
where the draft and the repository disagreed, the repository won.

The goal: discover, implement, falsify, compare, and rigorously evaluate successors to SESTINA
v1.0 — from minimal modifications to independent architectures — and produce a scientific account
of what improved play, what did not, why, how strength scales with computation, and what
exploitability remains. This is a research task, not a leaderboard task. Negative results stay in
the record. Nothing is optimized on holdout material. No near-optimality, unexploitable, or
strongest-known claim is made unless the evidence genuinely warrants it.

---

## 1. Ground truth about this repository (verified 2026-09-01)

Do not re-derive these; do not contradict them without checking the cited source.

**The lineage and artifact.** SESTINA v1.0 = FishBot v0.7, frozen as a 55-coordinate vector in
`engine/fishbot_v07.json` (`v07:r12=25,rtie=1,...,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26`).
The paper is `paper/sestina_v10.tex` (sections in `paper/sections_v07/`), every number a macro
generated or transcribed under `paper/check_provenance.py` discipline. Release notes:
`docs/RELEASE-v1.0.md`.

**Headline results.** +3.33 pp [+2.88, +3.78] vs F-cheap over 48,000 sealed games, sign replicated
on both banks. **+0.15 pp [−0.29, +0.59] vs the phase-2 composite** — the architecture work after
the composite added no measurable strength. 3rd of 4 on minimax regret over the 31-member panel;
worst panel cell −0.04 pp [−1.41, +1.33], not sign-replicated. Partner-substitution worst row
−0.19, unresolved.

**The known engine defect (the paper's §12.6).** Agents are constructed once per worker thread and
reused across deals; some per-agent state survives the inter-deal reset, reachable only under
search. The carrier was identified as the rollout engine; **the specific field was never found**.
Incidence ≈ 1 deal in 800; 4 of 16 audited holdout cells show a nonzero seat-isolation residual,
all searching configurations. The zero-tolerance gate mode (one thread, agents rebuilt per deal) is
clean but is not the deployed mode. The paper itself directs: repair it and re-measure everything it
moves *before* the next configuration is frozen.

**The measurement instrument.** Scoreboard: ~9,604 games per cell to resolve 1 pp; the registered
detection floor for the v0.7 primary rule was 1.53 pp; the responder-side floor does not fall with
more games (search power, not sample size, is the constraint). **The per-decision estimator of
§9 (sec:development-perdecision) resolves a one-point-equivalent effect in 284–560 games on the
declaration channel** — a ~20–30× design effect. It is confirmed as an estimator and rejected as a
fitting objective (all three self-oriented per-decision fits produced worse policies; the ask and
declaration channels trade against each other). The paper names "use it as an estimator while
fitting against the scoreboard" as the obvious untried experiment. **This programme's Stage 2 runs
on it.**

**Search facts.** The deployed search is tiny: 12 determinizations, 4 candidates, LCB guard
κ=2.5, depth-12 rollouts, gated by entry thresholds. ~84% of the search's own selection signal is
winner's curse on its Monte Carlo draw (~95.5% inside tied sets). Tied asks are mostly genuinely
exchangeable; nothing learned survived the free public-history tie-break, and the correct control
for any tied-set mechanism is the *decorrelated* policy, not the deployed one. A 74×-better leaf
evaluator changed play by nothing because the deviation rule is nearly leaf-invariant. An
unrestricted learned re-ranker lost outright (−1.83 pp).

**The declaration-allocation channel is closed.** The sequential shortcut's first chain-rule factor
is the marginal; the exact allocator agrees with it at every recorded decision. Declaration
*timing* remains open.

**The adversarial record is a lower bound produced by small searches.** Eight arms, none positive —
but the white-box transcript inverter effectively searched 37 live coordinates at 8 generations of
population 12, **against a v0.6 model of the target rather than the target itself**. The threat
model's principal regime (ex-ante correlated adversary) was never built; the synchronised regime
substituted for it. The C6 scripted-adaptive class now has one member (`engine/src/v07_adapt.hpp`).
Adversary calibration against planted vulnerabilities is existing practice — keep it.

**External opponents exist now (post-paper).** `engine/src/kv.hpp` is a faithful C++ port of KV's
SearchAgent (particle belief + sampled-world search, github.com/kv1514/fish-researchp12), parity-
checked against the original (`scripts/kv_parity_ref.py`). `engine/src/kv6.hpp` plays KV's deployed
FishBot v0.6 over a JSON bridge, core untouched. `engine/src/botpkg.hpp` / `extbot.hpp` /
`botserve.hpp` host arbitrary external bot packages. The paper's "no independently authored
opponents" limitation is stale; the KV lineage is also, note, an independently authored
particle-belief architecture — the thing Track C/L would otherwise build from scratch.

**Infrastructure that already exists — reuse it, do not reinvent it.** Sealed banks with digests
computed by a second build (`bank`/`bankdigest` modes); per-row `argv` provenance on every recorded
result; registered protocols with advance-named non-confirmation conditions; a deviation register
(paper Appendix F) and corrections register (Appendix E); decision capture (`capture` mode, K5
digests); a brute-force enumeration oracle for the exact block DP (`engine/src/oracle.hpp`); the
seat-isolation/side-channel audit (`engine/src/v07_side.hpp`); dialect batteries (Appendix A);
switchable rollout belief (`rbelief=`); deterministic seeded streams (`engine/src/v07_seeds.hpp`).

---

## 2. Compute reality and budgeting

All experiments run on one 15-core Apple M5 Pro, 24 GB. SESTINA costs ≥4.52× the blueprint per
game. The v0.7 sealed battery (~180k games across its cells) ran about two days flat-out. Planning
units, to be re-benchmarked in Phase 0:

- one 1-pp-resolution scoreboard cell (~10k games) ≈ 2–3 machine-hours;
- a 0.5-pp cell costs 4× that; a 0.25-pp cell 16×;
- a per-decision Stage-2 cell (a few hundred games + capture) ≈ minutes.

Rules that follow:

1. **Powered scoreboard cells are the scarce resource.** Every one is planned in the registry
   before it runs, with a power calculation. Never buy a cell that cannot resolve the effect it is
   testing.
2. **Stage 2 (per-decision, captured-state) is the default instrument.** A mechanism reaches the
   scoreboard only after it has moved decisions in capture.
3. Batteries run as background/overnight jobs from a frozen binary and are verified against
   expected cell counts or process exit — never declared complete from the existence of an
   artifact file.
4. `COMPUTE-LEDGER.md` records machine-hours per experiment, kept honest enough to reprioritize.
5. Model-side parallelism does not accelerate the machine. Fan out reading and review, never
   engine-editing, and never spawn agents to wait on batteries.

---

## 3. Two baselines

- **SESTINA-v1.0-HISTORICAL** — the shipped artifact, bit-reproducible, never modified. The
  published numbers rest on it; the provenance chain (v07-named banks, digests, freeze artifact)
  is load-bearing and stays exactly as sealed.
- **SESTINA-v1.0-CORRECTED** — the same vector on the engine with the cross-deal residual
  repaired. The repair lives behind an explicit switch or clearly separated code path such that a
  default build still reproduces HISTORICAL play bit-for-bit; the measured engine core stays
  consistent with the published paper. CORRECTED is the scientific baseline for every successor
  comparison. If the repair changes strength, that is a result, and it is reported.

---

## 4. Governance

New namespace: `docs/frontier/`. Maintain: `BASELINE-AUDIT.md`, `HYPOTHESES.md`,
`EXPERIMENT-REGISTRY.md`, `NEGATIVE-RESULTS.md`, `COMPUTE-LEDGER.md`, `LITERATURE-MAP.md`,
`RESULTS/`, and at the end `FINAL-REPORT.md` + `RECOMMENDED-FRONTIER.md`. Sealing, digests, seeds,
and per-row argv provenance use the *existing* engine machinery.

Every experiment gets a registry entry **before it runs**: unique ID; hypothesis and expected
direction; code commit + config hash; seeds; sample size chosen in advance with the effect size it
can resolve; primary and secondary metrics; stopping rule; exploratory or confirmatory. Afterwards:
result, CI, per-bank split, compute cost, interpretation. Exploratory work may iterate;
confirmatory claims may not be reruns-until-favorable. Negative results go to
`NEGATIVE-RESULTS.md`, not to /dev/null.

**Sealed material.** Before any candidate development: construct new disjoint deal banks with the
existing bank machinery; commit seeds/digests unplayed; partition development / validation /
adversarial / final-holdout and record the rule. Existing played banks are development material
only. Final comparisons are frozen before final holdout is opened; a defect found after opening
means a new cycle with new sealed material, not a patch-and-continue.

---

## 5. Evaluation standard

Primary comparisons: paired duplicate deal blocks, both orientations of the identical deal, the
deal (not the rotation) as the clustering unit, deal-clustered bootstrap CIs, replication across
two disjoint banks required in advance — i.e., the v0.7 protocol, which is sound. Report per
candidate: edge + CI, per-bank edges and sign replication, worst panel cell, minimax regret,
cross-play and one-seat/three-seat substitution, declaration accuracy, ask hit rate, latency and
throughput, game length and cap behavior, soundness gate, seat-isolation status.

**Measure the consumer.** A belief that predicts cards better, a leaf that predicts returns
better, a loss that goes down — none of these is a stronger Fish team, and this repository has
already demonstrated each of those gaps empirically (74× leaf, per-decision fits, re-ranker).
Surrogates must be validated against game strength before being trusted, and abandoned if
movement in the surrogate fails to predict movement in strength.

**Three frontiers, kept separate.** (A) Matched-compute: stronger at SESTINA-CORRECTED's budget?
(B) Compute-constrained: best team at each of several budgets (log-spaced after profiling).
(C) Maximum-strength: how strong with latency relaxed. Maintain a strength-vs-decision-compute
Pareto set rather than one winner. A +3 pp candidate at 1000× cost is a (C) result only.

---

## 6. Research portfolio, in priority order

Each item is a hypothesis with a stage gate, not a mandate. Kill weak branches early. **You will
not do all of this; the ordering is the plan.**

### P1 — run these first

**T1. Residual repair + corrected baseline** (Phase 0; prerequisite to everything). Locate the
surviving per-agent state (start: rollout engine, worker-thread agent reuse — `arena.hpp`,
`v06_rollout.hpp`, and every `mutable`/lazily-initialized member reachable under search); minimal
reproduction; repair; mechanical proof of per-deal isolation (S6 machinery); quantify the play
delta vs HISTORICAL (expected order: 1 deal in 800).

**T2. External validity via KV** (cheap, headline value). Duplicate-bank batteries:
SESTINA-CORRECTED vs `kv` (ported SearchAgent) and vs `kv6` (bridge), home/away, plus mixed-team
cells where the protocol permits. First comparative claims beyond the lineage; also
characterize *where* KV's particle belief fails vs the exact/Sinkhorn machinery (feeds T5/T6).
The black-box-first caveat is moot for KV (already ported); preserve it for future external bots.

**T3. Compute–strength scaling (Track D).** One axis at a time from the CORRECTED baseline:
determinizations (12→24→48→…), candidates (4→8→…), rollout depth, entry thresholds, rollout
policy quality (blueprint vs cheap — `rbelief`/rollout switches exist), then joint sweeps of the
winners. Distinguish variance-limited from model-bias-limited regimes: added samples cannot remove
rollout-model bias, and the 84%-winner's-curse figure predicts where the curve bends. Stage 2
(decision flips on captured states) before any scoreboard cell. Deliverable: strength = f(compute)
with the plateau located and attributed (variance / rollout bias / inference error / pruning).

**T4. Exact behavior-weighted deal inference (Track A).** Replace block count tables
T_h(v) = |{α ∈ F_h : σ(α)=v}| with weighted tables T̃_h(v) = Σ_{α∈F_h} Π_{c∈B_h} w(c, α(c)) and
likewise the forced-card tables T̃_h^{c→p}(v); propagate through the same count-signature
convolution. Because C1–C5 constraints live inside block enumeration and per-seat capacities join
blocks only through count signatures, a per-card factorized prior preserves both the factorization
and **exact** C5 — removing the Sinkhorn approximation *and* its independence-conditioning of C5
in one move, at the same complexity (floats + log-space instead of counts). Controls: w≡1 must
reduce to the count DP to numerical precision; brute-force agreement on small states via
`oracle.hpp`; sampled frequencies must reproduce weighted marginals; candidate sets never exclude
the true owner. Evaluate as predictor (log loss, calibration, allocation accuracy) and then as
consumer (deployed belief and rollout belief) — the repository's record warns loudly that the
first does not imply the second.

### P2 — one or two of these, chosen on P1 evidence

**T5. Policy-aware / action-likelihood inference (Track B).** Beyond ask counts: weight deals by
P(observed action | hand) under the actor's policy. Teammates first (known deterministic policy —
strongest inversion, and coordination is where it pays); opponents via softened likelihoods
(ε-contaminated, Boltzmann, archetype mixtures — never assume they are SESTINA). Explicitly
measure the inference/exploitability trade-off jointly with T7: sharper inversion of teammates is
also a gift to an inverting opponent.

**T6. Adaptive search allocation (Track E).** Successive elimination / best-arm style allocation
over paired candidate−blueprint deltas on shared sampled worlds (keep the pairing). Compare
against fixed-budget at matched compute. Tied sets: the burden of proof is on any mechanism, given
the exchangeability findings; control = decorrelated policy.

**T7. Red team v2 (Track N; separate operator, standing).** Scale the white-box transcript
inverter far beyond 37 coordinates × 8 gens × pop 12 and point it at **the actual CORRECTED
candidate, not a v0.6 surrogate**. Attempt the ex-ante correlated regime the threat model names as
principal. Extend C6 (adaptive) beyond one member. Calibrate every family on planted
vulnerabilities before interpreting a null. Every statement carries adversary class, seats,
correlation regime, budget, calibration floor; a failed search is a statement about the search.

### P3 — only with P1/P2 evidence in hand

**T8. Compounding small effects (Track M).** The hypothesis both cycles declined to test. Use the
per-decision estimator to identify mechanisms with distinct causal channels; predict composites
before evaluating them; confirm a small number on fresh sealed material at the sample size a
0.5-pp effect actually needs (~4× a 1-pp cell). This is the designed use of the Stage-2
instrument, not a license for selecting the best of hundreds of noisy combinations.

**T9. Declaration timing (Track J).** Allocation is closed; timing is open. Race risk, opponent
declaration hazard, value of waiting; exact solutions in restricted endgames as oracle labels.

**T10. Team architecture (Track K).** Two tracks kept distinct: fair-comparison (homogeneous,
same information rules) and expanded (role/seat-conditioned, jointly optimized, legal public
correlation). Include partner-aware adaptation — policy conditioned on whether teammates are
lineage bots, foreign bots (KV), or a human — which is a standing interest of this project.
Distinguish genuine coordination from brittle convention via one-seat substitution and cross-play;
note the paper's caveat that common-ancestor conventions are invisible to the existing cross-play.

**T11. Restricted endgame oracle (Track G, narrowed).** An exact belief-state solver for small
endgames (unresolved cards ≤ k), valuable primarily as oracle labels for search-decision quality
(T3/T6) even if never deployable. Full ISMCTS/POMCP ports are out of scope unless T3 shows
determinization is the binding limitation, and any port must state which of its theoretical
assumptions Fish violates (strategy fusion, non-locality).

**Deferred by default.** From-scratch RL/neural bots (Track L): at most one time-boxed spike, and
only if a concrete training budget exists; KV already supplies the independent-architecture
comparison. Alternative belief architectures (Track C) beyond KV-as-baseline and T4/T5. Nonlinear
ask policies (Track H): revisit only through the T8 pipeline with a validated decision-level
surrogate, given that three per-decision objectives and one learned re-ranker have already lost.

---

## 7. Stage gates

- **Stage 0 — audit.** T1. No architecture claims.
- **Stage 1 — theory + cheap prototypes.** Math soundness, complexity, unit tests, `oracle.hpp`
  exhaustive checks on tiny states, microbenchmarks.
- **Stage 2 — decision-level evaluation on captured states.** Belief quality, action flips,
  declaration changes, per-decision channels, cost. Kills mechanisms that cannot change decisions.
- **Stage 3 — small exploratory batteries** on development banks.
- **Stage 4 — powered evaluation** on disjoint validation banks.
- **Stage 5 — selection + freeze.** Small candidate set; code, config, panel, protocol, sample
  sizes, and non-confirmation criteria frozen and registered.
- **Stage 6 — sealed final evaluation.** Untouched holdout; no development after opening; defects
  found after opening start a new cycle with new material.

**Frontier acceptance:** soundness and seat-isolation gates clean; termination behavior
acceptable; positive edge vs SESTINA-CORRECTED on both final banks with pooled CI supporting the
claim; no worst-case regression beyond a pre-declared tolerance; cost reported; matched-compute
and unconstrained results kept separate.

---

## 8. Working protocol

Roles (a session per role; artifacts, not chat, carry state):

- **Lead** — owns this charter, the registry, sequencing, and merges. One lead at a time.
- **Builder** — implements the current phase. **Single-writer rule: at most one session edits the
  engine at a time**; worktrees only for genuinely concurrent code experiments. Batteries run in
  the background from frozen binaries while the builder moves on.
- **Red team** — separate session and branch; white-box read access to frozen candidates; never
  edits them; reports through its own artifacts.
- **Reviewer/statistician** — a *fresh* session invoked before any battery is promoted to
  confirmatory and before any freeze; checks pairing, clustering unit, power, selection effects,
  leakage, and registry completeness; has veto; signs the registry entry. Fresh context is the
  point.
- **Literature** — read-only sessions filling `LITERATURE-MAP.md`; every imported idea answers:
  designed for what problem, requires what assumptions, does Fish satisfy them, what would falsify
  it here. Citations tie to design decisions or are cut.

Session hygiene: begin by reading this charter + the phase prompt + the registry; end by updating
the registry and the phase log so the next session needs no chat context. Independent replication
(from spec, not from code) is reserved for the one or two results that will carry a paper claim.

---

## 9. Questions this programme answers, and deliverables

Primary: Q1 weighted-exact inference vs Sinkhorn (T4); Q2/Q3 action-likelihood information and
whether it buys win rate (T5); Q4 the inference/exploitability trade-off (T5+T7); Q5/Q6 the
compute–strength curve and what limits it (T3); Q7 adaptive vs fixed search (T6); Q11 whether
sub-floor effects compound (T8); Q13 lineage vs independent architecture (T2); Q14 strongest
adversarial response found (T7); Q15/Q16 the matched-compute and maximum-strength frontiers.

Final deliverables: the governance docs of §4 filled in; corrected baseline + candidate
implementations with tests; decision-capture datasets; machine-readable results; plots (strength
vs compute, inference quality vs compute, search error vs samples, panel robustness, ablations);
adversarial report; `FINAL-REPORT.md`; `RECOMMENDED-FRONTIER.md` separating best matched-compute /
moderate / high-compute / strongest-regardless, with confidence and limitations. Choose the paper
narrative **after** results exist; a strong paper centers one finding, and a well-calibrated null
beats a noisy win.

## 10. The attitude

Assume a skeptical professor audits the repository. For every positive result ask: deal variance,
selection, leakage, side channel, shared stream, opponent overfitting, self-play convention, proxy
misalignment, wrong bootstrap unit, weak adversary, insufficient power, implementation artifact,
winner's curse, holdout contamination? For every negative: is the mechanism absent, or is the
instrument incapable? Do not protect SESTINA, the new architectures, or elegant hypotheses. Try to
kill them. What survives is the frontier.
