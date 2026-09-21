# FishBot v0.7 — ADVERSARIES: what beats the v0.6 frontier, and by what mechanism

Dylan Nguyen, FishLab Research Project
Repository `https://github.com/dylann4500/fishbot.git`, built on `f4581da`.
Phase 2 of the v0.7 programme (`docs/v07/PHASE-PROMPTS.md`). Read with `docs/v07/THREAT-MODEL.md`
(what "exploitable" means), `docs/v07/INSTRUMENT.md` (what the instrument can see) and
`docs/v07/SUBOPTIMALITY-LEDGER.md` (the hypotheses). The session's working record, including every
defect found in the phase-1 apparatus, is `docs/v07/RESEARCH-LOG.md` §2.

This document is a **taxonomy of adversaries by mechanism**, a **severity per cluster with
intervals**, and an accounting of **how much of the frontier's measured exploitability each cluster
carries**. It proposes no architecture and changes no policy. Every table is generated from the
artifacts by `engine/build_adversaries_v07.py` and spliced in by `engine/assemble_adversaries_v07.py`;
no number below is hand-typed.

---

## 0. Summary

> **The verdict, stated plainly, because the phase brief asks for it plainly.**
>
> **Nothing phase 2 built exploits the v0.6 frontier beyond the detection floor.** The best measured
> exploitability of the deployed policy is still the in-class figure — **+0.79 [+0.48, +1.10]** over
> 96,000 games, a correctly-specified C1 responder against the unhandicapped incumbent, confirmed by
> seven independently fitted searches whose best reaches +0.98 and **none of which clears its class
> floor** — and that sits *below* the class's re-measured detection floor of **1.53 points**. Every arm that beats `v06` by
> more than that turns out, on measurement, to be a **better policy rather than an exploit**: the
> target's own test-time search at +1.89, and a single out-of-class coordinate at
> +2.71 whose gain survives against opponents that share none of v0.6's machinery once
> the win-rate scale is corrected. Both are handed to phase 3 as candidates. Neither is exploitability.
>
> That is a result, and it reshapes phase 3 exactly as the brief anticipates: **the v0.7 case rests on
> beating the frontier, not on closing exploits.** And phase 2 does not hand that case over empty:
> one configuration, built from the frontier's own search plus one new linear coordinate plus a
> switch, **dominates every measured point of the frontier** — +4.40 over the deployed policy, +2.40
> over the cheap search, +2.21 over the strongest measured configuration, positive on every bank and
> commit-gate clean. Alongside it is a list of *defects in the incumbent*, measured, replicated and
> gate-checked, worth about two points between them.

| # | Finding | Where |
|---|---|---|
| **A1** | **One arm clears the bar against the deployed policy — and loses to the frontier.** A single hand-set coordinate of the extended responder class, with no fit at all, scores +2.71 [+2.27, +3.15] against the deployed policy over 48,000 games on two banks, dose-responsive and commit-gate clean. It is not the information-denial mechanism the coordinate was built to express, not reducible to either of the feature's factors, and not a mis-set hit weight; all three controls were run. But it beats **every** member of the panel, not just the incumbent, so it is a measured strength gain rather than exploitability under E1 — and against `F-mid` its edge is **−0.99**, negative on both banks. Its worst cell over the frontier is negative. | §4A, §5 |
| **A2** | **The strongest thing that beats the deployed policy is still the deployed policy's own search**, at +1.89 [+1.51, +2.26], and it is explicitly not an exploitability number — there is no KPI that separates it from being stronger. Phase 1's class ladder read C3's +1.86 as the strongest exploit; most of that was the search improving the policy, not the adversarial opponent model attacking it. | §4B |
| **A3** | **The channel phase 2 expected to be the headline has a measured ceiling of +0.38 points [-0.06, +0.81] and is dead** — but the *same* channel is worth **+1.23 to +1.62 points, replicated on three banks, to a v0.7 that switches it off**. The adversary owns the smaller half of a two-point defect. That is the clearest thing phase 2 hands to phase 3, and it is a configuration change rather than an architecture. | §3, §4C |
| **A4** | **v0.6 carries a fifteen-point cliff behind an event counter; no adversary reaches it, but a candidate can reach it in self-play.** At `pub.nEvents >= 220` its declaration rule drops its team-ownership floor from 0.849 to 0.25 and cashes half-suits at a coin flip; a target permanently in that regime loses +15.18 points. Against an unmodified `v06` the longest single game any adversary produced is **149 events**. But the escalation is also v0.5's termination guarantee, and a candidate that switches it off has a **self-play** tail of 405 events — three times the incumbent's. Phase 3 must report game length in mirror. | §3 |
| **A5** | **Phase 1's claim that the detection floor buys down with evaluation games is refuted at its own predicted value.** Phase 1 predicted ≈0.9 points at 4× the games; at exactly 4× the games the floor is **1.53** (against phase 1's 1.68, the same rung), and the +0.88 rung is still undetected — not because the interval is wide but because the excess resolves *to* zero, **−0.01 [−0.45, +0.42]**. Below ~1.5 points the responder is empty, not unresolved. Phase 2's claims are read against 1.53, and against **2.13** on the declaration family. | §2 |
| **A6** | **The belief neighbourhood is exhausted; the deception family is not.** Every unfitted belief and policy-prior deviation loses, and every belief mode other than the shipped one costs 7.9 to 34.0 points. The deception family is closed too — its best arm is −1.21 — but **not for the reason the corpus records**: the `tol` sweep, a parameter no committed artifact has ever set, spans 6.75 points for `feint` alone and the published value is three points off the family's best, while `k` is **completely inert** at every setting. | §4E, §6 |
| **A7** | **Ledger entry L13 is closed, and one of its numbers was wrong.** The *game's* forced-endgame incidence is raisable by three orders of magnitude (0.0056 → 4.20 a game), but the declarations are made by the **adversary**: the target's own incidence stays at exactly zero and cannot be raised this way, and raising it costs −17 points. And v0.6's forced machinery, exercised 1,400× more often than normal, resolves correctly **99.3–99.5%**, against the 0.286 the corpus records from two to eight observations. | §4F |
| **A8** | **Class C6 exists.** `engine/src/v07_adapt.hpp` is the corpus's first adversary carrying an online model of the target's policy that updates within a match. It is a clean negative in an uncommitted smoke test — both polarities lose — and the structural reason generalises: the only lever it has on the `oppCards` clause is its own hand count, and moving that means declaring earlier, which is the `decl` handicap applied to itself. | §4D |
| **A9** | **Five harness findings, three refutations.** The deal-seed inversion channel and the confidence broadcast are open and unused; `DealDP`'s shared pool and two contradictory residue rules are latent and were tested to be unreachable; the measurement flag is readable by the policies it measures. The action-cap adjudication farm, arm asymmetry and `BlockDP` aliasing are refuted — the last of them is a correction to THREAT-MODEL.md §6.3. | §7 |
| **A10** | **The evaluation material is sealed, physically and before anything can be tuned against it**: 4 training banks and 7 sealed holdout banks with commitment digests, an adversary bank split 14 / 14 by a rule fixed in advance, and a seal `runMatch` enforces rather than a battery script remembers. | §8 |

---

## 1. What was searched, and why it is not one search

The phase brief's constraint is not "run many searches", it is that the searches must not share a
bias: *"Fifteen runs of one search are one run."* Every exploiter search in this corpus before phase 2
— the v0.6 probe and phase 1's four classes alike — is the same search: a cross-entropy fit of a
linear vector, maximising win rate, from one seed, against one target, with three identical copies of
the result seated as a team. Phase 2 varies six axes, and five of them did not exist at the end of
phase 1 (RESEARCH-LOG.md §2.0).

| axis | phase 1 | phase 2 |
|---|---|---|
| **responder class** | C1 in-class linear, C2 extended (12 coords), C3 search-based, C5 white-box inversion | the same four, plus **C2 widened to 18 coordinates** with two groups no feature in the lineage carries, plus **C6 scripted-adaptive** — built here, and empty in the corpus until now |
| **objective** | win rate | win rate, **half-suit margin**, and five **mechanism objectives** that climb a per-decision failure mode of the *target*: its misdeclaration rate, its forced-endgame incidence, its ask hit rate, its declaration count, the action-limit rate, and game length |
| **seed** | one fitting bank | 31 fitting banks, one per search, all registered |
| **target** | `v06` and planted-handicap variants of it | 4 points of the frontier, from the 364 games/s deployed policy to the 6.4 games/s search |
| **seat count / regime** | k = 3, A1 | k = 3 and k = 1, A1 and **A2 ex-ante correlated** — fitted rather than merely measured |
| **starting basin** | the incumbent's own vector | the incumbent, the v0.5 defaults, a wide-sigma exploration, and a vector seeded at the direction the coordinate sweep found |

and, cutting across all six, a search of a genuinely different *kind*: an **unfitted coordinate
sweep** over the axis-aligned neighbourhood of the incumbent, which explores exhaustively at zero
fitting cost and cannot be trapped by the CEM's covariance collapse. It is the search that found
this phase's strongest arm.

**The fitted searches.**

| id | class | responder base | target | objective | fitting bank | budget (gens x pop x deals) | games spent | regime / seats | hypothesis |
|---|---|---|---|---|---:|---|---:|---|---|
| X01 | C1 | `v06` | Ffast | `win` | 7040001 | 12x16x250 | 96000 | A1, k = 3 | L0 in-class control at phase-1's standard budget |
| X02 | C1 | `v06` | Ffast | `win` | 7040002 | 20x20x350 | 280000 | A1, k = 3 | L0 budget-curve top rung: is the null a budget artifact |
| X03 | C1 | `v06` | Ffast | `win` | 7040003 | 8x12x150 | &mdash; | A1, k = 3 | L0 independent replicate at a third seed |
| X04 | C1 | `v06` | Ffast | `setdiff` | 7040004 | 8x12x150 | &mdash; | A1, k = 3 | L0 margin in half-suits, not games: the low-variance objective |
| X05 | C1 | `v06` | Ffast | `declerr` | 7040005 | 8x12x150 | 28800 | A1, k = 3 | L1 drive the target's misdeclaration rate |
| X06 | C1 | `v06` | Ffast | `asksupp` | 7040006 | 8x12x150 | &mdash; | A1, k = 3 | L3/L10 suppress the target's ask hit rate |
| X07 | C1 | `v06` | Ffast | `forced` | 7040007 | 8x12x150 | &mdash; | A1, k = 3 | L13 raise forced-endgame incidence under pressure |
| X08 | C1 | `v06` | Ffast | `events` | 7040008 | 8x12x150 | &mdash; | A1, k = 3 | stalling: lengthen the game |
| X09 | C1 | `v06` | Ffast | `declsupp` | 7040009 | 8x12x150 | &mdash; | A1, k = 3 | L1 timing: deny the target its declarations |
| X10 | C2 | `v07` | Ffast | `win` | 7040010 | 12x16x250 | &mdash; | A1, k = 3 | L11/L1 extended features incl. information denial |
| X11 | C2 | `v07` | Ffast | `declerr` | 7040011 | 8x12x150 | &mdash; | A1, k = 3 | L1 information denial, scored on the target's errors |
| X12 | C2 | `v07` | Ffast | `setdiff` | 7040012 | 8x12x150 | &mdash; | A1, k = 3 | L1 information denial, scored on the margin |
| X13 | C2 | `v07:dead7=1` | Ffast | `win` | 7040013 | 8x12x150 | 28800 | A1, k = 3 | L14 the deliberate miss admitted to the scored set |
| X14 | C2 | `v07:corr=3` | Ffast | `win` | 7040014 | 8x12x150 | 28800 | A2 correlated | A2 ex-ante correlated role plans |
| X15 | C2 | `v07` | Ffast | `asksupp` | 7040015 | 8x12x150 | &mdash; | A1, k = 3 | L10 gate pressure via the extended class |
| X16 | C1 | `v06` | Ffast | `limit` | 7040016 | 8x12x200 | &mdash; | A1, k = 3 | harness probe: drive games to the action cap |
| X17 | C1 | `v06` | Ffast | `win` | 7040017 | 8x12x150 | &mdash; | k = 1 | k=1 one-seat deviation column, fitted |
| X18 | C1 | `v06` | Fcheap | `win` | 7040018 | 8x12x120 | 23040 | A1, k = 3 | attack the cheap search end in class |
| X19 | C2 | `v07` | Fcheap | `win` | 7040019 | 8x12x120 | 23040 | A1, k = 3 | attack the cheap search end, extended |
| X20 | C1 | `v06` | Fcheap | `declerr` | 7040020 | 8x12x120 | 23040 | A1, k = 3 | L1 against a searching target |
| X21 | C1 | `v06` | Fmid | `win` | 7040021 | 6x10x60 | &mdash; | A1, k = 3 | attack the strongest measured configuration directly |
| X22 | C1 | `v06` | FRONT | `win` | 7040022 | 8x12x120 | &mdash; | A1, k = 3 | dominate the FRONTIER: min over {Ffast, Fcheap} |
| X23 | C2 | `v07` | FRONT | `win` | 7040023 | 8x12x120 | &mdash; | A1, k = 3 | dominate the frontier, extended class |
| X24 | C2 | `v07:dead7=1,corr=3` | Ffast | `win` | 7040024 | 8x12x150 | &mdash; | A2 correlated | deliberate miss x correlated roles |
| X25 | C1 | `v06` | Ffast | `win` | 7040025 | 8x12x150 | &mdash; | A1, k = 3, v0.5 basin | different starting basin: v0.5 defaults, not the incumbent |
| X26 | C1 | `v06` | Ffast | `win` | 7040026 | 8x12x150 | &mdash; | A1, k = 3, wide sigma | wide exploration: is the CEM trapped near the incumbent |
| X27 | C2 | `v07` | Ffast | `win` | 7040027 | 8x12x150 | &mdash; | A1, k = 3, wide sigma | wide exploration in the extended class |
| X28 | C1 | `v06` | Ffast | `setdiff` | 7040028 | 8x12x150 | &mdash; | k = 1 | k=1 one-seat deviation, margin objective |
| X29 | C2 | `v07` | Ffast | `win` | 7040029 | 12x16x250 | 96000 | A1, k = 3, seeded at the denial direction | refine the denial direction rather than rediscover it |
| X30 | C2 | `v07` | Ffast | `setdiff` | 7040030 | 8x12x150 | &mdash; | A1, k = 3, seeded at the denial direction | the same, on the low-variance margin objective |
| X31 | C2 | `v07` | Fcheap | `win` | 7040031 | 8x12x120 | &mdash; | A1, k = 3, seeded at the denial direction | does the refined denial direction survive the frontier's search |

**9 of the 31 designed searches completed** within the session's simulator
budget, at a total fitting cost of 627,520 games. The rows that ran are the ones carrying a
figure in the "games spent" column above; the rest were retired, in the order the table lists them,
when the machine was reassigned to the batteries this document actually quotes. What the completed
set still spans is the point of the axis table: **both classes** (C1 and C2), **both structural
switches** (`dead7` and `corr` with the A2 device live), **two objectives** (win rate and the target's
misdeclaration rate), **two frontier targets** (`Ffast` and `Fcheap`), **four budgets** from 23,040
to 280,000 games — including phase 1's standard rung and the top of its budget curve — and **one
registered fitting bank per search**. The axes are covered; the replication within each axis is not.

That is stated plainly rather than smoothed over, because it bears on the verdict: "the fitted
searches found nothing above the floor" is a claim about 9 searches, not 31.

**The unfitted arms.** 54 adversaries were measured without any fit at all, on registered
evaluation banks at 12,000 games each: single-knob deviations of the incumbent across the belief, the
policy prior, the declaration path, ask selection and the live-ask gate; the thirteen scripted
archetypes; the deception archetypes swept in both of the parameters no committed artifact has ever
set; the `psychTells` machinery that is compiled into the scripted baselines and enabled nowhere; the
frontier's own search configurations seated as adversaries; and the white-box class at settings other
than its struct defaults.

---

## 2. The resolution this phase can claim

An exploitability claim is a claim about a difference, and the difference is only as good as the
detection floor of the class that produced it. Phase 1 measured those floors at 12,000 evaluation
games a bank — C1 and C5 at 1.68 points, C2 at 2.31, C3 at 2.45 — and established that below about
1.7 points the binding constraint is **evaluation power, not search power**, so the floor should buy
down as (evaluation games)^−1/2. That is an assertion about a scaling law, and phase 2 cannot rest a
null on it. It was therefore re-measured, at four times the power, on the same rungs, the same
responders and the same two banks.

| class | rung | dTrue (pts) | dFound pooled | excess over control | excess &minus; dTrue | detected |
|---|---|---:|---|---|---:|:--:|
| C1 | `none` (control) | 0.00 &mdash; mirror | +0.79 [+0.48, +1.10] | &mdash; | &mdash; | &mdash; |
| C1 | `decl,hstr=0.05` | +0.67 [+0.28, +1.06] | +1.04 [+0.73, +1.34] | +0.25 [-0.19, +0.68] | -0.42 | no |
| C1 | `decl,hstr=0.08` | +0.88 [+0.47, +1.28] | +0.78 [+0.47, +1.08] | -0.01 [-0.45, +0.42] | -0.89 | no |
| C1 | `leak,hstr=1.5` | +1.53 [+0.96, +2.11] | +2.53 [+2.22, +2.84] | +1.74 [+1.30, +2.18] | +0.21 | **yes** |
| C1 | `decl,hstr=0.11` | +2.13 [+1.61, +2.65] | +3.51 [+3.20, +3.82] | +2.72 [+2.28, +3.16] | +0.59 | **yes** |
| C1 | `decl,hstr=0.15` | +2.34 [+1.79, +2.88] | +4.71 [+4.40, +5.01] | +3.91 [+3.48, +4.35] | +1.58 | **yes** |

| class | detection floor at this sample size | evaluation games per bank |
|---|---:|---:|
| C1 | 1.53 pts | 48,000 |

Rungs run at this power: the false-positive control plus 5 planted rungs (`decl,hstr=0.05`, `decl,hstr=0.08`, `decl,hstr=0.11`, `decl,hstr=0.15`, `leak,hstr=1.5`), spanning the declaration family and the readability rung phase 1 quoted its 1.68-point floor from.

The banks are *extended*, not replaced: a deal's seed is a function of its index alone, so the first
6,000 deals of each are bit-identical to phase 1's and the additional 18,000 are new. The comparison
is nested, and a floor that failed to buy down would have refuted the scaling law rather than changed
banks.

**It failed to buy down.** Phase 1's floor was 1.68 points at 12,000 evaluation games a bank, set by
the `leak,hstr=1.5` rung. At 48,000 games a bank the floor is **1.53 points**, set by the *same* rung
— whose `dTrue` simply re-measures slightly lower at higher power. Four times the games bought
0.15 points of floor, not the 0.78 the scaling law predicts.

The prediction is refuted at its own predicted value. Phase 1 wrote that phase 2 "can buy a
≈0.9-point floor with 4× the evaluation games per cell, without touching the responders". At exactly
4× the evaluation games, the +0.88-point rung is **still undetected**, and not because the interval
is too wide: the excess is **−0.01 [−0.45, +0.42]**. Where phase 1 saw +0.69 at that rung and could
not exclude zero, four times the data resolves it *to* zero. Below about 1.5 points the C1 responder
is not unresolved, it is **empty** — the binding constraint is search power, and buying games buys a
sharper zero.

**Only C1 was re-floored, and the floor is per class.** The battery re-ran phase 1's C1 responders;
C2, C3 and C5 keep phase 1's measured floors of 2.31, 2.45 and
1.68. That matters for reading §4A: the contestation arm is a **C2-class** spec, so its
+2.71 [+2.27, +3.15] is read against **2.31**, and under the lower-bound
criterion it does **not** clear its own class floor — [+2.27, +3.15] has a lower bound below 2.31.
It is reported as a measured edge over the deployed policy, which it is, and not as a certified
detection.

Two further consequences, and both raise the bar. The declaration family — the channel carrying the
whole of v0.6's measured margin — sits worse than the class floor at **2.13**. And phase 3 cannot buy
resolution below ~1.5 points with games alone: a sub-1.5-point claim needs a better responder or a
per-decision estimator, which is what ledger entry L5 has said all along.

---

## 3. Channel ceilings, measured before any adversary was fitted

Phase 1's most useful construction was `dTrue`: the size of a planted edge measured by a *known*
exploiter before any responder was fitted against it. Phase 2 applies the same idea to mechanisms
rather than to handicaps. An adversary that raises some quantity in the target is only interesting if
raising that quantity is worth something, so the worth is measured directly — by modifying the
**target** so the mechanism is permanently on, and letting the unmodified incumbent play it. The
resulting edge is the **ceiling** on what any adversary attacking that mechanism could ever collect.

| arm | target the incumbent plays | `v06` edge (pts) | 95% CI | target decl. acc. | target decl./game | target lock hold | n |
|---|---|---:|---|---:|---:|---:|---:|
| `control` | `v06` | +0.00 | mirror &mdash; no interval | 0.9777 | 4.49 | 4.67 | 12,000 |
| `urg-always-pool` | `v06:pool=45` | +0.38 | [-0.06, +0.81] | 0.9768 | 4.49 | 4.85 | 12,000 |
| `urg-always-oppcard` | `v06:oppfloor=54` | +0.38 | [-0.06, +0.81] | 0.9768 | 4.49 | 4.85 | 12,000 |
| `urg-always-events` | `v06:force=1` | +15.18 | [+14.41, +15.94] | 0.8115 | 4.75 | 2.62 | 12,000 |
| `urg-always-askfl` | `v06:askfloor=1.1` | +0.38 | [-0.06, +0.81] | 0.9768 | 4.49 | 4.85 | 12,000 |
| `urg-always-all` | `v06:pool=45,oppfloor=54,force=1,askfloor=1.1` | +15.18 | [+14.41, +15.94] | 0.8115 | 4.75 | 2.62 | 12,000 |
| `urg-never` | `v06:pool=-1,oppfloor=-1,force=1000000,askfloor=-1` | -1.62 | [-1.99, -1.24] | 0.9902 | 4.47 | 4.79 | 12,000 |
| `m2-off` | `v06:m2=0` | -0.87 | [-1.04, -0.71] | 0.9867 | 4.47 | 4.66 | 12,000 |
| `m2-off-never-urg` | `v06:m2=0,pool=-1,oppfloor=-1,force=1000000,askfloor=-1` | -1.62 | [-1.99, -1.24] | 0.9902 | 4.47 | 4.79 | 12,000 |
| `no-declare` | `v06:declare=0` | +7.75 | [+6.92, +8.58] | 0.0000 | 0.00 | 48.37 | 12,000 |

Two results, and the first of them is the most important negative in the phase.

**The declaration-urgency channel has a ceiling of +0.38 points
[-0.06, +0.81], and is dead.** v0.6's declaration rule replaces an expected-value
comparison with a bare threshold whenever `urgent` fires, and on a half-suit the team provably owns
that threshold is a coin flip on the allocation — which is exactly the error class ledger entry L1
sizes at 88.1% of remaining misdeclarations. Measured per decision, half of v0.6's declarations are
taken under urgency and those are 3.36 points less accurate. At the ledger's own conversion that
reads as about two points of win rate, and it made this the phase's leading hypothesis. A target that
is *permanently* urgent loses +0.38 points and its declaration accuracy falls from
0.9777 to 0.9768 — nine hundredths of a point, not 3.36.
**The per-decision gap is confounded: urgency fires in hard positions.** The three clauses probed
separately agree to the last digit in every column — which tests nothing about the instrumentation,
because setting any one of them makes `urgent` permanently true and the three are then the same
policy. What it does establish is that each clause alone is sufficient to saturate the predicate, so
the ceiling is a property of urgency rather than of any particular route into it. RESEARCH-LOG.md §2.2 carries the full decomposition, including the C6 adversary built to attack
the channel and the structural reason it cannot: the only lever on the clause an adversary controls is
its own hand count, the only way to move it is to declare earlier, and declaring earlier is the
`decl` handicap family applied to oneself.

**v0.6 carries a fifteen-point cliff behind an event counter, and nothing can reach it.**
`pub.nEvents >= forceDeclareEvents` is a clause of `urgent` *and* the trigger of `pressure()`, and the
two are not the same mechanism: at pressure rung 1 the target drops its team-ownership floor from
0.849 to 0.25, bypasses the capacity gate, and cashes any half-suit at `pAlloc >= 0.5`. Measured
ceiling: +15.18 points [+14.41, +15.94], with the target's declaration
accuracy collapsing to 0.8115. The counter is set to 220.

**Against an unmodified `v06` no adversary gets close.** The mean is not the quantity a threshold
cares about, so this is measured per game: 800 games a configuration on bank 7051001, from
`fish7 pathology`.

| adversary (vs `v06`) | mean events | median | p90 | p99 | **max** |
|---|---:|---:|---:|---:|---:|
| `v06` (mirror) | 95.3 | 95 | 109 | 119 | 130 |
| `v07:r12=25` | 97.1 | 97 | 111 | 124 | 135 |
| `v06:rtie=1,…,askfloor=-1` | 95.6 | 96 | 110 | 121 | 132 |
| `v06:declare=0` | 104.5 | 104 | 119 | 133 | **149** |
| the most aggressive stall in the grammar | 105.7 | 105 | 122 | 140 | **148** |

The longest single game any adversary produced against the incumbent is **149 events** against a
220-ask rung. Filed as a standing hazard rather than an exploit.

**The hazard is nonetheless real for phase 3, in a way that is not obvious and that showed up in the
gate rather than in the strength table.** A configuration that switches the urgency escalation off
also switches off `pressure()`, and `pressure()` is v0.5's *termination guarantee*. In **self-play** —
which is the v0.7 deployment configuration, three identical copies — the `m1=0` urgency-off stack's
game-length tail reaches **405 events**, three times the incumbent's 130, because neither side has an
escalation left to break a stall. Against an unmodified `v06` the same configuration maxes at 135,
because the opponent still has one. **A candidate that disables the escalation is safe against v0.6
and unbounded against itself**, and the deployment configuration is the second case. Phase 3 must
print `eventsPerGame`, its p99 and its maximum **in mirror**, against the 220 rung.

**The cliff is graded, not a step, and the sweep says exactly how much lengthening would be needed.**
Moving the threshold down is the same operation as lengthening the game, so the curve below is a dose
response read backwards. A target pressured from event 1 loses 15.31 points; from 40, 10.61; from 60,
6.24; from 80, 2.50; from 100, **0.12**; and from 120 upward it is **bit-identical to the shipped
configuration** — same edge, same declaration accuracy to four places, same lock hold, same events per
game — because essentially no game reaches 120 (p90 = 109, p99 = 119).

So the exposed fraction of the cliff at the shipped threshold of 220 is not "small", it is **exactly
zero**, and it stays zero until the threshold falls below about 110. Turned around: an adversary would
have to roughly **double** the mean game length to expose any of it at all, and roughly **quadruple**
it to reach the +6-point rung. The longest game produced against the incumbent is 149 events against a
mean of 95.

The last two rows are the second escalation rung. `forceStage2` cashes the best candidate whatever it
is, and at a threshold of 40 it is worth **+39.75** with the target's declaration accuracy collapsing
to 0.4893 — a coin flip. It is off by default and its threshold is 7/5 x 220 = 308, so it is further
out of reach than the first rung; it is recorded because it is the largest number in the phase and
because a candidate that both lengthens games and enables stage 2 would find it.

| target | `v06` edge | 95% CI | target decl. acc. | target decl./game | target lock hold | events/game | n |
|---|---:|---|---:|---:|---:|---:|---:|
| `v06` | +0.00 | mirror &mdash; no interval | 0.9791 | 4.49 | 4.66 | 95.1 | 8,000 |
| `v06:force=1` | +15.31 | [+14.37, +16.26] | 0.8115 | 4.76 | 2.59 | 90.7 | 8,000 |
| `v06:force=40` | +10.61 | [+9.76, +11.46] | 0.8630 | 4.67 | 3.35 | 92.2 | 8,000 |
| `v06:force=60` | +6.24 | [+5.52, +6.95] | 0.9057 | 4.61 | 3.99 | 93.3 | 8,000 |
| `v06:force=80` | +2.50 | [+2.05, +2.95] | 0.9512 | 4.54 | 4.50 | 94.5 | 8,000 |
| `v06:force=100` | +0.12 | [+0.01, +0.24] | 0.9768 | 4.50 | 4.64 | 95.0 | 8,000 |
| `v06:force=110` | +0.02 | [-0.01, +0.06] | 0.9789 | 4.50 | 4.66 | 95.1 | 8,000 |
| `v06:force=120` | +0.00 | [-1.10, +1.10] | 0.9791 | 4.49 | 4.66 | 95.1 | 8,000 |
| `v06:force=140` | +0.00 | [-1.10, +1.10] | 0.9791 | 4.49 | 4.66 | 95.1 | 8,000 |
| `v06:force=160` | +0.00 | [-1.10, +1.10] | 0.9791 | 4.49 | 4.66 | 95.1 | 8,000 |
| `v06:force=180` | +0.00 | [-1.10, +1.10] | 0.9791 | 4.49 | 4.66 | 95.1 | 8,000 |
| `v06:force=200` | +0.00 | [-1.10, +1.10] | 0.9791 | 4.49 | 4.66 | 95.1 | 8,000 |
| `v06:stage2=1,force=40` | +39.75 | [+39.12, +40.38] | 0.4893 | 5.72 | 3.81 | 72.4 | 8,000 |
| `v06:stage2=1,force=100` | +0.12 | [+0.01, +0.24] | 0.9768 | 4.50 | 4.64 | 95.0 | 8,000 |
| `v06` | +0.00 | mirror &mdash; no interval | 0.9791 | 4.49 | 4.66 | 95.1 | 8,000 |
| `v06:force=1` | +15.31 | [+14.37, +16.26] | 0.8115 | 4.76 | 2.59 | 90.7 | 8,000 |
| `v06:force=40` | +10.61 | [+9.76, +11.46] | 0.8630 | 4.67 | 3.35 | 92.2 | 8,000 |
| `v06:force=60` | +6.24 | [+5.52, +6.95] | 0.9057 | 4.61 | 3.99 | 93.3 | 8,000 |
| `v06:force=80` | +2.50 | [+2.05, +2.95] | 0.9512 | 4.54 | 4.50 | 94.5 | 8,000 |
| `v06:force=100` | +0.12 | [+0.01, +0.24] | 0.9768 | 4.50 | 4.64 | 95.0 | 8,000 |
| `v06:force=110` | +0.02 | [-0.01, +0.06] | 0.9789 | 4.50 | 4.66 | 95.1 | 8,000 |
| `v06:force=120` | +0.00 | [-1.10, +1.10] | 0.9791 | 4.49 | 4.66 | 95.1 | 8,000 |
| `v06:force=140` | +0.00 | [-1.10, +1.10] | 0.9791 | 4.49 | 4.66 | 95.1 | 8,000 |
| `v06:force=160` | +0.00 | [-1.10, +1.10] | 0.9791 | 4.49 | 4.66 | 95.1 | 8,000 |
| `v06:force=180` | +0.00 | [-1.10, +1.10] | 0.9791 | 4.49 | 4.66 | 95.1 | 8,000 |
| `v06:force=200` | +0.00 | [-1.10, +1.10] | 0.9791 | 4.49 | 4.66 | 95.1 | 8,000 |
| `v06:stage2=1,force=40` | +39.75 | [+39.12, +40.38] | 0.4893 | 5.72 | 3.81 | 72.4 | 8,000 |
| `v06:stage2=1,force=100` | +0.12 | [+0.01, +0.24] | 0.9768 | 4.50 | 4.64 | 95.0 | 8,000 |

---

## 4. The taxonomy

Clusters are mechanisms, not scores. Two arms are in the same cluster when the same change to the
target would close both. The table is ordered by the **worst cell over the frontier** — the severity
that leads, per the project's standing rule — with arms measured against only one frontier point
sorted below every arm that has a full profile, because a single cell is not a frontier severity.
Note before reading it that **neither of the two arms with a full profile is an exploit**: §4A and
§4B establish that both are strength gains. Naming that is what keeps the rest of the table honest.

| arm | cluster | vs `Ffast` | vs `Fcheap` | vs `Fmid` | vs `Fsearch` | worst cell over the frontier |
|---|---|---|---|---|---|---|
| `composite` | contestation-x-search | +4.40 &plusmn;0.45 | +2.40 &plusmn;0.63 | +2.21 &plusmn;1.11 | &mdash; | **+2.21** |
| `SW-noURG` | one-switch | &mdash; | -0.69 &plusmn;0.53 | -1.55 &plusmn;0.95 | &mdash; | **-1.55** |
| `r12=25` | contestation | +2.71 &plusmn;0.44 | +0.55 &plusmn;0.63 | -0.99 &plusmn;1.10 | -1.69 &plusmn;2.41 | **-1.69** |
| `C5-unfitted` | whitebox | &mdash; | -1.08 &plusmn;0.59 | -2.14 &plusmn;1.04 | &mdash; | **-2.14** |
| `m2off-r12` | contestation-x-m2 | +3.17 &plusmn;0.44 | &mdash; | &mdash; | &mdash; | &mdash; (one point only) |
| `search-m2` | m2-x-search | +2.48 &plusmn;0.37 | &mdash; | &mdash; | &mdash; | &mdash; (one point only) |
| `search` | search-strength | +1.89 &plusmn;0.37 | &mdash; | &mdash; | &mdash; | &mdash; (one point only) |
| `denial-all` | contestation-group | +1.01 &plusmn;0.44 | &mdash; | &mdash; | &mdash; | &mdash; (one point only) |
| `m2off` | m2-defect | +0.70 &plusmn;0.08 | &mdash; | &mdash; | &mdash; | &mdash; (one point only) |
| `control` | reference | mirror | &mdash; | &mdash; | &mdash; | &mdash; (one point only) |
| `r17=25` | tally-inflation | -1.22 &plusmn;0.30 | &mdash; | &mdash; | &mdash; | &mdash; (one point only) |

### A. Half-suit contestation — the strongest arm phase 2 found, and not the mechanism it was built for

**The arm.** `v07:r12=25` — the extended responder class carrying a single non-zero
coordinate out of eighteen, hand-set, **with no fit at all**. Against `v06` it scores
+2.71 [+2.27, +3.15] over 48,000 games on two banks.

**The coordinate.** `oppCertDonate` = (1 − p) · (expected fraction of this half-suit held by the
opposing team) · (fraction of its cards whose holder the actor cannot place). It was added to express
*information denial* — the negative certificate a miss publishes about a target seat, which is the
only primitive that can separate two seats of one team, because `enumerateAsks` forbids asking a
teammate and so a team can never manufacture one about itself. The design intent was that the
coefficient would come out **negative**: refuse to donate.

**It came out positive, and the mechanism is a different one.** The negative branch — the sign the
feature was designed for, "refuse to donate the certificate" — was swept separately and is flat to
negative throughout:

| dose | edge vs `v06` | 95% CI | per bank | n |
|---|---:|---|---|---:|
| `r12=-25` | -1.47 | [-2.07, -0.87] | -1.57 / -1.37 | 24,000 |
| `r12=-10` | -0.26 | [-0.83, +0.30] | -0.57 / +0.05 | 24,000 |
| `r12=-5` | +0.14 | [-0.38, +0.65] | -0.25 / +0.52 | 24,000 |

The whole effect lives on the positive branch, and the design intent is **refuted** rather than merely
unexplored: pricing information denial the way the coordinate was written to price it is worth
nothing at a small dose and −1.47 at a large one.

| arm | cluster | edge vs `v06` | per bank | n | adversary decl. acc. | target decl. acc. | target lock hold | target decl./game | target ask acc. | limit hits |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `composite` | contestation-x-m2-x-search | +4.40 [+3.95, +4.84] | +4.43 / +4.36 | 48,000 | 0.9719 | 0.9818 | 5.39 | 4.28 | 0.5180 | 0.0000 |
| `search-r12` | contestation-x-search | +3.99 [+3.54, +4.43] | +4.03 / +3.95 | 48,000 | 0.9675 | 0.9818 | 5.39 | 4.28 | 0.5179 | 0.0000 |
| `m2off-r12` | contestation-x-m2 | +3.17 [+2.73, +3.61] | +3.31 / +3.03 | 48,000 | 0.9708 | 0.9822 | 5.33 | 4.31 | 0.5210 | 0.0000 |
| `r12=25` | contestation | +2.71 [+2.27, +3.15] | +2.85 / +2.58 | 48,000 | 0.9651 | 0.9822 | 5.33 | 4.31 | 0.5210 | 0.0000 |
| `r12=20` | contestation | +2.53 [+2.09, +2.98] | +2.69 / +2.38 | 48,000 | 0.9645 | 0.9825 | 5.36 | 4.30 | 0.5234 | 0.0000 |
| `search-m2` | m2-x-search | +2.48 [+2.11, +2.85] | +2.26 / +2.70 | 48,000 | 0.9876 | 0.9789 | 4.67 | 4.44 | 0.5365 | 0.0000 |
| `r12=15` | contestation | +2.23 [+1.79, +2.67] | +2.32 / +2.14 | 48,000 | 0.9649 | 0.9818 | 5.27 | 4.33 | 0.5274 | 0.0000 |
| `search` | search-strength | +1.89 [+1.51, +2.26] | +1.66 / +2.11 | 48,000 | 0.9815 | 0.9789 | 4.67 | 4.44 | 0.5364 | 0.0000 |
| `r12=30` | contestation | +1.25 [+0.81, +1.70] | +1.65 / +0.85 | 48,000 | 0.9643 | 0.9818 | 5.35 | 4.35 | 0.5162 | 0.0000 |
| `denial-all` | contestation-group | +1.01 [+0.56, +1.45] | +0.95 / +1.06 | 48,000 | 0.9642 | 0.9842 | 5.23 | 4.36 | 0.5402 | 0.0000 |
| `m2off` | m2-defect | +0.70 [+0.62, +0.78] | +0.63 / +0.77 | 48,000 | 0.9870 | 0.9784 | 4.61 | 4.50 | 0.5416 | 0.0000 |
| `r12=5` | contestation | +0.65 [+0.25, +1.04] | +0.19 / +1.11 | 48,000 | 0.9787 | 0.9786 | 4.66 | 4.48 | 0.5400 | 0.0000 |
| `r12=10` | contestation | +0.62 [+0.18, +1.07] | +0.80 / +0.45 | 48,000 | 0.9743 | 0.9801 | 4.88 | 4.44 | 0.5356 | 0.0000 |
| `r14=25` | contestation | +0.06 [+0.01, +0.10] | +0.05 / +0.06 | 48,000 | 0.9784 | 0.9784 | 4.62 | 4.49 | 0.5416 | 0.0000 |
| `control` | reference | +0.00 mirror &mdash; no interval | +0.00 / +0.00 | 48,000 | 0.9784 | 0.9784 | 4.61 | 4.49 | 0.5416 | 0.0000 |
| `r15=25` | contestation | -0.87 [-1.26, -0.47] | -1.24 / -0.50 | 48,000 | 0.9762 | 0.9818 | 4.68 | 4.50 | 0.5496 | 0.0000 |
| `r17=25` | tally-inflation | -1.22 [-1.52, -0.92] | -1.21 / -1.23 | 48,000 | 0.9787 | 0.9770 | 4.56 | 4.55 | 0.5374 | 0.0000 |
| `r13=25` | contestation | -2.31 [-2.74, -1.88] | -2.12 / -2.50 | 48,000 | 0.9776 | 0.9808 | 4.51 | 4.57 | 0.5595 | 0.0000 |
| `r16=25` | tally-inflation | -2.33 [-2.65, -2.01] | -2.34 / -2.33 | 48,000 | 0.9786 | 0.9769 | 4.51 | 4.59 | 0.5365 | 0.0000 |
| `r12=40` | contestation | -4.89 [-5.33, -4.45] | -4.92 / -4.87 | 48,000 | 0.9672 | 0.9798 | 5.54 | 4.60 | 0.5025 | 0.0000 |

The dose response is a clean inverted U — +0.65 at 5, +0.62 at 10, +2.23 at 15, +2.53 at 20, a peak
of +2.71 at 25, +1.25 at 30 and **−4.89 at 40** — with the two banks agreeing to within
0.4 points at every dose. A vector that wins by luck does not do that.

**The other three coordinates of the group do nothing, which is the sharpest evidence about what the
mechanism is.** All four were written to express information denial and all four were measured at the
same dose: `r13` (the same opponent-mass term multiplied by *p* instead of by 1 − *p*) is **−2.31**;
`r14` (a step term for "the opposing team is within one card of owning this half-suit outright") is
**+0.06**, inert; `r15` (opponent mass weighted by game phase) is **−0.87**. Only the coordinate that
combines *a likely miss* with *a half-suit the opponents dominate* with *residual ambiguity* does
anything at all.

**What it does to the target, measured per decision on the target arm.** `fish7 v7decide
--capture=b`, bank 7050001, 3,000 deals x 2 = 6,000 games, ~26,000 target declarations and ~260,000
target ask decisions a cell:

| target-arm KPI | `v06` mirror | under `v07:r12=25` | n | direction |
|---|---:|---:|---:|---|
| `askHitRate` | 0.54038 | 0.52138 | 260,987 | **the target's asks get 1.9 points worse** |
| `declAccuracy` | 0.97953 | 0.98197 | 25,847 | the target declares *better* |
| `declUrgentShare` | 0.50274 | 0.43278 | 25,847 | and is *less* often urgent |
| `declAllocErrorShare` | 0.85734 | 0.83047 | 466 | its allocation errors do not rise |
| `gateBindRate` | 0.00994 | 0.00982 | 260,987 | the live-ask gate is not the route |
| `ownLockedAskRate` | 0.10630 | 0.10443 | 260,987 | nor is the own-locked channel |

**This rules out the declaration channel, which is where the phase expected to find everything.** The
target does not misdeclare more against this arm: it misdeclares *less*, it is urgent *less* often,
and its allocation-error share does not move. What moves is its **ask accuracy**, by 1.9 points over
260,000 decisions. At the game level the same story appears as the target completing fewer half-suits
and holding the ones it has longer — declarations per game 4.31 against 4.49, lock hold
5.33 against 4.61 — and the adversary paying in the same coin: its own ask accuracy
falls from 0.5426 to 0.5336 and its own declaration accuracy from 0.9784 to
0.9651.

**It plays a worse game by its own KPIs and takes more half-suits.** That is the signature the phase
brief asks for — but the cross-opponent test below is what decides whether the signature means
"exploit", and it decides against.

**It is not reducible to either of its factors, and it is not a mis-set hit weight.** Three controls,
each at 4,000 games on bank 7030004:

| control | what it isolates | edge |
|---|---|---:|
| `v07:r2=25` | prefer half-suits the opposing team dominates, without the (1−p) and ambiguity factors | −1.30 [−2.88, +0.25] |
| `v07:r2=10` | the same, at a lower dose | −0.77 [−2.22, +0.68] |
| `v06:w0=8` | simply weight hit probability less (11.27 → 8) | +0.40 [−1.02, +1.82] |
| `v06:w0=6` | … less still | −0.62 [−2.10, +0.82] |
| `v06:w0=4` | … less again | −1.97 [−3.48, −0.50] |
| `v07:r12=25,r2=-25` | the target coordinate with the plain opponent-mass term cancelled | −11.50 [−13.00, −10.00] |

Neither factor alone reproduces the effect and neither does a blunt reduction in the hit-probability
weight. The product is doing the work.

**Is it an exploit of v0.6 or simply a better policy? Measured, and it is a better policy.**

The KPI signature says *exploit*: a strength gain does not lower the adversary's own declaration
accuracy by 1.5 points while raising the target's lock-hold time. The cross-opponent profile is the
independent read, and it was run twice — once at 8,000 games a cell, and then again at 48,000 a cell,
because the first read was not powered to separate the hypotheses.

| opponent | `v06` | `v07:r12=25` | increment | n a cell |
|---|---:|---:|---:|---:|
| `v06` | mirror | +2.56 &plusmn;0.44 | +2.56 | 48,000 |
| `v05` | +0.92 &plusmn;0.44 | +2.98 &plusmn;0.44 | **+2.05** | 48,000 |
| `v04` | +0.83 &plusmn;0.44 | +2.81 &plusmn;0.44 | **+1.97** | 48,000 |
| `detective` | +27.50 &plusmn;0.37 | +28.29 &plusmn;0.37 | **+0.79** | 48,000 |

On win rate the increment — the arm's edge over an opponent minus the incumbent's edge over the same
opponent — looks concentrated: +2.56 against the incumbent and its two predecessors, and +0.79 against
a v0.3-era scripted baseline, a difference that is comfortably separated. That reads as an exploit of
something the FishBot family shares, which would be the declaration path, since `V06Agent` inherits it
unchanged from `V05Agent` and `V04Agent` and the scripted baselines have their own.

**It is a scale artifact.** A win rate is compressed near its extremes: the same strength difference
buys fewer points at 77.5% than at 50%, so an increment measured against a near-parity opponent and
one measured against a weak opponent are not on the same scale. The corpus's own linear unit is the
mean half-suit differential — the ledger fits one unit of it to 14.7 win-rate points — and it is not
compressed.

| opponent | `v06` differential | `v07:r12=25` differential | increment (sets) | x 14.7 |
|---|---:|---:|---:|---:|
| `v06` | mirror | +0.171 &plusmn;0.009 | +0.171 | +2.52 |
| `v05` | +0.072 &plusmn;0.011 | +0.226 &plusmn;0.001 | **+0.154** | +2.26 |
| `v04` | +0.066 &plusmn;0.006 | +0.191 &plusmn;0.002 | **+0.124** | +1.83 |
| `detective` | +2.049 &plusmn;0.008 | +2.144 &plusmn;0.002 | **+0.095** | +1.40 |

On the linear scale the increment is +2.52, +2.26, +1.83,
+1.40 win-rate-equivalent points — a factor of 1.8 across the panel rather than the
factor of 3.2 the win-rate column shows. The gradient is monotone and it *is* resolved bank to bank
(the per-bank spreads are ±0.001 to ±0.011 sets against differences of 0.017 to 0.030), so something
real is falling off. What phase 2 cannot do is say what: **the three strong opponents are the three
FishBots**, so "it exploits the lineage's shared machinery" and "there is simply more to win against
a strong opponent" are confounded, and no opponent exists in the corpus that is near parity with
`v06` and does not share its declaration code.

What both readings agree on is the part that matters here: **the gain is not specific to v0.6.** It
survives at 55% of its size against an opponent that shares none of v0.6's machinery, and the
per-decision evidence says its route is the ask side rather than the declaration side — which is a
route the scripted baselines have too.

So the classification is settled and it is the unglamorous one: **this is a candidate architecture,
not an exploit.** It is reported in §5 as a measured edge over the deployed policy and it is **not**
counted as exploitability under E1. That is the same shape as phase 1's finding about the white-box
class — readable, convertible into general strength, not into target-specific exploitation — reached
by a different route, and it is the second time in this programme that a promising adversary has
turned out to be a candidate instead. Phase 3 should treat it as the latter.

The n = 8,000 first pass is retained below, because the two passes disagree in exactly the way an
underpowered comparison disagrees with a powered one, and that is worth being able to see.

| opponent | `v06` | `v07:r12=25` | difference |
|---|---|---|---|
| `v06` | mirror | +3.17 &plusmn;1.08 | &mdash; |
| `v05` | +0.98 &plusmn;1.07 | +2.89 &plusmn;1.10 | **+1.91** |
| `v04` | +0.95 &plusmn;1.09 | +4.00 &plusmn;1.11 | **+3.05** |
| `v03` | +25.92 &plusmn;0.92 | +27.15 &plusmn;0.91 | **+1.22** |
| `detective` | +26.48 &plusmn;0.92 | +28.10 &plusmn;0.89 | **+1.62** |
| `lockout` | +28.66 &plusmn;0.89 | +29.40 &plusmn;0.87 | **+0.74** |

**What phase 2 will not claim.** The mechanism is *characterised* — a tempo effect in the race to
claim half-suits, bought with ask accuracy — but it is not *attributed*: which of the two lock-hold
movements is cause and which is consequence is not settled by these measurements, and the coordinate
is a product of three quantities whose separate contributions phase 2 did not have the budget to
resolve. That attribution is phase 3/4 work and it is exactly the kind of work the v0.6 record says
it could not afford ("v0.6 shipped its search off partly because it could not afford to locate its own
gain"). What is settled is that the effect is real, replicated, dose-responsive, monotone in the right
region, commit-gate clean (0.022% dead asks, longest dead run 1, zero action-limit games), and
**outside every class the corpus had before this phase**.

### B. Test-time search — the strength baseline, and explicitly not an exploitability number

`v06:s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26` seated as the adversary scores
+1.89 [+1.51, +2.26] against `v06` on the phase-2 training banks. It is the
strongest thing in the corpus that beats the deployed policy, and it is **not an exploit**: it is the
target's own search, which the deployed configuration ships switched off for cost. There is no KPI
that separates "the target was exploited" from "the adversary was stronger", and the phase brief asks
for that case to be named rather than smuggled into a ranking. It is named here, and it is the number
every other cluster has to beat before the word "exploit" is used.

This also corrects the reading of phase 1's class ladder. Phase 1 reported C3 (search-based) at +1.86
against the unhandicapped incumbent and read it as the strongest class. Most of that is the search
improving the policy rather than the `roppo=` opponent model attacking it: the same search *without*
any opponent model measures +2.19 in phase 1's own T2 battery, at a stronger operating point than the
one the floor battery used. **The adversarial apparatus in C3 was worth approximately nothing; the
search was worth everything.**

### C. The urgency–M2 defect complex — a defect in the target, not an attack on it

This is the cluster phase 2 expected to be the headline, and it inverted.

v0.6's declaration rule replaces an expected-value comparison with a bare threshold whenever `urgent`
fires, and on a half-suit the team provably owns that threshold is `pAlloc >= 0.5` — a coin flip on
the allocation, which is exactly the error class ledger entry **L1** sizes at 88.1% of remaining
misdeclarations. Half of v0.6's declarations are taken under urgency and those are 3.36 points less
accurate (RESEARCH-LOG.md §2.2).

**An adversary can add urgency and it is worth +0.38 points
[-0.06, +0.81] — the ceiling, measured by making the target permanently urgent.** The
per-decision gap is confounded: urgency fires in hard positions, and forcing it in easy ones costs
almost nothing.

**But a target that can never be urgent beats `v06` by +1.23 to +1.62 points, replicated on three
banks, and declares 1.25 points more accurately.** The whole channel is worth about two points across
its range and the adversary owns only the smaller half of it. **The larger half is a v0.7 candidate,
not an exploit, and it is the clearest thing phase 2 hands to phase 3.**

| arm | seating | edge over `v06` | 95% CI | n | its declaration accuracy |
|---|---|---:|---|---:|---:|
| `v06:pool=-1,oppfloor=-1,force=1000000,askfloor=-1`, bank 7030001 | as adversary | +1.23 | [+0.88, +1.60] | 12,000 | 0.9907 |
| the same, bank 7030003 | as adversary | +1.34 | [+0.98, +1.71] | 12,000 | 0.9906 |
| the same, bank 7030002 | as **target**, in the ceiling battery | +1.62 | [+1.24, +1.99] | 12,000 | 0.9902 |

The first two are two banks of one measurement; the third is the same configuration measured from the
other side in a different battery, and is quoted as corroboration rather than as a third replicate.

On one common base — 400 deals x 2 at seed 31, `research/v07/results/P0-gate2.txt` — its mirror
misdeclaration rate is **1.278%** against the incumbent's **2.556%**, and it
passes the commit gate: 0.0295% dead asks, longest dead run 1, zero action-limit games.

**M2 is the same defect, not a second one.** `feasibleAllocation` promotes the team-ownership
probability to the allocation probability (`v.pTeam = std::max(v.pTeam, v.pAlloc)`), so the
team-ownership floor is passed on the strength of a quantity computed by a different approximation.
Switching it off is worth +0.57 to +0.87 on its own — and a target with urgency disabled is
**bit-identical** whether M2 is on or off: the same md5 over a 400-deal pathology digest, recorded in
`research/v07/results/P0-gate2.txt`. M2 only ever matters through the urgency branch, so the two
entries are one entry. (The `pTeam = max(pTeam, pAlloc)` promotion appears twice in `evaluateSet`, at
v05.hpp:851 inside the M2 branch and again at :865 in the branch taken without it, so M2 is not
uniquely responsible for the promotion — what `m2=0` changes is which allocator supplies `pAlloc`.)

### D. Declaration-timing induction, and the C6 class built to do it — a clean negative

`engine/src/v07_adapt.hpp` is the corpus's first scripted-adaptive adversary (threat-model class C6,
which THREAT-MODEL.md §4.4 records as empty and INSTRUMENT.md defers to phase 2). It carries an online
model of whether the target is currently holding a half-suit it owns outright but cannot allocate,
estimated every decision from the adversary's own posterior, and times its own declarations against
that estimate. Both polarities lose in a smoke test at n = 1,600 — holding declarations while the target is safe
costs 2.1 to 3.4 points, accelerating them when it is vulnerable costs 2.3, and doing both costs 5.3,
with the target's declaration accuracy unmoved and the adversary's falling two to three points.
**Those cells are an uncommitted smoke measurement**: the C6 arms are in the deception battery, whose
artifact is the one still outstanding at the foot of §6, so they are quoted here as what they are
rather than as registered results.

The structural reason generalises and is worth stating: **the only lever an adversary has on the
`oppCards` clause is its own hand count, the only way to lower its own hand count is to declare, and
declaring earlier is the `decl` handicap family applied to oneself** — which phase 1 measured at +0.62
to +2.45 points of cost to the holder. The adversary pays exactly the cost it is trying to impose.
`v07c:mode=0` is `v06` bit for bit, which is the class's identity control.

### E. Belief and policy-prior corruption — the deception family, and it is exhausted

Every unfitted single-knob deviation of the incumbent's belief and policy prior was measured at
12,000 games. **Every one of them loses.** The closest to parity is `v06:ptheta=0.2` at −0.84
[−1.67, −0.02]; zeroing the prior outright costs 2.68; setting it to 1.5 costs 9.62; and every belief
mode other than the shipped Sinkhorn `Fast` costs between 7.9 and 34.0 points. The shipped
configuration is a local optimum in its own axis-aligned neighbourhood, and that is a stronger
statement about it than any of the fits produced.

The three deception archetypes are the same story from the other side — **at their struct defaults**.
They are `V04Agent`s whose only departure is the set of half-suits they will ask in, and at the
parameter values every committed artifact in the corpus uses, all three score below the unrestricted
v0.4 they are built from: `feint` −4.41, `withholder` −29.38, `silent` −33.53, against `v04` at −1.33.

**But the defaults are not the family, and the sweep says so in two separate ways.** Neither `tol` nor
`k` has ever been set by a committed artifact in this corpus — which is precisely the "wider deception
space" the phase brief names as unmeasured — and setting them produces one large effect and one dead
knob.

**`tol` matters, monotonically, and the published value is a bad one.** For `feint` the sweep runs
−1.21 at `tol=0.02`, −1.74 at 0.05, −4.21 at 0.10 (the default), −7.96 at 0.20: a **6.75-point range**
across one parameter, with the corpus's published value sitting three points below the family's best.
`withholder` runs −2.44 / −3.19 / −3.94 / −8.81 over the same ladder and `silent` −5.59 / −6.45. The
tolerance is how much hit probability the archetype will sacrifice per deviation, so the curve is
telling us the same thing the rest of this section does: **the deception is not what costs, the
sacrifice is**, and the archetypes were published at a sacrifice three to twenty-eight points larger
than necessary.

**`k` is inert.** `feint:k=1`, `k=3`, `k=6` and `k=12` all measure **−4.21**, bit-identical to
`feint:tol=0.10` and to each other across every column of the table — win rate, target declaration
accuracy, lock hold and events per game. The cooldown parameter does nothing at any setting, which
puts it alongside `patient` and `lockthr` in ledger entry C12's list of knobs that cannot change an
output. Any future ablation on `k` measures nothing.

**What survives is the conclusion, not the reasoning behind it.** The family's best measured arm is
`feint:tol=0.02` at **−1.21 [−2.32, −0.11]**, which reaches parity with the unrestricted `v04` it is
built from (−1.33) rather than falling below it — so the tidy claim that "every restriction costs more
than the corruption it buys" is not right as stated. But −1.21 is still a loss, and the family at its
best is no threat to `v06`. The deception cluster is closed on its size, not on the argument.

### E2. Tally inflation — lying to an outcome-blind prior, and it does not pay

`Knowledge::priorWeight` reads `askCount[p][S]` and `totalAsks[p]` and **nothing else**. `missCount`
exists and no belief quantity reads it. The prior is therefore **outcome-blind**: an ask that misses is
exactly as much evidence that the asker holds cards of that half-suit as an ask that hits. Since the
rules require the asker to hold only *one* card of the half-suit, an adversary holding a single card of
S can ask in S repeatedly and drive the target's marginal for its own team's other five cards of S up
by as much as exp(2.6) = 13.5×, the clip — at a price of one lost turn per repetition. That is the
channel the `feint` archetype gropes at, and phase 2 priced it linearly for the first time with two
coordinates: `selfTally` (the actor's own public tally in this half-suit, which is the only thing the
target's theta term reads) and `tallyLie` (that tally times how little of the half-suit the actor
actually holds — the *size* of the lie the next ask there would tell).

**Both lose**: `r16=25` measures **−2.33** and `r17=25` measures **−1.22**, each replicated on two
banks at 24,000 games. The lie is available, it is cheap to tell, the target believes it, and telling
it costs more than it is worth. Together with §4E and §6 this closes the prior-corruption family from
the coordinate side as well as from the archetype side.

### F. Forced-endgame induction — measured, and it closes ledger entry L13

The ledger keeps L13 open on one condition: incidence is an adversarial variable, and phase 2 should
measure it under pressure before closing the entry. Measured, and the answer has two halves that must
not be run together.

**The incidence can be raised by three orders of magnitude, but on the wrong side of the table.**
`v06:declare=0` — a team that never claims its half-suits voluntarily — produces **4.20 forced
declarations a game** against a mirror baseline of 0.0056. They are made by the `declare=0` team
itself: in the paired cell `forcedPerGameA` = 4.20167 with `forcedAccB` = 0.000000, and the reason is
structural — a team that never claims keeps its cards while the other team empties, so it is the
*other* team that goes cardless and the non-declaring team that must then declare everything. The
ledger's question, whether an adversary can raise the **target's** forced-endgame incidence, is
therefore answered **no** for this construction. The construction that raises the target's is the
adversary going cardless itself, and that was measured at **−17 points**.

**And the machinery it worried about is not broken.** v0.6's forced path, exercised at 4.20
declarations a game instead of 0.0031, resolves correctly **99.3–99.5%** of the time — 23 wrong in
3,396 over 800 games (99.32%), and 0.994863 over 12,000 games in the paired cell. The corpus's 0.286 rests on two to eight
observations per battery and is a small-sample artifact; the v0.5 study's ~40.6% "feasible ceiling"
is a ceiling on a quantity measured in a regime the forced endgame is almost never in. Whichever team
is making them, it is the same `willingForced` ladder.

The adversary pays 6.5 to 7.8 points for the construction. **L13 is closed on both counts**: the
target's incidence is not raisable cheaply, and the mechanism is 99.4% accurate when it runs.

### F2. Partner coordination — attacked, and ledger entry L7 settled

The phase brief names "attack partner coordination" as one of the four structural hypotheses. It has
two sides, and phase 2 measured both.

**The adversary's side** is the A2 ex-ante correlation regime, which phase 1 built and could only
*measure*; the fitter now carries it, so a responder reading the shared signal with `corr=K` is fitted
in the regime it will be scored in rather than transplanted into it.

**The target's side** is ledger entry **L7** — "independent multi-agent search corrupts partners'
beliefs" — whose cheapest decisive experiment the ledger states as *"run the search on one seat of the
team only and compare, paired, to all three"*. It **was** expressible: `--partners` has seated a mixed
A-arm team since v0.6's E5 battery. It had simply never been run, and the ledger records it as
outstanding. It is run here. (`--partnersb`, added this phase, is the complement — a mixed *B*-arm
team — which is what a one-seat deviation column on the target side needs when the adversary occupies
the A arm.)

| arm | seats deviating | edge over `v06` | 95% CI | per bank | n | one-seat share of three |
|---|---:|---:|---|---|---:|---:|
| `search3-cheap` | 3 | +1.91 | [+1.38, +2.44] | +1.58 / +2.24 | 24,000 | &mdash; |
| `search1-cheap` | 1 | +0.39 | [-0.08, +0.85] | +0.48 / +0.30 | 24,000 | 20% |
| `search3-mid` | 3 | +2.78 | [+1.72, +3.84] | +2.77 / +2.80 | 6,000 | &mdash; |
| `search1-mid` | 1 | +1.27 | [+0.32, +2.21] | +1.40 / +1.13 | 6,000 | 46% |
| `contest3` | 3 | +2.77 | [+2.14, +3.39] | +3.00 / +2.53 | 24,000 | &mdash; |
| `contest1` | 1 | +0.54 | [-0.07, +1.15] | +0.80 / +0.28 | 24,000 | 19% |
| `noURG3` | 3 | +1.29 | [+1.03, +1.55] | +1.23 / +1.34 | 24,000 | &mdash; |
| `noURG1` | 1 | +0.70 | [+0.51, +0.89] | +0.58 / +0.82 | 24,000 | 54% |

SPARTA (Lerer, Hu, Foerster & Brown, AAAI 2020) measures two agents each searching while assuming the
partner plays the blueprint and reports 22.99 → 14.41, a 37% relative collapse; three agents is
strictly worse on that axis, because each deviation corrupts two partners. **If Fish shared that
pathology, one searching seat would be worth *more* than a third of three.** It is worth
**20%** — below linear, not above it. Whatever Fish's public-action structure does, it does not
reproduce the Hanabi collapse: three searching seats are worth more than three times one, not less.
**L7 closes as the negative the ledger itself called "genuinely interesting"**, and the reason it
gave for expecting that is the one that holds: Fish's belief is dominated by hard certificates rather
than by a soft partner model, so a partner's deviation does not corrupt what the others know.

**And the same column is the threat model's mandatory T2 report**, which had never been produced for
any adversary. It ranks the arms differently from the three-seat column, which is exactly why T2
requires it:

| arm | k = 3 | k = 1 | k=1 as a share of k=3 | reading |
|---|---:|---:|---:|---|
| the target's own search | +1.91 | +0.39 | **20%** | super-additive across seats: three searchers are worth more than three times one |
| contestation (`r12=25`) | +2.77 | +0.54 | **19%** | the same shape — the half-suit race compounds when all three seats contest |
| the urgency escalation off | +1.29 | +0.70 | **54%** | sub-additive: a declaration is a per-seat decision, and one seat declaring well captures most of it |

n = 12,000 games a cell on two banks. The two mechanisms that compound across the team are the two
that operate on the *ask* side; the one that is nearly per-seat operates on the declaration side.
That is a coherent picture and it is the first time the corpus has had one.

### G. The white-box class, the in-class refit, and the extended class

These are the four classes phase 1 built, re-measured against the *unhandicapped* incumbent at four
times phase 1's evaluation power. They are the continuity column: what a like-for-like exploitability
probe reaches when it is correctly specified.

| class | phase 1, 24,000 games | phase 2, this battery | n | what it is |
|---|---|---|---:|---|
| C1 | +0.76 [+0.15, +1.37] | +0.79 [+0.48, +1.10] | 96,000 | the incumbent's own 37-coordinate family, refit and seeded at the incumbent |
| C2 | +1.05 [+0.43, +1.66] | &mdash; (not re-run) | &mdash; | C1 plus the v0.7 responder coordinates |
| C3 | +1.86 [+0.78, +2.94] | &mdash; (not re-run) | &mdash; | the v0.6 rollout with the target's policy in the rollout's opposing seats |
| C5 | +1.52 [+0.92, +2.13] | &mdash; (not re-run) | &mdash; | the incumbent with its posterior sharpened by inverting the target's transcript |

### G2. What the fitted searches reached

| id | class | fitted against | objective | budget (games) | edge vs `v06` | 95% CI | per bank | n | clears its class floor |
|---|---|---|---|---:|---:|---|---|---:|:--:|
| X18 | C1 | `Fcheap` | `win` | 23,040 | +0.98 | [+0.54, +1.43] | +0.90 / +1.07 | 48,000 | no (1.53) |
| X01 | C1 | `Ffast` | `win` | 96,000 | +0.98 | [+0.54, +1.42] | +0.88 / +1.08 | 48,000 | no (1.53) |
| X14 | C2 | `Ffast` | `win` | 28,800 | +0.70 | [+0.26, +1.15] | +0.85 / +0.56 | 48,000 | no (2.31) |
| X13 | C2 | `Ffast` | `win` | 28,800 | +0.56 | [+0.13, +1.00] | +0.67 / +0.46 | 48,000 | no (2.31) |
| X19 | C2 | `Fcheap` | `win` | 23,040 | -0.89 | [-1.33, -0.46] | -0.78 / -1.00 | 48,000 | no (2.31) |
| X05 | C1 | `Ffast` | `declerr` | 28,800 | -8.42 | [-8.85, -7.99] | -8.23 / -8.61 | 48,000 | no (1.53) |
| X20 | C1 | `Fcheap` | `declerr` | 23,040 | -11.53 | [-11.96, -11.11] | -11.65 / -11.42 | 48,000 | no (1.53) |

**Not one of them clears its class floor.** The best two reach **+0.98** — and they reach it
independently: `X01`, fitted against the deployed policy at phase 1's standard 96,000-game budget, and
`X18`, fitted against the *cheap search* at a quarter of that budget, land on the same number to two
decimals from different targets, different budgets and different banks. That is the same place phase
1's C1 reached (+0.76) and the same place this phase's floor-battery control rung reached (+0.79) at
four times the power. **Four independent routes, one answer: a correctly-specified in-class responder
gets about a point out of `v06` and no more.**

**The mechanism-objective rows are the interesting ones.** `X05` and `X20` were fitted to maximise the
*target's* misdeclaration rate rather than their own win rate, and they score **−8.42** and **−11.53**
in games. That is not a failure of the objective; it is the objective doing its job as a diagnostic.
It says the conversion the ledger assumes — drive the target's declarations wrong and the wins follow
— does not survive being pursued single-mindedly: a policy that sacrifices everything to raise the
target's error rate loses far more than it extracts. Read with §3's ceiling of
+0.38 on the same channel and with §4A's per-decision finding that the phase's
strongest arm makes the target declare *better*, three independent measurements agree: **the
declaration channel is not where an adversary's points are**, which is the opposite of where this
phase started.

### H. The one-switch defects — v0.6 is not at a local optimum on its own declaration channel

The unfitted screen was built to find adversaries. It found something else as well, and it is the most
useful by-product of the phase: **four separate one-switch deviations of the incumbent each beat the
incumbent, and every one of them lowers the incumbent's own misdeclaration rate.**

| switch | edge over `v06` | 95% CI | mirror misdecl. rate | what it turns off |
|---|---:|---|---:|---|
| — | +0.00 | mirror | 2.556% | the shipped policy |
| `xf=0` | +1.20 | [+0.29, +2.11] | 2.39%\* | v0.6's three extra ask terms — **and, as ledger C4 records, this silently restores the v0.5 chain pass, so it is confounded** |
| `m1=0` | +1.11 | [+0.28, +1.93] | 2.44%\* | the live-ask gate |
| `rtie=1` | +1.04 | [+0.14, +1.92] | 1.94%\* | the array-order tie-break, replaced by a hash of the public event stream |
| `m2=0` | +0.57 | [+0.43, +0.71] | 2.00%\* | the capacity-feasible allocator's promotion of `pTeam` to `pAlloc` |
| the urgency escalation off | +1.23 to +1.34 | §4C | **1.278%** | the whole `urgent` disjunction |
| `rtie=1` + urgency off | +1.91 | [+1.29, +2.54] | **1.028%** | both |

The incumbent and the last two rows are on one common base (400 deals x 2, seed 31,
`P0-gate2.txt`). The four starred rows come from an earlier 200-deal digest on the same seed, where
the incumbent measures 2.94%; the incumbent's own mirror misdeclaration rate moves 0.39 points
between the two sample sizes, which is the same order as the reductions the starred rows claim, so
those four are indicative and the two unstarred ones are the measured comparison.

All five pass the commit gate: `xf=0` and `rtie=1` play **zero** provably-dead asks, and `m1=0` plays
1.89% with a longest run of 2 and no game killed by the action limit — nothing like the 46.28% that
removing M1 produces at v0.5's weights, which is the measured form of phase 1's finding that *M1 is
load-bearing for v0.5 and very nearly inert for v0.6*. Here it is slightly worse than inert.

Two of these are the corpus's own ablation nulls, now separated. `F1-chain2x2.json` reports `rtie=1` at
a paired `deltaFromRef` of −0.69 [−2.54, +1.13] and the three extra ask terms at −0.46 [−1.15, +0.23],
both read as "worth nothing"; ledger entry P-2 notes that "not separated at ±1.8 points" and "worth
exactly nothing" license very different conclusions. At 12,000 games with intervals excluding zero,
both are small positives.

**Do they stack? Yes — and the commit gate catches the best-scoring combination.**

| configuration | edge over `v06` | 95% CI | per bank | n | its own decl. acc. | its own ask acc. |
|---|---:|---|---|---:|---:|---:|
| `v06:rtie=1,m1=0,pool=-1,oppfloor=-1,force=1000000,askfloor=-1` | +2.68 | [+2.05, +3.30] | +2.44 / +2.92 | 24,000 | 0.9908 | 0.5375 |
| `v06:rtie=1,m1=0,m2=0,pool=-1,oppfloor=-1,force=1000000,askfloor=-1` | +2.68 | [+2.05, +3.30] | +2.44 / +2.92 | 24,000 | 0.9908 | 0.5375 |
| `v06:rtie=1,m1=0,xf=0,pool=-1,oppfloor=-1,force=1000000,askfloor=-1` | +2.07 | [+1.44, +2.70] | +2.38 / +1.77 | 24,000 | 0.9890 | 0.5411 |
| `v06:rtie=1,pool=-1,oppfloor=-1,force=1000000,askfloor=-1` | +1.91 | [+1.29, +2.54] | +1.74 / +2.08 | 24,000 | 0.9914 | 0.5406 |
| `v06:rtie=1,m1=0` | +1.67 | [+1.04, +2.29] | +1.44 / +1.89 | 24,000 | 0.9810 | 0.5374 |
| `v06:pool=-1,oppfloor=-1,force=1000000,askfloor=-1` | +1.29 | [+1.03, +1.55] | +1.23 / +1.34 | 24,000 | 0.9907 | 0.5428 |
| `v06:rtie=1` | +1.14 | [+0.52, +1.77] | +1.04 / +1.24 | 24,000 | 0.9808 | 0.5408 |
| `v06:xf=0,rtie=1` | +0.92 | [+0.29, +1.56] | +1.20 / +0.65 | 24,000 | 0.9792 | 0.5458 |
| `v06:m1=0` | +0.76 | [+0.18, +1.34] | +1.11 / +0.41 | 24,000 | 0.9787 | 0.5386 |
| `v06` | +0.00 | mirror &mdash; no interval | +0.00 / +0.00 | 24,000 | 0.9785 | 0.5426 |

Two configurations matter here and the ordering between them is the point.

`v06:rtie=1,m1=0,pool=-1,oppfloor=-1,force=1000000,askfloor=-1` scores **+2.68 [+2.05, +3.30]**,
replicated (+2.44 / +2.92), with its own declaration accuracy at **0.9908** against the incumbent's
0.9785. It is the highest-scoring *target-side switch stack* in the phase — three arms in §5 score higher — and it **fails the commit gate**: 2.91%
provably-dead asks against the incumbent's 0.012%, a **326-ask dead run**, 0.33% of games killed by
the action limit, and a game-length tail reaching **405 events** against the incumbent's 131. Ledger
entry **C14** is the same shape — "M8-alone scores 56.60% against v0.4 while carrying 44.83% dead
asks, a 373-ask dead run, and 14% of games killed by the action limit" — and the corpus's standing
rule, that the gate runs before any strength number, is what stops it being reported as the headline.
`m1=0` is the offending component: without it the same stack has a longest dead run of 1 and a maximum
game length of 129.

`v06:rtie=1,pool=-1,oppfloor=-1,force=1000000,askfloor=-1` scores **+1.91 [+1.29, +2.54]**, replicated
(+1.74 / +2.08), cuts the mirror misdeclaration rate from 2.37% to **1.11%**, and passes the gate
cleanly: 0.0203% dead asks, longest dead run 1, **zero** action-limit games. That is
the configuration phase 3 should inherit. On the criterion this document applies everywhere else — the
*lower bound* must exceed the floor — **+1.91 [+1.29, +2.54] does not clear the C1 floor of 1.53**;
its point estimate does and its interval does not. It is reported as a candidate on that basis and not
as a certified effect.

Adding `m2=0` to either changes nothing — bit-identical, for the reason in §4C. Adding `xf=0` makes
the stack *worse* (+2.07), which is a useful negative given that `xf=0` is the largest single switch.

**One warning that came out of the gate rather than out of the strength table.** The `m1=0` stack's
game-length tail reaches 405 events, past the 220 rung at which the pressure escalation fires. In this
particular configuration that is harmless, because `force=1000000` disables the escalation and the
cliff along with it — but a phase-3 candidate that lengthens games *without* disabling the escalation
walks toward its own fifteen-point cliff (§3). `eventsPerGame` and its p99 belong in every candidate's
gate.

The mechanism controls for cluster A are in the same register and are printed here so the two can be
read together:

| control | edge vs `v06` | 95% CI | n | adversary ask acc. | adversary decl./game | adversary decl. acc. | adversary lock hold | target lock hold |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `v06` | +0.00 | [-1.55, +1.55] | 4,000 | 0.5426 | 4.49 | 0.9778 | 4.54 | 4.54 |
| `v07:r12=25` | +3.43 | [+1.91, +4.94] | 4,000 | 0.5336 | 4.69 | 0.9643 | 3.18 | 5.25 |
| `v07:r2=25` | -1.30 | [-2.86, +0.26] | 4,000 | 0.5572 | 4.55 | 0.9638 | 4.18 | 5.17 |
| `v07:r2=10` | -0.77 | [-2.22, +0.68] | 4,000 | 0.5505 | 4.47 | 0.9786 | 5.04 | 4.53 |
| `v06:w0=8` | +0.40 | [-1.02, +1.82] | 4,000 | 0.5335 | 4.49 | 0.9789 | 4.14 | 4.63 |
| `v06:w0=6` | -0.62 | [-2.09, +0.84] | 4,000 | 0.5257 | 4.45 | 0.9789 | 3.98 | 4.59 |
| `v06:w0=4` | -1.97 | [-3.46, -0.49] | 4,000 | 0.5138 | 4.41 | 0.9793 | 3.94 | 4.54 |
| `v07:r12=25,r2=-25` | -11.50 | [-13.00, -10.00] | 4,000 | 0.4684 | 4.00 | 0.9753 | 4.29 | 4.22 |

---

## 5. How much of the frontier's measured exploitability each cluster accounts for

**The decomposition is not a partition, and saying so is part of the answer.** Exploitability under
E1 is a maximum over adversaries, not a sum over mechanisms, so "cluster C accounts for x%" can only
mean "the best arm in C reaches x% of the best arm anywhere". Two clusters can each account for most
of it and still be the same three points found twice. The only evidence about composition is the
crossed arms, which is why phase 2 ran them.

| cluster | best arm | edge vs `v06` | 95% CI | share of the best measured | replicated in sign |
|---|---|---:|---|---:|:--:|
| contestation-x-m2-x-search | `v07:m2=0,r12=25,s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26` | +4.40 | [+3.95, +4.84] | 100% | yes |
| contestation-x-search | `v07:r12=25,s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26` | +3.99 | [+3.54, +4.43] | 91% | yes |
| cross-inclass-search | `v06:allparams=9.60189|6.15330|3.83683|3.12220|6.65640|6.64437|1.29917|-0.63211|-2.07810|-4.73361|-3.36019|1.69692|2.23122|5.94852|0.89676|0.11936|5.11158|-0.59790|1.81511|-1.21220|0.86813|0.74469|0.29558|5.91993|2.55634|6.95324|0.68312|0.87672|-0.03021|0.40022|0.19007|5.73206|3.57881|3.69737|0.34414|-1.30664|0.01140,s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26,roppo=v06` | +3.58 | [+3.15, +4.01] | 81% | yes |
| contestation-x-m2 | `v07:m2=0,r12=25` | +3.17 | [+2.73, +3.61] | 72% | yes |
| cross-inclass-whitebox | `v07i:allparams=9.60189|6.15330|3.83683|3.12220|6.65640|6.64437|1.29917|-0.63211|-2.07810|-4.73361|-3.36019|1.69692|2.23122|5.94852|0.89676|0.11936|5.11158|-0.59790|1.81511|-1.21220|0.86813|0.74469|0.29558|5.91993|2.55634|6.95324|0.68312|0.87672|-0.03021|0.40022|0.19007|5.73206|3.57881|3.69737|0.34414|-1.30664|0.01140,idet=48,imodel=v06` | +2.87 | [+2.44, +3.31] | 65% | yes |
| contestation | `v07:r12=25` | +2.71 | [+2.27, +3.15] | 62% | yes |
| m2-x-search | `v06:m2=0,s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26` | +2.48 | [+2.11, +2.85] | 56% | yes |
| search-strength | `v06:s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26` | +1.89 | [+1.51, +2.26] | 43% | yes |
| consensus-median | `v06:allparams=10.65701|4.49654|2.81921|3.18930|4.52070|6.64437|1.94891|-0.63211|-2.07810|-7.03545|-3.36019|1.69692|2.23122|3.46434|0.89676|-0.81971|5.11158|-0.59790|1.81511|-1.21220|0.85372|0.74469|0.29558|5.41076|3.02993|6.95324|0.74807|0.86254|-0.02900|0.40022|0.18982|5.73206|3.15604|3.85802|0.32859|-0.35205|-0.54433` | +1.15 | [+0.72, +1.57] | 26% | yes |
| contestation-group | `v07:r12=20,r13=10,r14=5,r15=5` | +1.01 | [+0.56, +1.45] | 23% | yes |
| anti-search | `v06:allparams=10.88642|5.45148|3.08622|2.60621|1.23358|6.71067|1.33987|0.10283|-0.07741|-5.89914|-4.22573|2.19085|0.67779|3.42813|2.83997|-0.73452|6.80792|1.70377|1.81262|-5.88808|0.84584|0.74827|0.24368|4.15674|3.39203|5.50852|0.75888|0.81736|-0.02002|0.38967|0.15473|6.10077|2.79653|4.31269|0.63762|-0.47487|-0.73261` | +0.98 | [+0.54, +1.43] | 22% | yes |
| inclass-linear | `v06:allparams=9.60189|6.15330|3.83683|3.12220|6.65640|6.64437|1.29917|-0.63211|-2.07810|-4.73361|-3.36019|1.69692|2.23122|5.94852|0.89676|0.11936|5.11158|-0.59790|1.81511|-1.21220|0.86813|0.74469|0.29558|5.91993|2.55634|6.95324|0.68312|0.87672|-0.03021|0.40022|0.19007|5.73206|3.57881|3.69737|0.34414|-1.30664|0.01140` | +0.98 | [+0.54, +1.42] | 22% | yes |
| correlated-roles | `v07:corr=3,allparams=10.93235|3.84246|5.39019|3.44998|5.17941|7.80133|1.67007|-2.85106|-1.51045|-5.44792|-4.13674|0.72539|2.94226|4.69699|0.77276|-1.58619|4.68915|0.75792|2.01566|-0.34524|0.83417|0.75717|0.23269|4.65304|2.50029|7.23053|0.71549|0.84603|-0.02713|0.33723|0.14526|6.01071|3.13828|3.47756|0.71380|-0.65792|-0.64675|1.13249|0.84997|-0.02595|-0.97772|0.77239|0.64914|-1.32846|0.10564|1.33920|0.42725|-0.39210|0.53344|-0.49065|0.18453|0.40350|0.80095|-0.11058|-0.17790` | +0.70 | [+0.26, +1.15] | 16% | yes |
| m2-defect | `v06:m2=0` | +0.70 | [+0.62, +0.78] | 16% | yes |
| deadask-turnrouting | `v07:dead7=1,allparams=12.34003|5.77698|3.01280|3.15879|2.18166|6.93533|0.50231|-0.41603|-2.92129|-5.29695|-1.54504|1.21655|2.12483|4.67018|0.43247|-2.38785|4.18155|1.01245|1.61377|-1.50855|0.85344|0.74102|0.29524|5.64691|3.57890|8.97650|0.75725|0.81061|-0.01047|0.37589|0.16957|6.08466|3.50400|3.70347|0.71649|-0.73436|-1.76790|-1.51966|-0.33821|-0.29468|-0.52715|0.56908|-0.16538|0.01109|-1.18642|-1.61075|-0.23280|1.56231|-1.77070|0.61949|-1.43323|0.71314|-0.13769|-0.87270|0.82603` | +0.56 | [+0.13, +1.00] | 13% | yes |
| reference | `v06` | +0.00 | [-0.45, +0.45] | 0% | yes |
| tally-inflation | `v07:r17=25` | -1.22 | [-1.52, -0.92] | -28% | yes |
| urgency-declaration | `v06:allparams=10.89837|3.57354|-0.11033|6.08470|4.52070|12.53318|3.04417|1.97172|-1.84251|-10.58477|2.01230|0.46190|3.88880|3.01613|0.32374|-4.31450|7.55703|-1.39547|0.49575|-1.17613|0.85372|0.76187|0.30551|2.77896|4.32900|2.06702|0.74807|0.85416|-0.02585|0.29919|0.18982|4.54382|2.95441|3.85802|0.00000|-0.16728|-1.53080` | -8.42 | [-8.85, -7.99] | -191% | yes |

**And the frontier-dominance rule bites hard.** Severity leads with the *worst* cell over the
frontier, because an arm that beats the deployed policy and loses to the search has beaten an
operating point rather than the frontier. On that reading the contestation coordinate alone falls from
+2.71 against `v06` to **+0.55** against `F-cheap` — where the two banks disagree in sign — and to
**−0.99** against `F-mid`, negative on both banks. **The phase's strongest single arm loses to the
frontier's strongest measured configuration.**

The composite does not. It holds **+4.40 / +2.40 / +2.21** across the three measured frontier points,
positive on every point and on every bank, for a worst cell of **+2.21**. That is the one configuration
in this phase that **dominates the frontier** rather than an operating point of it — and it is built
from the frontier's own search plus one new linear coordinate plus a switch, so it belongs to phase 3
as a candidate and not to the threat model as exploitability. The arithmetic is almost exactly additive across the frontier (`F-cheap` is +1.89
over `v06`; 2.71 − 1.89 = 0.82 against a measured +0.55, and 4.40 − 1.89 = 2.51 against a measured
+2.40), which says the coordinate and the search are **largely independent mechanisms** rather than
two routes to the same margin. It also says the only thing in this phase that dominates the frontier
is a configuration built out of the frontier's own search plus a strength coordinate — a candidate,
not an exploit, which is the verdict again by a different route.

Read it as follows.

* **Neither of the two arms that beat the deployed policy is an exploit, so the honest share table has
  no entries.** The target's own search reaches +1.89 and half-suit contestation
  reaches +2.71, and §4A and §4B establish that both are strength gains: the first by
  construction (it is the target's own search, which the deployed configuration ships switched off for
  cost) and the second by measurement (its gain survives against `detective`, which shares none of
  v0.6's declaration machinery, once the win-rate scale is corrected). **The measured exploitability of
  `v06` under E1 is therefore the in-class figure of +0.79 [+0.48, +1.10], and it is below the
  class's detection floor of 1.53.**
* Everything else the corpus has ever built — the extended class, the white-box inverter, the whole
  deception family, the tally-inflation coordinates, the C6 adaptive class — sits below both, and most
  of it sits below zero.
* **Composition is real, substantial and sub-additive.** The three mechanisms were measured alone, in
  pairs and together, on two banks at 48,000 games a cell:

  | configuration | edge over `v06` | naive sum of parts | shortfall |
  |---|---:|---:|---:|
  | `search` alone | +1.89 [+1.51, +2.26] | — | — |
  | `r12=25` alone | +2.71 [+2.27, +3.15] | — | — |
  | `m2=0` alone | +0.70 [+0.62, +0.78] | — | — |
  | `search` + `r12` | +3.99 [+3.54, +4.43] | +4.60 | −0.61 |
  | `search` + `m2` | +2.48 [+2.11, +2.85] | +2.59 | −0.11 |
  | `m2` + `r12` | +3.17 [+2.73, +3.61] | +3.41 | −0.24 |
  | **all three** | **+4.40 [+3.95, +4.84]** | +5.30 | −0.90 |

  Three mechanisms built independently, one of them the target's own machinery and one of them a
  coordinate that did not exist a day ago, compose to **+4.40** over the deployed policy —
  83% of their naive sum. They are competing for the same margin, the half-suit race, rather than
  opening separate ones, but the competition costs less than half a point.

  **The composite passes the commit gate**: zero provably-dead asks, longest dead run zero, no game
  killed by the action limit, maximum game length 123 events. What it does carry is an accepted risk
  trade rather than a soundness failure — its *mirror* misdeclaration rate is 3.63% against the
  incumbent's 2.37%, and its self-play forced-endgame rate rises from zero. It wins by tempo while
  declaring worse, and against itself that costs it. Phase 3 inherits both halves of that.
* **The clusters that account for nothing account for nothing decisively.** Urgency induction has a
  measured ceiling of +0.38; the action cap is unreachable — every stalling probe at
  the shipped cap returns a limit-hit rate of exactly 0.0000; the deception family is uniformly
  negative; forced-endgame induction costs the adversary six to eight points. These are not "not
  found" — they are measured at or below zero with intervals, which is what makes them usable by
  phase 3 as closed directions rather than as unexplored ones.

---

## 6. The unfitted screen, in full

Every arm below is three identical copies of the named spec seated against the frozen target on
registered evaluation banks, with no fitting of any kind. The reference row is the mirror, which
carries no information by construction and is printed as such.

| adversary | cluster | edge vs `v06` (pts) | 95% CI | n | 98/&radic;N | target decl. acc. | target ask acc. | target decl./game | limit hits | events/game |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `v06:s1=1,det=16,cand=6,kappa=2.0,maxq=26` | search | +2.47 | [+1.72, +3.22] | 12,000 | 0.89 | 0.9780 | 0.5335 | 4.43 | 0.0000 | 95.5 |
| `v06:s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=24,maxq=26` | search | +1.63 | [+0.89, +2.37] | 12,000 | 0.89 | 0.9784 | 0.5378 | 4.44 | 0.0000 | 95.5 |
| `v06:s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26` | search | +1.58 | [+0.83, +2.33] | 12,000 | 0.89 | 0.9784 | 0.5381 | 4.46 | 0.0000 | 95.4 |
| `v06:xf=0` | ask | +1.20 | [+0.29, +2.11] | 12,000 | 0.89 | 0.9829 | 0.5425 | 4.45 | 0.0000 | 96.0 |
| `v06:m1=0` | ask | +1.11 | [+0.28, +1.93] | 12,000 | 0.89 | 0.9791 | 0.5371 | 4.47 | 0.0000 | 95.6 |
| `v06:rtie=1` | ask | +1.04 | [+0.15, +1.93] | 12,000 | 0.89 | 0.9811 | 0.5387 | 4.47 | 0.0000 | 95.3 |
| `v07i:idet=48,imodel=v06` | invert | +0.59 | [-0.25, +1.43] | 12,000 | 0.89 | 0.9801 | 0.5414 | 4.45 | 0.0000 | 94.6 |
| `v06:m2=0` | decl | +0.57 | [+0.43, +0.70] | 12,000 | 0.89 | 0.9785 | 0.5426 | 4.50 | 0.0000 | 95.2 |
| `v06:chain2=1` | ask | +0.30 | [-0.55, +1.15] | 12,000 | 0.89 | 0.9807 | 0.5438 | 4.47 | 0.0000 | 95.4 |
| `v06:vmargin=-0.02` | decl | +0.02 | [-0.09, +0.14] | 12,000 | 0.89 | 0.9785 | 0.5424 | 4.49 | 0.0000 | 95.2 |
| `v06` | reference | +0.00 | mirror &mdash; no interval | 12,000 | 0.89 | 0.9785 | 0.5426 | 4.49 | 0.0000 | 95.2 |
| `v06:gmap=1` | decl | +0.00 | [-0.89, +0.89] | 12,000 | 0.89 | 0.9785 | 0.5426 | 4.49 | 0.0000 | 95.2 |
| `v06:m1p=0` | ask | +0.00 | [-0.89, +0.89] | 12,000 | 0.89 | 0.9785 | 0.5426 | 4.49 | 0.0000 | 95.2 |
| `v06:dead=1` | ask | +0.00 | [-0.08, +0.08] | 12,000 | 0.89 | 0.9785 | 0.5425 | 4.49 | 0.0000 | 95.2 |
| `v06:dead=1,deadbudget=3` | ask | +0.00 | [-0.08, +0.08] | 12,000 | 0.89 | 0.9785 | 0.5425 | 4.49 | 0.0000 | 95.2 |
| `v06:topk=0` | ask | +0.00 | [-0.89, +0.89] | 12,000 | 0.89 | 0.9785 | 0.5426 | 4.49 | 0.0000 | 95.2 |
| `v06:dead=1,deadbudget=1` | ask | -0.01 | [-0.09, +0.08] | 12,000 | 0.89 | 0.9785 | 0.5425 | 4.49 | 0.0000 | 95.2 |
| `v06:oppfloor=0` | ask | -0.02 | [-0.06, +0.01] | 12,000 | 0.89 | 0.9785 | 0.5426 | 4.49 | 0.0000 | 95.2 |
| `v06:vdecl=0` | decl | -0.10 | [-0.86, +0.66] | 12,000 | 0.89 | 0.9795 | 0.5276 | 4.56 | 0.0000 | 97.5 |
| `v06:askfloor=0.3` | ask | -0.12 | [-0.28, +0.03] | 12,000 | 0.89 | 0.9784 | 0.5423 | 4.49 | 0.0000 | 95.2 |
| `v06:vmargin=-0.01` | decl | -0.16 | [-0.55, +0.24] | 12,000 | 0.89 | 0.9784 | 0.5410 | 4.50 | 0.0000 | 95.4 |
| `v06:decl=0.90` | decl | -0.30 | [-0.55, -0.05] | 12,000 | 0.89 | 0.9791 | 0.5421 | 4.50 | 0.0000 | 95.3 |
| `v06:value=0` | ask | -0.33 | [-1.12, +0.45] | 12,000 | 0.89 | 0.9797 | 0.5262 | 4.55 | 0.0000 | 97.5 |
| `v06:minteam=0.20` | decl | -0.42 | [-0.70, -0.14] | 12,000 | 0.89 | 0.9783 | 0.5424 | 4.48 | 0.0000 | 95.0 |
| `v06:minteam=0.70` | decl | -0.42 | [-0.70, -0.14] | 12,000 | 0.89 | 0.9783 | 0.5424 | 4.48 | 0.0000 | 95.0 |
| `v06:decl=0.99` | decl | -0.71 | [-1.00, -0.41] | 12,000 | 0.89 | 0.9794 | 0.5417 | 4.50 | 0.0000 | 95.3 |
| `v06:vmargin=0.01` | decl | -0.82 | [-1.39, -0.24] | 12,000 | 0.89 | 0.9785 | 0.5371 | 4.52 | 0.0000 | 96.0 |
| `v06:ptheta=0.2` | prior | -0.84 | [-1.67, -0.01] | 12,000 | 0.89 | 0.9791 | 0.5369 | 4.52 | 0.0000 | 95.8 |
| `v06:pphi=0` | prior | -0.94 | [-1.74, -0.14] | 12,000 | 0.89 | 0.9787 | 0.5375 | 4.53 | 0.0000 | 95.8 |
| `v04` | reference | -1.33 | [-2.23, -0.43] | 12,000 | 0.89 | 0.9826 | 0.5340 | 4.52 | 0.0000 | 99.6 |
| `v06:pphi=0.4` | prior | -1.34 | [-2.19, -0.50] | 12,000 | 0.89 | 0.9795 | 0.5440 | 4.48 | 0.0000 | 94.8 |
| `v06:vmargin=0.03` | decl | -1.37 | [-2.09, -0.66] | 12,000 | 0.89 | 0.9785 | 0.5282 | 4.54 | 0.0000 | 97.3 |
| `v05` | reference | -1.84 | [-2.72, -0.96] | 12,000 | 0.89 | 0.9848 | 0.5505 | 4.51 | 0.0000 | 96.0 |
| `v06:ptheta=0` | prior | -2.68 | [-3.52, -1.85] | 12,000 | 0.89 | 0.9788 | 0.5355 | 4.57 | 0.0000 | 96.6 |
| `feint` | archetype | -4.41 | [-5.30, -3.52] | 12,000 | 0.89 | 0.9832 | 0.5513 | 4.52 | 0.0000 | 104.6 |
| `v06:ptheta=0.9` | prior | -4.93 | [-5.80, -4.07] | 12,000 | 0.89 | 0.9781 | 0.5506 | 4.41 | 0.0000 | 92.9 |
| `v06:norepeat=1` | ask | -6.74 | [-7.38, -6.11] | 12,000 | 0.89 | 0.9789 | 0.5506 | 4.74 | 0.0000 | 95.7 |
| `v06:ptheta=0,pphi=0` | prior | -6.85 | [-7.70, -6.00] | 12,000 | 0.89 | 0.9793 | 0.5391 | 4.71 | 0.0000 | 97.2 |
| `v06:declare=0` | decl | -7.79 | [-8.61, -6.98] | 12,000 | 0.89 | 0.9858 | 0.4824 | 4.80 | 0.0000 | 104.5 |
| `v06:belief=block` | belief | -7.92 | [-8.80, -7.05] | 12,000 | 0.89 | 0.9714 | 0.5276 | 4.78 | 0.0000 | 97.3 |
| `v06:ptheta=1.5` | prior | -9.62 | [-10.47, -8.78] | 12,000 | 0.89 | 0.9780 | 0.5523 | 4.38 | 0.0000 | 91.8 |
| `v06:belief=exactdisj` | belief | -17.08 | [-17.92, -16.25] | 12,000 | 0.89 | 0.9787 | 0.5197 | 5.06 | 0.0000 | 100.3 |
| `v03` | archetype | -26.15 | [-26.91, -25.39] | 12,000 | 0.89 | 0.9811 | 0.5679 | 4.87 | 0.0000 | 97.4 |
| `v06:belief=hybrid` | belief | -26.23 | [-27.00, -25.47] | 12,000 | 0.89 | 0.9719 | 0.5152 | 5.42 | 0.0000 | 102.6 |
| `detective` | archetype | -27.57 | [-28.32, -26.83] | 12,000 | 0.89 | 0.9864 | 0.5640 | 5.18 | 0.0000 | 100.4 |
| `lockout` | archetype | -28.41 | [-29.14, -27.67] | 12,000 | 0.89 | 0.9862 | 0.5546 | 5.21 | 0.0000 | 101.6 |
| `v06:belief=sinkhorn` | belief | -29.33 | [-30.05, -28.60] | 12,000 | 0.89 | 0.9708 | 0.5179 | 5.33 | 0.0000 | 101.5 |
| `withholder` | archetype | -29.38 | [-30.11, -28.66] | 12,000 | 0.89 | 0.9645 | 0.5074 | 5.76 | 0.0000 | 94.1 |
| `v02` | archetype | -31.20 | [-31.89, -30.51] | 12,000 | 0.89 | 0.9863 | 0.5590 | 5.20 | 0.0000 | 101.4 |
| `silent` | archetype | -33.53 | [-34.20, -32.86] | 12,000 | 0.89 | 0.9765 | 0.5919 | 5.86 | 0.0000 | 105.6 |
| `v06:belief=indep` | belief | -33.99 | [-34.64, -33.34] | 12,000 | 0.89 | 0.9758 | 0.5160 | 5.51 | 0.0000 | 105.7 |
| `diversifier` | archetype | -44.89 | [-45.29, -44.50] | 12,000 | 0.89 | 0.9681 | 0.6359 | 6.39 | 0.0000 | 106.4 |
| `hunter` | archetype | -48.33 | [-48.55, -48.10] | 12,000 | 0.89 | 0.9684 | 0.5417 | 5.45 | 0.0000 | 90.6 |
| `bluffer` | archetype | -49.94 | [-49.99, -49.90] | 12,000 | 0.89 | 0.9729 | 0.5438 | 3.98 | 0.0000 | 84.8 |

> **80 arms landed.**  The battery was **still running** when this was built; re-running the generator after it finishes splices in the rest.

| adversary | cluster | edge vs `v06` | 95% CI | n | 98/&radic;N | target decl. acc. | target lock hold | events/game |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `v06:allparams=11.01959|3.76864|1.26247|2.02293|5.38365|7.35170|2.72276|-1.37625|-3.68164|-5.39828|-2.30428|-0.94811|3.23905|4.68089|-1.66255|-1.57374|5.66882|0.16416|0.50099|-2.35415|0.83516|0.74019|0.26813|6.09034|1.54718|7.77694|0.58685|0.77592|0.00473|0.53508|0.14616|5.46498|3.45467|3.67914|0.24753|-0.71832|-0.75982,s1=1,det=16,cand=6,kappa=2.0,maxq=26,roppo=v06` | c3 | +4.50 | [+3.84, +5.17] | 20,000 | 0.69 | 0.9768 | 4.73 | 97.0 |
| `v06:s1=1,det=16,cand=6,kappa=2.0,maxq=26,roppo=v06` | c3 | +3.53 | [+2.61, +4.44] | 8,000 | 1.10 | 0.9779 | 4.75 | 95.8 |
| `v06:allparams=11.01959|3.76864|1.26247|2.02293|5.38365|7.35170|2.72276|-1.37625|-3.68164|-5.39828|-2.30428|-0.94811|3.23905|4.68089|-1.66255|-1.57374|5.66882|0.16416|0.50099|-2.35415|0.83516|0.74019|0.26813|6.09034|1.54718|7.77694|0.58685|0.77592|0.00473|0.53508|0.14616|5.46498|3.45467|3.67914|0.24753|-0.71832|-0.75982,s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26,roppo=v06` | c3 | +3.38 | [+2.69, +4.06] | 20,000 | 0.69 | 0.9798 | 4.68 | 97.0 |
| `v07i:idet=48,imodel=v06,s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26` | invert | +3.00 | [+1.96, +4.04] | 8,000 | 1.10 | 0.9797 | 4.79 | 95.0 |
| `v06:allparams=11.01959|3.76864|1.26247|2.02293|5.38365|7.35170|2.72276|-1.37625|-3.68164|-5.39828|-2.30428|-0.94811|3.23905|4.68089|-1.66255|-1.57374|5.66882|0.16416|0.50099|-2.35415|0.83516|0.74019|0.26813|6.09034|1.54718|7.77694|0.58685|0.77592|0.00473|0.53508|0.14616|5.46498|3.45467|3.67914|0.24753|-0.71832|-0.75982,s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26` | c3 | +2.62 | [+1.94, +3.30] | 20,000 | 0.69 | 0.9797 | 4.61 | 96.8 |
| `v06:s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26,roppo=v06` | c3 | +2.19 | [+1.60, +2.78] | 20,000 | 0.69 | 0.9801 | 4.72 | 95.6 |
| `v07i:igain=0.5,imodel=v06` | invert | +2.06 | [+1.05, +3.08] | 8,000 | 1.10 | 0.9794 | 4.77 | 94.7 |
| `v06:w3=-2.0,w19=-6.0` | inclass-feint | +1.82 | [+1.13, +2.52] | 20,000 | 0.69 | 0.9822 | 5.30 | 97.4 |
| `v06:w3=1.9,w19=-12.0` | inclass-feint | +1.63 | [+0.95, +2.31] | 20,000 | 0.69 | 0.9816 | 5.35 | 97.1 |
| `v07i:idet=96,imodel=v06` | invert | +1.48 | [+0.46, +2.49] | 8,000 | 1.10 | 0.9802 | 4.60 | 94.6 |
| `v06:w19=-12.0` | inclass-feint | +1.38 | [+0.69, +2.07] | 20,000 | 0.69 | 0.9811 | 5.23 | 97.2 |
| `v06:allparams=11.01959|3.76864|1.26247|2.02293|5.38365|7.35170|2.72276|-1.37625|-3.68164|-5.39828|-2.30428|-0.94811|3.23905|4.68089|-1.66255|-1.57374|5.66882|0.16416|0.50099|-2.35415|0.83516|0.74019|0.26813|6.09034|1.54718|7.77694|0.58685|0.77592|0.00473|0.53508|0.14616|5.46498|3.45467|3.67914|0.24753|-0.71832|-0.75982` | c3 | +1.33 | [+0.66, +2.01] | 20,000 | 0.69 | 0.9781 | 4.60 | 96.4 |
| `v07i:ikappa=1.0,imodel=v06` | invert | +1.20 | [+0.18, +2.22] | 8,000 | 1.10 | 0.9798 | 4.73 | 94.7 |
| `v07i:imaxq=26,imodel=v06` | invert | +1.15 | [+0.34, +1.96] | 8,000 | 1.10 | 0.9796 | 4.65 | 94.9 |
| `v07i:istep=2.5,imodel=v06` | invert | +1.06 | [+0.03, +2.09] | 8,000 | 1.10 | 0.9809 | 4.70 | 94.8 |
| `v06:s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26,deadsearch=2` | c3 | +1.00 | [+0.07, +1.93] | 8,000 | 1.10 | 0.9767 | 4.66 | 95.8 |
| `v07i:idet=24,imodel=v06` | invert | +0.59 | [-0.43, +1.61] | 8,000 | 1.10 | 0.9786 | 4.66 | 94.7 |
| `v06:w3=-4.0` | inclass-feint | +0.49 | [-0.20, +1.19] | 20,000 | 0.69 | 0.9827 | 5.14 | 96.8 |
| `v06:decl=0.70` | urgency | +0.16 | [-0.30, +0.62] | 8,000 | 1.10 | 0.9778 | 4.64 | 94.8 |
| `v07c:mode=0` | c6 | +0.00 | [-1.10, +1.10] | 8,000 | 1.10 | 0.9782 | 4.64 | 95.2 |
| `v06:dead=1,deadbudget=8` | urgency | -0.01 | [-0.11, +0.09] | 8,000 | 1.10 | 0.9783 | 4.65 | 95.2 |
| `v06:dead=1,deadbudget=20` | urgency | -0.01 | [-0.11, +0.09] | 8,000 | 1.10 | 0.9783 | 4.65 | 95.2 |
| `v06:pool=20` | urgency | -0.05 | [-0.36, +0.26] | 8,000 | 1.10 | 0.9782 | 4.64 | 95.2 |
| `v06:w3=-2.0` | inclass-feint | -0.11 | [-0.77, +0.54] | 20,000 | 0.69 | 0.9787 | 4.64 | 95.5 |
| `v07c:mode=2,lockp=0.99,ambig=3` | c6 | -0.12 | [-0.26, +0.01] | 8,000 | 1.10 | 0.9781 | 4.64 | 95.1 |
| `v06:vmargin=-0.05` | urgency | -0.21 | [-0.81, +0.38] | 8,000 | 1.10 | 0.9784 | 4.65 | 94.5 |
| `v07i:igain=2.0,imodel=v06` | invert | -0.24 | [-1.27, +0.79] | 8,000 | 1.10 | 0.9799 | 4.58 | 94.9 |
| `v06:decl=0.95` | urgency | -0.34 | [-0.68, -0.00] | 8,000 | 1.10 | 0.9788 | 4.63 | 95.3 |
| `v06:pool=45` | urgency | -0.35 | [-0.88, +0.18] | 8,000 | 1.10 | 0.9780 | 4.66 | 95.4 |
| `v06:w19=-6.0` | inclass-feint | -0.38 | [-1.02, +0.25] | 20,000 | 0.69 | 0.9784 | 4.60 | 95.3 |
| `v06:decl=0.60` | urgency | -0.45 | [-1.01, +0.11] | 8,000 | 1.10 | 0.9774 | 4.67 | 94.5 |
| `v07i:idet=96,igain=2.0,imodel=v06` | invert | -0.65 | [-1.67, +0.37] | 8,000 | 1.10 | 0.9809 | 4.58 | 94.7 |
| `v07c:mode=1,holdmax=30` | c6 | -0.65 | [-1.58, +0.28] | 8,000 | 1.10 | 0.9771 | 4.70 | 96.2 |
| `v07c:mode=1` | c6 | -1.30 | [-2.33, -0.28] | 8,000 | 1.10 | 0.9785 | 4.83 | 97.7 |
| `v04` | deception | -1.43 | [-2.13, -0.74] | 20,000 | 0.69 | 0.9824 | 4.83 | 99.5 |
| `feint:tol=0.02` | deception | -1.44 | [-2.14, -0.74] | 20,000 | 0.69 | 0.9828 | 6.14 | 103.1 |
| `v07c:mode=2,aggr=-2.0` | c6 | -1.62 | [-2.12, -1.13] | 8,000 | 1.10 | 0.9786 | 4.65 | 94.6 |
| `v07c:mode=2` | c6 | -1.62 | [-2.12, -1.13] | 8,000 | 1.10 | 0.9786 | 4.65 | 94.6 |
| `v06:vmargin=-0.10,dead=1,deadbudget=8` | urgency | -1.77 | [-2.49, -1.06] | 8,000 | 1.10 | 0.9781 | 4.65 | 93.9 |
| `v06:vmargin=-0.10` | urgency | -1.77 | [-2.49, -1.06] | 8,000 | 1.10 | 0.9781 | 4.65 | 93.9 |
| `feint:tol=0.05` | deception | -1.97 | [-2.66, -1.28] | 20,000 | 0.69 | 0.9837 | 6.16 | 103.5 |
| `v06:vmargin=-0.10,decl=0.60` | urgency | -2.12 | [-2.88, -1.37] | 8,000 | 1.10 | 0.9771 | 4.67 | 93.6 |
| `v07c:mode=3,holdmax=30` | c6 | -2.14 | [-3.11, -1.17] | 8,000 | 1.10 | 0.9774 | 4.72 | 95.6 |
| `withholder:tol=0.02` | deception | -2.40 | [-3.09, -1.72] | 20,000 | 0.69 | 0.9824 | 4.82 | 99.6 |
| `withholder:tol=0.05` | deception | -3.17 | [-3.85, -2.49] | 20,000 | 0.69 | 0.9822 | 4.78 | 99.8 |
| `v07c:mode=1,holdmax=120` | c6 | -3.17 | [-4.18, -2.17] | 8,000 | 1.10 | 0.9794 | 4.87 | 99.2 |
| `v07c:mode=2,lockp=0.80,ambig=1` | c6 | -3.28 | [-3.92, -2.63] | 8,000 | 1.10 | 0.9782 | 4.66 | 94.1 |
| `v07c:mode=3` | c6 | -3.43 | [-4.45, -2.40] | 8,000 | 1.10 | 0.9785 | 4.85 | 96.9 |
| `feint:tol=0.10,k=6` | deception | -4.33 | [-5.02, -3.64] | 20,000 | 0.69 | 0.9831 | 6.15 | 104.6 |
| `feint:tol=0.10` | deception | -4.33 | [-5.02, -3.64] | 20,000 | 0.69 | 0.9831 | 6.15 | 104.6 |
| `feint:k=6` | deception | -4.33 | [-5.02, -3.64] | 20,000 | 0.69 | 0.9831 | 6.15 | 104.6 |
| `feint:k=3` | deception | -4.33 | [-5.02, -3.64] | 20,000 | 0.69 | 0.9831 | 6.15 | 104.6 |
| `feint:k=12` | deception | -4.33 | [-5.02, -3.64] | 20,000 | 0.69 | 0.9831 | 6.15 | 104.6 |
| `feint:k=1` | deception | -4.33 | [-5.02, -3.64] | 20,000 | 0.69 | 0.9831 | 6.15 | 104.6 |
| `withholder:tol=0.10,k=6` | deception | -4.45 | [-5.13, -3.77] | 20,000 | 0.69 | 0.9830 | 4.73 | 100.3 |
| `withholder:tol=0.10` | deception | -4.45 | [-5.13, -3.77] | 20,000 | 0.69 | 0.9830 | 4.73 | 100.3 |
| `v06:vmargin=-0.20` | urgency | -4.96 | [-5.74, -4.19] | 8,000 | 1.10 | 0.9781 | 4.68 | 93.4 |
| `v06:vmargin=-0.40` | urgency | -5.08 | [-5.86, -4.29] | 8,000 | 1.10 | 0.9782 | 4.68 | 93.4 |
| `v06:vmargin=-0.20,decl=0.60` | urgency | -5.14 | [-5.95, -4.33] | 8,000 | 1.10 | 0.9772 | 4.70 | 93.2 |
| `silent:tol=0.02` | deception | -5.40 | [-6.08, -4.71] | 20,000 | 0.69 | 0.9848 | 5.05 | 100.0 |
| `silent:tol=0.05` | deception | -6.30 | [-6.99, -5.62] | 20,000 | 0.69 | 0.9843 | 5.05 | 100.1 |
| `v07i:ifocus=1,imodel=v06` | invert | -7.00 | [-8.07, -5.93] | 8,000 | 1.10 | 0.9800 | 4.50 | 92.6 |
| `feint:tol=0.20,k=3` | deception | -8.23 | [-8.91, -7.55] | 20,000 | 0.69 | 0.9825 | 6.18 | 106.2 |
| `feint:tol=0.20` | deception | -8.23 | [-8.91, -7.55] | 20,000 | 0.69 | 0.9825 | 6.18 | 106.2 |
| `silent:tol=0.10` | deception | -9.01 | [-9.69, -8.33] | 20,000 | 0.69 | 0.9840 | 4.99 | 101.1 |
| `withholder:tol=0.20` | deception | -9.07 | [-9.75, -8.39] | 20,000 | 0.69 | 0.9816 | 4.60 | 101.4 |
| `withholder:k=1` | deception | -9.50 | [-10.17, -8.82] | 20,000 | 0.69 | 0.9739 | 4.21 | 97.8 |
| `feint:tol=0.40` | deception | -14.16 | [-14.81, -13.50] | 20,000 | 0.69 | 0.9793 | 6.03 | 108.8 |
| `silent:tol=0.20` | deception | -14.39 | [-15.05, -13.72] | 20,000 | 0.69 | 0.9835 | 4.93 | 103.0 |
| `withholder:tol=0.40` | deception | -14.98 | [-15.63, -14.33] | 20,000 | 0.69 | 0.9784 | 4.32 | 103.5 |
| `withholder:k=3` | deception | -21.51 | [-22.13, -20.89] | 20,000 | 0.69 | 0.9680 | 3.69 | 95.6 |
| `silent:tol=0.40` | deception | -23.26 | [-23.87, -22.65] | 20,000 | 0.69 | 0.9816 | 4.71 | 105.8 |
| `detective:tells=1` | tells | -27.58 | [-28.15, -27.01] | 20,000 | 0.69 | 0.9864 | 4.19 | 100.2 |
| `lockout:tells=1` | tells | -28.88 | [-29.45, -28.32] | 20,000 | 0.69 | 0.9866 | 4.25 | 100.9 |
| `withholder:k=6` | deception | -29.32 | [-29.88, -28.76] | 20,000 | 0.69 | 0.9644 | 3.41 | 94.2 |
| `withholder:k=12` | deception | -32.96 | [-33.48, -32.45] | 20,000 | 0.69 | 0.9653 | 3.35 | 94.1 |
| `v07i:ifocus=0,imodel=v06` | invert | -35.94 | [-36.69, -35.18] | 8,000 | 1.10 | 0.9777 | 4.61 | 87.9 |
| `diversifier:tells=1` | tells | -45.48 | [-45.76, -45.20] | 20,000 | 0.69 | 0.9708 | 3.87 | 106.3 |
| `hunter:tells=1` | tells | -48.33 | [-48.51, -48.15] | 20,000 | 0.69 | 0.9706 | 3.78 | 91.1 |
| `bluffer:tells=1` | tells | -49.97 | [-49.99, -49.95] | 20,000 | 0.69 | 0.9637 | 4.03 | 82.5 |

---

## 7. Harness findings

The phase brief: *"An exploiter that works only through a harness defect is a finding about the
harness — file it as such."* Phase 2 found five, refuted three, and closed one. None of the
exploitability numbers in this document rides on any of them; the two that are open are open because
no policy uses them, and the three that are refuted were candidates the threat model or this phase's
own reconnaissance had proposed.

| # | finding | status | evidence |
|---|---|---|---|
| **H-1** | The per-seat reset seed is an invertible function of the deal seed (threat-model **E-1**), so any policy can reconstruct all six hands in closed form. | **open** | `Game::setup` hands seat *p* `mixSeed(s, p + 77)` where the same `s` generates the deal; every stage of `mixSeed` is a bijection. Phase 0 verified the inversion numerically over 2,000 round trips. No policy reads it. Threat-model **T10** — an independent per-seat stream — remains the single most important harness repair outstanding. |
| **H-2** | `Event::confidence` is not merely broadcast to `observe`, it is **persisted in `PublicState::history`**, so the declarer's private `pAlloc` is a permanent field of the public state. | **open** | `Game::emit` pushes the event, confidence included, onto `g.pub.history` before calling `observe` on all six agents. Re-verified by grep that no policy reads it; the readers are all diagnostic probes. Repair is one line plus a NaN-poisoned identity control. |
| **H-3** | `DealDP` never received the shared-pool repair `BlockDP` was given. | **latent** | `BlockDP` carries a generation stamp; `DealDP` — the sampler the v0.6 determinization and the phase-1 inverter both use — carries none. **Tested**: a build with the pool made per-instance is bit-identical on `v06`, `v05`, `v07i:idet=48`, the truncated search and `v06:belief=block`. Unreachable today; live the moment anything holds a `DealDP` result across a nested build. |
| **H-4** | The measurement flag is visible to the thing it measures. | **open** | `decisionCapture()` is a `static thread_local bool` behind a free function in the same namespace as every policy. From phase 3 onward candidates are developed by the programme that measures them; a per-decision channel a policy can detect cannot adjudicate between candidates. |
| **H-5** | Two residue rules resolve the same physical 3–3 split by different and mutually contradictory rules, one of them an unconditional bias to team 0; both are dead code. | **latent** | `forcedEndgame`'s residue awards a 3–3 tie to team 0 unconditionally; `adjudicateRemaining`, thirty lines later, awards it to the holder of the lowest card. Neither is a rule of the game. Both were instrumented with counters and **neither fired** in 1,600 mirror games or 1,600 games against `v06:declare=0`, which produces 4.20 forced declarations a game — the willingness ladder's `-1.0` rung forces `bestGuess`, so nothing survives to the residue. |
| **R-1** | The action-cap adjudication farm. | **refuted twice over** | At the shipped cap every stalling configuration in the spec grammar returns a limit-hit rate of exactly 0.0000, and the longest single game any adversary produced against the incumbent is **149 events** against a 400-**ask** cap. Lowering the cap to 120 to force the mechanism reaches it on 0.0367 of games, and the stalling adversary still scores **0.00%**: adjudication by physical majority does not rescue a team that has already lost its cards. `limitHitRate` is printed with every phase-2 cell; the only non-zero values anywhere in the phase are 8.3e-05 in three of the switch-stack cells and the deliberately-lowered-cap probe. |
| **R-2** | Arm asymmetry — that being the A arm is worth something. | **refuted, but by construction rather than by measurement** | On bank 7051001 at 3,000 games, `v06:vmargin=-0.02` as A scores 49.77% and `v06` as A against it scores 50.23%, summing to **100.0000**; with one-seat partners on either side, **100.0000**. **That is a harness identity, not evidence.** At `--rotations=2` the arena plays both orientations of every deal, so exchanging the arm labels replays the same game multiset — the two cells report identical `eventsPerGame` to four decimals, which is the giveaway. The identity holding is worth recording because it would break if the orientation loop or the seat construction were asymmetric, but a real test of arm asymmetry needs a single fixed orientation, which the arena cannot currently express. Filed as **not tested**, with the construction that would test it named. |
| **R-3** | `BlockDP` aliasing (threat-model **E-2**). | **refuted** | The generation stamp, `current()` and `ensureCurrent()` are present and checked at every query site. The 175 raw field mismatches the v0.6 record reports are reads of the raw pool, which the query-level guard makes unobservable. **THREAT-MODEL.md §6.3 E-2 should be corrected in phase 6.** |

Two defects were also found in **phase 1's own instrument** and are recorded in RESEARCH-LOG.md §2.1
rather than here, because they are measurement defects rather than game defects: five early returns in
the v0.6 ask path recorded decisions with the previous decision's diagnostics (fixed; the correction
moves `gateBindRate` by nothing and `tieShare` by 0.0008), and phase 1's claim that the detection
floor buys down as (evaluation games)^−1/2 is refuted at four times the power (§2 above).

---

## 8. The sealed evaluation material

**train half** — 4 banks, written at commit `f4581da58ff0`.

| seed | deals | role | unseal phase | commitment digest | note |
|---:|---:|---|---:|---|---|
| 7030001 | 24,000 | train | 0 | `4bafec09b74925d6` | phase-2 adversary evaluation bank A; phases 3-4 training |
| 7030002 | 24,000 | train | 0 | `fb56483b3fffe0bb` | phase-2 adversary evaluation bank B; phases 3-4 training |
| 7030003 | 24,000 | train | 0 | `031ce425e3fdef3d` | phase-2 replication / transfer; phases 3-4 training |
| 7030004 | 24,000 | train | 0 | `e4acaae8326faed9` | phases 3-4 reserve |

**holdout half** — 7 banks, written at commit `f4581da58ff0`.

| seed | deals | role | unseal phase | commitment digest | note |
|---:|---:|---|---:|---|---|
| 7090001 | 24,000 | holdout | 5 | `896dbc89be124d85` | phase-5 holdout bank 1 |
| 7090002 | 24,000 | holdout | 5 | `0b6e40d834ac0ca1` | phase-5 holdout bank 2 |
| 7090003 | 24,000 | holdout | 5 | `863bea69baf6e73c` | phase-5 holdout bank 3 |
| 7090004 | 24,000 | holdout | 5 | `54f257c3f8ae9fab` | phase-5 fresh adversary search against the frozen v0.7 |
| 7090005 | 24,000 | holdout | 5 | `268a1dae71a31713` | phase-5 negative controls / planted-edge recovery |
| 7091001 | 24,000 | holdout | 5 | `958ada042cc26900` | sealed adversary half: evaluation bank |
| 7091002 | 24,000 | holdout | 5 | `5c39af3b5e0bd9a0` | sealed adversary half: fitting bank for the phase-5 fresh search |

The adversary bank is split 14 / 14 by a rule fixed before any result was known — rows sorted by id, alternating. The sealed half's plaintext SHA-256 is `1ca0346a332586c70a750f1523b10548…`, recorded in the file header and in `SEAL.json`, so phase 5 can verify it was not changed.

---

## 9. What phase 3 inherits

**Instruments.**

* Six responder classes with the objective axis opened: `--kpi=` climbs a per-decision failure mode of
  the *target* (its misdeclaration rate, forced-endgame incidence, ask hit rate, declaration count,
  half-suit margin, action-limit rate, game length) rather than a scoreboard, on fields the arena
  already accumulated and no objective had ever read.
* **C6 exists.** `engine/src/v07_adapt.hpp` is the corpus's first adversary carrying an online model
  of the target's policy that updates within a match, with `mode=0` as its identity control.
* The extended responder class widened from twelve coordinates to eighteen, with two groups no
  feature in the lineage carries.
* `--capture=a|b|both` on the per-decision channel, so the *target* arm can be the recorded one; and
  `urgent` / `pressure` / `urgWhy` recorded per declaration, so the urgency predicate is measurable
  by clause.
* `--partnersb`, the complement of the pre-existing `--partners`: a mixed **B**-arm team, which is
  what a one-seat deviation column needs when the adversary occupies the A arm.
* `fish7 bankdigest` and a seal `runMatch` enforces rather than a script remembers.

**Facts phase 3 should treat as established.**

1. **One configuration dominates the entire measured frontier.**
   `v07:m2=0,r12=25,s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26` scores **+4.40** over
   the deployed policy, **+2.40** over the cheap search and **+2.21** over the strongest measured
   configuration — positive on every point and every bank, commit-gate clean (zero dead asks, zero
   action-limit games, maximum game length 142). Its parts are measured separately and compose
   sub-additively (§5). **This is the phase-2 gate's "beating the frontier" case, already measured.**
2. **Take `rtie=1` on top of urgency-off: +1.91 [+1.29, +2.54], replicated, gate-clean — but it beats
   the deployed policy only, not the frontier.** Measured against the search end, urgency-off alone is
   **−0.69** against `F-cheap` and **−1.55** against `F-mid`: it fixes a real defect and is worth 1.3
   points over `v06`, and on its own it does not dominate the frontier. Combining it with the search —
   the one combination phase 2 did not run — is the obvious next cell. It is two one-line switches and it cuts the mirror misdeclaration rate from
   2.37% to 1.11%. Adding `m1=0` scores higher (+2.68) and **fails the commit gate** — 2.91% dead
   asks, a 326-ask dead run, 0.33% of games at the action limit. Take the first.
3. **The urgency escalation costs v0.6 about 1.4 points and an adversary can only add 0.38 of it.**
   Turning it off beats `v06` by +1.23 to +1.62 on three banks and cuts the mirror misdeclaration rate
   from 2.556% to 1.278% on a common base. M2 is the same defect and is
   bit-identical inert once urgency is off. This is
   the cleanest candidate phase 2 produced and it is a two-line configuration change, not an
   architecture.
4. **v0.6 carries a fifteen-point cliff at `pub.nEvents >= 220`. No adversary reaches it — 149 events
   is the longest game produced against the incumbent — but the escalation that guards it is also the
   termination guarantee, and a candidate that switches it off has a self-play tail of 405 events.
   Report `eventsPerGame`, p99 and max **in mirror** for every candidate.
5. **Half-suit contestation is real, replicated and out of class** at +2.71
   [+2.27, +3.15] against the deployed policy, with a KPI signature that is not a strength gain.
   Its attribution is unresolved and is phase 3/4 work.
6. **The detection floor does not buy down with evaluation games.** Four times the power bought 0.15
   points of floor (1.68 → 1.53), not the 0.78 the scaling law predicts, and the sub-point excesses
   resolve *to* zero rather than sharpening. A sub-1.5-point claim needs a better responder or a
   per-decision estimator, not a bigger bank.
7. **The deception family is closed on size, not on the argument the corpus gives for it.** Its best
   measured arm, `feint:tol=0.02`, reaches −1.21 — parity with the unrestricted `v04` it restricts,
   not below it — so "every restriction costs more than the corruption it buys" is not right as
   stated. `tol` spans 6.75 points for `feint` and the published value is three points off the best;
   `k` is inert at every setting and belongs on ledger C12's dead-knob list.
8. **The declaration rule is shared byte for byte across the entire frontier** — `V06Agent` overrides
   only `reset`, `resetWithKnowledge`, `observe` and `chooseAsk` — but the *inputs* to that rule are
   not, so an attack on the rule transfers across the frontier while its size need not.

---

**Correction (21 September 2026, added during the editorial revision of the v1.0 report; the
text above is left as written).** Summary item A7 and §4F state that the construction which
raises the *target's* forced-endgame incidence "was measured at −17 points". No cell in the
results tree supports that figure: no recorded configuration string describes such a
construction, the only measurement near −17 in this phase is the unrelated `v06:belief=exactdisj`
belief-objective screen row, and the planned search arm X07 (objective `forced`) has no recorded
result. The one construction that was measured is a team that never declares voluntarily (the
`no-declare` screen row), which raises its *own* forced-endgame incidence, not the target's. The
v1.0 report's Appendix E records this correction, and its §9.4.5 describes the
incidence-raising route as untested rather than closed by a price.
