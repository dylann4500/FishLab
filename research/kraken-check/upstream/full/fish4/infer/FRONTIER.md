# Hidden-state inference for FishBot v0.4: accuracy versus compute

What this answers: **which inference backend should v0.4 ship, at what setting.**

Everything below comes from code in this repository. The study is
`scripts4/infer_frontier.py`; its stages are cached in `results/` so it can be
re-run stage by stage:

```
py scripts4/infer_frontier.py --stage harvest   # -> results/infer_positions.json.gz
py scripts4/infer_frontier.py --stage stats     # -> results/infer_position_stats.json
py scripts4/infer_frontier.py --stage gold      # -> results/infer_gold.json.gz
py scripts4/infer_frontier.py --stage diag      # -> results/infer_diagnostics.json
py scripts4/infer_frontier.py --stage sweep     # -> results/infer_frontier.json
py scripts4/infer_frontier.py --stage hybrid    # -> results/infer_hybrid.json
py scripts4/infer_frontier.py --stage report    # renders the tables in section 7
```

Correctness is tested separately, against brute-force enumeration, in
`tests4/test_infer.py` (21 tests, under a minute).

---

## 0. Read every absolute time in this document twice

This machine shares 8 cores with several other jobs and the contention is
severe and time-varying. A bare Python loop iteration
(`for i in range(1e6): x += i`) measured between **222 ns and 469 ns** over the
course of this study, against roughly 30-50 ns on an unloaded modern core. So:

- **Ratios between backends transfer. Absolute milliseconds do not.**
- Every timing in this document is a **minimum over repetitions** (3 for the
  diagnostics, 4 for the frontier sweep). Contention can only make a timing
  larger, so the minimum is the least contaminated estimator available.
- Accuracy numbers are unaffected by any of this.

A consequence worth stating: on an unloaded machine every backend would move
left on the cost axis by something like a factor of 5, and the *ordering* of the
recommendations would not change, but the budget thresholds would.

---

## 1. The corpus: 360 real positions

`results/infer_positions.json.gz`. Seven complete games played by six
`tuned(w_turn=0.6, w_scarce=0.2)` agents -- the current champion -- snapshotting
the acting seat's propagated `BeliefState` at every decision, then subsampled
evenly across all 751 decisions to 360 positions so openings, middlegames and
endgames are all represented. Snapshots store the constraint store only
(candidates, public locations, ORs), so the study does not depend on the agent
code staying fixed.

| structure | mean | median | p90 | p99 | max |
|---|---|---|---|---|---|
| free cards | 25.1 | 25 | 42 | 45 | 45 |
| distinct candidate masks | 4.09 | 4 | 6 | 8 | 9 |
| live OR constraints | 4.66 | 5 | 9 | 12 | 14 |
| cards per OR | 3.17 | 3 | 5 | 5 | 5 |
| quota state space `prod_p (q_p+1)` | 23859 | 7560 | 72900 | 100000 | 100000 |
| half-suits carrying an OR | 3.0 | 3 | 5 | 7 | 7 |

**6.9% of positions have no live OR at all** (mostly the opening, before anyone
has asked). On those the OR-free closed form *is* the posterior, and three of
the four backends short-circuit to it.

Error is reported as **mean L1 per free card**: for each free card, the L1
distance between the estimated and gold 6-vectors, averaged over the free cards
of the position, then averaged over positions. Pinned and publicly-located cards
are excluded because every backend gets them exactly right by construction, and
including them would just divide every number by a constant.

---

## 2. The gold standard, and why you should believe it

This is the single most important part of the study: everything else is measured
against it.

**Method.** For each position, `exact_or` with a 2 x 10^8 work budget. It fitted
**360 / 360** positions, so no position needed the long-rejection fallback. On
60 of the 360 a second, structurally unrelated estimate was computed and the two
compared.

**Cross-check 1: exact against exact.** On the 5 cross-checked positions with no
live OR, `exact_or`'s block DP and `counting.py`'s group-count DP are two
completely different exact algorithms for the same quantity. They agreed to
**L1/card = 0.000000** -- floating-point identity, on all 5.

**Cross-check 2: exact against a long rejection run.** Rejection sampling from
the exactly-uniform proposal is unbiased by construction, so its disagreement
with `exact_or` should be pure Monte Carlo error.

| tier | positions | accepted draws | L1/card disagreement (mean / median / max) | conservative 1-sigma bound (mean) |
|---|---|---|---|---|
| deep | 7 | 106k - 200k | 0.00261 / 0.00228 / 0.00403 | 0.00416 |
| broad | 53 | 20k | 0.00560 / 0.00566 / 0.01044 | 0.01092 |

The bound is the binomial standard error of the accepted sample, summed over
players and averaged over cards. It is an **upper** bound on the rejection
estimator's true error, because most cards inside that estimator are
Rao-Blackwellised (section 4) and therefore have strictly lower variance than
the binomial formula assumes.

Across all 55 sampled cross-checks, **disagreement / bound** had mean 0.547,
median 0.475 and **maximum 1.329**. Not one position disagreed by more than 1.33
conservative sigma. That is exactly the signature of two estimators of the same
quantity differing only by the sampler's noise, and it is the strongest
statement available short of enumeration.

**Cross-check 3: enumeration.** `tests4/test_infer.py` enumerates every
consistent world, ORs included, on 34 synthetic systems and asserts `exact_or`
matches to `1e-12` and that its partition function equals the world count. The
enumerator is a plain depth-first search sharing no code with the DP.

**Statistical care in what follows.** Because gold is exact on all 360
positions, backend errors below are *not* inflated by gold's own error: there
isn't any. The only place Monte Carlo error in a reference matters is the
cross-check table above, where it is stated.

---

## 3. Backend A: `v03` (the incumbent)

A thin wrapper over `BeliefState.sample_current_hands`. Knob: number of sampled
worlds; its shipped default is 32.

The incumbent is **biased, not merely noisy**. Its error falls with more samples
and then stops falling: from `n=8` to `n=512` (a 64x increase in compute) the
L1/card only drops to about 0.095, and the drop from 256 to 512 is already small
relative to the compute spent. `tests4/test_infer.py::
test_v03_worlds_are_valid_but_marginals_are_biased` pins this down on small
systems: with 20000 draws, at which the per-entry standard error is under 0.004,
v0.3 still leaves errors above 0.05, up to 0.27. Its worlds satisfy every
constraint -- that is asserted in the same test -- it just does not weight them
uniformly.

---

## 4. Backend B: `uniform_reject`

Exactly-uniform draws from the candidate-mask + quota system, rejected against
the ORs. Unbiased by construction; the whole problem is cost.

### 4.1 The draw, optimised

`counting.GroupSystem.sample_counts` re-enumerates integer compositions in
Python on every draw. `fish4/infer/fastdraw.py` keeps the auxiliary tables that
the backward pass already builds and throws away, which turns each player's take
into a table peel, then memoises the cumulative weights per
`(player, peel level, state)`.

Measured on 60 real positions, minimum of 3 timings each:

| | mean | median | p90 | max |
|---|---|---|---|---|
| `sample_counts` (reference) | 553 us | 190 us | 1632 us | 4436 us |
| `sample_counts_fast` | **33.8 us** | **23.8 us** | 73.0 us | 110 us |
| speedup | 11.5x | 6.8x | 31x | 54x |
| `prepare_fast` (one-off setup) | 5.85 ms | 1.61 ms | 12.8 ms | 84.7 ms |
| auxiliary tables | 823 KB | 162 KB | 2.3 MB | 7.6 MB |

The brief's target was under 10 us per draw. On this machine the answer is
23.8 us at the median; scaled by the interpreter calibration in section 0
(222-469 ns per loop iteration against 30-50 ns typical) that is 2-5 us of
unloaded-machine time, so the target is met in the only sense that transfers,
and missed in raw wall-clock here. The distribution is unchanged: the fast draw
is asserted against `expected_counts` in `tests4`.

Setup pays for itself after roughly ten draws; below that the reference sampler
is cheaper.

### 4.2 The control variate: it works, by about 20%

The OR-free marginals are known exactly, and the all-draws mean estimates them,
so `m_hat = m_closedform + (mean over accepted - mean over all)`. Paired
comparison on 55 OR-active positions, **identical draws for both estimators**:

| accepted draws | L1/card, control variate | L1/card, plain | ratio | CV better on |
|---|---|---|---|---|
| 32 | 0.1371 | 0.1725 | 0.795 | 50 / 55 |
| 128 | 0.0690 | 0.0849 | 0.812 | 49 / 55 |
| 512 | 0.0343 | 0.0428 | 0.801 | 44 / 55 |

A stable ~20% reduction in L1, i.e. ~36% in variance, i.e. worth about 1.56x
more draws. Real, consistent across sample sizes, and free at draw time.

**But the control variate has a cost that undercuts the whole backend.** It
needs the *exact* OR-free posterior. Computing that with
`counting.GroupSystem.expected_counts` measured **93 ms mean / 9.6 ms median /
812 ms max** on top of the backward pass -- the single most expensive thing in
the v0.4 stack, because it carries a `(groups, states, quota)` accumulator and
copies it once per group per player. Computing the identical numbers with the
flat-array DP from backend C costs **7.9 ms mean / 5.1 ms median / 25.6 ms max**
and agrees to 1.7e-16. `uniform_reject` now uses the DP (and computes it lazily,
so `control_variate=False` never pays for it).

Which is the point: **the control-variate variant of backend B contains most of
backend C.** If you can afford the OR-free DP you can nearly afford the exact
one.

### 4.3 Rao-Blackwellisation

Acceptance depends only on which players got the OR-relevant cards, so only
those are dealt individually; every other card of a mask group contributes its
exact conditional expectation `residual[p] / residual_total` instead of a 0/1
indicator. Same expectation, lower variance, no extra work, and it removes the
per-draw shuffle. This is also why the gold cross-check's binomial standard
error is a conservative bound rather than the actual error.

---

## 5. Backend C: `exact_or`

Exact including the ORs. See the module docstring for the derivation; the four
things that made it practical were: blocks defined by OR-components rather than
half-suits, enumerating each block over exchangeability classes rather than
card-by-card, dropping one redundant quota coordinate, and running the DP on
flat contiguous arrays instead of strided multi-dimensional slices.

That last one was worth the most and is the least obvious. A last-axis slice-add
on a `(10,10,10,10,8)` numpy array measured **580 us**, against **32 us** for
the same element count contiguously -- an 18x penalty for striding at these
sizes. The rewrite from the natural multi-dimensional implementation to flat
arrays with cached 0/1 overflow masks, together with the class-based block
enumeration, took the backend from **99 ms mean / 76 ms median** per position to
the numbers below.

Measured on 60 real positions, minimum of 3 timings:

| | mean | median | p90 | p99 | max |
|---|---|---|---|---|---|
| total (build + solve) | 16.8 ms | 10.7 ms | 34.0 ms | 88.8 ms | 88.8 ms |
| of which planning/enumeration | 2.6 ms | 1.9 ms | 5.4 ms | 22.4 ms | 22.4 ms |
| DP state space (reduced) | 2727 | 1260 | 8000 | 10000 | 10000 |
| transitions | 137 | 149 | 257 | 314 | 314 |
| OR blocks | 2.9 | 3 | 5 | 7 | 7 |
| cards in the largest block | 3.7 | 4 | 5 | 6 | 6 |

Fraction of real positions inside a budget:

| budget | 5 ms | 10 ms | 25 ms | 50 ms |
|---|---|---|---|---|
| fits | 40% | 50% | 75% | 92% |

(A second measurement of the same 60 positions taken when the machine was less
loaded gave 40% / 54% / 96% / 100%. The spread between those two runs is a
direct illustration of section 0.)

It never refused at a 2 x 10^8 work budget on any of the 360 positions. The
budget guard exists for positions this corpus does not contain; when it trips,
`ok` is False and the caller must fall back rather than receive a wrong answer.

---

## 6. Backend D: `mcmc`

A swap/3-cycle chain on worlds, uniform by proposal symmetry.

### 6.1 Irreducibility: 2-swaps alone are not enough, measured

The smallest counterexample, now a regression test: three cards with candidate
masks {0,1}, {1,2}, {0,2} and one card each for players 0, 1, 2. The consistent
worlds are `(0,1,2)` and `(1,2,0)`; every 2-swap between them is blocked by a
mask. Measured: a 2-swap-only chain run for 3000 steps visits **1 of 2** worlds.
With 3-cycles enabled it visits **2 of 2**.

On random small systems, `chain_coverage` compares the chain's visited set
against full enumeration. Across the 34 synthetic systems used by `tests4`
(2 to 23 consistent worlds each, median 4), the 3-cycle chain reached **every**
enumerated world on **34 / 34**; the 2-swap-only chain failed on **8 / 34**
(24%). That is a measurement, not an argument, and it is asserted in
`test_three_cycles_reach_every_world`.

### 6.2 Burn-in: one sweep

Measured across an **ensemble of 800 independent chains** on 6 OR-active
positions. The marginal of the state at sweep `t`, averaged over chains, *is*
the chain's distribution at time `t`; its distance from gold is the transient
that burn-in has to remove. (A time average cannot show this, because it mixes
the transient in with the stationary part.)

| sweeps | 0 | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| L1/card vs gold | 0.0811 | 0.0550 | 0.0561 | 0.0542 | 0.0546 | 0.0564 | 0.0555 | 0.0552 |

The floor of ~0.0552 is the 800-chain ensemble's own Monte Carlo noise, not
bias: it scales as predicted with chain count (600 chains gave 0.0642, and
0.0642 x sqrt(600/800) = 0.0556). **The transient is gone after one sweep**: from
sweep 1 onward every value sits within +-0.001 of the floor, so any residual
burn-in bias is below the ~0.002 resolution of this measurement. Sweep 0 (the
greedy initialiser, before any MCMC) is resolvably worse at 0.0811.

Practical consequence: `burn_sweeps=2` is already generous, and the sweep
settings below use `max(2, n_sweeps/8)`.

### 6.3 Autocorrelation: about two sweeps

One long chain per position, on the indicator "free card j belongs to player p"
for the three cards whose marginal is closest to 1/2 -- the slowest coordinates
available -- with the integrated autocorrelation time taken by Geyer's
initial-positive truncation over 200 lags.

| tau_int (sweeps) | mean | median | p90 | max |
|---|---|---|---|---|
| all statistics | 2.20 | 2.00 | 4.02 | 4.07 |
| worst per position | 2.45 | 2.13 | 4.07 | 4.07 |

So one sweep buys roughly half an independent sample of the whole configuration,
and the chain mixes fast enough that the estimator is limited by wall-clock, not
by correlation.

The marginal estimator uses the **time integral over the whole post-burn-in
trajectory** (credit a card's owner with the dwell time whenever it changes)
rather than thinned snapshots: O(1) per accepted move instead of O(n_free) per
snapshot, and it uses every step.

---

## 7. The frontier

`results/infer_frontier.json`. Accuracy on all 349 positions with free cards
(one run each); wall-clock on every 6th position, minimum of 4 runs. Cost
includes everything a caller pays: building the `FreeSystem`, any DP setup, and
the query itself.

| backend | knob | ms/decision (median) | ms (mean) | ms (p90) | L1/card mean | L1/card median | L1/card p90 | refused |
|---|---|---|---|---|---|---|---|---|
| A v03 | n_samples=8 | 0.932 | 0.891 | 1.340 | 0.5101 | 0.5275 | 0.5890 | 0 |
| A v03 | n_samples=16 | 1.760 | 1.644 | 2.585 | 0.3589 | 0.3706 | 0.4152 | 0 |
| A v03 | n_samples=32 | 3.204 | 2.991 | 5.031 | 0.2582 | 0.2687 | 0.2956 | 0 |
| A v03 | n_samples=64 | 6.673 | 6.432 | 10.465 | 0.1889 | 0.1935 | 0.2186 | 0 |
| A v03 | n_samples=128 | 14.403 | 13.183 | 21.925 | 0.1446 | 0.1475 | 0.1696 | 0 |
| A v03 | n_samples=256 | 25.179 | 29.284 | 66.895 | 0.1142 | 0.1130 | 0.1417 | 0 |
| A v03 | n_samples=512 | 88.765 | 92.326 | 171.033 | 0.0951 | 0.0917 | 0.1296 | 0 |
| B uniform_reject (CV) | n_accepted=8 | 14.965 | 26.167 | 83.495 | 0.2618 | 0.2541 | 0.4407 | 0 |
| B uniform_reject (CV) | n_accepted=16 | 14.480 | 21.660 | 59.446 | 0.1997 | 0.1884 | 0.3327 | 0 |
| B uniform_reject (CV) | n_accepted=32 | 20.316 | 35.120 | 80.369 | 0.1417 | 0.1332 | 0.2368 | 0 |
| B uniform_reject (CV) | n_accepted=64 | 36.510 | 75.924 | 250.018 | 0.1004 | 0.0944 | 0.1664 | 0 |
| B uniform_reject (CV) | n_accepted=128 | 77.531 | 148.650 | 531.733 | 0.0700 | 0.0653 | 0.1149 | 0 |
| B uniform_reject (CV) | n_accepted=256 | 92.470 | 198.460 | 510.595 | 0.0494 | 0.0458 | 0.0812 | 0 |
| B uniform_reject (CV) | n_accepted=512 | 166.810 | 299.183 | 948.692 | 0.0344 | 0.0326 | 0.0560 | 0 |
| B uniform_reject (plain) | n_accepted=8 | 5.440 | 8.443 | 24.608 | 0.3348 | 0.3117 | 0.5252 | 0 |
| B uniform_reject (plain) | n_accepted=16 | 9.567 | 17.191 | 52.743 | 0.2433 | 0.2177 | 0.3712 | 0 |
| B uniform_reject (plain) | n_accepted=32 | 13.723 | 39.373 | 131.249 | 0.1705 | 0.1583 | 0.2618 | 0 |
| B uniform_reject (plain) | n_accepted=64 | 21.930 | 64.683 | 188.785 | 0.1209 | 0.1102 | 0.1878 | 0 |
| B uniform_reject (plain) | n_accepted=128 | 63.387 | 112.888 | 444.003 | 0.0838 | 0.0784 | 0.1289 | 0 |
| B uniform_reject (plain) | n_accepted=256 | 137.051 | 221.015 | 655.734 | 0.0593 | 0.0557 | 0.0916 | 0 |
| B uniform_reject (plain) | n_accepted=512 | 166.478 | 326.857 | 863.013 | 0.0411 | 0.0383 | 0.0608 | 0 |
| D mcmc | n_sweeps=4, burn_sweeps=2 | 0.574 | 0.548 | 0.872 | 0.7344 | 0.7440 | 0.8473 | 0 |
| D mcmc | n_sweeps=8, burn_sweeps=2 | 0.841 | 0.811 | 1.315 | 0.5443 | 0.5509 | 0.6456 | 0 |
| D mcmc | n_sweeps=16, burn_sweeps=2 | 1.446 | 1.319 | 2.141 | 0.3943 | 0.3988 | 0.4740 | 0 |
| D mcmc | n_sweeps=32, burn_sweeps=4 | 2.524 | 2.685 | 4.860 | 0.2845 | 0.2866 | 0.3418 | 0 |
| D mcmc | n_sweeps=64, burn_sweeps=8 | 5.436 | 5.277 | 9.136 | 0.2028 | 0.2029 | 0.2439 | 0 |
| D mcmc | n_sweeps=128, burn_sweeps=16 | 9.730 | 9.502 | 16.234 | 0.1435 | 0.1446 | 0.1704 | 0 |
| D mcmc | n_sweeps=256, burn_sweeps=32 | 26.323 | 25.213 | 47.166 | 0.1020 | 0.1014 | 0.1224 | 0 |
| D mcmc | n_sweeps=512, burn_sweeps=64 | 63.398 | 58.169 | 105.003 | 0.0717 | 0.0718 | 0.0856 | 0 |
| D mcmc | n_sweeps=1024, burn_sweeps=128 | 103.877 | 103.366 | 184.743 | 0.0510 | 0.0516 | 0.0609 | 0 |
| D mcmc | n_sweeps=2048, burn_sweeps=256 | 211.302 | 200.544 | 346.480 | 0.0357 | 0.0358 | 0.0438 | 0 |
| C exact_or | work_budget=500000 | 3.453 | 4.828 | 11.295 | 0.0000 | 0.0000 | 0.0000 | 118 |
| C exact_or | work_budget=2000000 | 6.971 | 8.072 | 18.731 | 0.0000 | 0.0000 | 0.0000 | 29 |
| C exact_or | work_budget=10000000 | 7.547 | 7.934 | 17.514 | 0.0000 | 0.0000 | 0.0000 | 0 |
| C exact_or | work_budget=40000000 | 7.545 | 8.022 | 17.714 | 0.0000 | 0.0000 | 0.0000 | 0 |
| C exact_or | work_budget=200000000 | 7.309 | 8.949 | 20.431 | 0.0000 | 0.0000 | 0.0000 | 0 |

Reading it:

- **`exact_or`'s 0.0000 is circular in this table and must not be read as an
  accuracy result.** Gold *is* `exact_or`, so of course it agrees with itself.
  The evidence that `exact_or` is right is section 2 (agreement with a long
  rejection run to within 1.33 conservative sigma on 55 positions, and
  floating-point identity with a second exact algorithm on 5) and the
  enumeration tests. What this table does contribute for `exact_or` is its
  **cost** and its **refusal rate**, which are not circular at all.
- At a 10^7 work budget `exact_or` refused **0 of 349** positions, at 2 x 10^6
  it refused 29 (8.3%), and at 5 x 10^5 it refused 118 (34%).
- The three `exact_or` rows at 10^7, 4 x 10^7 and 2 x 10^8 do identical work and
  differ only in measurement noise (7.55 / 7.55 / 7.31 ms median). Treat that
  0.24 ms spread as the noise floor of the timing method, even at best-of-4.
- **v0.3 plateaus.** From `n=256` to `n=512` it spends 3.5x the time to move
  0.114 -> 0.095. It is converging to its bias, not to the posterior.
- **`mcmc` overtakes v0.3 at around 7 ms and pulls away from there.** Below
  that the two are within noise of each other per millisecond (0.185 for the
  chain interpolated at 6.7 ms against 0.189 for v0.3), but the chain is
  converging and v0.3 is heading for a floor: at 63 ms the chain is at 0.0717
  and v0.3 needs 89 ms to reach 0.0951 and stop there.
- **`uniform_reject` and `mcmc` are roughly equally efficient** per millisecond,
  with the chain marginally ahead (0.0717 at 63 ms against 0.0700 at 78 ms), and
  both are far behind `exact_or`.
- Measured OR acceptance rate across the corpus: **0.27** (mean over positions).
  At `n_accepted=8` the draw cap (25x the target) left 6 of 349 positions with
  no accepted draw at all, and those fell back to the closed form; from
  `n_accepted=16` up, none did.

### 7.1 The hybrid: exact where it fits, fallback where it does not

An `exact_or` row with refusals reports error only over the positions it
accepted, which flatters it. `results/infer_hybrid.json` measures the whole
policy instead -- try the exact backend at a work budget, and when it declines,
pay for a fallback *on top of* the wasted planning time.

| work budget | fallback | fallback used on | L1/card mean | ms median | ms p90 |
|---|---|---|---|---|---|
| 2 x 10^5 | v03 n=32 | 47.9% | 0.1327 | 5.54 | 9.85 |
| 2 x 10^5 | mcmc 64 sweeps | 47.9% | 0.0965 | 8.37 | 14.68 |
| 2 x 10^5 | mcmc 256 sweeps | 47.9% | 0.0483 | 18.09 | 43.71 |
| 5 x 10^5 | v03 n=32 | 33.8% | 0.0940 | 6.23 | 10.35 |
| 5 x 10^5 | mcmc 64 sweeps | 33.8% | 0.0677 | 6.76 | 14.12 |
| 5 x 10^5 | uniform_reject n=32 | 33.8% | 0.0486 | 9.49 | 74.50 |
| 5 x 10^5 | mcmc 256 sweeps | 33.8% | 0.0337 | 8.25 | 45.17 |
| 2 x 10^6 | v03 n=32 | 8.3% | 0.0239 | 6.15 | 18.92 |
| 2 x 10^6 | mcmc 64 sweeps | 8.3% | 0.0158 | 8.51 | 20.12 |
| 2 x 10^6 | mcmc 256 sweeps | 8.3% | 0.0078 | 8.79 | 27.80 |
| 2 x 10^6 | uniform_reject n=32 | 8.3% | 0.0065 | 7.12 | 19.78 |
| 10^7 | (never used) | 0.0% | **0.0000** | 6.30 - 8.24 | 15.05 - 19.56 |

The last row is four separate measurements of the same computation -- the
fallback never fires at 10^7 -- so its spread is pure timing noise and is the
honest error bar on every median in this document.

---

## 8. Recommendation

**Ship backend C, `exact_or`, with `work_budget = 10^7`.** It is exact, it
refused none of 349 real positions, and it costs about 7.5 ms at the median and
18 ms at p90 on this (heavily loaded) machine. Every sampling backend is both
slower and wrong.

That is unusual enough to say plainly: the accuracy-versus-compute frontier for
this problem is not a curve, it is a point. The exact backend sits below and to
the left of every sampled one. There is no budget above roughly 8 ms at which a
sampler is the right answer.

Per requested budget, using **median** wall-clock (p90 in brackets):

| budget | recommendation | measured cost | L1/card |
|---|---|---|---|
| ~5 ms | `exact_or(work_budget=2e5)` + `v03(n_samples=32)` fallback | 5.5 ms (9.9) | 0.1327 |
| ~15 ms | **`exact_or(work_budget=1e7)`** | 7.5 ms (18.5) | **0.0000** |
| ~40 ms | **`exact_or(work_budget=1e7)`** | 7.5 ms (18.5) | **0.0000** |

Notes on the 5 ms row, which is the only genuinely contested one:

- Nothing exact fits 5 ms on every position, so this budget forces an
  approximation on 34-48% of positions and the question becomes which one.
- Even so, the hybrid beats every pure sampler that fits the same budget by a
  wide margin: **0.1327 at 5.5 ms**, against 0.2028 for pure `mcmc(64)` at
  5.4 ms and 0.2582 for the incumbent `v03(32)` at 3.2 ms. Half the positions
  are being answered exactly instead of approximately, and that is worth more
  than any amount of extra sampling spread over all of them.
- The budget is worth stretching. At 6.8 ms, `exact_or(5e5)` + `mcmc(64)` gives
  **0.0677** -- half the error for 1.3 ms more. At 8.5 ms, `exact_or(2e6)` +
  `mcmc(64)` gives **0.0158**. The error falls roughly an order of magnitude per
  2 ms in this range, because each increment moves another slice of positions
  from "approximated" to "exact". Nothing in the sampler-only frontier behaves
  like that.

At 15 ms and 40 ms there is nothing to decide. Raise the work budget to 10^7,
take the exact answer, and keep the refusal path only as a guard for positions
this 360-position corpus does not contain -- if it ever fires, `mcmc` with
256 sweeps is the fallback the data supports.

### What to do with the other three

- **`v03`**: retire it as an inference backend. Keep the code as the baseline
  the tests compare against, and keep `_sample_initial` as the MCMC
  initialiser's fallback.
- **`uniform_reject`**: keep it as the *validation* tool it turned out to be. It
  is the only backend that can check `exact_or` on a real position without
  enumerating, and section 2 is the reason to believe anything in this document.
  Do not ship it on the hot path.
- **`mcmc`**: keep it as the fallback for the refusal path and for any future
  position class where the exact DP genuinely does not fit. It is the best
  sampler measured, and it has the useful property of degrading gracefully with
  the budget rather than refusing.

---

## 9. Negative results worth keeping

1. **Rejection sampling is a dead end for production, despite being exactly
   correct.** After the fast draw (11.5x), Rao-Blackwellisation, and the control
   variate, `uniform_reject` still needs **167 ms** at the median to reach
   L1/card 0.034, on positions where `exact_or` returns the exact answer in
   **7.5 ms**. The measured acceptance rate is 0.27 and the optimised draw is
   23.8 us; neither number has a further order of magnitude in it. This is the
   measurement that killed it.

2. **The control variate works and it does not matter.** It removes a stable
   ~20% of L1 (36% of variance), consistently, on 44-50 of 55 OR-active
   positions. But it needs the exact OR-free posterior, and the cheapest way to
   get that is the same flat DP that backend C uses. The variance reduction is
   real; the backend it was meant to rescue is not rescuable.

3. **More samples do not fix a biased sampler.** v0.3 spends 3.5x the compute
   between `n=256` and `n=512` to move L1/card from 0.114 to 0.095, and on small
   enumerable systems it is still 0.05-0.27 away from truth at 20000 draws,
   where its own standard error is under 0.004. Sixty-four times the compute
   buys a factor of 5, and then stops.

4. **`counting.GroupSystem.expected_counts` is the most expensive single thing
   in the v0.4 inference stack**: 93 ms mean, 812 ms max, on top of the backward
   pass. The flat-array DP produces bit-identical numbers (agreement 1.7e-16) in
   7.9 ms mean, 25.6 ms max. `counting.py` is untouched -- its API and semantics
   are unchanged and its tests still pass -- but nothing on the hot path should
   call `expected_counts` any more.

5. **Multi-dimensional numpy striding was the hidden cost in the exact DP.** A
   last-axis slice-add on a `(10,10,10,10,8)` array measured 580 us against
   32 us for the same elements contiguously. Rewriting the DP onto flat arrays
   with cached overflow masks, plus enumerating blocks over exchangeability
   classes, took `exact_or` from 99 ms mean / 76 ms median to 17 ms / 11 ms. If
   the first version had been accepted, the recommendation in section 8 would
   have gone the other way.

6. **2-swaps alone do not connect the state space.** 8 of 34 enumerable
   synthetic systems (24%) were not fully covered by a 2-swap chain; 0 of 34
   failed once 3-cycles were added. The three-card counterexample in
   `test_two_swaps_alone_are_reducible` shows why: masks can block every
   pairwise exchange between two worlds that a single rotation connects.

7. **A micro-optimisation deliberately NOT applied, so the numbers match the
   code.** `uniform_reject._prep` builds the sampler even on the 6.9% of
   positions that have no live OR and therefore short-circuit to the closed
   form. Skipping it would save an average of about 1.4 ms across the corpus,
   concentrated in the cheapest settings. It was not applied because the sweep
   had already been measured and shaving cost off a backend that is not being
   recommended would bias the comparison in favour of the one that is.
