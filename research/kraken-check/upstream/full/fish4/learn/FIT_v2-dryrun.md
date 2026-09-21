# Learning the ask objective

ROADMAP item 2. Every number below was produced by `scripts4/learn_ask_objective.py`; nothing here is an estimate or a recollection. Regenerate with `report`, which reads `results/ask_objective_fit.json`.

## 1. What was measured

v0.3 scored an ask as `P(success) + 0.06*depth`, then found by 600-pair duplicate-deal experiments that a turn-risk term at weight 0.6 and a scarcity term at weight 0.2 were together worth +1.41 sets per deal-pair, while heavier weights were actively harmful. Those three numbers are the incumbent, and they are what the learned objective has to beat.

The learning target is a common-random-numbers paired rollout estimate `Qhat(a)` of each candidate ask, and the fit is the paired regression

```
  Qhat(a) - Qhat(a')  -  [p(a) - p(a')]  =  [f(a) - f(a')] . w  +  error
```

with the `p_success` coefficient pinned to 1.0 (the `AskWeights` scale convention) and a per-position intercept removed by construction.

* **Feature basis.** `fish4.askfeat.TERM_NAMES`, 11 terms: `suit`, `turn`, `scarce`, `reveal`, `deplete`, `expose`, `claim`, `info`, `certain`, `concent`, `signal`. The basis is recorded in every harvested position and re-checked at fit time, because it is not a constant of this project: it gained an eleventh term while a completed harvest was on disk, and fitting the stale matrix would have attributed one term's effect to another without any visible symptom.
* **Rollouts.** 91 positions evaluated, 8,720 games played out to the end by the continuation policy at 67 actions each, 0 hitting the action cap and 0 illegal-candidate sentinels. 40 worker-minutes. Totals are read back off the append-only rollout file rather than taken from a run summary, because the pass was interrupted once and resumed.

## 2. Noise-to-signal under common random numbers

v0.3's diagnosis of why determinized search lost was a noise-to-signal ratio of 2.4: the standard deviation of one action's rollout value across sampled layouts was 0.698 while the mean gap between the best and worst candidate was 0.293. The same quantities here, measured on the final set differential rather than on a depth-limited evaluation, so the scales are related but not identical:

| quantity | v0.3 | this run |
|---|---|---|
| sd of one candidate's value across worlds | 0.698 | 2.276 |
| mean best-worst candidate gap | 0.293 | 1.454 |
| **noise-to-signal, unpaired** | **2.4** | **1.57** |
| sd of the *paired* difference (CRN) | - | 2.992 |
| standard error of a pairwise comparison at K=16 | - | 0.748 |
| **noise-to-signal, paired** | - | **1.16** |
| CRN variance ratio (1.0 = no coupling) | - | 0.86 |

Common random numbers cut the variance of a pairwise comparison to 0.86 of what independent worlds would give. The per-position comparison is still noisy - which is precisely why this is a regression over 91 positions and not a search inside one.

## 2a. Does the rollout target respond to success probability at all?

The single sanity check the whole study rests on, and it belongs before any coefficient table. Within each position, subtract the mean `Qhat` over the evaluated candidates - which removes the position's own value level exactly, the same thing pairing does - and bin the deviations by `p_success`. The incumbent objective is built on the premise that a likelier ask is a better ask, so this is that premise, measured.

| p_success | candidates | mean position-centred Qhat | 95% CI (position bootstrap) |
|---|---|---|---|
| [0.000, 0.001) | 63 | -0.153 | [-0.253, -0.048] |
| [0.001, 0.150) | 67 | +0.021 | [-0.061, +0.110] |
| [0.150, 0.350) | 284 | +0.012 | [-0.015, +0.040] |
| [0.350, 0.650) | 86 | +0.019 | [-0.059, +0.105] |
| [0.650, 0.999) | 5 | -0.312 | [-0.875, +0.198] |
| [0.999, 1.000) | 40 | +0.116 | [-0.063, +0.282] |

Slope of position-centred `Qhat` on `p`: +0.149 sets per unit probability, against the 1.0 that the `AskWeights` convention pins it at.

## 3. The fit

1,360 pairs from 91 positions; ridge penalty 100 chosen by 5-fold cross-validation split on positions.

| term | incumbent | learned | cluster SE | naive SE | SE ratio | t (cluster) | bootstrap 95% | perm p |
|---|---|---|---|---|---|---|---|---|
| `suit` | +0.06 | **+0.0120** | 0.0538 | 0.0257 | 2.09x | +0.22 | [-0.088, +0.122] | 0.805 |
| `turn` | +0.60 | **+0.3391** | 0.2491 | 0.1434 | 1.74x | +1.36 | [-0.090, +0.735] | 0.195 |
| `scarce` | +0.20 | **+0.3790** | 0.1363 | 0.0838 | 1.63x | +2.78 | [+0.168, +0.721] | 0.073 |
| `reveal` | +0.00 | **-0.1833** | 0.0811 | 0.0517 | 1.57x | -2.26 | [-0.311, -0.051] | 0.122 |
| `deplete` | +0.00 | **+0.5918** | 0.3616 | 0.1810 | 2.00x | +1.64 | [-0.254, +1.620] | 0.220 |
| `expose` | +0.00 | **-0.0194** | 0.4441 | 0.2460 | 1.80x | -0.04 | [-0.695, +1.043] | 0.976 |
| `claim` | +0.00 | **-0.9454** | 0.4867 | 0.2215 | 2.20x | -1.94 | [-2.223, -0.040] | 0.073 |
| `info` | +0.00 | **-0.0986** | 0.1319 | 0.0736 | 1.79x | -0.75 | [-0.324, +0.135] | 0.512 |
| `certain` | +0.00 | **-0.6092** | 0.1672 | 0.0779 | 2.15x | -3.64 | [-0.832, -0.333] | 0.024 |
| `concent` | +0.00 | **+0.0076** | 0.2854 | 0.1492 | 1.91x | +0.03 | [-0.783, +0.438] | 1.000 |
| `signal` | +0.00 | **-0.5112** | 0.2115 | 0.1170 | 1.81x | -2.42 | [-0.949, -0.136] | 0.024 |

**Standard errors are clustered by position (CR1).** The naive column is shown only to make the size of the error visible: it is on average 1.88x too small, because pairs from one position share candidates, share the K determinized worlds and share the rollout seeds. Reporting the naive numbers as if they were standard errors would be a serious methodological error, and it is exactly the kind of error that produces an authoritative-looking table with intervals that do not cover.

| goodness of fit | value |
|---|---|
| R^2 on the paired residual `dQ - dp` | 0.1058 |
| R^2 on `dQ`, learned weights | 0.0224 |
| R^2 on `dQ`, incumbent weights | -0.0839 |
| R^2 on `dQ`, `p_success` alone | -0.0932 |

**Permutation test** (40 position-level sign flips, which preserve the within-position dependence exactly): observed R^2 0.1058 against a null mean of 0.0336 and a null 95th percentile of 0.0643, p = 0.0244.

A second null that re-labels the candidates within each position - carrying `Qhat` and `p` together, so only the `f` rows are re-paired - gives p = 0.0164. Note that the obvious third option, permuting `Qhat` while leaving `p` behind, would be invalid: the target keeps its `-dp` term, which genuinely correlates with the features, so the null would be contaminated with real signal and the test would quietly lose its power.

### 3a. What the pinned `p_success` coefficient costs

Refitting with the `p_success` coefficient FREE gives `c_p = -0.182` (cluster SE 0.156, t = -7.6 against the pinned value of 1.0).

That is not a rounding difference, and it has a consequence that has to be stated plainly. Pinning `c_p = 1` does not merely rescale the answer: it forces every feature that correlates with `p` to take a compensating coefficient. `claim` (`p * p_team_all^(1/6)`), `certain` (`1` exactly when `p = 1`) and `deplete` (`p * (1 - count/per)`) are all direct functions of `p`, so under a pinned overshoot they will come out negative whether or not anything strategic is happening. Reading those three as strategy would be a mistake.

Because a policy's ranking is invariant to positive rescaling, `b / c_p` from the free fit is a second, genuinely different candidate `AskWeights` vector. Both are played below; play decides.

One caveat on that column, stated rather than buried: `c_p` is itself only 1.2 standard errors from zero, so `b / c_p` is a ratio with an unstable denominator and has no usable confidence interval. Read it as a direction, not as an estimate. It is played anyway, because a policy either wins games or it does not and that question does not care how the weights were derived.

| term | pinned fit | free fit `b` | free SE | rescaled `b/c_p` |
|---|---|---|---|---|
| `suit` | +0.0120 | -0.0014 | 0.0525 | +0.0076 |
| `turn` | +0.3391 | +0.2664 | 0.2475 | -1.4671 |
| `scarce` | +0.3790 | +0.2820 | 0.1340 | -1.5533 |
| `reveal` | -0.1833 | -0.1192 | 0.0797 | +0.6565 |
| `deplete` | +0.5918 | +0.9383 | 0.3542 | -5.1682 |
| `expose` | -0.0194 | +0.1800 | 0.3971 | -0.9917 |
| `claim` | -0.9454 | +0.0567 | 0.4671 | -0.3122 |
| `info` | -0.0986 | +0.1072 | 0.1309 | -0.5902 |
| `certain` | -0.6092 | +0.3092 | 0.1740 | -1.7030 |
| `concent` | +0.0076 | +0.2776 | 0.2768 | -1.5291 |
| `signal` | -0.5112 | -0.4932 | 0.2099 | +2.7166 |

R^2 on `dQ` for the free model: 0.0532.

### 3b. Known-answer calibration of the target

Three of these coefficients already have answers that were established by PLAY, not by regression: v0.3's 600-pair duplicate-deal sweeps put `suit` at 0.06, `turn` at 0.60 and `scarce` at 0.20, and measured that the last two together are worth +1.41 sets per deal-pair. So this is a calibration test with a known answer. If the rollout target cannot recover a coefficient that duplicate-deal play has already shown to be worth more than a set per pair, the target is the problem, not the regression.

| term | established by play | learned | cluster 95% | covers the established value? |
|---|---|---|---|---|
| `suit` | +0.06 | +0.0120 | [-0.094, +0.117] | yes |
| `turn` | +0.60 | +0.3391 | [-0.149, +0.827] | yes |
| `scarce` | +0.20 | +0.3790 | [+0.112, +0.646] | yes |

The standard error on `turn` is 0.249, so a coefficient of 0.60 would sit 2 standard errors from zero. This dataset is not short of power to see it; it does not see it.

**Robustness.** Restricting to candidates inside the incumbent's top 8 - the region where decisions are actually close - gives 416 pairs and R^2 0.1359; the coefficients are in the JSON under `linear_top8`.

## 4. Does a nonlinear objective buy anything?

Not measured: torch not installed

## 5. Does it win games?

This is the deliverable. Duplicate-deal duels, every deal played twice with the teams swapped on identical cards, identical rotated starting seat and identical agent seeds, so per-pair set differentials are i.i.d. across deals. X is `("fishbot4", <weights>)`; Y is `("fishbot4", {})`, which carries v0.3's champion weights `suit=0.06, turn=0.6, scarce=0.2`.

_Not yet run._

## 6. What this says

_Validation not yet run; no conclusion._

## 7. Reproducing

```
py scripts4/learn_ask_objective.py harvest
py scripts4/learn_ask_objective.py rollout
py scripts4/learn_ask_objective.py fit
py scripts4/learn_ask_objective.py validate --pairs 120
py scripts4/learn_ask_objective.py report
```

All stages cap themselves at 3 worker processes. Both append-only data files are resumable: re-running a stage picks up where it stopped rather than starting over.

