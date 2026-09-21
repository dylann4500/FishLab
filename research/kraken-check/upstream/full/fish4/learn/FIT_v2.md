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
* **Rollouts.** 1023 positions evaluated, 97,920 games played out to the end by the continuation policy at 63 actions each, 0 hitting the action cap and 0 illegal-candidate sentinels. 401 worker-minutes. Totals are read back off the append-only rollout file rather than taken from a run summary, because the pass was interrupted once and resumed.

## 2. Noise-to-signal under common random numbers

v0.3's diagnosis of why determinized search lost was a noise-to-signal ratio of 2.4: the standard deviation of one action's rollout value across sampled layouts was 0.698 while the mean gap between the best and worst candidate was 0.293. The same quantities here, measured on the final set differential rather than on a depth-limited evaluation, so the scales are related but not identical:

| quantity | v0.3 | this run |
|---|---|---|
| sd of one candidate's value across worlds | 0.698 | 2.272 |
| mean best-worst candidate gap | 0.293 | 1.446 |
| **noise-to-signal, unpaired** | **2.4** | **1.57** |
| sd of the *paired* difference (CRN) | - | 2.937 |
| standard error of a pairwise comparison at K=16 | - | 0.734 |
| **noise-to-signal, paired** | - | **1.12** |
| CRN variance ratio (1.0 = no coupling) | - | 0.84 |

Common random numbers cut the variance of a pairwise comparison to 0.84 of what independent worlds would give. The per-position comparison is still noisy - which is precisely why this is a regression over 1023 positions and not a search inside one.

## 2a. Does the rollout target respond to success probability at all?

The single sanity check the whole study rests on, and it belongs before any coefficient table. Within each position, subtract the mean `Qhat` over the evaluated candidates - which removes the position's own value level exactly, the same thing pairing does - and bin the deviations by `p_success`. The incumbent objective is built on the premise that a likelier ask is a better ask, so this is that premise, measured.

| p_success | candidates | mean position-centred Qhat | 95% CI (position bootstrap) |
|---|---|---|---|
| [0.000, 0.001) | 715 | -0.150 | [-0.181, -0.116] |
| [0.001, 0.150) | 780 | -0.063 | [-0.090, -0.037] |
| [0.150, 0.350) | 2959 | -0.018 | [-0.028, -0.008] |
| [0.350, 0.650) | 1017 | +0.077 | [+0.054, +0.097] |
| [0.650, 0.999) | 65 | -0.079 | [-0.170, +0.003] |
| [0.999, 1.000) | 584 | +0.232 | [+0.192, +0.271] |

Slope of position-centred `Qhat` on `p`: +0.355 sets per unit probability, against the 1.0 that the `AskWeights` convention pins it at.

## 3. The fit

15,268 pairs from 1023 positions; ridge penalty 100 chosen by 5-fold cross-validation split on positions.

| term | incumbent | learned | cluster SE | naive SE | SE ratio | t (cluster) | bootstrap 95% | perm p |
|---|---|---|---|---|---|---|---|---|
| `suit` | +0.06 | **-0.0122** | 0.0258 | 0.0127 | 2.03x | -0.47 | [-0.060, +0.036] | 0.681 |
| `turn` | +0.60 | **+0.1079** | 0.0789 | 0.0390 | 2.02x | +1.37 | [-0.048, +0.267] | 0.216 |
| `scarce` | +0.20 | **+0.2400** | 0.0723 | 0.0356 | 2.03x | +3.32 | [+0.099, +0.377] | 0.010 |
| `reveal` | +0.00 | **-0.1178** | 0.0346 | 0.0178 | 1.94x | -3.41 | [-0.192, -0.051] | 0.007 |
| `deplete` | +0.00 | **-0.2190** | 0.1131 | 0.0501 | 2.25x | -1.94 | [-0.449, -0.003] | 0.106 |
| `expose` | +0.00 | **+0.8246** | 0.1413 | 0.0621 | 2.28x | +5.83 | [+0.531, +1.068] | 0.003 |
| `claim` | +0.00 | **+0.0000** | 0.0000 | 0.0000 | nanx | +nan | [+0.000, +0.000] | 1.000 |
| `info` | +0.00 | **-0.1553** | 0.0503 | 0.0247 | 2.04x | -3.08 | [-0.258, -0.061] | 0.013 |
| `certain` | +0.00 | **-0.5017** | 0.0562 | 0.0259 | 2.17x | -8.93 | [-0.604, -0.396] | 0.003 |
| `concent` | +0.00 | **+0.1244** | 0.1395 | 0.0677 | 2.06x | +0.89 | [-0.156, +0.384] | 0.415 |
| `signal` | +0.00 | **-0.2488** | 0.0866 | 0.0431 | 2.01x | -2.87 | [-0.426, -0.086] | 0.023 |

**Standard errors are clustered by position (CR1).** The naive column is shown only to make the size of the error visible: it is on average 2.08x too small, because pairs from one position share candidates, share the K determinized worlds and share the rollout seeds. Reporting the naive numbers as if they were standard errors would be a serious methodological error, and it is exactly the kind of error that produces an authoritative-looking table with intervals that do not cover.

| goodness of fit | value |
|---|---|
| R^2 on the paired residual `dQ - dp` | 0.0644 |
| R^2 on `dQ`, learned weights | 0.0567 |
| R^2 on `dQ`, incumbent weights | -0.0016 |
| R^2 on `dQ`, `p_success` alone | -0.0082 |

**Permutation test** (300 position-level sign flips, which preserve the within-position dependence exactly): observed R^2 0.0644 against a null mean of 0.0034 and a null 95th percentile of 0.0063, p = 0.0033.

A second null that re-labels the candidates within each position - carrying `Qhat` and `p` together, so only the `f` rows are re-paired - gives p = 0.0099. Note that the obvious third option, permuting `Qhat` while leaving `p` behind, would be invalid: the target keeps its `-dp` term, which genuinely correlates with the features, so the null would be contaminated with real signal and the test would quietly lose its power.

### 3a. What the pinned `p_success` coefficient costs

Refitting with the `p_success` coefficient FREE gives `c_p = 0.383` (cluster SE 0.091, t = -6.8 against the pinned value of 1.0).

That is not a rounding difference, and it has a consequence that has to be stated plainly. Pinning `c_p = 1` does not merely rescale the answer: it forces every feature that correlates with `p` to take a compensating coefficient. `claim` (`p * p_team_all^(1/6)`), `certain` (`1` exactly when `p = 1`) and `deplete` (`p * (1 - count/per)`) are all direct functions of `p`, so under a pinned overshoot they will come out negative whether or not anything strategic is happening. Reading those three as strategy would be a mistake.

Because a policy's ranking is invariant to positive rescaling, `b / c_p` from the free fit is a second, genuinely different candidate `AskWeights` vector. Both are played below; play decides.


| term | pinned fit | free fit `b` | free SE | rescaled `b/c_p` |
|---|---|---|---|---|
| `suit` | -0.0122 | -0.0148 | 0.0257 | -0.0386 |
| `turn` | +0.1079 | -0.0003 | 0.0816 | -0.0007 |
| `scarce` | +0.2400 | +0.2330 | 0.0720 | +0.6083 |
| `reveal` | -0.1178 | -0.0559 | 0.0350 | -0.1461 |
| `deplete` | -0.2190 | -0.0459 | 0.1162 | -0.1198 |
| `expose` | +0.8246 | +0.8932 | 0.1400 | +2.3321 |
| `claim` | +0.0000 | +0.0000 | 0.0000 | +0.0000 |
| `info` | -0.1553 | +0.0800 | 0.0584 | +0.2090 |
| `certain` | -0.5017 | +0.0008 | 0.0827 | +0.0022 |
| `concent` | +0.1244 | +0.3229 | 0.1398 | +0.8431 |
| `signal` | -0.2488 | -0.2163 | 0.0856 | -0.5648 |

R^2 on `dQ` for the free model: 0.0647.

### 3b. Known-answer calibration of the target

Three of these coefficients already have answers that were established by PLAY, not by regression: v0.3's 600-pair duplicate-deal sweeps put `suit` at 0.06, `turn` at 0.60 and `scarce` at 0.20, and measured that the last two together are worth +1.41 sets per deal-pair. So this is a calibration test with a known answer. If the rollout target cannot recover a coefficient that duplicate-deal play has already shown to be worth more than a set per pair, the target is the problem, not the regression.

| term | established by play | learned | cluster 95% | covers the established value? |
|---|---|---|---|---|
| `suit` | +0.06 | -0.0122 | [-0.063, +0.038] | **no** |
| `turn` | +0.60 | +0.1079 | [-0.047, +0.263] | **no** |
| `scarce` | +0.20 | +0.2400 | [+0.098, +0.382] | yes |

The standard error on `turn` is 0.079, so a coefficient of 0.60 would sit 8 standard errors from zero. This dataset is not short of power to see it; it does not see it.

**Robustness.** Restricting to candidates inside the incumbent's top 8 - the region where decisions are actually close - gives 4,671 pairs and R^2 0.0665; the coefficients are in the JSON under `linear_top8`.

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

