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
* **Positions.** 874 ask decisions harvested from 200 `("fishbot4", {})` self-play games, thinned at rate 0.05 with a cap of 8 per game so the sample is spread over game phases rather than piling up in the opening. Fully determined positions (nothing hidden left) are skipped: there is no objective to learn where the answer is already forced.
* **Rollouts.** 874 positions evaluated, 83,168 games played out to the end by the continuation policy at 341 actions each, 0 hitting the action cap and 0 illegal-candidate sentinels. 120 worker-minutes. Totals are read back off the append-only rollout file rather than taken from a run summary, because the pass was interrupted once and resumed.

## 2. Noise-to-signal under common random numbers

v0.3's diagnosis of why determinized search lost was a noise-to-signal ratio of 2.4: the standard deviation of one action's rollout value across sampled layouts was 0.698 while the mean gap between the best and worst candidate was 0.293. The same quantities here, measured on the final set differential rather than on a depth-limited evaluation, so the scales are related but not identical:

| quantity | v0.3 | this run |
|---|---|---|
| sd of one candidate's value across worlds | 0.698 | 1.627 |
| mean best-worst candidate gap | 0.293 | 0.848 |
| **noise-to-signal, unpaired** | **2.4** | **1.92** |
| sd of the *paired* difference (CRN) | - | 1.839 |
| standard error of a pairwise comparison at K=16 | - | 0.460 |
| **noise-to-signal, paired** | - | **1.19** |
| CRN variance ratio (1.0 = no coupling) | - | 0.64 |

Common random numbers cut the variance of a pairwise comparison to 0.64 of what independent worlds would give. The per-position comparison is still noisy - which is precisely why this is a regression over 874 positions and not a search inside one.

## 2a. Does the rollout target respond to success probability at all?

The single sanity check the whole study rests on, and it belongs before any coefficient table. Within each position, subtract the mean `Qhat` over the evaluated candidates - which removes the position's own value level exactly, the same thing pairing does - and bin the deviations by `p_success`. The incumbent objective is built on the premise that a likelier ask is a better ask, so this is that premise, measured.

| p_success | candidates | mean position-centred Qhat | 95% CI (position bootstrap) |
|---|---|---|---|
| [0.000, 0.001) | 609 | -0.035 | [-0.053, -0.019] |
| [0.001, 0.150) | 737 | -0.030 | [-0.048, -0.013] |
| [0.150, 0.350) | 2388 | -0.003 | [-0.010, +0.004] |
| [0.350, 0.650) | 877 | +0.014 | [-0.002, +0.031] |
| [0.650, 0.999) | 71 | +0.010 | [-0.036, +0.065] |
| [0.999, 1.000) | 516 | +0.074 | [+0.046, +0.103] |

Slope of position-centred `Qhat` on `p`: +0.101 sets per unit probability, against the 1.0 that the `AskWeights` convention pins it at.

## 3. The fit

12,925 pairs from 874 positions; ridge penalty 100 chosen by 5-fold cross-validation split on positions.

| term | incumbent | learned | cluster SE | naive SE | SE ratio | t (cluster) | bootstrap 95% | perm p |
|---|---|---|---|---|---|---|---|---|
| `suit` | +0.06 | **+0.0014** | 0.0141 | 0.0084 | 1.67x | +0.10 | [-0.027, +0.032] | 0.950 |
| `turn` | +0.60 | **+0.1571** | 0.0580 | 0.0257 | 2.26x | +2.71 | [+0.043, +0.275] | 0.030 |
| `scarce` | +0.20 | **+0.1078** | 0.0473 | 0.0250 | 1.89x | +2.28 | [+0.002, +0.197] | 0.110 |
| `reveal` | +0.00 | **-0.0236** | 0.0248 | 0.0127 | 1.96x | -0.95 | [-0.072, +0.020] | 0.445 |
| `deplete` | +0.00 | **-0.2209** | 0.0780 | 0.0340 | 2.29x | -2.83 | [-0.378, -0.072] | 0.060 |
| `expose` | +0.00 | **-0.1933** | 0.0895 | 0.0452 | 1.98x | -2.16 | [-0.389, -0.010] | 0.133 |
| `claim` | +0.00 | **-0.8048** | 0.1010 | 0.0555 | 1.82x | -7.97 | [-0.975, -0.606] | 0.003 |
| `info` | +0.00 | **-0.1375** | 0.0399 | 0.0203 | 1.96x | -3.44 | [-0.215, -0.065] | 0.007 |
| `certain` | +0.00 | **-0.6614** | 0.0429 | 0.0177 | 2.42x | -15.42 | [-0.752, -0.589] | 0.003 |
| `concent` | +0.00 | **-0.1489** | 0.0846 | 0.0449 | 1.88x | -1.76 | [-0.309, -0.005] | 0.233 |
| `signal` | +0.00 | **+0.1051** | 0.0548 | 0.0299 | 1.83x | +1.92 | [+0.008, +0.226] | 0.193 |

**Standard errors are clustered by position (CR1).** The naive column is shown only to make the size of the error visible: it is on average 2.00x too small, because pairs from one position share candidates, share the K determinized worlds and share the rollout seeds. Reporting the naive numbers as if they were standard errors would be a serious methodological error, and it is exactly the kind of error that produces an authoritative-looking table with intervals that do not cover.

| goodness of fit | value |
|---|---|
| R^2 on the paired residual `dQ - dp` | 0.2448 |
| R^2 on `dQ`, learned weights | -0.0269 |
| R^2 on `dQ`, incumbent weights | -0.4173 |
| R^2 on `dQ`, `p_success` alone | -0.3598 |

**Permutation test** (300 position-level sign flips, which preserve the within-position dependence exactly): observed R^2 0.2448 against a null mean of 0.0051 and a null 95th percentile of 0.0100, p = 0.0033.

A second null that re-labels the candidates within each position - carrying `Qhat` and `p` together, so only the `f` rows are re-paired - gives p = 0.0099. Note that the obvious third option, permuting `Qhat` while leaving `p` behind, would be invalid: the target keeps its `-dp` term, which genuinely correlates with the features, so the null would be contaminated with real signal and the test would quietly lose its power.

### 3a. What the pinned `p_success` coefficient costs

Refitting with the `p_success` coefficient FREE gives `c_p = 0.063` (cluster SE 0.062, t = -15.1 against the pinned value of 1.0).

That is not a rounding difference, and it has a consequence that has to be stated plainly. Pinning `c_p = 1` does not merely rescale the answer: it forces every feature that correlates with `p` to take a compensating coefficient. `claim` (`p * p_team_all^(1/6)`), `certain` (`1` exactly when `p = 1`) and `deplete` (`p * (1 - count/per)`) are all direct functions of `p`, so under a pinned overshoot they will come out negative whether or not anything strategic is happening. Reading those three as strategy would be a mistake.

Because a policy's ranking is invariant to positive rescaling, `b / c_p` from the free fit is a second, genuinely different candidate `AskWeights` vector. Both are played below; play decides.

One caveat on that column, stated rather than buried: `c_p` is itself only 1.0 standard errors from zero, so `b / c_p` is a ratio with an unstable denominator and has no usable confidence interval. Read it as a direction, not as an estimate. It is played anyway, because a policy either wins games or it does not and that question does not care how the weights were derived.

| term | pinned fit | free fit `b` | free SE | rescaled `b/c_p` |
|---|---|---|---|---|
| `suit` | +0.0014 | -0.0078 | 0.0133 | -0.1237 |
| `turn` | +0.1571 | +0.0451 | 0.0589 | +0.7193 |
| `scarce` | +0.1078 | +0.0004 | 0.0458 | +0.0061 |
| `reveal` | -0.0236 | +0.0612 | 0.0251 | +0.9751 |
| `deplete` | -0.2209 | -0.0073 | 0.0812 | -0.1162 |
| `expose` | -0.1933 | -0.0313 | 0.0905 | -0.4984 |
| `claim` | -0.8048 | -0.0117 | 0.1013 | -0.1858 |
| `info` | -0.1375 | +0.0544 | 0.0401 | +0.8680 |
| `certain` | -0.6614 | +0.0787 | 0.0554 | +1.2547 |
| `concent` | -0.1489 | +0.0650 | 0.0811 | +1.0360 |
| `signal` | +0.1051 | +0.1432 | 0.0525 | +2.2828 |

R^2 on `dQ` for the free model: 0.0146.

### 3b. Known-answer calibration of the target

Three of these coefficients already have answers that were established by PLAY, not by regression: v0.3's 600-pair duplicate-deal sweeps put `suit` at 0.06, `turn` at 0.60 and `scarce` at 0.20, and measured that the last two together are worth +1.41 sets per deal-pair. So this is a calibration test with a known answer. If the rollout target cannot recover a coefficient that duplicate-deal play has already shown to be worth more than a set per pair, the target is the problem, not the regression.

| term | established by play | learned | cluster 95% | covers the established value? |
|---|---|---|---|---|
| `suit` | +0.06 | +0.0014 | [-0.026, +0.029] | **no** |
| `turn` | +0.60 | +0.1571 | [+0.043, +0.271] | **no** |
| `scarce` | +0.20 | +0.1078 | [+0.015, +0.201] | yes |

The standard error on `turn` is 0.058, so a coefficient of 0.60 would sit 10 standard errors from zero. This dataset is not short of power to see it; it does not see it.

**Robustness.** Restricting to candidates inside the incumbent's top 8 - the region where decisions are actually close - gives 3,929 pairs and R^2 0.2484; the coefficients are in the JSON under `linear_top8`.

## 4. Does a nonlinear objective buy anything?

A small torch MLP `g: R^11 -> R` (2 hidden layers of 32, tanh) trained on the *same* paired target, `dQ - dp ~ g(f(a)) - g(f(a'))`, so a linear `g` recovers the linear model exactly and the only thing under test is the shape of the per-ask objective. Train and validation are split by position (656 / 218), and the linear baseline is refitted on the same training split.

| model | validation MSE | validation R^2 |
|---|---|---|
| predict zero | 0.3744 | 0.0000 |
| linear | 0.2870 | 0.2336 |
| MLP | 0.2813 | 0.2486 |

The MLP improves validation MSE by 0.0056, which is 6% of what the linear model itself explains.

## 5. Does it win games?

This is the deliverable. Duplicate-deal duels, every deal played twice with the teams swapped on identical cards, identical rotated starting seat and identical agent seeds, so per-pair set differentials are i.i.d. across deals. X is `("fishbot4", <weights>)`; Y is `("fishbot4", {})`, which carries v0.3's champion weights `suit=0.06, turn=0.6, scarce=0.2`.

| weights | pairs | pair score | 95% CI | set diff per pair | 95% CI | dropped | verdict |
|---|---|---|---|---|---|---|---|
| pinned | 120 | 0.037 | [0.015, 0.088] | -5.992 | [-6.686, -5.297] | 0 | learned weights LOSE |
| pinned x0.25 | 120 | 0.250 | [0.181, 0.334] | -2.458 | [-3.139, -1.778] | 0 | learned weights LOSE |
| rescaled | 120 | 0.275 | [0.203, 0.361] | -2.183 | [-2.746, -1.621] | 0 | learned weights LOSE |

Learned weights, as agent kwargs:

```json
{
  "w_suit": 0.0014,
  "w_turn": 0.1571,
  "w_scarce": 0.1078,
  "w_reveal": -0.0236,
  "w_deplete": -0.2209,
  "w_expose": -0.1933,
  "w_claim": -0.8048,
  "w_info": -0.1375,
  "w_certain": -0.6614,
  "w_concent": -0.1489,
  "w_signal": 0.1051
}
```

## 6. What this says

**The learned objective does not beat the hand-tuned one.** The best variant played, rescaled, scored -2.183 sets per deal-pair [-2.746, -1.621]. That is the result, and dressing it up would be worse than useless.

The interesting part is *why*, and the diagnostics above locate it precisely rather than leaving it to speculation.

1. **The rollout target barely responds to the quantity the objective is built on.** Position-centred `Qhat` rises by only +0.101 sets across the whole range of `p_success`, against the 1.0 that the `AskWeights` convention pins it at. The unpinned fit agrees: `c_p = 0.063` (SE 0.062).
2. **The target fails a known-answer calibration.** v0.3 established by 600-pair duplicate-deal play that `turn` at 0.60 and `scarce` at 0.20 are together worth +1.41 sets per pair. The regression puts `turn` at +0.157 with a cluster standard error of 0.058. An effect the size play has already measured would be impossible to miss at this precision. The instrument, not the sample size, is the limitation.
3. **The apparent fit is mostly the model undoing the pinned term.** R^2 on the paired residual is 0.245, but R^2 on `dQ` itself for the unpinned model is only 0.0146. Almost all of the first number is the p-correlated features (`claim`, `certain`, `deplete`) taking negative coefficients to cancel a `Delta_p` term the data does not support. Quoting the residual R^2 as evidence that the objective had been learned would have been a real error, and it is the exact shape of error this project's own methodological warning is about.

The likeliest mechanism is the continuation policy. A rollout has to be finished by a policy that can attach to a determinized mid-game position, and the public-information heuristic that does throws away most of the value of a marginal card, so a card won by a good ask is largely squandered before the game ends and the ask stops mattering to the final differential. The measured variance is not the binding constraint: common random numbers already cut the paired variance to 0.64 of independent worlds and brought the noise-to-signal ratio from v0.3's 2.4 down to 1.19. What is missing is not precision, it is a continuation strong enough for the difference between two asks to survive to the end of the game.

**This paragraph used to end differently, and the difference mattered.** It said `fish.beliefs.BeliefState` *cannot* attach to a determinized mid-game position - anchored on the initial deal, and refuses - and the whole line was closed on that. It is anchored on the initial deal but does not have to be *handed* one: given the current hand and the public log, `initial_hand()` back-computes a consistent deal and the tracker attaches to it. `scripts4/rollout_target.py` does exactly that over 110 positions with no `BeliefContradiction`, and on identical positions, worlds and root asks the target's slope on P(success) goes from +0.040 +/- 0.045 with the heuristic to +0.681 +/- 0.142 with the engine - a paired difference of +0.641 +/- 0.145. The heuristic needs 181 plies to finish a position the engine finishes in 26. So the mechanism above is right and the wall was not there; see `fish4/NEGATIVE_CLAIMS.md`.

So the honest summary is narrower than the roadmap item hoped for: the *machinery* for learning an ask objective works - the pairing, the clustering, the permutation test and the synthetic-recovery tests all behave - but the *target* it is pointed at is not yet informative enough to improve on weights that were tuned against actual play. Duplicate-deal play remains the strongest measurement instrument this project has, and it is measuring something these rollouts cannot see.

## 7. Reproducing

```
py scripts4/learn_ask_objective.py harvest
py scripts4/learn_ask_objective.py rollout
py scripts4/learn_ask_objective.py fit
py scripts4/learn_ask_objective.py validate --pairs 120
py scripts4/learn_ask_objective.py report
```

All stages cap themselves at 3 worker processes. Both append-only data files are resumable: re-running a stage picks up where it stopped rather than starting over.

