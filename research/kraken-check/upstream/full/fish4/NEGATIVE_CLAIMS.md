# The load-bearing "cannot"s, and which of them anybody checked

This project closed an entire research direction on a sentence that turned out to
be false. The paper and `fish4/learn/dataset.py` both said that
`fish.beliefs.BeliefState` "is anchored on the initial deal and refuses to attach
to a mid-game position, so a posterior cannot be reconstructed after the fact",
and the objective-learning line was written off on that basis. It is anchored on
the initial deal, but it does not have to be *handed* one: given the current hand
and the public log, `initial_hand()` back-computes a consistent deal in a few
lines. `scripts4/rollout_target.py` now does exactly that over 110 positions
without a single `BeliefContradiction`, and the target the regression could not
learn from turns out to carry seven times the signal.

The failure mode is structural, not careless. **A negative claim generates no
experiment.** A positive result invites replication; "this cannot be done"
invites moving on. So the sentence that closes a direction is exactly the
sentence nobody tests, and it can sit in a docstring for a whole version
collecting citations from things built on top of it.

This file is the response: a register of every load-bearing impossibility claim
in the engine, with its status. It was produced by sweeping the source for
`cannot`, `impossible`, `never` and `refuses`, discarding the ones that are local
implementation notes, and giving each survivor a verdict.

## Refuted

| where | claim | what happened |
|---|---|---|
| `learn/dataset.py`, `learn/rollout.py`, paper §Learning the ask objective | the belief tracker cannot attach to a determinized mid-game position | **False.** Corrected in all three. The slope it was used to explain goes from +0.101 to +0.681 once the strong continuation actually runs. |

## True in the limit, false where the engine operates

| where | claim | what happened |
|---|---|---|
| `sis.py`, `OpponentModel._build_tilt` | a proposal twist "cannot change what the policy computes — only how precisely it computes it" | **True asymptotically, false at the operating point.** Importance sampling is unbiased for any proposal with the right support, so at infinite draws the claim holds. At the shipped 160 draws, tuning the twist raised effective sample size from 83.7 to 105.8 and made the posterior marginals 3.4× *worse*. The docstring's own stated test — "do the marginals agree while the ESS rises" — is what caught it, and the parameter ships at zero. An asymptotic `cannot` is not a finite-sample `cannot`. |

## True, and checked rather than assumed

| where | claim | why it holds |
|---|---|---|
| `claim4.py` | an opponent cannot take a set from us by claiming it | Rules: a claim by a team that does not hold all six *gives* the set away. This is the basis of "waiting is nearly free", and the one screening cell that appears to strain it (`claim threshold 0.90`, +0.035) has a pre-registered confirmatory run. |
| `oppmodel.py` | the choice likelihood cannot go into the constraint propagator | It is a weight, not a constraint; the propagator is exact over a support and a likelihood does not restrict support. Structural. |
| `lookahead.py` | a card taken cannot be asked for again, and a resolved half-suit cannot be re-asked | Rules, and the reason the search's action set shrinks monotonically. |
| `perpetual.py` | a team that provably owns a half-suit but cannot place the split makes no progress | Proven against the exact solver, and the dead-position analysis is built on it. |
| `learn/fit.py` | the fit cannot learn a better ask than the rollout policy can exploit | A property of one-step policy improvement. It is *true*, and the response was not to argue with it but to make the rollout policy stronger — which is what the correction at the top of this file enabled. |
| `exact2_study.py` | v1 refuses to generate a deliberately losing claim | Stated as a limitation and then **measured** (`giveaway_study`): does allowing it ever change a value? That is the right shape for a claim like this. |

## The rule this leaves

Before a negative claim is allowed to close a direction, it needs one of:

1. a proof from the rules, stated as such;
2. a measurement, with the range it was measured over; or
3. a note that it is untested, so the next person knows it is load-bearing and
   unexamined rather than settled.

The middle case carries a rider learned from the tilt: **say what range you
measured over.** "Cannot, asymptotically" and "cannot, at 160 draws" are
different claims, and only one of them is about the engine that ships.


## The no-declaration signal is a null (2026-08-24: RETRACTED)

Two screening cells put `opp_lambda` at 0.9 and 1.8 and returned +0.205 and
+0.190, both intervals containing zero, and the paper reported them as nulls
alongside the rest of the failed-experiment table.

They measured nothing. Three bugs made the term a no-op:

- `fish4/sisbatch.py` assembled it into `logl` and then the opponent-model
  branch that follows did `logl = ...` rather than `logl += ...`, discarding it
  whenever any non-self player had asked -- 632 of 641 decisions in
  `results/ess_probe.json`;
- the half-suit column lists were indices into the caller's free-card order
  applied to a `picks` matrix in the sampler's own sort, so on a typical
  position six of eight columns pointed at cards from other half-suits;
- the scalar reference sampler stored the term and never applied it, so the
  batch/scalar agreement test could not detect either of the above.

All three are fixed and `tests4/test_opp_lambda.py` fails against the previous
code. The cells are now reported as INVALID rather than null. The feature has
not been re-measured.

The lesson is the same one this file keeps recording, in a new place: a null
from a feature nobody checked was switched on is not evidence about the
feature. Two cells agreeing on it was not corroboration either -- they agreed
because they were both running the champion.
