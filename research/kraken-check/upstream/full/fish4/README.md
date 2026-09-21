# FishBot v0.4

An engine for **Literature** (Fish / Canadian Fish): six players, two teams of
three in alternating seats, nine six-card half-suits, and a scoring rule that
requires declaring the *exact* distribution of a half-suit among your own team.

v0.4 keeps v0.3's engine, rules, information boundary and evaluation protocol as
the substrate --- so every comparison against the previous version is
apples-to-apples --- and replaces three layers: **inference**, the **ask
objective**, and **claiming**.

---

## The headline

| comparison | duplicate deal-pairs | set differential per pair | 95% CI |
|---|---|---|---|
| **v0.4 champion vs v0.3 champion** | **500** | **+1.854** | **[+1.542, +2.166]** |
| v0.4 champion vs v0.3 champion (replication) | 300 | +2.440 | [+2.052, +2.828] |
| v0.4 champion vs `probabilistic` | 200 | +2.785 | [+2.274, +3.296] |
| v0.4 champion vs `memory` | 200 | +5.360 | [+4.856, +5.864] |
| v0.4 champion vs `heuristic` | 120 | +14.092 | 120/120 pairs won |
| v0.4 champion vs `random` | 80 | +16.300 | 80/80 pairs won |

It also holds under every rule variant tested, including declare-at-any-time as
the rules are usually written (+2.175), the 48-card deck (+1.080), and
wrong-split-goes-to-the-opponents (+1.690).

Absolute strength, over 988 exactly solved positions (278 information-resolved,
776 with two live half-suits), agreement with **progress-optimal** play:

| agent | resolved | uncertain |
|---|---|---|
| v0.3 champion | 100.0% | 67.6% |
| v0.4, opponent model off | 100.0% | 67.5% |
| **v0.4 champion** | **100.0%** | **72.5%** |

---

## The result that shaped the version

The plan was to remove a known defect: v0.3's belief sampler was documented as
*not uniform* over the deals consistent with the public record, so every
probability it reported inherited an unquantified bias. We removed it --- exactly,
with a dynamic program validated against exhaustive enumeration --- and **it
changed nothing**. Against the v0.3 champion at matched seeds, 250 duplicate
deal-pairs:

```
exact inference, no opponent model    -0.008 sets/pair  [-0.482, +0.466]
exact inference + opponent model      +1.920 sets/pair  [+1.479, +2.361]
```

Scoring posteriors against the **true hidden hands** (available in simulation,
used only to score, never to act) explains why. Over 517 decisions and 13,640
uncertain-card predictions, the *exactly uniform* posterior was **worse** at
predicting reality than the biased sampler it replaced (negative log-likelihood
1.362 vs 1.341 at a matched budget). The uniform posterior is a correct theorem
about a false hypothesis --- that players' choices carry no information beyond
legality --- and v0.3's bias happened to encode a crude opponent model. Making
that model explicit, with one parameter, beats both.

---

## Layout

```
fish4/
  counting.py     exact counting, marginals, uniform sampling, joint probabilities
  sis.py          importance sampler with closed-form proposal densities
  sisbatch.py     the same, vectorised across the batch (5-6x)
  oppmodel.py     the opponent choice model and the no-declaration signal
  posterior.py    the layer the policy talks to: exact where possible, unbiased always
  askfeat.py      the 11-term ablatable ask objective
  hsvalue.py      a learned half-suit value, and an objective in units of sets
  claim4.py       exact-MAP claiming and expected-value forced claims
  tablebase4.py   exact endgame play on the v2 solver
  exact2.py       the widened exact solver (two half-suits, dense tablebase)
  agent4.py       FishBot4: every strategic choice is a constructor argument
  match.py        duplicate-deal matches spanning v0.3 and v0.4 policies
  evalx/          calibrated evaluation: sequential tests, ablation guard, registry
  infer/          alternative inference backends and the accuracy/compute study
  learn/          rollout regression for the ask objective
```

## Using it

```bash
py -m fish4 serve            # browser UI at http://127.0.0.1:8420  <- start here
py -m fish4 play             # play it in the terminal
py -m fish4 analyse --ply 40 # analyse a position: eval, ranked asks, card table
py -m fish4 watch --explain  # watch it play itself, with commentary
py -m fish4 sim   --games 100
py -m fish4 duel  --games 200 -x fishbot4:'{"opponent_gamma":0.35}' -y memory
py -m fish4 perpetual        # the dead-position / deadlock study
py -m fish4 bench
```

In the terminal client, `hint` shows what the engine would do and why, `cards`
prints its posterior over every hidden card, and `auto` hands the turn to the
engine. Everything you see is built from your seat's own view: the card table is
an inference from the public record, not a peek at the hidden hands.

The engine reports what a chess engine reports, translated:

| chess | here |
|---|---|
| evaluation in pawns | evaluation in **sets** you expect to finish ahead |
| ranked move list | every legal ask with P(it lands) and its eval on each branch |
| principal variation | the **possession**: the ask chain if each one lands |
| — | the **card table**: the posterior over who holds every hidden card |
| — | the **declaration table**: P(your team holds all six), and P(this split is exactly right) |

Analysis takes 130-270 ms, so it is interactive.

## Reproducing the research

```bash
py -m pytest tests4 -q                          # v0.4 tests
py -m pytest tests -q                           # v0.3 tests, still green
py scripts4/duel.py jobs/j6_ladder.json 4       # the strength ladder
py scripts4/posterior_accuracy.py 14 3          # posteriors vs ground truth
py scripts4/exact_bench4.py                     # agreement with exact play
py scripts4/perpetual_study.py 200 4            # dead positions and repetition
py scripts4/fit_hsvalue.py 200 3                # fit the half-suit value model
py scripts4/analytics4.py 120 3                 # strategy statistics
py scripts4/check_tex.py                        # structural check on the paper
```

Every duel appends to `results/v04_duels.jsonl`; `py scripts4/summarise_duels.py`
prints them with, alongside each interval, **the smallest effect that run could
have detected**. That column matters: the measured per-pair standard deviation
of the set differential is 3.869 sets, so a 150-pair cell resolves nothing below
about 0.9 sets per pair, and a "no significant difference" below that is absence
of evidence rather than evidence of absence.

## The champion

```python
from fish4.registry4 import make_agent
agent = make_agent(("fishbot4", {"opponent_gamma": 0.35}))
```

Everything else is at its default, which is deliberately v0.3's own ask
objective: the gain reported above comes from inference, not from re-tuning the
objective on top of it.

## What did not work

Recorded because a list of only the things that worked would misrepresent the
study. All measured by duplicate-deal duels against the champion:

- eleven-term ask objective: every new term unresolved, none positive
- learned half-suit value as the objective: **-7.36 sets/pair** (it is very good
  at predicting which team scores a half-suit, and a disastrous thing to
  maximise)
- learned ask objective by rollout regression: -2.18 sets/pair, with the cause
  located in the rollout continuation policy rather than in the fit
- the "nobody has declared it, so they probably do not hold it" signal: null
- more or fewer belief draws (96 / 160 / 320): null, so precision has saturated
- exact endgame tablebase: changes not one of 988 solved positions
- claim threshold 0.97 -> 0.99: changes essentially no decisions
- the one screening cell that did resolve (+0.490): failed to replicate twice
  (+0.068, -0.066)
