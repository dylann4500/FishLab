# EXACT2: the v2 exact solver, what it can do, and what it found

`fish4/exact2.py` replaces `fish/exact.py` as the source of absolute ground
truth. `fish/exact.py` is untouched and still passes its own tests; v2 is
additive.

**Measurement environment.** Windows 11, Python 3.13.5, numpy 2.5.1, 8 logical
cores and 15.8 GB RAM *shared with other jobs*. Most timings below were taken
while the machine sat at ~91% memory load with 1.3-1.4 GB free, which matters
for one number only (the table build) and is called out where it does. At most
3 worker processes were ever used, and only for the v1 baseline.

Everything here is reproduced by:

```
py -m fish4.exact2_study fuzz            # abstraction + playout validation
py -m fish4.exact2_study cross --positions 300 --workers 3
py -m fish4.exact2_study fixpoint        # cyclic-game semantics
py -m fish4.exact2_study closed          # the structural result
py -m fish4.exact2_study giveaway
py -m fish4.exact2_study timing --workers 3
py -m fish4.exact2_corpus --games 260 --per-game 6 \
      --generator probabilistic,memory,tuned,heuristic
py -m pytest tests4/test_exact2.py -q    # 33 tests, 17-35 s
```

Raw output: `results/exact2_study.json`, `results/exact2_cross_check.json`,
`results/exact_positions_v2.json`.

Note that `fish4/`, `tests4/` and `results/exact_positions_v2.json` are listed
in `.git/info/exclude` by another session working in this checkout, so they sit
on disk untracked. Nothing here modifies `fish/`, `tests/`, `fish4/counting.py`,
`fish4/posterior.py`, `fish4/evalx/` or any top-level document; the v1 suite
still passes unchanged (193 tests).

---

## 1. The wall v1 hit was in the encoding, not the game

v1 documents its limit as "roughly seven live cards in a single unresolved
half-suit". That description hides the real shape of the constraint. A claim
strips all six of its cards from every hand, so

> **live cards == 6 x (number of unresolved half-suits), exactly, always.**

There is no such thing as a 7-, 8- or 9-live-card Fish position. Asserted over
random play in `tests4/test_exact2.py::test_live_cards_is_six_times_unresolved`.
So v1 solves the one-half-suit layer and nothing else, and its own benchmark
harness (`fish.benchmark_exact.collect_solvable_positions`, which filters on
`2 <= live_cards <= 7`) was only ever collecting 6-live-card positions.

v1 keys a state on `(tuple(hands), turn, tuple(set_winner))` with 54-bit hand
masks, so a layer is a set of *card placements*: 6^6 x 6 = 279,936 of them for
one half-suit, 6^12 x 6 = 13,060,694,016 for two.

## 2. The abstraction: count, do not enumerate

Under perfect information the value depends only on **how many** cards of each
live half-suit each player holds, never on which ones. The proof is in the
module docstring of `fish4/exact2.py`: any permutation of the deck that maps
each half-suit onto itself is a game automorphism (it preserves ask legality,
ask success, claim exactness, the "some card is with an opponent" test, and
turn normalisation), and two states have equal per-(half-suit, player) counts
exactly when such a permutation relates them.

This is an exact symmetry of the perfect-information game, not a heuristic; no
rule can tell 2C from 3C. It is checked, not merely argued:

* `abstraction_fuzz` - 600 random positions; for each, the abstract successor
  set is compared against the successors the **real engine** produces. 0
  mismatches.
* `representative_invariance` - 300 abstract states, each materialised 3 ways
  with a random permutation of the half-suit's cards. 0 value mismatches.

A half-suit's counts are a composition of 6 into 6 ordered parts: 462 of them.

| layer | v1 card placements | v2 abstract states | legal (turn-normalised) | shrink |
|---|---|---|---|---|
| m = 1 | 279,936 | 2,772 | 2,604 | 101x |
| m = 2 | 13,060,694,016 | 1,280,664 | 1,275,960 | 10,198x |
| m = 3 | ~6.9e14 | 591,667,368 | - | 1.2e6x |

m = 1 and m = 2 are solved **in full**, once, as dense tables - an actual Fish
tablebase. m = 3 does not fit in full but its *reachable* subgraph does, and
`solve_reachable` handles it (section 3).

## 3. Before and after, measured

### Same positions, both solvers

300 varied one-half-suit positions (`generate_m1_positions`, deliberately
weighted towards one-team-holds-everything, single-holder, two-holder and
fully-spread layouts, each embedded in a random mix of already-won,
already-lost and nulled half-suits).

| | v1 `ExactSolver` | v2 `Exact2Solver` |
|---|---|---|
| positions solved | 300 / 300 | 300 / 300 |
| total CPU | **2,240.2 s** | **0.052 s** |
| mean per position | 7.47 s | 0.17 ms |
| median per position | 0.99 s | 0.091 ms |
| worst position seen | 139.8 s (a six-way split, timed separately in `tests4`) | not separable from timer noise |
| wall clock, 3 workers | 769.5 s | n/a (single process) |
| card/abstract states enumerated | 2,368,371 total (7,895 mean) | 0 - table lookup |

That is a **43,000x** reduction in per-query cost, but the honest figure has to
charge the one-off table build. Including it:

| positions answered | v1 total | v2 total (build + queries) | speedup |
|---|---|---|---|
| 1 | 7.5 s | 26 s | 0.3x (v2 loses) |
| 300 | 2,240 s | 26 s | 86x |
| 988 (the corpus) | n/a - v1 can only answer the 212 one-half-suit records, ~1,580 s | 26 s | - |

v2 is the wrong tool for a single position and the right one for a corpus.

### Positions v1 cannot solve at all

60 two-half-suit positions: v1 solved **0 of 60**, every one refused with
`SubgameTooLarge: layer with 12 live cards is ~6^12 placements, far beyond the
400000-state budget`. v2 answers all 60 from the table:

| | value | full optimal-action set |
|---|---|---|
| m = 1 | 12.8 us | 99.5 us |
| m = 2 | 13.8 us | 270 us |

### Table build (once per process)

| layer | states | sweeps to converge | wall | peak allocation |
|---|---|---|---|---|
| m = 1 | 2,772 | 8 | 0.010-0.013 s | negligible |
| m = 2 | 1,280,664 | 8 | **5.4 s** unloaded; 17-26 s at 91% memory load | 222 MB |

The build includes solving the layer **twice** (value iteration and the
independent attractor computation) and asserting agreement. The memory figure
is not incidental: the first implementation materialised three (n, 6) helper
arrays and peaked at 360 MB, which on this machine turned a 5 s build into a
31 s one purely through paging. Rewriting `_layer_slots` to keep every
intermediate (n,)-shaped cut the peak to 222 MB.

### Three half-suits, 18 live cards

`solve_reachable` enumerates only the states reachable from the root. That is a
real reduction, not wishful thinking: an ask only moves a card **to** a player
who already holds one of that half-suit, so the set of card-holding players
never grows.

90 random m = 3 roots, budget 400,000 abstract states:

| | |
|---|---|
| solved | 61 |
| refused as too large | 29 |
| median reachable states | 33,569 |
| max reachable states | 323,789 |
| median solve time | 2.59 s |
| max solve time | 50.0 s |

So the practical frontier moved from **one half-suit (6 live cards)** to **two
half-suits everywhere (12 live cards) and roughly two thirds of three-half-suit
positions (18 live cards)**.

## 4. The cyclic game, pinned down

### 4.1 What kind of object a layer is

Asks and passes never resolve a half-suit, so they cycle inside a layer; claims
strictly descend one. Inside a layer the game is therefore: *reach an exit
(a claim) worth v, or play forever for 0*. Formally this is a zero-sum
**stopping game on a finite graph**, equivalently an Everett recursive game
with terminal payoffs and payoff 0 for infinite play, equivalently a
deterministic game whose payoff is the limsup (= liminf here, the payoff
sequence is eventually constant) of a bounded reward stream. Finite
deterministic games with such Borel objectives are determined (Martin) and both
players have positional optimal strategies, so "the value" is well defined.

### 4.2 "Solve to a fixpoint" does not define the value

The Bellman operator of such a game generally has **many** fixpoints. Smallest
example, in `tests4/test_exact2.py::test_a_loopy_bellman_operator_can_have_many_fixpoints`:
two maximiser states in a cycle, one holding a claim worth -1. The value is 0
(loop rather than take a losing claim), but **every constant vector >= -1 is a
fixpoint**, including the all-5 vector, which value iteration leaves untouched.
So v1's line "every non-terminal in-layer state starts at value 0" is not an
implementation note - it is the specification, and it needs a justification.

The justification is exact rather than asymptotic. Let V_k be the value of the
k-step truncated game (payoff 0 if not stopped by k); V_k = T^k(0). If
V* > 0 the maximiser forces an exit worth >= V* within at most |S| moves, so
V_k >= V* for k >= |S|; if V* <= 0 he may instead simply refuse to exit and
truncation only helps him, so again V_k >= V*. Symmetrically V_k <= V*. Hence
**V_k = V* for every k >= |S|**, and any k at which V stops changing already
carries the true value.

v2 therefore does not rely on the fixpoint being unique. Every layer is solved
a second, independent way, by attractors over thresholds, which has no
initialisation and no sweep order at all:

* for c > 0: the maximiser must genuinely REACH an exit worth >= c (looping
  pays 0 < c), so the winning region is `Attr_max(exits >= c)`;
* for c <= 0: looping already pays >= c, so the maximiser loses only if the
  minimiser can FORCE an exit worth < c; the region is the complement of
  `Attr_min(exits < c)`;
* V(s) = max{c : s is in the region for c}, over the values that can actually
  occur (an exit value, or 0).

Every `layer_table` build asserts the two agree. They do, on all 2,604 m = 1
states and all 1,275,960 m = 2 states.

### 4.3 Empirically, the Fish fixpoint IS unique - and that is a fact about Fish

The brief asked whether the computed value depends on the initialisation. It
does not, and this was worth checking rather than assuming, because the general
theory gives no such guarantee.

| initialisation | m = 1 sweeps | m = 2 sweeps | states differing from the attractor value |
|---|---|---|---|
| zeros | 8 | 8 | 0 |
| optimistic (+m) | 8 | 8 | 0 |
| very optimistic (+5) | 12 | 17 | 0 |
| pessimistic (-m) | 8 | 8 | 0 |
| very pessimistic (-5) | 12 | 17 | 0 |
| random x3 | 8 | 17 | 0 |
| Gauss-Seidel from zero, 3 random sweep orders | 6, 7, 7 | (m = 1 only) | 0 |

So v1's answer never depended on its zero start or on its Python dict ordering.
The reason is structural, and follows from section 5: **no Fish position's value
rests on a cycle**, because whoever is on move can always force termination.
The pathological fixpoints of section 4.2 need a state where both sides prefer
looping to every available exit, and Fish has none.

Two consequences worth stating plainly, because the paper currently says
otherwise:

* PAPER.md section 1.1 says the solver "treats an unbroken cycle as scoring
  nothing further - which correctly predicts that a side able only to lose by
  claiming will stall indefinitely". The first half is right and the second is
  not. A side that can only lose by claiming can decline to claim, but it
  cannot stop the opponents from claiming: it keeps the turn only while its
  asks keep hitting, and those run out. Under perfect information **optimal
  play always terminates**. Cycles exist in the graph (v1's cycle-finding test
  still passes, and is reproduced in
  `test_cycles_exist_but_optimal_play_never_needs_them`) but no optimal line
  needs one.
* The "0 for an unbroken cycle" rule is still the right semantics; it is simply
  never load-bearing at m <= 3. It would become load-bearing under rule
  variants that let a team refuse a forced claim.

### 4.4 Bellman-optimal is not the same as optimal

In a loopy game an action that preserves the value can still throw it away, by
walking around a cycle forever. `optimal_actions` (which matches v1's
definition) returns the Bellman-optimal set; `progress_optimal_actions`
additionally requires the side the value favours to strictly descend the
attractor rank. Measured over the full tables, on states with a non-zero value:

| layer | decided states | states with a value-preserving move that makes NO progress |
|---|---|---|
| m = 1 | 2,604 | **0** (0.0%) |
| m = 2 | 1,130,136 | **366,660** (28.6%) |

At m = 1 the two notions coincide, which is why v1 never had to distinguish
them. At m = 2 nearly a third of decided positions admit a value-preserving
stall. The corpus carries both sets, and tags such positions
`stalling_action_is_bellman_optimal` (132 of 988 positions in the shipped
corpus, since real games do not visit the table uniformly).

## 5. What the tables found: the endgame has a closed form

Both full tables match a one-line rule exactly. Let *m* be the number of
unresolved half-suits and *f* the number of them in which the **team on move**
holds at least one card ("has a foothold"):

> **V = sign(mover's team) x (2f - m)**, where sign is +1 for team {0,2,4}.

| layer | states checked | closed form exact | value histogram |
|---|---|---|---|
| m = 1 | 2,604 | yes, 100% | -1: 1,302, +1: 1,302 |
| m = 2 | 1,275,960 | yes, 100% | -2: 565,068, 0: 145,824, +2: 565,068 |
| m = 3 | 61 sampled, solved by reachability | yes, 61/61 | - |
| m = 1..9 | 360 sampled, greedy playout | yes, 360/360 | see below |

The last row is no longer a sample of a conjecture. The sketch below is a
proof, and `scripts4/closed_form_proof.py` checks its three premises against
the engine rather than against a reading of it -- see section 5.1.

The mechanism, which is also a proof sketch:

1. **Footholds only shrink.** Asking for a card of half-suit h requires already
   holding one of h. A team with none of h can never acquire one, so it can
   never claim h.
2. **The team on move takes every half-suit it has a foothold in.** With
   perfect information every ask can be made to hit, and a hit retains the
   turn, so the mover drains all opponent cards from those half-suits without
   ever surrendering the turn, then claims them; when a claim leaves them
   empty-handed they pass to the teammate holding the next foothold.
3. **The opponents take the rest**, because the mover's team is empty-handed
   once its f half-suits are claimed and the turn normalises to an opponent.
4. So the mover's team scores f and the opponents m - f.

This is checked, not just argued: `constructive_playout` plays that exact
strategy for BOTH teams from 160 positions (m = 1 and m = 2) and the realised
set differential equals the table value in **160 / 160** cases, terminating
every time.

**What this means for the project.** The perfect-information Fish endgame is
much shallower than the exact-solver machinery suggests. In particular PAPER.md
section 5.3 reports belief-tracking agents at 100% agreement with exact optimal
play on resolved positions; that benchmark was run entirely on m = 1 positions,
where the exact value is just "the team on move wins the half-suit". The
agreement is real but it certifies less than it appears to. The corpus in
section 6 is 79% two-half-suit positions precisely so that the v0.4 benchmark
is not measuring this.

Note the closed form is a statement about *values*, not about *actions*: which
asks and claims are optimal, and in which order, is not closed-form trivial
(mean 4.3 optimal actions out of 8.1 legal in the corpus), and that is what the
corpus scores agents on.

### 5.1 Why it holds at every m, with no table

Steps 1-4 use only three facts, and each is a property of the RULES rather than
of any position, so none of them needs a layer to be enumerated:

**A. Footholds never grow.** `check_legal` refuses an ask unless
`self.hands[player] & half_suit_mask(hs)`, so a team holding no card of h can
never receive one. Its foothold set only shrinks. Hence the mover's team can
claim at most the f half-suits it has a foothold in *now*.

**B. A team with no foothold in h cannot deny h.** All six cards of h are in
opponents' hands, so in `_apply_claim` the `any(team_of(h) != team)` branch
fires for every assignment it could submit and `winner = 1 - team` -- under
BOTH misdeclaration rules. Under the legacy null variant the further point
was that no spite-null is available: the mover cannot convert a certain -1
into a 0. Under the opponent-award baseline the premise only strengthens,
since nulls do not exist at all (not even the owning team can void its own
set; a wrong order gifts it instead). Hence the opponents take all m - f
regardless of what the mover does.

**C. The mover takes all f without surrendering the turn.** A hit retains the
turn; `_apply_claim` never touches `self.turn`; a claim that empties the
claimant leaves `_normalize_turn` waiting for a `Pass`, which
`legal_passes` restricts to teammates with cards. With perfect information
every ask can be made to hit, so the mover drains and claims each of its f
half-suits in turn, and only when its team is cardless does the turn normalise
to an opponent -- who by symmetry then takes the rest.

A and B give V <= 2f - m; C gives V >= 2f - m.

The optimal perfect-information policy is correspondingly short: *take a
hitting ask if one exists; otherwise claim a half-suit your team wholly owns;
otherwise pass to a teammate with cards.* Played out from 360 random positions
across m = 1 to 9 it realises 2f - m every time, and at m = 1 and m = 2 it
agrees with `Exact2Solver.value` on every position tested -- the one step in
the argument anchored to a solver rather than to internal consistency.

**What the answer actually is.** A team lacks a foothold in a half-suit only if
all six of its cards landed in the opponents' 27, with probability
C(27,6)/C(54,6) = 1.15%. So at a fresh deal

    E[V] = 2 * (9 - 9 * 0.01146) - 9 = +8.794

out of 9. The team on move takes everything. Perfect-information Fish is not a
hard game that the tables partly solve; it is a trivial game, and the whole
difficulty of Fish is the hidden information. `results/determinization_gap.json`
measures that from the other side: the tables overstate the mover by +5.29 sets
on positions real play reaches, +8.18 at a fresh deal. Extending the tablebase
to larger m would therefore buy nothing -- the closed form already answers
every layer, and the answer is the wrong game's answer.

### Deliberately losing claims

v1 generates only TRUE claims, arguing that a knowingly-wrong claim is never
better. Mis-splitting a half-suit your team wholly owns is indeed strictly
dominated (same continuation, 0 banked instead of +1). Handing a half-suit to
the opponents is **not** dominated a priori - it strips cards off the table -
so v2 can generate it (`include_giveaway_claims=True`) and the question was
measured rather than argued:

| layer | states | states where the value changes | states where a give-away claim TIES the optimum |
|---|---|---|---|
| m = 1 | 2,604 | 0 | 0 |
| m = 2 | 1,275,960 | **0** | **84,672 (6.6%)** |

So v1's restriction never changes a value, but at m = 2 it does make
`optimal_actions` an incomplete list of the actions that achieve the value. v2
keeps the v1-compatible action set by default (an agent that gives a half-suit
away is not doing something praiseworthy, and scoring it "optimal" would
flatter it) and records the tie as the corpus field
`giveaway_claim_ties_optimum` so a benchmark author can choose deliberately.

A related modelling choice both solvers make: a cardless player on move is
given only PASS, although SPEC 4.2 lets them claim. That never changes a value,
by the closed form - passing keeps the same team on move with the same
footholds, so 2f - m is unchanged.

## 6. Corpus schema: `results/exact_positions_v2.json`

Harvested from 260 real games played by four different registry agents
(`probabilistic`, `memory`, `tuned`, `heuristic`) rotating game by game, so the
corpus is not shaped by a single policy's habits. Snapshots are taken whenever
1 or 2 half-suits remain live, deduplicated by abstract state.

Top level:

| field | meaning |
|---|---|
| `schema_version` | 2 |
| `generated_by`, `solver` | provenance |
| `generator_agents`, `games_played`, `seed` | how to regenerate exactly |
| `seconds_playing`, `seconds_solving` | cost split |
| `summary` | the counts reproduced below |
| `positions` | the records |

Each record in `positions`:

| field | type | meaning |
|---|---|---|
| `id` | int | index within the file |
| `rules` | dict | `RuleConfig.to_dict()`; feed to `RuleConfig(**rules)` |
| `hands` | 6 ints | per-player card bitmask, the true hidden state |
| `turn` | int | player to move, already turn-normalised |
| `set_winner` | 9 entries | `null` = unresolved, 0/1 = team, -1 = nulled |
| `history` | list | full public log; `{"t":"ask"/"claim"/"pass", ...}` |
| `source_game`, `source_ply` | int | provenance inside the harvest |
| `live_half_suits` | list[int] | the unresolved half-suit indices |
| `n_live_half_suits` | int | 1 or 2 |
| `live_cards` | int | always `6 * n_live_half_suits` |
| `abstract_compositions` | list[6 ints] | per live half-suit, cards per player |
| `information_resolved` | bool | **the key slice**: every live card's location is publicly determined, so the perfect-information optimum IS the optimum and a strong agent should score 100% |
| `mover_team_footholds` | int | the *f* of section 5 |
| `value` | float | exact remaining set differential, team 0's point of view |
| `n_legal_actions` | int | size of the legal action set at this node |
| `optimal_actions` | list | **full** set achieving `value`, v1-compatible action set |
| `progress_optimal_actions` | list | subset that also descends the attractor rank (section 4.4) |
| `giveaway_claim_ties_optimum` | bool | a deliberately losing claim also achieves `value` |
| `tags` | list[str] | see below |

Actions serialise as `{"type":"ask","target":t,"card":c,"card_name":"7C"}`,
`{"type":"claim","half_suit":h,"assignment":[6 player ids]}`, or
`{"type":"pass","teammate":t}`. `fish4.exact2_corpus.action_from_dict` and
`state_from_record` round-trip them; `state_from_record` rebuilds the exact
`GameState` **including the public log**, which is what
`information_resolved` is derived from.

Tags: `cardless_mover`, `forced_claim` (no legal ask exists at all),
`empty_hand_present`, `team_holds_nothing`, `nulled_half_suit_present`,
`stalling_action_is_bellman_optimal`, `giveaway_claim_ties_optimum`.

Shipped corpus (regenerate for exact numbers; `results/exact_positions_v2.json`
carries its own `summary`):

* 988 positions, 278 of them `information_resolved`, 11.7 MB (the public logs
  dominate the file and are not optional: an agent has to be handed the same
  observation a player would have)
* 212 one-half-suit, 776 two-half-suit
* values: -2: 166, -1: 97, 0: 443, +1: 115, +2: 167
* mean 4.27 optimal actions out of 8.06 legal
* tags: `empty_hand_present` 792, `giveaway_claim_ties_optimum` 401,
  `nulled_half_suit_present` 322, `forced_claim` 250,
  `team_holds_nothing` 250, `stalling_action_is_bellman_optimal` 132,
  `cardless_mover` 75

The give-away tie rate is 401 / 776 = 52% of the two-half-suit records against
6.6% of the table as a whole, because real games arrive disproportionately at
the f = 1 positions where a team's second half-suit is already lost and handing
it over costs nothing. It is a good example of why the corpus is harvested from
play rather than sampled from the table.

Scoring an agent: give it `Observation.from_state(state, rec["turn"])`, take
its action, and check membership in `optimal_actions`. Report
`information_resolved` positions separately from the rest, exactly as
`fish.benchmark_exact` does - the perfect-information optimum is not the
optimum under genuine uncertainty, and only the resolved slice licenses a
claim about optimality.

## 7. Correctness summary

| check | scope | result |
|---|---|---|
| value agreement with `fish.exact.ExactSolver` | 300 varied positions | **300 / 300** |
| full optimal-action-set agreement with v1 | same 300 | **300 / 300** |
| abstract successors == engine successors | 600 random positions | 0 mismatches |
| value invariant to which cards realise a composition | 300 states x 3 permutations | 0 mismatches |
| value iteration == attractor solution | every state of both tables | equal, asserted at every build |
| constructive optimal line realises the value, and terminates | 160 positions | 160 / 160 |
| reachable solver == full table | random m = 2 roots, all reached states | equal |
| every decided state has a non-zero progress rank | both tables in full | holds |
| corpus records reproduce their own ground truth | 60 sampled records | equal, and every listed optimal action is legal |

The 300-position cross-check covers, by construction: half-suits held entirely
by one team, forced-claim endings, single-holder and two-holder layouts,
six-way splits, cardless movers, and every position embedded in a random mix of
won, lost and **nulled** resolved half-suits.

## 8. Rejected ideas

**Abstracting to the partition of cards among players (unsound).** The brief
suggested that the value might depend only on how the cards are partitioned.
It does not - which player holds which share decides the teams and the turn
order. Counterexample, one live half-suit: counts `(6,0,0,0,0,0)` with turn 0
is worth **+1**, counts `(0,6,0,0,0,0)` with turn 0 normalises to turn 1 and is
worth **-1**. Same partition {6,0,0,0,0,0}, opposite values.

**Canonicalising the stored corpus positions (unsound for the corpus, sound
for the solver).** The within-half-suit card symmetry is genuine for the
perfect-information value, and the solver uses it. It must not be pushed out to
the stored positions: `information_resolved` is computed from the public log,
and two states with identical counts can differ in whether the log pins the
cards. So the corpus stores real hands and the real history, and abstracts only
inside `Exact2Solver.value`.

**Seat reflection p -> -p (mod 6) (rejected despite passing).** Reflection
preserves teams and, empirically, preserves the value on **all** 1,275,960
m = 2 states. It is nonetheless **not** a game automorphism, because turn
normalisation walks clockwise: with cards at players {1,3} and player 0 on
move, the turn normalises to 1; reflect to {5,3} and it normalises to 3, which
is not the reflection of 1. The value happens to survive only because the
closed form of section 5 depends on team footholds and not on seating
direction. Exploiting it would have been a correct answer reached by an
incorrect argument, which is exactly the failure mode the brief warns about, so
it is not used.

**Seat rotation and half-suit exchange (sound, verified, not needed).** Rotating
every seat by 2 preserves the value and by 1 negates it; exchanging the two
live half-suits preserves it. Both verified over the whole m = 2 table. Together
they would shrink the table by roughly another 6x. They are not implemented
because 1.28 M states already fit comfortably, and every extra canonicalisation
is another place for ground truth to be silently corrupted. They are the first
thing to reach for if m = 3 is ever tabulated in full.

**Attractor sweeps over the full integer threshold range (replaced).** The
first version swept every integer in [-m, m]. Restricting to the values that
can actually occur (exit values plus 0) cut m = 2 from five threshold passes to
three, since a two-half-suit claim is only ever worth -2, 0 or +2. Worth
recording because the change introduced a bug - the rank lookup for a losing
state assumed threshold v+1 existed, which it no longer did, silently emptying
`progress_optimal_actions` on every losing position. It was caught by
`test_progress_optimal_is_a_subset_of_optimal` and is now guarded by
`test_every_decided_state_has_a_progress_rank`, which checks the whole table
rather than a sample.

**Parallelism inside the solver (measured, not worth it).** The m = 2 build is
8 numpy sweeps over 1.28 M states; splitting the state vector over 3 processes
would need the 2.5 MB value vector shipped every sweep and would save a handful
of seconds, once per process. The 3 permitted workers were spent where they
measurably paid: running the v1 baseline, 2,240 CPU-seconds compressed into
769 s of wall clock (2.9x).

**Naive backward induction (v1 already rejected this, still true).** The graph
is cyclic; see `test_cycles_exist_but_optimal_play_never_needs_them`.

## 9. Known limits

* `claims_any_time=True` is refused (`UnsupportedRules`). It makes every player
  a mover at every node, which neither solver models.
* `allow_bluff_asks=True` is supported and builds its own tables; the 48-card
  variant needs no special handling, since half-suits are six cards either way.
* m = 3 goes through the pure-Python reachable solver: about a third of random
  roots exceed a 400,000-state budget. Tabulating m = 3 in full would need the
  rotation and exchange symmetries of section 8 plus an out-of-core layout.
* The tables are rebuilt per process (5-26 s). They are deliberately not
  cached to disk: a stale tablebase silently corrupting ground truth is a worse
  failure than a slow start.
* Everything here solves the **perfect-information** game. Section 5 makes the
  usual caveat sharper rather than weaker: on positions with genuine hidden
  information these values are a determinized upper bound, not the optimum.
