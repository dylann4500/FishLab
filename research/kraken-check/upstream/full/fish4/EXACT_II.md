# The imperfect-information solver: what is load-bearing, and what was tried

`fish4/exact_ii.py` computes an exact best response over information sets at
`m = 1` (and, more slowly, above it) against a deterministic realisation of the
champion. `fish4/EXACT2.md` covers the perfect-information solver; this covers
the one where the deviator cannot see.

## The invariant that does the work

A best response may copy the champion at every information set. So

    best response value  >=  champion value

at every position, with no ground truth required, at any layer, for any prior.
It is the only cheap check that can catch a broken tree, and it is the check
that caught one.

The pinned control -- exact value equals the closed form `2f - m` wherever the
belief pins every card -- cannot. Where the support is a single deal there is no
hidden information, and a fault that needs several deals to show is invisible to
it. Both controls are in `scripts4/ii_endgame.py`; neither replaces the other.

`ExactII.champion_tree_value` is the third: it plays the champion *inside the
recursion* rather than as a rollout, so the tree and `champion_value` evaluate
the same strategy by different code paths and must agree.

## The fault this file exists because of

The transposition key was

    (depth, sorted((hands, turn, set_winner, weight) for each state))

with no history in it. The opponents are the champion, whose action is a
function of its whole observation, so two nodes identical in every one of those
fields but reached by different move orders have different continuations and
different values. Merging them returned one branch's value for the other, and
because the maximisation reads those values, the max came out **below one of its
own options** -- five `m = 2` positions with a negative gain, which cannot
happen.

Scale of the merge, on the position in `tests4/fixtures/ii_memo_position.json`:
154 nodes with the broken key, 12,086 with the fixed one. It reported +0.2500
for a move whose rollout is +0.7500.

The key now carries the path since the root. Keying on `states[0].history`
directly is equally correct and unusably slow -- it re-`repr`s a hundred events
at every node, and a single game did not finish in four minutes.

## The budget is nodes, not seconds

`MAX_NODES = 300_000`, with `DEFAULT_DEADLINE` demoted to a backstop. A
wall-clock budget measures the machine: the first coverage figure after the fix
was 21% of `m = 1` positions over 60s, measured with three studies competing for
four cores, and it would not reproduce on an idle machine or on anyone else's.

## The cutoff, and what it is not

`ExactII._upper` bounds what a node can still pay. An action attaining that
bound ends the action loop -- no window, no stored bound, so every memoised
value is still exact. That last part is the whole reason it is not alpha-beta: a
pruned underestimate written into the memo is indistinguishable from a real one
on the next lookup, which is the fault above wearing a different hat.

Two exemptions, both deliberate:

* **the root**, because `scripts4/ii_action_diff.py` reads `action_values` for
  every action including the champion's, and an unpriced champion move is not
  reported as unpriced -- it is silently reclassified as a disagreement;
* **the root's action order.** Claims go first below the root because they are
  likeliest to attain the bound, but `best_action` is the first maximiser in
  list order, so reordering at the root would change which of several
  tied-optimal moves is reported. 78 of the 185 disagreements in that study cost
  exactly zero, i.e. are ties rather than errors; reordering them moves the
  headline while looking like a speedup.

Worth is the wrong single number, and the first one recorded here was the
wrong number. Over 21 real `m = 1` positions, values, root action values and
the reported optimum are identical throughout, and the node counts are:

| | plain | pruned | |
|---|---|---|---|
| median position | -- | -- | **21%** |
| hardest position | 2,511,080 | 2,855 | **0.1%** |
| all 21 together | 2,998,454 | 20,779 | **1%** |

The first measurement here said "about 5x", taken over seven positions that
happened to be easy. That is the median, and it understates the effect exactly
where it matters: the saving grows with the position, so the cheap positions
that dominate a small sample show 5x while the expensive ones that were blowing
the node budget show several hundred. That is why coverage came back to 98%
rather than merely improving.

At `m = 2` I first recorded it as worth **nothing**, on the strength of one
fixture position that went 12,086 -> 12,086, and gave a reason: the bound is +2
and no action attains it. The reason is wrong. Once one half-suit is banked a
node's bound is +1 for the decided one plus +1 for the rest, and an action
attains it by taking the second -- the same condition as an `m = 1` node, where
actions attain the bound in 79 of 305 positions. Measured over 11 real `m = 2`
positions:

| | plain | pruned | |
|---|---|---|---|
| median position | -- | -- | **38%** |
| best position | -- | -- | **1%** |
| all 11 together | 428,227 | 29,824 | **7%** |

Zero value disagreements, and it saved nodes in 9 of the 11. So roughly
fourteen-fold in aggregate rather than nothing.

That figure **understates** it, for a reason worth naming: the harness runs the
unpruned solver first and drops any position it cannot finish inside the node
budget. Those are precisely the positions where the cutoff helps most, so they
are missing from the sample by construction.

None of this changes `m = 2`'s 32% coverage, because the cutoff was already on
by default when that was measured. The error was documentary -- but a file that
tells the next reader a component is worthless is how a component gets removed.

## Tried, sound, and worthless: the foothold tightening

A team holding no card of half-suit `h` can never hold one -- `legal_asks` in
`fish/engine.py` refuses an ask in a half-suit you have no card of (`if not
mine: continue`), and cards move only by successful asks. So it can never
declare `h` for itself. Under the LEGACY `wrong_distribution_outcome =
"null"` variant a wrong declaration by the other team also scored for
nobody, so such a half-suit could pay the footholdless team at most 0, not
+1 -- the closed form's foothold fact (`scripts4/closed_form_proof.py`,
premise C) reused as a search bound, correct THERE.

**Under the opponent-award baseline this bound is UNSOUND and must not be
resurrected**: the owning team's misdeclaration of `h` now awards it to the
footholdless team, which is a +1 that the bound denies. It survives only
against an infallible declarer, and the champion opponent model misdeclares
with positive probability. Since the tightening was never merged (below),
nothing computes wrongly; this note exists so nobody re-implements it from
the old argument.

Measured against the loose bound: **0 disagreements and 0 nodes saved.** The
`m = 2` fixture went 12,086 -> 12,086; 22 real `m = 1` positions went
28,780 -> 28,780. It never binds, because a deviator with no foothold in the
only live half-suit is in a position nobody reaches with anything left to
decide.

It is not in the code. A tightening that changes no value and saves no node is
complexity with a correctness proof attached, and the proof is the part that
makes it tempting.

## Files

* `fish4/exact_ii.py` -- the solver.
* `scripts4/ii_endgame.py` -- the layered study and its controls.
* `scripts4/ii_action_diff.py` -- champion move vs exact optimum, per decision.
* `scripts4/ii_first_endgame.py` -- one deviation per game, comparable to
  `scripts4/exploitability.py`.
* `scripts4/ii_negative_repro.py` -- reproduces the negative-gain positions.
* `scripts4/ii_memo_effect.py` -- the matched old-vs-new comparison.
* `tests4/test_exact_ii.py` -- the invariants, each checked against the broken
  solver to confirm it can fail.
