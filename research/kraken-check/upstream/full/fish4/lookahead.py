"""Lookahead over the *belief*, not over sampled worlds.

WHY THIS IS THE ONE SEARCH DESIGN NOT YET TRIED
-----------------------------------------------
Every search this project has attempted - PIMC, Information-Set MCTS, the
value-network variants - evaluates a candidate move by sampling hidden layouts
and solving each one. v0.3 measured why they all failed: the standard deviation
of a position's value across possible layouts is about 2.4x the gap between the
best and the worst candidate move, so a search that scores different candidates
against different sampled worlds is ranking sampling noise. Common random
numbers removed the deficit and never produced a surplus.

The lookahead here never samples. Its state is the posterior marginal matrix
``M[card, player]``, and its transitions are exact local edits of that matrix.
Given a belief it is a deterministic function - run it twice, get the same
answer - so the variance that killed the previous attempts is structurally
absent rather than merely reduced.

WHAT IT COMPUTES: THE POSSESSION CHAIN
--------------------------------------
A Literature turn is a run of asks that ends the instant one misses. Greedy
scoring maximises the immediate P(success); what a possession is actually worth
is how many cards it banks before the turn flips. Those differ whenever taking
one card changes what else is askable::

    W(B, d) = max over legal asks a of  p_a * [1 + W(B | a succeeded, d-1)]

``W`` is in units of *cards expected to be banked in the rest of this
possession*. The failure branch contributes nothing to ``W`` because a miss
ends the possession, and what a miss costs positionally is already priced by
the incumbent objective's turn-risk and exposure terms.

HOW IT COMBINES WITH THE INCUMBENT OBJECTIVE
--------------------------------------------
It does not replace it. The paper's sharpest negative result is that replacing
P(success) with a learned half-suit value - correct units, calibrated model -
costs 7.4 sets per deal-pair. The lesson taken here is that P(success) is the
objective and everything else is a tie-break, so the lookahead enters as an
additive bonus::

    score(a) = incumbent_score(a) + w_lookahead * p_a * W(B | a succeeded, d-1)

At ``depth <= 1``, or at ``w_lookahead == 0``, the bonus is identically zero and
the policy reproduces the incumbent decision for decision. That is not a
convention, it is the ablation discipline this project enforces by test: a
weight of zero must reduce to the baseline exactly, or the cell is measuring two
changes at once.

THE SECOND CURRENCY: DECLARABILITY AT THE LEAF (``w_declare``)
--------------------------------------------------------------
``W`` above counts CARDS. Cards are not what the game pays for -- half-suits
are -- and the project's error ledger says where the loss is: 0.1676 of our
0.1759 wrong declarations a game are ALLOCATION class, our team holding all six
and naming the wrong split, against 0.0083 ownership errors.

`prereg/declaration_timing.md` then priced that. Routing a teammate-oracle to
one decision at a time: the declaration channel alone is worth +1.08 sets a
game, the ask channel alone +0.76, and BOTH together +3.41. D + K = 1.84
against T = 3.41 -- **46% of the prize, 1.57 sets a game, is in neither channel
alone.** It is interaction, and only an intervention that COUPLES asking to
declaring can reach it.

`prereg/locate_term.md` tried the cheap coupling: an additive one-ply ask
feature priced by the declaration it enables. Null at 3,000 pairs, +0.047
[-0.075, +0.168], and diagnosed rather than shrugged at -- it re-ranked 3.9% of
asks by 1% of the objective's scale, because a small additive bonus cannot
out-argue a P(success) term at weight 1.0.

This is the same coupling moved to where it can compound. ``w_declare`` prices
each edge of the possession chain by the DECLARABILITY it creates (see
:func:`declarability`), so a chain that ends with a half-suit our team can name
the split of is worth more than a chain that banks the same number of cards
scattered across five. The search multiplies, where a feature could only add.

Zero by default, and at zero the tree is the same shape, the same cost, and the
same numbers it has always been -- including the inert-last-ply short-circuit.

A CAUTION, STATED UP FRONT
--------------------------
There is a scheduling argument that predicts this will do nothing. If the
candidate asks were independent and all remained available regardless of which
you took first, then ordering them by descending p_a already maximises the
expected number banked before the first miss, and greedy is optimal - no
lookahead can improve on it. The bonus can only pay where that argument breaks:

1. asks are *not* independent - taking a card edits the belief other asks are
   scored against;
2. asks *disappear* - a card taken cannot be asked for again, and a half-suit
   can be exhausted;
3. which opponent receives the turn on a miss depends on the ask.

So this module is built to test a specific hypothesis, and the scheduling
argument above is the null it is being tested against.

INFORMATION BOUNDARY
--------------------
Every input is observation-derived: the posterior marginals (an inference from
public events plus this seat's own hand), our own hand, public hand counts, and
which half-suits are resolved. ``GameState`` is not reachable from here, and the
hypothetical future states the search builds are edits of the *belief*, never
peeks at a layout.
"""

from __future__ import annotations

import numpy as np

from fish.cards import NUM_PLAYERS, half_suit_mask, mask_to_cards, team_of

#: Default plies of possession to look ahead. Depth 1 is the incumbent.
DEFAULT_DEPTH = 3
#: Default number of candidate asks expanded per ply, in the CONTINUATION only.
#: A real mid-game position offers a median of 45 legal asks (mean 45.0, p90 81,
#: max 102, measured over 615 decisions), so an unbeamed depth-3 tree would be
#: ~45^3 nodes for a quantity dominated by its best few branches.
#:
#: The root ply is deliberately not beamed: lookahead_bonus has to return a
#: bonus for every candidate the policy is ranking. A naive depth-3 tree would
#: therefore cost n(1 + b + b^2) = 21n, about 945 expansions at the median n,
#: not the 84 a uniformly-beamed tree would give. Short-circuiting the inert
#: last ply (see possession_value) brings that to n(1 + b) = 225.
DEFAULT_BEAM = 4

# WHERE THE TIME ACTUALLY GOES, AND ONE OPTIMISATION NOT TO ATTEMPT
# -----------------------------------------------------------------
# Rebuilding the candidate list at every internal node, and sorting all ~45 of
# them to read 4, looks like the dominant cost and profiles at ~40% of
# possession_value under cProfile. It is not, and the obvious fix is a no-op.
# Recorded here so the next person does not spend a day on it:
#
#   * Wall-clock (perf_counter, not cProfile) splits it as generation 9.8% and
#     ranking 12.6% - about half the profiled figure. cProfile over-charges
#     per-call overhead across ~1.9M lambda calls.
#   * Since the last ply short-circuits before legal_asks is reached, most
#     nodes never build a list at all.
#   * An incremental-candidate implementation, verified exact, measured
#     0.998x / 1.003x / 1.015x - break-even. The delta maintenance at every
#     apply_success costs what the avoided rebuild saves.
#
# The real hot spot was apply_success -> _rebalance -> _renormalise_rows, which
# is what the two optimisations above address.


def declarability(M, team, live, n_hs: int) -> float:
    """Expected number of half-suits our team could name the SPLIT of, now.

    THE QUANTITY, AND WHY IT IS NOT ``p_team_all``
    ----------------------------------------------
    ``prod_c sum_{p in team} M[c, p]`` is the probability our team OWNS a
    half-suit -- the quantity the `claim` term prices and the one the claim
    evaluator gates ownership on. This is instead::

        prod_c max_{p in team} M[c, p]

    the probability that, naming for each card the teammate most likely to hold
    it, we get all six right. It is at most the ownership product, and the two
    are equal exactly when every card's team mass sits on a single teammate.

    THAT GAP IS THE PROJECT'S ERROR LEDGER, WRITTEN IN BELIEF TERMS. 0.1676 of
    our 0.1759 wrong declarations a game are ALLOCATION class -- our team held
    all six and we named the wrong split -- against 0.0083 ownership errors. A
    quantity built on team ownership is blind to 95% of what actually goes
    wrong; the difference between the two products above is precisely the part
    it cannot see.

    WHY A MAX AND NOT AN ENTROPY
    ----------------------------
    The declaration is a single all-or-nothing guess at one assignment, so what
    matters is the probability of the modal assignment, not the spread around
    it. An entropy would score a half-suit split 0.5/0.5 between two teammates
    the same as one split 0.5/0.25/0.25, and the first is strictly the better
    position to declare from.

    Resolved half-suits contribute nothing: they cannot be declared again.
    """
    best = M[:, team].max(axis=1)
    tot = 0.0
    for hs in range(n_hs):
        if live[hs]:
            tot += float(best[hs * 6:hs * 6 + 6].prod())
    return tot


class ChainState:
    """A hypothetical continuation of our own possession, held as a belief.

    Mutable by design and always restored: the search descends by applying an
    edit and ascends by undoing it, so one array is reused for the whole tree
    rather than copied per node.
    """

    __slots__ = ("M", "hand", "counts", "me", "live", "n_hs", "couple",
                 "team", "_undo")

    def __init__(self, M: np.ndarray, hand: int, counts, me: int, live,
                 n_hs: int, couple: bool = True):
        self.M = M.copy()
        self.hand = hand
        self.counts = list(counts)
        self.me = me
        self.live = list(live)
        self.n_hs = n_hs
        self.couple = couple
        #: Our own three seats, held once so the declarability evaluation is
        #: not recomputing a parity test at every node of the tree.
        self.team = [p for p in range(NUM_PLAYERS)
                     if team_of(p) == team_of(me)]
        self._undo: list = []

    def declarability(self) -> float:
        """This state's declarability. See the module function of that name."""
        return declarability(self.M, self.team, self.live, self.n_hs)

    # -- candidate generation -------------------------------------------------

    def legal_asks(self, allow_bluff: bool = False):
        """Mirror of ``Observation.legal_asks`` over the hypothetical hand.

        Same three conditions the engine enforces: the half-suit is unresolved,
        we hold at least one card of it, and (absent bluffing) we do not hold
        the card asked for. Targets are opponents who still have cards.
        """
        opps = [o for o in range(NUM_PLAYERS)
                if team_of(o) != team_of(self.me) and self.counts[o] > 0]
        if not opps:
            return []
        out = []
        for hs in range(self.n_hs):
            if not self.live[hs]:
                continue
            hs_mask = half_suit_mask(hs)
            if not self.hand & hs_mask:
                continue
            askable = hs_mask if allow_bluff else (hs_mask & ~self.hand)
            for card in mask_to_cards(askable):
                for o in opps:
                    out.append((o, card))
        return out

    # -- transitions ----------------------------------------------------------

    def apply_success(self, target: int, card: int) -> None:
        """The ask landed: the card is now provably ours.

        WHY THE QUOTA COUPLING IS NOT OPTIONAL
        --------------------------------------
        Collapsing the asked card's row to a point mass on us is the exact and
        obvious part. On its own it is also *inert*, and that is worth spelling
        out, because it determines whether this whole module can do anything at
        all.

        A success edits only the asked card's row, so if nothing else moved, the
        success probability of every other candidate would be unchanged for the
        rest of the possession. The search would then be maximising

            p_1 * (1 + p_2 * (1 + ...))

        over a *fixed* set of numbers, and an exchange argument makes descending
        p optimal: the lookahead would provably collapse to greedy, up to the
        two bookkeeping effects of a target running out of cards and of one card
        being askable from several targets. It would be a no-op by construction,
        and measuring it would be measuring nothing.

        What actually breaks that is the quota system. Learning that the target
        held this card makes it *less* likely they hold the others: they are now
        known to hold one card fewer, and that mass has to go somewhere. So we
        scale the target's column down to the count they now have and let each
        affected row renormalise across the remaining players - one step of the
        same proportional fitting the exact posterior does globally. It is this
        coupling, not the point mass, that lets a second ask be worth more after
        one ask than it was before, which is the only thing a possession-chain
        search can see that greedy cannot.

        ``couple=False`` disables it, and is retained precisely so the claim
        above can be measured rather than argued.
        """
        self._undo.append((card, target, self.M.copy(), self.hand))
        self.M[card] = 0.0
        self.M[card, self.me] = 1.0
        self.hand |= 1 << card
        self.counts[target] -= 1
        self.counts[self.me] += 1
        if self.couple:
            self._rebalance(target)

    def _rebalance(self, target: int) -> None:
        """Shrink ``target``'s column to their new count, then renormalise rows.

        Before the ask the target's expected holding over still-unlocated cards
        summed to their hand count; having given one up it must sum to one less.
        Rows that lose mass get it back spread over the other players in
        proportion to what they already had, which keeps every row a
        distribution and is exactly what a single proportional-fitting sweep
        does. Rows that are already certain carry no free mass and are left
        alone by the same arithmetic.
        """
        M = self.M
        mass = float(M[:, target].sum())
        want = float(self.counts[target])
        if mass <= 1e-12:
            return                      # we already knew where all of theirs was
        # want >= mass would mean inventing probability the counts do not
        # support, so the sweep only ever shrinks.
        M[:, target] *= max(0.0, min(1.0, want / mass))
        self._renormalise_rows()

    def _renormalise_rows(self) -> None:
        """Return each row to a distribution, keeping relative odds elsewhere.

        A row whose whole mass sat on the shrunk column would renormalise
        straight back to where it started, which is right: a publicly located
        card cannot be made less certain by a quota argument. A row left with no
        mass at all is impossible under the counts and is passed over rather
        than turned into a division by zero.
        """
        M = self.M
        # Fused: `M[live] /= tot[live][:, None]` is a get-copy, divide and
        # scatter-back through boolean fancy indexing, allocating two
        # temporaries per call in the hottest function in the module.
        tot = M.sum(axis=1)[:, None]
        np.divide(M, tot, out=M, where=tot > 1e-12)

    def undo(self) -> None:
        card, target, M, hand = self._undo.pop()
        self.M[:] = M
        self.hand = hand
        self.counts[target] += 1
        self.counts[self.me] -= 1


#: Optional instrumentation hook, ``None`` in every shipped path. Set it to a
#: callable ``(depth, n_branches, chosen_index, qs) -> None`` to record what
#: search picked at each multi-branch node; ``chosen_index == 0`` is the greedy
#: choice, since the branches are sorted by descending probability. It exists so
#: Proposition 1's empirical shadow can be MEASURED rather than asserted. The
#: cost when unset is one identity comparison per multi-branch node.
_RECORDER = None


def possession_value(state: ChainState, depth: int, beam: int,
                     allow_bluff: bool = False,
                     w_declare: float = 0.0) -> float:
    """Expected cards banked in the remainder of this possession.

    Zero at ``depth <= 0`` and at a position with no legal ask, which is what
    makes the depth-1 policy identical to the incumbent: the bonus term it
    multiplies is this function at depth 0.

    ``w_declare`` ADDS A SECOND CURRENCY, AND ONE PER EDGE, NOT PER LEAF
    -------------------------------------------------------------------
    With ``w_declare > 0`` the recursion becomes::

        V(B, d) = max_a  p_a * [ 1 + w * (D(B|a) - D(B)) + V(B|a, d-1) ]

    where ``D`` is :func:`declarability`. ``w`` is an exchange rate: how many
    banked cards one half-suit made declarable is worth.

    The delta is attached to the EDGE that causes it, not evaluated once at the
    leaf, and the difference is not cosmetic. A leaf evaluation
    ``D(leaf) - D(root)`` would discount the first ask's own gain by the
    probability of every ask after it -- so an ask that single-handedly
    completes a declarable half-suit but leaves no follow-up would be scored
    near zero, which is backwards. Per edge, each gain is discounted by exactly
    the chain that must land to reach it, and the terms still telescope to
    ``D(leaf) - D(root)`` when every ask succeeds.

    It is also the `concent` v1 -> v2 lesson, which this project has now paid
    for twice: score the CHANGE a move causes, never the LEVEL it leaves
    behind. A level would make ``V`` a rescaling of ``p_a`` -- the objective
    already carries P(success) at weight 1.0 and does not need it twice.

    ``D`` is non-decreasing along a successful chain, so ``V`` stays >= 0: a
    taken card's row becomes a point mass on us, and the quota rebalance
    divides every row by a total below one, which can only raise a teammate's
    entry. ``tests4/test_declare_leaf.py`` asserts it rather than trusting it.
    """
    if depth <= 0:
        return 0.0
    asks = state.legal_asks(allow_bluff)
    if not asks:
        return 0.0

    M = state.M
    if depth == 1 and not w_declare:
        # The last ply is inert. Its continuation is possession_value(.., 0),
        # which is 0 unconditionally, so q = p * (1 + 0) = p and the maximum
        # over the beam is just the largest probability - which the sort key
        # would compute anyway. Expanding it did up to `beam` full
        # apply_success/undo cycles, each an M.copy() plus a column scale and a
        # 54-row renormalise, to rediscover a number already in hand. With
        # depth 3 and beam 4 roughly three quarters of all expansions sit here.
        return max([0.0] + [float(M[c, t]) for t, c in asks])
    # The last ply stops being inert the moment declarability is priced: its
    # continuation is 0 but its EDGES still create declarability, so the
    # short-circuit above is gated on `w_declare` rather than removed. At
    # w_declare == 0 the champion's tree is the same shape and the same cost it
    # has always been, which is what keeps this weight a true ablation.
    d_here = state.declarability() if w_declare else 0.0
    # Sorted by descending p, which is the exact ordering when w_declare == 0
    # and a heuristic one above it: a branch with modest p and a large
    # declarability gain can fall outside the beam. Not corrected, because
    # correcting it means evaluating D for all ~45 candidates at every node to
    # choose 4. The cost is bounded by where it applies -- lookahead_bonus does
    # NOT beam the root ply, so every candidate the policy ranks gets its own
    # exact edge gain; the beam only prunes how the continuation is valued.
    scored = sorted(asks, key=lambda a: -float(M[a[1], a[0]]))
    best = 0.0
    best_i = -1
    qs = [] if _RECORDER is not None else None
    for i, (target, card) in enumerate(scored[:beam]):
        p = float(M[card, target])
        if p <= 0.0:
            if qs is not None:
                # Keep qs aligned with `scored`, or a recorder indexing it by
                # branch reads a different branch's value. A skipped branch is
                # not a zero-valued one; it is not a candidate at all.
                qs.append(None)
            continue
        state.apply_success(target, card)
        try:
            gain = (state.declarability() - d_here) if w_declare else 0.0
            cont = possession_value(state, depth - 1, beam, allow_bluff,
                                    w_declare)
        finally:
            state.undo()
        # A miss ends the possession, so it contributes no cards. What it costs
        # positionally is priced by the incumbent objective, not here.
        q = p * (1.0 + w_declare * gain + cont)
        if qs is not None:
            qs.append(q)
        if q > best:
            best, best_i = q, i
    if _RECORDER is not None and len(scored) > 1:
        # `scored` is sorted by descending p, so index 0 IS the greedy choice
        # and best_i != 0 is precisely a node where the search departed from it.
        # Proposition 1 says that cannot happen with the coupling off, and the
        # module has always said so; until now nothing stored the count, so the
        # empirical shadow of the proposition was a number in the paper with no
        # file behind it.
        _RECORDER(depth, len(scored), best_i, qs)
    return best


def lookahead_bonus(ctx, asks, depth: int = DEFAULT_DEPTH,
                    beam: int = DEFAULT_BEAM, couple: bool = True,
                    w_declare: float = 0.0) -> np.ndarray:
    """Per-ask lookahead bonus, in units of expected extra cards banked.

    ``ctx`` is an :class:`~fish4.askfeat.DecisionContext`. The returned vector
    is all zeros whenever ``depth <= 1``, so a caller that adds it to the
    incumbent score is guaranteed to reproduce the incumbent at depth 1.

    THAT ZERO HOLDS AT ``depth <= 1`` EVEN WITH ``w_declare > 0``, and it is a
    deliberate refusal rather than an oversight. A depth-1 declarability bonus
    would be an additively weighted ONE-PLY feature, and
    `prereg/locate_term.md` measured that family and closed it: 3,000 pairs,
    +0.047 [-0.075, +0.168], re-ranking 3.9% of asks by 1% of the objective's
    scale. Whatever this term is worth, it is not worth anything at one ply,
    and a knob that let it be run there would invite exactly the run that has
    already been done. Depth 2 is the shallowest live setting; the depth ladder
    at fixed ``w_declare`` is what separates "the quantity" from "the search".

    The root ply is NOT beamed -- every candidate the policy ranks gets its own
    exact edge gain -- so the beam's p-ordering (see possession_value) prunes
    only the continuation.
    """
    n = len(asks)
    out = np.zeros(n)
    if depth <= 1 or n == 0:
        return out

    obs = ctx.obs
    live = [w is None for w in obs.set_winner]
    allow_bluff = bool(obs.rules.allow_bluff_asks)
    state = ChainState(ctx.M, obs.hand, obs.hand_counts, obs.player,
                       live, ctx.n_hs, couple=couple)
    d_root = state.declarability() if w_declare else 0.0

    for i, a in enumerate(asks):
        p = float(ctx.M[a.card, a.target])
        if p <= 0.0:
            continue
        state.apply_success(a.target, a.card)
        try:
            gain = (state.declarability() - d_root) if w_declare else 0.0
            cont = possession_value(state, depth - 1, beam, allow_bluff,
                                    w_declare)
        finally:
            state.undo()
        # No `1 +` here, and the declarability gain is treated the same way:
        # the incumbent objective already prices this ask's own P(success), so
        # crediting the card again would double-count it. The gain the ROOT ask
        # itself creates is NOT double-counted by anything, though, which is
        # why it is added -- `locate` was the only term that ever priced it and
        # it is not in the shipped basis.
        out[i] = p * (w_declare * gain + cont)
    return out
