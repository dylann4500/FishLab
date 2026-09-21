"""Unbiased sampling of Fish-consistent deals under the OR constraints.

WHY THIS EXISTS
---------------
``counting.py`` solves the candidate-mask and quota system exactly, but the
constraint set also contains OR clauses: "the asker held at least one card of
that half-suit at that moment". Three measurements decide the design.

1. Ignoring the clauses is not an option. On 40 real positions the OR-free exact
   marginals were off by a mean L1 of **0.127 per card** against the
   OR-conditioned truth, which is *worse* than v0.3's biased sampler (0.074).
2. Rejection is not an option either. A typical clause covers about 3.35 live
   cards, each landing with the relevant player about one time in five, so a
   clause holds with probability roughly 0.53 and five clauses hold about 4% of
   the time. Measured acceptance on real positions: mean 0.30, median 0.18, and
   16% of positions below 0.05.
3. Folding the clauses into the exact DP requires refining the mask groups by
   OR membership, and the refined state space has a bad tail: median 5,760 but
   p90 230,000 and p99 27,000,000.

So this module does the thing that is cheap, always applicable, and *unbiased*:
sequential importance sampling with exactly computable proposal probabilities.
Cards are assigned in a fixed order, so each world has exactly one generating
path and its proposal density is the product of the per-card choice
probabilities. Self-normalised importance weights then make every estimate
consistent for the target distribution, which is what v0.3's sampler was
missing: it drew from a similar proposal but reported the draws as if they were
uniform. Support completeness, density correctness, marginal accuracy against
exhaustive enumeration, and the fact that dropping the weights visibly breaks
the estimate are all checked in ``tests4/test_sis.py``.

OPPONENT MODELLING
------------------
The uniform target treats every consistent world as equally likely, which is
correct only if players' *choices* carry no information beyond legality. They
do. A player who asks in a half-suit is more likely to be deep in it, and under
the simplest defensible choice model - a player picks which half-suit to ask in
in proportion to how many cards of it they hold - the choice probability is
``depth / hand_size``, whose denominator is public and therefore constant across
worlds. That makes the likelihood of the observed ask sequence proportional to a
product of depths, which this sampler can accumulate during the draw for the
cost of one integer increment per card. ``gamma`` controls how sharply the model
is believed; ``gamma = 0`` recovers the uniform target exactly, so the whole
idea is a single ablatable number.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

from fish.cards import NUM_PLAYERS


class SISFailure(Exception):
    """The sampler could not complete a draw within its restart budget."""


class OpponentModel:
    """Likelihood of the observed asks under a depth-proportional choice model.

    ``weight[slot]`` is
    ``gamma`` times the number of times that player asked in that half-suit, and
    ``base[slot]`` is the number of cards of the half-suit already pinned to that
    player (so only the free cards need counting per draw).
    """

    __slots__ = ("weight", "base", "n_slots", "set_cards", "opp_lambda",
                 "my_team", "tilt", "depth_table", "convention",
                 "convention_beta", "convention_q", "convention_aim",
                 "convention_book")

    #: Cap on a single step's twist factor. The depth-0 term of the likelihood
    #: is ``log(1e-9)``, so the 0 -> 1 ratio is ``1e9 ** w`` and would otherwise
    #: make the proposal effectively deterministic at that step. Capping costs
    #: variance reduction, never correctness: whatever the proposal does, the
    #: density used in the weight is the one actually sampled from.
    TILT_CAP = 1e4

    #: Depth is bounded by the cards in a half-suit, so the whole tilt table is
    #: this many entries per slot and is built once per decision.
    TILT_MAX_DEPTH = 7

    def __init__(self, weight, base, set_cards=None, opp_lambda: float = 0.0,
                 my_team: int = 0, tilt_strength: float = 0.0,
                 depth_table=None, convention=None,
                 convention_beta: float = 0.0,
                 convention_q: float = 0.0,
                 convention_aim: bool = False,
                 convention_book: str = "depth"):
        self.weight = list(weight)
        self.base = list(base)
        #: (asker, half_suit, card_asked, const_mask, free_cards) per ask by
        #: our own side, where const_mask is the cards publicly known to be
        #: with the asker AT THAT ASK and free_cards are the ones the draw
        #: decides. See fish4/convention.py for why this is a soft weight
        #: rather than a constraint: the receiver cannot know whether a given
        #: ask was carrying a message, so a hard decode would let one
        #: unencoded ask eliminate the true world.
        self.convention = convention
        self.convention_beta = convention_beta
        #: Sender carry rate for the mixture likelihood; 0 means
        #: no agreement exists and the term is skipped entirely.
        self.convention_q = convention_q
        #: Aim the code book at the most-unlocated half-suit
        #: rather than at the one asked in. See
        #: fish4/convention.py: 4.03x the entropy, same cost.
        self.convention_aim = convention_aim
        #: 'depth' | 'locate'. See fish4/convention.py: a count
        #: constrains the joint, only a location pins a card.
        self.convention_book = convention_book
        self.n_slots = len(self.weight)
        #: Per-slot, per-sampled-depth log terms, already scaled. Present only
        #: for the at-ask-time model, where each ask carries its own public
        #: delta and the terms cannot be folded into one weight times one log.
        self.depth_table = depth_table
        # The proposal twist reads weight and base to compute an incremental
        # likelihood ratio, which the table's per-ask structure does not expose.
        # It ships at zero and is a documented negative result, so it is simply
        # unavailable here rather than approximated.
        self.tilt = None if depth_table is not None else \
            self._build_tilt(tilt_strength)
        # Columns (in sampler order) of the still-unlocated cards of each
        # half-suit that COULD be entirely with the opposing team. See the
        # "no-declaration" signal in oppmodel.py.
        #: Tuples of CARD IDS -- see fish4/oppmodel.py. Consumers map them
        #: through their own column order.
        self.set_cards = set_cards or []
        self.opp_lambda = opp_lambda
        self.my_team = my_team

    def _build_tilt(self, strength: float):
        """Per-slot table of ``L(depth+1)/L(depth)``, raised to ``strength``.

        The sampler's proposal is quota-proportional and knows nothing about the
        choice model, so at ``gamma > 0`` the entire tilt of the target lands in
        the importance weights and the batch degenerates. Measured on real
        positions, 160 draws were worth 99.9 effective at ``gamma = 0`` but only
        83.7 at ``gamma = 0.35``.

        This is the standard remedy: fold one step of the likelihood into the
        proposal, so the weight carries only the residual. ``strength = 0``
        returns ``None`` and the sampler takes the untouched incumbent path;
        ``strength = 1`` is the locally optimal twist.

        This term is unlike every other weight in the engine. It changes the
        *proposal*, not the target, so it cannot change what the policy computes
        - only how precisely it computes it. The test is therefore not "does it
        change a decision" but "do the marginals agree while the ESS rises".
        """
        if not strength or not self.n_slots:
            return None
        cap = self.TILT_CAP
        table = []
        for i in range(self.n_slots):
            w = self.weight[i] * strength
            if not w:
                table.append(None)
                continue
            base = self.base[i]
            row = []
            for d in range(self.TILT_MAX_DEPTH):
                lo = base + d
                ratio = (lo + 1) / (lo if lo > 0 else 1e-9)
                try:
                    f = ratio ** w
                except OverflowError:
                    f = cap
                row.append(f if f < cap else cap)
            table.append(tuple(row))
        return table

    def log_convention(self, deal) -> float:
        """Log weight from the pre-play naming agreement, given one drawn deal.

        ``deal`` maps each free card id to the player it was assigned, in
        INITIAL-DEAL space. The holding the code book depends on is the one the
        asker had at the moment of the ask, so each ask carries a precomputed
        ``const_mask`` of the cards publicly known to be with them then, and
        only the never-publicly-moved cards are decided by the draw.

        A world in which the asker would have named the card they DID name is
        multiplied by exp(beta); one in which they would have named something
        else is left alone. beta = 0 returns 0.0 and the model is the incumbent
        exactly.
        """
        conv = self.convention
        q = self.convention_q
        if not conv or not (self.convention_beta or q):
            return 0.0
        from .convention import (depth_in, encoded_position, is_encoded,
                                 legal_cards, locate_payload,
                                 mixture_logp)
        total = 0.0
        for (asker, hs, card, const_mask, free_cards,
             g_hs, g_const, g_free, targets) in conv:
            hand = const_mask
            for c in free_cards:
                if deal.get(c) == asker:
                    hand |= 1 << c
            if self.convention_book == "locate" and targets:
                free = legal_cards(hand, hs)
                hand_all = const_mask
                for c in free_cards:
                    if deal.get(c) == asker:
                        hand_all |= 1 << c
                gh = g_const
                for c in g_free:
                    if deal.get(c) == asker:
                        gh |= 1 << c
                tg = targets[:len(free)]
                match = bool(free) and card == free[
                    locate_payload(hand_all | gh, tg) % len(free)]
            elif self.convention_aim and g_hs is not None:
                gh = g_const
                for c in g_free:
                    if deal.get(c) == asker:
                        gh |= 1 << c
                free = legal_cards(hand, hs)
                match = bool(free) and card == free[
                    encoded_position(depth_in(gh, g_hs), len(free))]
            else:
                match = is_encoded(hand, hs, card)
            if q > 0.0:
                total += mixture_logp(6 - depth_in(hand, hs), match, q)
            elif match:
                total += self.convention_beta
        return total

    def log_likelihood_from_depths(self, depth) -> float:
        table = self.depth_table
        if table is not None:
            total = 0.0
            last = len(table[0]) - 1 if table else 0
            for i in range(self.n_slots):
                d = depth[i]
                total += table[i][d if d <= last else last]
            return total
        total = 0.0
        for i in range(self.n_slots):
            w = self.weight[i]
            if w:
                d = depth[i] + self.base[i]
                total += w * math.log(d if d > 0 else 1e-9)
        return total


class SISSampler:
    """Sequential importance sampler over initial-deal assignments.

    ``card_masks`` maps each free card id to its 6-bit candidate mask,
    ``quotas`` gives the residual per-player deal counts, and ``ors`` is a list
    of ``(tuple_of_card_ids, player)`` clauses (clauses already satisfied with
    certainty must have been dropped by the propagator, which they are).
    """

    __slots__ = ("free", "masks", "quotas", "boost", "max_restarts", "ors",
                 "order", "_by_card", "_cand", "_or_player", "_n", "_ocand",
                 "_oclause", "_oc_player", "_otilt", "opponent_model",
                 "_n_slots", "_batch_plan")

    def __init__(self, free_cards, card_masks, quotas, ors,
                 boost: float = 3.0, max_restarts: int = 40,
                 opponent_model: Optional[OpponentModel] = None,
                 card_slot=None):
        self.opponent_model = opponent_model
        self.free = list(free_cards)
        self.masks = dict(card_masks)
        self.quotas = list(quotas)
        self.boost = boost
        self.max_restarts = max_restarts
        self.ors = []
        for cards, p in ors:
            live = tuple(c for c in cards
                         if c in self.masks and (self.masks[c] >> p & 1))
            if live:
                self.ors.append((live, p))
        self._or_player = [p for _, p in self.ors]
        # Assign the most constrained cards first: fewer candidates and
        # membership in a clause both make a card harder to place, and placing
        # hard cards early is what keeps the restart rate near zero (measured
        # 0 failures in 10,050 draws on real positions).
        membership: dict[int, int] = {}
        for live, _ in self.ors:
            for c in live:
                membership[c] = membership.get(c, 0) + 1
        self.order = sorted(
            self.free,
            key=lambda c: (self.masks[c].bit_count() - membership.get(c, 0), c))
        self._by_card: dict[int, list[int]] = {}
        for i, (live, _) in enumerate(self.ors):
            for c in live:
                self._by_card.setdefault(c, []).append(i)
        self._cand = {c: tuple(p for p in range(NUM_PLAYERS)
                               if self.masks[c] >> p & 1)
                      for c in self.free}
        # Flattened, order-indexed views: the inner loop runs millions of times
        # per game, and dict lookups there measured 219us per draw before this
        # was flattened.
        self._n = len(self.order)
        self._ocand = [self._cand[c] for c in self.order]
        self._oclause = [tuple(self._by_card.get(c, ())) for c in self.order]
        self._oc_player = [tuple(self._or_player[i] for i in cl)
                           for cl in self._oclause]
        self._batch_plan = None
        self._n_slots = 0 if opponent_model is None else opponent_model.n_slots
        if card_slot is None:
            self._otilt = None
        else:
            # slot index for (candidate player, this card's half-suit), aligned
            # with _ocand so the inner loop is a single tuple index
            self._otilt = [tuple(card_slot.get((p, c), -1) for p in cand)
                           for c, cand in zip(self.order, self._ocand)]

    # -- one draw ------------------------------------------------------------

    def draw(self, rng: random.Random) -> tuple[dict, float, float]:
        """Return ``(assignment, log_proposal_density, log_likelihood)``.

        Restarts on a dead end. Restarting conditions the proposal on success,
        which multiplies every density by the same constant, so self-normalised
        weights are unaffected.
        """
        for _ in range(self.max_restarts):
            out = self._attempt(rng)
            if out is not None:
                return out
        raise SISFailure("sequential sampler exhausted its restart budget")

    def _attempt(self, rng: random.Random):
        quota = self.quotas[:]
        order = self.order
        ocand = self._ocand
        oclause = self._oclause
        ocp = self._oc_player
        otilt = self._otilt
        om = self.opponent_model
        tilt = om.tilt if (om is not None and otilt is not None) else None
        boost = self.boost
        n_or = len(self.ors)
        live_left = [len(live) for live, _ in self.ors]
        satisfied = [False] * n_or
        depth = [0] * self._n_slots if self._n_slots else None
        picks = [0] * self._n
        prodq = 1.0
        random_ = rng.random
        wbuf = [0.0] * NUM_PLAYERS
        for i in range(self._n):
            clauses = oclause[i]
            forced = -1
            if clauses:
                for j, ci in enumerate(clauses):
                    if not satisfied[ci] and live_left[ci] == 1:
                        p = ocp[i][j]
                        if forced >= 0 and forced != p:
                            return None      # two clauses need the same card
                        forced = p
            if forced >= 0:
                if quota[forced] <= 0:
                    return None
                pick = forced
                pick_pos = -1
                # a forced choice carries probability 1: prodq is unchanged
            else:
                cand = ocand[i]
                total = 0.0
                k = 0
                for p in cand:
                    q = quota[p]
                    if q <= 0:
                        wbuf[k] = 0.0
                    else:
                        w = float(q)
                        if clauses:
                            cps = ocp[i]
                            for j, ci in enumerate(clauses):
                                if not satisfied[ci] and cps[j] == p:
                                    w *= boost
                                    break
                        if tilt is not None:
                            # One step of the choice-model likelihood, folded
                            # into the proposal. prodq below still records the
                            # density actually sampled from, so the estimator is
                            # untouched and only its variance moves.
                            slot = otilt[i][k]
                            if slot >= 0:
                                row = tilt[slot]
                                if row is not None:
                                    d = depth[slot]
                                    w *= row[d] if d < len(row) else row[-1]
                        wbuf[k] = w
                        total += w
                    k += 1
                if total <= 0.0:
                    return None
                r = random_() * total
                acc = 0.0
                pick = -1
                pick_pos = -1
                for k in range(len(cand)):
                    w = wbuf[k]
                    if w <= 0.0:
                        continue
                    acc += w
                    if r < acc:
                        pick = cand[k]
                        pick_pos = k
                        prodq *= w / total
                        break
                if pick < 0:
                    # float edge: take the last positive-weight candidate
                    for k in range(len(cand) - 1, -1, -1):
                        if wbuf[k] > 0.0:
                            pick = cand[k]
                            pick_pos = k
                            prodq *= wbuf[k] / total
                            break
                    if pick < 0:
                        return None
            picks[i] = pick
            quota[pick] -= 1
            if depth is not None and otilt is not None:
                if pick_pos < 0:
                    cand = ocand[i]
                    pick_pos = cand.index(pick)
                slot = otilt[i][pick_pos]
                if slot >= 0:
                    depth[slot] += 1
            if clauses:
                cps = ocp[i]
                for j, ci in enumerate(clauses):
                    live_left[ci] -= 1
                    if cps[j] == pick:
                        satisfied[ci] = True
        for q in quota:
            if q != 0:
                return None
        for s in satisfied:
            if not s:
                return None
        deal = {}
        for i in range(self._n):
            deal[order[i]] = picks[i]
        logq = math.log(prodq) if prodq > 0.0 else -1e300
        logl = 0.0
        if self.opponent_model is not None:
            if depth is not None:
                logl = self.opponent_model.log_likelihood_from_depths(depth)
            # The convention needs the ASSIGNMENT, not the depth counts: which
            # card was named depends on which cards of the half-suit the asker
            # holds, not merely how many.
            logl += self.opponent_model.log_convention(deal)
        return deal, logq, logl


class WeightedSample:
    """A batch of importance-weighted draws, with diagnostics.

    Draws are carried as an integer matrix ``picks`` of shape
    ``(n_draws, n_free)`` in ``order`` column order, because every consumer
    wants a vectorised view; the dictionary form is materialised lazily and
    only for callers that ask. Building 128 dicts of 26 entries per decision
    was, after the sampler itself was vectorised, the largest remaining cost.
    """

    __slots__ = ("_deals", "logw", "w", "ess", "picks", "order", "n")

    def __init__(self, deals, logq, logl=None, picks=None, order=None):
        self._deals = deals
        self.picks = picks
        self.order = order
        self.n = (len(deals) if deals is not None
                  else (0 if picks is None else picks.shape[0]))
        if not self.n:
            self.logw = np.zeros(0)
            self.w = np.zeros(0)
            self.ess = 0.0
            self._deals = []
            return
        lw = -np.asarray(logq, dtype=np.float64)      # weight is L(x)/q(x)
        if logl is not None:
            lw = lw + np.asarray(logl, dtype=np.float64)
        lw -= lw.max()
        w = np.exp(lw)
        s = w.sum()
        self.w = w / s if s > 0 else np.full(len(w), 1.0 / len(w))
        self.logw = lw
        # Kish effective sample size: how many equally-weighted draws this batch
        # is worth. Reported rather than assumed, because the whole point of the
        # scheme is that the cost of steering is measurable.
        self.ess = float(1.0 / np.sum(self.w ** 2))

    @property
    def deals(self):
        if self._deals is None:
            order = self.order
            self._deals = [{c: int(row[j]) for j, c in enumerate(order)}
                           for row in self.picks]
        return self._deals

    def __len__(self) -> int:
        return self.n


def sample_batch(sampler: SISSampler, rng: random.Random, n: int,
                 vectorised: bool = True) -> WeightedSample:
    """``n`` weighted draws.

    The vectorised path advances every draw through the card order in lockstep,
    which is the same algorithm executed differently; it exists because the
    scalar loop measured 67% of the policy's total runtime. Set
    ``vectorised=False`` to use the reference implementation, which the tests do
    when comparing the two.
    """
    if n <= 0:
        return WeightedSample([], [], [])
    if vectorised and sampler._n > 0:
        from .sisbatch import draw_batch
        picks, logq, logl, alive = draw_batch(sampler, rng, n)
        keep = np.asarray(alive, dtype=bool)
        if keep.any():
            return WeightedSample(None, logq[keep], logl[keep],
                                  picks=picks[keep], order=sampler.order)
        # fall through to the scalar path if every draw died, which should not
        # happen on a feasible system but must not silently return nothing
    deals, logs, lls = [], [], []
    for _ in range(n):
        try:
            d, lq, ll = sampler.draw(rng)
        except SISFailure:
            break
        deals.append(d)
        logs.append(lq)
        lls.append(ll)
    return WeightedSample(deals, logs, lls)


def weighted_marginals(batch: WeightedSample, free_cards, n_cards: int) -> np.ndarray:
    """``(n_cards, 6)`` array of P(card was dealt to player), rows for free cards
    only; everything else is left at zero for the caller to fill in."""
    M = np.zeros((n_cards, NUM_PLAYERS), dtype=np.float64)
    if not len(batch):
        return M
    for w, deal in zip(batch.w, batch.deals):
        for c in free_cards:
            M[c, deal[c]] += w
    return M
