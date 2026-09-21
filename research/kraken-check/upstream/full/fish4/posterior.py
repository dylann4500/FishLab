"""Posterior over hidden hands: exact where possible, unbiased everywhere.

This is the v0.4 replacement for v0.3's sampled beliefs. It sits on top of the
v0.3 constraint propagator (``fish.beliefs.BeliefState``), which is sound and
well tested, and replaces the *inference* layer:

  v0.3                                v0.4
  ----                                ----
  heuristic sampler, non-uniform      exact DP when no OR clause is active,
                                      unbiased importance sampling otherwise
  marginals from 32 draws             closed form, or weighted draws with a
                                      reported effective sample size
  claim probability from 32 draws     exact whenever the clause set is empty,
                                      and exactly 0/1 whenever deduction
                                      settles it either way
  bias never quantified               bias eliminated; every approximation is
                                      counted in ``PosteriorStats``

THE SPLIT, AND WHY IT IS WHERE IT IS
------------------------------------
The constraint system has two parts. Candidate masks plus exact per-player deal
counts are solved *exactly* by the dynamic program in ``counting.py``. The OR
clauses ("the asker held at least one card of that half-suit at the time") are
not absorbed by that DP, and three measurements decided how to handle them:

* Ignoring them is not an option: on 40 real positions the OR-free exact
  marginals were off by a mean L1 of 0.127 per card against the OR-conditioned
  truth, worse than v0.3's biased sampler at 0.074.
* Rejection is not an option: measured acceptance was mean 0.30, median 0.18,
  and below 0.05 on 16% of positions.
* Folding them into the DP exactly needs the mask groups refined by OR
  membership, whose state space has a bad tail (median 5,760, p99 27,000,000).

So: exact DP when no clause is active, and the importance sampler of ``sis.py``
when one is - which is unbiased, has a measured ESS/N around 0.8, and reaches
every consistent world (all three properties are checked against exhaustive
enumeration in ``tests4/test_sis.py``).
"""

from __future__ import annotations

import random
from itertools import product as iproduct
from typing import Optional

import numpy as np

from fish.beliefs import RESOLVED, BeliefContradiction, BeliefState
from fish.cards import NUM_PLAYERS, half_suit_cards

from .counting import GroupSystem, Infeasible
from .sis import SISFailure, SISSampler, sample_batch


class PosteriorStats:
    """Counters, so approximations are reported rather than hidden."""

    __slots__ = ("decisions", "exact_decisions", "sis_decisions", "draws",
                 "ess_sum", "failures", "infeasible", "capped_set_queries")

    def __init__(self) -> None:
        self.decisions = 0
        self.exact_decisions = 0
        self.sis_decisions = 0
        self.draws = 0
        self.ess_sum = 0.0
        self.failures = 0
        self.infeasible = 0
        #: prob_all_with() queries that hit the enumeration cap on the exact
        #: path and fell back to the independence product. Counted because an
        #: approximation nobody counts is one nobody notices.
        self.capped_set_queries = 0

    def to_dict(self) -> dict:
        d = {s: getattr(self, s) for s in self.__slots__}
        if self.sis_decisions:
            d["mean_ess"] = self.ess_sum / self.sis_decisions
        return d


class Posterior:
    """Posterior queries for one seat at one decision point.

    Construct once per decision from a propagated ``BeliefState``; all queries
    share the same DP tables and the same batch of weighted draws.
    """

    __slots__ = ("bel", "obs", "rng", "n_draws", "n_worlds", "mode", "gamma", "gamma_team", "convention_beta", "convention_q", "convention_aim", "convention_book",
                 "stats", "n", "_sys", "_card_group", "_free", "_marg",
                 "_worlds", "_batch", "_sampler", "_exact_ok", "_idx",
                 "_free_pos", "depth_mode", "count_mode", "opp_lambda",
                 "gamma_schedule", "sis_tilt", "silence_delta",
                 "w_unlocated")

    def __init__(self, belief: BeliefState, rng: random.Random,
                 n_draws: int = 128, n_worlds: int = 32,
                 mode: str = "auto", obs=None, gamma: float = 0.0,
                 depth_mode: str = "initial", count_mode: str = "linear",
                 opp_lambda: float = 0.0, gamma_schedule: float = 0.0,
                 sis_tilt: float = 0.0, silence_delta: float = 1.0,
                 gamma_team: Optional[float] = None,
                 w_unlocated: float = 0.0,
                 convention_beta: float = 0.0,
                 convention_q: float = 0.0,
                 convention_aim: bool = False,
                 convention_book: str = "depth",
                 stats: Optional[PosteriorStats] = None):
        self.bel = belief
        self.obs = obs
        self.gamma = gamma if obs is not None else 0.0
        # A separate sharpness for our own side's asks. None means one number
        # for both sides, which is the incumbent and is bit-identical to it.
        self.gamma_team = gamma_team if obs is not None else None
        #: Weight on the pre-play naming agreement. Inert at 0.
        self.convention_beta = convention_beta if obs is not None else 0.0
        self.convention_q = convention_q if obs is not None else 0.0
        self.convention_aim = bool(convention_aim)
        self.convention_book = convention_book
        # Like gamma, the silence prior conditions on behaviour, so it needs
        # the observation; without one it is inert.
        self.silence_delta = float(silence_delta) if obs is not None else 1.0
        #: Opportunity exponent on the choice model, per
        #: prereg/unlocated_belief.md. Inert at 0.0 and, like gamma, needs the
        #: observation: without one there are no asks to weight.
        self.w_unlocated = float(w_unlocated) if obs is not None else 0.0
        self.depth_mode = depth_mode
        self.count_mode = count_mode
        self.opp_lambda = opp_lambda
        self.gamma_schedule = gamma_schedule
        self.sis_tilt = sis_tilt
        self.rng = rng
        self.n_draws = n_draws
        self.n_worlds = n_worlds
        self.mode = mode
        self.stats = stats if stats is not None else PosteriorStats()
        self.stats.decisions += 1
        self.n = belief.n
        self._sys: Optional[GroupSystem] = None
        self._card_group: dict[int, int] = {}
        self._free: list[int] = []
        self._marg: Optional[np.ndarray] = None
        self._worlds: Optional[list[list[int]]] = None
        self._batch = None
        self._sampler: Optional[SISSampler] = None
        self._exact_ok = False
        self._idx = None
        self._free_pos: dict[int, int] = {}
        self._build()

    # -- construction --------------------------------------------------------

    def _build(self) -> None:
        bel = self.bel
        quotas = [bel.per] * NUM_PLAYERS
        free: list[int] = []
        for c in range(self.n):
            cand = bel.candidates[c]
            if cand & (cand - 1) == 0:
                quotas[cand.bit_length() - 1] -= 1
            else:
                free.append(c)
        self._free = free
        if not free:
            self._exact_ok = True
            return
        index: dict[int, int] = {}
        masks: list[int] = []
        sizes: list[int] = []
        for c in free:
            m = bel.candidates[c]
            gi = index.get(m)
            if gi is None:
                gi = index[m] = len(masks)
                masks.append(m)
                sizes.append(0)
            sizes[gi] += 1
            self._card_group[c] = gi
        try:
            self._sys = GroupSystem(masks, sizes, quotas)
        except Infeasible:
            self._sys = None
            self.stats.infeasible += 1
        active = self._active_clauses()
        opp = slot = None
        # `!= 0`, not `> 0`. The model is a log-linear tilt -- the weight
        # multiplies log(depth) -- so a NEGATIVE gamma is well defined and
        # means "this seat asks where it is SHALLOW". That is not academic:
        # v0.7's measured exponent is -1.0041 (results/choice_curve_foreign).
        # The old `> 0` guard silently turned a negative gamma into gamma = 0,
        # which made an experiment arm collapse into another arm and report a
        # bit-identical result -- a null that looked like a measurement. A
        # value of exactly 0 still means off, and every gamma > 0 path is
        # untouched.
        # gamma_team can make the model live even when gamma itself is zero:
        # "believe nothing about opponents, something about teammates" is a
        # coherent configuration and one the sweep visits.
        if (self.gamma != 0.0 or self.opp_lambda > 0.0
                or (self.gamma_team is not None and self.gamma_team != 0.0)
                or self.convention_beta != 0.0
                or self.convention_q != 0.0):
            from .oppmodel import build as build_opponent
            opp, slot = build_opponent(bel, self.obs, self.gamma,
                                       depth_mode=self.depth_mode,
                                       count_mode=self.count_mode,
                                       opp_lambda=self.opp_lambda,
                                       gamma_schedule=self.gamma_schedule,
                                       sis_tilt=self.sis_tilt,
                                       gamma_team=self.gamma_team,
                                       w_unlocated=self.w_unlocated,
                                       convention_beta=self.convention_beta,
                                       convention_q=self.convention_q,
                                       convention_aim=self.convention_aim,
                                       convention_book=self.convention_book,
                                       order=free)
        # The exact DP answers the uniform-target question only. An opponent
        # model changes the target, so it forces the sampling path even when no
        # clause is active.
        #
        # mode="exact" used to force the DP even with clauses LIVE, and then
        # report ``Posterior.exact is True`` and increment ``exact_decisions``.
        # The DP does not see OR clauses at all -- that is the whole reason
        # this module has a sampler -- so its draws come from a strict superset
        # of the feasible worlds, and the module docstring measures exactly how
        # wrong that is: a mean L1 error of 0.127 per card over 40 positions,
        # WORSE than the biased sampler v0.4 replaced. A wrong answer labelled
        # exact is the worst of the three available outcomes, so asking for
        # exact where exact does not exist is now an error rather than a
        # silently OR-free answer.
        if self.mode == "exact" and active:
            raise ValueError(
                f"mode='exact' was asked for at a position with "
                f"{len(active)} active OR clause(s). The counting DP does not "
                f"represent them, so it would draw from a superset of the "
                f"feasible worlds and report exact=True for it. Use "
                f"mode='auto', which takes the DP wherever it is genuinely "
                f"exact and the importance sampler everywhere else.")
        use_exact = ((self.mode == "exact"
                      or (self.mode == "auto" and not active))
                     and opp is None)
        if use_exact and self._sys is not None:
            self._exact_ok = True
            self.stats.exact_decisions += 1
        else:
            self._exact_ok = False
            card_masks = {c: bel.candidates[c] for c in free}
            self._sampler = SISSampler(free, card_masks, quotas, active,
                                       opponent_model=opp, card_slot=slot)
            self.stats.sis_decisions += 1

    def _active_clauses(self):
        """Clauses that still constrain a free card.

        The propagator already drops clauses satisfied with certainty and
        unit-propagates the rest, so what is left here genuinely bites.
        """
        free = set(self._free)
        out = []
        for cards, p in self.bel.ors:
            live = tuple(c for c in cards
                         if c in free and (self.bel.candidates[c] >> p & 1))
            if live:
                out.append((live, p))
        return out

    # -- draws ---------------------------------------------------------------

    def _get_batch(self):
        if self._batch is None and self._sampler is not None:
            try:
                self._batch = sample_batch(self._sampler, self.rng, self.n_draws)
            except SISFailure:
                self.stats.failures += 1
                self._batch = None
            if self._batch is not None and len(self._batch):
                self.stats.draws += len(self._batch)
                self.stats.ess_sum += self._batch.ess
                # One integer matrix of owners, draws x free cards. Every
                # downstream query is then a vectorised operation instead of a
                # Python double loop.
                if self._batch.picks is not None:
                    self._free_pos = {c: j for j, c in enumerate(self._batch.order)}
                    self._idx = self._batch.picks
                else:
                    free = self._free
                    self._free_pos = {c: j for j, c in enumerate(free)}
                    self._idx = np.empty((len(self._batch), len(free)),
                                         dtype=np.int64)
                    for i, deal in enumerate(self._batch.deals):
                        row = self._idx[i]
                        for j, c in enumerate(free):
                            row[j] = deal[c]
                if self.silence_delta < 1.0:
                    self._apply_silence_prior()
        return self._batch

    def _apply_silence_prior(self) -> None:
        """Down-weight draws in which a live half-suit sits wholly within one
        team right now.

        The behavioural argument, proposed by a viewer of the exhibition and
        priced here as a knob rather than assumed: a team that holds all six
        cards of a half-suit and can place them declares (this engine at a
        $0.97$ bar, most policies similarly), so the table's SILENCE about a
        live half-suit is evidence against worlds where one team already owns
        it outright. The evidence is deliberately weak -- a team can hold all
        six and be genuinely stuck on the split -- which is why the factor is
        a tunable ``silence_delta`` per concentrated suit and not a hard
        constraint. At 1.0 nothing runs and the batch is untouched.

        Current holders are exact per draw: any card that ever publicly moved
        is located by the record, so a free card's current holder IS its
        drawn initial owner, and located cards are draw-independent.
        """
        b, obs, bel = self._batch, self.obs, self.bel
        if b is None or not len(b) or self._idx is None or obs is None:
            return
        factors = np.ones(len(b), dtype=np.float64)
        touched = False
        for hs, w in enumerate(obs.set_winner):
            if w is not None:
                continue
            loc_team = -1
            usable = True
            cols = []
            for c in half_suit_cards(hs):
                m = bel.current_holder_mask(c)
                if m and (m & (m - 1)) == 0:
                    t = (m.bit_length() - 1) % 2
                    if loc_team == -1:
                        loc_team = t
                    elif t != loc_team:
                        usable = False   # located on both teams: never whole
                        break
                else:
                    j = self._free_pos.get(c)
                    if j is None:
                        usable = False   # not free, not located: stay out
                        break
                    cols.append(j)
            if not usable or not cols:
                # A fully located concentrated suit carries no draw
                # information (the factor would be a constant), so skip it.
                continue
            teams = self._idx[:, cols] % 2
            if loc_team == -1:
                conc = np.all(teams == teams[:, :1], axis=1)
            else:
                conc = np.all(teams == loc_team, axis=1)
            if conc.any():
                factors[conc] *= self.silence_delta
                touched = True
        if not touched:
            return
        w = b.w * factors
        s = w.sum()
        if s <= 0:
            return
        b.w = w / s
        b.ess = float(1.0 / np.sum(b.w ** 2))

    # -- marginals -----------------------------------------------------------

    def initial_marginals(self) -> np.ndarray:
        """``P(card c was dealt to player p)``."""
        M = np.zeros((self.n, NUM_PLAYERS), dtype=np.float64)
        for c in range(self.n):
            cand = self.bel.candidates[c]
            if cand & (cand - 1) == 0:
                M[c, cand.bit_length() - 1] = 1.0
        if not self._free:
            return M
        if self._exact_ok and self._sys is not None:
            try:
                E = self._sys.expected_counts()
                sizes = self._sys.sizes
                for c in self._free:
                    g = self._card_group[c]
                    M[c] = E[g] / sizes[g]
                return M
            except Infeasible:
                self.stats.infeasible += 1
        batch = self._get_batch()
        if batch is None or not len(batch):
            # Last resort: fall back to the v0.3 sampler so a decision is still
            # possible. Counted, never silent.
            self.stats.failures += 1
            hands = [self.bel.sample_current_hands(self.rng)
                     for _ in range(max(8, self.n_worlds))]
            for c in self._free:
                row = np.zeros(NUM_PLAYERS)
                for h in hands:
                    for p in range(NUM_PLAYERS):
                        if h[p] >> c & 1:
                            row[p] += 1
                            break
                M[c] = row / len(hands)
            return M
        free = self._free
        for c in free:
            M[c] = 0.0
        idx = self._idx
        if idx is None:
            for w, deal in zip(batch.w, batch.deals):
                for c in free:
                    M[c, deal[c]] += w
            return M
        cols = batch.order if batch.order is not None else free
        nf = idx.shape[1]
        flat = (idx + NUM_PLAYERS * np.arange(nf, dtype=np.int64)[None, :]).ravel()
        wts = np.repeat(batch.w, nf)
        acc = np.bincount(flat, weights=wts, minlength=NUM_PLAYERS * nf)
        acc = acc.reshape(nf, NUM_PLAYERS)
        for j, c in enumerate(cols):
            M[c] = acc[j]
        return M

    def marginals(self) -> np.ndarray:
        """``P(player p currently holds card c)``.

        Resolved cards give an all-zero row; publicly located cards give a
        one-hot row; everything else inherits the initial-deal marginal, because
        an unmoved card still sits with whoever was dealt it.
        """
        if self._marg is not None:
            return self._marg
        M = self.initial_marginals()
        for c in range(self.n):
            loc = self.bel.public_loc[c]
            if loc == RESOLVED:
                M[c] = 0.0
            elif loc is not None:
                M[c] = 0.0
                M[c, loc] = 1.0
        self._marg = M
        return M

    # -- worlds --------------------------------------------------------------

    def worlds(self) -> list[list[int]]:
        """``n_worlds`` current-hand samples, as 6-element bitmask lists.

        Under the exact path these are drawn exactly uniformly; under the
        sampling path they are the weighted draws resampled to unit weight, so
        they are asymptotically uniform rather than exactly so.
        """
        if self._worlds is not None:
            return self._worlds
        out: list[list[int]] = []
        if not self._free:
            out = [self._materialize({}) for _ in range(self.n_worlds)]
        elif self._exact_ok and self._sys is not None:
            try:
                for _ in range(self.n_worlds):
                    out.append(self._materialize(self._exact_draw()))
            except Infeasible:
                self.stats.infeasible += 1
        else:
            batch = self._get_batch()
            if batch is not None and len(batch):
                sel = self.rng.choices(range(len(batch)), weights=list(batch.w),
                                       k=self.n_worlds)
                if batch.picks is not None:
                    order = batch.order
                    out = [self._materialize_row(batch.picks[i], order)
                           for i in sel]
                else:
                    out = [self._materialize(batch.deals[i]) for i in sel]
        if not out:
            self.stats.failures += 1
            try:
                out = [self.bel.sample_current_hands(self.rng)
                       for _ in range(self.n_worlds)]
            except BeliefContradiction:
                out = []
        self._worlds = out
        return out

    def _exact_draw(self) -> dict:
        k = self._sys.sample_counts(self.rng)
        by_group: dict[int, list[int]] = {}
        for c in self._free:
            by_group.setdefault(self._card_group[c], []).append(c)
        deal: dict[int, int] = {}
        for g, cards in by_group.items():
            cards = list(cards)
            self.rng.shuffle(cards)
            i = 0
            for p in range(NUM_PLAYERS):
                for _ in range(int(k[g, p])):
                    deal[cards[i]] = p
                    i += 1
        return deal

    def _materialize_row(self, row, order) -> list[int]:
        hands = [0] * NUM_PLAYERS
        bel = self.bel
        owner_of = {}
        for j, c in enumerate(order):
            owner_of[c] = int(row[j])
        for c in range(self.n):
            loc = bel.public_loc[c]
            if loc == RESOLVED:
                continue
            if loc is not None:
                hands[loc] |= 1 << c
                continue
            owner = owner_of.get(c)
            if owner is None:
                cand = bel.candidates[c]
                owner = cand.bit_length() - 1
            hands[owner] |= 1 << c
        return hands

    def _materialize(self, deal: dict) -> list[int]:
        hands = [0] * NUM_PLAYERS
        bel = self.bel
        for c in range(self.n):
            loc = bel.public_loc[c]
            if loc == RESOLVED:
                continue
            if loc is not None:
                hands[loc] |= 1 << c
                continue
            owner = deal.get(c)
            if owner is None:
                cand = bel.candidates[c]
                owner = cand.bit_length() - 1
            hands[owner] |= 1 << c
        return hands

    # -- joint queries --------------------------------------------------------

    def prob_assignment(self, cards, owners) -> float:
        """``P(card_i is currently held by owner_i for all i)``.

        Exact when the clause set is empty. When it is not, deduction still
        settles the extremes exactly - a zero under the OR-free system implies a
        zero under the (smaller) OR-constrained one, and likewise for one - and
        only the genuine middle is estimated from the weighted draws.
        """
        bel = self.bel
        pins: list[tuple[int, int]] = []
        free_pins: list[tuple[int, int]] = []
        for c, want in zip(cards, owners):
            loc = bel.public_loc[c]
            if loc == RESOLVED:
                return 0.0
            if loc is not None:
                if loc != want:
                    return 0.0
                continue
            cand = bel.candidates[c]
            if cand & (cand - 1) == 0:
                if cand.bit_length() - 1 != want:
                    return 0.0
                continue
            if not cand >> want & 1:
                return 0.0
            pins.append((self._card_group[c], want))
            free_pins.append((c, want))
        if not pins:
            return 1.0
        if self._sys is None:
            return 0.0
        if self._exact_ok:
            Z = self._sys.partition()
            if not (Z > 0):
                return 0.0
            return self._sys.pinned_partition(pins) / Z
        # With clauses active the exact DP would answer a different question
        # (it ignores them) at the cost of a full extra dynamic program per
        # query. Measured, that was the dominant cost of a decision: up to nine
        # claimable half-suits times three candidate distributions each. The
        # weighted draws answer the right question for the price of one pass.
        batch = self._get_batch()
        if batch is None or not len(batch):
            return 0.0
        idx = self._idx
        if idx is None:
            tot = 0.0
            for w, deal in zip(batch.w, batch.deals):
                if all(deal[c] == want for c, want in free_pins):
                    tot += w
            return float(tot)
        mask = np.ones(idx.shape[0], dtype=bool)
        for c, want in free_pins:
            mask &= idx[:, self._free_pos[c]] == want
        return float(batch.w[mask].sum())

    def prob_all_with(self, cards, players, max_enumerate: int = 512) -> float:
        """``P(every one of `cards` is currently held by someone in `players`)``.

        The JOINT, not the product of per-card marginals.

        ``claim4.best_for_half_suit`` returns a pair -- the probability that a
        specific split is right, and the probability the half-suit is ours at
        all -- and ``forced_claim`` scores a declaration with both, as
        ``p_exact - (1 - p_team)``. Under the legacy null variant the two
        carried EQUAL weight in that ranking; under the opponent-award
        baseline ``p_team`` cancels out of it entirely (every wrong outcome
        costs the same set), so this joint query now serves diagnostics and
        the null-variant path rather than the shipped forced ranking. The first came from this posterior; the
        second was ``prod(sum of team marginals per card)``, an independence
        product over cards that compete for the same quota slots. The same
        method's own docstring says three lines earlier that the product of
        marginals is not the joint, which is precisely why the MAP split is
        shortlisted on marginals and then SCORED on the posterior.

        Mixing the two is worse than using either twice: the difference
        ``p_team - p_exact`` is read as "ours but wrongly split" and can come
        out negative, which is only hidden by a clamp.

        On the exact path this enumerates team-only assignments of the cards
        that are not already pinned; the cap keeps a rare wide half-suit from
        costing hundreds of dynamic programs, and falling back to the product
        is counted in ``PosteriorStats.capped_set_queries`` rather than passed
        off as the joint. On the sampling path it is one vectorised pass over
        the weighted draws -- the same estimator ``prob_assignment`` uses, and
        no more expensive.
        """
        bel = self.bel
        pset = set(int(p) for p in players)
        pmask = 0
        for p in pset:
            pmask |= 1 << p
        free_cards: list[int] = []
        for c in cards:
            loc = bel.public_loc[c]
            if loc == RESOLVED:
                return 0.0
            if loc is not None:
                if loc not in pset:
                    return 0.0
                continue
            cand = bel.candidates[c]
            if cand == 0:
                return 0.0
            if not cand & pmask:
                return 0.0
            if not cand & ~pmask:
                continue                  # every candidate is on the team
            free_cards.append(c)
        if not free_cards:
            return 1.0
        if self._sys is None:
            return 0.0

        if self._exact_ok:
            opts = []
            for c in free_cards:
                cand = bel.candidates[c]
                opts.append([(self._card_group[c], p) for p in sorted(pset)
                             if cand >> p & 1])
            total = 1
            for o in opts:
                total *= len(o)
            if total > max_enumerate:
                self.stats.capped_set_queries += 1
                M = self.marginals()
                out = 1.0
                for c in free_cards:
                    out *= float(sum(M[c, p] for p in pset))
                return out
            Z = self._sys.partition()
            if not (Z > 0):
                return 0.0
            acc = 0.0
            for combo in iproduct(*opts):
                acc += self._sys.pinned_partition(list(combo))
            return float(acc / Z)

        batch = self._get_batch()
        if batch is None or not len(batch):
            return 0.0
        idx = self._idx
        if idx is None:
            tot = 0.0
            for w, deal in zip(batch.w, batch.deals):
                if all(deal[c] in pset for c in free_cards):
                    tot += w
            return float(tot)
        mask = np.ones(idx.shape[0], dtype=bool)
        for c in free_cards:
            col = idx[:, self._free_pos[c]]
            ok = np.zeros(col.shape[0], dtype=bool)
            for p in pset:
                ok |= col == p
            mask &= ok
        return float(batch.w[mask].sum())

    def prob_holds(self, player: int, card: int) -> float:
        return float(self.marginals()[card, player])

    @property
    def exact(self) -> bool:
        return self._exact_ok
