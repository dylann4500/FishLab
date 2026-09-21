"""One interface for every hidden-state inference backend.

WHY THIS EXISTS
---------------
v0.3 had exactly one inference method: a heuristic sampler bolted onto the
constraint propagator, whose bias was never quantified. v0.4 has several
candidates that trade accuracy against wall-clock in different ways, and the
only honest way to pick one is to measure them against a common gold standard
through a common interface. So every backend here takes the same input (a
*propagated* ``fish.beliefs.BeliefState`` plus an RNG) and answers the same two
questions:

  ``marginals()``  -> (n_cards, 6) array, P(player p currently holds card c)
  ``worlds(n)``    -> n determinizations, each a list of 6 hand bitmasks

plus ``cost_model()``, which names the single knob that trades accuracy for
time. The frontier study (``scripts4/infer_frontier.py``) sweeps exactly that
knob.

THE TARGET DISTRIBUTION
-----------------------
After propagation each card is either pinned (candidate mask is a singleton) or
*free*. A world is a map f from free cards to players with

    f(c) in S_c,  |f^-1(p)| = q_p,  and  for each OR (cards, p): some c with f(c)=p

where the OR constraints come from "an asker holds a card of the half-suit it
asks in". Under a uniform deal prior conditioned on those hard constraints the
posterior is uniform over that set (PAPER.md section 3). Everything in this
package is measured against that uniform posterior; nothing here is a new
modelling assumption, only a different way of computing the same object.

WHAT ``FreeSystem`` FACTORS OUT
-------------------------------
All four backends need the same preprocessing: which cards are free, the
residual quotas, the mask groups (cards sharing a candidate mask are
exchangeable *for the OR-free part* of the problem), the live OR constraints
re-indexed onto free cards, and the per-half-suit partition of the free cards
(which is what makes the exact OR treatment factor -- every OR lives inside a
single half-suit, because a legal ask certifies a holding in the half-suit it
asks in). Building it once per decision keeps the backend comparison about
inference and not about bookkeeping; its measured cost is charged to every
backend equally and reported in FRONTIER.md.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import numpy as np

from fish.beliefs import RESOLVED, BeliefState
from fish.cards import NUM_PLAYERS
from fish.rules import RuleConfig


# mask -> tuple of players, for all 64 six-bit masks. Building this once
# removed 164 generator steps per FreeSystem construction, which profiling
# showed was the single biggest line in the constructor.
_MASK_PLAYERS = tuple(
    tuple(p for p in range(NUM_PLAYERS) if m >> p & 1) for m in range(64))


class FreeSystem:
    """Backend-independent decomposition of a propagated belief state.

    Read-only with respect to the belief state: nothing here mutates ``bel``,
    so the same ``BeliefState`` can be handed to several backends in a row and
    the comparison stays fair.

    Built once per decision, so its own cost is part of every backend's
    measured wall-clock. Measured on 60 real positions at 79 us mean / 81 us
    median after the optimisations noted inline, against 288 us before them.
    """

    __slots__ = ("bel", "n", "per", "free", "n_free", "pos", "cand", "players",
                 "quotas", "masks", "sizes", "group_of", "cards_of_group",
                 "n_groups", "ors", "is_or", "or_cards", "hs_of",
                 "half_suits", "base_hands", "free_bits", "_free_idx",
                 "_base_marg", "feasible", "_pinned")

    def __init__(self, bel: BeliefState):
        self.bel = bel
        n = self.n = bel.n
        self.per = bel.per
        quotas = [bel.per] * NUM_PLAYERS
        free: list[int] = []
        cand: list[int] = []
        pinned: list[int] = [-1] * n
        base_hands = [0] * NUM_PLAYERS
        candidates = bel.candidates
        public_loc = bel.public_loc
        # one pass over the deck: split pinned from free, accumulate quotas,
        # and build the fixed part of every world at the same time
        for c in range(n):
            m = candidates[c]
            if m & (m - 1) == 0:
                p = m.bit_length() - 1
                pinned[c] = p
                quotas[p] -= 1
                loc = public_loc[c]
                if loc != RESOLVED:
                    base_hands[p if loc is None else loc] |= 1 << c
            else:
                free.append(c)
                cand.append(m)
        self.free = free
        self.cand = cand
        self.n_free = len(free)
        self._free_idx = None
        self._base_marg = None
        self._pinned = pinned
        self.base_hands = base_hands
        self.pos = {c: i for i, c in enumerate(free)}
        self.quotas = quotas
        self.players = [_MASK_PLAYERS[m] for m in cand]
        self.feasible = (sum(quotas) == self.n_free and min(quotas) >= 0)

        # -- mask groups: cards with identical candidate masks are
        # exchangeable under the OR-free posterior.
        index: dict[int, int] = {}
        masks: list[int] = []
        sizes: list[int] = []
        group_of: list[int] = []
        for m in cand:
            g = index.get(m)
            if g is None:
                g = index[m] = len(masks)
                masks.append(m)
                sizes.append(0)
            sizes[g] += 1
            group_of.append(g)
        self.masks = masks
        self.sizes = sizes
        self.group_of = group_of
        self.n_groups = len(masks)
        self.cards_of_group = [[] for _ in masks]
        for i, g in enumerate(group_of):
            self.cards_of_group[g].append(i)

        # -- live OR constraints, re-indexed onto free-card indices.
        # After ``_propagate`` every card of a live OR is free and has p in its
        # mask (a card pinned to p would have satisfied and dropped the OR, a
        # card pinned elsewhere is filtered out of ``live``). We re-derive that
        # anyway rather than trusting it, because a caller could hand us a
        # belief state built by other means.
        ors: list[tuple[tuple[int, ...], int]] = []
        for cards, p in bel.ors:
            bit = 1 << p
            if any(bel.candidates[c] == bit for c in cards):
                continue                       # already certainly satisfied
            live = tuple(self.pos[c] for c in cards
                         if c in self.pos and bel.candidates[c] & bit)
            if live:
                ors.append((live, p))
        self.ors = ors
        is_or = [False] * self.n_free
        for live, _p in ors:
            for i in live:
                is_or[i] = True
        self.is_or = is_or
        self.or_cards = [i for i in range(self.n_free) if is_or[i]]

        # -- half-suit partition of the free cards
        self.hs_of = [c // 6 for c in free]
        hs: dict[int, list[int]] = {}
        for i, h in enumerate(self.hs_of):
            hs.setdefault(h, []).append(i)
        self.half_suits = hs
        self.free_bits = [1 << c for c in free]

    # -- lazily built, because ``free_marginals`` never needs them -----------

    @property
    def free_idx(self) -> np.ndarray:
        if self._free_idx is None:
            self._free_idx = np.array(self.free, dtype=np.int64)
        return self._free_idx

    @property
    def base_marg(self) -> np.ndarray:
        """``(n_cards, 6)`` with the publicly-determined rows already filled."""
        if self._base_marg is None:
            base = np.zeros((self.n, NUM_PLAYERS), dtype=np.float64)
            rows, cols = [], []
            for c in range(self.n):
                loc = self.bel.public_loc[c]
                if loc == RESOLVED:
                    continue                   # out of play: row stays zero
                p = self._pinned[c] if loc is None else loc
                if p >= 0:
                    rows.append(c)
                    cols.append(p)
            if rows:
                base[rows, cols] = 1.0
            self._base_marg = base
        return self._base_marg

    def without_ors(self) -> "FreeSystem":
        """The same system with the OR constraints dropped.

        The OR-free posterior is a useful object in its own right: it is the
        control variate in ``uniform_reject``, and it is what the closed form
        in ``counting.py`` computes. Sharing every other field keeps it free.
        """
        other = FreeSystem.__new__(FreeSystem)
        for name in FreeSystem.__slots__:
            setattr(other, name, getattr(self, name))
        other.ors = []
        other.is_or = [False] * self.n_free
        other.or_cards = []
        return other

    # -- helpers shared by every backend -------------------------------------

    def materialize(self, owners: Sequence[int]) -> list[int]:
        """Turn per-free-card owners into 6 current-hand bitmasks."""
        hands = list(self.base_hands)
        bits = self.free_bits
        for i in range(self.n_free):
            hands[owners[i]] |= bits[i]
        return hands

    def satisfies_ors(self, owners: Sequence[int]) -> bool:
        for live, p in self.ors:
            for i in live:
                if owners[i] == p:
                    break
            else:
                return False
        return True

    def valid_world(self, owners: Sequence[int]) -> bool:
        """Full constraint check -- masks, quotas and ORs. Used by tests."""
        if len(owners) != self.n_free:
            return False
        cnt = [0] * NUM_PLAYERS
        for i, p in enumerate(owners):
            if not self.cand[i] >> p & 1:
                return False
            cnt[p] += 1
        if cnt != self.quotas:
            return False
        return self.satisfies_ors(owners)

    def enumerate_worlds(self, cap: int = 2_000_000):
        """Every consistent world, as tuples of owners. Reference only."""
        out: list[tuple[int, ...]] = []
        quotas = self.quotas
        players = self.players
        n = self.n_free
        # last-position of each OR so we can prune a doomed branch early
        last_of: list[int] = [max(live) for live, _ in self.ors]
        owners = [0] * n
        used = [0] * NUM_PLAYERS
        ors = self.ors

        def rec(i: int) -> None:
            if len(out) > cap:
                raise RuntimeError("enumerate_worlds cap exceeded")
            if i == n:
                out.append(tuple(owners))
                return
            for p in players[i]:
                if used[p] >= quotas[p]:
                    continue
                owners[i] = p
                used[p] += 1
                ok = True
                for oi, (live, q) in enumerate(ors):
                    if last_of[oi] == i:
                        if not any(owners[j] == q for j in live):
                            ok = False
                            break
                if ok:
                    rec(i + 1)
                used[p] -= 1
            owners[i] = 0

        rec(0)
        return out

    def brute_force_marginals(self) -> np.ndarray:
        """Exact free-card marginals by enumeration. Reference only."""
        worlds = self.enumerate_worlds()
        M = np.zeros((self.n_free, NUM_PLAYERS), dtype=np.float64)
        for w in worlds:
            for i, p in enumerate(w):
                M[i, p] += 1.0
        if worlds:
            M /= len(worlds)
        return M


class InferenceBackend(ABC):
    """Common interface. Construction is cheap; work happens on first query."""

    name: str = "?"

    def __init__(self, belief: BeliefState, rng: random.Random,
                 fs: Optional[FreeSystem] = None):
        self.bel = belief
        self.rng = rng
        self.fs = fs if fs is not None else FreeSystem(belief)

    # -- what each backend must supply ---------------------------------------

    @abstractmethod
    def free_marginals(self) -> np.ndarray:
        """``(n_free, 6)`` array of P(player p currently holds free card i).

        Free cards have never moved publicly, so their current holder is
        whoever was dealt them; this is the initial-deal marginal.
        """

    @abstractmethod
    def worlds(self, n: int) -> list[list[int]]:
        """``n`` determinizations, each 6 current-hand bitmasks."""

    @abstractmethod
    def cost_model(self) -> dict:
        """``{"knob": name, "value": v, "unit": what one unit buys}``."""

    # -- derived, identical for every backend --------------------------------

    def marginals(self) -> np.ndarray:
        M = self.fs.base_marg.copy()
        if self.fs.n_free:
            M[self.fs.free_idx] = self.free_marginals()
        return M

    def prob_holds(self, player: int, card: int) -> float:
        return float(self.marginals()[card, player])


def snapshot_belief(bel: BeliefState) -> dict:
    """Serialisable capture of everything the backends read.

    The frontier study harvests hundreds of positions from real games; replaying
    those games to regenerate them costs minutes and depends on agent code that
    may change. Snapshotting the constraint store instead makes the study
    reproducible from a cached file and independent of the agents.
    """
    return {
        "rules": bel.rules.to_dict(),
        "observer": bel.observer,
        "candidates": list(bel.candidates),
        "public_loc": [(-1 if v is None else v) for v in bel.public_loc],
        "ors": [[list(cards), p] for cards, p in bel.ors],
    }


def restore_belief(d: dict) -> BeliefState:
    """Rebuild a BeliefState from ``snapshot_belief`` output.

    The result is query-ready but NOT update-ready: ``update()`` is not valid on
    it because the event cursor is gone. Backends only read the constraint
    store, so that is enough, and it keeps snapshots small.
    """
    bel = BeliefState(RuleConfig.from_dict(d["rules"]), observer=d["observer"])
    bel.candidates = list(d["candidates"])
    bel.public_loc = [(None if v == -1 else v) for v in d["public_loc"]]
    bel.ors = [(tuple(cards), p) for cards, p in d["ors"]]
    bel._observer_initialized = True
    bel._cursor = -1                     # poisoned: update() must not be called
    bel._pinned_count = [0] * NUM_PLAYERS
    for c in range(bel.n):
        cand = bel.candidates[c]
        if cand & (cand - 1) == 0:
            bel._pinned_count[cand.bit_length() - 1] += 1
    bel._version += 1
    bel._cache = None
    bel._cache_version = -1
    return bel
