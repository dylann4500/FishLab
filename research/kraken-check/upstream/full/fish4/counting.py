"""Exact counting, marginals and uniform sampling over Fish-consistent deals.

THE INFERENCE PROBLEM
---------------------
Every card movement in Literature is public, so a card's current location is a
deterministic function of the initial deal plus the public log, and all hidden
state reduces to constraints on the *initial deal* (see ``fish/beliefs.py``).
After propagation each card is either publicly located ("pinned") or free. A
free card ``c`` carries a candidate mask ``S_c`` of players who could have been
dealt it, and each player ``p`` has a residual quota ``q_p``.

A *world* is a map f from free cards to players with ``f(c) in S_c`` and
``|f^-1(p)| = q_p``. Under a uniform deal prior, and conditioning only on the
hard logical constraints, the posterior over worlds is **uniform** on this set
(Proposition 1 of the paper). v0.3 sampled from this set with a heuristic that
satisfied every constraint but was *not* uniform, so every reported probability
inherited an unquantified bias. This module removes that bias.

WHY IT IS TRACTABLE
-------------------
Free cards are numerous (mean 25.6, max 45 in measured play) but the number of
DISTINCT candidate masks is tiny (mean 4.0, max 8). Cards sharing a mask are
exchangeable, so the problem collapses onto the group-count matrix
``k[g][p]`` = how many cards of mask-group ``g`` went to player ``p``. The
number of worlds realising a given ``k`` is ``prod_g multinomial(n_g; k[g,.])``,
so

    Z = sum over feasible k of prod_g n_g! / prod_p k[g][p]!

We evaluate this by dynamic programming over players, carrying the vector of
cards still unassigned per group as state. Processing player ``p`` means
choosing ``k[g][p] <= rem_g`` from each group ``g`` with ``p in S_g`` subject to
``sum_g k[g][p] = q_p``, weighted by ``prod_g C(rem_g, k[g][p])``; the product
of binomials telescopes to the multinomial, so the recursion is exact. The
state space is ``prod_g (n_g + 1)``, a few hundred in practice (median 272).

WHAT THIS MODULE DOES AND DOES NOT COVER
----------------------------------------
It handles candidate masks and exact per-player quotas *exactly*. It does NOT
handle the OR constraints ("the asker held at least one card of that half-suit
at that moment"); those are layered on top in ``posterior.py`` by importance
weighting, because folding them in exactly requires refining the groups by
OR-membership and that blows up on a measured 4% tail of positions.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np

NUM_PLAYERS = 6

_MAXN = 64
_BINOM = np.zeros((_MAXN + 1, _MAXN + 1), dtype=np.float64)
for _n in range(_MAXN + 1):
    _BINOM[_n, 0] = 1.0
    for _k in range(1, _n + 1):
        _BINOM[_n, _k] = _BINOM[_n - 1, _k - 1] + _BINOM[_n - 1, _k]


class Infeasible(Exception):
    """The quota system admits no assignment at all."""


class GroupSystem:
    """Exact inference over ``(groups x players)`` quota systems.

    ``masks[g]`` is a 6-bit player mask, ``sizes[g]`` the number of free cards
    carrying that mask, ``quotas[p]`` how many free cards player ``p`` was
    dealt. Instances are cheap to build and cache their DP tables lazily.
    """

    __slots__ = ("masks", "sizes", "quotas", "G", "allowed", "shape",
                 "_fwd", "_bwd", "_Z", "_E", "_flat_strides")

    def __init__(self, masks: Sequence[int], sizes: Sequence[int],
                 quotas: Sequence[int]):
        self.masks = list(masks)
        self.sizes = list(sizes)
        self.quotas = list(quotas)
        self.G = len(self.masks)
        if sum(self.sizes) != sum(self.quotas):
            raise Infeasible(
                f"{sum(self.sizes)} free cards but quotas sum to "
                f"{sum(self.quotas)}")
        self.allowed = [tuple(g for g in range(self.G) if self.masks[g] >> p & 1)
                        for p in range(NUM_PLAYERS)]
        self.shape = tuple(n + 1 for n in self.sizes)
        self._fwd: Optional[list] = None
        self._bwd: Optional[list] = None
        self._Z: Optional[float] = None
        self._E: Optional[np.ndarray] = None

    # -- basic facts ---------------------------------------------------------

    @property
    def state_space(self) -> int:
        n = 1
        for s in self.sizes:
            n *= s + 1
        return n

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"GroupSystem(G={self.G}, sizes={self.sizes}, "
                f"quotas={self.quotas}, states={self.state_space})")

    # -- transitions ---------------------------------------------------------

    def _forward_step(self, table: np.ndarray, p: int) -> np.ndarray:
        """Table over remaining-counts, after player ``p`` takes their quota.

        The 'exactly q_p cards' constraint is enforced with an auxiliary axis
        counting cards taken so far, which avoids ever enumerating the
        compositions explicitly.
        """
        q = self.quotas[p]
        if q == 0:
            return table
        allowed = self.allowed[p]
        if not allowed:
            # No group can supply this player: zero completions. This is a
            # legitimate outcome for the residual systems built by
            # ``pinned_partition``, so it is a count of 0 rather than an error.
            return np.zeros_like(table)
        if len(allowed) == 1:
            # degenerate: the whole quota comes from one group
            g = allowed[0]
            n_g = self.sizes[g]
            if q > n_g:
                return np.zeros_like(table)
            out = np.zeros_like(table)
            src = [slice(None)] * table.ndim
            dst = [slice(None)] * table.ndim
            src[g] = slice(q, n_g + 1)
            dst[g] = slice(0, n_g + 1 - q)
            coef = _BINOM[q:n_g + 1, q]
            bshape = [1] * table.ndim
            bshape[g] = coef.size
            out[tuple(dst)] = table[tuple(src)] * coef.reshape(bshape)
            return out
        cur = np.zeros(table.shape + (q + 1,), dtype=np.float64)
        cur[..., 0] = table
        for g in allowed:
            n_g = self.sizes[g]
            nxt = cur.copy()                      # the k = 0 term
            for k in range(1, min(n_g, q) + 1):
                src = [slice(None)] * cur.ndim
                dst = [slice(None)] * cur.ndim
                src[g] = slice(k, n_g + 1)
                dst[g] = slice(0, n_g + 1 - k)
                src[-1] = slice(0, q + 1 - k)
                dst[-1] = slice(k, q + 1)
                coef = _BINOM[k:n_g + 1, k]
                bshape = [1] * cur.ndim
                bshape[g] = coef.size
                nxt[tuple(dst)] += cur[tuple(src)] * coef.reshape(bshape)
            cur = nxt
        return np.ascontiguousarray(cur[..., q])

    def _backward_step(self, table: np.ndarray, p: int) -> np.ndarray:
        """B_p[rem] = sum_k prod_g C(rem_g, k_g) * B_{p+1}[rem - k]."""
        q = self.quotas[p]
        if q == 0:
            return table
        allowed = self.allowed[p]
        if not allowed:
            return np.zeros_like(table)
        if len(allowed) == 1:
            g = allowed[0]
            n_g = self.sizes[g]
            out = np.zeros_like(table)
            if q > n_g:
                return out
            src = [slice(None)] * table.ndim
            dst = [slice(None)] * table.ndim
            src[g] = slice(0, n_g + 1 - q)
            dst[g] = slice(q, n_g + 1)
            coef = _BINOM[q:n_g + 1, q]
            bshape = [1] * table.ndim
            bshape[g] = coef.size
            out[tuple(dst)] = table[tuple(src)] * coef.reshape(bshape)
            return out
        cur = np.zeros(table.shape + (q + 1,), dtype=np.float64)
        cur[..., 0] = table
        for g in allowed:
            n_g = self.sizes[g]
            nxt = cur.copy()
            for k in range(1, min(n_g, q) + 1):
                src = [slice(None)] * cur.ndim
                dst = [slice(None)] * cur.ndim
                src[g] = slice(0, n_g + 1 - k)
                dst[g] = slice(k, n_g + 1)
                src[-1] = slice(0, q + 1 - k)
                dst[-1] = slice(k, q + 1)
                coef = _BINOM[k:n_g + 1, k]
                bshape = [1] * cur.ndim
                bshape[g] = coef.size
                nxt[tuple(dst)] += cur[tuple(src)] * coef.reshape(bshape)
            cur = nxt
        return np.ascontiguousarray(cur[..., q])

    # -- tables --------------------------------------------------------------

    def forward(self) -> list:
        if self._fwd is None:
            F = np.zeros(self.shape, dtype=np.float64)
            F[tuple(self.sizes)] = 1.0
            out = [F]
            for p in range(NUM_PLAYERS):
                F = self._forward_step(F, p)
                out.append(F)
            self._fwd = out
            self._Z = float(out[-1][(0,) * self.G])
        return self._fwd

    def backward(self) -> list:
        if self._bwd is None:
            B = np.zeros(self.shape, dtype=np.float64)
            B[(0,) * self.G] = 1.0
            tables = [None] * (NUM_PLAYERS + 1)
            tables[NUM_PLAYERS] = B
            for p in range(NUM_PLAYERS - 1, -1, -1):
                tables[p] = self._backward_step(tables[p + 1], p)
            self._bwd = tables
        return self._bwd

    def partition(self) -> float:
        """Exact number of consistent worlds (float; can be astronomically large)."""
        if self._Z is None:
            self.forward()
        return self._Z

    # -- exact marginals -----------------------------------------------------

    def expected_counts(self) -> np.ndarray:
        """``E[k[g][p]]`` under the uniform distribution over worlds.

        Exact, not sampled. Computed by a forward-backward sweep that carries
        one accumulator per group, so all G*6 expectations come out of a single
        pass rather than G*6 separate DP runs.
        """
        if self._E is not None:
            return self._E
        F = self.forward()
        B = self.backward()
        Z = self.partition()
        if not (Z > 0):
            raise Infeasible("no consistent world")
        E = np.zeros((self.G, NUM_PLAYERS), dtype=np.float64)
        for p in range(NUM_PLAYERS):
            q = self.quotas[p]
            if q == 0:
                continue
            E[:, p] = self._expected_for_player(F[p], B[p + 1], p) / Z
        self._E = E
        return E

    def _expected_for_player(self, Fp: np.ndarray, Bnext: np.ndarray,
                             p: int) -> np.ndarray:
        """Vector over groups of sum_{rem,k} Fp[rem] k_g W(rem,k) Bnext[rem-k]."""
        q = self.quotas[p]
        allowed = self.allowed[p]
        G = self.G
        cur = np.zeros(Fp.shape + (q + 1,), dtype=np.float64)
        cur[..., 0] = Fp
        # curw[g] mirrors cur but weighted by the number of group-g cards taken
        curw = np.zeros((G,) + cur.shape, dtype=np.float64)
        for g in allowed:
            n_g = self.sizes[g]
            nxt = cur.copy()
            nxtw = curw.copy()
            for k in range(1, min(n_g, q) + 1):
                src = [slice(None)] * cur.ndim
                dst = [slice(None)] * cur.ndim
                src[g] = slice(k, n_g + 1)
                dst[g] = slice(0, n_g + 1 - k)
                src[-1] = slice(0, q + 1 - k)
                dst[-1] = slice(k, q + 1)
                coef = _BINOM[k:n_g + 1, k]
                bshape = [1] * cur.ndim
                bshape[g] = coef.size
                cf = coef.reshape(bshape)
                s, d = tuple(src), tuple(dst)
                contrib = cur[s] * cf
                nxt[d] += contrib
                nxtw[(slice(None),) + d] += curw[(slice(None),) + s] * cf
                nxtw[(g,) + d] += contrib * k
            cur, curw = nxt, nxtw
        return curw[..., q].reshape(G, -1) @ Bnext.reshape(-1)

    # -- exact uniform sampling ----------------------------------------------

    def sample_counts(self, rng: random.Random) -> np.ndarray:
        """Draw ``k[g][p]`` exactly from the uniform distribution over worlds.

        Sequential: for each player in turn, draw their whole take from the
        exact conditional, which the backward table supplies in closed form.
        """
        B = self.backward()
        rem = list(self.sizes)
        k = np.zeros((self.G, NUM_PLAYERS), dtype=np.int64)
        for p in range(NUM_PLAYERS):
            q = self.quotas[p]
            if q == 0:
                continue
            allowed = self.allowed[p]
            if len(allowed) == 1:
                g = allowed[0]
                if rem[g] < q:
                    raise Infeasible("sampler dead end")
                k[g, p] = q
                rem[g] -= q
                continue
            caps = [min(rem[g], q) for g in allowed]
            best = None
            tot = 0.0
            opts = []
            for comp in _compositions(q, caps):
                w = 1.0
                nrem = list(rem)
                for gi, g in enumerate(allowed):
                    c = comp[gi]
                    if c:
                        w *= _BINOM[rem[g], c]
                        nrem[g] -= c
                if w == 0.0:
                    continue
                w *= B[p + 1][tuple(nrem)]
                if w > 0.0:
                    opts.append((comp, nrem, w))
                    tot += w
            if not opts:
                raise Infeasible("sampler dead end (backward table empty)")
            r = rng.random() * tot
            acc = 0.0
            best = opts[-1]
            for opt in opts:
                acc += opt[2]
                if r < acc:
                    best = opt
                    break
            comp, nrem, _ = best
            for gi, g in enumerate(allowed):
                k[g, p] = comp[gi]
            rem = nrem
        return k

    # -- exact conditional probabilities --------------------------------------

    def pinned_partition(self, pins: Sequence[tuple[int, int]]) -> float:
        """Number of worlds in which ``pins`` (group, player) placements hold.

        Each entry removes one card from that group and one slot from that
        player's quota. Used for exact joint probabilities, e.g. the chance a
        declared claim distribution is exactly right.
        """
        sizes = list(self.sizes)
        quotas = list(self.quotas)
        for g, p in pins:
            if not (self.masks[g] >> p & 1):
                return 0.0
            sizes[g] -= 1
            quotas[p] -= 1
            if sizes[g] < 0 or quotas[p] < 0:
                return 0.0
        keep = [g for g in range(self.G) if sizes[g] > 0]
        if not keep:
            return 1.0 if sum(quotas) == 0 else 0.0
        try:
            sub = GroupSystem([self.masks[g] for g in keep],
                              [sizes[g] for g in keep], quotas)
            return sub.partition()
        except Infeasible:
            return 0.0
        # Cards are labelled throughout (the multinomial counts labelled
        # assignments), so pinning named cards needs no extra choosing factor:
        # the residual system counts exactly the completions. Verified against
        # brute-force enumeration in tests.


def _compositions(total: int, caps: Sequence[int]):
    """Non-negative integer vectors summing to ``total`` with ``v[i] <= caps[i]``."""
    m = len(caps)
    if m == 1:
        if total <= caps[0]:
            yield (total,)
        return
    tail_cap = sum(caps[1:])
    lo = max(0, total - tail_cap)
    hi = min(caps[0], total)
    for v in range(lo, hi + 1):
        for rest in _compositions(total - v, caps[1:]):
            yield (v,) + rest


# ---------------------------------------------------------------------------
# Reference implementation, for tests only
# ---------------------------------------------------------------------------

def brute_force(card_masks: Sequence[int], quotas: Sequence[int]):
    """Enumerate every consistent assignment: (count, per-card marginals)."""
    from itertools import product as iproduct
    n = len(card_masks)
    choices = [[p for p in range(NUM_PLAYERS) if card_masks[c] >> p & 1]
               for c in range(n)]
    count = 0
    marg = [[0] * NUM_PLAYERS for _ in range(n)]
    target = list(quotas)
    for assign in iproduct(*choices):
        cnt = [0] * NUM_PLAYERS
        for p in assign:
            cnt[p] += 1
        if cnt != target:
            continue
        count += 1
        for c, p in enumerate(assign):
            marg[c][p] += 1
    if count:
        marg = [[v / count for v in row] for row in marg]
    return count, marg


def group_cards(card_masks: Sequence[int]):
    """Group identical masks. Returns (masks, sizes, card_to_group)."""
    index: dict[int, int] = {}
    masks: list[int] = []
    sizes: list[int] = []
    card_group: list[int] = []
    for m in card_masks:
        gi = index.get(m)
        if gi is None:
            gi = index[m] = len(masks)
            masks.append(m)
            sizes.append(0)
        sizes[gi] += 1
        card_group.append(gi)
    return masks, sizes, card_group
