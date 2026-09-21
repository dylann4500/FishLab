"""Backend D: a swap/3-cycle Markov chain on worlds.

THE CHAIN
---------
State = an assignment of every free card to a player satisfying masks, quotas
and ORs. Two proposals, both quota-preserving:

  * **2-swap**: pick an ordered pair of free cards, exchange their owners.
  * **3-cycle**: pick an ordered triple ``(i, j, l)`` and rotate owners
    ``own[i], own[j], own[l] <- own[j], own[l], own[i]``.

A proposal is accepted iff the result still satisfies every mask and every OR;
otherwise the chain stays put. Both proposal kernels are symmetric -- a 2-swap
is its own inverse, and the inverse of the rotation on the ordered triple
``(i, j, l)`` is the same rotation applied to ``(i, l, j)``, which is drawn with
equal probability -- so Metropolis-Hastings with a uniform target accepts with
probability 1 whenever the move is legal, and the uniform distribution over
worlds is stationary.

IRREDUCIBILITY IS NOT OBVIOUS AND IS NOT ASSUMED
------------------------------------------------
2-swaps alone do NOT connect the state space. Smallest counterexample, which is
in ``tests4/test_infer.py`` as a regression test: three cards with candidate
masks {0,1}, {1,2}, {0,2} and quotas (1,1,1). The two worlds are (0,1,2) and
(1,2,0), and every 2-swap between them is blocked by a mask. The 3-cycle
connects them. ``chain_coverage`` below measures reachability directly against
full enumeration: on the 34 enumerable synthetic systems the tests use, a
2-swap-only chain failed to reach every world on **8 of 34**, and the 3-cycle
chain reached every world on **34 of 34**.

Note the OR constraints only ever *remove* moves, so adding them can in
principle disconnect a space that masks alone would connect. That is measured,
not argued.

ESTIMATOR
---------
Marginals are the time average over the whole post-burn-in trajectory, not over
thinned snapshots: whenever a card changes owner at step t we credit its
previous owner with ``t - last_change``. That is O(1) per accepted move instead
of O(n_free) per snapshot, and it uses every step of the chain, so it strictly
beats thinning at equal wall-clock.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from fish.beliefs import BeliefContradiction, BeliefState
from fish.cards import NUM_PLAYERS

from .base import FreeSystem, InferenceBackend


class MCMCBackend(InferenceBackend):
    """Knob: ``n_sweeps`` (one sweep = ``n_free`` proposals)."""

    name = "mcmc"

    def __init__(self, belief: BeliefState, rng: random.Random,
                 n_sweeps: int = 40, burn_sweeps: int = 5,
                 p_three: float = 0.5, fs: Optional[FreeSystem] = None,
                 allow_v03_init: bool = True):
        super().__init__(belief, rng, fs)
        self.n_sweeps = int(n_sweeps)
        self.burn_sweeps = int(burn_sweeps)
        self.p_three = float(p_three)
        self.allow_v03_init = allow_v03_init
        self.accepted = 0
        self.proposed = 0
        self.init_failed = False
        self._marg: Optional[np.ndarray] = None
        fs = self.fs
        self._or_of_card: list[list[tuple[int, int]]] = [[] for _ in range(fs.n_free)]
        for oi, (live, p) in enumerate(fs.ors):
            for i in live:
                self._or_of_card[i].append((oi, p))
        self._order = sorted(range(fs.n_free), key=lambda i: len(fs.players[i]))

    def cost_model(self) -> dict:
        return {"knob": "n_sweeps", "value": self.n_sweeps,
                "unit": f"one sweep = {self.fs.n_free} proposals"}

    # -- initial state --------------------------------------------------------

    def initial_state(self) -> Optional[list[int]]:
        """A valid world, built greedily. Distribution is irrelevant (that is
        what burn-in is for), speed is not: this runs before every chain."""
        fs = self.fs
        if fs.n_free == 0:
            return []
        rng = self.rng
        random_ = rng.random
        players = fs.players
        for _ in range(50):
            quota = list(fs.quotas)
            own = [-1] * fs.n_free
            ok = True
            ors = list(fs.ors)
            rng.shuffle(ors)
            for live, p in ors:
                if any(own[i] == p for i in live):
                    continue
                cands = [i for i in live if own[i] == -1]
                if not cands or quota[p] <= 0:
                    ok = False
                    break
                i = cands[int(random_() * len(cands))]
                own[i] = p
                quota[p] -= 1
            if not ok:
                continue
            for i in self._order:
                if own[i] != -1:
                    continue
                tot = 0
                for p in players[i]:
                    if quota[p] > 0:
                        tot += quota[p]
                if tot == 0:
                    ok = False
                    break
                r = random_() * tot
                acc = 0
                pick = -1
                for p in players[i]:
                    q = quota[p]
                    if q > 0:
                        acc += q
                        if r < acc:
                            pick = p
                            break
                if pick < 0:
                    for p in players[i]:
                        if quota[p] > 0:
                            pick = p
                if pick < 0:
                    ok = False
                    break
                own[i] = pick
                quota[pick] -= 1
            if ok and fs.satisfies_ors(own):
                return own
        if self.allow_v03_init:
            try:
                deal = self.bel._sample_initial(self.rng)
                return [deal[c] for c in fs.free]
            except BeliefContradiction:
                return None
        return None

    # -- the chain ------------------------------------------------------------

    def run(self, own: list[int], steps: int, acc: Optional[list[float]] = None):
        """Run ``steps`` proposals in place. If ``acc`` is given, accumulate the
        time-integral of the one-hot owner indicators into it (flat n_free*6)."""
        fs = self.fs
        n = fs.n_free
        if n < 2:
            return own
        cand = fs.cand
        or_of = self._or_of_card
        ors = fs.ors
        orcnt = [0] * len(ors)
        for oi, (live, p) in enumerate(ors):
            orcnt[oi] = sum(1 for i in live if own[i] == p)
        random_ = self.rng.random
        p3 = self.p_three if n >= 3 else 0.0
        last = [0] * n
        accepted = 0
        for t in range(1, steps + 1):
            if random_() < p3:
                i = int(random_() * n)
                j = int(random_() * n)
                l = int(random_() * n)
                if i == j or j == l or i == l:
                    continue
                a, b, c = own[i], own[j], own[l]
                if not (cand[i] >> b & 1 and cand[j] >> c & 1
                        and cand[l] >> a & 1):
                    continue
                moves = ((i, a, b), (j, b, c), (l, c, a))
            else:
                i = int(random_() * n)
                j = int(random_() * n)
                if i == j:
                    continue
                a, b = own[i], own[j]
                if a == b:
                    continue
                if not (cand[i] >> b & 1 and cand[j] >> a & 1):
                    continue
                moves = ((i, a, b), (j, b, a))
            # OR feasibility of the resulting state
            if ors:
                delta: dict[int, int] = {}
                for idx, old, new in moves:
                    for oi, p in or_of[idx]:
                        if old == p:
                            delta[oi] = delta.get(oi, 0) - 1
                        if new == p:
                            delta[oi] = delta.get(oi, 0) + 1
                bad = False
                for oi, d in delta.items():
                    if d and orcnt[oi] + d <= 0:
                        bad = True
                        break
                if bad:
                    continue
                for oi, d in delta.items():
                    if d:
                        orcnt[oi] += d
            accepted += 1
            for idx, old, new in moves:
                if acc is not None:
                    acc[idx * NUM_PLAYERS + old] += t - last[idx]
                    last[idx] = t
                own[idx] = new
        if acc is not None:
            for idx in range(n):
                acc[idx * NUM_PLAYERS + own[idx]] += steps - last[idx]
        self.accepted += accepted
        self.proposed += steps
        return own

    # -- queries --------------------------------------------------------------

    def free_marginals(self) -> np.ndarray:
        if self._marg is not None:
            return self._marg
        fs = self.fs
        M = np.zeros((fs.n_free, NUM_PLAYERS))
        if fs.n_free == 0:
            self._marg = M
            return M
        own = self.initial_state()
        if own is None:
            self.init_failed = True
            self._marg = M
            return M
        if fs.n_free == 1:
            M[0, own[0]] = 1.0
            self._marg = M
            return M
        self.run(own, self.burn_sweeps * fs.n_free)
        steps = max(1, self.n_sweeps * fs.n_free)
        acc = [0.0] * (fs.n_free * NUM_PLAYERS)
        self.run(own, steps, acc)
        M = np.array(acc, dtype=np.float64).reshape(fs.n_free, NUM_PLAYERS)
        M /= steps
        self._marg = M
        return M

    def worlds(self, n: int) -> list[list[int]]:
        fs = self.fs
        if fs.n_free == 0:
            return [list(fs.base_hands) for _ in range(n)]
        own = self.initial_state()
        if own is None:
            self.init_failed = True
            return []
        self.run(own, self.burn_sweeps * fs.n_free)
        thin = max(1, self.n_sweeps * fs.n_free // max(1, n))
        out = []
        for _ in range(n):
            self.run(own, thin)
            out.append(fs.materialize(own))
        return out


# ---------------------------------------------------------------------------
# Diagnostics (used by tests4 and by the frontier study, not on the hot path)
# ---------------------------------------------------------------------------

def chain_coverage(fs: FreeSystem, rng: random.Random, steps: int,
                   p_three: float, bel: Optional[BeliefState] = None):
    """Distinct states visited by the chain, for irreducibility checking.

    Returns ``(visited_set, start_state)``. Compare against
    ``fs.enumerate_worlds()`` on systems small enough to enumerate.
    """
    back = MCMCBackend(bel, rng, n_sweeps=0, burn_sweeps=0, p_three=p_three,
                       fs=fs, allow_v03_init=False)
    own = back.initial_state()
    if own is None:
        return set(), None
    start = tuple(own)
    seen = {start}
    for _ in range(steps):
        back.run(own, 1)
        seen.add(tuple(own))
    return seen, start


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation of a 1-d series, lags 0..max_lag."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    var = float(np.dot(x, x))
    if var <= 0:
        return np.zeros(max_lag + 1)
    out = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        out[k] = float(np.dot(x[:len(x) - k], x[k:])) / var
    return out


def integrated_act(rho: np.ndarray) -> float:
    """Integrated autocorrelation time with the standard initial-positive
    truncation (Geyer): stop at the first non-positive value."""
    tau = 1.0
    for k in range(1, len(rho)):
        if rho[k] <= 0:
            break
        tau += 2.0 * rho[k]
    return tau
