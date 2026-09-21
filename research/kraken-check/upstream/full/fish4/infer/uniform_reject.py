"""Backend B: exactly-uniform draws, rejected against the OR constraints.

CORRECTNESS IS FREE HERE; COST IS THE WHOLE PROBLEM
---------------------------------------------------
``counting.GroupSystem`` samples exactly uniformly over the candidate-mask +
quota system. Keeping only the draws that also satisfy the OR constraints is
rejection sampling from the true posterior, so this backend is unbiased by
construction and its only error is Monte Carlo error. The measured acceptance
rate over 349 real positions is **0.27** on average, so the interesting question
is entirely how cheap a draw can be made and how much variance can be squeezed
out of the accepted ones.

The measured answer, in FRONTIER.md, is that it is not enough: after every
optimisation below this backend still needs 167 ms at the median to reach an
L1/card of 0.034, on positions where the exact backend returns the exact answer
in 7.5 ms. It is kept, and kept correct, because it is the only way to validate
the exact backend on a real position without enumerating -- not because it
should be on the hot path.

Three things make it cheaper than the naive loop:

1. ``FastGroupSystem`` (see ``fastdraw.py``) replaces per-draw composition
   enumeration with table lookups.

2. **Only OR-relevant cards are dealt individually.** Acceptance depends on
   nothing else. Cards of a mask group that appear in no OR stay aggregated as
   a residual count vector, which also removes the per-draw shuffle.

3. **Rao-Blackwellisation.** Given the group counts and the owners of the
   OR-relevant cards, the remaining cards of a group are uniform over the
   residual counts, and acceptance is already determined. So instead of a
   0/1 indicator we accumulate the exact conditional expectation
   ``residual[p] / residual_total`` for those cards. Same expectation, strictly
   lower variance, and no extra work.

And one thing makes it more accurate at equal cost:

4. **Control variate.** The OR-free marginals are known in closed form from the
   DP, so with ``Y`` the sampled indicator we can use

       m_hat = m_closedform + (mean over accepted draws - mean over all draws)

   because the all-draws mean is an unbiased estimator of the closed form. The
   two means are strongly correlated, so most of the noise cancels. FRONTIER.md
   reports whether that actually pays off; it is claimed here as a mechanism,
   not as a result.

When there are no live OR constraints the closed form IS the posterior and this
backend returns it without drawing anything. That is not a special case being
swept under the rug -- 7% of real positions are OR-free, and any deployed
inference layer would do the same.

A caveat worth stating plainly, because it shapes the recommendation: the
control variate needs the *exact* OR-free posterior, and the cheapest way to
compute that turns out to be the same flat-array DP that backend C uses. So the
control-variate variant of this backend is not really an alternative to the
exact backend -- it contains most of it.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from fish.beliefs import BeliefState
from fish.cards import NUM_PLAYERS
from fish4.counting import Infeasible

from .base import FreeSystem, InferenceBackend
from .exact_or import ExactOrBackend
from .fastdraw import FastGroupSystem


def closed_form_marginals(fs: FreeSystem) -> np.ndarray:
    """Exact OR-free per-card marginals: the control variate.

    ``counting.GroupSystem.expected_counts`` computes exactly these numbers,
    and it is the most expensive single thing in the v0.4 inference stack:
    measured on 30 real positions at **93 ms mean, 9.6 ms median, 812 ms max**
    on top of the backward pass, because it carries a ``(groups, states,
    quota)`` accumulator and copies it once per group per player.

    The same numbers come out of the flat-array DP in ``exact_or`` with the ORs
    removed, at **7.9 ms mean, 5.1 ms median, 25.6 ms max** -- 12x cheaper on
    the mean and, more usefully, with no 800 ms tail. The two agree to
    1.7e-16, i.e. to floating-point identity, which is checked in tests4.
    """
    free = fs.without_ors()
    eo = ExactOrBackend(None, random.Random(0), work_budget=10 ** 15, fs=free)
    if eo.ok:
        return eo.free_marginals()
    # extremely unlikely (the OR-free DP is the cheap case); keep the old path
    from fish4.counting import GroupSystem
    E = GroupSystem(fs.masks, fs.sizes, fs.quotas).expected_counts()
    out = np.zeros((fs.n_free, NUM_PLAYERS))
    for i in range(fs.n_free):
        g = fs.group_of[i]
        out[i] = E[g] / fs.sizes[g]
    return out


class UniformRejectBackend(InferenceBackend):
    """Knob: ``n_accepted`` -- how many OR-satisfying draws to collect."""

    name = "uniform_reject"

    def __init__(self, belief: BeliefState, rng: random.Random,
                 n_accepted: int = 64, max_draw_factor: int = 25,
                 control_variate: bool = True, use_fast: bool = True,
                 fs: Optional[FreeSystem] = None):
        super().__init__(belief, rng, fs)
        self.n_accepted = int(n_accepted)
        self.max_draw_factor = int(max_draw_factor)
        self.control_variate = bool(control_variate)
        self.use_fast = bool(use_fast)
        self.draws = 0
        self.accepted = 0
        self.exact = False          # True when no ORs are live: closed form
        self.starved = False        # True when rejection produced nothing
        self._marg: Optional[np.ndarray] = None
        self._sys: Optional[FastGroupSystem] = None
        self._closed: Optional[np.ndarray] = None
        self._prep()

    def cost_model(self) -> dict:
        return {"knob": "n_accepted", "value": self.n_accepted,
                "unit": "one OR-satisfying exactly-uniform world"}

    # -- setup ---------------------------------------------------------------

    def _prep(self) -> None:
        fs = self.fs
        self._or_idx: list[list[int]] = [[] for _ in range(fs.n_groups)]
        self._nonor_idx: list[list[int]] = [[] for _ in range(fs.n_groups)]
        for i in range(fs.n_free):
            g = fs.group_of[i]
            (self._or_idx if fs.is_or[i] else self._nonor_idx)[g].append(i)
        if fs.n_free == 0:
            return
        self._sys = FastGroupSystem(fs.masks, fs.sizes, fs.quotas)
        if self.use_fast:
            self._sys.prepare_fast()
        else:
            self._sys.backward()
        self._own = [0] * fs.n_free

    def closed(self) -> np.ndarray:
        """OR-free marginals, computed only when something actually wants them.

        Three callers: an OR-free position (where they ARE the posterior), the
        control variate, and the starvation fallback. With
        ``control_variate=False`` on an OR-active position nothing wants them,
        and computing them eagerly was measured to cost more than every draw
        put together.
        """
        if self._closed is None:
            self._closed = closed_form_marginals(self.fs)
        return self._closed

    # -- one draw ------------------------------------------------------------

    def _draw(self):
        """Returns ``(residual_counts, residual_totals, accepted)``.

        Side effect: ``self._own`` holds the owners of the OR-relevant cards.
        """
        fs = self.fs
        sysm = self._sys
        k = (sysm.sample_counts_fast(self.rng) if self.use_fast
             else [[int(v) for v in row] for row in sysm.sample_counts(self.rng)])
        own = self._own
        random_ = self.rng.random
        resid = []
        totals = []
        for g in range(fs.n_groups):
            row = k[g]
            oi = self._or_idx[g]
            if not oi:
                resid.append(row)
                totals.append(fs.sizes[g])
                continue
            rc = list(row)
            total = fs.sizes[g]
            for i in oi:
                r = random_() * total
                acc = 0
                pick = -1
                for p in range(NUM_PLAYERS):
                    cnt = rc[p]
                    if cnt:
                        acc += cnt
                        if r < acc:
                            pick = p
                            break
                if pick < 0:                       # float edge case
                    for p in range(NUM_PLAYERS - 1, -1, -1):
                        if rc[p]:
                            pick = p
                            break
                own[i] = pick
                rc[pick] -= 1
                total -= 1
            resid.append(rc)
            totals.append(total)
        ok = True
        for live, p in fs.ors:
            for i in live:
                if own[i] == p:
                    break
            else:
                ok = False
                break
        return resid, totals, ok

    # -- marginals -----------------------------------------------------------

    def free_marginals(self) -> np.ndarray:
        if self._marg is not None:
            return self._marg
        fs = self.fs
        if fs.n_free == 0:
            self._marg = np.zeros((0, NUM_PLAYERS))
            return self._marg
        if not fs.ors:
            self.exact = True
            self._marg = self.closed().copy()
            return self._marg

        G = fs.n_groups
        acc_all_g = [[0.0] * NUM_PLAYERS for _ in range(G)]
        acc_ok_g = [[0.0] * NUM_PLAYERS for _ in range(G)]
        or_cards = fs.or_cards
        acc_all_c = {i: [0.0] * NUM_PLAYERS for i in or_cards}
        acc_ok_c = {i: [0.0] * NUM_PLAYERS for i in or_cards}
        cap = max(self.n_accepted * self.max_draw_factor, self.n_accepted)
        n_ok = 0
        n_all = 0
        own = self._own
        try:
            while n_ok < self.n_accepted and n_all < cap:
                resid, totals, ok = self._draw()
                n_all += 1
                for g in range(G):
                    t = totals[g]
                    if not t:
                        continue
                    rowa = acc_all_g[g]
                    r = resid[g]
                    inv = 1.0 / t
                    for p in range(NUM_PLAYERS):
                        if r[p]:
                            rowa[p] += r[p] * inv
                    if ok:
                        rowo = acc_ok_g[g]
                        for p in range(NUM_PLAYERS):
                            if r[p]:
                                rowo[p] += r[p] * inv
                for i in or_cards:
                    acc_all_c[i][own[i]] += 1.0
                if ok:
                    n_ok += 1
                    for i in or_cards:
                        acc_ok_c[i][own[i]] += 1.0
        except Infeasible:
            pass
        self.draws = n_all
        self.accepted = n_ok
        if n_ok == 0 or n_all == 0:
            self.starved = True
            self._marg = self.closed().copy()
            return self._marg

        m_all = np.zeros((fs.n_free, NUM_PLAYERS))
        m_ok = np.zeros((fs.n_free, NUM_PLAYERS))
        for i in range(fs.n_free):
            g = fs.group_of[i]
            if fs.is_or[i]:
                m_all[i] = acc_all_c[i]
                m_ok[i] = acc_ok_c[i]
            else:
                m_all[i] = acc_all_g[g]
                m_ok[i] = acc_ok_g[g]
        m_all /= n_all
        m_ok /= n_ok
        if self.control_variate:
            out = self.closed() + (m_ok - m_all)
            np.clip(out, 0.0, 1.0, out=out)
        else:
            out = m_ok
        s = out.sum(axis=1, keepdims=True)
        out = out / np.where(s > 0, s, 1.0)
        self._marg = out
        return out

    # -- worlds --------------------------------------------------------------

    def worlds(self, n: int) -> list[list[int]]:
        fs = self.fs
        if fs.n_free == 0:
            return [list(fs.base_hands) for _ in range(n)]
        out: list[list[int]] = []
        cap = max(n * self.max_draw_factor, n)
        tries = 0
        own = self._own
        shuffle = self.rng.shuffle
        try:
            while len(out) < n and tries < cap:
                tries += 1
                resid, totals, ok = self._draw()
                self.draws += 1
                if not ok:
                    continue
                self.accepted += 1
                for g in range(fs.n_groups):
                    cards = self._nonor_idx[g]
                    if not cards:
                        continue
                    pool: list[int] = []
                    r = resid[g]
                    for p in range(NUM_PLAYERS):
                        if r[p]:
                            pool.extend([p] * r[p])
                    shuffle(pool)
                    for j, i in enumerate(cards):
                        own[i] = pool[j]
                out.append(fs.materialize(own))
        except Infeasible:
            pass
        return out

    # -- diagnostics ----------------------------------------------------------

    @property
    def acceptance(self) -> float:
        return self.accepted / self.draws if self.draws else 1.0
