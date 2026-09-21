"""Backend A: the v0.3 incumbent, wrapped unchanged.

This is the sampler in ``fish.beliefs.BeliefState._sample_initial``: fill the
most-constrained mask group first, choosing each card's owner with probability
proportional to that player's remaining quota, after seeding the disjoint OR
constraints and repairing the overlapping ones by swaps. Every world it
produces satisfies every constraint -- that part is tested to death in
``tests/`` -- but the draws are NOT uniform over the consistent set, because
quota-proportional filling over-weights worlds that spread cards evenly and the
OR seeding tilts the ORs' own cards.

It is here as the baseline the other three have to beat. Nothing in this module
changes it; ``fish/`` is off limits and, more importantly, the comparison is
only meaningful if the incumbent is the incumbent.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from fish.beliefs import BeliefContradiction, BeliefState
from fish.cards import NUM_PLAYERS

from .base import FreeSystem, InferenceBackend


class V03Backend(InferenceBackend):
    """Knob: number of sampled worlds (v0.3's own ``n_samples``, default 32)."""

    name = "v03"

    def __init__(self, belief: BeliefState, rng: random.Random,
                 n_samples: int = 32, fs: Optional[FreeSystem] = None):
        super().__init__(belief, rng, fs)
        self.n_samples = int(n_samples)
        self._marg: Optional[np.ndarray] = None
        self.failures = 0

    def cost_model(self) -> dict:
        return {"knob": "n_samples", "value": self.n_samples,
                "unit": "one heuristic world draw"}

    # -- sampling -------------------------------------------------------------

    def _draw_owners(self) -> Optional[list[int]]:
        """One v0.3 draw, projected onto free-card indices."""
        try:
            deal = self.bel._sample_initial(self.rng)
        except BeliefContradiction:
            self.failures += 1
            return None
        return [deal[c] for c in self.fs.free]

    def free_marginals(self) -> np.ndarray:
        if self._marg is not None:
            return self._marg
        fs = self.fs
        acc = np.zeros((fs.n_free, NUM_PLAYERS), dtype=np.float64)
        got = 0
        for _ in range(self.n_samples):
            owners = self._draw_owners()
            if owners is None:
                continue
            for i, p in enumerate(owners):
                acc[i, p] += 1.0
            got += 1
        if got:
            acc /= got
        self._marg = acc
        return acc

    def worlds(self, n: int) -> list[list[int]]:
        out: list[list[int]] = []
        for _ in range(n):
            try:
                out.append(self.bel.sample_current_hands(self.rng))
            except BeliefContradiction:
                self.failures += 1
        return out
