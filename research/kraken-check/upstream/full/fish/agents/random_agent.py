"""Uniform-random legal agent. Exists to validate the environment and to act
as the rating anchor. Claims randomly with small probability so games
terminate (asks alone never resolve a half-suit)."""

from __future__ import annotations

from ..cards import teammates
from ..engine import Action, Claim
from ..observation import Observation
from .base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, claim_prob: float = 0.02) -> None:
        super().__init__()
        self.claim_prob = claim_prob

    def _random_claim(self, obs: Observation) -> Claim:
        hs = self.rng.choice(obs.claimable_half_suits())
        team = teammates(self.player)
        assignment = tuple(self.rng.choice(team) for _ in range(6))
        return Claim(hs, assignment)

    def act(self, obs: Observation) -> Action:
        if obs.must_pass():
            return self.rng.choice(obs.legal_passes())
        asks = obs.legal_asks()
        if not asks or (obs.claimable_half_suits()
                        and self.rng.random() < self.claim_prob):
            return self._random_claim(obs)
        return self.rng.choice(asks)
