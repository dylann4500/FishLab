"""Agent interface. Agents see Observations only, never engine state."""

from __future__ import annotations

import random

from ..engine import Action
from ..observation import Observation
from ..rules import RuleConfig


class Agent:
    name = "agent"

    def __init__(self) -> None:
        self.player: int = -1
        self.rules: RuleConfig | None = None
        self.rng = random.Random()

    def begin_game(self, player: int, rules: RuleConfig, seed: int) -> None:
        self.player = player
        self.rules = rules
        self.rng = random.Random(seed)

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    @staticmethod
    def stalled(obs: Observation, window: int = 80) -> bool:
        """True when no half-suit has been RESOLVED in the last ``window``
        actions, meaning the game is making no real progress and somebody
        has to gamble a claim to break the deadlock.

        Earlier versions looked for a run of purely failed asks, which
        missed the common livelock where cards shuttle back and forth
        between two opponents: successful asks kept resetting the counter
        while the game went nowhere. Resolution progress is the honest
        measure, so that is what we track.
        """
        from ..engine import ClaimEvent
        h = obs.history
        if len(h) < window:
            return False
        for ev in h[-window:]:
            if isinstance(ev, ClaimEvent):
                return False
        return True
