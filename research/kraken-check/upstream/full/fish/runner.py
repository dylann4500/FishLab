"""Game loop: connects agents to the engine across the observation boundary.

Agents NEVER receive the GameState, only Observations. This module is the
single place where the two meet.

INFORMATION INTEGRITY NOTE: agent RNG seeds must NOT be a function of the
deal seed. Otherwise a policy could, in principle, invert its own seed to
reconstruct the hidden deal - a genuine (if exotic) leakage channel, and
exactly the kind of correlation a learned policy could latch onto. Agent
seeds therefore come from an independent stream (``agent_seed``); when it is
not supplied they are drawn from fresh OS entropy and recorded on the state,
so runs stay reproducible by recording rather than by coupling.
"""

from __future__ import annotations

import random
import secrets
from typing import Optional, Sequence

from .engine import GameState
from .observation import Observation
from .rules import RuleConfig


class GameTimeout(Exception):
    """A game exceeded max_actions: agents failed to drive it to termination."""


def play_game(agents: Sequence, rules: RuleConfig,
              seed: Optional[int] = None,
              deck_order: Optional[list[int]] = None,
              debug: bool = False,
              max_actions: int = 20000,
              agent_seed: Optional[int] = None) -> GameState:
    """Play one full game and return the terminal GameState.

    ``seed`` / ``deck_order`` control the DEAL only. ``agent_seed`` controls
    agent randomness and is deliberately independent of the deal; pass it
    explicitly for reproducible agent behavior.
    """
    state = GameState.deal(rules, seed=seed, deck_order=deck_order, debug=debug)
    if agent_seed is None:
        agent_seed = secrets.randbits(64)
    state.agent_seed = agent_seed
    agent_rng = random.Random(agent_seed)
    for p, agent in enumerate(agents):
        agent.begin_game(p, rules, agent_rng.getrandbits(64))
    n = 0
    while not state.is_terminal:
        p = state.turn
        obs = Observation.from_state(state, p)
        action = agents[p].act(obs)
        state.apply(p, action)
        n += 1
        if n >= max_actions:
            raise GameTimeout(f"game exceeded {max_actions} actions")
    return state
