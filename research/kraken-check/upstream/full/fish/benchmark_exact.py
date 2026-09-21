"""Absolute strength benchmark: agreement with EXACT optimal play.

Every other metric in this project is relative ("beats the previous
version"). This one is absolute: in positions small enough to solve exactly,
how often does an agent choose a provably optimal action?

Method
------
1. Play real games until only one half-suit remains live and few enough
   cards are on the table to enumerate (<= 9).
2. Solve the position exactly (perfect information, layered value
   iteration).
3. Ask the agent for its action from its LEGAL observation only, and check
   whether that action is among the exact optima.

Honest caveat, stated up front: the exact solution is the
perfect-information optimum. An agent acting under uncertainty cannot always
match it, and *should* not be expected to when information is genuinely
missing. Two readings are therefore reported separately:

- ``resolved``: positions where every remaining card's location is already
  publicly determined. Here hidden information is not a factor, the
  perfect-information optimum IS the optimum, and a strong agent should
  approach 100%. Shortfalls here are real, diagnosable defects.
- ``unresolved``: positions with genuine uncertainty. Agreement is a useful
  comparative signal between agents but is NOT expected to reach 100%.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .agents import AGENT_REGISTRY
from .engine import GameState
from .exact import (SubgameTooLarge, information_is_resolved, live_card_count,
                    solve_position)
from .observation import Observation
from .rules import RuleConfig


@dataclass
class ExactBenchResult:
    agent: str
    resolved_total: int = 0
    resolved_optimal: int = 0
    unresolved_total: int = 0
    unresolved_optimal: int = 0
    value_loss: list = field(default_factory=list)   # optimal - chosen value

    @property
    def resolved_rate(self) -> float:
        return self.resolved_optimal / max(1, self.resolved_total)

    @property
    def unresolved_rate(self) -> float:
        return self.unresolved_optimal / max(1, self.unresolved_total)

    @property
    def mean_value_loss(self) -> float:
        return sum(self.value_loss) / max(1, len(self.value_loss))

    def summary(self) -> str:
        return (f"{self.agent:16s} "
                f"resolved {self.resolved_rate:6.1%} "
                f"({self.resolved_optimal}/{self.resolved_total})   "
                f"uncertain {self.unresolved_rate:6.1%} "
                f"({self.unresolved_optimal}/{self.unresolved_total})   "
                f"mean value loss {self.mean_value_loss:.3f} sets")

    def to_dict(self) -> dict:
        return {"agent": self.agent,
                "resolved_rate": self.resolved_rate,
                "resolved_n": self.resolved_total,
                "unresolved_rate": self.unresolved_rate,
                "unresolved_n": self.unresolved_total,
                "mean_value_loss": self.mean_value_loss}


def collect_solvable_positions(n_games: int = 200, seed: int = 0,
                               generator: str = "probabilistic",
                               max_live_cards: int = 7,
                               max_per_game: int = 3) -> list:
    """Default max_live_cards is 7: at 8 the layer is ~1.7M placements, which
    exceeds the solver budget and merely wastes time before being refused."""
    """Play games and snapshot positions small enough to solve exactly."""
    rules = RuleConfig()
    rng = random.Random(seed)
    out = []
    for g in range(n_games):
        state = GameState.deal(rules, seed=seed * 100003 + g)
        agents = [AGENT_REGISTRY[generator]() for _ in range(6)]
        for p, a in enumerate(agents):
            a.begin_game(p, rules, rng.getrandbits(64))
        taken = 0
        guard = 0
        while not state.is_terminal and guard < 3000:
            live_sets = sum(1 for w in state.set_winner if w is None)
            live_cards = live_card_count(state)
            if (live_sets == 1 and 2 <= live_cards <= max_live_cards
                    and taken < max_per_game):
                p = state.turn
                obs = Observation.from_state(state, p)
                if obs.legal_asks() or obs.claimable_half_suits():
                    out.append({
                        "hands": list(state.hands),
                        "set_winner": list(state.set_winner),
                        "turn": state.turn,
                        "history": tuple(state.history),
                        "resolved": information_is_resolved(state),
                    })
                    taken += 1
            p = state.turn
            state.apply(p, agents[p].act(Observation.from_state(state, p)))
            guard += 1
    return out


def solve_positions(positions: list) -> list:
    """Solve every position ONCE and attach the solution.

    Solving dominates benchmark runtime, so doing it per-agent wasted a
    factor equal to the number of agents compared.
    """
    rules = RuleConfig()
    solved = []
    for pos in positions:
        state = GameState.from_components(
            rules, pos["hands"], pos["turn"], pos["set_winner"])
        state.history = list(pos["history"])
        try:
            sol = solve_position(state)
        except Exception:
            continue
        if not sol["optimal_actions"]:
            continue
        solved.append({**pos, "state": state, "solution": sol})
    return solved


def benchmark_agent(agent_name: str, positions: list, kwargs: dict | None = None,
                    seed: int = 0) -> ExactBenchResult:
    """``positions`` may be raw snapshots or the output of solve_positions."""
    rules = RuleConfig()
    res = ExactBenchResult(agent=agent_name)
    rng = random.Random(seed)
    for pos in positions:
        if "solution" in pos:
            state, sol = pos["state"], pos["solution"]
        else:
            state = GameState.from_components(
                rules, pos["hands"], pos["turn"], pos["set_winner"])
            state.history = list(pos["history"])
            try:
                sol = solve_position(state)
            except Exception:
                continue
        solver = sol["solver"]
        optimal = sol["optimal_actions"]
        if not optimal:
            continue
        seat = state.turn
        agent = AGENT_REGISTRY[agent_name](**(kwargs or {}))
        agent.begin_game(seat, rules, rng.getrandbits(64))
        obs = Observation.from_state(state, seat)
        try:
            action = agent.act(obs)
        except Exception:
            continue
        try:
            chosen_value = solver.action_value(state, action)
        except Exception:
            continue
        is_opt = any(action == o for o in optimal)
        res.value_loss.append(max(0.0, sol["value"] - chosen_value))
        if pos["resolved"]:
            res.resolved_total += 1
            res.resolved_optimal += int(is_opt)
        else:
            res.unresolved_total += 1
            res.unresolved_optimal += int(is_opt)
    return res
