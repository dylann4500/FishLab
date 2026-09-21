"""MemoryAgent: perfect logical bookkeeping, certainty-only decisions.

Uses the exact BeliefState but acts ONLY on facts derivable with certainty:
guaranteed steals, guaranteed-correct claims. Everything else falls back to
informed-random play (never asking for a card the target provably lacks).
"""

from __future__ import annotations

from ..beliefs import BeliefState
from ..cards import half_suit_cards, half_suit_mask, team_of, teammates
from ..engine import Action, Ask, Claim
from ..observation import Observation
from .base import Agent
from .tablebase import ExactEndgameMixin


class MemoryAgent(ExactEndgameMixin, Agent):
    name = "memory"

    def __init__(self, use_tablebase: bool = True,
                 tablebase_max_cards: int = 8):
        super().__init__()
        self.use_tablebase = use_tablebase
        self.tablebase_max_cards = tablebase_max_cards

    def begin_game(self, player, rules, seed):
        super().begin_game(player, rules, seed)
        self.bel = BeliefState(rules, observer=player)

    def _certain_claim(self, obs: Observation):
        """A claim guaranteed correct: all six current locations known and
        within our team."""
        me_team = team_of(self.player)
        for hs in obs.claimable_half_suits():
            assignment = []
            for c in half_suit_cards(hs):
                m = self.bel.current_holder_mask(c)
                if m == 0 or m & (m - 1):
                    break  # resolved or uncertain
                holder = m.bit_length() - 1
                if team_of(holder) != me_team:
                    break
                assignment.append(holder)
            else:
                return Claim(hs, tuple(assignment))
        return None

    def _forced_claim(self, obs: Observation) -> Claim:
        team = teammates(self.player)
        me_team = team_of(self.player)
        best = None
        for hs in obs.claimable_half_suits():
            assignment, known = [], 0
            for c in half_suit_cards(hs):
                m = self.bel.current_holder_mask(c)
                if m and m & (m - 1) == 0:
                    holder = m.bit_length() - 1
                    known += 1
                    assignment.append(holder if team_of(holder) == me_team
                                      else self.rng.choice(team))
                else:
                    assignment.append(self.rng.choice(team))
            if best is None or known > best[0]:
                best = (known, hs, assignment)
        return Claim(best[1], tuple(best[2]))

    def act(self, obs: Observation) -> Action:
        self.bel.update(obs)
        if obs.must_pass():
            return max(obs.legal_passes(),
                       key=lambda p: obs.hand_counts[p.teammate])
        exact = self.tablebase_action(obs)
        if exact is not None:
            return exact          # nothing hidden: solve, do not guess
        claim = self._certain_claim(obs)
        if claim is not None:
            return claim
        asks = obs.legal_asks()
        if not asks or (self.stalled(obs) and obs.claimable_half_suits()):
            return self._forced_claim(obs)

        def strength(a: Ask) -> int:
            return (obs.hand & half_suit_mask(a.card // 6)).bit_count()

        steals = [a for a in asks
                  if self.bel.current_holder_mask(a.card) == 1 << a.target]
        if steals:
            best = max(strength(a) for a in steals)
            return self.rng.choice([a for a in steals if strength(a) == best])
        viable = [a for a in asks
                  if self.bel.current_holder_mask(a.card) & (1 << a.target)]
        pool = viable or asks
        best = max(strength(a) for a in pool)
        return self.rng.choice([a for a in pool if strength(a) == best])
