"""BasicHeuristicAgent: cheap, belief-free baseline.

Tracks only directly-public card locations (successful asks, claims) with an
incremental cursor. Strategy: claim publicly-certain sets, steal known
opponent cards, otherwise ask from its strongest suits.
"""

from __future__ import annotations

from ..cards import (half_suit_cards, half_suit_mask, mask_to_cards, team_of,
                     teammates)
from ..engine import Action, Ask, Claim
from ..observation import Observation
from .base import Agent


class HeuristicAgent(Agent):
    name = "heuristic"

    def begin_game(self, player, rules, seed):
        super().begin_game(player, rules, seed)
        self._cursor = 0
        self._public_loc: dict[int, int] = {}  # card -> publicly-known holder

    def _sync(self, obs: Observation) -> None:
        from ..engine import AskEvent, ClaimEvent
        h = obs.history
        while self._cursor < len(h):
            ev = h[self._cursor]
            if isinstance(ev, AskEvent) and ev.success:
                self._public_loc[ev.card] = ev.asker
            elif isinstance(ev, ClaimEvent):
                for i in range(6):
                    self._public_loc.pop(ev.half_suit * 6 + i, None)
            self._cursor += 1
        for c in mask_to_cards(obs.hand):
            self._public_loc[c] = self.player

    def _pass(self, obs: Observation):
        return max(obs.legal_passes(), key=lambda p: obs.hand_counts[p.teammate])

    def _claim_publicly_certain(self, obs: Observation):
        """Claim a set fully covered by own hand + publicly-known teammate
        cards (guaranteed correct)."""
        me_team = team_of(self.player)
        for hs in obs.claimable_half_suits():
            if obs.hand & half_suit_mask(hs) == half_suit_mask(hs):
                return Claim(hs, (self.player,) * 6)
            assignment = []
            for c in half_suit_cards(hs):
                if obs.hand & (1 << c):
                    assignment.append(self.player)
                    continue
                holder = self._public_loc.get(c)
                if holder is None or team_of(holder) != me_team:
                    break
                assignment.append(holder)
            else:
                return Claim(hs, tuple(assignment))
        return None

    def _forced_claim(self, obs: Observation) -> Claim:
        team = teammates(self.player)
        best_hs, best_known = None, -1
        for hs in obs.claimable_half_suits():
            known = sum(1 for c in half_suit_cards(hs)
                        if obs.hand & (1 << c) or c in self._public_loc)
            if known > best_known:
                best_hs, best_known = hs, known
        assignment = []
        for c in half_suit_cards(best_hs):
            if obs.hand & (1 << c):
                assignment.append(self.player)
            elif (c in self._public_loc
                  and team_of(self._public_loc[c]) == team_of(self.player)):
                assignment.append(self._public_loc[c])
            else:
                assignment.append(self.rng.choice(team))
        return Claim(best_hs, tuple(assignment))

    def act(self, obs: Observation) -> Action:
        self._sync(obs)
        if obs.must_pass():
            return self._pass(obs)
        claim = self._claim_publicly_certain(obs)
        if claim is not None:
            return claim
        asks = obs.legal_asks()
        if not asks or (self.stalled(obs) and obs.claimable_half_suits()):
            return self._forced_claim(obs)
        steals = [a for a in asks if self._public_loc.get(a.card) == a.target]
        if steals:
            return self.rng.choice(steals)

        def suit_strength(a: Ask) -> int:
            return (obs.hand & half_suit_mask(a.card // 6)).bit_count()

        best = max(suit_strength(a) for a in asks)
        pool = [a for a in asks if suit_strength(a) == best]
        informed = [a for a in pool
                    if self._public_loc.get(a.card, a.target) == a.target]
        return self.rng.choice(informed or pool)
