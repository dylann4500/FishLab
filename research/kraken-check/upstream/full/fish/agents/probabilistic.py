"""ProbabilisticAgent: full belief tracking + sampled-world decision making.

Asks maximize P(success) weighted by suit progress; claims fire when the
modal declared distribution is correct in >= claim_threshold of sampled
worlds. Sampling comes from the exact BeliefState, so probabilities respect
every logical constraint observed so far.
"""

from __future__ import annotations

from collections import Counter

from ..beliefs import BeliefState
from ..cards import (NUM_PLAYERS, half_suit_cards, half_suit_mask, team_of,
                     teammates)
from ..engine import Action, Ask, Claim
from ..observation import Observation
from .base import Agent
from .tablebase import ExactEndgameMixin


class ProbabilisticAgent(ExactEndgameMixin, Agent):
    name = "probabilistic"

    def __init__(self, n_samples: int = 32, claim_threshold: float = 0.97,
                 suit_bonus: float = 0.06, use_tablebase: bool = True,
                 tablebase_max_cards: int = 8):
        super().__init__()
        self.n_samples = n_samples
        self.claim_threshold = claim_threshold
        self.suit_bonus = suit_bonus
        self.use_tablebase = use_tablebase
        self.tablebase_max_cards = tablebase_max_cards

    def begin_game(self, player, rules, seed):
        super().begin_game(player, rules, seed)
        self.bel = BeliefState(rules, observer=player)

    # -- sampling utilities ------------------------------------------------------

    def _sample_worlds(self) -> list[list[int]]:
        return [self.bel.sample_current_hands(self.rng)
                for _ in range(self.n_samples)]

    @staticmethod
    def _holder_in_world(world: list[int], card: int) -> int:
        bit = 1 << card
        for p in range(NUM_PLAYERS):
            if world[p] & bit:
                return p
        return -1  # resolved

    def _claim_candidates(self, obs: Observation, worlds) -> list[tuple[float, Claim]]:
        """(P(exactly correct), Claim) for the modal all-in-team distribution
        of each claimable half-suit."""
        me_team = team_of(self.player)
        out = []
        for hs in obs.claimable_half_suits():
            cards = list(half_suit_cards(hs))
            counter: Counter = Counter()
            for w in worlds:
                dist = tuple(self._holder_in_world(w, c) for c in cards)
                if all(h >= 0 and team_of(h) == me_team for h in dist):
                    counter[dist] += 1
            if counter:
                dist, k = counter.most_common(1)[0]
                out.append((k / len(worlds), Claim(hs, dist)))
        return out

    def _forced_claim(self, obs: Observation, worlds) -> Claim:
        cands = self._claim_candidates(obs, worlds)
        if cands:
            return max(cands, key=lambda t: t[0])[1]
        team = teammates(self.player)
        hs = obs.claimable_half_suits()[0]
        assignment = tuple(self.player if obs.hand & (1 << c)
                           else self.rng.choice(team)
                           for c in half_suit_cards(hs))
        return Claim(hs, assignment)

    # -- policy -------------------------------------------------------------------

    def act(self, obs: Observation) -> Action:
        self.bel.update(obs)
        if obs.must_pass():
            return max(obs.legal_passes(),
                       key=lambda p: obs.hand_counts[p.teammate])
        # If our own beliefs already determine every remaining card, the
        # position has no hidden information left: solve it instead of
        # guessing at it.
        exact = self.tablebase_action(obs)
        if exact is not None:
            return exact
        worlds = self._sample_worlds()
        claim_cands = self._claim_candidates(obs, worlds)
        best_claim = max(claim_cands, key=lambda t: t[0]) if claim_cands else None
        if best_claim and best_claim[0] >= self.claim_threshold:
            return best_claim[1]
        asks = obs.legal_asks()
        if not asks or (self.stalled(obs) and obs.claimable_half_suits()):
            return self._forced_claim(obs, worlds)
        n = len(worlds)
        scored = []
        for a in asks:
            hits = sum(1 for w in worlds if w[a.target] & (1 << a.card))
            p = hits / n
            strength = (obs.hand & half_suit_mask(a.card // 6)).bit_count()
            scored.append((p + self.suit_bonus * strength, p, a))
        scored.sort(key=lambda t: t[0], reverse=True)
        best_score = scored[0][0]
        if scored[0][1] == 0.0:
            if best_claim and best_claim[0] >= 0.5:
                return best_claim[1]
        pool = [t for t in scored if t[0] >= best_score - 1e-9]
        return self.rng.choice(pool)[2]
