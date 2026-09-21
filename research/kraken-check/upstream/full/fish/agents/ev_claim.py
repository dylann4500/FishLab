"""EVClaimAgent: probabilistic play with a decision-theoretic claim policy.

Identical to ProbabilisticAgent except that claiming is decided by comparing
the expected value of claiming now against the expected value of waiting,
rather than by a hand-picked confidence threshold. Everything else is held
constant so a paired duel isolates exactly one change.
"""

from __future__ import annotations

from ..cards import half_suit_cards, half_suit_mask, team_of
from ..engine import Action
from ..observation import Observation
from .claim_policy import ClaimEV, best_claim_decision
from .probabilistic import ProbabilisticAgent


class EVClaimAgent(ProbabilisticAgent):
    name = "ev_claim"

    def __init__(self, n_samples: int = 32, suit_bonus: float = 0.06,
                 margin: float = 0.0, p_improve: float = 0.55,
                 p_opponent_takes: float = 0.10):
        # claim_threshold is inherited but deliberately unused; the EV rule
        # replaces it. Kept at 1.01 so no inherited path can fire it.
        super().__init__(n_samples=n_samples, claim_threshold=1.01,
                         suit_bonus=suit_bonus)
        self.margin = margin
        self.p_improve = p_improve
        self.p_opponent_takes = p_opponent_takes

    def _claim_evs(self, obs: Observation, worlds) -> list[ClaimEV]:
        """Per half-suit: P(exact declared split) and P(team holds all six)."""
        me_team = team_of(self.player)
        out = []
        n = max(1, len(worlds))
        for prob, claim in self._claim_candidates(obs, worlds):
            hs = claim.half_suit
            m = half_suit_mask(hs)
            team_holds = 0
            for w in worlds:
                k = sum((w[p] & m).bit_count() for p in range(6)
                        if team_of(p) == me_team)
                if k == 6:
                    team_holds += 1
            out.append(ClaimEV(p_exact=prob, p_team_holds=team_holds / n,
                               hs=hs, claim=claim))
        return out

    def act(self, obs: Observation) -> Action:
        self.bel.update(obs)
        if obs.must_pass():
            return max(obs.legal_passes(),
                       key=lambda p: obs.hand_counts[p.teammate])
        worlds = self._sample_worlds()
        evs = self._claim_evs(obs, worlds)
        decision = best_claim_decision(
            evs, obs.rules.wrong_distribution_outcome, margin=self.margin,
            p_improve=self.p_improve, p_opponent_takes=self.p_opponent_takes)
        if decision is not None:
            return decision[0]
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
        scored.sort(key=lambda t: -t[0])
        if scored[0][1] == 0.0 and evs:
            best = max(evs, key=lambda c: c.p_exact)
            if best.p_exact >= 0.5:
                return best.claim
        top = scored[0][0]
        pool = [t for t in scored if t[0] >= top - 1e-9]
        return self.rng.choice(pool)[2]
