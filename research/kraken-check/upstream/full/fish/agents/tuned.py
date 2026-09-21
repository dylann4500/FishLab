"""TunedAgent: every strategic hypothesis about ASKING is an ablatable weight.

The strongest agent scores asks as ``P(success) + 0.06 * suit_progress``.
That ignores several things a strong human obviously considers, most
glaringly **who receives the turn when the ask fails**. Two asks with equal
success probability are not equally good if one hands the turn to an
opponent holding nine cards and the other to one holding two.

Each term below is a separate strategy question from STRATEGY_BOOK.md, and
setting its weight to zero ablates exactly that idea, so paired duels and
the exact-agreement benchmark answer "is this real?" instead of "does my
intuition feel right?".

Ask score =        P(success)
  + w_suit    * own depth in the half-suit
  + w_turn    * turn-risk of the target receiving the turn on failure
  + w_reveal  * penalty for exposing a half-suit we have not shown yet
  + w_scarce  * bonus for contested suits our team is winning
  + w_deplete * bonus for draining an opponent toward zero cards
"""

from __future__ import annotations

from ..cards import (cards_per_player, half_suit_mask, num_half_suits,
                     team_of)
from ..engine import Ask
from ..observation import Observation
from .probabilistic import ProbabilisticAgent


class TunedAgent(ProbabilisticAgent):
    name = "tuned"

    def __init__(self, n_samples: int = 32, claim_threshold: float = 0.97,
                 w_suit: float = 0.06, w_turn: float = 0.0,
                 w_reveal: float = 0.0, w_scarce: float = 0.0,
                 w_deplete: float = 0.0, use_tablebase: bool = True):
        super().__init__(n_samples=n_samples, claim_threshold=claim_threshold,
                         suit_bonus=w_suit, use_tablebase=use_tablebase)
        self.w_suit = w_suit
        self.w_turn = w_turn
        self.w_reveal = w_reveal
        self.w_scarce = w_scarce
        self.w_deplete = w_deplete

    # -- strategic terms ---------------------------------------------------

    def _revealed_suits(self, obs: Observation) -> set[int]:
        """Half-suits in which we have already publicly shown a holding."""
        from ..engine import AskEvent
        out = set()
        me = obs.player
        for ev in obs.history:
            if isinstance(ev, AskEvent):
                if ev.asker == me or (ev.success and ev.target == me):
                    out.add(ev.card // 6)
        return out

    def _turn_risk(self, obs: Observation, target: int) -> float:
        """How dangerous is it to hand THIS opponent the turn?

        Proxy: how much of the live material they hold. An opponent with
        many cards converts the turn into more asks and more claims; an
        opponent down to one or two cards can do little with it.
        Normalized to roughly [-1, +1] and negated, since risk is a cost.
        """
        live = [c for c in obs.hand_counts if c > 0]
        if not live:
            return 0.0
        avg = sum(live) / len(live)
        per = cards_per_player(obs.rules.variant)
        return -(obs.hand_counts[target] - avg) / max(1.0, per)

    def _team_share(self, obs: Observation, hs: int, worlds) -> float:
        m = half_suit_mask(hs)
        my_team = team_of(obs.player)
        tot = 0.0
        for w in worlds:
            tot += sum((w[p] & m).bit_count() for p in range(6)
                       if team_of(p) == my_team)
        return tot / (6.0 * max(1, len(worlds)))

    def _score_ask(self, obs: Observation, ask: Ask, p_success: float,
                   revealed: set[int], worlds) -> float:
        hs = ask.card // 6
        score = p_success
        # NOTE: this must match ProbabilisticAgent's term EXACTLY (raw card
        # count, not normalized), so that all-zero weights reproduce the
        # baseline. An earlier version divided by 6, which silently changed
        # two things at once and made every ablation a confounded comparison.
        score += self.w_suit * (obs.hand & half_suit_mask(hs)).bit_count()
        if self.w_turn:
            # only the failure branch hands over the turn
            score += self.w_turn * (1.0 - p_success) * self._turn_risk(
                obs, ask.target)
        if self.w_reveal and hs not in revealed:
            score -= self.w_reveal
        if self.w_scarce or self.w_deplete:
            if self.w_scarce:
                share = self._team_share(obs, hs, worlds)
                score += self.w_scarce * (share - 0.5) * 2.0
            if self.w_deplete:
                per = cards_per_player(obs.rules.variant)
                score += self.w_deplete * p_success * (
                    1.0 - obs.hand_counts[ask.target] / per)
        return score

    # -- policy ------------------------------------------------------------

    def act(self, obs: Observation):
        self.bel.update(obs)
        if obs.must_pass():
            return max(obs.legal_passes(),
                       key=lambda p: obs.hand_counts[p.teammate])
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
        revealed = self._revealed_suits(obs) if self.w_reveal else set()
        n = len(worlds)
        scored = []
        for a in asks:
            hits = sum(1 for w in worlds if w[a.target] & (1 << a.card))
            p = hits / n
            scored.append((self._score_ask(obs, a, p, revealed, worlds), p, a))
        scored.sort(key=lambda t: -t[0])
        if scored[0][1] == 0.0 and best_claim and best_claim[0] >= 0.5:
            return best_claim[1]
        top = scored[0][0]
        pool = [t for t in scored if t[0] >= top - 1e-9]
        return self.rng.choice(pool)[2]
