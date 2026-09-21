"""SearchAgent: determinized Monte Carlo search (PIMC-style), plus the
shared static evaluation used by every search variant.

Kept as a baseline: it LOSES to its own prior policy (see RESEARCH_LOG.md),
which is what motivated the variance diagnosis and PairedSearchAgent.
"""

from __future__ import annotations

from ..cards import half_suit_mask, num_half_suits, team_of
from ..engine import Action, Ask, Claim, GameState
from ..observation import Observation
from .heuristic import HeuristicAgent
from .probabilistic import ProbabilisticAgent


def evaluate(state: GameState, my_team: int) -> float:
    """Team-perspective evaluation: resolved set differential plus a
    card-control potential over unresolved half-suits. A half-suit fully
    held by one team is essentially banked (the holders can always claim it
    once located), so it is scored close to a full set."""
    a, b, _ = state.scores()
    diff = (a - b) if my_team == 0 else (b - a)
    potential = 0.0
    for hs in range(num_half_suits(state.rules.variant)):
        if state.set_winner[hs] is not None:
            continue
        m = half_suit_mask(hs)
        mine = other = 0
        for p in range(6):
            k = (state.hands[p] & m).bit_count()
            if team_of(p) == my_team:
                mine += k
            else:
                other += k
        if mine == 6:
            potential += 0.9
        elif other == 6:
            potential -= 0.9
        else:
            potential += 0.35 * (mine - other) / 6.0
    return diff + potential


class SearchAgent(ProbabilisticAgent):
    """Same belief machinery and claim logic as its parent, but ask
    selection is decided by determinized rollouts."""

    name = "search"

    def __init__(self, n_worlds: int = 8, n_candidates: int = 5,
                 rollout_depth: int = 32, n_samples: int = 32,
                 claim_threshold: float = 0.97):
        super().__init__(n_samples=n_samples, claim_threshold=claim_threshold)
        self.n_worlds = n_worlds
        self.n_candidates = n_candidates
        self.rollout_depth = rollout_depth

    def _rollout_value(self, obs: Observation, world: list[int],
                       action: Action) -> float:
        state = GameState.from_components(
            obs.rules, list(world), obs.turn, list(obs.set_winner))
        try:
            state.apply(self.player, action)
        except Exception:
            return -99.0
        rollers = [HeuristicAgent() for _ in range(6)]
        for p, r in enumerate(rollers):
            r.begin_game(p, self.rules, self.rng.getrandbits(64))
        for _ in range(self.rollout_depth):
            if state.is_terminal:
                break
            p = state.turn
            state.apply(p, rollers[p].act(Observation.from_state(state, p)))
        return evaluate(state, team_of(self.player))

    def act(self, obs: Observation) -> Action:
        self.bel.update(obs)
        if obs.must_pass():
            return max(obs.legal_passes(),
                       key=lambda p: obs.hand_counts[p.teammate])
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
            strength = (obs.hand & half_suit_mask(a.card // 6)).bit_count()
            scored.append((hits / n + self.suit_bonus * strength, a))
        scored.sort(key=lambda t: t[0], reverse=True)
        candidates: list[Action] = [a for _, a in scored[: self.n_candidates]]
        if best_claim and best_claim[0] >= 0.5:
            candidates.append(best_claim[1])
        if len(candidates) == 1:
            return candidates[0]
        sim_worlds = worlds[: self.n_worlds]
        best_action, best_value = None, float("-inf")
        for action in candidates:
            total = sum(self._rollout_value(obs, w, action) for w in sim_worlds)
            value = total / len(sim_worlds)
            if value > best_value:
                best_action, best_value = action, value
        return best_action
