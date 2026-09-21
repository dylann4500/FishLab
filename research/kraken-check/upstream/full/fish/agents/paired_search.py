"""PairedSearchAgent: search v2, built from the search-failure diagnosis.

Measured problem (scripts/diagnose_search.py): the spread of rollout values
across sampled worlds is ~2.4x LARGER than the gap between the best and
worst candidate action. Any search that evaluates different actions on
different worlds (UCT/ISMCTS with per-iteration resampling, or independent
PIMC batches) is therefore comparing apples to oranges, which is exactly
why both search v1 and ISMCTS lost to their own prior.

Fixes applied here:
1. COMMON RANDOM NUMBERS - every candidate action is evaluated on the SAME
   worlds with the SAME rollout seeds, so world luck cancels in the
   comparison.
2. PAIRED SIGNIFICANCE - an action only displaces the prior's choice when
   its per-world *difference* against the prior choice is significantly
   positive. Search refines a strong prior instead of replacing it with
   noise.
"""

from __future__ import annotations

import math
import random

from ..cards import half_suit_mask, team_of
from ..engine import Action, GameState
from ..observation import Observation
from .heuristic import HeuristicAgent
from .probabilistic import ProbabilisticAgent
from .search import evaluate


class PairedSearchAgent(ProbabilisticAgent):
    name = "paired_search"

    def __init__(self, n_worlds: int = 12, n_candidates: int = 5,
                 rollout_depth: int = 30, n_samples: int = 32,
                 claim_threshold: float = 0.97, t_threshold: float = 1.5,
                 suit_bonus: float = 0.06):
        super().__init__(n_samples=n_samples, claim_threshold=claim_threshold,
                         suit_bonus=suit_bonus)
        self.n_worlds = n_worlds
        self.n_candidates = n_candidates
        self.rollout_depth = rollout_depth
        self.t_threshold = t_threshold

    def _rollout(self, obs: Observation, world: list[int], action: Action,
                 roll_seed: int) -> float:
        state = GameState.from_components(
            obs.rules, list(world), obs.turn, list(obs.set_winner))
        try:
            state.apply(self.player, action)
        except Exception:
            return float("nan")  # illegal in this world; excluded from stats
        rng = random.Random(roll_seed)
        rollers = [HeuristicAgent() for _ in range(6)]
        for p, r in enumerate(rollers):
            r.begin_game(p, self.rules, rng.getrandbits(64))
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
        exact = self.tablebase_action(obs)
        if exact is not None:
            return exact          # nothing hidden: solve, do not estimate
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
        scored.sort(key=lambda t: -t[0])
        candidates: list[Action] = [a for _, a in scored[: self.n_candidates]]
        if best_claim and best_claim[0] >= 0.5:
            candidates.append(best_claim[1])
        baseline = candidates[0]
        if len(candidates) == 1:
            return baseline

        sim_worlds = worlds[: self.n_worlds]
        roll_seeds = [self.rng.getrandbits(32) for _ in sim_worlds]
        values = [[self._rollout(obs, w, action, s)
                   for w, s in zip(sim_worlds, roll_seeds)]
                  for action in candidates]

        base_vals = values[0]
        best_idx, best_stat = 0, 0.0
        for i in range(1, len(candidates)):
            diffs = [v - b for v, b in zip(values[i], base_vals)
                     if not (math.isnan(v) or math.isnan(b))]
            if len(diffs) < 3:
                continue
            mean = sum(diffs) / len(diffs)
            var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
            se = math.sqrt(var / len(diffs)) if var > 0 else 0.0
            stat = (math.inf if mean > 0 else 0.0) if se == 0 else mean / se
            if stat > self.t_threshold and stat > best_stat:
                best_idx, best_stat = i, stat
        return candidates[best_idx]
