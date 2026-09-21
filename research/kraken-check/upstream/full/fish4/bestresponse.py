"""A best response that is allowed to leave the champion's policy class.

WHY THE EARLIER EXPLOIT SCREEN WAS NOT THIS
-------------------------------------------
``jobs/j30_exploit_screen.json`` swept one parameter of the champion's own
objective and found nothing. That result is real but narrow: it searched INSIDE
the class, so it lower-bounds exploitability along a single axis and says
nothing about a policy shaped differently. Every other number in this project
has the same shape -- vs champion, vs v0.3 -- a ladder whose rungs are all made
of the same material.

This is different in kind. It does not score asks with a hand-designed
objective at all. It samples worlds from its own belief, plays each candidate
action out against actual champion opponents, and takes the action with the
best simulated outcome. What it prefers is therefore determined by how the
champion actually responds, not by any feature anybody chose -- which is what
"outside the class" has to mean if it is to mean anything.

IT IS NOT A CHEAT
-----------------
Everything it reads comes from its own ``BeliefState``: public events plus its
own hand. The worlds it rolls out are samples from that belief, not the true
deal. It is a legal policy, and its margin against the champion is an honest
duel result rather than a bound like ``fish4/oracle``'s.

WHAT IT IS AND IS NOT AS A BOUND
--------------------------------
This is perfect-information Monte Carlo, and PIMC has a known pathology:
inside each rollout the world is fixed and known, so the search implicitly
assumes it will be equally well informed at every future decision. That is
strategy fusion, and it makes PIMC value some lines it cannot actually reach.

The consequence for reading the result is asymmetric, and worth stating before
the number rather than after:

  * If it BEATS the champion, that margin is real -- it was won in honest play
    by a legal policy. A win lower-bounds exploitability.
  * If it LOSES, that is weak evidence. Strategy fusion may simply have made it
    play badly, and the champion could still be exploitable by a search that
    reasons about its own future uncertainty properly.

So this can demonstrate an exploit and cannot demonstrate the absence of one.
"""

from __future__ import annotations

import random
from typing import Optional

from fish.cards import NUM_PLAYERS, team_of
from fish.engine import GameState
from fish.observation import Observation

from .agent4 import FishBot4
from .askfeat import AskWeights, DecisionContext, score_asks
from .posterior import Posterior


class RolloutBR(FishBot4):
    """Choose the ask whose simulated continuation scores best.

    ``beam`` candidate actions, shortlisted by the incumbent objective purely to
    keep the cost finite, are each played out in ``worlds`` sampled worlds
    against ``opponent`` agents in every other seat. The shortlist is the one
    concession to the policy class and it only decides what gets SIMULATED --
    the ranking among the shortlisted actions comes entirely from the rollouts.
    """

    def __init__(self, beam: int = 4, worlds: int = 4, max_depth: int = 220,
                 opponent=("fishbot4", {"opponent_gamma": 0.35}), **kwargs):
        super().__init__(**kwargs)
        self.beam = beam
        self.n_rollout_worlds = worlds
        self.max_depth = max_depth
        self.opponent = opponent
        self.rollouts = 0
        self.br_decisions = 0

    # -- the rollout ---------------------------------------------------------

    def _playout(self, hands, set_winner, history, turn, action, seed) -> float:
        """Apply ``action`` in this world, then let champions finish. Returns
        the final differential from THIS seat's team's point of view."""
        from .registry4 import make_agent

        st = GameState(self.rules, list(hands), turn)
        st.set_winner = list(set_winner)
        st.history = list(history)
        try:
            st.apply(turn, action)
        except Exception:
            return float("-inf")        # illegal in this world; never chosen

        rng = random.Random(seed)
        agents = []
        for p in range(NUM_PLAYERS):
            a = make_agent((self.opponent[0], dict(self.opponent[1])))
            a.begin_game(p, self.rules, rng.getrandbits(64))
            agents.append(a)

        n = 0
        while not st.is_terminal and n < self.max_depth:
            p = st.turn
            try:
                act = agents[p].act(Observation.from_state(st, p))
                st.apply(p, act)
            except Exception:
                break                   # a stuck rollout scores what it banked
            n += 1
        a, b, _ = st.scores()
        mine = a if team_of(self.player) == 0 else b
        theirs = b if team_of(self.player) == 0 else a
        return float(mine - theirs)

    # -- policy --------------------------------------------------------------

    def act(self, obs: Observation):
        self.bel.update(obs)

        exact = self.tablebase_action(obs)
        if exact is not None:
            return exact                # solved is better than simulated

        asks = obs.legal_asks()
        if len(asks) < 2:
            return super().act(obs)

        post = Posterior(self.bel, self.rng, n_draws=self.n_draws,
                         n_worlds=max(self.n_worlds, self.n_rollout_worlds),
                         mode=self.infer_mode, obs=obs,
                         gamma=self.opponent_gamma)
        ctx = DecisionContext(obs, self.bel, post)
        scores, _ = score_asks(ctx, asks, self.weights)
        order = sorted(range(len(asks)), key=lambda i: -scores[i])
        shortlist = [asks[i] for i in order[:self.beam]]

        worlds = post.worlds()[:self.n_rollout_worlds]
        if not worlds:
            return super().act(obs)

        self.br_decisions += 1
        best, best_v = None, float("-inf")
        for a in shortlist:
            tot = 0.0
            for wi, w in enumerate(worlds):
                tot += self._playout(w, obs.set_winner, obs.history,
                                     obs.turn, a,
                                     self.rng.getrandbits(31))
                self.rollouts += 1
            v = tot / len(worlds)
            if v > best_v:
                best, best_v = a, v
        return best if best is not None else super().act(obs)
