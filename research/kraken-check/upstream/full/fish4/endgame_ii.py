"""Play the m = 1 endgame from the exact solution instead of estimating it.

``fish4/tablebase4.py`` already solves a position when the belief pins every
live card -- there the game is perfect-information and the closed form applies.
This covers the other case: $m = 1$ with cards genuinely hidden, where
``scripts4/ii_action_diff.py`` measures that the champion picks the exact
optimum at under a third of decisions and the ones it gets wrong cost about
half a set each.

WHAT THIS ASSUMES, AND WHERE THAT IS WRONG
------------------------------------------
``fish4.exact_ii`` computes a best response to CHAMPION opponents. Every seat
but ours is modelled as a deterministic realisation of the champion, seeded
from its observation. Two consequences, both real:

* Against champions, the model is right about the opponents and right about a
  teammate that is also a champion. That is the configuration
  ``scripts4/duel.py`` measures, and the configuration this is honest in.
* If our TEAMMATE also runs this, the model is wrong about them: they are no
  longer playing the champion's move. Nothing here detects that. A team of two
  such agents is playing a best response to a partner that does not exist, and
  whether that is better or worse than the heuristic is an empirical question
  this module does not answer -- the duel does.

It is off by default for that reason, and because a best response is not an
equilibrium strategy: being unexploitable is not what it optimises.

THE BUDGET IS SMALL ON PURPOSE
------------------------------
The study version allows 300,000 nodes and 24 deals of support because it is
measuring, not playing. In play the same search sits inside a move clock and
inside a duel of thousands of games, so the caps here are much tighter and a
position that exceeds them falls back to the heuristic rather than stalling.
Falling back is not a failure: it is the same move the champion would have
played anyway.
"""

from __future__ import annotations

from fish.engine import Claim, GameState
from fish.observation import Observation

from .exact_ii import ExactII, SolveTimeout, consistent_deals


class ExactEndgameMixin:
    """Mixed into the agent ahead of ``Agent``; see fish4/agent4.py."""

    def exact_ii_action(self, obs: Observation):
        if not getattr(self, "exact_endgame", False):
            return None
        live = [h for h, w in enumerate(obs.set_winner) if w is None]
        if len(live) != 1:
            return None
        try:
            deals = consistent_deals(obs, self.bel, live[0])
        except Exception:
            return None
        # Support 1 is the tablebase's position, not ours, and it gets there
        # first in agent4.act. Anything above the cap is left to the heuristic.
        if not 1 < len(deals) <= self.exact_endgame_max_support:
            return None
        states = []
        for hands in deals:
            t = GameState.from_components(self.rules, list(hands), obs.player,
                                          list(obs.set_winner))
            t.history = list(obs.history)
            states.append(t)
        w = [1.0 / len(states)] * len(states)
        sv = ExactII(self.rules, live[0], obs.player, self.exact_endgame_spec)
        sv.max_nodes = self.exact_endgame_max_nodes
        try:
            sv.solve(states, w)
        except SolveTimeout:
            return None
        except Exception:
            return None
        action = sv.best_action
        if action is None:
            return None
        # The same two guards the tablebase path carries. A claim the solver
        # likes can still be one a counterfactual replay is deliberately
        # holding open, and an action legal in the solver's reconstruction has
        # to be legal in the real position too.
        banned = getattr(getattr(self, "claim_cfg", None), "banned", ())
        if banned and isinstance(action, Claim) and action.half_suit in banned:
            return None
        try:
            states[0].check_legal(obs.player, action)
        except Exception:
            return None
        return action
