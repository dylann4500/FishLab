"""Exact endgame play for v0.4, on the wider and faster solver.

WHAT CHANGES FROM v0.3
----------------------
v0.3 solved a position exactly when the agent's own beliefs pinned every live
card, using a solver that enumerated card placements and therefore stopped at
one unresolved half-suit. Two things improve here, and both matter in play
rather than only on paper.

**Speed.** The count abstraction in ``exact2`` (the perfect-information value of
a position depends only on how many cards of each live half-suit each player
holds, never on which ones) shrinks a one-half-suit layer from 279,936 card
placements to 2,772 abstract states. Measured on the same position, 0.051s
against 1.774s: a 35x speedup on what profiling showed to be roughly 40% of the
policy's runtime.

**Correctness in a loopy game.** Fish's state graph is cyclic, and in a cyclic
game Bellman-optimality is NOT optimality: if a side can force +1, every action
that keeps the value at +1 looks optimal, including one that walks a cycle
forever and actually banks nothing. ``exact2`` carries an attractor rank so the
favoured side must strictly descend it, and this module asks for a
progress-optimal action rather than merely a Bellman-optimal one.

LEAK-FREE BY CONSTRUCTION
-------------------------
Unchanged from v0.3, and worth restating because it is the part that could
quietly become a leak. The state handed to the solver is reconstructed from the
agent's beliefs, which come only from public events plus the agent's own hand.
It is used only when those beliefs pin EVERY live card, in which case the
reconstruction happens to equal the true state -- but the agent got there by a
deduction any player at the table could perform. The reconstruction additionally
refuses unless it reproduces every public hand count and the agent's own hand.
"""

from __future__ import annotations

from typing import Optional

from fish.agents.tablebase import pinned_state
from fish.engine import GameState
from fish.observation import Observation


class Tablebase4Mixin:
    """Mix in before an agent base class for exact endgame play.

    Subclasses must expose ``self.bel``. ``use_tablebase`` ablates the feature;
    ``tablebase_max_half_suits`` bounds the work. Note that the natural knob is
    half-suits, not cards: resolved half-suits are stripped from every hand, so
    the number of live cards is always exactly six times the number of
    unresolved half-suits, which makes any threshold between 6 and 11 the same
    policy. (An audit of v0.3 found that every "max live cards" knob in the
    codebase was in fact a binary switch for this reason.)
    """

    use_tablebase: bool = True
    tablebase_max_half_suits: int = 2
    #: fall back to the v0.3 solver if the v2 one refuses or errors
    tablebase_fallback: bool = True

    def tablebase_action(self, obs: Observation):
        if not self.use_tablebase:
            return None
        # _claim_bel is the identity everywhere but the gated oracle; the
        # tablebase reads it because when it fires it usually ends the
        # half-suit by claiming. See fish4/agent4.py.
        state = pinned_state(obs, self._claim_bel())
        if state is None:
            return None
        live = sum(1 for w in state.set_winner if w is None)
        if live == 0 or live > self.tablebase_max_half_suits:
            return None
        action = self._solve_v2(state)
        if action is None and self.tablebase_fallback:
            action = self._solve_v1(state)
        if action is None:
            return None
        try:
            state.check_legal(obs.player, action)
        except Exception:
            return None
        # The claim ban has to bite here too. A tablebase declaration is
        # certainly correct -- pinned_state only returns a state when the
        # belief pins every live card -- so it can never misfire under either
        # misdeclaration rule, but it CAN resolve a
        # half-suit that a counterfactual is deliberately holding open, which
        # silently ends the replay. Empty on every shipped path.
        from fish.engine import Claim
        banned = getattr(getattr(self, "claim_cfg", None), "banned", ())
        if banned and isinstance(action, Claim) and action.half_suit in banned:
            return None
        return action

    # -- solvers ---------------------------------------------------------------

    def _solve_v2(self, state: GameState):
        try:
            from .exact2 import Exact2Solver
        except Exception:
            return None
        try:
            solver = Exact2Solver(state.rules)
            best = solver.progress_optimal_actions(state)
            if best:
                return best[0] if len(best) == 1 else self.rng.choice(best)
            _, action = solver.solve(state)
            return action
        except Exception:
            return None

    def _solve_v1(self, state: GameState):
        from fish.exact import ExactSolver, SubgameTooLarge, live_card_count
        if live_card_count(state) > 8:
            return None
        try:
            solver = ExactSolver(state.rules)
            _, action = solver.solve(state)
            return action
        except (SubgameTooLarge, Exception):
            return None
