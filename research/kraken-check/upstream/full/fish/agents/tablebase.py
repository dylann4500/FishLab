"""Exact endgame resolution at inference time (a Fish "tablebase").

When an agent's OWN belief state pins the location of every remaining card,
the position contains no hidden information at all. At that point sampling
worlds is pointless (every sample is the same world) and heuristics are
strictly worse than simply solving the position.

This is legitimate and leak-free by construction: the state handed to the
solver is built from the agent's beliefs, which are derived only from public
events plus the agent's own hand. If those beliefs pin every card, the
reconstruction happens to equal the true state, but the agent got there by
deduction that any player at the table could perform.

Two guards keep it honest and affordable:
  * ``pins_everything`` requires EVERY live card to be pinned. Partial
    knowledge falls through to the normal policy.
  * the live card count must fit the solver budget (6^k placements).

Chess engines get exact endgame play from precomputed tables; Fish endgames
are small enough to solve on the spot, so we do.
"""

from __future__ import annotations

from typing import Optional

from ..engine import GameState
from ..exact import SubgameTooLarge, live_card_count, ExactSolver
from ..observation import Observation


def pinned_state(obs: Observation, belief) -> Optional[GameState]:
    """Reconstruct the position iff beliefs determine it completely."""
    hands = [0] * 6
    for c in range(belief.n):
        m = belief.current_holder_mask(c)
        if m == 0:
            continue                       # resolved half-suit
        if m & (m - 1):
            return None                    # ambiguous: not fully determined
        hands[m.bit_length() - 1] |= 1 << c
    # sanity: the reconstruction must reproduce every public fact
    if tuple(h.bit_count() for h in hands) != tuple(obs.hand_counts):
        return None
    if hands[obs.player] != obs.hand:
        return None
    state = GameState.from_components(obs.rules, hands, obs.turn,
                                      list(obs.set_winner))
    return state


class ExactEndgameMixin:
    """Mix in before an agent base class to add exact endgame play.

    Subclasses must expose ``self.bel`` (a BeliefState) and implement
    ``act_policy`` with the normal policy.
    """

    #: solve only when at most this many cards remain live
    tablebase_max_cards: int = 8
    #: set False to ablate the feature
    use_tablebase: bool = True

    def tablebase_action(self, obs: Observation):
        if not self.use_tablebase:
            return None
        state = pinned_state(obs, self.bel)
        if state is None:
            return None
        if live_card_count(state) > self.tablebase_max_cards:
            return None
        try:
            solver = ExactSolver(obs.rules)
            _, action = solver.solve(state)
        except (SubgameTooLarge, Exception):
            return None
        if action is None:
            return None
        # Only return actions this seat may legally take right now.
        try:
            state.check_legal(obs.player, action)
        except Exception:
            return None
        return action
