"""Exact solver for small Fish subgames: GROUND TRUTH.

Why this module matters most for validation: every other measure of strength
is relative ("beats the previous version"). An exact solution says what the
RIGHT answer is. If the engine disagrees with the exact optimum on positions
we can solve, that is a genuine defect, not a preference.

STRUCTURAL FACT discovered here: the Fish state graph is **cyclic**. Two
opponents can trade a card back and forth and return to an identical
position, so naive backward induction never terminates. Fish is a *loopy*
game and must be solved by value iteration within layers (see ExactSolver).

Interpreting the result honestly
--------------------------------
This solves the PERFECT-INFORMATION game: both teams see all cards. That is
not the same as optimal play under hidden information. Two legitimate uses:

1. **Consistency check.** When every remaining card's location is already
   publicly determined (routine in Fish endgames, detected by
   ``information_is_resolved``), perfect-information optimal play IS optimal
   play, and the engine must match it.
2. **Determinized upper bound.** Averaged over deals sampled from a belief
   state, the perfect-information value bounds achievable value and gives a
   regret proxy.
"""

from __future__ import annotations

from typing import Optional

from .cards import (CARDS_PER_HALF_SUIT, NUM_PLAYERS, half_suit_mask,
                    num_half_suits, team_of)
from .engine import Ask, Claim, GameState, Pass, NULL_TEAM
from .rules import RuleConfig


class SubgameTooLarge(Exception):
    """Refuse to solve a position whose graph we cannot exhaust."""


def position_key(state: GameState) -> tuple:
    return (tuple(state.hands), state.turn, tuple(state.set_winner))


def count_unresolved(state: GameState) -> int:
    return sum(1 for w in state.set_winner if w is None)


def live_card_count(state: GameState) -> int:
    return sum(h.bit_count() for h in state.hands)


def information_is_resolved(state: GameState, history=None) -> bool:
    """True when every remaining card's location is publicly known, so hidden
    information no longer affects optimal play."""
    from .beliefs import BeliefState
    from .observation import Observation
    bel = BeliefState(state.rules, observer=None)
    try:
        bel.update(Observation.from_state(state, 0))
    except Exception:
        return False
    for c in range(bel.n):
        m = bel.current_holder_mask(c)
        if m and m & (m - 1):     # live and ambiguous
            return False
    return True


def _delta(before: GameState, after: GameState) -> float:
    """Set differential (team 0 perspective) banked between two states."""
    a0, b0, _ = before.scores()
    a1, b1, _ = after.scores()
    return float((a1 - a0) - (b1 - b0))


class ExactSolver:
    """Exact solver over the perfect-information game GRAPH.

    Solved in layers:

    - CLAIMS strictly reduce the number of unresolved half-suits, so they
      always move to a strictly simpler layer.
    - ASKS and PASSES keep the same resolved configuration and are the only
      source of cycles, so within one layer the value is computed by
      **value iteration** to a fixpoint rather than by recursion.
    - Every non-terminal in-layer state starts at value 0, encoding the
      honest semantics of an unbroken cycle: if play never progresses, no
      further half-suit is claimed and neither team scores again. A side
      that can only lose by claiming will correctly prefer to loop, which is
      exactly the stalemate behavior real Fish exhibits.

    Value is the REMAINING set differential from team 0's perspective.
    Team 0 maximizes, team 1 minimizes.
    """

    # Tractability, measured: a layer with k live cards has up to 6^k card
    # placements times 6 turns. One half-suit (6 cards) solves in seconds;
    # two half-suits (12 cards) is hopeless by enumeration. The default
    # budget fails fast and loudly rather than hanging.
    def __init__(self, rules: RuleConfig, max_nodes: int = 400_000,
                 max_iterations: int = 500):
        self.rules = rules
        self.max_nodes = max_nodes
        self.max_iterations = max_iterations
        self.nodes = 0
        self.cache: dict[tuple, float] = {}
        self.best: dict[tuple, object] = {}
        self._layer_done: set[tuple] = set()

    # -- action generation ---------------------------------------------------

    def actions(self, state: GameState) -> list:
        p = state.turn
        if state.hands[p] == 0:
            return list(state.legal_passes(p))
        acts: list = list(state.legal_asks(p))
        # With perfect information the only claims worth considering are TRUE
        # ones: a knowingly-wrong claim is never better than a correct one,
        # and claiming a set the opponents hold simply gifts it to them.
        team = team_of(p)
        for hs in state.claimable_half_suits(p):
            holders = []
            ok = True
            for i in range(CARDS_PER_HALF_SUIT):
                h = state.holder_of(hs * CARDS_PER_HALF_SUIT + i)
                if h is None or team_of(h) != team:
                    ok = False
                    break
                holders.append(h)
            if ok:
                acts.append(Claim(hs, tuple(holders)))
        return acts

    # -- search --------------------------------------------------------------

    def solve(self, state: GameState) -> tuple[float, object]:
        """Returns (optimal REMAINING set differential for team 0, action)."""
        if state.is_terminal:
            return 0.0, None
        key = position_key(state)
        if key not in self.cache:
            self._solve_layer(state)
        return self.cache[key], self.best.get(key)

    def _child(self, state: GameState, p: int, act) -> GameState:
        child = GameState(self.rules, list(state.hands), state.turn)
        child.set_winner = list(state.set_winner)
        child.apply(p, act)
        return child

    def _solve_layer(self, root: GameState) -> None:
        config = tuple(root.set_winner)
        if config in self._layer_done:
            return
        # Cheap upfront tractability check. A layer with k live cards has at
        # most 6^k placements times 6 turns; without this we would enumerate
        # all the way to max_nodes before giving up, wasting tens of seconds
        # per hopeless position.
        live = live_card_count(root)
        if live > 0:
            est = 6 ** min(live, 20) * NUM_PLAYERS
            if est > self.max_nodes * 8:
                raise SubgameTooLarge(
                    f"layer with {live} live cards is ~6^{live} placements, "
                    f"far beyond the {self.max_nodes}-state budget")
        # --- 1. enumerate the layer via ask/pass edges
        states: dict[tuple, GameState] = {}
        stack = [root]
        while stack:
            st = stack.pop()
            k = position_key(st)
            if k in states or st.is_terminal:
                continue
            states[k] = st
            self.nodes += 1
            if self.nodes > self.max_nodes:
                raise SubgameTooLarge(
                    f"layer too large (> {self.max_nodes} states)")
            p = st.turn
            for act in self.actions(st):
                if isinstance(act, Claim):
                    continue           # leaves the layer; handled below
                stack.append(self._child(st, p, act))
        # --- 2. claim edges recurse into strictly simpler layers
        exits: dict[tuple, list[tuple[object, float]]] = {}
        for k, st in states.items():
            p = st.turn
            out = []
            for act in self.actions(st):
                if not isinstance(act, Claim):
                    continue
                child = self._child(st, p, act)
                gained = _delta(st, child)
                if child.is_terminal:
                    out.append((act, gained))
                else:
                    sub, _ = self.solve(child)
                    out.append((act, gained + sub))
            exits[k] = out
        # --- 3. value iteration inside the layer
        V = {k: 0.0 for k in states}   # unbroken cycle => no further scoring
        best: dict[tuple, object] = {}
        for _ in range(self.max_iterations):
            delta = 0.0
            for k, st in states.items():
                p = st.turn
                maximizing = team_of(p) == 0
                cur = float("-inf") if maximizing else float("inf")
                arg = None
                for act, val in exits[k]:
                    if (val > cur) if maximizing else (val < cur):
                        cur, arg = val, act
                for act in self.actions(st):
                    if isinstance(act, Claim):
                        continue
                    ck = position_key(self._child(st, p, act))
                    val = V.get(ck, 0.0)
                    if (val > cur) if maximizing else (val < cur):
                        cur, arg = val, act
                if arg is None:
                    cur = 0.0
                if abs(cur - V[k]) > 1e-12:
                    delta = max(delta, abs(cur - V[k]))
                    V[k] = cur
                best[k] = arg
            if delta < 1e-12:
                break
        else:
            raise SubgameTooLarge("value iteration did not converge")
        self.cache.update(V)
        self.best.update(best)
        self._layer_done.add(config)

    def action_value(self, state: GameState, act) -> float:
        child = self._child(state, state.turn, act)
        if isinstance(act, Claim):
            gained = _delta(state, child)
            if child.is_terminal:
                return gained
            sub, _ = self.solve(child)
            return gained + sub
        if child.is_terminal:
            return _delta(state, child)
        v, _ = self.solve(child)
        return v

    def optimal_actions(self, state: GameState) -> list:
        """ALL actions achieving the optimal value (ties matter when
        comparing an engine's choice fairly)."""
        val, _ = self.solve(state)
        return [act for act in self.actions(state)
                if abs(self.action_value(state, act) - val) < 1e-9]


def solve_position(state: GameState, max_nodes: int = 400_000):
    """Solve exactly, or raise SubgameTooLarge with a useful message."""
    live = live_card_count(state)
    if live > 9:
        raise SubgameTooLarge(
            f"{live} live cards is beyond exact enumeration "
            f"(state space grows as 6^cards); solve a smaller endgame")
    solver = ExactSolver(state.rules, max_nodes=max_nodes)
    value, action = solver.solve(state)
    return {"value": value, "action": action, "nodes": solver.nodes,
            "optimal_actions": solver.optimal_actions(state), "solver": solver}


# ---------------------------------------------------------------------------
# Building tractable subgames
# ---------------------------------------------------------------------------

def make_endgame(rules: RuleConfig, live_half_suits: list[int],
                 hands: list[int], turn: int,
                 resolved: Optional[dict[int, int]] = None) -> GameState:
    """Construct an endgame with only ``live_half_suits`` unresolved."""
    n_hs = num_half_suits(rules.variant)
    live = set(live_half_suits)
    live_mask = 0
    for hs in live:
        live_mask |= half_suit_mask(hs)
    union = 0
    for h in hands:
        if h & union:
            raise ValueError("card assigned to two players")
        union |= h
    if union != live_mask:
        raise ValueError("hands must cover exactly the live half-suits")
    state = GameState(rules, list(hands), turn)
    resolved = resolved or {}
    for hs in range(n_hs):
        if hs not in live:
            state.set_winner[hs] = resolved.get(hs, NULL_TEAM)
    state._normalize_turn()
    return state
