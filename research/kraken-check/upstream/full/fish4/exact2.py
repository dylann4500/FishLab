"""Exact solver v2: ground truth for TWO unresolved half-suits.

Why a rewrite rather than a tune-up
-----------------------------------
``fish.exact.ExactSolver`` is the project's only source of ABSOLUTE truth, and
it stops one half-suit short of where the interesting decisions live. Its
stated limit ("about seven live cards") is really a hard structural wall:
resolved half-suits are stripped from every hand, so

    live cards == 6 * (number of unresolved half-suits)   -- exactly, always

There is no such thing as a 7-, 8- or 9-live-card Fish position (measured, see
``tests4/test_exact2.py::test_live_cards_is_six_times_unresolved``). The old
solver therefore solves exactly the one-half-suit layer (6 cards, 6**6 * 6 =
279,936 card placements) and refuses everything else, because two half-suits
are 6**12 * 6 = 1.3e10 placements.

That wall is an artifact of the state ENCODING, not of the game.

The abstraction that breaks the wall
------------------------------------
Under perfect information the value of a position depends only on **how many
cards of each live half-suit each player holds**, never on which ones.

  Proof. Let sigma be any permutation of the deck that maps each half-suit
  onto itself (it may permute cards inside a half-suit arbitrarily). sigma
  acts on states by relabelling hands. It is a game automorphism:
    * ASK(t, c) is legal in s iff ASK(t, sigma(c)) is legal in sigma(s): the
      target is unchanged, hand counts are unchanged, "asker holds a card of
      this half-suit" is preserved because sigma preserves half-suits, and
      "asker does not hold c" is preserved because sigma is a bijection of
      each hand. The ask succeeds iff the target holds c iff sigma(target)
      holds sigma(c), and the resulting transfer commutes with sigma.
    * CLAIM(h, a) maps to CLAIM(h, a o pi^-1) where pi is sigma restricted to
      the positions of h. Exactness, "some card sits with an opponent", and
      the wrong-distribution case are all preserved, and both states lose
      exactly the six cards of h.
    * PASS and turn normalisation depend only on which hands are empty.
  Hence V(sigma(s)) = V(s). Two states have equal per-(half-suit, player)
  counts iff they are related by such a sigma, so V factors through the count
  vector. QED.

This is an exact symmetry of the perfect-information game, not a heuristic:
no rule can tell 2C from 3C. It is NOT a symmetry of the imperfect-information
game, where the public log does distinguish cards - see the "rejected ideas"
section of EXACT2.md.

A half-suit's counts are a composition of 6 into 6 ordered parts: 462 of them.
So a layer with m unresolved half-suits has at most 462**m * 6 abstract states:

    m = 1:         2,772   (vs   279,936 card states)      101x smaller
    m = 2:     1,280,664   (vs 1.3e10   card states)    10,000x smaller
    m = 3:   591,667,368                                 too big in full; use
                                                         the reachable solver

m = 1 and m = 2 are therefore solved ONCE, in full, as dense tables - a real
Fish tablebase - after which every query is an array lookup.

Cyclic semantics, pinned down
-----------------------------
Asks and passes never resolve a half-suit, so they cycle inside a layer;
claims strictly descend a layer. Inside a layer the game is: reach an exit
(claim) worth v, or play forever for 0. That is a zero-sum stopping game on a
finite graph. Its Bellman operator has MANY fixpoints (EXACT2.md section 4),
so "the fixpoint" is not well defined and the initialisation is part of the
SPECIFICATION, not an implementation detail.

The correct value is the limit of SYNCHRONOUS value iteration from the
all-zero vector, and that limit is reached exactly rather than asymptotically:

  V_k = value of the k-step truncated game (payoff 0 if not stopped by k).
  If V* > 0 the maximiser forces an exit worth >= V* within R <= |S| moves, so
  V_k >= V* once k >= |S|; if V* <= 0 he can also simply refuse to exit, and
  truncation only helps him, so again V_k >= V*. Symmetrically V_k <= V*.
  Hence V_k = V* for every k >= |S|, so any k at which V stops changing
  already carries V*.

Every layer is also solved a second, independent way - attractor-style
backward induction over the 2m+1 reachable integer thresholds, which involves
no fixpoint iteration at all - and the two are asserted equal
(``solve_layer_attractor``).

Measurements quoted in comments were taken on the machine and numpy version
recorded at the top of fish4/EXACT2.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from fish.cards import CARDS_PER_HALF_SUIT, NUM_PLAYERS, team_of
from fish.engine import Claim, GameState
from fish.rules import RuleConfig


class Exact2Error(Exception):
    """Base class for refusals from the v2 solver."""


class LayerTooLarge(Exact2Error):
    """The layer does not fit the configured budget."""


class UnsupportedRules(Exact2Error):
    """Rule options the count abstraction does not model."""


# ---------------------------------------------------------------------------
# Compositions: the abstract description of one half-suit
# ---------------------------------------------------------------------------

def _compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


#: every way six cards of one half-suit can sit in six hands: C(11,5) = 462
COMPOSITIONS: tuple[tuple[int, ...], ...] = tuple(
    _compositions(CARDS_PER_HALF_SUIT, NUM_PLAYERS))
NC = len(COMPOSITIONS)
COMP_INDEX: dict[tuple[int, ...], int] = {c: i for i, c in enumerate(COMPOSITIONS)}
COMP = np.array(COMPOSITIONS, dtype=np.int8)              # (462, 6)

#: MOVE[c, src, dst] = composition after one card walks src -> dst (-1 if none)
MOVE = np.full((NC, NUM_PLAYERS, NUM_PLAYERS), -1, dtype=np.int32)
for _i, _c in enumerate(COMPOSITIONS):
    for _src in range(NUM_PLAYERS):
        if _c[_src] == 0:
            continue
        for _dst in range(NUM_PLAYERS):
            if _dst == _src:
                continue
            _t = list(_c)
            _t[_src] -= 1
            _t[_dst] += 1
            MOVE[_i, _src, _dst] = COMP_INDEX[tuple(_t)]
MOVE_PY = MOVE.tolist()

#: TEAM_HOLDS_ALL[c, t] - team t holds all six, i.e. a TRUE claim exists
TEAM_HOLDS_ALL = np.zeros((NC, 2), dtype=bool)
for _i, _c in enumerate(COMPOSITIONS):
    for _t in (0, 1):
        TEAM_HOLDS_ALL[_i, _t] = (_c[_t] + _c[_t + 2] + _c[_t + 4]
                                  == CARDS_PER_HALF_SUIT)
TEAM_HOLDS_ALL_PY = [[bool(x) for x in row] for row in TEAM_HOLDS_ALL.tolist()]

_LOW = -30000          # sentinel for "no such action"; int16-safe


# ---------------------------------------------------------------------------
# Count-space transitions (reference implementation, also used for m >= 3)
# ---------------------------------------------------------------------------

def hand_sizes(comps: Sequence[int]) -> list[int]:
    sizes = [0] * NUM_PLAYERS
    for c in comps:
        cc = COMPOSITIONS[c]
        for p in range(NUM_PLAYERS):
            sizes[p] += cc[p]
    return sizes


def normalize_turn(turn: int, sizes: Sequence[int]) -> Optional[int]:
    """Mirror of ``GameState._normalize_turn`` in count space."""
    if sizes[turn]:
        return turn
    if sizes[(turn + 2) % 6] or sizes[(turn + 4) % 6]:
        return turn                 # cardless, but a teammate can be passed to
    for step in range(1, NUM_PLAYERS):
        q = (turn + step) % NUM_PLAYERS
        if sizes[q]:
            return q
    return None                     # nobody holds anything: terminal


def layer_transitions(comps: tuple[int, ...], turn: int, bluff: bool = False,
                      giveaway: bool = False):
    """(in-layer moves, layer exits) for one abstract state.

    ``moves`` -> list of ((comps, turn), label)
    ``exits`` -> list of (delta, child_or_None, label); ``delta`` is the set
    differential banked immediately from team 0's point of view, and ``child``
    is the (m-1)-layer state, or None when the claim ends the game.
    """
    sizes = hand_sizes(comps)
    moves: list = []
    exits: list = []
    if sizes[turn] == 0:
        for t in ((turn + 2) % 6, (turn + 4) % 6):
            if sizes[t]:
                moves.append(((comps, t), ("pass", t)))
        return moves, exits
    team = team_of(turn)
    opps = [o for o in ((turn + 1) % 6, (turn + 3) % 6, (turn + 5) % 6)
            if sizes[o]]
    for j, cidx in enumerate(comps):
        cj = COMPOSITIONS[cidx]
        if cj[turn] == 0:
            continue                # cannot ask inside a half-suit you are out of
        for o in opps:
            if cj[o] > 0:
                nxt = list(comps)
                nxt[j] = MOVE_PY[cidx][o][turn]
                moves.append(((tuple(nxt), turn), ("hit", j, o)))
            spare = (CARDS_PER_HALF_SUIT - cj[o] if bluff
                     else CARDS_PER_HALF_SUIT - cj[turn] - cj[o])
            if spare > 0:
                moves.append(((comps, o), ("miss", j, o)))
    for j, cidx in enumerate(comps):
        if TEAM_HOLDS_ALL_PY[cidx][team]:
            delta = 1 if team == 0 else -1
        elif giveaway:
            delta = -1 if team == 0 else 1
        else:
            continue
        child_comps = tuple(c for k, c in enumerate(comps) if k != j)
        if not child_comps:
            exits.append((delta, None, ("claim", j)))
        else:
            ct = normalize_turn(turn, hand_sizes(child_comps))
            assert ct is not None, "a non-empty layer always has a card holder"
            exits.append((delta, (child_comps, ct), ("claim", j)))
    return moves, exits


# ---------------------------------------------------------------------------
# Dense layer tables (the tablebase)
# ---------------------------------------------------------------------------

@dataclass
class LayerTable:
    """Exact values for every abstract state of an m-half-suit layer."""

    m: int
    values: np.ndarray                  # int16, length 462**m * 6
    valid: np.ndarray                   # bool, states obeying turn normalisation
    iterations: int = 0
    build_seconds: float = 0.0
    n_states: int = 0
    ranks: Optional[np.ndarray] = None  # attractor rank of the pinning threshold
    is_max: Optional[np.ndarray] = None

    def index(self, comps: Sequence[int], turn: int) -> int:
        idx = 0
        for c in comps:
            idx = idx * NC + int(c)
        return idx * NUM_PLAYERS + int(turn)

    def value(self, comps: Sequence[int], turn: int) -> int:
        return int(self.values[self.index(comps, turn)])


def _layer_slots(m: int, bluff: bool, giveaway: bool,
                 sub: Optional[LayerTable]):
    """Vectorised transition arrays for the whole m-layer.

    Returns (moves, exits, is_max, valid) where ``moves`` is a list of
    (mask, child_index) and ``exits`` a list of (mask, exit_value). Building
    these once and reusing them every sweep is the single biggest constant
    factor: the v1 solver rebuilt a ``GameState`` per edge per sweep.

    Written to keep peak memory down rather than to read prettily. An earlier
    version materialised a (n, 6) "relative seating" index array and two more
    (n, 6) count arrays; on a machine with 1.4 GB free that pushed the m=2
    build from 5 s to 31 s of paging. Everything here is (n,)-shaped.
    """
    n = NC ** m * NUM_PLAYERS
    idx = np.arange(n, dtype=np.int32)
    ar = np.arange(n, dtype=np.intp)               # reused gather index
    turn = (idx % NUM_PLAYERS).astype(np.int32)
    rest = idx // NUM_PLAYERS
    comps: list[np.ndarray] = [None] * m                       # type: ignore
    place = [0] * m                     # positional weight of each half-suit
    for j in range(m - 1, -1, -1):
        comps[j] = (rest % NC).astype(np.int32)
        rest = rest // NC
        place[j] = (NC ** (m - 1 - j)) * NUM_PLAYERS

    def seat(k):
        """Absolute id of the player k seats clockwise from the mover."""
        return ((turn + k) % NUM_PLAYERS).astype(np.intp)

    def cards_at(j, who):
        return COMP[comps[j], who]

    def sizes_at(who):
        total = cards_at(0, who).astype(np.int8)
        for j in range(1, m):
            total = total + cards_at(j, who)
        return total

    self_has = sizes_at(turn.astype(np.intp)) > 0
    mate_has = (sizes_at(seat(2)) > 0) | (sizes_at(seat(4)) > 0)
    valid = self_has | mate_has         # states that survive turn normalisation
    is_max = (turn % 2 == 0)

    moves: list[tuple[np.ndarray, np.ndarray]] = []
    exits: list[tuple[np.ndarray, np.ndarray]] = []

    # PASS: only from a cardless mover with a card-holding teammate
    for k in (2, 4):
        who = seat(k)
        mask = (~self_has) & (sizes_at(who) > 0)
        moves.append((mask, (idx - turn + who).astype(np.int32)))

    # ASKS
    for k in (1, 3, 5):
        opp = seat(k)
        base_opp = self_has & (sizes_at(opp) > 0)
        for j in range(m):
            mine = cards_at(j, turn.astype(np.intp))
            theirs = cards_at(j, opp)
            base = base_opp & (mine > 0)
            # hit: the target really holds one, so the card walks to the mover
            hit_mask = base & (theirs > 0)
            new_c = MOVE[comps[j], opp, turn.astype(np.intp)]
            hit_child = idx + (np.where(hit_mask, new_c, comps[j])
                               - comps[j]) * place[j]
            moves.append((hit_mask, hit_child.astype(np.int32)))
            # miss: some card of the half-suit is held by neither side
            spare = (CARDS_PER_HALF_SUIT - theirs.astype(np.int16) if bluff
                     else CARDS_PER_HALF_SUIT - mine.astype(np.int16)
                     - theirs.astype(np.int16))
            moves.append((base & (spare > 0),
                          (idx - turn + opp).astype(np.int32)))

    # CLAIMS: leave the layer
    team = (turn % 2).astype(np.intp)
    sign = np.where(team == 0, 1, -1).astype(np.int16)
    for j in range(m):
        holds_all = TEAM_HOLDS_ALL[comps[j], team]
        if giveaway:
            # a claim is legal on any unresolved half-suit; if the team does
            # not own all six the opponents score it, whatever is declared
            mask = self_has
            delta = np.where(holds_all, sign, -sign).astype(np.int16)
        else:
            mask = self_has & holds_all
            delta = sign
        if m == 1:
            exits.append((mask, delta))
            continue
        assert sub is not None and sub.m == m - 1
        keep = [comps[k] for k in range(m) if k != j]

        def child_size(who, keep=keep):
            total = COMP[keep[0], who].astype(np.int8)
            for c in keep[1:]:
                total = total + COMP[c, who]
            return total

        # turn normalisation after the claim strips six cards off the table
        stay = np.zeros(n, dtype=bool)
        for off in (0, 2, 4):
            stay |= child_size(seat(off)) > 0
        nxt = np.zeros(n, dtype=np.int32)
        found = np.zeros(n, dtype=bool)
        for step in range(1, NUM_PLAYERS):
            who = seat(step)
            take = (~found) & (child_size(who) > 0)
            nxt = np.where(take, who.astype(np.int32), nxt)
            found |= take
        child_turn = np.where(stay, turn, nxt)
        cidx = np.zeros(n, dtype=np.int64)
        for c in keep:
            cidx = cidx * NC + c
        cidx = cidx * NUM_PLAYERS + child_turn
        exits.append((mask, (delta + sub.values[cidx]).astype(np.int16)))
        del cidx, child_turn, nxt, found, stay
    return moves, exits, is_max, valid


def exit_thresholds(exits) -> list[int]:
    """The only values a state can take: an exit value, or 0 for looping.

    Restricting the attractor sweep to these instead of every integer in
    [-m, m] cut the m=2 cross-check from five threshold passes to three,
    because a two-half-suit claim can only ever be worth -2, 0 or +2.
    """
    seen = {0}
    for mask, val in exits:
        seen.update(int(v) for v in np.unique(val[mask]))
    return sorted(seen)


def _value_iterate(moves, exits, is_max, valid, max_iterations: int = 20000):
    """Synchronous value iteration from the all-zero vector.

    Zero is not an arbitrary seed: it IS the semantics of an unbroken cycle,
    and V_k is exactly the value of the k-step truncated game, so the first k
    at which V stops moving already carries the true value (module docstring).
    """
    n = is_max.shape[0]
    sign = np.where(is_max, 1, -1).astype(np.int16)
    static = [(mask, (sign * val).astype(np.int16)) for mask, val in exits]
    values = np.zeros(n, dtype=np.int16)
    scratch = np.empty(n, dtype=np.int16)
    for it in range(1, max_iterations + 1):
        best = np.full(n, _LOW, dtype=np.int16)
        for mask, child in moves:
            np.multiply(sign, values[child], out=scratch)
            np.maximum(best, np.where(mask, scratch, _LOW), out=best)
        for mask, sval in static:
            np.maximum(best, np.where(mask, sval, _LOW), out=best)
        if it == 1 and np.any(valid & (best == _LOW)):
            raise Exact2Error("a legal position with no legal action")
        nxt = (sign * best).astype(np.int16)
        nxt[~valid] = 0
        if np.array_equal(nxt, values):
            return values, it
        values = nxt
    raise Exact2Error(f"value iteration did not settle in {max_iterations} sweeps")


def solve_layer_attractor(moves, exits, is_max, valid, thresholds):
    """Independent solution by attractors: no fixpoint iteration at all.

    For a threshold c > 0 the maximiser must genuinely REACH an exit worth
    >= c (playing forever pays 0 < c), so the winning region is the maximiser
    attractor of those exits. For c <= 0 playing forever already pays >= c, so
    the maximiser loses only if the minimiser can FORCE an exit worth < c;
    the winning region is the complement of the minimiser attractor of those
    exits. V(s) = max{c : s is in the region for c}. This is exact, has no
    initialisation, and no sweep order, which is why it is the cross-check on
    the value-iteration semantics.
    """
    n = is_max.shape[0]
    values = np.full(n, min(thresholds), dtype=np.int16)
    ranks: dict[int, np.ndarray] = {}
    for c in sorted(thresholds):
        if c > 0:
            # the maximiser must genuinely reach an exit worth >= c
            controls = is_max
            seed = [(mask & (val >= c)) for mask, val in exits]
        else:
            # the minimiser must force an exit worth < c; anything else pays
            # >= c already, looping included
            controls = ~is_max
            seed = [(mask & (val < c)) for mask, val in exits]
        # start from the empty set: a node only joins through the any/all rule,
        # so a MIN node holding one qualifying exit but also a safe move stays
        # out, which a naive "seed with the qualifying exits" start gets wrong.
        inset = np.zeros(n, dtype=bool)
        rank = np.full(n, np.iinfo(np.int32).max, dtype=np.int32)
        # A node controlled by the attractor player joins if ANY action leads
        # in; a node controlled by the opponent joins only if EVERY action
        # does, its own exits included.
        round_no = 0
        while True:
            any_in = np.zeros(n, dtype=bool)
            all_in = np.ones(n, dtype=bool)
            for mask, child in moves:
                lead = inset[child]
                any_in |= mask & lead
                all_in &= (~mask) | lead
            for (mask, _val), s in zip(exits, seed):
                any_in |= s
                all_in &= (~mask) | s
            grown = (inset | np.where(controls, any_in, all_in)) & valid
            if np.array_equal(grown, inset):
                break
            round_no += 1
            rank[grown & ~inset] = round_no
            inset = grown
        ranks[c] = rank
        if c > 0:
            values = np.where(inset, np.maximum(values, c), values)
        else:
            values = np.where(inset, values, np.maximum(values, c))
    values[~valid] = 0
    # The rank that matters for a state is the one of the attractor that
    # actually pins its value: the maximiser reaching an exit >= v when v > 0,
    # the minimiser forcing an exit <= v when v < 0, and none at all when
    # v == 0, where looping forever is already worth the value.
    #
    # For v < 0 the pinning attractor is "the minimiser forces an exit < c"
    # for the SMALLEST THRESHOLD ABOVE v, not for v + 1: with the thresholds
    # restricted to values that actually occur, v + 1 need not be one of them,
    # and looking it up silently left every rank at 0, which emptied
    # ``progress_optimal_actions`` on every losing position.
    order = sorted(thresholds)
    out = np.zeros(n, dtype=np.int32)
    for i, c in enumerate(order):
        if c > 0:
            out = np.where(values == c, ranks[c], out)
        elif c < 0 and i + 1 < len(order):
            out = np.where(values == c, ranks[order[i + 1]], out)
    return values, out


# ---------------------------------------------------------------------------
# Table cache
# ---------------------------------------------------------------------------

_TABLES: dict[tuple, LayerTable] = {}


def layer_table(m: int, bluff: bool = False, giveaway: bool = False,
                ranks: bool = True, max_states: int = 8_000_000) -> LayerTable:
    """Full dense table for an m-half-suit layer, built once and cached.

    Every table is solved TWICE, by value iteration from zero and by threshold
    attractors, and the two are asserted equal. That is not belt-and-braces
    paranoia: the two computations disagree on which fixpoint they select
    whenever the semantics of an unbroken cycle are wrong, so the assert is
    the empirical test of the cyclic-game semantics (EXACT2.md section 4).
    """
    if m < 1:
        raise ValueError("m must be at least 1")
    key = (m, bluff, giveaway)
    hit = _TABLES.get(key)
    if hit is not None and (hit.ranks is not None or not ranks):
        return hit
    n = NC ** m * NUM_PLAYERS
    if n > max_states:
        raise LayerTooLarge(
            f"a full {m}-half-suit table is {n:,} abstract states, over the "
            f"{max_states:,} budget; use the reachable-set path instead")
    sub = layer_table(m - 1, bluff, giveaway, ranks=False,
                      max_states=max_states) if m > 1 else None
    t0 = time.perf_counter()
    moves, exits, is_max, valid = _layer_slots(m, bluff, giveaway, sub)
    values, iterations = _value_iterate(moves, exits, is_max, valid)
    attractor_values, rank = solve_layer_attractor(
        moves, exits, is_max, valid, exit_thresholds(exits))
    if not np.array_equal(attractor_values, values):
        bad = int(np.argmax(attractor_values != values))
        raise Exact2Error(
            f"value iteration and attractors disagree at abstract state {bad} "
            f"of layer m={m}: VI={values[bad]}, attractor={attractor_values[bad]}")
    table = LayerTable(m=m, values=values, valid=valid, iterations=iterations,
                       build_seconds=time.perf_counter() - t0,
                       n_states=int(valid.sum()), ranks=rank, is_max=is_max)
    _TABLES[key] = table
    return table


def clear_tables() -> None:
    _TABLES.clear()


# ---------------------------------------------------------------------------
# Reachable-set solver: for layers too big to tabulate in full (m >= 3)
# ---------------------------------------------------------------------------

@dataclass
class ReachableSolution:
    values: dict[tuple, int]
    n_states: int
    iterations: int
    seconds: float


def solve_reachable(root: tuple, sub_value, bluff: bool = False,
                    giveaway: bool = False,
                    max_states: int = 3_000_000) -> ReachableSolution:
    """Solve one layer over only the states reachable from ``root``.

    Reachability is a real reduction here, not wishful thinking: an ask only
    ever moves a card TO a player who already holds one of that half-suit, so
    the set of players holding cards never grows. A layer entered with four
    live hands can never reach a six-hand placement.

    ``root`` is (comps, turn); ``sub_value`` maps an (m-1)-layer abstract state
    to its exact value.
    """
    t0 = time.perf_counter()
    order: list[tuple] = []
    index: dict[tuple, int] = {}
    stack = [root]
    while stack:
        st = stack.pop()
        if st in index:
            continue
        index[st] = len(order)
        order.append(st)
        if len(order) > max_states:
            raise LayerTooLarge(
                f"reachable layer exceeded {max_states:,} abstract states")
        moves, _ = layer_transitions(st[0], st[1], bluff, giveaway)
        for nxt, _label in moves:
            if nxt not in index:
                stack.append(nxt)
    n = len(order)
    move_edges: list[list[int]] = []
    exit_vals: list[list[int]] = []
    maximizing = [False] * n
    for i, st in enumerate(order):
        moves, exits = layer_transitions(st[0], st[1], bluff, giveaway)
        move_edges.append([index[nxt] for nxt, _ in moves])
        exit_vals.append([delta + (0 if child is None else sub_value(child))
                          for delta, child, _ in exits])
        maximizing[i] = team_of(st[1]) == 0
        if not move_edges[i] and not exit_vals[i]:
            raise Exact2Error(f"legal position with no legal action: {st}")
    values = [0] * n
    iterations = 0
    for iterations in range(1, 100_000):
        nxt_values = [0] * n
        changed = False
        for i in range(n):
            cand = [values[j] for j in move_edges[i]] + exit_vals[i]
            v = max(cand) if maximizing[i] else min(cand)
            nxt_values[i] = v
            if v != values[i]:
                changed = True
        values = nxt_values
        if not changed:
            break
    else:
        raise Exact2Error("reachable value iteration did not settle")
    return ReachableSolution({st: values[i] for st, i in index.items()},
                             n, iterations, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Query API on real GameStates
# ---------------------------------------------------------------------------

def live_half_suits(state: GameState) -> tuple[int, ...]:
    return tuple(hs for hs, w in enumerate(state.set_winner) if w is None)


def abstract_state(state: GameState) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """(live half-suits, composition indices, turn) - the count abstraction."""
    live = live_half_suits(state)
    comps = []
    for hs in live:
        counts = tuple(
            ((state.hands[p] >> (hs * CARDS_PER_HALF_SUIT)) & 0b111111).bit_count()
            for p in range(NUM_PLAYERS))
        if sum(counts) != CARDS_PER_HALF_SUIT:
            raise Exact2Error(
                f"half-suit {hs} is unresolved but only {sum(counts)} of its "
                f"cards sit in hands; the state breaks SPEC invariant 1")
        comps.append(COMP_INDEX[counts])
    return live, tuple(comps), state.turn


def _delta(before: GameState, after: GameState) -> float:
    a0, b0, _ = before.scores()
    a1, b1, _ = after.scores()
    return float((a1 - a0) - (b1 - b0))


class Exact2Solver:
    """Exact perfect-information values, drop-in for ``fish.exact.ExactSolver``.

    Same contract: ``solve`` returns the remaining set differential from team
    0's point of view together with an optimal action, ``action_value`` scores
    one action, and ``optimal_actions`` returns every action achieving the
    value. Unlike v1 it never enumerates card placements, so two live
    half-suits are routine rather than refused.
    """

    def __init__(self, rules: RuleConfig, max_full_layer: int = 2,
                 include_giveaway_claims: bool = False,
                 max_reachable_states: int = 3_000_000):
        if rules.claims_any_time:
            raise UnsupportedRules(
                "claims_any_time makes every player a mover at every node; "
                "neither v1 nor v2 models that")
        self.rules = rules
        self.bluff = rules.allow_bluff_asks
        self.giveaway = include_giveaway_claims
        self.max_full_layer = max_full_layer
        self.max_reachable_states = max_reachable_states
        self._reach: dict[tuple, dict[tuple, int]] = {}
        self.nodes = 0

    # -- tables ------------------------------------------------------------

    def table(self, m: int) -> LayerTable:
        return layer_table(m, self.bluff, self.giveaway)

    def _sub_value(self, child: tuple) -> int:
        comps, turn = child
        return self.table(len(comps)).value(comps, turn)

    # -- values ------------------------------------------------------------

    def value(self, state: GameState) -> float:
        if state.is_terminal:
            return 0.0
        live, comps, turn = abstract_state(state)
        m = len(comps)
        if m <= self.max_full_layer:
            return float(self.table(m).value(comps, turn))
        cache = self._reach.setdefault((m, live), {})
        key = (comps, turn)
        if key not in cache:
            sol = solve_reachable(key, self._sub_value, self.bluff,
                                  self.giveaway, self.max_reachable_states)
            cache.update(sol.values)
            self.nodes += sol.n_states
        return float(cache[key])

    def solve(self, state: GameState) -> tuple[float, object]:
        if state.is_terminal:
            return 0.0, None
        value = self.value(state)
        for act in self.actions(state):
            if abs(self.action_value(state, act) - value) < 1e-9:
                return value, act
        raise Exact2Error("no action achieves the computed value")

    # -- actions -----------------------------------------------------------

    def actions(self, state: GameState) -> list:
        """The same action set as ``fish.exact.ExactSolver.actions``.

        Under perfect information the only claims worth generating are TRUE
        ones: mis-splitting a half-suit your team wholly owns leads to the very
        same continuation and banks 0 (legacy null variant) or -1
        (opponent-award baseline) instead of +1, so it is strictly dominated
        under either rule -- the action set, and with it every
        perfect-information value here, is invariant to the misdeclaration
        rule. Handing a half-suit to the opponents is NOT dominated a
        priori - it strips cards off the table, which can matter once a second
        half-suit is live - so it is available behind
        ``include_giveaway_claims``. EXACT2.md section 5 reports the measured
        effect.
        """
        p = state.turn
        if state.hands[p] == 0:
            return list(state.legal_passes(p))
        acts: list = list(state.legal_asks(p))
        team = team_of(p)
        for hs in state.claimable_half_suits(p):
            holders = [state.holder_of(hs * CARDS_PER_HALF_SUIT + i)
                       for i in range(CARDS_PER_HALF_SUIT)]
            if all(h is not None and team_of(h) == team for h in holders):
                acts.append(Claim(hs, tuple(holders)))
            elif self.giveaway:
                # every declaration resolves the same way once an opponent
                # holds one of the six, so one representative is enough
                acts.append(Claim(hs, (p,) * CARDS_PER_HALF_SUIT))
        return acts

    def child(self, state: GameState, act) -> GameState:
        child = GameState(self.rules, list(state.hands), state.turn)
        child.set_winner = list(state.set_winner)
        child.apply(state.turn, act)
        return child

    def action_value(self, state: GameState, act) -> float:
        child = self.child(state, act)
        gained = _delta(state, child)
        if child.is_terminal:
            return gained
        return gained + self.value(child)

    def optimal_actions(self, state: GameState) -> list:
        value = self.value(state)
        return [act for act in self.actions(state)
                if abs(self.action_value(state, act) - value) < 1e-9]

    def progress_optimal_actions(self, state: GameState) -> list:
        """Optimal actions that also make PROGRESS towards the value.

        Bellman-optimality is not optimality in a loopy game. If team 0 can
        force +1, every action that keeps the value at +1 looks optimal,
        including one that walks around a cycle forever and actually banks 0.
        The attractor rank fixes this: when the value is non-zero the side it
        favours must strictly descend the rank. When the value is 0 there is
        nothing to progress towards - looping IS the value - so every
        Bellman-optimal action qualifies.
        """
        value = self.value(state)
        best = self.optimal_actions(state)
        if value == 0.0:
            return best
        live, comps, turn = abstract_state(state)
        m = len(comps)
        if m > self.max_full_layer:
            raise LayerTooLarge("progress ranks need a full table for the layer")
        table = self.table(m)
        here = int(table.ranks[table.index(comps, turn)])
        out = []
        for act in best:
            if isinstance(act, Claim):
                out.append(act)              # an exit is progress by definition
                continue
            child = self.child(state, act)
            _clive, ccomps, cturn = abstract_state(child)
            if int(table.ranks[table.index(ccomps, cturn)]) < here:
                out.append(act)
        return out


def solve_position2(state: GameState, **kwargs) -> dict:
    """``fish.exact.solve_position``-shaped result, without the size refusal."""
    solver = Exact2Solver(state.rules, **kwargs)
    value, action = solver.solve(state)
    return {"value": value, "action": action, "nodes": solver.nodes,
            "optimal_actions": solver.optimal_actions(state), "solver": solver}
