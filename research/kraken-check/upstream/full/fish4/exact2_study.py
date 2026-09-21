"""Validation and measurement harness for ``fish4.exact2``.

Nothing in here is imported by the solver; it exists so that every number in
fish4/EXACT2.md has a command that reproduces it. Five studies:

1. ``cross_check``      - v2 must agree with fish.exact.ExactSolver on every
                          position v1 can solve, on the value AND on the full
                          set of optimal actions.
2. ``timing``           - the before/after numbers. v1 is the "before".
3. ``fixpoint_study``   - is the cyclic-game fixpoint unique? Initialise value
                          iteration from zeros, random, optimistic and
                          pessimistic vectors, sweep synchronously and in
                          place, and compare against the attractor solution
                          which has no initialisation at all.
4. ``closed_form_study``- the structural result the tables exposed, checked
                          exhaustively at m<=2 and by sampling at m=3.
5. ``giveaway_study``   - does allowing a deliberately losing claim (the one
                          action v1 refuses to generate) ever change a value?

Run:  py -m fish4.exact2_study all --workers 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fish.cards import NUM_PLAYERS, half_suit_cards, num_half_suits, team_of
from fish.engine import Ask, Claim, GameState, NULL_TEAM
from fish.rules import RuleConfig

from fish4 import exact2 as e

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")


# ---------------------------------------------------------------------------
# Position generation
# ---------------------------------------------------------------------------

def build_position(live_hs, comps, turn, resolved=None, rules=None):
    """Materialise one card-level state from an abstract (comps, turn).

    Cards inside a half-suit are handed out in ascending order; by the count
    symmetry every other materialisation has the same value, which
    ``abstraction_fuzz`` checks rather than assumes.
    """
    rules = rules or RuleConfig()
    hands = [0] * NUM_PLAYERS
    for hs, comp in zip(live_hs, comps):
        cards = list(half_suit_cards(hs))
        i = 0
        for p, count in enumerate(e.COMPOSITIONS[comp]):
            for _ in range(count):
                hands[p] |= 1 << cards[i]
                i += 1
    state = GameState(rules, hands, turn)
    resolved = resolved or {}
    for hs in range(num_half_suits(rules.variant)):
        if hs not in live_hs:
            state.set_winner[hs] = resolved.get(hs, NULL_TEAM)
    state._normalize_turn()
    return state


def _tagged_compositions(rng, n):
    """A spread of compositions that deliberately covers the awkward cases."""
    picks: list[tuple[int, str]] = []
    # a team with zero cards (and therefore forced-claim endings)
    one_team = [i for i, c in enumerate(e.COMPOSITIONS)
                if c[0] + c[2] + c[4] == 0 or c[1] + c[3] + c[5] == 0]
    # a single player holding everything
    solo = [i for i, c in enumerate(e.COMPOSITIONS) if max(c) == 6]
    # maximally spread: every player holds one
    spread = [i for i, c in enumerate(e.COMPOSITIONS) if max(c) == 1]
    # exactly two players hold cards
    duo = [i for i, c in enumerate(e.COMPOSITIONS) if sum(1 for x in c if x) == 2]
    for name, pool in (("one_team_holds_all", one_team), ("solo", solo),
                       ("spread", spread), ("two_holders", duo)):
        for idx in rng.sample(pool, min(len(pool), max(2, n // 8))):
            picks.append((idx, name))
    while len(picks) < n:
        picks.append((rng.randrange(e.NC), "random"))
    rng.shuffle(picks)
    return picks[:n]


def generate_m1_positions(n=240, seed=11):
    """Varied single-half-suit positions, the regime v1 can also solve."""
    rng = random.Random(seed)
    out = []
    for comp, tag in _tagged_compositions(rng, n):
        hs = rng.randrange(9)
        turn = rng.randrange(6)
        # mix of already-scored, already-nulled and untouched contexts so the
        # "remaining differential" contract is exercised, not assumed
        resolved = {}
        for other in range(9):
            if other == hs:
                continue
            resolved[other] = rng.choice([0, 1, NULL_TEAM])
        state = build_position((hs,), (comp,), turn, resolved)
        out.append({"hands": list(state.hands), "turn": state.turn,
                    "set_winner": list(state.set_winner), "tag": tag,
                    "comp": comp, "hs": hs})
    return out


def generate_m2_positions(n=200, seed=23):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a, tag_a = _tagged_compositions(rng, 1)[0]
        b, _ = _tagged_compositions(rng, 1)[0]
        hs = sorted(rng.sample(range(9), 2))
        turn = rng.randrange(6)
        resolved = {o: rng.choice([0, 1, NULL_TEAM]) for o in range(9)
                    if o not in hs}
        state = build_position(tuple(hs), (a, b), turn, resolved)
        out.append({"hands": list(state.hands), "turn": state.turn,
                    "set_winner": list(state.set_winner), "tag": tag_a})
    return out


def rebuild(pos, rules=None):
    rules = rules or RuleConfig()
    return GameState.from_components(rules, pos["hands"], pos["turn"],
                                     pos["set_winner"])


# ---------------------------------------------------------------------------
# 1. cross check against fish.exact
# ---------------------------------------------------------------------------

def _act_key(act):
    if isinstance(act, Ask):
        return ("ask", act.target, act.card)
    if isinstance(act, Claim):
        return ("claim", act.half_suit, tuple(act.assignment))
    return ("pass", act.teammate)


def _solve_one_v1(pos):
    from fish.exact import ExactSolver, SubgameTooLarge
    rules = RuleConfig()
    state = rebuild(pos, rules)
    t0 = time.perf_counter()
    try:
        solver = ExactSolver(rules)
        value, _ = solver.solve(state)
        acts = sorted(_act_key(a) for a in solver.optimal_actions(state))
    except Exception as exc:                       # SubgameTooLarge and friends
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "seconds": time.perf_counter() - t0, "nodes": 0}
    return {"ok": True, "value": value, "actions": acts, "nodes": solver.nodes,
            "seconds": time.perf_counter() - t0}


def cross_check(positions, workers=3):
    """v2 must reproduce v1 exactly wherever v1 has an answer."""
    rules = RuleConfig()
    solver = e.Exact2Solver(rules)
    solver.value(rebuild(positions[0], rules))      # warm the tables

    t0 = time.perf_counter()
    v2 = []
    for pos in positions:
        state = rebuild(pos, rules)
        t = time.perf_counter()
        value = solver.value(state)
        acts = sorted(_act_key(a) for a in solver.optimal_actions(state))
        v2.append({"value": value, "actions": acts,
                   "seconds": time.perf_counter() - t})
    v2_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            v1 = list(pool.map(_solve_one_v1, positions, chunksize=4))
    else:
        v1 = [_solve_one_v1(p) for p in positions]
    v1_wall = time.perf_counter() - t0

    solvable = [i for i, r in enumerate(v1) if r["ok"]]
    value_agree = [i for i in solvable if v1[i]["value"] == v2[i]["value"]]
    action_agree = [i for i in solvable if v1[i]["actions"] == v2[i]["actions"]]
    disagreements = []
    for i in solvable:
        if v1[i]["value"] != v2[i]["value"] or v1[i]["actions"] != v2[i]["actions"]:
            disagreements.append({
                "position": positions[i], "v1_value": v1[i]["value"],
                "v2_value": v2[i]["value"],
                "v1_only": [a for a in v1[i]["actions"] if a not in v2[i]["actions"]],
                "v2_only": [a for a in v2[i]["actions"] if a not in v1[i]["actions"]]})
    return {
        "n_positions": len(positions),
        "n_v1_solvable": len(solvable),
        "n_v1_refused": len(positions) - len(solvable),
        "value_agreement": len(value_agree) / max(1, len(solvable)),
        "optimal_action_set_agreement": len(action_agree) / max(1, len(solvable)),
        "disagreements": disagreements[:20],
        "v1_cpu_seconds": sum(r["seconds"] for r in v1),
        "v1_wall_seconds": v1_wall,
        "v1_nodes": sum(r.get("nodes", 0) for r in v1),
        "v2_seconds": v2_seconds,
        "v2_median_us": float(np.median([r["seconds"] for r in v2]) * 1e6),
        "v1_median_ms": float(np.median([r["seconds"] for r in v1
                                         if r["ok"]]) * 1e3) if solvable else None,
    }


# ---------------------------------------------------------------------------
# 2. does the count abstraction really match the engine?
# ---------------------------------------------------------------------------

def abstraction_fuzz(trials=400, seed=5):
    """Every count-space transition must be an engine transition and back.

    The whole solver rests on "cards inside a half-suit are interchangeable".
    This walks random positions and checks, at the level of the real engine,
    that the abstract successor multiset equals the concrete one - so a wrong
    symmetry would show up here rather than silently corrupting ground truth.
    """
    rng = random.Random(seed)
    rules = RuleConfig()
    solver = e.Exact2Solver(rules)
    mismatches = []
    for _ in range(trials):
        m = rng.choice([1, 1, 2, 2])
        hs = sorted(rng.sample(range(9), m))
        comps = tuple(rng.randrange(e.NC) for _ in range(m))
        turn = rng.randrange(6)
        state = build_position(tuple(hs), comps, turn)
        live, acomps, aturn = e.abstract_state(state)
        moves, exits = e.layer_transitions(acomps, aturn)
        abstract_children = sorted(set(moves_key for moves_key, _ in moves))
        abstract_exits = sorted((d, c) for d, c, _ in exits)
        concrete_children, concrete_exits = set(), []
        for act in solver.actions(state):
            child = solver.child(state, act)
            if isinstance(act, Claim):
                delta = int(e._delta(state, child))
                if child.is_terminal:
                    concrete_exits.append((delta, None))
                else:
                    _l, cc, ct = e.abstract_state(child)
                    concrete_exits.append((delta, (cc, ct)))
            else:
                _l, cc, ct = e.abstract_state(child)
                concrete_children.add((cc, ct))
        if (sorted(concrete_children) != abstract_children
                or sorted(set(concrete_exits)) != sorted(set(abstract_exits))):
            mismatches.append({"hs": hs, "comps": comps, "turn": turn})
    return {"trials": trials, "mismatches": mismatches[:10],
            "n_mismatches": len(mismatches)}


def representative_invariance(trials=200, seed=9):
    """Different card-level materialisations of one abstract state must have
    the same exact value - the count symmetry, tested rather than trusted."""
    rng = random.Random(seed)
    rules = RuleConfig()
    solver = e.Exact2Solver(rules)
    bad = []
    for _ in range(trials):
        hs = rng.randrange(9)
        comp = rng.randrange(e.NC)
        turn = rng.randrange(6)
        base = build_position((hs,), (comp,), turn)
        target = solver.value(base)
        cards = list(half_suit_cards(hs))
        for _ in range(3):
            perm = cards[:]
            rng.shuffle(perm)
            hands = [0] * 6
            i = 0
            for p, count in enumerate(e.COMPOSITIONS[comp]):
                for _ in range(count):
                    hands[p] |= 1 << perm[i]
                    i += 1
            state = GameState(rules, hands, turn)
            for other in range(9):
                if other != hs:
                    state.set_winner[other] = NULL_TEAM
            state._normalize_turn()
            if solver.value(state) != target:
                bad.append({"hs": hs, "comp": comp, "turn": turn})
    return {"trials": trials, "n_mismatches": len(bad), "examples": bad[:5]}


def constructive_action(state):
    """The drain-then-claim strategy the closed form is proved from.

    On move: take any opponent card in a half-suit you have a foothold in (a
    perfectly informed asker never misses, so the turn is never surrendered);
    when nothing is left to drain, claim a half-suit your team owns outright;
    when empty-handed, pass. Every step is legal by construction.
    """
    from fish.cards import half_suit_mask
    p = state.turn
    if state.hands[p] == 0:
        return state.legal_passes(p)[0]
    live = e.live_half_suits(state)
    for hs in live:
        hs_mask = half_suit_mask(hs)
        if not state.hands[p] & hs_mask:
            continue                              # no foothold: cannot ask here
        for o in (p + 1) % 6, (p + 3) % 6, (p + 5) % 6:
            theirs = state.hands[o] & hs_mask
            if theirs:
                return Ask(o, (theirs & -theirs).bit_length() - 1)
    team = team_of(p)
    for hs in live:
        holders = [state.holder_of(hs * 6 + i) for i in range(6)]
        if all(h is not None and team_of(h) == team for h in holders):
            return Claim(hs, tuple(holders))
    raise AssertionError(
        "the drain-then-claim strategy ran out of moves, which contradicts "
        f"the foothold argument: {state.hands} turn={state.turn}")


def constructive_playout(state, solver, guard=300):
    """(computed value, value actually realised by the constructive line).

    Both teams play the constructive strategy, so if the two numbers differ
    either the table or the closed-form argument is wrong.
    """
    value = solver.value(state)
    live = GameState.from_components(state.rules, list(state.hands),
                                     state.turn, list(state.set_winner))
    a0, b0, _ = live.scores()
    steps = 0
    while not live.is_terminal and steps < guard:
        live.apply(live.turn, constructive_action(live))
        steps += 1
    if not live.is_terminal:
        raise AssertionError("constructive play failed to terminate")
    a1, b1, _ = live.scores()
    return value, float((a1 - a0) - (b1 - b0))


def playout_study(n=120, seed=31):
    """Does the constructive line realise the table value, and does it end?"""
    rules = RuleConfig()
    solver = e.Exact2Solver(rules)
    positions = generate_m1_positions(n // 2, seed=seed)
    positions += generate_m2_positions(n // 2, seed=seed + 1)
    bad = []
    for pos in positions:
        state = rebuild(pos, rules)
        value, realised = constructive_playout(state, solver)
        if value != realised:
            bad.append({"position": pos, "value": value, "realised": realised})
    return {"positions": len(positions), "mismatches": len(bad),
            "examples": bad[:5]}


# ---------------------------------------------------------------------------
# 3. is the cyclic-game fixpoint unique?
# ---------------------------------------------------------------------------

def _jacobi_sweep(values, moves, exits, is_max, valid):
    n = values.shape[0]
    sign = np.where(is_max, 1, -1).astype(np.int32)
    best = np.full(n, e._LOW, dtype=np.int32)
    for mask, child in moves:
        best = np.maximum(best, np.where(mask, sign * values[child], e._LOW))
    for mask, val in exits:
        best = np.maximum(best, np.where(mask, sign * val.astype(np.int32),
                                         e._LOW))
    nxt = sign * best
    nxt[~valid] = 0
    return nxt


def _gauss_seidel_sweep(values, moves, exits, is_max, order):
    """Update in place, in ``order`` - the shape of v1's dict-order sweep."""
    for i in order:
        cand = [int(values[child[i]]) for mask, child in moves if mask[i]]
        cand += [int(val[i]) for mask, val in exits if mask[i]]
        values[i] = max(cand) if is_max[i] else min(cand)
    return values


def toy_nonunique_fixpoint():
    """The general fact, on the smallest graph that shows it.

    Two maximiser states in a cycle; one of them holds a claim worth -1. The
    value is 0 (loop rather than take the losing claim), but every constant
    vector >= -1 is a Bellman fixpoint, so a solver that merely "iterates to a
    fixpoint" is under-specified. This is why the zero start is part of the
    semantics and why fish4 keeps an attractor cross-check.
    """
    is_max = np.array([True, True])
    valid = np.array([True, True])
    moves = [(np.array([True, True]), np.array([1, 0], dtype=np.int32))]
    exits = [(np.array([False, True]), np.array([0, -1], dtype=np.int16))]
    from_zero, _ = e._value_iterate(moves, exits, is_max, valid)
    attractor, _ = e.solve_layer_attractor(moves, exits, is_max, valid,
                                           [-1, 0, 1])
    optimistic = np.array([5, 5], dtype=np.int32)
    stays = _jacobi_sweep(optimistic.copy(), moves, exits, is_max, valid)
    return {"value_iteration_from_zero": [int(x) for x in from_zero],
            "attractor": [int(x) for x in attractor],
            "optimistic_vector": [5, 5],
            "optimistic_vector_after_a_sweep": [int(x) for x in stays],
            "optimistic_vector_is_also_a_fixpoint":
                bool(np.array_equal(stays, optimistic))}


def fixpoint_study(m=1, seeds=(1, 2, 3), max_sweeps=400, gauss_seidel_orders=3):
    """Initialise value iteration differently and see where it lands.

    v1 documents "every non-terminal in-layer state starts at value 0" as an
    implementation note. In a general loopy zero-sum game it is not a detail
    at all - see ``toy_nonunique_fixpoint`` - so the question has to be asked
    of real Fish layers rather than answered from theory.
    """
    sub = e.layer_table(m - 1, ranks=False) if m > 1 else None
    moves, exits, is_max, valid = e._layer_slots(m, False, False, sub)
    truth, _ranks = e.solve_layer_attractor(moves, exits, is_max, valid,
                                            list(range(-m, m + 1)))
    n = truth.shape[0]
    starts = {
        "zeros": np.zeros(n, dtype=np.int32),
        "optimistic": np.full(n, m, dtype=np.int32),
        "very_optimistic": np.full(n, 5, dtype=np.int32),
        "pessimistic": np.full(n, -m, dtype=np.int32),
        "very_pessimistic": np.full(n, -5, dtype=np.int32),
    }
    for s in seeds:
        r = np.random.default_rng(s)
        starts[f"random_{s}"] = r.integers(-m - 1, m + 2, size=n).astype(np.int32)
    out = {}
    for name, start in starts.items():
        values = start.copy()
        values[~valid] = 0
        for sweeps in range(1, max_sweeps + 1):
            nxt = _jacobi_sweep(values, moves, exits, is_max, valid)
            if np.array_equal(nxt, values):
                break
            values = nxt
        else:
            out[f"{name}/jacobi"] = {"converged": False}
            continue
        diff = int(np.sum((values != truth) & valid))
        out[f"{name}/jacobi"] = {
            "converged": True, "sweeps": sweeps,
            "states_differing_from_the_attractor_value": diff,
            "examples": [{"state": int(i), "fixpoint": int(values[i]),
                          "attractor_value": int(truth[i])}
                         for i in np.flatnonzero((values != truth) & valid)[:5]]}
    # sweep-order dependence: v1 is Gauss-Seidel in dict order, so try several
    if n <= 20000:
        live = list(np.flatnonzero(valid))
        rng = random.Random(0)
        for trial in range(gauss_seidel_orders):
            order = live[:]
            rng.shuffle(order)
            values = np.zeros(n, dtype=np.int32)
            for sweeps in range(1, max_sweeps + 1):
                before = values.copy()
                values = _gauss_seidel_sweep(values, moves, exits, is_max, order)
                if np.array_equal(before, values):
                    break
            out[f"zeros/gauss_seidel_order_{trial}"] = {
                "converged": True, "sweeps": sweeps,
                "states_differing_from_the_attractor_value":
                    int(np.sum((values != truth) & valid))}
    agree = all(r.get("states_differing_from_the_attractor_value") == 0
                for r in out.values())
    return {"m": m, "n_states": int(valid.sum()),
            "every_initialisation_agrees": agree, "results": out}


# ---------------------------------------------------------------------------
# 4. the closed form the tables exposed
# ---------------------------------------------------------------------------

def closed_form_predicate(comps, turn):
    """sign(mover team) * (2f - m), f = live half-suits the mover's team is in."""
    m = len(comps)
    f = 0
    for c in comps:
        counts = e.COMPOSITIONS[c]
        if any(counts[(turn + off) % 6] for off in (0, 2, 4)):
            f += 1
    sign = 1 if team_of(turn) == 0 else -1
    return sign * (2 * f - m)


def closed_form_study(m3_samples=40, seed=3):
    """Exhaustive at m = 1 and m = 2; sampled at m = 3 via the reachable path."""
    report = {}
    for m in (1, 2):
        table = e.layer_table(m)
        n = table.values.shape[0]
        idx = np.arange(n)
        turn = idx % 6
        rest = idx // 6
        comps = []
        for _ in range(m):
            comps.append(rest % e.NC)
            rest = rest // e.NC
        comps.reverse()
        f = np.zeros(n, dtype=np.int32)
        for c in comps:
            counts = e.COMP[c]
            has = np.zeros(n, dtype=bool)
            for off in (0, 2, 4):
                has |= counts[np.arange(n), (turn + off) % 6] > 0
            f += has
        sign = np.where(turn % 2 == 0, 1, -1)
        pred = (sign * (2 * f - m)).astype(np.int16)
        ok = table.valid
        report[f"m={m}"] = {
            "states": int(ok.sum()),
            "closed_form_exact": bool(np.array_equal(pred[ok], table.values[ok])),
            "agreement": float((pred[ok] == table.values[ok]).mean()),
            "value_histogram": {int(k): int(v) for k, v in
                                zip(*np.unique(table.values[ok],
                                               return_counts=True))},
        }
    # m = 3 by reachable-set solve, which never builds the 5.9e8-state table
    rng = random.Random(seed)
    solver = e.Exact2Solver(RuleConfig(), max_full_layer=2)
    agree, checked, refused, sizes, secs, fails = 0, 0, 0, [], [], []
    for _ in range(m3_samples):
        comps = tuple(rng.randrange(e.NC) for _ in range(3))
        turn = rng.randrange(6)
        if e.normalize_turn(turn, e.hand_sizes(comps)) != turn:
            continue                       # not a normalised (legal) turn
        t0 = time.perf_counter()
        try:
            sol = e.solve_reachable((comps, turn), solver._sub_value,
                                    max_states=400_000)
        except e.LayerTooLarge:
            refused += 1
            continue
        secs.append(time.perf_counter() - t0)
        sizes.append(sol.n_states)
        checked += 1
        got = sol.values[(comps, turn)]
        want = closed_form_predicate(comps, turn)
        if got == want:
            agree += 1
        else:
            fails.append({"comps": comps, "turn": turn, "solver": got,
                          "closed_form": want})
    report["m=3"] = {"sampled": m3_samples, "checked": checked,
                     "agreeing": agree, "refused_as_too_large": refused,
                     "failures": fails[:5],
                     "median_reachable_states": (int(np.median(sizes))
                                                 if sizes else None),
                     "max_reachable_states": int(max(sizes)) if sizes else None,
                     "median_seconds": float(np.median(secs)) if secs else None,
                     "max_seconds": float(max(secs)) if secs else None}
    return report


# ---------------------------------------------------------------------------
# 5. do deliberately losing claims ever matter?
# ---------------------------------------------------------------------------

def giveaway_study():
    """v1 refuses to generate a claim that hands the half-suit to the
    opponents. That is a real restriction of the action set, so it needs a
    measurement rather than an argument."""
    out = {}
    for m in (1, 2):
        plain = e.layer_table(m, giveaway=False)
        wide = e.layer_table(m, giveaway=True)
        ok = plain.valid
        diff = (plain.values != wide.values) & ok
        out[f"m={m}"] = {
            "states": int(ok.sum()),
            "states_where_value_changes": int(diff.sum()),
            "value_histogram_with_giveaway": {
                int(k): int(v) for k, v in
                zip(*np.unique(wide.values[ok], return_counts=True))},
        }
    return out


# ---------------------------------------------------------------------------
# 6. timing: the honest before/after
# ---------------------------------------------------------------------------

def timing(n_m1=60, n_m2=60, workers=3):
    rules = RuleConfig()
    m1 = generate_m1_positions(n_m1, seed=101)
    m2 = generate_m2_positions(n_m2, seed=202)

    e.clear_tables()
    t0 = time.perf_counter()
    e.layer_table(1)
    t_build1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    e.layer_table(2)
    t_build2 = time.perf_counter() - t0

    solver = e.Exact2Solver(rules)
    out = {"table_build_seconds": {"m=1": t_build1, "m=2": t_build2}}
    for name, positions in (("m=1", m1), ("m=2", m2)):
        t0 = time.perf_counter()
        for pos in positions:
            solver.value(rebuild(pos, rules))
        out[f"v2_value_{name}_seconds_per_position"] = (
            (time.perf_counter() - t0) / len(positions))
        t0 = time.perf_counter()
        for pos in positions:
            solver.optimal_actions(rebuild(pos, rules))
        out[f"v2_optimal_actions_{name}_seconds_per_position"] = (
            (time.perf_counter() - t0) / len(positions))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        t0 = time.perf_counter()
        v1_m1 = list(pool.map(_solve_one_v1, m1, chunksize=2))
        wall1 = time.perf_counter() - t0
        t0 = time.perf_counter()
        v1_m2 = list(pool.map(_solve_one_v1, m2, chunksize=2))
        wall2 = time.perf_counter() - t0
    solved1 = [r for r in v1_m1 if r["ok"]]
    out["v1_m=1_seconds_per_position"] = (sum(r["seconds"] for r in solved1)
                                          / max(1, len(solved1)))
    out["v1_m=1_solved"] = len(solved1)
    out["v1_m=1_wall_seconds_3_workers"] = wall1
    out["v1_m=1_mean_nodes"] = (sum(r["nodes"] for r in solved1)
                                / max(1, len(solved1)))
    out["v1_m=2_solved"] = sum(1 for r in v1_m2 if r["ok"])
    out["v1_m=2_refusal_example"] = next(
        (r["error"] for r in v1_m2 if not r["ok"]), None)
    out["v1_m=2_wall_seconds_3_workers"] = wall2
    return out


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study", choices=["all", "cross", "fuzz", "fixpoint",
                                      "closed", "giveaway", "timing"])
    ap.add_argument("--positions", type=int, default=240)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(RESULTS, "exact2_study.json"))
    args = ap.parse_args(argv)

    report = {}
    if os.path.exists(args.out):                 # studies are run separately
        with open(args.out, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    if args.study in ("all", "fuzz"):
        report["abstraction_fuzz"] = abstraction_fuzz()
        report["representative_invariance"] = representative_invariance()
        report["playout"] = playout_study()
    if args.study in ("all", "cross"):
        report["cross_check_m1"] = cross_check(
            generate_m1_positions(args.positions), workers=args.workers)
    if args.study in ("all", "fixpoint"):
        report["toy_nonunique_fixpoint"] = toy_nonunique_fixpoint()
        report["fixpoint_m1"] = fixpoint_study(1)
        report["fixpoint_m2"] = fixpoint_study(2)
    if args.study in ("all", "closed"):
        report["closed_form"] = closed_form_study()
    if args.study in ("all", "giveaway"):
        report["giveaway"] = giveaway_study()
    if args.study in ("all", "timing"):
        report["timing"] = timing(workers=args.workers)

    os.makedirs(RESULTS, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str)[:12000])
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
