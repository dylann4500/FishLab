"""Harvest ``results/exact_positions_v2.json``: the v0.4 absolute benchmark.

Why this file exists in this shape
----------------------------------
The v0.3 benchmark (``fish.benchmark_exact``) re-solved every position inside
the loop, could only reach one-half-suit endgames, and kept the solution in
memory rather than on disk, so no two runs were comparable and no one could
audit which positions were being scored. This writes the positions and their
ground truth down ONCE, with enough context to reconstruct the exact
``GameState`` and the exact public log, so an agent evaluated next month is
evaluated on the same objects.

Positions come from real games played by the registry agents, not from random
placements: the point is to score agents on positions they will actually meet.
Both regimes are harvested - one live half-suit (what v0.3 could solve) and
two (what only v2 can) - and every record carries the tags needed to slice
the results honestly, above all whether the hidden information is already
publicly resolved.

The schema is documented in fish4/EXACT2.md section 6. Run:

    py -m fish4.exact2_corpus --games 400 --out results/exact_positions_v2.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fish.agents import AGENT_REGISTRY
from fish.cards import card_name, team_of
from fish.engine import (Ask, AskEvent, Claim, ClaimEvent, GameState, Pass,
                         PassEvent)
from fish.exact import information_is_resolved, live_card_count
from fish.observation import Observation
from fish.rules import RuleConfig

from fish4 import exact2 as e

SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------

def event_to_dict(ev) -> dict:
    if isinstance(ev, AskEvent):
        return {"t": "ask", "asker": ev.asker, "target": ev.target,
                "card": ev.card, "success": ev.success}
    if isinstance(ev, ClaimEvent):
        return {"t": "claim", "claimer": ev.claimer, "half_suit": ev.half_suit,
                "declared": list(ev.declared), "revealed": list(ev.revealed),
                "winner": ev.winner}
    return {"t": "pass", "player": ev.player, "teammate": ev.teammate}


def event_from_dict(d: dict):
    if d["t"] == "ask":
        return AskEvent(d["asker"], d["target"], d["card"], d["success"])
    if d["t"] == "claim":
        return ClaimEvent(d["claimer"], d["half_suit"], tuple(d["declared"]),
                          tuple(d["revealed"]), d["winner"])
    return PassEvent(d["player"], d["teammate"])


def action_to_dict(act) -> dict:
    if isinstance(act, Ask):
        return {"type": "ask", "target": act.target, "card": act.card,
                "card_name": card_name(act.card)}
    if isinstance(act, Claim):
        return {"type": "claim", "half_suit": act.half_suit,
                "assignment": list(act.assignment)}
    return {"type": "pass", "teammate": act.teammate}


def action_from_dict(d: dict):
    if d["type"] == "ask":
        return Ask(d["target"], d["card"])
    if d["type"] == "claim":
        return Claim(d["half_suit"], tuple(d["assignment"]))
    return Pass(d["teammate"])


def state_from_record(rec: dict, rules: RuleConfig | None = None) -> GameState:
    """Rebuild the exact GameState, public log included."""
    rules = rules or RuleConfig(**rec["rules"])
    state = GameState.from_components(rules, rec["hands"], rec["turn"],
                                      [None if w is None else w
                                       for w in rec["set_winner"]])
    state.history = [event_from_dict(d) for d in rec["history"]]
    return state


def load_corpus(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# harvesting
# ---------------------------------------------------------------------------

def harvest(n_games=400, seed=0, generator="probabilistic", max_live_sets=2,
            per_game=4, guard=3000):
    """Play real games and snapshot every position with <= max_live_sets left.

    ``generator`` may be a comma-separated list, in which case games rotate
    through them. A corpus grown by one policy only would be biased towards
    the endgames that policy happens to produce, which is precisely the sort
    of quiet circularity an absolute benchmark exists to avoid.
    """
    rules = RuleConfig()
    rng = random.Random(seed)
    pool = [g.strip() for g in generator.split(",") if g.strip()]
    raw = []
    for g in range(n_games):
        state = GameState.deal(rules, seed=seed * 100003 + g)
        agents = [AGENT_REGISTRY[pool[g % len(pool)]]() for _ in range(6)]
        for p, a in enumerate(agents):
            a.begin_game(p, rules, rng.getrandbits(64))
        taken = 0
        steps = 0
        while not state.is_terminal and steps < guard:
            live_sets = sum(1 for w in state.set_winner if w is None)
            if 1 <= live_sets <= max_live_sets and taken < per_game:
                p = state.turn
                obs = Observation.from_state(state, p)
                if obs.legal_asks() or obs.claimable_half_suits() or obs.must_pass():
                    raw.append({"hands": list(state.hands), "turn": state.turn,
                                "set_winner": list(state.set_winner),
                                "history": [event_to_dict(ev)
                                            for ev in state.history],
                                "game": g, "ply": steps})
                    taken += 1
            p = state.turn
            state.apply(p, agents[p].act(Observation.from_state(state, p)))
            steps += 1
    return raw


def annotate(raw, rules=None, dedupe=True):
    """Attach exact ground truth and the slicing tags to each snapshot."""
    rules = rules or RuleConfig()
    solver = e.Exact2Solver(rules)
    # A claim that hands the half-suit to the opponents is legal, and it ties
    # the best legitimate line in 6.6% of the whole two-half-suit table but in
    # 52% of the two-half-suit positions real play actually reaches (EXACT2.md
    # sections 5 and 6). v1 never generated one, so ``optimal_actions`` keeps
    # the v1-compatible action set and the tie is recorded as a flag instead,
    # for a benchmark author to include or exclude deliberately.
    wide = e.Exact2Solver(rules, include_giveaway_claims=True)
    seen = set()
    out = []
    for rec in raw:
        state = state_from_record({**rec, "rules": rules.to_dict()}, rules)
        live, comps, turn = e.abstract_state(state)
        key = (live, comps, turn)
        if dedupe and key in seen:
            continue
        seen.add(key)
        try:
            value = solver.value(state)
            optimal = solver.optimal_actions(state)
            progress = solver.progress_optimal_actions(state)
        except e.Exact2Error:
            continue
        legal = solver.actions(state)
        mover = state.turn
        opp_cards = sum(state.hands[o].bit_count()
                        for o in range(6) if team_of(o) != team_of(mover))
        footholds = sum(
            1 for c in comps
            if any(e.COMPOSITIONS[c][(mover + off) % 6] for off in (0, 2, 4)))
        tags = []
        if state.hands[mover] == 0:
            tags.append("cardless_mover")
        if opp_cards == 0:
            tags.append("forced_claim")           # no legal ask exists at all
        if any(state.hands[p] == 0 for p in range(6)):
            tags.append("empty_hand_present")
        if sum(state.hands[p].bit_count() for p in range(6)
               if team_of(p) == 0) == 0 or sum(
                   state.hands[p].bit_count() for p in range(6)
                   if team_of(p) == 1) == 0:
            tags.append("team_holds_nothing")
        if any(w == -1 for w in state.set_winner):
            tags.append("nulled_half_suit_present")
        if len(progress) < len(optimal):
            tags.append("stalling_action_is_bellman_optimal")
        giveaway_ties = any(
            isinstance(a, Claim) and a not in optimal
            for a in wide.optimal_actions(state))
        if giveaway_ties:
            tags.append("giveaway_claim_ties_optimum")
        out.append({
            "id": len(out),
            "rules": rules.to_dict(),
            "hands": list(state.hands),
            "turn": state.turn,
            "set_winner": list(state.set_winner),
            "history": rec["history"],
            "source_game": rec.get("game"),
            "source_ply": rec.get("ply"),
            "live_half_suits": list(live),
            "n_live_half_suits": len(live),
            "live_cards": live_card_count(state),
            "abstract_compositions": [list(e.COMPOSITIONS[c]) for c in comps],
            "information_resolved": bool(information_is_resolved(state)),
            "mover_team_footholds": footholds,
            "value": value,
            "n_legal_actions": len(legal),
            "optimal_actions": [action_to_dict(a) for a in optimal],
            "progress_optimal_actions": [action_to_dict(a) for a in progress],
            "giveaway_claim_ties_optimum": giveaway_ties,
            "tags": tags,
        })
    return out


def summarise(records) -> dict:
    def frac(pred):
        return sum(1 for r in records if pred(r))
    tags: dict[str, int] = {}
    for r in records:
        for t in r["tags"]:
            tags[t] = tags.get(t, 0) + 1
    values: dict[str, int] = {}
    for r in records:
        values[str(r["value"])] = values.get(str(r["value"]), 0) + 1
    return {
        "n_positions": len(records),
        "n_information_resolved": frac(lambda r: r["information_resolved"]),
        "by_live_half_suits": {
            str(k): frac(lambda r, k=k: r["n_live_half_suits"] == k)
            for k in sorted({r["n_live_half_suits"] for r in records})},
        "by_value": values,
        "tag_counts": tags,
        "mean_optimal_actions": (sum(len(r["optimal_actions"]) for r in records)
                                 / max(1, len(records))),
        "mean_legal_actions": (sum(r["n_legal_actions"] for r in records)
                               / max(1, len(records))),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generator", default="probabilistic")
    ap.add_argument("--per-game", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "exact_positions_v2.json"))
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    raw = harvest(args.games, seed=args.seed, generator=args.generator,
                  per_game=args.per_game)
    t_play = time.perf_counter() - t0
    t0 = time.perf_counter()
    records = annotate(raw)
    t_solve = time.perf_counter() - t0
    summary = summarise(records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "fish4.exact2_corpus",
        "solver": "fish4.exact2.Exact2Solver",
        "generator_agents": args.generator,
        "games_played": args.games,
        "seed": args.seed,
        "seconds_playing": t_play,
        "seconds_solving": t_solve,
        "summary": summary,
        "positions": records,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        # compact: the public logs dominate the file and indentation was
        # costing 10 MB of the 24 MB for no benefit to any reader
        json.dump(payload, fh, separators=(",", ":"))
    print(json.dumps(summary, indent=2))
    print(f"play {t_play:.1f}s  solve {t_solve:.1f}s  -> {args.out}")


if __name__ == "__main__":
    main()
