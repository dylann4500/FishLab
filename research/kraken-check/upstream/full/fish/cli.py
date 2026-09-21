"""Fish engine CLI.

  py -m fish.cli analyze <observation.json>   # best-move analysis
  py -m fish.cli demo [seed] [steps]          # write a sample observation
  py -m fish.cli selfplay [seed]              # engine-vs-engine transcript
  py -m fish.cli replay [seed]                # readable game replay
  py -m fish.cli play                         # play against the engine
  py -m fish.cli serve                        # web simulation platform
  py -m fish.cli bench [n]                    # simulator throughput
  py -m fish.cli solve                        # exact endgame ground truth
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

from .agents import (HeuristicAgent, MemoryAgent, ProbabilisticAgent,
                     ValueSearchAgent)
from .cards import (HALF_SUIT_NAMES, card_long_name, card_name,
                    half_suit_mask, mask_to_cards, team_of)
from .engine import Ask, Claim, GameState, Pass
from .observation import Observation
from .rules import RuleConfig
from .runner import play_game

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "checkpoints" / "pi_value_v1.pt"


def describe_action(action) -> str:
    if isinstance(action, Ask):
        return f"ASK P{action.target} for {card_long_name(action.card)}"
    if isinstance(action, Claim):
        parts = ", ".join(f"{card_name(action.half_suit * 6 + i)}->P{p}"
                          for i, p in enumerate(action.assignment))
        return f"CLAIM {HALF_SUIT_NAMES[action.half_suit]} ({parts})"
    if isinstance(action, Pass):
        return f"PASS to P{action.teammate}"
    return str(action)


def analyze(path: str, n_worlds: int = 16, n_samples: int = 64) -> None:
    obs = Observation.from_json(Path(path).read_text())
    me = obs.player
    if obs.turn != me:
        print(f"NOTE: it is P{obs.turn}'s turn, not P{me}'s; the analysis "
              f"assumes you act next anyway.")
    agent = ValueSearchAgent(model_path=str(DEFAULT_MODEL),
                             n_worlds=n_worlds, n_candidates=8,
                             n_samples=n_samples)
    agent.begin_game(me, obs.rules, seed=0)
    agent.bel.update(obs)
    bel = agent.bel

    print(f"=== Fish analysis: P{me} (team {team_of(me)}) ===")
    print(f"Hand: {' '.join(card_name(c) for c in mask_to_cards(obs.hand))}")
    a, b, n = (sum(1 for w in obs.set_winner if w == t) for t in (0, 1, -1))
    print(f"Sets: team0={a} team1={b} null={n} | counts={obs.hand_counts}")
    print(f"Evaluator: {'learned value net' if agent.uses_model else 'static'}")

    if obs.must_pass():
        print("You are cardless: PASS to the teammate with the most cards.")
        return

    worlds = [bel.sample_current_hands(agent.rng) for _ in range(n_samples)]
    claim_cands = agent._claim_candidates(obs, worlds)
    asks = obs.legal_asks()

    print(f"\nBest action:\n  {describe_action(agent.act(obs))}")

    if claim_cands:
        print("\nClaim confidence:")
        for p, cl in sorted(claim_cands, key=lambda t: -t[0])[:4]:
            print(f"  {HALF_SUIT_NAMES[cl.half_suit]:22s} "
                  f"P(exactly right) = {p:.0%}")

    scored = []
    for ask in asks:
        hits = sum(1 for w in worlds if w[ask.target] & (1 << ask.card))
        scored.append((hits / len(worlds), ask))
    scored.sort(key=lambda t: -t[0])
    if scored:
        print("\nAsks by success probability:")
        for p, ask in scored[:10]:
            print(f"  {describe_action(ask):46s} {p:5.0%}")

    print("\nBeliefs (cards in your live half-suits):")
    marg = [[0.0] * 6 for _ in range(obs.deck_size)]
    for w in worlds:
        for p in range(6):
            h = w[p]
            while h:
                low = h & -h
                marg[low.bit_length() - 1][p] += 1 / len(worlds)
                h ^= low
    my_suits = {hs for hs in range(obs.num_half_suits)
                if obs.set_winner[hs] is None and obs.hand & half_suit_mask(hs)}
    for hs in sorted(my_suits):
        for c in range(hs * 6, hs * 6 + 6):
            if obs.hand & (1 << c) or bel.public_loc[c] == -2:
                continue
            probs = " ".join(f"P{p}:{marg[c][p]:.0%}"
                             for p in range(6) if marg[c][p] > 0.01)
            certain = "  (CERTAIN)" if max(marg[c]) > 0.999 else ""
            print(f"  {card_name(c):3s} -> {probs}{certain}")

    print("\nReasoning notes:")
    known = bel.known_current_hands()
    for p in range(6):
        if p != me and known[p]:
            print(f"  P{p} certainly holds: "
                  f"{' '.join(card_name(c) for c in mask_to_cards(known[p]))}")
    for cards, p in bel.ors[:8]:
        print(f"  P{p} holds at least one of: "
              f"{' '.join(card_name(c) for c in cards)}")


def demo(seed: int = 0, steps: int = 40) -> None:
    rules = RuleConfig()
    state = GameState.deal(rules, seed=seed)
    agents = [MemoryAgent() for _ in range(6)]
    rng = random.Random(seed)
    for p, ag in enumerate(agents):
        ag.begin_game(p, rules, rng.getrandbits(64))
    for _ in range(steps):
        if state.is_terminal:
            break
        p = state.turn
        state.apply(p, agents[p].act(Observation.from_state(state, p)))
    obs = Observation.from_state(state, state.turn)
    path = f"observation_seed{seed}_step{len(state.history)}.json"
    Path(path).write_text(obs.to_json())
    print(f"Wrote {path} (P{obs.turn} to act)")


def selfplay(seed: int = 0) -> None:
    from .play import describe_event
    rules = RuleConfig()
    state = play_game([ProbabilisticAgent() for _ in range(6)], rules,
                      seed=seed, agent_seed=seed)
    for i, ev in enumerate(state.history):
        print(f"{i:4d}  {describe_event(ev)}")
    a, b, n = state.scores()
    print(f"\nFinal: team0={a} team1={b} null={n}")


def bench(n: int = 200) -> None:
    rules = RuleConfig()
    for name, cls in (("heuristic", HeuristicAgent), ("memory", MemoryAgent)):
        agents = [cls() for _ in range(6)]
        t0 = time.perf_counter()
        acts = 0
        for seed in range(n):
            st = play_game(agents, rules, seed=seed, agent_seed=seed)
            acts += len(st.history)
        dt = time.perf_counter() - t0
        print(f"{name:12s} {n} games in {dt:.2f}s = {n/dt:.1f} games/s, "
              f"{acts/dt:.0f} actions/s")


def solve() -> None:
    """Demonstrate exact ground truth on a small endgame."""
    from .exact import make_endgame, solve_position
    from .cards import card_id as cid
    rules = RuleConfig()
    hands = [
        (1 << cid("2C")) | (1 << cid("3C")) | (1 << cid("4C")),
        (1 << cid("5C")),
        0,
        (1 << cid("6C")) | (1 << cid("7C")),
        0, 0,
    ]
    st = make_endgame(rules, [0], hands, turn=0)
    r = solve_position(st)
    print("Exact solution of a Low Clubs endgame:")
    print(f"  P0 holds 2C 3C 4C | P1 holds 5C | P3 holds 6C 7C, P0 to move")
    print(f"  exact value (team 0 remaining set differential): {r['value']:+.0f}")
    print(f"  optimal action: {describe_action(r['action'])}")
    print(f"  all optimal actions: "
          f"{[describe_action(a) for a in r['optimal_actions']]}")
    print(f"  states searched: {r['nodes']}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="fish")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_an = sub.add_parser("analyze")
    p_an.add_argument("path")
    p_an.add_argument("--worlds", type=int, default=16)
    p_an.add_argument("--samples", type=int, default=64)
    p_de = sub.add_parser("demo")
    p_de.add_argument("seed", type=int, nargs="?", default=0)
    p_de.add_argument("steps", type=int, nargs="?", default=40)
    p_sp = sub.add_parser("selfplay")
    p_sp.add_argument("seed", type=int, nargs="?", default=0)
    p_rp = sub.add_parser("replay")
    p_rp.add_argument("seed", type=int, nargs="?", default=0)
    p_rp.add_argument("--engine", default="memory")
    p_pl = sub.add_parser("play")
    p_pl.add_argument("--seat", type=int, default=0)
    p_pl.add_argument("--engine", default="probabilistic")
    p_pl.add_argument("--seed", type=int, default=None)
    p_sv = sub.add_parser("serve")
    p_sv.add_argument("--port", type=int, default=8777)
    p_sv.add_argument("--host", default="127.0.0.1")
    p_be = sub.add_parser("bench")
    p_be.add_argument("n", type=int, nargs="?", default=200)
    sub.add_parser("solve")
    args = ap.parse_args()

    if args.cmd == "analyze":
        analyze(args.path, n_worlds=args.worlds, n_samples=args.samples)
    elif args.cmd == "demo":
        demo(args.seed, args.steps)
    elif args.cmd == "selfplay":
        selfplay(args.seed)
    elif args.cmd == "replay":
        from .play import replay
        replay(args.seed, args.engine)
    elif args.cmd == "play":
        from .play import play
        play(seat=args.seat, engine=args.engine, seed=args.seed)
    elif args.cmd == "serve":
        from .web.server import serve
        serve(host=args.host, port=args.port)
    elif args.cmd == "bench":
        bench(args.n)
    elif args.cmd == "solve":
        solve()


if __name__ == "__main__":
    main()
