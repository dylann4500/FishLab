"""Interactive play and replay viewing.

  py -m fish.cli play [--seat 0] [--engine probabilistic]
  py -m fish.cli replay [seed] [--engine memory]

The human sees exactly what the engine's Observation exposes, no more.
"""

from __future__ import annotations

import random

from .agents import AGENT_REGISTRY
from .cards import (HALF_SUIT_NAMES, card_id, card_long_name, card_name,
                    half_suit_cards, mask_to_cards, team_of)
from .engine import (Ask, AskEvent, Claim, ClaimEvent, GameState, Pass,
                     PassEvent, IllegalAction, NULL_TEAM)
from .observation import Observation
from .rules import RuleConfig


def describe_event(ev) -> str:
    if isinstance(ev, AskEvent):
        mark = "GOT IT" if ev.success else "no"
        return (f"P{ev.asker} asks P{ev.target} for "
                f"{card_long_name(ev.card)} -> {mark}")
    if isinstance(ev, ClaimEvent):
        who = ("team 0" if ev.winner == 0 else
               "team 1" if ev.winner == 1 else "NOBODY (null)")
        detail = " ".join(f"{card_name(ev.half_suit * 6 + i)}=P{h}"
                          for i, h in enumerate(ev.revealed))
        return (f"P{ev.claimer} claims {HALF_SUIT_NAMES[ev.half_suit]} "
                f"-> {who}   [{detail}]")
    if isinstance(ev, PassEvent):
        return f"P{ev.player} passes the turn to P{ev.teammate}"
    return str(ev)


def show_observation(obs: Observation) -> None:
    me = obs.player
    print(f"\n--- You are P{me} (team {team_of(me)}) ---")
    by_suit: dict[int, list[str]] = {}
    for c in mask_to_cards(obs.hand):
        by_suit.setdefault(c // 6, []).append(card_name(c))
    for hs in sorted(by_suit):
        print(f"  {HALF_SUIT_NAMES[hs]:22s} {' '.join(by_suit[hs])}")
    counts = "  ".join(f"P{p}:{c}" for p, c in enumerate(obs.hand_counts))
    print(f"  cards: {counts}")
    a = sum(1 for w in obs.set_winner if w == 0)
    b = sum(1 for w in obs.set_winner if w == 1)
    n = sum(1 for w in obs.set_winner if w == NULL_TEAM)
    print(f"  sets: team0={a} team1={b} null={n}")
    live = [HALF_SUIT_NAMES[i] for i, w in enumerate(obs.set_winner)
            if w is None]
    print(f"  live half-suits: {', '.join(live)}")


def parse_action(text: str, obs: Observation):
    """'ask 3 6H' | 'claim 0 0,0,2,4,4,4' | 'pass 2'"""
    parts = text.strip().split()
    if not parts:
        return None
    verb = parts[0].lower()
    try:
        if verb in ("ask", "a") and len(parts) >= 3:
            return Ask(int(parts[1].lstrip("Pp")), card_id(" ".join(parts[2:])))
        if verb in ("claim", "c") and len(parts) >= 3:
            owners = tuple(int(x) for x in
                           parts[2].replace(" ", "").split(","))
            return Claim(int(parts[1]), owners)
        if verb in ("pass", "p") and len(parts) >= 2:
            return Pass(int(parts[1].lstrip("Pp")))
    except (ValueError, KeyError):
        return None
    return None


def help_text(obs: Observation) -> str:
    lines = ["Commands:",
             "  ask <player> <card>          e.g. ask 3 6H  |  ask 1 Red Joker",
             "  claim <hs#> <o1,..,o6>       e.g. claim 0 0,0,2,4,4,4",
             "  pass <teammate>              e.g. pass 2",
             "  hand | log | legal | help | quit",
             "Half-suit numbers:"]
    for i, name in enumerate(HALF_SUIT_NAMES[:obs.num_half_suits]):
        cards = " ".join(card_name(c) for c in half_suit_cards(i))
        lines.append(f"  {i}: {name:22s} {cards}")
    return "\n".join(lines)


def play(seat: int = 0, engine: str = "probabilistic",
         seed: int | None = None, engine_kwargs: dict | None = None) -> None:
    rules = RuleConfig()
    if seed is None:
        seed = random.randrange(1 << 30)
    state = GameState.deal(rules, seed=seed)
    kw = engine_kwargs or {}
    agents = [None if p == seat else AGENT_REGISTRY[engine](**kw)
              for p in range(6)]
    rng = random.Random()
    for p, ag in enumerate(agents):
        if ag is not None:
            ag.begin_game(p, rules, rng.getrandbits(64))
    print(f"Fish: you are P{seat} on team {team_of(seat)}; "
          f"opponents play '{engine}'. (deal seed {seed})")
    print(help_text(Observation.from_state(state, seat)))
    shown = 0
    while not state.is_terminal:
        p = state.turn
        if p != seat:
            state.apply(p, agents[p].act(Observation.from_state(state, p)))
            continue
        for ev in state.history[shown:]:
            print("   ", describe_event(ev))
        shown = len(state.history)
        obs = Observation.from_state(state, seat)
        show_observation(obs)
        while True:
            try:
                text = input("your move> ")
            except EOFError:
                print("\n(input closed)")
                return
            low = text.strip().lower()
            if low in ("quit", "q", "exit"):
                return
            if low in ("help", "?"):
                print(help_text(obs))
                continue
            if low == "hand":
                show_observation(obs)
                continue
            if low == "log":
                for i, ev in enumerate(state.history):
                    print(f"  {i:4d} {describe_event(ev)}")
                continue
            if low == "legal":
                asks = obs.legal_asks()
                print(f"  {len(asks)} legal asks; e.g. " + ", ".join(
                    f"ask {a.target} {card_name(a.card)}" for a in asks[:8]))
                print(f"  claimable half-suits: {obs.claimable_half_suits()}")
                continue
            action = parse_action(text, obs)
            if action is None:
                print("  ? try 'help'")
                continue
            try:
                ev = state.apply(seat, action)
            except IllegalAction as e:
                print(f"  illegal: {e}")
                continue
            print("   ", describe_event(ev))
            shown = len(state.history)
            break
    for ev in state.history[shown:]:
        print("   ", describe_event(ev))
    a, b, n = state.scores()
    mine, theirs = (a, b) if team_of(seat) == 0 else (b, a)
    verdict = ("YOU WIN" if mine > theirs
               else "you lose" if theirs > mine else "tie")
    print(f"\n=== {verdict}: your team {mine}, opponents {theirs}, null {n} ===")


def replay(seed: int = 0, engine: str = "memory") -> None:
    rules = RuleConfig()
    state = GameState.deal(rules, seed=seed)
    initial = list(state.hands)
    agents = [AGENT_REGISTRY[engine]() for _ in range(6)]
    rng = random.Random(seed)
    for p, ag in enumerate(agents):
        ag.begin_game(p, rules, rng.getrandbits(64))
    print(f"=== replay: {engine} self-play, deal seed {seed} ===")
    for p in range(6):
        print(f"  P{p} (team {team_of(p)}): "
              f"{' '.join(card_name(c) for c in mask_to_cards(initial[p]))}")
    print()
    i = 0
    while not state.is_terminal:
        p = state.turn
        ev = state.apply(p, agents[p].act(Observation.from_state(state, p)))
        print(f"{i:4d}  {describe_event(ev)}")
        i += 1
    a, b, n = state.scores()
    print(f"\nFinal: team0={a} team1={b} null={n} "
          f"({'team 0' if a > b else 'team 1' if b > a else 'tie'})")
