"""FishBot v0.4 command line: play it, analyse with it, simulate with it.

    py -m fish4 play                 sit at seat 0 and play against the engine
    py -m fish4 watch                watch the engine play itself
    py -m fish4 analyse              analyse a position from a played-out game
    py -m fish4 sim      --games 100 run simulations and report statistics
    py -m fish4 duel     --games 200 duplicate-deal head-to-head
    py -m fish4 perpetual            the dead-position / deadlock study
    py -m fish4 bench                speed
    py -m fish4 serve                browser UI

Every command takes ``--help``.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fish.cards import (CARD_NAMES, HALF_SUIT_NAMES, NUM_PLAYERS, card_id,
                        card_name, half_suit_cards, half_suit_mask,
                        mask_to_cards, num_half_suits, opponents, team_of,
                        teammates)
from fish.engine import Ask, AskEvent, Claim, ClaimEvent, GameState, Pass, PassEvent
from fish.observation import Observation
from fish.rules import RuleConfig

CHAMPION = ("fishbot4", {"opponent_gamma": 0.35})


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------

def _hand_str(hand: int) -> str:
    """Own hand, grouped by half-suit, which is how a player actually reads it."""
    parts = []
    for hs in range(9):
        cards = [c for c in half_suit_cards(hs) if hand >> c & 1]
        if cards:
            parts.append(f"  {HALF_SUIT_NAMES[hs]:22s} "
                         + " ".join(card_name(c) for c in cards))
    return "\n".join(parts) if parts else "  (no cards)"


def _event_str(ev) -> str:
    if isinstance(ev, AskEvent):
        return (f"P{ev.asker} asked P{ev.target} for {card_name(ev.card)}"
                f" -> {'GOT IT' if ev.success else 'no'}")
    if isinstance(ev, ClaimEvent):
        w = ("team " + str(ev.winner)) if ev.winner in (0, 1) else "NOBODY (null)"
        return (f"P{ev.claimer} declared {HALF_SUIT_NAMES[ev.half_suit]}"
                f" -> {w}")
    if isinstance(ev, PassEvent):
        return f"P{ev.player} passed the turn to P{ev.teammate}"
    return str(ev)


def _board(state: GameState, seat: int) -> str:
    a, b, n = state.scores()
    mine = team_of(seat)
    lines = [
        f"  score   you {a if mine == 0 else b} - {b if mine == 0 else a}"
        f"   nulled {n}",
        "  cards   " + "  ".join(
            f"{'YOU' if p == seat else 'P' + str(p)}"
            f"{'*' if team_of(p) == mine else ' '}:{state.hand_count(p)}"
            for p in range(NUM_PLAYERS)) + "     (* = your team)",
    ]
    settled = []
    for hs, w in enumerate(state.set_winner):
        if w is None:
            continue
        tag = "ours" if w == mine else ("theirs" if w in (0, 1) else "null")
        settled.append(f"{HALF_SUIT_NAMES[hs]}={tag}")
    if settled:
        lines.append("  settled " + ", ".join(settled))
    return "\n".join(lines)


def _print_analysis(a, top: int = 6, cards: bool = False) -> None:
    ev = a.evaluation
    bar = "=" * min(30, int(abs(ev) * 6))
    side = "you" if ev >= 0 else "them"
    print(f"\n  EVAL {ev:+.2f} sets  ({side} {bar})")
    if a.notes:
        for n in a.notes:
            print(f"  ! {n}")
    if a.exact:
        print(f"  [solved] exact value {a.exact['value_for_me']:+.2f} - "
              f"{a.exact['note']}")
    if a.moves:
        print(f"\n  {'ask':>22s} {'P(land)':>8s} {'eval':>8s}  breakdown")
        for m in a.moves[:top]:
            terms = " ".join(f"{k}{v:+.2f}" for k, v in
                             sorted(m.terms.items(), key=lambda t: -abs(t[1]))[:3])
            mark = "*" if m.certain else " "
            print(f"  {mark}ask P{m.target} for {m.card_name:>3s}      "
                  f"{m.p_success:8.3f} {m.eval_expected:+8.2f}  {terms}")
    if a.principal_variation:
        pv = " -> ".join(f"P{x['target']}:{x['card_name']}"
                         for x in a.principal_variation)
        print(f"\n  plan (if each lands): {pv}")
    live = [c for c in a.claims if c.p_declaration_exact > 0.001]
    if live:
        print("\n  declarations:")
        for c in live[:4]:
            d = ("  ".join(f"{card_name(h)}=P{p}" for h, p in
                           zip(half_suit_cards(c.half_suit), c.declaration))
                 if c.declaration else "-")
            print(f"    {c.half_suit_name:22s} team {c.p_team_holds_all:5.3f} "
                  f"exact {c.p_declaration_exact:5.3f}  {c.verdict}")
            if c.p_declaration_exact > 0.2:
                print(f"      {d}")
    if a.deadlock and a.deadlock.get("summary"):
        print(f"\n  {a.deadlock['summary']}")
    if cards:
        print("\n  card table  P(holder)   " + "  ".join(
            f"P{p}" for p in range(NUM_PLAYERS)))
        for row in a.card_table:
            if row["mine"]:
                continue
            probs = "  ".join(f"{x:.2f}" for x in row["probs"])
            flag = " <-" if row["certain"] else ""
            print(f"    {row['name']:>3s}  {probs}{flag}")


# ---------------------------------------------------------------------------
# play
# ---------------------------------------------------------------------------

HELP = """
  commands
    ask <player> <card>       e.g.  ask 1 7S      (or: a 1 7s)
    claim <half-suit> p p p p p p   e.g.  claim 3 0 0 2 4 4 0
    pass <teammate>
    hint                      what the engine would do, and why
    cards                     the engine's posterior over every hidden card
    auto                      let the engine take this turn for you
    board                     redraw
    help / quit
  half-suits: """ + ", ".join(f"{i}={n}" for i, n in enumerate(HALF_SUIT_NAMES))


def cmd_play(args) -> int:
    rules = RuleConfig(variant=args.variant, claims_any_time=args.any_time_claims)
    seat = args.seat
    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    state = GameState.deal(rules, seed=seed)
    from .analyse import Analyser
    from .hsvalue import HalfSuitValue
    from .registry4 import make_agent

    vm = _load_value(args.value_model)
    bots = {}
    rng = random.Random(seed ^ 0x9E3779B9)
    for p in range(NUM_PLAYERS):
        if p == seat:
            continue
        bots[p] = make_agent((CHAMPION[0], dict(CHAMPION[1])))
        bots[p].begin_game(p, rules, rng.getrandbits(64))
    me = Analyser(rules, seat, value_model=vm, gamma=args.gamma,
                  n_draws=args.draws, seed=rng.getrandbits(31))
    # 'auto' plays the real policy - claim rule and exact endgame included -
    # rather than the analyser's move ranking, which knows nothing about when
    # to declare.
    helper = make_agent((CHAMPION[0], dict(CHAMPION[1])))
    helper.begin_game(seat, rules, rng.getrandbits(64))

    print(f"\nFishBot v0.4 - you are seat {seat} (team "
          f"{team_of(seat)}, with P{teammates(seat)[1]} and P{teammates(seat)[2]})")
    print(f"deal seed {seed}.  'help' for commands.\n")

    n = 0
    while not state.is_terminal and n < args.max_actions:
        p = state.turn
        obs = Observation.from_state(state, p)
        if p != seat:
            action = bots[p].act(obs)
            ev = state.apply(p, action)
            print("  " + _event_str(ev))
            n += 1
            continue
        print("\n" + "-" * 68)
        print(_board(state, seat))
        print("\n  your hand:")
        print(_hand_str(state.hands[seat]))
        if state.history:
            print("\n  recent:")
            for e in state.history[-4:]:
                print("    " + _event_str(e))
        while True:
            try:
                raw = input("\n  your move > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  (stopping)")
                return 0
            if not raw:
                continue
            act, err = _parse(raw, state, seat, me, obs, args, helper)
            if act == "handled":
                continue
            if err:
                print(f"  ! {err}")
                continue
            try:
                ev = state.apply(seat, act)
            except Exception as e:
                print(f"  ! illegal: {e}")
                continue
            print("  " + _event_str(ev))
            n += 1
            break
    a, b, nulls = state.scores()
    mine = a if team_of(seat) == 0 else b
    theirs = b if team_of(seat) == 0 else a
    print("\n" + "=" * 68)
    print(f"  FINAL   you {mine} - {theirs}   (nulled {nulls})   "
          f"{'YOU WIN' if mine > theirs else 'you lose' if theirs > mine else 'tie'}")
    return 0


def _parse(raw, state, seat, analyser, obs, args, helper=None):
    """Returns (action, error) or ('handled', None) for informational commands."""
    parts = raw.split()
    c = parts[0].lower()
    if c in ("help", "h", "?"):
        print(HELP)
        return "handled", None
    if c in ("quit", "q", "exit"):
        raise SystemExit(0)
    if c == "board":
        print(_board(state, seat))
        print(_hand_str(state.hands[seat]))
        return "handled", None
    if c in ("hint", "eval", "e"):
        _print_analysis(analyser.analyse(obs), top=args.top)
        return "handled", None
    if c in ("cards", "table"):
        _print_analysis(analyser.analyse(obs), top=0, cards=True)
        return "handled", None
    if c == "auto":
        if helper is None:
            return None, "no engine attached"
        try:
            act = helper.act(obs)
        except Exception as e:
            return None, f"engine found no move: {e}"
        if isinstance(act, Ask):
            print(f"  engine plays: ask P{act.target} for {card_name(act.card)}")
        elif isinstance(act, Claim):
            print(f"  engine declares {HALF_SUIT_NAMES[act.half_suit]}")
        return act, None
    if c in ("ask", "a"):
        if len(parts) != 3:
            return None, "usage: ask <player> <card>   e.g. ask 1 7S"
        try:
            t = int(parts[1])
        except ValueError:
            return None, "player must be 0-5"
        try:
            cid = card_id(parts[2])
        except KeyError:
            return None, f"unknown card {parts[2]!r}"
        return Ask(t, cid), None
    if c in ("claim", "declare", "c"):
        if len(parts) != 8:
            return None, ("usage: claim <half-suit 0-8> then six player ids, "
                          "in card order")
        try:
            hs = int(parts[1])
            who = tuple(int(x) for x in parts[2:])
        except ValueError:
            return None, "half-suit and holders must be integers"
        return Claim(hs, who), None
    if c in ("pass", "p"):
        if len(parts) != 2:
            return None, "usage: pass <teammate>"
        return Pass(int(parts[1])), None
    return None, f"unknown command {c!r} - try 'help'"


def _load_value(path):
    from .hsvalue import HalfSuitValue
    p = Path(path) if path else ROOT / "checkpoints" / "hsvalue_v1.json"
    try:
        return HalfSuitValue.load(p)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# watch / analyse / sim / duel / bench
# ---------------------------------------------------------------------------

def cmd_watch(args) -> int:
    rules = RuleConfig(variant=args.variant)
    from .analyse import Analyser
    from .registry4 import make_agent
    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    state = GameState.deal(rules, seed=seed)
    rng = random.Random(seed ^ 0x5DEECE66D)
    bots = [make_agent((CHAMPION[0], dict(CHAMPION[1])))
            for _ in range(NUM_PLAYERS)]
    for p, b in enumerate(bots):
        b.begin_game(p, rules, rng.getrandbits(64))
    an = (Analyser(rules, args.seat, value_model=_load_value(args.value_model),
                   seed=7) if args.explain else None)
    print(f"deal seed {seed}\n")
    n = 0
    while not state.is_terminal and n < args.max_actions:
        p = state.turn
        obs = Observation.from_state(state, p)
        if an is not None and p == args.seat:
            _print_analysis(an.analyse(obs), top=3)
        ev = state.apply(p, bots[p].act(obs))
        print(f"  {n:3d}  " + _event_str(ev))
        n += 1
    a, b, nulls = state.scores()
    print(f"\n  FINAL  team0 {a} - team1 {b}  (nulled {nulls}) in {n} actions")
    return 0


def cmd_analyse(args) -> int:
    """Play a game forward N actions, then analyse the position."""
    rules = RuleConfig(variant=args.variant)
    from .analyse import Analyser
    from .registry4 import make_agent
    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    state = GameState.deal(rules, seed=seed)
    rng = random.Random(seed ^ 0x2545F491)
    bots = [make_agent((CHAMPION[0], dict(CHAMPION[1])))
            for _ in range(NUM_PLAYERS)]
    for p, b in enumerate(bots):
        b.begin_game(p, rules, rng.getrandbits(64))
    for _ in range(args.ply):
        if state.is_terminal:
            break
        p = state.turn
        state.apply(p, bots[p].act(Observation.from_state(state, p)))
    seat = args.seat if args.seat is not None else state.turn
    an = Analyser(rules, seat, value_model=_load_value(args.value_model),
                  gamma=args.gamma, n_draws=args.draws, seed=11)
    obs = Observation.from_state(state, seat)
    print(f"\ndeal seed {seed}, after {len(state.history)} actions, "
          f"analysing seat {seat} (turn: P{state.turn})")
    print(_board(state, seat))
    print("\n  hand:")
    print(_hand_str(state.hands[seat]))
    _print_analysis(an.analyse(obs), top=args.top, cards=args.cards)
    return 0


def cmd_sim(args) -> int:
    from .match import play_capped
    from .registry4 import make_agent
    rules = RuleConfig(variant=args.variant,
                       claims_any_time=args.any_time_claims)
    specs = _lineup(args)
    t0 = time.time()
    diffs, nulls, acts, timeouts = [], 0, 0, 0
    for g in range(args.games):
        rng = random.Random(args.seed + g if args.seed is not None else None)
        deck = list(range(54 if args.variant == "54" else 48))
        random.Random((args.seed or 0) + g).shuffle(deck)
        bots = [make_agent(specs[p]) for p in range(NUM_PLAYERS)]
        st, to = play_capped(bots, rules, deck, (args.seed or 0) * 7919 + g)
        a, b, n = st.scores()
        diffs.append(a - b)
        nulls += n
        acts += len(st.history)
        timeouts += int(to)
        if args.verbose:
            print(f"  game {g+1}: team0 {a} - team1 {b} (null {n}) "
                  f"{len(st.history)} actions")
    import statistics
    dt = time.time() - t0
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    wins = sum(1 for d in diffs if d > 0)
    ties = sum(1 for d in diffs if d == 0)
    print(f"\n  {args.games} games in {dt:.1f}s ({dt/args.games:.2f} s/game)")
    print(f"  team0 lineup: {specs[0][0]}{specs[0][1]}")
    print(f"  team1 lineup: {specs[1][0]}{specs[1][1]}")
    print(f"  team0 wins {wins}, ties {ties}, losses {args.games-wins-ties}")
    print(f"  set differential  mean {mean:+.3f}  sd {sd:.3f}"
          f"  se {sd/max(1,len(diffs))**0.5:.3f}")
    print(f"  nulls/game {nulls/args.games:.3f}   actions/game "
          f"{acts/args.games:.1f}   timeouts {timeouts}")
    return 0


def cmd_duel(args) -> int:
    from .match import play_matchup
    x = _spec(args.x)
    y = _spec(args.y)
    rules = RuleConfig(variant=args.variant,
                       claims_any_time=args.any_time_claims)
    res = play_matchup(x, y, n_deals=args.games, n_jobs=args.jobs, rules=rules,
                       base_seed=args.seed or 0, agent_seed_base=(args.seed or 0) + 1,
                       progress=True)
    print("\n  " + res.summary())
    m, lo, hi = res.diff_ci()
    sd = 3.869
    mde = 2.8016 * sd / max(1, res.n) ** 0.5
    print(f"  minimum effect this run could resolve at 80% power: "
          f"{mde:.2f} sets/pair")
    if lo > 0:
        print("  VERDICT: X is stronger.")
    elif hi < 0:
        print("  VERDICT: Y is stronger.")
    else:
        print("  VERDICT: unresolved at this sample size.")
    return 0


def cmd_bench(args) -> int:
    from .match import play_capped
    from .registry4 import make_agent
    rules = RuleConfig()
    for label, spec in (("v0.4 champion", CHAMPION),
                        ("v0.4, model off", ("fishbot4", {})),
                        ("v0.3 champion", ("tuned", {"w_turn": 0.6,
                                                     "w_scarce": 0.2}))):
        t0 = time.time()
        acts = 0
        for i in range(args.games):
            bots = [make_agent((spec[0], dict(spec[1])))
                    for _ in range(NUM_PLAYERS)]
            deck = list(range(54))
            random.Random(4242 + i).shuffle(deck)
            st, _ = play_capped(bots, rules, deck, 999 + i)
            acts += len(st.history)
        dt = time.time() - t0
        print(f"  {label:18s} {dt/args.games:6.2f} s/game   "
              f"{dt/max(1,acts)*1000:6.2f} ms/action")
    return 0


def _public_tunnel(port: int) -> None:
    r"""Ask a public relay for a URL that forwards to this machine.

    Both relays here speak plain SSH remote-forwarding and need no account and
    nothing installed, which matters because the alternative is asking somebody
    to sign up for a tunnelling service before they can deal a hand of cards.
    The trade is that the URL is temporary and the relay sees the traffic.
    """
    import re
    import subprocess
    import threading

    common = ["ssh", "-o", "StrictHostKeyChecking=no",
              "-o", "ServerAliveInterval=30", "-R", f"80:localhost:{port}"]
    relays = [common + ["serveo.net"], common + ["nokey@localhost.run"]]
    pat = re.compile(r"https://[\w.-]+\.(?:serveo\.net|lhr\.life|localhost\.run)")

    def run():
        for cmd in relays:
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        bufsize=1)
            except FileNotFoundError:
                print("  [public] no ssh on PATH; skipping the tunnel")
                return
            shown = False
            for line in proc.stdout:
                m = pat.search(line)
                if m and not shown:
                    shown = True
                    print(f"\n  public link  ->  {m.group(0)}\n"
                          "  anyone with that link can join a table.\n")
            proc.wait()
            if shown:
                return          # the relay worked then dropped; do not retry
            print(f"  [public] {cmd[-1]} gave no URL, trying the next relay")
        print("  [public] no relay answered; the LAN address above still works")

    threading.Thread(target=run, daemon=True).start()


def cmd_serve(args) -> int:
    from .web.server import serve
    if args.public:
        print("\n  --public puts this table on the open internet through a"
              "\n  third-party relay. There is no password on it: anyone"
              "\n  with the link can sit down. Ctrl-C when you are done.\n")
        _public_tunnel(args.port)
    return serve(args.host, args.port)


def cmd_perpetual(args) -> int:
    from scripts4.perpetual_study import main as study
    return study(args.games, args.jobs)


def _spec(text: str):
    import json
    if text.startswith("["):
        n, kw = json.loads(text)
        return (n, kw)
    if ":" in text:
        n, kw = text.split(":", 1)
        return (n, json.loads(kw))
    return (text, {})


def _lineup(args):
    a = _spec(args.team0)
    b = _spec(args.team1)
    return [a if p % 2 == 0 else b for p in range(NUM_PLAYERS)]


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fish4", description="FishBot v0.4: play, analyse, simulate.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--variant", default="54", choices=("48", "54"))
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--any-time-claims", action="store_true",
                       help="allow declaring outside your own turn, as the "
                            "printed rules do")
        return p

    def engine(p):
        p.add_argument("--gamma", type=float, default=0.35,
                       help="opponent-model strength (0 disables it)")
        p.add_argument("--draws", type=int, default=320)
        p.add_argument("--value-model", default=None)
        p.add_argument("--top", type=int, default=6)
        return p

    p = common(engine(sub.add_parser("play", help="play against the engine")))
    p.add_argument("--seat", type=int, default=0, choices=range(6))
    p.add_argument("--max-actions", type=int, default=4000)
    p.set_defaults(func=cmd_play)

    p = common(sub.add_parser("watch", help="watch the engine play itself"))
    p.add_argument("--seat", type=int, default=0)
    p.add_argument("--explain", action="store_true")
    p.add_argument("--value-model", default=None)
    p.add_argument("--max-actions", type=int, default=4000)
    p.set_defaults(func=cmd_watch)

    p = common(engine(sub.add_parser("analyse", help="analyse a position",
                                     aliases=["analyze"])))
    p.add_argument("--ply", type=int, default=40,
                   help="play this many actions first")
    p.add_argument("--seat", type=int, default=None)
    p.add_argument("--cards", action="store_true", help="show the card table")
    p.set_defaults(func=cmd_analyse)

    p = common(sub.add_parser("sim", help="run simulations"))
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--team0", default="fishbot4:{\"opponent_gamma\":0.35}")
    p.add_argument("--team1", default="tuned:{\"w_turn\":0.6,\"w_scarce\":0.2}")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_sim)

    p = common(sub.add_parser("duel", help="duplicate-deal head-to-head"))
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("-x", default="fishbot4:{\"opponent_gamma\":0.35}")
    p.add_argument("-y", default="tuned:{\"w_turn\":0.6,\"w_scarce\":0.2}")
    p.set_defaults(func=cmd_duel)

    p = sub.add_parser("bench", help="speed")
    p.add_argument("--games", type=int, default=3)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("perpetual", help="dead-position / deadlock study")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--jobs", type=int, default=4)
    p.set_defaults(func=cmd_perpetual)

    p = sub.add_parser("serve", help="browser UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--public", action="store_true",
                   help="also expose a temporary public URL via an ssh relay")
    p.set_defaults(func=cmd_serve)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
