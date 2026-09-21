"""KRAKEN v1.1 as a language-neutral decision service.

This is the mirror image of the bridge this project built to run dylann4500's
FishBot v0.7 inside our engine: it lets ANY host -- a C++ engine, a
TypeScript web app, another Python program -- ask this bot for a move without
importing it, sharing a process model, or knowing anything about how it
thinks.

PROTOCOL. One JSON object per line on stdin, one JSON object per line on
stdout, flushed immediately. The process is a loop, so a host may keep it
alive for a whole match (fast) or send a single line and close the pipe
(simple). It holds no state between requests: every request carries the
whole public record, and the bot's belief is rebuilt from it. That is a
deliberate cost -- the host stays the single source of truth about the
game, and a lost or duplicated request cannot corrupt anything.

REQUEST
    {"seat": 0,
     "hand": ["2C", "3C", "AH"],
     "hand_counts": [3, 9, 9, 9, 9, 9],
     "set_winner": [null, null, 0, 1, null, null, null, null, null],
     "turn": 0,
     "history": [ ...events, oldest first... ],
     "rules": {"wrong_distribution_outcome": "opponent"}}

  seat          which seat we are deciding for, 0..5. Teams are by parity:
                {0,2,4} play {1,3,5}.
  hand          our own cards, as names (see CARD NAMES below).
  hand_counts   how many cards each seat holds now, all six, public.
  set_winner    per half-suit: 0, 1, or null if unresolved. Nine entries.
  turn          whose turn it is; must equal `seat` except for a forced
                pass request.
  history       every public event so far, oldest first:
                  {"t":"ask","asker":0,"target":1,"card":"2H","success":true}
                  {"t":"claim","claimer":0,"half_suit":3,
                   "declared":[0,0,2,4,4,4],"revealed":[0,0,2,4,4,4],
                   "winner":0}
                  {"t":"pass","player":0,"teammate":2}
  rules         optional; defaults to this project's baseline. The only
                field that changes play is wrong_distribution_outcome,
                "opponent" (a misdeclared set goes to the other team --
                standard, and what v0.7's engine does) or "null" (the
                legacy void variant).

RESPONSE
    {"action": {"type": "ask", "target": 3, "card": "2H"}}
    {"action": {"type": "declare", "half_suit": 3,
                "assignment": [0, 0, 2, 4, 4, 4]}}
    {"action": {"type": "pass", "teammate": 2}}
    {"error": "..."}                 on a malformed or illegal request

  "declare" is this project's name for the move v0.7 calls a declaration and
  older code here calls a claim; they are the same move. `assignment[i]` is
  the seat declared to hold the i-th card of the half-suit, and every entry
  must be on the declarer's own team.

CARD NAMES. Rank then suit: "2C".."7C" and "9C".."AC" for clubs, likewise
D, H, S; "BJ" and "RJ" for the black and red jokers. Half-suit h contains
cards 6h..6h+5 in this project's ordering; ask for the mapping with
{"op": "cards"} and a full table comes back, so a host never has to
reimplement it:

    {"op": "cards"} -> {"cards": ["2C", ...54 names in index order...],
                        "half_suits": [{"index":0,"name":"Low Clubs",
                                        "cards":["2C",...]}, ...]}

DETERMINISM. A seed may be supplied per request ({"seed": 12345}); with the
same seed and the same public record the same move comes back, which is
what makes a host's replays reproducible. Omitted, a seed is derived from
the history length, so a match is still reproducible from its own log.

    python -m kraken.decide          # speaks the protocol on stdio
    python -m kraken.decide --self-test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fish.cards import (CARD_IDS, CARD_NAMES, HALF_SUIT_NAMES, NUM_PLAYERS,
                        card_name, half_suit_cards)
from fish.engine import Ask, AskEvent, Claim, ClaimEvent, Pass, PassEvent
from fish.observation import Observation
from fish.rules import RuleConfig

#: The one shipped configuration. Named here so a host can print what it is
#: playing against, and so there is exactly one answer to "which bot is
#: this?". Mirrors fish4.registry4.V06_DEPLOYED; the self-test asserts it.
SPEC = {"opponent_gamma": 0.35, "n_draws": 480, "w_lookahead": 0.25,
        "lookahead_depth": 3, "lookahead_beam": 4, "endgame_m": 0,
        "claim_forced_exhaustive": 1}
VERSION = "1.1"


def _card_id(x) -> int:
    """Cards are NAMES ("2C"), never integers. This is deliberate.

    An earlier version of this file also accepted an integer and read it as
    an index into OUR card ordering. That is a silent catastrophe across a
    bridge: v0.7 numbers its cards set*6+idx over a DIFFERENT permutation of
    the sets, so a host passing its own integers would have had this bot
    play a scrambled hand, legally, badly, and with no error anywhere. It
    would look exactly like a weak bot rather than a broken integration.
    Names are unambiguous between any two implementations of a 54-card deck,
    so names are the contract.
    """
    if isinstance(x, bool) or isinstance(x, int):
        raise ValueError(
            f"card must be a NAME like '2C' or 'RJ', not the integer {x}: "
            f"card indices are not portable between engines (ask "
            f'{{"op":"cards"}} for our ordering if you need it)')
    try:
        return CARD_IDS[str(x).strip().upper()]
    except KeyError:
        raise ValueError(f"unknown card name: {x!r}")


def _seat(x, what="seat") -> int:
    v = int(x)
    if not 0 <= v < NUM_PLAYERS:
        raise ValueError(f"{what} out of range: {v}")
    return v


def _event(d):
    t = str(d.get("t") or d.get("type") or "").lower()
    if t == "ask":
        return AskEvent(_seat(d["asker"], "asker"), _seat(d["target"], "target"),
                        _card_id(d["card"]), bool(d["success"]))
    if t in ("claim", "declare", "declaration"):
        declared = tuple(_seat(p, "declared holder") for p in d["declared"])
        revealed = tuple(_seat(p, "revealed holder")
                         for p in d.get("revealed", d["declared"]))
        if len(declared) != 6 or len(revealed) != 6:
            raise ValueError("a declaration names exactly six holders")
        w = d.get("winner")
        return ClaimEvent(_seat(d["claimer"], "claimer"),
                          int(d["half_suit"]), declared, revealed,
                          -1 if w is None else int(w))
    if t == "pass":
        return PassEvent(_seat(d["player"], "player"),
                         _seat(d["teammate"], "teammate"))
    raise ValueError(f"unknown event type: {t!r}")


def _observation(req) -> Observation:
    seat = _seat(req["seat"])
    rd = dict(req.get("rules") or {})
    rd.setdefault("wrong_distribution_outcome", "opponent")
    rd.pop("starting_player", None)
    rules = RuleConfig(**rd)
    hand = 0
    for c in req.get("hand", []):
        hand |= 1 << _card_id(c)
    counts = tuple(int(x) for x in req["hand_counts"])
    if len(counts) != NUM_PLAYERS:
        raise ValueError("hand_counts needs one entry per seat")
    if bin(hand).count("1") != counts[seat]:
        raise ValueError(
            f"hand has {bin(hand).count('1')} cards but hand_counts says "
            f"{counts[seat]} for seat {seat}")
    sw = tuple(None if w is None else int(w) for w in req["set_winner"])
    history = tuple(_event(e) for e in req.get("history", []))
    return Observation(player=seat, rules=rules, hand=hand,
                       turn=_seat(req.get("turn", seat), "turn"),
                       hand_counts=counts, set_winner=sw, history=history)


def _encode(action) -> dict:
    if isinstance(action, Ask):
        return {"type": "ask", "target": action.target,
                "card": card_name(action.card)}
    if isinstance(action, Claim):
        return {"type": "declare", "half_suit": action.half_suit,
                "assignment": list(action.assignment)}
    if isinstance(action, Pass):
        return {"type": "pass", "teammate": action.teammate}
    raise ValueError(f"unencodable action: {action!r}")


def _card_table() -> dict:
    return {"cards": list(CARD_NAMES),
            "half_suits": [
                {"index": h, "name": HALF_SUIT_NAMES[h],
                 "cards": [card_name(c) for c in half_suit_cards(h)]}
                for h in range(len(HALF_SUIT_NAMES))]}


def _offturn(req: dict) -> dict:
    """Does this seat have a declaration it can make RIGHT NOW, off-turn?

    Hosts whose rules allow declaring at any moment (v0.7's engine does, by
    default: ``outOfTurnDeclare = true``) must poll every seat, not only the
    turn holder. Skipping this is not a small omission. Measured over 240
    games against v0.7 on identical deals
    (``scripts4/dialect_gap.py``, ``results/dialect_gap.json``): with both
    sides declaring off-turn our margin is +2.375 sets/game; with only their
    side allowed to -- which is what a host gets if it never polls us
    off-turn -- it falls to +1.575. That 0.8-set swing is a rule we were not
    playing, not a weakness in the bot.

    Only CERTAIN declarations are offered here: the answer is a declaration
    exactly when the public record alone pins every card of a half-suit to a
    named teammate. A speculative off-turn declaration would be a gamble
    with a full set under the award rule, and a seat that is merely
    confident can wait for its turn and use the full policy.

    Returns {"action": ...} or {"action": null} -- never an error for the
    ordinary case of having nothing to say.
    """
    obs = _observation(req)
    from fish.beliefs import BeliefState
    from fish4.match import _deduced_claim

    class _Seat:
        pass

    seat = _Seat()
    seat.bel = BeliefState(obs.rules, observer=obs.player)
    seat.bel.update(obs)
    claim = _deduced_claim(seat, obs.player, obs)
    return {"action": _encode(claim) if claim is not None else None}


def decide(req: dict) -> dict:
    """One request in, one response out. Never raises: errors are data."""
    try:
        op = str(req.get("op", "decide")).lower()
        if op == "cards":
            return _card_table()
        if op in ("version", "hello"):
            return {"bot": f"KRAKEN v{VERSION}", "spec": SPEC,
                    "protocol": 2,
                    "ops": ["decide", "offturn", "cards", "version"],
                    "declares_off_turn": True}
        if op in ("offturn", "offturn_declare"):
            return _offturn(req)
        if op != "decide":
            return {"error": f"unknown op: {op!r}"}

        obs = _observation(req)
        seed = req.get("seed")
        if seed is None:
            # Reproducible from the log alone when the host does not care to
            # manage seeds: same public record, same move.
            seed = 1_000_003 + 7919 * len(obs.history) + 31 * obs.player
        from fish4.registry4 import make_agent
        agent = make_agent(("fishbot4", dict(SPEC)))
        agent.begin_game(obs.player, obs.rules, int(seed))
        action = agent.act(obs)
        out = {"action": _encode(action)}
        if req.get("explain"):
            out["explain"] = {
                "bot": f"KRAKEN v{VERSION}",
                "unresolved_half_suits": sum(1 for w in obs.set_winner
                                             if w is None),
                "seed": int(seed)}
        return out
    except Exception as e:                    # noqa: BLE001 - errors are data
        return {"error": f"{type(e).__name__}: {e}"}


def _self_test() -> int:
    """Play a whole game through the protocol and check it against the
    in-process agent.

    The point is not that the bot is good; it is that this file is a FAITHFUL
    door to it. A host integrating through JSON must get the same moves it
    would get by importing the agent, or the integration is a different bot
    wearing the same name.
    """
    from fish.engine import GameState
    from fish4.registry4 import make_agent

    rules = RuleConfig(wrong_distribution_outcome="opponent")
    assert SPEC == dict(__import__(
        "fish4.registry4", fromlist=["V06_DEPLOYED"]).V06_DEPLOYED[1]), (
        "SPEC has drifted from fish4.registry4.V06_DEPLOYED")

    mismatches = checked = 0
    for game in range(3):
        st = GameState.deal(rules, seed=770_000 + game)
        direct = [make_agent(("fishbot4", dict(SPEC))) for _ in range(6)]
        for p, a in enumerate(direct):
            a.begin_game(p, rules, 0)
        for _ in range(600):
            if st.is_terminal:
                break
            seat = st.turn
            obs = Observation.from_state(st, seat)
            seed = 4242 + 13 * len(obs.history)

            # through the protocol
            req = {"seat": seat, "turn": seat, "seed": seed,
                   "hand": [card_name(c) for c in range(54)
                            if obs.hand >> c & 1],
                   "hand_counts": list(obs.hand_counts),
                   "set_winner": list(obs.set_winner),
                   "rules": {"wrong_distribution_outcome": "opponent"},
                   "history": []}
            for ev in obs.history:
                if isinstance(ev, AskEvent):
                    req["history"].append(
                        {"t": "ask", "asker": ev.asker, "target": ev.target,
                         "card": card_name(ev.card), "success": ev.success})
                elif isinstance(ev, ClaimEvent):
                    req["history"].append(
                        {"t": "claim", "claimer": ev.claimer,
                         "half_suit": ev.half_suit,
                         "declared": list(ev.declared),
                         "revealed": list(ev.revealed),
                         "winner": None if ev.winner < 0 else ev.winner})
                else:
                    req["history"].append(
                        {"t": "pass", "player": ev.player,
                         "teammate": ev.teammate})
            resp = decide(req)
            assert "action" in resp, f"protocol error: {resp}"

            # and directly, same seed
            fresh = make_agent(("fishbot4", dict(SPEC)))
            fresh.begin_game(seat, rules, seed)
            want = _encode(fresh.act(obs))

            checked += 1
            if resp["action"] != want:
                mismatches += 1
                print(f"  MISMATCH game {game} ply {len(obs.history)}: "
                      f"{resp['action']} != {want}")
            st.apply(seat, direct[seat].act(obs))
    print(f"{checked} decisions compared, {mismatches} mismatches")
    tbl = decide({"op": "cards"})
    assert len(tbl["cards"]) == 54 and len(tbl["half_suits"]) == 9
    assert decide({"op": "version"})["spec"] == SPEC
    bad = decide({"op": "decide", "seat": 9})
    assert "error" in bad, "a malformed request must return an error"

    # Integer card ids must be REFUSED, not silently misread: the whole
    # failure mode this guards against is a host passing its own ordering.
    ints = decide({"seat": 0, "turn": 0, "hand": [0, 1],
                   "hand_counts": [2, 9, 9, 9, 9, 9],
                   "set_winner": [None] * 9, "history": []})
    assert "error" in ints and "not portable" in ints["error"], (
        "integer card ids must be refused loudly")

    # The off-turn channel must actually FIRE somewhere in real play, or a
    # host polling it gets nothing and the rule stays unplayed. Replay a
    # game and poll every non-turn seat at every ply.
    fired = polled = 0
    st = GameState.deal(rules, seed=771_001)
    agents = [make_agent(("fishbot4", dict(SPEC))) for _ in range(6)]
    for p, a in enumerate(agents):
        a.begin_game(p, rules, 5)
    for _ in range(600):
        if st.is_terminal:
            break
        for q in range(6):
            if q == st.turn:
                continue
            o = Observation.from_state(st, q)
            req = {"op": "offturn", "seat": q, "turn": st.turn,
                   "hand": [card_name(c) for c in range(54) if o.hand >> c & 1],
                   "hand_counts": list(o.hand_counts),
                   "set_winner": list(o.set_winner), "history": []}
            for ev in o.history:
                if isinstance(ev, AskEvent):
                    req["history"].append(
                        {"t": "ask", "asker": ev.asker, "target": ev.target,
                         "card": card_name(ev.card), "success": ev.success})
                elif isinstance(ev, ClaimEvent):
                    req["history"].append(
                        {"t": "claim", "claimer": ev.claimer,
                         "half_suit": ev.half_suit,
                         "declared": list(ev.declared),
                         "revealed": list(ev.revealed),
                         "winner": None if ev.winner < 0 else ev.winner})
                else:
                    req["history"].append(
                        {"t": "pass", "player": ev.player,
                         "teammate": ev.teammate})
            r = decide(req)
            assert "action" in r, f"off-turn poll errored: {r}"
            polled += 1
            if r["action"] is not None:
                fired += 1
                assert r["action"]["type"] == "declare"
                # and it must be TRUE: an off-turn declaration is offered
                # only when the public record pins the whole half-suit
                hs = r["action"]["half_suit"]
                for i, c in enumerate(half_suit_cards(hs)):
                    assert st.holder_of(c) == r["action"]["assignment"][i], (
                        "off-turn declaration offered a WRONG assignment")
        st.apply(st.turn, agents[st.turn].act(
            Observation.from_state(st, st.turn)))
    print(f"off-turn channel: {polled} polls, {fired} declarations offered, "
          f"all correct")
    assert fired > 0, ("the off-turn channel never fired in a whole game; "
                       "a host polling it would get nothing")
    print("card table, version, strict cards and error handling: ok")
    return 1 if mismatches else 0


def main() -> int:
    if "--self-test" in sys.argv:
        return _self_test()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            resp = {"error": f"bad JSON: {e}"}
        else:
            resp = decide(req)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
