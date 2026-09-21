"""One-shot decision entry point: run this engine from someone else's arbiter.

THE MIRROR IMAGE OF `external_v07/shim_decide.cpp`. That shim lets this project
drive dylann4500's C++ FishBot from Python. This module is the other direction:
it lets any external arbiter drive *this* engine, one decision at a time, over
a line protocol on stdin/stdout. Stateless by design -- the caller replays the
public log on every request, exactly as the existing shim requires of us, so
the two integrations are symmetric and neither side has to hold session state
for the other.

    python -m fish4.decide  <  request.txt

WHAT IT SEES, AND WHY THAT IS THE WHOLE POINT. It builds an `Observation`, and
an Observation is precisely this project's information boundary: the seat's own
hand plus the public event stream, and nothing else. `hand_counts` and
`set_winner` are DERIVED from the log rather than supplied, so a caller cannot
accidentally leak private state through them and cannot get them wrong.

CARDS ARE NAMED, NOT NUMBERED. The two engines number the deck differently and
`fish4/dylan_v07.py` bridges them by name. Doing the same here removes the id
mapping from the protocol: "TD", "JC", "RJ", "BJ" mean the same thing to both
projects, and a bijection built on names cannot silently rotate.

--------------------------------------------------------------------- protocol

Header lines, any order, before the first EV or DECIDE:

    RULES <n_half_suits> <opponent|void>   misdeclaration outcome; default 9 opponent
    SEAT  <0..5>
    HAND  <card> <card> ...                the seat's CURRENT hand, by name
    SEED  <uint64>                         optional; fixes sampling, default 1
    TURN  <0..5>                           optional; defaults to SEAT

Then zero or more public events, oldest first:

    EV ASK  <asker> <target> <card> <0|1>
    EV DECL <claimer> <half_suit> <o0> ... <o5>   revealed owners of the six cards
    EV PASS <player> <teammate>

Then exactly one:

    DECIDE

Reply, one line on stdout:

    ASK  <target> <card>
    DECL <half_suit> <o0> ... <o5>
    PASS <teammate>

Anything malformed exits non-zero with a message on stderr rather than
guessing, because a bridge that guesses is a bridge that silently plays a
different game -- which is the failure `external_v07/README.md` records this
project already making once, in the other direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fish.cards import (CARD_NAMES, NUM_PLAYERS, cards_per_player,
                        half_suit_cards, half_suit_of, num_half_suits)
from fish.engine import Ask, AskEvent, Claim, ClaimEvent, Pass, PassEvent
from fish.observation import Observation
from fish.rules import RuleConfig

_BY_NAME = {n.upper(): i for i, n in enumerate(CARD_NAMES)}


class ProtocolError(Exception):
    """The request does not parse. Never recovered from by guessing."""


def _card(tok: str) -> int:
    i = _BY_NAME.get(tok.strip().upper())
    if i is None:
        raise ProtocolError(f"unknown card {tok!r}; names are like 2S TD JC RJ")
    return i


def _seat(tok: str) -> int:
    try:
        p = int(tok)
    except ValueError:
        raise ProtocolError(f"seat {tok!r} is not an integer")
    if not 0 <= p < NUM_PLAYERS:
        raise ProtocolError(f"seat {p} out of range 0..{NUM_PLAYERS - 1}")
    return p


class Request:
    """A parsed request, and the Observation it implies."""

    def __init__(self) -> None:
        self.n_hs = 9
        self.outcome = "opponent"
        self.seat: int | None = None
        self.hand: list[int] | None = None
        self.seed = 1
        self.turn: int | None = None
        self.events: list = []
        self.decide = False

    # -- parsing ------------------------------------------------------------

    def feed(self, line: str) -> None:
        tok = line.split()
        if not tok or tok[0].startswith("#"):
            return
        tag, rest = tok[0].upper(), tok[1:]
        if tag == "RULES":
            if rest:
                self.n_hs = int(rest[0])
            if len(rest) > 1:
                if rest[1] not in ("opponent", "void"):
                    raise ProtocolError(
                        f"misdeclaration outcome {rest[1]!r} must be "
                        f"'opponent' or 'void'")
                self.outcome = rest[1]
        elif tag == "SEAT":
            self.seat = _seat(rest[0])
        elif tag == "HAND":
            self.hand = [_card(c) for c in rest]
        elif tag == "SEED":
            self.seed = int(rest[0])
        elif tag == "TURN":
            self.turn = _seat(rest[0])
        elif tag == "EV":
            self.events.append(self._event(rest))
        elif tag == "DECIDE":
            self.decide = True
        else:
            raise ProtocolError(f"unknown line tag {tag!r}")

    def _event(self, rest: list[str]):
        if not rest:
            raise ProtocolError("EV with no kind")
        kind, a = rest[0].upper(), rest[1:]
        if kind == "ASK":
            if len(a) != 4:
                raise ProtocolError("EV ASK needs <asker> <target> <card> <0|1>")
            return AskEvent(asker=_seat(a[0]), target=_seat(a[1]),
                            card=_card(a[2]), success=a[3] not in ("0", "false"))
        if kind == "DECL":
            if len(a) != 8:
                raise ProtocolError(
                    "EV DECL needs <claimer> <half_suit> and six owners")
            claimer, hs = _seat(a[0]), int(a[1])
            owners = tuple(_seat(x) for x in a[2:])
            # The winner is implied by the revealed owners and the rule, so the
            # caller does not send it and cannot contradict it.
            team = claimer % 2
            allours = all(o % 2 == team for o in owners)
            winner = team if allours else (1 - team)
            return ClaimEvent(claimer=claimer, half_suit=hs,
                              declared=owners, revealed=owners, winner=winner)
        if kind == "PASS":
            if len(a) != 2:
                raise ProtocolError("EV PASS needs <player> <teammate>")
            return PassEvent(player=_seat(a[0]), teammate=_seat(a[1]))
        raise ProtocolError(f"unknown event kind {kind!r}")

    # -- the Observation ----------------------------------------------------

    def observation(self) -> Observation:
        if self.seat is None:
            raise ProtocolError("no SEAT")
        if self.hand is None:
            raise ProtocolError("no HAND")
        rules = RuleConfig(wrong_distribution_outcome=self.outcome)
        if num_half_suits(rules.variant) != self.n_hs:
            raise ProtocolError(
                f"RULES asks for {self.n_hs} half-suits; this build plays "
                f"{num_half_suits(rules.variant)}")
        hand = 0
        for c in self.hand:
            hand |= 1 << c

        # Derived, never supplied. See the module docstring: a caller that
        # cannot state these cannot get them wrong, and cannot use them to
        # smuggle private information in.
        per = cards_per_player(rules.variant)
        counts = [per] * NUM_PLAYERS
        winners: list[int | None] = [None] * self.n_hs
        for ev in self.events:
            if isinstance(ev, AskEvent) and ev.success:
                counts[ev.asker] += 1
                counts[ev.target] -= 1
            elif isinstance(ev, ClaimEvent):
                if not 0 <= ev.half_suit < self.n_hs:
                    raise ProtocolError(f"half-suit {ev.half_suit} out of range")
                if winners[ev.half_suit] is not None:
                    raise ProtocolError(
                        f"half-suit {ev.half_suit} declared twice")
                winners[ev.half_suit] = ev.winner
                for o in ev.revealed:
                    counts[o] -= 1
        if any(c < 0 for c in counts):
            raise ProtocolError(
                f"the event stream implies a negative hand: {counts}. "
                f"Events must be in order, oldest first.")
        if counts[self.seat] != len(self.hand):
            raise ProtocolError(
                f"HAND has {len(self.hand)} cards but the event stream implies "
                f"{counts[self.seat]} for seat {self.seat}. One of the two is "
                f"wrong, and guessing which would play a different game.")
        return Observation(
            player=self.seat, rules=rules, hand=hand,
            turn=self.seat if self.turn is None else self.turn,
            hand_counts=tuple(counts), set_winner=tuple(winners),
            history=tuple(self.events))


def _format(action) -> str:
    if isinstance(action, Ask):
        return f"ASK {action.target} {CARD_NAMES[action.card]}"
    if isinstance(action, Claim):
        return "DECL " + str(action.half_suit) + " " + " ".join(
            str(o) for o in action.assignment)
    if isinstance(action, Pass):
        return f"PASS {action.teammate}"
    raise ProtocolError(f"engine returned {action!r}, which has no wire form")


def decide(text: str) -> str:
    req = Request()
    for n, line in enumerate(text.splitlines(), 1):
        try:
            req.feed(line)
        except ProtocolError as e:
            raise ProtocolError(f"line {n}: {e}") from None
        except (ValueError, IndexError) as e:
            raise ProtocolError(f"line {n}: {e}") from None
    if not req.decide:
        raise ProtocolError("no DECIDE line; nothing was asked")
    obs = req.observation()

    from fish.beliefs import BeliefContradiction
    from fish4.registry4 import V06_DEPLOYED, make_agent
    agent = make_agent(V06_DEPLOYED)
    agent.begin_game(obs.player, obs.rules, req.seed)
    try:
        return _format(agent.act(obs))
    except BeliefContradiction as e:
        # The belief tracker proved the event stream cannot describe any real
        # deal. That is the caller's bug, not ours, and it deserves a sentence
        # rather than a Python traceback: an integrator seeing a stack trace
        # from someone else's engine learns nothing about their own protocol
        # mistake. The usual cause is an ask whose asker did not hold a card of
        # that half-suit, or events out of order.
        raise ProtocolError(
            f"the event stream is not consistent with any deal ({e}). "
            f"Most often: an ASK whose asker held no card of that half-suit, "
            f"a card given away twice, or events out of chronological order."
        ) from None


def main(argv=None) -> int:
    try:
        print(decide(sys.stdin.read()))
    except ProtocolError as e:
        print(f"protocol error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
