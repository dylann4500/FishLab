"""Live coaching: advise a human during a real game of Fish.

The player tells us their seat and their dealt hand, then logs each public
event as it happens at the table (who asked whom for what, and whether they
got it; who claimed what and how it resolved). From that we rebuild exactly
the Observation that player legally has and recommend a move.

This works because of the fact the whole engine is built on: every card
movement in Literature is public. The player's own hand plus the public
record is enough to reconstruct their entire legal view, which is precisely
what ``Observation.reconstruct`` does. Nothing here needs the real deal, and
the advice is available to any human who tracks the same public information.

The session is a small state machine over the event log, so "undo" is just
dropping the last event and rebuilding.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .agents import AGENT_REGISTRY
from .beliefs import BeliefContradiction, BeliefState
from .cards import (CARD_NAMES, HALF_SUIT_NAMES, card_id, card_long_name,
                    card_name, cards_per_player, deck_size, half_suit_cards,
                    half_suit_mask, is_red, mask_to_cards, num_half_suits,
                    team_of)
from .engine import (Ask, AskEvent, Claim, ClaimEvent, Event, NULL_TEAM,
                     PassEvent)
from .observation import Observation
from .rules import RuleConfig


class CoachError(Exception):
    """A problem with what the user entered, phrased for a human."""


@dataclass
class CoachSession:
    seat: int
    initial_hand: int
    rules: RuleConfig
    events: list = field(default_factory=list)
    id: str = ""

    # -- construction ------------------------------------------------------

    @classmethod
    def create(cls, seat: int, hand_names: list[str], variant: str = "54",
               starting_player: int = 0, **rule_kwargs) -> "CoachSession":
        if not 0 <= seat < 6:
            raise CoachError("Seat must be between 0 and 5.")
        rules = RuleConfig(variant=variant, starting_player=starting_player,
                           **rule_kwargs)
        expected = cards_per_player(variant)
        ids = []
        for nm in hand_names:
            nm = (nm or "").strip()
            if not nm:
                continue
            try:
                ids.append(card_id(nm))
            except KeyError:
                raise CoachError(f"'{nm}' is not a card I recognise.")
        if len(set(ids)) != len(ids):
            raise CoachError("That hand lists the same card twice.")
        if len(ids) != expected:
            raise CoachError(
                f"A {variant}-card game deals {expected} cards each; "
                f"you entered {len(ids)}.")
        limit = deck_size(variant)
        bad = [card_name(c) for c in ids if c >= limit]
        if bad:
            raise CoachError(
                f"{', '.join(bad)} are not in the {variant}-card deck.")
        hand = 0
        for c in ids:
            hand |= 1 << c
        return cls(seat=seat, initial_hand=hand, rules=rules)

    # -- state -------------------------------------------------------------

    def observation(self) -> Observation:
        return Observation.reconstruct(self.rules, self.seat,
                                       self.initial_hand, tuple(self.events))

    def validate(self) -> None:
        """Check the log is self-consistent from the player's viewpoint."""
        bel = BeliefState(self.rules, observer=self.seat)
        bel.update(self.observation())

    # -- editing the log ---------------------------------------------------

    def add_ask(self, asker: int, target: int, card: str, success: bool):
        for p, label in ((asker, "asker"), (target, "target")):
            if not 0 <= p < 6:
                raise CoachError(f"The {label} must be a seat from 0 to 5.")
        if team_of(asker) == team_of(target):
            raise CoachError(
                f"P{asker} and P{target} are teammates; you cannot ask a "
                "teammate in Literature.")
        try:
            cid = card_id(card)
        except KeyError:
            raise CoachError(f"'{card}' is not a card I recognise.")
        if cid >= deck_size(self.rules.variant):
            raise CoachError(
                f"{card_name(cid)} is not in the {self.rules.variant}-card deck.")

        # Friendly checks for the mistakes people actually make when typing
        # a live game in. The belief engine would reject these anyway, but
        # it would say "card 12 has no possible owner", which helps nobody.
        obs = self.observation()
        held = bool(obs.hand & (1 << cid))
        name = card_long_name(cid)
        if target == self.seat:
            if held and not success:
                raise CoachError(
                    f"You are holding {name}, so you could not have said no "
                    f"to P{asker}. Did you mean 'Got it'?")
            if not held and success:
                raise CoachError(
                    f"You are not holding {name}, so you could not have given "
                    f"it to P{asker}.")
        if asker == self.seat:
            if held:
                raise CoachError(
                    f"You already hold {name}; asking for a card you hold is "
                    "not legal in Literature.")
            if not obs.hand & half_suit_mask(cid // 6):
                raise CoachError(
                    f"You hold no {HALF_SUIT_NAMES[cid // 6]}, so you may not "
                    f"ask for {name}.")
        if success and obs.hand_counts[target] == 0:
            raise CoachError(f"P{target} has no cards left to give.")

        self._push(AskEvent(asker, target, cid, bool(success)))

    def add_claim(self, claimer: int, half_suit: int, holders: list[int],
                  winner: Optional[int] = None, nulled: bool = False):
        """Record a resolved claim.

        ``holders`` is where each of the six cards ACTUALLY turned out to be,
        which becomes public when a claim resolves.

        ``winner`` should be given when known, because it cannot always be
        derived: if every card was within the claiming team but the claimer
        named the wrong teammate for one of them, the outcome depends on the
        table's misdeclaration rule -- under the baseline the OPPONENTS score
        it (pass ``winner=1 - claimer's team``), and under the legacy null
        variant it is voided (pass ``nulled=True`` or ``winner=-1``) -- and
        either looks identical from the holders alone. The ``winner=None``
        default credits the claiming team whenever all six were theirs, i.e.
        it records the claim as exact; if the split was actually misdeclared,
        say so explicitly. Getting this wrong would silently credit a set to
        a team that did not score it, and the scoreboard drives claim
        decisions later in the game.
        """
        if not 0 <= claimer < 6:
            raise CoachError("The claimer must be a seat from 0 to 5.")
        n_hs = num_half_suits(self.rules.variant)
        if not 0 <= half_suit < n_hs:
            raise CoachError(f"Half-suit must be between 0 and {n_hs - 1}.")
        if len(holders) != 6 or any(not 0 <= h < 6 for h in holders):
            raise CoachError("A claim reveals six holders, each a seat 0-5.")
        obs = self.observation()
        if obs.set_winner[half_suit] is not None:
            raise CoachError(
                f"{HALF_SUIT_NAMES[half_suit]} has already been claimed.")
        # Friendly check against the one thing we can verify directly: what
        # the player themself is holding. Otherwise the belief engine reports
        # "player 0 count infeasible", which tells nobody anything.
        base = half_suit * 6
        for i, h in enumerate(holders):
            card = base + i
            if h == self.seat and not obs.hand & (1 << card):
                raise CoachError(
                    f"You are not holding {card_long_name(card)}, so it "
                    "cannot have been revealed with you.")
            if h != self.seat and obs.hand & (1 << card):
                raise CoachError(
                    f"You are holding {card_long_name(card)}, so it cannot "
                    f"have been revealed with P{h}.")
        team = team_of(claimer)
        all_in_team = all(team_of(h) == team for h in holders)
        if nulled:
            if not all_in_team:
                raise CoachError(
                    "A half-suit can only be nulled when the claiming team "
                    "held all six cards. Here an opponent held one, so the "
                    "opposing team scores it.")
            winner = NULL_TEAM
        elif winner is None:
            winner = team if all_in_team else 1 - team
        elif winner not in (0, 1, NULL_TEAM):
            raise CoachError("Winner must be team 0, team 1, or -1 for null.")
        self._push(ClaimEvent(claimer, half_suit, tuple(holders),
                              tuple(holders), winner))

    def add_pass(self, player: int, teammate: int):
        if team_of(player) != team_of(teammate) or player == teammate:
            raise CoachError("A player may only pass to a different teammate.")
        self._push(PassEvent(player, teammate))

    def undo(self):
        if not self.events:
            raise CoachError("Nothing to undo.")
        self.events.pop()

    def _push(self, ev: Event):
        self.events.append(ev)
        try:
            self.validate()
        except BeliefContradiction as e:
            self.events.pop()
            raise CoachError(
                "That does not fit what has already happened at this table "
                f"({e}). Check the seats and the card, or undo an earlier "
                "entry.")
        except Exception:
            self.events.pop()
            raise CoachError("That event is not consistent with the log.")

    # -- advice ------------------------------------------------------------

    #: Current champion policy (see checkpoints/league.json). Beat the plain
    #: belief prior by +1.275 sets per duplicate deal-pair [+1.03, +1.52]
    #: over 800 pairs, by weighting turn-risk and suit scarcity as
    #: tie-breakers among asks of similar success probability.
    CHAMPION = ("tuned", {"w_turn": 0.6, "w_scarce": 0.2})

    def advise(self, engine: Optional[str] = None, n_samples: int = 64,
               engine_kwargs: Optional[dict] = None) -> dict:
        obs = self.observation()
        bel = BeliefState(self.rules, observer=self.seat)
        bel.update(obs)
        rng = random.Random(12345)          # deterministic advice
        worlds = [bel.sample_current_hands(rng) for _ in range(n_samples)]

        if engine is None:
            engine, default_kwargs = self.CHAMPION
        else:
            default_kwargs = {}
        agent = AGENT_REGISTRY[engine](**(engine_kwargs or default_kwargs))
        agent.begin_game(self.seat, self.rules, 999)
        on_move = obs.turn == self.seat
        best = None
        if on_move:
            try:
                best = agent.act(obs)
            except Exception:
                best = None

        n = max(1, len(worlds))
        asks = []
        for a in obs.legal_asks():
            hits = sum(1 for w in worlds if w[a.target] & (1 << a.card))
            asks.append({
                "target": a.target, "card": _card_payload(a.card),
                "p_success": hits / n,
                "text": f"Ask P{a.target} for {card_long_name(a.card)}",
            })
        asks.sort(key=lambda r: -r["p_success"])

        claims = []
        if hasattr(agent, "_claim_candidates"):
            try:
                for prob, cl in sorted(agent._claim_candidates(obs, worlds),
                                       key=lambda t: -t[0])[:5]:
                    claims.append({
                        "half_suit": cl.half_suit,
                        "half_suit_name": HALF_SUIT_NAMES[cl.half_suit],
                        "assignment": list(cl.assignment),
                        "p_correct": prob,
                        "detail": [
                            {"card": _card_payload(cl.half_suit * 6 + i),
                             "holder": h}
                            for i, h in enumerate(cl.assignment)],
                    })
            except Exception:
                pass

        return {
            "on_move": on_move,
            "turn": obs.turn,
            "seat": self.seat,
            "team": team_of(self.seat),
            "hand": [_card_payload(c) for c in mask_to_cards(obs.hand)],
            "hand_counts": list(obs.hand_counts),
            "set_winner": list(obs.set_winner),
            "half_suits": [
                {"index": i, "name": HALF_SUIT_NAMES[i],
                 "winner": obs.set_winner[i],
                 "mine": (obs.hand & half_suit_mask(i)).bit_count(),
                 "cards": [_card_payload(c) for c in half_suit_cards(i)]}
                for i in range(num_half_suits(self.rules.variant))],
            "recommendation": _describe(best) if best is not None else None,
            "recommendation_kind": _kind(best),
            "asks": asks[:15],
            "claims": claims,
            "beliefs": _beliefs(obs, worlds, bel),
            "certain": _certain(bel, self.seat),
            "deductions": _deductions(bel),
            "events": [_event_payload(e, i) for i, e in enumerate(self.events)],
            "scores": {
                "team0": sum(1 for w in obs.set_winner if w == 0),
                "team1": sum(1 for w in obs.set_winner if w == 1),
                "null": sum(1 for w in obs.set_winner if w == -1)},
        }


# ---------------------------------------------------------------------------
# payload helpers
# ---------------------------------------------------------------------------

def _card_payload(c: int) -> dict:
    return {"id": c, "name": card_name(c), "long": card_long_name(c),
            "red": is_red(c), "hs": c // 6}


def _kind(a) -> Optional[str]:
    if isinstance(a, Ask):
        return "ask"
    if isinstance(a, Claim):
        return "claim"
    if isinstance(a, PassEvent) or a.__class__.__name__ == "Pass":
        return "pass"
    return None


def _describe(a) -> str:
    if isinstance(a, Ask):
        return f"Ask P{a.target} for {card_long_name(a.card)}"
    if isinstance(a, Claim):
        parts = ", ".join(
            f"{card_name(a.half_suit * 6 + i)} with P{p}"
            for i, p in enumerate(a.assignment))
        return f"Claim {HALF_SUIT_NAMES[a.half_suit]} ({parts})"
    if a.__class__.__name__ == "Pass":
        return f"Pass the turn to P{a.teammate}"
    return str(a)


def _beliefs(obs: Observation, worlds, bel) -> list[dict]:
    n = max(1, len(worlds))
    out = []
    for c in range(obs.deck_size):
        if obs.set_winner[c // 6] is not None or obs.hand & (1 << c):
            continue
        probs = [0.0] * 6
        for w in worlds:
            for p in range(6):
                if w[p] & (1 << c):
                    probs[p] += 1 / n
        if not any(probs):
            continue
        out.append({"card": _card_payload(c), "probs": probs,
                    "best": max(range(6), key=lambda p: probs[p]),
                    "certainty": max(probs)})
    out.sort(key=lambda b: (-b["certainty"], b["card"]["id"]))
    return out


def _certain(bel, seat: int) -> list[dict]:
    hands = bel.known_current_hands()
    out = []
    for p in range(6):
        if p == seat or not hands[p]:
            continue
        out.append({"player": p,
                    "cards": [_card_payload(c) for c in mask_to_cards(hands[p])]})
    return out


def _deductions(bel) -> list[str]:
    out = []
    for cards, p in bel.ors[:12]:
        names = " or ".join(card_name(c) for c in cards)
        out.append(f"P{p} holds at least one of: {names}")
    return out


def _event_payload(ev: Event, i: int) -> dict:
    if isinstance(ev, AskEvent):
        return {"i": i, "type": "ask", "asker": ev.asker, "target": ev.target,
                "card": card_name(ev.card), "success": ev.success,
                "text": (f"P{ev.asker} asked P{ev.target} for "
                         f"{card_long_name(ev.card)}: "
                         f"{'got it' if ev.success else 'no'}")}
    if isinstance(ev, ClaimEvent):
        who = ("Team A" if ev.winner == 0 else "Team B" if ev.winner == 1
               else "nobody")
        return {"i": i, "type": "claim", "claimer": ev.claimer,
                "half_suit": ev.half_suit, "winner": ev.winner,
                "text": (f"P{ev.claimer} claimed "
                         f"{HALF_SUIT_NAMES[ev.half_suit]}: {who} scored")}
    return {"i": i, "type": "pass", "player": ev.player,
            "teammate": ev.teammate,
            "text": f"P{ev.player} passed to P{ev.teammate}"}
