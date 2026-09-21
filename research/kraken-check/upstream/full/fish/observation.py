"""The information boundary between the engine and acting agents.

An Observation contains exactly the information a human player in seat
``player`` would legally have (SPEC.md section 5): their own hand, the public
event log, and public derived state. Policies receive ONLY this object.

``Observation.reconstruct`` rebuilds the derived public state purely from the
event log + rules + own initial hand; the leakage test asserts it matches what
the engine reports, proving observations carry nothing beyond public data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from .cards import (
    CARDS_PER_HALF_SUIT, NUM_PLAYERS, card_id, card_name, cards_per_player,
    deck_size, half_suit_mask, mask_to_cards, num_half_suits, opponents,
)
from .engine import Ask, AskEvent, ClaimEvent, Event, GameState, Pass, PassEvent
from .rules import RuleConfig


@dataclass(frozen=True)
class Observation:
    player: int
    rules: RuleConfig
    hand: int                       # own current hand bitmask
    turn: int
    hand_counts: tuple[int, ...]
    set_winner: tuple[Optional[int], ...]
    history: tuple[Event, ...]

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_state(cls, state: GameState, player: int) -> "Observation":
        return cls(
            player=player,
            rules=state.rules,
            hand=state.hands[player],
            turn=state.turn,
            hand_counts=state.hand_counts(),
            set_winner=tuple(state.set_winner),
            history=tuple(state.history),
        )

    # -- public helpers -------------------------------------------------------

    @property
    def num_half_suits(self) -> int:
        return num_half_suits(self.rules.variant)

    @property
    def deck_size(self) -> int:
        return deck_size(self.rules.variant)

    def initial_hand(self) -> int:
        """Reconstruct own initial hand from current hand + public transfers."""
        hand = self.hand
        me = self.player
        for ev in reversed(self.history):
            if isinstance(ev, AskEvent) and ev.success:
                bit = 1 << ev.card
                if ev.asker == me:
                    hand &= ~bit          # received later; not initially mine
                elif ev.target == me:
                    hand |= bit           # gave it away later; it was mine before
            elif isinstance(ev, ClaimEvent):
                for i, holder in enumerate(ev.revealed):
                    if holder == me:
                        hand |= 1 << (ev.half_suit * CARDS_PER_HALF_SUIT + i)
        return hand

    def legal_asks(self) -> list[Ask]:
        """Legal asks for this observer (empty if not on move), computable
        from public info + own hand only."""
        me = self.player
        if self.turn != me or self.hand == 0:
            return []
        opps = [o for o in opponents(me) if self.hand_counts[o] > 0]
        if not opps:
            return []
        asks = []
        for hs in range(self.num_half_suits):
            if self.set_winner[hs] is not None:
                continue
            hs_mask = half_suit_mask(hs)
            if not self.hand & hs_mask:
                continue
            askable = hs_mask if self.rules.allow_bluff_asks else (hs_mask & ~self.hand)
            for card in mask_to_cards(askable):
                for o in opps:
                    asks.append(Ask(o, card))
        return asks

    def claimable_half_suits(self) -> list[int]:
        if not self.rules.claims_any_time and self.turn != self.player:
            return []
        return [hs for hs in range(self.num_half_suits)
                if self.set_winner[hs] is None]

    def legal_passes(self) -> list[Pass]:
        me = self.player
        if self.turn != me or self.hand != 0:
            return []
        team = me & 1
        return [Pass(t) for t in (team, team + 2, team + 4)
                if t != me and self.hand_counts[t] > 0]

    def must_claim(self) -> bool:
        return (self.turn == self.player and self.hand != 0
                and not self.legal_asks())

    def must_pass(self) -> bool:
        return self.turn == self.player and self.hand == 0

    # -- reconstruction (leakage proof) ---------------------------------------

    @staticmethod
    def reconstruct(rules: RuleConfig, player: int, initial_hand: int,
                    history: tuple[Event, ...]) -> "Observation":
        """Rebuild the observation for ``player`` using ONLY public events and
        the player's own initial hand. Used by tests to prove observations are
        leak-free, and by tools that ingest transcripts."""
        n_hs = num_half_suits(rules.variant)
        per = cards_per_player(rules.variant)
        counts = [per] * NUM_PLAYERS
        hand = initial_hand
        set_winner: list[Optional[int]] = [None] * n_hs
        turn = rules.starting_player

        def normalize(turn: int) -> int:
            # mirrors GameState._normalize_turn, using public counts
            if all(w is not None for w in set_winner) or counts[turn]:
                return turn
            team = turn & 1
            if any(counts[t] for t in (team, team + 2, team + 4) if t != turn):
                return turn  # player must PASS
            for step in range(1, NUM_PLAYERS):
                q = (turn + step) % NUM_PLAYERS
                if counts[q]:
                    return q
            return turn

        for ev in history:
            if isinstance(ev, AskEvent):
                if ev.success:
                    counts[ev.target] -= 1
                    counts[ev.asker] += 1
                    bit = 1 << ev.card
                    if ev.asker == player:
                        hand |= bit
                    elif ev.target == player:
                        hand &= ~bit
                    turn = ev.asker
                else:
                    turn = ev.target
            elif isinstance(ev, ClaimEvent):
                for i, holder in enumerate(ev.revealed):
                    counts[holder] -= 1
                    if holder == player:
                        hand &= ~(1 << (ev.half_suit * CARDS_PER_HALF_SUIT + i))
                set_winner[ev.half_suit] = ev.winner
            elif isinstance(ev, PassEvent):
                turn = ev.teammate
            turn = normalize(turn)

        return Observation(
            player=player, rules=rules, hand=hand, turn=turn,
            hand_counts=tuple(counts), set_winner=tuple(set_winner),
            history=tuple(history),
        )

    # -- serialization --------------------------------------------------------

    def to_json(self) -> str:
        def ev_dict(ev: Event) -> dict:
            if isinstance(ev, AskEvent):
                return {"type": "ask", "asker": ev.asker, "target": ev.target,
                        "card": card_name(ev.card), "success": ev.success}
            if isinstance(ev, ClaimEvent):
                return {"type": "claim", "claimer": ev.claimer,
                        "half_suit": ev.half_suit,
                        "declared": list(ev.declared),
                        "revealed": list(ev.revealed),
                        "winner": ev.winner}
            return {"type": "pass", "player": ev.player, "teammate": ev.teammate}

        return json.dumps({
            "player": self.player,
            "rules": self.rules.to_dict(),
            "hand": [card_name(c) for c in mask_to_cards(self.hand)],
            "turn": self.turn,
            "hand_counts": list(self.hand_counts),
            "set_winner": list(self.set_winner),
            "history": [ev_dict(e) for e in self.history],
        }, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Observation":
        d = json.loads(text)
        rules = RuleConfig.from_dict(d["rules"])
        history: list[Event] = []
        for e in d["history"]:
            if e["type"] == "ask":
                history.append(AskEvent(e["asker"], e["target"],
                                        card_id(e["card"]), e["success"]))
            elif e["type"] == "claim":
                history.append(ClaimEvent(e["claimer"], e["half_suit"],
                                          tuple(e["declared"]),
                                          tuple(e["revealed"]), e["winner"]))
            else:
                history.append(PassEvent(e["player"], e["teammate"]))
        hand = 0
        for name in d["hand"]:
            hand |= 1 << card_id(name)
        return cls(
            player=d["player"], rules=rules, hand=hand, turn=d["turn"],
            hand_counts=tuple(d["hand_counts"]),
            set_winner=tuple(d["set_winner"]),
            history=tuple(history),
        )
