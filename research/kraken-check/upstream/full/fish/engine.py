"""Core Literature/Fish game engine.

Design notes:
- Hands are int bitmasks over card ids (54 bits max), so membership tests,
  transfers, and half-suit intersections are single integer ops.
- Every event that occurs is public (SPEC.md section 5); the full event log
  IS the public transcript. Hidden information lives only in ``hands``.
- ``apply`` validates legality and raises IllegalAction on violations, so
  agents can never corrupt state; ``debug=True`` additionally checks global
  invariants after every action.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Union

from .cards import (
    CARDS_PER_HALF_SUIT, NUM_PLAYERS, cards_per_player, deck_size,
    half_suit_mask, half_suit_of, mask_to_cards, num_half_suits, opponents,
    team_of, teammates,
)
from .rules import RuleConfig

NULL_TEAM = -1  # set_winner value for a nulled half-suit


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ask:
    target: int
    card: int


@dataclass(frozen=True)
class Claim:
    half_suit: int
    # assignment[i] = player declared to hold card (half_suit*6 + i);
    # every entry must be on the claimant's team.
    assignment: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class Pass:
    teammate: int


Action = Union[Ask, Claim, Pass]


# ---------------------------------------------------------------------------
# Public events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AskEvent:
    asker: int
    target: int
    card: int
    success: bool


@dataclass(frozen=True)
class ClaimEvent:
    claimer: int
    half_suit: int
    declared: tuple[int, int, int, int, int, int]
    revealed: tuple[int, int, int, int, int, int]  # actual holders at resolution
    winner: int  # 0, 1, or NULL_TEAM


@dataclass(frozen=True)
class PassEvent:
    player: int
    teammate: int


Event = Union[AskEvent, ClaimEvent, PassEvent]


class IllegalAction(Exception):
    pass


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class GameState:
    __slots__ = ("rules", "hands", "turn", "set_winner", "history", "debug",
                 "_num_hs", "_deck_size", "agent_seed")

    def __init__(self, rules: RuleConfig, hands: list[int], turn: int,
                 debug: bool = False):
        self.rules = rules
        self.hands = hands                        # bitmask per player
        self.turn = turn
        self._num_hs = num_half_suits(rules.variant)
        self._deck_size = deck_size(rules.variant)
        self.set_winner: list[Optional[int]] = [None] * self._num_hs
        self.history: list[Event] = []
        self.debug = debug
        self.agent_seed: Optional[int] = None
        self._normalize_turn()

    # -- construction -------------------------------------------------------

    @classmethod
    def deal(cls, rules: RuleConfig, seed: Optional[int] = None,
             rng: Optional[random.Random] = None, debug: bool = False,
             deck_order: Optional[list[int]] = None) -> "GameState":
        """Deal a fresh game. Exactly one of seed/rng/deck_order drives the deal."""
        n = deck_size(rules.variant)
        if deck_order is not None:
            deck = list(deck_order)
            if sorted(deck) != list(range(n)):
                raise ValueError("deck_order must be a permutation of the deck")
        else:
            if rng is None:
                rng = random.Random(seed)
            deck = list(range(n))
            rng.shuffle(deck)
        per = cards_per_player(rules.variant)
        hands = [0] * NUM_PLAYERS
        for i, card in enumerate(deck):
            hands[i % NUM_PLAYERS] |= 1 << card
        assert all(h.bit_count() == per for h in hands)
        return cls(rules, hands, rules.starting_player, debug=debug)

    @classmethod
    def from_components(cls, rules: RuleConfig, hands: list[int], turn: int,
                        set_winner: list[Optional[int]],
                        debug: bool = False) -> "GameState":
        """Construct a mid-game state (used by search determinization)."""
        state = cls(rules, list(hands), turn, debug=debug)
        state.set_winner = list(set_winner)
        state._normalize_turn()
        return state

    # -- queries -------------------------------------------------------------

    def hand_count(self, player: int) -> int:
        return self.hands[player].bit_count()

    def hand_counts(self) -> tuple[int, ...]:
        return tuple(h.bit_count() for h in self.hands)

    def holder_of(self, card: int) -> Optional[int]:
        bit = 1 << card
        for p in range(NUM_PLAYERS):
            if self.hands[p] & bit:
                return p
        return None

    def resolved_mask(self) -> int:
        m = 0
        for hs, w in enumerate(self.set_winner):
            if w is not None:
                m |= 1 << hs
        return m

    def scores(self) -> tuple[int, int, int]:
        """(team0 sets, team1 sets, null sets)."""
        a = sum(1 for w in self.set_winner if w == 0)
        b = sum(1 for w in self.set_winner if w == 1)
        n = sum(1 for w in self.set_winner if w == NULL_TEAM)
        return a, b, n

    @property
    def is_terminal(self) -> bool:
        return all(w is not None for w in self.set_winner)

    def winner(self) -> Optional[int]:
        """0 or 1 for the winning team, None for tie. Only valid when terminal."""
        a, b, _ = self.scores()
        if a > b:
            return 0
        if b > a:
            return 1
        return None

    # -- legality ------------------------------------------------------------

    def legal_asks(self, player: int) -> list[Ask]:
        if player != self.turn:
            return []
        hand = self.hands[player]
        if hand == 0:
            return []
        opps = [o for o in opponents(player) if self.hands[o]]
        if not opps:
            return []
        asks = []
        for hs in range(self._num_hs):
            if self.set_winner[hs] is not None:
                continue
            hs_mask = half_suit_mask(hs)
            mine = hand & hs_mask
            if not mine:
                continue
            askable = hs_mask if self.rules.allow_bluff_asks else (hs_mask & ~hand)
            for card in mask_to_cards(askable):
                for o in opps:
                    asks.append(Ask(o, card))
        return asks

    def claimable_half_suits(self, player: int) -> list[int]:
        if not self.rules.claims_any_time and player != self.turn:
            return []
        return [hs for hs in range(self._num_hs) if self.set_winner[hs] is None]

    def legal_passes(self, player: int) -> list[Pass]:
        if player != self.turn or self.hands[player]:
            return []
        return [Pass(t) for t in teammates(player)
                if t != player and self.hands[t]]

    def must_claim(self, player: int) -> bool:
        """True when the on-move player has cards but no legal ask exists."""
        return (player == self.turn and self.hands[player] != 0
                and not self.legal_asks(player))

    def check_legal(self, player: int, action: Action) -> None:
        if isinstance(action, Ask):
            if player != self.turn:
                raise IllegalAction("not your turn to ask")
            if self.hands[player] == 0:
                raise IllegalAction("cardless player cannot ask")
            if not 0 <= action.target < NUM_PLAYERS:
                raise IllegalAction("bad target")
            if team_of(action.target) == team_of(player):
                raise IllegalAction("cannot ask a teammate")
            if self.hands[action.target] == 0:
                raise IllegalAction("target has no cards")
            if not 0 <= action.card < self._deck_size:
                raise IllegalAction("bad card")
            hs = half_suit_of(action.card)
            if self.set_winner[hs] is not None:
                raise IllegalAction("half-suit already resolved")
            if not self.hands[player] & half_suit_mask(hs):
                raise IllegalAction("must hold a card of the half-suit")
            if (not self.rules.allow_bluff_asks
                    and self.hands[player] & (1 << action.card)):
                raise IllegalAction("cannot ask for a card you hold")
        elif isinstance(action, Claim):
            if not self.rules.claims_any_time and player != self.turn:
                raise IllegalAction("claims only on your own turn")
            if not 0 <= action.half_suit < self._num_hs:
                raise IllegalAction("bad half-suit")
            if self.set_winner[action.half_suit] is not None:
                raise IllegalAction("half-suit already resolved")
            if len(action.assignment) != CARDS_PER_HALF_SUIT:
                raise IllegalAction("assignment must cover all six cards")
            team = team_of(player)
            if any(not 0 <= q < NUM_PLAYERS or team_of(q) != team
                   for q in action.assignment):
                raise IllegalAction("assignment must map to the claimant's team")
        elif isinstance(action, Pass):
            if player != self.turn:
                raise IllegalAction("not your turn")
            if self.hands[player] != 0:
                raise IllegalAction("only a cardless player may pass")
            # Bounds check FIRST: without it, Pass(-1) slipped through because
            # team_of(-1) == 1 and hands[-1] negative-indexes into hands[5],
            # corrupting state.turn to -1; Pass(7) raised IndexError instead of
            # IllegalAction. check_legal is the safety boundary, so it must
            # reject both cleanly.
            if not 0 <= action.teammate < NUM_PLAYERS:
                raise IllegalAction("bad teammate id")
            if action.teammate == player or team_of(action.teammate) != team_of(player):
                raise IllegalAction("must pass to a teammate")
            if self.hands[action.teammate] == 0:
                raise IllegalAction("teammate has no cards")
        else:
            raise IllegalAction(f"unknown action {action!r}")

    # -- application ---------------------------------------------------------

    def apply(self, player: int, action: Action) -> Event:
        self.check_legal(player, action)
        if isinstance(action, Ask):
            event = self._apply_ask(player, action)
        elif isinstance(action, Claim):
            event = self._apply_claim(player, action)
        else:
            event = self._apply_pass(player, action)
        self.history.append(event)
        self._normalize_turn()
        if self.debug:
            self.check_invariants()
        return event

    def _apply_ask(self, player: int, action: Ask) -> AskEvent:
        bit = 1 << action.card
        success = bool(self.hands[action.target] & bit)
        if success:
            self.hands[action.target] ^= bit
            self.hands[player] |= bit
            # asker retains the turn
        else:
            self.turn = action.target
        return AskEvent(player, action.target, action.card, success)

    def _apply_claim(self, player: int, action: Claim) -> ClaimEvent:
        hs = action.half_suit
        base = hs * CARDS_PER_HALF_SUIT
        revealed = []
        for i in range(CARDS_PER_HALF_SUIT):
            holder = self.holder_of(base + i)
            assert holder is not None, "unresolved card must be in a hand"
            revealed.append(holder)
        revealed_t = tuple(revealed)
        team = team_of(player)
        # Coerce before comparing: a factually exact claim submitted as a list
        # (natural for RL action decoding) compared unequal to the revealed
        # tuple and was scored as a wrong distribution, nulling a correct
        # claim and emitting a transcript where declared == revealed but
        # nobody scored.
        declared = tuple(action.assignment)
        if revealed_t == declared:
            winner = team
        elif any(team_of(h) != team for h in revealed_t):
            winner = 1 - team
        elif self.rules.wrong_distribution_outcome == "opponent":
            winner = 1 - team
        else:
            winner = NULL_TEAM
        hs_mask = half_suit_mask(hs)
        for p in range(NUM_PLAYERS):
            self.hands[p] &= ~hs_mask
        self.set_winner[hs] = winner
        return ClaimEvent(player, hs, declared, revealed_t, winner)

    def _apply_pass(self, player: int, action: Pass) -> PassEvent:
        self.turn = action.teammate
        return PassEvent(player, action.teammate)

    def _normalize_turn(self) -> None:
        """If the on-move player is cardless and no teammate has cards, advance
        clockwise to the first player with cards (SPEC section 4.4)."""
        if self.is_terminal:
            return
        p = self.turn
        if self.hands[p]:
            return
        if any(self.hands[t] for t in teammates(p) if t != p):
            return  # player must PASS; that is their action
        for step in range(1, NUM_PLAYERS):
            q = (p + step) % NUM_PLAYERS
            if self.hands[q]:
                self.turn = q
                return
        raise AssertionError("non-terminal state but nobody has cards")

    # -- invariants (SPEC section 7) ------------------------------------------

    def check_invariants(self) -> None:
        seen = 0
        for p in range(NUM_PLAYERS):
            assert self.hands[p] & seen == 0, "card in two hands"
            seen |= self.hands[p]
        resolved_cards = 0
        num_resolved = 0
        for hs, w in enumerate(self.set_winner):
            if w is not None:
                resolved_cards |= half_suit_mask(hs)
                num_resolved += 1
                assert w in (0, 1, NULL_TEAM)
        assert seen & resolved_cards == 0, "resolved card still in a hand"
        full = (1 << self._deck_size) - 1
        assert seen | resolved_cards == full, "card vanished"
        assert (sum(self.hand_counts()) + CARDS_PER_HALF_SUIT * num_resolved
                == self._deck_size)
        a, b, n = self.scores()
        assert a + b + n == num_resolved
        assert 0 <= self.turn < NUM_PLAYERS
        if not self.is_terminal:
            assert any(self.hands[p] for p in range(NUM_PLAYERS))
