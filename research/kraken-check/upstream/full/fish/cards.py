"""Card and half-suit definitions.

Card ids are stable across variants: ``card // 6`` is the half-suit index,
``card % 6`` the position within it. Half-suits 0-3 are the low suits
(2-7 of C, D, H, S), 4-7 the high suits (9-A), and 8 is the Specials
half-suit (the four 8s plus the Red and Black Jokers), used only in the
54-card variant.
"""

from __future__ import annotations

SUITS = "CDHS"
LOW_RANKS = ["2", "3", "4", "5", "6", "7"]
HIGH_RANKS = ["9", "T", "J", "Q", "K", "A"]

NUM_HALF_SUITS_48 = 8
NUM_HALF_SUITS_54 = 9
CARDS_PER_HALF_SUIT = 6
NUM_PLAYERS = 6

HALF_SUIT_NAMES = [
    "Low Clubs", "Low Diamonds", "Low Hearts", "Low Spades",
    "High Clubs", "High Diamonds", "High Hearts", "High Spades",
    "Specials (8s+Jokers)",
]

# The ninth half-suit: the four 8s plus the two jokers. The jokers are
# distinct, individually askable cards - a Red Joker and a Black Joker, so
# players can name exactly which one they want.
_SPECIAL_NAMES = ["8C", "8D", "8H", "8S", "RJ", "BJ"]

RED_JOKER = 52
BLACK_JOKER = 53

CARD_LONG_NAMES = {"RJ": "Red Joker", "BJ": "Black Joker"}

# Accepted aliases when parsing card names (case-insensitive).
_CARD_ALIASES = {
    "X1": "RJ", "X2": "BJ",
    "JOKER1": "RJ", "JOKER2": "BJ",
    "REDJOKER": "RJ", "BLACKJOKER": "BJ",
    "RED": "RJ", "BLACK": "BJ", "JR": "RJ", "JB": "BJ",
}


def _build_card_names() -> list[str]:
    names = []
    for hs in range(8):
        suit = SUITS[hs % 4]
        ranks = LOW_RANKS if hs < 4 else HIGH_RANKS
        names.extend(rank + suit for rank in ranks)
    names.extend(_SPECIAL_NAMES)
    return names


CARD_NAMES = _build_card_names()
CARD_IDS = {name: i for i, name in enumerate(CARD_NAMES)}


def card_name(card: int) -> str:
    return CARD_NAMES[card]


def card_long_name(card: int) -> str:
    """Human-facing name, e.g. 'Red Joker' instead of 'RJ'."""
    n = CARD_NAMES[card]
    return CARD_LONG_NAMES.get(n, n)


def is_red(card: int) -> bool:
    """True for hearts, diamonds, and the Red Joker."""
    if card == RED_JOKER:
        return True
    if card == BLACK_JOKER:
        return False
    return CARD_NAMES[card][-1] in ("H", "D")


def card_id(name: str) -> int:
    key = name.strip().upper().replace(" ", "").replace("_", "")
    key = _CARD_ALIASES.get(key, key)
    return CARD_IDS[key]


def half_suit_of(card: int) -> int:
    return card // 6


def half_suit_cards(hs: int) -> range:
    return range(hs * 6, hs * 6 + 6)


def half_suit_mask(hs: int) -> int:
    return 0b111111 << (hs * 6)


def deck_size(variant: str) -> int:
    return 48 if variant == "48" else 54


def num_half_suits(variant: str) -> int:
    return NUM_HALF_SUITS_48 if variant == "48" else NUM_HALF_SUITS_54


def cards_per_player(variant: str) -> int:
    return deck_size(variant) // NUM_PLAYERS


def team_of(player: int) -> int:
    """Team 0 = players 0,2,4; team 1 = players 1,3,5."""
    return player & 1


def teammates(player: int) -> tuple[int, int, int]:
    t = player & 1
    return (t, t + 2, t + 4)


def opponents(player: int) -> tuple[int, int, int]:
    t = 1 - (player & 1)
    return (t, t + 2, t + 4)


def mask_to_cards(mask: int) -> list[int]:
    cards = []
    while mask:
        low = mask & -mask
        cards.append(low.bit_length() - 1)
        mask ^= low
    return cards


def popcount(mask: int) -> int:
    return mask.bit_count()
