"""Compact binary game logging.

A game is stored as a fixed header plus one byte-packed record per action,
NOT verbose JSON, so millions of games stay tractable: a full 54-card game
with ~130 actions costs roughly 0.4 KB.

(Named ``gamelog`` rather than ``logging`` so it cannot shadow the standard
library module for anything importing this package.)

Layout (little-endian):
  magic  4s   b"FISH"
  ver    B    format version (1)
  flags  B    bit0: variant 54 (else 48)
  seed   Q    deal seed
  aseed  Q    agent seed (recorded, since agent randomness is independent
              of the deal and reproducibility requires both)
  start  B    starting player
  deal   54B  initial owner per card (0..5; 0xFF unused in 48-card games)
  nacts  I    number of actions
  then per action, 1 tag byte + payload:
    tag 0 ASK:   asker<<4|target (B), card (B), success (B)
    tag 1 CLAIM: claimer (B), half_suit (B), declared 6B, revealed 6B,
                 winner (b)
    tag 2 PASS:  player (B), teammate (B)
  footer: scores a, b, null (3B)

``detail_level`` lets callers disable logging entirely (0), record outcomes
only (1), or store full transcripts (2).
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

from .cards import deck_size
from .engine import AskEvent, ClaimEvent, GameState, PassEvent

MAGIC = b"FISH"
VERSION = 1
_HEADER = "<4sBBQQB"


def encode_game(state: GameState, initial_hands: list[int], seed: int,
                agent_seed: int = 0) -> bytes:
    deal = bytearray([0xFF] * 54)
    for p in range(6):
        h = initial_hands[p]
        while h:
            low = h & -h
            deal[low.bit_length() - 1] = p
            h ^= low
    flags = 1 if state.rules.variant == "54" else 0
    mask64 = (1 << 64) - 1
    out = bytearray()
    out += struct.pack(_HEADER, MAGIC, VERSION, flags, seed & mask64,
                       (agent_seed or 0) & mask64, state.rules.starting_player)
    out += bytes(deal)
    out += struct.pack("<I", len(state.history))
    for ev in state.history:
        if isinstance(ev, AskEvent):
            out += struct.pack("<BBBB", 0, (ev.asker << 4) | ev.target,
                               ev.card, 1 if ev.success else 0)
        elif isinstance(ev, ClaimEvent):
            out += struct.pack("<BBB", 1, ev.claimer, ev.half_suit)
            out += bytes(ev.declared) + bytes(ev.revealed)
            out += struct.pack("<b", ev.winner)
        else:
            out += struct.pack("<BBB", 2, ev.player, ev.teammate)
    a, b, n = state.scores()
    out += struct.pack("<BBB", a, b, n)
    return bytes(out)


def decode_game(buf: bytes, offset: int = 0):
    """Returns (record_dict, new_offset)."""
    magic, ver, flags, seed, aseed, start = struct.unpack_from(
        _HEADER, buf, offset)
    if magic != MAGIC or ver != VERSION:
        raise ValueError("bad record header")
    offset += struct.calcsize(_HEADER)
    deal = list(buf[offset:offset + 54])
    offset += 54
    (nacts,) = struct.unpack_from("<I", buf, offset)
    offset += 4
    events = []
    for _ in range(nacts):
        tag = buf[offset]
        if tag == 0:
            _, packed, card, success = struct.unpack_from("<BBBB", buf, offset)
            events.append(AskEvent(packed >> 4, packed & 0xF, card,
                                   bool(success)))
            offset += 4
        elif tag == 1:
            _, claimer, hs = struct.unpack_from("<BBB", buf, offset)
            offset += 3
            declared = tuple(buf[offset:offset + 6])
            revealed = tuple(buf[offset + 6:offset + 12])
            offset += 12
            (winner,) = struct.unpack_from("<b", buf, offset)
            offset += 1
            events.append(ClaimEvent(claimer, hs, declared, revealed, winner))
        else:
            _, player, teammate = struct.unpack_from("<BBB", buf, offset)
            events.append(PassEvent(player, teammate))
            offset += 3
    a, b, n = struct.unpack_from("<BBB", buf, offset)
    offset += 3
    return ({"seed": seed, "agent_seed": aseed,
             "variant": "54" if flags & 1 else "48",
             "starting_player": start, "deal": deal, "history": events,
             "scores": (a, b, n)}, offset)


class GameLogWriter:
    """Append-only binary log. detail_level: 0 off, 1 outcomes, 2 full."""

    def __init__(self, path: Path | str, detail_level: int = 2):
        self.path = Path(path)
        self.detail_level = detail_level
        self._fh = None
        if detail_level > 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "ab")

    def write(self, state: GameState, initial_hands: list[int], seed: int,
              agent_seed: int = 0) -> None:
        if self.detail_level == 0:
            return
        if self.detail_level == 1:
            a, b, n = state.scores()
            self._fh.write(struct.pack("<QBBB", seed & ((1 << 64) - 1),
                                       a, b, n))
            return
        self._fh.write(encode_game(state, initial_hands, seed, agent_seed))

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_games(path: Path | str) -> Iterator[dict]:
    buf = Path(path).read_bytes()
    off = 0
    while off < len(buf):
        rec, off = decode_game(buf, off)
        yield rec
