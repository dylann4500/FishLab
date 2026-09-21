"""Perpetual states in Literature: when can the game stop making progress?

THE QUESTION
------------
Chess has stalemate and threefold repetition. Does Fish have an analogue - a
position from which the game cannot progress, or from which rational players
would decline to progress? The answer has three layers, and only the third one
is interesting.

**Layer 1: the state graph is cyclic.** Two opponents can trade a card back and
forth and return to an identical position, so positions can repeat. This was
established by exhaustive search and is a genuine property of the rules, not an
artefact. It follows that *termination is a property of the players, not of the
rules*: any claim that "every game of Fish ends" describes a convention.

**Layer 2: under perfect information, optimal play terminates anyway.** Cycles
exist, but no optimal line needs one. So the cyclicity is not, by itself, a
strategic phenomenon.

**Layer 3: dead positions.** This is the real Fish analogue, and it has no chess
counterpart. Call a live half-suit **dead** when all six of its cards sit with
one team. Then no member of the other team holds a card of it, so no member of
the other team may legally ask in it; and members of the holding team may only
ask opponents, who hold none of it. So:

    Proposition. No ask in a dead half-suit can ever succeed.

Call a position **dead** when every live half-suit is dead. In a dead position
no card can ever change hands again: every legal ask is guaranteed to fail, the
turn simply circulates, and the material is frozen for the rest of the game.
That is as close to a stalemate as Fish gets, and unlike a chess stalemate it is
not a draw - the sets are already decided, they just have not been claimed yet.

THE PART THAT IS NOT OBVIOUS
----------------------------
A dead position looks like a deadlock for a team that holds a half-suit but
cannot place the split among its own members: no card will move, so no new
information can arrive, so the team must either gamble on a declaration or stall
forever. That reasoning is wrong, and the reason it is wrong is a strategy.

Information does not only arrive when cards move. Under the no-bluff rule a
player may not ask for a card they hold, so **asking for a card publicly proves
you do not hold it**. In a dead position that ask is guaranteed to fail, which
means it costs no material at all - there is no card to lose, and the turn it
concedes is worthless because no opponent can win a card either. The ask is a
*free* one-bit broadcast to your partners.

So a team stuck in a dead position can talk its way out: each member asks for
cards of the contested half-suit that they do not hold, and after a few such
asks the split is common knowledge within the team and the declaration is safe.
Literature's only communication channel is normally costly and read by the
opposition; in a dead position it becomes free, and the opposition can do
nothing with what it hears because the half-suit is already lost to them.

``signalling_ask`` implements that protocol, and
``scripts4/perpetual_study.py`` measures whether it is worth anything.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from fish.beliefs import RESOLVED
from fish.cards import (HALF_SUIT_NAMES, NUM_PLAYERS, card_name,
                        half_suit_cards, team_of)
from fish.engine import Ask


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

#: bitmask of seats on each team
_TEAM_MASK = (0b010101, 0b101010)


def half_suit_owner_team(bel, hs: int) -> Optional[int]:
    """The team that provably holds every card of ``hs``, or None.

    Provably, and WITHOUT needing to know which member holds what: it is enough
    that every possible holder of every card is on the same team. That
    distinction is the whole point. An earlier version of this function required
    each card's holder to be pinned, which made "we hold the half-suit but
    cannot place the split" a contradiction in terms - pinning the holders IS
    placing the split - and the interesting case could never be detected. It
    never fired once in 1,047 measured plies.
    """
    for team in (0, 1):
        mask = _TEAM_MASK[team]
        if all(0 < bel.current_holder_mask(c) <= mask
               and bel.current_holder_mask(c) & ~mask == 0
               for c in half_suit_cards(hs)):
            return team
    return None


def dead_half_suits(obs, bel) -> list:
    """Live half-suits in which one team provably holds all six cards."""
    out = []
    for hs in range(len(obs.set_winner)):
        if obs.set_winner[hs] is not None:
            continue
        t = half_suit_owner_team(bel, hs)
        if t is not None:
            out.append((hs, t))
    return out


def is_dead_position(obs, bel) -> bool:
    """True when no ask anywhere on the board can succeed."""
    live = [hs for hs in range(len(obs.set_winner))
            if obs.set_winner[hs] is None]
    if not live:
        return False
    dead = {hs for hs, _ in dead_half_suits(obs, bel)}
    return all(hs in dead for hs in live)


def unplaceable(obs, bel, ctx, hs: int) -> bool:
    """Our team provably holds the half-suit, but we cannot place the split.

    This is the position the deadlock argument is about: we know the six cards
    are ours, we know no opponent can ever take one, and we still cannot declare
    because we do not know which teammate has which.
    """
    if half_suit_owner_team(bel, hs) != ctx.my_team:
        return False
    return any(bel.current_holder_mask(c) & (bel.current_holder_mask(c) - 1)
               for c in half_suit_cards(hs))


# ---------------------------------------------------------------------------
# The signalling protocol
# ---------------------------------------------------------------------------

def signalling_ask(obs, bel, ctx, require_dead: bool = True,
                   exclude: frozenset = frozenset()):
    """A deliberately doomed ask that tells our partners where a card is not.

    The ask is placed in a half-suit our team provably owns, so it cannot land -
    which is the point. Under the no-bluff rule it proves publicly that we do
    not hold the card we asked for, and that is exactly the fact a partner is
    missing when the set is ours but the split is not placed.

    ``require_dead`` controls the price. In a DEAD position no ask anywhere can
    land, so the turn we concede is worthless and the broadcast is free. In a
    live position the same ask throws away a turn that might have won a card, so
    the caller must decide it is worth it - see ``FishBot4.signal_mode``.

    Measured motivation: a half-suit that is at some point provably ours but
    unplaceable is nulled 17.5% of the time, against 2.8% for one that never
    gets stuck, and such half-suits account for 27% of all nulls.

    ``exclude`` names cards this seat has already signalled. Proving *this seat
    does not hold X* removes only OUR bit from X's holder mask, so with two
    teammates left X keeps two bits and can stay the top pick forever, saying
    nothing new after the first time -- measured at 40.8 of 42.5 asks per
    episode in results/signal_deadline.json. Empty by default, which is
    bit-identical to the behaviour every published figure was taken on. See
    prereg/signal_no_repeat.md.
    """
    asks = obs.legal_asks()
    if not asks:
        return None
    if require_dead and not is_dead_position(obs, bel):
        return None
    M = ctx.M
    mine = [p for p in range(NUM_PLAYERS) if team_of(p) == ctx.my_team]
    best, best_gain = None, 0.0
    for a in asks:
        hs = a.card // 6
        if a.card in exclude:
            continue                       # already proved; says nothing new
        if half_suit_owner_team(bel, hs) != ctx.my_team:
            continue                       # only worth signalling about ours
        m = bel.current_holder_mask(a.card)
        if m == 0 or m & (m - 1) == 0:
            continue                       # already placed; no bit to send
        w = np.array([float(M[a.card, p]) for p in mine])
        s = w.sum()
        if s <= 0:
            continue
        w = w / s
        ent = float(-(w * np.log(np.maximum(w, 1e-12))).sum())
        if ent > best_gain:
            best, best_gain = a, ent
    return best


def stuck_half_suits(obs, bel, ctx) -> list:
    """Live half-suits our team provably owns but cannot yet place."""
    return [hs for hs, t in dead_half_suits(obs, bel)
            if t == ctx.my_team and unplaceable(obs, bel, ctx, hs)]


# ---------------------------------------------------------------------------
# Diagnosis, for the analyser and the UI
# ---------------------------------------------------------------------------

def diagnose(obs, bel, ctx) -> Optional[dict]:
    """A human-readable account of whether this position can still progress."""
    live = [hs for hs in range(len(obs.set_winner))
            if obs.set_winner[hs] is None]
    if not live:
        return None
    dead = dead_half_suits(obs, bel)
    dead_map = {hs: t for hs, t in dead}
    all_dead = all(hs in dead_map for hs in live)
    ours = [hs for hs, t in dead if t == ctx.my_team]
    theirs = [hs for hs, t in dead if t != ctx.my_team]
    stuck = [hs for hs in ours if unplaceable(obs, bel, ctx, hs)]

    left = sum(obs.hand_counts[o] for o in range(len(obs.hand_counts))
               if team_of(o) != ctx.my_team)
    if not dead:
        return {"is_dead": False, "dead_half_suits": [], "frozen": False,
                "unplaceable": [], "opponent_cards": left, "summary": ""}

    if all_dead:
        summary = (
            f"DEAD POSITION: all {len(live)} live half-suits sit entirely "
            f"with one team, so no ask anywhere can succeed and no card will "
            f"change hands again. The sets are already decided "
            f"({len(ours)} to us, {len(theirs)} to them); only the "
            f"declarations remain.")
        if stuck:
            summary += (
                f" We cannot yet place the split in "
                f"{', '.join(HALF_SUIT_NAMES[h] for h in stuck)} - but asking "
                f"is now free, and an ask proves we do not hold the card we "
                f"asked for, so we can signal the split to our partners.")
    else:
        summary = (
            f"{len(dead)} of {len(live)} live half-suits are dead (one team "
            f"holds all six), so no ask in them can ever succeed.")
        if stuck:
            # The part the earlier version only said once the WHOLE position
            # was dead, which is far too late to act on. A half-suit our team
            # owns is one no opponent can ask in -- legal_asks requires the
            # asker to hold a card of it -- so from the moment we collect it,
            # nothing anyone else does can tell us how it is split. The one
            # remaining channel is an ask of our own that we know will fail,
            # and that costs a turn. The deadline for spending one is public:
            # when the opponents run out of cards there is no legal ask left
            # at all, and the split must be named from whatever is known then.
            summary += (
                f" {', '.join(HALF_SUIT_NAMES[h] for h in stuck)} "
                f"{'is' if len(stuck) == 1 else 'are'} already ours and not "
                f"yet placeable, and no opponent can ask there again, so "
                f"nothing they do will place it. Only a deliberately failed "
                f"ask of our own can, and it costs the turn. The opponents "
                f"hold {left} card{'' if left == 1 else 's'} between them; "
                f"when that reaches zero there is no legal ask left and the "
                f"split must be named from whatever is known by then.")

    return {
        "is_dead": all_dead,
        "frozen": all_dead,
        "dead_half_suits": [
            {"half_suit": hs, "name": HALF_SUIT_NAMES[hs],
             "owner": "us" if t == ctx.my_team else "them",
             "placeable": hs not in stuck}
            for hs, t in dead],
        "unplaceable": [HALF_SUIT_NAMES[h] for h in stuck],
        #: cards the opponents still hold between them. Public, and it is the
        #: clock on the only channel that can still place a frozen half-suit.
        "opponent_cards": left,
        "summary": summary,
    }
