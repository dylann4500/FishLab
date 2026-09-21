"""Why the engine played what it played, captured where the numbers exist.

WHY THIS IS NOT THE ANALYSER. ``fish4.analyse.Analyser`` can rank the legal
asks from any position, and the site already uses it for the human's own
"Think" panel. It cannot explain a move the bot ALREADY MADE, because it
builds its own Posterior with its own RNG: over 14 mid-game positions, six
independent 480-draw passes over the same belief disagreed about the
top-ranked ask at 8 of them, with a median per-ask score spread of 0.122. A
post-hoc pass therefore explains a different sample than the one that moved,
and an explanation of a decision nobody made is worse than no explanation --
it is confidently wrong, and it is wrong exactly where the position is
interesting.

So a trace is taken INSIDE the decision, from the arrays the policy has
already computed. It costs no sampling and no extra search: at most a slice,
a sort that already happened, and a few float conversions.

THE COST OF BEING OFF. Tracing is opt-in and every builder here returns a
plain dict from values the caller already holds. No builder touches the RNG,
so a traced agent and an untraced one make bit-identical decisions from the
same seed -- which ``tests4/test_trace.py`` asserts rather than assumes,
because "surely it cannot change anything" is how a diagnostic becomes a
policy change nobody meant.

CARDS ARE NAMES. Every card in a trace is a name, never an index. This is the
same rule the v0.6 integration package enforces at its door, for the same
reason: an index is only meaningful next to the table that produced it, and a
trace is meant to be read by a person and shipped over a wire.
"""

from __future__ import annotations

from fish.cards import HALF_SUIT_NAMES, card_name, half_suit_cards

#: How many ranked candidates a trace carries. Five is what fits a panel
#: without scrolling, and the runner-up margin -- the only comparison most
#: readers actually want -- needs just two.
TOP_N = 5


def ask_trace(obs, asks, scores, p, order, pool, pick, *, top_n: int = TOP_N):
    """The ranked candidate list behind one ask.

    ``pick`` is the index the policy actually chose, AFTER the tie-break, so a
    reader sees the move that happened rather than the move that scored best.
    Those differ whenever the top group has more than one member, which is
    common: the objective genuinely cannot separate two cards of one half-suit
    at one target, and saying so is more honest than presenting the first as
    if it were preferred.
    """
    ranked = []
    for rank, i in enumerate(order[:top_n]):
        a = asks[i]
        ranked.append({
            "rank": rank,
            "target": int(a.target),
            "card": card_name(a.card),
            "half_suit": HALF_SUIT_NAMES[a.card // 6],
            "score": round(float(scores[i]), 4),
            "p_hit": round(float(p[i]), 4),
            "chosen": bool(i == pick),
        })
    best, second = order[0], (order[1] if len(order) > 1 else None)
    return {
        "kind": "ask",
        "n_legal": len(asks),
        "ranked": ranked,
        # The tie group is the honest unit of choice: within it the policy is
        # indifferent and picks at random, so a panel that implies a preference
        # inside it is inventing one.
        "tie_group": len(pool),
        "margin": (None if second is None
                   else round(float(scores[best] - scores[second]), 4)),
        "chosen": {"target": int(asks[pick].target),
                   "card": card_name(asks[pick].card)},
    }


def claim_trace(claim, *, why: str, confidence=None, alternatives=None):
    """A declaration, with the confidence the policy actually held.

    ``why`` separates the three ways a declaration happens -- chosen because it
    cleared the bar, forced because no ask was legal, or taken instead of an
    ask that could not land -- because they carry very different information
    about the engine. A forced declaration at low confidence is not a mistake;
    a voluntary one at low confidence is.
    """
    out = {
        "kind": "declare",
        "why": why,
        "half_suit": HALF_SUIT_NAMES[claim.half_suit],
        # Claim.assignment, not ClaimEvent.declared: this is the ACTION the
        # policy is about to take, before the engine has resolved it.
        "split": [{"card": card_name(c), "holder": int(h)}
                  for c, h in zip(half_suit_cards(claim.half_suit),
                                  claim.assignment)],
    }
    if confidence is not None:
        out["confidence"] = round(float(confidence), 4)
    if alternatives:
        out["alternatives"] = alternatives
    return out


def simple_trace(kind: str, **fields):
    """A decision with no ranking behind it: a pass, or an exact-solver move."""
    out = {"kind": kind}
    out.update(fields)
    return out


def marginals_trace(bel, obs, *, max_rows: int = 60):
    """Where the engine thinks the unresolved cards are.

    Only cards that are still in play and NOT already pinned to a single
    holder: a certainty is not a belief, and listing it pads the panel with
    rows that carry no information. Rows are ordered by how uncertain they
    are, so the top of the list is where the engine is actually guessing.
    """
    rows = []
    for hs, winner in enumerate(obs.set_winner):
        if winner is not None:
            continue
        for c in half_suit_cards(hs):
            m = bel.current_holder_mask(c)
            if m == 0 or not (m & (m - 1)):
                continue                    # gone, or already certain
            holders = [i for i in range(6) if m >> i & 1]
            rows.append({"card": card_name(c),
                         "half_suit": HALF_SUIT_NAMES[hs],
                         "candidates": holders})
    rows.sort(key=lambda r: -len(r["candidates"]))
    return {"rows": rows[:max_rows], "n_uncertain": len(rows)}
