"""Feature extraction for learned policy/value models.

STRICT RULE: every belief-based feature is computed from an Observation plus
the agent's own BeliefState (itself derived only from public info and the
agent's own hand). Nothing here may touch GameState during action selection.
The leakage test suite asserts this by recomputing features from a
public-only reconstruction.

Two feature spaces live here, for two different jobs:

1. ``extract`` - belief features of a real, uncertain position. Used for
   learning a value function over information states.
2. ``extract_perfect`` - features of a FULLY KNOWN position, used to
   evaluate determinized search leaves. See the note above that function
   for why mixing the two was a measured failure.
"""

from __future__ import annotations

import numpy as np

from .cards import (NUM_PLAYERS, cards_per_player, half_suit_mask,
                    num_half_suits, team_of)
from .observation import Observation

HS_FEATURES = 12
GLOBAL_FEATURES = 14


def feature_size(variant: str = "54") -> int:
    return num_half_suits(variant) * HS_FEATURES + GLOBAL_FEATURES


def extract(obs: Observation, belief, worlds=None,
            n_worlds: int = 16, rng=None) -> np.ndarray:
    """Belief-space features for the acting player."""
    n_hs = obs.num_half_suits
    me = obs.player
    my_team = team_of(me)
    x = np.zeros(n_hs * HS_FEATURES + GLOBAL_FEATURES, dtype=np.float32)

    if worlds is None:
        worlds = [belief.sample_current_hands(rng) for _ in range(n_worlds)]
    nw = max(1, len(worlds))
    known = belief.known_current_hands()
    revealed = _revealed_suits(obs)

    for hs in range(n_hs):
        base = hs * HS_FEATURES
        m = half_suit_mask(hs)
        w = obs.set_winner[hs]
        if w is None:
            x[base + 3] = 1.0
        elif w == -1:
            x[base + 2] = 1.0
        elif w == my_team:
            x[base + 0] = 1.0
        else:
            x[base + 1] = 1.0
        x[base + 4] = (obs.hand & m).bit_count() / 6.0
        if w is None:
            mine = theirs = 0.0
            per_opp = [0.0] * NUM_PLAYERS
            for world in worlds:
                for p in range(NUM_PLAYERS):
                    k = (world[p] & m).bit_count()
                    if team_of(p) == my_team:
                        mine += k
                    else:
                        theirs += k
                        per_opp[p] += k
            x[base + 5] = mine / (6.0 * nw)
            x[base + 6] = theirs / (6.0 * nw)
            x[base + 7] = max(per_opp) / (6.0 * nw)
            spread = [per_opp[p] / nw for p in range(NUM_PLAYERS)
                      if team_of(p) != my_team]
            tot = sum(spread) or 1.0
            probs = [s / tot for s in spread if s > 0]
            x[base + 8] = (-sum(p * np.log(p) for p in probs) / np.log(3)
                           if probs else 0.0)
            x[base + 9] = sum((known[p] & m).bit_count()
                              for p in range(NUM_PLAYERS)
                              if team_of(p) == my_team) / 6.0
            x[base + 10] = sum((known[p] & m).bit_count()
                               for p in range(NUM_PLAYERS)
                               if team_of(p) != my_team) / 6.0
        x[base + 11] = 1.0 if hs in revealed else 0.0

    g = n_hs * HS_FEATURES
    per = cards_per_player(obs.rules.variant)
    for i in range(NUM_PLAYERS):
        x[g + i] = obs.hand_counts[(me + i) % NUM_PLAYERS] / per
    my_sets = sum(1 for w in obs.set_winner if w == my_team)
    their_sets = sum(1 for w in obs.set_winner if w == 1 - my_team)
    nulls = sum(1 for w in obs.set_winner if w == -1)
    x[g + 6] = my_sets / n_hs
    x[g + 7] = their_sets / n_hs
    x[g + 8] = nulls / n_hs
    x[g + 9] = min(1.0, len(obs.history) / 200.0)
    x[g + 10] = 1.0 if obs.turn == me else 0.0
    x[g + 11] = 1.0 if obs.must_claim() else 0.0
    x[g + 12] = (my_sets - their_sets) / n_hs
    x[g + 13] = obs.hand.bit_count() / per
    return x


# ---------------------------------------------------------------------------
# Perfect-information features (for evaluating DETERMINIZED search leaves)
# ---------------------------------------------------------------------------
#
# Motivation, measured: a value net trained on belief-derived features and
# then applied inside determinized search LOST 34-6 to its own prior. The
# cause is a train/inference distribution mismatch: inside a sampled world
# every card location is certain, so belief features (entropy, spread,
# expected share) take values the net never saw in training.
#
# The honest fix is a value function whose input distribution matches its
# use: a perfect-information evaluator, applied to a world sampled from the
# agent's OWN beliefs. This leaks nothing (the world is a hypothesis drawn
# from legal information) and is exactly the role a static evaluation plays
# in a determinized search.

PI_HS_FEATURES = 10
PI_GLOBAL_FEATURES = 12


def perfect_feature_size(variant: str = "54") -> int:
    return num_half_suits(variant) * PI_HS_FEATURES + PI_GLOBAL_FEATURES


def extract_perfect(hands, set_winner, turn: int, player: int,
                    variant: str = "54") -> np.ndarray:
    """Features of a FULLY KNOWN position, from ``player``'s team's view."""
    n_hs = num_half_suits(variant)
    per = cards_per_player(variant)
    my_team = team_of(player)
    x = np.zeros(n_hs * PI_HS_FEATURES + PI_GLOBAL_FEATURES, dtype=np.float32)
    for hs in range(n_hs):
        base = hs * PI_HS_FEATURES
        w = set_winner[hs]
        if w is None:
            x[base + 3] = 1.0
        elif w == -1:
            x[base + 2] = 1.0
        elif w == my_team:
            x[base + 0] = 1.0
        else:
            x[base + 1] = 1.0
        if w is not None:
            continue
        m = half_suit_mask(hs)
        mine = theirs = 0
        best_mine = best_theirs = 0
        holders_mine = holders_theirs = 0
        for p in range(NUM_PLAYERS):
            k = (hands[p] & m).bit_count()
            if not k:
                continue
            if team_of(p) == my_team:
                mine += k
                best_mine = max(best_mine, k)
                holders_mine += 1
            else:
                theirs += k
                best_theirs = max(best_theirs, k)
                holders_theirs += 1
        x[base + 4] = mine / 6.0
        x[base + 5] = theirs / 6.0
        x[base + 6] = 1.0 if mine == 6 else 0.0     # banked by us
        x[base + 7] = 1.0 if theirs == 6 else 0.0   # banked by them
        # concentration matters: a set spread over three teammates is harder
        # to claim correctly than one held by a single player
        x[base + 8] = (best_mine / 6.0) if mine else 0.0
        x[base + 9] = (holders_mine - holders_theirs) / 3.0
    g = n_hs * PI_HS_FEATURES
    for i in range(NUM_PLAYERS):
        x[g + i] = hands[(player + i) % NUM_PLAYERS].bit_count() / per
    my_sets = sum(1 for w in set_winner if w == my_team)
    their_sets = sum(1 for w in set_winner if w == 1 - my_team)
    nulls = sum(1 for w in set_winner if w == -1)
    live = sum(1 for w in set_winner if w is None)
    x[g + 6] = my_sets / n_hs
    x[g + 7] = their_sets / n_hs
    x[g + 8] = nulls / n_hs
    x[g + 9] = live / n_hs
    x[g + 10] = (my_sets - their_sets) / n_hs
    x[g + 11] = 1.0 if team_of(turn) == my_team else 0.0   # turn control
    return x


def _revealed_suits(obs: Observation) -> set[int]:
    from .engine import AskEvent
    out = set()
    for ev in obs.history:
        if isinstance(ev, AskEvent):
            if ev.asker == obs.player or (ev.success and ev.target == obs.player):
                out.add(ev.card // 6)
    return out


ASK_FEATURE_SIZE = 6


def ask_features(obs: Observation, ask, worlds, belief) -> np.ndarray:
    """Per-action features for a policy head (small, cheap)."""
    hs = ask.card // 6
    nw = max(1, len(worlds))
    hits = sum(1 for w in worlds if w[ask.target] & (1 << ask.card))
    m = half_suit_mask(hs)
    my_team = team_of(obs.player)
    team_share = 0.0
    for w in worlds:
        team_share += sum((w[p] & m).bit_count() for p in range(NUM_PLAYERS)
                          if team_of(p) == my_team)
    return np.array([
        hits / nw,                                   # P(success)
        (obs.hand & m).bit_count() / 6.0,            # my depth in the suit
        team_share / (6.0 * nw),                     # team control of suit
        obs.hand_counts[ask.target] / cards_per_player(obs.rules.variant),
        1.0 if hs in _revealed_suits(obs) else 0.0,  # info already leaked
        1.0 if belief.current_holder_mask(ask.card) == (1 << ask.target) else 0.0,
    ], dtype=np.float32)
