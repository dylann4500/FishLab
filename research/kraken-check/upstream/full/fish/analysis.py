"""Offline strategy analytics over collected games.

IMPORTANT: this module analyzes finished games with full ground truth. That
is legitimate precisely because nothing here feeds back into acting
policies: it is the research microscope, not the player.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Optional

from .agents import AGENT_REGISTRY
from .cards import NUM_PLAYERS, half_suit_mask, num_half_suits, team_of
from .engine import AskEvent, ClaimEvent, GameState, PassEvent
from .observation import Observation
from .rules import RuleConfig


@dataclass
class GameRecord:
    seed: int
    initial_hands: list[int]
    history: tuple
    scores: tuple[int, int, int]
    starting_player: int
    variant: str = "54"


def _play_one(args) -> Optional[GameRecord]:
    specs, rules_dict, seed, agent_seed = args
    rules = RuleConfig.from_dict(rules_dict)
    state = GameState.deal(rules, seed=seed)
    initial = list(state.hands)
    agents = [AGENT_REGISTRY[name](**kw) for name, kw in specs]
    rng = random.Random(agent_seed)
    for p, ag in enumerate(agents):
        ag.begin_game(p, rules, rng.getrandbits(64))
    n = 0
    while not state.is_terminal:
        p = state.turn
        state.apply(p, agents[p].act(Observation.from_state(state, p)))
        n += 1
        if n > 20000:
            return None
    return GameRecord(seed, initial, tuple(state.history), state.scores(),
                      rules.starting_player, rules.variant)


def collect_games(specs, rules: RuleConfig, n_games: int, base_seed: int = 0,
                  n_jobs: int = 8) -> list[GameRecord]:
    srng = random.Random(base_seed ^ 0xA5A5)
    jobs = [(specs, {**rules.to_dict(), "starting_player": i % 6},
             base_seed + i, srng.getrandbits(64)) for i in range(n_games)]
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            recs = pool.map(_play_one, jobs, chunksize=4)
    else:
        recs = [_play_one(j) for j in jobs]
    return [r for r in recs if r is not None]


# ---------------------------------------------------------------------------
# Replay utilities
# ---------------------------------------------------------------------------

def replay_hands(rec: GameRecord):
    """Yield (index, event, hands_before) over the game."""
    hands = list(rec.initial_hands)
    for i, ev in enumerate(rec.history):
        yield i, ev, hands
        if isinstance(ev, AskEvent) and ev.success:
            bit = 1 << ev.card
            hands[ev.target] &= ~bit
            hands[ev.asker] |= bit
        elif isinstance(ev, ClaimEvent):
            m = half_suit_mask(ev.half_suit)
            for p in range(NUM_PLAYERS):
                hands[p] &= ~m


def team_hs_holdings(hands, hs: int) -> tuple[int, int]:
    m = half_suit_mask(hs)
    a = sum((hands[p] & m).bit_count() for p in (0, 2, 4))
    b = sum((hands[p] & m).bit_count() for p in (1, 3, 5))
    return a, b


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _mean_ci(xs):
    n = len(xs)
    if n == 0:
        return {"n": 0}
    mean = sum(xs) / n
    if n < 2:
        return {"n": n, "mean": mean, "ci95": (mean, mean)}
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    se = (var / n) ** 0.5
    return {"n": n, "mean": mean, "ci95": (mean - 1.96 * se, mean + 1.96 * se)}


def seat_advantage(records) -> dict:
    diffs = []
    for r in records:
        st = team_of(r.starting_player)
        a, b, _ = r.scores
        diffs.append((a - b) if st == 0 else (b - a))
    out = _mean_ci(diffs)
    out["metric"] = "set differential for the starting player's team"
    return out


def turn_retention(records) -> dict:
    """Ask-streak statistics: how many consecutive successful asks players
    chain before losing the turn."""
    streaks = []
    for r in records:
        cur = 0
        for ev in r.history:
            if isinstance(ev, AskEvent):
                if ev.success:
                    cur += 1
                else:
                    streaks.append(cur)
                    cur = 0
        if cur:
            streaks.append(cur)
    n = len(streaks)
    mean = sum(streaks) / n if n else 0.0
    dist = {}
    for s in streaks:
        dist[s] = dist.get(s, 0) + 1
    top = {k: dist[k] / n for k in sorted(dist)[:8]}
    return {"n_turn_possessions": n, "mean_cards_per_possession": mean,
            "streak_distribution": top}


def ask_success_by_phase(records, bins: int = 3) -> dict:
    tot = [[0, 0] for _ in range(bins)]
    for r in records:
        n = len(r.history)
        if not n:
            continue
        for i, ev in enumerate(r.history):
            if isinstance(ev, AskEvent):
                b = min(bins - 1, i * bins // n)
                tot[b][1] += 1
                tot[b][0] += 1 if ev.success else 0
    return {f"phase_{i}": {"success_rate": (t[0] / t[1] if t[1] else None),
                           "n": t[1]}
            for i, t in enumerate(tot)}


def completion_to_claim_delay(records) -> dict:
    """For each half-suit eventually WON: actions between the moment that
    team first held all six cards and the claim."""
    delays = []
    for r in records:
        n_hs = num_half_suits(r.variant)
        control_start = [None] * n_hs
        controller = [None] * n_hs
        for i, ev, hands in replay_hands(r):
            for hs in range(n_hs):
                a, b = team_hs_holdings(hands, hs)
                if a == 6 or b == 6:
                    t = 0 if a == 6 else 1
                    if controller[hs] != t:
                        controller[hs] = t
                        control_start[hs] = i
                else:
                    controller[hs] = None
                    control_start[hs] = None
            if isinstance(ev, ClaimEvent) and ev.winner in (0, 1):
                hs = ev.half_suit
                if controller[hs] == ev.winner and control_start[hs] is not None:
                    delays.append(i - control_start[hs])
    if not delays:
        return {"n": 0}
    delays.sort()
    n = len(delays)
    return {"n": n, "mean": sum(delays) / n, "median": delays[n // 2],
            "p90": delays[int(n * 0.9)]}


def specials_vs_normal(records) -> dict:
    """Does the 8s+jokers half-suit behave differently?"""
    order_specials, order_normal = [], []
    nulls_s = nulls_n = won_s = won_n = 0
    for r in records:
        claim_idx = 0
        for ev in r.history:
            if isinstance(ev, ClaimEvent):
                if ev.half_suit == 8:
                    order_specials.append(claim_idx)
                    if ev.winner == -1:
                        nulls_s += 1
                    else:
                        won_s += 1
                else:
                    order_normal.append(claim_idx)
                    if ev.winner == -1:
                        nulls_n += 1
                    else:
                        won_n += 1
                claim_idx += 1

    def _m(xs):
        return sum(xs) / len(xs) if xs else None

    return {
        "mean_resolution_rank_specials": _m(order_specials),
        "mean_resolution_rank_normal": _m(order_normal),
        "null_rate_specials": nulls_s / max(1, nulls_s + won_s),
        "null_rate_normal": nulls_n / max(1, nulls_n + won_n),
        "n_specials": nulls_s + won_s, "n_normal": nulls_n + won_n,
    }


def failed_ask_stats(records) -> dict:
    ta = tf = 0
    for r in records:
        for ev in r.history:
            if isinstance(ev, AskEvent):
                ta += 1
                if not ev.success:
                    tf += 1
    return {"total_asks": ta, "failed_share": tf / max(1, ta),
            "mean_asks_per_game": ta / max(1, len(records))}


def run_all(records) -> dict:
    return {
        "n_games": len(records),
        "seat_advantage": seat_advantage(records),
        "turn_retention": turn_retention(records),
        "ask_success_by_phase": ask_success_by_phase(records),
        "completion_to_claim_delay": completion_to_claim_delay(records),
        "specials_vs_normal": specials_vs_normal(records),
        "failed_asks": failed_ask_stats(records),
    }
