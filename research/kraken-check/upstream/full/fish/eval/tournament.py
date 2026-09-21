"""Paired-deal evaluation: luck-controlled matchups between agent policies.

The core variance-reduction device: every deal is played twice with the
teams swapped (X on seats 0/2/4, then X on 1/3/5), on the SAME hidden deal,
the same rotated starting seat, and the same agent RNG seed. Per-deal paired
set-differentials are then i.i.d. across deals, giving honest intervals.

Agent specs are (name, kwargs) so they can cross process boundaries.
"""

from __future__ import annotations

import math
import random
import secrets
import statistics
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Optional

from ..agents import AGENT_REGISTRY
from ..cards import deck_size
from ..rules import RuleConfig
from ..runner import GameTimeout, play_game

AgentSpec = tuple[str, dict]


def make_agent(spec: AgentSpec):
    name, kwargs = spec
    return AGENT_REGISTRY[name](**kwargs)


def _t_critical(df: int, conf: float = 0.95) -> float:
    """Two-sided Student-t critical value via the Cornish-Fisher expansion
    (accurate to well under 1% for df >= 4)."""
    if df <= 0:
        return float("inf")
    z = statistics.NormalDist().inv_cdf(0.5 + conf / 2)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


@dataclass
class MatchupStats:
    spec_x: AgentSpec
    spec_y: AgentSpec
    n_pairs: int = 0
    x_pair_wins: int = 0
    y_pair_wins: int = 0
    pair_ties: int = 0
    sum_diff: float = 0.0
    sum_diff_sq: float = 0.0
    x_sets: int = 0
    y_sets: int = 0
    null_sets: int = 0
    timeouts: int = 0
    incomplete_pairs: int = 0   # pairs dropped because a game timed out
    total_actions: int = 0
    per_deal_diffs: list = field(default_factory=list)

    def pair_score(self) -> float:
        """X's paired score in [0,1] (ties = 0.5)."""
        if self.n_pairs == 0:
            return 0.5
        return (self.x_pair_wins + 0.5 * self.pair_ties) / self.n_pairs

    def wilson_ci(self, z: float = 1.96) -> tuple[float, float]:
        n = self.n_pairs
        if n == 0:
            return (0.0, 1.0)
        p = self.pair_score()
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))

    def diff_mean_ci(self, conf: float = 0.95) -> tuple[float, float, float]:
        """Mean paired set-differential with a t-based CI (sample variance and
        a Student t critical value; z undercovers at small n)."""
        n = self.n_pairs
        if n == 0:
            return (0.0, 0.0, 0.0)
        mean = self.sum_diff / n
        if n < 2:
            return (mean, float("-inf"), float("inf"))
        var = max(0.0, (self.sum_diff_sq - n * mean * mean) / (n - 1))
        se = math.sqrt(var / n)
        t = _t_critical(n - 1, conf)
        return (mean, mean - t * se, mean + t * se)

    def summary(self) -> str:
        lo, hi = self.wilson_ci()
        m, mlo, mhi = self.diff_mean_ci()
        return (f"{self.spec_x[0]} vs {self.spec_y[0]}: "
                f"pair score {self.pair_score():.3f} [{lo:.3f},{hi:.3f}] "
                f"({self.x_pair_wins}W/{self.pair_ties}T/{self.y_pair_wins}L "
                f"of {self.n_pairs} pairs), "
                f"set diff {m:+.2f} [{mlo:+.2f},{mhi:+.2f}], "
                f"nulls {self.null_sets}, timeouts {self.timeouts}")

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("per_deal_diffs")
        d["pair_score"] = self.pair_score()
        d["wilson_ci"] = self.wilson_ci()
        d["diff_mean_ci"] = self.diff_mean_ci()
        return d


def _team_sets(state, team: int) -> int:
    a, b, _ = state.scores()
    return a if team == 0 else b


def _play_one_deal(args) -> dict:
    """Worker: one deal, two games (teams swapped)."""
    spec_x, spec_y, rules_dict, deal_seed, start_seat, agent_seed = args
    rng = random.Random(deal_seed)
    base_rules = RuleConfig.from_dict(rules_dict)
    deck = list(range(deck_size(base_rules.variant)))
    rng.shuffle(deck)
    rules = RuleConfig(**{**rules_dict, "starting_player": start_seat})
    rec = {"deal_seed": deal_seed, "timeout": 0, "actions": 0,
           "x_sets": 0, "y_sets": 0, "nulls": 0, "diff": 0, "complete": True}
    for swap in (0, 1):
        agents = []
        for seat in range(6):
            on_x_team = (seat % 2 == 0) if swap == 0 else (seat % 2 == 1)
            agents.append(make_agent(spec_x if on_x_team else spec_y))
        try:
            state = play_game(agents, rules, deck_order=deck,
                              agent_seed=agent_seed)
        except GameTimeout:
            rec["timeout"] += 1
            rec["complete"] = False
            continue
        x_team = 0 if swap == 0 else 1
        xs = _team_sets(state, x_team)
        ys = _team_sets(state, 1 - x_team)
        rec["x_sets"] += xs
        rec["y_sets"] += ys
        rec["nulls"] += state.scores()[2]
        rec["diff"] += xs - ys
        rec["actions"] += len(state.history)
    return rec


def play_matchup(spec_x: AgentSpec, spec_y: AgentSpec, n_deals: int,
                 rules: Optional[RuleConfig] = None, base_seed: int = 0,
                 n_jobs: int = 1, pool: Optional[Pool] = None,
                 agent_seed_base: Optional[int] = None) -> MatchupStats:
    rules = rules or RuleConfig()
    rules_dict = rules.to_dict()
    # Agent seeds come from a stream independent of the deck seeds so agent
    # randomness carries no information about the deal.
    seed_rng = random.Random(agent_seed_base if agent_seed_base is not None
                             else secrets.randbits(64))
    jobs = [(spec_x, spec_y, rules_dict, base_seed + i, i % 6,
             seed_rng.getrandbits(64)) for i in range(n_deals)]
    stats = MatchupStats(spec_x, spec_y)
    if pool is not None:
        records = pool.map(_play_one_deal, jobs, chunksize=4)
    elif n_jobs > 1:
        with Pool(n_jobs) as p:
            records = p.map(_play_one_deal, jobs, chunksize=4)
    else:
        records = [_play_one_deal(j) for j in jobs]
    for rec in records:
        stats.timeouts += rec["timeout"]
        if not rec["complete"]:
            # A half-played pair is not a paired observation; counting it
            # would bias the differential toward the completed side.
            stats.incomplete_pairs += 1
            continue
        stats.n_pairs += 1
        d = rec["diff"]
        stats.sum_diff += d
        stats.sum_diff_sq += d * d
        stats.per_deal_diffs.append(d)
        stats.x_sets += rec["x_sets"]
        stats.y_sets += rec["y_sets"]
        stats.null_sets += rec["nulls"]
        stats.total_actions += rec["actions"]
        if d > 0:
            stats.x_pair_wins += 1
        elif d < 0:
            stats.y_pair_wins += 1
        else:
            stats.pair_ties += 1
    return stats
