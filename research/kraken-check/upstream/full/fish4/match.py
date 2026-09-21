"""Duplicate-deal matches spanning v0.3 and v0.4 agents.

A thin, self-contained copy of the v0.3 pairing protocol whose worker function
lives in a module that imports ``registry4``. That matters on Windows, where
multiprocessing uses spawn: the worker re-imports the module holding the
function it was given, so the v0.4 agents are only constructible in a worker if
the worker's entry point is here rather than in ``fish.eval.tournament``.

Pairing protocol (unchanged from v0.3, because it is correct): every deal is
played twice with the teams swapped, on the same cards, the same rotated
starting seat, and the same agent RNG seed. Only the policy assignment differs,
so per-pair set differentials are i.i.d. across deals.
"""

from __future__ import annotations

import math
import random
import secrets
import statistics
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Optional

import random as _random
import secrets as _secrets

from fish.cards import deck_size, team_of
from fish.engine import NULL_TEAM, ClaimEvent, GameState
from fish.observation import Observation
from fish.rules import RuleConfig

from .registry4 import make_agent


def side_seeds(agent_seed: int, side: int, n: int = 3) -> list:
    """``n`` agent seeds for one side of the match, derived from the pair seed.

    Used only when ``independent_seeds`` is on. Domain-separated by side so the
    two sides draw from streams that cannot coincide.
    """
    r = _random.Random((int(agent_seed) << 1) | (side & 1))
    return [r.getrandbits(64) for _ in range(n)]


def play_capped(agents, rules, deck_order, agent_seed, max_actions=4000,
                seat_seeds=None):
    """Play a game, returning ``(state, timed_out)`` instead of raising.

    WHY NOT ``fish.runner.play_game``: it raises on the action cap, and the v0.3
    harness then DROPPED the whole deal-pair. An audit of that behaviour found it
    biases the estimate rather than merely losing precision - on 60 replayed
    deals the dropped pairs had a true mean differential of -7.14 against -10.13
    for the kept ones, and at one cap the reported 95% interval excluded the true
    value. Dropping a pair conditions on the outcome, which is exactly what a
    paired design exists to avoid.

    So a capped game is scored on the half-suits actually resolved, with the
    unresolved ones counting for nobody. That is the same semantics the exact
    solver gives an unbroken cycle ("if play never progresses, neither side
    scores again"), so it is the consistent choice as well as the unbiased one.
    """
    state = GameState.deal(rules, deck_order=deck_order)
    state.agent_seed = agent_seed
    agent_rng = _random.Random(agent_seed)
    for p, agent in enumerate(agents):
        agent.begin_game(p, rules, seat_seeds[p] if seat_seeds is not None
                         else agent_rng.getrandbits(64))
    poll = rules.claims_any_time
    n = 0
    while not state.is_terminal:
        p = state.turn
        action = agents[p].act(Observation.from_state(state, p))
        state.apply(p, action)
        n += 1
        if poll:
            n += _poll_offturn_claims(state, agents, max_actions - n)
        if n >= max_actions:
            return state, True
    return state, False


def _deduced_claim(agent, seat, obs):
    """A claim this seat can make with certainty, from ANY belief-tracking agent.

    Duck-typed on ``agent.bel`` rather than on a v0.4-only method, so that a v0.3
    policy gets exactly the same out-of-turn opportunity in a mixed match. Giving
    the new generation a capability the old one lacks would make every
    ``claims_any_time`` comparison a comparison of two things at once.
    """
    from fish.cards import half_suit_cards, team_of
    from fish.engine import Claim
    bel = getattr(agent, "bel", None)
    if bel is None:
        return None
    try:
        bel.update(obs)
    except Exception:
        return None
    my_team = team_of(seat)
    for hs in range(len(obs.set_winner)):
        if obs.set_winner[hs] is not None:
            continue
        assignment = []
        for c in half_suit_cards(hs):
            m = bel.current_holder_mask(c)
            if m == 0 or m & (m - 1):
                break
            holder = m.bit_length() - 1
            if team_of(holder) != my_team:
                break
            assignment.append(holder)
        else:
            return Claim(hs, tuple(assignment))
    return None


def _poll_offturn_claims(state, agents, budget) -> int:
    """Let seats declare outside their own turn, as the printed rules allow.

    v0.3's loop only ever asked the player on move, so ``claims_any_time`` was a
    rule the engine accepted but no policy could ever exercise. That is a real
    gap rather than a cosmetic one: a seat with no cards left never gets a turn
    of its own while a teammate still has cards, so under an own-turn-only loop
    it can never declare a set it has fully deduced.

    Only the deductive test is run here (see ``FishBot4.certain_claim``), so the
    poll costs a belief update rather than a posterior per seat.
    """
    used = 0
    changed = True
    while changed and used < budget:
        changed = False
        for q in range(6):
            if state.is_terminal or q == state.turn:
                continue
            claim = _deduced_claim(agents[q], q,
                                   Observation.from_state(state, q))
            if claim is None:
                continue
            try:
                state.apply(q, claim)
            except Exception:
                continue
            used += 1
            changed = True
            if used >= budget:
                break
    return used


def _t_critical(df: int, conf: float = 0.95) -> float:
    if df <= 0:
        return float("inf")
    z = statistics.NormalDist().inv_cdf(0.5 + conf / 2)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


@dataclass
class Result:
    spec_x: tuple
    spec_y: tuple
    diffs: list = field(default_factory=list)
    x_wins: int = 0
    y_wins: int = 0
    ties: int = 0
    nulls: int = 0
    x_nulls: int = 0
    y_nulls: int = 0
    x_gifts: int = 0        # sets handed to the opponents by X's own claim
    y_gifts: int = 0
    #: gifts whose half-suit was wholly on the declarer's team but misordered
    #: -- the case the misdeclaration rule governs. Under the baseline
    #: (opponent-award) these are a subset of gifts; under the legacy null
    #: variant they appear as nulls instead and this stays 0.
    x_misdeclares: int = 0
    y_misdeclares: int = 0
    #: the rules the games were actually played under, so an archived record
    #: is self-describing. Every row in results/v04_duels.jsonl written
    #: before this field existed was played under the legacy null variant.
    rules: Optional[dict] = None
    timeouts: int = 0
    dropped_pairs: int = 0
    actions: int = 0
    seconds: float = 0.0

    @property
    def n(self) -> int:
        return len(self.diffs)

    def pair_score(self) -> float:
        return (self.x_wins + 0.5 * self.ties) / self.n if self.n else 0.5

    def wilson(self, z: float = 1.96) -> tuple[float, float]:
        n = self.n
        if not n:
            return (0.0, 1.0)
        p = self.pair_score()
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (max(0.0, c - h), min(1.0, c + h))

    def diff_ci(self, conf: float = 0.95) -> tuple[float, float, float]:
        n = self.n
        if not n:
            return (0.0, 0.0, 0.0)
        m = sum(self.diffs) / n
        if n < 2:
            return (m, float("-inf"), float("inf"))
        var = statistics.variance(self.diffs)
        se = math.sqrt(var / n)
        t = _t_critical(n - 1, conf)
        return (m, m - t * se, m + t * se)

    def summary(self) -> str:
        lo, hi = self.wilson()
        m, ml, mh = self.diff_ci()
        return (f"{self.spec_x[0]} vs {self.spec_y[0]}: "
                f"score {self.pair_score():.3f} [{lo:.3f},{hi:.3f}] "
                f"({self.x_wins}W/{self.ties}T/{self.y_wins}L of {self.n}), "
                f"diff {m:+.3f} [{ml:+.3f},{mh:+.3f}], "
                f"nulls {self.nulls} (X {self.x_nulls}/Y {self.y_nulls}), "
                f"gifts (X {self.x_gifts}/Y {self.y_gifts}), "
                f"timeouts {self.timeouts}, "
                f"dropped {self.dropped_pairs}, {self.seconds:.0f}s")

    def to_dict(self) -> dict:
        m, ml, mh = self.diff_ci()
        lo, hi = self.wilson()
        return {"spec_x": list(self.spec_x), "spec_y": list(self.spec_y),
                "n_pairs": self.n, "pair_score": self.pair_score(),
                "wilson_ci": [lo, hi], "diff_mean": m, "diff_ci": [ml, mh],
                "nulls": self.nulls, "x_nulls": self.x_nulls,
                "y_nulls": self.y_nulls, "x_gifts": self.x_gifts,
                "y_gifts": self.y_gifts,
                "x_misdeclares": self.x_misdeclares,
                "y_misdeclares": self.y_misdeclares,
                "rules": self.rules, "timeouts": self.timeouts,
                "dropped_pairs": self.dropped_pairs,
                "actions": self.actions, "seconds": self.seconds,
                # The per-pair differentials themselves. Without these a run's
                # raw observations are unrecoverable and any re-analysis - a
                # different interval, a variance decomposition, a pooling across
                # runs - needs the whole run repeated. They are the actual
                # measurement; everything else in this record is a summary of
                # them. A 500-pair run costs about 2 kB.
                "diffs": list(self.diffs)}


def _one_deal(args) -> dict:
    spec_x, spec_y, rules_dict, deal_seed, start_seat, agent_seed = args[:6]
    independent = bool(args[6]) if len(args) > 6 else False
    rng = random.Random(deal_seed)
    base = RuleConfig.from_dict(rules_dict)
    deck = list(range(deck_size(base.variant)))
    rng.shuffle(deck)
    rules = RuleConfig(**{**rules_dict, "starting_player": start_seat})
    rec = {"diff": 0, "nulls": 0, "x_nulls": 0, "y_nulls": 0,
           "x_gifts": 0, "y_gifts": 0,
           "x_misdeclares": 0, "y_misdeclares": 0,
           "timeout": 0, "actions": 0, "complete": True}
    xs_seeds = side_seeds(agent_seed, 0) if independent else None
    ys_seeds = side_seeds(agent_seed, 1) if independent else None
    for swap in (0, 1):
        agents = []
        seat_seeds = [] if independent else None
        for seat in range(6):
            on_x = (seat % 2 == 0) if swap == 0 else (seat % 2 == 1)
            agents.append(make_agent(spec_x if on_x else spec_y))
            if independent:
                # Seeded by position WITHIN THE SIDE, not by seat, so a policy
                # carries the same randomness into both halves of the pair while
                # the two sides stay on streams that never coincide. Without
                # this an A/A is identical in both swaps and its differential is
                # exactly zero for every deal - see the docstring of
                # play_matchup.
                seat_seeds.append((xs_seeds if on_x else ys_seeds)[seat // 2])
        st, timed_out = play_capped(agents, rules, deck, agent_seed,
                                    seat_seeds=seat_seeds)
        if timed_out:
            rec["timeout"] += 1
        a, b, nulls = st.scores()
        x_team = 0 if swap == 0 else 1
        xs, ys = (a, b) if swap == 0 else (b, a)
        rec["diff"] += xs - ys
        rec["nulls"] += nulls
        rec["actions"] += len(st.history)
        # Attribute each nulled or gifted half-suit to the side whose claim
        # caused it. A null is a bookkeeping failure and a gift is worse, so
        # which policy produced them is a diagnostic, not a footnote.
        for ev in st.history:
            if not isinstance(ev, ClaimEvent):
                continue
            claimer_is_x = (team_of(ev.claimer) == x_team)
            if ev.winner == NULL_TEAM:
                rec["x_nulls" if claimer_is_x else "y_nulls"] += 1
            elif team_of(ev.claimer) != ev.winner:
                rec["x_gifts" if claimer_is_x else "y_gifts"] += 1
                # A gift whose cards were all on the declarer's own team is a
                # misdeclaration: the case the misdeclaration rule governs,
                # kept separate so the diagnostic keeps meaning what it meant
                # across the rule change (a plain gift named a card the
                # opponents actually held).
                if all(team_of(h) == team_of(ev.claimer)
                       for h in ev.revealed):
                    rec["x_misdeclares" if claimer_is_x
                        else "y_misdeclares"] += 1
    return rec


def play_matchup(spec_x, spec_y, n_deals: int, rules: Optional[RuleConfig] = None,
                 base_seed: int = 0, n_jobs: int = 4,
                 agent_seed_base: Optional[int] = None,
                 progress: bool = False,
                 independent_seeds: bool = False) -> Result:
    """Duplicate-deal matchup of two agent specs.

    ``independent_seeds`` gives each SIDE its own agent-randomness stream rather
    than seeding by seat. It is off by default because turning it on changes
    every number this harness produces, and the archive in
    ``results/v04_duels.jsonl`` was built without it.

    It exists because the default makes the harness unable to measure its own
    null. With seat seeding, an A/A - one spec against a copy of itself - plays a
    bit-identical game in both swaps, so its differential is ``(a-b) + (b-a) = 0``
    for every deal, exactly, and no amount of sampling reveals anything about the
    harness's variance. With independent streams the two sides break ties
    differently, the differential becomes a real random variable with mean zero,
    and a null distribution can actually be measured.
    """
    rules = rules or RuleConfig()
    rd = rules.to_dict()
    seed_rng = random.Random(agent_seed_base if agent_seed_base is not None
                             else secrets.randbits(64))
    jobs = [(spec_x, spec_y, rd, base_seed + i, i % 6, seed_rng.getrandbits(64),
             independent_seeds) for i in range(n_deals)]
    res = Result(spec_x, spec_y)
    # Self-describing archive: starting_player is per-deal (i % 6 above), so
    # it is stripped rather than stamped with a misleading single value.
    res.rules = {k: v for k, v in rd.items() if k != "starting_player"}
    t0 = time.time()
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            recs = []
            for i, r in enumerate(pool.imap_unordered(_one_deal, jobs, chunksize=2)):
                recs.append(r)
                if progress and (i + 1) % 25 == 0:
                    print(f"  ... {i+1}/{n_deals} pairs "
                          f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
    else:
        recs = [_one_deal(j) for j in jobs]
    for r in recs:
        res.timeouts += r["timeout"]
        if not r["complete"]:
            res.dropped_pairs += 1
            continue
        d = r["diff"]
        res.diffs.append(d)
        res.nulls += r["nulls"]
        res.x_nulls += r["x_nulls"]
        res.y_nulls += r["y_nulls"]
        res.x_gifts += r["x_gifts"]
        res.y_gifts += r["y_gifts"]
        res.x_misdeclares += r.get("x_misdeclares", 0)
        res.y_misdeclares += r.get("y_misdeclares", 0)
        res.actions += r["actions"]
        if d > 0:
            res.x_wins += 1
        elif d < 0:
            res.y_wins += 1
        else:
            res.ties += 1
    res.seconds = time.time() - t0
    return res
