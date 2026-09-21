"""Common-random-numbers paired rollout evaluation of candidate asks.

WHAT THIS COMPUTES
------------------
Given a real position (rules, whose turn it is, which half-suits are resolved),
the acting seat, and K determinized worlds drawn from THAT SEAT'S OWN posterior,
this module returns a matrix ``V`` of shape ``(n_candidates, K)``:

    V[a, k] = acting team's set differential gained from world k after playing
              candidate ask ``a`` and then letting a fixed continuation policy
              finish the game.

Two properties are enforced, and they are the whole point of the module.

**Common random numbers.** World ``k`` is identical for every candidate, and
the six continuation policies are seeded identically for every candidate on
world ``k``. Only the candidate ask differs. v0.3 measured that allocating
different sampled worlds to different candidates is what destroyed determinized
search (noise-to-signal 2.4); the paired differences ``V[a,k] - V[a',k]``
computed here share the world and the seeds, so the position's own value level
- which is the dominant variance term - cancels exactly.

**No information leaks into the continuation.** Each rollout seat is handed an
``Observation`` for its own seat and nothing else, so it sees its own hand in
the determinized world plus the public log. Nothing inside a rollout consults
the world it is being played in; the world only generates observations. That is
what separates determinization from strategy fusion, and it holds under both
continuations below. The knowledge set of the public one is spelled out on
``PublicInfoHeuristic`` and audited by ``tests4/test_learn.py``.

WHY THE WORLDS COME FROM OUTSIDE
--------------------------------
Worlds are drawn at harvest time, inside the game, from the acting agent's own
live ``BeliefState`` (see ``dataset.py``), and stored with the position. That is
the leak-free choice: the worlds are a function of the acting seat's own
information only.

The original reason given here was stronger and was wrong. It said a posterior
"cannot be rebuilt from a stored position after the fact", because
``fish.beliefs.BeliefState`` is anchored on the initial deal. It is anchored on
the initial deal, but it does not have to be handed one: given the current hand
and the real public history, ``initial_hand()`` back-computes a deal consistent
with both, and the tracker attaches to that. ``scripts4/ask_regret.py`` does
exactly this and ``scripts4/rollout_target.py`` runs it over 110 positions
without a single ``BeliefContradiction``. Harvest-time sampling is therefore a
convenience, not a necessity, and it is kept because the stored worlds make
every rollout batch reproducible from the file alone.

WHY THE CONTINUATION POLICY IS WHAT IT IS
-----------------------------------------
For the same reason, the claim that once stood here - that the strong v0.4
policy cannot be used as a continuation - is false, and it mattered. The paper
diagnosed the failure of the whole objective-learning line on the continuation
being weak, and that diagnosis was correct: with the full v0.4 policy finishing
the rollout, position-centred value rises by +0.681 +/- 0.142 sets across the
range of P(success), against +0.101 for the public-information heuristic
(``results/rollout_target.json``, 110 positions, robustness in
``results/rollout_target_robust.json``). The target the regression could not
learn from was flat because of what finished the game, and finishing it properly
un-flattens it.

``RolloutConfig.policy`` therefore selects the continuation, and
``POLICY_PUBLIC`` remains the default so that every rollout file already on disk
still means what it meant. ``POLICY_V04`` requires ``seed_history=True``: the
strong policy attaches to the position through the real public log, so a
configuration that withheld it would silently anchor the tracker on a game that
never happened. That combination is refused rather than run.

``POLICY_V04`` ALSO CONSTRAINS THE WORLDS, and the constraint is stronger than
it looks. The public continuation accepts any determinized world that preserves
hand counts, because it has no model to contradict. The strong one has six of
them: every seat rebuilds an initial deal from its own hand plus the public log,
so a world must satisfy everything that log implies -- cards a successful ask
located sit with their holder, and every FAILED ask asserts, under the no-bluff
rule, that the asker holds at least one card of that half-suit. A count-
preserving shuffle violates the second constraint routinely. ``BeliefState``
enforces both, so worlds for this continuation must come from the belief
sampler, which is what ``dataset.py`` stores at harvest time anyway. A world
that does not qualify raises ``BeliefContradiction`` from inside the rollout,
loudly, and ``tests4/test_learn.py`` pins that it keeps doing so.

The public continuation and its knobs, below, are unchanged.
``fish.agents.heuristic.HeuristicAgent`` is the
documented starting point - it tracks only publicly-established card locations -
but it is slow to terminate, because it will happily shuttle a card back and
forth until its stall rule fires. ``PublicInfoHeuristic`` therefore offers two
optional changes: a shorter stall window, and the one piece of public
information the heuristic throws away (a failed ask proves the target did not
hold that card, until they are publicly seen to acquire it).

**Both options are OFF by default, because measuring them said so.** On 9
harvested positions at K=8 the four configurations gave, in signal variance
recovered per position (variance of the paired difference minus the variance the
common random numbers leave behind), 0.76 for the plain heuristic (window 80, no
negative information), roughly 0 for window 40 with or without it, and 0.43 for
window 25 with it. Shortening the window makes the continuation policy gamble a
forced claim before the ask under evaluation has had time to matter, which is
exactly the quiescence failure v0.3 hit from the other direction. The negative
information additionally *hurt* the common-random-numbers coupling
(``crn_var_ratio`` 0.90 against 0.57), because a sharper policy reacts more to
the one public event that distinguishes the candidates. The plain heuristic is
therefore the default, and the two knobs stay as ablation handles.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from fish.agents.base import Agent
from fish.agents.heuristic import HeuristicAgent
from fish.cards import NUM_PLAYERS, half_suit_mask, team_of
from fish.engine import (NULL_TEAM, Ask, AskEvent, ClaimEvent, GameState,
                         IllegalAction)
from fish.observation import Observation
from fish.rules import RuleConfig

#: Value returned for a candidate whose ask turns out to be illegal in the
#: determinized world. That should never happen - ask legality is a function of
#: public information plus the asker's own hand, both of which every world
#: reproduces - so it is a loud sentinel rather than a silent zero.
ILLEGAL_VALUE = -99.0


# ---------------------------------------------------------------------------
# Continuation policy
# ---------------------------------------------------------------------------

class PublicInfoHeuristic(HeuristicAgent):
    """The rollout continuation policy. Public information only.

    KNOWLEDGE SET, exhaustively
    ---------------------------
    Everything this policy consults is reachable from the single
    ``Observation`` it is handed each turn:

    1. ``obs.hand``            - its own hand in the determinized world;
    2. ``obs.history``         - the public event log OF THE ROLLOUT ONLY. The
                                 determinized ``GameState`` is built with an
                                 empty history, so no event of the real game is
                                 visible here;
    3. ``obs.hand_counts``, ``obs.set_winner``, ``obs.turn``, ``obs.rules``
                               - public derived state (SPEC.md section 5);
    4. its own RNG, seeded from the rollout seed, which is independent of the
       deal.

    It never receives a ``GameState`` and holds no reference to one. Derived
    state it caches:

    * ``_public_loc``  card -> holder, set only from successful asks in the
                       rollout log, from claims, and from its own hand;
    * ``_not_holding`` card -> seats publicly proven not to hold it. A failed
                       ask proves the target did not hold the card; the only
                       way a seat can subsequently acquire it is a successful
                       ask by that seat, which is public and resets the entry.
                       Under baseline rules the asker also cannot hold the card
                       they asked for.

    Both caches are audited in ``tests4/test_learn.py`` against the set of cards
    actually named by public events plus the policy's own hand.
    """

    name = "rollout-public"

    def __init__(self, stall_window: int = 80, use_negative: bool = False):
        super().__init__()
        self.stall_window = stall_window
        self.use_negative = use_negative

    def begin_game(self, player, rules, seed):
        super().begin_game(player, rules, seed)
        self._not_holding: dict[int, set[int]] = {}
        self._neg_cursor = 0

    # -- bookkeeping -------------------------------------------------------

    def stalled(self, obs, window: Optional[int] = None) -> bool:  # type: ignore[override]
        # HeuristicAgent.act calls ``self.stalled(obs)`` with the base class's
        # default window of 80. Overriding it as an instance method is how the
        # window becomes a parameter without touching ``fish/``.
        return Agent.stalled(obs, self.stall_window if window is None else window)

    def _sync_negative(self, obs: Observation) -> None:
        h = obs.history
        while self._neg_cursor < len(h):
            ev = h[self._neg_cursor]
            if isinstance(ev, AskEvent):
                if ev.success:
                    # Publicly located with the asker: everybody else is out.
                    self._not_holding[ev.card] = {
                        p for p in range(NUM_PLAYERS) if p != ev.asker}
                else:
                    s = self._not_holding.setdefault(ev.card, set())
                    s.add(ev.target)
                    s.add(ev.asker)   # baseline rules forbid asking for a held card
            elif isinstance(ev, ClaimEvent):
                for i in range(6):
                    self._not_holding.pop(ev.half_suit * 6 + i, None)
            self._neg_cursor += 1

    # -- policy ------------------------------------------------------------

    def act(self, obs: Observation):
        self._sync(obs)
        if self.use_negative:
            self._sync_negative(obs)
        if obs.must_pass():
            return self._pass(obs)
        claim = self._claim_publicly_certain(obs)
        if claim is not None:
            return claim
        asks = obs.legal_asks()
        if not asks or (self.stalled(obs) and obs.claimable_half_suits()):
            return self._forced_claim(obs)
        steals = [a for a in asks if self._public_loc.get(a.card) == a.target]
        if steals:
            return self.rng.choice(steals)
        if self.use_negative:
            plausible = [a for a in asks
                         if a.target not in self._not_holding.get(a.card, ())]
        else:
            plausible = [a for a in asks
                         if self._public_loc.get(a.card, a.target) == a.target]
        pool0 = plausible or asks

        def suit_strength(a: Ask) -> int:
            return (obs.hand & half_suit_mask(a.card // 6)).bit_count()

        best = max(suit_strength(a) for a in pool0)
        pool = [a for a in pool0 if suit_strength(a) == best]
        return self.rng.choice(pool)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: The strong continuation. Fixed by name so that a rollout file records which
#: policy produced it and a later run cannot silently mean something else.
V04_SPEC = ("fishbot4", {"opponent_gamma": 0.35})

POLICY_PUBLIC = "rollout-public"
POLICY_V04 = "v04"


@dataclass(frozen=True)
class RolloutConfig:
    """Everything that makes a rollout batch reproducible."""

    #: hard cap on actions inside one rollout; a hit is counted and reported.
    max_actions: int = 900
    #: stall window of the continuation policy (see PublicInfoHeuristic).
    stall_window: int = 80
    #: whether the continuation policy uses negative information.
    use_negative: bool = False
    #: which continuation finishes the rollout: ``POLICY_PUBLIC`` (the
    #: incumbent, public information only) or ``POLICY_V04`` (the full engine).
    policy: str = POLICY_PUBLIC
    #: whether the determinized state is seeded with the REAL public history of
    #: the position, rather than starting from an empty log.
    seed_history: bool = False

    def __post_init__(self) -> None:
        if self.policy not in (POLICY_PUBLIC, POLICY_V04):
            raise ValueError(f"unknown continuation policy {self.policy!r}")
        if self.policy == POLICY_V04 and not self.seed_history:
            # Not a taste question. The belief tracker anchors on an initial
            # deal it back-computes from (current hand, public history); with an
            # empty history it would anchor on the determinized mid-game hand as
            # if it were the deal, and every inference downstream would be drawn
            # from a game that never happened. Refuse rather than produce it.
            raise ValueError(
                "policy='v04' requires seed_history=True: the belief tracker "
                "has nothing to attach to without the real public history")

    def to_dict(self) -> dict:
        return {"max_actions": self.max_actions,
                "stall_window": self.stall_window,
                "use_negative": self.use_negative,
                "seed_history": self.seed_history,
                "policy": (PublicInfoHeuristic.name
                           if self.policy == POLICY_PUBLIC else self.policy)}

    @classmethod
    def from_dict(cls, d: dict) -> "RolloutConfig":
        pol = d.get("policy", POLICY_PUBLIC)
        if pol == PublicInfoHeuristic.name:
            pol = POLICY_PUBLIC
        return cls(max_actions=int(d.get("max_actions", 900)),
                   stall_window=int(d.get("stall_window", 80)),
                   use_negative=bool(d.get("use_negative", False)),
                   policy=pol,
                   seed_history=bool(d.get("seed_history",
                                           pol == POLICY_V04)))


@dataclass
class RolloutStats:
    rollouts: int = 0
    actions: int = 0
    cap_hits: int = 0
    illegal: int = 0

    def to_dict(self) -> dict:
        return {"rollouts": self.rollouts, "actions": self.actions,
                "cap_hits": self.cap_hits, "illegal": self.illegal}

    def merge(self, other: dict) -> None:
        self.rollouts += other.get("rollouts", 0)
        self.actions += other.get("actions", 0)
        self.cap_hits += other.get("cap_hits", 0)
        self.illegal += other.get("illegal", 0)


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------

def banked_differential(set_winner: Sequence, my_team: int) -> int:
    """Set differential already resolved at the position, from ``my_team``."""
    a = sum(1 for w in set_winner if w == 0)
    b = sum(1 for w in set_winner if w == 1)
    return (a - b) if my_team == 0 else (b - a)


def _seat_seeds(world_seed: int) -> list[int]:
    """Six continuation seeds for one world, shared by every candidate."""
    rng = random.Random(world_seed)
    return [rng.getrandbits(64) for _ in range(NUM_PLAYERS)]


def _make_agents(cfg: RolloutConfig) -> list:
    """Six continuation agents, one per seat.

    ``fish4.registry4`` is imported here rather than at module scope: it pulls
    in the whole engine, and a rollout file written by the public continuation
    must stay loadable without it.
    """
    if cfg.policy == POLICY_V04:
        from fish4.registry4 import make_agent
        return [make_agent(V04_SPEC) for _ in range(NUM_PLAYERS)]
    return [PublicInfoHeuristic(stall_window=cfg.stall_window,
                                use_negative=cfg.use_negative)
            for _ in range(NUM_PLAYERS)]


def _play_out(state: GameState, seat_seeds: Sequence[int],
              cfg: RolloutConfig, stats: RolloutStats) -> None:
    agents = _make_agents(cfg)
    for p, ag in enumerate(agents):
        ag.begin_game(p, state.rules, seat_seeds[p])
    n = 0
    while not state.is_terminal and n < cfg.max_actions:
        p = state.turn
        # THE information boundary: a per-seat Observation, never the state.
        action = agents[p].act(Observation.from_state(state, p))
        state.apply(p, action)
        n += 1
    stats.actions += n
    if not state.is_terminal:
        stats.cap_hits += 1


def _differential(state: GameState, my_team: int) -> int:
    a, b, _ = state.scores()
    return (a - b) if my_team == 0 else (b - a)


#: Distinguishes "the caller did not pass a history" from "the caller passed an
#: empty one". They are the same value and opposite claims: the first is an
#: omission, the second is an assertion that the game has not started.
_NO_HISTORY = object()


def rollout_matrix(rules: RuleConfig, turn: int, set_winner: Sequence,
                   seat: int, worlds: Sequence[Sequence[int]],
                   asks: Sequence[Ask], seed: int,
                   cfg: Optional[RolloutConfig] = None,
                   stats: Optional[RolloutStats] = None,
                   history: Sequence = _NO_HISTORY) -> np.ndarray:
    """``V[a, k]``: set differential GAINED by the acting team, in world k.

    The value is incremental - the differential already banked at the position
    is subtracted - so values are comparable across positions. Within a position
    the subtraction is a constant and cancels in every pairwise difference
    anyway.

    ``worlds[k]`` is a list of six current-hand bitmasks, as returned by
    ``fish4.posterior.Posterior.worlds()``. The same ``seed`` reproduces the
    matrix exactly, and world k uses the same six continuation seeds for every
    candidate - that is the common-random-numbers guarantee.
    """
    cfg = cfg or RolloutConfig()
    if cfg.policy == POLICY_V04 and history is _NO_HISTORY:
        # RolloutConfig.__post_init__ refuses policy='v04' with
        # seed_history=False, which reads like the guard is in place. It is
        # not: it checks the FLAG, while `history` used to default to (). So
        # policy='v04' with the flag set and the argument forgotten seeded an
        # empty log, anchored six belief trackers on a determinized mid-game
        # hand as though it were the deal, and returned an ordinary-looking
        # number computed from a game that could not have happened. The engine
        # often notices -- the back-computed deal is usually infeasible and
        # BeliefContradiction fires -- but "usually" is not a guarantee, and a
        # world that happens to be consistent produces a wrong number quietly.
        #
        # Inferring the omission from the position does not work: 20 plies of
        # asking with no claim leaves every hand full and no set decided, so it
        # is indistinguishable from an opening. A sentinel makes "not passed"
        # exactly detectable, which is the thing actually worth refusing. An
        # empty history explicitly passed is a caller's assertion that this is
        # ply zero, and is honoured.
        raise ValueError(
            "policy='v04' needs the real public history: pass history=... "
            "(explicitly history=() at an opening position). Without it the "
            "belief tracker anchors on a mid-game hand as if it were the deal")
    if history is _NO_HISTORY:
        history = ()
    stats = stats if stats is not None else RolloutStats()
    my_team = team_of(seat)
    base = banked_differential(set_winner, my_team)
    V = np.zeros((len(asks), len(worlds)), dtype=np.float64)
    seed_rng = random.Random(seed)
    world_seeds = [seed_rng.getrandbits(64) for _ in range(len(worlds))]
    for k, world in enumerate(worlds):
        seat_seeds = _seat_seeds(world_seeds[k])
        for i, ask in enumerate(asks):
            state = GameState.from_components(rules, list(world), turn,
                                              list(set_winner))
            if cfg.seed_history:
                state.history = list(history)
            try:
                state.apply(seat, ask)
            except IllegalAction:
                stats.illegal += 1
                V[i, k] = ILLEGAL_VALUE
                continue
            _play_out(state, seat_seeds, cfg, stats)
            stats.rollouts += 1
            V[i, k] = _differential(state, my_team) - base
    return V


# ---------------------------------------------------------------------------
# Noise-to-signal reporting
# ---------------------------------------------------------------------------

def noise_to_signal(mats: Iterable[np.ndarray]) -> dict:
    """Variance diagnostics over a collection of per-position ``V`` matrices.

    v0.3's headline numbers, for comparison, were sd of a single action's value
    across sampled layouts = 0.698 and mean best-worst candidate gap = 0.293,
    giving 2.4. Those were measured on a depth-limited static evaluation; here
    the value is the realised final set differential, so the scales are related
    but not identical, and the comparison is indicative rather than exact.

    Reported:
      ``sd_world``      mean over candidates of sd_k(V[a,k])  - v0.3's numerator
      ``gap``           mean over positions of max_a Qhat(a) - min_a Qhat(a)
      ``nsr_unpaired``  sd_world / gap                        - v0.3's 2.4
      ``sd_paired``     mean over candidate pairs of sd_k(V[a,k] - V[a',k])
      ``se_paired``     sd_paired / sqrt(K), the standard error CRN actually
                        delivers on a pairwise comparison
      ``nsr_paired``    se_paired / mean |Qhat(a) - Qhat(a')|
      ``crn_var_ratio`` variance of the paired difference divided by the
                        variance it would have with independent worlds
                        (2*sd_world^2); below 1 is the reduction CRN buys.
    """
    sd_world: list[float] = []
    gaps: list[float] = []
    sd_paired: list[float] = []
    absdq: list[float] = []
    ks: list[int] = []
    for V in mats:
        V = np.asarray(V, dtype=np.float64)
        if V.ndim != 2 or V.shape[0] < 2 or V.shape[1] < 2:
            continue
        A, K = V.shape
        ks.append(K)
        sd_world.extend(V.std(axis=1, ddof=1).tolist())
        Q = V.mean(axis=1)
        gaps.append(float(Q.max() - Q.min()))
        for i in range(A):
            for j in range(i + 1, A):
                d = V[i] - V[j]
                sd_paired.append(float(d.std(ddof=1)))
                absdq.append(abs(float(Q[i] - Q[j])))
    if not gaps:
        return {}
    K = float(np.mean(ks))
    m_sd_world = float(np.mean(sd_world))
    m_gap = float(np.mean(gaps))
    m_sd_paired = float(np.mean(sd_paired))
    m_absdq = float(np.mean(absdq))
    se_paired = m_sd_paired / np.sqrt(K)
    indep = np.sqrt(2.0) * m_sd_world
    return {
        "n_positions": len(gaps),
        "K": K,
        "sd_world": m_sd_world,
        "gap": m_gap,
        "nsr_unpaired": m_sd_world / m_gap if m_gap else float("inf"),
        "sd_paired": m_sd_paired,
        "se_paired": se_paired,
        "mean_abs_dq": m_absdq,
        "nsr_paired": se_paired / m_absdq if m_absdq else float("inf"),
        "crn_var_ratio": (m_sd_paired / indep) ** 2 if indep else float("nan"),
        "v03_reference": {"sd_world": 0.698, "gap": 0.293, "nsr": 2.4},
    }


# ---------------------------------------------------------------------------
# The rollout pass over a harvested dataset
# ---------------------------------------------------------------------------
#
# Kept here rather than in ``dataset.py`` because it is rollout machinery; the
# import of ``dataset`` is deferred into the functions so the two modules do not
# form an import cycle (``dataset`` needs ``banked_differential`` at import
# time).

def rollouts_path(root) -> "Path":
    from pathlib import Path
    return Path(root) / "rollouts.jsonl"


def completed_rollouts(root) -> set:
    """Position ids already evaluated, for resuming an interrupted pass."""
    import json
    p = rollouts_path(root)
    if not p.exists():
        return set()
    out = set()
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.add(json.loads(line)["pid"])
    return out


def recorded_policies(root) -> dict:
    """How many stored rollout records were produced by each policy.

    ``None`` counts records written before the policy was recorded at all.
    Resuming keys on the position id alone, so without this a pass run with
    ``--continuation v04`` on top of a public pass would skip every position
    the weak policy had already done and leave one file holding both -- values
    that differ by a factor of seven in the slope they support, with nothing on
    disk saying which line is which.
    """
    import json
    p = rollouts_path(root)
    if not p.exists():
        return {}
    seen: dict = {}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                k = json.loads(line).get("policy")
                seen[k] = seen.get(k, 0) + 1
    return seen


def load_rollouts(root) -> dict:
    """``pid -> V`` matrix over the position's ``eval_idx`` candidates."""
    import json
    p = rollouts_path(root)
    out: dict = {}
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["pid"]] = np.asarray(d["v"], dtype=np.float64)
    return out


def rollout_summary(root) -> dict:
    """Aggregate the rollout pass from the file itself, not from a run summary.

    A pass can be interrupted - this one was, at 874 of 1030 positions, by the
    parent being killed - and the totals reported have to be the totals actually
    on disk rather than whatever the last completed invocation happened to
    return.
    """
    import json
    p = rollouts_path(root)
    agg = RolloutStats()
    n = 0
    seconds = 0.0
    if not p.exists():
        return {"positions": 0}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            agg.merge(d.get("stats", {}))
            seconds += float(d.get("seconds", 0.0))
            n += 1
    out = agg.to_dict()
    out["positions"] = n
    out["seconds"] = seconds
    return out


def _eval_one(args) -> dict:
    """Worker: evaluate one harvested position. Pure function of its input."""
    import time
    from .dataset import (decode_history, record_asks, record_rules,
                          record_worlds)
    rec, cfg_dict, seed = args
    cfg = RolloutConfig.from_dict(cfg_dict)
    hist = ()
    if cfg.seed_history:
        if "history" not in rec:
            # A schema-1 position, harvested before the log was stored. Seeding
            # an empty history here would anchor six belief trackers on a deal
            # that never happened and the rollout would run, silently, on a
            # different game. Refuse by name.
            raise ValueError(
                f"position {rec.get('pid')} predates the stored public history "
                f"(schema askobj-positions/1) and cannot be finished by "
                f"policy={cfg.policy!r}; re-harvest it")
        hist = tuple(decode_history(rec["history"]))
    asks = record_asks(rec)
    cands = [asks[i] for i in rec["eval_idx"]]
    stats = RolloutStats()
    t0 = time.time()
    V = rollout_matrix(record_rules(rec), rec["seat"], rec["set_winner"],
                       rec["seat"], record_worlds(rec), cands, seed,
                       cfg=cfg, stats=stats, history=hist)
    return {"pid": rec["pid"], "policy": cfg.policy,
            "eval_idx": list(rec["eval_idx"]),
            "v": V.astype(int).tolist(), "stats": stats.to_dict(),
            "seconds": round(time.time() - t0, 3)}


def evaluate_positions(root, cfg: Optional[RolloutConfig] = None,
                       n_workers: int = 2, seed: int = 90210,
                       limit: Optional[int] = None,
                       progress: bool = True) -> dict:
    """Run the CRN rollout pass over every harvested position not yet done.

    Resumable: positions already present in ``rollouts.jsonl`` are skipped. The
    parent process is the only writer, so there is no interleaved-write hazard.
    The rollout seed for a position is derived from ``seed`` and the position id,
    so re-running reproduces the same matrix.
    """
    import json
    import sys
    import time
    import zlib
    from multiprocessing import Pool
    from pathlib import Path

    from .dataset import MAX_WORKERS, iter_positions

    root = Path(root)
    cfg = cfg or RolloutConfig()
    n_workers = max(1, min(n_workers, MAX_WORKERS))
    seen = recorded_policies(root)
    wrong = {k: v for k, v in seen.items() if k is not None and k != cfg.policy}
    if wrong:
        raise ValueError(
            f"{rollouts_path(root)} already holds "
            f"{sum(wrong.values())} record(s) from "
            f"{sorted(wrong)}; this pass is {cfg.policy!r}. Resuming keys on "
            f"the position id alone, so continuing here would leave one file "
            f"mixing two continuations. Use a different --run.")
    if seen.get(None):
        print(f"note: {seen[None]} rollout record(s) predate the policy tag "
              f"and cannot be checked against {cfg.policy!r}", file=sys.stderr)
    done = completed_rollouts(root)
    jobs = []
    for rec in iter_positions(root):
        if rec["pid"] in done:
            continue
        # zlib.crc32, not hash(): Python string hashing is salted per process,
        # so hash() would make the "same seed reproduces the matrix" promise
        # false across runs.
        pos_seed = (seed * 1_000_003) ^ zlib.crc32(rec["pid"].encode())
        jobs.append((rec, cfg.to_dict(), pos_seed))
        if limit is not None and len(jobs) >= limit:
            break
    summary = {"skipped": len(done), "evaluated": 0, "seconds": 0.0,
               "config": cfg.to_dict()}
    agg = RolloutStats()
    if not jobs:
        return summary
    t0 = time.time()
    fh = rollouts_path(root).open("a", encoding="utf-8")
    try:
        if n_workers > 1:
            with Pool(n_workers) as pool:
                it = pool.imap_unordered(_eval_one, jobs, chunksize=1)
                for i, out in enumerate(it):
                    fh.write(json.dumps(out) + "\n")
                    fh.flush()
                    agg.merge(out["stats"])
                    summary["evaluated"] += 1
                    if progress and (i + 1) % 20 == 0:
                        print(f"  rollout {i+1}/{len(jobs)} positions, "
                              f"{time.time()-t0:.0f}s", file=sys.stderr,
                              flush=True)
        else:
            for i, job in enumerate(jobs):
                out = _eval_one(job)
                fh.write(json.dumps(out) + "\n")
                fh.flush()
                agg.merge(out["stats"])
                summary["evaluated"] += 1
                if progress and (i + 1) % 20 == 0:
                    print(f"  rollout {i+1}/{len(jobs)} positions, "
                          f"{time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    finally:
        fh.close()
    summary["seconds"] = time.time() - t0
    summary["rollout_stats"] = agg.to_dict()
    return summary
