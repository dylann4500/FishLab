"""Harvesting real ask positions from v0.4 self-play.

WHAT A POSITION RECORD IS, AND WHY IT HOLDS WHAT IT HOLDS
--------------------------------------------------------
A record is one ask decision actually taken by ``("fishbot4", {})`` in
self-play. It stores everything needed to (a) re-run rollouts later and (b) fit
the objective, and deliberately stores nothing else:

    pid          "<game_seed>:<ply>", unique and reproducible
    game, ply    provenance; ``ply`` is the length of the public history
    seat         the acting seat (also ``turn``)
    rules        RuleConfig as a dict
    set_winner   resolved half-suits at the position
    asks         every legal ask, as ``[target, card]``
    p            P(success) for each legal ask, from the acting seat's posterior
    f            the 10-column ``fish4.askfeat.ask_feature_matrix`` block, in
                 ``TERM_NAMES`` order, for each legal ask
    eval_idx     indices of the asks that the rollout pass will evaluate
    worlds       K determinized worlds, each six current-hand bitmasks in hex,
                 drawn from the ACTING SEAT'S OWN posterior
    banked       set differential already resolved, from the acting team's view
    n_hidden     number of live cards whose location is not yet pinned

**The worlds are stored, not the true hands.** The reason that matters is
leak-freedom: storing the posterior's worlds and nothing else makes it
structural rather than a matter of discipline, because the true layout is never
written to disk and no later stage can accidentally read it. Capturing the
posterior while the game runs also makes every batch reproducible from the file
alone.

This paragraph used to give a second, stronger reason, and it was false.
``fish.beliefs.BeliefState`` is anchored on the initial deal, so it was said
that "a posterior cannot be reconstructed after the fact". It is anchored on the
initial deal but does not have to be HANDED one: given the current hand and the
public history, ``initial_hand()`` back-computes a deal consistent with both and
the tracker attaches to that. ``scripts4/rollout_target.py`` does it over 110
positions without a single ``BeliefContradiction``.

The correction is recorded rather than quietly deleted because the false half of
that sentence cost more than the true half ever earned: the paper closed the
whole objective-learning line on it (see ``fish4/learn/rollout.py``). An
impossibility claim generates no experiment -- a positive result invites
replication, "this cannot be done" invites moving on -- so the sentence that
closes a direction is exactly the sentence nobody tests.

WHICH DECISIONS ARE RECORDED
----------------------------
* the agent actually chose an ASK (positions where it claimed are not ask
  decisions and would contribute nothing to an ask objective);
* at least two legal asks exist (a pair needs two candidates);
* the position is not fully determined. When every live card is already pinned
  there is no hidden information, the posterior is a point mass, all K worlds
  are identical, and every rollout value is the same number - infinite pairs
  with zero variance and zero information. Those positions are skipped, which
  also saves the rollout budget for positions that can teach something.

Sampling is thinned per game (``max_per_game``, Bernoulli ``sample_rate``) so
that positions are spread over game phases and the within-game correlation that
would otherwise dominate the dataset is kept small. The thinning RNG is seeded
from the harvest seed and the game seed only, never from the deal.

FORMAT AND RESUMABILITY
-----------------------
One JSON object per line in ``positions.jsonl``, written by the parent process
only, so there is no interleaved-write hazard. ``meta.json`` holds the config
and the schema tag. Restarting the harvest re-reads the game seeds already
present and skips them, so an interrupted run resumes at the game boundary.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from fish.beliefs import RESOLVED, BeliefContradiction
from fish.cards import deck_size, team_of
from fish.engine import Ask, AskEvent, ClaimEvent, PassEvent
from fish.observation import Observation
from fish.rules import RuleConfig
from fish.runner import GameTimeout, play_game

from ..agent4 import FishBot4
from ..askfeat import (TERM_NAMES, ask_feature_matrix, DecisionContext,
                       term_versions)
from ..posterior import Posterior
from .rollout import banked_differential

#: Bump when the record shape changes.
POSITION_SCHEMA = "askobj-positions/2"
#: Schemas this module can still read. Version 1 lacks ``history`` and so
#: cannot feed the strong continuation; the rollout stage refuses it by name
#: rather than by seeding an empty log, which would anchor six belief trackers
#: on a game that never happened.
READABLE_SCHEMAS = ("askobj-positions/1", "askobj-positions/2")

#: Hard cap on worker processes: 8 cores, other jobs running alongside.

def encode_history(history) -> list:
    """The public log as JSON, one compact row per event.

    Tags mirror ``fish.gamelog``: ``a`` ask, ``c`` claim, ``p`` pass. Stored
    because the strong rollout continuation reconstructs each seat's initial
    deal from its hand plus this log, and a position without it can only be
    finished by a policy that has no model to attach.
    """
    out = []
    for ev in history:
        if isinstance(ev, AskEvent):
            out.append(["a", ev.asker, ev.target, ev.card,
                        1 if ev.success else 0])
        elif isinstance(ev, ClaimEvent):
            out.append(["c", ev.claimer, ev.half_suit, list(ev.declared),
                        list(ev.revealed), ev.winner])
        elif isinstance(ev, PassEvent):
            out.append(["p", ev.player, ev.teammate])
        else:
            raise TypeError(f"cannot encode event {ev!r}")
    return out


def decode_history(rows) -> list:
    """Inverse of :func:`encode_history`. Unknown tags raise."""
    out = []
    for r in rows:
        tag = r[0]
        if tag == "a":
            out.append(AskEvent(r[1], r[2], r[3], bool(r[4])))
        elif tag == "c":
            out.append(ClaimEvent(r[1], r[2], tuple(r[3]), tuple(r[4]), r[5]))
        elif tag == "p":
            out.append(PassEvent(r[1], r[2]))
        else:
            raise ValueError(f"unknown history tag {tag!r}")
    return out


def _worker_cap() -> int:
    """How many processes a harvest or rollout pass may use.

    Was a hard-coded 2, chosen when this ran on a machine with eight cores
    shared with other jobs. Two is right while duels occupy the rest and wrong
    the moment they finish, and a constant cannot tell the difference. The cap
    now leaves one core for whatever else is running and can be overridden
    explicitly, so a long resumable pass can be restarted wider once the queue
    beside it drains.

    An explicit FISH_LEARN_WORKERS is honoured down to 1: the docstring used to
    promise "never below 2" while the override path returned max(1, ...), and a
    promise the code does not keep is worse than either behaviour. Setting it
    to 1 deliberately is a legitimate thing to want on a small box.
    """
    override = os.environ.get("FISH_LEARN_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return max(2, (os.cpu_count() or 4) - 1)


MAX_WORKERS = _worker_cap()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HarvestConfig:
    """Everything that determines which positions land in the dataset."""

    n_games: int = 200
    #: determinized worlds captured per position (the rollout budget K)
    n_worlds: int = 16
    #: candidates the rollout pass will evaluate per position
    n_eval: int = 6
    #: of those, how many are the top of the incumbent objective; the rest are
    #: drawn uniformly from the remaining legal asks so the feature contrasts
    #: are not confined to the incumbent's own preferred region.
    n_top: int = 3
    #: Bernoulli thinning rate over eligible ask decisions. With ~80 ask
    #: decisions per game this yields ~4 positions per game, and the cap
    #: below almost never binds - which is the point. An earlier setting
    #: (rate 0.20, cap 4) filled the cap within the first 25 plies of every
    #: game and produced a dataset of openings only.
    sample_rate: float = 0.05
    #: cap per game, shared across all six seats
    max_per_game: int = 8
    seed: int = 20260821
    base_deal_seed: int = 4_100_000
    agent_seed_base: int = 771
    variant: str = "54"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HarvestConfig":
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


# ---------------------------------------------------------------------------
# The harvesting agent
# ---------------------------------------------------------------------------

class _Sink:
    """Per-game shared state: the thinning RNG, the cap, and the records."""

    __slots__ = ("cfg", "rng", "records", "game_seed", "taken")

    def __init__(self, cfg: HarvestConfig, game_seed: int, rng_seed: int):
        self.cfg = cfg
        self.rng = random.Random(rng_seed)
        self.records: list[dict] = []
        self.game_seed = game_seed
        self.taken = 0

    def wants(self) -> bool:
        if self.taken >= self.cfg.max_per_game:
            return False
        return self.rng.random() < self.cfg.sample_rate


def select_candidates(scores: np.ndarray, rng: random.Random,
                      n_top: int, n_eval: int) -> list[int]:
    """Indices of the asks to evaluate by rollout.

    Top-``n_top`` by the incumbent objective plus a uniform sample of the rest.
    The top slice keeps the dataset anchored on decisions that are actually
    close to being made; the uniform slice supplies feature contrasts the
    incumbent ranking would never generate, which is what identifies the
    coefficients. The split is reported so the fit can be re-run on the top
    slice alone as a robustness check.
    """
    order = list(np.argsort(-np.asarray(scores, dtype=np.float64)))
    top = [int(i) for i in order[:n_top]]
    rest = [int(i) for i in order[n_top:]]
    k = max(0, min(n_eval - len(top), len(rest)))
    extra = rng.sample(rest, k) if k else []
    return sorted(top + extra)


class HarvestBot(FishBot4):
    """``FishBot4`` that records a thinned sample of its own ask decisions.

    The recording happens AFTER the policy has chosen, and only when the choice
    was an ask, so the harvested distribution is exactly the distribution of ask
    decisions the v0.4 agent actually faces. The extra ``Posterior`` built for
    the record costs one more inference per sampled decision; at a thinning rate
    of ~0.12 that is a few percent of harvest time.
    """

    name = "harvestbot"

    def __init__(self, sink: Optional[_Sink] = None, **kw):
        super().__init__(**kw)
        self.sink = sink

    def act(self, obs: Observation):
        action = super().act(obs)
        if (self.sink is not None and isinstance(action, Ask)
                and self.sink.wants()):
            try:
                rec = self._record(obs)
            except (BeliefContradiction, ValueError):
                rec = None
            if rec is not None:
                self.sink.records.append(rec)
                self.sink.taken += 1
        return action

    # -- recording ---------------------------------------------------------

    def _record(self, obs: Observation) -> Optional[dict]:
        cfg = self.sink.cfg
        asks = obs.legal_asks()
        if len(asks) < 2:
            return None
        post = Posterior(self.bel, self.rng, n_draws=self.n_draws,
                         n_worlds=cfg.n_worlds, mode=self.infer_mode)
        M = post.marginals()
        n_hidden = 0
        for c in range(self.bel.n):
            if self.bel.public_loc[c] == RESOLVED:
                continue
            if float(M[c].max()) < 1.0 - 1e-9:
                n_hidden += 1
        if n_hidden == 0:
            return None            # nothing hidden: no objective to learn here
        worlds = post.worlds()
        if len(worlds) < cfg.n_worlds:
            return None            # sampler failed; counted by refusal, not faked
        ctx = DecisionContext(obs, self.bel, post)
        p, F = ask_feature_matrix(ctx, asks)
        scores = p + F @ self.weights.as_vector()
        eval_idx = select_candidates(scores, self.sink.rng, cfg.n_top,
                                     cfg.n_eval)
        if len(eval_idx) < 2:
            return None
        return {
            "pid": f"{self.sink.game_seed}:{len(obs.history)}",
            "game": self.sink.game_seed,
            "ply": len(obs.history),
            "history": encode_history(obs.history),
            "seat": obs.player,
            "rules": obs.rules.to_dict(),
            "set_winner": list(obs.set_winner),
            "asks": [[a.target, a.card] for a in asks],
            "p": [round(float(x), 6) for x in p],
            "f": [[round(float(x), 6) for x in row] for row in F],
            "eval_idx": eval_idx,
            "worlds": [[format(h, "x") for h in w] for w in worlds[:cfg.n_worlds]],
            "banked": banked_differential(obs.set_winner, team_of(obs.player)),
            "n_hidden": n_hidden,
            "terms": list(TERM_NAMES),
            # Not just the names: the DEFINITION version of each, so a formula
            # that changes under a stable name is caught too. See
            # fish4.askfeat.TERM_VERSIONS.
            "tv": term_versions(),
        }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _play_one(args) -> dict:
    game_seed, agent_seed, start_seat, cfg_dict = args
    cfg = HarvestConfig.from_dict(cfg_dict)
    rules = RuleConfig(variant=cfg.variant, starting_player=start_seat)
    deck = list(range(deck_size(cfg.variant)))
    random.Random(game_seed).shuffle(deck)
    sink = _Sink(cfg, game_seed, rng_seed=(cfg.seed * 1_000_003) ^ game_seed)
    agents = [HarvestBot(sink=sink) for _ in range(6)]
    t0 = time.time()
    out = {"game": game_seed, "records": [], "timeout": 0,
           "actions": 0, "seconds": 0.0}
    try:
        st = play_game(agents, rules, deck_order=deck, agent_seed=agent_seed)
        out["actions"] = len(st.history)
    except GameTimeout:
        out["timeout"] = 1
    out["records"] = sink.records
    out["seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def positions_path(root: Path) -> Path:
    return Path(root) / "positions.jsonl"


def meta_path(root: Path) -> Path:
    return Path(root) / "meta.json"


def _meta(root: Path) -> dict:
    """The harvest's recorded metadata, or an empty dict if there is none."""
    p = meta_path(root)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_positions(root: Path) -> list[dict]:
    p = positions_path(root)
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_positions(root: Path) -> Iterator[dict]:
    p = positions_path(root)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def completed_games(root: Path) -> set[int]:
    """Game seeds already harvested, read from the sidecar progress file.

    A game that produced zero records still counts as done, so a resumed run
    does not replay it. That is why progress is tracked separately from
    ``positions.jsonl`` rather than inferred from it.
    """
    f = Path(root) / "games_done.txt"
    if not f.exists():
        return set()
    return {int(x) for x in f.read_text(encoding="utf-8").split() if x.strip()}


def harvest(root: Path, cfg: HarvestConfig, n_workers: int = MAX_WORKERS,
            progress: bool = True) -> dict:
    """Play ``cfg.n_games`` self-play games and append sampled positions.

    Resumable at the game boundary. Returns a summary dict.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    n_workers = max(1, min(n_workers, MAX_WORKERS))
    # The schema tag was written unconditionally and before this check, so
    # resuming a directory whose positions.jsonl holds version-1 rows stamped
    # it "askobj-positions/2" -- a dataset claiming to carry histories while
    # only partly doing so. READABLE_SCHEMAS existed to police exactly that and
    # was referenced nowhere, so nothing ever read the tag it was policing.
    old = _meta(root)
    prior = old.get("schema") if old else None
    if prior is not None and prior != POSITION_SCHEMA:
        if prior not in READABLE_SCHEMAS:
            raise ValueError(
                f"{root} holds schema {prior!r}, which this module cannot "
                f"read (known: {list(READABLE_SCHEMAS)})")
        raise ValueError(
            f"{root} holds schema {prior!r} and this harvest writes "
            f"{POSITION_SCHEMA!r}. Appending would leave one file mixing both, "
            f"with the tag naming only the newer. Harvest into a new root.")
    meta_path(root).write_text(json.dumps(
        {"schema": POSITION_SCHEMA, "config": cfg.to_dict(),
         "terms": list(TERM_NAMES), "term_versions": term_versions(),
         "agent": ["fishbot4", {}]},
        indent=2), encoding="utf-8")
    done = completed_games(root)
    seed_rng = random.Random(cfg.agent_seed_base)
    jobs = []
    for i in range(cfg.n_games):
        gs = cfg.base_deal_seed + i
        agent_seed = seed_rng.getrandbits(64)
        if gs in done:
            continue
        jobs.append((gs, agent_seed, i % 6, cfg.to_dict()))
    summary = {"games_requested": cfg.n_games, "games_skipped": len(done),
               "games_played": 0, "positions": 0, "timeouts": 0,
               "actions": 0, "seconds": 0.0}
    if not jobs:
        summary["positions"] = len(load_positions(root))
        return summary
    t0 = time.time()
    fh = positions_path(root).open("a", encoding="utf-8")
    gh = (Path(root) / "games_done.txt").open("a", encoding="utf-8")
    try:
        if n_workers > 1:
            with Pool(n_workers) as pool:
                it = pool.imap_unordered(_play_one, jobs, chunksize=1)
                for i, rec in enumerate(it):
                    _absorb(rec, fh, gh, summary)
                    if progress and (i + 1) % 10 == 0:
                        print(f"  harvest {i+1}/{len(jobs)} games, "
                              f"{summary['positions']} positions, "
                              f"{time.time()-t0:.0f}s", file=sys.stderr,
                              flush=True)
        else:
            for i, job in enumerate(jobs):
                _absorb(_play_one(job), fh, gh, summary)
                if progress and (i + 1) % 10 == 0:
                    print(f"  harvest {i+1}/{len(jobs)} games, "
                          f"{summary['positions']} positions, "
                          f"{time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    finally:
        fh.close()
        gh.close()
    summary["seconds"] = time.time() - t0
    summary["positions_total"] = sum(1 for _ in iter_positions(root))
    return summary


def _absorb(rec: dict, fh, gh, summary: dict) -> None:
    for r in rec["records"]:
        fh.write(json.dumps(r) + "\n")
    fh.flush()
    gh.write(f"{rec['game']}\n")
    gh.flush()
    summary["games_played"] += 1
    summary["positions"] += len(rec["records"])
    summary["timeouts"] += rec["timeout"]
    summary["actions"] += rec["actions"]


# ---------------------------------------------------------------------------
# Record helpers used by the rollout pass and the fit
# ---------------------------------------------------------------------------

def record_asks(rec: dict) -> list[Ask]:
    return [Ask(t, c) for t, c in rec["asks"]]


def record_worlds(rec: dict) -> list[list[int]]:
    return [[int(h, 16) for h in w] for w in rec["worlds"]]


def record_rules(rec: dict) -> RuleConfig:
    return RuleConfig.from_dict(rec["rules"])


def record_features(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(rec["p"], dtype=np.float64),
            np.asarray(rec["f"], dtype=np.float64))
