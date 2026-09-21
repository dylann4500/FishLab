"""Perfect-information value dataset.

Records (features of the TRUE state, remaining set differential) at decision
points of self-play games. This is a *training-time* use of ground truth,
which is legitimate: the resulting network is a static evaluation function
applied inside search to worlds the agent SAMPLED FROM ITS OWN BELIEFS. No
acting policy ever sees the real state.

The target is the differential still to be decided from that point on
(final minus already-banked), so the evaluator learns the value of the
remaining game rather than memorizing the scoreboard.
"""

from __future__ import annotations

import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..agents import AGENT_REGISTRY
from ..cards import team_of
from ..engine import GameState
from ..features import extract_perfect
from ..observation import Observation
from ..rules import RuleConfig


def _one_game(args):
    spec, rules_dict, deal_seed, agent_seed, stride = args
    rules = RuleConfig.from_dict(rules_dict)
    state = GameState.deal(rules, seed=deal_seed)
    agents = [AGENT_REGISTRY[spec[0]](**spec[1]) for _ in range(6)]
    rng = random.Random(agent_seed)
    for p, ag in enumerate(agents):
        ag.begin_game(p, rules, rng.getrandbits(64))
    rows, meta = [], []
    step = guard = 0
    while not state.is_terminal:
        p = state.turn
        obs = Observation.from_state(state, p)
        if step % stride == 0:
            a0, b0, _ = state.scores()
            rows.append(extract_perfect(state.hands, state.set_winner,
                                        state.turn, p, rules.variant))
            meta.append((team_of(p), a0, b0))
        try:
            state.apply(p, agents[p].act(obs))
        except Exception:
            return None
        step += 1
        guard += 1
        if guard > 20000:
            return None
    a, b, _ = state.scores()
    if not rows:
        return None
    X = np.stack(rows)
    y = np.array([((a - a0) - (b - b0)) if t == 0 else ((b - b0) - (a - a0))
                  for (t, a0, b0) in meta], dtype=np.float32)
    return X, y


def generate_pi_dataset(spec=("probabilistic", {}), n_games: int = 2000,
                        rules: RuleConfig | None = None, base_seed: int = 0,
                        n_jobs: int = 8, stride: int = 2,
                        out: Path | str | None = None):
    rules = rules or RuleConfig()
    rd = rules.to_dict()
    srng = random.Random(base_seed ^ 0xC0FFEE)
    jobs = [(spec, {**rd, "starting_player": i % 6}, base_seed + i,
             srng.getrandbits(64), stride) for i in range(n_games)]
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            res = pool.map(_one_game, jobs, chunksize=4)
    else:
        res = [_one_game(j) for j in jobs]
    res = [r for r in res if r is not None]
    X = np.concatenate([r[0] for r in res])
    y = np.concatenate([r[1] for r in res])
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, X=X, y=y)
    return X, y


def load_dataset(path: Path | str):
    d = np.load(path)
    return d["X"], d["y"]
