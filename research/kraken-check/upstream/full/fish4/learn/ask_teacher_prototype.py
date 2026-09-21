"""Label the value of an ask by playing the game out, and learn from it.

The champion's ask score is a hand-picked weighting: success probability,
suit depth, turn-risk and scarcity, with weights found by sweeping. Those
terms were *guessed*, and the sweep showed the good ones behave like
tie-breakers. A model trained to predict what an ask is actually worth
should beat any hand-tuned combination, and it is the natural next step now
that a champion exists worth learning from.

Why THIS teacher and not the ones that failed
---------------------------------------------
Earlier attempts scored candidate actions with depth-limited rollouts under
a weak policy, and lost to the very prior they were built on. Two measured
reasons, both fixed here:

1. **Variance swamped signal.** The spread of a position's value across
   hidden layouts is ~2.4x the gap between candidate actions. So every
   candidate here is continued from the SAME sampled layouts with the SAME
   agent seeds (common random numbers). The comparison is paired, exactly
   like the duplicate-deal design used for evaluation.
2. **The continuation policy was too weak and too short.** Here the game is
   played to completion by the current champion in every seat, so the label
   is "what this ask is worth under strong play", not "what a heuristic
   makes of it in thirty moves".

The label is the final set differential from the acting team's perspective,
averaged over continuations. Ground truth is used only for *labelling*,
offline. The features the student consumes are computed from the acting
player's own observation and beliefs, so the resulting policy stays inside
the information boundary.
"""

from __future__ import annotations

import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..agents import AGENT_REGISTRY
from ..cards import cards_per_player, half_suit_mask, num_half_suits, team_of
from ..engine import Ask, GameState
from ..observation import Observation
from ..rules import RuleConfig

CHAMPION = ("tuned", {"w_turn": 0.6, "w_scarce": 0.2})

# Feature layout, all computable from the acting player's legal view.
FEATURE_NAMES = [
    "p_success",           # sampled probability the target holds it
    "my_depth",            # cards I hold in that half-suit / 6
    "team_share",          # expected fraction of the suit held by my team
    "target_hand",         # target's hand size / cards per player
    "target_rel_hand",     # target's hand size relative to the live average
    "suit_revealed",       # have I already shown interest in this half-suit
    "certain_hit",         # do I KNOW the target holds it
    "suit_live_cards",     # unresolved cards of that suit still in play / 6
    "my_hand",             # my hand size / cards per player
    "game_progress",       # actions so far / 200
    "set_diff",            # my sets minus theirs, normalized
    "opp_max_hand",        # largest opponent hand / cards per player
]
N_FEATURES = len(FEATURE_NAMES)


def ask_feature_vector(obs: Observation, ask: Ask, worlds, belief,
                       revealed: set[int]) -> np.ndarray:
    hs = ask.card // 6
    m = half_suit_mask(hs)
    per = cards_per_player(obs.rules.variant)
    n_hs = num_half_suits(obs.rules.variant)
    my_team = team_of(obs.player)
    nw = max(1, len(worlds))

    hits = sum(1 for w in worlds if w[ask.target] & (1 << ask.card))
    share = 0.0
    live = 0.0
    for w in worlds:
        share += sum((w[p] & m).bit_count() for p in range(6)
                     if team_of(p) == my_team)
        live += sum((w[p] & m).bit_count() for p in range(6))
    counts = obs.hand_counts
    alive = [c for c in counts if c > 0]
    avg = (sum(alive) / len(alive)) if alive else 1.0
    my_sets = sum(1 for w in obs.set_winner if w == my_team)
    their_sets = sum(1 for w in obs.set_winner if w == 1 - my_team)
    opp_max = max((counts[p] for p in range(6)
                   if team_of(p) != my_team), default=0)

    return np.array([
        hits / nw,
        (obs.hand & m).bit_count() / 6.0,
        share / (6.0 * nw),
        counts[ask.target] / per,
        (counts[ask.target] - avg) / max(1.0, per),
        1.0 if hs in revealed else 0.0,
        1.0 if belief.current_holder_mask(ask.card) == (1 << ask.target) else 0.0,
        live / (6.0 * nw),
        obs.hand.bit_count() / per,
        min(1.0, len(obs.history) / 200.0),
        (my_sets - their_sets) / n_hs,
        opp_max / per,
    ], dtype=np.float32)


def _revealed_suits(obs: Observation) -> set[int]:
    from ..engine import AskEvent
    out = set()
    for ev in obs.history:
        if isinstance(ev, AskEvent):
            if ev.asker == obs.player or (ev.success and ev.target == obs.player):
                out.add(ev.card // 6)
    return out


def _play_out(rules: RuleConfig, hands, set_winner, turn, history,
              first_actor, first_action, seed: int, max_actions: int = 4000):
    """Apply one action then let the champion finish the game. Returns the
    final set differential for ``first_actor``'s team, or None if the
    continuation failed to terminate.

    The real public history is carried into the continuation. Without it the
    belief-tracking agents refuse to attach (correctly: their constraint
    system is anchored on the initial deal), and every continuation would
    silently fail. Replaying under the true record also means the
    continuation is played by agents who know what actually happened, which
    is the whole point of labelling under strong play.
    """
    state = GameState.from_components(rules, list(hands), turn,
                                      list(set_winner))
    state.history = list(history)
    try:
        state.apply(first_actor, first_action)
    except Exception:
        return None
    agents = [AGENT_REGISTRY[CHAMPION[0]](**CHAMPION[1]) for _ in range(6)]
    rng = random.Random(seed)
    for p, ag in enumerate(agents):
        ag.begin_game(p, rules, rng.getrandbits(64))
    n = 0
    while not state.is_terminal:
        p = state.turn
        try:
            state.apply(p, agents[p].act(Observation.from_state(state, p)))
        except Exception:
            return None
        n += 1
        if n > max_actions:
            return None
    a, b, _ = state.scores()
    mine = a if team_of(first_actor) == 0 else b
    theirs = b if team_of(first_actor) == 0 else a
    return float(mine - theirs)


def _one_position(args):
    """Sample a position, then label several candidate asks by continuation."""
    (rules_dict, deal_seed, agent_seed, stop_at, n_candidates,
     n_continuations, n_worlds) = args
    rules = RuleConfig.from_dict(rules_dict)
    state = GameState.deal(rules, seed=deal_seed)
    agents = [AGENT_REGISTRY[CHAMPION[0]](**CHAMPION[1]) for _ in range(6)]
    rng = random.Random(agent_seed)
    for p, ag in enumerate(agents):
        ag.begin_game(p, rules, rng.getrandbits(64))

    for _ in range(stop_at):
        if state.is_terminal:
            return None
        p = state.turn
        try:
            state.apply(p, agents[p].act(Observation.from_state(state, p)))
        except Exception:
            return None
    if state.is_terminal:
        return None

    actor = state.turn
    obs = Observation.from_state(state, actor)
    asks = obs.legal_asks()
    if len(asks) < 2:
        return None

    agent = agents[actor]
    agent.bel.update(obs)
    worlds = [agent.bel.sample_current_hands(agent.rng) for _ in range(n_worlds)]
    revealed = _revealed_suits(obs)

    # Candidates: the policy's own favourites plus random others, so the
    # training set is not confined to moves the champion already likes.
    scored = []
    for a in asks:
        hits = sum(1 for w in worlds if w[a.target] & (1 << a.card))
        scored.append((hits / max(1, len(worlds)), a))
    scored.sort(key=lambda t: -t[0])
    top = [a for _, a in scored[:max(1, n_candidates // 2)]]
    rest = [a for a in asks if a not in top]
    rng.shuffle(rest)
    candidates = top + rest[: n_candidates - len(top)]
    if len(candidates) < 2:
        return None

    # Common random numbers: identical continuation seeds for every
    # candidate, so world luck cancels in the comparison between them.
    cont_seeds = [rng.getrandbits(32) for _ in range(n_continuations)]
    rows, labels = [], []
    for a in candidates:
        vals = [v for cs in cont_seeds
                if (v := _play_out(rules, state.hands, state.set_winner,
                                   state.turn, state.history, actor, a, cs))
                is not None]
        if len(vals) < max(2, n_continuations // 2):
            continue
        rows.append(ask_feature_vector(obs, a, worlds, agent.bel, revealed))
        labels.append(sum(vals) / len(vals))
    if len(rows) < 2:
        return None
    return np.stack(rows), np.array(labels, dtype=np.float32)


def generate_ask_dataset(n_positions: int = 800, n_candidates: int = 4,
                         n_continuations: int = 6, n_worlds: int = 24,
                         base_seed: int = 0, n_jobs: int = 8,
                         rules: RuleConfig | None = None,
                         out: Path | str | None = None):
    rules = rules or RuleConfig()
    rd = rules.to_dict()
    srng = random.Random(base_seed ^ 0x5A5A)
    jobs = []
    for i in range(n_positions):
        jobs.append((
            {**rd, "starting_player": i % 6},
            base_seed + i,
            srng.getrandbits(64),
            srng.randint(12, 70),          # how far into the game to stop
            n_candidates, n_continuations, n_worlds))
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            res = pool.map(_one_position, jobs, chunksize=2)
    else:
        res = [_one_position(j) for j in jobs]
    res = [r for r in res if r is not None]
    if not res:
        raise RuntimeError("no positions produced labels")
    X = np.concatenate([r[0] for r in res])
    y = np.concatenate([r[1] for r in res])
    # group id lets the student learn a RANKING within a position rather
    # than an absolute value, which is what actually matters for choosing.
    groups = np.concatenate([np.full(len(r[1]), i, dtype=np.int32)
                             for i, r in enumerate(res)])
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, X=X, y=y, groups=groups,
                            names=np.array(FEATURE_NAMES))
    return X, y, groups
