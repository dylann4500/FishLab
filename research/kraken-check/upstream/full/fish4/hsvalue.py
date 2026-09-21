"""A learned value for a half-suit, and an ask objective built out of it.

THE PROBLEM WITH A WEIGHTED SUM OF HEURISTICS
---------------------------------------------
v0.3's ask objective was ``P(success) + 0.06*depth``, later
``+ 0.6*turn_risk + 0.2*scarcity``. Those terms are in incommensurable units:
a probability, a card count, a normalised hand-size difference, a share. The
weights that combine them therefore have no meaning beyond "what won a sweep",
which is why the sweep found an inverted-U (helpful small, destructive large)
and why v0.3 concluded the terms are tie-breakers rather than objectives.

What the game actually pays out is half-suits. So define, for each unresolved
half-suit \\(H\\),

    V(H) = P(our team ends up scoring H) - P(the opposing team does)

which is measured in sets, is bounded in [-1, 1], and is exactly the quantity
whose sum over half-suits is the final differential. An ask is then worth the
expected change it produces:

    score(ask) = p * [V(H | success) - V(H)]
               + (1-p) * [V(H | failure) - V(H)]
               + (1-p) * turn_cost(target)

with every term in the same unit. That is a marginal value, not a level, which
is the substantive difference from the heuristic basis: "our team is winning
this half-suit" is a level, and what should drive a decision is how much one
more card moves the outcome.

HOW V IS OBTAINED
-----------------
Not by guessing a functional form. We play self-play games, record for every
unresolved half-suit at every decision a feature vector computable from the
acting seat's posterior alone, and record how that half-suit eventually
resolved. Fitting a three-way classifier (we score it / they score it / null)
gives a calibrated V. This is supervised learning on an abundant, cheaply
labelled target - roughly nine labels per decision - rather than the rollout
regression that the variance analysis of v0.3 made expensive.

LEAKAGE
-------
Features come from the acting seat's posterior, which is derived from public
events plus that seat's own hand. The *label* is the eventual outcome, which is
ground truth, and is used only in training. The same line v0.3 drew for its
value network.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fish.cards import NUM_PLAYERS, half_suit_mask, team_of

# Feature names, fixed order. Everything is computed from the marginal block of
# one half-suit plus public counts, so the vector is cheap and seat-relative.
HS_FEATURES = (
    "my_depth",        # cards of H in our own hand, /6
    "team_share",      # expected cards of H with our team, /6
    "opp_share",       # expected cards of H with the opponents, /6
    "team_known",      # cards of H whose holder we have pinned, on our team, /6
    "opp_known",       # ... on theirs, /6
    "concentration",   # largest single teammate share of our team's holding
    "opp_concentration",
    "entropy",         # mean holder entropy over the six cards, /log 6
    "p_team_all",      # independence estimate that all six are ours
    "p_opp_any",       # 1 - p_team_all
    "turn_is_ours",    # does our team hold the turn
    "my_hand",         # our hand size / cards per player
    "team_hand",       # our team's total cards / (3 * cards per player)
    "opp_hand",
    "live_fraction",   # unresolved half-suits / total
    "revealed_by_me",  # have we already shown a holding in H
    "bias",
)
N_HS_FEATURES = len(HS_FEATURES)

#: labels
OUTCOME_US, OUTCOME_THEM, OUTCOME_NULL = 0, 1, 2


def half_suit_features(ctx, hs: int) -> np.ndarray:
    """Feature vector for one unresolved half-suit, from the acting seat's view."""
    x = np.zeros(N_HS_FEATURES, dtype=np.float64)
    obs = ctx.obs
    M = ctx.M
    lo = hs * 6
    block = M[lo:lo + 6]
    my_team = ctx.my_team
    mine = [p for p in range(NUM_PLAYERS) if team_of(p) == my_team]
    theirs = [p for p in range(NUM_PLAYERS) if team_of(p) != my_team]
    per = ctx.per

    x[0] = (obs.hand & half_suit_mask(hs)).bit_count() / 6.0
    team_exp = float(block[:, mine].sum())
    opp_exp = float(block[:, theirs].sum())
    x[1] = team_exp / 6.0
    x[2] = opp_exp / 6.0
    known_t = known_o = 0
    for i in range(6):
        row = block[i]
        j = int(np.argmax(row))
        if row[j] > 0.999:
            if team_of(j) == my_team:
                known_t += 1
            else:
                known_o += 1
    x[3] = known_t / 6.0
    x[4] = known_o / 6.0
    te = block[:, mine].sum(axis=0)
    oe = block[:, theirs].sum(axis=0)
    x[5] = float(te.max() / te.sum()) if te.sum() > 0 else 0.0
    x[6] = float(oe.max() / oe.sum()) if oe.sum() > 0 else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(block > 0, np.log(np.maximum(block, 1e-12)), 0.0)
    x[7] = float(-(block * lg).sum() / (6.0 * math.log(6.0)))
    pteam = block[:, mine].sum(axis=1)
    p_all = float(np.prod(np.clip(pteam, 0.0, 1.0)))
    x[8] = p_all
    x[9] = 1.0 - p_all
    x[10] = 1.0 if team_of(obs.turn) == my_team else 0.0
    x[11] = obs.hand.bit_count() / per
    x[12] = sum(obs.hand_counts[p] for p in mine) / (3.0 * per)
    x[13] = sum(obs.hand_counts[p] for p in theirs) / (3.0 * per)
    x[14] = sum(1 for w in obs.set_winner if w is None) / float(ctx.n_hs)
    x[15] = 1.0 if ctx.revealed[hs] else 0.0
    x[16] = 1.0
    return x


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass
class HalfSuitValue:
    """Multinomial logistic model over (us, them, null).

    Carries the standardisation it was fitted with, so inference cannot
    silently apply a model to unscaled features - a mistake that produces a
    plausible-looking but badly miscalibrated value.
    """
    W: np.ndarray                        # (3, N_HS_FEATURES)
    mu: np.ndarray = None                # (N_HS_FEATURES,)
    sigma: np.ndarray = None

    def __post_init__(self):
        if self.mu is None:
            self.mu = np.zeros(self.W.shape[1])
        if self.sigma is None:
            self.sigma = np.ones(self.W.shape[1])

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sigma

    def probs(self, X: np.ndarray) -> np.ndarray:
        z = self._scale(X) @ self.W.T
        z -= z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def value(self, X: np.ndarray) -> np.ndarray:
        """``P(us) - P(them)``, in sets."""
        p = self.probs(X)
        return p[..., OUTCOME_US] - p[..., OUTCOME_THEM]

    def to_dict(self) -> dict:
        return {"features": list(HS_FEATURES), "W": self.W.tolist(),
                "mu": self.mu.tolist(), "sigma": self.sigma.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "HalfSuitValue":
        feats = tuple(d["features"])
        if feats != HS_FEATURES:
            raise ValueError(
                "half-suit value model was fitted on a different feature basis "
                f"({feats}); refusing to load it silently onto {HS_FEATURES}")
        return cls(np.asarray(d["W"], dtype=np.float64),
                   np.asarray(d.get("mu", np.zeros(len(feats)))),
                   np.asarray(d.get("sigma", np.ones(len(feats)))))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path) -> "HalfSuitValue":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def zero(cls) -> "HalfSuitValue":
        return cls(np.zeros((3, N_HS_FEATURES)))


def fit_multinomial(X: np.ndarray, y: np.ndarray, l2: float = 1.0,
                    iters: int = 400, lr: float = 0.1) -> HalfSuitValue:
    """Multinomial logistic regression, fitted with Adam on standardised inputs.

    Plain gradient descent on raw features did not converge here: the columns
    span two orders of magnitude in scale, and an under-converged fit reported a
    held-out log-loss WORSE than the class prior while looking superficially
    reasonable. Standardising (leaving the intercept column alone) and using an
    adaptive optimiser fixes it. Convergence is not assumed: the caller compares
    against the class prior and a one-feature baseline on games held out whole.
    """
    try:
        import torch
    except ImportError as e:                      # pragma: no cover
        #: Not a declared runtime dependency: the engine, every duel and every
        #: figure in the strength ladder run on numpy alone, and torch is
        #: needed only to FIT the half-suit value model. A bare
        #: ModuleNotFoundError here sent a reader of the paper's reproduction
        #: section looking for a missing file.
        raise ImportError(
            "fitting the half-suit value model needs PyTorch, which is an "
            "OPTIONAL dependency of this project:\n"
            "    pip install -r requirements-learn.txt\n"
            "Nothing else in the repository requires it -- the engine, the "
            "duels and the strength results all run on numpy."
        ) from e

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    const = sd < 1e-9
    mu = np.where(const, 0.0, mu)
    sd = np.where(const, 1.0, sd)
    Xs = (X - mu) / sd

    xt = torch.tensor(Xs, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)
    W = torch.zeros((3, X.shape[1]), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([W], lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(iters):
        opt.zero_grad()
        loss = lossf(xt @ W.T, yt) + l2 * (W * W).sum() / len(y)
        loss.backward()
        opt.step()
    return HalfSuitValue(W.detach().numpy(), mu, sd)


def log_loss(model: HalfSuitValue, X: np.ndarray, y: np.ndarray) -> float:
    P = model.probs(X)
    return float(-np.mean(np.log(np.maximum(P[np.arange(len(y)), y], 1e-12))))


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------

def _renormalise(row: np.ndarray) -> np.ndarray:
    s = row.sum()
    return row / s if s > 0 else row


def ask_delta_values(ctx, asks, model: HalfSuitValue) -> tuple:
    """``(delta_success, delta_failure, p_success)`` per candidate ask.

    The two counterfactual posteriors are local updates on the half-suit's
    marginal block, which is what makes this affordable: a success pins the
    asked card to us, a failure removes the target from that card's candidates.
    Both are exact statements about the card in question; the second-order
    effect on the other cards of the block (they compete for the same quota
    slots) is not modelled, and that approximation is stated rather than hidden.
    """
    M = ctx.M
    me = ctx.me
    n = len(asks)
    ds = np.zeros(n)
    df = np.zeros(n)
    ps = np.zeros(n)
    base_cache: dict[int, float] = {}
    for i, a in enumerate(asks):
        hs = a.card // 6
        lo = hs * 6
        pi = float(M[a.card, a.target])
        ps[i] = pi
        if hs not in base_cache:
            base_cache[hs] = float(model.value(half_suit_features(ctx, hs)))
        v0 = base_cache[hs]

        saved = M[lo:lo + 6].copy()
        # success: the card is ours, and perfectly localised
        M[a.card] = 0.0
        M[a.card, me] = 1.0
        ds[i] = float(model.value(half_suit_features(ctx, hs))) - v0
        M[lo:lo + 6] = saved
        # failure: the target does not have it
        row = M[a.card].copy()
        row[a.target] = 0.0
        M[a.card] = _renormalise(row)
        df[i] = float(model.value(half_suit_features(ctx, hs))) - v0
        M[lo:lo + 6] = saved
    return ds, df, ps


def score_asks_by_value(ctx, asks, model: HalfSuitValue,
                        turn_weight: float = 0.0,
                        expose_weight: float = 0.0,
                        keep_value: float = 0.0) -> np.ndarray:
    """Expected change in sets from each candidate ask.

    ``keep_value`` is the worth, in sets, of STILL BEING ON THE MOVE after a
    successful ask. Without it this function prices the card and not the tempo,
    which is the larger half of what an ask buys: a hit means you ask again,
    and again, until you miss. ``ask_delta_values`` cannot see that -- it builds
    both counterfactuals by editing the half-suit's marginal block, so
    ``turn_is_ours``, feature 10 of the value model, is identical in the success
    and failure branches. The turn entered only through ``turn_weight``, which
    is a cost on failure with no matching credit for success.

    That asymmetry is not cosmetic, and it was measured rather than argued
    (``results/value_objective_diag.json``, scripts4/value_objective_diag.py):
    over real positions the objective picks asks whose probability of success
    is far below the ones the champion picks, and REMOVING the turn term makes
    it pick better ones, because ``turn_weight * (1 - p) * turn_risk`` is
    largest exactly where ``p`` is smallest. A cost on failure, with nothing on
    the other side, is a subsidy for long shots.

    Since the credit enters as ``keep_value * p``, it is a probability-of-
    success term with a weight in sets -- which is what the heuristic objective
    has carried at weight 1.0 all along, in its own units. The claim here is
    only that the same quantity belongs in this objective too, expressed in
    sets; what that number should BE is a measurement, not a guess, and until
    one exists this defaults to 0.0 and changes nothing.
    """
    ds, df, ps = ask_delta_values(ctx, asks, model)
    score = ps * ds + (1.0 - ps) * df
    if keep_value:
        score = score + keep_value * ps
    if turn_weight or expose_weight:
        for i, a in enumerate(asks):
            fail = 1.0 - ps[i]
            if turn_weight:
                score[i] += turn_weight * fail * ctx.turn_risk[a.target]
            if expose_weight:
                score[i] -= expose_weight * fail * ctx.exposure[a.target]
    return score
