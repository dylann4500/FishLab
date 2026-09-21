"""Bradley-Terry rating fit over pairwise matchup results.

Design notes (this file was rewritten after an audit found the first version
unsound; see RESEARCH_LOG.md "ratings v1 retracted"):

- PERFECT SEPARATION IS THE NORM HERE. When a policy wins 400/400 paired
  deals, the unregularized maximum-likelihood rating gap is INFINITE, and a
  fixed-iteration gradient loop silently reports whatever it reached when it
  stopped. We therefore fit a MAP estimate with a Gaussian prior on ratings
  (sd = ``prior_sd`` Elo). Separated pairs then yield finite, reproducible
  numbers that are honest LOWER BOUNDS on the true gap, flagged as such.
- Optimization is damped Newton on the full Hessian, run to a gradient
  tolerance, so results do not depend on an iteration budget.
- Uncertainty uses the inverse of the full Fisher information (plus prior),
  not per-player diagonals, so correlations between ratings are respected.
"""

from __future__ import annotations

import math

import numpy as np

ANCHOR_RATING = 200.0
SCALE = 400.0 / math.log(10.0)   # chess-Elo logistic scale


def _expected(diff: float) -> float:
    return 1.0 / (1.0 + math.exp(-diff / SCALE))


def fit_ratings(results: list[tuple[str, str, float, int]],
                anchor: str = "random", prior_sd: float = 1500.0,
                tol: float = 1e-9, max_iter: int = 200) -> dict:
    """results: (name_x, name_y, pair_score_x in [0,1], n_pairs).

    Returns {name: {"rating", "stderr", "separated"}} where ``separated``
    marks a policy whose every matchup was a shutout (its rating is a
    prior-regularized bound, not a measured value).
    """
    names = sorted({n for r in results for n in (r[0], r[1])})
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    rows = [(idx[a], idx[b], s, n) for a, b, s, n in results if n > 0]
    r = np.zeros(k)
    prior_prec = 1.0 / (prior_sd ** 2)
    hess = np.eye(k) * prior_prec

    for _ in range(max_iter):
        grad = -prior_prec * r
        hess = np.eye(k) * prior_prec
        for i, j, s, n in rows:
            e = _expected(r[i] - r[j])
            g = n * (s - e) / SCALE
            grad[i] += g
            grad[j] -= g
            w = n * e * (1 - e) / (SCALE ** 2)
            hess[i, i] += w
            hess[j, j] += w
            hess[i, j] -= w
            hess[j, i] -= w
        if np.max(np.abs(grad)) < tol:
            break
        step = np.linalg.solve(hess, grad)
        norm = np.max(np.abs(step))
        if norm > 400.0:               # damping keeps steps sane near separation
            step *= 400.0 / norm
        r += step

    cov = np.linalg.inv(hess)
    played: dict[int, list[float]] = {i: [] for i in range(k)}
    for i, j, s, n in rows:
        played[i].append(s)
        played[j].append(1.0 - s)
    separated = {names[i]: (all(x >= 0.9999 for x in v)
                            or all(x <= 0.0001 for x in v))
                 for i, v in played.items() if v}

    shift = ANCHOR_RATING - (r[idx[anchor]] if anchor in idx else 0.0)
    out = {}
    for name in names:
        i = idx[name]
        if anchor in idx and name != anchor:
            a = idx[anchor]
            var = cov[i, i] + cov[a, a] - 2 * cov[i, a]
        else:
            var = cov[i, i]
        out[name] = {
            "rating": float(r[i] + shift),
            "stderr": float(math.sqrt(max(0.0, var))),
            "separated": bool(separated.get(name, False)),
        }
    return out


def format_table(ratings: dict) -> str:
    lines = [f"{'policy':18s} {'rating':>8s} {'+/-':>7s}  note"]
    for name, d in sorted(ratings.items(), key=lambda kv: -kv[1]["rating"]):
        note = "bound only (undefeated/winless)" if d["separated"] else ""
        lines.append(f"{name:18s} {d['rating']:8.0f} {d['stderr']:7.0f}  {note}")
    return "\n".join(lines)
