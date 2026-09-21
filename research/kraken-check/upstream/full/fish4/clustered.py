"""One cluster-robust interval, used everywhere, so it is right in one place.

Two mistakes this exists to prevent, both found on 2026-08-30:

1. **Dividing by the wrong count.** ``ask_regret.harvest`` walks games in order
   and emits every qualifying ply, so a run reported as "162 positions" is 162
   consecutive plies from EIGHT deals. Positions inside a deal share the hands,
   the history and every earlier decision; an interval that divides by 162
   divides by a number that is not the sample size.

2. **Pairing a cluster-robust standard error with 1.96.** Cluster-robust
   variance is asymptotic in the NUMBER OF CLUSTERS, and these runs have four
   to ten deals. At six clusters the normal critical value understates the
   interval by 31%, at four by 62%. Degrees of freedom are ``k - 1``: the
   estimator has as many independent pieces of information as it has clusters.

Correcting only the first would have replaced one too-narrow interval with
another.
"""

from __future__ import annotations

import math

from fish4.match import _t_critical


def cluster_ci(values, groups, conf: float = 0.95):
    """Mean of ``values`` and a half-width clustered on ``groups``.

    Returns ``(mean, half_width, n_clusters)``. ``half_width`` is ``None`` when
    there is only one cluster -- one cluster is not a sample of clusters, and a
    plausible-looking interval from it is worse than no interval.
    """
    vals = list(values)
    grps = list(groups)
    if len(vals) != len(grps):
        raise ValueError(f"{len(vals)} values against {len(grps)} group labels")
    if not vals:
        raise ValueError("no values")
    by: dict[object, list[float]] = {}
    for v, g in zip(vals, grps):
        by.setdefault(g, []).append(float(v))
    n = len(vals)
    k = len(by)
    mu = sum(vals) / n
    if k < 2:
        return mu, None, k
    acc = sum((sum(v) - mu * len(v)) ** 2 for v in by.values())
    se = math.sqrt(acc * k / (k - 1.0)) / n
    return mu, _t_critical(k - 1, conf) * se, k


def fmt(mean, half_width, n_clusters) -> str:
    """A one-line rendering that always says what the interval rests on."""
    if half_width is None:
        return f"{mean:+.4f}  (one cluster; no interval available)"
    return (f"{mean:+.4f} [{mean - half_width:+.4f}, {mean + half_width:+.4f}]"
            f"  ({n_clusters} clusters)")
