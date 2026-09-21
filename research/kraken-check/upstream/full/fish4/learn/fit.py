"""Paired regression of the ask objective.

THE MODEL
---------
The v0.4 objective is linear by construction (``fish4.askfeat.score_asks``):

    score(a) = P(success | a) + sum_j w_j * f_j(a)

so the thing to learn is the vector ``w`` over ``fish4.askfeat.TERM_NAMES``. Let
``Qhat(a)`` be the CRN rollout estimate of an ask's value (``rollout.py``). For
candidates ``a``, ``a'`` in the SAME position,

    Delta_Q = Qhat(a) - Qhat(a'),   Delta_p = p(a) - p(a'),
    Delta_f = f(a) - f(a')

and the regression is

    Delta_Q - Delta_p  =  Delta_f . w  +  error.

Three things about that equation are load-bearing.

**Differencing removes the position.** A position's own value level - who is
ahead, how many sets are left - is the dominant source of variance in ``Qhat``
and is completely uninformative about which ask to pick. It is a per-position
intercept, and it vanishes exactly under differencing. There is deliberately no
intercept term in the fit: the data are antisymmetric, so an intercept would
have to be zero anyway, and each unordered pair is used exactly once (including
both orientations would duplicate every row and halve the standard errors for no
information).

**The p_success coefficient is pinned to 1.0.** That is the scale convention of
``AskWeights``: every other weight is read "in units of probability of success".
Pinning it is what makes the fitted vector drop straight into ``FishBot4``. It
is imposed by moving ``Delta_p`` to the left-hand side, so it costs no degree of
freedom and cannot be quietly relaxed.

**Standard errors must be clustered by position.** Pairs from one position share
candidates, share the K determinized worlds, and share the rollout seeds; their
errors are strongly correlated. Naive OLS standard errors treat 15 pairs from
one position as 15 independent observations and are far too small. This module
reports cluster-robust (CR1) standard errors and a position-level bootstrap, and
reports the naive ones only to show how badly they mislead.

WHAT THE FIT CANNOT DO
----------------------
It cannot learn a better ask than the rollout policy can exploit, and it cannot
learn anything about claiming (see the package docstring: claims must never be
selected by determinized search). It also inherits whatever the continuation
policy is blind to. If the fitted objective does not beat the hand-tuned one in
actual play, the regression diagnostics here are the first place to look for the
reason, and ``FIT.md`` records that reasoning.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from ..askfeat import TERM_NAMES, AskWeights, stale_terms

N_TERMS = len(TERM_NAMES)


# ---------------------------------------------------------------------------
# Assembling pairs
# ---------------------------------------------------------------------------

@dataclass
class PositionBlock:
    """One harvested position with its rollout values, ready to be paired."""

    pid: str
    game: int
    ply: int
    seat: int
    #: (n_eval,) CRN rollout estimate per evaluated candidate
    Q: np.ndarray
    #: (n_eval,) success probability per evaluated candidate
    p: np.ndarray
    #: (n_eval, N_TERMS) feature rows per evaluated candidate
    F: np.ndarray
    #: (n_eval, K) raw per-world rollout values, kept for variance diagnostics
    V: Optional[np.ndarray] = None
    #: rank of each evaluated candidate under the incumbent objective, so the
    #: fit can be restricted to the near-top candidates as a robustness check
    top_rank: Optional[np.ndarray] = None
    #: memo of the per-block pair layout. The bootstrap and the permutation
    #: test rebuild the pair matrix hundreds of times over the same blocks, and
    #: the feature differences never change, so they are computed once.
    _memo: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return int(self.Q.shape[0])

    def layout(self, max_rank: Optional[int] = None):
        """``(keep, ii, jj, dF)`` for this block, memoised per ``max_rank``."""
        got = self._memo.get(max_rank)
        if got is not None:
            return got
        keep = np.arange(self.n)
        if max_rank is not None and self.top_rank is not None:
            keep = keep[self.top_rank[keep] < max_rank]
        if len(keep) < 2:
            got = (keep, None, None, None)
        else:
            ii, jj = np.triu_indices(len(keep), k=1)
            Fk = self.F[keep]
            got = (keep, ii, jj, Fk[ii] - Fk[jj])
        self._memo[max_rank] = got
        return got


def build_blocks(positions: Iterable[dict], rollouts: dict,
                 zero_terms: Sequence[str] = ()) -> list[PositionBlock]:
    """Join harvested positions with their rollout matrices.

    ``zero_terms`` names columns whose stored values are known to have been
    computed by a formula the engine no longer uses. Their columns are set to
    zero rather than trusted, which makes a ridge fit return exactly 0.0 for
    them. That is not the same statement as "fitted and came out at zero", so
    the caller is required to NAME them -- there is no automatic fallback --
    and is expected to say so in whatever it writes out.

    The alternative, dropping the column from the basis, changes the width of
    the weight vector and every consumer of it. Zeroing keeps the shape and
    costs one term's worth of information, which is the right trade when the
    term is one of eleven and the champion weights it at 0.0 anyway.
    """
    from .rollout import ILLEGAL_VALUE

    # Validated once, before anything is read: a typo in zero_terms would
    # otherwise leave the stale column in the fit and raise -- or not raise --
    # depending on what else happened to be stale.
    for t in zero_terms:
        if t not in TERM_NAMES:
            raise ValueError(f"zero_terms names {t!r}, which is not one of "
                             f"{tuple(TERM_NAMES)}")
    out: list[PositionBlock] = []
    for rec in positions:
        V = rollouts.get(rec["pid"])
        if V is None:
            continue
        V = np.asarray(V, dtype=np.float64)
        if V.ndim != 2 or V.shape[0] < 2:
            continue
        if np.any(V <= ILLEGAL_VALUE + 1e-9):
            continue                      # loud sentinel: drop, never average in
        idx = list(rec["eval_idx"])
        p_all = np.asarray(rec["p"], dtype=np.float64)
        F_all = np.asarray(rec["f"], dtype=np.float64)
        # The feature basis is versioned by the data, not assumed. If someone
        # adds or reorders a term in fish4/askfeat.py after a harvest, a fit
        # against the stale matrix would silently attribute one term's effect
        # to another, so refuse loudly instead. (This has already happened once
        # during development: TERM_NAMES gained an eleventh term, "signal",
        # while a harvest was in flight.)
        stored = rec.get("terms")
        if stored is not None and tuple(stored) != tuple(TERM_NAMES):
            raise ValueError(
                f"{rec['pid']}: harvested with basis {tuple(stored)} but "
                f"fish4.askfeat.TERM_NAMES is now {tuple(TERM_NAMES)}; "
                f"re-harvest before fitting")
        # The names matching is only half the check. A term whose FORMULA
        # changed while its name stayed put produces a column that no longer
        # means what the fitted weight would claim, and the name check cannot
        # see it. `claim` has already been corrected this way, after this
        # dataset was harvested.
        bad = [t for t in stale_terms(rec.get("tv")) if t not in zero_terms]
        if bad:
            raise ValueError(
                f"{rec['pid']}: harvested with definition version(s) "
                f"{ {n: (rec.get('tv') or [1] * len(TERM_NAMES))[TERM_NAMES.index(n)] for n in bad} } "
                f"for {bad}, which fish4/askfeat.py has since changed. A "
                f"weight fitted on that column would describe a feature the "
                f"engine no longer computes. Re-harvest, or drop the term "
                f"from the basis deliberately -- not by ignoring this.")
        if F_all.shape[1] != N_TERMS:
            raise ValueError(f"{rec['pid']}: feature width {F_all.shape[1]} "
                             f"!= {N_TERMS}; TERM_NAMES changed under the data")
        for t in zero_terms:
            F_all[:, TERM_NAMES.index(t)] = 0.0
        p = p_all[idx]
        F = F_all[idx]
        # Incumbent ranking, recomputed from the stored features so the
        # robustness check does not depend on anything not on disk.
        inc = p_all + F_all @ AskWeights().as_vector()
        order = np.argsort(-inc)
        rank_of = np.empty(len(p_all), dtype=int)
        rank_of[order] = np.arange(len(p_all))
        out.append(PositionBlock(
            pid=rec["pid"], game=rec["game"], ply=rec["ply"], seat=rec["seat"],
            Q=V.mean(axis=1), p=p, F=F, V=V, top_rank=rank_of[idx]))
    return out


@dataclass
class PairSet:
    X: np.ndarray            # (n_pairs, N_TERMS)   Delta_f
    y: np.ndarray            # (n_pairs,)           Delta_Q - Delta_p
    dq: np.ndarray           # (n_pairs,)           Delta_Q
    dp: np.ndarray           # (n_pairs,)           Delta_p
    g: np.ndarray            # (n_pairs,)           cluster index = position
    pids: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def n_clusters(self) -> int:
        return int(len(self.pids))


def assemble(blocks: Sequence[PositionBlock],
             perm: Optional[Sequence[np.ndarray]] = None,
             sign: Optional[Sequence[float]] = None,
             max_rank: Optional[int] = None) -> PairSet:
    """Every unordered pair of evaluated candidates, one orientation each.

    ``perm[b]`` re-labels the candidates of block ``b``: ``Qhat`` and ``p``
    travel together under the permutation while the feature rows stay put, so
    the null it generates is "the ``f`` terms explain nothing beyond
    ``p_success``" rather than something weaker.

    ``sign[b]`` flips the sign of every row of block ``b`` (the sign-flip null;
    ``X`` is deliberately NOT flipped, since flipping both would leave the
    estimate unchanged and test nothing).

    ``max_rank`` restricts to candidates inside the top ``max_rank`` of the
    incumbent objective.
    """
    Xs, dqs, dps, gs, pids = [], [], [], [], []
    for b, blk in enumerate(blocks):
        keep, ii, jj, dF = blk.layout(max_rank)
        if ii is None:
            continue
        if perm is None:
            Qk, pk = blk.Q[keep], blk.p[keep]
        else:
            o = np.asarray(perm[b], dtype=int)
            Qk, pk = blk.Q[keep][o], blk.p[keep][o]
        s = 1.0 if sign is None else float(sign[b])
        Xs.append(dF)
        dqs.append(s * (Qk[ii] - Qk[jj]))
        dps.append(s * (pk[ii] - pk[jj]))
        gs.append(np.full(len(ii), len(pids), dtype=np.int64))
        pids.append(blk.pid)
    if not pids:
        raise ValueError("no pairs assembled")
    dq = np.concatenate(dqs)
    dp = np.concatenate(dps)
    return PairSet(X=np.vstack(Xs), y=dq - dp, dq=dq, dp=dp,
                   g=np.concatenate(gs), pids=pids)


# ---------------------------------------------------------------------------
# Ridge with a pinned p_success coefficient
# ---------------------------------------------------------------------------

def _scales(X: np.ndarray) -> np.ndarray:
    """Column RMS, used to make the ridge penalty scale-free.

    Not centred: the rows are antisymmetric differences and centring them would
    reintroduce the intercept that differencing just removed. A column that is
    identically zero (for instance ``certain`` in a batch where no candidate is
    a provable steal) gets scale 1 here -- but see :func:`dead_columns`. This
    docstring used to go on to say such a column "will simply receive weight
    0", which was an assertion, was never tested, and is FALSE at ``lam = 0``:
    a zero column makes the Gram matrix singular and ``np.linalg.solve``
    raises. The lambda grid starts at 0.0, so that path is reachable. It is
    now handled explicitly by the two callers rather than asserted here.
    """
    s = np.sqrt((X ** 2).mean(axis=0))
    s[s < 1e-12] = 1.0
    return s


def dead_columns(X: np.ndarray) -> np.ndarray:
    """Boolean mask of columns that are identically zero.

    Such a column carries no information and must be held out of the solve:
    with ``lam = 0`` it makes ``Z'Z`` singular, and with ``lam > 0`` the
    penalty hides that by returning a coefficient that is zero anyway. Holding
    it out and writing 0 into its slot gives the same answer everywhere the
    solve succeeded before, and an answer instead of an exception where it did
    not.
    """
    return np.sqrt((X ** 2).mean(axis=0)) < 1e-12


def ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge coefficients, no intercept, penalty applied in scaled space.

    Identically-zero columns are held out of the solve and receive exactly
    zero. With no such column the original expression is evaluated unchanged,
    so every previously recorded coefficient is reproduced to the bit -- an
    equivalent-but-resliced expression is not good enough here, because BLAS
    rounds a differently-laid-out matrix differently and "only zero columns are
    affected" then stops being exactly true.
    """
    s = _scales(X)
    Z = X / s
    live = ~dead_columns(X)
    if live.all():
        # The overwhelmingly common case, and taken by the ORIGINAL code path
        # rather than by an equivalent one. Boolean-mask indexing would copy Z
        # into a different memory layout, and BLAS then rounds differently --
        # enough to move a recorded coefficient in its last bits. "Only zero
        # columns are affected" has to be exactly true, not nearly.
        G = Z.T @ Z + lam * np.eye(Z.shape[1])
        return np.linalg.solve(G, Z.T @ y) / s
    w_s = np.zeros(Z.shape[1], dtype=np.float64)
    if live.any():
        Zl = Z[:, live]
        G = Zl.T @ Zl + lam * np.eye(Zl.shape[1])
        w_s[live] = np.linalg.solve(G, Zl.T @ y)
    return w_s / s


def _sandwich(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float,
              g: Optional[np.ndarray]) -> np.ndarray:
    """Covariance of the ridge estimator, in ORIGINAL feature units.

    ``g is None`` gives the naive homoskedastic OLS covariance
    ``sigma^2 (Z'Z + lam I)^-1 Z'Z (Z'Z + lam I)^-1``; ``g`` given gives the
    CR1 cluster-robust sandwich. For ``lam = 0`` the first reduces to the
    textbook ``sigma^2 (X'X)^-1`` and the second to the usual clustered
    sandwich; for ``lam > 0`` both ignore the shrinkage bias, which is standard
    and is stated in the report rather than hidden.
    """
    s = _scales(X)
    Z_full = X / s
    live = ~dead_columns(X)
    # A dead column has zero variance, so its coefficient is not estimated and
    # its variance is zero. Inverting the full Gram would raise at lam = 0.
    # With nothing dead the full matrix is used unsliced, so the arithmetic is
    # identical to what it was rather than merely equivalent.
    Z = Z_full if live.all() else Z_full[:, live]
    n, k = Z.shape
    if k == 0:
        return np.zeros((Z_full.shape[1], Z_full.shape[1]))
    A = np.linalg.inv(Z.T @ Z + lam * np.eye(k))
    e = y - Z @ ((w * s) if live.all() else (w * s)[live])
    if g is None:
        dof = max(1, n - k)
        sigma2 = float(e @ e) / dof
        meat = sigma2 * (Z.T @ Z)
    else:
        meat = np.zeros((k, k))
        uniq = np.unique(g)
        for c in uniq:
            m = g == c
            Zc, ec = Z[m], e[m]
            u = Zc.T @ ec
            meat += np.outer(u, u)
        G = len(uniq)
        corr = (G / max(1, G - 1)) * ((n - 1) / max(1, n - k))
        meat *= corr
    V_s = A @ meat @ A
    if not live.all():
        # Scatter back to full width with zero rows and columns for dead terms,
        # so every caller keeps indexing by TERM_NAMES position.
        V_full = np.zeros((Z_full.shape[1], Z_full.shape[1]))
        idx = np.flatnonzero(live)
        V_full[np.ix_(idx, idx)] = V_s
        V_s = V_full
    S = np.diag(1.0 / s)
    return S @ V_s @ S


def r2_uncentred(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float((y ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def cv_lambda(blocks: Sequence[PositionBlock], lams: Sequence[float],
              n_folds: int = 5, seed: int = 0,
              max_rank: Optional[int] = None) -> tuple[float, list[dict]]:
    """Choose the ridge penalty by K-fold CV **split on positions**.

    Splitting on rows would put pairs from the same position on both sides of
    the split and report a validation error that is really a training error.
    """
    idx = list(range(len(blocks)))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::n_folds] for i in range(n_folds)]
    report = []
    best, best_mse = lams[0], float("inf")
    for lam in lams:
        tot, cnt = 0.0, 0
        for f in range(n_folds):
            va = set(folds[f])
            tr_blocks = [blocks[i] for i in idx if i not in va]
            va_blocks = [blocks[i] for i in folds[f]]
            if len(tr_blocks) < 2 or not va_blocks:
                continue
            tr = assemble(tr_blocks, max_rank=max_rank)
            va_ps = assemble(va_blocks, max_rank=max_rank)
            w = ridge(tr.X, tr.y, lam)
            e = va_ps.y - va_ps.X @ w
            tot += float(e @ e)
            cnt += va_ps.n
        mse = tot / max(1, cnt)
        report.append({"lambda": lam, "cv_mse": mse, "n": cnt})
        if mse < best_mse:
            best, best_mse = lam, mse
    return best, report


# ---------------------------------------------------------------------------
# Inference: bootstrap and permutation
# ---------------------------------------------------------------------------

def cluster_bootstrap(blocks: Sequence[PositionBlock], lam: float,
                      n_boot: int = 400, seed: int = 11,
                      max_rank: Optional[int] = None) -> np.ndarray:
    """Resample POSITIONS with replacement; refit each time. (n_boot, N_TERMS)."""
    rng = random.Random(seed)
    n = len(blocks)
    out = np.zeros((n_boot, N_TERMS))
    for b in range(n_boot):
        pick = [blocks[rng.randrange(n)] for _ in range(n)]
        ps = assemble(pick, max_rank=max_rank)
        out[b] = ridge(ps.X, ps.y, lam)
    return out


def permutation_test(blocks: Sequence[PositionBlock], lam: float,
                     n_perm: int = 300, seed: int = 23,
                     mode: str = "signflip",
                     max_rank: Optional[int] = None) -> dict:
    """Is the fit distinguishable from noise?

    ``mode="signflip"`` (primary) flips the sign of every pair belonging to a
    randomly chosen subset of positions. The rows of a position flip together,
    so the within-position dependence is preserved exactly. Under the null that
    ``w = 0`` the residual is the pure rollout noise of an arbitrarily oriented
    difference, which is symmetric about zero, so this is a valid randomization
    test of exactly the coefficients being reported.

    ``mode="labels"`` re-labels the candidates within each position, carrying
    ``Qhat`` and ``p`` together so that each rollout value keeps its own success
    probability and only the ``f`` rows are re-paired. That tests the same
    hypothesis by a different route and is reported as a cross-check.

    Permuting ``Qhat`` while leaving ``p`` behind would be INVALID here, and it
    is the obvious thing to reach for: the target keeps its ``-Delta_p`` term,
    which genuinely correlates with the features, so the null distribution would
    be contaminated with real signal and the test would silently lose power.
    """
    rng = random.Random(seed)
    obs = assemble(blocks, max_rank=max_rank)
    w0 = ridge(obs.X, obs.y, lam)
    r0 = r2_uncentred(obs.y, obs.X @ w0)
    ge = np.zeros(N_TERMS)
    r2_ge = 0
    r2_null = []
    for _ in range(n_perm):
        if mode == "signflip":
            sign = [1.0 if rng.random() < 0.5 else -1.0 for _ in blocks]
            ps = assemble(blocks, sign=sign, max_rank=max_rank)
        elif mode == "labels":
            perm = []
            for blk in blocks:
                keep = blk.layout(max_rank)[0]
                order = list(range(len(keep)))
                rng.shuffle(order)
                perm.append(order)
            ps = assemble(blocks, perm=perm, max_rank=max_rank)
        else:
            raise ValueError(f"unknown permutation mode {mode!r}")
        w = ridge(ps.X, ps.y, lam)
        ge += (np.abs(w) >= np.abs(w0) - 1e-15).astype(float)
        r = r2_uncentred(ps.y, ps.X @ w)
        r2_null.append(r)
        if r >= r0:
            r2_ge += 1
    # (count + 1) / (B + 1): the observed value is one of the possible
    # randomizations, so a p-value of exactly zero is not a thing.
    return {
        "mode": mode,
        "n_perm": n_perm,
        "p_coef": {n: float((ge[i] + 1) / (n_perm + 1))
                   for i, n in enumerate(TERM_NAMES)},
        "p_r2": float((r2_ge + 1) / (n_perm + 1)),
        "r2_observed": r0,
        "r2_null_mean": float(np.mean(r2_null)),
        "r2_null_p95": float(np.quantile(r2_null, 0.95)),
    }


# ---------------------------------------------------------------------------
# The linear fit, packaged
# ---------------------------------------------------------------------------

def p_response(blocks: Sequence[PositionBlock],
               edges: Sequence[float] = (0.0, 0.001, 0.15, 0.35, 0.65, 0.999,
                                         1.0001),
               n_boot: int = 400, seed: int = 31) -> dict:
    """Rollout value against success probability, with the position removed.

    This is the sanity check the whole study rests on, and it is worth doing
    before believing any coefficient. Within each position, subtract the mean
    ``Qhat`` over the evaluated candidates - that removes the position's own
    value level exactly, the same thing pairing does - then bin the remaining
    deviations by ``p``. If a guaranteed steal is not worth measurably more
    than an ask that cannot possibly land, then the rollout target carries no
    information about the single quantity the incumbent objective is built on,
    and no regression on top of it can be trusted.

    The interval is a position-level bootstrap, because candidates within a
    position are not independent.
    """
    rows = []
    for b, blk in enumerate(blocks):
        c = blk.Q - blk.Q.mean()
        for i in range(blk.n):
            rows.append((b, float(blk.p[i]), float(c[i])))
    if not rows:
        return {}
    bidx = np.array([r[0] for r in rows])
    pv = np.array([r[1] for r in rows])
    cq = np.array([r[2] for r in rows])
    lab = np.digitize(pv, np.asarray(edges[1:-1]))
    rng = np.random.default_rng(seed)
    n_blocks = len(blocks)
    out = []
    for k in range(len(edges) - 1):
        m = lab == k
        if not m.any():
            continue
        boot = np.empty(n_boot)
        for t in range(n_boot):
            pick = rng.integers(0, n_blocks, size=n_blocks)
            sel = np.isin(bidx, pick) & m
            boot[t] = cq[sel].mean() if sel.any() else np.nan
        lo, hi = np.nanquantile(boot, [0.025, 0.975])
        out.append({"p_lo": float(edges[k]), "p_hi": float(edges[k + 1]),
                    "n": int(m.sum()), "mean_centred_Q": float(cq[m].mean()),
                    "ci": [float(lo), float(hi)]})
    # Slope of centred Qhat on p, which is c_p measured a second way.
    A = np.column_stack([pv, np.ones_like(pv)])
    coef, *_ = np.linalg.lstsq(A, cq, rcond=None)
    return {"bins": out, "slope_on_p": float(coef[0])}


def free_p_fit(ps: PairSet, lam: float) -> dict:
    """The same regression with the ``p_success`` coefficient NOT pinned.

    Why this has to be reported alongside the pinned fit. Suppose the rollout
    value really behaves like ``c_p * Delta_p + Delta_f . b`` with ``c_p`` well
    below 1. Pinning ``c_p = 1`` does not just rescale the answer: it forces
    every feature that correlates with ``p`` to take a NEGATIVE coefficient
    purely to cancel the overshoot, and those coefficients then look like
    strategic findings when they are an artefact of the scale convention.
    ``claim``, ``certain`` and ``deplete`` are all direct functions of ``p``, so
    this is not a hypothetical failure mode in this basis.

    The ranking a policy actually uses is invariant to positive rescaling, so
    ``b / c_p`` is the ``AskWeights`` vector implied by the unpinned fit, and it
    is a genuinely different candidate policy from the pinned one. Both get
    played, and play decides.
    """
    Xa = np.column_stack([ps.dp, ps.X])
    w = ridge(Xa, ps.dq, lam)
    V = _sandwich(Xa, ps.dq, w, lam, ps.g)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    c_p = float(w[0])
    b = w[1:]
    rescaled = (b / c_p) if abs(c_p) > 1e-6 else None
    return {
        "c_p": c_p,
        "c_p_se_cluster": float(se[0]),
        "c_p_t_vs_one": float((c_p - 1.0) / se[0]) if se[0] > 0 else float("nan"),
        "b": {n: float(b[i]) for i, n in enumerate(TERM_NAMES)},
        "b_se_cluster": {n: float(se[i + 1]) for i, n in enumerate(TERM_NAMES)},
        "weights_rescaled": ({n: float(rescaled[i])
                              for i, n in enumerate(TERM_NAMES)}
                             if rescaled is not None else None),
        "r2_dq": r2_uncentred(ps.dq, Xa @ w),
    }


def fit_linear(blocks: Sequence[PositionBlock],
               lams: Sequence[float] = (0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
               n_boot: int = 400, n_perm: int = 300, seed: int = 5,
               max_rank: Optional[int] = None) -> dict:
    lam, cv = cv_lambda(blocks, lams, seed=seed, max_rank=max_rank)
    ps = assemble(blocks, max_rank=max_rank)
    w = ridge(ps.X, ps.y, lam)
    yhat = ps.X @ w
    V_cl = _sandwich(ps.X, ps.y, w, lam, ps.g)
    V_naive = _sandwich(ps.X, ps.y, w, lam, None)
    se_cl = np.sqrt(np.clip(np.diag(V_cl), 0, None))
    se_nv = np.sqrt(np.clip(np.diag(V_naive), 0, None))
    boot = cluster_bootstrap(blocks, lam, n_boot=n_boot, seed=seed + 1,
                             max_rank=max_rank)
    perm = permutation_test(blocks, lam, n_perm=n_perm, seed=seed + 2,
                            mode="signflip", max_rank=max_rank)
    perm_lab = permutation_test(blocks, lam, n_perm=max(60, n_perm // 3),
                                seed=seed + 3, mode="labels",
                                max_rank=max_rank)
    # OLS (lam = 0) coefficients too: with lam > 0 the sandwich ignores the
    # shrinkage bias, so the unpenalised fit is the honest reference point.
    w_ols = ridge(ps.X, ps.y, 0.0)
    se_ols_cl = np.sqrt(np.clip(np.diag(
        _sandwich(ps.X, ps.y, w_ols, 0.0, ps.g)), 0, None))
    free = free_p_fit(ps, lam)

    coefs = {}
    for i, name in enumerate(TERM_NAMES):
        lo, hi = np.quantile(boot[:, i], [0.025, 0.975])
        coefs[name] = {
            "w": float(w[i]),
            "se_cluster": float(se_cl[i]),
            "se_naive": float(se_nv[i]),
            "se_ratio": float(se_cl[i] / se_nv[i]) if se_nv[i] > 0 else float("nan"),
            "t_cluster": float(w[i] / se_cl[i]) if se_cl[i] > 0 else float("nan"),
            "boot_ci": [float(lo), float(hi)],
            "perm_p": perm["p_coef"][name],
            "w_ols": float(w_ols[i]),
            "se_ols_cluster": float(se_ols_cl[i]),
            "incumbent": float(getattr(AskWeights(), name)),
        }
    return {
        "n_positions": len(blocks),
        "n_pairs": ps.n,
        "n_clusters": ps.n_clusters,
        "lambda": lam,
        "cv": cv,
        "coefficients": coefs,
        "weights": {n: float(w[i]) for i, n in enumerate(TERM_NAMES)},
        "r2_residual": r2_uncentred(ps.y, yhat),
        "r2_dq": r2_uncentred(ps.dq, ps.dp + yhat),
        "r2_dq_incumbent": r2_uncentred(
            ps.dq, ps.dp + ps.X @ AskWeights().as_vector()),
        "r2_dq_p_only": r2_uncentred(ps.dq, ps.dp),
        "free_p": free,
        "permutation": perm,
        "permutation_labels": perm_lab,
        "se_ratio_mean": float(np.mean(
            [c["se_ratio"] for c in coefs.values() if np.isfinite(c["se_ratio"])])),
        "max_rank": max_rank,
    }


def weights_from_fit(fit: dict) -> AskWeights:
    return AskWeights(**{n: float(fit["weights"][n]) for n in TERM_NAMES})


def agent_kwargs(weights: AskWeights, ndigits: int = 4) -> dict:
    return {f"w_{n}": round(float(getattr(weights, n)), ndigits)
            for n in TERM_NAMES}


# ---------------------------------------------------------------------------
# Nonlinear alternative: a small antisymmetric MLP
# ---------------------------------------------------------------------------

def fit_mlp(blocks: Sequence[PositionBlock], hidden: int = 32,
            layers: int = 2, epochs: int = 600, lr: float = 3e-3,
            weight_decay: float = 1e-4, val_frac: float = 0.25,
            seed: int = 17, max_rank: Optional[int] = None) -> dict:
    """Does a nonlinear ask objective beat the linear one?

    The network is ``g_theta: R^10 -> R`` and the loss is on the SAME paired
    target the linear model uses:

        Delta_Q - Delta_p  ~  g(f(a)) - g(f(a'))

    so the comparison is like for like: a linear ``g`` recovers the linear model
    exactly, and the only difference under test is the functional form of the
    per-ask objective. Train and validation are split BY POSITION, and the
    linear baseline is refitted on the same training split, so the reported
    comparison is out-of-sample for both.

    If this does not beat the linear fit, that is a result worth reporting: it
    says the feature basis, not the shape of the objective, is what limits the
    fit.
    """
    try:
        import torch
        from torch import nn
    except ImportError:  # pragma: no cover - torch is OPTIONAL, see
                         # requirements-learn.txt; this path is the reason the
                         # rest of the pipeline still runs without it
        return {"available": False, "reason": "torch not installed"}

    # One thread, deliberately. This machine has 8 cores shared with other
    # jobs and the rest of the pipeline is capped at 2 workers; torch's default
    # is to grab every core, which both breaks that budget and - measured here -
    # made this fit 100x SLOWER through thread contention on a problem far too
    # small to parallelise (294s at the default, ~3s at one thread).
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    idx = list(range(len(blocks)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac))
    va_blocks = [blocks[i] for i in idx[:n_val]]
    tr_blocks = [blocks[i] for i in idx[n_val:]]
    tr = assemble(tr_blocks, max_rank=max_rank)
    va = assemble(va_blocks, max_rank=max_rank)

    # Feature rows of both members of each pair are needed, not just their
    # difference, because g is nonlinear. Rebuild them alongside the pair set.
    def sides(bs):
        """Both members of each pair, in exactly ``assemble``'s row order."""
        A, B, Y = [], [], []
        for blk in bs:
            keep, ii, jj, _ = blk.layout(max_rank)
            if ii is None:
                continue
            Fk, Qk, pk = blk.F[keep], blk.Q[keep], blk.p[keep]
            A.append(Fk[ii])
            B.append(Fk[jj])
            Y.append((Qk[ii] - Qk[jj]) - (pk[ii] - pk[jj]))
        return (np.vstack(A), np.vstack(B), np.concatenate(Y))

    Atr, Btr, Ytr = sides(tr_blocks)
    Ava, Bva, Yva = sides(va_blocks)
    s = np.sqrt((np.vstack([Atr, Btr]) ** 2).mean(axis=0))
    s[s < 1e-12] = 1.0

    def T(x):
        return torch.tensor(x / s, dtype=torch.float32)

    at, bt, yt = T(Atr), T(Btr), torch.tensor(Ytr, dtype=torch.float32)
    av, bv, yv = T(Ava), T(Bva), torch.tensor(Yva, dtype=torch.float32)

    mods: list = []
    d = N_TERMS
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    mods += [nn.Linear(d, 1)]
    net = nn.Sequential(*mods)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_state, best_epoch = float("inf"), None, -1
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        pred = (net(at) - net(bt)).squeeze(-1)
        loss = ((pred - yt) ** 2).mean()
        loss.backward()
        opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            with torch.no_grad():
                pv = (net(av) - net(bv)).squeeze(-1)
                vl = float(((pv - yv) ** 2).mean())
            hist.append({"epoch": ep, "train_mse": float(loss.detach()),
                         "val_mse": vl})
            if vl < best_val:
                best_val, best_epoch = vl, ep
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
    if best_state is not None:
        net.load_state_dict(best_state)

    lam, _ = cv_lambda(tr_blocks, (0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
                       seed=seed, max_rank=max_rank)
    w_lin = ridge(tr.X, tr.y, lam)
    lin_val = float(((va.y - va.X @ w_lin) ** 2).mean())
    zero_val = float((va.y ** 2).mean())
    with torch.no_grad():
        pv = (net(av) - net(bv)).squeeze(-1).numpy()
    mlp_val = float(((Yva - pv) ** 2).mean())
    return {
        "available": True,
        "hidden": hidden, "layers": layers, "epochs": epochs,
        "best_epoch": best_epoch,
        "n_train_pairs": int(tr.n), "n_val_pairs": int(va.n),
        "n_train_positions": len(tr_blocks), "n_val_positions": len(va_blocks),
        "val_mse_mlp": mlp_val,
        "val_mse_linear": lin_val,
        "val_mse_zero": zero_val,
        "val_r2_mlp": 1.0 - mlp_val / zero_val if zero_val else float("nan"),
        "val_r2_linear": 1.0 - lin_val / zero_val if zero_val else float("nan"),
        "improvement": lin_val - mlp_val,
        "improvement_pct_of_linear_r2": (
            (lin_val - mlp_val) / (zero_val - lin_val) * 100.0
            if zero_val - lin_val > 0 else float("nan")),
        "history_tail": hist[-5:],
        "lambda_linear": lam,
    }


# ---------------------------------------------------------------------------
# Synthetic data, for the tests
# ---------------------------------------------------------------------------

def synthetic_blocks(w_true: np.ndarray, n_positions: int = 300,
                     n_cand: int = 6, K: int = 16,
                     cand_noise: float = 0.4, world_noise: float = 1.2,
                     seed: int = 3) -> list[PositionBlock]:
    """Positions with a planted linear objective, for recovery tests.

    The noise has the structure the real data has: a per-position level (which
    differencing removes), a per-CANDIDATE shock (which it does not, and which
    is what makes pairs sharing a candidate correlated - the reason naive
    standard errors are wrong), and per-world rollout noise.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    for i in range(n_positions):
        F = rng.normal(size=(n_cand, N_TERMS))
        F[:, 0] = rng.integers(0, 6, size=n_cand)     # depth is a small count
        p = rng.uniform(0.0, 1.0, size=n_cand)
        level = rng.normal(0.0, 2.0)
        true = level + p + F @ w_true + rng.normal(0.0, cand_noise, size=n_cand)
        V = true[:, None] + rng.normal(0.0, world_noise, size=(n_cand, K))
        blocks.append(PositionBlock(
            pid=f"syn:{i}", game=i, ply=0, seat=0,
            Q=V.mean(axis=1), p=p, F=F, V=V,
            top_rank=np.argsort(np.argsort(-(p + F @ w_true)))))
    return blocks
