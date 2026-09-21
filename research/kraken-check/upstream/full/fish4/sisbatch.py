"""Vectorised batch drawing for the sequential importance sampler.

WHY
---
Profiling the v0.4 policy showed a single function, ``SISSampler._attempt``,
taking 67% of total runtime (24.9s of 37.4s over three games). That is
unsurprising: it is a tight Python loop over roughly 26 cards, run 128 times per
decision and roughly 110 times per game, so it executes several hundred thousand
times per game.

The loop is sequential *within* a draw but completely independent *across*
draws, so the whole batch can advance one card at a time with every draw
progressing in lockstep. Each card then costs a handful of numpy operations on
``(N, 6)`` arrays instead of ``N`` passes of scalar Python. The proposal
distribution is unchanged - this is the same algorithm, executed differently -
and ``tests4/test_sisbatch.py`` asserts that by comparing the realised
distribution against the analytic path density that the scalar implementation is
already validated against.

DEAD ENDS
---------
A draw can paint itself into a corner (a card with no candidate that still has
quota, or two clauses demanding the same card for different players). In the
scalar sampler that triggers a restart. Here such draws are simply marked dead
and dropped from the batch; the caller may top up. Both are the same
conditioning-on-success, which multiplies every density by the same constant and
so leaves self-normalised weights unaffected. The measured dead-end rate on real
positions is zero in 10,050 draws, so this path is a safety net rather than a
hot one.
"""

from __future__ import annotations

import numpy as np

from fish.cards import NUM_PLAYERS


def _plan(sampler):
    """Per-card constants, precomputed once and cached on the sampler.

    Everything here was previously recomputed inside the per-card loop of every
    batch: candidate arrays, the column each clause's player occupies, and the
    opponent-model slot per candidate column. Hoisting them out is most of the
    difference between the batch path being 1.4x and 5.6x faster than the scalar
    one.
    """
    cached = sampler._batch_plan
    if cached is not None:
        return cached
    plan = []
    for i in range(sampler._n):
        cand = np.asarray(sampler._ocand[i], dtype=np.int64)
        clauses = sampler._oclause[i]
        cps = sampler._oc_player[i]
        cols = []
        for j, ci in enumerate(clauses):
            k = int(np.searchsorted(cand, cps[j]))
            cols.append(k if k < cand.size and cand[k] == cps[j] else -1)
        slots = (np.asarray(sampler._otilt[i], dtype=np.int64)
                 if sampler._otilt is not None else None)
        plan.append((cand, tuple(clauses), tuple(cps), tuple(cols), slots))
    sampler._batch_plan = plan
    return plan


def draw_batch(sampler, rng, n: int):
    """Draw ``n`` worlds at once.

    Returns ``(picks, logq, logl, alive)`` where ``picks`` is an ``(n, n_free)``
    integer array of owners in ``sampler.order`` column order.
    """
    n_free = sampler._n
    if n_free == 0 or n <= 0:
        return (np.zeros((0, 0), dtype=np.int64), np.zeros(0),
                np.zeros(0), np.zeros(0, dtype=bool))
    n_or = len(sampler.ors)
    # numpy's Generator is seeded from the caller's Random so the batch stays a
    # deterministic function of the agent's own RNG stream.
    gen = np.random.default_rng(rng.getrandbits(63))
    plan = _plan(sampler)

    quota = np.tile(np.asarray(sampler.quotas, dtype=np.float64), (n, 1))
    picks = np.zeros((n, n_free), dtype=np.int64)
    logq = np.zeros(n, dtype=np.float64)
    alive = np.ones(n, dtype=bool)
    if n_or:
        live_left = np.tile(
            np.asarray([len(live) for live, _ in sampler.ors], dtype=np.int64),
            (n, 1))
        satisfied = np.zeros((n, n_or), dtype=bool)
    else:
        live_left = satisfied = None
    n_slots = sampler._n_slots
    depth = np.zeros((n, n_slots), dtype=np.int64) if n_slots else None
    boost = sampler.boost
    rows = np.arange(n)
    randoms = gen.random((n_free, n))

    # Proposal twist (see OpponentModel._build_tilt). One row per slot plus a
    # trailing all-ones row, so a candidate with no slot indexes a no-op instead
    # of needing a mask in the inner loop.
    om0 = sampler.opponent_model
    tiltT = None
    if om0 is not None and getattr(om0, "tilt", None) is not None and n_slots:
        maxd = om0.TILT_MAX_DEPTH
        tiltT = np.ones((n_slots + 1, maxd), dtype=np.float64)
        for si, row in enumerate(om0.tilt):
            if row is not None:
                tiltT[si, :len(row)] = row

    for i in range(n_free):
        cand, clauses, cps, cols, slots = plan[i]
        m = cand.size
        W = quota[:, cand].copy()
        np.maximum(W, 0.0, out=W)
        if tiltT is not None and slots is not None:
            have = slots >= 0
            dcol = depth[:, np.where(have, slots, 0)]
            np.clip(dcol, 0, tiltT.shape[1] - 1, out=dcol)
            W *= tiltT[np.where(have, slots, n_slots)[None, :], dcol]

        forced = None
        if clauses:
            boosted = np.zeros((n, m), dtype=bool)
            for j, ci in enumerate(clauses):
                unsat = ~satisfied[:, ci]
                k = cols[j]
                if k >= 0:
                    col = unsat & (~boosted[:, k])
                    if col.any():
                        W[col, k] *= boost
                        boosted[col, k] = True
                need = unsat & (live_left[:, ci] == 1)
                if need.any():
                    want = cps[j]
                    if forced is None:
                        forced = np.full(n, -1, dtype=np.int64)
                    conflict = need & (forced >= 0) & (forced != want)
                    if conflict.any():
                        alive &= ~conflict
                    forced[need & (forced < 0)] = want

        total = W.sum(axis=1)
        if forced is None:
            free_mask = alive
        else:
            free_mask = alive & (forced < 0)
        dead = free_mask & (total <= 0.0)
        if dead.any():
            alive &= ~dead
            free_mask &= ~dead

        r = randoms[i] * np.maximum(total, 1e-300)
        csum = np.cumsum(W, axis=1)
        sel = (csum < r[:, None]).sum(axis=1)
        np.clip(sel, 0, m - 1, out=sel)
        bad = free_mask & (W[rows, sel] <= 0.0)
        if bad.any():
            sel[bad] = np.argmax(W[bad], axis=1)
            alive &= ~(free_mask & (W[rows, sel] <= 0.0))
        chosen = W[rows, sel]
        np.add(logq, np.where(free_mask,
                              np.log(np.where(chosen > 0.0, chosen, 1.0)
                                     / np.where(total > 0.0, total, 1.0)),
                              0.0), out=logq)

        if forced is None:
            idx = sel
            pick = cand[sel]
        else:
            fmask = alive & (forced >= 0)
            fpos = np.zeros(n, dtype=np.int64)
            if fmask.any():
                fp = np.searchsorted(cand, np.where(fmask, forced, cand[0]))
                np.clip(fp, 0, m - 1, out=fp)
                alive &= ~(fmask & (cand[fp] != forced))
                alive &= ~(fmask & (quota[rows, np.clip(forced, 0, 5)] <= 0.0))
                fpos = fp
            idx = np.where(forced >= 0, fpos, sel)
            pick = cand[idx]

        upd = alive
        quota[rows[upd], pick[upd]] -= 1.0
        picks[:, i] = np.where(upd, pick, 0)
        if depth is not None and slots is not None:
            sidx = slots[idx]
            take = upd & (sidx >= 0)
            if take.any():
                # each row appears at most once here, so the indices are unique
                # and a direct scatter is safe (and much faster than np.add.at)
                depth[rows[take], sidx[take]] += 1
        if clauses:
            for j, ci in enumerate(clauses):
                live_left[upd, ci] -= 1
                satisfied[:, ci] |= upd & (pick == cps[j])

    alive &= (quota == 0.0).all(axis=1)
    if n_or:
        alive &= satisfied.all(axis=1)

    logl = np.zeros(n, dtype=np.float64)
    om = sampler.opponent_model
    if om is not None and depth is not None:
        if getattr(om, "depth_table", None) is not None:
            # Each slot's contribution is a function of its sampled depth alone
            # -- the per-ask deltas and the base are already inside the table --
            # so the whole likelihood is one gather and a row sum.
            T = np.asarray(om.depth_table, dtype=np.float64)
            d = np.clip(depth, 0, T.shape[1] - 1)
            logl += np.take_along_axis(
                T[None, :, :], d[:, :, None], axis=2)[:, :, 0].sum(axis=1)
        else:
            w = np.asarray(om.weight, dtype=np.float64)
            base = np.asarray(om.base, dtype=np.int64)
            d = depth + base[None, :]
            logl += (np.log(np.where(d > 0, d, 1e-9)) * w[None, :]).sum(axis=1)
    # AFTER the depth term, and with += rather than =. Both of those were wrong:
    # this block came first and the depth branches ASSIGNED, so every decision
    # in which any non-self player had already asked -- which is almost all of
    # them, 632 of 641 in results/ess_probe.json -- silently threw the
    # no-declaration term away before it could be used.
    col_of = None
    if om is not None and om.opp_lambda and getattr(om, "set_cards", None):
        # "If the opponents held this whole half-suit, one of them would very
        # likely have declared it by now." Not a constraint - they may hold it
        # and be unable to place the split - so it enters as a soft weight.
        col_of = {c: j for j, c in enumerate(sampler.order)}
        for cards in om.set_cards:
            idx = [col_of[c] for c in cards if c in col_of]
            if not idx:
                continue
            idxc = np.asarray(idx, dtype=np.int64)
            all_opp = ((picks[:, idxc] & 1) != om.my_team).all(axis=1)
            logl -= om.opp_lambda * all_opp
    # ---- the pre-play naming convention --------------------------------
    # THIS BLOCK MUST LIVE HERE, not in SISSampler._attempt. The scalar
    # sampler is no longer the path any decision takes: sample_batch calls
    # draw_batch, which never materialises the per-draw `deal` dict the
    # scalar likelihood reads. A first version of the convention was wired
    # into _attempt only, and the inertness check duly reported the decoder
    # as bit-identical to the incumbent on every seed -- a dead term reading
    # exactly like a measured null, which is the second time this project
    # has produced that artefact.
    #
    # Unlike the depth term, this one needs the ASSIGNMENT and not the
    # counts: which card the convention names depends on WHICH cards of the
    # half-suit the asker holds, not how many. It is still one gather, since
    # the encoding is a function of the six-bit holding alone.
    conv = getattr(om, "convention", None) if om is not None else None
    if conv and (om.convention_beta or getattr(om, "convention_q", 0.0)):
        from .convention import aimed_position_table, encoded_position_table
        aim = bool(getattr(om, "convention_aim", False))
        table = np.asarray(aimed_position_table() if aim
                           else encoded_position_table(), dtype=np.int64)
        if col_of is None:
            col_of = {c: j for j, c in enumerate(sampler.order)}
        q = getattr(om, "convention_q", 0.0) or 0.0
        book = getattr(om, "convention_book", "depth")
        for (asker, hs, card, const_mask, free_cards,
             g_hs, g_const, g_free, targets) in conv:
            lo = hs * 6
            # Cards publicly known to have been with this asker AT THIS ASK
            # are the same in every world; see the `where` ledger in
            # oppmodel.build for why the snapshot is taken at the ask and not
            # at the end of the log.
            held = np.full(n, (const_mask >> lo) & 0x3F, dtype=np.int64)
            for c in free_cards:
                j = col_of.get(c)
                if j is None:
                    continue          # not free in this decision; contributes
                held |= ((picks[:, j] == asker).astype(np.int64)
                         << (c - lo))
            if book == "locate" and targets:
                # The locating book: j is the index in U of the first target
                # card this world gives the asker. One pass over at most six
                # columns -- the same shape of work as a depth, except that it
                # names cards instead of counting them.
                k = 6 - _POPCOUNT6[held]
                tg = list(targets)[:6]
                j = np.full(n, max(len(tg) - 1, 0), dtype=np.int64)
                found = np.zeros(n, dtype=bool)
                for idx, c in enumerate(tg):
                    w = _holds(picks, col_of, c, asker, const_mask, g_const)
                    j = np.where((~found) & w, idx, j)
                    found |= w
                pos = np.where(k > 0, j % np.maximum(k, 1), 0)
                match = _pos_match(held, pos, card - lo)
            elif aim and g_hs is not None:
                # The payload is the asker's depth in the TARGET half-suit,
                # reconstructed the same way and in the same space.
                glo = g_hs * 6
                gheld = np.full(n, (g_const >> glo) & 0x3F, dtype=np.int64)
                for c in g_free:
                    j = col_of.get(c)
                    if j is None:
                        continue
                    gheld |= ((picks[:, j] == asker).astype(np.int64)
                              << (c - glo))
                match = table[held, _POPCOUNT6[gheld]] == (card - lo)
            else:
                match = table[held] == (card - lo)
            if q > 0.0:
                # The mixture (see fish4/convention.py). k varies BY WORLD,
                # which is the whole point: the flat weight below scores every
                # match alike and so over-credits matches in low-k -- deep --
                # worlds.
                k = 6 - _POPCOUNT6[held]
                pr = np.where(match, q, 0.0) + (1.0 - q) / np.maximum(k, 1)
                logl += np.where(k > 0, np.log(np.maximum(pr, 1e-13)), 0.0)
            else:
                logl += om.convention_beta * match
    return picks, logq, logl, alive


def _holds(picks, col_of, c, asker, const_mask, g_const):
    """Does the asker hold card ``c``, per drawn world?

    Constant when the card was fixed at snapshot time -- publicly placed, or
    pinned by the propagator -- and read off the draw otherwise. The two
    const masks cover the two half-suits an ask can talk about.
    """
    if (const_mask >> c) & 1 or (g_const is not None and (g_const >> c) & 1):
        return np.ones(picks.shape[0], dtype=bool)
    j = col_of.get(c)
    if j is None:
        return np.zeros(picks.shape[0], dtype=bool)
    return picks[:, j] == asker


#: ``_FREE_AT[mask][p]`` = position of the p-th card this holding does NOT
#: have, so a payload becomes a card with one gather, exactly as the depth
#: books do.
_FREE_AT = np.array(
    [[[i for i in range(6) if not (m >> i & 1)][p]
      if p < 6 - bin(m).count("1") else -1
      for p in range(6)] for m in range(64)], dtype=np.int64)


def _pos_match(held, pos, want):
    """Is the ``pos``-th free card of ``held`` the one at index ``want``?"""
    return _FREE_AT[held, np.clip(pos, 0, 5)] == want


#: Population count of every six-bit holding. A table rather than SWAR
#: arithmetic: the first version of this used the wrong magic constants
#: (0x15/0x13 instead of 0x55/0x33) and a 64-entry lookup cannot be wrong in
#: that way. It is also the same shape as the encoding table beside it.
_POPCOUNT6 = np.array([bin(i).count("1") for i in range(64)], dtype=np.int64)
