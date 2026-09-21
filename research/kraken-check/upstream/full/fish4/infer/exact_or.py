"""Backend C: exact marginals INCLUDING the OR constraints, by block factorisation.

THE FACTORISATION
-----------------
Every OR constraint in this game says "the asker held at least one card of
half-suit h at that moment", so every OR involves cards of a *single* half-suit
and the constraint system splits into small blocks that are independent of each
other except through the global quota vector. Block b contributes a count vector
``v in N^6`` (how many of its cards each player was dealt); the only global
condition is that all contributions sum to the residual quota vector ``q``. So:
enumerate each block exactly, aggregate into a histogram over ``v``, then run a
forward-backward DP over the blocks with state = accumulated quota vector.
Marginals come out of the same tables:

    P(card c -> p) = (1/Z) * sum_v  #{block assignments with c->p and count v}
                                    * sum_u F_b[u] B_{b+1}[u+v]

FOUR REFINEMENTS, ALL EXACT, ALL MEASURED
------------------------------------------
1. **Blocks are OR-components, not half-suits.** A block is the union of the
   cards of a maximal set of ORs that share cards. A free card in no OR is not
   part of any block even if it sits in a half-suit that has one: nothing
   couples it beyond the quota vector, so it becomes its own single-card item
   with transitions ``e_p`` for p in its mask. Half-suits with two disjoint ORs
   become two blocks, not one. This is what keeps enumeration small (measured:
   3 blocks per position at the median, 4 cards in the largest block, 6 at the
   worst) and it is why the opening -- 45 free cards, no live OR at all --
   costs nothing extra here.

2. **One quota coordinate is dropped.** After item i the number of cards
   assigned so far is fixed, so the count for one chosen player is determined by
   the other five. Dropping the largest-quota player shrinks every DP array by
   ``q_max + 1``: measured 6x at the median (full product 7560, reduced 1260).
   Within a single item the
   full count vector is recoverable from the reduced one, because the item's
   total is fixed, so nothing is lost.

3. **The DP runs on flat contiguous arrays.** The obvious implementation keeps a
   6-d (or 5-d) array and expresses a transition as a multi-axis slice-add. That
   is a disaster in numpy: a last-axis slice-add on a (10,10,10,10,8) array
   measured **580 us**, versus 32 us for the same number of elements
   contiguously. Instead every state is one flat index, a transition by ``v`` is
   a constant flat offset, and the states from which the transition would
   overflow a quota are removed by multiplying with a precomputed 0/1 mask
   (cached per ``v``). Two contiguous vector ops per transition, no striding.

4. Marginals reuse the same forward/backward arrays; nothing is recomputed.

WHERE IT DIES
-------------
Cost is ``state_space * transitions``. The state space is ``prod_p (q_p + 1)``
divided by the dropped axis: median about 1.3k on real positions but 10k in the
opening, where every player still has 8 or 9 unknown cards. The backend carries
an explicit work budget and refuses (``ok = False``) rather than pretending; the
caller falls back. FRONTIER.md reports the measured cost distribution and the
fraction of real positions inside 10 / 25 / 50 ms.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from fish.beliefs import BeliefState
from fish.cards import NUM_PLAYERS

from fish4.counting import _BINOM

from .base import FreeSystem, InferenceBackend

_BINOM_L = _BINOM.tolist()


class BlockTooBig(Exception):
    """A block enumerated past its leaf cap."""


class ExactOrBackend(InferenceBackend):
    """Knob: ``work_budget`` (state_space * transitions).

    Not a sample count: this backend is exact or it refuses, so its cost
    setting is the size of problem it is willing to accept.
    """

    name = "exact_or"

    def __init__(self, belief: BeliefState, rng: random.Random,
                 work_budget: int = 40_000_000, leaf_cap: int = 200_000,
                 fs: Optional[FreeSystem] = None):
        super().__init__(belief, rng, fs)
        self.work_budget = int(work_budget)
        self.leaf_cap = int(leaf_cap)
        self.ok = False
        self.reason = ""
        self.state_space = 0
        self.full_state_space = 0
        self.n_items = 0
        self.n_transitions = 0
        self.n_blocks = 0
        self.max_block = 0
        self.work = 0
        self.Z = 0.0
        self._items: list[dict] = []
        self._F: list[np.ndarray] = []
        self._B: list[np.ndarray] = []
        self._masks: dict[tuple, np.ndarray] = {}
        self._marg: Optional[np.ndarray] = None
        self._plan()

    def cost_model(self) -> dict:
        return {"knob": "work_budget", "value": self.work_budget,
                "unit": "state_space * transitions the DP may touch"}

    # -- planning -------------------------------------------------------------

    def _plan(self) -> None:
        fs = self.fs
        if not fs.feasible:
            self.reason = "infeasible quotas"
            return
        self.drop = max(range(NUM_PLAYERS), key=lambda p: fs.quotas[p])
        self.keep = [p for p in range(NUM_PLAYERS) if p != self.drop]
        self.shape = tuple(fs.quotas[p] + 1 for p in self.keep)
        self.naxes = len(self.shape)
        full = 1
        for q in fs.quotas:
            full *= q + 1
        self.full_state_space = full
        ss = 1
        for x in self.shape:
            ss *= x
        self.state_space = ss
        strides = [1] * self.naxes
        for a in range(self.naxes - 2, -1, -1):
            strides[a] = strides[a + 1] * self.shape[a + 1]
        self.strides = strides
        if fs.n_free == 0:
            self.ok = True
            self.Z = 1.0
            return

        # -- blocks: connected components of ORs that share cards
        n_or = len(fs.ors)
        parent = list(range(n_or))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        owner_of_card: dict[int, int] = {}
        for oi, (live, _p) in enumerate(fs.ors):
            for i in live:
                other = owner_of_card.get(i)
                if other is None:
                    owner_of_card[i] = oi
                else:
                    ra, rb = find(oi), find(other)
                    if ra != rb:
                        parent[rb] = ra
        comps: dict[int, list[int]] = {}
        for oi in range(n_or):
            comps.setdefault(find(oi), []).append(oi)

        blocked: set[int] = set()
        blocks: list[tuple[list[int], list]] = []
        for root, ois in sorted(comps.items()):
            cards: set[int] = set()
            for oi in ois:
                cards.update(fs.ors[oi][0])
            blocked |= cards
            blocks.append((sorted(cards), [fs.ors[oi] for oi in ois]))

        items: list[dict] = []
        try:
            for cards, ors_b in blocks:
                items.append(self._block_item(cards, ors_b))
        except BlockTooBig as e:
            self.reason = str(e)
            return
        self.n_blocks = len(blocks)
        self.max_block = max((len(c) for c, _ in blocks), default=0)
        for i in range(fs.n_free):
            if i not in blocked:
                items.append(self._card_item(i))
        self._items = items
        self.n_items = len(items)
        self.n_transitions = sum(len(it["vs"]) for it in items)
        self.work = ss * self.n_transitions
        if self.work > self.work_budget:
            self.reason = (f"work {self.work} > budget {self.work_budget} "
                           f"(states {ss}, transitions {self.n_transitions})")
            return
        self.ok = True

    # -- items ----------------------------------------------------------------

    def _finish_item(self, cards, vs, w, contrib, leaves) -> dict:
        offs = []
        for v in vs:
            off = 0
            for a in range(self.naxes):
                off += v[a] * self.strides[a]
            offs.append(off)
        return {"cards": cards, "vs": vs, "offs": offs, "w": w,
                "contrib": contrib, "leaves": leaves}

    def _card_item(self, i: int) -> dict:
        keep = self.keep
        vs, w = [], []
        contrib: dict[tuple[int, int], dict[int, float]] = {}
        leaves: dict[int, list[tuple[int, ...]]] = {}
        for p in self.fs.players[i]:
            if self.fs.quotas[p] == 0:
                continue
            v = [0] * NUM_PLAYERS
            v[p] = 1
            j = len(vs)
            vs.append(tuple(v[q] for q in keep))
            w.append(1.0)
            contrib[(i, p)] = {j: 1.0}
            leaves[j] = p
        item = self._finish_item([i], vs, w, contrib, leaves)
        item["classes"] = None
        return item

    def _block_item(self, cards: list[int], ors_b: list) -> dict:
        """Enumerate one OR-component exactly, over exchangeability classes.

        Enumerating card-by-card is the obvious thing and it is wasteful. Two
        cards of a block with the same candidate mask AND the same set of ORs
        containing them are exchangeable: nothing in the constraint system can
        tell them apart. So enumerate the per-CLASS count vectors instead of the
        per-CARD assignments, weighting each by the multinomial that counts the
        card-level assignments it stands for. The ORs are still exact, because
        an OR's card set is exactly a union of classes, so "at least one of
        those cards went to p" is "the classes in that OR gave p at least one".

        Measured on the worst real block seen (5 cards, 5 candidates each, one
        OR): 3906 recursion nodes and 19.0 ms card-by-card, against 251 nodes
        and 1.4 ms by class.

        Per-card marginals survive the collapse: within a class every card is
        equally likely to be the one that went to p, so a class configuration
        contributes ``weight * v_class[p] / n_class`` to each of its cards.
        """
        fs = self.fs
        keep = self.keep
        quotas = fs.quotas
        n_or = len(ors_b)
        or_sets = [set(live) for live, _p in ors_b]
        or_players = [p for _live, p in ors_b]
        sig: dict[tuple, list[int]] = {}
        for i in cards:
            key = (fs.cand[i],
                   frozenset(o for o in range(n_or) if i in or_sets[o]))
            sig.setdefault(key, []).append(i)
        cls_cards = [cs for cs in sig.values()]
        cls_players = [fs.players[cs[0]] for cs in cls_cards]
        cls_n = [len(cs) for cs in cls_cards]
        C = len(cls_cards)
        or_classes = [[c for c in range(C)
                       if cls_cards[c][0] in or_sets[o]] for o in range(n_or)]

        cnt = [0] * NUM_PLAYERS
        chosen = [[0] * NUM_PLAYERS for _ in range(C)]
        table: dict[tuple, list] = {}
        raw_leaves: dict[tuple, list] = {}
        row_len = 1 + C * NUM_PLAYERS
        cap_leaves = self.leaf_cap
        n_leaf = 0

        def leaf(weight: float) -> None:
            nonlocal n_leaf
            for o in range(n_or):
                q = or_players[o]
                for cc in or_classes[o]:
                    if chosen[cc][q]:
                        break
                else:
                    return
            n_leaf += 1
            if n_leaf > cap_leaves:
                raise BlockTooBig(
                    f"block of {len(cards)} cards exceeded {cap_leaves} leaves")
            v = tuple(cnt[q] for q in keep)
            row = table.get(v)
            if row is None:
                row = table[v] = [0.0] * row_len
                raw_leaves[v] = []
            row[0] += weight
            base = 1
            for cc in range(C):
                ch = chosen[cc]
                for q in range(NUM_PLAYERS):
                    if ch[q]:
                        row[base + q] += weight * ch[q]
                base += NUM_PLAYERS
            raw_leaves[v].append((weight, tuple(tuple(x) for x in chosen)))

        def rec(c: int, k: int, rem: int, weight: float) -> None:
            if c == C:
                leaf(weight)
                return
            pl = cls_players[c]
            p = pl[k]
            room = quotas[p] - cnt[p]
            vec = chosen[c]
            if k == len(pl) - 1:
                if rem > room:
                    return
                vec[p] = rem
                cnt[p] += rem
                rec(c + 1, 0, cls_n[c + 1] if c + 1 < C else 0, weight)
                cnt[p] -= rem
                vec[p] = 0
                return
            hi = room if room < rem else rem
            brow = _BINOM_L[rem]
            for x in range(hi + 1):
                vec[p] = x
                cnt[p] += x
                rec(c, k + 1, rem - x, weight * brow[x])
                cnt[p] -= x
            vec[p] = 0

        if C:
            rec(0, 0, cls_n[0], 1.0)

        vs = sorted(table)
        w = [table[v][0] for v in vs]
        contrib: dict[tuple[int, int], dict[int, float]] = {}
        for j, v in enumerate(vs):
            row = table[v]
            base = 1
            for cc in range(C):
                inv = 1.0 / cls_n[cc]
                for q in range(NUM_PLAYERS):
                    c_val = row[base + q]
                    if c_val:
                        share = c_val * inv
                        for card in cls_cards[cc]:
                            contrib.setdefault((card, q), {})[j] = share
                base += NUM_PLAYERS
        leaves = {}
        for j, v in enumerate(vs):
            entries = raw_leaves[v]
            cum = []
            tot = 0.0
            for wt, _cc in entries:
                tot += wt
                cum.append(tot)
            leaves[j] = (cum, [cc for _wt, cc in entries])
        flat_cards = [i for cs in cls_cards for i in cs]
        item = self._finish_item(flat_cards, vs, w, contrib, leaves)
        item["classes"] = cls_cards
        return item

    # -- the DP ---------------------------------------------------------------

    def _mask(self, v) -> np.ndarray:
        """0/1 over flat states: 1 where adding ``v`` keeps every quota legal."""
        m = self._masks.get(v)
        if m is None:
            box = np.zeros(self.shape, dtype=np.float64)
            box[tuple(slice(0, self.shape[a] - v[a])
                      for a in range(self.naxes))] = 1.0
            m = self._masks[v] = box.ravel()
        return m

    def _run(self) -> None:
        if self._F:
            return
        S = self.state_space
        N = len(self._items)
        tmp = np.empty(S, dtype=np.float64)
        F = [np.zeros(S) for _ in range(N + 1)]
        F[0][0] = 1.0
        for i, it in enumerate(self._items):
            src, dst = F[i], F[i + 1]
            for j, off in enumerate(it["offs"]):
                w = it["w"][j]
                if off == 0:
                    if w == 1.0:
                        dst += src
                    else:
                        dst += src * w
                    continue
                m = self._mask(it["vs"][j])
                lim = S - off
                np.multiply(src[:lim], m[:lim], out=tmp[:lim])
                if w != 1.0:
                    tmp[:lim] *= w
                dst[off:] += tmp[:lim]
        B = [np.zeros(S) for _ in range(N + 1)]
        B[N][S - 1] = 1.0
        for i in range(N - 1, -1, -1):
            it = self._items[i]
            src, dst = B[i + 1], B[i]
            for j, off in enumerate(it["offs"]):
                w = it["w"][j]
                if off == 0:
                    if w == 1.0:
                        dst += src
                    else:
                        dst += src * w
                    continue
                m = self._mask(it["vs"][j])
                lim = S - off
                np.multiply(src[off:], m[:lim], out=tmp[:lim])
                if w != 1.0:
                    tmp[:lim] *= w
                dst[:lim] += tmp[:lim]
        self._F, self._B = F, B
        self.Z = float(F[N][S - 1])

    def free_marginals(self) -> np.ndarray:
        if self._marg is not None:
            return self._marg
        fs = self.fs
        if not self.ok:
            raise RuntimeError(f"exact_or refused this position: {self.reason}")
        M = np.zeros((fs.n_free, NUM_PLAYERS))
        if fs.n_free == 0:
            self._marg = M
            return M
        self._run()
        Z = self.Z
        if not (Z > 0):
            raise RuntimeError("exact_or: no consistent world (Z = 0)")
        S = self.state_space
        tmp = np.empty(S, dtype=np.float64)
        for i, it in enumerate(self._items):
            F, B = self._F[i], self._B[i + 1]
            D = []
            for j, off in enumerate(it["offs"]):
                if off == 0:
                    D.append(float(np.dot(F, B)))
                    continue
                m = self._mask(it["vs"][j])
                lim = S - off
                np.multiply(F[:lim], m[:lim], out=tmp[:lim])
                D.append(float(np.dot(tmp[:lim], B[off:])))
            for (card, p), d in it["contrib"].items():
                M[card, p] = sum(c * D[j] for j, c in d.items()) / Z
        self._marg = M
        return M

    # -- exact uniform world sampling from the same tables --------------------

    def worlds(self, n: int) -> list[list[int]]:
        fs = self.fs
        if not self.ok:
            raise RuntimeError(f"exact_or refused this position: {self.reason}")
        if fs.n_free == 0:
            return [list(fs.base_hands) for _ in range(n)]
        self._run()
        if not (self.Z > 0):
            return []
        out = []
        own = [0] * fs.n_free
        rnd = self.rng.random
        S = self.state_space
        for _ in range(n):
            u = 0
            for i, it in enumerate(self._items):
                Bn = self._B[i + 1]
                offs = it["offs"]
                w = it["w"]
                tot = 0.0
                cum = []
                for j, off in enumerate(offs):
                    idx = u + off
                    val = 0.0
                    if idx < S and self._mask(it["vs"][j])[u]:
                        val = w[j] * float(Bn[idx])
                    tot += val
                    cum.append(tot)
                r = rnd() * tot
                pick = len(cum) - 1
                for j, c in enumerate(cum):
                    if r < c:
                        pick = j
                        break
                lv = it["leaves"][pick]
                if it["classes"] is None:
                    own[it["cards"][0]] = lv
                else:
                    cum2, configs = lv
                    if len(configs) > 1:
                        r2 = rnd() * cum2[-1]
                        ci = len(configs) - 1
                        for t, c2 in enumerate(cum2):
                            if r2 < c2:
                                ci = t
                                break
                    else:
                        ci = 0
                    conf = configs[ci]
                    for cc, cards_c in enumerate(it["classes"]):
                        pool = []
                        counts = conf[cc]
                        for q in range(NUM_PLAYERS):
                            if counts[q]:
                                pool.extend([q] * counts[q])
                        if len(pool) > 1:
                            self.rng.shuffle(pool)
                        for t, card in enumerate(cards_c):
                            own[card] = pool[t]
                u += offs[pick]
            out.append(fs.materialize(own))
        return out
