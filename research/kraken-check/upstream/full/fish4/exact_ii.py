"""Exact best response at m = 1, under imperfect information.

Everything this project calls exact is a statement about the
PERFECT-information game. ``scripts4/closed_form_proof.py`` solves that game at
every m, and the answer is degenerate: the team on move takes every half-suit it
can touch. The v0.3 benchmark's headline -- belief agents choosing a provably
optimal move in 100% of information-resolved positions -- was scored against
that same perfect-information optimum, and every one of those positions had a
single live half-suit.

So the one thing nobody has measured is whether the champion plays the m = 1
endgame well when it does NOT know where the cards are. That is a different
game with a different value, and this solves it exactly for one deviating seat.

WHAT IS EXACT AND WHAT IS ASSUMED
---------------------------------
Exact: the enumeration of deals, the deviator's optimisation over its own
information sets, and the backward induction. No sampling anywhere.

Assumed, and both stated rather than buried:

* **The prior is uniform over consistent deals.** A deal is consistent when
  every unseen card sits with a player the deviator's belief still allows and
  the public hand counts are met. Uniform is the maximum-entropy choice given
  those constraints and is what the constraint system alone licenses; the
  champion's own posterior tilts this by an opponent model, and using that
  instead would make the answer a statement about the model rather than about
  the position.
* **The opponents are a deterministic realisation of the champion.** The
  champion samples, so "the champion" is a distribution over policies. Seeding
  it from a hash of the observation makes it a pure strategy -- a well-defined
  object to best-respond to -- and one that a fresh agent reproduces exactly,
  checked: over 629 real decisions a freshly constructed agent given the same
  observation computes a bit-identical belief to one attached from the deal.

THE CONTROL THAT MAKES IT CHECKABLE
-----------------------------------
When the deviator's belief pins every card the support collapses to one deal,
imperfect information is gone, and the exact value MUST equal the closed form
``2f - m``. 43% of real m = 1 decisions are of that kind, so the solver is
checked against an independently proved answer on nearly half its input. A
solver that cannot reproduce ``+/-1`` there is wrong, whatever it says
elsewhere.
"""

from __future__ import annotations

import hashlib
import time
from itertools import product
from typing import Optional

from fish.cards import (CARDS_PER_HALF_SUIT, NUM_PLAYERS, half_suit_cards,
                        team_of)
from fish.engine import Ask, Claim, GameState, NULL_TEAM, Pass
from fish.observation import Observation
from fish.rules import RuleConfig

#: Depth cap. A cycle of mutual misses can run forever; the harness scores an
#: unresolved half-suit for nobody (fish4/match.play_capped), so this does too.
MAX_PLIES = 24

#: Search nodes a single position may take before the solver gives up on it.
#:
#: THE BUDGET IS NODES, NOT SECONDS, because a wall-clock budget measures the
#: machine. Three of these studies were running at once on four cores when the
#: corrected solver first met a real workload, and 21% of m = 1 positions
#: exceeded 60s at about 190,000 nodes each -- a coverage figure that would not
#: reproduce on an idle machine, or on anyone else's. A node cap gives every
#: position the same budget in every run, and the seconds below are kept only
#: as a backstop against a pathological case wedging a study for hours.
MAX_NODES = 300_000

#: Wall-clock backstop, in seconds. Secondary to MAX_NODES above: a run should
#: end on the node cap, and reach this only if a single node somehow becomes
#: pathologically slow.
#:
#: This exists because "solvable" and "solvable in the time available" were
#: conflated once already. A probe capped at support 8 solved every m = 2
#: position in under a second, which read as "m = 2 is reachable"; the study
#: allowed support up to 24 and ground on ONE position for six and a half
#: hours, writing nothing. An exact solver with no deadline does not fail
#: loudly, it fails silently and forever, and a layer that is out of reach
#: should say so rather than hang.
DEFAULT_DEADLINE = 900.0


class SolveTimeout(Exception):
    """The position exceeded its budget. It is unsolved, not zero."""


def consistent_deals_multi(obs: Observation, bel, live,
                           limit: Optional[int] = None) -> list:
    """The same enumeration over SEVERAL live half-suits at once.

    m = 1 is the only layer where one half-suit's cards are all the live cards,
    so the single-half-suit version below silently assumes it. Above m = 1 the
    hand counts cover every live card and the enumeration has to as well.

    WHY THIS IS A BACKTRACKING WALK AND NOT A PRODUCT. The first version built
    the full cartesian product of allowed holders and filtered it on the hand
    counts afterwards, and REFUSED outright when the product exceeded two
    million -- returning an empty list, which the caller could not tell apart
    from "the belief admits no deal at all", so the position vanished from the
    study without being counted anywhere. On twelve synthetic m = 2 positions
    with a wide belief it refused nine. Descending with the counts as a
    constraint visits only prefixes that can still be completed: same deals, in
    the same ORDER (checked, because the order decides which of several tied
    actions gets reported as optimal), about sixty times faster where the
    product version ran at all.

    ``limit`` stops the walk once more than that many deals are found. A caller
    that only needs to know whether the support exceeds its cap pays for the
    cap, not for the belief's full width.
    """
    me = obs.player
    cards = [c for h in live for c in half_suit_cards(h)]
    mine = [c for c in cards if (obs.hand >> c) & 1]
    unseen = [c for c in cards if not (obs.hand >> c) & 1]
    opts = []
    for c in unseen:
        mask = bel.current_holder_mask(c)
        allowed = [q for q in range(NUM_PLAYERS)
                   if q != me and (mask >> q) & 1]
        if not allowed:
            return []
        opts.append(allowed)
    need = [obs.hand_counts[q] if q != me else 0 for q in range(NUM_PLAYERS)]
    base = 0
    for c in mine:
        base |= 1 << c
    n = len(unseen)
    out: list = []
    assign: list = []

    def rec(i: int, left: int) -> None:
        # The count test comes FIRST, so that i == n is reached only with every
        # player's count exactly met -- including the n == 0 case, where the
        # product version still had to check that nobody else was owed a card.
        if sum(need) != left:
            return
        if i == n:
            hands = [0] * NUM_PLAYERS
            hands[me] = base
            for c, q in zip(unseen, assign):
                hands[q] |= 1 << c
            out.append(tuple(hands))
            return
        for q in opts[i]:
            if need[q] > 0:
                need[q] -= 1
                assign.append(q)
                rec(i + 1, left - 1)
                assign.pop()
                need[q] += 1
                if limit is not None and len(out) > limit:
                    return

    rec(0, n)
    return out


def consistent_deals(obs: Observation, bel, hs: int) -> list:
    """Every assignment of the unseen cards of ``hs`` the public record allows.

    Constrained by the belief's per-card holder masks and by the public hand
    counts, both of which every seat can see.
    """
    me = obs.player
    cards = list(half_suit_cards(hs))
    mine = [c for c in cards if (obs.hand >> c) & 1]
    unseen = [c for c in cards if not (obs.hand >> c) & 1]
    opts = []
    for c in unseen:
        mask = bel.current_holder_mask(c)
        allowed = [q for q in range(NUM_PLAYERS)
                   if q != me and (mask >> q) & 1]
        if not allowed:
            return []
        opts.append(allowed)
    counts = list(obs.hand_counts)
    out = []
    for combo in (product(*opts) if opts else [()]):
        need = [0] * NUM_PLAYERS
        for q in combo:
            need[q] += 1
        if any(need[q] != counts[q] for q in range(NUM_PLAYERS) if q != me):
            continue
        hands = [0] * NUM_PLAYERS
        for c in mine:
            hands[me] |= 1 << c
        for c, q in zip(unseen, combo):
            hands[q] |= 1 << c
        out.append(tuple(hands))
    return out


#: Memo for the champion oracle, ONE BUCKET PER SPEC. It is a pure function of
#: the information set AND of the policy being asked -- that is the whole point
#: of seeding it from the observation hash -- so caching it changes no value,
#: only the time to get one. Without it the solver rebuilt an agent and replayed
#: a hundred-event history at every node of every branch, and twenty games took
#: over two hours to reach four.
#:
#: THE SPEC WAS MISSING FROM THE KEY. Every script here solves with a single
#: spec per process, so nothing published was affected; the first run that used
#: two -- solving the same positions against the champion and against a
#: corrected policy, to ask whether the correction reduces exploitability --
#: got byte-identical answers for both on all 290 positions, because the second
#: policy was reading the first one's cached moves. A cache key that omits
#: something the value depends on is the same fault as the memo key that
#: omitted the history, found the same way: by a control that could not have
#: passed if the code were right.
#:
#: Bucketing rather than folding the spec into the digest is deliberate. The
#: first eight bytes of that digest seed the champion's own RNG, so extending
#: it would change what the champion PLAYS and reprice every number measured
#: against it. The info key stays byte-identical; only which dictionary it is
#: looked up in changes.
_CHAMP_CACHE: dict = {}


def _info_key(seat, obs, hist=None) -> bytes:
    """Hash the whole information set.

    ``hist`` is ``tuple(repr(e) for e in obs.history)``, which a caller may
    pass in when it already has it. Every deal in an information set shares the
    public history by definition -- that is what makes it an information set --
    so at an opponent node this tuple is the same for all of them, and building
    it per deal meant re-running a hundred generated dataclass ``__repr__``
    calls to get the same strings back. The profile put that at a sixth of the
    solver's runtime.

    Passing it in does NOT change the key: the same tuple goes into the same
    ``repr`` and the same digest comes out, byte for byte. That matters more
    than the speed does, because the first eight bytes of this digest seed the
    champion's own RNG -- it samples its posterior and breaks ties between
    equally scored asks with it. A key that changed here would change what the
    champion PLAYS, and every measured number with it. So the shared work moves
    and the key does not.
    """
    if hist is None:
        hist = tuple(repr(e) for e in obs.history)
    return hashlib.sha256(repr((seat, obs.hand, tuple(obs.hand_counts),
                                tuple(obs.set_winner), hist))
                          .encode()).digest()


def _champion_action(spec, rules, seat, st, hist=None):
    """The champion's move, as a pure function of its information set.

    ``hist`` is the pre-built history reprs shared by every deal at a node; see
    ``_info_key``. Omitted by the rollout, whose history changes every ply.
    """
    from fish4.registry4 import make_agent
    obs = Observation.from_state(st, seat)
    key = _info_key(seat, obs, hist)
    bucket = _CHAMP_CACHE.get(repr(spec))
    if bucket is None:
        bucket = _CHAMP_CACHE[repr(spec)] = {}
    hit = bucket.get(key)
    if hit is not None:
        return hit[0]
    a = make_agent(spec)
    a.begin_game(seat, rules, int.from_bytes(key[:8], "big"))
    try:
        act = a.act(obs)
    except Exception:
        act = None
    bucket[key] = (act,)
    return act


def _clone(st: GameState) -> GameState:
    """A shallow copy of a state, which is all the search ever needs.

    ``copy.deepcopy`` was 60%+ of the solver's time: it walked the rule config
    and every event in the history at every node. Events are frozen dataclasses
    and the rules are never mutated, so only the three mutable containers have
    to be new. Checked against deepcopy for identical values on every position
    of a full game before it was adopted -- a faster search that returns a
    different number is not a faster search.
    """
    t = GameState.__new__(GameState)
    t.rules = st.rules
    t.hands = list(st.hands)
    t.turn = st.turn
    t._num_hs = st._num_hs
    t._deck_size = st._deck_size
    t.set_winner = list(st.set_winner)
    t.history = list(st.history)
    t.debug = st.debug
    t.agent_seed = st.agent_seed
    return t


class ExactII:
    """Best response for one seat at m = 1, against champion opponents."""

    def __init__(self, rules: RuleConfig, hs, deviator: int, spec):
        self.rules = rules
        #: one half-suit or several. Kept as a tuple internally so the m = 1
        #: path and the general path are the same code; ``self.hs`` stays an
        #: int there so nothing that reads it has to change.
        self.live = (hs,) if isinstance(hs, int) else tuple(hs)
        self.hs = self.live[0]
        self.me = deviator
        self.spec = spec
        self.nodes = 0
        self._memo: dict = {}
        #: Transposition memo, OFF by default. Measured on a support-18 m = 2
        #: position: 19 hits in 200,001 lookups, one in ten thousand. That is
        #: not a near miss, it is the structure of the problem -- once the fix
        #: for the history bug put the path since the root into the key, two
        #: nodes can only share a key by being reached through the same public
        #: events, and the search almost never does that twice. What remained
        #: was a dict growing one large entry per node, 199,982 of them here
        #: and millions on the positions this solver most needs to reach.
        #: Switchable, not deleted: tests4/test_exact_ii.py runs a position
        #: both ways and requires the same value, which is the only thing that
        #: makes turning it off a speedup rather than a change of answer.
        self.use_memo = False
        #: the optimal action AT THE ROOT, once solve() has run. The value
        #: alone says the champion leaves something on the table; this says
        #: what it should have done instead, which is what a policy change has
        #: to be built from.
        self.best_action = None
        self.action_values: dict = {}
        self.deadline = None          # wall-clock backstop; None = no limit
        #: the reproducible budget. None = no limit, which is what an
        #: unbounded exact search should never be given.
        self.max_nodes = None
        #: Exact cutoff, on by default. Switchable because the only honest way
        #: to keep a cutoff honest is to be able to run without it and compare;
        #: tests4/test_exact_ii.py does exactly that.
        self.prune = True
        #: States the opponent walk dropped, and actions the deviator skipped.
        #: Both SHOULD be zero and both are argued to be: the champion's action
        #: is computed from its own view of a state and applied to that same
        #: state, and a wrong claim is legal rather than an error. Arguments of
        #: that shape are what this file spent a day disproving, so they are
        #: counted and reported instead.
        self.opp_dropped = 0
        self.dev_skipped = 0
        #: reprs of the history at the ROOT, filled on the first solve. A node's
        #: history is the root's plus one event per ply since, and ``path``
        #: already holds the repr of each of those events, so the champion's
        #: cache key can be assembled by concatenation instead of by walking a
        #: hundred events again at every node. Guarded by a length check rather
        #: than by trusting that identity: if the two ever disagree the tuple is
        #: rebuilt from the state.
        self._root_hist = None

    # -- terminal value ------------------------------------------------------

    def _value(self, st: GameState) -> Optional[float]:
        total = 0.0
        for h in self.live:
            w = st.set_winner[h]
            if w is None:
                return None
            if w == NULL_TEAM:
                continue
            total += 1.0 if w == team_of(self.me) else -1.0
        return total

    # -- what the champion itself gets ---------------------------------------

    def champion_value(self, states, weights, max_plies=MAX_PLIES) -> float:
        """The same expectation with the CHAMPION in the deviator's seat.

        The best response is only interesting beside this. Their difference is
        the exact gain from deviating at m = 1 -- exploitability restricted to
        the endgame, computed rather than sampled, which is what
        scripts4/exploitability.py could not do.
        """
        tot = 0.0
        for st, w in zip(states, weights):
            t = _clone(st)
            # The rollout's history grows by one event a ply, so the key's
            # expensive part can be EXTENDED instead of rebuilt: without this
            # a forty-ply rollout re-reprs the same hundred-event prefix forty
            # times. The length check is the guard -- if a move ever appends
            # something other than exactly one event the tuple is rebuilt from
            # the state, so the key stays a function of the real history and
            # not of an assumption about the engine.
            hist = tuple(repr(e) for e in t.history)
            for _ in range(max_plies):
                v = self._value(t)
                if v is not None:
                    break
                a = _champion_action(self.spec, self.rules, t.turn, t, hist)
                if a is None:
                    break
                try:
                    t.apply(t.turn, a)
                    hist = (hist + (repr(t.history[-1]),)
                            if len(t.history) == len(hist) + 1
                            else tuple(repr(e) for e in t.history))
                except Exception:
                    break
            v = self._value(t)
            tot += w * (0.0 if v is None else v)
        return tot

    def champion_tree_value(self, states, weights) -> float:
        """The champion's value computed BY THE RECURSION instead of a playout.

        ``champion_value`` rolls each deal forward independently; this walks the
        same tree the best response walks, but plays the champion's move at the
        deviator's nodes rather than maximising. They evaluate the same strategy
        by two different code paths, so they must return the same number, and
        that number must not exceed the best response -- the optimiser may copy
        the champion.

        This is the check that catches a broken tree. A memo key that omitted
        the history made the maximisation return less than one of its own
        options for five m = 2 positions; the negative gain was the symptom,
        and this is the test that localises it. Neither the pinned control nor
        the closed form can see it, because both agree wherever the support
        collapses to one deal and the fault needs several.
        """
        # A SEPARATE MEMO, and this is not a detail. A node's value depends on
        # the deviator's policy below it: solve() maximises there, this copies.
        # Same key, different value. Sharing the instance's memo would have had
        # this control read back the numbers of the search it exists to check
        # -- a memo bug inside the control written to catch a memo bug. The
        # node counter is saved and restored for the same reason: a search that
        # has already spent 250,000 nodes would push this one straight over the
        # budget and the control would report a timeout that is not one.
        save_dev, save_memo, save_nodes = (self._deviator, self._memo,
                                           self.nodes)
        try:
            self._deviator = self._deviator_copies_champion
            self._memo = {}
            self.nodes = 0
            return self.solve(states, weights)
        finally:
            self._deviator = save_dev
            self._memo = save_memo
            self.nodes = save_nodes

    def _deviator_copies_champion(self, states, weights, depth, path=()):
        a = _champion_action(self.spec, self.rules, self.me, states[0])
        if a is None:
            return 0.0
        buckets = {}
        for s, w in zip(states, weights):
            t = _clone(s)
            try:
                ev = t.apply(self.me, a)
            except Exception:
                return 0.0
            sig = repr(ev)
            buckets.setdefault(sig, ([], []))
            buckets[sig][0].append(t)
            buckets[sig][1].append(w)
        v = 0.0
        for sig, (ss, ws) in buckets.items():
            tot = sum(ws)
            v += tot * self.solve(ss, [x / tot for x in ws],
                                  depth + 1, path + (sig,))
        return v

    def _upper(self, states, weights) -> float:
        """The most this node can still pay the deviator.

        Every live half-suit is worth at most +1, and one already decided
        against us is worth exactly what it is worth. If an action ATTAINS this
        bound there is nothing better and the remaining actions can be skipped.

        This is not alpha-beta. No window is threaded and no bound is ever
        stored, so a memoised value is still an exact value -- which matters,
        because a pruned underestimate written into the memo is
        indistinguishable from a real one afterwards.
        """
        mine = team_of(self.me)
        tot = 0.0
        for s, w in zip(states, weights):
            best = 0.0
            for h in self.live:
                win = s.set_winner[h]
                if win is None or win == mine:
                    best += 1.0
                elif win != NULL_TEAM:
                    best -= 1.0
            tot += w * best
        return tot

    # -- the search ----------------------------------------------------------

    def solve(self, states: list, weights: list, depth: int = 0,
              path: tuple = ()) -> float:
        """Expected value to the deviator's team over a weighted belief set.

        ``states`` all share the deviator's information: same public history,
        same own hand. Its action must therefore be the same in all of them,
        which is what makes this a best response over INFORMATION SETS rather
        than a per-deal cheat.
        """
        if depth == 0:
            self._root_hist = tuple(repr(e) for e in states[0].history)
        self.nodes += 1
        if self.max_nodes is not None and self.nodes > self.max_nodes:
            raise SolveTimeout(f"exceeded {self.max_nodes} nodes")
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise SolveTimeout(f"exceeded the wall-clock backstop after "
                               f"{self.nodes} nodes")
        # Memo on the node itself: two branches reaching the same weighted
        # belief set at the same depth have the same value by construction.
        #
        # THE HISTORY IS PART OF THE NODE. Leaving it out cost five impossible
        # negative gains at m = 2 -- positions where the "best response" scored
        # BELOW the champion it may freely copy. The opponents here are the
        # champion, whose action is a function of its whole observation, so two
        # nodes with identical hands, turn, winners and weights but different
        # histories have different continuations and different values. Merging
        # them returned one branch's value for the other, and because the
        # maximisation reads those values, the max came out below one of its
        # own options. Every solved position at m = 1 and m = 2 was recomputed
        # after this line changed; see results/ii_endgame*.json.
        # ``path`` is the events since the root, and every state in this node
        # shares them. The root's own history is common to the whole search, so
        # the path identifies the history without rebuilding it -- keying on
        # ``states[0].history`` directly was equally correct and unusably slow,
        # since it re-reprs a hundred events at every node.
        key = None
        if self.use_memo:
            key = (depth, path, tuple(sorted(
                (tuple(s.hands), s.turn, tuple(s.set_winner), round(w, 12))
                for s, w in zip(states, weights))))
            hit = self._memo.get(key)
            if hit is not None:
                return hit
        done = [self._value(s) for s in states]
        if all(v is not None for v in done):
            r = sum(w * v for w, v in zip(weights, done))
            if key is not None:
                self._memo[key] = r
            return r
        if depth >= MAX_PLIES:
            if key is not None:
                self._memo[key] = 0.0
            return 0.0              # unresolved scores for nobody, as the
                                    # harness does

        turn = states[0].turn
        if any(s.turn != turn for s in states):
            # The deviator can see whose turn it is, so this cannot happen.
            raise AssertionError("information set spans different movers")

        r = (self._deviator(states, weights, depth, path) if turn == self.me
             else self._opponent(states, weights, depth, turn, path))
        if key is not None:
            self._memo[key] = r
        return r

    def _legal(self, st: GameState):
        p = st.turn
        if st.hands[p] == 0:
            return list(st.legal_passes(p))
        acts = list(st.legal_asks(p))
        team = team_of(p)
        for h in self.live:
            if st.set_winner[h] is not None:
                continue
            base = h * CARDS_PER_HALF_SUIT
            holders = [st.holder_of(base + i)
                       for i in range(CARDS_PER_HALF_SUIT)]
            if all(x is not None and team_of(x) == team for x in holders):
                acts.append(Claim(h, tuple(holders)))
        return acts

    def _deviator(self, states, weights, depth, path=()):
        # Legality is information-set measurable: an ask needs a card of the
        # half-suit in hand (own), a target holding cards (public counts), and
        # a card not held (own). So the action set is the same in every state.
        acts = self._legal(states[0])
        # Claims are the exception: which assignment is TRUE differs by state,
        # so the deviator may only offer assignments over its own team, and is
        # scored on whether each happens to be right.
        acts = [a for a in acts if not isinstance(a, Claim)]
        claims = self._claim_candidates(states)
        root = depth == 0
        # Claims first BELOW the root, because a claim is the action most
        # likely to attain the bound and end the loop early. NOT at the root:
        # every action is evaluated there anyway, and best_action is the first
        # maximiser in list order, so reordering would change which of several
        # tied-optimal moves gets reported -- and ii_action_diff.py counts a
        # disagreement whenever the champion's move is not the one reported.
        # That would be a change to the headline dressed up as a speedup.
        acts = (acts + claims) if root else (claims + acts)
        if not acts:
            return 0.0
        best = None
        # The root is never pruned: ii_action_diff.py reads action_values for
        # every action, the champion's included, and an unpriced champion move
        # is silently reclassified rather than reported.
        ub = None if (root or not self.prune) else self._upper(states, weights)
        for a in acts:
            buckets = {}
            illegal = False
            for s, w in zip(states, weights):
                t = _clone(s)
                try:
                    ev = t.apply(self.me, a)
                except Exception:
                    self.dev_skipped += 1
                    # Skip the action, do not abandon the loop. Returning the
                    # best found SO FAR here made the answer depend on the
                    # order actions happen to be generated in, and could have
                    # returned -1.0 for a position with a fine move later in
                    # the list. It never fired on the positions I instrumented,
                    # which is the only reason it did no damage.
                    illegal = True
                    break
                sig = repr(ev)
                buckets.setdefault(sig, ([], []))
                buckets[sig][0].append(t)
                buckets[sig][1].append(w)
            if illegal:
                continue
            v = 0.0
            for sig, (ss, ws) in buckets.items():
                tot = sum(ws)
                v += tot * self.solve(ss, [x / tot for x in ws],
                                      depth + 1, path + (sig,))
            if root:
                self.action_values[repr(a)] = v
            if best is None or v > best:
                best = v
                if root:
                    self.best_action = a
            if ub is not None and best >= ub - 1e-9:
                break               # attains the bound; nothing can beat it
        return best if best is not None else 0.0

    def _claim_candidates(self, states):
        """The declarations worth considering: those TRUE in some candidate.

        All 3^6 = 729 assignments over the team are legal, but a declaration
        true in no candidate deal scores its worst outcome in every one of
        them (-1 under the opponent-award baseline; 0 or -1 under the legacy
        null variant), while one true in some candidate scores +1 there and
        no worse elsewhere. So the rest are weakly dominated under either
        rule and enumerating them only costs tree. Outcomes are priced by
        the engine through ``t.apply`` with the state's real rules, so the
        search stays exact under both settings -- but the VALUE of a partly
        wrong declaration differs between them (worth p under null, 2p-1
        under opponent-award), which is why solver outputs are stamped with
        the rules via the callers' fingerprints.
        """
        team = team_of(self.me)
        seen = set()
        out = []
        for h in self.live:
            base = h * CARDS_PER_HALF_SUIT
            for st in states:
                if st.set_winner[h] is not None:
                    continue
                holders = tuple(st.holder_of(base + i)
                                for i in range(CARDS_PER_HALF_SUIT))
                if any(x is None or team_of(x) != team for x in holders):
                    continue        # not ours in this deal; cannot be true
                if (h, holders) not in seen:
                    seen.add((h, holders))
                    out.append(Claim(h, holders))
        return out

    def _opponent(self, states, weights, depth, seat, path=()):
        """A deal the champion cannot move in is STUCK, not impossible.

        This used to drop such a state and renormalise the rest, which quietly
        asserts that the world does not exist and inflates every other deal's
        weight. ``champion_value`` does the opposite: it stops that deal's
        rollout and scores it where it stands, which is what the harness does
        with an unresolved half-suit -- it counts for nobody. The two therefore
        disagreed about the same strategy, and the tree/rollout control caught
        it at m = 2 with a single dropped state: tree -0.4167 against a rollout
        of -0.3333 on six deals whose true values are 0, 0, 0, -1, 0, -1.

        The rollout is the one that is right, so the tree now scores a stuck
        deal in place and keeps its weight. With nothing stuck this is
        arithmetically identical to the old code -- the weights already sum to
        one, so the old ``tot / norm`` was ``tot`` -- which is why m = 1, where
        nothing was ever stuck, is unaffected.
        """
        buckets = {}
        stuck = 0.0
        # Once for the node, not once per deal: the deals share the history.
        # And by concatenation when the lengths line up, not by walking it.
        hist = None
        if self._root_hist is not None and \
                len(self._root_hist) + len(path) == len(states[0].history):
            hist = self._root_hist + path
        else:
            hist = tuple(repr(e) for e in states[0].history)
        for s, w in zip(states, weights):
            a = _champion_action(self.spec, self.rules, seat, s, hist)
            t = _clone(s)
            if a is None:
                self.opp_dropped += 1
                v = self._value(s)
                stuck += w * (0.0 if v is None else v)
                continue
            try:
                ev = t.apply(seat, a)
            except Exception:
                self.opp_dropped += 1
                v = self._value(s)
                stuck += w * (0.0 if v is None else v)
                continue
            sig = repr(ev)
            buckets.setdefault(sig, ([], []))
            buckets[sig][0].append(t)
            buckets[sig][1].append(w)
        if not buckets:
            return stuck
        # NO renormalisation. The incoming weights sum to one over ALL states
        # including the stuck ones, so dividing by the surviving weight would
        # be the very error this method was fixed for.
        v = stuck
        for sig, (ss, ws) in buckets.items():
            tot = sum(ws)
            v += tot * self.solve(ss, [x / tot for x in ws],
                                  depth + 1, path + (sig,))
        return v
