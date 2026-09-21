"""Exact belief tracking over hidden cards.

Theory (see SPEC.md section 5): every card movement in Literature is public,
a successful ask names the card and a claim reveals all six locations. So the
current location of any card is a deterministic function of the INITIAL DEAL
plus the public log, and every piece of hidden-state inference reduces to
constraints on the initial deal:

- per-card candidate sets over initial owners (6-bit masks),
- exact per-player initial hand sizes (8 or 9),
- OR-constraints "at least one of these cards was initially player p's"
  (from "asker must hold a card of the half-suit"),

all maintained incrementally from public events (plus the observer's own
hand, which pins its own cards). A randomized sampler draws full deals
consistent with all constraints, yielding both marginal probabilities and
determinizations for search agents.

SOUNDNESS vs COMPLETENESS (be precise about which we have):

- SOUND: every fact this engine asserts as certain is genuinely implied by
  public information. The true world is never excluded from the support.
  Tested by (a) truth-in-support checks at every step for all six seats plus
  an external spectator, and (b) independent replay-validation of sampled
  worlds against the whole public history.
- NOT FULLY COMPLETE: propagation runs candidate-set filtering, exact
  per-player count reasoning, OR unit-propagation, OR subsumption, and a
  disjoint-OR pigeonhole rule. It is NOT full arc-consistency over the
  combined constraint system, so exotic positions can hold a logically
  certain fact that the propagator leaves merely "very likely". Such facts
  still show up as ~1.0 sampled marginals, so agents lose nothing but the
  proof. Do not describe this engine as complete.

SAMPLING BIAS: the sampler produces worlds that satisfy every constraint,
but not uniformly over the consistent set (OR seeding and quota weighting
skew it). Marginals are therefore good heuristic estimates, not exact
posteriors. Correcting this (importance weighting or MCMC refinement) is an
open item in RESEARCH_LOG.md.
"""

from __future__ import annotations

import random
from typing import Optional

from .cards import (
    CARDS_PER_HALF_SUIT, NUM_PLAYERS, cards_per_player, deck_size,
    half_suit_cards, half_suit_of,
)
from .engine import AskEvent, ClaimEvent, Event, PassEvent
from .observation import Observation
from .rules import RuleConfig

RESOLVED = -2
ALL_PLAYERS_MASK = (1 << NUM_PLAYERS) - 1


class BeliefContradiction(Exception):
    """Constraints became unsatisfiable, indicating an engine or logic bug."""


class BeliefState:
    """Tracks everything logically derivable about hidden cards.

    ``observer`` is the seat whose private hand is known (None = external
    spectator who sees only public events).
    """

    def __init__(self, rules: RuleConfig, observer: Optional[int] = None):
        self.rules = rules
        self.observer = observer
        self.n = deck_size(rules.variant)
        self.per = cards_per_player(rules.variant)
        # candidates[c]: bitmask of players who may have initially held c
        self.candidates = [ALL_PLAYERS_MASK] * self.n
        # public_loc[c]: None (never publicly seen), player id, or RESOLVED
        self.public_loc: list[Optional[int]] = [None] * self.n
        # active OR constraints: (tuple_of_cards, player)
        self.ors: list[tuple[tuple[int, ...], int]] = []
        self._cursor = 0            # events processed so far
        self._pinned_count = [0] * NUM_PLAYERS
        self._observer_initialized = False
        # Sampler cache: rebuilt only when the constraint store changes.
        self._version = 0
        self._cache = None
        self._cache_version = -1

    # -- primitives -----------------------------------------------------------

    def _pin(self, card: int, player: int) -> None:
        cur = self.candidates[card]
        bit = 1 << player
        if not cur & bit:
            raise BeliefContradiction(
                f"card {card} pinned to {player} but candidates={cur:06b}")
        if cur == bit:
            return
        self.candidates[card] = bit
        self._pinned_count[player] += 1
        self._version += 1

    def _exclude(self, card: int, player: int) -> None:
        cur = self.candidates[card]
        bit = 1 << player
        if not cur & bit:
            return
        new = cur & ~bit
        if new == 0:
            raise BeliefContradiction(f"card {card} has no possible owner")
        self.candidates[card] = new
        self._version += 1
        if new & (new - 1) == 0:  # became singleton
            self._pinned_count[new.bit_length() - 1] += 1

    def is_pinned(self, card: int) -> bool:
        c = self.candidates[card]
        return c & (c - 1) == 0

    def pinned_owner(self, card: int) -> Optional[int]:
        c = self.candidates[card]
        if c & (c - 1) == 0:
            return c.bit_length() - 1
        return None

    # -- event ingestion ------------------------------------------------------

    def update(self, obs: Observation) -> None:
        """Ingest all new events from the observation (incremental).

        Must be attached from the START of a game: the constraint system is
        anchored on the initial deal (``per`` cards each). Attaching to a
        mid-game position with a truncated history is unsupported and raises
        rather than silently producing wrong beliefs.
        """
        if self._cursor == 0 and not self._observer_initialized:
            dealt = sum(obs.hand_counts) + 6 * sum(
                1 for w in obs.set_winner if w is not None)
            if not obs.history and dealt != self.n:
                raise BeliefContradiction(
                    "BeliefState must be attached at game start; this "
                    f"position accounts for {dealt} of {self.n} cards with an "
                    "empty history (truncated transcript?)")
        if self.observer is not None and not self._observer_initialized:
            if obs.player != self.observer:
                raise ValueError("observation is for a different seat")
            initial = obs.initial_hand()
            for c in range(self.n):
                if initial & (1 << c):
                    self._pin(c, self.observer)
                else:
                    self._exclude(c, self.observer)
            self._observer_initialized = True
        events = obs.history
        while self._cursor < len(events):
            self._ingest(events[self._cursor])
            self._cursor += 1
        self._propagate()

    def _ingest(self, ev: Event) -> None:
        if isinstance(ev, AskEvent):
            self._ingest_ask(ev)
        elif isinstance(ev, ClaimEvent):
            base = ev.half_suit * CARDS_PER_HALF_SUIT
            for i, holder in enumerate(ev.revealed):
                c = base + i
                if self.public_loc[c] is None:
                    self._pin(c, holder)
                self.public_loc[c] = RESOLVED
            self._version += 1
        elif isinstance(ev, PassEvent):
            pass  # passing carries no hidden-card information

    def _ingest_ask(self, ev: AskEvent) -> None:
        a, g, c = ev.asker, ev.target, ev.card
        hs = half_suit_of(c)
        # (1) asker held >=1 card of the half-suit at ask time
        satisfied = False
        or_cards = []
        for c2 in half_suit_cards(hs):
            loc = self.public_loc[c2]
            if loc == a:
                satisfied = True  # publicly held a card of the suit
                break
            if loc is None:
                if not self.rules.allow_bluff_asks and c2 == c:
                    continue  # asker cannot hold the asked card itself
                or_cards.append(c2)
        if not satisfied:
            if not or_cards:
                raise BeliefContradiction("legal ask but no possible suit card")
            self.ors.append((tuple(or_cards), a))
            self._version += 1
        # (2) asker did not hold the asked card (no-bluff rules)
        if not self.rules.allow_bluff_asks and self.public_loc[c] is None:
            self._exclude(c, a)
        # (3) outcome
        if ev.success:
            if self.public_loc[c] is None:
                self._pin(c, g)       # it was with the target, hence dealt there
            self.public_loc[c] = a    # and now publicly sits with the asker
            self._version += 1
        else:
            if self.public_loc[c] is None:
                self._exclude(c, g)

    # -- constraint propagation ------------------------------------------------

    def _propagate(self) -> None:
        changed = True
        while changed:
            changed = False
            quota = [self.per] * NUM_PLAYERS
            possible = [0] * NUM_PLAYERS   # unpinned cards that could be p's
            for c in range(self.n):
                cand = self.candidates[c]
                if cand & (cand - 1) == 0:
                    quota[cand.bit_length() - 1] -= 1
                else:
                    for p in range(NUM_PLAYERS):
                        if cand & (1 << p):
                            possible[p] += 1
            for p in range(NUM_PLAYERS):
                if quota[p] < 0 or possible[p] < quota[p]:
                    raise BeliefContradiction(f"player {p} count infeasible")
                if quota[p] == 0:
                    for c in range(self.n):
                        cand = self.candidates[c]
                        if cand & (1 << p) and cand & (cand - 1):
                            self._exclude(c, p)
                            changed = True
                elif possible[p] == quota[p]:
                    for c in range(self.n):
                        cand = self.candidates[c]
                        if cand & (1 << p) and cand & (cand - 1):
                            self._pin(c, p)
                            changed = True
            # OR-constraint filtering (unit propagation)
            if self.ors:
                still = []
                for cards, p in self.ors:
                    bit = 1 << p
                    live = tuple(c for c in cards if self.candidates[c] & bit)
                    if any(self.candidates[c] == bit for c in live):
                        continue  # satisfied with certainty
                    if not live:
                        raise BeliefContradiction("OR constraint unsatisfiable")
                    if len(live) == 1:
                        self._pin(live[0], p)
                        changed = True
                        continue
                    still.append((live, p))
                if still != self.ors:
                    self.ors = still
                    self._version += 1
                if self._propagate_or_pigeonhole(quota):
                    changed = True

    def _propagate_or_pigeonhole(self, quota: list[int]) -> bool:
        """Cross-OR reasoning that unit propagation alone misses.

        (a) COUNT PIGEONHOLE: if player p has k pairwise-disjoint OR
            requirements, at least k of p's remaining unknown slots are
            consumed by them. If k equals p's remaining quota, every card
            OUTSIDE those groups cannot be p's.
        (b) SUBSUMPTION: an OR whose live set is a subset of another OR for
            the same player makes the superset redundant.
        """
        changed = False
        by_player: dict[int, list[tuple[int, ...]]] = {}
        for cards, p in self.ors:
            by_player.setdefault(p, []).append(cards)
        for p, groups in by_player.items():
            groups = sorted(set(groups), key=len)
            kept: list[tuple[int, ...]] = []
            for g in groups:
                gs = set(g)
                if any(set(k) <= gs for k in kept):
                    continue
                kept.append(g)
            disjoint: list[set[int]] = []
            used: set[int] = set()
            for g in kept:
                gs = set(g)
                if not (gs & used):
                    disjoint.append(gs)
                    used |= gs
            if not disjoint:
                continue
            bit = 1 << p
            remaining = quota[p]
            if len(disjoint) > remaining:
                raise BeliefContradiction(
                    f"player {p}: {len(disjoint)} disjoint requirements but "
                    f"only {remaining} unknown slots")
            if len(disjoint) == remaining and remaining > 0:
                for c in range(self.n):
                    cand = self.candidates[c]
                    if (cand & bit and cand & (cand - 1) and c not in used):
                        self._exclude(c, p)
                        changed = True
        return changed

    # -- queries ---------------------------------------------------------------

    def current_holder_mask(self, card: int) -> int:
        """Bitmask of players who might hold ``card`` RIGHT NOW (0 if resolved)."""
        loc = self.public_loc[card]
        if loc == RESOLVED:
            return 0
        if loc is not None:
            return 1 << loc
        return self.candidates[card]  # unmoved: current owner == initial owner

    def known_current_hands(self) -> list[int]:
        """Cards known with certainty to currently sit with each player."""
        hands = [0] * NUM_PLAYERS
        for c in range(self.n):
            m = self.current_holder_mask(c)
            if m and m & (m - 1) == 0:
                hands[m.bit_length() - 1] |= 1 << c
        return hands

    # -- world sampling ----------------------------------------------------------

    def sample_current_hands(self, rng: random.Random,
                             max_restarts: int = 200) -> list[int]:
        """Sample a full assignment of current hands consistent with all
        constraints. Raises BeliefContradiction if none is found."""
        deal = self._sample_initial(rng, max_restarts)
        hands = [0] * NUM_PLAYERS
        for c in range(self.n):
            loc = self.public_loc[c]
            if loc == RESOLVED:
                continue
            hands[deal[c] if loc is None else loc] |= 1 << c
        return hands

    def _build_sample_cache(self) -> None:
        """Precompute everything about the constraint store that does not
        change between draws.

        Measured motivation: sampling dominated agent compute at 736 us per
        world, and profiling showed most of that was re-deriving this
        structure (pinned cards, quotas, constraint ordering) on EVERY draw
        even though beliefs change at most once per public action while
        dozens of worlds are drawn in between.
        """
        n = self.n
        pinned = [-1] * n
        base_quota = [self.per] * NUM_PLAYERS
        free_cards = []
        cand_players: list[tuple[int, ...]] = [()] * n
        for c in range(n):
            cand = self.candidates[c]
            if cand & (cand - 1) == 0:
                p = cand.bit_length() - 1
                pinned[c] = p
                base_quota[p] -= 1
            else:
                free_cards.append(c)
                cand_players[c] = tuple(p for p in range(NUM_PLAYERS)
                                        if cand & (1 << p))
        free_cards.sort(key=lambda c: len(cand_players[c]))
        groups: list[list[int]] = []
        start = 0
        for i in range(1, len(free_cards) + 1):
            if (i == len(free_cards)
                    or len(cand_players[free_cards[i]])
                    != len(cand_players[free_cards[start]])):
                groups.append(free_cards[start:i])
                start = i
        # Disjoint OR groups can be satisfied during construction instead of
        # repaired afterwards (repair was 66% of sampling time).
        seeded: list[tuple[int, tuple[int, ...]]] = []
        overlapping: list[tuple[tuple[int, ...], int]] = []
        used_cards: set[int] = set()
        for cards, p in sorted(self.ors, key=lambda t: len(t[0])):
            if any(pinned[c] == p for c in cards):
                continue  # already satisfied by a pinned card
            live = tuple(c for c in cards
                         if pinned[c] == -1 and self.candidates[c] & (1 << p))
            if live and not (set(live) & used_cards):
                seeded.append((p, live))
                used_cards |= set(live)
            else:
                overlapping.append((cards, p))
        self._cache = {
            "pinned": pinned,
            "base_quota": base_quota,
            "free_groups": groups,
            "cand_players": cand_players,
            "seeded_ors": seeded,
            "residual_ors": overlapping,
            "unpinned_by_player": [
                [c for c in free_cards if self.candidates[c] & (1 << p)]
                for p in range(NUM_PLAYERS)],
        }
        self._cache_version = self._version

    def _sample_initial(self, rng: random.Random,
                        max_restarts: int = 200) -> list[int]:
        """Sample initial owners for every card, satisfying candidate sets,
        exact per-player counts, and OR constraints."""
        if self._cache is None or self._cache_version != self._version:
            self._build_sample_cache()
        cache = self._cache
        pinned = cache["pinned"]
        base_quota = cache["base_quota"]
        groups = cache["free_groups"]
        cand_players = cache["cand_players"]
        seeded = cache["seeded_ors"]
        residual = cache["residual_ors"]
        shuffle = rng.shuffle
        random_ = rng.random

        for _ in range(max_restarts):
            quota = base_quota[:]
            deal = pinned[:]
            ok = True
            # 1) satisfy disjoint OR requirements up front
            for p, live in seeded:
                if quota[p] <= 0:
                    ok = False
                    break
                c = live[int(random_() * len(live))]
                if deal[c] != -1:
                    continue
                deal[c] = p
                quota[p] -= 1
            if not ok:
                continue
            # 2) fill the rest, most-constrained group first.
            # Shuffle a COPY: mutating the cached ordering in place made
            # sampling depend on how many draws had happened before, so two
            # belief states with identical constraints and identical seeds
            # produced different worlds. Sampling must be a pure function of
            # (constraints, rng).
            for cached_group in groups:
                group = cached_group
                if len(group) > 1:
                    group = list(cached_group)
                    shuffle(group)
                for c in group:
                    if deal[c] != -1:
                        continue  # already fixed by an OR seed
                    total = 0
                    for p in cand_players[c]:
                        q = quota[p]
                        if q > 0:
                            total += q
                    if total == 0:
                        ok = False
                        break
                    r = random_() * total
                    acc = 0
                    pick = -1
                    for p in cand_players[c]:
                        q = quota[p]
                        if q > 0:
                            acc += q
                            if r < acc:
                                pick = p
                                break
                    if pick < 0:  # float edge case: take the last feasible
                        for p in cand_players[c]:
                            if quota[p] > 0:
                                pick = p
                    deal[c] = pick
                    quota[pick] -= 1
                if not ok:
                    break
            if not ok:
                continue
            # Repair must validate against ALL constraints, not just the
            # residual ones: a swap that fixes an overlapping constraint can
            # break one satisfied during seeding. (Caught by the sampled-world
            # replay-validation test.)
            if not residual or self._repair_ors(deal, rng, self.ors):
                return deal
        raise BeliefContradiction("sampler failed; constraints too tight?")

    def _repair_ors(self, deal: list[int], rng: random.Random,
                    constraints=None, max_swaps: int = 100) -> bool:
        """Swap-repair the assignment until the given OR constraints hold."""
        constraints = self.ors if constraints is None else constraints
        cache = self._cache
        unpinned = (cache["unpinned_by_player"] if cache else None)
        for _ in range(max_swaps):
            violated = None
            for cards, p in constraints:
                if not any(deal[c] == p for c in cards):
                    violated = (cards, p)
                    break
            if violated is None:
                return True
            cards, p = violated
            movable = [c for c in cards if self.candidates[c] & (1 << p)
                       and not self.is_pinned(c)]
            rng.shuffle(movable)
            fixed = False
            for c in movable:
                q = deal[c]
                pool = (unpinned[p] if unpinned is not None
                        else [x for x in range(self.n) if not self.is_pinned(x)])
                swaps = [c2 for c2 in pool
                         if deal[c2] == p and self.candidates[c2] & (1 << q)]
                if swaps:
                    c2 = rng.choice(swaps)
                    deal[c], deal[c2] = p, q
                    fixed = True
                    break
            if not fixed:
                return False
        return False

    def marginals(self, rng: random.Random, n_samples: int = 64) -> list[list[float]]:
        """P(player p currently holds card c), estimated over sampled worlds.
        Returns [card][player] floats; resolved cards are all-zero rows."""
        counts = [[0] * NUM_PLAYERS for _ in range(self.n)]
        for _ in range(n_samples):
            hands = self.sample_current_hands(rng)
            for p in range(NUM_PLAYERS):
                h = hands[p]
                while h:
                    low = h & -h
                    counts[low.bit_length() - 1][p] += 1
                    h ^= low
        return [[k / n_samples for k in row] for row in counts]


# ---------------------------------------------------------------------------
# Independent gold-standard validator (used by tests and search sanity checks)
# ---------------------------------------------------------------------------

def validate_deal_against_history(rules: RuleConfig, initial_hands: list[int],
                                  history: tuple[Event, ...]) -> bool:
    """Replay the public history assuming ``initial_hands`` and check every
    event is consistent (legality-relevant facts only). Completely independent
    of BeliefState's constraint encoding."""
    from .cards import half_suit_mask
    hands = list(initial_hands)
    for ev in history:
        if isinstance(ev, AskEvent):
            bit = 1 << ev.card
            hs_mask = half_suit_mask(half_suit_of(ev.card))
            if not rules.allow_bluff_asks and hands[ev.asker] & bit:
                return False           # asker held the card they asked for
            if not hands[ev.asker] & hs_mask:
                return False           # asker had no card of the half-suit
            if ev.success:
                if not hands[ev.target] & bit:
                    return False       # target didn't actually have it
                hands[ev.target] ^= bit
                hands[ev.asker] |= bit
            else:
                if hands[ev.target] & bit:
                    return False       # target had it but said no
        elif isinstance(ev, ClaimEvent):
            base = ev.half_suit * CARDS_PER_HALF_SUIT
            for i, holder in enumerate(ev.revealed):
                if not hands[holder] & (1 << (base + i)):
                    return False       # revealed location wrong
            for p in range(NUM_PLAYERS):
                hands[p] &= ~half_suit_mask(ev.half_suit)
        elif isinstance(ev, PassEvent):
            if hands[ev.player] != 0:
                return False           # only cardless players pass
    return True
