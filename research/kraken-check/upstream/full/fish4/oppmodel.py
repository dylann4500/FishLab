"""Inferring hidden cards from what players CHOSE to ask, not just legality.

THE GAP THIS FILLS
------------------
v0.3's belief tracker extracts from an ask exactly three hard facts: the asker
held at least one card of that half-suit, the asker did not hold the asked card,
and the target did or did not have it. All three are consequences of the rules.
None of them uses the fact that the asker *chose* that half-suit out of the
several they could legally have asked in.

That choice is informative. Under the simplest defensible model of how a player
picks - proportional to how many cards of a half-suit they hold - the
probability of the observed choice is

    P(ask in H | hand) = depth_H(hand) / |hand|

and the denominator is public, so it is the same in every candidate world and
cancels. The likelihood of the whole observed ask sequence is therefore
proportional to a product of depths, one factor per ask, which turns "who asked
where" into a soft weight over worlds.

WHY THIS IS THE RIGHT PLACE TO PUT IT
-------------------------------------
The likelihood cannot go into the constraint propagator, because it is not a
constraint: it excludes nothing, it only re-weights. It fits naturally into the
importance sampler, which is already carrying a per-world weight, and costs one
integer increment per card during a draw.

HONESTY ABOUT WHAT THIS IS
--------------------------
This is a *model*, not a deduction, and the rest of the engine is deduction. If
the model is wrong the posterior gets worse, and against opponents who
deliberately vary their asks it could be exploited. So it is controlled by a
single parameter ``gamma``, with ``gamma = 0`` recovering the uninformative-
choice assumption exactly, and it is validated against ground truth (the true
hidden hands in simulation) rather than by whether it feels reasonable. See
``scripts4/posterior_accuracy.py``.

A known limitation, stated rather than hidden: depth is evaluated on the INITIAL
deal, not at the moment of the ask. Cards move, so late in a game the two
diverge. Modelling the depth at ask time would require replaying each candidate
world through the whole public history, which costs O(history) per draw instead
of O(1). Measured, that refinement is a null (paper, Remark 1).

WHY AN ASK EARLY IS BETTER EVIDENCE THAN AN ASK LATE
----------------------------------------------------
The model treats every ask as equally informative about depth. It should not.
The hypothesis is that a player asks in a half-suit *in proportion to how many
cards of it they hold* - which presumes they had a choice. Early, with nine cards
across many live half-suits, they do. Late, holding two cards in one half-suit
with most of the deck resolved, legality binds: they ask where they *can*, not
where they are deep, and the same observation carries far less signal about their
hand.

``gamma_schedule`` scales the per-ask weight by where in the game the ask
happened, measured by the fraction of half-suits already resolved at that moment
- a public quantity, recoverable from the log alone.

THE PROFILE IS MEASURED, NOT CHOSEN
-----------------------------------
The first version of this used a symmetric tilt, ``1 + s*(1 - 2*frac)``, picked
for being the simplest thing with the right sign. It is the wrong shape.

``scripts4/choice_curve.py`` fits the exponent directly from self-play: at every
ask it records which half-suits were legal, the asker's initial-deal depth in
each, and which was chosen, then fits ``P(ask in H) ~ depth_H ** alpha`` by
exact conditional likelihood. Over 200 games and 17,005 decisions with a genuine
choice, with standard errors block-bootstrapped over games - which doubles them,
because decisions inside one deal are not independent draws:

    half-suits resolved   0      1      2      3      4      5     6-8
    alpha                 2.00   1.28   1.20   0.68   0.30   0.41  -0.02
    clustered SE          0.08   0.08   0.12   0.12   0.13   0.18   0.16

So against the covariate the model actually uses, the shipped constant is wrong
at both ends and in opposite directions. Pooled, the best constant is
1.207 +/- 0.046.

WHAT THAT DECAY TURNED OUT TO BE
--------------------------------
The first reading was behavioural: late asks carry no depth signal, because
legality binds and a player asks where they can rather than where they are deep.
Refitting on depth AT THE MOMENT OF THE ASK - see the ``"at_ask"`` mode below -
says most of it was not behaviour:

    covariate                 opening   1-2     3-4     5-8    pooled
    initial-deal depth          2.00    1.25    0.52    0.23    1.207
    depth at the ask            2.90    2.32    1.72    1.34    2.195

The decay survives and shrinks. Late asks do carry depth signal, about the hand
the asker held; what decays is how well the initial deal still describes that
hand. Noise in a covariate attenuates its coefficient, and here the noise grows
monotonically with the game - mean absolute disagreement between the two rises
from 0.16 cards to 0.66 - which manufactures a decay out of drift.

That is not an argument against ``gamma_schedule``, it is a better argument for
it. The engine conditions on initial-deal depth, and a covariate that degrades as
the game runs on SHOULD be believed less as the game runs on; the profile below
is a dilution correction for the covariate in use, which is a modest and
defensible thing rather than the claim about player behaviour it was first
written up as. It should also become unnecessary under ``"at_ask"``, where the
covariate does not drift - and if it does not, that is evidence the behavioural
reading had something in it after all.

``ALPHA_*`` below is a weighted quadratic through those seven bands (chi-square
8.3 on 4 dof), clamped flat past its vertex because past there the parabola
turns up and the measurements do not. ``gamma_schedule`` is the strength with
which that profile replaces the constant:

    gamma_eff(ask) = gamma * [(1 - s) + s * alpha(frac) / ALPHA_MEAN]

``ALPHA_MEAN`` is the profile averaged over the observed distribution of asks,
so ``s = 1`` redistributes the model's belief across the game without changing
how much of it there is in total. That matters: gamma was tuned by duels to a
broad plateau, and this term is a claim about SHAPE, not strength. Confounding
the two would leave a positive result unattributable. ``s = 0`` is the
incumbent, exactly.
"""

from __future__ import annotations

import math

from fish.cards import NUM_PLAYERS, half_suit_of
from fish.engine import AskEvent, ClaimEvent

from .sis import OpponentModel


#: Weighted quadratic through the seven measured bands (see the module
#: docstring). Fitted on the 54-card variant; the argument is a fraction, so it
#: carries to the 48-card one under the assumption that what matters is how far
#: the deal has run rather than how many half-suits it had.
#: Sampled free-card depths run 0..6, so a per-slot log table needs 7 entries.
DEPTH_TABLE_MAX = 7

ALPHA_Q, ALPHA_L, ALPHA_C = 3.0281, -4.7983, 1.9344

#: The parabola's vertex. Past it the fit turns upward; the measurements do not,
#: they flatten, so the profile is held at its vertex value beyond this point.
ALPHA_FLAT = 0.7923

#: The profile averaged over the observed distribution of asks (17,005 of them),
#: so dividing by it leaves the model's total strength alone and moves only its
#: distribution across the game.
ALPHA_MEAN = 1.0626


def measured_alpha(frac: float) -> float:
    """The fitted depth exponent for an ask at this point in the game."""
    f = frac if frac < ALPHA_FLAT else ALPHA_FLAT
    v = ALPHA_C + ALPHA_L * f + ALPHA_Q * f * f
    return v if v > 0.0 else 0.0


def schedule_factor(resolved: int, n_half_suits: int, s: float) -> float:
    """Weight for an ask made when ``resolved`` half-suits were already decided.

    ``s`` interpolates between the shipped constant and the measured profile:
    ``0`` returns 1 everywhere and is the incumbent exactly, ``1`` is the profile
    normalised to leave the model's average strength unchanged. Clamped at zero,
    because a negative weight is the claim that asking in a half-suit is evidence
    of being SHALLOW in it - a different model rather than a weaker form of this
    one, and not what the measurement found even at the end of the game.
    """
    if not s:
        return 1.0
    frac = (resolved / n_half_suits) if n_half_suits else 0.0
    v = (1.0 - s) + s * measured_alpha(frac) / ALPHA_MEAN
    return v if v > 0.0 else 0.0


def _snapshot(bel, where, asker, lo):
    """The asker's holding of one half-suit at one ask: what is fixed, what the
    draw decides.

    Three cases per card, and only the third is world-dependent:

      publicly moved            the log says where it is; held iff that is us
      never moved, pinned       never publicly moved means it is still with
                                whoever was dealt it, and the propagator has
                                DEDUCED who that is -- so it is fixed too
      never moved, not pinned   the draw decides

    The middle case is the one worth spelling out. A first version of this
    lumped every never-publicly-moved card the asker could hold into the free
    list, including the ones the propagator had already pinned to them. Those
    cards are not in the sampler's column map -- it only carries cards with more
    than one candidate -- so the consumer skipped them and the holding came out
    SHORT by exactly the cards we were surest about. Under the unaimed code book
    that shifts a depth; under the aimed one it corrupts the payload, and the
    round-trip check showed sender and receiver disagreeing about a message they
    had in fact both computed correctly.
    """
    const_mask = 0
    free_cards = []
    for c in range(lo, lo + 6):
        w = where.get(c)
        if w is not None:
            if w == asker:
                const_mask |= 1 << c
            continue
        cand = bel.candidates[c]
        if cand & (cand - 1) == 0:          # a single candidate: deduced
            if cand.bit_length() - 1 == asker:
                const_mask |= 1 << c
        elif cand >> asker & 1:
            free_cards.append(c)
    return const_mask, tuple(free_cards)


def build(bel, obs, gamma: float, include_self: bool = False,
          depth_mode: str = "initial", count_mode: str = "linear",
          w_unlocated: float = 0.0,
          opp_lambda: float = 0.0, order=None,
          gamma_schedule: float = 0.0, sis_tilt: float = 0.0,
          gamma_team: float | None = None, convention_beta: float = 0.0,
          convention_q: float = 0.0, convention_aim: bool = False,
          convention_book: str = "depth"):
    """Build an ``(OpponentModel, card_slot)`` pair, or ``(None, None)``.

    ``card_slot`` maps ``(player, card)`` to the model slot for that player and
    that card's half-suit, which is the form the sampler's inner loop wants.

    ``depth_mode`` chooses what "depth" means for the already-located cards of a
    half-suit, which is the part of the count the sampler does not re-draw:

    ``"initial"``
        cards the propagator has pinned to that player in the INITIAL deal. This
        matches the constraint system, which is stated over the initial deal.
    ``"current"``
        cards the player publicly holds RIGHT NOW. Superficially closer to the
        quantity the choice model is about, and measured to be much worse (see
        the results): it mixes a present-tense count with free cards that are
        being drawn over the INITIAL deal, so the two halves of the depth are
        counted at different times.
    ``"attime"``
        cards publicly located with that player AT THE MOMENT THEY ASKED,
        averaged over their asks in that half-suit. This is the quantity the
        model actually wants, and it is consistent with the free half: a card
        that is still free now was never publicly located, so its holder at the
        time of the ask was whoever was dealt it.

    ``"attime"`` needs one replay of the public log per decision, not per
    candidate world, because only the already-located half of the count depends
    on time. It remains an approximation in one respect: several asks by the same
    player in the same half-suit are collapsed onto one slot, so their per-ask
    counts are averaged rather than kept separate. Measured in play it costs
    $-1.544$ sets per deal-pair over 250 pairs, which is a refutation rather
    than a null.

    ``"at_ask"``
        the asker's true depth at the moment they asked, kept per ask rather
        than averaged. Cards move only when an ask succeeds, and a successful
        ask is entirely public -- card, asker and target are all in the log --
        so

            depth_at_ask(p, H) = depth_initial(p, H) + delta(p, H, t)

        where ``delta`` is the net of that half-suit's publicly transferred
        cards and is the SAME number in every hypothesised world. Verified
        directly: 23,268 (player, half-suit, time) triples across six games,
        zero mismatches. So the quantity ``"attime"`` was reaching for costs a
        table built once per decision, not a replay per draw, and it is exact
        rather than approximate.

        The covariate matters more than the implementation detail suggests.
        Refitting the choice model on 17,005 real decisions, at-ask-time depth
        beats initial-deal depth by 4,654 nats: the shipped covariate captures
        under a quarter of the signal the same one-parameter family can reach
        (1,403 nats above uniform against 6,057). The two diverge exactly as the
        game runs on -- mean absolute disagreement 0.16 cards at the opening
        against 0.66 by the endgame -- which also means much of the apparent
        decay of the fitted exponent over a game is regression dilution in the
        old covariate rather than a change in behaviour.

    ``count_mode`` decides how repeated asks in the same half-suit accumulate.
    Under the choice model each ask is an independent draw, so the likelihood is
    the depth raised to the number of asks: that is ``"linear"``, and it is the
    principled reading. But asks in one half-suit are plainly *not* independent
    - a player keeps working a suit because they are working it - so the linear
    form over-counts correlated evidence. ``"sqrt"`` and ``"capped"`` are the
    two obvious hedges, and which is right is an empirical question.
    """
    # See the note in posterior.py: a negative gamma is a real setting (tilt
    # toward shallow), not a synonym for off. Only exactly zero is off.
    #
    # gamma_team has to be in this guard too. "Believe nothing about opponents,
    # something about teammates" is a coherent configuration and one the sweep
    # in scripts4/gamma_split.py visits; without the extra clause it returned
    # the uniform posterior for EVERY gamma_team at gamma == 0, so a whole row
    # of the grid reported bit-identical numbers that looked like a measured
    # null. Exactly the collapse the `> 0` guard on gamma caused before it.
    off_team = gamma_team is None or gamma_team == 0.0
    if (gamma == 0.0 and off_team and opp_lambda <= 0.0
            and convention_beta == 0.0 and convention_q == 0.0):
        return None, None
    counts: dict[tuple[int, int], int] = {}
    #: Sum of per-ask schedule factors per slot, for gamma_schedule. Left equal
    #: to the raw count when the schedule is off, so the mean factor is 1.
    sched: dict[tuple[int, int], float] = {}
    #: Net publicly-transferred cards of each half-suit held by each player,
    #: running forward through the log. World-independent by construction: an
    #: ask succeeds or fails identically in every hypothesised deal, because the
    #: log says which.
    pub: dict[tuple[int, int], int] = {}
    #: Per slot, the value of `pub` at each of that slot's asks.
    deltas: dict[tuple[int, int], list] = {}
    #: (asker, half_suit, card, const_mask, free_cards) for every ask by our
    #: own side, when the convention is live. See fish4/convention.py.
    conv_asks: list = []
    #: Cards the public record can place, for the aimed code book's target.
    #: A successful ask locates one card; a declaration resolves six. A FAILED
    #: ask locates nothing -- it proves the target does not hold the card and
    #: that the asker holds SOME other card of the half-suit, both of which are
    #: constraints and neither of which is a location.
    located: set = set()
    #: Publicly-known current location of each card, or absent when the card
    #: has never been publicly moved and so is still wherever it was dealt.
    #: THE CONVENTION NEEDS THIS AND THE DEPTH MODEL DOES NOT, which is the
    #: whole reason it is here. The sampler works in INITIAL-DEAL space --
    #: `bel.candidates[c]` is the set of players who may have initially held c
    #: -- but the code book is a function of the hand the asker held AT THE
    #: MOMENT OF THE ASK. Cards move. A teammate who asked in a half-suit
    #: twenty moves ago and has since been given three of its cards would be
    #: decoded against a holding they never had, and the further back the ask,
    #: the more wrong the test. Measured: the number of distinct half-suit
    #: holdings the sampler entertains for an asker falls from 4.97 at the most
    #: recent of our asks to 1.30 by fifteen asks back, so the stale end of the
    #: ledger is where the term is both least informative and most misread.
    where: dict[int, int] = {}
    me = obs.player
    n_hs = len(obs.set_winner)
    resolved = 0
    for ev in obs.history:
        if isinstance(ev, ClaimEvent):
            # The game clock: how much of the deck is already decided. Public,
            # and recoverable from the log alone.
            resolved += 1
            hsc = getattr(ev, "half_suit", None)
            if hsc is not None:
                located.update(range(hsc * 6, hsc * 6 + 6))
            continue
        if not isinstance(ev, AskEvent):
            continue
        if not include_self and ev.asker == me:
            # Still record the transfer: our own asks move cards too, and the
            # deltas of OTHER players' slots depend on every movement, not only
            # the ones we are modelling.
            if ev.success:
                h = half_suit_of(ev.card)
                pub[(ev.asker, h)] = pub.get((ev.asker, h), 0) + 1
                pub[(ev.target, h)] = pub.get((ev.target, h), 0) - 1
                where[ev.card] = ev.asker
                located.add(ev.card)
            continue
        hs = half_suit_of(ev.card)
        # The convention reads only OUR OWN side's asks. It is an agreement
        # between us and our partners about how to name a card; an opponent
        # never agreed to it and reading their asks through it would be
        # inventing a message nobody sent.
        if (convention_beta or convention_q) and \
                (ev.asker % 2) == (me % 2):
            # The aimed target, computed from the public record AS OF THIS ASK.
            # Both sides must name the same half-suit without communicating, so
            # it can only be a function of common knowledge -- and it must be
            # the common knowledge the SENDER had, not the receiver's later
            # view, which is why it is snapshotted here rather than derived at
            # the end. See fish4/convention.py.
            g_hs = g_const = None
            g_free: tuple = ()
            targets: tuple = ()
            if convention_aim or convention_book == "locate":
                best, g_hs = -1, 0
                for h in range(n_hs):
                    u = sum(1 for c in range(h * 6, h * 6 + 6)
                            if c not in located)
                    if u > best:
                        best, g_hs = u, h
                g_const, g_free = _snapshot(bel, where, ev.asker, g_hs * 6)
                if convention_book == "locate":
                    # U: the unlocated cards of the target half-suit, in index
                    # order, AS OF THIS ASK. Public on both sides, so no part
                    # of the message has to carry it.
                    targets = tuple(c for c in range(g_hs * 6, g_hs * 6 + 6)
                                    if c not in located)
            # Snapshot the asked half-suit as of THIS ask, before its own
            # transfer is applied. Three cases per card, and only the third is
            # world-dependent:
            #   publicly with the asker      -> definitely held, constant
            #   publicly with someone else   -> definitely not held
            #   never publicly moved         -> held iff dealt to the asker
            lo = hs * 6
            const_mask, free_cards = _snapshot(bel, where, ev.asker, lo)
            conv_asks.append((ev.asker, hs, ev.card, const_mask,
                              tuple(free_cards), g_hs, g_const, g_free,
                              targets))
        if ev.success:
            where[ev.card] = ev.asker
            located.add(ev.card)
        key = (ev.asker, hs)
        counts[key] = counts.get(key, 0) + 1
        sched[key] = sched.get(key, 0.0) + schedule_factor(
            resolved, n_hs, gamma_schedule)
        # The asker's depth AT THIS MOMENT is their initial depth plus the net
        # of the half-suit's cards that have publicly moved to them since the
        # deal. Recorded before this ask's own transfer, which had not happened
        # when they chose it.
        deltas.setdefault(key, []).append(pub.get(key, 0))
        if ev.success:
            pub[key] = pub.get(key, 0) + 1
            tkey = (ev.target, hs)
            pub[tkey] = pub.get(tkey, 0) - 1
    if not counts and opp_lambda <= 0.0:
        return None, None
    slots = {key: i for i, key in enumerate(counts)}
    weight = [0.0] * len(slots)
    base = [0] * len(slots)

    # ---- opportunity, per prereg/unlocated_belief.md -----------------------
    # How many cards of the half-suit still sit with nobody the PUBLIC record
    # can name. `public_loc` and not `candidates`: candidates is narrowed by
    # the observing seat's own hand, so it is this seat's private view, while
    # the choice model is a claim about what the ASKER was looking at. The fit
    # in scripts4/choice_curve.py recorded `bel.public_loc[c] is None` for
    # exactly that reason -- it is common knowledge, identical for every
    # observer and under every candidate world. Scoring the belief against a
    # differently-defined covariate would be the transfer error that closed
    # the w_expose direction without a single game being played.
    #
    # max(u, 1): the fitted exponent is NEGATIVE, u = 0 is legal (hold a card,
    # know where all six are), and 0 ** -3.96 is unbounded. The fit clamps at
    # 1e-12, which is harmless in a log-likelihood and would hand one half-suit
    # a weight near 1e47 here. Fixed in the pre-registration, not after.
    #
    # At the default this loop does not run and `weight[i]` keeps its exact
    # incumbent value -- bit-identical, like w_contest=0.0 and endgame_m=0,
    # and tests4/test_unlocated_inert.py asserts it rather than trusting it.
    unloc_factor: dict[int, float] = {}
    if w_unlocated:
        for _key in slots:
            hs_ = _key[1]
            if hs_ not in unloc_factor:
                lo_ = hs_ * 6
                u = sum(1 for c_ in range(lo_, lo_ + 6)
                        if bel.public_loc[c_] is None)
                unloc_factor[hs_] = float(max(u, 1)) ** w_unlocated
    for key, i in slots.items():
        n = counts[key]
        # The schedule enters as the MEAN factor over that slot's asks, so it
        # rescales the weight without disturbing how count_mode shapes it.
        mean_f = (sched[key] / n) if n else 1.0
        if count_mode == "sqrt":
            n = math.sqrt(n)
        elif count_mode == "capped":
            n = 1.0
        # One gamma per SIDE. The choice model is a property of the asker's
        # policy and is the same for everyone; gamma is not the model, it is
        # how sharply the model is believed as a likelihood weight, and the two
        # sides are believed for different jobs. Teammate evidence is what
        # resolves an allocation -- which of our own seats holds which card of
        # a half-suit our team already owns outright -- and 95.3% of this
        # engine's residual errors are of exactly that kind. Opponent evidence
        # is what picks the next ask. A single gamma forces one compromise
        # between two jobs whose returns were measured to differ by 2.6x
        # (prereg/information_ceiling_split.md). gamma_team=None keeps one
        # number for both and is bit-identical to the incumbent.
        g = gamma
        if gamma_team is not None and (key[0] % 2) == (me % 2):
            g = gamma_team
        w = g * n * mean_f
        if w_unlocated:
            w *= unloc_factor[key[1]]
        weight[i] = w
    # base[i] = cards of that half-suit already pinned to that player by the
    # propagator, i.e. depth contributed by cards the sampler will not re-draw
    if depth_mode == "attime":
        _attime_base(bel, obs, slots, base, include_self, me)
    else:
      for (player, hs), i in slots.items():
        lo = hs * 6
        n = 0
        for c in range(lo, lo + 6):
            if depth_mode == "current":
                m = bel.current_holder_mask(c)
                if m and m & (m - 1) == 0 and m.bit_length() - 1 == player:
                    n += 1
            else:
                cand = bel.candidates[c]
                if cand & (cand - 1) == 0 and cand.bit_length() - 1 == player:
                    n += 1
        base[i] = n
    card_slot: dict[tuple[int, int], int] = {}
    for (player, hs), i in slots.items():
        lo = hs * 6
        for c in range(lo, lo + 6):
            card_slot[(player, c)] = i

    # ---- the "nobody declared it" signal -----------------------------------
    # A team that holds an entire half-suit and can place the split declares it.
    # So a world in which the OPPONENTS hold all six cards of a half-suit that
    # is still unresolved is less likely than the count constraints alone would
    # suggest. It is not impossible - they may hold it and not be able to place
    # it - so this is a soft weight, not a constraint, with its own parameter.
    # Only the opponents' side is used: our own team may well hold a set without
    # us being able to prove it, which is precisely the situation this engine
    # spends its time in.
    # CARD IDS, not column indices. They used to be indices into whatever
    # `order` the caller passed -- which is the unsorted free list -- while the
    # thing they end up indexing is a `picks` matrix whose columns are in
    # SISSampler.order, a different sort. Six of eight columns differ on a
    # typical position, so the "did the opponents take this whole half-suit"
    # test was reading a mixture of cards from other half-suits. Handing out
    # card ids removes the coupling: the consumer maps them through the order
    # that actually applies to its own array.
    set_cards = []
    if opp_lambda > 0.0 and order is not None:
        free_set = set(order)
        my_team = obs.player & 1
        for hs in range(len(obs.set_winner)):
            if obs.set_winner[hs] is not None:
                continue
            lo = hs * 6
            cols = []
            possible = True
            for c in range(lo, lo + 6):
                if c in free_set:
                    cols.append(c)
                    continue
                m = bel.current_holder_mask(c)
                if m == 0 or (m & (m - 1) == 0
                              and (m.bit_length() - 1) & 1 == my_team):
                    possible = False       # a card of ours, or already resolved
                    break
            if possible and cols:
                set_cards.append(tuple(cols))
    #: Per-slot log terms for the at-ask-time model. Depth at the moment of an
    #: ask is the initial depth plus a delta the public log fixes, identically
    #: in every world, so the better covariate costs a table built once per
    #: decision rather than a replay per draw. Verified on 23,268
    #: (player, half-suit, time) triples across six games: zero mismatches.
    table = None
    if depth_mode == "at_ask" and slots:
        table = []
        for key, i in slots.items():
            n = counts[key]
            scale = weight[i] / (gamma * n) if (gamma and n) else 0.0
            ds = deltas.get(key, [0] * n)
            row = []
            for d in range(DEPTH_TABLE_MAX):
                tot = 0.0
                for dl in ds:
                    v = d + base[i] + dl
                    tot += math.log(v if v > 0 else 1e-9)
                row.append(gamma * scale * tot)
            table.append(tuple(row))
    # ---- the convention ----------------------------------------------------
    # Per ask by our own side, everything the receiver's forward test needs:
    # which cards of that half-suit are already PINNED to the asker (the same
    # in every world, so precomputed once), and which are still free and could
    # fall to them (so the test can be finished per draw). The test itself is
    # fish4.convention.is_encoded, which asks the forward question -- would
    # this hypothesised hand have named this card? -- and so never has to
    # invert an encoding whose modulus depends on the answer.
    # Built inline above, at each ask, because the holding it needs is the one
    # that existed then and cannot be reconstructed from the final belief.
    conv = tuple(conv_asks) if conv_asks else None

    return (OpponentModel(weight, base, set_cards=set_cards,
                          opp_lambda=opp_lambda, my_team=obs.player & 1,
                          tilt_strength=sis_tilt, depth_table=table,
                          convention=conv, convention_beta=convention_beta,
                          convention_q=convention_q,
                          convention_aim=convention_aim,
                          convention_book=convention_book),
            card_slot)


def _attime_base(bel, obs, slots, base, include_self, me) -> None:
    """Fill ``base`` with the located depth at the moment of each ask.

    "Located" has to mean *known to the propagator*, not merely *publicly
    transferred*. A first implementation counted only cards that had visibly
    changed hands, which silently omitted every card the propagator had DEDUCED
    to be someone's - the large majority - so depths came out far too small and
    the model was much worse than the initial-deal version on both instruments
    (posterior NLL 2.14 against 1.35, and -1.54 sets per deal-pair in play).
    That was a bug, not a finding.

    The correct bookkeeping: a card's holder at time t is its deduced initial
    owner unless a successful ask moved it before t, and a card the propagator
    has NOT pinned is exactly a card the sampler will draw, so it must not be
    counted here.
    """
    from fish.engine import ClaimEvent

    n_hs = len(obs.set_winner)
    holder: dict[int, int] = {}
    for c in range(bel.n):
        cand = bel.candidates[c]
        if cand & (cand - 1) == 0:
            holder[c] = cand.bit_length() - 1
    held = [[0] * n_hs for _ in range(NUM_PLAYERS)]
    for c, o in holder.items():
        held[o][c // 6] += 1
    sums = {k: 0 for k in slots}
    hits = {k: 0 for k in slots}
    for ev in obs.history:
        if isinstance(ev, AskEvent):
            hs = half_suit_of(ev.card)
            if (include_self or ev.asker != me) and (ev.asker, hs) in slots:
                key = (ev.asker, hs)
                sums[key] += held[ev.asker][hs]
                hits[key] += 1
            if ev.success:
                prev = holder.get(ev.card)
                if prev is not None:
                    held[prev][hs] -= 1
                holder[ev.card] = ev.asker
                held[ev.asker][hs] += 1
        elif isinstance(ev, ClaimEvent):
            hs = ev.half_suit
            for p in range(NUM_PLAYERS):
                held[p][hs] = 0
            for i in range(6):
                holder.pop(hs * 6 + i, None)
    for key, i in slots.items():
        base[i] = int(round(sums[key] / hits[key])) if hits[key] else 0
