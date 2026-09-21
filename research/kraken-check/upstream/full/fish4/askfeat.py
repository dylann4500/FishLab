"""Ask scoring for FishBot v0.4: every strategic hypothesis as one term.

v0.3 scored an ask as ``P(success) + 0.06*own_depth`` and later added two
hand-guessed terms (turn-risk and scarcity) that together were worth +1.41 sets
per duplicate deal-pair. Its own conclusion was that the *objective*, not the
depth of search, is what binds in this game. This module takes that seriously:
it defines a wider basis of terms, computes them all from the posterior in one
vectorised pass, and exposes each as an ablatable weight so that measurement,
not intuition, decides which survive.

Two design changes from v0.3 matter for cost as well as quality.

**Everything is computed from marginals, not from per-ask loops over sampled
worlds.** v0.3 recomputed a per-half-suit team-share statistic inside the loop
over candidate asks; profiling showed that single quantity was 47% of the
champion's runtime (13.4s of 28.3s). Here all half-suit and per-target
aggregates are computed once per decision and indexed.

**Two of the new terms are attempts to say what v0.3's proxies were groping
at.** Its turn-risk term used the target's hand *size* as a proxy for "how
dangerous is it to hand this opponent the turn". The quantity that actually
matters is how much they can immediately take from us, which we can compute:
cards of ours whose location is already PUBLIC and that sit in a half-suit the
opponent can legally ask in. Similarly its scarcity term used the team's
expected share of a half-suit; what actually scores is the probability the team
ends up able to claim it, which is a different function of the same marginals.

INFORMATION BOUNDARY: every quantity here is derived from the Observation plus
the acting seat's own BeliefState (public events + own hand). ``public_loc`` in
particular is set only from public events, so "what the opponents can see" is
genuinely public knowledge and not a leak.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np

from fish.beliefs import RESOLVED
from fish.cards import (NUM_PLAYERS, cards_per_player, half_suit_mask,
                        num_half_suits, team_of)

# Names of the ablatable terms, in a fixed order. ``score = sum_i w_i * f_i``
# with the success probability always carrying weight 1.0 by convention, so the
# other weights are read as "in units of probability of success".
TERM_NAMES = (
    "suit",        # own depth in the half-suit (v0.3 baseline term)
    "turn",        # hand-size turn risk on failure (v0.3 win, +0.56)
    "scarce",      # team share of the half-suit (v0.3 win, +0.65)
    "reveal",      # cost of exposing a half-suit not previously shown
    "deplete",     # bonus for draining a short-handed opponent (v0.3 null)
    "expose",      # NEW: what the target could immediately take back from us
    "claim",       # NEW: progress toward a half-suit we can actually claim
    "info",        # NEW: expected information gained about the half-suit
    "certain",     # NEW: explicit bonus for a provably certain steal
    "concent",     # v2: the CHANGE in team concentration this ask causes
    "signal",      # NEW: the BENEFIT side of revealing a holding to teammates
    "locate",      # NEW: location-uncertainty this ask removes from a half-suit
                   #      our team is on course to declare
    "reach",       # NEW: the entry point this ask spends -- P(the half-suit
                   #      stops being askable by us once we take this card)
)

#: Definition version of each term, bumped whenever a term's FORMULA changes
#: while its name stays the same.
#:
#: fish4/learn/fit.py already refused a harvest whose TERM_NAMES differed from
#: the current ones, because adding or reordering a term silently attributes
#: one term's effect to another. It could not see the other half of the same
#: hazard: a name that stays put while the column beneath it changes meaning.
#: That is not hypothetical -- ``claim`` was corrected from a product over all
#: six cards of the half-suit to a product over the OTHER five, which flips it
#: from scoring zero on a certain steal to scoring highest there. Any fit over
#: a harvest predating that correction would attribute a weight to a feature
#: the engine no longer computes.
#:
#: A harvest with no versions recorded is treated as all-1s, which is exactly
#: what it is.
TERM_VERSIONS = {
    "suit": 1, "turn": 1, "scarce": 1, "reveal": 1, "deplete": 1,
    "expose": 1,
    "claim": 2,        # v2: excludes the asked card from the product
    "info": 1, "certain": 1,
    "concent": 2,      # v2: expected change caused, not level observed
    "signal": 1,
    "locate": 1,
    "reach": 2,       # v2: no 1/n_askable divisor; see the comment in the
                      #     feature block and results/term_bite_reach.json
}
assert tuple(TERM_VERSIONS) == TERM_NAMES, "TERM_VERSIONS must mirror TERM_NAMES"


def term_versions() -> list:
    """Current definition versions, aligned to :data:`TERM_NAMES`."""
    return [TERM_VERSIONS[n] for n in TERM_NAMES]


def stale_terms(recorded) -> list:
    """Names whose stored definition version is not the current one.

    ``recorded`` is a list aligned to TERM_NAMES, or None for a harvest taken
    before versions were written -- which means every term was at version 1.
    """
    have = list(recorded) if recorded else [1] * len(TERM_NAMES)
    if len(have) != len(TERM_NAMES):
        raise ValueError(f"recorded {len(have)} term versions for "
                         f"{len(TERM_NAMES)} terms")
    return [n for n, v in zip(TERM_NAMES, have) if v != TERM_VERSIONS[n]]


@dataclass(frozen=True)
class AskWeights:
    """Weights for the ask objective. All default to the v0.3 champion."""
    suit: float = 0.06
    turn: float = 0.6
    scarce: float = 0.2
    reveal: float = 0.0
    deplete: float = 0.0
    expose: float = 0.0
    claim: float = 0.0
    info: float = 0.0
    certain: float = 0.0
    concent: float = 0.0
    signal: float = 0.0
    locate: float = 0.0
    reach: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def as_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in TERM_NAMES], dtype=np.float64)

    @classmethod
    def from_vector(cls, v) -> "AskWeights":
        return cls(**{n: float(x) for n, x in zip(TERM_NAMES, v)})

    @classmethod
    def zeros(cls) -> "AskWeights":
        return cls(**{n: 0.0 for n in TERM_NAMES})


class DecisionContext:
    """Per-decision aggregates shared by every candidate ask.

    Built once per turn; the per-ask feature rows are then pure indexing.
    """

    __slots__ = ("obs", "bel", "post", "M", "me", "my_team", "n_hs", "per",
                 "mine", "theirs",
                 "hs_live", "my_depth", "team_exp", "opp_exp", "player_exp",
                 "revealed", "turn_risk", "exposure", "hs_entropy",
                 "team_concentration", "p_team_all", "p_team_card", "avg_live")

    def __init__(self, obs, bel, post):
        self.obs = obs
        self.bel = bel
        self.post = post
        self.M = post.marginals()
        self.me = obs.player
        self.my_team = team_of(self.me)
        self.n_hs = num_half_suits(obs.rules.variant)
        self.per = cards_per_player(obs.rules.variant)
        self._compute()

    # -- aggregates ---------------------------------------------------------

    def _compute(self) -> None:
        obs, M = self.obs, self.M
        n_hs, me, my_team = self.n_hs, self.me, self.my_team
        mine = [p for p in range(NUM_PLAYERS) if team_of(p) == my_team]
        theirs = [p for p in range(NUM_PLAYERS) if team_of(p) != my_team]
        # Kept because the per-ask concentration feature needs the roster and
        # rebuilding it once per candidate ask is the whole cost of the term.
        self.mine, self.theirs = mine, theirs

        self.hs_live = np.array([w is None for w in obs.set_winner], dtype=bool)
        self.my_depth = np.zeros(n_hs)
        self.team_exp = np.zeros(n_hs)
        self.opp_exp = np.zeros(n_hs)
        self.player_exp = np.zeros((n_hs, NUM_PLAYERS))
        self.hs_entropy = np.zeros(n_hs)
        self.team_concentration = np.zeros(n_hs)
        self.p_team_all = np.zeros(n_hs)
        #: Per-card P(this card sits with our team), by half-suit.
        #: Kept because the claim feature needs the product over the
        #: OTHER five cards, and dividing the six-card product by one
        #: factor is exactly wrong when that factor is zero -- which
        #: is precisely the case the feature exists to reward.
        self.p_team_card = np.zeros((n_hs, 6))

        for hs in range(n_hs):
            if not self.hs_live[hs]:
                continue
            lo = hs * 6
            block = M[lo:lo + 6]                       # (6 cards, 6 players)
            self.my_depth[hs] = (obs.hand & half_suit_mask(hs)).bit_count()
            self.player_exp[hs] = block.sum(axis=0)
            self.team_exp[hs] = self.player_exp[hs, mine].sum()
            self.opp_exp[hs] = self.player_exp[hs, theirs].sum()
            # Shannon entropy over holders, averaged across the six cards:
            # how much is still unknown about where this half-suit lives.
            with np.errstate(divide="ignore", invalid="ignore"):
                lg = np.where(block > 0, np.log(np.maximum(block, 1e-12)), 0.0)
            self.hs_entropy[hs] = float(-(block * lg).sum() / 6.0)
            # Concentration: a set spread over three teammates is harder to
            # declare correctly than one held by a single player, so a team
            # holding is worth more when it is concentrated.
            te = self.player_exp[hs, mine]
            s = te.sum()
            self.team_concentration[hs] = float(te.max() / s) if s > 0 else 0.0
            # Probability every card of the half-suit sits with our team, under
            # an independence approximation across cards. Exact joint queries
            # are available from the posterior but cost a DP each; this is the
            # cheap screen and is only ever used as a *relative* term.
            pteam = block[:, mine].sum(axis=1)
            self.p_team_all[hs] = float(np.prod(pteam))
            self.p_team_card[hs] = pteam

        counts = obs.hand_counts
        live_counts = [c for c in counts if c > 0]
        self.avg_live = (sum(live_counts) / len(live_counts)) if live_counts else 0.0
        self.turn_risk = np.array(
            [-(counts[p] - self.avg_live) / max(1.0, self.per)
             for p in range(NUM_PLAYERS)])
        self.revealed = self._revealed_suits()
        self.exposure = self._exposure()

    def _revealed_suits(self) -> np.ndarray:
        """Half-suits in which WE have already publicly shown a holding."""
        from fish.engine import AskEvent
        out = np.zeros(self.n_hs, dtype=bool)
        me = self.me
        for ev in self.obs.history:
            if isinstance(ev, AskEvent):
                if ev.asker == me or (ev.success and ev.target == me):
                    out[ev.card // 6] = True
        return out

    def _exposure(self) -> np.ndarray:
        """For each opponent, how much they could take from us on their turn.

        v0.3's turn-risk used hand size as a proxy for danger. The sharper
        quantity is how many of OUR team's cards are already publicly located
        and sit in a half-suit that opponent can legally ask in, because those
        are cards they can take with certainty the moment they get the turn.
        Both parts are public: ``public_loc`` is set only from public events,
        and the opponent's ability to ask in a half-suit is estimated from our
        posterior over their holdings.
        """
        bel, M = self.bel, self.M
        out = np.zeros(NUM_PLAYERS)
        my_team = self.my_team
        # p_can_ask[p][hs] = P(player p holds at least one card of hs)
        for hs in range(self.n_hs):
            if not self.hs_live[hs]:
                continue
            lo = hs * 6
            block = M[lo:lo + 6]
            # public cards of this half-suit sitting with our team
            n_public_ours = 0
            for i in range(6):
                loc = bel.public_loc[lo + i]
                if loc is not None and loc != RESOLVED and team_of(loc) == my_team:
                    n_public_ours += 1
            if not n_public_ours:
                continue
            for p in range(NUM_PLAYERS):
                if team_of(p) == my_team:
                    continue
                # P(p holds >= 1 card of hs), independence approximation
                p_none = float(np.prod(1.0 - np.clip(block[:, p], 0.0, 1.0)))
                out[p] += n_public_ours * (1.0 - p_none)
        return out / 6.0


def ask_feature_matrix(ctx: DecisionContext, asks) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(p_success, features)`` for every candidate ask.

    ``features`` has one row per ask and one column per name in TERM_NAMES, in
    that order. The success probability is returned separately because it
    always carries weight 1.
    """
    n = len(asks)
    p = np.empty(n)
    F = np.zeros((n, len(TERM_NAMES)))
    M = ctx.M
    per = ctx.per
    counts = ctx.obs.hand_counts
    # -- `reach`: the entry point an ask spends ------------------------------
    #
    # results/forced_locus.json, 15,929 decisions with the game stage
    # controlled for: a seat five of its own decisions away from a `gate` or
    # `forced` declaration has 6.9 FEWER live asks than a seat at the same
    # cards-left, and 4.0 fewer at eight decisions. Those two paths carry 62 of
    # 63 wrong declarations. And the seat about to be stuck holds 0.6 to 0.9
    # MORE cards than the control: it is not short of cards, it is short of
    # places to reach.
    #
    # Every successful ask takes a card from an OPPONENT, so every success
    # spends a little of the asker's own future askability. Taking cards is how
    # the game is won and how a seat strands itself, and the basis prices only
    # the first half of that -- `concent` rewards concentrating the team's
    # holding and `scarce` rewards team share, both of which consume entry
    # points, while nothing charges for them.
    #
    # opp_mass[c] is P(an opponent currently holds c), which is 0 for a card in
    # our own hand and for a resolved one, so a product over it needs no
    # special cases for either.
    opp_mass = M[:, ctx.theirs].sum(axis=1)
    for i, a in enumerate(asks):
        hs = a.card // 6
        t = a.target
        pi = float(M[a.card, t])
        p[i] = pi
        fail = 1.0 - pi
        F[i, 0] = ctx.my_depth[hs]
        F[i, 1] = fail * ctx.turn_risk[t]
        F[i, 2] = (ctx.team_exp[hs] / 6.0 - 0.5) * 2.0
        F[i, 3] = -1.0 if not ctx.revealed[hs] else 0.0
        F[i, 4] = pi * (1.0 - counts[t] / per)
        F[i, 5] = -fail * ctx.exposure[t]
        # Claim progress: how much closer a success brings this half-suit to
        # being one WE can declare. Gaining a card puts it in our own hand,
        # which is the only perfectly localised place it can be, so the gain is
        # largest when THE REST of the suit is already ours.
        #
        # "The rest" is the whole point and was the bug. This used
        # ctx.p_team_all[hs], the product over all SIX cards -- including the
        # one being asked for, which by construction we do not hold. For a
        # provably-certain steal that card's factor is exactly 0, so the
        # product is 0 and the feature is 0: it scored zero on precisely the
        # asks it was written to reward, and rose as the steal became less
        # certain. Excluding the asked card is what the comment above always
        # described.
        others = ctx.p_team_card[hs]
        rest = 1.0
        for k in range(6):
            if hs * 6 + k != a.card:
                rest *= float(others[k])
        F[i, 6] = pi * rest ** (1.0 / 5.0)
        # Information: a maximally uncertain ask (p ~ 0.5) in a maximally
        # uncertain half-suit resolves the most.
        F[i, 7] = 4.0 * pi * fail * ctx.hs_entropy[hs]
        F[i, 8] = 1.0 if pi >= 0.999999 else 0.0
        # Concentration, v2: the change this ask would cause, not the level it
        # observes.
        #
        # v1 was `ctx.team_concentration[hs]` -- one number per half-suit,
        # identical for every candidate ask in it and independent of the target
        # and of who would end up holding the card. A term that takes the same
        # value on every ask in a half-suit cannot express a preference BETWEEN
        # asks; it can only tilt the choice of half-suit. Worse, its sign is
        # wrong in the case the term exists for: when the concentration sits
        # with a TEAMMATE, my taking a card breaks it up, and v1 scored that
        # ask highest precisely because the half-suit was concentrated.
        #
        # This is the same defect `claim` had at v1 -- a formula that cannot
        # reward what its own comment describes -- and gets the same remedy:
        # corrected in place, TERM_VERSIONS bumped, every harvest fitted
        # against the old column marked stale by stale_terms().
        #
        # Why it is worth correcting rather than deleting: 0.1676 of our 0.1759
        # wrong declarations a game are allocation class, our own team holding
        # all six and naming the wrong split, against 0.0083 ownership errors
        # (results/margin_decomposition.json). A holding in one hand needs no
        # split named at all. Concentration is the only term in the basis that
        # points at the dominant error class.
        #
        # On success the card moves from the target to me: my expectation gains
        # one, every teammate's probability mass on that card is discharged,
        # and the team total moves by the same amount. Scaled by pi, so the
        # feature is the EXPECTED change.
        e = ctx.player_exp[hs]
        t = float(ctx.team_exp[hs])
        if t > 1e-9:
            cur = float(max(e[q] for q in ctx.mine)) / t
            spent = float(sum(M[a.card, q] for q in ctx.mine))
            new_t = t + 1.0 - spent
            if new_t > 1e-9:
                best = float(e[ctx.me]) + 1.0 - float(M[a.card, ctx.me])
                for q in ctx.mine:
                    if q != ctx.me:
                        best = max(best, float(e[q]) - float(M[a.card, q]))
                F[i, 9] = pi * (best / new_t - cur)
        # Signalling. An ask publicly certifies that we hold a card of this
        # half-suit - the only legal communication channel in Literature, and
        # one that is simultaneously read by the opponents. v0.3 modelled only
        # the cost of that (its "reveal" term, worth a marginal +0.29). The
        # benefit is that a teammate learns where a card of a suit OUR team is
        # winning must be, which is precisely the localisation that lets a set
        # be declared. So the sign should depend on who gains more, and the
        # natural proxy for that is our team's share of the half-suit. Both
        # terms are carried with independent weights so the data, not the
        # story, decides.
        if not ctx.revealed[hs]:
            F[i, 10] = (ctx.team_exp[hs] / 6.0 - 0.5) * 2.0
        # Location. THE ONE TERM THAT PRICES AN ASK BY THE DECLARATION IT
        # ENABLES, which is why it exists.
        #
        # results/declaration_timing.json decomposed the +3.41 sets a game that
        # perfect knowledge of a teammate's cards is worth, by routing the same
        # cheat to one decision at a time. Neither channel carries it: the
        # declaration channel alone is worth +1.08, the ask channel alone
        # +0.76, and 46% of the ceiling -- 1.57 sets a game -- lives in NEITHER
        # ALONE. It is interaction. Four separate attempts had each improved
        # one channel and returned nothing, which is what reaching for a third
        # of a prize looks like.
        #
        # The only intervention that can reach an interaction term is one that
        # couples the two decisions, and this is the cheapest such coupling:
        # score an ask by what it will let the team DECLARE later.
        #
        # The quantity is fixed by the mediator finding, not chosen: what a
        # declaration risks is how many of the half-suit's six cards have never
        # been publicly LOCATED, not how many the declarer holds. A successful
        # ask locates exactly one card, permanently, for the whole table --
        # including our partners. So the value of an ask, to a future
        # declaration, is the share of that half-suit's remaining location
        # uncertainty it removes.
        #
        #     pi          it only locates anything if it lands
        #     1 / u       the fraction of the remaining uncertainty removed:
        #                 going from two unlocated to one is worth more than
        #                 six to five, which is the shape "risk tracks
        #                 unlocated count" implies
        #     rest^(1/5)  weighted by our team owning the REST of the suit,
        #                 because locating a card of a half-suit we will never
        #                 declare buys nothing. Same per-card geometric mean
        #                 the `claim` term uses, and excluding the asked card
        #                 for the same reason: on a provably certain steal its
        #                 own factor is 0, and a term that scores zero exactly
        #                 where it should score highest is the bug `claim` and
        #                 `concent` both already had.
        #
        # Zero when the asked card is ALREADY publicly located: the ask then
        # adds no location and the term must not pay for one. That is the case
        # this feature would otherwise reward twice, since a located card
        # sitting with the target is a certain steal and `certain` already
        # prices it.
        if ctx.bel.public_loc[a.card] is None:
            u = 0
            for k in range(6):
                if ctx.bel.public_loc[hs * 6 + k] is None:
                    u += 1
            if u:
                F[i, 11] = pi * (rest ** (1.0 / 5.0)) / u
        # `reach`, and it is a COST: negative, so a positive weight penalises.
        #
        #     -pi * P(no other card of this half-suit is with an opponent)
        #
        #   pi        it spends nothing if it does not land.
        #   the product  P(h stops being askable by us). Taking the last card
        #             an opponent held closes the half-suit as an entry point;
        #             taking one of four leaves it wide open. The asked card is
        #             excluded because we are about to hold it either way --
        #             including it would zero the term exactly on a certain
        #             steal, the bug `claim`, `concent` and `locate` all had.
        #
        # v1 DIVIDED BY the number of half-suits we could still ask in, and
        # results/term_bite_reach.json says that shape cannot reach the
        # decision at all: a top-ask change in 1.6% of positions at w = 0.3 and
        # only 7.6% at w = 1.2. That is `locate`'s 1/u again, and `locate` is a
        # measured null. v2 drops it; the weight sets the exchange rate instead
        # of a divisor chosen by me.
        #
        # READ THIS BEFORE RAISING THE WEIGHT. The term is REFUTED at a
        # positive weight and the screen is in prereg/reach_term.md. `keep` is
        # largest exactly when our team already holds the rest of the
        # half-suit, so a positive weight penalises the ask that COMPLETES a
        # set -- the ask that banks one. Measured over 480 games at w = 0.8:
        # voluntary declarations fall 33%, the displaced ones reappear on the
        # `gate` path (up 165%, 22.4% wrong), total declarations fall 16.3%,
        # wrong declarations per game rise from 0.1313 to 0.2167, and the
        # margin is -1.67 sets a game. Both pre-registered screen rules fired
        # and no duel was run.
        #
        # The NEGATIVE weight is the one that works mechanically -- at -0.4 it
        # cuts gate+forced declarations 19% and wrong declarations 24% while
        # voluntary holds -- and it still does not pay: margin -0.075. That
        # sign was chosen after reading the screen, so it is a diagnostic and
        # not a registered arm, and nothing was confirmed on it. The finding is
        # that the trajectory is steerable and steering it is not worth
        # anything, which is why this ships at 0.0 and stays there.
        #
        keep = 1.0
        for k in range(6):
            d = hs * 6 + k
            if d != a.card:
                keep *= 1.0 - float(opp_mass[d])
        F[i, 12] = -pi * keep
    return p, F


def score_asks(ctx: DecisionContext, asks, weights: AskWeights) -> tuple:
    """``(scores, p_success)`` for every candidate ask."""
    p, F = ask_feature_matrix(ctx, asks)
    return p + F @ weights.as_vector(), p
