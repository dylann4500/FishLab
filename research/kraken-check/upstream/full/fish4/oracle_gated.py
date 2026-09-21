"""A CHEATING agent that cheats in only ONE of its two decisions.

READ THIS FIRST. Like `fish4/oracle.py`, everything here is handed the true
deal. Nothing it produces is a strength measurement, nothing belongs in a
ladder beside an honest engine, and every figure carries the word *ceiling*
wherever it is written down.

WHY THIS EXISTS
---------------
`prereg/information_ceiling_split.md` measured that perfect knowledge of a
TEAMMATE's cards is worth **+3.41 sets/game** [+3.16, +3.66], against +1.31 for
an opponent's. That is the largest unclaimed number in the project, and three
separate attempts to reach any of it through better inference have now returned
nothing:

  * `prereg/gamma_split.md`   -- believe the teammate model harder: refuted
  * the at-ask-time covariate -- better posterior, worth nothing in play
  * `prereg/convention_duel.md` -- an actual message on an actual channel:
    the belief improved and replicated, and the duel came back at -0.002

Three nulls in a row against one large ceiling is not bad luck; it is evidence
that the ceiling is not measuring what everyone assumed. The hypothesis this
module tests:

    THE +3.41 IS NOT A CARD-READING EFFECT. A seat handed its teammates' cards
    does not ASK better. It DECLARES -- at moments it would otherwise never
    dare, and with splits it could otherwise never name.

If that is right, no amount of belief accuracy reaches the number, because the
belief was never the binding constraint. The lever is the declaration policy.

HOW THE CHEAT IS GATED
----------------------
`OracleBot` pins the belief to the truth and lets *everything* downstream
follow, so its cheat reaches both decisions at once and cannot say which one
carries the value. This keeps two beliefs side by side -- one honest, one
collapsed -- and routes each channel to one of them through the identity hooks
in `FishBot4`:

    declare   the claim machinery sees the truth; the asks are honest
    ask       the asks see the truth; the claim machinery is honest
    both      both see the truth -- i.e. OracleBot(side="team"), carried as a
              replication so the decomposition is anchored to a known figure

The cut is `ClaimEvaluator` + `certain_claim` + the tablebase on the
declaration side, everything else on the ask side. The tablebase is on the
declaration side because when it fires it usually ends the half-suit by
claiming.

WHAT THE ARMS DO AND DO NOT SUM TO
----------------------------------
Not a partition, and the report has to say so. The two decisions interact: a
different ask reaches a different position, so the declaration the other arm
faces is not the same one. `both` is carried precisely so that
`declare + ask == both` is visibly a question rather than an assumption.
"""

from __future__ import annotations

from typing import Optional

from fish.beliefs import BeliefState
from fish.cards import team_of
from fish.observation import Observation

from .agent4 import DecisionContext, FishBot4
from .posterior import Posterior


class GatedOracleBot(FishBot4):
    """The champion, with the truth substituted into one channel only."""

    MODES = ("declare", "ask", "both")

    def __init__(self, mode: str = "declare", side: str = "team", **kwargs):
        super().__init__(**kwargs)
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if side not in ("all", "team", "opp"):
            raise ValueError(f"bad side {side!r}")
        self.mode = mode
        self.side = side
        self._owners: Optional[list[int]] = None
        #: The second belief. Which of the two is the cheat depends on `mode`:
        #: in "ask" mode `self.bel` is collapsed and this one stays honest, so
        #: that the ask channel -- which reads `self.bel` throughout -- is the
        #: one that sees truth without any of its call sites being rerouted.
        self._other: Optional[BeliefState] = None
        self._collapsed = False
        self.pinned_by_cheat = 0
        self.pinned_first = None
        self.decisions = 0

    # -- the cheat ------------------------------------------------------------

    def see_deal(self, owners: list[int]) -> None:
        self._owners = list(owners)

    def begin_game(self, player: int, rules, seed: int) -> None:
        super().begin_game(player, rules, seed)
        self._other = BeliefState(rules, observer=player)
        self._collapsed = False
        self.pinned_by_cheat = 0
        self.pinned_first = None
        self.decisions = 0

    def _cheat_bel(self) -> BeliefState:
        """Whichever of the two beliefs is the one being fed the truth."""
        return self.bel if self.mode == "ask" else self._other

    def _honest_bel(self) -> BeliefState:
        return self._other if self.mode == "ask" else self.bel

    def _collapse(self, obs: Observation) -> None:
        """Pin every card of the chosen side to whoever was actually dealt it."""
        bel = self._cheat_bel()
        if self.mode == "both":
            targets = (self.bel, self._other)
        else:
            targets = (bel,)
        mine = team_of(self.player)
        want = self.side == "team"
        # In "both" mode the same cards go into both beliefs. Count them once:
        # a cheat that reports twice the cards it was told makes every
        # per-pinned-card figure wrong by a factor of two.
        first = targets[0]
        for b in targets:
            for card in range(len(self._owners)):
                if b.is_pinned(card):
                    continue
                if self.side != "all" and \
                        (team_of(self._owners[card]) == mine) != want:
                    continue
                # raises if the honest belief had already excluded the truth,
                # which would be an inference bug rather than a cheat failure
                b._pin(card, self._owners[card])
                if b is first:
                    self.pinned_by_cheat += 1
        if self.pinned_first is None:
            self.pinned_first = self.pinned_by_cheat

    # -- routing --------------------------------------------------------------

    def _claim_ctx(self, ctx: DecisionContext) -> DecisionContext:
        """Score splits with whichever belief this arm gives the claim channel.

        In "ask" mode that is the honest one, and it needs its own posterior:
        the claim evaluator reads marginals, so handing it the ask channel's
        context would leak the whole cheat through the one hook that exists to
        stop exactly that.
        """
        if self.mode == "both":
            return ctx
        # ALWAYS `_other`, in both modes, and that is the whole trick rather
        # than a missing branch. `self.bel` is the ask channel's belief by
        # construction -- every ask call site reads it and none of them is
        # rerouted -- so whichever belief the ask channel is NOT using is the
        # one the claim channel gets. In "declare" mode `_other` is the
        # collapsed one; in "ask" mode it is the honest one. One assignment
        # inverts the experiment.
        bel = self._other
        post = Posterior(bel, self.rng, n_draws=self.n_draws,
                         n_worlds=self.n_worlds, mode=self.infer_mode,
                         obs=ctx.obs, gamma=self.opponent_gamma,
                         depth_mode=self.depth_mode,
                         count_mode=self.count_mode,
                         opp_lambda=self.opp_lambda,
                         gamma_schedule=self.gamma_schedule,
                         sis_tilt=self.sis_tilt,
                         silence_delta=self.silence_delta)
        return DecisionContext(ctx.obs, bel, post)

    def _claim_bel(self) -> BeliefState:
        if self.mode == "both":
            return self.bel
        return self._other          # see _claim_ctx for why this is unbranched

    # -- policy ---------------------------------------------------------------

    def act(self, obs: Observation):
        if self._owners is None:
            raise RuntimeError(
                "GatedOracleBot.act called without see_deal: this agent is a "
                "deliberate cheat and refuses to run as an honest one")
        self.bel.update(obs)
        self._other.update(obs)
        self._collapse(obs)
        self.decisions += 1
        return super().act(obs)

    def certain_claim(self, obs: Observation):
        # keep the second belief in step: the base class only updates self.bel
        if self._other is not None:
            self._other.update(obs)
            if self._owners is not None:
                self._collapse(obs)
        return super().certain_claim(obs)


def gated_spec(mode: str, **kwargs):
    return ("oracle_gated", dict(mode=mode, **kwargs))
