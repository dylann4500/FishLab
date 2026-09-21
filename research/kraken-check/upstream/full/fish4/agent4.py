"""FishBot v0.4: exact-posterior play with a wide, ablatable ask objective.

Relationship to v0.3
--------------------
v0.3's strongest policy sampled 32 worlds from a *biased* sampler, scored asks
as ``P(success) + 0.06*depth + 0.6*turn_risk + 0.2*scarcity``, and claimed at a
fixed 0.97 confidence. Its own diagnosis was that the objective, not the depth
of search, was the binding constraint, and that belief precision had not
saturated (32 -> 96 samples was still worth +0.54 sets per deal-pair).

v0.4 changes three things and holds everything else fixed so the changes are
attributable:

1. **Inference.** Marginals come from exact combinatorial counting over the
   candidate-mask and quota system rather than from a heuristic sampler, with
   the OR constraints handled by a correction whose error is measured rather
   than assumed. See ``counting.py`` and ``posterior.py``.
2. **Objective.** Ten ablatable terms instead of three, all computed from the
   marginals in one vectorised pass. See ``askfeat.py``.
3. **Claiming.** The declared distribution is the exact posterior MAP, not the
   mode of 32 samples, and forced claims maximise expected sets rather than
   confidence. See ``claim4.py``.

Everything an acting policy touches comes from the Observation plus this seat's
own BeliefState, which is derived from public events and this seat's own hand.
``GameState`` is never visible here.
"""

from __future__ import annotations

import random
from typing import Optional

from fish.agents.base import Agent
from .endgame_ii import ExactEndgameMixin
from .tablebase4 import Tablebase4Mixin
from . import trace as _tr
from fish.beliefs import BeliefContradiction, BeliefState
from fish.engine import Action
from fish.observation import Observation

from dataclasses import replace

from .askfeat import AskWeights, DecisionContext, score_asks
from .claim4 import ClaimConfig, ClaimEvaluator, choose_pass
from .hsvalue import HalfSuitValue, score_asks_by_value
from .posterior import Posterior, PosteriorStats

#: Value models are small and immutable; six agents per game times thousands of
#: games would otherwise re-parse the same JSON constantly.
_VALUE_CACHE: dict = {}


def _load_value(path):
    if path is None:
        return None
    m = _VALUE_CACHE.get(path)
    if m is None:
        m = _VALUE_CACHE[path] = HalfSuitValue.load(path)
    return m


#: Optional instrumentation hook, ``None`` in every shipped path. Called as
#: ``(bot, ctx, asks, scores)`` with the ask ranking complete and not yet
#: sorted, so a diagnostic can see the objective exactly as the policy does.
#:
#: It exists because the most useful thing to come out of the `locate` null was
#: its size / overlap / bite diagnosis -- how big the new term is against the
#: objective's own scale, how much it merely re-states, and how often it
#: actually changes the top-ranked ask. That was measured against a hand-copied
#: fragment of act(), which is a duplicate that rots the first time this
#: function changes. Any weight fitted or refuted against a stale copy of the
#: objective is measuring the copy.
#:
#: The cost when unset is one identity comparison per decision.
_SCORE_RECORDER = None


class FishBot4(ExactEndgameMixin, Tablebase4Mixin, Agent):
    """The v0.4 policy. Every strategic choice is a constructor argument."""

    name = "fishbot4"

    def __init__(self,
                 # -- inference
                 n_worlds: int = 32,
                 n_draws: int = 160,
                 infer_mode: str = "auto",
                 opponent_gamma: float = 0.0,
                 gamma_team: float | None = None,
                 convention_beta: float = 0.0,
                 convention_q: float = 0.0,
                 convention_aim: bool = False,
                 convention_book: str = "depth",
                 convention_max_cost: float = 0.0,
                 depth_mode: str = "initial",
                 count_mode: str = "linear",
                 opp_lambda: float = 0.0,
                 gamma_schedule: float = 0.0,
                 sis_tilt: float = 0.0,
                 # -- ask objective
                 w_suit: float = 0.06,
                 w_turn: float = 0.6,
                 w_scarce: float = 0.2,
                 w_reveal: float = 0.0,
                 w_deplete: float = 0.0,
                 w_expose: float = 0.0,
                 w_claim: float = 0.0,
                 w_info: float = 0.0,
                 w_certain: float = 0.0,
                 w_concent: float = 0.0,
                 w_signal: float = 0.0,
                 w_locate: float = 0.0,
                 #: Charge an ask for the entry point it spends. See
                 #: fish4/askfeat.py and results/forced_locus.json.
                 w_reach: float = 0.0,
                 # -- learned half-suit value objective
                 objective: str = "linear",
                 hsvalue_path: str = None,
                 w_value: float = 0.0,
                 value_turn: float = 0.0,
                 value_expose: float = 0.0,
                 value_keep: float = 0.0,
                 #: never make an ask that provably cannot land while one
                 #: that can is available. See the note at the use site.
                 avoid_doomed_asks: bool = False,
                 # -- claiming
                 claim_threshold: float = 0.97,
                 #: refuse declarations no complete consistent deal allows
                 claim_feasibility: bool = False,
                 claim_exact: bool = True,
                 claim_exact_candidates: int = 3,
                 #: The bar the doomed-ask claim gate uses (see the long note
                 #: at its use site). 0.5 is the incumbent, hard-coded since
                 #: v0.3 and never measured.
                 claim_stuck_threshold: float = 0.5,
                 #: ...but the raised bar applies only where p_team is at or
                 #: above this. At 1.01 the test can never pass, so the gate
                 #: keeps its single 0.5 bar everywhere and the default is
                 #: bit-identical -- the same discipline as endgame_m=0 and
                 #: w_contest=0.0.
                 stuck_team_certain: float = 1.01,
                 #: search the FULL team space at the forced declaration when
                 #: at most this many half-suits are live. See
                 #: prereg/forced_exhaustive.md. 0 = never; bit-identical.
                 claim_forced_exhaustive: int = 0,
                 # -- adaptive style
                 w_retake: float = 0.0,
                 retake_window: int = 8,
                 retake_min_depth: int = 0,
                 w_behind: float = 0.0,
                 #: Scale the tempo term down when the turn it is protecting
                 #: is worth nothing. See prereg/tempo_regime.md: the paper
                 #: measured a turn at -0.043 +- 0.169 below p_best 0.25 and
                 #: +0.004 +- 0.143 in [0.25, 0.50), and the objective charges
                 #: the same 0.6*(1-p)*turn_risk at every one of them -- which
                 #: is 57% of all ask decisions. At 0.0 the test can never pass
                 #: and the champion is bit-identical.
                 turn_free_below: float = 0.0,
                 turn_free_scale: float = 0.0,
                 # -- endgame-only ask weights. The exact solver shows the ask
                 # objective is wrong in a specific way once few half-suits are
                 # live, and wrong in a way that is not the same as being wrong
                 # everywhere -- so the correction is applied there and nowhere
                 # else. Zero deltas leave the incumbent weights untouched at
                 # every m, which is what makes the default bit-identical to
                 # the champion rather than merely close to it.
                 endgame_m: int = 0,
                 endgame_d_info: float = 0.0,
                 endgame_d_certain: float = 0.0,
                 # -- belief-space lookahead
                 w_lookahead: float = 0.0,
                 lookahead_depth: int = 1,
                 lookahead_beam: int = 4,
                 lookahead_couple: bool = True,
                 #: Build the DECLARATION's posterior at a different
                 #: action-model weight than the ask objective's. An ask is
                 #: scored on the argmax and a declaration against a 0.97
                 #: threshold, so the two want different things from the same
                 #: model, and nothing in this engine had ever priced them
                 #: apart. 0.0 is off and the champion is bit-identical.
                 #: See prereg/claim_gamma.md and results/split_why.json.
                 claim_gamma: float = 0.0,
                 #: Price each edge of the possession chain by the
                 #: DECLARABILITY it creates, in banked-cards per half-suit
                 #: made nameable. 0.0 is the champion, exactly: the tree keeps
                 #: its inert-last-ply short-circuit and every number it has
                 #: ever produced. See fish4/lookahead.py, and
                 #: prereg/declarability_leaf.md for why the search rather than
                 #: another ask feature.
                 lookahead_declare: float = 0.0,
                 # -- endgame
                 use_tablebase: bool = True,
                 tablebase_max_half_suits: int = 2,
                 #: Play the m = 1 endgame from fish4.exact_ii's exact best
                 #: response instead of the heuristic. OFF by default: it
                 #: models every other seat as the champion, which is wrong
                 #: about a teammate that also runs it, and a best response is
                 #: not an equilibrium strategy. See fish4/endgame_ii.py.
                 exact_endgame: bool = False,
                 exact_endgame_max_support: int = 12,
                 exact_endgame_max_nodes: int = 50_000,
                 # -- misc
                 stall_window: int = 80,
                 smart_pass: bool = False,
                 signal_mode: str = "off",
                 signal_max_p: float = 0.15,
                 #: Do not signal a card this seat has already signalled this
                 #: game. The signal branch re-asks one card a mean of 40.8
                 #: times an episode to re-prove a fact already public, and
                 #: those turns come out of the seat's own stall window.
                 #: False is bit-identical. prereg/signal_no_repeat.md.
                 signal_no_repeat: bool = False,
                 #: Cap on signals a GAME; 0 is unlimited and bit-identical.
                 #: Each signal is a deliberately doomed ask and a doomed ask
                 #: loses the turn, so the mechanism pays -0.1695 of margin in
                 #: half-suits it never gets to declare for the +0.2625 of
                 #: opponent errors it buys. If the first few fires carry the
                 #: confusion, a budget keeps the second number and returns
                 #: some of the first. prereg/signal_budget.md.
                 signal_budget: int = 0,
                 #: Signed weight on adaptive.contest_bonus. Positive fights
                 #: in opponent-dominated ambiguous half-suits (Dylan's v0.7
                 #: carries the analogous term strongly positive); negative is
                 #: the off-limits reading (avoid them unless the ask is a
                 #: certain steal). 0.0 = incumbent, bit-identical.
                 w_contest: float = 0.0,
                 trace: bool = False,
                 #: Silence prior: down-weight sampled worlds in which a live
                 #: half-suit sits wholly within one team right now, because
                 #: a team that held it all and could place it would usually
                 #: have declared. 1.0 = off, bit-identical.
                 silence_delta: float = 1.0):
        super().__init__()
        self.n_worlds = n_worlds
        self.n_draws = n_draws
        self.infer_mode = infer_mode
        self.opponent_gamma = opponent_gamma
        #: Sharpness for our OWN side's asks. None -> one gamma for both
        #: sides, which is the incumbent and bit-identical to it.
        self.gamma_team = gamma_team
        #: Decoder weight: how sharply a partner's ask is read as
        #: carrying the agreed message. Inert at 0.
        self.convention_beta = convention_beta
        self.convention_q = convention_q
        self.convention_aim = bool(convention_aim)
        self.convention_book = convention_book
        #: Encoder gate: the most probability of success this seat
        #: will give up to send one. At 0 it never encodes, so the
        #: two halves are independently ablatable and the pair can
        #: be measured apart -- a decoder with no encoder is the
        #: control that says whether the DECODER alone is harmful.
        self.convention_max_cost = convention_max_cost
        self.depth_mode = depth_mode
        self.count_mode = count_mode
        self.opp_lambda = opp_lambda
        self.gamma_schedule = gamma_schedule
        self.sis_tilt = sis_tilt
        self.weights = AskWeights(
            suit=w_suit, turn=w_turn, scarce=w_scarce, reveal=w_reveal,
            deplete=w_deplete, expose=w_expose, claim=w_claim, info=w_info,
            certain=w_certain, concent=w_concent, signal=w_signal,
            locate=w_locate, reach=w_reach)
        self.objective = objective
        self.hsvalue_path = hsvalue_path
        self.w_value = w_value
        self.value_turn = value_turn
        self.value_expose = value_expose
        self.value_keep = value_keep
        # value_keep is read ONLY by the pure-value objective. In the blend
        # path the heuristic already carries P(success) at weight 1.0, so the
        # term is deliberately not applied there -- and a parameter that
        # silently does nothing is how an ablation gets attributed to the wrong
        # cause. Refuse the combination instead of ignoring half of it.
        if value_keep and objective != "value":
            raise ValueError(
                f"value_keep={value_keep} has no effect with objective="
                f"{objective!r}; it applies only to objective='value'")
        self.avoid_doomed_asks = bool(avoid_doomed_asks)
        self.claim_cfg = ClaimConfig(feasibility=bool(claim_feasibility),
                                     threshold=claim_threshold,
                                     exact_candidates=claim_exact_candidates,
                                     use_exact=claim_exact,
                                     forced_exhaustive=int(
                                         claim_forced_exhaustive))
        self.claim_stuck_threshold = float(claim_stuck_threshold)
        self.stuck_team_certain = float(stuck_team_certain)
        self.w_retake = w_retake
        self.retake_window = retake_window
        self.retake_min_depth = retake_min_depth
        self.w_behind = w_behind
        self.turn_free_below = float(turn_free_below)
        self.turn_free_scale = float(turn_free_scale)
        self.endgame_m = endgame_m
        self.endgame_d_info = endgame_d_info
        self.endgame_d_certain = endgame_d_certain
        self.w_lookahead = w_lookahead
        self.lookahead_depth = lookahead_depth
        self.lookahead_beam = lookahead_beam
        self.lookahead_couple = lookahead_couple
        self.lookahead_declare = lookahead_declare
        self.claim_gamma = claim_gamma
        #: How many extra posteriors the claim gamma has paid for. Reported so
        #: the cost of the gate is visible rather than assumed.
        self.claim_posteriors = 0
        self.use_tablebase = use_tablebase
        self.tablebase_max_half_suits = tablebase_max_half_suits
        self.exact_endgame = bool(exact_endgame)
        self.exact_endgame_max_support = int(exact_endgame_max_support)
        self.exact_endgame_max_nodes = int(exact_endgame_max_nodes)
        #: What the solver models the other five seats as. The champion, and
        #: named here rather than inlined so it is obvious that changing the
        #: agent's own configuration does NOT change the opponent model the
        #: endgame search assumes.
        self.exact_endgame_spec = ("fishbot4", {"opponent_gamma": 0.35})
        self.stall_window = stall_window
        self.smart_pass = smart_pass
        self.signal_mode = signal_mode
        self.signal_max_p = signal_max_p
        self.signal_no_repeat = signal_no_repeat
        self.signal_budget = int(signal_budget)
        self._signalled: set = set()
        self._signals = 0
        self.w_contest = float(w_contest)
        self.silence_delta = float(silence_delta)
        self.stats = PosteriorStats()
        #: Capture WHY each decision was made, for the site's explanation
        #: panels. Off by default and free when off: the builders in
        #: fish4/trace.py read arrays the policy has already computed and
        #: never touch the RNG, so a traced agent and an untraced one play
        #: bit-identical games from the same seed (tests4/test_trace.py).
        self.trace = bool(trace)
        #: The trace of the most recent act(), or None.
        self.last_trace = None
        self.bel: Optional[BeliefState] = None

    def _t(self, build, *args, **kw) -> None:
        """Record a trace, or do nothing at all.

        Deliberately takes the BUILDER rather than a built dict, so that with
        tracing off none of the formatting work happens either.
        """
        if self.trace:
            self.last_trace = build(*args, **kw)

    # -- lifecycle -----------------------------------------------------------

    def begin_game(self, player: int, rules, seed: int) -> None:
        super().begin_game(player, rules, seed)
        self.bel = BeliefState(rules, observer=player)
        # per GAME, not per agent: an agent instance is reused across deals in
        # some harnesses, and a set that outlived its game would silently
        # suppress signals in the next one.
        self._signalled = set()
        self._signals = 0

    # -- which belief each channel reads --------------------------------------
    #
    # Both hooks are the IDENTITY for the champion and every shipped
    # configuration, so nothing here changes any measured number. They exist so
    # one experiment can feed the DECLARATION channel and the ASK channel
    # different beliefs, which is the only way to ask whether the value of
    # knowing a teammate's cards is in what you ask or in when you dare to
    # declare. See fish4/oracle_gated.py and prereg/declaration_timing.md.
    #
    # The cut is: ClaimEvaluator, certain_claim and the tablebase are the
    # declaration channel -- they decide whether to name a split and which one.
    # Everything else is the ask channel. The tablebase sits on the declaration
    # side because when it fires it usually ends the half-suit by claiming.

    def _claim_ctx(self, ctx: DecisionContext) -> DecisionContext:
        """The context the claim machinery scores splits with.

        Identity unless ``claim_gamma`` is set, in which case the declaration
        reads a posterior built at that action-model weight while the ask
        objective keeps the configured one.

        WHY THE TWO DECISIONS DESERVE DIFFERENT MODELS
        ----------------------------------------------
        results/split_why.json, 1,619 frozen (decision, half-suit) pairs
        re-scored on the same belief: the engine's P(split right | ours) is
        under-confident by 0.219 in the half-suits our team owns outright, and
        that bias is FLAT in the draw count (0.219 / 0.244 / 0.237 at 480 /
        1920 / 5760) and MONOTONE in gamma (0.337 / 0.219 / 0.143 / 0.025 at
        0.0 / 0.35 / 0.7 / 1.4). It is the action model, not the sampler.

        Raising gamma globally is refuted -- prereg/gamma_split.md killed a
        uniform raise on teammate top-1 -- and the reason is that the two
        decisions are scored differently. An ask reads the argmax; a
        declaration is compared against 0.97. At that gate, gamma 0.7 clears it
        on 19.3% of these positions against the deployed 17.1%, at precision
        0.974 against 0.996.

        THE COST GATE. Building a second posterior at every decision would
        roughly double inference for a quantity only the declaration reads, so
        it is built only when something is near claimable -- `p_team_all` is
        already computed on this context, and the bar is ClaimEvaluator's own
        screen. Below it, tier 3 would never run and the extra posterior would
        be discarded unread.
        """
        if not self.claim_gamma:
            return ctx
        live = [hs for hs in range(ctx.n_hs) if ctx.hs_live[hs]]
        if not live or max(float(ctx.p_team_all[hs])
                           for hs in live) < self.claim_cfg.screen:
            return ctx
        self.claim_posteriors += 1
        return DecisionContext(
            ctx.obs, ctx.bel,
            self.build_posterior(ctx.obs, gamma=self.claim_gamma))

    def _claim_bel(self):
        """The belief the purely deductive claim paths read."""
        return self.bel

    # -- policy --------------------------------------------------------------

    def build_posterior(self, obs, n_draws: int = 0, gamma=None):
        """The posterior this seat would form at this decision.

        Extracted from ``act`` so an instrument wanting the SAME posterior
        under a different draw count gets it from the same seventeen arguments
        the policy uses. The alternative -- a script that re-lists them -- is
        the duplication that has already cost this project two results: a
        decoder wired into a sampler path no decision takes, and a size/bite
        diagnosis run against a hand-copied fragment of this function.

        ``n_draws`` overrides the configured count and changes nothing else, so
        the difference between two calls is sampling and only sampling. Zero
        means the configured count, which is what ``act`` passes. ``gamma``
        overrides the action-model weight the same way; None means the
        configured one. Both exist so the difference between two posteriors is
        exactly the named argument -- see scripts4/split_why.py, which used
        them to establish that the split joint's under-confidence is flat in
        draws and monotone in gamma.
        """
        return Posterior(self.bel, self.rng, n_draws=n_draws or self.n_draws,
                         n_worlds=self.n_worlds, mode=self.infer_mode,
                         obs=obs,
                         gamma=(self.opponent_gamma if gamma is None
                                else float(gamma)),
                         gamma_team=self.gamma_team,
                         convention_beta=self.convention_beta,
                         convention_q=self.convention_q,
                         convention_aim=self.convention_aim,
                         convention_book=self.convention_book,
                         depth_mode=self.depth_mode,
                         count_mode=self.count_mode,
                         opp_lambda=self.opp_lambda,
                         gamma_schedule=self.gamma_schedule,
                         sis_tilt=self.sis_tilt,
                         silence_delta=self.silence_delta,
                         stats=self.stats)

    def act(self, obs: Observation) -> Action:
        self.bel.update(obs)
        self.last_trace = None

        # Nothing hidden left? Solve the position instead of estimating it.
        # (Leak-free: the reconstruction uses only public events plus our own
        # hand, and refuses unless every live card is already pinned.)
        exact = self.tablebase_action(obs)
        if exact is not None:
            self._t(_tr.simple_trace, "exact",
                    solver="tablebase", note="every live card already pinned")
            return exact

        # Nothing pinned, but one half-suit left and few enough deals to solve
        # it exactly under imperfect information. Off by default.
        exact = self.exact_ii_action(obs)
        if exact is not None:
            self._t(_tr.simple_trace, "exact", solver="imperfect-information",
                    note="one half-suit left and few enough deals to solve")
            return exact

        post = self.build_posterior(obs)
        ctx = DecisionContext(obs, self.bel, post)

        if obs.must_pass():
            passes = obs.legal_passes()
            if self.smart_pass:
                pick = choose_pass(ctx, passes)
                if pick is not None:
                    self._t(_tr.simple_trace, "pass",
                            teammate=int(pick.teammate), chosen="scored")
                    return pick
            fallback = max(passes, key=lambda q: obs.hand_counts[q.teammate])
            self._t(_tr.simple_trace, "pass",
                    teammate=int(fallback.teammate), chosen="largest hand")
            return fallback

        claims = ClaimEvaluator(self._claim_ctx(ctx), self.claim_cfg)
        voluntary = claims.voluntary_claim()
        if voluntary is not None:
            best = claims.best_candidate()
            self._t(_tr.claim_trace, voluntary, why="voluntary",
                    confidence=(best[0] if best else None))
            return voluntary

        asks = obs.legal_asks()
        stalled = self.stalled(obs, window=self.stall_window)
        if not asks or (stalled and obs.claimable_half_suits()):
            forced = claims.forced_claim()
            if forced is not None:
                best = claims.best_candidate()
                self._t(_tr.claim_trace, forced,
                        why=("forced: no legal ask" if not asks
                             else "forced: stalled with a claimable half-suit"),
                        confidence=(best[0] if best else None))
                return forced
        if not asks:
            raise BeliefContradiction("no legal ask and no claimable half-suit")

        model = _load_value(self.hsvalue_path)
        if self.objective == "value" and model is not None:
            scores = score_asks_by_value(ctx, asks, model,
                                         turn_weight=self.value_turn,
                                         expose_weight=self.value_expose,
                                         keep_value=self.value_keep)
            _, p = score_asks(ctx, asks, AskWeights.zeros())
        else:
            # Style adapts to the match before the ask is scored: a team that
            # is behind weighs its tie-breakers differently from one that is
            # ahead. Zero leaves the incumbent weights untouched.
            wts = self.weights
            if self.endgame_m:
                live = sum(1 for x in obs.set_winner if x is None)
                if live <= self.endgame_m:
                    wts = replace(
                        wts, info=wts.info + self.endgame_d_info,
                        certain=wts.certain + self.endgame_d_certain)
            if self.w_behind:
                from .adaptive import adjust_weights
                wts = adjust_weights(wts, obs, self.w_behind)
            scores, p = score_asks(ctx, asks, wts)
            # Two passes, and the second one only sometimes. p_best is defined
            # as the success probability of the ask the INCUMBENT objective
            # would have chosen, because that is exactly the quantity the
            # paper's tempo section bucketed its price by. Using max(p) would
            # be cheaper and would not be the same number, so the first pass
            # stands and the tempo column is re-weighted only when it turns
            # out to have been charging for a turn worth nothing.
            if self.turn_free_below > 0.0 and len(p):
                top = max(range(len(scores)), key=lambda i: scores[i])
                if float(p[top]) < self.turn_free_below:
                    scores, p = score_asks(
                        ctx, asks,
                        replace(wts, turn=wts.turn * self.turn_free_scale))
            if self.w_value and model is not None:
                # No keep_value here, deliberately: this branch ADDS the value
                # objective to the heuristic one, and the heuristic already
                # carries P(success) at weight 1.0. Crediting the turn again
                # would price the same tempo twice.
                scores = scores + self.w_value * score_asks_by_value(
                    ctx, asks, model)
        # Belief-space lookahead, as an additive bonus rather than a
        # replacement. The bonus is identically zero at depth <= 1, so the
        # baseline is reproduced decision for decision and the weight ablates
        # exactly one idea. See fish4/lookahead.py for why this is the one
        # search design the variance diagnosis does not rule out.
        # Breaking a duel: penalise taking back a card this seat just lost to
        # this same opponent. See fish4/adaptive.py for why this is expected to
        # lose and why it is worth measuring regardless.
        if self.w_retake:
            from .adaptive import retake_flags
            scores = scores - self.w_retake * retake_flags(
                obs, asks, self.retake_window, self.retake_min_depth)

        # Half-suit contestation, signed: see adaptive.contest_bonus. At the
        # default 0.0 this branch never runs and the incumbent is reproduced
        # decision for decision.
        if self.w_contest:
            from .adaptive import contest_bonus
            scores = scores + self.w_contest * contest_bonus(ctx, asks, p)

        if self.w_lookahead and self.lookahead_depth > 1:
            from .lookahead import lookahead_bonus
            scores = scores + self.w_lookahead * lookahead_bonus(
                ctx, asks, depth=self.lookahead_depth,
                beam=self.lookahead_beam, couple=self.lookahead_couple,
                w_declare=self.lookahead_declare)
        if _SCORE_RECORDER is not None:
            _SCORE_RECORDER(self, ctx, asks, scores)
        order = sorted(range(len(asks)), key=lambda i: -scores[i])
        top = scores[order[0]]
        # If the ask we are ABOUT TO MAKE cannot land, it hands over the turn
        # for certain, so a declaration is worth considering instead - but only
        # if it is more likely right than wrong, since otherwise it gifts a set
        # on top of the lost turn, which is strictly worse than a doomed ask.
        # v0.3 used the same 0.5 bar. Under the opponent-award baseline the
        # bar is exactly break-even in sets (EV = 2p - 1 = 0 at p = 0.5, where
        # under the legacy null variant an all-ours candidate at the bar was
        # worth up to +0.5), so 0.5 remains defensible but is no longer
        # conservative; whether a higher bar plays better is an open
        # empirical question, deliberately not settled here by fiat.
        #
        # Note what this is NOT. An earlier version of this comment said "if no
        # ask can possibly land", which is a stricter condition - max(p) <= 0 -
        # and not what the line below tests. p[order[0]] is the probability of
        # the highest-SCORING ask, so the gate can open while some other ask
        # could still have landed, and it depends on every objective weight
        # rather than on a fact about the position. The two readings disagree on
        # roughly one decision in a few hundred.
        #
        # The code is left as it is deliberately. Both readings are defensible -
        # "the move I would make is doomed" is a reasonable trigger, not only
        # "every move is doomed" - and which plays better is an empirical
        # question nobody has measured. Changing it would silently move the
        # champion out from under every number in the paper to settle a question
        # by fiat. If it is ever measured, max(p) is the other arm.
        # Signalling. An ask placed in a half-suit our own team fully owns
        # cannot land, so it throws the turn away - but under the no-bluff rule
        # it publicly proves we do not hold the card we asked for, which is the
        # one fact a partner needs to place a split. Measured: a half-suit that
        # is provably ours but unplaceable is nulled 17.5% of the time against
        # 2.8% otherwise, and such half-suits are 27% of all nulls.
        #   "dead"  - only when NO ask anywhere can land, so the turn is free.
        #   "stuck" - also when our best ask is unlikely to land anyway, which
        #             makes the turn cheap rather than free.
        if self.signal_mode != "off":
            from .perpetual import signalling_ask, stuck_half_suits
            cheap = p[order[0]] <= (self.signal_max_p
                                    if self.signal_mode == "stuck" else 0.0)
            spent = (self.signal_budget
                     and self._signals >= self.signal_budget)
            if cheap and not spent and stuck_half_suits(obs, self.bel, ctx):
                sig = signalling_ask(
                    obs, self.bel, ctx,
                    require_dead=(self.signal_mode == "dead"),
                    exclude=(frozenset(self._signalled)
                             if self.signal_no_repeat else frozenset()))
                if sig is not None:
                    self._signalled.add(int(sig.card))
                    self._signals += 1
                    self._t(_tr.simple_trace, "signal",
                            target=int(sig.target),
                            # the gate's own input, recorded so an instrument
                            # can use it as a NEGATIVE control: widening this
                            # threshold 3.3x moved three declarations in a
                            # thousand games (prereg/deadline_signalling.md),
                            # so a probe that finds it predictive is suspect.
                            p_best=float(p[order[0]]),
                            note="a deliberately dead ask that proves to a "
                                 "partner which card this seat does not hold")
                    return sig

        if p[order[0]] <= 0.0:
            best = claims.best_candidate()
            # One bar or two. `best` is (p_exact, p_team, Claim), and the
            # incumbent reads only p_exact -- it declares at a coin flip
            # without asking the question the rest of this module is built
            # around. claim4's docstring says waiting is nearly free while our
            # own team holds the set, because an opponent who claims it is
            # wrong and hands it to us; that is precisely the p_team = 1 case,
            # and it is precisely where this gate does not look. At the
            # default stuck_team_certain = 1.01 the test below can never pass
            # and the single 0.5 bar is reproduced exactly.
            bar = (self.claim_cfg.threshold
                   if best is not None and best[1] >= self.stuck_team_certain
                   else self.claim_stuck_threshold)
            if best is not None and best[0] >= bar:
                self._t(_tr.claim_trace, best[2],
                        why="the best-scoring ask cannot land, so the turn is "
                            "lost anyway and this claim is better than even",
                        confidence=best[0])
                return best[2]
            # No claim, and the ask we are about to make cannot land -- so it
            # surrenders the turn for certain. Measured over 15,542 decisions
            # (results/doomed_ask_diag.json) that happens 269 times in 150
            # games, and in 229 of them ANOTHER ask could still have landed,
            # with a median success probability of 0.385. So 1.5% of all
            # decisions throw the turn away when a better-than-one-in-three
            # chance of keeping it was on the table.
            #
            # This restricts the choice to asks that can land and then ranks
            # them by the SAME objective, so it ablates exactly one idea: which
            # ask to make when the best-scoring one is doomed. The claim gate
            # above still sees the unfiltered order, so the claiming behaviour
            # is bit-identical and this cannot be two changes wearing one flag.
            #
            # Off by default. It is a hypothesis, not a fix: the objective
            # ranked the doomed ask top for reasons, and under the no-bluff
            # rule a failed ask publicly proves the asker holds another card of
            # that set, which is real information for a partner. Whether that
            # is worth a certain turn is what the duel is for.
            if self.avoid_doomed_asks:
                live = [i for i in order if p[i] > 0.0]
                if live:
                    order = live
                    top = scores[order[0]]
        pool = [i for i in order if scores[i] >= top - 1e-9]
        pick = int(self.rng.choice(pool))

        # ---- the convention, sender side -----------------------------------
        # The half-suit is already decided by the objective above and is NOT
        # touched here: only WHICH CARD of it we name. That keeps the change to
        # the one degree of freedom the objective was never using, and means a
        # measured effect cannot be a half-suit choice wearing a convention's
        # name.
        #
        # The cost gate is computed from our own posterior, so a partner cannot
        # reproduce it and cannot know whether any given ask carried a message.
        # That is exactly why the receiver's side is a soft weight rather than
        # a decode -- see fish4/convention.py.
        if self.convention_max_cost > 0.0:
            from .convention import encoded_card
            from fish.cards import half_suit_of
            chosen = asks[pick]
            hs = half_suit_of(chosen.card)
            hand = obs.hand
            if self.convention_book == "locate":
                # The locating book: name the card whose position tells a
                # partner the index of the first unlocated target card we hold.
                # They learn j negatives and one positive -- j + 1 cards
                # located -- from an ask that was happening anyway.
                from .convention import (half_suit_cards, legal_cards,
                                         locate_payload)
                best_u, g_hs = -1, 0
                for h in range(len(obs.set_winner)):
                    u = sum(1 for c in half_suit_cards(h)
                            if self.bel.public_loc[c] is None)
                    if u > best_u:
                        best_u, g_hs = u, h
                cards = legal_cards(hand, hs)
                tg = [c for c in half_suit_cards(g_hs)
                      if self.bel.public_loc[c] is None][:len(cards)]
                enc = (cards[locate_payload(hand, tg) % len(cards)]
                       if cards else None)
            elif self.convention_aim:
                # Aim at the most-unlocated half-suit rather than at the one
                # being asked in. The receiver reconstructs the same target
                # from the same public record, snapshotted at this ask -- see
                # the `located` ledger in oppmodel.build. We know our own hand
                # exactly, so the payload is exact; only the receiver has to
                # entertain worlds about it.
                from .convention import (encoded_position, half_suit_cards,
                                         legal_cards)
                n_hs = len(obs.set_winner)
                best_u, g_hs = -1, 0
                for h in range(n_hs):
                    u = sum(1 for c in half_suit_cards(h)
                            if self.bel.public_loc[c] is None)
                    if u > best_u:
                        best_u, g_hs = u, h
                payload = sum(1 for c in half_suit_cards(g_hs)
                              if hand >> c & 1)
                cards = legal_cards(hand, hs)
                enc = (cards[encoded_position(payload, len(cards))]
                       if cards else None)
            else:
                enc = encoded_card(hand, hs)
            # Legality is taken from the engine's own list, not recomputed. An
            # earlier version picked the target as the likeliest holder over
            # ALL opponents and raised IllegalAction("target has no cards") the
            # first time the likeliest holder was out of cards -- a seat can be
            # empty and still be the best guess for where a card went. The
            # objective's own candidate list has already excluded those, so ask
            # it rather than re-deriving the rule.
            # THE GATE IS PRICED IN THE OBJECTIVE'S OWN CURRENCY, and an
            # earlier version was not. It compared the drop in P(SUCCESS)
            # between the best legal card and the agreed one, and let anything
            # under `convention_max_cost` through. But `scores` is not P(success)
            # -- it carries lookahead, tempo, concentration and the information
            # the ask leaks -- so a swap that looked like it cost 0.009
            # probability could be discarding a large amount of what the
            # objective was actually ranking on. Measured over 120 duplicate-deal
            # pairs with the DECODER OFF, that mis-priced gate cost
            # -1.467 [-2.116, -0.818] sets a game: speaking, not listening, was
            # the expensive half. The gate now reads the same scores the pick
            # itself was made from.
            enc_idx = [i for i, a in enumerate(asks) if a.card == enc]
            if enc is not None and enc != chosen.card and enc_idx:
                best_enc = max(enc_idx, key=lambda i: scores[i])
                cost = scores[pick] - scores[best_enc]
                if cost <= self.convention_max_cost:
                    tgt = asks[best_enc].target
                    from fish.engine import Ask as _Ask
                    self._t(_tr.simple_trace, "convention",
                            card=int(enc), target=int(tgt),
                            cost=float(cost),
                            note="named the agreed card instead of the "
                                 "best-scoring one, to tell a partner how "
                                 "deep this seat is in the half-suit")
                    return _Ask(target=tgt, card=enc)

        self._t(_tr.ask_trace, obs, asks, scores, p, order, pool, pick)
        return asks[pick]

    # -- out-of-turn claiming --------------------------------------------------

    def certain_claim(self, obs: Observation):
        """A claim this seat could make with certainty, or None.

        Used only when the rules permit declaring outside your own turn. It is
        deliberately the cheap, purely deductive test - every card of some
        half-suit pinned by this seat's beliefs, all of them on this seat's team
        - because polling five off-turn seats after every action must not cost a
        full posterior each. A seat that merely *suspects* the split waits for
        its own turn, where the full machinery runs.
        """
        from fish.cards import half_suit_cards, team_of
        from fish.engine import Claim
        self.bel.update(obs)
        bel = self._claim_bel()
        my_team = team_of(self.player)
        for hs in range(len(obs.set_winner)):
            if obs.set_winner[hs] is not None:
                continue
            assignment = []
            for c in half_suit_cards(hs):
                m = bel.current_holder_mask(c)
                if m == 0 or m & (m - 1):
                    break
                holder = m.bit_length() - 1
                if team_of(holder) != my_team:
                    break
                assignment.append(holder)
            else:
                return Claim(hs, tuple(assignment))
        return None

    # -- introspection --------------------------------------------------------

    def diagnostics(self) -> dict:
        return self.stats.to_dict()
