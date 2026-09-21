"""The analysis engine: what FishBot v0.4 thinks about a position.

This is the layer that makes the engine usable rather than merely strong. A
chess engine hands you an evaluation, a ranked move list and a principal
variation; the same three things exist here, they just mean something
different in a game whose hidden state is the whole problem.

**Evaluation.** In sets, from the acting seat's team's point of view. The
banked differential plus, for every unresolved half-suit, the learned
probability that our team ends up scoring it minus the probability that
theirs does (``hsvalue.py``). So ``+1.4`` reads as "we expect to finish 1.4
sets ahead of where the score already is". The model is calibrated against
outcomes rather than asserted: mean bias -0.006, reliable in every decile.

**Move list.** Every legal ask, with the probability it lands (which the
measurements say is the dominant term and not one consideration among
several), the change in evaluation it produces on each branch, and the
per-term breakdown of the policy's own score so a disagreement with the
engine can be attributed to a term rather than left as a mystery.

**Principal variation.** In chess the PV is the expected line. Here the
analogous object is the *possession*: the chain of asks the engine would make
if each one landed, which is what actually decides a turn in Fish. The chain is
computed by pinning each card as it is won and re-ranking, so it is the
engine's actual plan rather than a static list.

Two things a chess engine does not have to report, and this one must:

**The card table.** The posterior over who holds every unresolved card. This is
where the engine's strength lives, so it is shown, not hidden.

**The claim table.** For each live half-suit, the probability our team holds all
six, the most likely distribution among us, and the exact probability that
distribution is right - which is the number the claim decision turns on.

INFORMATION BOUNDARY: everything here is computed from an ``Observation`` plus
the seat's own belief state. The analyser can be pointed at any seat, but it
only ever sees what that seat is entitled to see.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fish.beliefs import RESOLVED, BeliefState
from fish.cards import (HALF_SUIT_NAMES, NUM_PLAYERS, card_name,
                        half_suit_cards, half_suit_mask, half_suit_of,
                        num_half_suits, team_of)
from fish.engine import Ask, Claim
from fish.observation import Observation

from .askfeat import TERM_NAMES, AskWeights, DecisionContext, ask_feature_matrix
from .hsvalue import HalfSuitValue, half_suit_features
from .posterior import Posterior


@dataclass
class MoveLine:
    """One candidate ask, as the engine sees it."""
    target: int
    card: int
    card_name: str
    half_suit: int
    half_suit_name: str
    p_success: float
    score: float                  # the policy's own ranking score
    eval_if_success: float        # evaluation in sets after it lands
    eval_if_fail: float
    eval_expected: float
    terms: dict = field(default_factory=dict)
    certain: bool = False

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["terms"] = {k: round(v, 4) for k, v in self.terms.items()}
        for k in ("p_success", "score", "eval_if_success", "eval_if_fail",
                  "eval_expected"):
            d[k] = round(float(d[k]), 4)
        return d


@dataclass
class ClaimLine:
    half_suit: int
    half_suit_name: str
    p_team_holds_all: float
    p_declaration_exact: float
    declaration: Optional[list]
    verdict: str

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["p_team_holds_all"] = round(float(d["p_team_holds_all"]), 4)
        d["p_declaration_exact"] = round(float(d["p_declaration_exact"]), 4)
        return d


@dataclass
class Analysis:
    seat: int
    team: int
    turn: int
    banked: tuple
    evaluation: float
    eval_breakdown: list
    moves: list
    claims: list
    principal_variation: list
    card_table: list
    exact: Optional[dict]
    deadlock: Optional[dict]
    notes: list
    ms: float

    def to_dict(self) -> dict:
        return {
            "seat": self.seat, "team": self.team, "turn": self.turn,
            "banked": list(self.banked),
            "evaluation": round(float(self.evaluation), 4),
            "eval_breakdown": self.eval_breakdown,
            "moves": [m.to_dict() for m in self.moves],
            "claims": [c.to_dict() for c in self.claims],
            "principal_variation": self.principal_variation,
            "card_table": self.card_table,
            "exact": self.exact, "deadlock": self.deadlock,
            "notes": self.notes, "ms": round(self.ms, 1),
        }


class Analyser:
    """Analyse a position from one seat's point of view.

    Reusable across a game: keep one per seat and call ``analyse`` after each
    action, so the belief state is maintained incrementally rather than replayed.
    """

    def __init__(self, rules, seat: int, weights: Optional[AskWeights] = None,
                 value_model: Optional[HalfSuitValue] = None,
                 n_draws: int = 320, gamma: float = 0.35, seed: int = 0,
                 w_lookahead: float = 0.0, lookahead_depth: int = 1,
                 lookahead_beam: int = 4, lookahead_couple: bool = True,
                 claim_forced_exhaustive: int = 0):
        self.rules = rules
        self.seat = seat
        #: Threaded through to ClaimConfig so the split this class SUGGESTS is
        #: the split the engine would actually play. It shipped in
        #: V06_DEPLOYED on 2026-08-28, and an explanation of a policy the site
        #: is not playing is the exact failure api/_engine._analyser_spec was
        #: written to make loud rather than silent.
        self.claim_forced_exhaustive = int(claim_forced_exhaustive)
        self.weights = weights or AskWeights()
        self.value = value_model
        self.n_draws = n_draws
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.bel = BeliefState(rules, observer=seat)
        # The lookahead has to be here, and not only in the agent, because this
        # class explains the agent's ranking to a human. A caller playing a
        # policy with the search on and rendering an explanation with it off
        # shows a score that omits a term the engine used, and a decomposition
        # that leaves a term out is worse than no decomposition. Defaults are
        # the incumbent's, so an explanation of the champion is unchanged.
        self.w_lookahead = w_lookahead
        self.lookahead_depth = lookahead_depth
        self.lookahead_beam = lookahead_beam
        self.lookahead_couple = lookahead_couple

    # -- evaluation ------------------------------------------------------------

    def _half_suit_values(self, ctx) -> list:
        """Per-half-suit expected contribution, in sets."""
        out = []
        my_team = ctx.my_team
        for hs in range(ctx.n_hs):
            w = ctx.obs.set_winner[hs]
            if w is not None:
                out.append({"half_suit": hs, "name": HALF_SUIT_NAMES[hs],
                            "state": ("ours" if w == my_team else
                                      "theirs" if w in (0, 1) else "nulled"),
                            "value": (1.0 if w == my_team else
                                      -1.0 if w in (0, 1) else 0.0),
                            "settled": True})
                continue
            if self.value is not None:
                v = float(self.value.value(half_suit_features(ctx, hs)))
            else:
                # Without a fitted model, fall back on the share of the
                # half-suit our team is expected to hold, mapped to [-1, 1].
                v = float(ctx.team_exp[hs] / 6.0 - ctx.opp_exp[hs] / 6.0)
            out.append({"half_suit": hs, "name": HALF_SUIT_NAMES[hs],
                        "state": "live", "value": round(v, 4),
                        "team_share": round(float(ctx.team_exp[hs] / 6.0), 4),
                        "my_cards": int(ctx.my_depth[hs]), "settled": False})
        return out

    # -- move ranking -----------------------------------------------------------

    def _rank_moves(self, ctx, asks, banked_diff: float = 0.0) -> list:
        """Move evaluations are reported on the SAME scale as the headline
        evaluation - banked sets included - so that a move line reads as
        "this takes you from +0.78 to +0.62" rather than as a number on a
        private axis."""
        p, F = ask_feature_matrix(ctx, asks)
        scores = p + F @ self.weights.as_vector()
        bonus = None
        if self.w_lookahead and self.lookahead_depth > 1:
            from .lookahead import lookahead_bonus
            bonus = self.w_lookahead * lookahead_bonus(
                ctx, asks, depth=self.lookahead_depth,
                beam=self.lookahead_beam, couple=self.lookahead_couple)
            scores = scores + bonus
        out = []
        M = ctx.M
        for i, a in enumerate(asks):
            hs = a.card // 6
            saved = M[hs * 6:hs * 6 + 6].copy()
            M[a.card] = 0.0
            M[a.card, ctx.me] = 1.0
            ev_s = banked_diff + self._live_eval(ctx)
            M[hs * 6:hs * 6 + 6] = saved
            row = M[a.card].copy()
            row[a.target] = 0.0
            s = row.sum()
            M[a.card] = row / s if s > 0 else row
            ev_f = banked_diff + self._live_eval(ctx)
            M[hs * 6:hs * 6 + 6] = saved
            out.append(MoveLine(
                target=a.target, card=a.card, card_name=card_name(a.card),
                half_suit=hs, half_suit_name=HALF_SUIT_NAMES[hs],
                p_success=float(p[i]), score=float(scores[i]),
                eval_if_success=ev_s, eval_if_fail=ev_f,
                eval_expected=p[i] * ev_s + (1 - p[i]) * ev_f,
                terms={**{n: float(F[i, j] * getattr(self.weights, n))
                          for j, n in enumerate(TERM_NAMES)
                          if getattr(self.weights, n)},
                       **({"lookahead": float(bonus[i])}
                          if bonus is not None else {})},
                certain=bool(p[i] >= 0.999999)))
        out.sort(key=lambda m: -m.score)
        return out

    def _live_eval(self, ctx) -> float:
        """Expected half-suit differential, read from the CURRENT marginals.

        The caller evaluates a candidate by mutating ``ctx.M`` -- forcing the
        asked card to us for the success branch, redistributing it away from
        the target for the failure branch -- and calling this in between. That
        only works if this function reads M.

        It used to read ``ctx.team_exp`` / ``ctx.opp_exp``, which
        DecisionContext computes once at construction and never refreshes
        (fish4/askfeat.py:178-186). So the two branches returned the same
        number by construction: measured on the live site, 8 of 8 ranked asks
        had eval_if_success == eval_if_fail == eval_expected, all of them
        0.000000. Three fields the dataclass documents as "evaluation in sets"
        were inert, and any caption built on their difference -- "this move
        costs you X" -- would have printed zero forever.

        The quantity is the same one askfeat computes, just re-derived from the
        block that was mutated: a column sum over the six cards of the
        half-suit, split by team.
        """
        tot = 0.0
        mine = ctx.my_team
        for hs in range(ctx.n_hs):
            if ctx.obs.set_winner[hs] is not None:
                continue
            if self.value is not None:
                tot += float(self.value.value(half_suit_features(ctx, hs)))
            else:
                block = ctx.M[hs * 6:hs * 6 + 6]
                per_player = block.sum(axis=0)
                team = sum(per_player[p] for p in range(NUM_PLAYERS)
                           if p % 2 == mine)
                opp = sum(per_player[p] for p in range(NUM_PLAYERS)
                          if p % 2 != mine)
                tot += float(team / 6.0 - opp / 6.0)
        return tot

    # -- principal variation ----------------------------------------------------

    def _pv(self, ctx, asks, depth: int = 5) -> list:
        """The possession the engine is planning: best ask, then the best ask
        assuming it landed, and so on. This is the Fish analogue of a chess PV,
        because a turn in Fish is a run of successful asks."""
        if not asks:
            return []
        M = ctx.M
        saved = M.copy()
        hand = ctx.obs.hand
        line = []
        try:
            for _ in range(depth):
                p, F = ask_feature_matrix(ctx, asks)
                if not len(p):
                    break
                scores = p + F @ self.weights.as_vector()
                i = int(np.argmax(scores))
                a = asks[i]
                line.append({"target": a.target, "card": a.card,
                             "card_name": card_name(a.card),
                             "p_success": round(float(p[i]), 4)})
                if p[i] <= 0.0:
                    break
                # assume it lands: the card is now ours and perfectly located
                M[a.card] = 0.0
                M[a.card, ctx.me] = 1.0
                hand |= 1 << a.card
                asks = [b for b in asks if b.card != a.card]
                ctx.M = M
        finally:
            ctx.M = saved
        return line

    # -- card and claim tables ---------------------------------------------------

    def _card_table(self, ctx) -> list:
        rows = []
        M = ctx.M
        for hs in range(ctx.n_hs):
            if ctx.obs.set_winner[hs] is not None:
                continue
            for c in half_suit_cards(hs):
                loc = self.bel.public_loc[c]
                if loc == RESOLVED:
                    continue
                probs = [round(float(M[c, p]), 3) for p in range(NUM_PLAYERS)]
                best = int(np.argmax(M[c]))
                rows.append({
                    "card": c, "name": card_name(c), "half_suit": hs,
                    "probs": probs, "most_likely": best,
                    "certain": bool(M[c, best] >= 0.999999),
                    "public": loc is not None,
                    "mine": bool(ctx.obs.hand >> c & 1)})
        return rows

    def _claim_table(self, ctx) -> list:
        from .claim4 import ClaimConfig, ClaimEvaluator
        ev = ClaimEvaluator(ctx, ClaimConfig(
            forced_exhaustive=self.claim_forced_exhaustive))
        out = []
        for hs in ctx.obs.claimable_half_suits():
            r = ev.best_for_half_suit(hs)
            if r is None:
                out.append(ClaimLine(hs, HALF_SUIT_NAMES[hs], 0.0, 0.0, None,
                                     "an opponent certainly holds a card"))
                continue
            p_exact, p_team, claim = r
            if p_exact >= 0.97:
                verdict = "CLAIM: declare it now"
            elif p_team >= 0.9:
                verdict = "your team has it; the split is not yet placed"
            elif p_team >= 0.4:
                verdict = "contested"
            else:
                verdict = "the opponents hold at least one"
            out.append(ClaimLine(hs, HALF_SUIT_NAMES[hs], p_team, p_exact,
                                 list(claim.assignment), verdict))
        out.sort(key=lambda c: -c.p_declaration_exact)
        return out

    # -- exact endgame -----------------------------------------------------------

    def _exact(self, obs) -> Optional[dict]:
        from fish.agents.tablebase import pinned_state
        try:
            from .exact2 import Exact2Solver
        except Exception:
            return None
        state = pinned_state(obs, self.bel)
        if state is None:
            return None
        live = sum(1 for w in state.set_winner if w is None)
        if live == 0 or live > 2:
            return None
        try:
            solver = Exact2Solver(state.rules)
            value, _ = solver.solve(state)
            best = solver.progress_optimal_actions(state)
        except Exception:
            return None
        my_team = team_of(obs.player)
        return {
            "solved": True, "live_half_suits": live,
            "value_for_me": round(float(value if my_team == 0 else -value), 3),
            "optimal": [_action_repr(a) for a in best[:8]],
            "note": ("every remaining card's location is deduced, so this is "
                     "the exact game value, not an estimate"),
        }

    # -- the main entry point ------------------------------------------------------

    def context(self, obs: Observation) -> DecisionContext:
        """The belief update and posterior draw behind one decision.

        Factored out so that a caller wanting one number off the posterior --
        the price of a split the player built themselves, say -- reads it from
        the same context `analyse` ranks moves with, rather than from a second
        construction that could drift away from it.
        """
        self.bel.update(obs)
        post = Posterior(self.bel, self.rng, n_draws=self.n_draws,
                         mode="auto", obs=obs, gamma=self.gamma)
        return DecisionContext(obs, self.bel, post)

    def analyse(self, obs: Observation, pv_depth: int = 5) -> Analysis:
        import time
        t0 = time.perf_counter()
        ctx = self.context(obs)
        asks = obs.legal_asks()
        a, b, nulls = _scores(obs)
        my_team = team_of(obs.player)
        banked = (a if my_team == 0 else b, b if my_team == 0 else a, nulls)

        breakdown = self._half_suit_values(ctx)
        live_eval = self._live_eval(ctx)
        evaluation = (banked[0] - banked[1]) + live_eval

        moves = (self._rank_moves(ctx, asks, banked[0] - banked[1])
                 if asks else [])
        claims = self._claim_table(ctx)
        pv = self._pv(ctx, asks, pv_depth) if asks else []
        exact = self._exact(obs)
        from .perpetual import diagnose
        dead = diagnose(obs, self.bel, ctx)

        notes = []
        if not asks and obs.turn == obs.player:
            notes.append("No legal ask exists: you must declare a half-suit.")
        if moves and moves[0].certain:
            notes.append("A certain steal is available: it cannot fail, so it "
                         "costs nothing to take first.")
        if claims and claims[0].p_declaration_exact >= 0.97:
            notes.append(
                f"{claims[0].half_suit_name} is ready to declare "
                f"(P={claims[0].p_declaration_exact:.3f}).")
        if dead and dead.get("is_dead"):
            notes.append(dead["summary"])
        elif dead and dead.get("unplaceable"):
            # Not a dead position, but a frozen half-suit inside a live one:
            # ours, unplaceable, and beyond reach of anything an opponent can
            # do. Worth saying while there are still turns to spend on it --
            # the previous behaviour said nothing until the whole position had
            # died, by which point the choice has usually been made for you.
            notes.append(dead["summary"])

        return Analysis(
            seat=obs.player, team=my_team, turn=obs.turn, banked=banked,
            evaluation=evaluation, eval_breakdown=breakdown, moves=moves,
            claims=claims, principal_variation=pv,
            card_table=self._card_table(ctx), exact=exact, deadlock=dead,
            notes=notes, ms=(time.perf_counter() - t0) * 1000.0)


def _scores(obs) -> tuple:
    a = sum(1 for w in obs.set_winner if w == 0)
    b = sum(1 for w in obs.set_winner if w == 1)
    n = sum(1 for w in obs.set_winner if w == -1)
    return a, b, n


def _action_repr(act) -> dict:
    if isinstance(act, Ask):
        return {"type": "ask", "target": act.target, "card": act.card,
                "card_name": card_name(act.card)}
    if isinstance(act, Claim):
        return {"type": "claim", "half_suit": act.half_suit,
                "half_suit_name": HALF_SUIT_NAMES[act.half_suit],
                "assignment": list(act.assignment)}
    return {"type": "pass", "teammate": getattr(act, "teammate", None)}
