"""A CHEATING agent, and the ceiling it measures.

READ THIS BEFORE USING ANYTHING HERE
------------------------------------
``OracleBot`` sees the true deal. It is not a policy, it is not a baseline, and
no number produced with it belongs in a "vs champion" table beside the honest
ones. It exists to measure a bound that honest play cannot: **how much is
perfect card-reading worth?**

Why that bound is the one worth having. The largest demonstrated gain in this
project is posterior precision -- +0.340 sets per deal-pair for 160 -> 480
draws -- and the next rung, 480 -> 1440, came back at +0.094 with an interval
touching zero over 6000 pairs. So *more sampling* has stopped paying. That is
not the same as *inference is exhausted*: a sampler can converge on a posterior
that is correct and still leave value on the table, or be wrong in a way more
draws never fixes. The two are distinguishable only against a ceiling.

``OracleBot`` is that ceiling. It keeps the champion's objective, its claim
rule, its tablebase, its every weight -- and replaces only its beliefs with the
truth. The margin it wins by is therefore the whole remaining value of card
reading FOR THIS POLICY, and every future inference improvement is bounded by
it. If the margin is small, better inference cannot be where the strength is
and the search should move elsewhere. If it is large, the sampler is leaving
real sets on the table and it is worth knowing how.

HOW IT CHEATS, AND WHY THAT IS THE WHOLE IMPLEMENTATION
-------------------------------------------------------
It does not bypass the posterior, or the objective, or the search. It pins the
BELIEF to the true initial deal and lets everything downstream follow. The
posterior built from a fully pinned belief is a point mass on the truth, so
``marginals()`` becomes a 0/1 matrix, every joint query becomes exact, and the
tablebase fires wherever it legally can. One intervention, at the one place the
hidden information enters.

That also makes the cheat self-checking. ``BeliefState._pin`` raises
``BeliefContradiction`` when the true holder is not already a candidate -- so
pinning to truth can only fail if the honest belief had already excluded the
truth, which would be an inference bug. Every game played here is therefore
also a test that the belief tracker never rules out the real world.

WHAT IT DOES NOT MEASURE
------------------------
Not exploitability. An opponent cannot see the deal, so this margin is not
available to any real adversary; it is an upper bound on a bound. Not the value
of optimal play either -- the oracle still plays the champion's heuristic
objective, merely on perfect information, so it is beaten by a
perfect-information *solver*. Those are separate measurements and they are not
this one.
"""

from __future__ import annotations

from typing import Optional

from fish.cards import NUM_PLAYERS, cards_per_player, deck_size
from fish.observation import Observation

from .agent4 import FishBot4


def initial_owners(deck_order, variant: str = "54") -> list[int]:
    """``owner[card]`` for a deal, from the deck order that produced it.

    ``GameState.deal`` assigns ``deck[i]`` to seat ``i % 6``, so this is the
    same deal the game will play -- derived from the caller's own shuffle
    rather than read off a state, which is what lets the oracle be configured
    before the game starts and never touch the running game.
    """
    n = deck_size(variant)
    if sorted(deck_order) != list(range(n)):
        raise ValueError("deck_order must be a permutation of the deck")
    owners = [0] * n
    for i, card in enumerate(deck_order):
        owners[card] = i % NUM_PLAYERS
    return owners


class OracleBot(FishBot4):
    """The champion, with the truth substituted for its beliefs.

    Refuses to act until ``see_deal`` has been called: a cheating agent that
    silently degrades to an honest one would put a number in a results file
    that means neither thing.
    """

    #: Which side's cards `reveal` is drawn from. See __init__.
    SIDES = ("all", "team", "opp")

    def __init__(self, reveal: float = 1.0, side: str = "all", **kwargs):
        """``reveal`` is the fraction of the cards it cannot see that it is told.

        1.0 is omniscience, which turns out to be nearly the maximum possible
        margin and therefore a poor bound: with perfect information you never
        miss an ask, never lose the turn, and take every set. What discriminates
        between inference ideas is the SHAPE of the curve below that, so this
        interpolates.

        The revealed set is chosen once per game and held fixed, which is the
        interpretable model: "this seat has perfect knowledge of these cards and
        ordinary inference about the rest". Re-drawing it every decision would
        instead model a stream of fresh information, and because pins persist in
        the belief it would accumulate to omniscience after enough turns rather
        than holding any fraction at all.

        Drawn per seat from that seat's own randomness, so two oracles on the
        same team do not share a lucky subset.

        ``side`` restricts the pool the fraction is drawn from: "team" tells it
        only where its TEAMMATES' cards are, "opp" only the opponents', "all"
        (the default, and the historical behaviour) both. It exists because the
        two are different problems and the project's error ledger says so:
        0.1676 of our 0.1759 wrong declarations a game are allocation class --
        our team held all six and named the wrong split -- against 0.0083
        ownership errors. Knowing where teammates' cards are is the fix for the
        first; knowing where opponents' are is the fix for the second, and
        nothing had ever priced them apart.

        NOT A CLEAN DECOMPOSITION, and saying so is the point. Telling a seat
        every one of its teammates' cards also tells it, by elimination, that
        the remaining cards are with opponents -- it just does not say which
        opponent. So "team" and "opp" are two BOUNDS on two different
        questions, not two halves that sum to omniscience, and any report of
        them has to say that rather than let a reader add them up.
        """
        super().__init__(**kwargs)
        if not 0.0 <= reveal <= 1.0:
            raise ValueError(f"reveal must be in [0, 1], got {reveal}")
        if side not in self.SIDES:
            raise ValueError(f"side must be one of {self.SIDES}, got {side!r}")
        self.reveal = reveal
        self.side = side
        self._owners: Optional[list[int]] = None
        self._revealed: Optional[set] = None
        #: Cards pinned by this agent rather than deduced. Reported so the
        #: cheat's size is visible instead of implied.
        self.pinned_by_cheat = 0
        self.decisions = 0

    def see_deal(self, owners: list[int]) -> None:
        """Hand it the true initial deal, before the game starts."""
        self._owners = list(owners)

    def begin_game(self, player: int, rules, seed: int) -> None:
        super().begin_game(player, rules, seed)
        self.pinned_by_cheat = 0
        self.decisions = 0
        self._revealed = None          # redrawn on the first act of this game

    def _draw_revealed(self) -> None:
        """Choose, once per game, which unknown cards this seat is told.

        Drawn from the cards NOT already pinned at the first decision -- which
        excludes this seat's own hand, so ``reveal`` is a fraction of what is
        genuinely hidden from it rather than of the whole deck.
        """
        hidden = [c for c in range(len(self._owners))
                  if not self.bel.is_pinned(c)]
        if self.side != "all":
            from fish.cards import team_of
            mine = team_of(self.player)
            want = (self.side == "team")
            hidden = [c for c in hidden
                      if (team_of(self._owners[c]) == mine) == want]
        if self.reveal >= 1.0:
            self._revealed = set(hidden)
            return
        k = int(round(self.reveal * len(hidden)))
        self._revealed = set(self.rng.sample(hidden, k)) if k else set()

    def _collapse(self) -> None:
        """Pin the revealed cards to whoever was actually dealt them."""
        if self._owners is None or self.bel is None:
            return
        if self._revealed is None:
            self._draw_revealed()
        for card in self._revealed:
            if not self.bel.is_pinned(card):
                # raises if the honest belief had excluded the truth
                self.bel._pin(card, self._owners[card])
                self.pinned_by_cheat += 1

    def act(self, obs: Observation):
        if self._owners is None:
            raise RuntimeError(
                "OracleBot.act called without see_deal: this agent is a "
                "deliberate cheat and refuses to run as an honest one")
        # update first so the belief has ingested this observation's events,
        # then collapse what remains hidden. Pinning before the update would
        # race the tracker's own deductions.
        self.bel.update(obs)
        self._collapse()
        self.decisions += 1
        return super().act(obs)


def oracle_spec(**kwargs):
    """Spec tuple for harnesses that address agents by name."""
    return ("oracle4", dict(kwargs))
