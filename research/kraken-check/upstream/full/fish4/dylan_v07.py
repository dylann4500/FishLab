"""Dylan's FishBot v0.7 (github.com/dylann4500/fishbot), bridged into our engine.

The bot itself is untouched: the frozen ``fishbot_v07.json`` spec — the full
55-coordinate ``allparamsSpec`` vector plus the det=12 test-time search — runs
in their own C++ code, compiled from their tree into a one-shot ``decide``
binary (``external_v07/shim_decide.cpp``).  This file only translates.

STATELESS BY CONSTRUCTION
-------------------------
Each ``act`` spawns the binary, replays the WHOLE public history into it, asks
for one decision and exits.  That is not laziness: our website reconstructs
every session from the public log on every request, so a stateful bridge would
be a second source of truth that could drift from the first.  A replay costs a
few hundred events through a C++ belief — well under the price of one of our
own decisions.

WHAT IS TRANSLATED, AND WHAT IS NOT
-----------------------------------
* Cards.  Both engines encode ``card = set*6 + index`` but order the sets and
  the specials differently; ``_OURS_TO_THEIRS`` is the exact name-keyed
  bijection, built once and checked to be one (54 distinct images, set
  structure preserved).
* Seats and teams are identical (0-5, parity teams) — no translation.
* Events.  Our AskEvent/ClaimEvent/PassEvent map onto their Event kinds.  Two
  honest losses of information for their side, both disclosed rather than
  patched around: our ClaimEvent reveals the true holders even when the claim
  fails, and their Event format cannot carry that, so their belief learns only
  what their own game would have told it; and their out-of-turn declaration
  channel is never polled, because our rules do not have one.
* The forced declaration.  When a seat holds only complete sets it must
  declare, and the DRIVER picks which half-suit.  Theirs takes the first live
  one the mover holds a card in; ours took the first live one, period, which
  periodically asked their ``bestGuess`` to name owners in a half-suit it held
  nothing of.  See ``_forced_half_suit``.  This was a bridge defect biased in
  our favour and it is fixed; the published margin was re-measured after.
* Rules.  Their agent is constructed with ``outOfTurnDeclare=0`` so its own
  view of the rules matches the game it is actually playing.  Since the
  misdeclaration rule flipped to the opponent-award baseline, the two engines
  agree on what a wrong declaration costs -- their policy's native risk model
  is now exactly the rule it plays under.  (Games before the flip scored a
  within-team misdeclaration as a void here where their engine expected an
  award; that disclosed divergence is gone.)

LEGALITY IS VERIFIED, NOT ASSUMED
---------------------------------
Every action their side proposes is checked against our engine's own legality
lists before it is played.  A proposal that fails the check is replaced by a
fallback and COUNTED on ``self.fallbacks`` — a bridge that silently substituted
moves would be misrepresenting their bot, so the verification harness requires
that counter to be zero before any result is reported.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from fish.agents.base import Agent
from fish.cards import CARDS_PER_HALF_SUIT, half_suit_mask
from fish.engine import Ask, AskEvent, Claim, ClaimEvent, Pass, PassEvent
from fish.observation import Observation

_ROOT = Path(__file__).resolve().parents[1]

# The binary is looked for where the repo keeps it for local runs and where the
# deployment bundles it.  First hit wins.
# api/bin holds the DEPLOYABLE binary (static, portable flags); external_v07
# holds whatever the local build script last produced, which may be dynamic
# and tied to this machine's glibc. The deployable one therefore wins: the
# first deploy shipped both and the loader refused the local one on the
# function's older glibc ("version GLIBC_2.38 not found").
_BIN_CANDIDATES = [
    _ROOT / "api" / "bin" / "fish_v07_decide",
    Path("/var/task/api/bin/fish_v07_decide"),
    _ROOT / "external_v07" / "fish_v07_decide",
]

_SPEC_FILE = _ROOT / "external_v07" / "v07_spec.txt"


def _find_binary() -> Path:
    import os
    for p in _BIN_CANDIDATES:
        if p.exists():
            if not os.access(p, os.X_OK):
                # Some deploy pipelines drop the execute bit in transit; the
                # file's own directory may be read-only there, so failing to
                # restore it is reported, not swallowed.
                os.chmod(p, 0o755)
            return p
    raise FileNotFoundError(
        "fish_v07_decide binary not found; build it with external_v07/build.sh")


# ---------------------------------------------------------------- card mapping
def _their_card_names() -> list[str]:
    """Their 54 card names, in THEIR id order (mirrors engine/src/fish.hpp)."""
    low = ["2", "3", "4", "5", "6", "7"]
    high = ["9", "10", "J", "Q", "K", "A"]
    suit = ["S", "H", "D", "C"]
    out: list[str] = []
    for s in range(8):
        ranks = low if s % 2 == 0 else high
        out.extend(r + suit[s // 2] for r in ranks)
    out.extend(["8S", "8H", "8D", "8C", "RJ", "BJ"])
    return out


def _build_maps() -> tuple[list[int], list[int]]:
    from fish.cards import CARD_NAMES  # ours, in OUR id order
    theirs = {n: i for i, n in enumerate(_their_card_names())}
    ours_to_theirs = []
    for our_id, name in enumerate(CARD_NAMES):
        n = name.replace("T", "10") if name[0] == "T" else name
        ours_to_theirs.append(theirs[n])
    assert len(set(ours_to_theirs)) == 54, "card map is not a bijection"
    for our_id, their_id in enumerate(ours_to_theirs):
        # The bijection must respect set structure: our half-suit boundaries
        # land on their set boundaries, or asks change meaning in transit.
        assert all(ours_to_theirs[our_id // 6 * 6 + i] // 6 == their_id // 6
                   for i in range(6)), "half-suit torn by the card map"
    theirs_to_ours = [0] * 54
    for o, t in enumerate(ours_to_theirs):
        theirs_to_ours[t] = o
    return ours_to_theirs, theirs_to_ours


_OURS_TO_THEIRS, _THEIRS_TO_OURS = _build_maps()

#: The BRIDGE's behavioural revision -- not their engine's version, which is
#: frozen, and not this file's git hash, which changes for comments too. Bump
#: it only when a change here alters what their engine is asked or answers,
#: because that is exactly when games measured before and after stop being
#: comparable. Runners stamp it into every journal row and drop rows that do
#: not match on resume, so a repaired bridge can never be averaged together
#: with the games a defective one produced.
#:
#:   rev 1  the original bridge.
#:   rev 2  forced declarations now pick a half-suit they hold a card in, as
#:          their own driver does. See DylanV07._forced_half_suit. Priced at
#:          -0.0792 sets/game against us; results/BRIDGE_REVISIONS.md lists
#:          which stored journals are which.
BRIDGE_REV = 2


#: The frozen v0.7 spec (their engine/fishbot_v07.json, "allparamsSpec"),
#: embedded so the deployed function does not depend on any file beyond the
#: binary. external_v07/v07_spec.txt, when present, wins -- that is the copy a
#: refreshed upstream pin updates.
_EMBEDDED_SPEC = (
    "v07:rtie=1,pool=-1,oppfloor=-1,force=1000000,askfloor=-1,stall=12,s1=1,det=12,cand=4,kappa=2.5,rbelief=indep,depth=12,maxq=26,allparams=11.26561|4.18288|3.05733|3.80008|4.35318|6.42340|1.36441|-0.38185|-0.81545|-6.24532|-2.62967|1.49745|3.12582|3.96890|1.06205|-1.62528|4.04349|1.01429|1.21342|-0.79873|0.84218|0.76534|0.26573|4.77343|2.99879|7.12192|0.78874|0.84929|-0.02068|0.37062|0.14525|5.89200|3.27319|3.76581|0.17133|-0.47667|-0.77680|0.00000|0.00000|0.00000|0.00000|0.00000|0.00000|0.00000|0.00000|0.00000|0.00000|0.00000|0.00000|25.00000|0.00000|0.00000|0.00000|0.00000|0.00000"
)


def _load_spec() -> str:
    if _SPEC_FILE.exists():
        return _SPEC_FILE.read_text().strip()
    return _EMBEDDED_SPEC


class DylanV07(Agent):
    """Their v0.7, playing our game through the decide binary."""

    name = "dylan_v07"

    def __init__(self, timeout: float = 60.0, spec: str = None):
        """``spec`` overrides the frozen v0.7 spec string. Use with care.

        The default is and must remain the frozen spec: every headline number
        in this project is a statement about THAT policy, and a runner that
        quietly played a variant would be measuring something else under the
        same name. The override exists for one legitimate purpose --
        establishing, by paired measurement, that some coordinate of their
        spec does not change their strength, after which that variant may be
        used to buy throughput on SCREENS (never on a headline). See
        scripts4/v07_s1_equivalence.py.
        """
        super().__init__()
        self.timeout = timeout
        self.fallbacks = 0
        self.off_team_owners = 0
        #: (kind, detail) per fallback, for the verification harness — a
        #: fallback with no recorded reason is a silent substitution.
        self.fallback_log: list = []
        self._spec = spec or _load_spec()
        self._bin = str(_find_binary())
        self._seed = 1

    def begin_game(self, player: int, rules, seed: int) -> None:
        super().begin_game(player, rules, seed)
        self._seed = seed & 0x7FFFFFFFFFFFFFFF or 1

    # -- protocol assembly ---------------------------------------------------
    def _feed(self, obs: Observation) -> list[str]:
        their_hand = 0
        for c in range(54):
            if obs.hand >> c & 1:
                their_hand |= 1 << _OURS_TO_THEIRS[c]
        lines = [f"SPEC {self._spec}",
                 "RULES 9 0 1",
                 f"SEAT {self.player}",
                 f"HAND {their_hand}",
                 f"SEED {self._seed}"]
        # Replay with running hand counts and score, both of which their event
        # format carries as post-event state.  Counts derive from the events
        # alone; ClaimEvent.revealed says whose hand each retired card left.
        counts = [9] * 6
        score = [0, 0]
        for ev in obs.history:
            if isinstance(ev, AskEvent):
                if ev.success:
                    counts[ev.asker] += 1
                    counts[ev.target] -= 1
                hc = " ".join(str(x) for x in counts)
                lines.append(f"EV ASK {ev.asker} {ev.target} "
                             f"{_OURS_TO_THEIRS[ev.card]} {int(ev.success)} {hc}")
            elif isinstance(ev, ClaimEvent):
                for holder in ev.revealed:
                    counts[holder] -= 1
                if ev.winner in (0, 1):
                    score[ev.winner] += 1
                their_set = _OURS_TO_THEIRS[ev.half_suit * 6] // 6
                owners = [0] * 6
                for i in range(CARDS_PER_HALF_SUIT):
                    tc = _OURS_TO_THEIRS[ev.half_suit * 6 + i]
                    owners[tc % 6] = ev.declared[i]
                ok = int(ev.winner == (ev.claimer & 1))
                hc = " ".join(str(x) for x in counts)
                ow = " ".join(str(x) for x in owners)
                lines.append(f"EV DECL {ev.claimer} {their_set} {ok} {ow} {hc} "
                             f"{score[0]} {score[1]}")
            elif isinstance(ev, PassEvent):
                hc = " ".join(str(x) for x in counts)
                lines.append(f"EV PASS {ev.player} {ev.teammate} {hc}")
        lines.append(f"TURN {self.player}")
        return lines

    def _run(self, lines: list[str]) -> str:
        r = subprocess.run([self._bin], input="\n".join(lines) + "\n",
                           capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError(f"v07 decide failed rc={r.returncode}: "
                               f"{r.stderr[:200]}")
        return r.stdout.strip()

    # -- decisions -----------------------------------------------------------
    def act(self, obs: Observation):
        lines = self._feed(obs)
        if obs.must_pass():
            passes = obs.legal_passes()
            cand = [p.teammate for p in passes]
            out = self._run(lines + [
                f"DECIDE PASSTO {len(cand)} " + " ".join(map(str, cand))])
            if out.startswith("PASS "):
                t = int(out.split()[1])
                for p in passes:
                    if p.teammate == t:
                        return p
            self.fallbacks += 1
            self.fallback_log.append(("pass", out))
            return passes[0]

        # No legal ask at all (every opponent cardless) is their engine's
        # forced-declare phase, and their chooseAsk answers it with a junk
        # sentinel rather than an ask.  Detect it here and route to the same
        # forced path their own driver would take, so the sentinel never
        # reaches the legality check and every remaining fallback is a real
        # translation failure.
        if not obs.legal_asks():
            claimable = obs.claimable_half_suits()
            if claimable:
                hs = self._forced_half_suit(obs, claimable)
                their_set = _OURS_TO_THEIRS[hs * 6] // 6
                out = self._run(lines + [f"DECIDE FORCED {their_set}"])
                claim = self._claim_from(out.split(), obs, force_hs=hs)
                if claim is not None:
                    return claim
                self.fallbacks += 1
                self.fallback_log.append(("bad-forced-claim", out))
                return Claim(hs, tuple(self.player for _ in range(6)))
            return obs.legal_passes()[0]

        out = self._run(lines + ["DECIDE TURN"])
        parts = out.split()
        if parts and parts[0] == "DECL":
            claim = self._claim_from(parts, obs)
            if claim is not None:
                return claim
            self.fallbacks += 1
            self.fallback_log.append(("bad-claim", out))
        elif parts and parts[0] == "ASK":
            target, their_card = int(parts[1]), int(parts[2])
            ask = Ask(target=target, card=_THEIRS_TO_OURS[their_card])
            if any(a == ask for a in obs.legal_asks()):
                return ask
            self.fallbacks += 1
            self.fallback_log.append(
                ("illegal-ask", f"{out} -> ours t={target} c={ask.card}"))
        else:
            self.fallbacks += 1
            self.fallback_log.append(("garbled", out))

        # Fallback chain: any legal ask; else a forced claim on the first
        # claimable half-suit using THEIR best guess so the declaration is
        # still theirs; else the first legal pass.  All counted above.
        asks = obs.legal_asks()
        if asks:
            return asks[0]
        claimable = obs.claimable_half_suits()
        if claimable:
            hs = self._forced_half_suit(obs, claimable)
            their_set = _OURS_TO_THEIRS[hs * 6] // 6
            out = self._run(self._feed(obs) + [f"DECIDE FORCED {their_set}"])
            claim = self._claim_from(out.split(), obs, force_hs=hs)
            if claim is not None:
                return claim
            team = [p for p in range(6) if p % 2 == self.player % 2]
            return Claim(hs, tuple(self.player for _ in range(6)))
        return obs.legal_passes()[0]

    @staticmethod
    def _forced_half_suit(obs: Observation, claimable: list[int]) -> int:
        """Which half-suit their engine would be made to declare, their way.

        Their own driver (``engine/src/game.hpp:535``) does NOT hand the
        forced declaration an arbitrary live half-suit -- it takes the first
        active one the player still HOLDS A CARD IN::

            for (int st = 0; st < NSET; st++)
              if (g.pub.setActive[st] && (g.hand[g.turn] & setMask(st)))
                { chosen = st; break; }

        This bridge used to pass ``claimable[0]`` instead. That is not a
        translation of their rule, it is a harder question: forced to name
        the owners of a half-suit they hold nothing of, their ``bestGuess``
        has no anchor and is wrong nearly every time -- and under the
        opponent-award rule every one of those donates the set to US. An
        adversarial read of their engine caught it at six of eighteen forced
        claims, all six wrong, which flatters our published margin by roughly
        0.3 sets/game. A measurement that beats an opponent by mis-asking
        them a question their own driver never asks is not a measurement.

        The fallback is the same list as before, for the case their driver
        does not have to handle: it ends the game (``res.hitLimit``) when the
        mover holds no card in any live half-suit, and ours must still act.
        """
        for hs in claimable:
            if obs.hand & half_suit_mask(hs):
                return hs
        return claimable[0]

    def _claim_from(self, parts, obs: Observation, force_hs=None):
        try:
            their_set = int(parts[1])
            owners = [int(x) for x in parts[2:8]]
        except (IndexError, ValueError):
            return None
        hs = _THEIRS_TO_OURS[their_set * 6] // 6
        if force_hs is not None:
            hs = force_hs
        live = [h for h, w in enumerate(obs.set_winner) if w is None]
        if hs not in live:
            return None
        assignment = [0] * 6
        for i in range(CARDS_PER_HALF_SUIT):
            tc = _OURS_TO_THEIRS[hs * 6 + i]
            seat = owners[tc % 6]
            if seat % 2 != self.player % 2:
                # Our engine requires every declared owner on the claimant's
                # team; theirs does too for voluntary declarations, so this
                # exists only as a guard and is counted when it fires.
                self.off_team_owners += 1
                seat = self.player
            assignment[i] = seat
        return Claim(hs, tuple(assignment))
