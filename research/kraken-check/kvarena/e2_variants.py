"""Bridge variants for E3. Each is KV's DylanV07 with ONE change, so the margin shift on identical
deals isolates that change."""
import os
from fish4.dylan_v07 import DylanV07, _OURS_TO_THEIRS, _THEIRS_TO_OURS
from fish.engine import AskEvent, ClaimEvent, Claim
from fish.cards import CARDS_PER_HALF_SUIT

_ROOT = os.path.dirname(os.path.abspath(__file__))
_BIN2 = os.path.join(_ROOT, "external_v07", "fish_v07_decide2")

def dealt_hand(obs):
    """The hand this seat was DEALT, recovered from the current hand by undoing every public
    transfer that touched it, newest first."""
    me = obs.player
    hand = obs.hand
    for ev in reversed(obs.history):
        if isinstance(ev, AskEvent) and ev.success:
            if ev.asker == me: hand &= ~(1 << ev.card)
            elif ev.target == me: hand |= (1 << ev.card)
        elif isinstance(ev, ClaimEvent):
            base = ev.half_suit * CARDS_PER_HALF_SUIT
            for i, holder in enumerate(ev.revealed):
                if holder == me: hand |= (1 << (base + i))
    return hand

def replay_forward(hand, obs):
    me = obs.player; h = hand
    for ev in obs.history:
        if isinstance(ev, AskEvent) and ev.success:
            if ev.asker == me: h |= (1 << ev.card)
            elif ev.target == me: h &= ~(1 << ev.card)
        elif isinstance(ev, ClaimEvent):
            base = ev.half_suit * CARDS_PER_HALF_SUIT
            for i, holder in enumerate(ev.revealed):
                if holder == me: h &= ~(1 << (base + i))
    return h

class DealtHand(DylanV07):
    """V1: HAND is the hand as DEALT, which is what FishLab's Agent::reset receives natively;
    rev 2 passes the CURRENT hand and then replays the whole history on top of it."""
    name = "dylan_v07"
    def __init__(self, **kw):
        super().__init__(**kw); self.stats = {"hand_mismatch": 0}
    def _feed(self, obs):
        hand = dealt_hand(obs)
        if replay_forward(hand, obs) != obs.hand: self.stats["hand_mismatch"] += 1
        lines = super()._feed(obs)
        their = 0
        for c in range(54):
            if hand >> c & 1: their |= 1 << _OURS_TO_THEIRS[c]
        return [l if not l.startswith("HAND ") else f"HAND {their}" for l in lines]

class SeedVaries(DylanV07):
    """V3: the agent seed changes with the history length, so the fresh agent built per decision
    does not restart the same random stream at every decision."""
    name = "dylan_v07"
    def _feed(self, obs):
        lines = super()._feed(obs)
        s = (self._seed * 1000003 + 7919 * (len(obs.history) + 1)) & 0x7FFFFFFFFFFFFFFF or 1
        return [l if not l.startswith("SEED ") else f"SEED {s}" for l in lines]

class MoverLadder(DylanV07):
    """V2: when the mover must declare, run FishLab's forced ladder for that seat over every live
    half-suit (proposal, then willingness at descending thresholds, then the most confident best
    guess) instead of a best guess on the first half-suit it holds a card in."""
    name = "dylan_v07"
    def __init__(self, **kw):
        super().__init__(**kw); self._bin = _BIN2
        self.stats = {"ladder_calls": 0, "ladder_fallbacks": 0}
    def _ladder(self, obs, claimable):
        self.stats["ladder_calls"] += 1
        their = [_OURS_TO_THEIRS[hs * 6] // 6 for hs in claimable]
        out = self._run(self._feed(obs) + [f"DECIDE FORCEDLADDER {len(their)} " + " ".join(map(str, their))])
        claim = self._claim_from(out.split(), obs)
        if claim is not None and claim.half_suit in claimable:
            return claim
        self.stats["ladder_fallbacks"] += 1
        self.fallbacks += 1; self.fallback_log.append(("bad-ladder-claim", out))
        return Claim(claimable[0], tuple(self.player for _ in range(6)))
    def act(self, obs):
        if obs.must_pass():
            return super().act(obs)
        if not obs.legal_asks():
            claimable = obs.claimable_half_suits()
            if claimable:
                return self._ladder(obs, claimable)
            return obs.legal_passes()[0]
        return super().act(obs)

class Both(MoverLadder, DealtHand):
    name = "dylan_v07"
    def __init__(self, **kw):
        MoverLadder.__init__(self, **kw); self.stats.update({"hand_mismatch": 0})

class DealtSeed(DealtHand, SeedVaries):
    """V5: dealt hand AND a per-decision seed (the two faithfulness repairs together)."""
    name = "dylan_v07"
    def _feed(self, obs):
        lines = DealtHand._feed(self, obs)
        s = (self._seed * 1000003 + 7919 * (len(obs.history) + 1)) & 0x7FFFFFFFFFFFFFFF or 1
        return [l if not l.startswith("SEED ") else f"SEED {s}" for l in lines]

def make(variant):
    return {"dealthand": DealtHand, "seedvary": SeedVaries, "ladder": MoverLadder, "both": Both, "dealtseed": DealtSeed}[variant]()
