"""Configurable rule set. Baseline values follow SPEC.md (Wikipedia rules)."""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class RuleConfig:
    variant: str = "54"                       # "54" (main) or "48"
    claims_any_time: bool = False             # baseline: claim only on your own turn
    allow_bluff_asks: bool = False            # baseline: may not ask for a card you hold
    #: What a claim that is wrong ONLY in its within-team split scores. The
    #: baseline is "opponent": a misdeclared half-suit is awarded to the
    #: opposing team, exactly as when the claim names a card the opponents
    #: hold. "null" (the set is voided, nobody scores) was this engine's
    #: default through the v0.4 measurement era and remains supported so that
    #: every result from that era is reproducible; it is a variant, not the
    #: game.
    wrong_distribution_outcome: str = "opponent"
    mandatory_claim_known: bool = False       # reserved for house-rule experiments
    starting_player: int = 0

    def __post_init__(self) -> None:
        if self.variant not in ("48", "54"):
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.wrong_distribution_outcome not in ("null", "opponent"):
            raise ValueError(
                f"unknown wrong_distribution_outcome "
                f"{self.wrong_distribution_outcome!r}")
        if not 0 <= self.starting_player < 6:
            raise ValueError("starting_player must be 0..5")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RuleConfig":
        return cls(**d)
