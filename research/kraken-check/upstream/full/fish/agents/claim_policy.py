"""Decision-theoretic claim policy.

Replaces "claim when confidence >= 0.97" with an expected-value comparison.
The threshold was the least principled part of an otherwise
belief-driven agent: 0.97 was chosen by intuition, and the right threshold
demonstrably is not a constant. It should depend on:

- the payoff structure of being wrong (null vs gift to the opponents),
- how much the set is worth relative to the rest of the game,
- whether waiting is likely to improve the estimate or lose the set,
- the risk of an opponent claiming first,
- the score (a team that is behind should accept more variance).

EV of claiming now, in sets, from the claimant's perspective:

    EV_claim = p_exact * (+1)
             + p_team_but_wrong_split * (loss_if_wrong)
             + p_opponent_holds_one * (-1)

where ``loss_if_wrong`` is 0 under the baseline null rule (nobody scores)
and -1 under ``wrong_distribution_outcome="opponent"``.

EV of waiting is the same set's value later, discounted by:
- the chance opponents resolve it against us first,
- the chance our information improves (which is what makes waiting good).

STATUS: this model's headline prediction has been FALSIFIED and the module
is retained as an ablation handle, not as the recommended policy.

It derived an optimal claim threshold near 0.70. A direct sweep (150 paired
deals per cell) showed 0.70 is measurably WORSE than 0.97, by +0.45 sets per
deal-pair [+0.18, +0.72], and that any threshold from 0.85 to near-certainty
plays identically. See RESEARCH_LOG.md.

The error is identifiable: ``p_improve`` below defaults to 0.55, treating
the resolution of an unknown teammate split as roughly a coin flip. In real
play, continued asking localizes teammate holdings far more reliably, so
waiting is worth substantially more than this model credits. Fitting
``p_improve`` and ``p_opponent_takes`` from self-play data is the concrete
fix; until then the fixed-threshold policy is the better default.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClaimEV:
    p_exact: float          # P(declared distribution exactly right)
    p_team_holds: float     # P(all six within our team, any split)
    hs: int
    claim: object

    @property
    def p_wrong_split(self) -> float:
        return max(0.0, self.p_team_holds - self.p_exact)

    @property
    def p_opponent_holds(self) -> float:
        return max(0.0, 1.0 - self.p_team_holds)


def claim_now_ev(c: ClaimEV, wrong_distribution_outcome: str = "opponent") -> float:
    """Expected sets gained by claiming immediately."""
    loss_if_wrong = 0.0 if wrong_distribution_outcome == "null" else -1.0
    return (c.p_exact * 1.0
            + c.p_wrong_split * loss_if_wrong
            + c.p_opponent_holds * -1.0)


def wait_ev(c: ClaimEV, *, p_improve: float = 0.55,
            p_opponent_takes: float = 0.10,
            improvement_ceiling: float = 0.98,
            wrong_distribution_outcome: str = "opponent") -> float:
    """Expected sets from NOT claiming yet.

    Waiting is worth something only when the uncertainty is *resolvable*:
    if our team already holds the set, more asks can localize it, so the
    eventual claim is likely exact. If an opponent holds one, waiting lets
    us discover that and avoid gifting the set.

    ``p_improve`` is the chance that continued play resolves the split
    before the opportunity is lost; ``p_opponent_takes`` is the chance the
    opposing team claims it against us in the meantime.
    """
    # If our team holds it, waiting converges toward an exact claim. The
    # mass that stays unresolved is eventually declared wrong, and what that
    # costs is the misdeclaration rule's to say: 0 under the legacy null
    # variant, -1 under the opponent-award baseline.
    loss_if_wrong = 0.0 if wrong_distribution_outcome == "null" else -1.0
    resolved_p = c.p_exact + p_improve * (c.p_team_holds - c.p_exact)
    resolved_p = min(resolved_p, improvement_ceiling * c.p_team_holds)
    future = resolved_p * 1.0 + (c.p_team_holds - resolved_p) * loss_if_wrong
    # An opponent-held set stays a loss whether we claim it or not, but
    # waiting at least avoids handing it over on a bad claim.
    future += c.p_opponent_holds * -1.0 * 0.5
    return (1 - p_opponent_takes) * future + p_opponent_takes * (-1.0)


def should_claim(c: ClaimEV, wrong_distribution_outcome: str = "opponent",
                 margin: float = 0.0, **wait_kwargs) -> bool:
    """Claim iff claiming now beats waiting by ``margin`` sets."""
    return (claim_now_ev(c, wrong_distribution_outcome)
            >= wait_ev(c, wrong_distribution_outcome=wrong_distribution_outcome,
                       **wait_kwargs) + margin)


def best_claim_decision(candidates: list[ClaimEV],
                        wrong_distribution_outcome: str = "opponent",
                        margin: float = 0.0, **wait_kwargs):
    """Return (claim, ev_gain) for the best claim worth making, or None."""
    best = None
    for c in candidates:
        now = claim_now_ev(c, wrong_distribution_outcome)
        later = wait_ev(c, wrong_distribution_outcome=wrong_distribution_outcome,
                        **wait_kwargs)
        gain = now - later
        if gain >= margin and (best is None or gain > best[1]):
            best = (c.claim, gain)
    return best
