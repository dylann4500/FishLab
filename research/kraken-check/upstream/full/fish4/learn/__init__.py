"""Learning the ask objective for FishBot v0.4.

ROADMAP item 2: v0.3 scored an ask as ``P(success) + 0.06*depth`` and later
bolted on two hand-guessed terms. This package replaces the guessing with a
measurement: evaluate candidate asks by common-random-numbers paired rollouts,
then fit the coefficients of ``fish4.askfeat.TERM_NAMES`` by paired regression.

Why a regression can work where determinized search did not
-----------------------------------------------------------
v0.3 measured a noise-to-signal ratio of 2.4 for ranking candidate asks WITHIN
a single position (sd of one action's rollout value across sampled layouts
0.698, mean best-worst candidate gap 0.293). Search has to win that fight in
every position independently. A regression does not: it fits ten global
coefficients over thousands of paired observations, so the same per-position
noise averages down as 1/sqrt(number of positions) instead of having to be
beaten one position at a time. Pairing removes the dominant variance term (the
position's own value level) before any averaging happens at all.

Modules
-------
``rollout``  common-random-numbers paired rollout evaluator (leak-free
             continuation policy, documented knowledge set).
``dataset``  self-play position harvesting, resumable, JSONL on disk.
``fit``      paired ridge with the p_success coefficient pinned to 1.0,
             cluster-robust standard errors, permutation test, torch MLP
             nonlinear alternative.

CLAIMS ARE OUT OF SCOPE, ON PURPOSE. A claim must never be selected by
determinized search: inside a determinization the searcher knows the true
layout, so every claim evaluates as a certain gain and the declared assignment
differs between sampled worlds (strategy fusion). Everything here is about ASK
selection only; the claim policy is untouched.
"""

from __future__ import annotations

__all__ = ["rollout", "dataset", "fit"]
