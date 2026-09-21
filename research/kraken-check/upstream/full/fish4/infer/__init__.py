"""Interchangeable hidden-state inference backends for FishBot v0.4.

Pick one with ``BACKENDS[name](belief, rng, **knob)``. See ``base.py`` for the
interface and ``FRONTIER.md`` for the accuracy-versus-compute measurements that
decide which one ships.
"""

from .base import FreeSystem, InferenceBackend, restore_belief, snapshot_belief
from .exact_or import ExactOrBackend
from .fastdraw import FastGroupSystem
from .mcmc import MCMCBackend
from .uniform_reject import UniformRejectBackend
from .v03 import V03Backend

BACKENDS = {
    "v03": V03Backend,
    "uniform_reject": UniformRejectBackend,
    "exact_or": ExactOrBackend,
    "mcmc": MCMCBackend,
}

__all__ = ["FreeSystem", "InferenceBackend", "FastGroupSystem",
           "V03Backend", "UniformRejectBackend", "ExactOrBackend",
           "MCMCBackend", "BACKENDS", "snapshot_belief", "restore_belief"]
