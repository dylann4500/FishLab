from .base import Agent
from .random_agent import RandomAgent
from .heuristic import HeuristicAgent
from .memory import MemoryAgent
from .probabilistic import ProbabilisticAgent
from .search import SearchAgent
from .paired_search import PairedSearchAgent
from .value_search import ValueSearchAgent
from .ev_claim import EVClaimAgent
from .tuned import TunedAgent

AGENT_REGISTRY = {
    "ev_claim": EVClaimAgent,
    "tuned": TunedAgent,
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "memory": MemoryAgent,
    "probabilistic": ProbabilisticAgent,
    "search": SearchAgent,
    "paired_search": PairedSearchAgent,
    "value_search": ValueSearchAgent,
}

__all__ = ["Agent", "RandomAgent", "HeuristicAgent", "MemoryAgent",
           "ProbabilisticAgent", "SearchAgent", "PairedSearchAgent",
           "ValueSearchAgent", "EVClaimAgent", "TunedAgent", "AGENT_REGISTRY"]
