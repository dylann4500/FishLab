"""Checkpoint league: a permanent registry of engine versions.

Every entry records how to rebuild the policy, its rating, the evaluation
evidence behind it, and provenance. Nothing is ever deleted, and a new
champion must BEAT the incumbent convincingly (paired-deal CI strictly above
0.5) to take the default slot.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
LEAGUE_PATH = ROOT / "checkpoints" / "league.json"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


@dataclass
class Entry:
    version: str                    # e.g. "probabilistic-v1"
    agent: str                      # registry key
    kwargs: dict
    rating: Optional[float] = None
    rating_stderr: Optional[float] = None
    separated: bool = False
    created: str = ""
    commit: str = ""
    notes: str = ""
    evidence: list = field(default_factory=list)
    is_champion: bool = False

    def spec(self) -> tuple[str, dict]:
        return (self.agent, dict(self.kwargs))


class League:
    def __init__(self, path: Path = LEAGUE_PATH):
        self.path = Path(path)
        self.entries: dict[str, Entry] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            for k, v in data.get("entries", {}).items():
                self.entries[k] = Entry(**v)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"entries": {k: asdict(v) for k, v in self.entries.items()}},
            indent=2))

    def register(self, version: str, agent: str, kwargs: dict,
                 notes: str = "", rating: Optional[float] = None,
                 rating_stderr: Optional[float] = None,
                 separated: bool = False,
                 evidence: Optional[list] = None) -> Entry:
        if version in self.entries:
            raise ValueError(f"version {version!r} already exists; versions "
                             "are immutable")
        e = Entry(version=version, agent=agent, kwargs=kwargs, rating=rating,
                  rating_stderr=rating_stderr, separated=separated,
                  created=time.strftime("%Y-%m-%d %H:%M:%S"),
                  commit=_git_commit(), notes=notes, evidence=evidence or [])
        self.entries[version] = e
        self.save()
        return e

    def champion(self) -> Optional[Entry]:
        for e in self.entries.values():
            if e.is_champion:
                return e
        return None

    def promote(self, version: str, evidence: dict) -> None:
        """Promote to champion. The paired-deal CI must strictly beat 0.5."""
        lo = evidence.get("wilson_ci", [0, 1])[0]
        if lo <= 0.5:
            raise ValueError(
                f"cannot promote {version}: paired-deal CI lower bound "
                f"{lo:.3f} does not convincingly beat the incumbent")
        for e in self.entries.values():
            e.is_champion = False
        entry = self.entries[version]
        entry.is_champion = True
        entry.evidence.append(evidence)
        self.save()

    def table(self) -> str:
        rows = sorted(self.entries.values(), key=lambda e: -(e.rating or -1e9))
        out = [f"{'version':26s} {'rating':>12s}  {'commit':9s} notes"]
        for e in rows:
            r = (f"{e.rating:.0f} +/- {e.rating_stderr:.0f}"
                 if e.rating is not None else "-")
            star = " *" if e.is_champion else "  "
            out.append(f"{e.version:26s} {r:>12s}{star} {e.commit:9s} {e.notes}")
        return "\n".join(out)
