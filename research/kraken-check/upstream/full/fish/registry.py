"""Experiment registry and reproducibility manifests.

Every experiment writes a machine-readable record with enough information to
re-run it: the code commit, the agent specs, every seed, the rule config,
and the result. Results without a manifest are not citable.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "results" / "experiments.jsonl"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True,
            stderr=subprocess.DEVNULL).strip()[:12]
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT), text=True,
            stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return True


@dataclass
class Experiment:
    name: str
    kind: str                       # matchup | ablation | benchmark | training
    config: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    evidence_tier: str = "PROMISING"   # DEMONSTRATED | PROMISING | SPECULATIVE
    notes: str = ""
    id: str = ""
    created: str = ""
    commit: str = ""
    dirty_worktree: bool = False
    python: str = ""
    platform_: str = ""

    def finalize(self) -> "Experiment":
        self.id = self.id or uuid.uuid4().hex[:12]
        self.created = self.created or time.strftime("%Y-%m-%d %H:%M:%S")
        self.commit = self.commit or git_commit()
        self.dirty_worktree = git_dirty()
        self.python = sys.version.split()[0]
        self.platform_ = platform.platform()
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["platform"] = d.pop("platform_")
        return d


def record(experiment: Experiment, path: Path = REGISTRY_PATH) -> Experiment:
    """Append an experiment to the registry (JSON Lines, append-only)."""
    experiment.finalize()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(experiment.to_dict()) + "\n")
    return experiment


def load_all(path: Path = REGISTRY_PATH) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def summary(path: Path = REGISTRY_PATH) -> str:
    rows = load_all(path)
    if not rows:
        return "no experiments recorded"
    lines = [f"{'when':20s} {'kind':11s} {'tier':13s} {'commit':13s} name"]
    for r in rows:
        flag = "*" if r.get("dirty_worktree") else " "
        lines.append(f"{r['created']:20s} {r['kind']:11s} "
                     f"{r['evidence_tier']:13s} {r['commit']:12s}{flag} "
                     f"{r['name']}")
    lines.append("\n* = recorded from a modified working tree "
                 "(not exactly reproducible from the commit alone)")
    return "\n".join(lines)
