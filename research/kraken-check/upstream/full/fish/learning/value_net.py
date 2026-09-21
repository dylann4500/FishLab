"""Value network: features -> expected remaining set differential.

Purpose (from the search-failure diagnosis in RESEARCH_LOG.md): rollout noise
is ~2.4x the gap between candidate actions, so search on rollouts optimizes
noise. A learned value function evaluates a position in one deterministic
forward pass, with zero rollout variance, which is the standard fix.

Deliberately small: the feature set is already a strong summary, the dataset
is modest, and inference must be cheap enough to call thousands of times per
game on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.in_dim = in_dim

    def forward(self, x):
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def value(self, x: np.ndarray) -> float:
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return float(self.forward(t)[0])

    @torch.no_grad()
    def values(self, X: np.ndarray) -> np.ndarray:
        """Batched inference: one call for many leaves is far cheaper than
        many single calls."""
        t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        return self.forward(t).numpy()


def train_value_net(X: np.ndarray, y: np.ndarray, hidden: int = 128,
                    epochs: int = 40, batch_size: int = 512, lr: float = 1e-3,
                    val_frac: float = 0.15, seed: int = 0,
                    verbose: bool = True):
    torch.manual_seed(seed)
    n = len(X)
    idx = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xtr = torch.from_numpy(X[tr_idx]); ytr = torch.from_numpy(y[tr_idx])
    Xva = torch.from_numpy(X[val_idx]); yva = torch.from_numpy(y[val_idx])
    model = ValueNet(X.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    best_val, best_state = float("inf"), None
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        total = 0.0
        for i in range(0, len(Xtr), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            vpred = model(Xva)
            vloss = float(lossf(vpred, yva))
            base = float(lossf(torch.full_like(yva, float(ytr.mean())), yva))
            corr = (float(np.corrcoef(vpred.numpy(), yva.numpy())[0, 1])
                    if len(yva) > 2 else float("nan"))
        history.append({"epoch": ep, "train_mse": total / len(Xtr),
                        "val_mse": vloss, "baseline_mse": base,
                        "val_corr": corr})
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  train {total/len(Xtr):7.3f}  "
                  f"val {vloss:7.3f}  (mean-baseline {base:7.3f})  "
                  f"corr {corr:+.3f}", flush=True)
    if best_state:
        model.load_state_dict(best_state)
    return model, history


def save_value_net(model: ValueNet, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"in_dim": model.in_dim,
                "hidden": model.net[0].out_features,
                "state": model.state_dict()}, path)


def load_value_net(path: Path | str) -> ValueNet:
    d = torch.load(path, map_location="cpu", weights_only=True)
    m = ValueNet(d["in_dim"], d["hidden"])
    m.load_state_dict(d["state"])
    m.eval()
    return m
