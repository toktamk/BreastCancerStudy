from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

@dataclass
class BootstrapCI:
    mean: float
    std: float
    lo95: float
    hi95: float
    n_boot: int

def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 1337,
) -> BootstrapCI:
    y = np.asarray(y)
    p = np.asarray(p)
    m = np.isfinite(y) & np.isfinite(p)
    y = y[m]
    p = p[m]
    n = len(y)

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = float(metric_fn(y[idx], p[idx]))

    mean = float(vals.mean())
    std = float(vals.std(ddof=1))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return BootstrapCI(mean=mean, std=std, lo95=float(lo), hi95=float(hi), n_boot=int(n_boot))
