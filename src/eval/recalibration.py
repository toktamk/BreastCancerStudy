"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-12

def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)

def logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1.0 - p))

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

@dataclass
class LogisticRecalibrator:
    a: float
    b: float

    @classmethod
    def fit(cls, y: np.ndarray, p: np.ndarray) -> "LogisticRecalibrator":
        y = np.asarray(y)
        p = np.asarray(p)
        m = np.isfinite(y) & np.isfinite(p)
        y = y[m].astype(int)
        lp = logit(p[m]).reshape(-1, 1)

        # sklearn compatibility: try penalty=None then fallback
        try:
            lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
            lr.fit(lp, y)
        except Exception:
            lr = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=5000)
            lr.fit(lp, y)

        return cls(a=float(lr.intercept_[0]), b=float(lr.coef_[0][0]))

    def transform(self, p: np.ndarray) -> np.ndarray:
        lp = logit(p)
        return sigmoid(self.a + self.b * lp)

    def to_dict(self) -> Dict[str, float]:
        return {"a": float(self.a), "b": float(self.b)}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "LogisticRecalibrator":
        return cls(a=float(d["a"]), b=float(d["b"]))
