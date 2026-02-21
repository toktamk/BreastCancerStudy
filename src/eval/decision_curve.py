"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class DecisionCurve:
    thresholds: np.ndarray
    net_benefit: np.ndarray
    net_benefit_all: np.ndarray
    net_benefit_none: np.ndarray


def decision_curve_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Iterable[float] = np.linspace(0.01, 0.99, 99),
) -> DecisionCurve:
    """
    Decision Curve Analysis (DCA) for binary outcomes.

    Net benefit:
      NB(pt) = TP/N - FP/N * (pt/(1-pt))

    Baselines:
      NB_all: treat all as positive
      NB_none: treat none as positive (always 0)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thr = np.asarray(list(thresholds), dtype=float)

    if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must be 1D arrays of equal length")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must be binary {0,1}")

    n = len(y_true)
    prev = y_true.mean()

    nb = np.zeros_like(thr, dtype=float)
    nb_all = np.zeros_like(thr, dtype=float)
    nb_none = np.zeros_like(thr, dtype=float)

    for i, pt in enumerate(thr):
        y_hat = (y_prob >= pt).astype(int)

        tp = np.sum((y_hat == 1) & (y_true == 1))
        fp = np.sum((y_hat == 1) & (y_true == 0))

        w = pt / (1.0 - pt)
        nb[i] = (tp / n) - (fp / n) * w

        # Treat-all: everyone positive
        tp_all = np.sum(y_true == 1)
        fp_all = np.sum(y_true == 0)
        nb_all[i] = (tp_all / n) - (fp_all / n) * w

        nb_none[i] = 0.0

    return DecisionCurve(thresholds=thr, net_benefit=nb, net_benefit_all=nb_all, net_benefit_none=nb_none)
