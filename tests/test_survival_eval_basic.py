from __future__ import annotations

import numpy as np
from src.eval.survival_eval import harrell_c_index, survival_metrics_basic


def test_harrell_c_index_in_range():
    time = np.array([5, 10, 7, 12], dtype=float)
    event = np.array([1, 1, 0, 1], dtype=int)
    risk = np.array([0.9, 0.1, 0.2, 0.05], dtype=float)
    c = harrell_c_index(time, event, risk)
    assert 0.0 <= c <= 1.0


def test_survival_metrics_basic_keys():
    time = np.array([5, 10, 7, 12], dtype=float)
    event = np.array([1, 1, 0, 1], dtype=int)
    risk = np.array([0.9, 0.1, 0.2, 0.05], dtype=float)
    out = survival_metrics_basic(time, event, risk)
    assert "harrell_c_index" in out
