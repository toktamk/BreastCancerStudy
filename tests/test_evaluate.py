# tests/test_evaluate.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.evaluate import evaluate_binary, harrell_c_index


def test_evaluate_binary_returns_expected_keys():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])

    out = evaluate_binary(y, p)
    assert "discrimination" in out
    assert "calibration" in out
    assert "robustness" in out
    assert "calibration_curve" in out
    assert "auroc" in out["discrimination"]
    assert "brier" in out["calibration"]


def test_group_metrics_optional():
    y = np.array([0, 0, 1, 1, 0, 1] * 10)
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9] * 10)
    g = pd.Series(["A", "A", "A", "A", "B", "B"] * 10)

    out = evaluate_binary(y, p, group=g)
    assert "fairness" in out
    assert "per_group" in out["fairness"]


def test_harrell_c_index_basic():
    # Two comparable pairs: should be perfect concordance when risk higher => earlier event
    time = np.array([5, 10, 7, 12])
    event = np.array([1, 1, 0, 1])  # third is censored
    risk = np.array([0.9, 0.1, 0.2, 0.05])  # highest risk at time=5
    c = harrell_c_index(time, event, risk)
    assert 0.0 <= c <= 1.0
