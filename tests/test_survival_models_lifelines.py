# tests/test_survival_models_lifelines.py
from __future__ import annotations

import numpy as np
import pytest

lifelines = pytest.importorskip("lifelines")

from src.models.survival import (
    CoxPHConfig,
    fit_coxph_lifelines,
    predict_risk_coxph_lifelines,
    predict_survival_prob_at_horizon_coxph,
    survival_to_5yr_risk,
)


def test_coxph_fit_predict_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    time = rng.uniform(1, 100, size=50)
    event = rng.integers(0, 2, size=50)

    model = fit_coxph_lifelines(X, time, event, CoxPHConfig(penalizer=0.01))
    risk = predict_risk_coxph_lifelines(model, X)
    assert risk.shape == (50,)

    sp = predict_survival_prob_at_horizon_coxph(model, X, horizon_months=60.0)
    assert sp.shape == (50,)
    assert np.all((sp >= 0.0) & (sp <= 1.0))

    rp = survival_to_5yr_risk(sp)
    assert np.all((rp >= 0.0) & (rp <= 1.0))
