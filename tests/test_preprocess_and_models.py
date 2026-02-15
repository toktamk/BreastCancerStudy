from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.preprocess import FeatureSpec, fit_transform_split
from src.models.baselines import LogisticConfig, fit_logistic, predict_proba


def test_preprocess_fit_on_train_only_and_model_runs():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3", "P4", "P5"],
            "age": [50, 60, 55, 40, 65],
            "subtype": ["A", "B", "A", "A", "B"],
            "expr__G1": [0.1, 0.2, 0.0, -0.1, 0.3],
            "cna__C1": [0, 1, -1, 0, 1],
            "time_months": [70, 10, 80, 50, 90],
            "event": [0, 1, 1, 0, 1],
            "y60": [0, 1, 0, 0, 0],
        }
    )

    train_ids = pd.Index(["P1", "P2", "P3"])
    val_ids = pd.Index(["P4"])
    test_ids = pd.Index(["P5"])

    bundle = fit_transform_split(df, train_ids, val_ids, test_ids, spec=FeatureSpec(label_col="y60"))

    X_train, y_train = bundle["X_train"], bundle["y_train"]
    X_val, y_val = bundle["X_val"], bundle["y_val"]
    X_test, y_test = bundle["X_test"], bundle["y_test"]

    assert X_train.shape[0] == 3
    assert X_val.shape[0] == 1
    assert X_test.shape[0] == 1

    model = fit_logistic(X_train, y_train, LogisticConfig(penalty="l2", C=1.0))
    p = predict_proba(model, X_test)

    assert p.shape == (1,)
    assert float(p[0]) >= 0.0 and float(p[0]) <= 1.0
