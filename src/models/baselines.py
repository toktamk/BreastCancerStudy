# src/models/baselines.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


@dataclass(frozen=True)
class LogisticConfig:
    penalty: str = "l2"          # "l2" or "elasticnet"
    C: float = 1.0
    l1_ratio: Optional[float] = None  # required if elasticnet
    max_iter: int = 5000
    class_weight: Optional[str] = "balanced"
    random_state: int = 1337


def fit_logistic(X: np.ndarray, y: np.ndarray, cfg: LogisticConfig) -> LogisticRegression:
    """
    Binary baseline for 5-year OS. Strong and standard.
    """
    if cfg.penalty == "elasticnet":
        if cfg.l1_ratio is None:
            raise ValueError("ElasticNet requires l1_ratio.")
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=float(cfg.l1_ratio),
            C=float(cfg.C),
            max_iter=int(cfg.max_iter),
            class_weight=cfg.class_weight,
            random_state=int(cfg.random_state),
        )
    elif cfg.penalty == "l2":
        model = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            C=float(cfg.C),
            max_iter=int(cfg.max_iter),
            class_weight=cfg.class_weight,
            random_state=int(cfg.random_state),
        )
    else:
        raise ValueError(f"Unknown penalty: {cfg.penalty}")

    model.fit(X, y)
    return model


def predict_proba(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """
    Return P(y=1).
    """
    p = model.predict_proba(X)[:, 1]
    return p


# --- Survival optional: CoxPH using lifelines if installed
def fit_cox_lifelines(X: np.ndarray, time: np.ndarray, event: np.ndarray) -> Any:
    """
    Optional CoxPH baseline using lifelines. If lifelines is not installed, raise a clear error.
    """
    try:
        from lifelines import CoxPHFitter
        import pandas as pd
    except Exception as e:
        raise ImportError(
            "Survival Cox baseline requires `lifelines`. Install it or skip Cox runs."
        ) from e

    df = pd.DataFrame(X)
    df["time"] = time
    df["event"] = event

    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="event")
    return cph


def predict_risk_cox_lifelines(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Return risk scores (higher => higher risk).
    """
    import pandas as pd
    df = pd.DataFrame(X)
    # lifelines returns partial hazard
    risk = model.predict_partial_hazard(df).to_numpy().reshape(-1)
    return risk
