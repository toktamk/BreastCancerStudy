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
# --- Survival optional: CoxPH using lifelines if installed
def fit_cox_lifelines(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    max_features: int = 2000,
    var_floor: float = 1e-12,
    penalizer: float = 1e-2,
    l1_ratio: float = 0.0,
) -> Any:
    """
    CoxPH baseline using lifelines, with feature capping to avoid O(p^2) memory blowups.
    """
    try:
        from lifelines import CoxPHFitter
        import pandas as pd
    except Exception as e:
        raise ImportError(
            "Survival Cox baseline requires `lifelines`. Install it or skip Cox runs."
        ) from e

    X = np.asarray(X)
    p = X.shape[1]

    # 1) drop (near) zero-variance features
    var = X.var(axis=0)
    keep = np.flatnonzero(var > var_floor)

    if keep.size == 0:
        raise ValueError("All survival features have ~zero variance after filtering.")

    # 2) cap to top-K most variable features (if still huge)
    if keep.size > max_features:
        # sort keep by variance, take top-K
        keep = keep[np.argsort(var[keep])[-max_features:]]

    X_small = X[:, keep]

    df = pd.DataFrame(X_small)
    df["time"] = time
    df["event"] = event

    cph = CoxPHFitter(penalizer=float(penalizer), l1_ratio=float(l1_ratio))
    cph.fit(df, duration_col="time", event_col="event")

    # stash which columns we used so predict can match
    cph._selected_cols = keep
    return cph


def predict_risk_cox_lifelines(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Return risk scores (higher => higher risk). Uses the same feature subset as training.
    """
    import pandas as pd
    X = np.asarray(X)

    keep = getattr(model, "_selected_cols", None)
    if keep is not None:
        X = X[:, keep]

    df = pd.DataFrame(X)
    risk = model.predict_partial_hazard(df).to_numpy().reshape(-1)
    return risk

