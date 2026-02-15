from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoxPHConfig:
    """
    Cox proportional hazards baseline using lifelines.

    Notes:
      - This is a strong, interpretable baseline for clinical features.
      - Use penalization (penalizer, l1_ratio) for stability with more features.
    """
    penalizer: float = 0.0
    l1_ratio: float = 0.0  # 0 => ridge, 1 => lasso
    robust: bool = False   # robust variance estimation


@dataclass(frozen=True)
class CoxnetConfig:
    """
    ElasticNet Cox using scikit-survival CoxnetSurvivalAnalysis (optional dependency).
    Great for high-dimensional omics.

    alphas: if None => let model generate path.
    """
    l1_ratio: float = 0.5
    alphas: Optional[np.ndarray] = None
    max_iter: int = 100000
    tol: float = 1e-7


def _require_lifelines() -> Any:
    try:
        import lifelines  # noqa: F401
        from lifelines import CoxPHFitter
        import pandas as pd
        return CoxPHFitter, pd
    except Exception as e:
        raise ImportError(
            "CoxPH requires `lifelines`. Install it with: pip install lifelines"
        ) from e


def _require_sksurv() -> Any:
    try:
        from sksurv.linear_model import CoxnetSurvivalAnalysis
        from sksurv.util import Surv
        return CoxnetSurvivalAnalysis, Surv
    except Exception as e:
        raise ImportError(
            "Coxnet requires `scikit-survival`. Install it with: pip install scikit-survival"
        ) from e


# ---------------------------
# CoxPH (lifelines)
# ---------------------------

def fit_coxph_lifelines(X, t, e, cfg):
    import numpy as np
    import pandas as pd
    from lifelines import CoxPHFitter

    X = pd.DataFrame(X).copy()
    t = pd.Series(t).astype(float).reset_index(drop=True)
    e = pd.Series(e).astype(int).reset_index(drop=True)

    # 1) Ensure numeric and finite
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are entirely NA
    all_na = X.columns[X.isna().all()]
    if len(all_na) > 0:
        X = X.drop(columns=all_na)

    # 2) Drop constant / near-constant columns (prevents std==0 => NaN)
    # We can tune thresholds; these are safe defaults.
    nunique = X.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index
    if len(const_cols) > 0:
        X = X.drop(columns=list(const_cols))

    # Optional: drop near-zero variance columns
    # (variance computed after filling NaNs with column median)
    X_filled = X.copy()
    X_filled = X_filled.apply(lambda c: c.fillna(c.median()) if c.notna().any() else c)
    var = X_filled.var(axis=0)
    nzv_cols = var[var < 1e-12].index
    if len(nzv_cols) > 0:
        X = X.drop(columns=list(nzv_cols))

    # 3) Impute remaining NaNs (lifelines can't handle NaN)
    X = X.apply(lambda c: c.fillna(c.median()) if c.notna().any() else c.fillna(0.0))

    # 4) Build lifelines dataframe
    df = X.reset_index(drop=True)
    df["time"] = t
    df["event"] = e

    # 5) Penalization to handle separation / collinearity
    # If the cfg already has these fields, use them; otherwise set defaults:
    penalizer = getattr(cfg, "penalizer", 0.1)      # try 0.1; if still unstable try 1.0
    l1_ratio = getattr(cfg, "l1_ratio", 0.0)        # 0.0 = ridge, safer than lasso initially

    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(df, duration_col="time", event_col="event", robust=bool(getattr(cfg, "robust", False)))
    return cph



def predict_risk_coxph_lifelines(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Return risk scores (partial hazards), higher => higher risk.
    """
    import pandas as pd
    X = np.asarray(X, dtype=float)
    df = pd.DataFrame(X)
    r = model.predict_partial_hazard(df).to_numpy().reshape(-1)
    return r


def predict_survival_prob_at_horizon_coxph(
    model: Any,
    X: np.ndarray,
    horizon_months: float = 60.0,
) -> np.ndarray:
    """
    Predict P(T > horizon | x) for CoxPH using lifelines predicted survival function.

    Output is survival probability at the specified horizon.
    """
    import pandas as pd
    X = np.asarray(X, dtype=float)
    df = pd.DataFrame(X)

    # lifelines returns survival curves indexed by time
    sf = model.predict_survival_function(df)  # shape: (time_grid, n)
    # Find survival at horizon by interpolation on the time index
    times = sf.index.to_numpy(dtype=float)

    # If horizon outside range, clamp to nearest
    if horizon_months <= times.min():
        return sf.iloc[0].to_numpy(dtype=float)
    if horizon_months >= times.max():
        return sf.iloc[-1].to_numpy(dtype=float)

    # Interpolate per-column
    # sf values are monotone decreasing; linear interpolation is fine for evaluation
    svals = sf.to_numpy(dtype=float)  # [T, N]
    out = np.empty(svals.shape[1], dtype=float)
    for j in range(svals.shape[1]):
        out[j] = np.interp(horizon_months, times, svals[:, j])
    return out


def survival_to_5yr_risk(surv_prob_5yr: np.ndarray) -> np.ndarray:
    """
    Convert survival probability at 5 years to risk probability:
      risk = P(death by 5y) ≈ 1 - P(T > 5y)
    """
    p = np.asarray(surv_prob_5yr, dtype=float)
    return np.clip(1.0 - p, 0.0, 1.0)


# ---------------------------
# Coxnet (scikit-survival)
# ---------------------------

def fit_coxnet_sksurv(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    cfg: CoxnetConfig = CoxnetConfig(),
) -> Any:
    """
    Fit CoxnetSurvivalAnalysis (ElasticNet Cox). Returns fitted model.

    Important: this returns a model that is a path over alphas. We typically tune alpha on validation.
    """
    CoxnetSurvivalAnalysis, Surv = _require_sksurv()

    X = np.asarray(X, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)

    y = Surv.from_arrays(event=event, time=time)
    model = CoxnetSurvivalAnalysis(
        l1_ratio=float(cfg.l1_ratio),
        alphas=cfg.alphas,
        max_iter=int(cfg.max_iter),
        tol=float(cfg.tol),
    )
    model.fit(X, y)
    return model


def select_alpha_by_val_cindex(
    model: Any,
    X_val: np.ndarray,
    time_val: np.ndarray,
    event_val: np.ndarray,
) -> Tuple[Any, float, int]:
    """
    Coxnet produces coefficients for multiple alphas. We choose alpha index that maximizes C-index on validation.

    Returns:
      (selected_model, best_cindex, best_alpha_index)

    Implementation detail:
      We create a shallow wrapper carrying the chosen coef_.
    """
    from src.eval.survival_eval import harrell_c_index

    X_val = np.asarray(X_val, dtype=float)
    t = np.asarray(time_val, dtype=float)
    e = np.asarray(event_val, dtype=int)

    # model.coef_ is shape (n_features, n_alphas)
    coefs = model.coef_
    if coefs.ndim != 2:
        raise ValueError("Unexpected coef_ shape from Coxnet model.")

    best = -np.inf
    best_k = 0
    for k in range(coefs.shape[1]):
        risk = X_val @ coefs[:, k]
        c = harrell_c_index(t, e, risk)
        if np.isfinite(c) and c > best:
            best = c
            best_k = k

    class _CoxnetSelected:
        def __init__(self, base, k):
            self.base = base
            self.k = k
            self.coef_ = base.coef_[:, k]

    return _CoxnetSelected(model, best_k), float(best), int(best_k)


def predict_risk_coxnet(selected_model: Any, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    coef = np.asarray(selected_model.coef_, dtype=float).reshape(-1)
    return (X @ coef).reshape(-1)


def fit_cox_model(X: np.ndarray, time: np.ndarray, event: np.ndarray):
    """
    Minimal Cox wrapper.
    Replace internals with the existing Cox fitter if already present.
    """
    try:
        from lifelines import CoxPHFitter
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise ImportError("Cox modeling requires lifelines: pip install lifelines") from e

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["time"] = time.astype(float)
    df["event"] = event.astype(int)

    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(df, duration_col="time", event_col="event")
    return cph


def predict_risk_cox(model, X: np.ndarray) -> np.ndarray:
    """
    Returns a risk score (higher => higher risk).
    """
    import pandas as pd
    df = pd.DataFrame(X, columns=model.params_.index.tolist())
    # lifelines: partial_hazard is exp(beta x)
    ph = model.predict_partial_hazard(df).to_numpy().reshape(-1)
    return ph.astype(float)

