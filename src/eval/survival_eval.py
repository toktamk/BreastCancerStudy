from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """
    Harrell's C-index for right-censored data.
    risk: higher => higher risk (shorter survival expected).
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    r = np.asarray(risk, dtype=float)

    n_conc = 0.0
    n_tied = 0.0
    n_comp = 0.0

    for i in range(len(t)):
        if e[i] != 1:
            continue
        for j in range(len(t)):
            if t[i] < t[j]:
                n_comp += 1.0
                if r[i] > r[j]:
                    n_conc += 1.0
                elif r[i] == r[j]:
                    n_tied += 1.0

    if n_comp == 0:
        return float("nan")
    return float((n_conc + 0.5 * n_tied) / n_comp)


def _require_sksurv():
    try:
        from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score
        from sksurv.util import Surv
        return cumulative_dynamic_auc, integrated_brier_score, Surv
    except Exception as e:
        raise ImportError(
            "Time-dependent AUC / IBS requires scikit-survival. Install with: pip install scikit-survival"
        ) from e


def survival_metrics_basic(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
) -> Dict[str, float]:
    """
    Minimal survival metrics that require no special deps.
    """
    return {"harrell_c_index": float(harrell_c_index(time, event, risk))}


def survival_metrics_sksurv(
    time_train: np.ndarray,
    event_train: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
    risk_test: np.ndarray,
    horizons: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Time-dependent AUC and Integrated Brier Score using scikit-survival.

    horizons: time grid for AUC evaluation. If None, auto choose quantiles of test time.
    """
    cumulative_dynamic_auc, integrated_brier_score, Surv = _require_sksurv()

    ttr = np.asarray(time_train, dtype=float)
    etr = np.asarray(event_train, dtype=bool)
    tte = np.asarray(time_test, dtype=float)
    ete = np.asarray(event_test, dtype=bool)
    risk = np.asarray(risk_test, dtype=float)

    y_train = Surv.from_arrays(event=etr, time=ttr)
    y_test = Surv.from_arrays(event=ete, time=tte)

    if horizons is None:
        # choose reasonable horizon points within observed range
        qs = np.array([0.25, 0.50, 0.75], dtype=float)
        horizons = np.quantile(tte, qs)
        horizons = np.unique(np.clip(horizons, 1.0, float(np.max(tte))))

    # cumulative_dynamic_auc expects "risk scores" and returns auc at times
    times, aucs = cumulative_dynamic_auc(y_train, y_test, risk, horizons)

    # For IBS, we need predicted survival probabilities over a time grid.
    # scikit-survival IBS is typically used with models that can predict survival functions.
    # If we only have risk scores, we cannot compute IBS directly without a survival model object.
    # Here we expose time-dependent AUC, and leave IBS for models providing survival functions.
    return {
        "time_dependent_auc": {"times": times.tolist(), "auc": aucs.tolist()},
    }
