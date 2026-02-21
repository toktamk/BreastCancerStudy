"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeDependentResult:
    horizons_months: np.ndarray
    auc: np.ndarray
    auc_mean: float
    ibs: Optional[float]


def _require_sksurv() -> None:
    try:
        import sksurv  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Time-dependent survival evaluation requires scikit-survival.\n"
            "Install (one option): pip install scikit-survival\n"
            "If the code is running on Windows, conda-forge is often the easiest route:\n"
            "  conda install -c conda-forge scikit-survival\n"
        ) from e


def _to_sksurv_y(time: np.ndarray, event: np.ndarray):
    _require_sksurv()
    from sksurv.util import Surv
    return Surv.from_arrays(event.astype(bool), time.astype(float))


def time_dependent_auc_and_ibs(
    *,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    test_risk_score: np.ndarray,
    horizons_months: Iterable[float],
    # optional: survival probability predictions for IBS if model provides them
    test_surv_fn=None,
) -> TimeDependentResult:
    """
    Computes:
    - cumulative_dynamic_auc at supplied horizons (months)
    - optional IBS if survival functions are available

    Risk score convention:
    - Higher score => higher risk (earlier event).
    """
    _require_sksurv()
    from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score

    horizons = np.asarray(list(horizons_months), dtype=float)
    if horizons.ndim != 1 or len(horizons) == 0:
        raise ValueError("horizons_months must be a non-empty 1D iterable")

    y_train = _to_sksurv_y(train_time, train_event)
    y_test = _to_sksurv_y(test_time, test_event)

    # AUC(t)
    auc_t, auc_mean = cumulative_dynamic_auc(
        y_train,
        y_test,
        test_risk_score.astype(float),
        horizons,
    )

    # IBS requires survival probability estimates over time for each test sample.
    # If the model provides a survival function per patient, pass test_surv_fn.
    ibs = None
    if test_surv_fn is not None:
        # test_surv_fn should return survival probs at given times for each sample.
        # We expect callable(times)->(n_samples, n_times) OR already computed array.
        if callable(test_surv_fn):
            surv_probs = test_surv_fn(horizons)
        else:
            surv_probs = np.asarray(test_surv_fn)

        if surv_probs.shape != (len(test_time), len(horizons)):
            raise ValueError(
                f"surv_probs must have shape (n_test, n_times)=({len(test_time)}, {len(horizons)}), "
                f"got {surv_probs.shape}"
            )

        ibs = integrated_brier_score(y_train, y_test, surv_probs, horizons)

    return TimeDependentResult(
        horizons_months=horizons,
        auc=np.asarray(auc_t, dtype=float),
        auc_mean=float(auc_mean),
        ibs=float(ibs) if ibs is not None else None,
    )
