from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) using equal-width bins over [0,1].
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)

    return float(ece)


def calibration_summary(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calibration-in-the-large and slope (via simple linear regression of logit(p) -> y),
    plus Brier and ECE.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(y_prob).astype(float), 1e-6, 1 - 1e-6)

    brier = float(brier_score_loss(y_true, p))
    ece = float(ece_score(y_true, p, n_bins=10))

    # Calibration-in-the-large: mean(p) vs mean(y)
    cil = float(p.mean() - y_true.mean())

    # Calibration slope: regress y on logit(p) using least squares
    logit = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(logit), logit])
    beta, *_ = np.linalg.lstsq(X, y_true, rcond=None)
    slope = float(beta[1])

    return {"brier": brier, "ece": ece, "calibration_in_the_large": cil, "calibration_slope": slope}


def discrimination_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """
    Harrell's concordance index for right-censored survival data.
    risk: higher => higher risk (shorter survival expected).
    """
    t = np.asarray(time).astype(float)
    e = np.asarray(event).astype(int)
    r = np.asarray(risk).astype(float)

    # Comparable pairs: i has event and t_i < t_j
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


def group_metrics_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group: pd.Series,
    min_group_n: int = 30,
) -> Dict[str, Any]:
    """
    Fairness-style reporting: compute AUROC/AUPRC/Brier/ECE per group (if enough samples),
    plus worst-group AUROC.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    g = pd.Series(group).astype("object").fillna("NA")

    out: Dict[str, Any] = {"per_group": {}, "worst_group_auroc": None}

    worst = None
    for gv, idx in g.groupby(g).groups.items():
        idx = np.array(list(idx), dtype=int)
        if len(idx) < min_group_n:
            continue
        yt = y_true[idx]
        yp = y_prob[idx]
        try:
            disc = discrimination_metrics(yt, yp)
            cal = calibration_summary(yt, yp)
            out["per_group"][str(gv)] = {**disc, **cal, "n": int(len(idx))}
            worst = disc["auroc"] if worst is None else min(worst, disc["auroc"])
        except Exception:
            # Some groups may be single-class; skip
            continue

    out["worst_group_auroc"] = worst
    return out


def bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 1337,
    alpha: float = 0.05,
    max_attempts: int = 5000,
) -> Dict[str, float]:
    """
    Generic bootstrap CI on a scalar metric.

    Robustness: some bootstrap samples may be degenerate (e.g., single-class y_true),
    making metrics like AUROC undefined. We skip those draws.

    Returns median and percentile CI. If too few valid draws exist, returns NaNs.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    stats: list[float] = []
    n = len(y_true)

    attempts = 0
    while len(stats) < n_boot and attempts < max_attempts:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        try:
            val = metric_fn(y_true[idx], y_pred[idx])
        except Exception:
            continue
        if np.isfinite(val):
            stats.append(float(val))

    if len(stats) < max(30, int(0.2 * n_boot)):
        # Not enough stable draws to form a meaningful CI
        return {"median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_valid": int(len(stats))}

    stats = np.sort(np.asarray(stats, dtype=float))
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    mid = float(np.median(stats))
    return {"median": mid, "ci_low": lo, "ci_high": hi, "n_valid": int(len(stats))}



def evaluate_binary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Full evaluation bundle for 5-year risk prediction.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    out: Dict[str, Any] = {}
    out["discrimination"] = discrimination_metrics(y_true, y_prob)
    out["calibration"] = calibration_summary(y_true, y_prob)

    # Bootstrap CI for AUROC
    out["robustness"] = {
        "auroc_bootstrap": bootstrap_ci(
            lambda yt, yp: roc_auc_score(yt.astype(int), yp.astype(float)),
            y_true,
            y_prob,
            n_boot=300,  # keep runtime manageable; increase in final runs
        )
    }

    if group is not None:
        out["fairness"] = group_metrics_binary(y_true, y_prob, group=group, min_group_n=30)

    # Calibration curve data (for plotting later)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    out["calibration_curve"] = {"mean_pred": mean_pred.tolist(), "frac_pos": frac_pos.tolist()}

    return out
