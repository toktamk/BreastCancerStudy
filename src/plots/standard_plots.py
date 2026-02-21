"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

from src.eval.decision_curve import DecisionCurve


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, outpath: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, outpath: Path) -> None:
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, outpath: Path, n_bins: int = 10) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    plt.figure()
    plt.plot(mean_pred, frac_pos)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Event Fraction")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_decision_curve(dca_df, outpath: Path) -> None:
    plt.figure()
    plt.plot(dca_df["threshold"], dca_df["net_benefit_model"], label="Model")
    plt.plot(dca_df["threshold"], dca_df["net_benefit_all"], label="Treat all")
    plt.plot(dca_df["threshold"], dca_df["net_benefit_none"], label="Treat none")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_km_by_risk_group(
    *,
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    outpath: Path,
    n_bins: int = 3,
) -> None:
    try:
        from lifelines import KaplanMeierFitter
    except Exception as e:  # pragma: no cover
        raise ImportError("KM plot requires lifelines: pip install lifelines") from e

    kmf = KaplanMeierFitter()

    # Quantile bins by risk
    qs = np.quantile(risk, np.linspace(0, 1, n_bins + 1))
    # make bins stable
    qs[0] -= 1e-9
    groups = np.digitize(risk, qs[1:-1], right=True)  # 0..n_bins-1

    plt.figure()
    for g in range(n_bins):
        m = (groups == g)
        label = f"Risk bin {g+1}/{n_bins}"
        kmf.fit(time[m], event_observed=event[m], label=label)
        kmf.plot_survival_function(ci_show=False)

    plt.xlabel("Time (months)")
    plt.ylabel("Survival probability")
    plt.title("Kaplan–Meier by Predicted Risk Strata")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
