from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from scripts.sanity_check_calibration import calibration_slope_intercept, compute_ece  

@dataclass
class GroupMetrics:
    group_col: str
    group_value: str
    n: int
    prevalence: float
    mean_pred: float
    auroc: float
    auprc: float
    brier: float
    intercept: float
    slope: float
    ece: float

def group_metrics(
    df: pd.DataFrame,
    y_col: str,
    p_col: str,
    group_col: str,
    n_bins: int = 10,
    min_n: int = 30,
) -> List[GroupMetrics]:
    out: List[GroupMetrics] = []
    for gv, gdf in df.groupby(group_col, dropna=False):
        if len(gdf) < min_n:
            continue
        y = gdf[y_col].to_numpy()
        p = gdf[p_col].to_numpy()
        m = np.isfinite(y) & np.isfinite(p)
        y = y[m].astype(int)
        p = p[m].astype(float)
        if len(y) < min_n or len(np.unique(y)) < 2:
            continue

        auroc = float(roc_auc_score(y, p))
        auprc = float(average_precision_score(y, p))
        brier = float(brier_score_loss(y, p))
        a, b = calibration_slope_intercept(y, p)  
        ece_out = compute_ece(y, p, n_bins=n_bins)
        ece = float(ece_out[0]) if isinstance(ece_out, tuple) else float(ece_out)

        out.append(GroupMetrics(
            group_col=group_col,
            group_value=str(gv),
            n=int(len(y)),
            prevalence=float(y.mean()),
            mean_pred=float(p.mean()),
            auroc=auroc,
            auprc=auprc,
            brier=brier,
            intercept=float(a),
            slope=float(b),
            ece=ece,
        ))
    return out

@dataclass
class ThresholdParity:
    group_col: str
    group_value: str
    n: int
    threshold: float
    tpr: float
    fpr: float
    ppv: float
    pred_pos_rate: float

def threshold_parity(
    df: pd.DataFrame,
    y_col: str,
    p_col: str,
    group_col: str,
    threshold: float = 0.2,
    min_n: int = 30,
) -> List[ThresholdParity]:
    out: List[ThresholdParity] = []
    for gv, gdf in df.groupby(group_col, dropna=False):
        if len(gdf) < min_n:
            continue
        y = gdf[y_col].to_numpy()
        p = gdf[p_col].to_numpy()
        m = np.isfinite(y) & np.isfinite(p)
        y = y[m].astype(int)
        p = p[m].astype(float)
        if len(y) < min_n:
            continue

        pred = (p >= threshold).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())

        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) else float("nan")

        out.append(ThresholdParity(
            group_col=group_col,
            group_value=str(gv),
            n=int(len(y)),
            threshold=float(threshold),
            tpr=float(tpr),
            fpr=float(fpr),
            ppv=float(ppv),
            pred_pos_rate=float(pred.mean()),
        ))
    return out
