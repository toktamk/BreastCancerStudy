#!/usr/bin/env python3
"""
sanity_check_calibration.py

Purpose:
  Diagnose why calibration is bad by running sanity checks on:
   - label/horizon consistency (including censoring before horizon)
   - probability invariants (range, NaNs, monotonicity, survival->risk mapping)
   - split integrity (optional)
   - calibration diagnostics (calibration-in-the-large, slope/intercept, ECE, brier, AUROC)
   - stratified diagnostics by subgroup columns (optional)

Works with:
  A) Survival-model prediction files like scripts/run_survival_models.py produces:
      preds/test.parquet with columns:
        patient_id, risk_score, survival_prob_horizon, risk_prob_horizon
  B) Binary prediction files (csv/parquet) with columns:
        patient_id, y_true, p_pred   (or configurable via args)

It also supports joining predictions to a cohort file that contains:
  patient_id, time_months, event, y60 (or configurable label col)
so we can validate that y60 is consistent with (time,event) and horizon rules.

Examples:

  # Survival preds + cohort sanity checks at 60 months
  python sanity_check_calibration.py \
    --cohort data/processed/cohort.parquet \
    --preds runs/survival/coxph/preds/test.parquet \
    --mode survival \
    --horizon 60 \
    --time-col time_months --event-col event --label-col y60 \
    --group-cols age_group,ethnicity

  # Binary preds already include y_true and p_pred
  python sanity_check_calibration.py \
    --preds runs/binary/baseline/preds/test.csv \
    --mode binary \
    --y-col y_true --p-col p_pred

  # Check split files (train/val/test CSV with patient_id column)
  python sanity_check_calibration.py \
    --cohort data/processed/cohort.parquet \
    --preds runs/survival/coxph/preds/test.parquet \
    --mode survival \
    --splits-dir runs/survival/coxph/splits
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    log_loss,
)
from sklearn.linear_model import LogisticRegression


# -------------------------
# Utilities
# -------------------------

def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if ext in [".csv"]:
        return pd.read_csv(path)
    if ext in [".feather"]:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported file extension: {ext} for {path}")

def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def clip_prob(p: pd.Series, eps: float = 1e-6) -> pd.Series:
    return p.clip(eps, 1.0 - eps)

def logit(p: pd.Series, eps: float = 1e-6) -> pd.Series:
    p = clip_prob(p, eps=eps)
    return np.log(p / (1.0 - p))

def fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{100.0 * x:.2f}%"

def header(title: str) -> None:
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))

def subheader(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def info(msg: str) -> None:
    print(f"[INFO] {msg}")

def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")

def ok(msg: str) -> None:
    print(f"[OK] {msg}")


# -------------------------
# Label logic
# -------------------------

def recompute_fixed_horizon_label(
    time_months: pd.Series,
    event: pd.Series,
    horizon_months: float
) -> pd.Series:
    """
    Recompute y_horizon from (time,event) with typical fixed-horizon rules:
      - event==1 and time < horizon  => 1
      - time >= horizon and event==0 => 0  (known event-free at horizon)
      - censored before horizon (event==0 and time < horizon) => NaN (undefined)
      - edge cases:
          * event==1 and time >= horizon => 0 for event-by-horizon definition
            (event occurred after horizon)
    """
    t = safe_float_series(time_months)
    e = safe_float_series(event)

    y = pd.Series(np.nan, index=t.index, dtype="float64")

    # Event by horizon
    y[(e == 1) & (t < horizon_months)] = 1.0

    # Known event-free at or beyond horizon
    # If event==0 and t >= horizon => 0
    y[(e == 0) & (t >= horizon_months)] = 0.0

    # If event==1 but after horizon => 0 for "event-by-horizon" label
    y[(e == 1) & (t >= horizon_months)] = 0.0

    # Censored before horizon stays NaN
    return y


# -------------------------
# Calibration diagnostics
# -------------------------

@dataclass
class CalibrationStats:
    n: int
    prevalence: float
    mean_pred: float
    brier: float
    logloss: Optional[float]
    auroc: Optional[float]
    auprc: Optional[float]
    calib_intercept: Optional[float]
    calib_slope: Optional[float]
    ece: Optional[float]

def compute_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> Tuple[float, pd.DataFrame]:
    """
    Expected Calibration Error (ECE) with equal-width bins over [0,1].
    Returns: (ece, per_bin_df)
    """
    df = pd.DataFrame({"y": y, "p": p}).copy()
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    agg = df.groupby("bin", observed=True).agg(
        n=("y", "size"),
        mean_p=("p", "mean"),
        mean_y=("y", "mean"),
    ).reset_index()
    agg["abs_gap"] = (agg["mean_p"] - agg["mean_y"]).abs()
    total = agg["n"].sum()
    if total == 0:
        return float("nan"), agg
    ece = float((agg["n"] / total * agg["abs_gap"]).sum())
    return ece, agg

def calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate calibration intercept and slope via logistic regression:
      y ~ a + b * logit(p)

    Interpreting:
      b < 1 => predictions too extreme (overconfident)
      b > 1 => predictions too timid (underconfident)
      a != 0 => calibration-in-the-large error
    """
    # Need both classes
    if len(np.unique(y)) < 2:
        return None, None

    lp = logit(pd.Series(p)).to_numpy().reshape(-1, 1)
    # Use near-unregularized LR; fallback if solver complains.
    lr = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=5000)

    lr.fit(lp, y)
    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0, 0])
    return intercept, slope

def compute_binary_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> CalibrationStats:
    y = y.astype(float)
    p = p.astype(float)

    n = int(len(y))
    prevalence = float(np.mean(y)) if n else float("nan")
    mean_pred = float(np.mean(p)) if n else float("nan")

    brier = float(brier_score_loss(y, p)) if n else float("nan")

    # logloss requires both classes and valid probs
    logloss = None
    auroc = None
    auprc = None
    calib_int = None
    calib_slope = None
    ece = None

    if n and len(np.unique(y)) == 2:
        p_clip = clip_prob(pd.Series(p)).to_numpy()
        logloss = float(log_loss(y, p_clip))
        auroc = float(roc_auc_score(y, p))
        auprc = float(average_precision_score(y, p))
        calib_int, calib_slope = calibration_slope_intercept(y, p)
        ece, _ = compute_ece(y, p, n_bins=n_bins)
    else:
        warn("Binary metrics that require both classes (logloss/AUROC/AUPRC/slope/ECE) were skipped (only one class present).")

    return CalibrationStats(
        n=n,
        prevalence=prevalence,
        mean_pred=mean_pred,
        brier=brier,
        logloss=logloss,
        auroc=auroc,
        auprc=auprc,
        calib_intercept=calib_int,
        calib_slope=calib_slope,
        ece=ece,
    )


# -------------------------
# Split integrity
# -------------------------

def read_split_ids(splits_dir: str, id_col: str) -> Dict[str, pd.Index]:
    out: Dict[str, pd.Index] = {}
    for name in ["train", "val", "test"]:
        p_csv = os.path.join(splits_dir, f"{name}.csv")
        if os.path.exists(p_csv):
            df = pd.read_csv(p_csv)
            if id_col not in df.columns:
                raise ValueError(f"Split file {p_csv} missing id_col={id_col}. Columns: {list(df.columns)}")
            out[name] = pd.Index(df[id_col].astype(str))
        else:
            warn(f"Split file not found: {p_csv}")
    return out

def check_splits(splits: Dict[str, pd.Index], cohort_ids: Optional[pd.Index] = None) -> Dict[str, object]:
    report: Dict[str, object] = {}
    keys = [k for k in ["train", "val", "test"] if k in splits]
    for k in keys:
        report[f"{k}_n"] = int(len(splits[k]))
        report[f"{k}_unique_n"] = int(splits[k].nunique())

    # overlaps
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            overlap = splits[a].intersection(splits[b])
            report[f"overlap_{a}_{b}"] = int(len(overlap))

    # membership in cohort
    if cohort_ids is not None:
        for k in keys:
            missing = splits[k].difference(cohort_ids)
            report[f"{k}_missing_from_cohort"] = int(len(missing))

    return report


# -------------------------
# Core checks
# -------------------------

def check_probability_invariants_survival(preds: pd.DataFrame, eps: float = 1e-6) -> Dict[str, object]:
    """
    Expects columns: risk_score, survival_prob_horizon, risk_prob_horizon (names configurable upstream).
    """
    rs = safe_float_series(preds["risk_score"])
    s = safe_float_series(preds["survival_prob_horizon"])
    r = safe_float_series(preds["risk_prob_horizon"])

    out: Dict[str, object] = {}

    out["n"] = int(len(preds))
    out["survival_nan"] = int(s.isna().sum())
    out["risk_nan"] = int(r.isna().sum())
    out["risk_score_nan"] = int(rs.isna().sum())

    out["survival_out_of_range"] = int(((s < -eps) | (s > 1 + eps)).sum())
    out["risk_out_of_range"] = int(((r < -eps) | (r > 1 + eps)).sum())

    # risk should be 1 - survival
    diff = (r - (1.0 - s)).abs()
    out["risk_vs_1_minus_survival_mean_abs_diff"] = float(diff.mean(skipna=True))
    out["risk_vs_1_minus_survival_p99_abs_diff"] = float(diff.quantile(0.99))

    # monotonicity expectation
    tmp = pd.DataFrame({"risk_score": rs, "survival": s, "risk": r}).dropna()
    if len(tmp) > 10:
        out["spearman_risk_score_vs_risk_prob"] = float(tmp["risk_score"].corr(tmp["risk"], method="spearman"))
        out["spearman_risk_score_vs_survival_prob"] = float(tmp["risk_score"].corr(tmp["survival"], method="spearman"))
    else:
        out["spearman_risk_score_vs_risk_prob"] = None
        out["spearman_risk_score_vs_survival_prob"] = None

    out["risk_prob_min"] = float(r.min(skipna=True))
    out["risk_prob_max"] = float(r.max(skipna=True))
    out["risk_prob_mean"] = float(r.mean(skipna=True))
    out["survival_prob_min"] = float(s.min(skipna=True))
    out["survival_prob_max"] = float(s.max(skipna=True))
    out["survival_prob_mean"] = float(s.mean(skipna=True))

    return out

def join_preds_to_cohort(
    cohort: pd.DataFrame,
    preds: pd.DataFrame,
    id_col: str
) -> pd.DataFrame:
    c = cohort.copy()
    p = preds.copy()
    c[id_col] = c[id_col].astype(str)
    p[id_col] = p[id_col].astype(str)
    joined = p.merge(c, on=id_col, how="left", validate="many_to_one")
    return joined

def check_label_consistency(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    label_col: str,
    horizon: float
) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if time_col not in df.columns or event_col not in df.columns or label_col not in df.columns:
        warn(f"Cannot run label consistency check (need {time_col}, {event_col}, {label_col}).")
        return out

    t = safe_float_series(df[time_col])
    e = safe_float_series(df[event_col])
    y = safe_float_series(df[label_col])

    y_re = recompute_fixed_horizon_label(t, e, horizon)

    # stats
    out["n_total"] = int(len(df))
    out["n_label_defined"] = int(y.notna().sum())
    out["n_label_undefined"] = int(y.isna().sum())

    cens_before = ((e == 0) & (t < horizon))
    out["n_censored_before_horizon"] = int(cens_before.sum())

    # mismatches only where both defined
    both = y.notna() & y_re.notna()
    mism = (y[both] != y_re[both])
    out["n_both_defined"] = int(both.sum())
    out["n_mismatch"] = int(mism.sum())
    out["mismatch_rate_over_both_defined"] = float(mism.mean()) if both.sum() else float("nan")

    # suspicious: censored-before-horizon but label is finite
    susp = cens_before & y.notna()
    out["n_censored_before_horizon_with_finite_label"] = int(susp.sum())

    # show a few examples (indices only; keep lightweight)
    ex_idx = df.index[susp].tolist()[:10]
    out["example_indices_censored_but_labeled"] = ex_idx

    return out

def extract_binary_y_p(
    df: pd.DataFrame,
    y_col: str,
    p_col: str
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    y = safe_float_series(df[y_col])
    p = safe_float_series(df[p_col])

    mask = y.notna() & p.notna()
    y = y[mask].astype(int)
    p = p[mask].astype(float)

    # keep only {0,1}
    mask2 = y.isin([0, 1])
    if (~mask2).any():
        warn(f"Dropping {(~mask2).sum()} rows where {y_col} not in {{0,1}}.")
    y = y[mask2].to_numpy()
    p = p[mask2].to_numpy()
    df_used = df.loc[mask.index[mask][mask2], :].copy()
    return y, p, df_used

def print_calibration_summary(stats: CalibrationStats) -> None:
    print(f"n = {stats.n}")
    print(f"prevalence mean(y) = {stats.prevalence:.6f}")
    print(f"mean predicted = {stats.mean_pred:.6f}")
    print(f"Brier = {stats.brier:.6f}")
    if stats.logloss is not None:
        print(f"LogLoss = {stats.logloss:.6f}")
    if stats.auroc is not None:
        print(f"AUROC = {stats.auroc:.6f}")
    if stats.auprc is not None:
        print(f"AUPRC = {stats.auprc:.6f}")
    if stats.calib_intercept is not None and stats.calib_slope is not None:
        print(f"Calibration intercept (a) = {stats.calib_intercept:.6f}")
        print(f"Calibration slope (b) = {stats.calib_slope:.6f}")
    if stats.ece is not None:
        print(f"ECE = {stats.ece:.6f}")

def print_bin_table(y: np.ndarray, p: np.ndarray, n_bins: int) -> None:
    ece, tbl = compute_ece(y, p, n_bins=n_bins)
    subheader(f"Reliability bins (ECE={ece:.6f}, bins={n_bins})")
    if len(tbl) == 0:
        print("(no data)")
        return
    # pretty print
    tbl2 = tbl.copy()
    tbl2["bin"] = tbl2["bin"].astype(str)
    tbl2["mean_p"] = tbl2["mean_p"].round(6)
    tbl2["mean_y"] = tbl2["mean_y"].round(6)
    tbl2["abs_gap"] = tbl2["abs_gap"].round(6)
    print(tbl2.to_string(index=False))

def run_grouped_metrics(
    df_used: pd.DataFrame,
    y_col: str,
    p_col: str,
    group_cols: List[str],
    n_bins: int,
    min_group_n: int = 100
) -> Dict[str, object]:
    """
    Compute metrics per subgroup value for each group_col.
    Returns JSON-serializable summary.
    """
    out: Dict[str, object] = {}
    for g in group_cols:
        if g not in df_used.columns:
            warn(f"group col not found: {g}")
            continue
        groups = df_used.groupby(g, dropna=False)
        rows = []
        for val, sub in groups:
            y, p, _ = extract_binary_y_p(sub, y_col=y_col, p_col=p_col)
            if len(y) < min_group_n:
                continue
            stats = compute_binary_metrics(y, p, n_bins=n_bins)
            rows.append({
                "group": g,
                "value": None if (pd.isna(val)) else str(val),
                **asdict(stats),
            })
        out[g] = rows
    return out


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Predictions file (csv/parquet).")
    ap.add_argument("--cohort", default=None, help="Cohort file with time/event/label columns (csv/parquet). Optional but recommended.")
    ap.add_argument("--mode", choices=["survival", "binary"], required=True, help="Which prediction format is provided.")
    ap.add_argument("--horizon", type=float, default=60.0, help="Horizon in months for fixed-horizon label checks.")
    ap.add_argument("--id-col", default="patient_id")

    # Cohort columns
    ap.add_argument("--time-col", default="time_months")
    ap.add_argument("--event-col", default="event")
    ap.add_argument("--label-col", default="y60", help="Fixed-horizon binary label column in cohort (if present).")

    # Binary preds columns (mode=binary)
    ap.add_argument("--y-col", default="y_true", help="Name of true label column in preds/cohort-joined dataframe.")
    ap.add_argument("--p-col", default="p_pred", help="Name of predicted probability column in preds/cohort-joined dataframe.")

    # Survival preds columns (mode=survival)
    ap.add_argument("--risk-score-col", default="risk_score")
    ap.add_argument("--survival-prob-col", default="survival_prob_horizon")
    ap.add_argument("--risk-prob-col", default="risk_prob_horizon")

    # Optional split checks
    ap.add_argument("--splits-dir", default=None, help="Directory containing train.csv/val.csv/test.csv with id-col.")
    ap.add_argument("--n-bins", type=int, default=10, help="Bins for ECE/reliability table.")
    ap.add_argument("--group-cols", default="", help="Comma-separated subgroup columns to stratify calibration (must exist after join).")
    ap.add_argument("--min-group-n", type=int, default=100)

    ap.add_argument("--report-json", default=None, help="If set, write a JSON report to this path.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    header("Load inputs")
    preds = read_table(args.preds)
    info(f"preds: {args.preds}  shape={preds.shape}")

    cohort = None
    if args.cohort:
        cohort = read_table(args.cohort)
        info(f"cohort: {args.cohort}  shape={cohort.shape}")

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]

    report: Dict[str, object] = {
        "args": vars(args),
        "files": {"preds": args.preds, "cohort": args.cohort, "splits_dir": args.splits_dir},
        "checks": {},
    }

    # Basic column existence
    if args.id_col not in preds.columns:
        raise ValueError(f"Preds missing id_col={args.id_col}. Columns={list(preds.columns)}")

    # Split checks (optional)
    if args.splits_dir:
        header("Split integrity checks")
        splits = read_split_ids(args.splits_dir, id_col=args.id_col)
        cohort_ids = None
        if cohort is not None and args.id_col in cohort.columns:
            cohort_ids = pd.Index(cohort[args.id_col].astype(str))
        split_report = check_splits(splits, cohort_ids=cohort_ids)

        for k, v in split_report.items():
            print(f"{k}: {v}")

        # highlight overlaps/missing
        for k, v in split_report.items():
            if "overlap_" in k and v != 0:
                fail(f"{k} = {v} (non-zero overlap!)")
            if "missing_from_cohort" in k and v != 0:
                warn(f"{k} = {v} (split ids not present in current cohort)")

        report["checks"]["splits"] = split_report

    # Join preds to cohort if possible
    if cohort is not None:
        header("Join predictions to cohort")
        joined = join_preds_to_cohort(cohort, preds, id_col=args.id_col)
        missing = joined[args.time_col].isna().mean() if args.time_col in joined.columns else None
        info(f"joined shape={joined.shape}")
        if missing is not None:
            info(f"fraction missing {args.time_col} after join: {missing:.4f}")
        report["checks"]["join"] = {
            "joined_shape": list(joined.shape),
            "fraction_missing_time_col": missing,
        }
    else:
        joined = preds.copy()

    # Label/horizon sanity checks (only if cohort columns exist)
    header("Label / horizon consistency checks")
    label_report = check_label_consistency(
        joined,
        time_col=args.time_col,
        event_col=args.event_col,
        label_col=args.label_col,
        horizon=args.horizon,
    )
    if label_report:
        for k, v in label_report.items():
            print(f"{k}: {v}")
        if label_report.get("n_censored_before_horizon_with_finite_label", 0) > 0:
            fail("Some censored-before-horizon rows have finite labels. This will bias calibration.")
        if (label_report.get("mismatch_rate_over_both_defined") is not None and
            np.isfinite(label_report.get("mismatch_rate_over_both_defined", np.nan)) and
            label_report.get("mismatch_rate_over_both_defined", 0.0) > 0.001):
            warn("Stored label differs from recomputed horizon label. Investigate label generation logic.")
    else:
        warn("Skipped label consistency checks (missing required columns).")
    report["checks"]["label_consistency"] = label_report

    # Mode-specific checks and calibration metrics
    if args.mode == "survival":
        header("Survival prediction invariants")
        # rename if needed
        for col, argname in [
            (args.risk_score_col, "risk_score"),
            (args.survival_prob_col, "survival_prob_horizon"),
            (args.risk_prob_col, "risk_prob_horizon"),
        ]:
            if col not in joined.columns:
                raise ValueError(f"Missing survival preds column '{col}' in joined dataframe.")
        df_surv = joined.rename(columns={
            args.risk_score_col: "risk_score",
            args.survival_prob_col: "survival_prob_horizon",
            args.risk_prob_col: "risk_prob_horizon",
        }).copy()

        inv = check_probability_invariants_survival(df_surv)
        for k, v in inv.items():
            print(f"{k}: {v}")

        if inv["survival_out_of_range"] > 0 or inv["risk_out_of_range"] > 0:
            fail("Probabilities out of [0,1]. Fix survival->risk mapping.")
        if inv["risk_vs_1_minus_survival_p99_abs_diff"] > 1e-3:
            warn("risk_prob_horizon is not ~ 1 - survival_prob_horizon (large discrepancy).")

        if inv.get("spearman_risk_score_vs_risk_prob") is not None:
            if inv["spearman_risk_score_vs_risk_prob"] < 0.2:
                warn("Weak monotonicity between risk_score and risk_prob_horizon. Check sign/transform.")
            if inv["spearman_risk_score_vs_survival_prob"] > -0.2:
                warn("Weak negative monotonicity between risk_score and survival_prob_horizon. Check sign/transform.")

        report["checks"]["survival_invariants"] = inv

        # If we have a usable binary label in the joined frame, compute calibration on risk_prob_horizon
        if args.label_col in df_surv.columns:
            subheader(f"Calibration metrics using label_col={args.label_col} vs risk_prob_horizon")
            # Use only finite labels (exclude censored-before-horizon if label is NaN)
            tmp = df_surv[[args.label_col, "risk_prob_horizon"]].copy()
            tmp = tmp.rename(columns={args.label_col: "y_true", "risk_prob_horizon": "p_pred"})
            y, p, df_used = extract_binary_y_p(tmp, y_col="y_true", p_col="p_pred")

            if len(y) == 0:
                warn("No valid (y,p) pairs found for calibration metrics.")
            else:
                stats = compute_binary_metrics(y, p, n_bins=args.n_bins)
                print_calibration_summary(stats)
                print_bin_table(y, p, n_bins=args.n_bins)

                report["checks"]["calibration"] = asdict(stats)

                # subgroup
                if group_cols:
                    # Need group cols in df_surv; attach to df_used by index alignment
                    df_used_full = df_surv.loc[df_used.index, :].copy()
                    grp = run_grouped_metrics(
                        df_used_full.assign(y_true=y, p_pred=p),
                        y_col="y_true",
                        p_col="p_pred",
                        group_cols=group_cols,
                        n_bins=args.n_bins,
                        min_group_n=args.min_group_n,
                    )
                    report["checks"]["grouped_calibration"] = grp
        else:
            warn(f"label_col={args.label_col} not found; skipping calibration metrics against survival probabilities.")

    else:
        header("Binary prediction sanity checks + calibration")
        if args.y_col not in joined.columns:
            raise ValueError(f"Missing y_col={args.y_col} in joined dataframe.")
        if args.p_col not in joined.columns:
            raise ValueError(f"Missing p_col={args.p_col} in joined dataframe.")

        tmp = joined[[args.y_col, args.p_col] + group_cols].copy()
        y, p, df_used = extract_binary_y_p(tmp, y_col=args.y_col, p_col=args.p_col)

        # Probability range checks
        p_s = pd.Series(p)
        out_of_range = int(((p_s < -1e-6) | (p_s > 1 + 1e-6)).sum())
        if out_of_range > 0:
            fail(f"{out_of_range} probabilities are outside [0,1].")
        else:
            ok("All probabilities within [0,1] (tolerance).")

        print(f"p_pred min={p_s.min():.6f} max={p_s.max():.6f} mean={p_s.mean():.6f}")

        stats = compute_binary_metrics(y, p, n_bins=args.n_bins)
        print_calibration_summary(stats)
        print_bin_table(y, p, n_bins=args.n_bins)
        report["checks"]["calibration"] = asdict(stats)

        if group_cols:
            df_used2 = df_used.copy()
            df_used2["y_true"] = y
            df_used2["p_pred"] = p
            grp = run_grouped_metrics(
                df_used2,
                y_col="y_true",
                p_col="p_pred",
                group_cols=group_cols,
                n_bins=args.n_bins,
                min_group_n=args.min_group_n,
            )
            report["checks"]["grouped_calibration"] = grp

    # Write JSON report if requested
    if args.report_json:
        header("Write report")
        os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        ok(f"Wrote JSON report to: {args.report_json}")

    header("Done")


if __name__ == "__main__":
    main()

