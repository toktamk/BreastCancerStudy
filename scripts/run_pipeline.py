"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from src.datasets.load_cohort import CohortSpec, load_cohort_parquet
from src.splits.make_splits import SplitConfig, make_splits, load_splits, save_splits
from src.models.baselines import fit_cox_lifelines, predict_risk_cox_lifelines
from src.eval.evaluate import evaluate_binary
from src.eval.survival_eval import harrell_c_index
from src.eval.decision_curve import decision_curve_binary
from src.plots.standard_plots import (
    plot_roc_curve,
    plot_pr_curve,
    plot_calibration_curve,
    plot_decision_curve,
    plot_km_by_risk_group,
)
from src.pipelines.preprocessing import build_preprocess_pipeline
from src.pipelines.utils import (
    ensure_dir,
    dump_json,
    write_run_manifest,
    extract_groups_for_fairness,
    safe_float,
)

LABEL_COLS = {"patient_id", "time_months", "event", "y60", "horizon_months"}


# ==========================================================
# Utility Functions
# ==========================================================
def choose_stratify_col(df: pd.DataFrame) -> str:
    """Prefer y60 only if fully observed; otherwise fall back to event."""
    if "y60" in df.columns:
        y = pd.to_numeric(df["y60"], errors="coerce")
        if y.notna().all():
            return "y60"
    return "event"


def splits_to_ids(df: pd.DataFrame, splits: dict) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """Normalize splits to patient_id Indexes (splits may be row indices or ids)."""
    def to_ids(v) -> pd.Index:
        v_arr = np.asarray(v)
        if np.issubdtype(v_arr.dtype, np.integer):
            return pd.Index(df.iloc[v_arr]["patient_id"].astype(str).values)
        return pd.Index(pd.Series(v).astype(str).values)

    for k in ("train", "val", "test"):
        if k not in splits:
            raise KeyError(f"Splits missing key '{k}'. Expected train/val/test.")
    return to_ids(splits["train"]), to_ids(splits["val"]), to_ids(splits["test"])


def select_features_dim_control(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    top_k_numeric: int = 2000,
    low_card_nunique: int = 20,
) -> list[str]:
    """
    Dimensionality-controlled feature selection:
      - keep all categorical columns
      - keep numeric columns with low cardinality
      - keep top_k_numeric remaining numeric columns by variance (on train)
    """
    cols = [c for c in df.columns if c not in LABEL_COLS]
    Xtr = train_df[cols]

    cat_cols = [
        c for c in cols
        if Xtr[c].dtype == "object"
        or str(Xtr[c].dtype).startswith("string")
        or str(Xtr[c].dtype) == "category"
    ]
    num_cols = [c for c in cols if c not in cat_cols]

    keep = set(cat_cols)

    if num_cols:
        nunique = Xtr[num_cols].nunique(dropna=True)
        low_card = nunique[nunique <= low_card_nunique].index.tolist()
        keep.update(low_card)

        high_card = [c for c in num_cols if c not in set(low_card)]
        if high_card and top_k_numeric > 0:
            var = Xtr[high_card].astype(float).var(axis=0, skipna=True)
            keep_high = var.sort_values(ascending=False).head(top_k_numeric).index.tolist()
            keep.update(keep_high)

    feat = sorted(keep)
    if not feat:
        raise ValueError("No feature columns selected.")
    return feat


def train_xgb_with_validation(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    seed: int,
):
    """
    Train XGBoost with early stopping and a small hyperparameter grid.
    Select best by validation AUROC.
    """
    pos = float(ytr.sum())
    neg = float(len(ytr) - ytr.sum())
    scale_pos_weight = neg / max(pos, 1.0)

    param_grid = [
        {"max_depth": 3, "eta": 0.05},
        {"max_depth": 4, "eta": 0.05},
        {"max_depth": 3, "eta": 0.10},
        {"max_depth": 4, "eta": 0.10},
    ]

    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dval = xgb.DMatrix(Xva, label=yva)

    best_model = None
    best_params = None
    best_auc = -1e18

    for grid in param_grid:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "seed": seed,
            "max_depth": grid["max_depth"],
            "eta": grid["eta"],
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight,
            "lambda": 1.0,
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        p_val = model.predict(dval)
        # evaluate_binary returns dict with discrimination->auroc in the project
        auc = float(evaluate_binary(yva.astype(int), p_val)["discrimination"]["auroc"])

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_params = params

    if best_model is None or best_params is None:
        raise RuntimeError("XGBoost training failed to produce a model.")
    return best_model, best_params


# ==========================================================
# Main
# ==========================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--splits_path", default="")
    parser.add_argument("--do_binary", action="store_true")
    parser.add_argument("--do_survival", action="store_true")
    parser.add_argument("--calibration_bins", type=int, default=10)
    parser.add_argument("--risk_bins", type=int, default=3)
    parser.add_argument("--fairness_group", default="er_status")
    parser.add_argument("--top_k_numeric", type=int, default=2000)
    parser.add_argument("--low_card_nunique", type=int, default=20)
    parser.add_argument("--calibrate", action="store_true")

    args = parser.parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # ---------------- Load Cohort ----------------
    df = load_cohort_parquet(args.cohort, spec=CohortSpec())

    # ---------------- Splits ----------------
    splits = None
    if args.splits_path.strip():
        splits = load_splits(Path(args.splits_path))

    if splits is None:
        strat_col = choose_stratify_col(df)
        split_cfg = SplitConfig(id_col="patient_id", stratify_col=strat_col, seed=int(args.seed))
        splits = make_splits(df=df, config=split_cfg)

        # save splits if user requested a path
        if args.splits_path.strip():
            out_splits = Path(args.splits_path)
            ensure_dir(out_splits)
            try:
                save_splits(df=df, splits=splits, out_dir=out_splits, config=split_cfg)  # type: ignore
            except TypeError:
                save_splits(splits, out_splits)  # type: ignore

    train_ids, val_ids, test_ids = splits_to_ids(df, splits)

    train_df = df[df["patient_id"].astype(str).isin(train_ids)].copy()
    val_df = df[df["patient_id"].astype(str).isin(val_ids)].copy()
    test_df = df[df["patient_id"].astype(str).isin(test_ids)].copy()

    # ---------------- Feature Selection ----------------
    feature_cols = select_features_dim_control(
        df=df,
        train_df=train_df,
        top_k_numeric=int(args.top_k_numeric),
        low_card_nunique=int(args.low_card_nunique),
    )

    # ---------------- Preprocess (fit on train only) ----------------
    pre = build_preprocess_pipeline(
    train_df=train_df,
    feature_cols=feature_cols,
    id_col="patient_id",
    do_hybrid_reduction=True,
    top_k_features=3000,          # keep 2000 original selected features
    select_mode="hybrid",         # "variance" or "sparsity" or "hybrid"
    svd_max_components=2000,      # fit up to 1000
    svd_var_threshold=0.05,       # KEEP meaningful; could be <1000
    svd_min_components=100,
    random_state=args.seed,
    )


    X_train = pre.transform(train_df)
    X_val = pre.transform(val_df)
    X_test = pre.transform(test_df)

    # ---------------- Manifest ----------------
    write_run_manifest(
        outdir=outdir,
        args=vars(args),
        cohort_path=args.cohort,
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        n_features=len(feature_cols),
        feature_preview=feature_cols[:20],
    )

    group_test = extract_groups_for_fairness(test_df, group_col=args.fairness_group)

    results: Dict[str, Any] = {"binary": None, "survival": None}

    # =====================================================
    # Binary Modeling (XGBoost)
    # =====================================================
    if args.do_binary:
        if "y60" not in train_df.columns or "y60" not in val_df.columns or "y60" not in test_df.columns:
            raise KeyError("Binary run requested but y60 column is missing from cohort.")

        ytr = pd.to_numeric(train_df["y60"], errors="coerce").to_numpy()
        yva = pd.to_numeric(val_df["y60"], errors="coerce").to_numpy()
        yte = pd.to_numeric(test_df["y60"], errors="coerce").to_numpy()

        m_tr = np.isfinite(ytr)
        m_va = np.isfinite(yva)
        m_te = np.isfinite(yte)

        Xtr, ytr2 = X_train[m_tr], ytr[m_tr].astype(int)
        Xva, yva2 = X_val[m_va], yva[m_va].astype(int)
        Xte, yte2 = X_test[m_te], yte[m_te].astype(int)

        model_val, best_params = train_xgb_with_validation(Xtr, ytr2, Xva, yva2, seed=int(args.seed))

        # refit on train+val using best_iteration
        dtrain_full = xgb.DMatrix(np.vstack([Xtr, Xva]), label=np.concatenate([ytr2, yva2]))
        num_round = int(getattr(model_val, "best_iteration", 0) or 500)

        model = xgb.train(best_params, dtrain_full, num_boost_round=num_round)

        dtest = xgb.DMatrix(Xte)
        p_raw = model.predict(dtest)

        # optional calibration (fit on validation predictions from the val-selected model)
        p_cal = None
        if args.calibrate:
            dval = xgb.DMatrix(Xva)
            p_val = model_val.predict(dval)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_val, yva2)
            p_cal = iso.transform(p_raw)

        # evaluate raw (and calibrated if present)
        metrics_raw = evaluate_binary(yte2, p_raw)
        dump_json(metrics_raw, outdir / "binary_results_xgb_raw.json")

        if p_cal is not None:
            metrics_cal = evaluate_binary(yte2, p_cal)
            dump_json(metrics_cal, outdir / "binary_results_xgb_calibrated.json")

        # plots + decision curve (raw)
        plot_roc_curve(yte2, p_raw, outdir / "plot_roc_raw.png")
        plot_pr_curve(yte2, p_raw, outdir / "plot_pr_raw.png")
        plot_calibration_curve(yte2, p_raw, outdir / "plot_calibration_raw.png", n_bins=int(args.calibration_bins))

        dca_raw = decision_curve_binary(yte2, p_raw)
        dca_raw_df = pd.DataFrame(
            {
                "threshold": dca_raw.thresholds,
                "net_benefit_model": dca_raw.net_benefit,
                "net_benefit_all": dca_raw.net_benefit_all,
                "net_benefit_none": dca_raw.net_benefit_none,
            }
        )
        dca_raw_df.to_csv(outdir / "binary_decision_curve_raw.csv", index=False)
        plot_decision_curve(dca_raw_df, outdir / "plot_decision_curve_raw.png")

        # plots + decision curve (calibrated)
        if p_cal is not None:
            plot_roc_curve(yte2, p_cal, outdir / "plot_roc_cal.png")
            plot_pr_curve(yte2, p_cal, outdir / "plot_pr_cal.png")
            plot_calibration_curve(yte2, p_cal, outdir / "plot_calibration_cal.png", n_bins=int(args.calibration_bins))

            dca_cal = decision_curve_binary(yte2, p_cal)
            dca_cal_df = pd.DataFrame(
                {
                    "threshold": dca_cal.thresholds,
                    "net_benefit_model": dca_cal.net_benefit,
                    "net_benefit_all": dca_cal.net_benefit_all,
                    "net_benefit_none": dca_cal.net_benefit_none,
                }
            )
            dca_cal_df.to_csv(outdir / "binary_decision_curve_cal.csv", index=False)
            plot_decision_curve(dca_cal_df, outdir / "plot_decision_curve_cal.png")

        # save predictions
        pred_df = pd.DataFrame(
            {
                "patient_id": test_df.loc[m_te, "patient_id"].astype(str).values,
                "y60": yte2,
                "p_raw": p_raw,
            }
        )
        if p_cal is not None:
            pred_df["p_cal"] = p_cal
        pred_df.to_csv(outdir / "binary_predictions.csv", index=False)

        results["binary"] = {
            "model": "xgboost",
            "best_params": best_params,
            "n_test_defined": int(len(yte2)),
        }
        dump_json(results["binary"], outdir / "binary_results.json")

    # =====================================================
    # Survival Modeling (Cox baseline)
    # =====================================================
    if args.do_survival:
        if "time_months" not in train_df.columns or "event" not in train_df.columns:
            raise KeyError("Survival run requested but time_months/event missing from cohort.")

        t_train = pd.to_numeric(train_df["time_months"], errors="coerce").to_numpy(dtype=float)
        e_train = pd.to_numeric(train_df["event"], errors="coerce").to_numpy(dtype=int)

        t_test = pd.to_numeric(test_df["time_months"], errors="coerce").to_numpy(dtype=float)
        e_test = pd.to_numeric(test_df["event"], errors="coerce").to_numpy(dtype=int)

        sig = inspect.signature(fit_cox_lifelines)
        if "max_features" in sig.parameters:
            model_s = fit_cox_lifelines(X_train, t_train, e_train, max_features=1000, penalizer=1e-2)  # type: ignore
        else:
            model_s = fit_cox_lifelines(X_train, t_train, e_train)

        risk_test = predict_risk_cox_lifelines(model_s, X_test)
        cidx = harrell_c_index(t_test, e_test, risk_test)

        pd.DataFrame(
            {
                "patient_id": test_df["patient_id"].astype(str).values,
                "time_months": t_test,
                "event": e_test,
                "risk": risk_test,
            }
        ).to_csv(outdir / "survival_predictions.csv", index=False)

        # IMPORTANT: keyword-only call (matches the function signature)
        plot_km_by_risk_group(
            time=t_test,
            event=e_test,
            risk=risk_test,
            outpath=outdir / "plot_km_risk_strata.png",
            n_bins=int(args.risk_bins),
        )

        results["survival"] = {"harrell_c_index": safe_float(cidx)}
        dump_json(results["survival"], outdir / "survival_results.json")

    dump_json(results, outdir / "results_all.json")
    print(f"✅ Done. Outputs: {outdir}")


if __name__ == "__main__":
    main()
