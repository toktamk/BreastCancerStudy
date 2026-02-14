# scripts/run_pipeline.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.datasets.load_cohort import CohortSpec, load_cohort_parquet
from src.splits.make_splits import SplitConfig, make_splits, load_splits, save_splits

from src.models.baselines import (
    LogisticConfig,
    fit_logistic,
    predict_proba,
    fit_cox_lifelines,
    predict_risk_cox_lifelines,
)

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
    select_feature_columns,
    extract_groups_for_fairness,
    safe_float,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train + evaluate METABRIC pipeline (binary + survival, no external validation)."
    )
    ap.add_argument("--cohort", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--splits_path", type=str, default="")

    ap.add_argument("--do_binary", action="store_true")
    ap.add_argument("--do_survival", action="store_true")

    ap.add_argument("--calibration_bins", type=int, default=10)
    ap.add_argument("--risk_bins", type=int, default=3)
    ap.add_argument("--fairness_group", type=str, default="er_status")

    ap.add_argument("--log_penalty", type=str, default="l2", choices=["l2", "elasticnet"])
    ap.add_argument("--log_C", type=float, default=1.0)
    ap.add_argument("--log_l1_ratio", type=float, default=0.5)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # 1) Load cohort
    spec = CohortSpec()
    df = load_cohort_parquet(args.cohort, spec=spec)

    # 2) Splits
    splits = None
    if args.splits_path.strip():
        splits = load_splits(Path(args.splits_path))  # dict: {"train": Index(ids), ...}

    if splits is None:
        # pick a stratify label that is fully observed; otherwise fall back
        if "y60" in df.columns and pd.to_numeric(df["y60"], errors="coerce").notna().all():
            strat_label = "y60"
        else:
            strat_label = "event"

        splits = make_splits(
            df=df,
            config=SplitConfig(id_col="patient_id", stratify_col=strat_label, seed=int(args.seed)),
        )

        # if a path is provided, save splits there
        if args.splits_path.strip():
            out_dir = Path(args.splits_path)
            save_splits(df=df, splits=splits, out_dir=out_dir,
                    config=SplitConfig(id_col="patient_id", stratify_col=strat_label, seed=int(args.seed)))

    # --- normalize split IDs regardless of whether splits are indices or ids ---
    def _to_ids(split_obj, key: str):
        v = split_obj[key]
        # If it's integer row indices (np.ndarray/list), map to patient_id
        if hasattr(v, "dtype") and str(getattr(v, "dtype", "")) != "object":
            return pd.Index(df.iloc[v]["patient_id"].astype(str))
        # If it's already ids (Index/Series/list of str)
        return pd.Index(pd.Series(v).astype(str))

    train_ids = _to_ids(splits, "train")
    val_ids   = _to_ids(splits, "val")
    test_ids  = _to_ids(splits, "test")

    train_df = df[df["patient_id"].astype(str).isin(train_ids)].copy()
    val_df   = df[df["patient_id"].astype(str).isin(val_ids)].copy()
    test_df  = df[df["patient_id"].astype(str).isin(test_ids)].copy()

    
    # 3) Features
    feature_cols = select_feature_columns(df)
    if not feature_cols:
        raise ValueError("No feature columns found.")

    # 4) Preprocessing (fit on train only)
    pre = build_preprocess_pipeline(train_df=train_df, feature_cols=feature_cols, id_col="patient_id")
    X_train = pre.transform(train_df)
    X_test = pre.transform(test_df)

    # outcomes
    t_train = train_df["time_months"].to_numpy(dtype=float)
    e_train = train_df["event"].to_numpy(dtype=int)
    t_test = test_df["time_months"].to_numpy(dtype=float)
    e_test = test_df["event"].to_numpy(dtype=int)

    y60_train = train_df["y60"].to_numpy(dtype=float) if "y60" in train_df.columns else None
    y60_test = test_df["y60"].to_numpy(dtype=float) if "y60" in test_df.columns else None

    group_test = extract_groups_for_fairness(test_df, group_col=args.fairness_group)

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

    results: Dict[str, Any] = {"binary": None, "survival": None}

    # 5) Binary
    if args.do_binary:
        if y60_train is None or y60_test is None:
            raise KeyError("Binary run requested but y60 column is missing from cohort.")

        m_train = np.isfinite(y60_train)
        m_test = np.isfinite(y60_test)

        Xtr = X_train[m_train]
        ytr = y60_train[m_train].astype(int)

        Xte = X_test[m_test]
        yte = y60_test[m_test].astype(int)

        cfg = LogisticConfig(
            penalty=args.log_penalty,
            C=float(args.log_C),
            l1_ratio=float(args.log_l1_ratio) if args.log_penalty == "elasticnet" else None,
            random_state=int(args.seed),
        )

        model = fit_logistic(Xtr, ytr, cfg=cfg)
        pte = predict_proba(model, Xte)

        metrics = evaluate_binary(yte, pte)

        fairness = None
        if group_test is not None:
            fairness = _grouped_binary_metrics(yte, pte, group_test[m_test])

        dca = decision_curve_binary(yte, pte)
        dca_df = pd.DataFrame(
            {
                "threshold": dca.thresholds,
                "net_benefit_model": dca.net_benefit,
                "net_benefit_all": dca.net_benefit_all,
                "net_benefit_none": dca.net_benefit_none,
            }
        )
        dca_df.to_csv(outdir / "binary_decision_curve.csv", index=False)

        pd.DataFrame(
            {
                "patient_id": test_df.loc[m_test, "patient_id"].astype(str).values,
                "y60": yte,
                "p_y60": pte,
            }
        ).to_csv(outdir / "binary_predictions.csv", index=False)

        plot_roc_curve(yte, pte, outdir / "plot_roc.png")
        plot_pr_curve(yte, pte, outdir / "plot_pr.png")
        plot_calibration_curve(yte, pte, outdir / "plot_calibration.png", n_bins=int(args.calibration_bins))
        plot_decision_curve(dca_df, outdir / "plot_decision_curve.png")

        results["binary"] = {"metrics": metrics, "fairness": fairness, "n_test_defined": int(len(yte))}
        dump_json(results["binary"], outdir / "binary_results.json")

    # 6) Survival
    if args.do_survival:
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

        plot_km_by_risk_group(
            time=t_test, event=e_test, risk=risk_test, outpath=outdir / "plot_km_risk_strata.png", n_bins=int(args.risk_bins)
        )

        fairness_s = None
        if group_test is not None:
            fairness_s = _grouped_survival_cindex(t_test, e_test, risk_test, group_test)

        results["survival"] = {"harrell_c_index": safe_float(cidx), "fairness": fairness_s}
        dump_json(results["survival"], outdir / "survival_results.json")

    dump_json(results, outdir / "results_all.json")
    print(f"✅ Done. Outputs: {outdir}")


def _grouped_binary_metrics(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    gs = pd.Series(g).astype("string")
    for grp in sorted(gs.dropna().unique().tolist()):
        m = (gs == grp).to_numpy()
        if m.sum() < 20:
            continue
        out[str(grp)] = evaluate_binary(y[m], p[m])
    return out


def _grouped_survival_cindex(t: np.ndarray, e: np.ndarray, risk: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    gs = pd.Series(g).astype("string")
    for grp in sorted(gs.dropna().unique().tolist()):
        m = (gs == grp).to_numpy()
        if m.sum() < 30:
            continue
        out[str(grp)] = safe_float(harrell_c_index(t[m], e[m], risk[m]))
    return out


if __name__ == "__main__":
    main()
