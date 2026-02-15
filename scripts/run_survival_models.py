from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.experiments.ablation import (
    AblationSpec,
    apply_missing_modality_mask,
    select_columns_by_ablation,
)
from src.eval.evaluate import evaluate_binary
from src.eval.survival_eval import survival_metrics_basic
from src.features.preprocess import FeatureSpec, fit_transform_split
from src.models.survival import (
    CoxPHConfig,
    fit_coxph_lifelines,
    predict_risk_coxph_lifelines,
    predict_survival_prob_at_horizon_coxph,
    survival_to_5yr_risk,
)
from src.splits.make_splits import SplitConfig, make_splits, save_splits, load_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_survival_models")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="outputs/survival")
    ap.add_argument("--label", type=str, default="y60")  # fixed-horizon binary endpoint for evaluation only
    ap.add_argument("--horizon", type=float, default=60.0)

    # Ablation
    ap.add_argument("--use_clinical", action="store_true")
    ap.add_argument("--use_expr", action="store_true")
    ap.add_argument("--use_cna", action="store_true")

    # Missing modality robustness
    ap.add_argument("--miss_expr", type=float, default=0.0)
    ap.add_argument("--miss_cna", type=float, default=0.0)

    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cohort_path = Path(args.cohort)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    df_full = pd.read_parquet(cohort_path)

    # ---- Ablation selection (features + labels)
    # Keep patient_id so we can always export aligned IDs for predictions.
    required_cols = ["patient_id", "time_months", "event", args.label]
    ab_spec = AblationSpec(
        use_clinical=bool(args.use_clinical),
        use_expr=bool(args.use_expr),
        use_cna=bool(args.use_cna),
    )
    if not (ab_spec.use_clinical or ab_spec.use_expr or ab_spec.use_cna):
        # default: clinical-only (safe baseline)
        ab_spec = AblationSpec(use_clinical=True, use_expr=False, use_cna=False)

    # select_columns_by_ablation should include patient_id if present in df_full.
    # If it doesn't, we force-add it afterward.
    df = select_columns_by_ablation(df_full, label_cols=required_cols, spec=ab_spec)
    if "patient_id" not in df.columns and "patient_id" in df_full.columns:
        df = df.join(df_full[["patient_id"]])

    # ---- Missing modality simulation (stress test)
    df = apply_missing_modality_mask(
        df,
        frac_missing_expr=float(args.miss_expr),
        frac_missing_cna=float(args.miss_cna),
        seed=int(args.seed),
    )
    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df = df.set_index("patient_id", drop=False)


    # ---- Splits (patient-level)
    splits_dir = outdir / "splits"
    _ensure_dir(splits_dir)

    if (splits_dir / "train.csv").exists():
        pid_splits = load_splits(splits_dir)
        train_ids, val_ids, test_ids = pid_splits["train"], pid_splits["val"], pid_splits["test"]
    else:
        # For survival modeling, stratify on event (no NaNs), not on fixed-horizon label.
        scfg = SplitConfig(seed=int(args.seed), stratify_col="event")
        idx_splits = make_splits(df, scfg)
        save_splits(df, idx_splits, splits_dir, scfg)
        pid_splits = load_splits(splits_dir)
        train_ids, val_ids, test_ids = pid_splits["train"], pid_splits["val"], pid_splits["test"]

    # ---- Preprocess (fit on train only)
    spec = FeatureSpec(label_col=args.label)
    bundle = fit_transform_split(df, train_ids, val_ids, test_ids, spec=spec)

    X_train, X_val, X_test = bundle["X_train"], bundle["X_val"], bundle["X_test"]
    t_train, e_train = bundle["t_train"], bundle["e_train"]
    t_test, e_test = bundle["t_test"], bundle["e_test"]
    y_test = bundle["y_test"]

    # ---- Test IDs aligned with X_test/y_test
    # Prefer pulling IDs from the original df using the split IDs, not from idx order assumptions.
    if "patient_id" not in df.columns:
        raise ValueError("patient_id column is missing from cohort after ablation selection; cannot export aligned preds.")
    df_test = df.loc[test_ids].copy()
    id_test = df_test["patient_id"].astype(str).to_numpy()

    # ---- CoxPH model
    # Penalizer helps with separation/ill-conditioning; tune if needed (0.1 or 1.0 if still unstable).
    cox_cfg = CoxPHConfig(penalizer=0.01, l1_ratio=0.0, robust=False)
    model = fit_coxph_lifelines(X_train, t_train, e_train, cox_cfg)

    # Risk score (linear predictor / partial hazard ranking score)
    risk_score = predict_risk_coxph_lifelines(model, X_test)
    # ---- Validation predictions (for calibration fitting)
    risk_val = predict_risk_coxph_lifelines(model, X_val)
    surv_prob_val = predict_survival_prob_at_horizon_coxph(model, X_val, horizon_months=float(args.horizon))
    risk_prob_val = survival_to_5yr_risk(surv_prob_val)
    y_val = bundle["y_val"]
    # Survival discrimination
    surv_metrics = survival_metrics_basic(t_test, e_test, risk_score)

    # ---- Horizon probabilities: P(event by horizon) = 1 - S(horizon)
    survival_prob_horizon = predict_survival_prob_at_horizon_coxph(
        model, X_test, horizon_months=float(args.horizon)
    )
    risk_prob_horizon = survival_to_5yr_risk(survival_prob_horizon)

    # ---- Binary metrics (only where fixed-horizon label is defined)
    m_eval = np.isfinite(y_test) & np.isfinite(risk_prob_horizon)
    bin_metrics = evaluate_binary(y_test[m_eval], risk_prob_horizon[m_eval])

    # ---- Save outputs
    id_val = pd.Index(val_ids).astype(str).to_numpy()
    id_test = pd.Index(test_ids).astype(str).to_numpy()
    preds_dir = outdir / "preds"
    _ensure_dir(preds_dir)

    # VAL
    mval = np.isfinite(y_val)
    val_df = pd.DataFrame({
        "patient_id": id_val,
        "y_true": y_val.astype("float32"),
        "y_true_defined": mval.astype(np.int8),
        "risk_score": risk_val.astype("float32"),
        "survival_prob_horizon": surv_prob_val.astype("float32"),
        "risk_prob_horizon": risk_prob_val.astype("float32"),
    })
    val_df.to_parquet(preds_dir / "val.parquet", index=False)
    # TEST
    mtest = np.isfinite(y_test)
    test_df = pd.DataFrame({
        "patient_id": id_test,
        "y_true": y_test.astype("float32"),
        "y_true_defined": mtest.astype(np.int8),
        "risk_score": risk_score.astype("float32"),
        "survival_prob_horizon": survival_prob_horizon.astype("float32"),
        "risk_prob_horizon": risk_prob_horizon.astype("float32"),
    })
    test_df.to_parquet(preds_dir / "test.parquet", index=False)
   
    metrics = {"survival": surv_metrics, "binary_at_horizon": bin_metrics}
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    run_cfg: Dict[str, Any] = {
        "cohort": str(cohort_path),
        "seed": int(args.seed),
        "horizon_months": float(args.horizon),
        "label": str(args.label),
        "ablation": asdict(ab_spec),
        "missing_modality": {"expr": float(args.miss_expr), "cna": float(args.miss_cna)},
        "model": {"type": "coxph_lifelines", **asdict(cox_cfg)},
        "feature_blocks": bundle.get("meta", {}).get("block_order", None),
        "n_eval_defined": int(m_eval.sum()),
        "n_test_total": int(len(y_test)),
    }
    (outdir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    logger.info("Done. Saved to %s", outdir.resolve())


if __name__ == "__main__":
    main()
