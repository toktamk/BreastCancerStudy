# scripts/run_survival_models.py
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.splits.make_splits import SplitConfig, make_splits, save_splits, load_splits
from src.features.preprocess import FeatureSpec, fit_transform_split
from src.eval.evaluate import evaluate_binary
from src.eval.survival_eval import survival_metrics_basic
from src.models.survival import (
    CoxPHConfig,
    fit_coxph_lifelines,
    predict_risk_coxph_lifelines,
    predict_survival_prob_at_horizon_coxph,
    survival_to_5yr_risk,
)
from src.experiments.ablation import AblationSpec, select_columns_by_ablation, apply_missing_modality_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_survival_models")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="outputs/survival")
    ap.add_argument("--label", type=str, default="y60")  # for binary evaluation at horizon
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

    # ---- Apply ablation selection
    label_cols = [args.label, "time_months", "event"]
    ab_spec = AblationSpec(
        use_clinical=bool(args.use_clinical),
        use_expr=bool(args.use_expr),
        use_cna=bool(args.use_cna),
    )
    if not (ab_spec.use_clinical or ab_spec.use_expr or ab_spec.use_cna):
        # default: clinical-only (safe baseline)
        ab_spec = AblationSpec(use_clinical=True, use_expr=False, use_cna=False)

    df = select_columns_by_ablation(df_full, label_cols=label_cols, spec=ab_spec)

    # ---- Missing modality simulation (stress test)
    df = apply_missing_modality_mask(
        df,
        frac_missing_expr=float(args.miss_expr),
        frac_missing_cna=float(args.miss_cna),
        seed=int(args.seed),
    )

    # ---- Splits (patient-level)
    splits_dir = outdir / "splits"
    _ensure_dir(splits_dir)

    if (splits_dir / "train.csv").exists():
        pid_splits = load_splits(splits_dir)
        train_ids, val_ids, test_ids = pid_splits["train"], pid_splits["val"], pid_splits["test"]
    else:
        scfg = SplitConfig(seed=int(args.seed), stratify_col=args.label)
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

    # ---- CoxPH model
    cox_cfg = CoxPHConfig(penalizer=0.01, l1_ratio=0.0, robust=False)
    model = fit_coxph_lifelines(X_train, t_train, e_train, cox_cfg)

    risk_test = predict_risk_coxph_lifelines(model, X_test)

    # Survival discrimination
    surv_metrics = survival_metrics_basic(t_test, e_test, risk_test)

    # ---- Horizon calibration: P(death by horizon)
    surv_prob_h = predict_survival_prob_at_horizon_coxph(model, X_test, horizon_months=float(args.horizon))
    risk_prob_h = survival_to_5yr_risk(surv_prob_h)

    bin_metrics = evaluate_binary(y_test, risk_prob_h)

    # ---- Save outputs
    preds_dir = outdir / "preds"
    _ensure_dir(preds_dir)

    pred_df = pd.DataFrame(
        {
            "patient_id": test_ids.astype(str),
            "time_months": t_test.astype(float),
            "event": e_test.astype(int),
            "y_true": y_test.astype(int),
            "risk_score": risk_test.astype(float),
            "survival_prob_horizon": surv_prob_h.astype(float),
            "risk_prob_horizon": risk_prob_h.astype(float),
        }
    )
    pred_df.to_parquet(preds_dir / "test.parquet", index=False)

    metrics = {"survival": surv_metrics, "binary_at_horizon": bin_metrics}
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    run_cfg: Dict[str, Any] = {
        "cohort": str(cohort_path),
        "seed": int(args.seed),
        "horizon_months": float(args.horizon),
        "ablation": asdict(ab_spec),
        "missing_modality": {"expr": float(args.miss_expr), "cna": float(args.miss_cna)},
        "model": {"type": "coxph_lifelines", **asdict(cox_cfg)},
        "feature_blocks": bundle["meta"]["block_order"],
    }
    (outdir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    logger.info("Done. Saved to %s", outdir.resolve())


if __name__ == "__main__":
    main()
