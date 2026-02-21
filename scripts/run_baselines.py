"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from src.splits.make_splits import SplitConfig, make_splits, save_splits, load_splits
from src.features.preprocess import FeatureSpec, fit_transform_split
from src.models.baselines import LogisticConfig, fit_logistic, predict_proba
from src.eval.evaluate import evaluate_binary

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_baselines")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=str, required=True, help="Path to cohort parquet (patient-level).")
    ap.add_argument("--outdir", type=str, default="outputs/baselines", help="Output directory.")
    ap.add_argument("--label", type=str, default="y60", help="Binary label column for fixed-horizon task.")
    ap.add_argument("--group_col", type=str, default="", help="Optional group column for fairness reporting (e.g., age_bin).")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cohort_path = Path(args.cohort)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    df = pd.read_parquet(cohort_path)
    
    if "patient_id" not in df.columns:
        raise KeyError("cohort must include patient_id")
    if args.label not in df.columns:
        raise KeyError(f"cohort missing label '{args.label}'")
    # --- Filter to defined binary label
    df = df[df[args.label].notna()].copy()
    # --- Splits
    splits_dir = outdir / "splits"
    if (splits_dir / "train.csv").exists():
        logger.info("Loading existing splits from %s", splits_dir)
        pid_splits = load_splits(splits_dir)
        train_ids, val_ids, test_ids = pid_splits["train"], pid_splits["val"], pid_splits["test"]
    else:
        logger.info("Creating new splits")
        scfg = SplitConfig(seed=args.seed, stratify_col=args.label)
        idx_splits = make_splits(df, scfg)
        save_splits(df, idx_splits, splits_dir, scfg)
        pid_splits = load_splits(splits_dir)
        train_ids, val_ids, test_ids = pid_splits["train"], pid_splits["val"], pid_splits["test"]

    # --- Preprocess
    spec = FeatureSpec(label_col=args.label)
    bundle = fit_transform_split(df, train_ids, val_ids, test_ids, spec=spec)

    X_train, X_val, X_test = bundle["X_train"], bundle["X_val"], bundle["X_test"]
    y_train, y_val, y_test = bundle["y_train"], bundle["y_val"], bundle["y_test"]

    # --- Model: Logistic baseline (L2)
    mcfg = LogisticConfig(penalty="l2", C=1.0, random_state=args.seed)
    model = fit_logistic(X_train, y_train, mcfg)

    p_val = predict_proba(model, X_val)
    p_test = predict_proba(model, X_test)

    # --- Evaluation
    group_series = None
    if args.group_col and args.group_col in df.columns:
        df_i = df.set_index("patient_id")
        group_series = df_i.loc[test_ids][args.group_col]

    metrics = {
        "val": evaluate_binary(y_val, p_val),
        "test": evaluate_binary(y_test, p_test, group=group_series),
    }

    # --- Save predictions
    preds_dir = outdir / "preds"
    _ensure_dir(preds_dir)

    pred_val_df = pd.DataFrame({"patient_id": val_ids.astype(str), "y_true": y_val.astype(int), "y_prob": p_val})
    pred_test_df = pd.DataFrame({"patient_id": test_ids.astype(str), "y_true": y_test.astype(int), "y_prob": p_test})

    pred_val_df.to_parquet(preds_dir / "val.parquet", index=False)
    pred_test_df.to_parquet(preds_dir / "test.parquet", index=False)

    # --- Save metrics & run config
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    run_cfg: Dict[str, Any] = {
        "cohort": str(cohort_path),
        "label": args.label,
        "seed": args.seed,
        "split_dir": str(splits_dir),
        "model": {"type": "logistic_regression", **asdict(mcfg)},
        "feature_blocks": bundle["meta"]["block_order"],
        "feature_cols": bundle["meta"]["feature_cols"],
    }
    (outdir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    logger.info("Done. Outputs saved to: %s", outdir.resolve())


if __name__ == "__main__":
    main()
