from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.load_cohort import load_cohort_parquet, CohortSpec
from src.harmonize.genes import intersect_feature_space, HarmonizeConfig
from src.splits.make_splits import SplitConfig, make_splits
from src.models.survival import fit_cox_model, predict_risk_cox  # expected in your survival.py
from src.eval.evaluate import evaluate_binary  # expected in your evaluate.py
from src.eval.time_dependent import time_dependent_auc_and_ibs
from src.eval.decision_curve import decision_curve_binary
from src.plots.risk_strata_km import make_risk_strata, km_summary_by_group


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cohort", type=str, required=True, help="METABRIC cohort parquet")
    ap.add_argument("--test_cohort", type=str, required=True, help="TCGA BRCA cohort parquet")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--horizons", type=str, default="36,60,120", help="comma-separated months")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    spec = CohortSpec()
    train_df = load_cohort_parquet(args.train_cohort, spec=spec)
    test_df = load_cohort_parquet(args.test_cohort, spec=spec)

    # Harmonize omics feature space (intersection)
    train_df, test_df, feat_cols = intersect_feature_space(
        train_df, test_df, cfg=HarmonizeConfig()
    )

    # Make a train/val split *within train cohort* for threshold selection, optional.
    # External validation conceptually evaluates on the entire external cohort.
    splits = make_splits(
        df=train_df,
        config=SplitConfig(
            id_col="patient_id",
            label_col="y60" if "y60" in train_df.columns else "event",
            seed=int(args.seed),
        ),
    )
    train_ids = set(splits.train_ids)
    dev = train_df[train_df["patient_id"].isin(train_ids)].copy()

    # Prepare survival inputs
    X_train = dev[feat_cols].to_numpy(dtype=float)
    t_train = dev["time_months"].to_numpy(dtype=float)
    e_train = dev["event"].to_numpy(dtype=int)

    X_test = test_df[feat_cols].to_numpy(dtype=float)
    t_test = test_df["time_months"].to_numpy(dtype=float)
    e_test = test_df["event"].to_numpy(dtype=int)

    # Fit Cox on METABRIC-dev
    model = fit_cox_model(X_train, t_train, e_train)

    # Predict risk scores on TCGA
    risk_test = predict_risk_cox(model, X_test)

    # --- Survival evaluation (external) ---
    horizons = [float(x) for x in args.horizons.split(",") if x.strip()]
    td = time_dependent_auc_and_ibs(
        train_time=t_train,
        train_event=e_train,
        test_time=t_test,
        test_event=e_test,
        test_risk_score=risk_test,
        horizons_months=horizons,
        test_surv_fn=None,  # add if your Cox wrapper can output survival probs
    )

    surv_out = {
        "horizons_months": td.horizons_months.tolist(),
        "auc_t": td.auc.tolist(),
        "auc_mean": td.auc_mean,
        "ibs": td.ibs,
    }
    (outdir / "external_survival_metrics.json").write_text(json.dumps(surv_out, indent=2), encoding="utf-8")

    # --- Binary evaluation at 5y if y60 available on TCGA cohort ---
    if "y60" in test_df.columns and test_df["y60"].notna().any():
        df_bin = test_df.dropna(subset=["y60"]).copy()
        Xb = df_bin[feat_cols].to_numpy(dtype=float)

        # Risk → probability mapping (simple logistic calibration using train dev)
        # For research-grade work, you can replace this with Platt scaling / isotonic on train.
        # Here we transform risk score to probability via a monotonic link.
        # NOTE: This is intentionally conservative and explicit.
        rb = predict_risk_cox(model, Xb)
        pb = 1.0 / (1.0 + np.exp(-rb))

        yb = df_bin["y60"].to_numpy(dtype=int)
        bin_metrics = evaluate_binary(yb, pb)

        (outdir / "external_binary_metrics.json").write_text(
            json.dumps(bin_metrics, indent=2),
            encoding="utf-8",
        )

        # Decision curve analysis
        dca = decision_curve_binary(yb, pb)
        dca_df = pd.DataFrame(
            {
                "threshold": dca.thresholds,
                "net_benefit_model": dca.net_benefit,
                "net_benefit_all": dca.net_benefit_all,
                "net_benefit_none": dca.net_benefit_none,
            }
        )
        dca_df.to_csv(outdir / "decision_curve.csv", index=False)

    # Risk strata KM summary on external cohort
    km_df = pd.DataFrame(
        {
            "time_months": t_test,
            "event": e_test,
            "risk": risk_test,
        }
    )
    strata = make_risk_strata(df=km_df, risk_col="risk", n_bins=3)
    km_sum = km_summary_by_group(strata)
    km_sum.to_csv(outdir / "km_risk_strata_summary.csv", index=False)

    print(f"✅ External validation results saved to: {outdir}")


if __name__ == "__main__":
    main()
