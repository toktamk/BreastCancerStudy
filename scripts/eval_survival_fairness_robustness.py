from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.robustness.bootstrap import bootstrap_ci
from src.fairness.group_eval import group_metrics, threshold_parity


def _parse_group_cols(s: str) -> List[str]:
    return [c.strip() for c in (s or "").split(",") if c.strip()]


def _ensure_age_group(df: pd.DataFrame, requested_groups: List[str]) -> pd.DataFrame:
    """
    If 'age_group' is requested but missing, derive it from AGE_AT_DIAGNOSIS.
    Bins are clinically interpretable and stable.
    """
    if "age_group" not in requested_groups:
        return df

    if "age_group" in df.columns:
        return df

    if "AGE_AT_DIAGNOSIS" not in df.columns:
        raise KeyError("Requested group column 'age_group' but 'AGE_AT_DIAGNOSIS' not found in cohort.")

    df = df.copy()
    df["age_group"] = pd.cut(
        df["AGE_AT_DIAGNOSIS"].astype(float),
        bins=[-np.inf, 40, 50, 60, 70, np.inf],
        labels=["<=40", "41-50", "51-60", "61-70", "71+"],
        include_lowest=True,
    ).astype(str)

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, help="Path to cohort parquet (must include patient_id and subgroup cols).")
    ap.add_argument("--preds", required=True, help="Path to preds parquet (must include patient_id, y_true, p-col).")
    ap.add_argument("--outdir", required=True, help="Output directory for report.json and CSV tables.")

    ap.add_argument("--id-col", default="patient_id")
    ap.add_argument("--y-col", default="y_true")
    ap.add_argument("--p-col", default="risk_prob_horizon")

    ap.add_argument("--group-cols", default="", help="Comma-separated subgroup columns, e.g. 'age_group,SEX,ER_IHC'.")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--min-n", type=int, default=30, help="Minimum subgroup size for fairness metrics.")

    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cohort_path = Path(args.cohort)
    preds_path = Path(args.preds)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_parquet(cohort_path)
    preds = pd.read_parquet(preds_path)

    if args.id_col not in cohort.columns:
        raise KeyError(f"ID column '{args.id_col}' not found in cohort.")
    if args.id_col not in preds.columns:
        raise KeyError(f"ID column '{args.id_col}' not found in preds.")
    if args.y_col not in preds.columns:
        raise KeyError(f"y column '{args.y_col}' not found in preds.")
    if args.p_col not in preds.columns:
        raise KeyError(f"p column '{args.p_col}' not found in preds.")

    # Join predictions to cohort
    df = preds.merge(cohort, on=args.id_col, how="left")
    # Strong join sanity check: if any joined row has all cohort fields missing, flag it.
    # (We can't easily distinguish cohort columns vs pred columns without more bookkeeping,
    # so we use a pragmatic check that catches the worst failures.)
    if df.isna().all(axis=1).any():
        raise RuntimeError("Join appears broken: some rows are entirely NaN after merge.")

    # Group columns handling (auto-derive age_group if requested)
    group_cols = _parse_group_cols(args.group_cols)
    df = _ensure_age_group(df, group_cols)

    # Fail-fast if requested group columns are missing
    missing = [gc for gc in group_cols if gc not in df.columns]
    if missing:
        raise KeyError(
            f"Requested group column(s) not found: {missing}. "
            f"Example available columns: {list(df.columns[:40])}"
        )

    # Core vectors (binary-at-horizon; exclude undefined)
    y = df[args.y_col].to_numpy()
    p = df[args.p_col].to_numpy()
    m = np.isfinite(y) & np.isfinite(p)
    yv = y[m].astype(int)
    pv = p[m].astype(float)

    n_defined = int(m.sum())
    if n_defined == 0:
        raise ValueError("No defined labels/probabilities after filtering finite y and p.")

    # Core metrics (binary at horizon)
    auroc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) == 2 else float("nan")
    brier = float(brier_score_loss(yv, pv))

    # Bootstrap robustness
    boot_auroc = bootstrap_ci(
        yv,
        pv,
        lambda yy, pp: roc_auc_score(yy.astype(int), pp),
        n_boot=int(args.bootstrap),
        seed=int(args.seed),
    )
    boot_brier = bootstrap_ci(
        yv,
        pv,
        lambda yy, pp: brier_score_loss(yy.astype(int), pp),
        n_boot=int(args.bootstrap),
        seed=int(args.seed),
    )

    # Fairness
    fairness: Dict[str, Dict[str, List[dict]]] = {}
    for gc in group_cols:
        gm = group_metrics(
            df,
            y_col=args.y_col,
            p_col=args.p_col,
            group_col=gc,
            n_bins=int(args.n_bins),
            min_n=int(args.min_n),
        )
        tp = threshold_parity(
            df,
            y_col=args.y_col,
            p_col=args.p_col,
            group_col=gc,
            threshold=float(args.threshold),
            min_n=int(args.min_n),
        )
        fairness[gc] = {
            "group_metrics": [x.__dict__ for x in gm],
            "threshold_parity": [x.__dict__ for x in tp],
        }

        # Write per-group CSVs (convenience)
        pd.DataFrame(fairness[gc]["group_metrics"]).to_csv(outdir / f"fairness_{gc}_group_metrics.csv", index=False)
        pd.DataFrame(fairness[gc]["threshold_parity"]).to_csv(
            outdir / f"fairness_{gc}_threshold_parity.csv", index=False
        )

    report = {
        "inputs": {"cohort": str(cohort_path), "preds": str(preds_path)},
        "columns": {
            "id_col": args.id_col,
            "y_col": args.y_col,
            "p_col": args.p_col,
            "group_cols": group_cols,
        },
        "n_total": int(len(df)),
        "n_defined": n_defined,
        "binary_at_horizon": {
            "auroc": auroc,
            "brier": brier,
            "prevalence": float(np.mean(yv)),
            "mean_pred": float(np.mean(pv)),
        },
        "bootstrap": {
            "auroc": boot_auroc.__dict__,
            "brier": boot_brier.__dict__,
        },
        "fairness": fairness,
    }

    (outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[OK] wrote", outdir / "report.json")


if __name__ == "__main__":
    main()
