"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def dump_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def select_feature_columns(
    df: pd.DataFrame,
    *,
    train_df: pd.DataFrame | None = None,
    top_k_numeric: int = 2000,
    low_card_nunique: int = 20,
) -> list[str]:
    """
    Select modeling features with dimensionality control.

    Keeps:
      - all categorical columns
      - numeric columns with low cardinality (<= low_card_nunique)
      - top_k_numeric numeric columns (among remaining) by variance computed on train_df
    """
    exclude = {"patient_id", "time_months", "event", "y60", "horizon_months"}
    cols = [c for c in df.columns if c not in exclude]

    if train_df is None:
        return sorted(set(cols))

    Xtr = train_df[cols]

    cat_cols = [
        c for c in cols
        if Xtr[c].dtype == "object" or str(Xtr[c].dtype).startswith("string")
    ]
    num_cols = [c for c in cols if c not in cat_cols]

    # low-cardinality numeric -> keep
    nunique = Xtr[num_cols].nunique(dropna=True)
    low_card = nunique[nunique <= low_card_nunique].index.tolist()

    # high-card numeric -> choose top-K by variance
    high_card = [c for c in num_cols if c not in set(low_card)]
    if high_card:
        var = Xtr[high_card].astype(float).var(axis=0, skipna=True)
        keep_high = var.sort_values(ascending=False).head(top_k_numeric).index.tolist()
    else:
        keep_high = []

    feat = sorted(set(cat_cols + low_card + keep_high))
    return feat


def extract_groups_for_fairness(df: pd.DataFrame, group_col: str):
    if group_col and group_col in df.columns:
        return df[group_col].astype("string").to_numpy()
    return None


def write_run_manifest(
    *,
    outdir: Path,
    args: Dict[str, Any],
    cohort_path: str,
    n_train: int,
    n_val: int,
    n_test: int,
    n_features: int,
    feature_preview: list[str],
) -> None:
    manifest = {
        "cohort_path": cohort_path,
        "args": args,
        "counts": {"train": n_train, "val": n_val, "test": n_test},
        "n_features": n_features,
        "feature_preview": feature_preview,
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
