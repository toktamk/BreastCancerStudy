# src/pipelines/utils.py
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


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select modeling features.

    - Exclude ID and label columns
    - Include omics columns (expr__/cna__)
    - Include clinical covariates
    """
    exclude = {"patient_id", "time_months", "event", "y60", "horizon_months"}

    feat = []
    for c in df.columns:
        if c in exclude:
            continue
        feat.append(c)

    return sorted(set(feat))


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
