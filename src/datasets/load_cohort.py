"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class CohortSpec:
    """
    Defines which columns are required for modeling.

    Notes:
    - For binary 5-year models, we need y60.
    - For survival models, we need time_months and event.
    """
    id_col: str = "patient_id"
    time_col: str = "time_months"
    event_col: str = "event"
    y60_col: str = "y60"

    # Optional: group columns (fairness strata) if present
    group_cols: tuple[str, ...] = ("age_group", "er_status", "pr_status", "her2_status", "grade")


def _require_cols(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def load_cohort_parquet(path: str | Path, spec: CohortSpec = CohortSpec()) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Cohort file not found: {p}")

    df = pd.read_parquet(p)

    # Minimum survival contract
    _require_cols(df, [spec.id_col, spec.time_col, spec.event_col], "cohort")

    # Normalize basic types defensively
    df = df.copy()
    df[spec.id_col] = df[spec.id_col].astype("string")
    df[spec.time_col] = pd.to_numeric(df[spec.time_col], errors="coerce")
    df[spec.event_col] = pd.to_numeric(df[spec.event_col], errors="coerce")

    # Drop rows with broken survival labels
    df = df.dropna(subset=[spec.id_col, spec.time_col, spec.event_col])
    df = df[df[spec.time_col] >= 0]
    df = df[df[spec.event_col].isin([0, 1])]

    return df
