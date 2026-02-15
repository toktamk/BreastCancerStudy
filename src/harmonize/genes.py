from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HarmonizeConfig:
    """
    Harmonization focuses on column intersection and consistent ordering.

    For external validation we want:
    - same feature set
    - same feature order
    - same preprocessing parameters applied (train-fitted)
    """
    id_col: str = "patient_id"
    time_col: str = "time_months"
    event_col: str = "event"
    y60_col: str = "y60"

    # Feature prefixes used in the cohort assembly
    expr_prefix: str = "expr__"
    cna_prefix: str = "cna__"


def _feature_cols(df: pd.DataFrame, *, prefixes: tuple[str, ...]) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if any(c.startswith(p) for p in prefixes):
            cols.append(c)
    return cols


def intersect_feature_space(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: HarmonizeConfig = HarmonizeConfig(),
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Returns (train_aligned, test_aligned, feature_cols).

    - Keeps only features present in BOTH datasets.
    - Sorts features deterministically to ensure stable ordering.
    - Preserves label columns on both sides.
    """
    prefixes = (cfg.expr_prefix, cfg.cna_prefix)

    train_feats = set(_feature_cols(train_df, prefixes=prefixes))
    test_feats = set(_feature_cols(test_df, prefixes=prefixes))

    common = sorted(train_feats.intersection(test_feats))
    if not common:
        raise ValueError(
            "No common omics features found between train and test cohorts. "
            "Check naming (expr__/cna__) and gene identifiers."
        )

    keep_base = [cfg.id_col, cfg.time_col, cfg.event_col]
    keep_train = keep_base + ([cfg.y60_col] if cfg.y60_col in train_df.columns else []) + common
    keep_test = keep_base + ([cfg.y60_col] if cfg.y60_col in test_df.columns else []) + common

    train_aligned = train_df.loc[:, [c for c in keep_train if c in train_df.columns]].copy()
    test_aligned = test_df.loc[:, [c for c in keep_test if c in test_df.columns]].copy()

    # Ensure numeric feature dtypes (coerce)
    train_aligned[common] = train_aligned[common].apply(pd.to_numeric, errors="coerce")
    test_aligned[common] = test_aligned[common].apply(pd.to_numeric, errors="coerce")

    return train_aligned, test_aligned, common
