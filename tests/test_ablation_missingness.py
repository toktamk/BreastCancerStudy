"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.ablation import (
    AblationSpec,
    select_columns_by_ablation,
    apply_missing_modality_mask,
)


def test_select_columns_by_ablation():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2"],
            "y60": [0, 1],
            "time_months": [70, 10],
            "event": [0, 1],
            "age": [50, 60],
            "expr__G1": [0.1, 0.2],
            "cna__C1": [0, 1],
        }
    )
    out = select_columns_by_ablation(df, label_cols=["y60", "time_months", "event"], spec=AblationSpec(True, False, False))
    assert "age" in out.columns
    assert "expr__G1" not in out.columns
    assert "cna__C1" not in out.columns


def test_apply_missing_modality_mask_sets_nans():
    df = pd.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(10)],
            "expr__G1": np.arange(10, dtype=float),
            "cna__C1": np.arange(10, dtype=float),
        }
    )
    out = apply_missing_modality_mask(df, frac_missing_expr=0.3, frac_missing_cna=0.2, seed=0)
    assert out["expr__G1"].isna().sum() in (3, 4)  # rounding tolerance
    assert out["cna__C1"].isna().sum() in (2, 1, 3)  # rounding tolerance
