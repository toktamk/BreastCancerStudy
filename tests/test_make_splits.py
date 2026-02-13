# tests/test_make_splits.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.splits.make_splits import SplitConfig, make_splits


def test_make_splits_stratified_and_disjoint():
    df = pd.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(200)],
            "y60": ([0] * 150) + ([1] * 50),
        }
    )

    cfg = SplitConfig(seed=42, test_size=0.2, val_size=0.2, stratify_col="y60")
    splits = make_splits(df, cfg)

    tr, va, te = splits["train"], splits["val"], splits["test"]

    # disjoint
    assert len(set(tr) & set(va)) == 0
    assert len(set(tr) & set(te)) == 0
    assert len(set(va) & set(te)) == 0

    # coverage
    assert len(tr) + len(va) + len(te) == len(df)

    # label proportions roughly preserved
    def rate(ix):
        return df.iloc[ix]["y60"].mean()

    overall = df["y60"].mean()
    assert abs(rate(tr) - overall) < 0.05
    assert abs(rate(va) - overall) < 0.05
    assert abs(rate(te) - overall) < 0.05
