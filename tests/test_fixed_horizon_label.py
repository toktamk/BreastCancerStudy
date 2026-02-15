from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.labeling import build_fixed_horizon_label


def test_build_fixed_horizon_label_handles_censoring_before_horizon():
    df = pd.DataFrame(
        {
            "patient_id": ["P1", "P2", "P3", "P4"],
            "time_months": [12, 80, 30, 70],
            "event": [1, 0, 0, 1],
        }
    )
    out = build_fixed_horizon_label(df, horizon_months=60.0, label_col="y60").df

    # P1 died before 60 => 1
    assert out.loc[out["patient_id"] == "P1", "y60"].iloc[0] == 1.0

    # P2 censored after 60 => known negative (survived at least 60)
    assert out.loc[out["patient_id"] == "P2", "y60"].iloc[0] == 0.0

    # P3 censored before 60 => unknown
    assert np.isnan(out.loc[out["patient_id"] == "P3", "y60"].iloc[0])

    # P4 died after 60 => negative at 60 horizon
    assert out.loc[out["patient_id"] == "P4", "y60"].iloc[0] == 0.0
    assert "horizon_months" in out.columns
    assert (out["horizon_months"].dropna().unique() == [60.0]).all()

