"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskStrata:
    df: pd.DataFrame  # includes patient_id, time, event, risk, group


def make_risk_strata(
    *,
    df: pd.DataFrame,
    time_col: str = "time_months",
    event_col: str = "event",
    risk_col: str = "risk",
    n_bins: int = 3,
    labels: Optional[list[str]] = None,
) -> RiskStrata:
    """
    Splits patients into quantile bins by predicted risk score.
    Higher risk => higher bin.

    Output column: risk_group (categorical string)
    """
    if labels is None:
        if n_bins == 3:
            labels = ["low", "intermediate", "high"]
        else:
            labels = [f"q{i+1}" for i in range(n_bins)]

    out = df[[time_col, event_col, risk_col]].copy()
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    out[event_col] = pd.to_numeric(out[event_col], errors="coerce")
    out[risk_col] = pd.to_numeric(out[risk_col], errors="coerce")

    out = out.dropna()
    out = out[out[time_col] >= 0]
    out = out[out[event_col].isin([0, 1])]

    # Quantile-based bins
    q = pd.qcut(out[risk_col], q=n_bins, labels=labels, duplicates="drop")
    out["risk_group"] = q.astype("string")

    return RiskStrata(df=out)


def km_summary_by_group(
    strata: RiskStrata,
    time_col: str = "time_months",
    event_col: str = "event",
) -> pd.DataFrame:
    """
    Returns a simple KM survival estimate per group at a few key time points.
    (This is table-first, plot-optional. Plotting can be added later.)
    """
    try:
        from lifelines import KaplanMeierFitter
    except Exception as e:  # pragma: no cover
        raise ImportError("Kaplan–Meier summaries require lifelines: pip install lifelines") from e

    kmf = KaplanMeierFitter()
    rows = []
    times = [12, 36, 60, 120]  # months

    for g, gdf in strata.df.groupby("risk_group"):
        kmf.fit(gdf[time_col].astype(float), event_observed=gdf[event_col].astype(int))
        surv = kmf.survival_function_at_times(times).reset_index()
        # surv columns: index, KM_estimate
        row = {"risk_group": str(g), "n": int(len(gdf))}
        for t, s in zip(times, surv.iloc[:, 1].values):
            row[f"S({t}m)"] = float(s)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("risk_group")
