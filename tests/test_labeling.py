"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.labeling import (
    FiveYearLabels,
    SurvivalLabels,
    build_5yr_labels,
    build_survival_labels,
    summarize_label_distribution,
)


def test_build_survival_labels_happy_path():
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P1", "P2", "P3"],
            "time_months": [12, 80, 10],
            "event": [1, 0, 1],
        }
    )
    surv = build_survival_labels(df)
    assert isinstance(surv, SurvivalLabels)
    assert set(surv.df.columns) == {"patient_id", "time_months", "event"}
    assert len(surv.df) == 3
    assert surv.df["event"].isin([0, 1]).all()


def test_build_survival_labels_drops_invalid_rows():
    df = pd.DataFrame(
        {
            "PATIENT_ID": ["P1", "P2", "P3", None],
            "time_months": [12, -1, "x", 5],
            "event": [1, 2, 0, 1],
        }
    )
    surv = build_survival_labels(df)
    # Only P1 is valid here: time=12, event=1
    assert len(surv.df) == 1
    assert surv.df.iloc[0]["patient_id"] == "P1"


def test_build_5yr_labels_exclude_censored_before_horizon():
    surv_df = pd.DataFrame(
        {
            "patient_id": ["A", "B", "C", "D"],
            "time_months": [10, 70, 20, 70],
            "event": [1, 1, 0, 0],  # C is censored before 60 -> ambiguous
        }
    )
    surv = SurvivalLabels(df=surv_df)

    labels = build_5yr_labels(surv, horizon_months=60, include_strategy="exclude_censored_before_horizon")
    assert isinstance(labels, FiveYearLabels)

    # A: event within 60 => 1
    # B: time > 60 => 0 (even though event=1 at 70)
    # C: censored before 60 => dropped
    # D: time > 60 => 0
    assert set(labels.df["patient_id"]) == {"A", "B", "D"}
    y = dict(zip(labels.df["patient_id"], labels.df["y60"]))
    assert y["A"] == 1
    assert y["B"] == 0
    assert y["D"] == 0


def test_build_5yr_labels_keep_na_for_censored_before_horizon():
    surv_df = pd.DataFrame(
        {
            "patient_id": ["A", "B", "C"],
            "time_months": [10, 70, 20],
            "event": [1, 1, 0],
        }
    )
    surv = SurvivalLabels(df=surv_df)
    labels = build_5yr_labels(surv, horizon_months=60, include_strategy="keep_with_na")

    assert len(labels.df) == 3
    row_c = labels.df[labels.df["patient_id"] == "C"].iloc[0]
    assert pd.isna(row_c["y60"])


def test_summarize_label_distribution_runs():
    surv_df = pd.DataFrame(
        {
            "patient_id": ["A", "B", "C", "D"],
            "time_months": [10, 70, 20, 70],
            "event": [1, 1, 0, 0],
        }
    )
    labels = build_5yr_labels(SurvivalLabels(df=surv_df), horizon_months=60, include_strategy="keep_with_na")
    summary = summarize_label_distribution(labels)

    assert summary.shape[0] == 1
    assert "event_rate_within_horizon" in summary.columns
    assert summary["horizon_months"].iloc[0] == 60.0
