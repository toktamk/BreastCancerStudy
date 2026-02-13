# tests/test_assemble_cohort.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.assemble_cohort import (
    CohortAssemblyConfig,
    CohortPaths,
    assemble_multimodal_cohort,
)


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _make_minimal_processed_tree(tmp_path: Path) -> Path:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    return processed


def test_assemble_multimodal_cohort_complete_case_happy_path(tmp_path: Path):
    """
    Complete-case multimodal cohort:
      - all patients have labels
      - all patients have a sample mapping
      - all samples have expression & CNA values
    Expect:
      - all patients retained
      - expr__ and cna__ namespaced columns exist
      - labels merged into cohort_df
    """
    processed = _make_minimal_processed_tree(tmp_path)

    # --- Clinical patient table (ETL stage can keep lots of columns; we only need id + some covariate)
    clinical_patient = pd.DataFrame(
        {
            "PATIENT_ID": ["P1", "P2", "P3"],
            "AGE": [50, 60, 55],
            # These may exist in clinical_patient (from earlier standardization), but assembler excludes them by default.
            "time_months": [70, 10, 80],
            "event": [0, 1, 1],
        }
    )

    # --- Sample map (canonical)
    sample_map = pd.DataFrame(
        {"sample_id": ["S1", "S2", "S3"], "patient_id": ["P1", "P2", "P3"]}
    )

    # --- Labels
    labels_survival = pd.DataFrame(
        {"patient_id": ["P1", "P2", "P3"], "time_months": [70, 10, 80], "event": [0, 1, 1]}
    )
    labels_5yr = pd.DataFrame(
        {"patient_id": ["P1", "P2", "P3"], "y60": [0, 1, 0], "horizon_months": [60.0, 60.0, 60.0]}
    )

    # --- Expression long (sample-indexed)
    expression_long = pd.DataFrame(
        {
            "sample_id": ["S1", "S1", "S2", "S2", "S3", "S3"],
            "feature": ["G1", "G2", "G1", "G2", "G1", "G2"],
            "value": [0.1, 0.2, 1.0, 1.2, 0.0, -0.1],
        }
    )

    # --- CNA long (sample-indexed)
    cna_long = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "feature": ["C1", "C1", "C1"],
            "value": [0, 1, -1],
        }
    )

    # Write expected processed files
    _save(clinical_patient, processed / "clinical_patient.parquet")
    _save(pd.DataFrame({"dummy": []}), processed / "clinical_sample.parquet")  # not used, but path required
    _save(sample_map, processed / "sample_map.parquet")
    _save(labels_survival, processed / "labels_survival.parquet")
    _save(labels_5yr, processed / "labels_5yr.parquet")
    _save(expression_long, processed / "expression_long.parquet")
    _save(cna_long, processed / "cna_long.parquet")

    paths = CohortPaths(
        clinical_patient=processed / "clinical_patient.parquet",
        clinical_sample=processed / "clinical_sample.parquet",
        sample_map=processed / "sample_map.parquet",
        labels_survival=processed / "labels_survival.parquet",
        labels_5yr=processed / "labels_5yr.parquet",
        expression_long=processed / "expression_long.parquet",
        cna_long=processed / "cna_long.parquet",
    )

    # Keep all features (disable filtering) for predictable tests
    config = CohortAssemblyConfig(
        require_expression=True,
        require_cna=True,
        expr_min_non_missing_frac=0.0,
        cna_min_non_missing_frac=0.0,
        expr_top_n_by_variance=None,
        cna_top_n_by_variance=None,
        strict=True,
    )

    artifacts = assemble_multimodal_cohort(paths=paths, config=config)

    # Patient count
    assert artifacts.cohort_df["patient_id"].nunique() == 3

    # Namespaced modality columns
    expr_cols = [c for c in artifacts.cohort_df.columns if c.startswith("expr__")]
    cna_cols = [c for c in artifacts.cohort_df.columns if c.startswith("cna__")]
    assert set(expr_cols) == {"expr__G1", "expr__G2"}
    assert set(cna_cols) == {"cna__C1"}

    # Labels merged into the final cohort dataframe
    assert "time_months" in artifacts.cohort_df.columns
    assert "event" in artifacts.cohort_df.columns
    assert "y60" in artifacts.cohort_df.columns
    assert artifacts.cohort_df["y60"].isin([0, 1]).all()

    # Clinical matrix alignment
    assert artifacts.x_clinical.index.is_unique
    assert list(artifacts.x_clinical.index) == sorted(["P1", "P2", "P3"])


def test_complete_case_drops_patient_missing_expression(tmp_path: Path):
    """
    If require_expression=True, any patient without expression coverage must be dropped.
    """
    processed = _make_minimal_processed_tree(tmp_path)

    clinical_patient = pd.DataFrame({"PATIENT_ID": ["P1", "P2"], "AGE": [50, 60]})
    sample_map = pd.DataFrame({"sample_id": ["S1", "S2"], "patient_id": ["P1", "P2"]})

    labels_survival = pd.DataFrame({"patient_id": ["P1", "P2"], "time_months": [70, 10], "event": [0, 1]})
    labels_5yr = pd.DataFrame({"patient_id": ["P1", "P2"], "y60": [0, 1], "horizon_months": [60.0, 60.0]})

    # Expression only for S1 → P1
    expression_long = pd.DataFrame({"sample_id": ["S1"], "feature": ["G1"], "value": [0.1]})
    # CNA for both
    cna_long = pd.DataFrame({"sample_id": ["S1", "S2"], "feature": ["C1", "C1"], "value": [0, 1]})

    _save(clinical_patient, processed / "clinical_patient.parquet")
    _save(pd.DataFrame({"dummy": []}), processed / "clinical_sample.parquet")
    _save(sample_map, processed / "sample_map.parquet")
    _save(labels_survival, processed / "labels_survival.parquet")
    _save(labels_5yr, processed / "labels_5yr.parquet")
    _save(expression_long, processed / "expression_long.parquet")
    _save(cna_long, processed / "cna_long.parquet")

    paths = CohortPaths(
        clinical_patient=processed / "clinical_patient.parquet",
        clinical_sample=processed / "clinical_sample.parquet",
        sample_map=processed / "sample_map.parquet",
        labels_survival=processed / "labels_survival.parquet",
        labels_5yr=processed / "labels_5yr.parquet",
        expression_long=processed / "expression_long.parquet",
        cna_long=processed / "cna_long.parquet",
    )

    config = CohortAssemblyConfig(
        require_expression=True,
        require_cna=True,
        expr_min_non_missing_frac=0.0,
        cna_min_non_missing_frac=0.0,
        expr_top_n_by_variance=None,
        cna_top_n_by_variance=None,
        strict=True,
    )

    artifacts = assemble_multimodal_cohort(paths=paths, config=config)
    assert artifacts.cohort_df["patient_id"].nunique() == 1
    assert artifacts.cohort_df["patient_id"].iloc[0] == "P1"


def test_complete_case_drops_patient_missing_cna(tmp_path: Path):
    """
    If require_cna=True, any patient without CNA coverage must be dropped.
    """
    processed = _make_minimal_processed_tree(tmp_path)

    clinical_patient = pd.DataFrame({"PATIENT_ID": ["P1", "P2"], "AGE": [50, 60]})
    sample_map = pd.DataFrame({"sample_id": ["S1", "S2"], "patient_id": ["P1", "P2"]})

    labels_survival = pd.DataFrame({"patient_id": ["P1", "P2"], "time_months": [70, 10], "event": [0, 1]})
    labels_5yr = pd.DataFrame({"patient_id": ["P1", "P2"], "y60": [0, 1], "horizon_months": [60.0, 60.0]})

    # Expression for both
    expression_long = pd.DataFrame(
        {"sample_id": ["S1", "S2"], "feature": ["G1", "G1"], "value": [0.1, 0.2]}
    )
    # CNA only for S1
    cna_long = pd.DataFrame({"sample_id": ["S1"], "feature": ["C1"], "value": [0]})

    _save(clinical_patient, processed / "clinical_patient.parquet")
    _save(pd.DataFrame({"dummy": []}), processed / "clinical_sample.parquet")
    _save(sample_map, processed / "sample_map.parquet")
    _save(labels_survival, processed / "labels_survival.parquet")
    _save(labels_5yr, processed / "labels_5yr.parquet")
    _save(expression_long, processed / "expression_long.parquet")
    _save(cna_long, processed / "cna_long.parquet")

    paths = CohortPaths(
        clinical_patient=processed / "clinical_patient.parquet",
        clinical_sample=processed / "clinical_sample.parquet",
        sample_map=processed / "sample_map.parquet",
        labels_survival=processed / "labels_survival.parquet",
        labels_5yr=processed / "labels_5yr.parquet",
        expression_long=processed / "expression_long.parquet",
        cna_long=processed / "cna_long.parquet",
    )

    config = CohortAssemblyConfig(
        require_expression=True,
        require_cna=True,
        expr_min_non_missing_frac=0.0,
        cna_min_non_missing_frac=0.0,
        expr_top_n_by_variance=None,
        cna_top_n_by_variance=None,
        strict=True,
    )

    artifacts = assemble_multimodal_cohort(paths=paths, config=config)
    assert artifacts.cohort_df["patient_id"].nunique() == 1
    assert artifacts.cohort_df["patient_id"].iloc[0] == "P1"


def test_patient_multiple_samples_aggregation_first(tmp_path: Path):
    """
    A patient has multiple samples. With aggregation_policy='first',
    we keep lexicographically smallest sample_id per patient.
    """
    processed = _make_minimal_processed_tree(tmp_path)

    clinical_patient = pd.DataFrame({"PATIENT_ID": ["P1"], "AGE": [50]})
    # P1 has two samples, S1 and S2
    sample_map = pd.DataFrame({"sample_id": ["S2", "S1"], "patient_id": ["P1", "P1"]})

    labels_survival = pd.DataFrame({"patient_id": ["P1"], "time_months": [70], "event": [0]})
    labels_5yr = pd.DataFrame({"patient_id": ["P1"], "y60": [0], "horizon_months": [60.0]})

    # Expression differs by sample, so aggregation matters
    expression_long = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "feature": ["G1", "G1"],
            "value": [0.1, 9.9],
        }
    )
    cna_long = pd.DataFrame({"sample_id": ["S1", "S2"], "feature": ["C1", "C1"], "value": [0, 0]})

    _save(clinical_patient, processed / "clinical_patient.parquet")
    _save(pd.DataFrame({"dummy": []}), processed / "clinical_sample.parquet")
    _save(sample_map, processed / "sample_map.parquet")
    _save(labels_survival, processed / "labels_survival.parquet")
    _save(labels_5yr, processed / "labels_5yr.parquet")
    _save(expression_long, processed / "expression_long.parquet")
    _save(cna_long, processed / "cna_long.parquet")

    paths = CohortPaths(
        clinical_patient=processed / "clinical_patient.parquet",
        clinical_sample=processed / "clinical_sample.parquet",
        sample_map=processed / "sample_map.parquet",
        labels_survival=processed / "labels_survival.parquet",
        labels_5yr=processed / "labels_5yr.parquet",
        expression_long=processed / "expression_long.parquet",
        cna_long=processed / "cna_long.parquet",
    )

    config = CohortAssemblyConfig(
        aggregation_policy="first",
        require_expression=True,
        require_cna=True,
        expr_min_non_missing_frac=0.0,
        cna_min_non_missing_frac=0.0,
        expr_top_n_by_variance=None,
        cna_top_n_by_variance=None,
        strict=True,
    )

    artifacts = assemble_multimodal_cohort(paths=paths, config=config)

    # For policy='first', lexicographically smallest sample is S1; value should be 0.1 not 9.9
    g1_col = "expr__G1"
    assert pytest.approx(float(artifacts.cohort_df[g1_col].iloc[0]), rel=1e-9) == 0.1


def test_patient_multiple_samples_aggregation_mean(tmp_path: Path):
    """
    A patient has multiple samples. With aggregation_policy='mean',
    we average values across samples per feature.
    """
    processed = _make_minimal_processed_tree(tmp_path)

    clinical_patient = pd.DataFrame({"PATIENT_ID": ["P1"], "AGE": [50]})
    sample_map = pd.DataFrame({"sample_id": ["S1", "S2"], "patient_id": ["P1", "P1"]})

    labels_survival = pd.DataFrame({"patient_id": ["P1"], "time_months": [70], "event": [0]})
    labels_5yr = pd.DataFrame({"patient_id": ["P1"], "y60": [0], "horizon_months": [60.0]})

    expression_long = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "feature": ["G1", "G1"],
            "value": [0.0, 1.0],
        }
    )
    cna_long = pd.DataFrame({"sample_id": ["S1", "S2"], "feature": ["C1", "C1"], "value": [0, 0]})

    _save(clinical_patient, processed / "clinical_patient.parquet")
    _save(pd.DataFrame({"dummy": []}), processed / "clinical_sample.parquet")
    _save(sample_map, processed / "sample_map.parquet")
    _save(labels_survival, processed / "labels_survival.parquet")
    _save(labels_5yr, processed / "labels_5yr.parquet")
    _save(expression_long, processed / "expression_long.parquet")
    _save(cna_long, processed / "cna_long.parquet")

    paths = CohortPaths(
        clinical_patient=processed / "clinical_patient.parquet",
        clinical_sample=processed / "clinical_sample.parquet",
        sample_map=processed / "sample_map.parquet",
        labels_survival=processed / "labels_survival.parquet",
        labels_5yr=processed / "labels_5yr.parquet",
        expression_long=processed / "expression_long.parquet",
        cna_long=processed / "cna_long.parquet",
    )

    config = CohortAssemblyConfig(
        aggregation_policy="mean",
        require_expression=True,
        require_cna=True,
        expr_min_non_missing_frac=0.0,
        cna_min_non_missing_frac=0.0,
        expr_top_n_by_variance=None,
        cna_top_n_by_variance=None,
        strict=True,
    )

    artifacts = assemble_multimodal_cohort(paths=paths, config=config)

    # Mean of [0.0, 1.0] = 0.5
    g1_col = "expr__G1"
    assert pytest.approx(float(artifacts.cohort_df[g1_col].iloc[0]), rel=1e-9) == 0.5


def test_strict_mode_raises_on_duplicate_patient_labels(tmp_path: Path):
    """
    In strict=True mode, labels_survival and labels_5yr must have unique patient_id.
    """
    processed = _make_minimal_processed_tree(tmp_path)

    clinical_patient = pd.DataFrame({"PATIENT_ID": ["P1"], "AGE": [50]})
    sample_map = pd.DataFrame({"sample_id": ["S1"], "patient_id": ["P1"]})

    # Duplicate patient_id row in survival labels
    labels_survival = pd.DataFrame(
        {"patient_id": ["P1", "P1"], "time_months": [70, 80], "event": [0, 0]}
    )
    labels_5yr = pd.DataFrame({"patient_id": ["P1"], "y60": [0], "horizon_months": [60.0]})

    expression_long = pd.DataFrame({"sample_id": ["S1"], "feature": ["G1"], "value": [0.1]})
    cna_long = pd.DataFrame({"sample_id": ["S1"], "feature": ["C1"], "value": [0]})

    _save(clinical_patient, processed / "clinical_patient.parquet")
    _save(pd.DataFrame({"dummy": []}), processed / "clinical_sample.parquet")
    _save(sample_map, processed / "sample_map.parquet")
    _save(labels_survival, processed / "labels_survival.parquet")
    _save(labels_5yr, processed / "labels_5yr.parquet")
    _save(expression_long, processed / "expression_long.parquet")
    _save(cna_long, processed / "cna_long.parquet")

    paths = CohortPaths(
        clinical_patient=processed / "clinical_patient.parquet",
        clinical_sample=processed / "clinical_sample.parquet",
        sample_map=processed / "sample_map.parquet",
        labels_survival=processed / "labels_survival.parquet",
        labels_5yr=processed / "labels_5yr.parquet",
        expression_long=processed / "expression_long.parquet",
        cna_long=processed / "cna_long.parquet",
    )

    config = CohortAssemblyConfig(
        strict=True,
        require_expression=True,
        require_cna=True,
        expr_min_non_missing_frac=0.0,
        cna_min_non_missing_frac=0.0,
        expr_top_n_by_variance=None,
        cna_top_n_by_variance=None,
    )

    with pytest.raises(ValueError):
        _ = assemble_multimodal_cohort(paths=paths, config=config)
