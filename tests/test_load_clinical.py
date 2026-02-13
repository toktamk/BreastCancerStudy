# tests/test_load_clinical.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.load_clinical import (
    ClinicalColumnMap,
    ClinicalPaths,
    build_patient_sample_map,
    load_and_prepare_clinical,
    load_clinical_patient,
    load_clinical_sample,
    standardize_survival_fields,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_clinical_patient_reads_tsv_and_drops_all_empty_cols(tmp_path: Path):
    p = _write(
        tmp_path / "data_clinical_patient.txt",
        "# comment line\n"
        "PATIENT_ID\tOS_MONTHS\tOS_STATUS\tEMPTYCOL\n"
        "P1\t12\tDECEASED\t\n"
        "P2\t80\tLIVING\t\n",
    )

    df = load_clinical_patient(p)
    assert len(df) == 2
    assert "PATIENT_ID" in df.columns
    assert "OS_MONTHS" in df.columns
    assert "OS_STATUS" in df.columns
    # Column is entirely empty -> should be dropped
    assert "EMPTYCOL" not in df.columns


def test_load_clinical_sample_reads_ids(tmp_path: Path):
    p = _write(
        tmp_path / "data_clinical_sample.txt",
        "SAMPLE_ID\tPATIENT_ID\tER_STATUS\n"
        "S1\tP1\tPositive\n"
        "S2\tP2\tNegative\n",
    )

    df = load_clinical_sample(p)
    assert set(df.columns) == {"SAMPLE_ID", "PATIENT_ID", "ER_STATUS"}
    assert df["SAMPLE_ID"].tolist() == ["S1", "S2"]
    assert df["PATIENT_ID"].tolist() == ["P1", "P2"]


def test_standardize_survival_fields_with_explicit_map(tmp_path: Path):
    p = _write(
        tmp_path / "data_clinical_patient.txt",
        "PATIENT_ID\tOS_MONTHS\tOS_STATUS\n"
        "P1\t12\tDECEASED\n"
        "P2\t80\tLIVING\n"
        "P3\t10\tdead\n"
        "P4\t100\talive\n",
    )
    df = load_clinical_patient(p)

    cmap = ClinicalColumnMap(os_time_months="OS_MONTHS", os_status="OS_STATUS")
    out = standardize_survival_fields(df, column_map=cmap)

    assert "time_months" in out.columns
    assert "event" in out.columns
    assert out.loc[out["PATIENT_ID"] == "P1", "event"].iloc[0] == 1
    assert out.loc[out["PATIENT_ID"] == "P2", "event"].iloc[0] == 0
    assert out.loc[out["PATIENT_ID"] == "P3", "event"].iloc[0] == 1
    assert out.loc[out["PATIENT_ID"] == "P4", "event"].iloc[0] == 0
    assert float(out.loc[out["PATIENT_ID"] == "P1", "time_months"].iloc[0]) == 12.0


def test_standardize_survival_fields_infers_common_columns(tmp_path: Path):
    # Use canonical names that the inference function is designed to catch.
    p = _write(
        tmp_path / "data_clinical_patient.txt",
        "PATIENT_ID\tos_months\tos_status\n"
        "P1\t12\tDECEASED\n"
        "P2\t80\tLIVING\n",
    )
    df = load_clinical_patient(p)

    cmap = ClinicalColumnMap(os_time_months=None, os_status=None)
    out = standardize_survival_fields(df, column_map=cmap)

    assert out["time_months"].notna().all()
    assert out["event"].notna().all()
    assert out["event"].tolist() == [1, 0]


def test_standardize_survival_fields_raises_if_not_found(tmp_path: Path):
    p = _write(
        tmp_path / "data_clinical_patient.txt",
        "PATIENT_ID\tSOMETHING_ELSE\n"
        "P1\tX\n",
    )
    df = load_clinical_patient(p)

    cmap = ClinicalColumnMap(os_time_months=None, os_status=None)
    with pytest.raises(KeyError):
        _ = standardize_survival_fields(df, column_map=cmap)


def test_build_patient_sample_map_happy_path(tmp_path: Path):
    p = _write(
        tmp_path / "data_clinical_sample.txt",
        "SAMPLE_ID\tPATIENT_ID\tER_STATUS\n"
        "S1\tP1\tPositive\n"
        "S2\tP2\tNegative\n"
        "S2\tP2\tNegative\n",  # duplicate row should be dropped
    )
    sample_df = load_clinical_sample(p)
    m = build_patient_sample_map(sample_df)

    assert set(m.columns) == {"sample_id", "patient_id"}
    assert len(m) == 2
    assert m["sample_id"].nunique() == 2
    assert m["patient_id"].nunique() == 2


def test_build_patient_sample_map_raises_if_missing_cols():
    df = pd.DataFrame({"SAMPLE_ID": ["S1"]})
    with pytest.raises(KeyError):
        _ = build_patient_sample_map(df)  # PATIENT_ID missing


def test_load_and_prepare_clinical_end_to_end(tmp_path: Path):
    patient_path = _write(
        tmp_path / "data_clinical_patient.txt",
        "PATIENT_ID\tOS_MONTHS\tOS_STATUS\tAGE_AT_DIAGNOSIS\n"
        "P1\t12\tDECEASED\t45\n"
        "P2\t80\tLIVING\t62\n",
    )
    sample_path = _write(
        tmp_path / "data_clinical_sample.txt",
        "SAMPLE_ID\tPATIENT_ID\tER_STATUS\n"
        "S1\tP1\tPositive\n"
        "S2\tP2\tNegative\n",
    )

    paths = ClinicalPaths(patient=patient_path, sample=sample_path)
    cmap = ClinicalColumnMap(os_time_months="OS_MONTHS", os_status="OS_STATUS")

    out = load_and_prepare_clinical(paths=paths, column_map=cmap)

    assert set(out.keys()) == {"patient", "sample", "sample_map"}
    assert "time_months" in out["patient"].columns
    assert "event" in out["patient"].columns
    assert len(out["sample_map"]) == 2
