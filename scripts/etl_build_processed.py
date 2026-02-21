"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.load_clinical import load_clinical_patient, load_clinical_sample
from src.data.load_expression import load_expression_long
from src.data.load_cna import load_cna_long
from src.data.labeling import build_survival_labels, build_fixed_horizon_label


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _default_paths(raw_dir: Path) -> dict[str, Path]:
    """
    Defaults for cBioPortal METABRIC downloads.
    """
    return {
        "clinical_patient": raw_dir / "data_clinical_patient.txt",
        "clinical_sample": raw_dir / "data_clinical_sample.txt",
        "expression": raw_dir / "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt",
        "cna": raw_dir / "data_cna.txt",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw")
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--horizon_months", type=float, default=60.0)

    # Optional overrides
    ap.add_argument("--clinical_patient", type=str, default="")
    ap.add_argument("--clinical_sample", type=str, default="")
    ap.add_argument("--expression", type=str, default="")
    ap.add_argument("--cna", type=str, default="")

    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    defaults = _default_paths(raw_dir)

    clinical_patient_path = Path(args.clinical_patient) if args.clinical_patient else defaults["clinical_patient"]
    clinical_sample_path = Path(args.clinical_sample) if args.clinical_sample else defaults["clinical_sample"]
    expression_path = Path(args.expression) if args.expression else defaults["expression"]
    cna_path = Path(args.cna) if args.cna else defaults["cna"]

    for p in [clinical_patient_path, clinical_sample_path, expression_path, cna_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required raw file: {p}")

    # ---- Clinical
    clinical_patient = load_clinical_patient(clinical_patient_path)
    clinical_patient.to_parquet(out_dir / "clinical_patient.parquet", index=False)

    clinical_sample = load_clinical_sample(clinical_sample_path)
    clinical_sample.to_parquet(out_dir / "clinical_sample.parquet", index=False)

    # ---- Sample map
    # Expect clinical_sample includes sample and patient identifiers.
    cs = clinical_sample.copy()
    # These are common cBioPortal column names; your loader should standardize them,
    # but we keep a defensive fallback here.
    if "sample_id" in cs.columns and "patient_id" in cs.columns:
        sample_map = cs[["sample_id", "patient_id"]].copy()
    elif "SAMPLE_ID" in cs.columns and "PATIENT_ID" in cs.columns:
        sample_map = cs[["SAMPLE_ID", "PATIENT_ID"]].rename(columns={"SAMPLE_ID": "sample_id", "PATIENT_ID": "patient_id"})
    else:
        raise RuntimeError(
            "Cannot build sample_map: clinical_sample must contain either "
            "('sample_id','patient_id') or ('SAMPLE_ID','PATIENT_ID'). "
            f"Columns seen: {list(cs.columns)}"
        )

    sample_map["sample_id"] = sample_map["sample_id"].astype(str).str.strip()
    sample_map["patient_id"] = sample_map["patient_id"].astype(str).str.strip()
    sample_map = sample_map.dropna().drop_duplicates()
    sample_map.to_parquet(out_dir / "sample_map.parquet", index=False)

    # ---- Labels (from clinical_patient standardized by your loader)
    surv = build_survival_labels(clinical_patient)
    surv.df.to_parquet(out_dir / "labels_survival.parquet", index=False)

    y5 = build_fixed_horizon_label(surv, horizon_months=float(args.horizon_months))
    y5.df.to_parquet(out_dir / "labels_5yr.parquet", index=False)

    # ---- Expression & CNA (long format)
    expr_long = load_expression_long(expression_path)
    expr_long.to_parquet(out_dir / "expression_long.parquet", index=False)

    cna_long = load_cna_long(cna_path)
    cna_long.to_parquet(out_dir / "cna_long.parquet", index=False)

    print("✅ Wrote processed artifacts to:", out_dir.resolve())
    for f in [
        "clinical_patient.parquet",
        "clinical_sample.parquet",
        "sample_map.parquet",
        "labels_survival.parquet",
        "labels_5yr.parquet",
        "expression_long.parquet",
        "cna_long.parquet",
    ]:
        print(" -", f)


if __name__ == "__main__":
    main()
