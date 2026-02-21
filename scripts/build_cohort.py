"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.assemble_cohort import CohortPaths, CohortAssemblyConfig, assemble_multimodal_cohort


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed")
    ap.add_argument("--out_path", type=str, default="data/processed/cohorts/metabric_complete_case_v1.parquet")
    ap.add_argument("--require_expr", action="store_true")
    ap.add_argument("--require_cna", action="store_true")
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    out_path = Path(args.out_path)
    _ensure_dir(out_path.parent)

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
        require_expression=bool(args.require_expr),
        require_cna=bool(args.require_cna),
        expr_min_non_missing_frac=0.0,
        cna_min_non_missing_frac=0.0,
        expr_top_n_by_variance=None,
        cna_top_n_by_variance=None,
        strict=True,
    )


    artifacts = assemble_multimodal_cohort(paths=paths, config=config)

    artifacts.cohort_df.to_parquet(out_path, index=False)
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(artifacts.manifest, indent=2), encoding="utf-8"
    )

    print(f" Saved cohort: {out_path} (n={len(artifacts.cohort_df)})")
    print(f" Saved manifest: {out_path.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()
