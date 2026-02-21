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
from typing import Dict, List

import pandas as pd
import subprocess
import sys


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="outputs/ablation_suite")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cohort = Path(args.cohort)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    # Define ablations
    runs = [
        ("clinical_only", ["--use_clinical"]),
        ("clinical_expr", ["--use_clinical", "--use_expr"]),
        ("clinical_cna", ["--use_clinical", "--use_cna"]),
        ("all_modalities", ["--use_clinical", "--use_expr", "--use_cna"]),
    ]

    # Missing modality stress tests (only meaningful when omics used)
    stress = [
        ("no_stress", []),
        ("miss_expr_20", ["--miss_expr", "0.2"]),
        ("miss_cna_20", ["--miss_cna", "0.2"]),
        ("miss_both_20", ["--miss_expr", "0.2", "--miss_cna", "0.2"]),
    ]

    rows: List[Dict] = []

    for run_name, flags in runs:
        for stress_name, sflags in stress:
            tag = f"{run_name}__{stress_name}"
            run_out = outdir / tag
            _ensure_dir(run_out)

            cmd = [
                sys.executable,
                "scripts/run_survival_models.py",
                "--cohort", str(cohort),
                "--outdir", str(run_out),
                "--seed", str(args.seed),
            ] + flags + sflags

            subprocess.run(cmd, check=True)

            metrics = json.loads((run_out / "metrics.json").read_text(encoding="utf-8"))
            row = {
                "tag": tag,
                "harrell_c_index": metrics["survival"]["harrell_c_index"],
                "auroc_5yr": metrics["binary_at_horizon"]["discrimination"]["auroc"],
                "brier_5yr": metrics["binary_at_horizon"]["calibration"]["brier"],
                "ece_5yr": metrics["binary_at_horizon"]["calibration"]["ece"],
            }
            rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["harrell_c_index", "auroc_5yr"], ascending=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    print(f"Saved: {outdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
