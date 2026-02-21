"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AblationSpec:
    """
    Defines which feature blocks to keep.

    Blocks:
      - clinical: non-namespaced columns (excluding labels)
      - expr: columns starting with expr__
      - cna: columns starting with cna__
    """
    use_clinical: bool = True
    use_expr: bool = False
    use_cna: bool = False


def select_columns_by_ablation(
    cohort_df: pd.DataFrame,
    label_cols: List[str],
    id_col: str = "patient_id",
    expr_prefix: str = "expr__",
    cna_prefix: str = "cna__",
    spec: AblationSpec = AblationSpec(),
) -> pd.DataFrame:
    cols = list(cohort_df.columns)

    expr_cols = [c for c in cols if c.startswith(expr_prefix)]
    cna_cols = [c for c in cols if c.startswith(cna_prefix)]

    reserved = set([id_col] + label_cols)

    clinical_cols = [c for c in cols if c not in reserved and c not in expr_cols and c not in cna_cols]

    keep = [id_col] + label_cols
    if spec.use_clinical:
        keep += clinical_cols
    if spec.use_expr:
        keep += expr_cols
    if spec.use_cna:
        keep += cna_cols

    # preserve order, dedup
    seen = set()
    keep2 = []
    for c in keep:
        if c in cohort_df.columns and c not in seen:
            keep2.append(c)
            seen.add(c)
    return cohort_df[keep2].copy()


def apply_missing_modality_mask(
    cohort_df: pd.DataFrame,
    frac_missing_expr: float = 0.0,
    frac_missing_cna: float = 0.0,
    seed: int = 1337,
    expr_prefix: str = "expr__",
    cna_prefix: str = "cna__",
) -> pd.DataFrame:
    """
    Simulate missing modalities by setting modality block columns to NaN for a random subset of patients.

    This is a robustness stress test. Preprocessing must handle NaNs (median imputation).
    """
    rng = np.random.default_rng(seed)
    df = cohort_df.copy()

    n = len(df)
    idx = np.arange(n)

    expr_cols = [c for c in df.columns if c.startswith(expr_prefix)]
    cna_cols = [c for c in df.columns if c.startswith(cna_prefix)]

    if frac_missing_expr > 0 and expr_cols:
        m = int(round(frac_missing_expr * n))
        mask = rng.choice(idx, size=m, replace=False)
        df.loc[df.index[mask], expr_cols] = np.nan

    if frac_missing_cna > 0 and cna_cols:
        m = int(round(frac_missing_cna * n))
        mask = rng.choice(idx, size=m, replace=False)
        df.loc[df.index[mask], cna_cols] = np.nan

    return df
