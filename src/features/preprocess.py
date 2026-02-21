"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


@dataclass(frozen=True)
class FeatureSpec:
    id_col: str = "patient_id"
    label_col: str = "y60"
    time_col: str = "time_months"
    event_col: str = "event"

    # modality namespaces in cohort_df
    expr_prefix: str = "expr__"
    cna_prefix: str = "cna__"

    # clinical columns policy:
    # "all_non_label": keep all non-label, non-omics columns as clinical
    clinical_mode: str = "all_non_label"


def split_feature_columns(df: pd.DataFrame, spec: FeatureSpec) -> Dict[str, List[str]]:
    """
    Split columns into clinical / expression / CNA blocks based on naming conventions.
    """
    cols = list(df.columns)

    expr_cols = [c for c in cols if c.startswith(spec.expr_prefix)]
    cna_cols = [c for c in cols if c.startswith(spec.cna_prefix)]

    reserved = {spec.id_col, spec.label_col, spec.time_col, spec.event_col}
    reserved |= set(expr_cols) | set(cna_cols)

    if spec.clinical_mode == "all_non_label":
        clinical_cols = [c for c in cols if c not in reserved]
    else:
        raise ValueError(f"Unknown clinical_mode: {spec.clinical_mode}")

    return {"clinical": clinical_cols, "expr": expr_cols, "cna": cna_cols}


def build_preprocessor(df: pd.DataFrame, feature_cols: Dict[str, List[str]]) -> Tuple[ColumnTransformer, List[str]]:
    """
    Create sklearn ColumnTransformer that:
      - imputes + one-hot encodes categoricals in clinical
      - imputes + scales numeric clinical
      - imputes + scales omics blocks
    Returns transformer and final feature block order (for traceability).
    """
    clinical_cols = feature_cols["clinical"]
    expr_cols = feature_cols["expr"]
    cna_cols = feature_cols["cna"]

    # Determine clinical numeric vs categorical
    clinical_num = [c for c in clinical_cols if pd.api.types.is_numeric_dtype(df[c])]
    clinical_cat = [c for c in clinical_cols if c not in clinical_num]

    clinical_num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ]
    )
    clinical_cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    omics_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    transformers = []
    if clinical_num:
        transformers.append(("clinical_num", clinical_num_pipe, clinical_num))
    if clinical_cat:
        transformers.append(("clinical_cat", clinical_cat_pipe, clinical_cat))
    if expr_cols:
        transformers.append(("expr", omics_pipe, expr_cols))
    if cna_cols:
        transformers.append(("cna", omics_pipe, cna_cols))

    ct = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)

    # Feature block order: for logging/debug, not the expanded one-hot names.
    block_order = []
    if clinical_num:
        block_order.append("clinical_num")
    if clinical_cat:
        block_order.append("clinical_cat")
    if expr_cols:
        block_order.append("expr")
    if cna_cols:
        block_order.append("cna")

    return ct, block_order


def fit_transform_split(
    cohort_df: pd.DataFrame,
    train_ids: pd.Index,
    val_ids: pd.Index,
    test_ids: pd.Index,
    spec: FeatureSpec = FeatureSpec(),
) -> Dict[str, object]:
    """
    Fit preprocessing on train only, transform train/val/test.

    Returns:
      {
        "preprocessor": fitted ColumnTransformer,
        "X_train": np.ndarray, "X_val": np.ndarray, "X_test": np.ndarray,
        "y_train": np.ndarray, ...,
        "meta": {feature_cols, block_order}
      }
    """
    if spec.id_col not in cohort_df.columns:
        raise KeyError(f"cohort_df missing '{spec.id_col}'")

    df = cohort_df.copy()
    df[spec.id_col] = df[spec.id_col].astype(str)

    df = df.set_index(spec.id_col)
    df_train = df.loc[train_ids]
    df_val = df.loc[val_ids]
    df_test = df.loc[test_ids]

    # Labels
    y_train = pd.to_numeric(df_train[spec.label_col], errors="coerce").to_numpy()
    y_val = pd.to_numeric(df_val[spec.label_col], errors="coerce").to_numpy()
    y_test = pd.to_numeric(df_test[spec.label_col], errors="coerce").to_numpy()

    # Survival labels (for later)
    t_train = pd.to_numeric(df_train[spec.time_col], errors="coerce").to_numpy()
    e_train = pd.to_numeric(df_train[spec.event_col], errors="coerce").to_numpy()
    t_val = pd.to_numeric(df_val[spec.time_col], errors="coerce").to_numpy()
    e_val = pd.to_numeric(df_val[spec.event_col], errors="coerce").to_numpy()
    t_test = pd.to_numeric(df_test[spec.time_col], errors="coerce").to_numpy()
    e_test = pd.to_numeric(df_test[spec.event_col], errors="coerce").to_numpy()

    feature_cols = split_feature_columns(df.reset_index(), spec)
    preprocessor, block_order = build_preprocessor(df.reset_index(), feature_cols)

    X_train = preprocessor.fit_transform(df_train.reset_index())
    X_val = preprocessor.transform(df_val.reset_index())
    X_test = preprocessor.transform(df_test.reset_index())

    return {
        "preprocessor": preprocessor,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "t_train": t_train, "e_train": e_train,
        "t_val": t_val, "e_val": e_val,
        "t_test": t_test, "e_test": e_test,
        "meta": {"feature_cols": feature_cols, "block_order": block_order},
    }
