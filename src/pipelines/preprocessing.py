# src/pipelines/preprocessing.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessBundle:
    feature_cols: list[str]
    transformer: ColumnTransformer

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].copy()
        return self.transformer.transform(X)


def build_preprocess_pipeline(
    *,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    id_col: str = "patient_id",
) -> PreprocessBundle:
    X_train = train_df[feature_cols].copy()

    cat_cols = [c for c in feature_cols if X_train[c].dtype == "object" or str(X_train[c].dtype).startswith("string")]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    ct.fit(X_train)
    return PreprocessBundle(feature_cols=feature_cols, transformer=ct)
