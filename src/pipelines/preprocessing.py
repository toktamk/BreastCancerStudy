from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SelectMode = Literal["variance", "sparsity", "hybrid"]


@dataclass
class PreprocessBundle:
    feature_cols: list[str]
    transformer: Pipeline  # ColumnTransformer + Hybrid reducer

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].copy()
        out = self.transformer.transform(X)
        # Hybrid reducer returns dense np.ndarray
        return np.asarray(out)


def _sparse_col_variance(X: sp.spmatrix) -> np.ndarray:
    """
    Column-wise variance for sparse matrices without densifying.
    var = E[X^2] - (E[X])^2
    """
    n = X.shape[0]
    if n <= 1:
        return np.zeros(X.shape[1], dtype=float)

    # mean
    mu = np.asarray(X.mean(axis=0)).ravel()
    # mean of squares
    mu2 = np.asarray(X.multiply(X).mean(axis=0)).ravel()
    var = mu2 - mu ** 2
    # numerical safety
    var[var < 0] = 0.0
    return var


def _sparse_col_sparsity(X: sp.spmatrix) -> np.ndarray:
    """
    Column-wise sparsity: fraction of zeros.
    sparsity = 1 - nnz/n
    """
    n = X.shape[0]
    nnz_per_col = np.diff(X.tocsc().indptr)
    return 1.0 - (nnz_per_col / max(n, 1))


class HybridSelectAndSVD(BaseEstimator, TransformerMixin):
    """
    Fit on train:
      1) Fit TruncatedSVD with max_components, then keep only first k components
         where cumulative explained_variance_ratio_ >= svd_var_threshold.
      2) Select top_k_features original features based on variance/sparsity/hybrid.
      3) Output concatenation: [X_selected_original, X_svd_selected]
    """

    def __init__(
        self,
        *,
        top_k_features: int = 2000,
        select_mode: SelectMode = "variance",
        svd_max_components: int = 1000,
        svd_var_threshold: float = 0.90,
        svd_min_components: int = 100,
        random_state: int = 1337,
        output_dense: bool = True,
    ):
        self.top_k_features = int(top_k_features)
        self.select_mode = select_mode
        self.svd_max_components = int(svd_max_components)
        self.svd_var_threshold = float(svd_var_threshold)
        self.svd_min_components = int(svd_min_components)
        self.random_state = int(random_state)
        self.output_dense = bool(output_dense)

        # learned
        self.selected_idx_: Optional[np.ndarray] = None
        self.k_svd_: Optional[int] = None
        self.svd_: Optional[TruncatedSVD] = None

    def fit(self, X, y=None):
        X = self._ensure_2d(X)

        # --- Feature selection indices ---
        if sp.issparse(X):
            var = _sparse_col_variance(X)
            spars = _sparse_col_sparsity(X)
        else:
            Xd = np.asarray(X)
            var = Xd.var(axis=0)
            spars = (Xd == 0).mean(axis=0)

        if self.select_mode == "variance":
            score = var
            k = min(self.top_k_features, X.shape[1])
            self.selected_idx_ = np.argsort(score)[-k:]
        elif self.select_mode == "sparsity":
            # Prefer less sparse (more present) => lower sparsity
            score = 1.0 - spars
            k = min(self.top_k_features, X.shape[1])
            self.selected_idx_ = np.argsort(score)[-k:]
        elif self.select_mode == "hybrid":
            # 1) Remove very sparse features first
            # Assume spars is fraction missing per feature (0 = no missing, 1 = all missing)
            sparsity_threshold = getattr(self, "sparsity_threshold", 0.9)

            valid_mask = spars <= sparsity_threshold
            valid_idx = np.where(valid_mask)[0]

            if len(valid_idx) == 0:
                raise ValueError("No features left after sparsity filtering.")

            # 2) Rank remaining features by variance
            var_valid = var[valid_idx]
            k = min(self.top_k_features, len(valid_idx))

            top_local = np.argsort(var_valid)[-k:]
            self.selected_idx_ = valid_idx[top_local]
        else:
            raise ValueError("select_mode must be 'variance', 'sparsity', or 'hybrid'.")

        # --- Truncated SVD ---
        max_comp = min(self.svd_max_components, max(2, X.shape[0] - 1), X.shape[1])
        self.svd_ = TruncatedSVD(n_components=max_comp, random_state=self.random_state)
        self.svd_.fit(X)

        evr = np.asarray(self.svd_.explained_variance_ratio_, dtype=float)
        cum = np.cumsum(evr)

        # choose k_svd as first index reaching threshold
        k_svd = int(np.searchsorted(cum, self.svd_var_threshold) + 1)
        k_svd = max(k_svd, self.svd_min_components)
        k_svd = min(k_svd, max_comp)

        self.k_svd_ = k_svd
        return self

    def transform(self, X):
        if self.selected_idx_ is None or self.svd_ is None or self.k_svd_ is None:
            raise RuntimeError("HybridSelectAndSVD must be fit before transform().")

        X = self._ensure_2d(X)

        # selected original features
        X_sel = X[:, self.selected_idx_]

        # SVD components
        X_svd_full = self.svd_.transform(X)
        X_svd = X_svd_full[:, : self.k_svd_]

        # concatenate
        if sp.issparse(X_sel):
            X_sel = X_sel.toarray() if self.output_dense else X_sel
        else:
            X_sel = np.asarray(X_sel)

        out = np.hstack([X_sel, np.asarray(X_svd)])

        return out

    @staticmethod
    def _ensure_2d(X):
        if sp.issparse(X):
            return X
        X = np.asarray(X)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X


def build_preprocess_pipeline(
    *,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    id_col: str = "patient_id",
    # --- NEW: hybrid DR knobs ---
    do_hybrid_reduction: bool = True,
    top_k_features: int = 2000,
    select_mode: SelectMode = "variance",
    svd_max_components: int = 1000,
    svd_var_threshold: float = 0.90,
    svd_min_components: int = 10,
    random_state: int = 1337,
) -> PreprocessBundle:
    X_train = train_df[feature_cols].copy()

    cat_cols = [
        c for c in feature_cols
        if X_train[c].dtype == "object"
        or str(X_train[c].dtype).startswith("string")
        or str(X_train[c].dtype) == "category"
    ]
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

    if do_hybrid_reduction:
        reducer = HybridSelectAndSVD(
            top_k_features=top_k_features,
            select_mode=select_mode,
            svd_max_components=svd_max_components,
            svd_var_threshold=svd_var_threshold,
            svd_min_components=svd_min_components,
            random_state=random_state,
            output_dense=True,
        )
        full = Pipeline(
            steps=[
                ("preprocess", ct),
                ("hybrid", reducer),
            ]
        )
    else:
        full = Pipeline(steps=[("preprocess", ct)])

    full.fit(X_train)
    return PreprocessBundle(feature_cols=feature_cols, transformer=full)
