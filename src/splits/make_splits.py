# src/splits/make_splits.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 1337
    test_size: float = 0.20
    val_size: float = 0.20  # fraction of (train+val) reserved for val
    stratify_col: str = "y60"
    id_col: str = "patient_id"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _check_binary(y: pd.Series, name: str) -> None:
    vals = set(pd.Series(y).dropna().unique().tolist())
    if not vals.issubset({0, 1}):
        raise ValueError(f"{name} must be binary in {{0,1}}; got {sorted(vals)}")


def make_splits(df: pd.DataFrame, config: SplitConfig) -> Dict[str, np.ndarray]:
    """
    Deterministic patient-level splits: train/val/test using stratification on y (binary).

    Returns indices (row positions) for each split.
    """
    if config.id_col not in df.columns:
        raise KeyError(f"Missing id_col='{config.id_col}' in dataframe.")
    if config.stratify_col not in df.columns:
        raise KeyError(f"Missing stratify_col='{config.stratify_col}' in dataframe.")

    ids = df[config.id_col].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("patient_id must be unique per row in the modeling cohort.")

    y = pd.to_numeric(df[config.stratify_col], errors="coerce")
    n_nan = int(pd.isna(y).sum())
    if n_nan > 0:
        raise ValueError(
            f"Stratify column '{config.stratify_col}' contains {n_nan} NaNs. "
            f"Choose a stratify_col with no missing values (e.g., 'event'), "
            f"or filter your cohort to rows with defined {config.stratify_col}."
        )

    _check_binary(y, config.stratify_col)

    idx = np.arange(len(df))

    # First split: train_val vs test
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=float(config.test_size), random_state=int(config.seed)
    )
    train_val_idx, test_idx = next(sss1.split(idx, y))

    # Second split: train vs val inside train_val
    y_train_val = y.iloc[train_val_idx].reset_index(drop=True)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=float(config.val_size), random_state=int(config.seed) + 1
    )
    tr_rel, val_rel = next(sss2.split(np.arange(len(train_val_idx)), y_train_val))

    train_idx = train_val_idx[tr_rel]
    val_idx = train_val_idx[val_rel]

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_splits(df: pd.DataFrame, splits: Dict[str, np.ndarray], out_dir: Path, config: SplitConfig) -> None:
    """
    Save patient_id lists for train/val/test to CSV and config to JSON.
    """
    _ensure_dir(out_dir)

    for split_name, split_idx in splits.items():
        out = df.iloc[split_idx][[config.id_col]].copy()
        out.to_csv(out_dir / f"{split_name}.csv", index=False)

    (out_dir / "split_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def load_splits(out_dir: Path) -> Dict[str, pd.Index]:
    """
    Load saved split CSVs and return patient_id indices.
    """
    train = pd.read_csv(out_dir / "train.csv")["patient_id"]
    val = pd.read_csv(out_dir / "val.csv")["patient_id"]
    test = pd.read_csv(out_dir / "test.csv")["patient_id"]
    return {"train": pd.Index(train), "val": pd.Index(val), "test": pd.Index(test)}
