# tests/test_load_expression.py
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.load_expression import load_expression, load_expression_long, maybe_make_wide


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_expression_long_basic(tmp_path: Path):
    p = _write(
        tmp_path / "data_mrna.txt",
        "# comment\n"
        "Hugo_Symbol\tS1\tS2\n"
        "TP53\t0.1\t-0.2\n"
        "BRCA1\t1.0\tNA\n",
    )

    long = load_expression_long(p)
    # 3 numeric values (NA dropped)
    assert len(long) == 3
    assert set(long.columns) == {"sample_id", "feature", "value"}
    assert long["sample_id"].nunique() == 2
    assert long["feature"].nunique() == 2


def test_load_expression_long_feature_id_fallback_to_first_column(tmp_path: Path):
    p = _write(
        tmp_path / "data_mrna.txt",
        "GENE\tS1\n"
        "TP53\t0.5\n",
    )
    long = load_expression_long(p, feature_id_col=None)
    assert len(long) == 1
    assert long.iloc[0]["feature"] == "TP53"
    assert long.iloc[0]["sample_id"] == "S1"


def test_maybe_make_wide_builds_small_matrix(tmp_path: Path):
    p = _write(
        tmp_path / "data_mrna.txt",
        "Hugo_Symbol\tS1\tS2\n"
        "TP53\t0.1\t-0.2\n"
        "BRCA1\t1.0\t0.0\n",
    )
    long = load_expression_long(p)
    wide = maybe_make_wide(long, max_features=10)

    assert wide is not None
    assert wide.shape == (2, 2)  # 2 samples x 2 genes
    assert "TP53" in wide.columns
    assert "BRCA1" in wide.columns


def test_maybe_make_wide_skips_if_too_many_features(tmp_path: Path):
    # Construct long df with 6 distinct features, set max_features=5 to force skip
    p = _write(
        tmp_path / "data_mrna.txt",
        "Hugo_Symbol\tS1\n"
        "G1\t1\nG2\t1\nG3\t1\nG4\t1\nG5\t1\nG6\t1\n",
    )
    long = load_expression_long(p)
    wide = maybe_make_wide(long, max_features=5)
    assert wide is None


def test_load_expression_wrapper_returns_long_and_optional_wide(tmp_path: Path):
    p = _write(
        tmp_path / "data_mrna.txt",
        "Hugo_Symbol\tS1\tS2\n"
        "TP53\t0.1\t-0.2\n"
        "BRCA1\t1.0\t0.0\n",
    )

    res = load_expression(p, build_wide=True, max_features_for_wide=100)
    assert res.expression_long is not None
    assert res.expression_wide is not None
    assert res.expression_wide.shape == (2, 2)
