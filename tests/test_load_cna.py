"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.load_cna import load_cna, load_cna_long, maybe_make_wide


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_cna_long_basic(tmp_path: Path):
    p = _write(
        tmp_path / "data_cna.txt",
        "Hugo_Symbol\tS1\tS2\n"
        "TP53\t0\t-2\n"
        "BRCA1\t1\tNA\n",
    )

    long = load_cna_long(p)
    assert len(long) == 3  # NA dropped
    assert set(long.columns) == {"sample_id", "feature", "value"}
    assert long["sample_id"].nunique() == 2
    assert long["feature"].nunique() == 2


def test_maybe_make_wide_cna_small(tmp_path: Path):
    p = _write(
        tmp_path / "data_cna.txt",
        "Hugo_Symbol\tS1\tS2\n"
        "TP53\t0\t-2\n"
        "BRCA1\t1\t0\n",
    )
    long = load_cna_long(p)
    wide = maybe_make_wide(long, max_features=10)

    assert wide is not None
    assert wide.shape == (2, 2)
    assert "TP53" in wide.columns


def test_load_cna_wrapper(tmp_path: Path):
    p = _write(
        tmp_path / "data_cna.txt",
        "Hugo_Symbol\tS1\tS2\n"
        "TP53\t0\t-2\n"
        "BRCA1\t1\t0\n",
    )
    res = load_cna(p, build_wide=True, max_features_for_wide=100)
    assert res.cna_long is not None
    assert res.cna_wide is not None
    assert res.cna_wide.shape == (2, 2)
