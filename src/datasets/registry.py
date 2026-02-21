"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class DatasetPaths:
    """
    Simple contract: a dataset is represented by a cohort parquet.
    (Later we can expand this to include raw/processed roots.)
    """
    cohort_parquet: Path


class DatasetRegistry:
    """
    Registry for dataset locations.

    Why: for external validation and PI-level positioning, we want
    dataset selection to be explicit, versioned, and reproducible.

    This registry is intentionally minimal: it only points to cohort-level
    Parquet files (patient-level modeling matrices), not raw data.
    """

    def __init__(self) -> None:
        self._items: Dict[str, DatasetPaths] = {}

    def register(self, name: str, cohort_parquet: str | Path) -> None:
        p = Path(cohort_parquet)
        self._items[name.lower()] = DatasetPaths(cohort_parquet=p)

    def get(self, name: str) -> DatasetPaths:
        key = name.lower()
        if key not in self._items:
            known = ", ".join(sorted(self._items.keys()))
            raise KeyError(f"Unknown dataset '{name}'. Known: [{known}]")
        return self._items[key]

    def maybe_get(self, name: str) -> Optional[DatasetPaths]:
        try:
            return self.get(name)
        except KeyError:
            return None
