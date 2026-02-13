# tests/conftest.py
from __future__ import annotations

import pytest


@pytest.fixture
def seed() -> int:
    # Handy if you later add randomized tests.
    return 1337
