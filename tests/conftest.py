from __future__ import annotations

import pytest


@pytest.fixture
def seed() -> int:
    # Handy if we later add randomized tests.
    return 1337
