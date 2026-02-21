"""
BreastCancerStudy
Copyright (c) 2026 Toktam Khatibi
All rights reserved.

No permission is granted to use, reproduce, modify, or distribute this file
without prior written consent from the author.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def seed() -> int:
    # Handy if we later add randomized tests.
    return 1337
