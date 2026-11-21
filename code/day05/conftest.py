from __future__ import annotations

import pytest

from common.infra import postgres_available


@pytest.fixture(scope="session")
def pg_ok() -> bool:
    return postgres_available()
