from __future__ import annotations

import pytest

from object_store import ObjectStore


@pytest.fixture
def store(tmp_path):
    return ObjectStore(uri=str(tmp_path / "db"), table_name="objects")


@pytest.fixture
def store_factory(tmp_path):
    """Factory that yields stores pointing at the same uri (for reopen tests)."""
    counter = {"n": 0}
    uri = str(tmp_path / "shared")

    def make(table_name: str = "objects", **kw) -> ObjectStore:
        counter["n"] += 1
        return ObjectStore(uri=uri, table_name=table_name, **kw)

    return make
