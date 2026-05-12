from __future__ import annotations

import pyarrow as pa
import pytest

from agentic_rec.index import LanceIndex, LanceIndexConfig


@pytest.fixture
def index_config(tmp_path) -> LanceIndexConfig:
    return LanceIndexConfig(lancedb_path=str(tmp_path / "test_db"), table_name="test_table")


class TestLanceIndex:
    def test_init(self, index_config: LanceIndexConfig) -> None:
        index = LanceIndex(index_config)
        assert index.config == index_config
        assert index.table is None

    def test_index_data(self, index_config: LanceIndexConfig) -> None:
        index = LanceIndex(index_config)
        data = pa.table({"id": ["1", "2"], "text": ["foo", "bar"]})
        index.index_data(data)
        assert index.table is not None
        assert index.table.count_rows() == 2

    def test_get_ids(self, index_config: LanceIndexConfig) -> None:
        index = LanceIndex(index_config)
        data = pa.table({"id": ["1", "2"], "text": ["foo", "bar"]})
        index.index_data(data)

        result = index.get_ids(["1"])
        assert result.num_rows == 1
        assert result.to_pylist()[0]["text"] == "foo"


class TestSearch:
    def test_returns_table(self, index_config: LanceIndexConfig) -> None:
        index = LanceIndex(index_config)
        data = pa.table({"id": ["1", "2"], "text": ["foo", "bar"]})
        index.index_data(data)

        results = index.search("foo", limit=1)
        assert isinstance(results, pa.Table)
        assert results.num_rows == 1
        assert "score" in results.column_names
