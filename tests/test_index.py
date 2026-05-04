from __future__ import annotations

import datasets
import numpy as np
import pytest

from agentic_rec.index import LanceIndex, LanceIndexConfig
from agentic_rec.params import (
    EMBEDDER_NAME,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    RERANKER_NAME,
)

_DEFAULT_NPROBES = 8
_DEFAULT_REFINE_FACTOR = 4
_CUSTOM_NPROBES = 16


class TestLanceIndexConfig:
    def test_defaults(self) -> None:
        config = LanceIndexConfig()
        assert config.lancedb_path == LANCE_DB_PATH
        assert config.table_name == ITEMS_TABLE_NAME
        assert config.embedder_model_name == EMBEDDER_NAME
        assert config.embedder_device == "cpu"
        assert config.reranker_model_name == RERANKER_NAME
        assert config.reranker_model_type == "cross-encoder"
        assert config.nprobes == _DEFAULT_NPROBES
        assert config.refine_factor == _DEFAULT_REFINE_FACTOR

    def test_custom_values(self, tmp_path: str) -> None:
        config = LanceIndexConfig(lancedb_path=str(tmp_path), nprobes=_CUSTOM_NPROBES)
        assert config.lancedb_path == str(tmp_path)
        assert config.nprobes == _CUSTOM_NPROBES


# ── shared integration fixtures ────────────────────────────────────────────

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


@pytest.fixture(scope="session")
def test_dataset() -> datasets.Dataset:
    rng = np.random.default_rng(42)
    items = [
        {
            "id": str(i),
            "text": (
                f"Movie {i}: "
                f"{'Comedy' if i % 2 == 0 else 'Drama'} about "
                f"{'love' if i % 3 == 0 else 'adventure'}"
            ),
            "vector": rng.random(EMBEDDING_DIM).astype(np.float32).tolist(),
        }
        for i in range(100)
    ]
    return datasets.Dataset.from_list(items)


@pytest.fixture(scope="session")
def lance_config(tmp_path_factory: pytest.TempPathFactory) -> LanceIndexConfig:
    return LanceIndexConfig(
        lancedb_path=str(tmp_path_factory.mktemp("lance_db")),
    )


@pytest.fixture(scope="session")
def indexed_index(
    lance_config: LanceIndexConfig, test_dataset: datasets.Dataset
) -> LanceIndex:
    index = LanceIndex(lance_config)
    index.index_data(test_dataset)
    return index


# ── schema tests ───────────────────────────────────────────────────────────


class TestBuildSchema:
    def test_schema_fields(self, lance_config: LanceIndexConfig) -> None:
        from lancedb.pydantic import LanceModel

        index = LanceIndex(lance_config)
        schema = index.schema
        assert issubclass(schema, LanceModel)
        assert "id" in schema.field_names()
        assert "text" in schema.field_names()
        assert "vector" in schema.field_names()

    def test_schema_cached(self, lance_config: LanceIndexConfig) -> None:
        index = LanceIndex(lance_config)
        assert index.schema is index.schema


class TestIndexData:
    def test_table_created(
        self, indexed_index: LanceIndex, test_dataset: datasets.Dataset
    ) -> None:
        assert indexed_index.table is not None
        assert indexed_index.table.count_rows() == len(test_dataset)

    def test_table_columns(self, indexed_index: LanceIndex) -> None:
        cols = indexed_index.table.schema.names
        assert "id" in cols
        assert "text" in cols
        assert "vector" in cols

    def test_overwrite_false_skips(
        self, indexed_index: LanceIndex, test_dataset: datasets.Dataset
    ) -> None:
        original_table = indexed_index.table
        indexed_index.index_data(test_dataset, overwrite=False)
        assert indexed_index.table is original_table
