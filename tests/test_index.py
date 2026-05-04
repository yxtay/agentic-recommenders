from __future__ import annotations

from agentic_rec.index import LanceIndexConfig
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
