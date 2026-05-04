from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pydantic

from agentic_rec.params import (
    EMBEDDER_NAME,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    RERANKER_NAME,
)

if TYPE_CHECKING:
    import datasets
    import lancedb
    import lancedb.table


class LanceIndexConfig(pydantic.BaseModel):
    lancedb_path: str = LANCE_DB_PATH
    table_name: str = ITEMS_TABLE_NAME
    embedder_model_name: str = EMBEDDER_NAME
    embedder_device: str = "cpu"
    reranker_model_name: str = RERANKER_NAME
    reranker_model_type: str = "cross-encoder"
    nprobes: int = 8
    refine_factor: int = 4


class LanceIndex:
    def __init__(self, config: LanceIndexConfig) -> None:
        super().__init__()
        self.config = config
        self.table: lancedb.table.Table | None = None
        self._embedder: Any | None = None
        self._schema: type | None = None
        self._reranker: Any | None = None

    @property
    def embedder(self) -> Any:  # noqa: ANN401
        if self._embedder is None:
            from lancedb.embeddings import get_registry

            self._embedder = (
                get_registry()
                .get("sentence-transformers")
                .create(
                    name=self.config.embedder_model_name,
                    device=self.config.embedder_device,
                )
            )
        return self._embedder

    @property
    def schema(self) -> type:
        if self._schema is None:
            self._schema = self._build_schema()
        return self._schema

    @property
    def reranker(self) -> Any:  # noqa: ANN401
        if self._reranker is None:
            from rerankers import Reranker

            self._reranker = Reranker(
                self.config.reranker_model_name,
                model_type=self.config.reranker_model_type,
            )
        return self._reranker

    def _build_schema(self) -> type:
        from lancedb.pydantic import LanceModel, Vector

        embedder = self.embedder

        class ItemSchema(LanceModel):
            id: str
            text: str = embedder.SourceField()
            vector: Vector(embedder.ndims()) = embedder.VectorField()

        return ItemSchema

    def save(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, config: LanceIndexConfig) -> LanceIndex:
        return cls(config)

    def _open_table(self) -> lancedb.table.Table:
        pass

    def index_data(
        self, dataset: datasets.Dataset, *, overwrite: bool = False
    ) -> lancedb.table.Table:
        pass

    def search(
        self,
        text: str,
        exclude_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> datasets.Dataset:
        pass

    def get_ids(self, ids: list[str]) -> datasets.Dataset:
        pass
