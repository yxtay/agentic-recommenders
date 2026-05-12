from __future__ import annotations

import datetime
from functools import cached_property
from typing import Any

import lancedb
import lancedb.table
import pydantic
from loguru import logger

from agentic_rec.settings import settings


class LanceIndexConfig(pydantic.BaseModel):
    lancedb_path: str = settings.lance_db_path
    table_name: str = settings.items_table_name
    embedder_name: str = settings.embedder_name
    embedder_device: str = "cpu"
    reranker_name: str = settings.reranker_name
    reranker_type: str = settings.reranker_type


class BaseLanceRepository:
    def __init__(self, config: LanceIndexConfig) -> None:
        self.config = config
        self.table: lancedb.table.Table | None = None

    @cached_property
    def embedder(self) -> Any:  # noqa: ANN401
        from lancedb.embeddings import get_registry

        return (
            get_registry()
            .get("sentence-transformers")
            .create(
                name=self.config.embedder_name,
                device=self.config.embedder_device,
            )
        )

    @cached_property
    def reranker(self) -> Any:  # noqa: ANN401
        from lancedb.rerankers import AnswerdotaiRerankers

        return AnswerdotaiRerankers(
            model_name=self.config.reranker_name,
            model_type=self.config.reranker_type,
        )

    def open_table(self) -> lancedb.table.Table:
        """Connect to LanceDB and open the configured table."""
        db = lancedb.connect(self.config.lancedb_path)
        self.table = db.open_table(self.config.table_name)
        logger.info(f"{self.__class__.__name__}: {self.table}")
        return self.table

    def optimize(self) -> None:
        if self.table is not None:
            self.table.optimize(
                cleanup_older_than=datetime.timedelta(days=0),
                delete_unverified=True,
            )
