from __future__ import annotations

import datetime
import math
import shutil
from functools import cached_property
from typing import Any, Literal

import lancedb
import lancedb.table
import pyarrow as pa
import pydantic
from loguru import logger
from sqlalchemy import column, literal

from agentic_rec.settings import settings


class LanceIndexConfig(pydantic.BaseModel):
    lancedb_path: str = settings.lance_db_path
    table_name: str = settings.items_table_name
    embedder_name: str = settings.embedder_name
    embedder_device: str = "cpu"
    reranker_name: str = settings.reranker_name
    reranker_type: str = settings.reranker_type


class LanceIndex:
    def __init__(self, config: LanceIndexConfig) -> None:
        """Initialize index with config; call open_table() or index_data() to activate."""
        super().__init__()
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

    def save(self, path: str) -> None:
        """Copy the LanceDB directory to the given path."""
        shutil.copytree(self.config.lancedb_path, path)

    @classmethod
    def load(cls, config: LanceIndexConfig) -> LanceIndex:
        """Create an index and open the existing table."""
        index = cls(config)
        index.open_table()
        return index

    def open_table(self) -> lancedb.table.Table:
        """Connect to LanceDB and open the configured table."""
        db = lancedb.connect(self.config.lancedb_path)
        self.table = db.open_table(self.config.table_name)
        logger.info(f"{self.__class__.__name__}: {self.table}")
        logger.info(
            f"num_items: {self.table.count_rows()}, columns: {self.table.schema.names}"
        )
        return self.table

    def index_data(
        self, data: pa.Table, *, overwrite: bool = False
    ) -> lancedb.table.Table:
        """Embed data and create table with scalar, FTS, and vector indices.

        All columns from the table are indexed. The ``id`` and ``text`` columns
        are required; a ``vector`` column is computed automatically via the
        configured embedder.
        """
        if self.table is not None and not overwrite:
            return self.table

        from lancedb.embeddings import EmbeddingFunctionConfig

        # Scalar index requires pa.string(), not large_string
        id_idx = data.schema.get_field_index("id")
        if data.schema.field("id").type != pa.string():
            data = data.cast(data.schema.set(id_idx, pa.field("id", pa.string())))

        db = lancedb.connect(self.config.lancedb_path)
        embedding_function = EmbeddingFunctionConfig(
            source_column="text",
            vector_column="vector",
            function=self.embedder,
        )
        self.table = db.create_table(
            self.config.table_name,
            data=data.to_batches(max_chunksize=1024),
            embedding_functions=[embedding_function],
            mode="overwrite",
        )
        self.table.create_scalar_index("id")
        self.table.create_fts_index("text")

        num_partitions = 2 ** int(math.log2(max(1, data.num_rows)) / 2)
        self.table.create_index(
            vector_column_name="vector",
            metric="cosine",
            index_type="IVF_RQ",
            num_partitions=num_partitions,
        )
        self.table.optimize(
            cleanup_older_than=datetime.timedelta(days=0),
            delete_unverified=True,
        )

        logger.info(f"{self.__class__.__name__}: {self.table}")
        logger.info(
            f"num_items: {self.table.count_rows()}, columns: {self.table.schema.names}"
        )
        return self.table

    def index_parquet(
        self,
        parquet_path: str,
        *,
        overwrite: bool = False,
    ) -> lancedb.table.Table:
        """Read a Parquet file (memory-mapped) and build the LanceDB index."""
        import pyarrow.parquet as pq

        data = pq.read_table(parquet_path, memory_map=True)
        return self.index_data(data, overwrite=overwrite)

    def search(
        self,
        text: str,
        query_type: Literal["vector", "fts", "hybrid"] = "hybrid",
        exclude_ids: list[str] | None = None,
        limit: int = 10,
    ) -> pa.Table:
        """Search with reranking, returning scored results.

        query_type can be 'hybrid', 'vector', or 'fts'.
        """
        assert self.table is not None

        query = (
            self.table.search(text, query_type=query_type)
            .rerank(self.reranker)
            .limit(limit)
        )

        if exclude_ids:
            expr = column("id").not_in([literal(v) for v in exclude_ids])
            filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
            query = query.where(filter_str, prefilter=True)

        result = query.to_arrow().drop_columns(["vector"])
        return result.rename_columns(
            ["score" if n == "_relevance_score" else n for n in result.schema.names]
        )

    def get_ids(self, ids: list[str]) -> pa.Table:
        """Look up rows by ID list using the scalar index."""
        assert self.table is not None

        if not ids:
            return self.table.head(0)

        expr = column("id").in_([literal(v) for v in ids])
        filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
        return self.table.search().where(filter_str).to_arrow().drop_columns(["vector"])


def main(
    parquet_path: str = settings.items_parquet,
    table_name: str = settings.items_table_name,
    lancedb_path: str = settings.lance_db_path,
    *,
    overwrite: bool = True,
) -> None:
    """Build the LanceDB index from parquet and run a sample search."""
    import random

    import pyarrow.parquet as pq
    import rich

    import agentic_rec.ml_1m

    agentic_rec.ml_1m.main(overwrite=False)

    config = LanceIndexConfig(lancedb_path=lancedb_path, table_name=table_name)
    index = LanceIndex(config)
    if overwrite:
        index.index_parquet(parquet_path, overwrite=overwrite)
    else:
        try:
            index.open_table()
        except ValueError:
            index.index_parquet(parquet_path)

    data = pq.read_table(parquet_path, memory_map=True)
    logger.info("data loaded: {}, rows: {}", parquet_path, data.num_rows)

    sample_id = random.choice(data["id"].to_pylist())
    item = index.get_ids([sample_id])
    rich.print(item.to_pylist()[0])

    text = item.column("text")[0].as_py()
    results = index.search(text, exclude_ids=[sample_id], limit=5)
    rich.print(results.to_pylist())


def cli() -> None:
    from jsonargparse import auto_cli

    auto_cli(main, as_positional=False)


if __name__ == "__main__":
    cli()
