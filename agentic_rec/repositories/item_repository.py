from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pyarrow as pa
from loguru import logger
from sqlalchemy import column, literal

from agentic_rec.models import ItemResponse
from agentic_rec.repositories.base import BaseLanceRepository

if TYPE_CHECKING:
    import lancedb.table


class ItemRepository(BaseLanceRepository):
    def get_by_id(self, item_id: str) -> ItemResponse | None:
        """Look up an item by ID."""
        assert self.table is not None
        expr = column("id") == literal(item_id)
        filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
        result = self.table.search().where(filter_str).to_arrow().drop_columns(["vector"])
        if result.num_rows == 0:
            return None
        return ItemResponse.model_validate(result.to_pylist()[0])

    def get_by_ids(self, item_ids: list[str]) -> pa.Table:
        """Look up multiple items by ID list."""
        assert self.table is not None
        if not item_ids:
            return self.table.head(0)

        expr = column("id").in_([literal(v) for v in item_ids])
        filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
        return self.table.search().where(filter_str).to_arrow().drop_columns(["vector"])

    def search(
        self,
        query_text: str,
        exclude_ids: list[str] | None = None,
        limit: int = 20,
    ) -> pa.Table:
        """Hybrid vector + FTS search with reranking."""
        assert self.table is not None

        query = (
            self.table.search(query_text, query_type="hybrid")
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

    def index_parquet(self, parquet_path: str, *, overwrite: bool = False) -> lancedb.table.Table:
        import lancedb
        import pyarrow.parquet as pq
        from lancedb.embeddings import EmbeddingFunctionConfig

        if self.table is not None and not overwrite:
            return self.table

        data = pq.read_table(parquet_path, memory_map=True)

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
        self.optimize()
        logger.info(f"Indexed items from {parquet_path}")
        return self.table
