from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import column, literal

from agentic_rec.models import UserResponse
from agentic_rec.repositories.base import BaseLanceRepository

if TYPE_CHECKING:
    import lancedb.table


class UserRepository(BaseLanceRepository):
    def get_by_id(self, user_id: str) -> UserResponse | None:
        """Look up a user by ID."""
        assert self.table is not None
        expr = column("id") == literal(user_id)
        filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
        result = self.table.search().where(filter_str).to_arrow().drop_columns(["vector"])
        if result.num_rows == 0:
            return None
        return UserResponse.model_validate(result.to_pylist()[0])

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
        # Optional: create FTS for users if needed
        # self.table.create_fts_index("text")

        self.optimize()
        logger.info(f"Indexed users from {parquet_path}")
        return self.table
