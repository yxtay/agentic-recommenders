from __future__ import annotations

import datetime
import math
import shutil
from functools import cached_property
from typing import TYPE_CHECKING, Any

import datasets
import lancedb
import lancedb.table
import pydantic
from loguru import logger
from sqlalchemy import column, literal

from agentic_rec.params import (
    EMBEDDER_NAME,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    RERANKER_NAME,
)

if TYPE_CHECKING:
    from lancedb.pydantic import LanceModel


class LanceIndexConfig(pydantic.BaseModel):
    lancedb_path: str = LANCE_DB_PATH
    table_name: str = ITEMS_TABLE_NAME
    embedder_name: str = EMBEDDER_NAME
    embedder_device: str = "cpu"
    reranker_name: str = RERANKER_NAME
    reranker_type: str = "cross-encoder"


class LanceIndex:
    def __init__(self, config: LanceIndexConfig) -> None:
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
    def schema(self) -> type[LanceModel]:
        from lancedb.pydantic import LanceModel, Vector

        class ItemSchema(LanceModel):
            id: str
            text: str = self.embedder.SourceField()
            vector: Vector(self.embedder.ndims()) = self.embedder.VectorField()  # type: ignore[valid-type]

        return ItemSchema

    @cached_property
    def reranker(self) -> Any:  # noqa: ANN401
        from lancedb.rerankers import AnswerdotaiRerankers

        return AnswerdotaiRerankers(
            model_name=self.config.reranker_name,
            model_type=self.config.reranker_type,
        )

    def save(self, path: str) -> None:
        shutil.copytree(self.config.lancedb_path, path)

    @classmethod
    def load(cls, config: LanceIndexConfig) -> LanceIndex:
        index = cls(config)
        index._open_table()
        return index

    def _open_table(self) -> lancedb.table.Table:
        db = lancedb.connect(self.config.lancedb_path)
        self.table = db.open_table(self.config.table_name)
        logger.info(f"{self.__class__.__name__}: {self.table}")
        logger.info(
            f"num_items: {self.table.count_rows()}, columns: {self.table.schema.names}"
        )
        return self.table

    def index_data(
        self, dataset: datasets.Dataset, *, overwrite: bool = False
    ) -> lancedb.table.Table:
        if self.table is not None and not overwrite:
            return self.table

        import pyarrow as pa

        num_items = len(dataset)
        embedding_dim = self.embedder.ndims()

        # Scalar index requires pa.string(), not large_string
        arrow_data = dataset.data
        id_idx = arrow_data.schema.get_field_index("id")
        if arrow_data.schema.field("id").type != pa.string():
            arrow_data = arrow_data.cast(
                arrow_data.schema.set(id_idx, pa.field("id", pa.string()))
            )

        db = lancedb.connect(self.config.lancedb_path)
        self.table = db.create_table(
            self.config.table_name,
            data=arrow_data.to_batches(max_chunksize=1024),
            schema=self.schema,  # type: ignore[arg-type]
            mode="overwrite",
        )

        self.table.create_scalar_index("id")
        self.table.create_fts_index("text")

        # IVF_HNSW_PQ: rule-of-thumb num_partitions ~= 4 * sqrt(n)
        num_partitions = 2 ** int(math.log2(num_items) / 2)
        num_sub_vectors = embedding_dim // 8
        # PQ codebook: num_bits=8 requires >=256 training rows, num_bits=4 requires >=16
        num_bits = 8 if num_items >= 256 else 4  # noqa: PLR2004
        self.table.create_index(
            vector_column_name="vector",
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            num_bits=num_bits,
            index_type="IVF_HNSW_PQ",
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

    def search(
        self,
        text: str,
        exclude_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> datasets.Dataset:
        assert self.table is not None

        query = (
            self.table.search(text, query_type="hybrid")
            .rerank(self.reranker)
            .limit(top_k)
        )

        if exclude_ids:
            expr = column("id").not_in([literal(v) for v in exclude_ids])
            filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
            query = query.where(filter_str, prefilter=True)

        return datasets.Dataset(query.to_arrow())  # type: ignore[arg-type]

    def get_ids(self, ids: list[str]) -> datasets.Dataset:
        assert self.table is not None

        if not ids:
            return datasets.Dataset(self.table.head(0))

        expr = column("id").in_([literal(v) for v in ids])
        filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
        return datasets.Dataset(self.table.search().where(filter_str).to_arrow())  # type: ignore[arg-type]
