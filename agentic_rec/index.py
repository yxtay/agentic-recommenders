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
    ITEMS_PARQUET,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
    RERANKER_NAME,
    RERANKER_TYPE,
)

if TYPE_CHECKING:
    from lancedb.pydantic import LanceModel


class LanceIndexConfig(pydantic.BaseModel):
    lancedb_path: str = LANCE_DB_PATH
    table_name: str = ITEMS_TABLE_NAME
    embedder_name: str = EMBEDDER_NAME
    embedder_device: str = "cpu"
    reranker_name: str = RERANKER_NAME
    reranker_type: str = RERANKER_TYPE


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

        # Scalar index requires pa.string(), not large_string
        arrow_data = dataset.select_columns(["id", "text"]).data
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

        num_partitions = 2 ** int(math.log2(len(dataset)) / 2)
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

        return datasets.Dataset(query.to_arrow()).rename_column(
            "_relevance_score", "score"
        )  # type: ignore[arg-type]

    def get_ids(self, ids: list[str]) -> datasets.Dataset:
        assert self.table is not None

        if not ids:
            return datasets.Dataset(self.table.head(0))

        expr = column("id").in_([literal(v) for v in ids])
        filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
        return datasets.Dataset(self.table.search().where(filter_str).to_arrow())  # type: ignore[arg-type]


def main(
    parquet_path: str = ITEMS_PARQUET,
    table_name: str = ITEMS_TABLE_NAME,
    lancedb_path: str = LANCE_DB_PATH,
    *,
    overwrite: bool = True,
) -> None:
    import rich

    import agentic_rec.data

    agentic_rec.data.main(overwrite=False)
    dataset = datasets.Dataset.from_parquet(parquet_path)
    logger.info("dataset loaded: {}, shape: {}", parquet_path, dataset.shape)

    config = LanceIndexConfig(lancedb_path=lancedb_path, table_name=table_name)
    index = LanceIndex(config)
    index.index_data(dataset, overwrite=overwrite)

    sample_id = dataset.shuffle()[0]["id"]
    item = index.get_ids([sample_id])
    rich.print(item.select_columns(["id", "text"])[0])

    text = item["text"][0]
    results = index.search(text, exclude_ids=[sample_id], top_k=5)
    rich.print(results.select_columns(["id", "text", "score"]).to_list())


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main, as_positional=False)
