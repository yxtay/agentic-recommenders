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

from agentic_rec.settings import settings

if TYPE_CHECKING:
    from lancedb.pydantic import LanceModel


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
        self, dataset: datasets.Dataset, *, overwrite: bool = False
    ) -> lancedb.table.Table:
        """Embed dataset and create table with scalar, FTS, and vector indices."""
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
        limit: int = 20,
    ) -> datasets.Dataset:
        """Hybrid vector + FTS search with reranking, returning scored results."""
        assert self.table is not None

        query = (
            self.table.search(text, query_type="hybrid")
            .rerank(self.reranker)
            .limit(limit)
        )

        if exclude_ids:
            expr = column("id").not_in([literal(v) for v in exclude_ids])
            filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
            query = query.where(filter_str, prefilter=True)

        return datasets.Dataset(query.to_arrow()).rename_column(
            "_relevance_score", "score"
        )  # type: ignore[arg-type]

    def get_ids(self, ids: list[str]) -> datasets.Dataset:
        """Look up rows by ID list using the scalar index."""
        assert self.table is not None

        if not ids:
            return datasets.Dataset(self.table.head(0))

        expr = column("id").in_([literal(v) for v in ids])
        filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
        return datasets.Dataset(self.table.search().where(filter_str).to_arrow())  # type: ignore[arg-type]


def main(
    parquet_path: str = settings.items_parquet,
    table_name: str = settings.items_table_name,
    lancedb_path: str = settings.lance_db_path,
    *,
    overwrite: bool = True,
) -> None:
    """Build the LanceDB index from parquet and run a sample search."""
    import random

    import rich

    import agentic_rec.data

    agentic_rec.data.main(overwrite=False)
    dataset = datasets.Dataset.from_parquet(parquet_path)
    logger.info("dataset loaded: {}, shape: {}", parquet_path, dataset.shape)

    config = LanceIndexConfig(lancedb_path=lancedb_path, table_name=table_name)
    index = LanceIndex(config)
    if overwrite:
        index.index_data(dataset, overwrite=overwrite)
    else:
        try:
            index.open_table()
        except ValueError:
            index.index_data(dataset)

    sample_id = random.choice(dataset["id"])
    item = index.get_ids([sample_id])
    rich.print(item.select_columns(["id", "text"])[0])

    text = item["text"][0]
    results = index.search(text, exclude_ids=[sample_id], limit=5)
    rich.print(results.select_columns(["id", "text", "score"]).to_list())


def cli() -> None:
    from jsonargparse import auto_cli

    auto_cli(main, as_positional=False)


if __name__ == "__main__":
    cli()
