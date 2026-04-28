from __future__ import annotations

import datetime
import math
import shutil
from typing import TYPE_CHECKING, Any, Literal

import datasets
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
from loguru import logger

from agentic_rec.params import (
    CROSS_ENCODER_MODEL_NAME,
    EMBEDDER_MODEL_NAME,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
)

if TYPE_CHECKING:
    import lancedb
    import numpy as np


class LanceIndexConfig(pydantic.BaseModel):
    """Configuration for LanceDB index, including paths and text column.

    Attributes:
        lancedb_path (str): Path to the LanceDB store.
        table_name (str): Name of the table within the LanceDB store.
        text_col (str): Name of the column containing text for full-text search.
    """

    lancedb_path: str = LANCE_DB_PATH
    table_name: str = ITEMS_TABLE_NAME
    embedder_model_name: str = EMBEDDER_MODEL_NAME
    cross_encoder_model_name: str = CROSS_ENCODER_MODEL_NAME

    id_col: str = "item_id"
    text_col: str = "item_text"
    index_metric: Literal["l2", "dot", "cosine"] = "cosine"
    embedding_col: str = "vector"


class LanceIndex:
    """
    Index implementation using LanceDB for fast vector and text search.
    """

    def __init__(
        self, config: LanceIndexConfig, item_data_model: type[pydantic.BaseModel]
    ) -> None:
        """Initialize LanceIndex with configuration and optional table.

        Args:
            config (LanceIndexConfig): Configuration specifying paths and
                column names used by the index.
        """
        super().__init__()
        self.config = config
        self.table: lancedb.table.Table | None = None
        self.item_data_model = item_data_model

    def save(self, path: str) -> None:
        """Copy the underlying LanceDB store to a new path.

        This is a convenience that copies the directory backing the
        LanceDB instance to ``path`` using ``shutil.copytree``. No
        validation is performed on the target location.

        Args:
            path (str): Destination path where the LanceDB store will be
                copied.

        Returns:
            None: The function copies files on disk and does not return a value.
        """
        shutil.copytree(self.config.lancedb_path, path)

    @classmethod
    def load(cls, config: LanceIndexConfig) -> LanceIndex:
        """Load a LanceIndex from disk and infer column names from indices.

        This classmethod opens the LanceDB table configured in ``config``
        and inspects any indices created on the table to populate the
        configuration fields ``embedding_col``, ``text_col`` and
        ``id_col`` when possible.

        Args:
            config (LanceIndexConfig): Configuration pointing to the
                LanceDB store to load.

        Returns:
            LanceIndex: Configured LanceIndex with an opened table.
        """
        self = cls(config, item_data_model=item_data_model)
        self.open_table()

        assert self.table is not None
        for index in self.table.list_indices():
            match index.index_type:
                case "FTS":
                    self.config.text_col = index.columns[0]
                case "BTree":
                    self.config.id_col = index.columns[0]
                case _:
                    pass
        return self

    def open_table(self) -> lancedb.table.Table:
        """Open and return the LanceDB table specified by the config.

        This method connects to the LanceDB store at ``config.lancedb_path``
        and opens the table ``config.table_name``. The opened table is
        stored on the instance as ``self.table``.

        Returns:
            lancedb.table.Table: Opened LanceDB table object.
        """
        import lancedb

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
        """Create and index data in LanceDB from a HuggingFace Dataset.

        The provided dataset is used to create a LanceDB table; scalar
        and full-text-search (FTS) indices are created for the configured
        ID and text columns. If an embedding column is configured a
        vector index (IVF_HNSW_PQ) will be created using heuristics for
        partitioning and sub-vector size.

        Args:
            dataset (datasets.Dataset): HuggingFace Dataset containing
                the data to index.
            overwrite (bool): If ``True`` an existing table will be
                replaced. Defaults to ``False``.

        Returns:
            lancedb.table.Table: The created or existing LanceDB table.
        """
        if self.table is not None and not overwrite:
            return self.table

        import lancedb

        schema = dataset.data.schema
        schema = schema.set(
            # scalar index does not work on large_string
            schema.get_field_index(self.config.id_col),
            pa.field(self.config.id_col, pa.string()),
        )

        if self.config.embedding_col:
            embedding_dim = len(dataset[self.config.embedding_col][0])
            schema = schema.set(
                # embedding column must be fixed size float array
                schema.get_field_index(self.config.embedding_col),
                pa.field(
                    self.config.embedding_col, pa.list_(pa.float32(), embedding_dim)
                ),
            )

        db = lancedb.connect(self.config.lancedb_path)
        self.table = db.create_table(
            self.config.table_name,
            data=dataset.data.to_batches(max_chunksize=1024),
            schema=schema,
            mode="overwrite",
        )
        self.table.create_scalar_index(self.config.id_col)
        self.table.create_fts_index(self.config.text_col)

        if self.config.embedding_col:
            num_items = len(dataset)
            embedding_dim = len(dataset[self.config.embedding_col][0])
            # rule of thumb: nlist ~= 4 * sqrt(n_vectors)
            num_partitions = 2 ** int(math.log2(num_items) / 2)
            num_sub_vectors = embedding_dim // 8

            self.table.create_index(
                vector_column_name=self.config.embedding_col,
                metric=self.config.index_metric,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
                index_type="IVF_HNSW_PQ",
            )

        self.table.optimize(
            cleanup_older_than=datetime.timedelta(days=0),
            delete_unverified=True,
            retrain=True,
        )

        logger.info(f"{self.__class__.__name__}: {self.table}")
        logger.info(
            f"num_items: {self.table.count_rows()}, columns: {self.table.schema.names}"
        )
        return self.table

    def search(
        self,
        embedding: np.typing.NDArray[np.float32],
        exclude_item_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> datasets.Dataset:
        """Search the LanceDB vector index for the nearest items.

        The method performs a vector search with optional prefiltering to
        exclude specific item IDs. Returned results are converted into a
        HuggingFace ``datasets.Dataset`` and a cosine-like ``score`` is
        appended (computed as 1 - distance).

        Args:
            embedding (np.typing.NDArray[np.float32]): Query vector.
            exclude_item_ids (list[str] | None): Optional list of item
                IDs to exclude from results.
            top_k (int): Number of top results to return.

        Returns:
            datasets.Dataset: Dataset containing the search results with an
            additional ``score`` column. The ``score`` is computed as
            ``1 - _distance`` to resemble cosine similarity.
        """
        assert self.table is not None
        exclude_item_ids = exclude_item_ids or [""]
        exclude_filter = ", ".join(
            f"'{str(item).replace("'", "''")}'" for item in exclude_item_ids
        )
        exclude_filter = f"{self.config.id_col} NOT IN ({exclude_filter})"
        rec_table = (
            self.table.search(embedding)
            .where(exclude_filter, prefilter=True)
            .nprobes(8)
            .refine_factor(4)
            .limit(top_k)
            .to_arrow()
        )
        rec_table = rec_table.append_column(
            "score", pc.subtract(1, rec_table["_distance"])
        )
        return datasets.Dataset(rec_table)

    def get_ids(self, ids: list[str]) -> datasets.Dataset:
        """Fetch rows from the LanceDB table matching the provided IDs.

        Args:
            ids (list[str]): List of item identifiers to fetch.

        Returns:
            datasets.Dataset: Dataset containing the matching rows.
        """
        assert self.table is not None
        ids_filter = ", ".join(f"'{str(id_val).replace("'", "''")}'" for id_val in ids)
        result = (
            self.table.search()
            .where(f"{self.config.id_col} IN ({ids_filter})")
            .to_arrow()
        )
        return datasets.Dataset(result)

    def get_id(self, id_val: str | None) -> dict[str, Any]:
        """Retrieve a single item from LanceDB by its ID.

        Args:
            id_val (str | None): Item ID to fetch. If ``None`` an empty
                dictionary is returned.

        Returns:
            dict[str, Any]: The first matching row as a dictionary or an
            empty dictionary if no match is found.
        """
        if id_val is None:
            return {}

        result = self.get_ids([id_val])
        if len(result) == 0:
            return {}
        return result[0]
