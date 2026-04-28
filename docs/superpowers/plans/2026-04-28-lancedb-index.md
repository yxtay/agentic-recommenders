# LanceDB Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `LanceIndex` to use `LanceModel` for automatic embedding, hybrid search with answerdotai reranking, and sqlalchemy-built WHERE clauses for SQL injection safety.

**Architecture:** `LanceIndexConfig` holds tunable parameters (paths, model names, nprobes, refine_factor). `LanceIndex` wraps a LanceDB table; it lazily loads the embedder, schema, and reranker on first use. `index_data` creates the table and all three index types; `search` runs hybrid (vector + FTS) search and reranks results; `get_ids` does scalar-indexed point lookups.

**Tech Stack:** lancedb 0.30, sentence-transformers (via lancedb registry), answerdotai `rerankers`, sqlalchemy (filter string generation), HuggingFace `datasets`, pydantic.

---

## File Map

| File | Change |
|---|---|
| `pyproject.toml` | Add `rerankers`, `sqlalchemy` to `[project.dependencies]` |
| `agentic_rec/index.py` | Full rewrite — new config fields, LanceModel schema, hybrid search, reranker |
| `tests/__init__.py` | Create (empty) |
| `tests/test_index.py` | Create — unit + integration tests |

---

## Task 1: Add Dependencies

**Files:**

- Modify: `pyproject.toml`

- [ ] **Step 1: Add `rerankers` and `sqlalchemy` to `pyproject.toml`**

In the `[project.dependencies]` list, add after the existing entries:

```toml
  "rerankers[transformers]>=0.9",
  "sqlalchemy>=2.0",
```

The full dependencies block should look like:

```toml
dependencies = [
  "bentoml~=1.3",
  "datasets~=4.0",
  "lancedb>=0.19,<1.0",
  "loguru>=0.7,<1.0",
  "mlflow-skinny~=3.1",
  "polars[pandas]~=1.40",
  "pydantic-ai~=1.43",
  "pylance~=4.0",
  "rerankers[transformers]>=0.9",
  "sentence-transformers[onnx,train]~=5.4",
  "sqlalchemy>=2.0",
  "tensorboard~=2.18",
  "torch~=2.5",
  "pydantic-settings~=2.0",
]
```

- [ ] **Step 2: Sync dependencies**

```bash
uv sync
```

Expected: resolves and installs `rerankers` and any new transitive deps.

- [ ] **Step 3: Verify imports**

```bash
uv run python -c "from rerankers import Reranker; from sqlalchemy import column, literal; print('ok')"
```

Expected output: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(uv): Add rerankers and sqlalchemy dependencies"
```

---

## Task 2: Rewrite `LanceIndexConfig`

**Files:**

- Modify: `agentic_rec/index.py`
- Create: `tests/__init__.py`
- Create: `tests/test_index.py`

- [ ] **Step 1: Create test scaffold and write failing config tests**

Create `tests/__init__.py` (empty):

```python
```

Create `tests/test_index.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
import datasets

from agentic_rec.index import LanceIndex, LanceIndexConfig
from agentic_rec.params import (
    CROSS_ENCODER_MODEL_NAME,
    EMBEDDER_MODEL_NAME,
    ITEMS_TABLE_NAME,
    LANCE_DB_PATH,
)


class TestLanceIndexConfig:
    def test_defaults(self):
        config = LanceIndexConfig()
        assert config.lancedb_path == LANCE_DB_PATH
        assert config.table_name == ITEMS_TABLE_NAME
        assert config.embedder_model_name == EMBEDDER_MODEL_NAME
        assert config.embedder_device == "cpu"
        assert config.reranker_model_name == CROSS_ENCODER_MODEL_NAME
        assert config.reranker_model_type == "cross-encoder"
        assert config.nprobes == 8
        assert config.refine_factor == 4

    def test_custom_values(self):
        config = LanceIndexConfig(lancedb_path="/tmp/test", nprobes=16)
        assert config.lancedb_path == "/tmp/test"
        assert config.nprobes == 16
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_index.py::TestLanceIndexConfig -v
```

Expected: FAIL — `LanceIndexConfig` still has old fields (`cross_encoder_model_name`, `id_col`, etc.)

- [ ] **Step 3: Rewrite `LanceIndexConfig` in `agentic_rec/index.py`**

Replace the entire file content with:

```python
from __future__ import annotations

import datetime
import math
import shutil
from typing import TYPE_CHECKING, Any

import datasets
import pyarrow as pa
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
    import lancedb.table


class LanceIndexConfig(pydantic.BaseModel):
    lancedb_path: str = LANCE_DB_PATH
    table_name: str = ITEMS_TABLE_NAME
    embedder_model_name: str = EMBEDDER_MODEL_NAME
    embedder_device: str = "cpu"
    reranker_model_name: str = CROSS_ENCODER_MODEL_NAME
    reranker_model_type: str = "cross-encoder"
    nprobes: int = 8
    refine_factor: int = 4


class LanceIndex:
    pass
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_index.py::TestLanceIndexConfig -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agentic_rec/index.py tests/__init__.py tests/test_index.py
git commit -m "feat(index): rewrite LanceIndexConfig with embedder and reranker fields"
```

---

## Task 3: Implement Lazy Properties and `_build_schema()`

**Files:**

- Modify: `agentic_rec/index.py`
- Modify: `tests/test_index.py`

- [ ] **Step 1: Add failing schema tests to `tests/test_index.py`**

Append to the file:

```python

# ── shared integration fixtures ────────────────────────────────────────────

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


@pytest.fixture(scope="session")
def test_dataset():
    rng = np.random.default_rng(42)
    items = [
        {
            "id": str(i),
            "text": (
                f"Movie {i}: "
                f"{'Comedy' if i % 2 == 0 else 'Drama'} about "
                f"{'love' if i % 3 == 0 else 'adventure'}"
            ),
            "vector": rng.random(EMBEDDING_DIM).astype(np.float32).tolist(),
        }
        for i in range(100)
    ]
    return datasets.Dataset.from_list(items)


@pytest.fixture(scope="session")
def lance_config(tmp_path_factory):
    return LanceIndexConfig(
        lancedb_path=str(tmp_path_factory.mktemp("lance_db")),
    )


@pytest.fixture(scope="session")
def indexed_index(lance_config, test_dataset):
    index = LanceIndex(lance_config)
    index.index_data(test_dataset)
    return index


# ── schema tests ───────────────────────────────────────────────────────────

class TestBuildSchema:
    def test_schema_fields(self, lance_config):
        from lancedb.pydantic import LanceModel
        index = LanceIndex(lance_config)
        schema = index.schema
        assert issubclass(schema, LanceModel)
        assert "id" in schema.field_names()
        assert "text" in schema.field_names()
        assert "vector" in schema.field_names()

    def test_schema_cached(self, lance_config):
        index = LanceIndex(lance_config)
        assert index.schema is index.schema
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_index.py::TestBuildSchema -v
```

Expected: FAIL — `LanceIndex` is a stub with only `pass`

- [ ] **Step 3: Implement `LanceIndex.__init__`, lazy properties, and `_build_schema()`**

Replace `class LanceIndex:` stub with:

```python
class LanceIndex:
    def __init__(self, config: LanceIndexConfig) -> None:
        super().__init__()
        self.config = config
        self.table: lancedb.table.Table | None = None
        self._embedder: Any | None = None
        self._schema: type | None = None
        self._reranker: Any | None = None

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from lancedb.embeddings import get_registry

            self._embedder = get_registry().get("sentence-transformers").create(
                name=self.config.embedder_model_name,
                device=self.config.embedder_device,
            )
        return self._embedder

    @property
    def schema(self) -> type:
        if self._schema is None:
            self._schema = self._build_schema()
        return self._schema

    @property
    def reranker(self) -> Any:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_index.py::TestBuildSchema -v
```

Expected: PASS (downloads `all-MiniLM-L6-v2` ~90 MB on first run)

- [ ] **Step 5: Commit**

```bash
git add agentic_rec/index.py tests/test_index.py
git commit -m "feat(index): add lazy embedder/schema/reranker properties and _build_schema"
```

---

## Task 4: Implement `index_data()`

**Files:**

- Modify: `agentic_rec/index.py`
- Modify: `tests/test_index.py`

- [ ] **Step 1: Add failing `index_data` tests**

Append to `tests/test_index.py`:

```python

class TestIndexData:
    def test_table_created(self, indexed_index, test_dataset):
        assert indexed_index.table is not None
        assert indexed_index.table.count_rows() == len(test_dataset)

    def test_table_columns(self, indexed_index):
        cols = indexed_index.table.schema.names
        assert "id" in cols
        assert "text" in cols
        assert "vector" in cols

    def test_overwrite_false_skips(self, indexed_index, test_dataset):
        original_table = indexed_index.table
        indexed_index.index_data(test_dataset, overwrite=False)
        assert indexed_index.table is original_table
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_index.py::TestIndexData -v
```

Expected: FAIL — `index_data` returns `None`

- [ ] **Step 3: Implement `index_data()` in `agentic_rec/index.py`**

Replace the `index_data` stub:

```python
    def index_data(
        self, dataset: datasets.Dataset, *, overwrite: bool = False
    ) -> lancedb.table.Table:
        if self.table is not None and not overwrite:
            return self.table

        import lancedb

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
            schema=self.schema,
            mode="overwrite",
        )

        self.table.create_scalar_index("id")
        self.table.create_fts_index("text")

        # IVF_HNSW_PQ: rule-of-thumb num_partitions ~= 4 * sqrt(n)
        num_partitions = 2 ** int(math.log2(num_items) / 2)
        num_sub_vectors = embedding_dim // 8
        self.table.create_index(
            vector_column_name="vector",
            metric="cosine",
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_index.py::TestIndexData -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agentic_rec/index.py tests/test_index.py
git commit -m "feat(index): implement index_data with scalar, FTS, and IVF_HNSW_PQ indices"
```

---

## Task 5: Implement `search()`

**Files:**

- Modify: `agentic_rec/index.py`
- Modify: `tests/test_index.py`

- [ ] **Step 1: Add unit test for sqlalchemy filter and failing `search` tests**

Append to `tests/test_index.py`:

```python

class TestSQLFilters:
    def test_exclude_filter_escapes_injection(self):
        from sqlalchemy import column, literal

        ids = ["id1", "id2'--injection"]
        expr = column("id").not_in([literal(v) for v in ids])
        filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
        assert "id NOT IN" in filter_str
        assert "'id1'" in filter_str
        # single-quote in value is doubled (SQL escaping), not left raw
        assert "id2''--injection" in filter_str

    def test_include_filter(self):
        from sqlalchemy import column, literal

        ids = ["a", "b"]
        expr = column("id").in_([literal(v) for v in ids])
        filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
        assert "id IN" in filter_str
        assert "'a'" in filter_str
        assert "'b'" in filter_str


class TestSearch:
    def test_returns_dataset(self, indexed_index):
        result = indexed_index.search("comedy film about love", top_k=5)
        assert isinstance(result, datasets.Dataset)

    def test_top_k_respected(self, indexed_index):
        result = indexed_index.search("drama about adventure", top_k=5)
        assert len(result) <= 5

    def test_exclude_ids(self, indexed_index):
        result = indexed_index.search("comedy", exclude_ids=["0", "1", "2"], top_k=10)
        returned_ids = result["id"]
        assert "0" not in returned_ids
        assert "1" not in returned_ids
        assert "2" not in returned_ids

    def test_no_exclude_ids_returns_results(self, indexed_index):
        result = indexed_index.search("love story", top_k=5)
        assert len(result) > 0
```

- [ ] **Step 2: Run the unit filter tests to confirm they pass immediately**

```bash
uv run pytest tests/test_index.py::TestSQLFilters -v
```

Expected: PASS (these are pure unit tests, no lancedb needed)

- [ ] **Step 3: Run the search tests to confirm they fail**

```bash
uv run pytest tests/test_index.py::TestSearch -v
```

Expected: FAIL — `search` stub returns `None`

- [ ] **Step 4: Implement `search()` in `agentic_rec/index.py`**

Replace the `search` stub:

```python
    def search(
        self,
        text: str,
        exclude_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> datasets.Dataset:
        assert self.table is not None
        from sqlalchemy import column, literal

        filter_str = None
        if exclude_ids:
            expr = column("id").not_in([literal(v) for v in exclude_ids])
            filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))

        query = (
            self.table.search(text, query_type="hybrid")
            .nprobes(self.config.nprobes)
            .refine_factor(self.config.refine_factor)
            .rerank(self.reranker)
            .limit(top_k)
        )
        if filter_str:
            query = query.where(filter_str, prefilter=True)

        return datasets.Dataset(query.to_arrow())
```

- [ ] **Step 5: Run search tests to verify they pass**

```bash
uv run pytest tests/test_index.py::TestSearch -v
```

Expected: PASS (downloads cross-encoder model ~90 MB on first run)

- [ ] **Step 6: Commit**

```bash
git add agentic_rec/index.py tests/test_index.py
git commit -m "feat(index): implement hybrid search with sqlalchemy prefilter and reranker"
```

---

## Task 6: Implement `get_ids()`

**Files:**

- Modify: `agentic_rec/index.py`
- Modify: `tests/test_index.py`

- [ ] **Step 1: Add failing `get_ids` tests**

Append to `tests/test_index.py`:

```python

class TestGetIds:
    def test_returns_matching_rows(self, indexed_index):
        result = indexed_index.get_ids(["0", "1", "2"])
        assert isinstance(result, datasets.Dataset)
        assert set(result["id"]) == {"0", "1", "2"}

    def test_missing_id_not_returned(self, indexed_index):
        result = indexed_index.get_ids(["nonexistent-999"])
        assert len(result) == 0

    def test_subset_returned(self, indexed_index):
        result = indexed_index.get_ids(["5", "nonexistent-999"])
        assert len(result) == 1
        assert result["id"][0] == "5"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_index.py::TestGetIds -v
```

Expected: FAIL — `get_ids` stub returns `None`

- [ ] **Step 3: Implement `get_ids()` in `agentic_rec/index.py`**

Replace the `get_ids` stub:

```python
    def get_ids(self, ids: list[str]) -> datasets.Dataset:
        assert self.table is not None
        from sqlalchemy import column, literal

        expr = column("id").in_([literal(v) for v in ids])
        filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
        return datasets.Dataset(self.table.search().where(filter_str).to_arrow())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_index.py::TestGetIds -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agentic_rec/index.py tests/test_index.py
git commit -m "feat(index): implement get_ids with sqlalchemy IN filter"
```

---

## Task 7: Implement `save()` and `load()`

**Files:**

- Modify: `agentic_rec/index.py`
- Modify: `tests/test_index.py`

- [ ] **Step 1: Add failing `save`/`load` tests**

Append to `tests/test_index.py`:

```python

class TestSaveLoad:
    def test_save_creates_directory(self, indexed_index, tmp_path):
        import pathlib

        save_path = str(tmp_path / "saved_index")
        indexed_index.save(save_path)
        assert pathlib.Path(save_path).is_dir()

    def test_load_opens_table(self, indexed_index, tmp_path):
        save_path = str(tmp_path / "loaded_index")
        indexed_index.save(save_path)

        config = LanceIndexConfig(
            lancedb_path=save_path,
            table_name=indexed_index.config.table_name,
        )
        loaded = LanceIndex.load(config)
        assert loaded.table is not None
        assert loaded.table.count_rows() == indexed_index.table.count_rows()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_index.py::TestSaveLoad -v
```

Expected: FAIL — `save` is a stub, `load` doesn't open the table

- [ ] **Step 3: Implement `save()`, `_open_table()`, and `load()` in `agentic_rec/index.py`**

Replace the three stubs:

```python
    def save(self, path: str) -> None:
        shutil.copytree(self.config.lancedb_path, path)

    @classmethod
    def load(cls, config: LanceIndexConfig) -> LanceIndex:
        self = cls(config)
        self._open_table()
        return self

    def _open_table(self) -> lancedb.table.Table:
        import lancedb

        db = lancedb.connect(self.config.lancedb_path)
        self.table = db.open_table(self.config.table_name)
        logger.info(f"{self.__class__.__name__}: {self.table}")
        logger.info(
            f"num_items: {self.table.count_rows()}, columns: {self.table.schema.names}"
        )
        return self.table
```

- [ ] **Step 4: Run all tests to verify everything passes**

```bash
uv run pytest tests/test_index.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add agentic_rec/index.py tests/test_index.py
git commit -m "feat(index): implement save, load, and _open_table"
```
