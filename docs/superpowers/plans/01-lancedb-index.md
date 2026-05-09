# LanceDB Index Implementation Plan

**Goal:** Implement `LanceIndex` using `LanceModel` for automatic embedding, hybrid search with
answerdotai reranking, and sqlalchemy-built WHERE clauses for SQL injection safety.

**Architecture:** `LanceIndexConfig` holds tunable parameters (paths, model names). `LanceIndex`
wraps a LanceDB table; it lazily loads the embedder, schema, and reranker on first use via
`functools.cached_property`. `index_data` creates the table and all three index types; `search`
runs hybrid (vector + FTS) search and reranks results; `get_ids` does scalar-indexed point lookups.

**Tech Stack:** lancedb, sentence-transformers (via lancedb registry), answerdotai rerankers
(via lancedb), sqlalchemy (filter string generation), HuggingFace datasets, pydantic.

---

## File Map

| File                   | Change                                                          |
|------------------------|-----------------------------------------------------------------|
| `pyproject.toml`       | Add `rerankers`, `sqlalchemy` to `[project.dependencies]`       |
| `agentic_rec/index.py` | Config, LanceModel schema, hybrid search, rerank                |
| `tests/test_index.py`  | Unit + integration tests for all index operations               |

---

## Tasks

### 1. Add dependencies

Add `rerankers[transformers]` and `sqlalchemy>=2.0` to `pyproject.toml`. Run `uv sync` and
verify imports resolve.

### 2. Implement `LanceIndexConfig`

Pydantic `BaseModel` with fields: `lancedb_path`, `table_name`, `embedder_name`,
`embedder_device`, `reranker_name`, `reranker_type`. Defaults pulled from `settings`.

### 3. Implement lazy-loaded properties

Use `functools.cached_property` for three components:

- **`embedder`** — loads sentence-transformers model via `lancedb.embeddings.get_registry()`.
  Deferred import avoids loading the model at module import time.
- **`schema`** — builds a `LanceModel` subclass with `id: str`, `text: SourceField()`,
  `vector: VectorField()`. The embedder is captured via `self.embedder` reference.
- **`reranker`** — loads `AnswerdotaiRerankers` with configured model name and type.

### 4. Implement `index_data(dataset, overwrite=False)`

Input: `datasets.Dataset` with `id` and `text` columns. Steps:

1. Cast `id` column to `pa.string()` if it's `large_string` (required for scalar index).
2. Connect to LanceDB at `config.lancedb_path`.
3. Create table from Arrow batches using the `schema` (auto-embeds `text` → `vector`).
4. Create scalar index on `id` — enables fast IN/= lookups.
5. Create FTS index on `text` — enables BM25 full-text search.
6. Create `IVF_RQ` vector index with heuristic partitioning: `num_partitions = 2^(log2(n)/2)`.
7. Optimize table (cleanup old versions, delete unverified fragments).

Early-return if `self.table` exists and `overwrite=False`.

### 5. Implement `search(text, exclude_ids=None, limit=20)`

Hybrid (vector + FTS) search with reranking. Steps:

1. Build prefilter via sqlalchemy if `exclude_ids` is non-empty:
    `column("id").not_in([literal(v) for v in exclude_ids])` compiled with `literal_binds=True`.
2. Execute: `table.search(text, query_type="hybrid").rerank(reranker).limit(limit)`.
    Apply `.where(filter_str, prefilter=True)` if filter exists.
3. Convert result to `datasets.Dataset`, rename `_relevance_score` → `score`.

Returns dataset with columns: `id`, `text`, `vector`, `score`.

### 6. Implement `get_ids(ids)`

Scalar-indexed point lookup. Build `column("id").in_([literal(v) for v in ids])` via
sqlalchemy, then `table.search().where(filter_str).to_arrow()`. Return empty dataset
if `ids` is empty.

### 7. Implement `save(path)` and `load(config)`

- `save`: `shutil.copytree(config.lancedb_path, path)`.
- `load` (classmethod): create instance, call `open_table()`, return.
- `open_table`: connect to LanceDB, open existing table by name.

### 8. Write tests

- **Config tests**: defaults match settings, custom values propagate.
- **Schema tests**: `LanceModel` subclass with correct field names, cached.
- **Index tests**: table created with correct row count and columns, overwrite=False skips.
- **Search tests**: returns Dataset, respects limit, excludes specified IDs.
- **SQL filter tests**: injection attempts are escaped (quoted/doubled).
- **Get IDs tests**: returns matching rows, missing IDs omitted gracefully.
- **Save/load tests**: directory created, loaded table has same row count.

Use a session-scoped fixture with a small synthetic dataset (100 items) for integration tests.

---

## Design Notes

- **SQL injection safety**: never interpolate user-supplied IDs into SQL strings directly.
  Always use `sqlalchemy.literal()` with `literal_binds=True` compilation.
- **Deferred imports**: `lancedb`, `rerankers`, `sentence_transformers` are heavy; import
  inside methods/properties, not at module level. Suppress ruff `PLC0415`.
- **IVF_RQ over IVF_HNSW_PQ**: simpler, fewer hyperparameters, adequate for MovieLens scale.
- **`datasets.Dataset` as return type**: provides Arrow-backed memory efficiency and
  interoperability with the rest of the pipeline.
