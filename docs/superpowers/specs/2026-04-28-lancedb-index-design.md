# LanceDB Index Design

**Date:** 2026-04-28
**File:** `agentic_rec/index.py`

---

## Overview

Rewrite `LanceIndex` to use `LanceModel` for automatic embedding via lancedb's registry, hybrid search with answerdotai reranking, and sqlalchemy-built filters for SQL injection safety.

---

## `LanceIndexConfig`

```python
class LanceIndexConfig(pydantic.BaseModel):
    lancedb_path: str = LANCE_DB_PATH
    table_name: str = ITEMS_TABLE_NAME
    embedder_model_name: str = EMBEDDER_MODEL_NAME
    embedder_device: str = "cpu"
    reranker_model_name: str = CROSS_ENCODER_MODEL_NAME
    reranker_model_type: str = "cross-encoder"
    nprobes: int = 8
    refine_factor: int = 4
```

---

## `ItemSchema` (LanceModel)

Built at runtime via `_build_schema()`, using a normal class definition inside the factory to capture `embedder` via closure:

```python
def _build_schema(self) -> type[LanceModel]:
    embedder = get_registry().get("sentence-transformers").create(
        name=self.config.embedder_model_name,
        device=self.config.embedder_device,
    )

    class ItemSchema(LanceModel):
        id: str
        text: str = embedder.SourceField()
        vector: Vector(embedder.ndims()) = embedder.VectorField()

    return ItemSchema
```

Field names (`id`, `text`, `vector`) are hardcoded. The schema registers the sentence-transformers embedding function in lancedb's registry, so:

- Any row missing `vector` at index time gets it auto-computed from `text`.
- At query time, a text string passed to `.search()` is auto-embedded.

No `ID_COL`/`TEXT_COL`/`EMBEDDING_COL` constants needed. Filter expressions in `search` and `get_ids` use the string literals `"id"` and `"text"` directly.

---

## `LanceIndex`

### Lazy-loaded properties

- `_schema`: built once via `_build_schema()` (loads embedder model on first call)
- `_reranker`: built once via `Reranker(config.reranker_model_name, model_type=config.reranker_model_type)` from answerdotai `rerankers`
- `table`: `lancedb.table.Table | None`, set by `open_table()` or `index_data()`

### `index_data(dataset, overwrite)`

Input: `datasets.Dataset` with at least `id` (str) and `text` (str) columns. `vector` column is optional — auto-generated if absent.

Steps:

1. Connect to lancedb at `config.lancedb_path`.
2. `db.create_table(name, data=dataset.data.to_batches(), schema=schema, mode="overwrite")`.
3. `table.create_scalar_index(ID_COL)` — enables fast `IN` / `=` lookups.
4. `table.create_fts_index(TEXT_COL)` — BM25 full-text search.
5. Vector index (`IVF_HNSW_PQ`) on `EMBEDDING_COL` with heuristic sizing:
    - `num_partitions = 2 ** int(log2(n) / 2)`
    - `num_sub_vectors = ndims // 8`
6. `table.optimize(cleanup_older_than=timedelta(days=0), delete_unverified=True, retrain=True)`.

Returns `lancedb.table.Table`.

### `search(text, exclude_ids, top_k)`

```python
def search(
    self,
    text: str,
    exclude_ids: list[str] | None = None,
    top_k: int = 20,
) -> datasets.Dataset:
```

Steps:

1. Build prefilter using sqlalchemy (skip entirely if `exclude_ids` is empty):

    ```python
    from sqlalchemy import column, literal
    filter_str = None
    if exclude_ids:
        expr = column("id").not_in([literal(v) for v in exclude_ids])
        filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
    ```

2. Execute:

    ```python
    query = (
        table.search(text, query_type="hybrid")
        .nprobes(config.nprobes)
        .refine_factor(config.refine_factor)
        .rerank(reranker)
        .limit(top_k)
    )
    if filter_str:
        query = query.where(filter_str, prefilter=True)
    result = query.to_arrow()
    ```

    lancedb auto-embeds `text` for the vector leg, runs FTS in parallel, then passes both ranked lists to the answerdotai reranker's `rerank_hybrid()` method.
3. Return `datasets.Dataset(result_table)`.

### `get_ids(ids)`

```python
def get_ids(self, ids: list[str]) -> datasets.Dataset:
```

Builds `column("id").in_(...)` filter via sqlalchemy, then:

```python
table.search().where(filter_str).to_arrow()
```

Uses the scalar index for fast point lookups.

### `save(path)`

`shutil.copytree(config.lancedb_path, path)` — copies the full lancedb directory.

### `load(cls, config)` (classmethod)

Opens the existing table at `config.lancedb_path` / `config.table_name`. No index inference needed — column names are fixed constants.

---

## SQL Injection Safety

All lancedb `.where()` filter strings are built via sqlalchemy Core expressions with `literal_binds=True` compilation. This escapes string values rather than interpolating them directly, preventing SQL injection through item IDs.

---

## Dependencies to Add

| Package | Reason |
|---|---|
| `rerankers` (answerdotai) | answerdotai reranker with `rerank_hybrid` support |
| `sqlalchemy` | explicit pin (currently transitive via mlflow) |

Add to `pyproject.toml` dependencies.

---

## What Changes in `params.py`

No new constants needed. `CROSS_ENCODER_MODEL_NAME` stays as the default value source for `LanceIndexConfig.reranker_model_name`.
