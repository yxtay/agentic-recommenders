# LanceDB Index Implementation Plan

**Goal:** Rewrite `LanceIndex` to use `LanceModel` for automatic embedding, hybrid search with
answerdotai reranking, and sqlalchemy-built WHERE clauses for SQL injection safety.

**Architecture:** `LanceIndexConfig` holds tunable parameters (paths, model names). `LanceIndex`
wraps a LanceDB table; it lazily loads the embedder, schema, and reranker on first use via
`functools.cached_property`. `index_data` creates the table and all three index types; `search`
runs hybrid (vector + FTS) search and reranks results; `get_ids` does scalar-indexed point lookups.

**Tech Stack:** lancedb, sentence-transformers (via lancedb registry), answerdotai rerankers
(via lancedb), sqlalchemy (filter string generation), HuggingFace datasets, pydantic.

---

## File Map

| File                   | Change                                                                       |
|------------------------|------------------------------------------------------------------------------|
| `agentic_rec/index.py` | Full rewrite — config, LanceModel schema, hybrid search, reranker            |
| `tests/test_index.py`  | Unit + integration tests                                                     |
| `pyproject.toml`       | Add `rerankers`, `sqlalchemy` to dependencies                                |

---

## Tasks

1. Add `rerankers` and `sqlalchemy` dependencies to `pyproject.toml`
2. Implement `LanceIndexConfig` with embedder/reranker settings
3. Implement `LanceIndex` with lazy-loaded `embedder`, `schema`, `reranker` cached properties
4. Implement `index_data()` — create table, scalar/FTS/IVF_RQ indices, optimize
5. Implement `search()` — hybrid search with sqlalchemy-built exclude filter and reranking
6. Implement `get_ids()` — scalar-indexed point lookup with sqlalchemy IN filter
7. Implement `save()` / `load()` — copytree and open_table
8. Write tests for each component
