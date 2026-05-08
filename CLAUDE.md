# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session start

At the start of every conversation, read `README.md` and `CLAUDE.md` in full before doing anything else.
At the end of every conversation, review both files and update any sections that are out of date with changes
made during the session.

## Commands

```bash
# install dependencies
uv sync

# data preparation (downloads MovieLens 1M, converts to Parquet)
uv run data

# build LanceDB index for items (default) or users
uv run index
uv run index --parquet_path data/ml-1m/users.parquet --table_name users

# run agent sanity check (samples a user, runs recommendation)
uv run agent

# serve FastAPI endpoint
uv run fastapi run agentic_rec.app:app

# lint and format
uv run ruff check --fix .
uv run ruff format .

# run all tests
uv run pytest

# run a single test
uv run pytest tests/test_index.py::TestSearch::test_returns_dataset -v
```

## Architecture

ARAG (Agentic Retrieval-Augmented Generation) for MovieLens 1M. A single `pydantic-ai` agent receives
`text` (demographics/preferences) and `history` (past interactions, may be empty for cold-start) and
orchestrates two tools: item-text lookup by ID (`get_item_texts`) and hybrid candidate retrieval
(`search_items`). Context understanding and item ranking with per-item explanations are done by the agent's
LLM directly (no separate tool calls). Served via FastAPI.

### Modules

| File                       | Responsibility                                                      |
|----------------------------|---------------------------------------------------------------------|
| `agentic_rec/settings.py`  | pydantic-settings `Settings` class: paths, model names, table names |
| `agentic_rec/data.py`      | MovieLens download, Parquet conversion, train/val/test split        |
| `agentic_rec/index.py`     | LanceDB item index: embedding, hybrid search, reranking             |
| `agentic_rec/models.py`    | Pydantic models: request/response types                             |
| `agentic_rec/agent.py`     | pydantic-ai `Agent` singleton with tools                            |
| `agentic_rec/app.py`       | FastAPI `POST /recommend` endpoint _(planned)_                      |

### Data columns

All Parquet columns are prefixed by entity (`item_id`, `item_text`, `user_id`, `user_text`, `event_value`,
`event_datetime`). The LanceDB index uses unprefixed names: `id`, `text`, `vector`.

### LanceDB index

`LanceIndex` uses `functools.cached_property` for the sentence-transformers embedder, `LanceModel` schema,
and answerdotai reranker — each loaded once on first access. Key methods:

- `index_data(dataset, overwrite=False)` — creates the table from a `datasets.Dataset` with `id` and `text`
  columns; builds scalar index on `id`, FTS index on `text`, and `IVF_RQ` vector index. `vector` is
  auto-embedded if absent.
- `search(text, exclude_ids, limit)` — hybrid (vector + FTS) search with answerdotai reranker; `exclude_ids`
  filter built via `sqlalchemy` (injection safety). Returns a `datasets.Dataset` with columns `id`, `text`,
  `vector`, `score`.
- `get_ids(ids)` — scalar-indexed point lookup by ID list.
- `save(path)` / `load(config)` — copy/open the LanceDB directory.

`LanceIndexConfig` fields: `lancedb_path`, `table_name`, `embedder_name`, `embedder_device`,
`reranker_name`, `reranker_type`.

See `docs/superpowers/specs/2026-04-28-lancedb-index-design.md` for full spec.

### Agent

Module-level `pydantic_ai.Agent` singleton with `AgentDeps` (index + request) for dependency injection.

- **`system_prompt`** (static): generic item recommendation workflow (domain-agnostic).
- **`@agent.instructions`** (dynamic): per-request user context (JSON-serialized request).
- **Runtime `instructions`**: domain-specific context (e.g., "items are movies with title and genres")
  passed at `agent.run(instructions=..., deps=...)` call site.
- **Tools**: `get_item_texts` (delegates to `index.get_ids`) and `search_items` (delegates to
  `index.search`). Both access the index via `ctx.deps.index`.

Callers use `agent.run(instructions=DOMAIN_INSTRUCTIONS, deps=AgentDeps(...))` directly.

Request accepts `text` (required), `history` (list of interactions, defaults to `[]`),
and `limit` (default 10). Cold-start (empty history) is handled by the agent using `text`
alone for retrieval.

See `docs/superpowers/specs/2026-05-07-arag-request-redesign.md` for full spec.

### LLM configuration

Set `LLM_MODEL` (any pydantic-ai model string, default `cerebras:llama3.1-8b`) and the matching API key env var before serving.

## Key conventions

- Imports that require optional/heavy packages (`lancedb`, `rerankers`, `sentence_transformers`) are deferred
  inside methods rather than at module level. Ruff rule `PLC0415` is suppressed to allow this.
- `torch` is pinned to the CPU-only index — no GPU build.
- `uv.lock` must be committed; CI runs with `UV_LOCKED=1`.
