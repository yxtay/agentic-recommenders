# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# install dependencies
uv sync

# data preparation (downloads MovieLens 1M, converts to Parquet)
uv run data

# serve BentoML endpoint
uv run bentoml serve agentic_rec.service:RecommenderService

# lint and format
uv run ruff check --fix .
uv run ruff format .

# run all tests
uv run pytest

# run a single test
uv run pytest tests/test_index.py::TestSearch::test_returns_dataset -v
```

## Architecture

ARAG (Agentic Retrieval-Augmented Generation) for MovieLens 1M. A single `pydantic-ai` agent orchestrates five tools in sequence: user understanding → retrieval (semantic / FTS / hybrid) → NLI scoring → context summary → item ranking. Served via BentoML.

### Modules

| File                     | Responsibility                                               |
| ------------------------ | ------------------------------------------------------------ |
| `agentic_rec/params.py`  | All constants: paths, model names, table names               |
| `agentic_rec/data.py`    | MovieLens download, Parquet conversion, train/val/test split |
| `agentic_rec/index.py`   | LanceDB item index: embedding, hybrid search, reranking      |
| `agentic_rec/agent.py`   | pydantic-ai `Agent` with five tools _(planned)_              |
| `agentic_rec/service.py` | BentoML `POST /recommend` endpoint _(planned)_               |

### Data columns

All Parquet columns are prefixed by entity (`item_id`, `item_text`, `user_id`, `user_text`, `event_value`). The LanceDB index uses unprefixed names: `id`, `text`, `vector`.

### LanceDB index

`LanceIndex` lazily loads the sentence-transformers embedder and answerdotai reranker on first use. `index_data` takes a `datasets.Dataset` with `id` and `text` columns (pre-computed `vector` is optional — auto-generated if absent). `search` runs hybrid (vector + FTS) with answerdotai `rerank_hybrid`. Filters use `sqlalchemy` Core expressions compiled to strings (injection safety). See `docs/superpowers/specs/2026-04-28-lancedb-index-design.md` for full spec and `docs/superpowers/plans/2026-04-28-lancedb-index.md` for implementation plan.

### LLM configuration

Set `LLM_MODEL` (any pydantic-ai model string, default `openai:gpt-4o`) and the matching API key env var before serving.

## Key conventions

- Imports that require optional/heavy packages (`lancedb`, `rerankers`, `sentence_transformers`) are deferred inside methods rather than at module level. Ruff rule `PLC0415` is suppressed to allow this.
- `torch` is pinned to the CPU-only index — no GPU build.
- `uv.lock` must be committed; CI runs with `UV_LOCKED=1`.
