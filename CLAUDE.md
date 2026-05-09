# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session start

At the start of every conversation, read `README.md` and `CLAUDE.md` in full before doing anything else.
At the end of every conversation, review both files and update any sections that are out of date with changes
made during the session.

## Workflow

Commit changes regularly — after each logical unit of work (e.g., a completed feature, a bug fix,
a documentation update). Do not batch unrelated changes into a single commit.

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

# app sanity check (hits healthz, info, user/item lookup, user/item recommend)
uv run app

# serve FastAPI endpoint
uv run fastapi run

# lint and format
uv run ruff check --fix .
uv run ruff format .

# run all tests
uv run pytest

# run a single test
uv run pytest tests/test_index.py::TestSearch::test_returns_dataset -v
```

## Architecture

See `README.md` for overview and `docs/design.md` for full spec.

### Data columns

All Parquet columns are prefixed by entity (`item_id`, `item_text`, `user_id`, `user_text`, `event_value`,
`event_datetime`). The LanceDB index uses unprefixed names: `id`, `text`, `vector`.
Additional columns (e.g. `history`, `target`, split flags) are preserved with their original PyArrow types.

### App state

The FastAPI app loads two `LanceIndex` instances on startup:

- `app.state.items_index` — items table (default `LanceIndexConfig()`)
- `app.state.users_index` — users table (`LanceIndexConfig(table_name="users")`)

## Key conventions

- Imports that require optional/heavy packages (`lancedb`, `rerankers`, `sentence_transformers`) are deferred
  inside methods rather than at module level. Ruff rule `PLC0415` is suppressed to allow this.
- `torch` is pinned to the CPU-only index — no GPU build.
- `uv.lock` must be committed; CI runs with `UV_LOCKED=1`.
