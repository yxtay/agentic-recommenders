# FastAPI Web Service for Agentic Recommender

**Date:** 2026-05-08
**Stack:** FastAPI, pydantic-ai, LanceDB, datasets

---

## Overview

Web service exposing the ARAG recommender agent via REST API. Provides recommendation routes for
both user context and item context, plus demo lookup routes for retrieving sample users/items by ID.

Item-based recommendations reuse the same agent flow in cold-start mode: the item's text becomes the
`text` field with an empty history.

---

## Routes

| Method | Path                         | Purpose                                                          |
|--------|------------------------------|------------------------------------------------------------------|
| `POST` | `/recommend`                 | Recommend items given full user context (text + history + limit) |
| `GET`  | `/users/{user_id}`           | Retrieve a user record (text + history) from users parquet       |
| `POST` | `/users/{user_id}/recommend` | Fetch user by ID, run agent with their context                   |
| `GET`  | `/items/{item_id}`           | Retrieve an item record (id + text) from the items index         |
| `POST` | `/items/{item_id}/recommend` | Fetch item text, run agent in cold-start mode                    |
| `GET`  | `/info`                      | Return model configuration (embedder, reranker, LLM)             |

---

## Architecture

```text
Client
  │
  ├─ POST /recommend ──────────────────────┐
  │     Body: RecommendRequest             │
  │                                        ▼
  ├─ POST /users/{id}/recommend ──┐    ┌──────────┐    ┌───────┐
  │     Fetches user from parquet │───▶│  Agent   │───▶│ Index │
  │                               │    │  .run()  │    │(Lance)│
  ├─ POST /items/{id}/recommend ──┘    └──────────┘    └───────┘
  │     Fetches item from index,           │
  │     builds cold-start request          ▼
  │                                  RecommendResponse
  ├─ GET /users/{id} ──▶ users.parquet lookup
  ├─ GET /items/{id} ──▶ index.get_ids() lookup
  └─ GET /info ──▶ settings/config lookup
```

---

## Data Flow

### `POST /recommend`

Input: `RecommendRequest` (text, history, limit) — same as the existing spec.
Output: `RecommendResponse` (list of RankedItem with explanations).

Directly invokes `agent.run(instructions=INSTRUCTIONS, deps=AgentDeps(index, request))`.

### `POST /users/{user_id}/recommend`

1. Look up user by ID in users dataset (loaded at startup from `data/ml-1m/users.parquet`).
2. Build `RecommendRequest` from stored `text` and `history` (truncated to last 20 interactions).
3. Accept optional `limit` query param (default 10).
4. Run agent as above.

### `POST /items/{item_id}/recommend`

1. Look up item by ID via `index.get_ids([item_id])`.
2. Build `RecommendRequest(text=item_text, history=[], limit=limit)` — cold-start mode.
3. Accept optional `limit` query param (default 10).
4. Run agent — skips `get_item_texts` tool, uses item text directly for retrieval queries.

### `GET /users/{user_id}`

Returns `UserResponse(id, text, history)`. 404 if not found.

### `GET /items/{item_id}`

Returns `ItemResponse(id, text)`. 404 if not found.

### `GET /info`

Returns `InfoResponse` with service configuration:

- `embedder_name`: sentence-transformers model used for embedding (e.g. `"lightonai/DenseOn"`)
- `reranker_name`: reranker model (e.g. `"lightonai/LateOn"`)
- `llm_model`: pydantic-ai model string (e.g. `"cerebras:llama3.1-8b"`)

---

## Startup

Use FastAPI lifespan context manager to load resources once:

1. `LanceIndex.load(LanceIndexConfig())` — opens the items LanceDB table.
2. `datasets.Dataset.from_parquet(settings.users_parquet)` — loads users into memory for demo lookups.

Both are stored on `app.state` and injected into route handlers via FastAPI dependencies.

---

## Response Models

New models added to `agentic_rec/models.py`:

```python
class UserResponse(pydantic.BaseModel):
    id: str
    text: str
    history: list[Interaction] = []

class ItemResponse(pydantic.BaseModel):
    id: str
    text: str

class InfoResponse(pydantic.BaseModel):
    embedder_name: str
    reranker_name: str
    llm_model: str
```

---

## Modules Modified

| File                    | Change                                             |
|-------------------------|----------------------------------------------------|
| `agentic_rec/app.py`    | New — FastAPI app with lifespan, all routes        |
| `agentic_rec/models.py` | Add `UserResponse`, `ItemResponse`, `InfoResponse` |
| `tests/test_app.py`     | New — route tests with mocked agent/index          |

---

## Key Design Decisions

- **Same agent flow for items**: No separate "similar items" logic. The agent's cold-start path
  naturally handles item text as the sole signal, providing explanations and diversity.
- **Lifespan over `@app.on_event`**: FastAPI recommends lifespan context managers; events are deprecated.
- **Users in memory**: Dataset is small (6040 users for ML-1M). Filtered by ID at request time.
- **History truncation**: Cap at last 20 interactions for convenience routes (matches `agent.py:main()`).
- **No BentoML**: Bottleneck is external LLM calls, not local model inference. FastAPI's async
  support is sufficient.

---

## Running

```bash
# Prerequisites
uv run data    # download and prepare MovieLens data
uv run index   # build LanceDB index

# Serve
uv run fastapi dev agentic_rec/app.py

# Or production
uv run fastapi run agentic_rec/app.py
```

---

## Verification

```bash
# Lint and format
uv run ruff check --fix . && uv run ruff format .

# Tests
uv run pytest tests/test_app.py -v

# Manual smoke test (requires index + LLM API key)
curl localhost:8000/items/1
curl -X POST localhost:8000/items/1/recommend
curl localhost:8000/users/1
curl -X POST localhost:8000/users/1/recommend
curl -X POST localhost:8000/recommend -H 'Content-Type: application/json' \
  -d '{"text": "25-year-old male who enjoys sci-fi", "limit": 5}'
```
