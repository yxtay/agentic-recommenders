# FastAPI Service Implementation Plan

**Goal:** Expose the ARAG recommender agent via a REST API with routes for direct
recommendation, user/item lookup, and convenience recommend-by-ID endpoints.

**Architecture:** FastAPI app with lifespan context manager for startup resource loading.
The agent is a module-level singleton; per-request state flows through `AgentDeps`.
Users stored as memory-mapped `datasets.Dataset` with a `dict[str, int]` ID-to-index mapping.

**Tech Stack:** FastAPI, pydantic-ai, LanceDB, HuggingFace datasets.

---

## File Map

| File                    | Change                                              |
|-------------------------|-----------------------------------------------------|
| `agentic_rec/models.py` | Add `UserResponse`, `ItemResponse`, `InfoResponse`  |
| `agentic_rec/app.py`    | FastAPI app with lifespan, all routes, dependencies |
| `tests/test_app.py`     | Route tests with mocked agent/index                 |

---

## Tasks

### 1. Add response models to `models.py`

- **`UserResponse`**: `id`, `text`, `history` (list of Interaction). For user lookup.
- **`ItemResponse`**: `id`, `text`. For item lookup.
- **`InfoResponse`**: `embedder_name`, `reranker_name`, `llm_model`. For service info.

### 2. Implement FastAPI lifespan

Async context manager that loads resources once at startup:

1. `LanceIndex.load(LanceIndexConfig())` — opens the items LanceDB table.
2. `datasets.Dataset.from_parquet(settings.users_parquet)` — memory-mapped users.
3. `dict(zip(users["id"], range(len(users))))` — user ID → row index mapping.
4. `check_llm()` — verifies LLM API key is valid (non-blocking; app starts regardless).

All stored on `app.state` for access via dependency injection.

### 3. Define dependency functions

Three dependencies injected via FastAPI `Depends`:

- `get_index() → LanceIndex` from `app.state.index`.
- `get_users() → datasets.Dataset` from `app.state.users`.
- `get_userid2idx() → dict[str, int]` from `app.state.userid2idx`.

Define `Annotated` type aliases (`IndexDep`, `UsersDep`, `UserId2IdxDep`) for route signatures.

### 4. Implement `GET /healthz`

Return service readiness status: index loaded, user count, LLM connectivity. Used by
health check probes (Kubernetes, load balancers).

### 5. Implement `GET /info`

Return model configuration from settings: embedder name, reranker name, LLM model string.
Useful for debugging which models a deployed instance is running.

### 6. Implement `POST /recommend` (+ `/recommend/user` alias)

Accept `RecommendRequest` body, invoke `agent.run(instructions=USER_INSTRUCTIONS, deps=...)`.
Return `RecommendResponse`. Both paths share the same handler via stacked decorators.

### 7. Implement `POST /recommend/item`

Accept `RecommendRequest` body, invoke `agent.run(instructions=ITEM_INSTRUCTIONS, deps=...)`.
Return `RecommendResponse`. Separate handler because it uses different instructions.

### 8. Implement `GET /users/{user_id}`

Look up user in the memory-mapped dataset via `userid2idx[user_id]`. Return `UserResponse`
with ID, text, and full interaction history. 404 if not found.

### 9. Implement `POST /users/{user_id}/recommend`

Convenience route: look up user, truncate history to last 20 interactions (keeps prompt
manageable), build `RecommendRequest`, delegate to the `/recommend` handler.
Accept optional `limit` query parameter (default 10).

### 10. Implement `GET /items/{item_id}`

Look up item via `index.get_ids([item_id])`. Return `ItemResponse` with ID and text.
404 if not found.

### 11. Implement `POST /items/{item_id}/recommend`

Convenience route: look up item, build `RecommendRequest(text=item_text, history=[], limit=limit)`,
delegate to the `/recommend/item` handler. Cold-start mode — the item's text becomes the
sole retrieval signal.

### 12. Write route tests

Mock the agent (patch `agent.run` to return a fixed `RecommendResponse`) so tests don't
require an LLM API key. Test:

- Health/info endpoints return expected structure.
- Recommend endpoints accept valid requests and return valid responses.
- User/item lookup returns 404 for missing IDs.
- Convenience recommend routes delegate correctly.

---

## Design Notes

- **Lifespan over `@app.on_event`**: FastAPI-recommended pattern; events are deprecated.
- **Memory-mapped users**: `datasets.Dataset` (Arrow backend) keeps user data on disk;
  only accessed rows are paged into RAM. The `dict[str, int]` mapping is trivially small.
- **History truncation**: cap at 20 for convenience routes to keep LLM context reasonable.
  Direct `/recommend` accepts the full history as-is.
- **Same agent for both modes**: `USER_INSTRUCTIONS` vs `ITEM_INSTRUCTIONS` at the call
  site is the only difference. No separate agent instance needed.
- **No BentoML**: bottleneck is external LLM API calls, not local model inference.
  FastAPI's native async support is sufficient for concurrent requests.
