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

1. Add response models to `models.py`: `UserResponse`, `ItemResponse`, `InfoResponse`
2. Implement FastAPI lifespan (load index, users dataset, userid2idx dict, check LLM)
3. Define dependency functions (`get_index`, `get_users`, `get_userid2idx`)
4. Implement `GET /healthz` — service readiness check
5. Implement `GET /info` — return model configuration
6. Implement `POST /recommend` (+ `/recommend/user` alias) — user-based recommendations
7. Implement `POST /recommend/item` — item-based similar-item recommendations
8. Implement `GET /users/{user_id}` — user lookup from dataset
9. Implement `POST /users/{user_id}/recommend` — lookup + recommend with history truncation
10. Implement `GET /items/{item_id}` — item lookup from index
11. Implement `POST /items/{item_id}/recommend` — lookup + recommend in cold-start mode
12. Write route tests with mocked agent
