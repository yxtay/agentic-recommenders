# ARAG Design Specification

**Paper:**
[ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation](https://arxiv.org/abs/2506.21931)
**Stack:** pydantic-ai, LanceDB, sentence-transformers, datasets, FastAPI

---

## Overview

A simplified single-agent implementation inspired by the ARAG framework, applied to MovieLens 1M.
A single `pydantic-ai` `Agent` orchestrates recommendation via two tools: item text lookup (by ID)
and candidate retrieval (hybrid search). The agent derives a preference summary from the user's
interaction history, issues multiple targeted retrieval queries for diversity, then ranks candidates
with per-item explanations. Served via FastAPI.

### Relationship to the paper

The ARAG paper proposes a **multi-agent** pipeline with four specialized LLM agents:

1. **User Understanding Agent** — synthesizes long-term and session-level preferences into a
    natural-language summary (S_user)
2. **NLI Agent** — scores semantic alignment between each candidate item and user intent;
    filters items below threshold θ
3. **Context Summary Agent** — summarizes metadata of NLI-accepted items (S_ctx)
4. **Item Ranker Agent** — ranks items using S_user and S_ctx; produces final permutation

The paper's pipeline: embedding-based retrieval → NLI filtering → context summarization → ranking.
Evaluated on Amazon Reviews (Clothing, Electronics, Home & Kitchen) with NDCG@5 and Hit@5.

**This implementation simplifies** the 4-agent pipeline into a single agent that performs all
reasoning stages (understanding, retrieval, filtering, ranking) within one LLM call. Key
differences from the paper:

| Aspect          | Paper (ARAG)                     | This implementation                     |
|-----------------|----------------------------------|-----------------------------------------|
| Agents          | 4 specialized LLM agents         | 1 agent with tool access                |
| Retrieval       | Cosine similarity top-k          | Hybrid (vector + FTS) with reranking    |
| Filtering       | NLI agent with threshold scoring | Agent reasoning (no explicit threshold) |
| Diversity       | Not explicitly addressed         | Multi-query retrieval (2-4 calls)       |
| Dataset         | Amazon Reviews (3 categories)    | MovieLens 1M                            |
| Cold-start      | Not addressed                    | Supported via text-only input           |
| Item-based recs | Not addressed                    | Supported via ITEM_INSTRUCTIONS         |

---

## Architecture

```text
Request (text, history: [{item_id, event_datetime, event_name, event_value}], limit)
    │
    ├─ [Tool] get_item_texts(item_ids)              ← skipped if cold-start
    │         Returns {item_id: item_text} from LanceDB.
    │
    ├─ LLM: context understanding
    │         Builds preference summary from text + item texts + event values.
    │
    ├─ [Tool] search_items(query, query_type, ...)   ← called 2-4 times
    │         Vector/FTS/Hybrid search, interacted IDs excluded.
    │         Queries derived from text, preference summary, item texts.
    │
    └─ LLM: deduplicate, rank, explain
              → RecommendResponse
```

---

## Structured Types

Defined in `agentic_rec/models.py`:

```python
class Interaction(pydantic.BaseModel):
    item_id: str
    event_datetime: datetime
    event_name: str
    event_value: float

class ItemCandidate(pydantic.BaseModel):
    id: str
    text: str
    score: float = 0.0

class ItemRecommended(pydantic.BaseModel):
    id: str
    text: str
    explanation: str

class RecommendRequest(pydantic.BaseModel):
    text: str                       # user demographics / preferences, or item description
    history: list[Interaction] = [] # past interactions; empty for cold-start
    limit: int = 10

class RecommendResponse(pydantic.BaseModel):
    items: list[ItemRecommended]

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

## LanceDB Index

**File:** `agentic_rec/index.py`

### Configuration

```python
class LanceIndexConfig(pydantic.BaseModel):
    lancedb_path: str       # default from settings
    table_name: str         # "items" or "users"
    embedder_name: str      # e.g. "lightonai/DenseOn"
    embedder_device: str    # "cpu"
    reranker_name: str      # e.g. "lightonai/LateOn"
    reranker_type: str      # "pylate"
```

### LanceIndex class

Uses `functools.cached_property` for lazy-loaded components (each loaded once on first access):

- **`embedder`** — sentence-transformers model via lancedb registry
- **`reranker`** — `AnswerdotaiRerankers` with pylate model type

### Methods

**`index_data(dataset, overwrite=False)`**

Input: `datasets.Dataset` with required `id` and `text` columns, plus any additional columns.
The schema is inferred from the dataset's PyArrow schema (supports nested list/struct types).
The `vector` column is computed automatically via the configured embedder. Steps:

1. Connect to LanceDB, create table from Arrow batches with `EmbeddingFunctionConfig`
  (auto-embeds `text` → `vector`). All dataset columns are preserved.
2. Create scalar index on `id` (fast point lookups).
3. Create FTS index on `text` (BM25 full-text search).
4. Create `IVF_RQ` vector index with heuristic partitioning (`2^(log2(n)/2)`).
5. Optimize table (cleanup, delete unverified).

**`search(text, query_type="hybrid", exclude_ids=None, limit=20) → pa.Table`**

Vector, FTS, or Hybrid search with answerdotai reranking. Filter built via sqlalchemy
(`column("id").not_in(...)` compiled with `literal_binds=True`) for SQL injection safety.
Returns dataset with columns: `id`, `text`, `vector`, `score`.

**`get_ids(ids) → datasets.Dataset`**

Scalar-indexed point lookup by ID list. Filter built via sqlalchemy (`column("id").in_(...)`).

**`save(path)` / `load(config)`**

Copy/open the LanceDB directory.

---

## Recommender Agent

**File:** `agentic_rec/agent.py`

### Agent definition

Module-level `pydantic_ai.Agent[AgentDeps, RecommendResponse]` singleton configured with
`model=settings.llm_model` and `output_type=RecommendResponse`.

```python
@dataclass
class AgentDeps:
    item_repository: ItemRepository
    request: RecommendRequest
```

### Prompt structure

- **`system_prompt`** (static): domain-agnostic recommendation workflow — context understanding,
  candidate retrieval, cold-start handling, deduplication, and ranking. Output schema field
  descriptions (via `pydantic.Field`) guide the LLM on response format.
- **`@agent.instructions`** (dynamic): serializes `ctx.deps.request` as JSON per-request.
- **Runtime `instructions`** (per call site): domain-specific context passed at `agent.run(instructions=...)`.

### Instructions

Two instruction sets for different recommendation modes:

- **`USER_INSTRUCTIONS`** — user-based: text contains user demographics and preferences,
  use history and text to understand taste, vary queries for diversity.
- **`ITEM_INSTRUCTIONS`** — item-based (similar items): text contains source item description,
  find diverse but related items, no interaction history.

### Tools

| Tool             | Delegates to                 | Purpose                                           |
|------------------|------------------------------|---------------------------------------------------|
| `get_item_texts` | `item_repository.get_by_ids` | Fetch text of interacted items by ID list         |
| `search_items`   | `item_repository.search`     | Vector, FTS, or Hybrid search with exclude filter |

Both tools use `strict=True` for OpenAI-compatible API compliance.
`search_items` accepts `query_type: Literal["vector", "fts", "hybrid"]`.

### Cold-start handling

When `history` is empty, the agent skips `get_item_texts` and uses `text` alone as the retrieval
signal. This is the natural path for item-based recommendations (item text, no history).

### Invocation

```python
response = await agent.run(instructions=USER_INSTRUCTIONS, deps=AgentDeps(item_repository, request))
response = await agent.run(instructions=ITEM_INSTRUCTIONS, deps=AgentDeps(item_repository, request))
```

---

## FastAPI Service

**File:** `agentic_rec/main.py`

### Architecture (3-Tier)

This implementation follows a 3-tier architecture to separate concerns and improve testability:

1. **Repository Layer (`agentic_rec/repositories/`)**:
    Wraps `LanceIndex` to provide specialized data access for items and users.
    - `ItemRepository`: search and point lookups for items.
    - `UserRepository`: point lookups for users.

2. **Service Layer (`agentic_rec/services/`)**:
    Contains business logic and orchestrates the agent and repositories.
    - `RecommendationService`: handles recommendation workflows (user-based, item-based, by ID).
    - `UserService`: basic user management.
    - `ItemService`: basic item management.

3. **API Layer (`agentic_rec/routers/`)**:
    FastAPI routers that handle HTTP requests and delegate to services.
    - `recommendations.py`: all recommendation endpoints.
    - `users.py`: user lookup endpoints.
    - `items.py`: item lookup endpoints.
    - `health.py`: system health and metadata.

### Routes

| Route                        | Method | Description                                           |
|------------------------------|--------|-------------------------------------------------------|
| `/healthz`                   | GET    | Service health and model configuration                |
| `/recommend`                 | POST   | User-based recommendations (alias: `/recommend/user`) |
| `/recommend/item`            | POST   | Item-based similar-item recommendations               |
| `/users/{user_id}`           | GET    | Look up user by ID (text + history)                   |
| `/users/{user_id}/recommend` | POST   | Recommend for an existing user by ID                  |
| `/items/{item_id}`           | GET    | Look up item by ID (text)                             |
| `/items/{item_id}/recommend` | POST   | Similar-item recommendations for an item by ID        |

### Startup (lifespan)

Resources loaded once via FastAPI lifespan context manager:

1. `LanceIndex.load(LanceIndexConfig())` — opens items LanceDB table.
2. `LanceIndex.load(LanceIndexConfig(table_name="users"))` — opens users LanceDB table.
3. `check_llm()` — verifies LLM API connectivity.

### Dependencies

Injected via FastAPI `Depends`:

- `ItemsIndexDep` — items `LanceIndex` from app state
- `UsersIndexDep` — users `LanceIndex` from app state

### Data flow

**`POST /recommend`** — directly invokes `agent.run(instructions=USER_INSTRUCTIONS, deps=...)`.

**`POST /recommend/item`** — invokes `agent.run(instructions=ITEM_INSTRUCTIONS, deps=...)`.

**`POST /users/{user_id}/recommend`** — looks up user, builds `RecommendRequest` from user text
and full history, delegates to `/recommend`.

**`POST /items/{item_id}/recommend`** — looks up item text, builds `RecommendRequest(text=item_text,
history=[], limit=limit)`, delegates to `/recommend/item`.

---

## Settings

**File:** `agentic_rec/settings.py`

`pydantic-settings` `Settings` class with `env_prefix="AGENTIC_REC_"`:

| Setting            | Default                 | Env var                        |
|--------------------|-------------------------|--------------------------------|
| `llm_model`        | `cerebras:gpt-oss-120b` | `AGENTIC_REC_LLM_MODEL`        |
| `embedder_name`    | `lightonai/DenseOn`     | `AGENTIC_REC_EMBEDDER_NAME`    |
| `reranker_name`    | `lightonai/LateOn`      | `AGENTIC_REC_RERANKER_NAME`    |
| `reranker_type`    | `pylate`                | `AGENTIC_REC_RERANKER_TYPE`    |
| `lance_db_path`    | `lance_db`              | `AGENTIC_REC_LANCE_DB_PATH`    |
| `items_table_name` | `items`                 | `AGENTIC_REC_ITEMS_TABLE_NAME` |
| `data_dir`         | `data`                  | `AGENTIC_REC_DATA_DIR`         |

Set the LLM model and its matching API key:

```bash
export AGENTIC_REC_LLM_MODEL="cerebras:gpt-oss-120b"
export CEREBRAS_API_KEY="..."
```

---

## Key Design Decisions

- **Two tools, not five**: context understanding and ranking are LLM reasoning steps, not tool calls.
  Reduces latency and round-trips.
- **Input is text + history, not user_id**: caller supplies demographics/preferences and interaction
  history directly. Supports cold-start (empty history) without a separate endpoint.
- **Same agent for item-based recommendations**: the cold-start path naturally handles item text as
  the sole signal via `ITEM_INSTRUCTIONS`.
- **Multi-query retrieval for diversity**: agent issues 2-4 hybrid search calls with different query
  strategies rather than a single large retrieval.
- **Hypothetical item text**: when recent history is a poor query signal, the agent generates a
  synthetic item description matching real item schema, then uses it as the search query.
- **Per-item explanations**: `ItemRecommended.explanation` makes recommendations interpretable.
- **SQL injection safety**: all LanceDB `.where()` filters built via sqlalchemy Core with
  `literal_binds=True` compilation.
- **Users via LanceIndex**: users are stored in a LanceDB table (same as items), enabling scalar-indexed
  lookups by ID. The users table preserves all columns (demographics, history, splits) via PyArrow schema
  inference.
- **Lifespan over `@app.on_event`**: FastAPI-recommended pattern; events are deprecated.
- **Module-level agent singleton**: agent is stateless; all per-request context flows through `deps`.
