# agentic-recommenders

A simplified single-agent implementation inspired by
[ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation](https://arxiv.org/abs/2506.21931),
applied to the MovieLens 1M dataset.

## Overview

The ARAG paper proposes a 4-agent pipeline (User Understanding, NLI filtering, Context Summary, Item Ranker)
for personalized recommendation. This project distills that into a single `pydantic-ai` agent that performs
all reasoning stages within one LLM call, augmented with tool access for retrieval.

The agent receives `text` (demographics/preferences) and `history` (past interactions, may be empty for
cold-start users) and works in four stages:

1. **Item text lookup** — fetches the text of interacted items from LanceDB by ID
2. **Context understanding** — LLM builds a preference summary from interaction history and item texts,
    emphasising recent events
3. **Candidate retrieval** — agent issues multiple hybrid-search queries (using recent item texts and/or
    a generated hypothetical item description) for diversity; interacted items are excluded
4. **Ranking with explanations** — LLM ranks candidates by relevance and diversity, attaching a
    one-sentence explanation to each recommendation

The system is served via a FastAPI REST endpoint.

## Architecture

```text
Request (text, history: [{item_id, event_datetime, event_name, event_value}], limit)
    │
    ├─ [Tool 1] get_item_texts(item_ids)    → {item_id: item_text}  (skipped if cold-start)
    ├─ LLM: context understanding           → preference summary
    ├─ [Tool 2] search_items(query, exclude_ids) → candidates  (called 2-4×)
    └─ LLM: rank + explain                 → ItemRecommended list

POST /recommend      → { items: [{ id, text, explanation }] }
POST /recommend/item → { items: [{ id, text, explanation }] }
```

## Requirements

- Python 3.12+
- `uv` for environment and task management — see `pyproject.toml`
- An LLM API key matching the configured model

```bash
uv sync
```

## Setup

### 1. Prepare MovieLens data

Downloads, extracts, and converts MovieLens 1M into Parquet files under `data/`:

```bash
uv run data
```

If you already have `ml-1m.zip`, place it under `data/` before running the command.

### 2. Build the LanceDB indexes

Encodes items and users with sentence-transformers and writes to LanceDB:

```bash
uv run index
uv run index --parquet_path data/ml-1m/users.parquet --table_name users
```

### 3. Configure the LLM

```bash
export AGENTIC_REC_LLM_MODEL="cerebras:llama3.1-8b"   # any pydantic-ai model string
export CEREBRAS_API_KEY="..."
```

Supported model strings: `cerebras:llama3.1-8b`, `anthropic:claude-haiku-4-5`, `ollama:llama3`, and any other
[pydantic-ai provider](https://ai.pydantic.dev/models/).

### 4. Serve

```bash
uv run fastapi run agentic_rec.main:app
```

Request example:

```bash
curl -X POST http://localhost:3000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "text": "25-year-old male, software engineer, enjoys sci-fi and thriller films",
    "history": [
      {"item_id": "1193", "event_datetime": "2000-12-31T22:12:40", "event_name": "rating", "event_value": 5},
      {"item_id": "661",  "event_datetime": "2000-12-31T22:35:09", "event_name": "rating", "event_value": 3}
    ],
    "limit": 10
  }'
```

## Modules

| File                          | Responsibility                                                      |
|-------------------------------|---------------------------------------------------------------------|
| `agentic_rec/settings.py`     | pydantic-settings `Settings` class: paths, model names, table names |
| `agentic_rec/data.py`         | MovieLens download, Parquet conversion, train/val/test split        |
| `agentic_rec/index.py`        | LanceDB index: embedding, hybrid search, reranking (items & users)  |
| `agentic_rec/models.py`       | Pydantic models: request/response types                             |
| `agentic_rec/agent.py`        | pydantic-ai `Agent` singleton with tools                            |
| `agentic_rec/main.py`         | FastAPI service entry point                                         |
| `agentic_rec/repositories/`   | Repository layer (data access)                                      |
| `agentic_rec/services/`       | Service layer (business logic)                                      |
| `agentic_rec/routers/`        | API layer (routers)                                                 |
| `agentic_rec/dependencies.py` | Dependency injection                                                |

## API Routes

| Route                        | Method | Description                                           |
|------------------------------|--------|-------------------------------------------------------|
| `/healthz`                   | GET    | Service health (index, users, LLM readiness)          |
| `/info`                      | GET    | Model configuration (embedder, reranker, LLM)         |
| `/recommend`                 | POST   | User-based recommendations (alias: `/recommend/user`) |
| `/recommend/item`            | POST   | Item-based similar-item recommendations               |
| `/users/{user_id}`           | GET    | Look up user by ID (text + history)                   |
| `/users/{user_id}/recommend` | POST   | Recommend for an existing user by ID                  |
| `/items/{item_id}`           | GET    | Look up item by ID (text)                             |
| `/items/{item_id}/recommend` | POST   | Similar-item recommendations for an item by ID        |

The `/users/{user_id}/recommend` and `/items/{item_id}/recommend` convenience routes look up the entity and
then delegate to the corresponding `/recommend` or `/recommend/item` endpoint.

## Development

```bash
# lint and format
uv run ruff check --fix .
uv run ruff format .

# tests
uv run pytest
```

## References

- [ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation](https://arxiv.org/abs/2506.21931)
- [Design spec](docs/design.md)
