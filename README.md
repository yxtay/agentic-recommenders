# agentic-recommenders

Implementation of
[ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation](https://arxiv.org/abs/2506.21931)
on the MovieLens 1M dataset.

## Overview

ARAG replaces static retrieval heuristics with an LLM agent that reasons about user preferences and item relevance.
A single `pydantic-ai` agent receives `text` (demographics/preferences) and `history` (past interactions,
may be empty for cold-start users) and works in three stages:

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
    └─ LLM: rank + explain                 → RankedItem list

POST /recommend → [{ item_id, item_text, explanation }]
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
export LLM_MODEL="cerebras:llama3.1-8b"   # any pydantic-ai model string
export CEREBRAS_API_KEY="..."
```

Supported model strings: `cerebras:llama3.1-8b`, `anthropic:claude-haiku-4-5`, `ollama:llama3`, and any other
[pydantic-ai provider](https://ai.pydantic.dev/models/).

### 4. Serve

```bash
uv run fastapi run agentic_rec.app:app
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
- [Design spec](docs/superpowers/specs/2026-04-28-arag-design.md)
