# agentic-recommenders

Implementation of [ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation](https://arxiv.org/abs/2506.21931) on the MovieLens 1M dataset.

## Overview

ARAG replaces static retrieval heuristics with an LLM agent that reasons about user preferences and item relevance. A single `pydantic-ai` agent orchestrates five tools in sequence:

1. **User Understanding** — builds a natural-language preference summary from the user's rating history and demographics
2. **Agentic Retrieval** — agent chooses among semantic search, full-text search, or hybrid search against a LanceDB item index
3. **NLI Scoring** — scores each candidate for semantic alignment with the preference summary
4. **Context Summary** — filters candidates above a threshold and summarises accepted items
5. **Item Ranking** — produces a final ranked list of movie recommendations

The system is served via a BentoML REST endpoint.

## Architecture

```
Request (user_id, top_k)
    │
    ├─ user_understanding(user_id)         → preference summary
    ├─ semantic_search / fulltext_search
    │   / hybrid_search (agent chooses)    → ItemCandidate list
    ├─ nli_score(candidates, summary)      → scored candidates
    ├─ context_summary(scored, threshold)  → context string
    └─ rank_items(candidates, summaries)   → ranked item_ids

POST /recommend → { item_ids, explanation }
```

## Requirements

- Python 3.12+
- `uv` for environment and task management — see `pyproject.toml`
- An LLM API key matching the configured model (default: OpenAI)

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

### 2. Build the item index

Encodes movie titles/genres with sentence-transformers and writes to LanceDB:

```bash
uv run index
```

### 3. Configure the LLM

```bash
export LLM_MODEL="openai:gpt-4o"   # any pydantic-ai model string
export OPENAI_API_KEY="sk-..."
```

Supported model strings: `openai:gpt-4o`, `anthropic:claude-haiku-4-5`, `ollama:llama3`, and any other [pydantic-ai provider](https://ai.pydantic.dev/models/).

### 4. Serve

```bash
uv run bentoml serve agentic_rec.service:RecommenderService
```

Request example:

```bash
curl -X POST http://localhost:3000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "1", "top_k": 10}'
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
