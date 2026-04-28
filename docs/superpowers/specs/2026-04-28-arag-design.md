# ARAG: Agentic Retrieval Augmented Generation for MovieLens

**Date:** 2026-04-28
**Paper:** [ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation](https://arxiv.org/abs/2506.21931)
**Stack:** pydantic-ai, lancedb, sentence-transformers, datasets, BentoML, Polars

---

## Overview

Implementation of the ARAG framework on MovieLens 1M. A single `pydantic-ai` `Agent` orchestrates five recommendation roles as tools: user understanding, agentic retrieval (semantic / FTS / hybrid), NLI scoring, context summarisation, and item ranking. The agent decides which LanceDB retrieval strategy (semantic, full-text, or hybrid) to use based on the user preference summary it derives first.

---

## Architecture

```
Request (user_id, top_k)
    │
    ├─ [Tool 1] user_understanding(user_id)
    │           Reads users.parquet; builds preference summary from
    │           demographic user_text + rated item history (title, genres,
    │           rating, datetime). Returns a natural-language string.
    │
    ├─ Agent chooses retrieval strategy (may call multiple times):
    │   ├─ [Tool 2a] semantic_search(query, top_k)
    │   │            Embedding similarity via sentence-transformers +
    │   │            LanceDB IVF_HNSW_PQ vector index.
    │   ├─ [Tool 2b] fulltext_search(query, top_k)
    │   │            BM25/FTS index on item_text in LanceDB.
    │   └─ [Tool 2c] hybrid_search(query, top_k)
    │                Merges semantic and FTS results; deduplicates.
    │           → List[ItemCandidate]
    │
    ├─ [Tool 3] nli_score(candidates, preference_summary)
    │           LLM evaluates each candidate's item_text against the
    │           preference summary. Returns candidates annotated with
    │           an alignment score in [0, 1].
    │
    ├─ [Tool 4] context_summary(scored_candidates, threshold)
    │           Filters candidates above threshold θ.
    │           LLM produces a concise summary of accepted items.
    │           Returns summary string + filtered candidate list.
    │
    └─ [Tool 5] rank_items(candidates, preference_summary, context_summary)
                LLM ranks candidates by predicted purchase/watch likelihood
                considering session history, long-term preferences, and
                item-context alignment. Returns ordered list of item_ids.

BentoML /recommend → RecommendResponse(item_ids, explanations)
```

---

## Data Flow

### User context (Tool 1)

Built from `data/ml-1m/users.parquet`:
- `user_text`: demographic JSON (`gender`, `age`, `occupation`, `zipcode`)
- `history`: list of train-split interactions sorted by `datetime`, each containing `item_id`, `item_text` (title + genres), `event_value` (rating 1–5), `datetime`

The UUA tool formats these into a natural-language prompt summarising long-term taste and recent session behaviour. No LLM call in the tool body — the agent LLM produces the summary given the formatted context.

### Item candidates (Tools 2a/2b/2c)

LanceDB `items` table, populated from `data/ml-1m/items.parquet`. Each row stores `item_id` and `item_text` (JSON of `title` + `genres`) plus a `vector` column (sentence-transformer embedding of `item_text`).

- **semantic_search**: calls existing `LanceIndex.search()` with embedding of the query string
- **fulltext_search**: new method using existing FTS index via `LanceIndex.table.search(query).query_type("fts")`
- **hybrid_search**: calls both, merges by item_id using Reciprocal Rank Fusion (RRF), deduplicates, returns top_k

Items already in the user's history are excluded from candidates (already supported by `exclude_item_ids` in `LanceIndex.search`).

### Structured types

```python
class ItemCandidate(pydantic.BaseModel):
    item_id: str
    item_text: str
    score: float = 0.0
    nli_score: float = 0.0

class RecommendRequest(pydantic.BaseModel):
    user_id: str
    top_k: int = 10

class RecommendResponse(pydantic.BaseModel):
    item_ids: list[str]
    explanation: str
```

---

## Modules

| File | Responsibility |
|---|---|
| `agentic_rec/agent.py` | `Agent` definition, system prompt, all five tools, pydantic result types |
| `agentic_rec/index.py` | Extend `LanceIndex` with `fulltext_search` and `hybrid_search` methods |
| `agentic_rec/service.py` | BentoML `Service` wrapping the agent; `/recommend` POST endpoint |
| `agentic_rec/params.py` | Add `LLM_MODEL` constant (default `"openai:gpt-4o"`); `NLI_THRESHOLD` (default `0.5`) |

`data.py` is unchanged. `params.py` gains two new constants.

---

## LLM Configuration

The agent is constructed with `model=settings.llm_model` where `settings` is a `pydantic-settings` `Settings` object reading from environment variables. Default: `"openai:gpt-4o"`. Any pydantic-ai–supported model string works (e.g. `"anthropic:claude-haiku-4-5"`, `"ollama:llama3"`).

`pydantic-settings` must be added to `pyproject.toml` dependencies.

```bash
export LLM_MODEL="openai:gpt-4o"
export OPENAI_API_KEY="sk-..."
```

---

## BentoML Service

`agentic_rec/service.py` defines a `@bentoml.service` with:
- `POST /recommend` accepting `RecommendRequest`, returning `RecommendResponse`
- Dependencies: `LanceIndex` (loaded once on startup), `Settings` (from env)

Run with:
```bash
uv run bentoml serve agentic_rec.service:RecommenderService
```

---

## Index Preparation

Before serving, items must be embedded and indexed into LanceDB:

```bash
uv run index   # new CLI entry point in pyproject.toml
```

This loads `data/ml-1m/items.parquet`, encodes `item_text` with sentence-transformers (`all-MiniLM-L6-v2`), and calls `LanceIndex.index_data()`.

---

## Key Design Decisions

- **Single agent, five tools**: simpler than four separate agents; the LLM orchestrates tool order naturally from its system prompt. Retrieval tools can be called multiple times if the agent wants to try different queries.
- **Retrieval is agentic**: unlike vanilla RAG (fixed retrieval heuristic), the agent chooses semantic / FTS / hybrid based on what the user understanding step reveals.
- **No evaluation in v1**: NDCG@5 / Hit@5 measurement deferred; pipeline first.
- **MovieLens 1M**: item context is title + genres (no reviews). User context is rating history. Less rich than Amazon Reviews but the existing data pipeline is reused without changes.
- **NLI threshold `θ`**: configurable via `NLI_THRESHOLD` env var; defaults to 0.5. Items below threshold are excluded from context summary but may still appear in the candidate list passed to the ranker.
