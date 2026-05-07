# ARAG: Agentic Retrieval Augmented Generation for MovieLens

**Date:** 2026-04-28
**Paper:**
[ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation](https://arxiv.org/abs/2506.21931)
**Stack:** pydantic-ai, lancedb, sentence-transformers, datasets, BentoML, Polars

---

## Overview

Implementation of the ARAG framework on MovieLens 1M. A single `pydantic-ai` `Agent` orchestrates
recommendation via two tools: item text lookup (by ID) and candidate retrieval (hybrid search). The agent
derives a preference summary from the user's interaction history, issues multiple targeted retrieval queries
for diversity, then ranks candidates with per-item explanations.

---

## Architecture

```text
Request (text, history: list[{item_id, event_datetime, event_name, event_value}], top_k)
    │
    ├─ [Tool 1] fetch_item_texts(item_ids)  ← skipped if history is empty
    │           Looks up item_text in LanceDB by item_id.
    │           Returns {item_id: item_text} for all interacted items.
    │
    ├─ Agent generates context understanding (no tool call):
    │           Using text, interaction details (timestamp, event_name,
    │           event_value), and retrieved item texts, the agent produces a
    │           natural-language preference summary. Weights revealed behavior
    │           over stated preferences when they conflict.
    │
    ├─ [Tool 2] retrieve_candidates(query, top_k, exclude_item_ids)  ← called N times
    │           Hybrid search (vector + FTS) on LanceDB items table.
    │           Interacted item_ids are always excluded.
    │           Agent issues multiple queries for diversity, e.g.:
    │             • item_text of a recent highly-rated interaction
    │             • a hypothetical item text generated from context understanding
    │             • text-derived query (stated genre/style preferences)
    │           → List[ItemCandidate] (deduplicated across calls)
    │
    └─ Agent ranks and explains (no tool call):
                Ranks deduplicated candidates by predicted relevance.
                Promotes diversity so similar items are spread across the list.
                For each item, produces a short explanation (linked to a past
                interaction, a detected preference, etc.).
                Returns RecommendResponse.

BentoML /recommend → RecommendResponse(items: list[RankedItem])
```

---

## Data Flow

### Input

The caller supplies `text` (demographics/preferences) and an optional `history` of past interactions:

**`text`** (str, required): natural-language description of user demographics and stated preferences
(e.g. "25-year-old male, software engineer, enjoys sci-fi and thriller films").

**`history`** (list, defaults to `[]`): past interactions. Each interaction has:

| Field            | Type       | Description                               |
|------------------|------------|-------------------------------------------|
| `item_id`        | `str`      | Interacted item                           |
| `event_datetime` | `datetime` | When the interaction occurred             |
| `event_name`     | `str`      | Interaction type (e.g. `"rating"`)        |
| `event_value`    | `float`    | Strength of interaction (e.g. 1–5 rating) |

When history is non-empty, interactions are sorted by `event_datetime` descending so recency is preserved.
When history is empty (cold-start), the agent uses `text` as the sole signal for retrieval and ranking.

### Context understanding (between Tool 1 and Tool 2)

No tool call — the agent LLM reasons over the fetched item texts and interaction metadata to produce a
preference summary string. The summary should:

- Weight recent interactions more heavily than older ones.
- Distinguish between strong signals (high rating, explicit like) and weak signals.
- Capture genre/style preferences, not just individual titles.

### Item candidates (Tool 2)

LanceDB `items` table rows: `id`, `text`, `vector`. Hybrid search combines vector similarity and BM25/FTS,
reranked via `answerdotai/rerankers`.

The agent may call `retrieve_candidates` multiple times:

| Query strategy                                              | When to use                     |
|-------------------------------------------------------------|---------------------------------|
| Item text of a recent, high-value interaction               | Fast, anchored signal           |
| Generated hypothetical item text from context understanding | When history is sparse or stale |

The agent should limit `top_k` per call (e.g. 20) and issue 2–4 calls to ensure diversity. Candidates are
deduplicated by `item_id` across calls, keeping the highest retrieval score.

All interacted `item_id`s are passed as `exclude_item_ids` on every call.

### Ranking and explanation

No tool call — the agent LLM reasons over the deduplicated candidate list and context understanding to
produce the final ranked output. The ranking should:

- Prefer candidates that align with the preference summary.
- Penalise adjacent items that are too similar (genre/style diversity).
- Attach a one-sentence explanation per item.

### Structured types

```python
class Interaction(pydantic.BaseModel):
    item_id: str
    event_datetime: datetime
    event_name: str
    event_value: float

class ItemCandidate(pydantic.BaseModel):
    item_id: str
    item_text: str
    score: float = 0.0

class RankedItem(pydantic.BaseModel):
    item_id: str
    item_text: str
    explanation: str  # one sentence, references past interaction or preference

class RecommendRequest(pydantic.BaseModel):
    text: str
    history: list[Interaction] = []
    top_k: int = 10

class RecommendResponse(pydantic.BaseModel):
    items: list[RankedItem]
```

---

## Modules

| File                     | Responsibility                                                            |
|--------------------------|---------------------------------------------------------------------------|
| `agentic_rec/agent.py`   | `Agent` definition, system prompt, two tools, pydantic result types       |
| `agentic_rec/index.py`   | `LanceIndex` with `get_ids` (for Tool 1) and `search` hybrid (for Tool 2) |
| `agentic_rec/service.py` | BentoML `Service` wrapping the agent; `/recommend` POST endpoint          |
| `agentic_rec/params.py`  | `LLM_MODEL` constant (default `"openai:gpt-4o"`)                          |

`data.py` is unchanged. `NLI_THRESHOLD` is no longer needed.

---

## LLM Configuration

The agent is constructed with `model=settings.llm_model` where `settings` is a `pydantic-settings` `Settings`
object reading from environment variables. Default: `"openai:gpt-4o"`. Any pydantic-ai–supported model string
works (e.g. `"anthropic:claude-haiku-4-5"`, `"ollama:llama3"`).

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
uv run index   # CLI entry point in pyproject.toml
```

This loads `data/ml-1m/items.parquet`, encodes `item_text` with sentence-transformers (`lightonai/DenseOn`),
and calls `LanceIndex.index_data()`.

---

## Key Design Decisions

- **Two tools, not five**: removes NLI scoring and context summarisation as separate tool calls. Context
  understanding and ranking are agent reasoning steps, not tool calls, which reduces latency and round-trips.
- **Input is text + history, not user_id**: caller supplies demographics/preferences and interaction
  history directly; no server-side user profile store is required. Supports cold-start (empty history).
- **Multi-query retrieval for diversity**: agent issues 2–4 hybrid search calls with different query strategies
  rather than a single large retrieval. Deduplication happens client-side.
- **Hypothetical item text**: when recent history alone is a poor query signal, the agent generates a synthetic
  item description matching the schema of real items, then uses it as the search query.
- **Per-item explanations in output**: `RankedItem.explanation` makes recommendations interpretable and
  testable without a separate evaluation pipeline.
- **No evaluation in v1**: NDCG@5 / Hit@5 measurement deferred; pipeline first.
- **MovieLens 1M**: item context is title + genres (no reviews). Less rich than Amazon Reviews but the existing
  data pipeline is reused without changes.
