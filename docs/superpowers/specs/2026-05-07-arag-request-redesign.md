# ARAG Request Redesign: User Text + History

**Date:** 2026-05-07
**Modifies:** `agentic_rec/agent.py`, `agentic_rec/service.py`, ARAG design spec

---

## Overview

Restructure the `RecommendRequest` to accept `user_text` (demographics/preferences) and `history`
(past interactions, may be empty) as top-level fields. The agent uses `user_text` as both LLM context
and a retrieval signal. This enables cold-start recommendations when history is empty.

---

## Structured Types

```python
class Interaction(pydantic.BaseModel):
    item_id: str
    event_timestamp: datetime
    event_name: str
    event_value: float

class RecommendRequest(pydantic.BaseModel):
    user_text: str                  # user demographics / stated preferences
    history: list[Interaction] = [] # past interactions, may be empty
    top_k: int = 10

class RankedItem(pydantic.BaseModel):
    item_id: str
    item_text: str
    explanation: str

class RecommendResponse(pydantic.BaseModel):
    items: list[RankedItem]
```

### Changes from current design

| Field (old)    | Field (new) | Notes                         |
|----------------|-------------|-------------------------------|
| `interactions` | `history`   | Renamed; now defaults to `[]` |
| —              | `user_text` | New required field            |
| `top_k`        | `top_k`     | Unchanged                     |

---

## Agent Flow

### Normal case (history is non-empty)

```text
Request (user_text, history, top_k)
    │
    ├─ [Tool 1] get_ids(item_ids from history)
    │           → {item_id: item_text} for interacted items
    │
    ├─ LLM: context understanding
    │           Inputs: user_text + item texts + interaction metadata
    │           Produces preference summary weighing both stated preferences
    │           and revealed behavior (recent high-value interactions).
    │
    ├─ [Tool 2] search(query, exclude_ids) × 2-4 calls
    │           Query strategies:
    │             • item_text of a recent high-value interaction
    │             • hypothetical item text from preference summary
    │             • user_text-derived query (e.g., stated genre preferences)
    │           All interacted item_ids excluded.
    │
    └─ LLM: rank + explain → RecommendResponse
```

### Cold-start case (history is empty)

```text
Request (user_text, history=[], top_k)
    │
    ├─ Tool 1 skipped (no items to look up)
    │
    ├─ LLM: context understanding
    │           Inputs: user_text only
    │           Produces preference summary from stated demographics/preferences.
    │
    ├─ [Tool 2] search(query) × 1-2 calls
    │           Query strategies:
    │             • user_text directly (or key phrases extracted from it)
    │             • hypothetical item text generated from user_text
    │           No exclude_ids needed.
    │
    └─ LLM: rank + explain → RecommendResponse
```

---

## System Prompt Changes

The agent's system prompt includes `user_text` as part of the user message context. The prompt
instructs the agent to:

1. Always consider `user_text` when building the preference summary.
2. Use `user_text` as one of the retrieval query strategies (alongside history-derived queries).
3. When history is empty, rely entirely on `user_text` for retrieval signals — skip `get_ids`.
4. Weight revealed behavior (history) over stated preferences when they conflict.

---

## BentoML Endpoint

`POST /recommend` request body changes:

```json
{
  "user_text": "25-year-old male, software engineer, enjoys sci-fi and thriller films",
  "history": [
    {"item_id": "1193", "event_timestamp": "2000-12-31T22:12:40", "event_name": "rating", "event_value": 5},
    {"item_id": "661",  "event_timestamp": "2000-12-31T22:35:09", "event_name": "rating", "event_value": 3}
  ],
  "top_k": 10
}
```

Cold-start example:

```json
{
  "user_text": "35-year-old female, teacher, loves romantic comedies and drama",
  "history": [],
  "top_k": 10
}
```

---

## Files to Modify

| File                                               | Change                                                         |
|----------------------------------------------------|----------------------------------------------------------------|
| `agentic_rec/agent.py`                             | New request/response models, updated system prompt, flow logic |
| `agentic_rec/service.py`                           | Update endpoint to use new `RecommendRequest`                  |
| `docs/superpowers/specs/2026-04-28-arag-design.md` | Update structured types and architecture sections              |
| `README.md`                                        | Update request examples                                        |
| `CLAUDE.md`                                        | Update architecture description if needed                      |
| Tests                                              | Add cold-start and normal-case test scenarios                  |

---

## Key Design Decisions

- **`user_text` as retrieval signal**: the agent can use stated preferences (e.g., "prefers sci-fi")
  directly as search queries, not just as LLM reasoning context. This allows preference-based
  retrieval even when history is sparse.
- **Cold-start via `user_text` only**: no dependency on a users index or collaborative filtering.
  The agent generates retrieval queries from demographics/preferences alone.
- **History defaults to empty list**: makes the field optional in the request body, enabling
  cold-start without a separate endpoint.
- **Revealed > stated**: when history contradicts stated preferences, the agent should weight
  actual behavior more heavily. This is enforced via the system prompt.
