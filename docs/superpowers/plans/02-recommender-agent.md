# Recommender Agent Implementation Plan

**Goal:** Build a single pydantic-ai agent that accepts `text` + `history` and orchestrates
item text lookup and hybrid search to produce ranked recommendations with explanations.

**Architecture:** A module-level `pydantic_ai.Agent[AgentDeps, RecommendResponse]` singleton
with two tools (`get_item_texts`, `search_items`). The agent receives per-request context via
dynamic instructions (serialized request JSON) and domain-specific runtime instructions
(user-based vs item-based). Cold-start is handled naturally by skipping history lookup.

**Tech Stack:** pydantic-ai, pydantic, LanceDB (via `LanceIndex`), loguru.

---

## File Map

| File                    | Change                                                        |
|-------------------------|---------------------------------------------------------------|
| `agentic_rec/models.py` | Request/response models (Interaction, RecommendRequest, etc.) |
| `agentic_rec/agent.py`  | Agent singleton, tools, system prompt, instructions           |
| `tests/test_agent.py`   | Unit tests for models and tool logic                          |

---

## Tasks

### 1. Define structured types in `models.py`

- **`Interaction`**: `item_id`, `event_datetime`, `event_name`, `event_value`.
- **`ItemCandidate`**: `id`, `text`, `score` (default 0.0). Returned by search tool.
- **`RankedItem`**: `id`, `text`, `explanation`. Final output item.
- **`RecommendRequest`**: `text` (required), `history` (list of Interaction, default `[]`),
  `limit` (default 10).
- **`RecommendResponse`**: `items` (list of RankedItem). Used as `output_type`.

### 2. Define `AgentDeps`

Dataclass holding `index: LanceIndex` and `request: RecommendRequest`. Passed to
`agent.run(deps=...)` so tools can access the index and request context.

### 3. Write static system prompt

Domain-agnostic recommendation workflow covering:

1. Context understanding (fetch item texts if history non-empty).
2. Candidate retrieval (call search 2-4 times with diverse queries).
3. Cold-start handling (skip item lookup, use text as sole signal).
4. Ranking with explanations (select top items, attach one-sentence reasons).

Keep generic — no mention of movies, genres, or specific domains.

### 4. Write runtime instructions

Two instruction sets passed at call site via `agent.run(instructions=...)`:

- **`USER_INSTRUCTIONS`**: domain context for user-based recommendations. Tells the agent
  that items are films, text contains demographics, history reveals taste.
- **`ITEM_INSTRUCTIONS`**: domain context for item-based (similar items). Tells the agent
  that text is the source item, there's no history, and it should find related items.

This separation allows the same agent to serve both use cases without system prompt changes.

### 5. Create module-level Agent singleton

`pydantic_ai.Agent(model=settings.llm_model, system_prompt=SYSTEM_PROMPT,
output_type=RecommendResponse, defer_model_check=True)`.

Module-level so it's created once. `defer_model_check=True` avoids import-time API calls.

### 6. Add dynamic instructions via `@agent.instructions`

Decorator function that serializes `ctx.deps.request` as JSON. This gives the LLM the
full request context (text, history, limit) without embedding it in the system prompt.

### 7. Implement `get_item_texts` tool

Delegates to `ctx.deps.index.get_ids(item_ids)`. Returns `dict[str, str]` mapping
item ID to text. The agent calls this to understand what items the user interacted with.

### 8. Implement `search_items` tool

Delegates to `ctx.deps.index.search(query, exclude_ids, limit)`. Returns
`list[ItemCandidate]` via a `TypeAdapter` for validation. The agent calls this 2-4 times
with varied queries for diversity.

### 9. Add `check_llm()` helper

Async function that creates a throwaway agent and sends a trivial prompt. Returns `True`
if the LLM responds, `False` on any error. Used at app startup to verify API connectivity.

### 10. Add CLI entry point

Define `main()` function as a sanity check: load users from parquet, sample a random user,
build a `RecommendRequest` from their data (truncate history to 20), run agent, print results.

Register in `pyproject.toml` under `[project.scripts]`:
`agent = "agentic_rec.agent:main"`. Uses `jsonargparse.auto_cli(main)` for CLI arg parsing.

### 11. Write tests

- **Model tests**: field defaults, required fields, validation.
- **Tool unit tests**: mock the index, verify correct delegation and return types.
- **Agent creation tests**: verify tools are registered, output type is set.

---

## Conventions

- **`@logger.catch(reraise=True)`** on tool functions for structured error logging.
- **Deferred imports** inside `main()` for heavy packages. Suppress ruff `PLC0415`.

---

## Design Notes

- **Single agent, not four**: the paper's 4-agent pipeline is collapsed into one LLM call.
  The system prompt guides the agent through understanding → retrieval → ranking stages
  sequentially within a single conversation turn.
- **Structured output**: `output_type=RecommendResponse` ensures the LLM returns valid
  JSON matching the schema. Pydantic-ai handles retries on schema violations.
- **Tool-based retrieval**: the agent decides when and how many times to search, what
  queries to use, and what IDs to exclude. This gives it agency over the retrieval strategy.
- **Cold-start**: naturally handled — when history is empty, the agent skips `get_item_texts`
  and uses `text` alone for search queries. No special-case logic needed.
- **Invocation pattern**: callers use `agent.run(instructions=..., deps=...)` directly.
  No factory function or agent-per-request.
