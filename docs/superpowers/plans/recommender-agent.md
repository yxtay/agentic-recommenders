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

1. Define structured types in `models.py`: `Interaction`, `ItemCandidate`, `RankedItem`,
    `RecommendRequest`, `RecommendResponse`
2. Define `AgentDeps` dataclass holding `index` and `request`
3. Write static `SYSTEM_PROMPT` — domain-agnostic recommendation workflow
4. Write `USER_INSTRUCTIONS` and `ITEM_INSTRUCTIONS` for the two recommendation modes
5. Create module-level `Agent` singleton with `output_type=RecommendResponse`
6. Add `@agent.instructions` dynamic prompt (serializes request as JSON)
7. Implement `get_item_texts` tool (delegates to `index.get_ids()`)
8. Implement `search_items` tool (delegates to `index.search()`)
9. Add `check_llm()` helper for startup verification
10. Write tests for models and tool behavior
