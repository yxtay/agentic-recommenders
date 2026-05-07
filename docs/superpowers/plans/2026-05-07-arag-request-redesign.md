# ARAG Request Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure ARAG to accept `user_text` + `history` request fields and support cold-start recommendations.

**Architecture:** A pydantic-ai agent receives `user_text` (demographics/preferences)
and `history` (past interactions). When history is present, the agent fetches item texts
and uses both signals for retrieval. When history is empty (cold-start), the agent uses
`user_text` alone for retrieval queries. Served via BentoML.

**Tech Stack:** pydantic-ai, pydantic, lancedb, BentoML, datasets, pytest

---

## File Structure

| File                              | Responsibility                                                       |
|-----------------------------------|----------------------------------------------------------------------|
| `agentic_rec/agent.py` (create)   | Request/response models, pydantic-ai Agent, two tools, system prompt |
| `agentic_rec/service.py` (create) | BentoML service with `/recommend` POST endpoint                      |
| `agentic_rec/params.py` (modify)  | Add `LLM_MODEL` constant                                             |
| `tests/test_agent.py` (create)    | Unit tests for agent models and tool logic                           |
| `tests/test_service.py` (create)  | Integration tests for BentoML endpoint                               |
| `README.md` (modify)              | Update request examples                                              |
| `CLAUDE.md` (modify)              | Update architecture description                                      |

---

### Task 1: Add LLM_MODEL to params.py

**Files:**

- Modify: `agentic_rec/params.py`

- [ ] **Step 1: Add LLM_MODEL constant**

Add to the end of `agentic_rec/params.py`:

```python
# llm
LLM_MODEL = "openai:gpt-4o"
```

- [ ] **Step 2: Verify no import errors**

Run: `uv run python -c "from agentic_rec.params import LLM_MODEL; print(LLM_MODEL)"`
Expected: `openai:gpt-4o`

- [ ] **Step 3: Commit**

```bash
git add agentic_rec/params.py
git commit -m "feat: add LLM_MODEL constant to params"
```

---

### Task 2: Create agent.py with request/response models

**Files:**

- Create: `agentic_rec/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write failing tests for models**

Create `tests/test_agent.py`:

```python
from __future__ import annotations

from datetime import datetime

import pytest

from agentic_rec.agent import (
    Interaction,
    ItemCandidate,
    RankedItem,
    RecommendRequest,
    RecommendResponse,
)


class TestInteraction:
    def test_fields(self) -> None:
        i = Interaction(
            item_id="123",
            event_timestamp=datetime(2024, 1, 1, 12, 0, 0),
            event_name="rating",
            event_value=5.0,
        )
        assert i.item_id == "123"
        assert i.event_name == "rating"
        assert i.event_value == 5.0

    def test_timestamp_parsing(self) -> None:
        i = Interaction(
            item_id="1",
            event_timestamp="2024-01-01T12:00:00",
            event_name="rating",
            event_value=3.0,
        )
        assert i.event_timestamp == datetime(2024, 1, 1, 12, 0, 0)


class TestRecommendRequest:
    def test_required_user_text(self) -> None:
        with pytest.raises(Exception):
            RecommendRequest()

    def test_history_defaults_empty(self) -> None:
        req = RecommendRequest(user_text="25 year old male, likes sci-fi")
        assert req.history == []
        assert req.top_k == 10

    def test_with_history(self) -> None:
        req = RecommendRequest(
            user_text="test user",
            history=[
                Interaction(
                    item_id="1",
                    event_timestamp=datetime(2024, 1, 1),
                    event_name="rating",
                    event_value=5.0,
                )
            ],
            top_k=5,
        )
        assert len(req.history) == 1
        assert req.top_k == 5


class TestItemCandidate:
    def test_defaults(self) -> None:
        c = ItemCandidate(item_id="42", item_text="Some Movie (2024)")
        assert c.score == 0.0

    def test_with_score(self) -> None:
        c = ItemCandidate(item_id="42", item_text="Some Movie", score=0.95)
        assert c.score == 0.95


class TestRankedItem:
    def test_fields(self) -> None:
        item = RankedItem(
            item_id="42",
            item_text="Some Movie (2024)",
            explanation="Matches your preference for sci-fi",
        )
        assert item.item_id == "42"
        assert item.explanation == "Matches your preference for sci-fi"


class TestRecommendResponse:
    def test_items_list(self) -> None:
        resp = RecommendResponse(
            items=[
                RankedItem(
                    item_id="1", item_text="Movie A", explanation="reason A"
                ),
                RankedItem(
                    item_id="2", item_text="Movie B", explanation="reason B"
                ),
            ]
        )
        assert len(resp.items) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent.py -v`
Expected: FAIL (cannot import `agentic_rec.agent`)

- [ ] **Step 3: Write the models in agent.py**

Create `agentic_rec/agent.py`:

```python
from __future__ import annotations

from datetime import datetime

import pydantic


class Interaction(pydantic.BaseModel):
    item_id: str
    event_timestamp: datetime
    event_name: str
    event_value: float


class ItemCandidate(pydantic.BaseModel):
    item_id: str
    item_text: str
    score: float = 0.0


class RankedItem(pydantic.BaseModel):
    item_id: str
    item_text: str
    explanation: str


class RecommendRequest(pydantic.BaseModel):
    user_text: str
    history: list[Interaction] = []
    top_k: int = 10


class RecommendResponse(pydantic.BaseModel):
    items: list[RankedItem]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add agentic_rec/agent.py tests/test_agent.py
git commit -m "feat: add request/response models for ARAG agent"
```

---

### Task 3: Add pydantic-ai Agent with tools

**Files:**

- Modify: `agentic_rec/agent.py`
- Modify: `tests/test_agent.py`

- [ ] **Step 1: Write failing test for agent creation**

Append to `tests/test_agent.py`:

```python
from unittest.mock import MagicMock

from agentic_rec.agent import create_agent


class TestCreateAgent:
    def test_returns_agent(self) -> None:
        mock_index = MagicMock()
        agent = create_agent(mock_index)
        assert agent is not None

    def test_agent_has_tools(self) -> None:
        mock_index = MagicMock()
        agent = create_agent(mock_index)
        tool_names = [t.name for t in agent._function_tools.values()]
        assert "get_item_texts" in tool_names
        assert "search_items" in tool_names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent.py::TestCreateAgent -v`
Expected: FAIL (cannot import `create_agent`)

- [ ] **Step 3: Implement create_agent with tools**

Add to `agentic_rec/agent.py`:

```python
from typing import TYPE_CHECKING

import pydantic_ai

from agentic_rec.params import LLM_MODEL

if TYPE_CHECKING:
    import datasets

    from agentic_rec.index import LanceIndex

SYSTEM_PROMPT = """\
You are a movie recommendation agent. You receive:
- user_text: a description of the user's demographics and stated preferences
- history: their past interactions (may be empty for cold-start users)

Your task is to recommend {top_k} movies the user will enjoy.

## Workflow

1. If history is non-empty, call get_item_texts to fetch the text of interacted items.
2. Build a preference summary from user_text and (if available) the interaction history.
  Weight revealed behavior (history) over stated preferences when they conflict.
3. Call search_items 2-4 times with different queries for diversity:
  - Item text of a recent high-value interaction (if history exists)
  - A hypothetical item description based on the preference summary
  - A query derived from user_text (stated genre/style preferences)
  Always pass interacted item_ids as exclude_ids.
4. Rank deduplicated candidates by relevance and diversity.
5. Return exactly {top_k} RankedItem results with one-sentence explanations.

## Cold-start (empty history)

Skip step 1. Use user_text as the sole signal. Generate 1-2 search queries from
stated preferences and demographics.
"""


def create_agent(index: LanceIndex) -> pydantic_ai.Agent:
    agent = pydantic_ai.Agent(
        model=LLM_MODEL,
        system_prompt=SYSTEM_PROMPT,
        result_type=RecommendResponse,
    )

    @agent.tool
    async def get_item_texts(
        ctx: pydantic_ai.RunContext, item_ids: list[str]
    ) -> dict[str, str]:
        """Look up item texts by ID. Returns {item_id: item_text} mapping."""
        result: datasets.Dataset = index.get_ids(item_ids)
        return dict(zip(result["id"], result["text"]))

    @agent.tool
    async def search_items(
        ctx: pydantic_ai.RunContext,
        query: str,
        exclude_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> list[ItemCandidate]:
        """Hybrid search for candidate items. Returns scored candidates."""
        result: datasets.Dataset = index.search(
            query, exclude_ids=exclude_ids, top_k=top_k
        )
        return [
            ItemCandidate(item_id=row["id"], item_text=row["text"], score=row["score"])
            for row in result
        ]

    return agent
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_agent.py::TestCreateAgent -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add agentic_rec/agent.py tests/test_agent.py
git commit -m "feat: add pydantic-ai agent with get_item_texts and search_items tools"
```

---

### Task 4: Add the recommend entrypoint function

**Files:**

- Modify: `agentic_rec/agent.py`
- Modify: `tests/test_agent.py`

- [ ] **Step 1: Write failing test for recommend function**

Append to `tests/test_agent.py`:

```python
from unittest.mock import AsyncMock, patch

import pytest

from agentic_rec.agent import recommend


class TestRecommend:
    @pytest.mark.asyncio
    async def test_returns_response(self) -> None:
        mock_result = RecommendResponse(
            items=[
                RankedItem(
                    item_id="42",
                    item_text="Great Movie (2024)",
                    explanation="Matches your sci-fi preference",
                )
            ]
        )

        with patch("agentic_rec.agent.create_agent") as mock_create:
            mock_agent = AsyncMock()
            mock_agent.run.return_value = AsyncMock(data=mock_result)
            mock_create.return_value = mock_agent

            from agentic_rec.index import LanceIndex, LanceIndexConfig

            mock_index = MagicMock(spec=LanceIndex)

            request = RecommendRequest(
                user_text="likes sci-fi",
                history=[],
                top_k=1,
            )
            response = await recommend(request, mock_index)
            assert isinstance(response, RecommendResponse)
            assert len(response.items) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py::TestRecommend -v`
Expected: FAIL (cannot import `recommend`)

- [ ] **Step 3: Implement recommend function**

Add to `agentic_rec/agent.py`:

```python
async def recommend(request: RecommendRequest, index: LanceIndex) -> RecommendResponse:
    agent = create_agent(index)

    user_message_parts = [f"User profile: {request.user_text}"]

    if request.history:
        sorted_history = sorted(
            request.history, key=lambda i: i.event_timestamp, reverse=True
        )
        history_lines = [
            f"- {h.item_id} | {h.event_name}={h.event_value} | {h.event_timestamp.isoformat()}"
            for h in sorted_history
        ]
        user_message_parts.append("Interaction history:\n" + "\n".join(history_lines))
    else:
        user_message_parts.append("Interaction history: (none — cold-start user)")

    user_message_parts.append(f"Please recommend {request.top_k} items.")

    user_message = "\n\n".join(user_message_parts)

    result = await agent.run(user_message)
    return result.data
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent.py::TestRecommend -v`
Expected: PASS

- [ ] **Step 5: Lint check**

Run: `uv run ruff check agentic_rec/agent.py tests/test_agent.py`
Expected: No errors (or fix any that appear)

- [ ] **Step 6: Commit**

```bash
git add agentic_rec/agent.py tests/test_agent.py
git commit -m "feat: add recommend() entrypoint for agent"
```

---

### Task 5: Create BentoML service

**Files:**

- Create: `agentic_rec/service.py`
- Create: `tests/test_service.py`

- [ ] **Step 1: Write failing test for service**

Create `tests/test_service.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rec.agent import RankedItem, RecommendResponse


class TestRecommenderService:
    def test_service_importable(self) -> None:
        from agentic_rec.service import RecommenderService

        assert RecommenderService is not None

    @pytest.mark.asyncio
    async def test_recommend_endpoint(self) -> None:
        from agentic_rec.service import RecommenderService

        mock_response = RecommendResponse(
            items=[
                RankedItem(
                    item_id="1",
                    item_text="Test Movie",
                    explanation="Good match",
                )
            ]
        )

        with patch("agentic_rec.service.recommend", new_callable=AsyncMock) as mock_rec:
            mock_rec.return_value = mock_response
            service = RecommenderService()
            service.index = MagicMock()

            result = await service.recommend(
                user_text="likes action movies",
                history=[],
                top_k=1,
            )
            assert len(result.items) == 1
            assert result.items[0].item_id == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_service.py -v`
Expected: FAIL (cannot import `agentic_rec.service`)

- [ ] **Step 3: Implement service.py**

Create `agentic_rec/service.py`:

```python
from __future__ import annotations

import bentoml

from agentic_rec.agent import (
    Interaction,
    RecommendRequest,
    RecommendResponse,
    recommend,
)
from agentic_rec.index import LanceIndex, LanceIndexConfig


@bentoml.service
class RecommenderService:
    def __init__(self) -> None:
        self.index = LanceIndex.load(LanceIndexConfig())

    @bentoml.api
    async def recommend(
        self,
        user_text: str,
        history: list[Interaction] | None = None,
        top_k: int = 10,
    ) -> RecommendResponse:
        request = RecommendRequest(
            user_text=user_text,
            history=history or [],
            top_k=top_k,
        )
        return await recommend(request, self.index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_service.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add agentic_rec/service.py tests/test_service.py
git commit -m "feat: add BentoML service with /recommend endpoint"
```

---

### Task 6: Update README and CLAUDE.md

**Files:**

- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update README request example**

In `README.md`, replace the curl example and architecture diagram to reflect the new
`user_text` + `history` request format:

Architecture section:

```text
Request (user_text, history: [{item_id, event_timestamp, event_name, event_value}], top_k)
    │
    ├─ [Tool 1] get_ids(item_ids)           → {item_id: item_text}  (skipped if history empty)
    ├─ LLM: context understanding           → preference summary
    ├─ [Tool 2] search(query, exclude_ids)  → candidates  (called 2-4×)
    └─ LLM: rank + explain                 → RankedItem list

POST /recommend → [{ item_id, item_text, explanation }]
```

Curl example:

```bash
curl -X POST http://localhost:3000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_text": "25-year-old male, software engineer, enjoys sci-fi and thriller films",
    "history": [
      {"item_id": "1193", "event_timestamp": "2000-12-31T22:12:40", "event_name": "rating", "event_value": 5},
      {"item_id": "661",  "event_timestamp": "2000-12-31T22:35:09", "event_name": "rating", "event_value": 3}
    ],
    "top_k": 10
  }'
```

- [ ] **Step 2: Update CLAUDE.md architecture section**

Update the architecture description to mention `user_text` + `history` input format and cold-start support.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass

- [ ] **Step 4: Lint**

Run: `uv run ruff check --fix . && uv run ruff format .`
Expected: Clean

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: update README and CLAUDE.md for new request format"
```

---

### Task 7: Add pytest-asyncio dev dependency (prerequisite for Tasks 4-5)

**Files:**

- Modify: `pyproject.toml`

**IMPORTANT:** Execute this task before Task 4 if `pytest-asyncio` is not already installed.

- [ ] **Step 1: Check if pytest-asyncio is available**

Run: `uv run python -c "import pytest_asyncio; print(pytest_asyncio.__version__)"`
If it fails, proceed to step 2. If it succeeds, skip this task.

- [ ] **Step 2: Add pytest-asyncio to dev dependencies**

In `pyproject.toml`, update the dev dependency group:

```toml
[dependency-groups]
dev = ["huggingface-hub[cli]>=0.35", "pyarrow-stubs>=20.0", "pytest~=9.0", "pytest-asyncio~=1.0"]
```

- [ ] **Step 3: Sync dependencies**

Run: `uv sync`
Expected: Installs pytest-asyncio

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add pytest-asyncio dev dependency"
```
