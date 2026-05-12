from __future__ import annotations

import datetime
from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from agentic_rec.agent import AgentDeps, agent
from agentic_rec.models import (
    ItemCandidate,
    RecommendRequest,
    RecommendResponse,
)


@pytest.fixture
def mock_item_repo() -> MagicMock:
    repo = MagicMock()
    repo.get_by_ids.return_value = pa.table(
        {"id": ["1", "2"], "text": ["Movie A (1999)", "Movie B (2000)"]}
    )
    repo.search.return_value = pa.table(
        {
            "id": ["3", "4"],
            "text": ["Action Movie (2001)", "Drama Film (2002)"],
            "score": [0.9, 0.8],
        }
    )
    return repo


@pytest.fixture
def sample_request() -> RecommendRequest:
    request = {
        "text": "25-year-old male, enjoys sci-fi and thriller films",
        "history": [
            {
                "item_id": "1",
                "event_datetime": datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
                "event_name": "rating",
                "event_value": 5.0,
            },
            {
                "item_id": "2",
                "event_datetime": datetime.datetime(2024, 1, 2, tzinfo=datetime.UTC),
                "event_name": "rating",
                "event_value": 3.0,
            },
        ],
        "limit": 5,
    }
    return RecommendRequest.model_validate(request)


class TestAgentStructure:
    def test_agent_has_tools(self) -> None:
        tool_names: set[str] = set()
        for toolset in agent.toolsets:
            if hasattr(toolset, "tools"):
                tool_names.update(toolset.tools.keys())
        assert "get_item_texts" in tool_names
        assert "search_items" in tool_names

    def test_agent_output_type(self) -> None:
        assert agent._output_type is RecommendResponse  # noqa: SLF001


class TestGetItemTextsTool:
    def test_delegates_to_repository(
        self, mock_item_repo: MagicMock, sample_request: RecommendRequest
    ) -> None:
        toolset = agent.toolsets[0]
        tool = toolset.tools["get_item_texts"]
        ctx = MagicMock()
        ctx.deps = AgentDeps(item_repository=mock_item_repo, request=sample_request)

        result = tool.function(ctx, item_ids=["1", "2"])
        mock_item_repo.get_by_ids.assert_called_once_with(["1", "2"])
        assert result == {"1": "Movie A (1999)", "2": "Movie B (2000)"}


class TestSearchItemsTool:
    def test_delegates_to_repository(
        self, mock_item_repo: MagicMock, sample_request: RecommendRequest
    ) -> None:
        toolset = agent.toolsets[0]
        tool = toolset.tools["search_items"]
        ctx = MagicMock()
        ctx.deps = AgentDeps(item_repository=mock_item_repo, request=sample_request)

        result = tool.function(ctx, query="sci-fi action", exclude_ids=["1"], limit=10)
        mock_item_repo.search.assert_called_once_with(
            "sci-fi action", query_type="hybrid", exclude_ids=["1"], limit=10
        )
        assert len(result) == 2
        assert all(isinstance(item, ItemCandidate) for item in result)
        assert result[0].id == "3"
        assert result[0].score == pytest.approx(0.9)
