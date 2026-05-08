from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import datasets
import pytest

from agentic_rec.agent import agent
from agentic_rec.models import (
    AgentDeps,
    ItemCandidate,
    RecommendRequest,
    RecommendResponse,
)


@pytest.fixture
def mock_index() -> MagicMock:
    index = MagicMock()
    index.get_ids.return_value = datasets.Dataset.from_dict(
        {"id": ["1", "2"], "text": ["Movie A (1999)", "Movie B (2000)"]}
    )
    index.search.return_value = datasets.Dataset.from_dict(
        {
            "id": ["3", "4"],
            "text": ["Action Movie (2001)", "Drama Film (2002)"],
            "score": [0.9, 0.8],
        }
    )
    return index


@pytest.fixture
def sample_request() -> RecommendRequest:
    return RecommendRequest(
        text="25-year-old male, enjoys sci-fi and thriller films",
        history=[
            {
                "item_id": "1",
                "event_datetime": datetime(2024, 1, 1, tzinfo=UTC),
                "event_name": "rating",
                "event_value": 5.0,
            },
            {
                "item_id": "2",
                "event_datetime": datetime(2024, 1, 2, tzinfo=UTC),
                "event_name": "rating",
                "event_value": 3.0,
            },
        ],
        top_k=5,
    )


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
    def test_delegates_to_index(
        self, mock_index: MagicMock, sample_request: RecommendRequest
    ) -> None:
        toolset = agent.toolsets[0]
        tool = toolset.tools["get_item_texts"]
        ctx = MagicMock()
        ctx.deps = AgentDeps(index=mock_index, request=sample_request)

        result = tool.function(ctx, item_ids=["1", "2"])
        mock_index.get_ids.assert_called_once_with(["1", "2"])
        assert result == {"1": "Movie A (1999)", "2": "Movie B (2000)"}


class TestSearchItemsTool:
    def test_delegates_to_index(
        self, mock_index: MagicMock, sample_request: RecommendRequest
    ) -> None:
        toolset = agent.toolsets[0]
        tool = toolset.tools["search_items"]
        ctx = MagicMock()
        ctx.deps = AgentDeps(index=mock_index, request=sample_request)

        result = tool.function(ctx, query="sci-fi action", exclude_ids=["1"], top_k=10)
        mock_index.search.assert_called_once_with(
            "sci-fi action", exclude_ids=["1"], top_k=10
        )
        assert len(result) == 2  # noqa: PLR2004
        assert all(isinstance(item, ItemCandidate) for item in result)
        assert result[0].item_id == "3"
        assert result[0].score == pytest.approx(0.9)
