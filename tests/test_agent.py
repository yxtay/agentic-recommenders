from __future__ import annotations

from unittest.mock import MagicMock

import datasets
import pytest

from agentic_rec.agent import ItemCandidate, RecommendResponse, create_agent


@pytest.fixture
def mock_index() -> MagicMock:
    """Return a mock LanceIndex."""
    index = MagicMock()
    index.get_ids.return_value = datasets.Dataset.from_dict(
        {"id": ["1"], "text": ["Movie title (1999)"]}
    )
    index.search.return_value = datasets.Dataset.from_dict(
        {
            "id": ["2", "3"],
            "text": ["Action movie (2000)", "Drama film (2001)"],
            "score": [0.9, 0.8],
        }
    )
    return index


def test_create_agent_returns_agent(mock_index: MagicMock) -> None:
    """create_agent returns a pydantic_ai.Agent instance."""
    import pydantic_ai

    agent = create_agent(mock_index)
    assert isinstance(agent, pydantic_ai.Agent)


def test_create_agent_has_correct_output_type(mock_index: MagicMock) -> None:
    """create_agent produces an agent with RecommendResponse as output type."""
    agent = create_agent(mock_index)
    assert agent._output_type is RecommendResponse  # noqa: SLF001


def test_create_agent_registers_two_tools(mock_index: MagicMock) -> None:
    """create_agent registers exactly the get_item_texts and search_items tools."""
    agent = create_agent(mock_index)
    # Gather all tool names from all toolsets
    tool_names: set[str] = set()
    for toolset in agent.toolsets:
        if hasattr(toolset, "tools"):
            tool_names.update(toolset.tools.keys())
    assert "get_item_texts" in tool_names
    assert "search_items" in tool_names


def test_create_agent_top_k_in_system_prompt(mock_index: MagicMock) -> None:
    """create_agent embeds the top_k value into the system prompt."""
    agent = create_agent(mock_index, top_k=5)
    assert any("5" in p for p in agent._system_prompts)  # noqa: SLF001


def test_get_item_texts_tool_calls_index(mock_index: MagicMock) -> None:
    """The get_item_texts tool delegates to index.get_ids and maps id->text."""
    agent = create_agent(mock_index)
    # Retrieve the registered tool function directly
    tool_names: dict[str, object] = {}
    for toolset in agent.toolsets:
        if hasattr(toolset, "tools"):
            tool_names.update(toolset.tools)

    tool = tool_names["get_item_texts"]
    ctx = MagicMock()
    result = tool.function(ctx, item_ids=["1"])
    mock_index.get_ids.assert_called_once_with(["1"])
    assert result == {"1": "Movie title (1999)"}


def test_search_items_tool_calls_index(mock_index: MagicMock) -> None:
    """The search_items tool delegates to index.search and returns ItemCandidate list."""
    agent = create_agent(mock_index)
    tool_names: dict[str, object] = {}
    for toolset in agent.toolsets:
        if hasattr(toolset, "tools"):
            tool_names.update(toolset.tools)

    tool = tool_names["search_items"]
    ctx = MagicMock()
    result = tool.function(ctx, query="action movie", exclude_ids=["99"], top_k=5)
    mock_index.search.assert_called_once_with(
        "action movie", exclude_ids=["99"], top_k=5
    )
    assert len(result) == len(mock_index.search.return_value)
    assert all(isinstance(item, ItemCandidate) for item in result)
    assert result[0].item_id == "2"
    assert result[0].score == pytest.approx(0.9)
