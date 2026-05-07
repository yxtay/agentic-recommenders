from __future__ import annotations

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rec.agent import RankedItem, RecommendResponse
from agentic_rec.service import RecommenderService


@pytest.fixture
def mock_index() -> MagicMock:
    """Return a mock LanceIndex."""
    return MagicMock()


@pytest.fixture
def service(mock_index: MagicMock) -> RecommenderService:
    """Return a RecommenderService with a patched index."""
    with patch("agentic_rec.service.LanceIndex.load", return_value=mock_index):
        return RecommenderService()


def test_service_has_recommend_api() -> None:
    """RecommenderService exposes a /recommend API endpoint."""
    assert "recommend" in RecommenderService.apis


def test_service_init_loads_index(mock_index: MagicMock) -> None:
    """RecommenderService.__init__ calls LanceIndex.load with a LanceIndexConfig."""
    from agentic_rec.index import LanceIndexConfig

    with patch(
        "agentic_rec.service.LanceIndex.load", return_value=mock_index
    ) as mock_load:
        svc = RecommenderService()

    mock_load.assert_called_once()
    config_arg = mock_load.call_args[0][0]
    assert isinstance(config_arg, LanceIndexConfig)
    assert svc.index is mock_index


@pytest.mark.asyncio
async def test_recommend_delegates_to_agent(
    service: RecommenderService, mock_index: MagicMock
) -> None:
    """recommend() delegates to agent.recommend with a RecommendRequest."""
    from agentic_rec.agent import RecommendRequest

    expected_response = RecommendResponse(
        items=[RankedItem(item_id="1", item_text="Movie (1999)", explanation="Great")]
    )

    with patch(
        "agentic_rec.service.recommend",
        new_callable=AsyncMock,
        return_value=expected_response,
    ) as mock_recommend:
        response = await service.recommend(user_text="sci-fi fan", top_k=5)

    mock_recommend.assert_called_once()
    call_args = mock_recommend.call_args
    request: RecommendRequest = call_args[0][0]
    assert request.user_text == "sci-fi fan"
    assert request.top_k == 5  # noqa: PLR2004
    assert request.history == []
    assert call_args[0][1] is mock_index
    assert response is expected_response


@pytest.mark.asyncio
async def test_recommend_passes_history(
    service: RecommenderService,
) -> None:
    """recommend() forwards non-empty history to the agent."""
    from datetime import datetime

    from agentic_rec.agent import Interaction, RecommendRequest

    history = [
        Interaction(
            item_id="42",
            event_timestamp=datetime(2000, 1, 1, tzinfo=UTC),
            event_name="rating",
            event_value=5.0,
        )
    ]
    expected_response = RecommendResponse(items=[])

    with patch(
        "agentic_rec.service.recommend",
        new_callable=AsyncMock,
        return_value=expected_response,
    ) as mock_recommend:
        await service.recommend(user_text="drama lover", history=history, top_k=3)

    request: RecommendRequest = mock_recommend.call_args[0][0]
    assert len(request.history) == 1
    assert request.history[0].item_id == "42"


@pytest.mark.asyncio
async def test_recommend_none_history_becomes_empty(
    service: RecommenderService,
) -> None:
    """recommend() converts None history to an empty list."""
    from agentic_rec.agent import RecommendRequest

    expected_response = RecommendResponse(items=[])

    with patch(
        "agentic_rec.service.recommend",
        new_callable=AsyncMock,
        return_value=expected_response,
    ) as mock_recommend:
        await service.recommend(user_text="any user", history=None)

    request: RecommendRequest = mock_recommend.call_args[0][0]
    assert request.history == []
