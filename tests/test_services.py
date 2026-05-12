from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_rec.models import (
    ItemRecommended,
    ItemResponse,
    RecommendRequest,
    RecommendResponse,
    UserResponse,
)
from agentic_rec.services.item_service import ItemService
from agentic_rec.services.recommendation_service import RecommendationService
from agentic_rec.services.user_service import UserService


@pytest.mark.asyncio
async def test_recommendation_service_recommend() -> None:
    mock_item_repo = MagicMock()
    mock_user_repo = MagicMock()
    mock_agent = MagicMock()

    expected_response = RecommendResponse(
        items=[ItemRecommended(id="1", text="Movie", explanation="desc")]
    )
    mock_agent.run = AsyncMock(return_value=MagicMock(output=expected_response))

    service = RecommendationService(mock_item_repo, mock_user_repo, mock_agent)
    request = RecommendRequest(text="test", limit=5)

    response = await service.recommend(request)
    assert response == expected_response
    mock_agent.run.assert_called_once()


def test_user_service_get_user() -> None:
    mock_repo = MagicMock()
    expected_user = UserResponse(id="1", text="User 1")
    mock_repo.get_by_id.return_value = expected_user

    service = UserService(mock_repo)
    user = service.get_user("1")
    assert user == expected_user
    mock_repo.get_by_id.assert_called_once_with("1")


def test_item_service_get_item() -> None:
    mock_repo = MagicMock()
    expected_item = ItemResponse(id="1", text="Item 1")
    mock_repo.get_by_id.return_value = expected_item

    service = ItemService(mock_repo)
    item = service.get_item("1")
    assert item == expected_item
    mock_repo.get_by_id.assert_called_once_with("1")
