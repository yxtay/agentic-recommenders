from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

if TYPE_CHECKING:
    from collections.abc import Iterator

import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from agentic_rec.dependencies import get_item_repository, get_user_repository
from agentic_rec.main import app
from agentic_rec.models import ItemRecommended, RecommendResponse


@pytest.fixture
def mock_index() -> MagicMock:
    index = MagicMock()
    index.get_ids.return_value = pa.table(
        {"id": ["42"], "text": ["The Matrix (1999) | Action, Sci-Fi"]}
    )
    return index


@pytest.fixture
def mock_users_index() -> MagicMock:
    users_index = MagicMock()
    users_index.get_ids.return_value = pa.table(
        {
            "id": ["1"],
            "text": ["25-year-old male, software engineer"],
            "history": [
                [
                    {
                        "item_id": "42",
                        "event_datetime": "2024-01-01T00:00:00",
                        "event_name": "rating",
                        "event_value": 5.0,
                    }
                ]
            ],
        }
    )
    return users_index


@pytest.fixture
def mock_agent_response() -> RecommendResponse:
    return RecommendResponse(
        items=[
            ItemRecommended(
                id="99",
                text="Inception (2010) | Sci-Fi, Thriller",
                explanation="Similar to The Matrix",
            )
        ]
    )


@pytest.fixture
def client(
    mock_index: MagicMock,
    mock_users_index: MagicMock,
    mock_agent_response: RecommendResponse,
) -> Iterator[TestClient]:
    from agentic_rec.dependencies import get_recommendation_service
    from agentic_rec.repositories.item_repository import ItemRepository
    from agentic_rec.repositories.user_repository import UserRepository
    from agentic_rec.services.recommendation_service import RecommendationService

    item_repo = MagicMock(spec=ItemRepository)
    item_repo.count_rows.return_value = 1

    def get_item_by_id(item_id):
        res = mock_index.get_ids([item_id])
        if res.num_rows == 0:
            return None
        from agentic_rec.models import ItemResponse

        return ItemResponse.model_validate(res.to_pylist()[0])

    item_repo.get_by_id.side_effect = get_item_by_id

    user_repo = MagicMock(spec=UserRepository)
    user_repo.count_rows.return_value = 1

    def get_user_by_id(user_id):
        res = mock_users_index.get_ids([user_id])
        if res.num_rows == 0:
            return None
        from agentic_rec.models import UserResponse

        return UserResponse.model_validate(res.to_pylist()[0])

    user_repo.get_by_id.side_effect = get_user_by_id

    rec_service = MagicMock(spec=RecommendationService)
    rec_service.recommend = AsyncMock(return_value=mock_agent_response)
    rec_service.recommend_item = AsyncMock(return_value=mock_agent_response)
    rec_service.recommend_for_user = AsyncMock(return_value=mock_agent_response)
    rec_service.recommend_for_item = AsyncMock(return_value=mock_agent_response)

    app.dependency_overrides[get_item_repository] = lambda: item_repo
    app.dependency_overrides[get_user_repository] = lambda: user_repo
    app.dependency_overrides[get_recommendation_service] = lambda: rec_service

    yield TestClient(app)
    app.dependency_overrides.clear()


class TestGetInfo:
    def test_returns_model_config(self, client: TestClient) -> None:
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "embedder_name" in data
        assert "reranker_name" in data
        assert "llm_model" in data


class TestGetUser:
    def test_found(self, client: TestClient) -> None:
        resp = client.get("/users/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "1"
        assert "text" in data
        assert "history" in data

    def test_not_found(self, client: TestClient, mock_users_index: MagicMock) -> None:
        mock_users_index.get_ids.return_value = pa.table(
            {"id": [], "text": [], "history": []}
        )
        resp = client.get("/users/999")
        assert resp.status_code == 404


class TestGetItem:
    def test_found(self, client: TestClient, mock_index: MagicMock) -> None:
        resp = client.get("/items/42")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "42"
        assert "text" in data

    def test_not_found(self, client: TestClient, mock_index: MagicMock) -> None:
        mock_index.get_ids.return_value = pa.table({"id": [], "text": []})
        resp = client.get("/items/999")
        assert resp.status_code == 404


class TestRecommend:
    def test_post_recommend(self, client: TestClient) -> None:
        resp = client.post(
            "/recommend",
            json={"text": "likes sci-fi", "limit": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert data["items"][0]["id"] == "99"


class TestRecommendUser:
    def test_recommend_for_user(self, client: TestClient) -> None:
        resp = client.post("/users/1/recommend?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data

    def test_user_not_found(
        self, client: TestClient, mock_users_index: MagicMock
    ) -> None:
        # In our new architecture, the router calls rec_service.recommend_for_user
        # and if it returns None, raises 404.
        # So we need to mock rec_service to return None.
        from agentic_rec.dependencies import get_recommendation_service

        rec_service = app.dependency_overrides[get_recommendation_service]()
        rec_service.recommend_for_user.return_value = None

        resp = client.post("/users/999/recommend")
        assert resp.status_code == 404


class TestRecommendItem:
    def test_recommend_for_item(self, client: TestClient) -> None:
        resp = client.post("/items/42/recommend?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data

    def test_item_not_found(self, client: TestClient, mock_index: MagicMock) -> None:
        from agentic_rec.dependencies import get_recommendation_service

        rec_service = app.dependency_overrides[get_recommendation_service]()
        rec_service.recommend_for_item.return_value = None

        resp = client.post("/items/999/recommend")
        assert resp.status_code == 404
