from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import datasets
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agentic_rec.app import app
from agentic_rec.models import RankedItem, RecommendResponse


@pytest.fixture
def mock_index() -> MagicMock:
    index = MagicMock()
    index.get_ids.return_value = datasets.Dataset.from_dict(
        {"id": ["42"], "text": ["The Matrix (1999) | Action, Sci-Fi"]}
    )
    return index


@pytest.fixture
def mock_users() -> datasets.Dataset:
    return datasets.Dataset.from_dict(
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


@pytest.fixture
def mock_agent_response() -> RecommendResponse:
    return RecommendResponse(
        items=[
            RankedItem(
                id="99",
                text="Inception (2010) | Sci-Fi, Thriller",
                explanation="Similar to The Matrix",
            )
        ]
    )


@pytest.fixture(autouse=True)
def _setup_app_state(mock_index: MagicMock, mock_users: datasets.Dataset) -> None:
    app.state.index = mock_index
    app.state.users = mock_users


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestGetInfo:
    @pytest.mark.asyncio
    async def test_returns_model_config(self, client: AsyncClient) -> None:
        resp = await client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "embedder_name" in data
        assert "reranker_name" in data
        assert "llm_model" in data


class TestGetUser:
    @pytest.mark.asyncio
    async def test_found(self, client: AsyncClient) -> None:
        resp = await client.get("/users/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "1"
        assert "text" in data
        assert "history" in data

    @pytest.mark.asyncio
    async def test_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/users/999")
        assert resp.status_code == 404


class TestGetItem:
    @pytest.mark.asyncio
    async def test_found(self, client: AsyncClient, mock_index: MagicMock) -> None:
        resp = await client.get("/items/42")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "42"
        assert "text" in data
        mock_index.get_ids.assert_called_once_with(["42"])

    @pytest.mark.asyncio
    async def test_not_found(self, client: AsyncClient, mock_index: MagicMock) -> None:
        mock_index.get_ids.return_value = datasets.Dataset.from_dict(
            {"id": [], "text": []}
        )
        resp = await client.get("/items/999")
        assert resp.status_code == 404


class TestRecommend:
    @pytest.mark.asyncio
    async def test_post_recommend(
        self, client: AsyncClient, mock_agent_response: RecommendResponse
    ) -> None:
        mock_run = AsyncMock()
        mock_run.return_value.output = mock_agent_response
        with patch("agentic_rec.app.agent.run", mock_run):
            resp = await client.post(
                "/recommend",
                json={"text": "likes sci-fi", "limit": 5},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert data["items"][0]["id"] == "99"


class TestRecommendUser:
    @pytest.mark.asyncio
    async def test_recommend_for_user(
        self, client: AsyncClient, mock_agent_response: RecommendResponse
    ) -> None:
        mock_run = AsyncMock()
        mock_run.return_value.output = mock_agent_response
        with patch("agentic_rec.app.agent.run", mock_run):
            resp = await client.post("/users/1/recommend?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data

    @pytest.mark.asyncio
    async def test_user_not_found(self, client: AsyncClient) -> None:
        resp = await client.post("/users/999/recommend")
        assert resp.status_code == 404


class TestRecommendItem:
    @pytest.mark.asyncio
    async def test_recommend_for_item(
        self, client: AsyncClient, mock_agent_response: RecommendResponse
    ) -> None:
        mock_run = AsyncMock()
        mock_run.return_value.output = mock_agent_response
        with patch("agentic_rec.app.agent.run", mock_run):
            resp = await client.post("/items/42/recommend?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data

    @pytest.mark.asyncio
    async def test_item_not_found(
        self, client: AsyncClient, mock_index: MagicMock
    ) -> None:
        mock_index.get_ids.return_value = datasets.Dataset.from_dict(
            {"id": [], "text": []}
        )
        resp = await client.post("/items/999/recommend")
        assert resp.status_code == 404
