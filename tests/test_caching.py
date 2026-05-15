from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agentic_rec.main import app
from agentic_rec.models import ItemRecommended, RecommendResponse


@pytest.fixture(autouse=True)
def _mock_lifespan() -> None:
    with (
        patch("agentic_rec.index.LanceIndex.load") as mock_load,
        patch("agentic_rec.agent.check_llm", return_value=True),
    ):
        mock_idx = MagicMock()
        mock_idx.table.count_rows.return_value = 0
        mock_load.return_value = mock_idx
        yield


@pytest.fixture
def mock_agent_run() -> AsyncMock:
    with patch("agentic_rec.services.recommendation_service.agent") as mock_agent:
        mock_run_res = MagicMock()
        mock_run_res.output = RecommendResponse(
            items=[ItemRecommended(id="1", text="Movie 1", explanation="Because...")]
        )
        mock_agent.run = AsyncMock(return_value=mock_run_res)
        yield mock_agent.run


def test_recommend_caching(mock_agent_run: AsyncMock) -> None:
    with TestClient(app) as client:
        request_data = {"text": "action movies", "history": [], "limit": 5}

        # First request with TTL - should call agent and cache
        response1 = client.post(
            "/recommend", json=request_data, headers={"X-Cache-TTL": "60"}
        )
        assert response1.status_code == 200
        assert mock_agent_run.call_count == 1

        # Second request with same data and TTL - should be a cache hit, NOT call agent
        response2 = client.post(
            "/recommend", json=request_data, headers={"X-Cache-TTL": "60"}
        )
        assert response2.status_code == 200
        assert response1.json() == response2.json()
        assert mock_agent_run.call_count == 1  # Still 1


def test_cache_expiration(mock_agent_run: AsyncMock) -> None:
    mock_agent_run.return_value.output = RecommendResponse(items=[])

    with TestClient(app) as client:
        request_data = {"text": "comedy movies", "history": [], "limit": 5}

        # Request with 1 second TTL
        client.post("/recommend", json=request_data, headers={"X-Cache-TTL": "1"})
        assert mock_agent_run.call_count == 1

        # Wait for expiration
        time.sleep(1.1)

        # Request again - should be a cache miss, call agent again
        response = client.post(
            "/recommend", json=request_data, headers={"X-Cache-TTL": "1"}
        )
        assert response.status_code == 200
        assert mock_agent_run.call_count == 2


def test_default_cache_ttl(mock_agent_run: AsyncMock) -> None:
    mock_agent_run.return_value.output = RecommendResponse(items=[])

    with TestClient(app) as client:
        request_data = {"text": "sci-fi movies", "history": [], "limit": 5}

        # Request without TTL header - should use default 1h TTL
        client.post("/recommend", json=request_data)
        assert mock_agent_run.call_count == 1

        # Second request - should be a cache hit even without header
        client.post("/recommend", json=request_data)
        assert mock_agent_run.call_count == 1


def test_different_keys(mock_agent_run: AsyncMock) -> None:
    async def side_effect(instructions: str, deps: Any) -> MagicMock:  # noqa: ARG001, ANN401
        return MagicMock(
            output=RecommendResponse(
                items=[
                    ItemRecommended(
                        id=deps.request.text, text=deps.request.text, explanation="..."
                    )
                ]
            )
        )

    mock_agent_run.side_effect = side_effect

    with TestClient(app) as client:
        req1 = {"text": "horror", "history": [], "limit": 5}
        req2 = {"text": "romance", "history": [], "limit": 5}

        resp1 = client.post("/recommend", json=req1, headers={"X-Cache-TTL": "60"})
        resp2 = client.post("/recommend", json=req2, headers={"X-Cache-TTL": "60"})

        assert resp1.json() != resp2.json()
        assert mock_agent_run.call_count == 2
