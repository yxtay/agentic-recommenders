import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from agentic_rec.main import app
from agentic_rec.models import ItemRecommended, RecommendResponse


@patch("agentic_rec.services.recommendation_service.agent")
def test_recommend_caching(mock_agent: MagicMock) -> None:
    # Setup mock response from agent
    mock_run_res = MagicMock()
    mock_run_res.output = RecommendResponse(
        items=[ItemRecommended(id="1", text="Movie 1", explanation="Because...")]
    )
    mock_agent.run = AsyncMock(return_value=mock_run_res)

    with TestClient(app) as client:
        request_data = {"text": "action movies", "history": [], "limit": 5}

        # First request with TTL - should call agent and cache
        response1 = client.post(
            "/recommend", json=request_data, headers={"X-Cache-TTL": "60"}
        )
        assert response1.status_code == 200
        assert mock_agent.run.call_count == 1

        # Second request with same data and TTL - should be a cache hit, NOT call agent
        response2 = client.post(
            "/recommend", json=request_data, headers={"X-Cache-TTL": "60"}
        )
        assert response2.status_code == 200
        assert response1.json() == response2.json()
        assert mock_agent.run.call_count == 1  # Still 1


@patch("agentic_rec.services.recommendation_service.agent")
def test_cache_expiration(mock_agent: MagicMock) -> None:
    mock_run_res = MagicMock()
    mock_run_res.output = RecommendResponse(items=[])
    mock_agent.run = AsyncMock(return_value=mock_run_res)

    with TestClient(app) as client:
        request_data = {"text": "comedy movies", "history": [], "limit": 5}

        # Request with 1 second TTL
        client.post("/recommend", json=request_data, headers={"X-Cache-TTL": "1"})
        assert mock_agent.run.call_count == 1

        # Wait for expiration
        time.sleep(1.1)

        # Request again - should be a cache miss, call agent again
        response = client.post(
            "/recommend", json=request_data, headers={"X-Cache-TTL": "1"}
        )
        assert response.status_code == 200
        assert mock_agent.run.call_count == 2


@patch("agentic_rec.services.recommendation_service.agent")
def test_no_cache_header(mock_agent: MagicMock) -> None:
    mock_run_res = MagicMock()
    mock_run_res.output = RecommendResponse(items=[])
    mock_agent.run = AsyncMock(return_value=mock_run_res)

    with TestClient(app) as client:
        request_data = {"text": "sci-fi movies", "history": [], "limit": 5}

        # Request without TTL header
        client.post("/recommend", json=request_data)
        client.post("/recommend", json=request_data)

        # Should call agent every time
        assert mock_agent.run.call_count == 2


@patch("agentic_rec.services.recommendation_service.agent")
def test_different_keys(mock_agent: MagicMock) -> None:
    async def side_effect(instructions: str, deps: Any) -> MagicMock:  # noqa: ANN401
        return MagicMock(
            output=RecommendResponse(
                items=[
                    ItemRecommended(
                        id=deps.request.text, text=deps.request.text, explanation="..."
                    )
                ]
            )
        )

    mock_agent.run = AsyncMock(side_effect=side_effect)

    with TestClient(app) as client:
        req1 = {"text": "horror", "history": [], "limit": 5}
        req2 = {"text": "romance", "history": [], "limit": 5}

        resp1 = client.post("/recommend", json=req1, headers={"X-Cache-TTL": "60"})
        resp2 = client.post("/recommend", json=req2, headers={"X-Cache-TTL": "60"})

        assert resp1.json() != resp2.json()
        assert mock_agent.run.call_count == 2
