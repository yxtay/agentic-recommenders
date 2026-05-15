import time
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from agentic_rec.main import app
from agentic_rec.models import ItemRecommended, RecommendResponse


@patch("agentic_rec.services.recommendation_service.RecommendationService.recommend")
def test_recommend_caching(mock_recommend: MagicMock) -> None:
    # Setup mock response
    mock_response = RecommendResponse(
        items=[ItemRecommended(id="1", text="Movie 1", explanation="Because...")]
    )
    mock_recommend.return_value = mock_response

    with TestClient(app) as client:
        request_data = {"text": "action movies", "history": [], "limit": 5}

        # First request with TTL - should call service and cache
        response1 = client.post(
            "/recommend", json=request_data, headers={"X-Cache-TTL": "60"}
        )
        assert response1.status_code == 200
        assert mock_recommend.call_count == 1

        # Second request with same data and TTL - should be a cache hit, NOT call service
        response2 = client.post(
            "/recommend", json=request_data, headers={"X-Cache-TTL": "60"}
        )
        assert response2.status_code == 200
        assert response1.json() == response2.json()
        assert mock_recommend.call_count == 1  # Still 1


@patch("agentic_rec.services.recommendation_service.RecommendationService.recommend")
def test_cache_expiration(mock_recommend: MagicMock) -> None:
    mock_response = RecommendResponse(items=[])
    mock_recommend.return_value = mock_response

    with TestClient(app) as client:
        request_data = {"text": "comedy movies", "history": [], "limit": 5}

        # Request with 1 second TTL
        client.post("/recommend", json=request_data, headers={"X-Cache-TTL": "1"})
        assert mock_recommend.call_count == 1

        # Wait for expiration
        time.sleep(1.1)

        # Request again - should be a cache miss, call service again
        response = client.post(
            "/recommend", json=request_data, headers={"X-Cache-TTL": "1"}
        )
        assert response.status_code == 200
        assert mock_recommend.call_count == 2


@patch("agentic_rec.services.recommendation_service.RecommendationService.recommend")
def test_no_cache_header(mock_recommend: MagicMock) -> None:
    mock_response = RecommendResponse(items=[])
    mock_recommend.return_value = mock_response

    with TestClient(app) as client:
        request_data = {"text": "sci-fi movies", "history": [], "limit": 5}

        # Request without TTL header
        client.post("/recommend", json=request_data)
        client.post("/recommend", json=request_data)

        # Should call service every time
        assert mock_recommend.call_count == 2


@patch("agentic_rec.services.recommendation_service.RecommendationService.recommend")
def test_different_keys(mock_recommend: MagicMock) -> None:
    mock_recommend.side_effect = lambda req: RecommendResponse(
        items=[ItemRecommended(id=req.text, text=req.text, explanation="...")]
    )

    with TestClient(app) as client:
        req1 = {"text": "horror", "history": [], "limit": 5}
        req2 = {"text": "romance", "history": [], "limit": 5}

        resp1 = client.post("/recommend", json=req1, headers={"X-Cache-TTL": "60"})
        resp2 = client.post("/recommend", json=req2, headers={"X-Cache-TTL": "60"})

        assert resp1.json() != resp2.json()
        assert mock_recommend.call_count == 2
