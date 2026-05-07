from __future__ import annotations

from datetime import UTC, datetime

import pydantic
import pytest

from agentic_rec.agent import (
    Interaction,
    ItemCandidate,
    RankedItem,
    RecommendRequest,
    RecommendResponse,
)


class TestInteraction:
    def test_fields(self) -> None:
        i = Interaction(
            item_id="123",
            event_timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            event_name="rating",
            event_value=5.0,
        )
        assert i.item_id == "123"
        assert i.event_name == "rating"
        assert i.event_value == 5.0  # noqa: PLR2004

    def test_timestamp_parsing(self) -> None:
        i = Interaction(
            item_id="1",
            event_timestamp="2024-01-01T12:00:00",
            event_name="rating",
            event_value=3.0,
        )
        assert i.event_timestamp == datetime(2024, 1, 1, 12, 0, 0)  # noqa: DTZ001


class TestRecommendRequest:
    def test_required_user_text(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            RecommendRequest()

    def test_history_defaults_empty(self) -> None:
        req = RecommendRequest(user_text="25 year old male, likes sci-fi")
        assert req.history == []
        assert req.top_k == 10  # noqa: PLR2004

    def test_with_history(self) -> None:
        req = RecommendRequest(
            user_text="test user",
            history=[
                Interaction(
                    item_id="1",
                    event_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                    event_name="rating",
                    event_value=5.0,
                )
            ],
            top_k=5,
        )
        assert len(req.history) == 1
        assert req.top_k == 5  # noqa: PLR2004


class TestItemCandidate:
    def test_defaults(self) -> None:
        c = ItemCandidate(item_id="42", item_text="Some Movie (2024)")
        assert c.score == 0.0

    def test_with_score(self) -> None:
        c = ItemCandidate(item_id="42", item_text="Some Movie", score=0.95)
        assert c.score == 0.95  # noqa: PLR2004


class TestRankedItem:
    def test_fields(self) -> None:
        item = RankedItem(
            item_id="42",
            item_text="Some Movie (2024)",
            explanation="Matches your preference for sci-fi",
        )
        assert item.item_id == "42"
        assert item.explanation == "Matches your preference for sci-fi"


class TestRecommendResponse:
    def test_items_list(self) -> None:
        resp = RecommendResponse(
            items=[
                RankedItem(item_id="1", item_text="Movie A", explanation="reason A"),
                RankedItem(item_id="2", item_text="Movie B", explanation="reason B"),
            ]
        )
        assert len(resp.items) == 2  # noqa: PLR2004
