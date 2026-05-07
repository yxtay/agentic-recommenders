from __future__ import annotations

from datetime import datetime  # noqa: TC003

import pydantic


class Interaction(pydantic.BaseModel):
    item_id: str
    event_timestamp: datetime
    event_name: str
    event_value: float


class ItemCandidate(pydantic.BaseModel):
    item_id: str
    item_text: str
    score: float = 0.0


class RankedItem(pydantic.BaseModel):
    item_id: str
    item_text: str
    explanation: str


class RecommendRequest(pydantic.BaseModel):
    user_text: str
    history: list[Interaction] = []
    top_k: int = 10


class RecommendResponse(pydantic.BaseModel):
    items: list[RankedItem]
