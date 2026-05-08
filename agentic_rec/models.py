from __future__ import annotations

from datetime import datetime  # noqa: TC003

import pydantic


class Interaction(pydantic.BaseModel):
    item_id: str
    event_datetime: datetime
    event_name: str
    event_value: float


class ItemCandidate(pydantic.BaseModel):
    id: str
    text: str
    score: float = 0.0


class RankedItem(pydantic.BaseModel):
    id: str
    text: str
    explanation: str


class RecommendRequest(pydantic.BaseModel):
    text: str
    history: list[Interaction] = []
    top_k: int = 10


class RecommendResponse(pydantic.BaseModel):
    items: list[RankedItem]
