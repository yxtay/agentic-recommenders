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
    limit: int = 10


class RecommendResponse(pydantic.BaseModel):
    items: list[RankedItem]


class UserResponse(pydantic.BaseModel):
    id: str
    text: str
    history: list[Interaction] = []


class ItemResponse(pydantic.BaseModel):
    id: str
    text: str


class InfoResponse(pydantic.BaseModel):
    embedder_name: str
    reranker_name: str
    llm_model: str
