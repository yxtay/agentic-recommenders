from __future__ import annotations

from datetime import datetime  # noqa: TC003

import pydantic


class Interaction(pydantic.BaseModel):
    item_id: str
    event_datetime: datetime
    event_name: str
    event_value: float


class ItemCandidate(pydantic.BaseModel):
    id: str = pydantic.Field(description="Unique item identifier.")
    text: str = pydantic.Field(description="Full text description of the item.")
    score: float = pydantic.Field(
        default=0.0, description="Relevance score from search."
    )


class ItemRecommended(pydantic.BaseModel):
    id: str = pydantic.Field(
        description="Item ID, exactly as returned by search_items."
    )
    text: str = pydantic.Field(
        description="Item text, exactly as returned by search_items."
    )
    explanation: str = pydantic.Field(
        description="Concise reason for recommending this item, e.g. 'Because you...'."
    )


class RecommendRequest(pydantic.BaseModel):
    text: str
    history: list[Interaction] = []
    limit: int = 10


class RecommendResponse(pydantic.BaseModel):
    items: list[ItemRecommended] = pydantic.Field(
        description="Ranked list of recommended items, deduplicated and excluding previously interacted items."
    )


class UserResponse(pydantic.BaseModel):
    id: str
    text: str
    history: list[Interaction] = []


class ItemResponse(pydantic.BaseModel):
    id: str
    text: str


class HealthResponse(pydantic.BaseModel):
    status: str
    num_items: int
    num_users: int
    llm_ready: bool
    embedder_name: str
    reranker_name: str
    llm_model: str
