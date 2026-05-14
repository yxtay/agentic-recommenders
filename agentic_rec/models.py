from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Annotated

import pydantic

ItemId = Annotated[str, pydantic.Field(description="Unique item identifier.")]
ItemText = Annotated[
    str, pydantic.Field(description="Full text description of the item.")
]
UserId = Annotated[str, pydantic.Field(description="Unique user identifier.")]


class Interaction(pydantic.BaseModel):
    item_id: ItemId
    event_datetime: datetime = pydantic.Field(
        description="When the interaction occurred."
    )
    event_name: str = pydantic.Field(
        description="Type of interaction, e.g. 'rating', 'click', 'purchase', 'watch'."
    )
    event_value: float = pydantic.Field(
        description="Interaction value, e.g. rating score."
    )


InteractionHistory = Annotated[
    list[Interaction],
    pydantic.Field(default=[], description="Interaction history."),
]


class ItemCandidate(pydantic.BaseModel):
    id: ItemId
    text: ItemText
    score: float = pydantic.Field(
        default=0.0, description="Relevance score from search."
    )


class ItemRecommended(pydantic.BaseModel):
    id: ItemId
    text: ItemText
    explanation: str = pydantic.Field(
        description="Concise reason for recommending this item, e.g. 'Because you...'."
    )


class RecommendRequest(pydantic.BaseModel):
    text: str = pydantic.Field(
        description="User profile or item description as context."
    )
    history: InteractionHistory
    limit: int = pydantic.Field(default=10, description="Number of items to recommend.")


class RecommendResponse(pydantic.BaseModel):
    items: list[ItemRecommended] = pydantic.Field(
        description="Ranked list of recommended items, deduplicated and excluding previously interacted items."
    )


class UserResponse(pydantic.BaseModel):
    id: UserId
    text: str = pydantic.Field(description="User demographics and stated preferences.")
    history: InteractionHistory


class ItemResponse(pydantic.BaseModel):
    id: ItemId
    text: ItemText


class HealthResponse(pydantic.BaseModel):
    status: str = pydantic.Field(description="Service health status.")
    num_items: int = pydantic.Field(description="Number of items in the index.")
    num_users: int = pydantic.Field(description="Number of users in the index.")
    llm_ready: bool = pydantic.Field(description="Whether the LLM API is reachable.")
    embedder_name: str = pydantic.Field(description="Configured embedding model name.")
    reranker_name: str = pydantic.Field(description="Configured reranker model name.")
    llm_model: str = pydantic.Field(description="Configured LLM model identifier.")
