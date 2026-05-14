from __future__ import annotations

from datetime import datetime  # noqa: TC003

import pydantic


class Interaction(pydantic.BaseModel):
    item_id: str = pydantic.Field(description="ID of the interacted item.")
    event_datetime: datetime = pydantic.Field(
        description="When the interaction occurred."
    )
    event_name: str = pydantic.Field(description="Type of interaction, e.g. 'rating'.")
    event_value: float = pydantic.Field(
        description="Interaction value, e.g. rating score."
    )


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
    text: str = pydantic.Field(
        description="User profile or item description as context."
    )
    history: list[Interaction] = pydantic.Field(
        default=[],
        description="Past interactions; empty for cold-start or item-based requests.",
    )
    limit: int = pydantic.Field(default=10, description="Number of items to recommend.")


class RecommendResponse(pydantic.BaseModel):
    items: list[ItemRecommended] = pydantic.Field(
        description="Ranked list of recommended items, deduplicated and excluding previously interacted items."
    )


class UserResponse(pydantic.BaseModel):
    id: str = pydantic.Field(description="Unique user identifier.")
    text: str = pydantic.Field(description="User demographics and stated preferences.")
    history: list[Interaction] = pydantic.Field(
        default=[], description="User's interaction history."
    )


class ItemResponse(pydantic.BaseModel):
    id: str = pydantic.Field(description="Unique item identifier.")
    text: str = pydantic.Field(description="Full text description of the item.")


class HealthResponse(pydantic.BaseModel):
    status: str = pydantic.Field(description="Service health status.")
    num_items: int = pydantic.Field(description="Number of items in the index.")
    num_users: int = pydantic.Field(description="Number of users in the index.")
    llm_ready: bool = pydantic.Field(description="Whether the LLM API is reachable.")
    embedder_name: str = pydantic.Field(description="Configured embedding model name.")
    reranker_name: str = pydantic.Field(description="Configured reranker model name.")
    llm_model: str = pydantic.Field(description="Configured LLM model identifier.")
