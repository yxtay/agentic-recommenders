from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime  # noqa: TC003
from typing import TYPE_CHECKING

import pydantic

if TYPE_CHECKING:
    from agentic_rec.index import LanceIndex


class Interaction(pydantic.BaseModel):
    item_id: str
    event_datetime: datetime
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
    text: str
    history: list[Interaction] = []
    top_k: int = 10


class RecommendResponse(pydantic.BaseModel):
    items: list[RankedItem]


@dataclass
class AgentDeps:
    index: LanceIndex
    request: RecommendRequest
