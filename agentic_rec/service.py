from __future__ import annotations

import bentoml

from agentic_rec.agent import (
    Interaction,
    RecommendRequest,
    RecommendResponse,
    recommend,
)
from agentic_rec.index import LanceIndex, LanceIndexConfig


@bentoml.service
class RecommenderService:
    def __init__(self) -> None:
        self.index = LanceIndex.load(LanceIndexConfig())

    @bentoml.api
    async def recommend(
        self,
        user_text: str,
        history: list[Interaction] | None = None,
        top_k: int = 10,
    ) -> RecommendResponse:
        request = RecommendRequest(
            user_text=user_text,
            history=history or [],
            top_k=top_k,
        )
        return await recommend(request, self.index)
