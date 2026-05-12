from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from agentic_rec.agent import ITEM_INSTRUCTIONS, USER_INSTRUCTIONS, AgentDeps, agent
from agentic_rec.models import RecommendRequest, RecommendResponse

if TYPE_CHECKING:
    from agentic_rec.repositories.item_repository import ItemRepository
    from agentic_rec.repositories.user_repository import UserRepository


class RecommendationService:
    def __init__(
        self,
        item_repository: ItemRepository,
        user_repository: UserRepository,
    ) -> None:
        self.item_repository = item_repository
        self.user_repository = user_repository
        self.rec_agent = agent

    async def recommend(self, request: RecommendRequest) -> RecommendResponse:
        """Generate user-based recommendations."""
        deps = AgentDeps(index=self.item_repository.index, request=request)
        response = await self.rec_agent.run(instructions=USER_INSTRUCTIONS, deps=deps)
        logger.info("recommend: {} items", len(response.output.items))
        return response.output

    async def recommend_item(self, request: RecommendRequest) -> RecommendResponse:
        """Generate item-based recommendations."""
        deps = AgentDeps(index=self.item_repository.index, request=request)
        response = await self.rec_agent.run(instructions=ITEM_INSTRUCTIONS, deps=deps)
        logger.info("recommend_item: {} items", len(response.output.items))
        return response.output

    async def recommend_for_user(
        self, user_id: str, limit: int = 10
    ) -> RecommendResponse | None:
        """Look up user and generate recommendations."""
        user = self.user_repository.get_by_id(user_id)
        if not user:
            return None
        user.history = user.history[-20:]
        request = RecommendRequest.model_validate({**user.model_dump(), "limit": limit})
        return await self.recommend(request)

    async def recommend_for_item(
        self, item_id: str, limit: int = 10
    ) -> RecommendResponse | None:
        """Look up item and generate similar-item recommendations."""
        item = self.item_repository.get_by_id(item_id)
        if not item:
            return None
        request = RecommendRequest.model_validate({**item.model_dump(), "limit": limit})
        return await self.recommend_item(request)
