from __future__ import annotations

from typing import TYPE_CHECKING

from cachetools_async import cachedmethod
from loguru import logger

from agentic_rec.agent import ITEM_INSTRUCTIONS, USER_INSTRUCTIONS, AgentDeps
from agentic_rec.models import RecommendRequest, RecommendResponse

if TYPE_CHECKING:
    import pydantic_ai
    from cachetools import TLRUCache

    from agentic_rec.repositories.item_repository import ItemRepository
    from agentic_rec.repositories.user_repository import UserRepository


class RecommendationService:
    def __init__(
        self,
        item_repository: ItemRepository,
        user_repository: UserRepository,
        agent: pydantic_ai.Agent[AgentDeps, RecommendResponse],
        cache: TLRUCache,
    ) -> None:
        self.item_repository = item_repository
        self.user_repository = user_repository
        self.rec_agent = agent
        self.cache = cache

    @cachedmethod(cache=lambda self: self.cache)
    async def recommend(
        self, instructions: str, request: RecommendRequest
    ) -> RecommendResponse:
        """Generate recommendations via the agent."""
        deps = AgentDeps(item_repository=self.item_repository, request=request)
        response = await self.rec_agent.run(instructions=instructions, deps=deps)
        logger.info("recommend: {} items", len(response.output.items))
        return response.output

    async def recommend_user(self, request: RecommendRequest) -> RecommendResponse:
        """Generate user-based recommendations."""
        return await self.recommend(USER_INSTRUCTIONS, request)

    async def recommend_item(self, request: RecommendRequest) -> RecommendResponse:
        """Generate item-based recommendations."""
        return await self.recommend(ITEM_INSTRUCTIONS, request)

    async def recommend_for_user(
        self, user_id: str, limit: int = 10
    ) -> RecommendResponse | None:
        """Look up user and generate recommendations."""
        user = self.user_repository.get_by_id(user_id)
        if not user:
            return None
        request = RecommendRequest(text=user.text, history=user.history, limit=limit)
        return await self.recommend_user(request)

    async def recommend_for_item(
        self, item_id: str, limit: int = 10
    ) -> RecommendResponse | None:
        """Look up item and generate similar-item recommendations."""
        item = self.item_repository.get_by_id(item_id)
        if not item:
            return None
        request = RecommendRequest(text=item.text, limit=limit)
        return await self.recommend_item(request)
