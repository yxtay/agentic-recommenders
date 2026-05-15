from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from agentic_rec.agent import ITEM_INSTRUCTIONS, USER_INSTRUCTIONS, AgentDeps, agent
from agentic_rec.models import RecommendRequest, RecommendResponse
from agentic_rec.settings import settings
from agentic_rec.utils.cache import ResponseCache

if TYPE_CHECKING:
    from agentic_rec.repositories.item_repository import ItemRepository
    from agentic_rec.repositories.user_repository import UserRepository


class RecommendationService:
    def __init__(
        self,
        item_repository: ItemRepository,
        user_repository: UserRepository,
        cache: ResponseCache | None = None,
    ) -> None:
        self.item_repository = item_repository
        self.user_repository = user_repository
        self.rec_agent = agent
        self.cache = cache

    async def recommend(
        self, request: RecommendRequest, cache_ttl: int = settings.cache_ttl
    ) -> RecommendResponse:
        """Generate user-based recommendations."""
        cache_key = ResponseCache.generate_key("recommend", request)
        if self.cache is not None and (cached := self.cache.get(cache_key)):
            logger.info("Cache hit: recommend")
            return RecommendResponse.model_validate(cached)

        deps = AgentDeps(item_repository=self.item_repository, request=request)
        response = await self.rec_agent.run(instructions=USER_INSTRUCTIONS, deps=deps)
        logger.info("recommend: {} items", len(response.output.items))

        if self.cache is not None:
            self.cache.set(cache_key, response.output.model_dump(), cache_ttl)

        return response.output

    async def recommend_item(
        self, request: RecommendRequest, cache_ttl: int = settings.cache_ttl
    ) -> RecommendResponse:
        """Generate item-based recommendations."""
        cache_key = ResponseCache.generate_key("recommend_item", request)
        if self.cache is not None and (cached := self.cache.get(cache_key)):
            logger.info("Cache hit: recommend_item")
            return RecommendResponse.model_validate(cached)

        deps = AgentDeps(item_repository=self.item_repository, request=request)
        response = await self.rec_agent.run(instructions=ITEM_INSTRUCTIONS, deps=deps)
        logger.info("recommend_item: {} items", len(response.output.items))

        if self.cache is not None:
            self.cache.set(cache_key, response.output.model_dump(), cache_ttl)

        return response.output

    async def recommend_for_user(
        self, user_id: str, limit: int = 10, cache_ttl: int = settings.cache_ttl
    ) -> RecommendResponse | None:
        """Look up user and generate recommendations."""
        user = self.user_repository.get_by_id(user_id)
        if not user:
            return None
        request = RecommendRequest(text=user.text, history=user.history, limit=limit)
        return await self.recommend(request, cache_ttl=cache_ttl)

    async def recommend_for_item(
        self, item_id: str, limit: int = 10, cache_ttl: int = settings.cache_ttl
    ) -> RecommendResponse | None:
        """Look up item and generate similar-item recommendations."""
        item = self.item_repository.get_by_id(item_id)
        if not item:
            return None
        request = RecommendRequest(text=item.text, limit=limit)
        return await self.recommend_item(request, cache_ttl=cache_ttl)
