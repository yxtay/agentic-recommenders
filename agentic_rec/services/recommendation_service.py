from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pydantic
import pydantic_ai
from loguru import logger
from pydantic_ai import RunContext

from agentic_rec.models import ItemCandidate, RecommendRequest, RecommendResponse
from agentic_rec.settings import settings

if TYPE_CHECKING:
    from agentic_rec.repositories.item_repository import ItemRepository
    from agentic_rec.repositories.user_repository import UserRepository


@dataclass
class AgentDeps:
    item_repository: ItemRepository
    request: RecommendRequest


SYSTEM_PROMPT = """\
You are a personalized item recommender.

You receive a JSON object with:
- text: context description (user profile or item description, depending on the task)
- history: list of past interactions (may be empty), each with item_id, event_datetime, \
event_name, event_value
- limit: number of items to recommend

Workflow:

1. Context understanding:
    If history is not empty, call get_item_texts with all interacted item IDs.
    Analyze retrieved texts with event values to build a preference summary,
    emphasizing recent and highly-rated interactions.

2. Candidate retrieval: call search_items 2-4 times with diverse queries.
    - Derive queries from the context and taste profile.
    - Use the text field directly as one query if it contains useful preference signals.
    - Exclude already-interacted item IDs from all search calls.
    - Aim for diversity: vary query angles.

3. Cold-start: if history is empty, skip get_item_texts and rely solely on text.

4. Ranking with explanations: from all candidates, select the limit items.
    Rank by relevance and diversity.
    For each item, provide a concise explanation of why it is recommended,
    such as due to stated or inferred preferences, or recent activity.
    Use short explanations such as "Because you...", etc.

Return a RecommendResponse with the ranked list of items.
"""


USER_INSTRUCTIONS = """\
You are recommending movies for a user.
Items are films described by title and genres.
The text field contains user demographics and stated preferences.
Use history and text to understand taste.
Vary queries by genre, mood, era, and director style for diversity.
"""

ITEM_INSTRUCTIONS = """\
You are recommending movies similar to a given movie.
Items are films described by title and genres.
The text field contains the source movie's title and genres.
Find diverse but related films that someone who liked this movie would enjoy.
There is no interaction history.
Vary queries by genre, mood, era, and thematic similarity for diversity.
"""


_item_candidate_adapter = pydantic.TypeAdapter(list[ItemCandidate])


class RecommendationService:
    def __init__(
        self,
        item_repository: ItemRepository,
        user_repository: UserRepository,
        rec_agent: pydantic_ai.Agent[AgentDeps, RecommendResponse] | None = None,
    ) -> None:
        self.item_repository = item_repository
        self.user_repository = user_repository
        self.rec_agent = rec_agent or self._create_agent()

    def _create_agent(self) -> pydantic_ai.Agent[AgentDeps, RecommendResponse]:
        agent = pydantic_ai.Agent(
            model=settings.llm_model,
            system_prompt=SYSTEM_PROMPT,
            output_type=RecommendResponse,
            deps_type=AgentDeps,
            defer_model_check=True,
        )

        @agent.instructions
        def user_context(ctx: RunContext[AgentDeps]) -> str:
            return ctx.deps.request.model_dump_json()

        @agent.tool
        def get_item_texts(
            ctx: RunContext[AgentDeps],
            item_ids: list[str],
        ) -> dict[str, str]:
            """Look up the full text descriptions for items by their IDs."""
            logger.info("get_item_texts: {} ids", len(item_ids))
            result = ctx.deps.item_repository.get_by_ids(item_ids)
            logger.info("get_item_texts: {} results", result.num_rows)
            return {row["id"]: row["text"] for row in result.to_pylist()}

        @agent.tool
        def search_items(
            ctx: RunContext[AgentDeps],
            query: str,
            exclude_ids: list[str] | None = None,
            limit: int = 20,
        ) -> list[ItemCandidate]:
            """Search for candidate items using hybrid vector + full-text search."""
            result = ctx.deps.item_repository.search(query, exclude_ids=exclude_ids, limit=limit)
            logger.info("search_items: {} results", result.num_rows)
            return _item_candidate_adapter.validate_python(result.to_pylist())

        return agent

    async def recommend(self, request: RecommendRequest) -> RecommendResponse:
        """Generate user-based recommendations."""
        deps = AgentDeps(item_repository=self.item_repository, request=request)
        response = await self.rec_agent.run(instructions=USER_INSTRUCTIONS, deps=deps)
        logger.info("recommend: {} items", len(response.output.items))
        return response.output

    async def recommend_item(self, request: RecommendRequest) -> RecommendResponse:
        """Generate item-based recommendations."""
        deps = AgentDeps(item_repository=self.item_repository, request=request)
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
