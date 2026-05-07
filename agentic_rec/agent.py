from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime  # noqa: TC003
from typing import TYPE_CHECKING

import pydantic
import pydantic_ai
from pydantic_ai import RunContext

from agentic_rec.params import LLM_MODEL

if TYPE_CHECKING:
    from agentic_rec.index import LanceIndex


class Interaction(pydantic.BaseModel):
    item_id: str
    event_timestamp: datetime
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
    user_text: str
    history: list[Interaction] = []
    top_k: int = 10


class RecommendResponse(pydantic.BaseModel):
    items: list[RankedItem]


@dataclass
class AgentDeps:
    index: LanceIndex
    request: RecommendRequest


SYSTEM_PROMPT = """\
You are a personalized movie recommender.

You receive a JSON object with:
- user_text: user demographics and stated preferences
- history: list of past interactions (may be empty), each with item_id, event_timestamp, \
event_name, event_value
- top_k: number of items to recommend

Workflow:

1. Context understanding:
    If history is not empty, call get_item_texts with all interacted item IDs.
    Analyze retrieved texts with event values (ratings) to build a preference summary,
    emphasizing recent and highly-rated interactions.

2. Candidate retrieval: call search_items 2-4 times with diverse queries.
    - Use specific genre/theme queries derived from the user's taste profile.
    - Use user_text directly as one query if it contains useful preference signals.
    - Exclude already-interacted item IDs from all search calls.
    - Aim for diversity: vary query angles (genre, mood, era, director style).

3. Cold-start: if history is empty, skip get_item_texts and rely solely on user_text.

4. Ranking with explanations: from all candidates, select the top_k items.
    Rank by relevance and diversity.
    For each item, provide a concise one-sentence explanation of why it suits the user.

Return a RecommendResponse with the ranked list of items.
"""

agent: pydantic_ai.Agent[AgentDeps, RecommendResponse] = pydantic_ai.Agent(
    model=LLM_MODEL,
    system_prompt=SYSTEM_PROMPT,
    output_type=RecommendResponse,
    defer_model_check=True,
)


@agent.instructions
def user_context(ctx: RunContext[AgentDeps]) -> str:
    """Serialize the request as JSON for the agent to interpret."""
    return ctx.deps.request.model_dump_json()


@agent.tool
def get_item_texts(
    ctx: RunContext[AgentDeps],
    item_ids: list[str],
) -> dict[str, str]:
    """Look up item texts for the given item IDs."""
    dataset = ctx.deps.index.get_ids(item_ids)
    return {row["id"]: row["text"] for row in dataset}


@agent.tool
def search_items(
    ctx: RunContext[AgentDeps],
    query: str,
    exclude_ids: list[str] | None = None,
    top_k: int = 20,
) -> list[ItemCandidate]:
    """Search for items matching the query using hybrid vector + full-text search."""
    dataset = ctx.deps.index.search(query, exclude_ids=exclude_ids, top_k=top_k)
    return [
        ItemCandidate(item_id=row["id"], item_text=row["text"], score=row["score"])
        for row in dataset
    ]


async def recommend(request: RecommendRequest, index: LanceIndex) -> RecommendResponse:
    """Run the recommendation agent and return ranked items."""
    deps = AgentDeps(index=index, request=request)
    result = await agent.run(deps=deps)
    return result.output


def main(
    user_text: str = "25-year-old male, software engineer, enjoys sci-fi and thriller films",
    top_k: int = 5,
) -> None:
    """Sanity check: run a cold-start recommendation and print results."""
    import asyncio

    from loguru import logger

    import agentic_rec.index
    from agentic_rec.index import LanceIndex, LanceIndexConfig

    agentic_rec.index.main(overwrite=False)
    index = LanceIndex.load(LanceIndexConfig())
    request = RecommendRequest(user_text=user_text, top_k=top_k)
    logger.info("request: {}", request.model_dump_json())

    response = asyncio.run(recommend(request, index))
    for i, item in enumerate(response.items, 1):
        logger.info("{}. {} — {}", i, item.item_text, item.explanation)


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main, as_positional=False)
