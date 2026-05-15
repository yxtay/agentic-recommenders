from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pydantic
import pydantic_ai
from loguru import logger
from pydantic_ai import RunContext

from .models import ItemCandidate, RecommendRequest, RecommendResponse
from .settings import settings

if TYPE_CHECKING:
    from .repositories.item_repository import ItemRepository


@dataclass
class AgentDeps:
    item_repository: ItemRepository
    request: RecommendRequest


SYSTEM_PROMPT = """\
You are a personalized item recommender.

Workflow:

1. Context understanding:
    Analyze the text field for explicit preferences or item attributes.
    If history is not empty, call get_item_texts with all interacted item IDs.
    Combine text signals with retrieved item texts and event values to build
    a preference summary. Weight interactions with higher event_value and
    more recent event_datetime.
    If history is empty, rely solely on text.

2. Candidate retrieval: call search_items with at least 2 diverse queries.
    Use more queries when the preference summary reveals varied interests.
    Generate concise queries capturing different aspects of user taste.
    Exclude already-interacted item IDs from all search calls.

3. Ranking: from all candidates, select the limit items.
    Deduplicate candidates by item ID.
    Exclude any item IDs from the interaction history.
    Rank by relevance and diversity.
"""


USER_INSTRUCTIONS = """\
You are recommending items for a user.
The text field contains user demographics and stated preferences.
Prioritize taste patterns from recent history over older interactions.
"""

ITEM_INSTRUCTIONS = """\
You are recommending items similar to a given item.
The text field contains the source item's description and attributes.
Prioritize attribute and thematic similarity to the source item.
"""

agent: pydantic_ai.Agent[AgentDeps, RecommendResponse] = pydantic_ai.Agent(
    model=settings.llm_model,
    system_prompt=SYSTEM_PROMPT,
    output_type=RecommendResponse,
    defer_model_check=True,
)


@agent.instructions
def user_context(ctx: RunContext[AgentDeps]) -> str:
    """Serialize the request as JSON for the agent to interpret."""
    return ctx.deps.request.model_dump_json()


@agent.tool(strict=True)
def get_item_texts(
    ctx: RunContext[AgentDeps],
    item_ids: list[str],
) -> dict[str, str]:
    """Look up full text descriptions for items by their IDs."""
    logger.info("get_item_texts: {} ids", len(item_ids))
    result = ctx.deps.item_repository.get_by_ids(item_ids)
    logger.info("get_item_texts: {} results", result.num_rows)
    return {row["id"]: row["text"] for row in result.to_pylist()}


_item_candidate_adapter = pydantic.TypeAdapter(list[ItemCandidate])


@agent.tool(strict=True)
def search_items(
    ctx: RunContext[AgentDeps],
    query: str,
    query_type: Literal["vector", "fts", "hybrid"] = "hybrid",
    exclude_ids: list[str] | None = None,
    limit: int = 20,
) -> list[ItemCandidate]:
    """Search for candidate items using vector, full-text, or hybrid search.

    Use 'vector' for broad thematic similarity, 'fts' for specific terms
    or names, and 'hybrid' when both apply.
    """
    result = ctx.deps.item_repository.search(
        query, query_type=query_type, exclude_ids=exclude_ids, limit=limit
    )
    logger.info("search_items: {} results", result.num_rows)
    return _item_candidate_adapter.validate_python(result.to_pylist())


async def check_llm() -> bool:
    """Verify the LLM API key is set and valid."""
    try:
        test_agent: pydantic_ai.Agent[None, str] = pydantic_ai.Agent(
            model=settings.llm_model, output_type=str
        )
        await test_agent.run("Say hi")
    except (OSError, ValueError, RuntimeError):
        logger.exception("llm check: failed ({})", settings.llm_model)
        return False
    else:
        logger.info("llm check: ok ({})", settings.llm_model)
        return True


def main(limit: int = 5) -> None:
    """Sanity check: sample a user from parquet and run recommendation."""
    import asyncio
    import random

    import pyarrow.parquet as pq
    import rich

    from . import index
    from .index import LanceIndex, LanceIndexConfig
    from .repositories.item_repository import ItemRepository

    index.main(overwrite=False)
    index = LanceIndex(LanceIndexConfig())
    index.open_table()
    item_repository = ItemRepository(index)

    users_table = pq.read_table(settings.users_parquet, memory_map=True)
    sample_idx = random.randrange(users_table.num_rows)
    sample_user = users_table.slice(sample_idx, 1).to_pylist()[0]
    request = RecommendRequest.model_validate({**sample_user, "limit": limit})
    rich.print(request)

    deps = AgentDeps(item_repository=item_repository, request=request)
    response = asyncio.run(agent.run(instructions=USER_INSTRUCTIONS, deps=deps))
    rich.print(response.output.items)


def cli() -> None:
    from jsonargparse import auto_cli

    auto_cli(main, as_positional=False)


if __name__ == "__main__":
    cli()
