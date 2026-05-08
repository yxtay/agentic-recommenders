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
    from agentic_rec.index import LanceIndex


@dataclass
class AgentDeps:
    index: LanceIndex
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
    For each item, provide a concise explanation of why it is recommended.
    Use short explanations such as "Similar to...", "Matches your interest in...", etc.

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

agent: pydantic_ai.Agent[AgentDeps, RecommendResponse] = pydantic_ai.Agent(
    model=settings.llm_model,
    system_prompt=SYSTEM_PROMPT,
    output_type=RecommendResponse,
    defer_model_check=True,
)


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
    logger.info("get_item_texts: {} ids", len(item_ids))
    dataset = ctx.deps.index.get_ids(item_ids)
    logger.info("get_item_texts: {} results", len(dataset))
    return {row["id"]: row["text"] for row in dataset}


_item_candidate_adapter = pydantic.TypeAdapter(list[ItemCandidate])


@agent.tool
def search_items(
    ctx: RunContext[AgentDeps],
    query: str,
    exclude_ids: list[str] | None = None,
    limit: int = 20,
) -> list[ItemCandidate]:
    """Search for items matching the query using hybrid vector + full-text search."""
    dataset = ctx.deps.index.search(query, exclude_ids=exclude_ids, limit=limit)
    logger.info("search_items: {} results", len(dataset))
    return _item_candidate_adapter.validate_python(dataset.to_list())


def main(limit: int = 5) -> None:
    """Sanity check: sample a user from parquet and run recommendation."""
    import asyncio
    import random

    import datasets
    import rich

    import agentic_rec.index
    from agentic_rec.index import LanceIndex, LanceIndexConfig

    agentic_rec.index.main(overwrite=False)
    index = LanceIndex(LanceIndexConfig())
    index.open_table()

    users_dataset = datasets.Dataset.from_parquet(settings.users_parquet)
    sample_idx = random.randrange(len(users_dataset))
    sample_user = users_dataset[sample_idx]
    request = RecommendRequest.model_validate({**sample_user, "limit": limit})
    request.history = request.history[-20:]
    rich.print(request)

    deps = AgentDeps(index=index, request=request)
    response = asyncio.run(agent.run(instructions=USER_INSTRUCTIONS, deps=deps))
    rich.print(response.output.items)


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main, as_positional=False)
