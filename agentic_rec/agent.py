from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pydantic
import pydantic_ai
from pydantic_ai import RunContext

from agentic_rec.models import ItemCandidate, RecommendRequest, RecommendResponse
from agentic_rec.params import ITEMS_PARQUET, LLM_MODEL, USERS_PARQUET

if TYPE_CHECKING:
    from agentic_rec.index import LanceIndex


@dataclass
class AgentDeps:
    index: LanceIndex
    request: RecommendRequest


SYSTEM_PROMPT = """\
You are a personalized item recommender.

You receive a JSON object with:
- text: user demographics and stated preferences
- history: list of past interactions (may be empty), each with item_id, event_datetime, \
event_name, event_value
- limit: number of items to recommend

Workflow:

1. Context understanding:
    If history is not empty, call get_item_texts with all interacted item IDs.
    Analyze retrieved texts with event values to build a preference summary,
    emphasizing recent and highly-rated interactions.

2. Candidate retrieval: call search_items 2-4 times with diverse queries.
    - Derive queries from the user's taste profile.
    - Use the text field directly as one query if it contains useful preference signals.
    - Exclude already-interacted item IDs from all search calls.
    - Aim for diversity: vary query angles.

3. Cold-start: if history is empty, skip get_item_texts and rely solely on text.

4. Ranking with explanations: from all candidates, select the limit items.
    Rank by relevance and diversity.
    For each item, provide a concise explanation of why it is recommended,
    referencing specific preferences or past interactions when possible.
    Use short explanations such as "Similar to...", "Matches your interest in...", etc.

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
    return _item_candidate_adapter.validate_python(dataset.to_list())


MOVIE_INSTRUCTIONS = """\
You are recommending movies. Items are films described by title and genres.
Vary queries by genre, mood, era, and director style for diversity.
"""


def main(limit: int = 5) -> None:
    """Sanity check: sample a user from parquet and run recommendation."""
    import asyncio

    import datasets
    import rich

    import agentic_rec.data
    import agentic_rec.index
    from agentic_rec.index import LanceIndex, LanceIndexConfig

    agentic_rec.data.main(overwrite=False)
    index_config = LanceIndexConfig()
    try:
        index = LanceIndex.load(index_config)
    except FileNotFoundError:
        dataset = datasets.Dataset.from_parquet(ITEMS_PARQUET)
        index = LanceIndex(index_config)
        index.index_data(dataset)

    users_dataset = datasets.Dataset.from_parquet(USERS_PARQUET)
    sample_user = users_dataset.shuffle()[0]
    request = RecommendRequest.model_validate({**sample_user, "limit": limit})
    request.history = request.history[-20:]
    rich.print(request)

    deps = AgentDeps(index=index, request=request)
    response = asyncio.run(agent.run(instructions=MOVIE_INSTRUCTIONS, deps=deps))
    rich.print(response.output)


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main, as_positional=False)
