from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import TYPE_CHECKING

import pydantic
import pydantic_ai
from pydantic_ai import RunContext

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


SYSTEM_PROMPT = """\
You are a personalized movie recommender. Your job is to recommend {top_k} movies that the user will enjoy.

Workflow:

1. Context understanding: receive user profile (user_text) and interaction history.
    If history is not empty, call get_item_texts with all interacted item IDs.
    Analyze retrieved texts with event values (ratings) to build a preference summary,
    emphasizing recent and highly-rated interactions.

2. Candidate retrieval: call search_items 2-4 times with diverse queries.
    - Use specific genre/theme queries derived from the user's taste profile.
    - Use the user_text directly as one query if useful.
    - Exclude already-interacted item IDs from all search calls.
    - Aim for diversity: vary query angles (genre, mood, era, director style).

3. Cold-start: if history is empty, skip get_item_texts and rely solely on user_text.

4. Ranking with explanations: from all candidates, select the top {top_k} items.
    Rank by relevance and diversity.
    For each item, provide a concise one-sentence explanation of why it suits the user.

Return a RecommendResponse with the ranked list of items.
"""


def create_agent(
    index: LanceIndex, top_k: int = 10
) -> pydantic_ai.Agent[None, RecommendResponse]:
    """Create a pydantic-ai agent with item lookup and search tools bound to *index*.

    Args:
        index: The LanceDB index for item lookup and search.
        top_k: Number of items to recommend; embedded into the system prompt.
    """
    from agentic_rec.params import LLM_MODEL

    agent: pydantic_ai.Agent[None, RecommendResponse] = pydantic_ai.Agent(
        model=LLM_MODEL,
        system_prompt=SYSTEM_PROMPT.format(top_k=top_k),
        output_type=RecommendResponse,
        defer_model_check=True,
    )

    @agent.tool
    def get_item_texts(
        ctx: RunContext[None],  # noqa: ARG001
        item_ids: list[str],
    ) -> dict[str, str]:
        """Look up item texts for the given item IDs.

        Args:
            ctx: The run context.
            item_ids: List of item IDs to look up.
        """
        dataset = index.get_ids(item_ids)
        return {row["id"]: row["text"] for row in dataset}

    @agent.tool
    def search_items(
        ctx: RunContext[None],  # noqa: ARG001
        query: str,
        exclude_ids: list[str] | None = None,
        top_k: int = 20,
    ) -> list[ItemCandidate]:
        """Search for items matching the query using hybrid vector + full-text search.

        Args:
            ctx: The run context.
            query: The search query string describing desired items.
            exclude_ids: Item IDs to exclude from results (e.g. already-interacted items).
            top_k: Maximum number of candidates to return.
        """
        dataset = index.search(query, exclude_ids=exclude_ids, top_k=top_k)
        return [
            ItemCandidate(item_id=row["id"], item_text=row["text"], score=row["score"])
            for row in dataset
        ]

    return agent


async def recommend(request: RecommendRequest, index: LanceIndex) -> RecommendResponse:
    """Run the recommendation agent and return ranked items.

    Args:
        request: The recommendation request with user_text, history, and top_k.
        index: The LanceDB index for item lookup and search.
    """
    agent = create_agent(index, top_k=request.top_k)

    # Sort history by event_timestamp descending (most recent first)
    sorted_history = sorted(
        request.history, key=lambda i: i.event_timestamp, reverse=True
    )

    # Build the user message
    lines: list[str] = [f"User profile: {request.user_text}", ""]

    if sorted_history:
        lines.append("Interaction history (most recent first):")
        lines.extend(
            f"  - item_id={interaction.item_id}"
            f", event={interaction.event_name}"
            f", value={interaction.event_value}"
            f", timestamp={interaction.event_timestamp.isoformat()}"
            for interaction in sorted_history
        )
    else:
        lines.append("No interaction history (cold-start).")

    lines.append("")
    lines.append(f"Please recommend {request.top_k} items.")

    user_message = "\n".join(lines)

    result = await agent.run(user_message)
    return result.output
