from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pydantic_ai
from fastapi import FastAPI
from loguru import logger

from agentic_rec.repositories.base import LanceIndexConfig
from agentic_rec.repositories.item_repository import ItemRepository
from agentic_rec.repositories.user_repository import UserRepository
from agentic_rec.routers import health, info, items, recommendations, users
from agentic_rec.settings import settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load item and user indices, and verify LLM on startup."""
    app.state.items_index = ItemRepository(LanceIndexConfig())
    app.state.items_index.open_table()

    app.state.users_index = UserRepository(
        LanceIndexConfig(table_name=settings.users_table_name)
    )
    app.state.users_index.open_table()

    app.state.llm_ready = await check_llm()
    logger.info(
        "app ready: {} items, {} users, llm_ready={}",
        app.state.items_index.table.count_rows(),
        app.state.users_index.table.count_rows(),
        app.state.llm_ready,
    )
    yield


app = FastAPI(title="Agentic Recommender", lifespan=lifespan)

app.include_router(health.router, tags=["health"])
app.include_router(info.router, tags=["info"])
app.include_router(recommendations.router, tags=["recommendations"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(items.router, prefix="/items", tags=["items"])


def main(limit: int = 5) -> None:
    """Sanity check: hit each route with the TestClient."""
    import random

    import rich
    from fastapi.testclient import TestClient

    # Initialize data/indices if needed
    item_repo = ItemRepository(LanceIndexConfig())
    try:
        item_repo.open_table()
    except ValueError:
        item_repo.index_parquet(settings.items_parquet)

    user_repo = UserRepository(LanceIndexConfig(table_name=settings.users_table_name))
    try:
        user_repo.open_table()
    except ValueError:
        user_repo.index_parquet(settings.users_parquet)

    with TestClient(app, raise_server_exceptions=False) as client:
        user_count = app.state.users_index.table.count_rows()
        user_id = (
            app.state.users_index.table.search()
            .select(["id"])
            .offset(random.randint(0, user_count - 1))
            .limit(1)
            .to_list()[0]["id"]
        )

        logger.info(f"GET /users/{user_id}")
        resp = client.get(f"/users/{user_id}")
        resp.raise_for_status()
        rich.print(resp.json())

        logger.info(f"POST /users/{user_id}/recommend?limit={limit}")
        resp = client.post(f"/users/{user_id}/recommend?limit={limit}")
        resp.raise_for_status()
        rich.print(resp.json())


def cli() -> None:
    from jsonargparse import auto_cli

    auto_cli(main, as_positional=False)


if __name__ == "__main__":
    cli()
