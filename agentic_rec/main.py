from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from loguru import logger

from agentic_rec.agent import check_llm
from agentic_rec.index import LanceIndex, LanceIndexConfig
from agentic_rec.routers import health, info, items, recommendations, users
from agentic_rec.settings import settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load item and user indices, and verify LLM on startup."""
    app.state.items_index = LanceIndex.load(LanceIndexConfig())
    app.state.users_index = LanceIndex.load(
        LanceIndexConfig(table_name=settings.users_table_name)
    )
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

    import agentic_rec.index

    agentic_rec.index.main(overwrite=False)
    agentic_rec.index.main(
        parquet_path=settings.users_parquet,
        table_name=settings.users_table_name,
        overwrite=False,
    )

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
