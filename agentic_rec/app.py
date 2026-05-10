from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, FastAPI, HTTPException
from loguru import logger

from agentic_rec.agent import (
    ITEM_INSTRUCTIONS,
    USER_INSTRUCTIONS,
    AgentDeps,
    agent,
    check_llm,
)
from agentic_rec.index import LanceIndex, LanceIndexConfig
from agentic_rec.models import (
    InfoResponse,
    ItemResponse,
    RecommendRequest,
    RecommendResponse,
    UserResponse,
)
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


def get_items_index() -> LanceIndex:
    """Dependency: return the items LanceIndex from app state."""
    return app.state.items_index


def get_users_index() -> LanceIndex:
    """Dependency: return the users LanceIndex from app state."""
    return app.state.users_index


ItemsIndexDep = Annotated[LanceIndex, Depends(get_items_index)]
UsersIndexDep = Annotated[LanceIndex, Depends(get_users_index)]


@app.get("/healthz")
async def healthz(items_index: ItemsIndexDep, users_index: UsersIndexDep) -> dict:
    """Return service health status."""
    return {
        "status": "ok",
        "index_ready": items_index.table is not None,
        "num_items": items_index.table.count_rows() if items_index.table else 0,
        "num_users": users_index.table.count_rows() if users_index.table else 0,
        "llm_ready": app.state.llm_ready,
    }


@app.get("/info")
async def get_info() -> InfoResponse:
    """Return model configuration info."""
    return InfoResponse(
        embedder_name=settings.embedder_name,
        reranker_name=settings.reranker_name,
        llm_model=settings.llm_model,
    )


@app.post("/recommend")
@app.post("/recommend/user")
@logger.catch(reraise=True)
async def recommend(
    request: RecommendRequest, *, items_index: ItemsIndexDep
) -> RecommendResponse:
    """Generate user-based recommendations via the ARAG agent."""
    deps = AgentDeps(index=items_index, request=request)
    response = await agent.run(instructions=USER_INSTRUCTIONS, deps=deps)
    logger.info("recommend: {} items", len(response.output.items))
    return response.output


@app.post("/recommend/item")
@logger.catch(reraise=True)
async def recommend_item(
    request: RecommendRequest, *, items_index: ItemsIndexDep
) -> RecommendResponse:
    """Generate item-based (similar items) recommendations via the ARAG agent."""
    deps = AgentDeps(index=items_index, request=request)
    response = await agent.run(instructions=ITEM_INSTRUCTIONS, deps=deps)
    logger.info("recommend_item: {} items", len(response.output.items))
    return response.output


@app.get("/users/{user_id}")
async def get_user(user_id: str, *, users_index: UsersIndexDep) -> UserResponse:
    """Look up a user by ID."""
    result = users_index.get_ids([user_id])
    if result.num_rows == 0:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return UserResponse.model_validate(result.to_pylist()[0])


@app.post("/users/{user_id}/recommend")
async def recommend_user_id(
    user_id: str,
    limit: int = 10,
    *,
    items_index: ItemsIndexDep,
    users_index: UsersIndexDep,
) -> RecommendResponse:
    """Look up a user by ID and generate recommendations."""
    user = await get_user(user_id, users_index=users_index)
    user.history = user.history[-20:]
    request = RecommendRequest.model_validate({**user.model_dump(), "limit": limit})
    return await recommend(request, items_index=items_index)


@app.get("/items/{item_id}")
async def get_item(item_id: str, *, items_index: ItemsIndexDep) -> ItemResponse:
    """Look up an item by ID."""
    result = items_index.get_ids([item_id])
    if result.num_rows == 0:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    return ItemResponse.model_validate(result.to_pylist()[0])


@app.post("/items/{item_id}/recommend")
async def recommend_item_id(
    item_id: str, limit: int = 10, *, items_index: ItemsIndexDep
) -> RecommendResponse:
    """Look up an item by ID and generate similar-item recommendations."""
    item = await get_item(item_id, items_index=items_index)
    request = RecommendRequest.model_validate({**item.model_dump(), "limit": limit})
    return await recommend_item(request, items_index=items_index)


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
        rich.print("[bold]GET /healthz[/bold]")
        resp = client.get("/healthz")
        resp.raise_for_status()
        rich.print(resp.json())

        rich.print("\n[bold]GET /info[/bold]")
        resp = client.get("/info")
        resp.raise_for_status()
        rich.print(resp.json())

        user_count = app.state.users_index.table.count_rows()
        user_id = (
            app.state.users_index.table.search()
            .select(["id"])
            .offset(random.randint(0, user_count - 1))
            .limit(1)
            .to_list()[0]["id"]
        )

        rich.print(f"\n[bold]GET /users/{user_id}[/bold]")
        resp = client.get(f"/users/{user_id}")
        resp.raise_for_status()
        rich.print(resp.json())

        rich.print(f"\n[bold]POST /users/{user_id}/recommend?limit={limit}[/bold]")
        resp = client.post(f"/users/{user_id}/recommend?limit={limit}")
        resp.raise_for_status()
        rich.print(resp.json())

        item_count = app.state.items_index.table.count_rows()
        item_id = (
            app.state.items_index.table.search()
            .select(["id"])
            .offset(random.randint(0, item_count - 1))
            .limit(1)
            .to_list()[0]["id"]
        )

        rich.print(f"\n[bold]GET /items/{item_id}[/bold]")
        resp = client.get(f"/items/{item_id}")
        resp.raise_for_status()
        rich.print(resp.json())

        rich.print(f"\n[bold]POST /items/{item_id}/recommend?limit={limit}[/bold]")
        resp = client.post(f"/items/{item_id}/recommend?limit={limit}")
        resp.raise_for_status()
        rich.print(resp.json())


def cli() -> None:
    from jsonargparse import auto_cli

    auto_cli(main, as_positional=False)


if __name__ == "__main__":
    cli()
