from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated

import datasets
import pandas as pd
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
    app.state.index = LanceIndex.load(LanceIndexConfig())
    app.state.users = datasets.Dataset.from_parquet(settings.users_parquet)
    app.state.userid2idx = pd.Series(
        pd.RangeIndex(len(app.state.users)), index=app.state.users["id"]
    )
    app.state.llm_ready = await check_llm()
    logger.info(
        "app ready: {} items, {} users, llm_ready={}",
        app.state.index.table.count_rows(),
        len(app.state.users),
        app.state.llm_ready,
    )
    yield


app = FastAPI(title="Agentic Recommender", lifespan=lifespan)


def get_index() -> LanceIndex:
    return app.state.index


def get_users() -> datasets.Dataset:
    return app.state.users


def get_userid2idx() -> pd.Series:
    return app.state.userid2idx


IndexDep = Annotated[LanceIndex, Depends(get_index)]
UsersDep = Annotated[datasets.Dataset, Depends(get_users)]
UserId2IdxDep = Annotated[pd.Series, Depends(get_userid2idx)]


@app.get("/healthz")
@logger.catch(reraise=True)
async def healthz(index: IndexDep, users: UsersDep) -> dict:
    return {
        "status": "ok",
        "index_ready": index.table is not None,
        "num_items": index.table.count_rows() if index.table else 0,
        "num_users": len(users),
        "llm_ready": app.state.llm_ready,
    }


@app.get("/info")
@logger.catch(reraise=True)
async def get_info() -> InfoResponse:
    return InfoResponse(
        embedder_name=settings.embedder_name,
        reranker_name=settings.reranker_name,
        llm_model=settings.llm_model,
    )


@app.post("/recommend")
@app.post("/recommend/user")
@logger.catch(reraise=True)
async def recommend(request: RecommendRequest, *, index: IndexDep) -> RecommendResponse:
    deps = AgentDeps(index=index, request=request)
    response = await agent.run(instructions=USER_INSTRUCTIONS, deps=deps)
    logger.info("recommend: {} items", len(response.output.items))
    return response.output


@app.post("/recommend/item")
@logger.catch(reraise=True)
async def recommend_item(
    request: RecommendRequest, *, index: IndexDep
) -> RecommendResponse:
    deps = AgentDeps(index=index, request=request)
    response = await agent.run(instructions=ITEM_INSTRUCTIONS, deps=deps)
    logger.info("recommend_item: {} items", len(response.output.items))
    return response.output


@app.get("/users/{user_id}")
@logger.catch(reraise=True)
async def get_user(
    user_id: str, *, users: UsersDep, userid2idx: UserId2IdxDep
) -> UserResponse:
    if user_id not in userid2idx:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    user = users[userid2idx[user_id]]
    return UserResponse.model_validate(user)


@app.post("/users/{user_id}/recommend")
@logger.catch(reraise=True)
async def recommend_user_id(
    user_id: str,
    limit: int = 10,
    *,
    users: UsersDep,
    userid2idx: UserId2IdxDep,
    index: IndexDep,
) -> RecommendResponse:
    user = await get_user(user_id, users=users, userid2idx=userid2idx)
    user.history = user.history[-20:]
    request = RecommendRequest.model_validate({**user.model_dump(), "limit": limit})
    return await recommend(request, index=index)


@app.get("/items/{item_id}")
@logger.catch(reraise=True)
async def get_item(item_id: str, *, index: IndexDep) -> ItemResponse:
    result = index.get_ids([item_id])
    if len(result) == 0:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    return ItemResponse.model_validate(result[0])


@app.post("/items/{item_id}/recommend")
@logger.catch(reraise=True)
async def recommend_item_id(
    item_id: str, limit: int = 10, *, index: IndexDep
) -> RecommendResponse:
    item = await get_item(item_id, index=index)
    request = RecommendRequest.model_validate({**item.model_dump(), "limit": limit})
    return await recommend_item(request, index=index)


def main(limit: int = 5) -> None:
    """Sanity check: hit each route with the TestClient."""
    import random

    import rich
    from fastapi.testclient import TestClient

    import agentic_rec.index

    agentic_rec.index.main(overwrite=False)

    with TestClient(app, raise_server_exceptions=False) as client:
        rich.print("[bold]GET /healthz[/bold]")
        resp = client.get("/healthz")
        resp.raise_for_status()
        rich.print(resp.json())

        rich.print("\n[bold]GET /info[/bold]")
        resp = client.get("/info")
        resp.raise_for_status()
        rich.print(resp.json())

        users = datasets.Dataset.from_parquet(settings.users_parquet)
        user_id = random.choice(users["id"])

        rich.print(f"\n[bold]GET /users/{user_id}[/bold]")
        resp = client.get(f"/users/{user_id}")
        resp.raise_for_status()
        rich.print(resp.json())

        rich.print(f"\n[bold]POST /users/{user_id}/recommend?limit={limit}[/bold]")
        resp = client.post(f"/users/{user_id}/recommend?limit={limit}")
        resp.raise_for_status()
        rich.print(resp.json())

        items = datasets.Dataset.from_parquet(settings.items_parquet)
        item_id = random.choice(items["id"])

        rich.print(f"\n[bold]GET /items/{item_id}[/bold]")
        resp = client.get(f"/items/{item_id}")
        resp.raise_for_status()
        rich.print(resp.json())

        rich.print(f"\n[bold]POST /items/{item_id}/recommend?limit={limit}[/bold]")
        resp = client.post(f"/items/{item_id}/recommend?limit={limit}")
        resp.raise_for_status()
        rich.print(resp.json())


if __name__ == "__main__":
    from jsonargparse import auto_cli

    auto_cli(main, as_positional=False)
