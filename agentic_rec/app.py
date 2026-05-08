from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated

import datasets
from fastapi import Depends, FastAPI, HTTPException

from agentic_rec.agent import (
    ITEM_INSTRUCTIONS,
    USER_INSTRUCTIONS,
    AgentDeps,
    agent,
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
    yield


app = FastAPI(title="Agentic Recommender", lifespan=lifespan)


def get_index() -> LanceIndex:
    return app.state.index


def get_users() -> datasets.Dataset:
    return app.state.users


IndexDep = Annotated[LanceIndex, Depends(get_index)]
UsersDep = Annotated[datasets.Dataset, Depends(get_users)]


@app.get("/info")
async def get_info() -> InfoResponse:
    return InfoResponse(
        embedder_name=settings.embedder_name,
        reranker_name=settings.reranker_name,
        llm_model=settings.llm_model,
    )


@app.post("/recommend")
async def recommend(request: RecommendRequest, index: IndexDep) -> RecommendResponse:
    deps = AgentDeps(index=index, request=request)
    response = await agent.run(instructions=USER_INSTRUCTIONS, deps=deps)
    return response.output


@app.post("/recommend/item")
async def recommend_item(
    request: RecommendRequest, index: IndexDep
) -> RecommendResponse:
    deps = AgentDeps(index=index, request=request)
    response = await agent.run(instructions=ITEM_INSTRUCTIONS, deps=deps)
    return response.output


@app.get("/users/{user_id}")
async def get_user(user_id: str, users: UsersDep) -> UserResponse:
    filtered = users.filter(lambda row: row["id"] == user_id)
    if len(filtered) == 0:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return UserResponse.model_validate(filtered[0])


@app.post("/users/{user_id}/recommend")
async def recommend_user_id(
    user_id: str, users: UsersDep, index: IndexDep, limit: int = 10
) -> RecommendResponse:
    user = await get_user(user_id, users)
    user.history = user.history[-20:]
    request = RecommendRequest.model_validate({**user.model_dump(), "limit": limit})
    return await recommend(request, index)


@app.get("/items/{item_id}")
async def get_item(item_id: str, index: IndexDep) -> ItemResponse:
    result = index.get_ids([item_id])
    if len(result) == 0:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    return ItemResponse.model_validate(result[0])


@app.post("/items/{item_id}/recommend")
async def recommend_item_id(
    item_id: str, index: IndexDep, limit: int = 10
) -> RecommendResponse:
    item = await get_item(item_id, index)
    request = RecommendRequest.model_validate({**item.model_dump(), "limit": limit})
    return await recommend_item(request, index)
