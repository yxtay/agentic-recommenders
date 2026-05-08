from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import datasets
from fastapi import FastAPI, HTTPException

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


@app.get("/info")
async def get_info() -> InfoResponse:
    return InfoResponse(
        embedder_name=settings.embedder_name,
        reranker_name=settings.reranker_name,
        llm_model=settings.llm_model,
    )


@app.post("/recommend")
async def recommend(request: RecommendRequest) -> RecommendResponse:
    deps = AgentDeps(index=app.state.index, request=request)
    response = await agent.run(instructions=USER_INSTRUCTIONS, deps=deps)
    return response.output


@app.post("/recommend/item")
async def recommend_item(request: RecommendRequest) -> RecommendResponse:
    deps = AgentDeps(index=app.state.index, request=request)
    response = await agent.run(instructions=ITEM_INSTRUCTIONS, deps=deps)
    return response.output


@app.get("/users/{user_id}")
async def get_user(user_id: str) -> UserResponse:
    users: datasets.Dataset = app.state.users
    filtered = users.filter(lambda row: row["id"] == user_id)
    if len(filtered) == 0:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return UserResponse.model_validate(filtered[0])


@app.post("/users/{user_id}/recommend")
async def recommend_user_id(user_id: str, limit: int = 10) -> RecommendResponse:
    user = await get_user(user_id)
    user.history = user.history[-20:]
    request = RecommendRequest.model_validate({**user.model_dump(), "limit": limit})
    return await recommend(request)


@app.get("/items/{item_id}")
async def get_item(item_id: str) -> ItemResponse:
    index: LanceIndex = app.state.index
    result = index.get_ids([item_id])
    if len(result) == 0:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    return ItemResponse.model_validate(result[0])


@app.post("/items/{item_id}/recommend")
async def recommend_item_id(item_id: str, limit: int = 10) -> RecommendResponse:
    item = await get_item(item_id)
    request = RecommendRequest.model_validate({**item.model_dump(), "limit": limit})
    return await recommend_item(request)
