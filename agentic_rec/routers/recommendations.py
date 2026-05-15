from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from loguru import logger

from agentic_rec.dependencies import RecServiceDep  # noqa: TC001
from agentic_rec.models import RecommendRequest, RecommendResponse  # noqa: TC001

router = APIRouter()


@router.post("/recommend")
@router.post("/recommend/user")
@logger.catch(reraise=True)
async def recommend(
    request: RecommendRequest,
    rec_service: RecServiceDep,
    x_cache_ttl: Annotated[int | None, Header()] = None,
) -> RecommendResponse:
    """Generate user-based recommendations via the ARAG agent."""
    return await rec_service.recommend(request, cache_ttl=x_cache_ttl)


@router.post("/recommend/item")
@logger.catch(reraise=True)
async def recommend_item(
    request: RecommendRequest,
    rec_service: RecServiceDep,
    x_cache_ttl: Annotated[int | None, Header()] = None,
) -> RecommendResponse:
    """Generate item-based (similar items) recommendations via the ARAG agent."""
    return await rec_service.recommend_item(request, cache_ttl=x_cache_ttl)


@router.post("/users/{user_id}/recommend")
async def recommend_user_id(
    user_id: str,
    rec_service: RecServiceDep,
    limit: int = 10,
    x_cache_ttl: Annotated[int | None, Header()] = None,
) -> RecommendResponse:
    """Look up a user by ID and generate recommendations."""
    response = await rec_service.recommend_for_user(
        user_id, limit, cache_ttl=x_cache_ttl
    )
    if not response:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return response


@router.post("/items/{item_id}/recommend")
async def recommend_item_id(
    item_id: str,
    rec_service: RecServiceDep,
    limit: int = 10,
    x_cache_ttl: Annotated[int | None, Header()] = None,
) -> RecommendResponse:
    """Look up an item by ID and generate similar-item recommendations."""
    response = await rec_service.recommend_for_item(
        item_id, limit, cache_ttl=x_cache_ttl
    )
    if not response:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    return response
