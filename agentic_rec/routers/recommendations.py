from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request
from loguru import logger

from agentic_rec.dependencies import RecServiceDep  # noqa: TC001
from agentic_rec.models import RecommendRequest, RecommendResponse
from agentic_rec.utils.cache import generate_cache_key

router = APIRouter()


@router.post("/recommend")
@router.post("/recommend/user")
@logger.catch(reraise=True)
async def recommend(
    request: RecommendRequest,
    rec_service: RecServiceDep,
    fastapi_request: Request,
    x_cache_ttl: Annotated[int | None, Header()] = None,
) -> RecommendResponse:
    """Generate user-based recommendations via the ARAG agent."""
    if x_cache_ttl is not None:
        cache_key = generate_cache_key(fastapi_request.url.path, body=request)
        if cached := fastapi_request.app.state.response_cache.get(cache_key):
            logger.info("Cache hit for {}", fastapi_request.url.path)
            return RecommendResponse.model_validate(cached)

    response = await rec_service.recommend(request)

    if x_cache_ttl is not None:
        cache_key = generate_cache_key(fastapi_request.url.path, body=request)
        fastapi_request.app.state.response_cache.set(
            cache_key, response.model_dump(), x_cache_ttl
        )

    return response


@router.post("/recommend/item")
@logger.catch(reraise=True)
async def recommend_item(
    request: RecommendRequest,
    rec_service: RecServiceDep,
    fastapi_request: Request,
    x_cache_ttl: Annotated[int | None, Header()] = None,
) -> RecommendResponse:
    """Generate item-based (similar items) recommendations via the ARAG agent."""
    if x_cache_ttl is not None:
        cache_key = generate_cache_key(fastapi_request.url.path, body=request)
        if cached := fastapi_request.app.state.response_cache.get(cache_key):
            logger.info("Cache hit for {}", fastapi_request.url.path)
            return RecommendResponse.model_validate(cached)

    response = await rec_service.recommend_item(request)

    if x_cache_ttl is not None:
        cache_key = generate_cache_key(fastapi_request.url.path, body=request)
        fastapi_request.app.state.response_cache.set(
            cache_key, response.model_dump(), x_cache_ttl
        )

    return response


@router.post("/users/{user_id}/recommend")
async def recommend_user_id(
    user_id: str,
    rec_service: RecServiceDep,
    fastapi_request: Request,
    limit: int = 10,
    x_cache_ttl: Annotated[int | None, Header()] = None,
) -> RecommendResponse:
    """Look up a user by ID and generate recommendations."""
    if x_cache_ttl is not None:
        cache_key = generate_cache_key(
            fastapi_request.url.path, params={"limit": limit}
        )
        if cached := fastapi_request.app.state.response_cache.get(cache_key):
            logger.info("Cache hit for {}", fastapi_request.url.path)
            return RecommendResponse.model_validate(cached)

    response = await rec_service.recommend_for_user(user_id, limit)
    if not response:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    if x_cache_ttl is not None:
        cache_key = generate_cache_key(
            fastapi_request.url.path, params={"limit": limit}
        )
        fastapi_request.app.state.response_cache.set(
            cache_key, response.model_dump(), x_cache_ttl
        )

    return response


@router.post("/items/{item_id}/recommend")
async def recommend_item_id(
    item_id: str,
    rec_service: RecServiceDep,
    fastapi_request: Request,
    limit: int = 10,
    x_cache_ttl: Annotated[int | None, Header()] = None,
) -> RecommendResponse:
    """Look up an item by ID and generate similar-item recommendations."""
    if x_cache_ttl is not None:
        cache_key = generate_cache_key(
            fastapi_request.url.path, params={"limit": limit}
        )
        if cached := fastapi_request.app.state.response_cache.get(cache_key):
            logger.info("Cache hit for {}", fastapi_request.url.path)
            return RecommendResponse.model_validate(cached)

    response = await rec_service.recommend_for_item(item_id, limit)
    if not response:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    if x_cache_ttl is not None:
        cache_key = generate_cache_key(
            fastapi_request.url.path, params={"limit": limit}
        )
        fastapi_request.app.state.response_cache.set(
            cache_key, response.model_dump(), x_cache_ttl
        )

    return response
