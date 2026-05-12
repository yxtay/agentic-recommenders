from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Request

from agentic_rec.dependencies import get_item_repository, get_user_repository
from agentic_rec.models import HealthResponse
from agentic_rec.settings import settings

if TYPE_CHECKING:
    from agentic_rec.repositories.item_repository import ItemRepository
    from agentic_rec.repositories.user_repository import UserRepository

router = APIRouter()


@router.get("/healthz")
async def healthz(
    request: Request,
    items_repo: Annotated[ItemRepository, Depends(get_item_repository)],
    users_repo: Annotated[UserRepository, Depends(get_user_repository)],
) -> HealthResponse:
    """Return service health and configuration status."""
    return HealthResponse(
        status="ok",
        num_items=items_repo.count_rows(),
        num_users=users_repo.count_rows(),
        llm_ready=request.app.state.llm_ready,
        embedder_name=settings.embedder_name,
        reranker_name=settings.reranker_name,
        llm_model=settings.llm_model,
    )
