from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Request

from agentic_rec.dependencies import get_item_repository, get_user_repository

if TYPE_CHECKING:
    from agentic_rec.repositories.item_repository import ItemRepository
    from agentic_rec.repositories.user_repository import UserRepository

router = APIRouter()


@router.get("/healthz")
async def healthz(
    request: Request,
    items_repo: Annotated[ItemRepository, Depends(get_item_repository)],
    users_repo: Annotated[UserRepository, Depends(get_user_repository)],
) -> dict:
    """Return service health status."""
    return {
        "status": "ok",
        "num_items": items_repo.count_rows(),
        "num_users": users_repo.count_rows(),
        "llm_ready": request.app.state.llm_ready,
    }
