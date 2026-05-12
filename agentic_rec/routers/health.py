from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from agentic_rec.dependencies import get_item_repository, get_user_repository
from agentic_rec.repositories.item_repository import ItemRepository
from agentic_rec.repositories.user_repository import UserRepository

router = APIRouter()


@router.get("/healthz")
async def healthz(
    request: Request,
    items_repo: ItemRepository = Depends(get_item_repository),
    users_repo: UserRepository = Depends(get_user_repository),
) -> dict:
    """Return service health status."""
    return {
        "status": "ok",
        "index_ready": items_repo.index.table is not None,
        "num_items": items_repo.index.table.count_rows() if items_repo.index.table else 0,
        "num_users": users_repo.index.table.count_rows() if users_repo.index.table else 0,
        "llm_ready": request.app.state.llm_ready,
    }
