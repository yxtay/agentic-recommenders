from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request) -> dict:
    """Return service health status."""
    items_index = request.app.state.items_index
    users_index = request.app.state.users_index
    return {
        "status": "ok",
        "index_ready": items_index.table is not None,
        "num_items": items_index.table.count_rows() if items_index.table else 0,
        "num_users": users_index.table.count_rows() if users_index.table else 0,
        "llm_ready": request.app.state.llm_ready,
    }
