from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agentic_rec.dependencies import ItemServiceDep
from agentic_rec.models import ItemResponse

router = APIRouter()


@router.get("/{item_id}")
async def get_item(item_id: str, item_service: ItemServiceDep) -> ItemResponse:
    """Look up an item by ID."""
    item = item_service.get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    return item
