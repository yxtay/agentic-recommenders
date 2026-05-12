from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

if TYPE_CHECKING:
    from agentic_rec.dependencies import UserServiceDep
    from agentic_rec.models import UserResponse

router = APIRouter()


@router.get("/{user_id}")
async def get_user(user_id: str, user_service: UserServiceDep) -> UserResponse:
    """Look up a user by ID."""
    user = user_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user
