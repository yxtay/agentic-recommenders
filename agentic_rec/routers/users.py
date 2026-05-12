from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agentic_rec.dependencies import UserServiceDep  # noqa: TC001
from agentic_rec.models import UserResponse  # noqa: TC001

router = APIRouter()


@router.get("/{user_id}")
def get_user(user_id: str, user_service: UserServiceDep) -> UserResponse:
    """Look up a user by ID."""
    user = user_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user
