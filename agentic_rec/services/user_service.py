from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_rec.models import UserResponse
    from agentic_rec.repositories.user_repository import UserRepository


class UserService:
    def __init__(self, user_repository: UserRepository) -> None:
        self.user_repository = user_repository

    def get_user(self, user_id: str) -> UserResponse | None:
        return self.user_repository.get_by_id(user_id)
