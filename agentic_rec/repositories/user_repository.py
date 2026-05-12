from __future__ import annotations

from agentic_rec.models import UserResponse
from agentic_rec.repositories.base import BaseRepository


class UserRepository(BaseRepository[UserResponse]):
    model_class = UserResponse
