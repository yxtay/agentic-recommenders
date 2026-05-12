from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_rec.models import UserResponse

if TYPE_CHECKING:
    from agentic_rec.index import LanceIndex


class UserRepository:
    def __init__(self, index: LanceIndex) -> None:
        self.index = index

    def count_rows(self) -> int:
        if self.index.table is None:
            return 0
        return self.index.table.count_rows()

    def get_by_id(self, user_id: str) -> UserResponse | None:
        """Look up a user by ID."""
        if self.index.table is None:
            return None
        result = self.index.get_ids([user_id])
        if result.num_rows == 0:
            return None
        return UserResponse.model_validate(result.to_pylist()[0])
