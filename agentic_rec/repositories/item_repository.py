from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_rec.models import ItemResponse
from agentic_rec.repositories.base import BaseRepository

if TYPE_CHECKING:
    import pyarrow as pa


class ItemRepository(BaseRepository[ItemResponse]):
    model_class = ItemResponse

    def search(
        self,
        query: str,
        exclude_ids: list[str] | None = None,
        limit: int = 20,
    ) -> pa.Table:
        """Search for items."""
        return self.index.search(query, exclude_ids=exclude_ids, limit=limit)
