from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_rec.models import ItemResponse

if TYPE_CHECKING:
    import pyarrow as pa

    from agentic_rec.index import LanceIndex


class ItemRepository:
    def __init__(self, index: LanceIndex) -> None:
        self.index = index

    def count_rows(self) -> int:
        if self.index.table is None:
            return 0
        return self.index.table.count_rows()

    def get_by_id(self, item_id: str) -> ItemResponse | None:
        """Look up an item by ID."""
        if self.index.table is None:
            return None
        result = self.index.get_ids([item_id])
        if result.num_rows == 0:
            return None
        return ItemResponse.model_validate(result.to_pylist()[0])

    def get_by_ids(self, item_ids: list[str]) -> pa.Table:
        """Look up multiple items by ID list."""
        return self.index.get_ids(item_ids)

    def search(
        self,
        query: str,
        query_type: str = "hybrid",
        exclude_ids: list[str] | None = None,
        limit: int = 20,
    ) -> pa.Table:
        """Search for items."""
        return self.index.search(
            query, query_type=query_type, exclude_ids=exclude_ids, limit=limit
        )
