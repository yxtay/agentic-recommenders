from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_rec.models import ItemResponse
    from agentic_rec.repositories.item_repository import ItemRepository


class ItemService:
    def __init__(self, item_repository: ItemRepository) -> None:
        self.item_repository = item_repository

    def get_item(self, item_id: str) -> ItemResponse | None:
        return self.item_repository.get_by_id(item_id)
