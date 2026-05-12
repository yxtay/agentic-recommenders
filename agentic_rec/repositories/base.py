from __future__ import annotations

import datetime
from functools import cached_property
from typing import Any, Generic, TypeVar, TYPE_CHECKING

import lancedb
import lancedb.table
import pydantic
from loguru import logger
from sqlalchemy import column, literal

from agentic_rec.settings import settings

if TYPE_CHECKING:
    import pyarrow as pa
    from agentic_rec.index import LanceIndex

T = TypeVar("T", bound=pydantic.BaseModel)


class BaseRepository(Generic[T]):
    model_class: type[T]

    def __init__(self, index: LanceIndex) -> None:
        self.index = index

    def get_by_id(self, entity_id: str) -> T | None:
        """Look up an entity by ID."""
        result = self.index.get_ids([entity_id])
        if result.num_rows == 0:
            return None
        return self.model_class.model_validate(result.to_pylist()[0])

    def get_by_ids(self, entity_ids: list[str]) -> pa.Table:
        """Look up multiple entities by ID list."""
        return self.index.get_ids(entity_ids)
