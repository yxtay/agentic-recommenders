from __future__ import annotations

from unittest.mock import MagicMock

import pyarrow as pa
import pytest

from agentic_rec.repositories.item_repository import ItemRepository
from agentic_rec.repositories.user_repository import UserRepository
from agentic_rec.repositories.base import LanceIndexConfig


def test_item_repository_get_by_id():
    mock_table = MagicMock()
    mock_table.search().where().to_arrow().drop_columns.return_value = pa.table(
        {"id": ["42"], "text": ["The Matrix"]}
    )
    repo = ItemRepository(LanceIndexConfig())
    repo.table = mock_table
    item = repo.get_by_id("42")
    assert item is not None
    assert item.id == "42"
    assert item.text == "The Matrix"


def test_item_repository_get_by_id_not_found():
    mock_table = MagicMock()
    mock_table.search().where().to_arrow().drop_columns.return_value = pa.table(
        {"id": [], "text": []}
    )
    repo = ItemRepository(LanceIndexConfig())
    repo.table = mock_table
    item = repo.get_by_id("999")
    assert item is None


def test_user_repository_get_by_id():
    mock_table = MagicMock()
    mock_table.search().where().to_arrow().drop_columns.return_value = pa.table(
        {"id": ["1"], "text": ["User 1"], "history": [[]]}
    )
    repo = UserRepository(LanceIndexConfig())
    repo.table = mock_table
    user = repo.get_by_id("1")
    assert user is not None
    assert user.id == "1"
    assert user.text == "User 1"
