from __future__ import annotations

from unittest.mock import MagicMock

import pyarrow as pa

from agentic_rec.repositories.item_repository import ItemRepository
from agentic_rec.repositories.user_repository import UserRepository


def test_item_repository_get_by_id() -> None:
    mock_index = MagicMock()
    mock_index.get_ids.return_value = pa.table({"id": ["42"], "text": ["The Matrix"]})
    repo = ItemRepository(mock_index)
    item = repo.get_by_id("42")
    assert item is not None
    assert item.id == "42"
    assert item.text == "The Matrix"
    mock_index.get_ids.assert_called_once_with(["42"])


def test_item_repository_get_by_id_not_found() -> None:
    mock_index = MagicMock()
    mock_index.get_ids.return_value = pa.table({"id": [], "text": []})
    repo = ItemRepository(mock_index)
    item = repo.get_by_id("999")
    assert item is None


def test_user_repository_get_by_id() -> None:
    mock_index = MagicMock()
    mock_index.get_ids.return_value = pa.table(
        {"id": ["1"], "text": ["User 1"], "history": [[]]}
    )
    repo = UserRepository(mock_index)
    user = repo.get_by_id("1")
    assert user is not None
    assert user.id == "1"
    assert user.text == "User 1"
