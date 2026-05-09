from __future__ import annotations

import pathlib

import datasets
import numpy as np
import pytest

from agentic_rec.index import LanceIndex, LanceIndexConfig

# ── shared integration fixtures ────────────────────────────────────────────

EMBEDDING_DIM = 768


@pytest.fixture(scope="session")
def test_dataset() -> datasets.Dataset:
    rng = np.random.default_rng(42)
    items = [
        {
            "id": str(i),
            "text": (
                f"Movie {i}: "
                f"{'Comedy' if i % 2 == 0 else 'Drama'} about "
                f"{'love' if i % 3 == 0 else 'adventure'}"
            ),
            "vector": rng.random(EMBEDDING_DIM).astype(np.float32).tolist(),
            "rating": round(rng.uniform(1.0, 5.0), 1),
            "genres": ["Comedy", "Drama"] if i % 2 == 0 else ["Action"],
            "metadata": {"year": 2000 + i % 25, "studio": f"Studio {i % 5}"},
        }
        for i in range(100)
    ]
    return datasets.Dataset.from_list(items)


@pytest.fixture(scope="session")
def lance_config(tmp_path_factory: pytest.TempPathFactory) -> LanceIndexConfig:
    return LanceIndexConfig(
        lancedb_path=str(tmp_path_factory.mktemp("lance_db")),
    )


@pytest.fixture(scope="session")
def indexed_index(
    lance_config: LanceIndexConfig, test_dataset: datasets.Dataset
) -> LanceIndex:
    index = LanceIndex(lance_config)
    index.index_data(test_dataset)
    return index


# ── index data tests ──────────────────────────────────────────────────────


class TestIndexData:
    def test_table_created(
        self, indexed_index: LanceIndex, test_dataset: datasets.Dataset
    ) -> None:
        assert indexed_index.table is not None
        assert indexed_index.table.count_rows() == len(test_dataset)

    def test_table_columns(self, indexed_index: LanceIndex) -> None:
        cols = indexed_index.table.schema.names
        assert "id" in cols
        assert "text" in cols
        assert "vector" in cols
        assert "rating" in cols
        assert "genres" in cols
        assert "metadata" in cols

    def test_overwrite_false_skips(
        self, indexed_index: LanceIndex, test_dataset: datasets.Dataset
    ) -> None:
        original_table = indexed_index.table
        indexed_index.index_data(test_dataset, overwrite=False)
        assert indexed_index.table is original_table


class TestSQLFilters:
    def test_exclude_filter_escapes_injection(self) -> None:
        from sqlalchemy import column, literal

        ids = ["id1", "id2'--injection"]
        expr = column("id").not_in([literal(v) for v in ids])
        filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
        assert "id NOT IN" in filter_str
        assert "'id1'" in filter_str
        # single-quote in value is doubled (SQL escaping), not left raw
        assert "id2''--injection" in filter_str

    def test_include_filter(self) -> None:
        from sqlalchemy import column, literal

        ids = ["a", "b"]
        expr = column("id").in_([literal(v) for v in ids])
        filter_str = str(expr.compile(compile_kwargs={"literal_binds": True}))
        assert "id IN" in filter_str
        assert "'a'" in filter_str
        assert "'b'" in filter_str


class TestSearch:
    def test_returns_dataset(self, indexed_index: LanceIndex) -> None:
        result = indexed_index.search("comedy film about love", limit=5)
        assert isinstance(result, datasets.Dataset)

    def test_limit_respected(self, indexed_index: LanceIndex) -> None:
        result = indexed_index.search("drama about adventure", limit=5)
        assert len(result) <= 5

    def test_exclude_ids(self, indexed_index: LanceIndex) -> None:
        result = indexed_index.search("comedy", exclude_ids=["0", "1", "2"], limit=10)
        returned_ids = result["id"]
        assert "0" not in returned_ids
        assert "1" not in returned_ids
        assert "2" not in returned_ids

    def test_no_exclude_ids_returns_results(self, indexed_index: LanceIndex) -> None:
        result = indexed_index.search("love story", limit=5)
        assert len(result) > 0


class TestGetIds:
    def test_returns_matching_rows(self, indexed_index: LanceIndex) -> None:
        result = indexed_index.get_ids(["0", "1", "2"])
        assert isinstance(result, datasets.Dataset)
        assert set(result["id"]) == {"0", "1", "2"}

    def test_missing_id_not_returned(self, indexed_index: LanceIndex) -> None:
        result = indexed_index.get_ids(["nonexistent-999"])
        assert len(result) == 0

    def test_subset_returned(self, indexed_index: LanceIndex) -> None:
        result = indexed_index.get_ids(["5", "nonexistent-999"])
        assert len(result) == 1
        assert result["id"][0] == "5"


class TestSaveLoad:
    def test_save_creates_directory(
        self, indexed_index: LanceIndex, tmp_path: pathlib.Path
    ) -> None:
        save_path = str(tmp_path / "saved_index")
        indexed_index.save(save_path)
        assert pathlib.Path(save_path).is_dir()

    def test_load_opens_table(
        self, indexed_index: LanceIndex, tmp_path: pathlib.Path
    ) -> None:
        save_path = str(tmp_path / "loaded_index")
        indexed_index.save(save_path)

        config = LanceIndexConfig(
            lancedb_path=save_path,
            table_name=indexed_index.config.table_name,
        )
        loaded = LanceIndex.load(config)
        assert loaded.table is not None
        assert indexed_index.table is not None
        assert loaded.table.count_rows() == indexed_index.table.count_rows()
