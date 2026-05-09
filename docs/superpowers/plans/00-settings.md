# Settings Implementation Plan

**Goal:** Centralize all configuration (paths, model names, table names) in a single
pydantic-settings class with environment variable override support.

**Architecture:** A `Settings` class using `pydantic-settings` with `env_prefix="AGENTIC_REC_"`.
A module-level `settings` singleton is imported throughout the codebase. Supports `.env` file
loading for local development.

**Tech Stack:** pydantic-settings.

---

## File Map

| File                      | Change                                    |
|---------------------------|-------------------------------------------|
| `pyproject.toml`          | Add `pydantic-settings` to dependencies   |
| `agentic_rec/settings.py` | Settings class with all configuration     |

---

## Tasks

### 1. Add `pydantic-settings` dependency

Add `pydantic-settings~=2.0` to `pyproject.toml` dependencies. Run `uv sync`.

### 2. Implement `Settings` class

`pydantic_settings.BaseSettings` subclass with `SettingsConfigDict`:

- `env_prefix = "AGENTIC_REC_"` — all env vars prefixed.
- `env_file = ".env"` — loads from `.env` file if present.
- `extra = "ignore"` — silently ignore unknown env vars.

Fields:

| Field              | Type  | Default                                                      |
|--------------------|-------|--------------------------------------------------------------|
| `movielens_1m_url` | `str` | `https://files.grouplens.org/datasets/movielens/ml-1m.zip`   |
| `data_dir`         | `str` | `data`                                                       |
| `lance_db_path`    | `str` | `lance_db`                                                   |
| `items_table_name` | `str` | `items`                                                      |
| `users_table_name` | `str` | `users`                                                      |
| `embedder_name`    | `str` | `lightonai/DenseOn`                                          |
| `reranker_name`    | `str` | `lightonai/LateOn`                                           |
| `reranker_type`    | `str` | `pylate`                                                     |
| `llm_model`        | `str` | `cerebras:llama3.1-8b`                                       |

### 3. Add derived path properties

`@property` methods that compute paths from `data_dir`:

- `items_parquet` → `"{data_dir}/ml-1m/items.parquet"`
- `users_parquet` → `"{data_dir}/ml-1m/users.parquet"`
- `events_parquet` → `"{data_dir}/ml-1m/events.parquet"`

### 4. Create module-level singleton

```python
settings = Settings()
```

Imported as `from agentic_rec.settings import settings` by all other modules.

---

## Design Notes

- **Single source of truth**: all other modules import `settings` rather than defining
  their own constants. No `params.py` or scattered defaults.
- **Environment override**: any setting can be changed via env var (e.g.,
  `AGENTIC_REC_LLM_MODEL=anthropic:claude-haiku-4-5`).
- **`.env` support**: developers can create a local `.env` file for API keys and
  model overrides without modifying code.
