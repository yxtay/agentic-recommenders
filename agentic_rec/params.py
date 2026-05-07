# paths
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "data"

# data
ITEMS_PARQUET = "data/ml-1m/items.parquet"
USERS_PARQUET = "data/ml-1m/users.parquet"
EVENTS_PARQUET = "data/ml-1m/events.parquet"

# lancedb
LANCE_DB_PATH = "lance_db"
ITEMS_TABLE_NAME = "items"
USERS_TABLE_NAME = "users"
EMBEDDER_NAME = "lightonai/DenseOn"
RERANKER_NAME = "lightonai/LateOn"
RERANKER_TYPE = "pylate"

# llm
LLM_MODEL = "openai:gpt-4o"
