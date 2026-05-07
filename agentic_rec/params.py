# paths
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "data"

# data
ITEMS_PARQUET = "data/ml-1m/items.parquet"
USERS_PARQUET = "data/ml-1m/users.parquet"
EVENTS_PARQUET = "data/ml-1m/events.parquet"

# model
EMBEDDER_NAME = "lightonai/DenseOn"
RERANKER_NAME = "lightonai/LateOn"
RERANKER_TYPE = "pylate"

ITEMS_TABLE_NAME = "items"
LANCE_DB_PATH = "lance_db"
USERS_TABLE_NAME = "users"

# llm
LLM_MODEL = "openai:gpt-4o"
