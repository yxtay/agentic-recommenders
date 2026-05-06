# paths
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATA_DIR = "data"

# data
ITEMS_PARQUET = "data/ml-1m/items.parquet"
USERS_PARQUET = "data/ml-1m/users.parquet"
EVENTS_PARQUET = "data/ml-1m/events.parquet"

# model
EMBEDDER_NAME = "ibm-granite/granite-embedding-97m-multilingual-r2"
RERANKER_NAME = "mixedbread-ai/mxbai-edge-colbert-v0-32m"
RERANKER_TYPE = "colbert"

ITEMS_TABLE_NAME = "items"
LANCE_DB_PATH = "lance_db"
USERS_TABLE_NAME = "users"
