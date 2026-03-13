import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VECTOR_DB_PATH = "vector_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 5
