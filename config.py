# config.py

import os

# File paths
INDEX_PATH = "college_index.faiss"
METADATA_PATH = "college_metadata.pkl"
DATA_PATH = "college_pages_2"

# Model names
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o"

# FAISS settings
FAISS_TOP_K = 10

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Server settings
SERVER_PORT = 5001
