# core/settings.py
import os

from dotenv import load_dotenv

load_dotenv("../.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "advanced_rag_collection")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gemma3:1b")

CRITIC_MODEL = os.getenv("CRITIC_MODEL", "openai/gpt-oss-20b")
