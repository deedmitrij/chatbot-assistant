import os
from pathlib import Path
from dotenv import load_dotenv


# Get the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")

# Database configuration
DB_PATH = os.getenv("DB_PATH", str(PROJECT_ROOT / "pending_requests.db"))

# Vector DB configuration
VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", 1.2))
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000")) if os.getenv("CHROMA_PORT") else None

# Telegram configuration
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_ADMIN_ID = os.getenv("TG_ADMIN_ID")

# HuggingFace configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_BASE_URL = os.getenv("HF_BASE_URL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Knowledge base configuration
FAQ_PATH = PROJECT_ROOT / "knowledge_base.json"
OPERATOR_KNOWLEDGE_PATH = PROJECT_ROOT / "operator_knowledge.json"
