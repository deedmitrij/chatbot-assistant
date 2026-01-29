import os
from pathlib import Path
from dotenv import load_dotenv


# Get the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")

VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", 0.7))
LLM_CONFIDENCE_THRESHOLD = float(os.getenv("LLM_CONFIDENCE_THRESHOLD", 0.75))

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_ADMIN_ID = os.getenv("TG_ADMIN_ID")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
