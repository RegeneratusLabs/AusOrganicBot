import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR.parent / "data"

# LLM — any OpenAI-compatible API
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "2048"))

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000,http://127.0.0.1:3001",
).split(",")

CHROMA_PERSIST_DIR = DATA_DIR / "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Thinking mode — "enabled" or "disabled". DeepSeek defaults to enabled.
# Disable to reduce verbose chain-of-thought reasoning.
THINKING_ENABLED = os.getenv("THINKING_ENABLED", "false").lower() == "true"

# Whether to send the DeepSeek-specific `thinking` param in requests.
# auto = only send for DeepSeek/OpenCode Go (other providers reject unknown params)
# force = always send | off = never send
THINKING_PARAM_MODE = os.getenv("THINKING_PARAM", "auto").lower()

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
