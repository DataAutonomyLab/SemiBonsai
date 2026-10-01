# -*- coding: utf-8 -*-
import os

DELIMITER = "################"

T_LIST = 1
T_ARRT = 2
T_SEMI = 3
T_MIX = 4
T_OTHER = -1

SMALL_TABLE_ROWS = 1
SMALL_TABLE_COLUMNS = 1
BIG_TABLE_ROWS = 8
BIG_TABLE_COLUMNS = 8

DEFAULT_TABLE_NAME = "table"
DEFAULT_SUBTABLE_NAME = "subtable"
DEFAULT_SUBVALUE_NAME = "subvalue"
DEFAULT_SPLIT_SIG = "-"

DIRECTION_KEY = "direction_key"
VLM_SCHEMA_KEY = "vlm_schema_key"
SCHEMA_TOP = True
SCHEMA_LEFT = False
SCHEMA_FAIL = -1

STATUS_END = 1
STATUS_RETRIEVE = 2
STATUS_AGG = 3
STATUS_SPLIT = 4

TAG_DISCRETE = 1
TAG_CONTINUOUS = 2
TAG_TEXT = 3

MAX_ITER_META_INFORMATION_DETECTION = 5
MAX_ITER_PRIMITIVE = 5
MAX_RETRY_HOTREE = 3
MAX_RETRY_PRIMITIVE = 5

# ============================
# Paths (keep as-is if you want)
# ============================
BASE_DIR = os.getenv("SemiBonsai_BASE_DIR")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
LOG_DIR = os.path.join(BASE_DIR, "log")

FONT_PATH = "file://" + os.path.join(BASE_DIR, "static/simfang.ttf")
HTML_CACHE_DIR = os.path.join(CACHE_DIR, "html")
IMAGE_CACHE_DIR = os.path.join(CACHE_DIR, "image")
EXCEL_CACHE_DIR = os.path.join(CACHE_DIR, "excel")
SCHEMA_CACHE_DIR = os.path.join(CACHE_DIR, "schema")
JSON_CACHE_DIR = os.path.join(CACHE_DIR, "json")
OUTPUT_JSON_CACHE_DIR = os.path.join(CACHE_DIR, "output_json")

# ============================
# Model / API Config (NO secrets hardcoded)
# ============================

# ---- LLM (default: OpenAI compatible) ----
# Use environment variables:
#   OPENAI_API_KEY
#   OPENAI_BASE_URL (optional, default https://api.openai.com/v1)
# If you use OpenRouter:
#   OPENROUTER_API_KEY
#   OPENROUTER_BASE_URL (optional, default https://openrouter.ai/api/v1)
#
# You can switch between providers by setting LLM_PROVIDER:
#   LLM_PROVIDER=openai   (default)
#   LLM_PROVIDER=openrouter

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower().strip()

if LLM_PROVIDER == "openrouter":
    LLM_API_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    LLM_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
else:
    LLM_API_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Optional default model (your code sometimes sets this elsewhere)
LLM_MODEL_TYPE = os.getenv("LLM_MODEL_TYPE", "")

# ---- VLM (usually OpenAI-compatible) ----
VLM_API_URL = os.getenv("VLM_API_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
VLM_API_KEY = os.getenv("VLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
VLM_MODEL_TYPE = os.getenv("VLM_MODEL_TYPE", "gpt-4.1")

# ---- Embedding ----
EMBEDDING_TYPE = os.getenv("EMBEDDING_TYPE", "local")  # "api" or "local"


# If api:
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL_TYPE = os.getenv("EMBEDDING_MODEL_TYPE", "")

# ---- Optional: other provider keys (keep but DO NOT hardcode) ----
# Example:
#   VOLC_API_KEY=...
#   VOLC_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
MODEL_API_KEY1 = {
    "DeepSeek-V3": [
        os.getenv("VOLC_API_KEY", ""),
        os.getenv("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        "deepseek-v3-250324",
    ],
    "DeepSeek-R1": [
        os.getenv("VOLC_API_KEY", ""),
        os.getenv("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        "deepseek-r1-250528",
    ],
}

# ============================
# Display model name -> backend model id mapping (no secrets)
# ============================
MODEL_MAP = {
    "GPT-5": "gpt-5",
    "GPT-4o mini": "gpt-4o-mini",
    "GPT-4o": "gpt-4o",
    "GPT-4.1": "gpt-4.1",
    # not public: suggest nearest substitutes
    "GPT-5 mini": "gpt-5-mini",
    "DeepSeek-V3": "DeepSeek-V3",
    "DeepSeek-R1": "DeepSeek-R1",
    "GPT-41-mini": "openai/gpt-4.1-mini",
    "claude-3-5-haiku": "anthropic/claude-3.5-haiku",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "grok-3-mini": "x-ai/grok-3-mini",
    "grok-4": "x-ai/grok-4",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "deepseek-r1-0528": "deepseek/deepseek-r1-0528",
    "deepseek-v3-0324": "deepseek/deepseek-chat-v3-0324",
    "gpt-5-nano": "openai/gpt-5-nano",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    "gpt-4.1-nano": "openai/gpt-4.1-nano",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "claude-haiku-4.5": "anthropic/claude-haiku-4.5",
    "claude-3.5-haiku-20241022": "anthropic/claude-3.5-haiku-20241022",
    "gemini-2.5-flash-lite-preview-09-2025": "google/gemini-2.5-flash-lite-preview-09-2025",
    "gemini-2.5-flash-preview-09-2025": "google/gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash-lite": "google/gemini-2.5-flash-lite",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "gemini-2.0-flash-001": "google/gemini-2.0-flash-001",
    "grok-4-fast": "x-ai/grok-4-fast",
    "grok-code-fast-1": "x-ai/grok-code-fast-1",
    "grok-4.1-fast": "x-ai/grok-4.1-fast",
    "grok-3-mini-beta": "x-ai/grok-3-mini-beta",
    "nova-micro-v1": "amazon/nova-micro-v1",
    "nova-2-lite-v1": "amazon/nova-2-lite-v1",
    "nova-lite-v1": "amazon/nova-lite-v1",
    "nova-pro-v1": "amazon/nova-pro-v1",
    "qwen-plus": "qwen/qwen-plus",
    "qwen-2.5-72b-instruct": "qwen/qwen-2.5-72b-instruct",
    "ministral-3b-2512": "mistralai/ministral-3b-2512",
    "mistral-7b-instruct-v0.3": "mistralai/mistral-7b-instruct-v0.3",
    "ministral-8b": "mistralai/ministral-8b",
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "llama-4-scout": "meta-llama/llama-4-scout",
    "qwen3-30b-a3b": "qwen/qwen3-30b-a3b",
    "qwen3-14b": "qwen/qwen3-14b",
    "qwen3-30b-a3b-instruct-2507": "qwen/qwen3-30b-a3b-instruct-2507",
    "qwen3-32b": "qwen/qwen3-32b",
}