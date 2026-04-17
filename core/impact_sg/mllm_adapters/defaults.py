from __future__ import annotations

DEFAULT_CYCLE_PROVIDER = "qwen25_vl"
SUPPORTED_CYCLE_PROVIDERS = ("gemini_api", "chatgpt_api", "qwen25_vl", "manual")

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_API_KEY_ENV = "IMPACT_GEMINI_API_KEY"
DEFAULT_GEMINI_ONLINE_TIMEOUT_SEC = 180

DEFAULT_OPENAI_MODEL = "gpt-5.4"
DEFAULT_OPENAI_API_KEY_ENV = "IMPACT_OPENAI_API_KEY"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"

DEFAULT_API_TIMEOUT_SEC = 120
DEFAULT_API_MAX_OUTPUT_TOKENS = 256
LOW_QUOTA_API_MAX_OUTPUT_TOKENS = 128


def normalize_cycle_provider(value: object, *, default: str = DEFAULT_CYCLE_PROVIDER) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return str(default)
    aliases = {
        "gemini": "gemini_api",
        "google_gemini": "gemini_api",
        "chatgpt": "chatgpt_api",
        "openai": "chatgpt_api",
        "qwen": "qwen25_vl",
        "qwen2.5_vl": "qwen25_vl",
        "qwen2.5-vl": "qwen25_vl",
    }
    token = aliases.get(token, token)
    if token in SUPPORTED_CYCLE_PROVIDERS:
        return token
    return str(default)


def cycle_provider_display_name(value: object) -> str:
    token = str(value or "").strip().lower()
    labels = {
        "gemini_api": "Gemini API",
        "gemini_online": "Gemini Online",
        "chatgpt_api": "ChatGPT API",
        "qwen25_vl": "Qwen",
        "manual": "Manual",
        "mock": "Mock",
        "generic_api": "Generic API",
    }
    return labels.get(token, str(value or "").strip() or "Unknown")
