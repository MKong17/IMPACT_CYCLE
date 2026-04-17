from .base import MockVisionVerifier, VisionVerifier
from .api_verifier import APIVisionVerifier
from .gemini_online_verifier import GeminiOnlineVisionVerifier
from .factory import build_vision_verifier
from .service import cached_verifier_count, clear_cached_verifiers, get_shared_qwen25_vl_verifier

__all__ = [
    "VisionVerifier",
    "MockVisionVerifier",
    "APIVisionVerifier",
    "GeminiOnlineVisionVerifier",
    "build_vision_verifier",
    "Qwen25VLVerifier",
    "get_shared_qwen25_vl_verifier",
    "clear_cached_verifiers",
    "cached_verifier_count",
]


def __getattr__(name: str):
    if name == "Qwen25VLVerifier":
        from .qwen25_vl import Qwen25VLVerifier

        return Qwen25VLVerifier
    raise AttributeError(name)
