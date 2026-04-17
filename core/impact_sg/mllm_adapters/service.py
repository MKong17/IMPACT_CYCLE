from __future__ import annotations

from threading import RLock
from typing import Callable, Dict, Hashable, Tuple


class ThreadSafeVisionVerifier:
    def __init__(self, verifier) -> None:
        self._verifier = verifier
        self._lock = RLock()

    def answer_probe(self, *, image_path: str, question: str, regions, response_format=None, schema=None):
        with self._lock:
            try:
                return dict(
                    self._verifier.answer_probe(
                        image_path=image_path,
                        question=question,
                        regions=list(regions or []),
                        response_format=dict(response_format or {}) if isinstance(response_format, dict) else response_format,
                        schema=dict(schema or {}) if isinstance(schema, dict) else schema,
                    )
                    or {}
                )
            except TypeError:
                return dict(
                    self._verifier.answer_probe(
                        image_path=image_path,
                        question=question,
                        regions=list(regions or []),
                        response_format=dict(response_format or {}) if isinstance(response_format, dict) else response_format,
                    )
                    or {}
                )

    def answer_probe_batch(self, *, image_path: str, probes):
        with self._lock:
            if not hasattr(self._verifier, "answer_probe_batch"):
                raise AttributeError("wrapped verifier does not implement answer_probe_batch")
            return list(
                self._verifier.answer_probe_batch(
                    image_path=image_path,
                    probes=[
                        dict(item or {})
                        for item in list(probes or [])
                        if isinstance(item, dict)
                    ],
                )
                or []
            )

    def generate_caption(self, *, image_path: str, prompt: str, regions, video_or_frames=None, schema=None):
        with self._lock:
            try:
                return dict(
                    self._verifier.generate_caption(
                        image_path=image_path,
                        prompt=prompt,
                        regions=list(regions or []),
                        video_or_frames=video_or_frames,
                        schema=dict(schema or {}) if isinstance(schema, dict) else schema,
                    )
                    or {}
                )
            except TypeError:
                return dict(
                    self._verifier.generate_caption(
                        image_path=image_path,
                        prompt=prompt,
                        regions=list(regions or []),
                    )
                    or {}
                )


_CACHE_LOCK = RLock()
_VERIFIER_CACHE: Dict[Tuple[Hashable, ...], ThreadSafeVisionVerifier] = {}


def get_or_create_cached_verifier(
    cache_key: Tuple[Hashable, ...],
    factory: Callable[[], object],
) -> tuple[ThreadSafeVisionVerifier, bool]:
    key = tuple(cache_key)
    with _CACHE_LOCK:
        cached = _VERIFIER_CACHE.get(key)
        if cached is not None:
            return cached, True
        verifier = ThreadSafeVisionVerifier(factory())
        _VERIFIER_CACHE[key] = verifier
        return verifier, False


def get_shared_qwen25_vl_verifier(
    *,
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    max_new_tokens: int = 256,
    device_map: str = "auto",
) -> tuple[ThreadSafeVisionVerifier, bool]:
    from .qwen25_vl import Qwen25VLVerifier

    return get_or_create_cached_verifier(
        (
            "qwen25_vl",
            str(model_id or "").strip(),
            int(max_new_tokens),
            str(device_map or "auto").strip(),
        ),
        lambda: Qwen25VLVerifier(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            device_map=device_map,
        ),
    )


def clear_cached_verifiers() -> None:
    with _CACHE_LOCK:
        _VERIFIER_CACHE.clear()


def cached_verifier_count() -> int:
    with _CACHE_LOCK:
        return len(_VERIFIER_CACHE)
