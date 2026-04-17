from __future__ import annotations

from typing import Callable, Dict, Optional

from .api_verifier import APIVisionVerifier
from .base import MockVisionVerifier
from .defaults import (
    DEFAULT_API_MAX_OUTPUT_TOKENS,
    DEFAULT_API_TIMEOUT_SEC,
    DEFAULT_GEMINI_ONLINE_TIMEOUT_SEC,
    normalize_cycle_provider,
)
from .gemini_online_verifier import GeminiOnlineVisionVerifier
from .service import ThreadSafeVisionVerifier, get_shared_qwen25_vl_verifier


def build_vision_verifier(
    cycle_cfg: Dict[str, object],
    *,
    preferred_provider: str = "auto",
    local_model_id: str = "",
    api_key: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
    allow_mock_fallback: bool = True,
):
    local_cfg = dict(cycle_cfg.get("local_verifier") or {})
    api_cfg = dict(cycle_cfg.get("api_verifier") or {})
    preferred = str(preferred_provider or "auto").strip().lower() or "auto"
    warning = ""

    def _runtime_meta(**extra) -> Dict[str, object]:
        meta = {
            "verifier_provider": "",
            "verifier_model_id": "",
            "verifier_cached": False,
            "verifier_warning": warning,
        }
        meta.update(extra)
        return meta

    api_provider = str(api_cfg.get("provider", "generic_api") or "generic_api").strip().lower()
    if preferred != "auto":
        preferred = normalize_cycle_provider(preferred, default=preferred)
        if preferred == "manual":
            preferred = "mock"
    web_cfg = dict(cycle_cfg.get("web_verifier") or {})
    web_provider = str(web_cfg.get("provider", "") or "").strip().lower()
    runtime_cfg = dict(cycle_cfg.get("runtime") or {})
    safe_local_only = bool(runtime_cfg.get("safe_local_only", False))
    experimental_providers = {
        str(x or "").strip().lower()
        for x in list(runtime_cfg.get("experimental_providers") or [])
        if str(x or "").strip()
    }
    if preferred == "auto":
        if bool(api_cfg.get("enabled", False)) and api_provider in {"gemini", "google_gemini"}:
            preferred = "gemini_api"
        elif bool(api_cfg.get("enabled", False)) and api_provider in {"openai", "chatgpt"}:
            preferred = "chatgpt_api"
        elif str(local_cfg.get("provider", "mock") or "mock").strip().lower() in {"qwen25_vl", "qwen", "qwen2.5_vl", "qwen2.5-vl"}:
            preferred = "qwen25_vl"
        else:
            preferred = "auto"
    want_api = preferred in {"api", "gemini_api", "chatgpt_api"} or (preferred == "auto" and bool(api_cfg.get("enabled", False)))
    gemini_online_enabled = "gemini_online" in experimental_providers
    want_gemini_online = gemini_online_enabled and (
        preferred == "gemini_online" or (preferred == "auto" and web_provider == "gemini_online")
    )
    force_gemini = preferred == "gemini_api" or api_provider in {"gemini", "google_gemini"}
    force_openai = preferred == "chatgpt_api" or api_provider in {"openai", "chatgpt"}
    want_qwen = preferred == "qwen25_vl" or (
        preferred == "auto"
        and not want_api
        and str(local_cfg.get("provider", "mock") or "mock").strip().lower()
        in {"qwen25_vl", "qwen", "qwen2.5_vl", "qwen2.5-vl"}
    )

    if safe_local_only:
        if preferred in {"api", "gemini_api", "chatgpt_api", "gemini_online"}:
            warning = "runtime.safe_local_only=true blocked external verifier selection; using a local verifier instead."
        elif want_api or want_gemini_online:
            warning = "runtime.safe_local_only=true blocked external verifiers; using a local verifier instead."
        want_api = False
        want_gemini_online = False
        if preferred in {"api", "gemini_api", "chatgpt_api", "gemini_online", "auto"}:
            preferred = "qwen25_vl"
        want_qwen = preferred == "qwen25_vl" or (
            str(local_cfg.get("provider", "mock") or "mock").strip().lower()
            in {"qwen25_vl", "qwen", "qwen2.5_vl", "qwen2.5-vl"}
        )

    def _unified(verifier):
        if isinstance(verifier, ThreadSafeVisionVerifier):
            return verifier
        return ThreadSafeVisionVerifier(verifier)

    if preferred == "mock":
        return _unified(MockVisionVerifier()), _runtime_meta(verifier_provider="mock")

    if preferred == "gemini_online" and not gemini_online_enabled:
        warning = "Gemini Online is experimental and disabled by default; falling back."
    if want_gemini_online:
        try:
            verifier = GeminiOnlineVisionVerifier(
                model=str(web_cfg.get("model", "") or "").strip(),
                timeout_sec=int(
                    web_cfg.get("timeout_sec", DEFAULT_GEMINI_ONLINE_TIMEOUT_SEC)
                    or DEFAULT_GEMINI_ONLINE_TIMEOUT_SEC
                ),
                debug_log=progress_cb,
                user_data_dir=str(web_cfg.get("user_data_dir", "") or "").strip(),
                profile_directory=str(web_cfg.get("profile_directory", "") or "").strip(),
                chrome_binary=str(web_cfg.get("chrome_binary", "") or "").strip(),
                headless=bool(web_cfg.get("headless", False)),
                gemini_url=str(web_cfg.get("gemini_url", "https://gemini.google.com/app") or "https://gemini.google.com/app").strip(),
            )
            return _unified(verifier), _runtime_meta(
                verifier_provider="gemini_online",
                verifier_model_id=str(web_cfg.get("model", "") or "").strip(),
            )
        except Exception as exc:
            warning = f"Gemini online verifier unavailable, falling back: {exc}"
            if preferred == "gemini_online" and not allow_mock_fallback:
                raise

    if want_api:
        try:
            verifier = APIVisionVerifier(
                provider=(
                    "gemini"
                    if force_gemini
                    else ("openai" if force_openai else str(api_cfg.get("provider", "generic_api") or "generic_api"))
                ),
                model=str(api_cfg.get("model", "") or "").strip(),
                base_url=str(api_cfg.get("base_url", "") or "").strip(),
                answer_url=str(api_cfg.get("answer_url", "") or "").strip(),
                caption_url=str(api_cfg.get("caption_url", "") or "").strip(),
                timeout_sec=int(api_cfg.get("timeout_sec", DEFAULT_API_TIMEOUT_SEC) or DEFAULT_API_TIMEOUT_SEC),
                api_key=str(api_key or api_cfg.get("api_key", "") or "").strip(),
                api_key_env=str(api_cfg.get("api_key_env", "IMPACT_API_KEY") or "IMPACT_API_KEY"),
                api_key_header=str(api_cfg.get("api_key_header", "Authorization") or "Authorization"),
                api_key_prefix=str(api_cfg.get("api_key_prefix", "Bearer ") or "Bearer "),
                extra_headers=dict(api_cfg.get("extra_headers") or {}),
                include_image_base64=bool(api_cfg.get("include_image_base64", True)),
                max_output_tokens=int(
                    api_cfg.get("max_output_tokens", DEFAULT_API_MAX_OUTPUT_TOKENS)
                    or DEFAULT_API_MAX_OUTPUT_TOKENS
                ),
                debug_log=progress_cb,
            )
            target_url = str(api_cfg.get("answer_url", "") or api_cfg.get("base_url", "") or "").strip()
            if not target_url and not (force_gemini or force_openai):
                raise RuntimeError("api_verifier requires base_url or answer_url/caption_url.")
            return _unified(verifier), _runtime_meta(
                verifier_provider=(
                    "gemini_api"
                    if force_gemini
                    else ("chatgpt_api" if force_openai else str(api_cfg.get("provider", "generic_api") or "generic_api"))
                ),
                verifier_model_id=str(api_cfg.get("model", "") or "").strip(),
            )
        except Exception as exc:
            warning = f"API verifier unavailable, falling back: {exc}"
            if preferred in {"api", "gemini_api", "chatgpt_api"} and not allow_mock_fallback:
                raise

    if want_qwen:
        try:
            model_id = str(local_model_id or local_cfg.get("model_id") or "Qwen/Qwen2.5-VL-7B-Instruct").strip()
            max_new_tokens = int(local_cfg.get("max_new_tokens", 256) or 256)
            verifier, reused = get_shared_qwen25_vl_verifier(
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                device_map="auto",
            )
            return _unified(verifier), _runtime_meta(
                verifier_provider="qwen25_vl",
                verifier_model_id=model_id,
                verifier_cached=bool(reused),
            )
        except Exception as exc:
            warning = f"Local verifier unavailable, fell back to mock: {exc}"
            if preferred == "qwen25_vl" and not allow_mock_fallback:
                raise

    if allow_mock_fallback:
        return _unified(MockVisionVerifier()), _runtime_meta(verifier_provider="mock")
    raise RuntimeError("No usable verifier could be created from the current cycle configuration.")
