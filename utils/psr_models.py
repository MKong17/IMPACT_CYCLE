import json
import os
from typing import Any, Dict, List, Optional


DEFAULT_PSR_MODELS: List[Dict[str, Any]] = [
    {
        "id": "CG15-125BL",
        "display_name": "CG15-125BL",
        "aliases": ["CG15-125BL"],
        "enabled": True,
        "description": "Default assembly-state model profile.",
    },
    {
        "id": "WSG7-115A",
        "display_name": "WSG7-115A",
        "aliases": ["WSG7-115A"],
        "enabled": True,
        "description": "Alternative assembly-state model profile.",
    },
]


def _registry_path(path: Optional[str] = None) -> str:
    if path:
        return str(path)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "configs", "psr_models.json")


def _normalize_aliases(model_id: str, aliases: Any) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in [model_id] + list(aliases or []):
        text = str(raw or "").strip()
        if not text:
            continue
        key = text.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _normalize_model_entry(raw: Any, index: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    model_id = str(
        raw.get("id")
        or raw.get("model_type")
        or raw.get("name")
        or f"model_{index + 1}"
    ).strip()
    if not model_id:
        return None
    display_name = str(
        raw.get("display_name") or raw.get("name") or model_id
    ).strip() or model_id
    aliases = _normalize_aliases(model_id, raw.get("aliases"))
    return {
        "id": model_id,
        "display_name": display_name,
        "aliases": aliases,
        "enabled": bool(raw.get("enabled", True)),
        "description": str(raw.get("description") or "").strip(),
    }


def _coerce_registry(raw_models: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_models, list):
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for index, raw in enumerate(raw_models):
        spec = _normalize_model_entry(raw, index)
        if not spec:
            continue
        key = str(spec["id"]).upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    if not out:
        return []
    if not any(bool(spec.get("enabled", True)) for spec in out):
        out[0]["enabled"] = True
    return out


def load_psr_model_registry(path: Optional[str] = None) -> List[Dict[str, Any]]:
    config_path = _registry_path(path)
    raw_models: Any = None
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                raw_models = payload.get("models")
            else:
                raw_models = payload
        except Exception:
            raw_models = None
    models = _coerce_registry(raw_models)
    if models:
        return models
    return _coerce_registry(DEFAULT_PSR_MODELS)


def enabled_psr_models(
    registry: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    specs = list(registry or load_psr_model_registry())
    enabled = [dict(spec) for spec in specs if bool(spec.get("enabled", True))]
    return enabled or [dict(spec) for spec in specs]


def default_psr_model_id(registry: Optional[List[Dict[str, Any]]] = None) -> str:
    specs = enabled_psr_models(registry)
    if specs:
        return str(specs[0].get("id") or "").strip()
    return str(DEFAULT_PSR_MODELS[0]["id"])


def find_psr_model_spec(
    value: Any,
    registry: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    text = str(value or "").strip()
    if not text:
        return None
    key = text.upper()
    for spec in list(registry or load_psr_model_registry()):
        aliases = [str(alias or "").strip().upper() for alias in spec.get("aliases", [])]
        model_id = str(spec.get("id") or "").strip().upper()
        if key == model_id or key in aliases:
            return dict(spec)
    return None


def normalize_psr_model_type(
    value: Any,
    registry: Optional[List[Dict[str, Any]]] = None,
    *,
    allow_unknown: bool = True,
) -> str:
    text = str(value or "").strip()
    specs = list(registry or load_psr_model_registry())
    if not text:
        return default_psr_model_id(specs)
    spec = find_psr_model_spec(text, specs)
    if spec is not None:
        return str(spec.get("id") or text).strip()
    if allow_unknown:
        return text
    return default_psr_model_id(specs)


def psr_model_display_name(
    value: Any,
    registry: Optional[List[Dict[str, Any]]] = None,
) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    spec = find_psr_model_spec(text, registry)
    if spec is not None:
        return str(spec.get("display_name") or spec.get("id") or text).strip()
    return text
