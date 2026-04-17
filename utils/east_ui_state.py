import json
import os
import tempfile
from typing import Any, Dict, Tuple


def _settings_dir() -> str:
    base = os.environ.get("CVHCI_SETTINGS_DIR")
    if base:
        return os.path.abspath(base)
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".cvhci_local")
    )


def _state_file_path() -> str:
    return os.path.join(_settings_dir(), "east_ui_state.json")


def _backup_file_path() -> str:
    return os.path.join(_settings_dir(), "east_ui_state.backup.json")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    folder = os.path.dirname(path) or "."
    _ensure_dir(folder)
    fd, tmp_path = tempfile.mkstemp(prefix="east_ui_", suffix=".json", dir=folder)
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def default_east_ui_state() -> Dict[str, Any]:
    return {
        "shared_adapter_path": "",
        "east_ckpt": "",
        "east_cfg": "",
        "text_bank_backend": "",
        "masked_refresh_context_frames": 0,
    }


def _normalize_state(data: Any) -> Dict[str, Any]:
    out = default_east_ui_state()
    if not isinstance(data, dict):
        return out
    obj = data.get("east_ui") if isinstance(data.get("east_ui"), dict) else data
    out["shared_adapter_path"] = str(obj.get("shared_adapter_path") or "").strip()
    out["east_ckpt"] = str(obj.get("east_ckpt") or "").strip()
    out["east_cfg"] = str(obj.get("east_cfg") or "").strip()
    out["text_bank_backend"] = str(obj.get("text_bank_backend") or "").strip()
    try:
        value = int(obj.get("masked_refresh_context_frames", 0) or 0)
    except Exception:
        value = 0
    out["masked_refresh_context_frames"] = max(0, int(value))
    return out


def load_east_ui_state() -> Dict[str, Any]:
    for path in (_state_file_path(), _backup_file_path()):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return _normalize_state(data)
        except Exception:
            continue
    return default_east_ui_state()


def save_east_ui_state(state: Dict[str, Any]) -> Tuple[bool, str]:
    payload = {
        "schema": "cvhci.east_ui_state.v2",
        "east_ui": _normalize_state(state),
    }
    primary = _state_file_path()
    backup = _backup_file_path()
    try:
        _atomic_write_json(primary, payload)
        _atomic_write_json(backup, payload)
        return True, primary
    except Exception as ex:
        return False, str(ex)
