import copy
import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Any


SHORTCUT_DEFINITIONS: List[Dict[str, str]] = [
    # --- Action Segmentation: General ---
    {
        "id": "action.play_pause",
        "section": "Action Segmentation",
        "scope": "action_general",
        "label": "Play / Pause",
        "default": "Space",
    },
    {
        "id": "action.step_prev",
        "section": "Action Segmentation",
        "scope": "action_general",
        "label": "Step frame -1",
        "default": "A",
    },
    {
        "id": "action.step_next",
        "section": "Action Segmentation",
        "scope": "action_general",
        "label": "Step frame +1",
        "default": "D",
    },
    {
        "id": "action.step_prev_fast",
        "section": "Action Segmentation",
        "scope": "action_general",
        "label": "Step frame -10",
        "default": "Shift+A",
    },
    {
        "id": "action.step_next_fast",
        "section": "Action Segmentation",
        "scope": "action_general",
        "label": "Step frame +10",
        "default": "Shift+D",
    },
    {
        "id": "action.seek_back_1s",
        "section": "Action Segmentation",
        "scope": "action_general",
        "label": "Seek -1 second",
        "default": "J",
    },
    {
        "id": "action.pause",
        "section": "Action Segmentation",
        "scope": "action_general",
        "label": "Pause",
        "default": "K",
    },
    {
        "id": "action.seek_fwd_1s",
        "section": "Action Segmentation",
        "scope": "action_general",
        "label": "Seek +1 second",
        "default": "L",
    },
    {
        "id": "action.jump_start",
        "section": "Action Segmentation",
        "scope": "action_general",
        "label": "Jump to crop start",
        "default": "Home",
    },
    {
        "id": "action.jump_end",
        "section": "Action Segmentation",
        "scope": "action_general",
        "label": "Jump to crop end",
        "default": "End",
    },
    {
        "id": "action.undo",
        "section": "Action Segmentation",
        "scope": "action_general",
        "label": "Undo",
        "default": "Ctrl+Z",
    },
    {
        "id": "action.redo",
        "section": "Action Segmentation",
        "scope": "action_general",
        "label": "Redo",
        "default": "Ctrl+Y",
    },
    {
        "id": "action.adjust_uncertainty",
        "section": "Action Segmentation",
        "scope": "action_general",
        "label": "Adjust uncertainty margin",
        "default": "Ctrl+Shift+U",
    },
    {
        "id": "action.open_settings",
        "section": "Action Segmentation",
        "scope": "action_general",
        "label": "Open settings",
        "default": "Ctrl+,",
    },
    # --- Action Segmentation: Review ---
    {
        "id": "action.review_prev",
        "section": "Action Review",
        "scope": "action_review",
        "label": "Previous review item",
        "default": "Left",
    },
    {
        "id": "action.review_next",
        "section": "Action Review",
        "scope": "action_review",
        "label": "Next review item",
        "default": "Right",
    },
    # --- Action Segmentation: Assisted ---
    {
        "id": "action.assist_nudge_left",
        "section": "Action Assisted Review",
        "scope": "action_assisted",
        "label": "Nudge boundary left",
        "default": "Left",
    },
    {
        "id": "action.assist_nudge_right",
        "section": "Action Assisted Review",
        "scope": "action_assisted",
        "label": "Nudge boundary right",
        "default": "Right",
    },
    {
        "id": "action.assist_confirm",
        "section": "Action Assisted Review",
        "scope": "action_assisted",
        "label": "Confirm boundary",
        "default": "S",
    },
    {
        "id": "action.assist_confirm_down",
        "section": "Action Assisted Review",
        "scope": "action_assisted",
        "label": "Confirm boundary (Down)",
        "default": "Down",
    },
    {
        "id": "action.assist_prev",
        "section": "Action Assisted Review",
        "scope": "action_assisted",
        "label": "Previous assisted point",
        "default": "P",
    },
    {
        "id": "action.assist_next",
        "section": "Action Assisted Review",
        "scope": "action_assisted",
        "label": "Next assisted point",
        "default": "N",
    },
    {
        "id": "action.assist_skip",
        "section": "Action Assisted Review",
        "scope": "action_assisted",
        "label": "Skip assisted point",
        "default": "X",
    },
    {
        "id": "action.assist_merge",
        "section": "Action Assisted Review",
        "scope": "action_assisted",
        "label": "Merge boundary",
        "default": "Backspace",
    },
    {
        "id": "action.assist_merge_delete",
        "section": "Action Assisted Review",
        "scope": "action_assisted",
        "label": "Merge boundary (Delete)",
        "default": "Delete",
    },
    # --- HOI ---
    {
        "id": "hoi.step_prev",
        "section": "HandOI / HOI Detection",
        "scope": "hoi",
        "label": "Step frame -1",
        "default": "Left",
    },
    {
        "id": "hoi.step_next",
        "section": "HandOI / HOI Detection",
        "scope": "hoi",
        "label": "Step frame +1",
        "default": "Right",
    },
    {
        "id": "hoi.seek_prev_second",
        "section": "HandOI / HOI Detection",
        "scope": "hoi",
        "label": "Seek -1 second",
        "default": "Up",
    },
    {
        "id": "hoi.seek_next_second",
        "section": "HandOI / HOI Detection",
        "scope": "hoi",
        "label": "Seek +1 second",
        "default": "Down",
    },
    {
        "id": "hoi.play_pause",
        "section": "HandOI / HOI Detection",
        "scope": "hoi",
        "label": "Play / Pause",
        "default": "Space",
    },
    {
        "id": "hoi.pause",
        "section": "HandOI / HOI Detection",
        "scope": "hoi",
        "label": "Pause",
        "default": "K",
    },
    {
        "id": "hoi.detect",
        "section": "HandOI / HOI Detection",
        "scope": "hoi",
        "label": "Detect current frame",
        "default": "Ctrl+Shift+D",
    },
    {
        "id": "hoi.undo",
        "section": "HandOI / HOI Detection",
        "scope": "hoi",
        "label": "Undo",
        "default": "Ctrl+Z",
    },
    {
        "id": "hoi.redo",
        "section": "HandOI / HOI Detection",
        "scope": "hoi",
        "label": "Redo",
        "default": "Ctrl+Y",
    },
    # --- PSR ---
    {
        "id": "psr.undo",
        "section": "PSR/ASR/ASD",
        "scope": "psr",
        "label": "Undo",
        "default": "Ctrl+Z",
    },
    {
        "id": "psr.redo",
        "section": "PSR/ASR/ASD",
        "scope": "psr",
        "label": "Redo",
        "default": "Ctrl+Y",
    },
    {
        "id": "psr.split_at_playhead",
        "section": "PSR/ASR/ASD",
        "scope": "psr",
        "label": "Split segment at playhead",
        "default": "Ctrl+K",
    },
    {
        "id": "psr.scope_segment",
        "section": "PSR/ASR/ASD",
        "scope": "psr",
        "label": "Set scope to segment",
        "default": "Ctrl+Shift+S",
    },
    {
        "id": "psr.scope_from_here",
        "section": "PSR/ASR/ASD",
        "scope": "psr",
        "label": "Set scope to from-here",
        "default": "Ctrl+Shift+F",
    },
    {
        "id": "psr.reset_segment",
        "section": "PSR/ASR/ASD",
        "scope": "psr",
        "label": "Reset selected segment",
        "default": "Ctrl+Backspace",
    },
    {
        "id": "psr.invert_segment",
        "section": "PSR/ASR/ASD",
        "scope": "psr",
        "label": "Invert selected segment",
        "default": "Ctrl+I",
    },
    {
        "id": "psr.merge_identical",
        "section": "PSR/ASR/ASD",
        "scope": "psr",
        "label": "Merge adjacent identical states",
        "default": "Ctrl+M",
    },
]


def _shortcuts_dir() -> str:
    base = os.environ.get("CVHCI_SETTINGS_DIR")
    if base:
        return os.path.abspath(base)
    return os.path.join(os.path.expanduser("~"), ".cvhci_video_annotation_suite")


def shortcuts_file_path() -> str:
    return os.path.join(_shortcuts_dir(), "shortcuts.json")


def shortcuts_backup_path() -> str:
    return os.path.join(_shortcuts_dir(), "shortcuts.backup.json")


def logging_policy_file_path() -> str:
    return os.path.join(_shortcuts_dir(), "logging_policy.json")


def logging_policy_backup_path() -> str:
    return os.path.join(_shortcuts_dir(), "logging_policy.backup.json")


def _definition_map() -> Dict[str, Dict[str, str]]:
    return {item["id"]: item for item in SHORTCUT_DEFINITIONS}


def default_shortcut_bindings() -> Dict[str, str]:
    return {item["id"]: item["default"] for item in SHORTCUT_DEFINITIONS}


def shortcut_value(
    bindings: Optional[Dict[str, Any]],
    defaults: Optional[Dict[str, str]],
    sid: str,
    default_key: str,
) -> str:
    if isinstance(bindings, dict) and sid in bindings:
        try:
            return str(bindings.get(sid, default_key) or "").strip()
        except Exception:
            return str(default_key or "").strip()
    if isinstance(defaults, dict):
        try:
            return str(defaults.get(sid, default_key) or "").strip()
        except Exception:
            return str(default_key or "").strip()
    return str(default_key or "").strip()


def set_shortcut_key(shortcut: Any, key: str, default_key: str) -> None:
    if shortcut is None:
        return
    try:
        from PyQt5.QtGui import QKeySequence  # lazy import to keep tools lightweight
    except Exception:
        return
    try:
        shortcut.setKey(QKeySequence(key))
    except Exception:
        try:
            shortcut.setKey(QKeySequence(default_key))
        except Exception:
            pass


def shortcut_definitions_by_section() -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for item in SHORTCUT_DEFINITIONS:
        grouped.setdefault(item["section"], []).append(copy.deepcopy(item))
    return grouped


def _normalize_bindings(bindings: Optional[Dict[str, Any]]) -> Dict[str, str]:
    defs = _definition_map()
    out = default_shortcut_bindings()
    if not isinstance(bindings, dict):
        return out
    for sid, val in bindings.items():
        if sid not in defs:
            continue
        txt = ""
        if isinstance(val, str):
            txt = val.strip()
        elif val is None:
            txt = ""
        else:
            txt = str(val).strip()
        out[sid] = txt
    return out


def default_logging_policy() -> Dict[str, bool]:
    return {
        "ops_csv_enabled": False,
        "validation_summary_enabled": True,
        "validation_comment_prompt_enabled": True,
    }


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on", "enabled"}:
            return True
        if text in {"0", "false", "no", "off", "disabled"}:
            return False
    return bool(fallback)


def _normalize_logging_policy(
    data: Optional[Dict[str, Any]], base: Optional[Dict[str, bool]] = None
) -> Dict[str, bool]:
    out = dict(base or default_logging_policy())
    if not isinstance(data, dict):
        return out
    obj = data
    if isinstance(data.get("logging"), dict):
        obj = data.get("logging") or {}

    if "ops_csv_enabled" in obj:
        out["ops_csv_enabled"] = _coerce_bool(
            obj.get("ops_csv_enabled"), out["ops_csv_enabled"]
        )
    elif "oplog_enabled" in obj:
        out["ops_csv_enabled"] = _coerce_bool(
            obj.get("oplog_enabled"), out["ops_csv_enabled"]
        )
    elif "oplog" in obj:
        out["ops_csv_enabled"] = _coerce_bool(obj.get("oplog"), out["ops_csv_enabled"])
    elif "enabled" in obj:
        # Backward compatibility with old single-toggle files.
        out["ops_csv_enabled"] = _coerce_bool(
            obj.get("enabled"), out["ops_csv_enabled"]
        )

    if "validation_summary_enabled" in obj:
        out["validation_summary_enabled"] = _coerce_bool(
            obj.get("validation_summary_enabled"), out["validation_summary_enabled"]
        )
    if "validation_comment_prompt_enabled" in obj:
        out["validation_comment_prompt_enabled"] = _coerce_bool(
            obj.get("validation_comment_prompt_enabled"),
            out["validation_comment_prompt_enabled"],
        )

    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    folder = os.path.dirname(path) or "."
    _ensure_dir(folder)
    fd, tmp_path = tempfile.mkstemp(prefix="shortcuts_", suffix=".json", dir=folder)
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


def load_shortcut_bindings() -> Dict[str, str]:
    for path in (shortcuts_file_path(), shortcuts_backup_path()):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if isinstance(data.get("bindings"), dict):
                    return _normalize_bindings(data.get("bindings"))
                return _normalize_bindings(data)
        except Exception:
            continue
    return default_shortcut_bindings()


def load_logging_policy(
    default_ops_enabled: bool = False,
    default_validation_summary_enabled: bool = True,
) -> Dict[str, bool]:
    base = {
        "ops_csv_enabled": bool(default_ops_enabled),
        "validation_summary_enabled": bool(default_validation_summary_enabled),
        "validation_comment_prompt_enabled": True,
    }
    for path in (logging_policy_file_path(), logging_policy_backup_path()):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return _normalize_logging_policy(data, base=base)
        except Exception:
            continue

    # Backward compatibility: accept legacy logging keys if they were stored in
    # shortcut settings payloads.
    for path in (shortcuts_file_path(), shortcuts_backup_path()):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            if any(
                key in data
                for key in (
                    "logging",
                    "ops_csv_enabled",
                    "oplog_enabled",
                    "validation_summary_enabled",
                    "enabled",
                )
            ):
                return _normalize_logging_policy(data, base=base)
        except Exception:
            continue
    return base


def save_shortcut_bindings(bindings: Dict[str, Any]) -> Tuple[bool, str]:
    payload = {
        "schema": "cvhci.shortcut_bindings.v1",
        "bindings": _normalize_bindings(bindings),
    }
    primary = shortcuts_file_path()
    backup = shortcuts_backup_path()
    try:
        _atomic_write_json(primary, payload)
        _atomic_write_json(backup, payload)
        return True, primary
    except Exception as ex:
        return False, str(ex)


def save_logging_policy(policy: Dict[str, Any]) -> Tuple[bool, str]:
    payload = {
        "schema": "cvhci.logging_policy.v1",
        "logging": _normalize_logging_policy(policy),
    }
    primary = logging_policy_file_path()
    backup = logging_policy_backup_path()
    try:
        _atomic_write_json(primary, payload)
        _atomic_write_json(backup, payload)
        return True, primary
    except Exception as ex:
        return False, str(ex)


def detect_scope_conflicts(
    bindings: Optional[Dict[str, Any]],
) -> Dict[str, List[Tuple[str, List[str]]]]:
    defs = _definition_map()
    normalized = _normalize_bindings(bindings)
    scope_to_key_ids: Dict[str, Dict[str, List[str]]] = {}
    for sid, key in normalized.items():
        key = (key or "").strip()
        if not key:
            continue
        scope = defs.get(sid, {}).get("scope", "")
        if not scope:
            continue
        scope_to_key_ids.setdefault(scope, {}).setdefault(key, []).append(sid)
    conflicts: Dict[str, List[Tuple[str, List[str]]]] = {}
    for scope, mapping in scope_to_key_ids.items():
        for key, ids in mapping.items():
            if len(ids) > 1:
                conflicts.setdefault(scope, []).append((key, sorted(ids)))
    return conflicts


def conflict_messages(bindings: Optional[Dict[str, Any]]) -> List[str]:
    defs = _definition_map()
    messages: List[str] = []
    conflicts = detect_scope_conflicts(bindings)
    for scope, items in conflicts.items():
        for key, ids in items:
            labels = [defs.get(sid, {}).get("label", sid) for sid in ids]
            messages.append(f"[{scope}] {key}: " + ", ".join(labels))
    return messages
