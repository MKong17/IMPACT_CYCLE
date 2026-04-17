import json
import re
from typing import Dict, List, Any, Optional


_VERB_TOKENS = {
    "install",
    "remove",
    "mount",
    "dismount",
    "attach",
    "detach",
    "insert",
    "extract",
    "thread",
    "place",
    "fit",
    "seat",
    "align",
    "pull",
    "screw",
    "unscrew",
    "tighten",
    "loosen",
    "take",
    "put",
    "check",
    "browse",
}

_STATE_ERROR_TOKENS = {
    "error",
    "incorrect",
    "wrong",
    "mistake",
    "fault",
    "failure",
    "failed",
    "broken",
    "damage",
    "damaged",
    "misaligned",
    "mismatch",
    "anomaly",
}

_STATE_REMOVE_TOKENS = {
    "remove",
    "removed",
    "detach",
    "detached",
    "disassemble",
    "unmount",
    "uninstall",
    "unscrew",
    "loosen",
    "take",
    "extract",
    "pull",
    "unplug",
    "missing",
    "absent",
}

_STATE_INSTALL_TOKENS = {
    "install",
    "installed",
    "attach",
    "insert",
    "thread",
    "fit",
    "tighten",
    "mount",
    "seat",
}

_STATE_NEUTRAL_TOKENS = {
    "check",
    "inspect",
    "verify",
    "browse",
    "observe",
    "look",
    "hold",
    "transfer",
    "store",
    "pick",
    "move",
    "adjust",
    "align",
    "flip",
    "rotate",
    "turn",
    "place",
    "put",
    "spin",
}

_STATE_REMOVE_PHRASES = {
    "hand_loosen",
    "loosen",
    "unscrew",
    "dismount",
    "detach",
    "remove",
    "unmount",
    "uninstall",
    "extract",
    "pull_out",
    "pull",
    "take_off",
    "takeout",
}

_STATE_INSTALL_PHRASES = {
    "hand_tighten",
    "tighten",
    "install",
    "insert",
    "thread",
    "attach",
    "mount",
    "fit",
    "seat",
}

_STATE_NEUTRAL_PHRASES = {
    "pick_up",
    "pick",
    "hold",
    "transfer",
    "store",
    "adjust",
    "align",
    "flip",
    "rotate",
    "turn",
    "place",
    "put",
    "hand_spin",
    "spin",
    "check",
    "inspect",
    "verify",
    "browse",
}

_STATE_REMOVE_PREFIXES = {
    "dismount",
    "extract",
    "detach",
    "remove",
    "loosen",
    "hand_loosen",
}

_STATE_INSTALL_PREFIXES = {
    "mount",
    "insert",
    "attach",
    "seat",
    "thread",
    "tighten",
    "hand_tighten",
}

_STATE_NEUTRAL_PREFIXES = {
    "pick_up",
    "place",
    "transfer",
    "hold",
    "store",
    "adjust",
    "align",
    "flip",
    "hand_spin",
    "check",
    "inspect",
    "verify",
    "browse",
}


def _normalize_name(text: str) -> str:
    if text is None:
        return ""
    cleaned = str(text).strip().lower()
    cleaned = cleaned.replace("-", " ").replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Backward compatibility for legacy bevel-screw naming.
    cleaned = cleaned.replace("screw bevel", "bearing screw")
    cleaned = cleaned.replace("bevel screw", "bearing screw")
    cleaned = cleaned.replace("screw bearing", "bearing screw")
    cleaned = cleaned.replace("screw plate", "screw adaptor")
    cleaned = cleaned.replace("plate screw", "screw adaptor")
    return cleaned


def _normalize_key(text: str) -> str:
    cleaned = _normalize_name(text)
    return cleaned.replace(" ", "_")


def _state_from_text(value: Any, label_fallback: str = "") -> int:
    if value is None or value == "":
        return -1 if "error" in _normalize_key(label_fallback) else 1
    if isinstance(value, (int, float)):
        val = int(value)
        if val in (-1, 0, 1):
            return val
        return -1 if val < 0 else 1
    txt = str(value).strip().lower()
    if txt in {"-1", "incorrect", "error", "wrong", "misuse"}:
        return -1
    if txt in {"0", "missing", "not_installed", "not installed", "absent"}:
        return 0
    if txt in {"1", "installed", "correct", "ok"}:
        return 1
    return -1 if "error" in _normalize_key(label_fallback) else 1


def _tokenize(text: Any) -> List[str]:
    normalized = _normalize_name(str(text or ""))
    if not normalized:
        return []
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.split(" ") if normalized else []


def _infer_state_from_label_text(label: Any) -> Optional[int]:
    key = _normalize_key(label)
    if not key:
        return None
    # Prefer explicit verb-prefix mapping for HAS labels.
    for prefix in _STATE_REMOVE_PREFIXES:
        if key == prefix or key.startswith(prefix + "_"):
            return 0
    for prefix in _STATE_INSTALL_PREFIXES:
        if key == prefix or key.startswith(prefix + "_"):
            return 1
    for prefix in _STATE_NEUTRAL_PREFIXES:
        if key == prefix or key.startswith(prefix + "_"):
            return None
    if "not_install" in key or "not_installed" in key:
        return 0
    for phrase in _STATE_REMOVE_PHRASES:
        if phrase in key:
            return 0
    for phrase in _STATE_INSTALL_PHRASES:
        if phrase in key:
            return 1
    for phrase in _STATE_NEUTRAL_PHRASES:
        if phrase in key:
            return None
    tokens = key.split("_")
    if not tokens:
        return None
    if any(tok in _STATE_ERROR_TOKENS for tok in tokens):
        return -1
    if any(tok in _STATE_REMOVE_TOKENS for tok in tokens):
        return 0
    if any(tok in _STATE_INSTALL_TOKENS for tok in tokens):
        return 1
    if tokens[0] in _STATE_NEUTRAL_TOKENS:
        return None
    return None


def _is_truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        try:
            return int(value) != 0
        except Exception:
            return bool(value)
    txt = str(value).strip().lower()
    if txt in {"", "0", "false", "none", "null", "no"}:
        return False
    return True


def _segment_implies_error(seg: Dict[str, Any]) -> bool:
    phase = _normalize_key(seg.get("phase", ""))
    if phase in {"anomaly", "error", "incorrect", "wrong", "mistake"}:
        return True
    anomaly = seg.get("anomaly_type")
    if isinstance(anomaly, list):
        return any(_is_truthy_flag(v) for v in anomaly)
    if isinstance(anomaly, dict):
        return any(_is_truthy_flag(v) for v in anomaly.values())
    if anomaly is not None:
        return _is_truthy_flag(anomaly)
    return False


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current_key: Optional[str] = None
    current_item: Optional[Dict[str, Any]] = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip("\n").rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            key = key.strip()
            rest = rest.strip()
            current_key = key
            if rest:
                data[key] = rest.strip("'\"")
                current_item = None
            else:
                data[key] = []
                current_item = None
            continue
        if current_key is None:
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if current_key not in data or not isinstance(data[current_key], list):
                data[current_key] = []
            if ":" in item:
                k, v = item.split(":", 1)
                current_item = {k.strip(): v.strip().strip("'\"")}
                data[current_key].append(current_item)
            else:
                data[current_key].append(item.strip("'\""))
                current_item = None
        elif ":" in stripped and isinstance(data.get(current_key), list):
            k, v = stripped.split(":", 1)
            if current_item is None:
                current_item = {}
                data[current_key].append(current_item)
            current_item[k.strip()] = v.strip().strip("'\"")
    return data


def _load_json_or_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read()
    if path.lower().endswith(".json"):
        return json.loads(text)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        pass
    try:
        return json.loads(text)
    except Exception:
        return _parse_simple_yaml(text)


def load_components(path: str) -> List[Dict[str, Any]]:
    lower = path.lower()
    if lower.endswith(".txt") or lower.endswith(".csv"):
        comps = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "," in line:
                    left, right = line.split(",", 1)
                    key_txt = left.strip()
                    name = right.strip()
                else:
                    parts = line.split()
                    if not parts:
                        continue
                    key_txt = parts[0]
                    name = " ".join(parts[1:]).strip()
                try:
                    cid = int(key_txt)
                except Exception:
                    cid = None
                if not name:
                    name = str(key_txt)
                comps.append({"id": cid, "name": name})
        payload = {"components": comps}
    else:
        payload = _load_json_or_yaml(path)
    raw = payload.get("components", payload)
    comps: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        for key, val in raw.items():
            name = ""
            if isinstance(val, dict):
                name = str(val.get("name") or val.get("component") or key)
                cid = val.get("id", key)
            else:
                name = str(val)
                cid = key
            try:
                cid = int(cid)
            except Exception:
                cid = None
            comps.append({"id": cid, "name": name})
    elif isinstance(raw, list):
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                name = str(
                    item.get("name") or item.get("component") or item.get("label") or ""
                )
                cid = item.get("id", idx)
            else:
                name = str(item)
                cid = idx
            comps.append({"id": cid, "name": name})
    else:
        return []
    # normalize ids
    has_valid_ids = any(isinstance(c.get("id"), int) for c in comps)
    if has_valid_ids:
        comps.sort(key=lambda c: (c.get("id") is None, c.get("id", 0)))
        next_id = (
            max(
                [c.get("id") for c in comps if isinstance(c.get("id"), int)], default=-1
            )
            + 1
        )
        for c in comps:
            if c.get("id") is None:
                c["id"] = next_id
                next_id += 1
    else:
        for idx, c in enumerate(comps):
            c["id"] = idx
    return comps


def load_rules(path: str) -> Dict[str, Dict[str, Any]]:
    payload = _load_json_or_yaml(path)
    raw = payload.get("rules", payload)
    rules: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict):
        for label, entry in raw.items():
            if isinstance(entry, dict):
                if isinstance(entry.get("components"), list):
                    rules[str(label)] = {
                        "components": entry.get("components"),
                        "state": entry.get("state"),
                    }
                else:
                    rules[str(label)] = {
                        "components": [
                            {
                                "component": entry.get("component")
                                or entry.get("name"),
                                "component_id": entry.get("component_id"),
                                "state": entry.get("state"),
                            }
                        ],
                        "state": entry.get("state"),
                    }
            else:
                rules[str(label)] = {
                    "components": [{"component": entry}],
                    "state": None,
                }
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            label = item.get("label") or item.get("action") or item.get("name")
            if not label:
                continue
            if isinstance(item.get("components"), list):
                rules[str(label)] = {
                    "components": item.get("components"),
                    "state": item.get("state"),
                }
            else:
                rules[str(label)] = {
                    "components": [
                        {
                            "component": item.get("component") or item.get("part"),
                            "component_id": item.get("component_id"),
                            "state": item.get("state"),
                        }
                    ],
                    "state": item.get("state"),
                }
    return rules


def extract_components_from_labels(
    labels: List[str], ignore: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    ignore_set = {_normalize_key(x) for x in (ignore or [])}
    seen = set()
    comps: List[Dict[str, Any]] = []
    for label in labels:
        key = _normalize_key(label)
        if not key or key in ignore_set:
            continue
        words = _normalize_name(label).split()
        words = [
            w for w in words if w not in {"error", "incorrect", "wrong", "mistake"}
        ]
        if not words:
            continue
        if words[0] in _VERB_TOKENS and len(words) > 1:
            comp_name = " ".join(words[1:])
        else:
            comp_name = " ".join(words)
        comp_key = _normalize_key(comp_name)
        if comp_key and comp_key not in seen:
            seen.add(comp_key)
            comps.append({"id": len(comps), "name": comp_name})
    return comps


def _match_component(
    label: str, components: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    label_key = _normalize_key(label)
    if not label_key:
        return None
    best = None
    best_len = 0
    for comp in components:
        name = comp.get("name", "")
        comp_key = _normalize_key(name)
        if not comp_key:
            continue
        if comp_key in label_key:
            if len(comp_key) > best_len:
                best = comp
                best_len = len(comp_key)
    return best


def derive_events(
    segments: List[Dict[str, Any]],
    components: List[Dict[str, Any]],
    rules: Dict[str, Dict[str, Any]],
    ignore_labels: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    ignore_set = {_normalize_key(x) for x in (ignore_labels or [])}
    rule_map = {_normalize_key(k): v for k, v in (rules or {}).items()}
    events: List[Dict[str, Any]] = []
    for seg in segments:
        label = seg.get("label")
        if label is None:
            continue
        label_key = _normalize_key(label)
        if label_key in ignore_set:
            continue
        rule = rule_map.get(label_key)
        segment_error = _segment_implies_error(seg)
        inferred_state = _infer_state_from_label_text(label)
        try:
            frame = int(seg.get("end"))
        except Exception:
            continue
        if rule:
            applied = False
            mappings = rule.get("components") or []
            rule_state = rule.get("state")
            for mapping in mappings:
                comp = None
                comp_id = mapping.get("component_id")
                if comp_id is not None:
                    comp = next(
                        (c for c in components if str(c.get("id")) == str(comp_id)),
                        None,
                    )
                if comp is None and mapping.get("component"):
                    comp_name = str(mapping.get("component"))
                    comp = next(
                        (
                            c
                            for c in components
                            if _normalize_key(c.get("name"))
                            == _normalize_key(comp_name)
                        ),
                        None,
                    )
                if comp is None:
                    continue
                state_val = mapping.get("state", rule_state)
                if state_val in (None, ""):
                    if segment_error:
                        state = -1
                    elif inferred_state is None:
                        continue
                    else:
                        state = int(inferred_state)
                else:
                    state = _state_from_text(state_val, label_fallback=str(label))
                if segment_error and state is not None:
                    state = -1
                events.append(
                    {
                        "frame": frame,
                        "label": label,
                        "component_id": comp.get("id"),
                        "component_name": comp.get("name"),
                        "state": state,
                    }
                )
                applied = True
            if applied:
                continue
        comp = _match_component(label, components)
        if comp is None:
            continue
        if segment_error:
            state = -1
        elif inferred_state is None:
            continue
        else:
            state = int(inferred_state)
        events.append(
            {
                "frame": frame,
                "label": label,
                "component_id": comp.get("id"),
                "component_name": comp.get("name"),
                "state": state,
            }
        )
    events.sort(key=lambda x: (x.get("frame", 0), str(x.get("component_id"))))
    return events


def build_state_sequence(
    events: List[Dict[str, Any]],
    components: List[Dict[str, Any]],
    initial_state: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    if not components:
        return []
    if isinstance(initial_state, list) and len(initial_state) == len(components):
        state = []
        for val in initial_state:
            try:
                num = int(val)
            except Exception:
                num = 0
            if num not in (-1, 0, 1):
                num = 0
            state.append(num)
    else:
        state = [0] * len(components)
    seq: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(events):
        frame = events[idx]["frame"]
        while idx < len(events) and events[idx]["frame"] == frame:
            comp_id = events[idx].get("component_id")
            if comp_id is not None:
                try:
                    comp_idx = next(
                        i for i, c in enumerate(components) if c.get("id") == comp_id
                    )
                    state[comp_idx] = int(events[idx].get("state", 0))
                except Exception:
                    pass
            idx += 1
        seq.append({"frame": frame, "state": list(state)})
    return seq


def build_state_runs(
    events: List[Dict[str, Any]],
    components: List[Dict[str, Any]],
    start_frame: int,
    end_frame: int,
    initial_state: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    if not components or end_frame < start_frame:
        return []
    runs: List[Dict[str, Any]] = []
    if isinstance(initial_state, list) and len(initial_state) == len(components):
        state = []
        for val in initial_state:
            try:
                num = int(val)
            except Exception:
                num = 0
            if num not in (-1, 0, 1):
                num = 0
            state.append(num)
    else:
        state = [0] * len(components)
    idx = 0
    # Ignore out-of-window events to keep timeline length aligned to [start_frame, end_frame].
    # This also prevents stale events from earlier videos/ranges from shifting run starts.
    while idx < len(events):
        try:
            frame = int(events[idx].get("frame", 0))
        except Exception:
            idx += 1
            continue
        if frame >= start_frame:
            break
        idx += 1
    current = int(start_frame)
    while idx < len(events):
        try:
            frame = int(events[idx].get("frame", 0))
        except Exception:
            idx += 1
            continue
        if frame > end_frame:
            break
        if frame > current:
            runs.append(
                {
                    "start_frame": current,
                    "end_frame": frame - 1,
                    "state": list(state),
                }
            )
        while idx < len(events) and int(events[idx]["frame"]) == frame:
            comp_id = events[idx].get("component_id")
            if comp_id is not None:
                try:
                    comp_idx = next(
                        i for i, c in enumerate(components) if c.get("id") == comp_id
                    )
                    state[comp_idx] = int(events[idx].get("state", 0))
                except Exception:
                    pass
            idx += 1
        current = frame
    if current <= end_frame:
        runs.append(
            {
                "start_frame": current,
                "end_frame": int(end_frame),
                "state": list(state),
            }
        )
    return runs
