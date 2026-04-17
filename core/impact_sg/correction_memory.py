from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional


def default_correction_memory() -> Dict[str, object]:
    return {
        "label_confusions": {},
        "relation_confusions": {},
        "prompt_aliases": {},
        "verified_locks": {},
    }


def normalize_correction_memory(memory: Optional[Dict[str, object]]) -> Dict[str, object]:
    out = default_correction_memory()
    if isinstance(memory, dict):
        out.update(memory)
    return out


def summarize_correction_memory(memory: Optional[Dict[str, object]]) -> Dict[str, int]:
    payload = normalize_correction_memory(memory)
    return {
        "label_confusions": sum(
            int(v)
            for bucket in dict(payload.get("label_confusions") or {}).values()
            for v in dict(bucket or {}).values()
        ),
        "relation_confusions": sum(
            int(v)
            for bucket in dict(payload.get("relation_confusions") or {}).values()
            for v in dict(bucket or {}).values()
        ),
        "prompt_aliases": sum(len(list(bucket or [])) for bucket in dict(payload.get("prompt_aliases") or {}).values()),
        "verified_locks": len(dict(payload.get("verified_locks") or {})),
    }


def load_correction_memory(path: str) -> Dict[str, object]:
    abs_path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(abs_path):
        return default_correction_memory()
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return default_correction_memory()
    if not isinstance(payload, dict):
        return default_correction_memory()
    return normalize_correction_memory(payload)


def save_correction_memory(path: str, memory: Dict[str, object]) -> None:
    abs_path = os.path.abspath(os.path.expanduser(path))
    folder = os.path.dirname(abs_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def _safe_int(value: object, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _bump_nested_counter(table: Dict[str, object], corrected: str, proposed: str) -> Dict[str, object]:
    out = dict(table)
    bucket = dict(out.get(corrected) or {})
    bucket[proposed] = int(bucket.get(proposed, 0)) + 1
    out[corrected] = bucket
    return out


def _merge_counter_tables(*tables: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for table in tables:
        for corrected, raw_bucket in dict(table or {}).items():
            corrected_key = str(corrected or "").strip()
            if not corrected_key:
                continue
            bucket = dict(out.get(corrected_key) or {})
            for proposed, count in dict(raw_bucket or {}).items():
                proposed_key = str(proposed or "").strip()
                if not proposed_key:
                    continue
                bucket[proposed_key] = int(bucket.get(proposed_key, 0)) + max(0, _safe_int(count, 0))
            out[corrected_key] = bucket
    return out


def _merge_alias_tables(*tables: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for table in tables:
        for canonical, raw_aliases in dict(table or {}).items():
            canonical_key = str(canonical or "").strip()
            if not canonical_key:
                continue
            merged = list(out.get(canonical_key) or [])
            for item in list(raw_aliases or []):
                alias = str(item or "").strip()
                if not alias or alias == canonical_key or alias in merged:
                    continue
                merged.append(alias)
            out[canonical_key] = merged
    return out


def _merge_verified_locks(*tables: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for table in tables:
        for subject_id, raw_lock in dict(table or {}).items():
            sid = str(subject_id or "").strip()
            if not sid:
                continue
            lock = dict(raw_lock or {})
            incoming = {
                "status": str(lock.get("status", "") or "confirmed").strip() or "confirmed",
                "frame_start": _safe_int(lock.get("frame_start"), -1),
                "frame_end": _safe_int(lock.get("frame_end"), -1),
            }
            existing = dict(out.get(sid) or {})
            if not existing:
                out[sid] = incoming
                continue
            out[sid] = {
                "status": str(existing.get("status", "") or incoming["status"]).strip() or incoming["status"],
                "frame_start": min(_safe_int(existing.get("frame_start"), incoming["frame_start"]), incoming["frame_start"]),
                "frame_end": max(_safe_int(existing.get("frame_end"), incoming["frame_end"]), incoming["frame_end"]),
            }
    return out


def merge_correction_memories(memories: Iterable[Optional[Dict[str, object]]]) -> Dict[str, object]:
    rows = [normalize_correction_memory(item) for item in list(memories or []) if isinstance(item, dict)]
    if not rows:
        return default_correction_memory()
    return {
        "label_confusions": _merge_counter_tables(*(row.get("label_confusions") or {} for row in rows)),
        "relation_confusions": _merge_counter_tables(*(row.get("relation_confusions") or {} for row in rows)),
        "prompt_aliases": _merge_alias_tables(*(row.get("prompt_aliases") or {} for row in rows)),
        "verified_locks": _merge_verified_locks(*(row.get("verified_locks") or {} for row in rows)),
    }


def prompt_alias_candidates(
    memory: Optional[Dict[str, object]],
    canonical_label: str,
    *,
    limit: int = 4,
) -> List[str]:
    payload = normalize_correction_memory(memory)
    token = str(canonical_label or "").strip()
    if not token:
        return []
    bucket = list((payload.get("prompt_aliases") or {}).get(token) or [])
    out: List[str] = []
    for item in bucket:
        alias = str(item or "").strip()
        if not alias or alias == token or alias in out:
            continue
        out.append(alias)
        if len(out) >= max(1, int(limit)):
            break
    return out


def common_confusions(
    memory: Optional[Dict[str, object]],
    *,
    claim_type: str,
    canonical_value: str,
    limit: int = 3,
) -> List[str]:
    payload = normalize_correction_memory(memory)
    canonical_value = str(canonical_value or "").strip()
    if not canonical_value:
        return []
    claim_type = str(claim_type or "").strip().lower()
    if claim_type == "label":
        table = dict(payload.get("label_confusions") or {})
    elif claim_type == "relation":
        table = dict(payload.get("relation_confusions") or {})
    else:
        return []
    bucket = dict(table.get(canonical_value) or {})
    ranked = sorted(
        (
            (str(name or "").strip(), _safe_int(count, 0))
            for name, count in bucket.items()
            if str(name or "").strip()
        ),
        key=lambda row: (-row[1], row[0]),
    )
    out: List[str] = []
    for name, _count in ranked[: max(1, int(limit))]:
        if name not in out:
            out.append(name)
    return out


def confusion_frequency(
    memory: Optional[Dict[str, object]],
    *,
    claim_type: str,
    canonical_value: str,
) -> int:
    payload = normalize_correction_memory(memory)
    canonical_value = str(canonical_value or "").strip()
    if not canonical_value:
        return 0
    claim_type = str(claim_type or "").strip().lower()
    if claim_type == "label":
        table = dict(payload.get("label_confusions") or {})
    elif claim_type == "relation":
        table = dict(payload.get("relation_confusions") or {})
    else:
        return 0
    bucket = dict(table.get(canonical_value) or {})
    return sum(max(0, _safe_int(value, 0)) for value in bucket.values())


def is_verified_locked(
    memory: Optional[Dict[str, object]],
    *,
    subject_id: str,
    frame_idx: Optional[int],
) -> bool:
    if frame_idx is None:
        return False
    payload = normalize_correction_memory(memory)
    subject_id = str(subject_id or "").strip()
    if not subject_id:
        return False
    lock = dict((payload.get("verified_locks") or {}).get(subject_id) or {})
    if str(lock.get("status", "") or "").strip().lower() != "confirmed":
        return False
    frame_start = _safe_int(lock.get("frame_start"), -1)
    frame_end = _safe_int(lock.get("frame_end"), -1)
    if frame_start < 0 or frame_end < frame_start:
        return False
    return frame_start <= int(frame_idx) <= frame_end


def update_memory_from_human_decision(
    memory: Dict[str, object],
    *,
    claim_type: str,
    proposed: str,
    corrected: str,
    subject_id: str = "",
    frame_start: int = -1,
    frame_end: int = -1,
) -> Dict[str, object]:
    out = default_correction_memory()
    out.update(dict(memory or {}))
    claim_type = str(claim_type or "").strip().lower()
    proposed = str(proposed or "").strip()
    corrected = str(corrected or "").strip()

    if claim_type == "label" and proposed and corrected and proposed != corrected:
        out["label_confusions"] = _bump_nested_counter(
            dict(out.get("label_confusions") or {}),
            corrected,
            proposed,
        )
    elif claim_type == "relation" and proposed and corrected and proposed != corrected:
        out["relation_confusions"] = _bump_nested_counter(
            dict(out.get("relation_confusions") or {}),
            corrected,
            proposed,
        )

    if claim_type == "alias" and corrected:
        aliases = dict(out.get("prompt_aliases") or {})
        bucket = list(aliases.get(corrected) or [])
        if proposed and proposed not in bucket:
            bucket.append(proposed)
        aliases[corrected] = bucket
        out["prompt_aliases"] = aliases

    if subject_id and frame_start >= 0 and frame_end >= frame_start:
        locks = dict(out.get("verified_locks") or {})
        locks[str(subject_id)] = {
            "status": "confirmed",
            "frame_start": int(frame_start),
            "frame_end": int(frame_end),
        }
        out["verified_locks"] = locks

    return out
