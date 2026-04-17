from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _safe_video_token(video_id: str) -> str:
    token = str(video_id or "").strip()
    return token or "unknown_video"


def detection_record_relpath(video_id: str, frame_idx: int) -> str:
    return os.path.join("frames", _safe_video_token(video_id), f"{int(frame_idx):06d}.json")


def detection_record_path(root_dir: str, video_id: str, frame_idx: int) -> str:
    return os.path.join(os.path.abspath(root_dir), detection_record_relpath(video_id, frame_idx))


def save_detection_record(root_dir: str, video_id: str, frame_idx: int, payload: Dict[str, object]) -> str:
    path = detection_record_path(root_dir, video_id, frame_idx)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    return path


def load_detection_record(root_dir: str, video_id: str, frame_idx: int) -> Dict[str, object]:
    path = detection_record_path(root_dir, video_id, frame_idx)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Detection payload must be a JSON object: {path}")
    payload.setdefault("_loaded_from", path)
    return payload


def flatten_post_threshold_records(payload: Dict[str, object]) -> List[Dict[str, object]]:
    prompt_results = [dict(x) for x in list(payload.get("prompt_results") or []) if isinstance(x, dict)]
    out: List[Dict[str, object]] = []
    if prompt_results:
        for item in prompt_results:
            label = str(item.get("canonical_label", "") or "").strip().lower()
            prompt = str(item.get("prompt", "") or "").strip()
            for rec in list(item.get("post_threshold_records") or []):
                if not isinstance(rec, dict):
                    continue
                row = dict(rec)
                if label:
                    row["canonical_label"] = str(row.get("canonical_label", "") or label).strip().lower()
                if prompt:
                    row["prompt_used"] = str(row.get("prompt_used", "") or prompt)
                out.append(row)
        return out
    for rec in list(payload.get("post_threshold_records") or []):
        if isinstance(rec, dict):
            out.append(dict(rec))
    return out


def summarize_detection_payload(payload: Dict[str, object]) -> Dict[str, object]:
    prompt_results = [dict(x) for x in list(payload.get("prompt_results") or []) if isinstance(x, dict)]
    raw_total = 0
    post_total = 0
    cache_hits = 0
    score_values: List[float] = []
    for item in prompt_results:
        raw_total += int(len(list(item.get("raw_records") or [])))
        post_total += int(len(list(item.get("post_threshold_records") or [])))
        if bool(item.get("cache_hit", False)):
            cache_hits += 1
        for row in list(item.get("raw_records") or []):
            if not isinstance(row, dict):
                continue
            try:
                score_values.append(float(row.get("score", 0.0) or 0.0))
            except Exception:
                continue
    return {
        "prompt_count": len(prompt_results),
        "raw_detection_count": raw_total,
        "post_threshold_count": post_total,
        "cache_hit_prompts": cache_hits,
        "score_min": min(score_values) if score_values else 0.0,
        "score_max": max(score_values) if score_values else 0.0,
    }
