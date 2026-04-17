import bisect
import hashlib
import json
import os
import pickle
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch  # type: ignore
except Exception:
    torch = None


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _safe_load_object(path: str) -> Any:
    if torch is not None:
        try:
            return torch.load(path, map_location="cpu")
        except TypeError:
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                pass
        except Exception:
            try:
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                pass
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _safe_save_object(path: str, payload: Any) -> None:
    if torch is not None:
        try:
            torch.save(payload, path)
            return
        except Exception:
            pass
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _stable_records_hash(records: Sequence[Dict[str, Any]]) -> str:
    try:
        blob = json.dumps(
            list(records or []),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except Exception:
        blob = repr(list(records or [])).encode("utf-8", errors="ignore")
    return hashlib.sha1(blob).hexdigest()


def _infer_feature_table(feat: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if feat.ndim != 2:
        raise ValueError(f"features.npy must be 2D, got {feat.shape}")
    if meta and isinstance(meta.get("feature_dim"), (int, float)):
        dim = int(meta.get("feature_dim"))
        if feat.shape[0] == dim and feat.shape[1] != dim:
            return feat.T
        if feat.shape[1] == dim and feat.shape[0] != dim:
            return feat
    typical_dims = {256, 512, 768, 1024, 1536, 2048, 3072, 4096}
    if feat.shape[0] in typical_dims and feat.shape[1] not in typical_dims:
        return feat.T
    if feat.shape[1] in typical_dims and feat.shape[0] not in typical_dims:
        return feat
    return feat


def _build_frame_map(seq_len: int, meta: Optional[Dict[str, Any]] = None) -> List[int]:
    if meta:
        picked = meta.get("picked_indices")
        if isinstance(picked, list) and len(picked) == seq_len:
            try:
                return [int(x) for x in picked]
            except Exception:
                pass
        stride = meta.get("frame_stride")
        if stride is not None:
            try:
                stride = int(stride)
                return [i * stride for i in range(seq_len)]
            except Exception:
                pass
    return list(range(seq_len))


def _load_feature_series(features_dir: str) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    feat_path = os.path.join(features_dir, "features.npy")
    if not os.path.isfile(feat_path):
        raise FileNotFoundError(f"features.npy not found: {feat_path}")
    meta = _load_json(os.path.join(features_dir, "meta.json"))
    feat = np.load(feat_path, mmap_mode="r")
    table = _infer_feature_table(feat, meta=meta)
    seq_len = int(table.shape[0])
    frame_map = _build_frame_map(seq_len, meta=meta)
    if len(frame_map) > seq_len:
        frame_map = frame_map[:seq_len]
    elif len(frame_map) < seq_len:
        frame_map = list(frame_map) + list(range(len(frame_map), seq_len))
    return table, frame_map, meta


def _normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec or {})
    try:
        out["feedback_start"] = int(out.get("feedback_start", out.get("start", 0)))
    except Exception:
        out["feedback_start"] = 0
    try:
        out["feedback_end"] = int(out.get("feedback_end", out.get("end", out["feedback_start"])))
    except Exception:
        out["feedback_end"] = int(out["feedback_start"])
    if out["feedback_end"] < out["feedback_start"]:
        out["feedback_start"], out["feedback_end"] = out["feedback_end"], out["feedback_start"]
    out["point_type"] = str(out.get("point_type", "") or "").strip().lower()
    out["label"] = str(out.get("label", "") or "").strip()
    out["view_name"] = str(out.get("view_name", "") or "").strip()
    out["confirmed_kind"] = str(out.get("confirmed_kind", "") or "").strip().lower()
    return out


def _record_buffer_path(runtime_dir: str, *, finalized: bool = False) -> str:
    return os.path.join(
        runtime_dir,
        "finalized_record_buffer.pkl" if finalized else "record_buffer.pkl",
    )


def _is_runtime_finalized(runtime_dir: str) -> bool:
    meta = _load_json(os.path.join(runtime_dir, "meta.json"))
    return bool(meta.get("video_finalized", False))


def _load_record_buffer(
    runtime_dir: str,
    *,
    finalized: bool = False,
) -> List[Dict[str, Any]]:
    path = _record_buffer_path(runtime_dir, finalized=finalized)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    return [_normalize_record(x) for x in obj if isinstance(x, dict)]


def _load_supervision_records(runtime_dir: str) -> Tuple[List[Dict[str, Any]], str]:
    if _is_runtime_finalized(runtime_dir):
        rows = _load_record_buffer(runtime_dir, finalized=True)
        if rows:
            return rows, "finalized"
    return _load_record_buffer(runtime_dir, finalized=False), "confirmed"


def _confirmed_kind(rec: Dict[str, Any]) -> str:
    kind = str(rec.get("confirmed_kind", "") or "").strip().lower()
    if kind:
        return kind
    action_kind = str(rec.get("action_kind", "") or "").strip().lower()
    if action_kind.endswith("_accept"):
        return "accepted"
    return "corrected"


def _record_span_length(rec: Dict[str, Any]) -> int:
    try:
        s = int(rec.get("feedback_start", 0))
        e = int(rec.get("feedback_end", s))
    except Exception:
        return 0
    if e < s:
        s, e = e, s
    return max(0, int(e - s + 1))


def _record_quality_factor(rec: Dict[str, Any], *, finalized_video: bool = False) -> float:
    kind = _confirmed_kind(rec)
    action_kind = str(rec.get("action_kind", "") or "").strip().lower()
    span_len = _record_span_length(rec)
    coverage = 1.0
    if span_len > 0:
        coverage = 0.9 + min(0.22, float(np.log1p(span_len)) / 12.0)
    edit = 1.0
    if action_kind in {"label_replace", "boundary_move"}:
        edit += 0.12
    elif action_kind in {"label_remove", "boundary_add", "boundary_remove"}:
        edit += 0.08
    elif action_kind in {"label_assign", "transition_add", "transition_remove"}:
        edit += 0.05
    elif action_kind.endswith("_finalize"):
        edit += 0.10
    if action_kind == "boundary_move":
        try:
            old_frame = int(rec.get("old_boundary_frame", rec.get("boundary_frame", 0)) or 0)
            new_frame = int(rec.get("boundary_frame", old_frame) or old_frame)
            delta = abs(int(new_frame) - int(old_frame))
            edit += min(0.10, float(delta) / 40.0)
        except Exception:
            pass
    neg_count = len([x for x in (rec.get("hard_negative_labels") or []) if str(x or "").strip()])
    if kind != "accepted" and neg_count > 0:
        edit += min(0.09, 0.03 * float(neg_count))
    if kind == "accepted":
        edit = min(edit, 1.03)
    if finalized_video and kind == "finalized":
        coverage *= 1.05
    return float(max(0.85, min(1.40, coverage * edit)))


def _supervision_weight(rec: Dict[str, Any], *, finalized_video: bool = False) -> float:
    raw = rec.get("training_weight")
    if raw is not None:
        try:
            val = float(raw)
            if np.isfinite(val) and val > 0.0:
                return float(val)
        except Exception:
            pass
    kind = _confirmed_kind(rec)
    defaults = {
        "accepted": 0.7,
        "corrected": 1.0,
        "finalized": 1.35,
    }
    env_keys = {
        "accepted": "EAST_ACCEPTED_WEIGHT",
        "corrected": "EAST_CORRECTED_WEIGHT",
        "finalized": "EAST_FINALIZED_WEIGHT",
    }
    base = defaults.get(kind, defaults["corrected"])
    try:
        env_raw = str(os.environ.get(env_keys.get(kind, ""), "") or "").strip()
        if env_raw:
            env_val = float(env_raw)
            if np.isfinite(env_val) and env_val > 0.0:
                base = float(env_val)
    except Exception:
        pass
    if finalized_video and kind != "finalized":
        try:
            boost = float(str(os.environ.get("EAST_FINALIZED_VIDEO_WEIGHT_BOOST", "1.15") or "1.15").strip())
            if np.isfinite(boost) and boost > 0.0:
                base *= float(boost)
        except Exception:
            pass
    return float(max(0.05, base * _record_quality_factor(rec, finalized_video=finalized_video)))


def _segment_embedding_for_span(
    feature_table: np.ndarray,
    frame_map: Sequence[int],
    start: int,
    end: int,
    trim_ratio: float = 0.1,
) -> Optional[np.ndarray]:
    if feature_table.ndim != 2 or feature_table.shape[0] <= 0:
        return None
    try:
        s = int(start)
        e = int(end)
    except Exception:
        return None
    if e < s:
        s, e = e, s
    length = int(e - s + 1)
    if length <= 0:
        return None
    trim_ratio = max(0.0, float(trim_ratio))
    delta = int(trim_ratio * length)
    s2 = s + delta
    e2 = e - delta
    if e2 < s2:
        s2, e2 = s, e
    fmap = list(frame_map or [])
    if not fmap:
        return None
    data_len = int(feature_table.shape[0])
    if len(fmap) > data_len:
        fmap = fmap[:data_len]
    idx_start = bisect.bisect_left(fmap, s2)
    idx_end = bisect.bisect_right(fmap, e2) - 1
    if idx_start > idx_end:
        idx = max(0, min(idx_start, len(fmap) - 1))
        idx_start = idx_end = idx
    seg = feature_table[idx_start : idx_end + 1]
    if getattr(seg, "size", 0) <= 0:
        return None
    emb = np.asarray(np.mean(seg, axis=0), dtype=np.float32).reshape(-1)
    if emb.size == 0:
        return None
    norm = float(np.linalg.norm(emb))
    if norm > 0.0:
        emb = emb / norm
    return emb


def _mark_runtime_dirty(
    runtime_dir: str,
    *,
    record_hash: str,
    label_hash: str,
    update_step: int,
    record_count: int,
    label_record_count: int,
) -> bool:
    meta_path = os.path.join(runtime_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return False
    meta = _load_json(meta_path)
    meta["runtime_dirty"] = True
    meta["correction_record_hash"] = str(record_hash or "")
    meta["correction_label_hash"] = str(label_hash or "")
    meta["correction_step"] = int(update_step)
    meta["correction_record_count"] = int(record_count)
    meta["correction_label_record_count"] = int(label_record_count)
    meta["correction_updated_at"] = _utc_now_iso()
    _save_json(meta_path, meta)
    return True


def rebuild_runtime_model_delta(
    features_dir: str,
    *,
    trim_ratio: float = 0.1,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    def _emit(msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(str(msg))
            except Exception:
                pass

    features_dir = os.path.abspath(os.path.expanduser(str(features_dir or "")))
    runtime_dir = os.path.join(features_dir, "east_runtime")
    os.makedirs(runtime_dir, exist_ok=True)
    delta_path = os.path.join(runtime_dir, "model_delta.pt")

    records, supervision_source = _load_supervision_records(runtime_dir)
    finalized_video = supervision_source == "finalized"
    action_counts: Dict[str, int] = {}
    for rec in records:
        key = str(rec.get("action_kind", rec.get("point_type", "unknown")) or "unknown").strip() or "unknown"
        action_counts[key] = int(action_counts.get(key, 0)) + 1
    label_records = [
        rec
        for rec in records
        if str(rec.get("point_type", "") or "").strip().lower() == "label"
        and str(rec.get("label", "") or "").strip()
    ]
    transition_records = [
        rec
        for rec in records
        if str(rec.get("point_type", "") or "").strip().lower() == "transition"
        and str(rec.get("from_label", "") or "").strip()
        and str(rec.get("to_label", "") or "").strip()
    ]
    record_hash = _stable_records_hash(records)
    label_hash = _stable_records_hash(label_records)

    prev_obj = _safe_load_object(delta_path) if os.path.isfile(delta_path) else {}
    if not isinstance(prev_obj, dict):
        prev_obj = {}
    prev_record_hash = str(prev_obj.get("last_record_hash", "") or "")
    prev_label_hash = str(prev_obj.get("last_label_record_hash", "") or "")
    try:
        prev_step = int(prev_obj.get("step", 0))
    except Exception:
        prev_step = 0

    unchanged = (
        os.path.isfile(delta_path)
        and record_hash == prev_record_hash
        and label_hash == prev_label_hash
    )
    if unchanged:
        return {
            "ok": True,
            "changed": False,
            "reason": "unchanged",
            "features_dir": features_dir,
            "runtime_dir": runtime_dir,
            "supervision_source": str(supervision_source),
            "video_finalized": bool(finalized_video),
            "record_count": int(len(records)),
            "label_record_count": int(len(label_records)),
            "transition_record_count": int(len(transition_records)),
            "prototype_count": int(len(dict(prev_obj.get("prototypes") or {}))),
            "action_counts": dict(action_counts),
            "step": int(prev_step),
        }

    _emit("[EAST][ONLINE] Loading feature table...")
    feature_table, frame_map, meta = _load_feature_series(features_dir)
    feature_dim = int(feature_table.shape[1]) if feature_table.ndim == 2 else 0

    label_sums: Dict[str, np.ndarray] = {}
    label_counts: Dict[str, int] = {}
    used_label_records = 0
    skipped_label_records = 0
    _emit(
        f"[EAST][ONLINE] Rebuilding label prototypes from {len(label_records)} confirmed label corrections..."
    )
    for rec in label_records:
        rec_weight = _supervision_weight(rec, finalized_video=finalized_video)
        emb = _segment_embedding_for_span(
            feature_table,
            frame_map,
            int(rec.get("feedback_start", 0)),
            int(rec.get("feedback_end", 0)),
            trim_ratio=float(trim_ratio),
        )
        if emb is None or emb.size <= 0:
            skipped_label_records += 1
            continue
        label = str(rec.get("label", "") or "").strip()
        if not label:
            skipped_label_records += 1
            continue
        if label not in label_sums:
            label_sums[label] = np.zeros_like(emb, dtype=np.float32)
            label_counts[label] = 0.0
        label_sums[label] = np.asarray(label_sums[label], dtype=np.float32) + (emb * float(rec_weight))
        label_counts[label] = float(label_counts.get(label, 0.0) + float(rec_weight))
        used_label_records += 1

    prototypes: Dict[str, np.ndarray] = {}
    for label, vec in label_sums.items():
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm > 0.0:
            arr = arr / norm
        if arr.size > 0:
            prototypes[str(label)] = arr.astype(np.float32)

    transition_delta: Dict[str, Dict[str, float]] = {}
    confusion_delta: Dict[str, Dict[str, float]] = {}
    for rec in transition_records:
        src = str(rec.get("from_label", "") or "").strip()
        dst = str(rec.get("to_label", "") or "").strip()
        if not src or not dst:
            continue
        try:
            amount = float(rec.get("count", 1) or 1.0)
        except Exception:
            amount = 1.0
        if amount <= 0.0:
            continue
        sign = 1.0
        if str(rec.get("action_kind", "") or "").strip().lower() == "transition_remove":
            sign = -1.0
        amount *= float(_supervision_weight(rec, finalized_video=finalized_video))
        bucket = transition_delta.setdefault(src, {})
        bucket[dst] = float(bucket.get(dst, 0.0) + sign * amount)
    transition_delta = {
        str(src): {
            str(dst): float(val)
            for dst, val in row.items()
            if abs(float(val)) > 1e-6
        }
        for src, row in transition_delta.items()
        if any(abs(float(val)) > 1e-6 for val in row.values())
    }

    for rec in label_records:
        pos = str(rec.get("label", "") or "").strip()
        if not pos:
            continue
        rec_weight = float(_supervision_weight(rec, finalized_video=finalized_video))
        neg_weights: Dict[str, float] = {}
        old_label = str(rec.get("old_label", "") or "").strip()
        action_kind = str(rec.get("action_kind", "") or "").strip().lower()
        if (
            old_label
            and old_label != pos
            and action_kind in {"label_replace"}
        ):
            neg_weights[old_label] = float(neg_weights.get(old_label, 0.0) + 2.0 * rec_weight)
        for raw_name in rec.get("hard_negative_labels") or []:
            name = str(raw_name or "").strip()
            if not name or name == pos:
                continue
            neg_weights[name] = float(neg_weights.get(name, 0.0) + 1.0 * rec_weight)
        if not neg_weights:
            continue
        bucket = confusion_delta.setdefault(pos, {})
        for neg, weight in neg_weights.items():
            bucket[str(neg)] = float(bucket.get(str(neg), 0.0) + float(weight))

    confusion_delta = {
        str(pos): {
            str(neg): float(val)
            for neg, val in row.items()
            if abs(float(val)) > 1e-6
        }
        for pos, row in confusion_delta.items()
        if any(abs(float(val)) > 1e-6 for val in row.values())
    }

    step = int(prev_step + 1)
    payload = {
        "version": 2,
        "step": int(step),
        "source": "confirmed_correction_buffer",
        "supervision_source": str(supervision_source),
        "video_finalized": bool(finalized_video),
        "feature_dim": int(feature_dim),
        "trim_ratio": float(max(0.0, float(trim_ratio))),
        "record_count": int(len(records)),
        "label_record_count": int(len(label_records)),
        "transition_record_count": int(len(transition_records)),
        "used_label_record_count": int(used_label_records),
        "skipped_label_record_count": int(skipped_label_records),
        "action_counts": dict(action_counts),
        "prototypes": prototypes,
        "transition_deltas": transition_delta,
        "confusion_deltas": confusion_delta,
        "counts": {str(k): float(v) for k, v in label_counts.items()},
        "labels": sorted(str(k) for k in prototypes.keys()),
        "meta_backbone": str(
            meta.get("backbone")
            or meta.get("source")
            or meta.get("feature_backbone_version")
            or ""
        ),
        "last_record_hash": str(record_hash),
        "last_label_record_hash": str(label_hash),
        "updated_at": _utc_now_iso(),
    }
    _safe_save_object(delta_path, payload)
    meta_dirty = _mark_runtime_dirty(
        runtime_dir,
        record_hash=record_hash,
        label_hash=label_hash,
        update_step=step,
        record_count=len(records),
        label_record_count=len(label_records),
    )
    _emit(
        f"[EAST][ONLINE] Runtime update ready: {len(records)} records, {len(prototypes)} prototypes."
    )
    return {
        "ok": True,
        "changed": True,
        "features_dir": features_dir,
        "runtime_dir": runtime_dir,
        "supervision_source": str(supervision_source),
        "video_finalized": bool(finalized_video),
        "record_count": int(len(records)),
        "label_record_count": int(len(label_records)),
        "transition_record_count": int(len(transition_records)),
        "used_label_record_count": int(used_label_records),
        "skipped_label_record_count": int(skipped_label_records),
        "action_counts": dict(action_counts),
        "prototype_count": int(len(prototypes)),
        "transition_delta_count": int(
            sum(len(row) for row in transition_delta.values())
        ),
        "confusion_delta_pairs": int(
            sum(len(row) for row in confusion_delta.values())
        ),
        "step": int(step),
        "runtime_meta_marked_dirty": bool(meta_dirty),
    }
