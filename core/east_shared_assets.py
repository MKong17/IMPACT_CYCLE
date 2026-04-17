import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from core.east_online_adapter_train import train_shared_online_adapter_from_runtime_dirs
except Exception:
    train_shared_online_adapter_from_runtime_dirs = None


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_load(path: str) -> Any:
    if torch is not None:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
        except Exception:
            pass
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _safe_save(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if torch is not None:
        try:
            torch.save(obj, path)
            return
        except Exception:
            pass
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _clean_confusion_map(raw: Any) -> Dict[str, Dict[str, float]]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for pos, row in raw.items():
        if not isinstance(row, dict):
            continue
        clean_row: Dict[str, float] = {}
        for neg, val in row.items():
            try:
                score = float(val)
            except Exception:
                continue
            if abs(score) <= 1e-6:
                continue
            clean_row[str(neg)] = score
        if clean_row:
            out[str(pos)] = clean_row
    return out


def load_east_runtime_confusion_map(features_dir: str) -> Dict[str, Dict[str, float]]:
    features_dir = os.path.abspath(os.path.expanduser(str(features_dir or "").strip()))
    runtime_dir = os.path.join(features_dir, "east_runtime")
    delta_path = os.path.join(runtime_dir, "model_delta.pt")
    if not os.path.isfile(delta_path):
        return {}
    obj = _safe_load(delta_path)
    if not isinstance(obj, dict):
        return {}
    return _clean_confusion_map(obj.get("confusion_deltas"))


def inspect_east_runtime_assets(features_dir: str) -> Dict[str, Any]:
    features_dir = os.path.abspath(os.path.expanduser(str(features_dir or "").strip()))
    runtime_dir = os.path.join(features_dir, "east_runtime")
    if not os.path.isdir(features_dir):
        return {"ok": False, "reason": "features_dir_missing", "features_dir": features_dir}
    if not os.path.isdir(runtime_dir):
        return {"ok": False, "reason": "runtime_dir_missing", "features_dir": features_dir, "runtime_dir": runtime_dir}

    meta = _load_json(os.path.join(runtime_dir, "meta.json"))
    seg_payload = _load_json(os.path.join(runtime_dir, "segments.json"))
    record_path = os.path.join(runtime_dir, "record_buffer.pkl")
    online_path = os.path.join(runtime_dir, "online_adapter.pt")
    delta_path = os.path.join(runtime_dir, "model_delta.pt")
    boundary_path = os.path.join(runtime_dir, "boundary.npy")
    seg_embed_path = os.path.join(runtime_dir, "seg_embeds.npy")
    label_scores_path = os.path.join(runtime_dir, "label_scores.npy")
    transition_path = os.path.join(runtime_dir, "transition.npy")
    proto_path = os.path.join(runtime_dir, "prototype.npy")
    text_bank_path = os.path.join(runtime_dir, "label_text_bank.pt")
    finalized_record_path = os.path.join(runtime_dir, "finalized_record_buffer.pkl")

    record_count = 0
    action_counts: Dict[str, int] = {}
    if os.path.isfile(record_path):
        try:
            with open(record_path, "rb") as f:
                records = pickle.load(f)
            if isinstance(records, list):
                record_count = len(records)
                for rec in records:
                    if not isinstance(rec, dict):
                        continue
                    key = str(rec.get("action_kind", rec.get("point_type", "unknown")) or "unknown").strip() or "unknown"
                    action_counts[key] = int(action_counts.get(key, 0)) + 1
        except Exception:
            pass

    online_obj = _safe_load(online_path) if os.path.isfile(online_path) else {}
    delta_obj = _safe_load(delta_path) if os.path.isfile(delta_path) else {}
    text_bank_obj = _safe_load(text_bank_path) if os.path.isfile(text_bank_path) else {}
    online_obj = online_obj if isinstance(online_obj, dict) else {}
    delta_obj = delta_obj if isinstance(delta_obj, dict) else {}
    text_bank_obj = text_bank_obj if isinstance(text_bank_obj, dict) else {}

    segment_count = len(seg_payload.get("segments") or []) if isinstance(seg_payload.get("segments"), list) else 0
    classes = [str(x).strip() for x in (seg_payload.get("classes") or []) if str(x).strip()]
    text_table = np.asarray(online_obj.get("text_table"), dtype=np.float32) if online_obj.get("text_table") is not None else np.zeros((0, 0), dtype=np.float32)
    proto_map = dict(delta_obj.get("prototypes") or {})
    transition_map = dict(delta_obj.get("transition_deltas") or {})
    confusion_map = _clean_confusion_map(delta_obj.get("confusion_deltas"))

    return {
        "ok": True,
        "features_dir": features_dir,
        "runtime_dir": runtime_dir,
        "runtime_meta": meta,
        "runtime_origin": str(meta.get("runtime_origin") or ""),
        "bootstrap_source": str(meta.get("bootstrap_source") or ""),
        "bootstrap_pending": bool(meta.get("bootstrap_pending", False)),
        "bootstrap_completed": bool(meta.get("bootstrap_completed", False)),
        "bootstrap_initialized_at": str(meta.get("bootstrap_initialized_at") or ""),
        "bootstrap_confirmed_count": int(meta.get("bootstrap_confirmed_count", 0) or 0),
        "bootstrap_refresh_deferred": bool(meta.get("bootstrap_refresh_deferred", False)),
        "video_finalized": bool(meta.get("video_finalized", False)),
        "finalized_pending": bool(meta.get("finalized_pending", False)),
        "finalized_record_count": int(meta.get("finalized_record_count", 0) or 0),
        "has_finalized_record_buffer": bool(os.path.isfile(finalized_record_path)),
        "video_id": str(meta.get("video_id") or ""),
        "runtime_dirty": bool(meta.get("runtime_dirty", False)),
        "segment_count": int(segment_count),
        "class_count": int(len(classes)),
        "classes": list(classes),
        "record_count": int(record_count),
        "action_counts": dict(action_counts),
        "locked_segment_count": int(action_counts.get("segment_lock", 0) or 0),
        "has_boundary_score": bool(os.path.isfile(boundary_path)),
        "has_seg_embeddings": bool(os.path.isfile(seg_embed_path)),
        "has_label_scores": bool(os.path.isfile(label_scores_path)),
        "has_transition_matrix": bool(os.path.isfile(transition_path)),
        "has_runtime_prototype": bool(os.path.isfile(proto_path)),
        "has_label_text_bank": bool(text_bank_obj.get("text_table") is not None),
        "label_text_bank_backend": str(text_bank_obj.get("backend") or ""),
        "has_online_adapter": bool(online_obj.get("model")),
        "online_step": int(online_obj.get("step", 0) or 0),
        "online_class_count": int(len([str(x).strip() for x in (online_obj.get("classes") or []) if str(x).strip()])),
        "online_boundary_supervision_count": int(online_obj.get("boundary_supervision_count", 0) or 0),
        "online_positive_span_count": int(online_obj.get("positive_span_count", 0) or 0),
        "online_negative_span_count": int(online_obj.get("negative_span_count", 0) or 0),
        "online_ranking_pair_count": int(online_obj.get("ranking_pair_count", 0) or 0),
        "online_input_dim": int(online_obj.get("input_dim", 0) or 0),
        "online_hidden_ratio": float(online_obj.get("hidden_ratio", 0.0) or 0.0),
        "text_table_shape": tuple(int(x) for x in getattr(text_table, "shape", (0, 0))),
        "has_model_delta": bool(proto_map or transition_map),
        "model_delta_step": int(delta_obj.get("step", 0) or 0),
        "prototype_count": int(len(proto_map)),
        "transition_delta_pairs": int(
            sum(len(row) for row in transition_map.values() if isinstance(row, dict))
        ),
        "confusion_delta_pairs": int(
            sum(len(row) for row in confusion_map.values() if isinstance(row, dict))
        ),
        "shared_adapter_seed_path": str(meta.get("shared_adapter_seed_path") or ""),
    }


def export_east_runtime_report(features_dir: str, output_path: str) -> str:
    report = inspect_east_runtime_assets(features_dir)
    output_path = os.path.abspath(os.path.expanduser(str(output_path or "").strip()))
    if not output_path:
        raise ValueError("output_path is required")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return output_path


def export_runtime_as_shared_adapter(
    features_dir: str,
    output_path: str,
    *,
    category: str = "",
    label_bank: Optional[Sequence[str]] = None,
) -> str:
    features_dir = os.path.abspath(os.path.expanduser(str(features_dir or "").strip()))
    runtime_dir = os.path.join(features_dir, "east_runtime")
    if not os.path.isdir(runtime_dir):
        raise FileNotFoundError(f"runtime_dir not found: {runtime_dir}")
    meta = _load_json(os.path.join(runtime_dir, "meta.json"))
    online_path = os.path.join(runtime_dir, "online_adapter.pt")
    delta_path = os.path.join(runtime_dir, "model_delta.pt")
    text_bank_path = os.path.join(runtime_dir, "label_text_bank.pt")
    online_obj = _safe_load(online_path) if os.path.isfile(online_path) else {}
    delta_obj = _safe_load(delta_path) if os.path.isfile(delta_path) else {}
    text_bank_obj = _safe_load(text_bank_path) if os.path.isfile(text_bank_path) else {}
    online_obj = online_obj if isinstance(online_obj, dict) else {}
    delta_obj = delta_obj if isinstance(delta_obj, dict) else {}
    text_bank_obj = text_bank_obj if isinstance(text_bank_obj, dict) else {}
    bundle = {
        "version": 1,
        "kind": "east_shared_adapter",
        "created_at": _utc_now(),
        "category": str(category or "").strip(),
        "name": os.path.splitext(os.path.basename(str(output_path or "")))[0],
        "source_video_id": str(meta.get("video_id") or ""),
        "source_runtime_dir": runtime_dir,
        "label_bank": [str(x).strip() for x in (label_bank or []) if str(x).strip()],
        "runtime_meta": meta,
        "online_adapter": online_obj,
        "model_delta": delta_obj,
        "label_text_bank": text_bank_obj,
    }
    output_path = os.path.abspath(os.path.expanduser(str(output_path or "").strip()))
    if not output_path:
        raise ValueError("output_path is required")
    _safe_save(bundle, output_path)
    return output_path


def inspect_east_shared_adapter_bundle(path: str) -> Dict[str, Any]:
    bundle_path = os.path.abspath(os.path.expanduser(str(path or "").strip()))
    if not bundle_path or not os.path.isfile(bundle_path):
        return {"ok": False, "reason": "bundle_missing", "path": bundle_path}
    obj = _safe_load(bundle_path)
    if not isinstance(obj, dict):
        return {"ok": False, "reason": "bundle_invalid", "path": bundle_path}
    online_obj = obj.get("online_adapter") if isinstance(obj.get("online_adapter"), dict) else {}
    delta_obj = obj.get("model_delta") if isinstance(obj.get("model_delta"), dict) else {}
    text_bank_obj = obj.get("label_text_bank") if isinstance(obj.get("label_text_bank"), dict) else {}
    transition_map = dict(delta_obj.get("transition_deltas") or {}) if isinstance(delta_obj, dict) else {}
    confusion_map = _clean_confusion_map(delta_obj.get("confusion_deltas")) if isinstance(delta_obj, dict) else {}
    return {
        "ok": True,
        "path": bundle_path,
        "category": str(obj.get("category") or ""),
        "name": str(obj.get("name") or os.path.splitext(os.path.basename(bundle_path))[0]),
        "display_name": "/".join(
            [part for part in [str(obj.get("category") or "").strip(), str(obj.get("name") or "").strip()] if part]
        )
        or os.path.basename(bundle_path),
        "label_bank": [str(x).strip() for x in (obj.get("label_bank") or []) if str(x).strip()],
        "created_at": str(obj.get("created_at") or ""),
        "source_video_id": str(obj.get("source_video_id") or ""),
        "member_count": int(len(obj.get("members") or [])) if isinstance(obj.get("members"), list) else 0,
        "has_online_adapter": bool(online_obj.get("model")),
        "has_label_text_bank": bool(text_bank_obj.get("text_table") is not None),
        "label_text_bank_backend": str(text_bank_obj.get("backend") or ""),
        "online_step": int(online_obj.get("step", 0) or 0),
        "online_class_count": int(len([str(x).strip() for x in (online_obj.get("classes") or []) if str(x).strip()])),
        "online_ranking_pair_count": int(online_obj.get("ranking_pair_count", 0) or 0),
        "online_training_scope": str(online_obj.get("training_scope") or ""),
        "online_fusion_gate_trained": bool(online_obj.get("fusion_gate_trained", False)),
        "has_model_delta": bool(delta_obj),
        "model_delta_step": int(delta_obj.get("step", 0) or 0) if isinstance(delta_obj, dict) else 0,
        "prototype_count": int(len(dict(delta_obj.get("prototypes") or {}))) if isinstance(delta_obj, dict) else 0,
        "transition_delta_pairs": int(sum(len(row) for row in transition_map.values() if isinstance(row, dict))),
        "confusion_delta_pairs": int(sum(len(row) for row in confusion_map.values() if isinstance(row, dict))),
    }


def _find_runtime_dirs(root_dir: str) -> List[str]:
    root_dir = os.path.abspath(os.path.expanduser(str(root_dir or "").strip()))
    if not root_dir or not os.path.isdir(root_dir):
        return []
    out: List[str] = []
    for cur, dirs, _files in os.walk(root_dir):
        if os.path.basename(cur).lower() != "east_runtime":
            continue
        out.append(cur)
        dirs[:] = []
    return sorted(set(out))


def _merge_state_dicts(items: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
    if not items or torch is None:
        return {}
    keys = None
    base_shapes: Dict[str, Tuple[int, ...]] = {}
    for state, _weight in items:
        if not isinstance(state, dict):
            return {}
        skeys = set(state.keys())
        if keys is None:
            keys = skeys
            for name, tensor in state.items():
                if not hasattr(tensor, "shape"):
                    return {}
                base_shapes[name] = tuple(int(x) for x in tensor.shape)
        else:
            keys &= skeys
    if not keys:
        return {}
    out: Dict[str, Any] = {}
    total_weight = sum(max(0.0, float(w)) for _s, w in items) or float(len(items))
    for name in sorted(keys):
        acc = None
        valid = True
        for state, weight in items:
            tensor = state.get(name)
            if not hasattr(tensor, "shape") or tuple(int(x) for x in tensor.shape) != base_shapes.get(name):
                valid = False
                break
            arr = tensor.detach().cpu().float() if hasattr(tensor, "detach") else torch.tensor(tensor, dtype=torch.float32)
            acc = arr * float(weight) if acc is None else acc + arr * float(weight)
        if valid and acc is not None:
            merged = acc / float(total_weight)
            template = items[0][0][name]
            if hasattr(template, "detach"):
                out[name] = merged.to(dtype=getattr(template, "dtype", torch.float32))
            else:
                target_dtype = np.asarray(template).dtype if template is not None else np.float32
                out[name] = merged.detach().cpu().numpy().astype(target_dtype, copy=False)
    return out


def consolidate_east_shared_adapter(
    root_dir: str,
    output_path: str,
    *,
    category: str = "",
) -> Dict[str, Any]:
    runtime_dirs = _find_runtime_dirs(root_dir)
    if not runtime_dirs:
        return {"ok": False, "reason": "no_runtime_dirs", "root_dir": root_dir}

    online_ckpts: List[Tuple[Dict[str, Any], float]] = []
    delta_objs: List[Tuple[Dict[str, Any], float]] = []
    text_bank_objs: List[Tuple[Dict[str, Any], float]] = []
    label_bank: List[str] = []
    members: List[Dict[str, Any]] = []
    finalized_runtime_dirs: List[str] = []

    for runtime_dir in runtime_dirs:
        meta = _load_json(os.path.join(runtime_dir, "meta.json"))
        segs = _load_json(os.path.join(runtime_dir, "segments.json"))
        online_obj = _safe_load(os.path.join(runtime_dir, "online_adapter.pt"))
        delta_obj = _safe_load(os.path.join(runtime_dir, "model_delta.pt"))
        text_bank_obj = _safe_load(os.path.join(runtime_dir, "label_text_bank.pt"))
        online_obj = online_obj if isinstance(online_obj, dict) else {}
        delta_obj = delta_obj if isinstance(delta_obj, dict) else {}
        text_bank_obj = text_bank_obj if isinstance(text_bank_obj, dict) else {}
        classes = [str(x).strip() for x in (segs.get("classes") or []) if str(x).strip()]
        for name in classes:
            if name and name not in label_bank:
                label_bank.append(name)
        weight = float(
            max(
                int(online_obj.get("positive_span_count", 0) or 0)
                + int(online_obj.get("boundary_supervision_count", 0) or 0),
                int(delta_obj.get("label_record_count", 0) or 0)
                + int(delta_obj.get("transition_record_count", 0) or 0),
                1,
            )
        )
        if online_obj.get("model"):
            online_ckpts.append((online_obj, weight))
        if delta_obj:
            delta_objs.append((delta_obj, weight))
        if text_bank_obj:
            text_bank_objs.append((text_bank_obj, weight))
        if bool(meta.get("video_finalized", False)) and os.path.isfile(os.path.join(runtime_dir, "finalized_record_buffer.pkl")):
            finalized_runtime_dirs.append(runtime_dir)
        members.append(
            {
                "runtime_dir": runtime_dir,
                "video_id": str(meta.get("video_id") or ""),
                "online_step": int(online_obj.get("step", 0) or 0),
                "model_delta_step": int(delta_obj.get("step", 0) or 0),
                "weight": float(weight),
                "video_finalized": bool(meta.get("video_finalized", False)),
            }
        )

    merged_online: Dict[str, Any] = {}
    if online_ckpts:
        template = online_ckpts[0][0]
        input_dim = int(template.get("input_dim", 0) or 0)
        hidden_ratio = float(template.get("hidden_ratio", 0.5) or 0.5)
        source_buckets = int(template.get("source_buckets", 16) or 16)
        compatible = [
            (obj, w)
            for obj, w in online_ckpts
            if int(obj.get("input_dim", 0) or 0) == input_dim
            and abs(float(obj.get("hidden_ratio", 0.5) or 0.5) - hidden_ratio) <= 1e-6
        ]
        state_items = [(dict(obj.get("model") or {}), weight) for obj, weight in compatible if isinstance(obj.get("model"), dict)]
        model_state = _merge_state_dicts(state_items)
        class_to_row: Dict[str, Tuple[np.ndarray, float]] = {}
        for obj, weight in compatible:
            names = [str(x).strip() for x in (obj.get("classes") or []) if str(x).strip()]
            table = np.asarray(obj.get("text_table"), dtype=np.float32) if obj.get("text_table") is not None else np.zeros((0, 0), dtype=np.float32)
            if table.ndim != 2 or table.shape[0] != len(names) or (input_dim and table.shape[1] != input_dim):
                continue
            for idx, name in enumerate(names):
                row = np.asarray(table[idx], dtype=np.float32).reshape(-1)
                prev = class_to_row.get(name)
                if prev is None:
                    class_to_row[name] = (row * float(weight), float(weight))
                else:
                    class_to_row[name] = (prev[0] + row * float(weight), prev[1] + float(weight))
                if name not in label_bank:
                    label_bank.append(name)
        merged_classes = [name for name in label_bank if name in class_to_row]
        merged_text = np.zeros((len(merged_classes), input_dim), dtype=np.float32)
        for idx, name in enumerate(merged_classes):
            acc, weight = class_to_row[name]
            row = acc / max(weight, 1e-6)
            norm = float(np.linalg.norm(row))
            if norm > 0:
                row = row / norm
            merged_text[idx] = row.astype(np.float32)
        if model_state:
            merged_online = {
                "version": 1,
                "step": int(max(int(obj.get("step", 0) or 0) for obj, _w in compatible) + 1),
                "input_dim": int(input_dim),
                "hidden_ratio": float(hidden_ratio),
                "source_buckets": int(source_buckets),
                "feature_source": "mixed",
                "classes": merged_classes,
                "text_table": merged_text,
                "model": model_state,
                "boundary_supervision_count": int(sum(int(obj.get("boundary_supervision_count", 0) or 0) for obj, _w in compatible)),
                "positive_span_count": int(sum(int(obj.get("positive_span_count", 0) or 0) for obj, _w in compatible)),
                "negative_span_count": int(sum(int(obj.get("negative_span_count", 0) or 0) for obj, _w in compatible)),
                "updated_at": _utc_now(),
            }

    merged_delta: Dict[str, Any] = {}
    if delta_objs:
        proto_acc: Dict[str, Tuple[np.ndarray, float]] = {}
        trans_acc: Dict[str, Dict[str, float]] = {}
        confusion_acc: Dict[str, Dict[str, float]] = {}
        count_acc: Dict[str, int] = {}
        for obj, weight in delta_objs:
            protos = dict(obj.get("prototypes") or {})
            counts = dict(obj.get("counts") or {})
            for name, vec in protos.items():
                arr = np.asarray(vec, dtype=np.float32).reshape(-1)
                prev = proto_acc.get(str(name))
                if prev is None:
                    proto_acc[str(name)] = (arr * float(weight), float(weight))
                else:
                    proto_acc[str(name)] = (prev[0] + arr * float(weight), prev[1] + float(weight))
                count_acc[str(name)] = int(count_acc.get(str(name), 0) + int(counts.get(name, 0) or 0))
                if str(name) not in label_bank:
                    label_bank.append(str(name))
            for src, row in dict(obj.get("transition_deltas") or {}).items():
                if not isinstance(row, dict):
                    continue
                bucket = trans_acc.setdefault(str(src), {})
                for dst, val in row.items():
                    try:
                        score = float(val)
                    except Exception:
                        continue
                    bucket[str(dst)] = float(bucket.get(str(dst), 0.0) + score)
                    if str(src) not in label_bank:
                        label_bank.append(str(src))
                    if str(dst) not in label_bank:
                        label_bank.append(str(dst))
            for pos, row in _clean_confusion_map(obj.get("confusion_deltas")).items():
                bucket = confusion_acc.setdefault(str(pos), {})
                for neg, val in row.items():
                    bucket[str(neg)] = float(bucket.get(str(neg), 0.0) + float(val))
                    if str(pos) not in label_bank:
                        label_bank.append(str(pos))
                    if str(neg) not in label_bank:
                        label_bank.append(str(neg))
        merged_protos: Dict[str, np.ndarray] = {}
        for name, (acc, weight) in proto_acc.items():
            row = acc / max(weight, 1e-6)
            norm = float(np.linalg.norm(row))
            if norm > 0:
                row = row / norm
            merged_protos[str(name)] = row.astype(np.float32)
        merged_delta = {
            "version": 2,
            "step": int(max(int(obj.get("step", 0) or 0) for obj, _w in delta_objs) + 1),
            "source": "shared_consolidation",
            "record_count": int(sum(int(obj.get("record_count", 0) or 0) for obj, _w in delta_objs)),
            "label_record_count": int(sum(int(obj.get("label_record_count", 0) or 0) for obj, _w in delta_objs)),
            "transition_record_count": int(sum(int(obj.get("transition_record_count", 0) or 0) for obj, _w in delta_objs)),
            "prototypes": merged_protos,
            "counts": count_acc,
            "labels": sorted(str(x) for x in merged_protos.keys()),
            "transition_deltas": trans_acc,
            "confusion_deltas": confusion_acc,
            "updated_at": _utc_now(),
        }

    merged_text_bank: Dict[str, Any] = {}
    if text_bank_objs and label_bank:
        text_dim = 0
        row_acc: Dict[str, Tuple[np.ndarray, float]] = {}
        backend_names: List[str] = []
        for obj, weight in text_bank_objs:
            names = [str(x).strip() for x in (obj.get("classes") or []) if str(x).strip()]
            table = np.asarray(obj.get("text_table"), dtype=np.float32) if obj.get("text_table") is not None else np.zeros((0, 0), dtype=np.float32)
            if table.ndim != 2 or table.shape[0] != len(names):
                continue
            if text_dim <= 0:
                text_dim = int(table.shape[1])
            if int(table.shape[1]) != text_dim:
                continue
            backend_name = str(obj.get("backend") or "").strip()
            if backend_name and backend_name not in backend_names:
                backend_names.append(backend_name)
            for idx, name in enumerate(names):
                row = np.asarray(table[idx], dtype=np.float32).reshape(-1)
                prev = row_acc.get(name)
                if prev is None:
                    row_acc[name] = (row * float(weight), float(weight))
                else:
                    row_acc[name] = (prev[0] + row * float(weight), prev[1] + float(weight))
                if name not in label_bank:
                    label_bank.append(name)
        merged_names = [name for name in label_bank if name in row_acc]
        if merged_names and text_dim > 0:
            merged_table = np.zeros((len(merged_names), text_dim), dtype=np.float32)
            for idx, name in enumerate(merged_names):
                acc, weight = row_acc[name]
                row = acc / max(weight, 1e-6)
                norm = float(np.linalg.norm(row))
                if norm > 0:
                    row = row / norm
                merged_table[idx] = row.astype(np.float32)
            merged_text_bank = {
                "version": 1,
                "kind": "east_label_text_bank",
                "created_at": _utc_now(),
                "backend": "+".join(backend_names) if backend_names else "",
                "classes": merged_names,
                "feature_dim": int(text_dim),
                "text_table": merged_table,
            }

    offline_online_summary: Dict[str, Any] = {}
    if finalized_runtime_dirs and train_shared_online_adapter_from_runtime_dirs is not None:
        trained_online = train_shared_online_adapter_from_runtime_dirs(
            finalized_runtime_dirs,
            initial_online=merged_online if merged_online else None,
            label_bank=label_bank,
        )
        if bool(trained_online.get("ok", False)) and bool(trained_online.get("changed", False)):
            merged_online = {
                key: value
                for key, value in dict(trained_online).items()
                if key not in {"ok", "changed", "reason", "error"}
            }
            offline_online_summary = {
                "ok": True,
                "runtime_count": int(len(finalized_runtime_dirs)),
                "member_count": int(trained_online.get("member_count", len(finalized_runtime_dirs)) or len(finalized_runtime_dirs)),
                "training_scope": str(trained_online.get("training_scope") or "finalized_shared_offline"),
                "fusion_gate_trained": bool(trained_online.get("fusion_gate_trained", False)),
            }
        elif bool(trained_online):
            offline_online_summary = {
                "ok": False,
                "runtime_count": int(len(finalized_runtime_dirs)),
                "reason": str(trained_online.get("reason") or trained_online.get("error") or "offline_training_skipped"),
            }

    if not merged_online and not merged_delta and not merged_text_bank:
        return {"ok": False, "reason": "no_mergeable_assets", "root_dir": root_dir, "runtime_count": len(runtime_dirs)}

    bundle = {
        "version": 1,
        "kind": "east_shared_adapter",
        "created_at": _utc_now(),
        "category": str(category or "").strip(),
        "name": os.path.splitext(os.path.basename(str(output_path or "")))[0],
        "source_video_id": "",
        "source_runtime_dir": "",
        "label_bank": list(label_bank),
        "runtime_meta": {
            "consolidated_from_root": os.path.abspath(os.path.expanduser(str(root_dir or ""))),
            "member_count": len(runtime_dirs),
            "finalized_member_count": len(finalized_runtime_dirs),
            "offline_online_summary": offline_online_summary,
        },
        "members": members,
        "online_adapter": merged_online,
        "model_delta": merged_delta,
        "label_text_bank": merged_text_bank,
    }
    output_path = os.path.abspath(os.path.expanduser(str(output_path or "").strip()))
    if not output_path:
        raise ValueError("output_path is required")
    _safe_save(bundle, output_path)
    return {
        "ok": True,
        "output_path": output_path,
        "runtime_count": int(len(runtime_dirs)),
        "label_count": int(len(label_bank)),
        "has_online_adapter": bool(merged_online),
        "has_model_delta": bool(merged_delta),
        "has_label_text_bank": bool(merged_text_bank),
        "finalized_runtime_count": int(len(finalized_runtime_dirs)),
        "offline_online_trained": bool(offline_online_summary.get("ok", False)),
        "confusion_delta_pairs": int(
            sum(
                len(row)
                for row in dict(merged_delta.get("confusion_deltas") or {}).values()
                if isinstance(row, dict)
            )
        ) if merged_delta else 0,
        "member_count": int(len(members)),
    }
