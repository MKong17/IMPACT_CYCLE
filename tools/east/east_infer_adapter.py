from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.east.east_checkpoint_loader import collect_model_identity, model_identity_matches
from tools.east.east_model_infer import infer_east_detections
from tools.east.east_output_format import (
    build_segments_payload,
    normalize_label_bank,
    normalize_segments,
    validate_segments,
    write_segments_outputs,
)
from tools.label_utils import load_label_names, resolve_label_source
try:
    from tools.east.online_ieast_adapter import OnlineInteractiveAdapter
    from core.label_text_bank import ensure_label_text_bank, load_label_text_bank_map
    from core.east_label_fusion import (
        SOURCE_NAMES,
        apply_fusion_gate,
        build_gate_features,
    )
except Exception:  # pragma: no cover
    OnlineInteractiveAdapter = None
    ensure_label_text_bank = None
    load_label_text_bank_map = None
    SOURCE_NAMES = ("base", "text_prior", "class_table", "prototype")
    apply_fusion_gate = None
    build_gate_features = None


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def default_east_shared_adapter_root(base_dir: Optional[str] = None) -> str:
    root = str(os.environ.get("EAST_SHARED_ADAPTER_ROOT", "") or "").strip()
    if root:
        return os.path.abspath(os.path.expanduser(root))
    return os.path.join(os.path.abspath(base_dir or _REPO_ROOT), "adapters")


def _shared_adapter_signature(path: str) -> str:
    try:
        st = os.stat(path)
        return "|".join(
            [
                os.path.abspath(path),
                str(int(getattr(st, "st_size", 0))),
                str(int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))),
            ]
        )
    except Exception:
        return os.path.abspath(path)


def _safe_torch_or_pickle_load(path: str) -> Any:
    if torch is not None:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_torch_or_pickle_save(obj: Any, path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if torch is not None:
        torch.save(obj, path)
        return
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_east_shared_adapter_bundle(path: Optional[str]) -> Dict[str, Any]:
    bundle_path = os.path.abspath(os.path.expanduser(str(path or "").strip()))
    if not bundle_path or not os.path.isfile(bundle_path):
        return {}
    try:
        bundle = _safe_torch_or_pickle_load(bundle_path)
    except Exception:
        return {}
    if not isinstance(bundle, dict):
        return {}
    info = dict(bundle)
    info["_path"] = bundle_path
    info["_signature"] = _shared_adapter_signature(bundle_path)
    return info


def inspect_east_shared_adapter(path: Optional[str]) -> Dict[str, Any]:
    bundle = _load_east_shared_adapter_bundle(path)
    if not bundle:
        return {}
    online_obj = bundle.get("online_adapter")
    model_delta_obj = bundle.get("model_delta")
    category = str(bundle.get("category") or "").strip()
    name = str(bundle.get("name") or os.path.splitext(os.path.basename(str(bundle.get("_path") or "")))[0]).strip()
    display_name = "/".join([part for part in [category, name] if part]) or os.path.basename(str(bundle.get("_path") or ""))
    label_bank = [str(x).strip() for x in (bundle.get("label_bank") or []) if str(x).strip()]
    return {
        "path": str(bundle.get("_path") or ""),
        "signature": str(bundle.get("_signature") or ""),
        "category": category,
        "name": name,
        "display_name": display_name,
        "label_bank": label_bank,
        "created_at": str(bundle.get("created_at") or ""),
        "source_video_id": str(bundle.get("source_video_id") or ""),
        "has_online_adapter": isinstance(online_obj, dict) and bool(online_obj.get("model")),
        "has_model_delta": isinstance(model_delta_obj, dict) and bool(model_delta_obj.get("prototypes")),
        "online_step": int((online_obj or {}).get("step", 0)) if isinstance(online_obj, dict) else 0,
    }


def export_east_shared_adapter(
    runtime_dir: str,
    output_path: str,
    *,
    category: str = "",
    video_id: str = "",
    label_bank: Optional[Sequence[str]] = None,
) -> str:
    runtime_dir = os.path.abspath(os.path.expanduser(str(runtime_dir or "").strip()))
    if not runtime_dir or not os.path.isdir(runtime_dir):
        raise FileNotFoundError(f"runtime_dir not found: {runtime_dir}")
    meta_path = os.path.join(runtime_dir, "meta.json")
    meta: Dict[str, Any] = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    online_path = os.path.join(runtime_dir, "online_adapter.pt")
    model_delta_path = os.path.join(runtime_dir, "model_delta.pt")
    online_obj = _safe_torch_or_pickle_load(online_path) if os.path.isfile(online_path) else {}
    model_delta_obj = _safe_torch_or_pickle_load(model_delta_path) if os.path.isfile(model_delta_path) else {}
    bundle = {
        "version": 1,
        "kind": "east_shared_adapter",
        "created_at": _utc_now(),
        "category": str(category or "").strip(),
        "name": os.path.splitext(os.path.basename(str(output_path or "")))[0],
        "source_video_id": str(video_id or meta.get("video_id") or "").strip(),
        "source_runtime_dir": runtime_dir,
        "label_bank": [str(x).strip() for x in (label_bank or []) if str(x).strip()],
        "runtime_meta": meta,
        "online_adapter": online_obj if isinstance(online_obj, dict) else {},
        "model_delta": model_delta_obj if isinstance(model_delta_obj, dict) else {},
    }
    output_path = os.path.abspath(os.path.expanduser(str(output_path or "").strip()))
    if not output_path:
        raise ValueError("output_path is required")
    _safe_torch_or_pickle_save(bundle, output_path)
    return output_path


def seed_east_runtime_from_shared_adapter(
    adapter_path: Optional[str],
    runtime_dir: str,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    bundle = _load_east_shared_adapter_bundle(adapter_path)
    if not bundle:
        return {"seeded": False, "reason": "bundle_missing"}
    runtime_dir = os.path.abspath(os.path.expanduser(str(runtime_dir or "").strip()))
    if not runtime_dir:
        return {"seeded": False, "reason": "runtime_dir_missing"}
    os.makedirs(runtime_dir, exist_ok=True)
    written: List[str] = []
    for key, filename in (
        ("online_adapter", "online_adapter.pt"),
        ("model_delta", "model_delta.pt"),
    ):
        obj = bundle.get(key)
        if not isinstance(obj, dict) or not obj:
            continue
        dst = os.path.join(runtime_dir, filename)
        if (not overwrite) and os.path.isfile(dst):
            continue
        _safe_torch_or_pickle_save(obj, dst)
        written.append(filename)
    meta_path = os.path.join(runtime_dir, "meta.json")
    meta: Dict[str, Any] = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    meta["shared_adapter_seed_path"] = str(bundle.get("_path") or "")
    meta["shared_adapter_seed_signature"] = str(bundle.get("_signature") or "")
    meta["shared_adapter_seeded_at"] = _utc_now()
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)
    return {
        "seeded": bool(written),
        "written": written,
        "signature": str(bundle.get("_signature") or ""),
        "path": str(bundle.get("_path") or ""),
    }


class EastRuntime:
    _logged_meta_mismatch_keys: set = set()

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        cfg_path: Optional[str] = None,
        text_bank_version: str = "disabled",
        feature_backbone_version: Optional[str] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
        category_adapter_path: Optional[str] = None,
    ):
        self.ckpt_path = os.path.abspath(os.path.expanduser(ckpt_path)) if ckpt_path else ""
        self.cfg_path = os.path.abspath(os.path.expanduser(cfg_path)) if cfg_path else ""
        self.text_bank_version = str(text_bank_version or "disabled")
        self.feature_backbone_version = str(
            feature_backbone_version
            or os.environ.get("FEATURE_BACKBONE")
            or "unknown"
        )
        self.progress_cb = progress_cb

        self.model_identity = collect_model_identity(
            ckpt_path=self.ckpt_path,
            cfg_path=self.cfg_path,
            feature_backbone_version=self.feature_backbone_version,
        )
        self._last_runtime: Optional[Dict[str, Any]] = None
        self.category_adapter_path = os.path.abspath(os.path.expanduser(category_adapter_path)) if category_adapter_path else ""
        self._category_adapter_bundle: Optional[Dict[str, Any]] = None

    def _log(self, msg: str) -> None:
        if callable(self.progress_cb):
            try:
                self.progress_cb(msg)
                return
            except Exception:
                pass
        print(msg)

    def _runtime_dir(self, features_dir: str) -> str:
        return os.path.join(features_dir, "east_runtime")

    def _runtime_mismatch_key(self, features_dir: str) -> str:
        runtime_dir = os.path.abspath(self._runtime_dir(features_dir))
        return "|".join(
            [
                runtime_dir,
                str(self.ckpt_path or ""),
                str(self.cfg_path or ""),
                str(self.text_bank_version or "disabled"),
                str(self.feature_backbone_version or "unknown"),
                str(self._category_adapter_signature() or ""),
            ]
        )

    def _category_adapter(self) -> Dict[str, Any]:
        if self._category_adapter_bundle is None:
            self._category_adapter_bundle = _load_east_shared_adapter_bundle(self.category_adapter_path)
        return dict(self._category_adapter_bundle or {})

    def _category_adapter_signature(self) -> str:
        bundle = self._category_adapter()
        return str(bundle.get("_signature") or "")

    def _category_adapter_display_name(self) -> str:
        info = inspect_east_shared_adapter(self.category_adapter_path)
        return str(info.get("display_name") or "")

    def _sha1_text(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _label_bank_hash(self, label_bank: Sequence[str]) -> str:
        blob = "\n".join(normalize_label_bank(label_bank))
        return self._sha1_text(blob)

    def _load_features(self, features_dir: str) -> np.ndarray:
        path = os.path.join(features_dir, "features.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"features.npy not found: {path}")
        feat = np.load(path)
        if feat.ndim != 2:
            raise ValueError(f"features.npy must be 2D, got {feat.shape}")

        h, w = int(feat.shape[0]), int(feat.shape[1])
        if h == 2048 and w != 2048:
            feat = feat.T
        elif w == 2048 and h != 2048:
            pass
        elif h in (512, 768, 1024, 1536, 2048, 4096) and w not in (512, 768, 1024, 1536, 2048, 4096):
            # Heuristic fallback for D x T layouts.
            feat = feat.T
        return np.asarray(feat, dtype=np.float32)

    def _load_feature_meta(self, features_dir: str) -> Dict[str, Any]:
        path = os.path.join(features_dir, "meta.json")
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {}

    def _feature_source_name(self, features_dir: str, meta: Optional[Dict[str, Any]] = None) -> str:
        info = dict(meta or self._load_feature_meta(features_dir) or {})
        backbone = info.get("backbone") or info.get("feature_backbone_version") or ""
        if OnlineInteractiveAdapter is not None and hasattr(OnlineInteractiveAdapter, "normalize_source_name"):
            try:
                return str(OnlineInteractiveAdapter.normalize_source_name(backbone))
            except Exception:
                pass
        key = str(backbone or "").strip().lower()
        if key in {"i3d", "i3d_inception", "i3d_inception_rgb", "i3d_rgb"}:
            return "i3d"
        if key in {
            "east",
            "east_backbone",
            "east-backbone",
            "videomae",
            "videomaev2",
            "video_mae",
            "videomae2",
        }:
            return "east_backbone"
        return "unknown"

    def _normalize_scores(self, row: np.ndarray) -> np.ndarray:
        row = np.asarray(row, dtype=np.float32)
        if row.size == 0:
            return row
        row = np.maximum(row, 0.0)
        s = float(row.sum())
        if s <= 0:
            row[:] = 1.0 / float(row.size)
            return row
        return row / s

    def _normalize_rows(self, mat: np.ndarray) -> np.ndarray:
        arr = np.asarray(mat, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            return arr
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        return arr / norms

    def _softmax_rows(self, mat: np.ndarray) -> np.ndarray:
        arr = np.asarray(mat, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            return arr
        arr = arr - np.max(arr, axis=1, keepdims=True)
        np.exp(arr, out=arr)
        den = np.maximum(arr.sum(axis=1, keepdims=True), 1e-6)
        return arr / den

    def _load_record_buffer(self, features_dir: str) -> List[Dict[str, Any]]:
        path = os.path.join(self._runtime_dir(features_dir), "record_buffer.pkl")
        if not os.path.isfile(path):
            return []
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception:
            return []
        if not isinstance(obj, list):
            return []
        return [dict(x) for x in obj if isinstance(x, dict)]

    def _record_span(self, rec: Dict[str, Any], T: int) -> Optional[Tuple[int, int]]:
        try:
            s = int(rec.get("feedback_start", rec.get("start", 0)))
            e = int(rec.get("feedback_end", rec.get("end", s)))
        except Exception:
            return None
        if T <= 0:
            return None
        if e < s:
            s, e = e, s
        s = max(0, min(T - 1, s))
        e = max(0, min(T - 1, e))
        if e < s:
            return None
        return int(s), int(e)

    def _record_class_id(self, rec: Dict[str, Any], classes: Sequence[str]) -> int:
        try:
            cid = int(rec.get("class_id", -1))
            if 0 <= cid < len(classes):
                return cid
        except Exception:
            pass
        name = str(rec.get("label", "") or "").strip()
        if name and name in classes:
            return int(classes.index(name))
        return -1

    def _collect_hard_protection(
        self,
        features_dir: str,
        classes: Sequence[str],
        T: int,
    ) -> Dict[str, Any]:
        records = self._load_record_buffer(features_dir)
        label_spans: List[Dict[str, Any]] = []
        negative_label_spans: List[Dict[str, Any]] = []
        locked_segments: List[Dict[str, Any]] = []
        boundary_frames: List[int] = []
        removed_boundary_frames: List[int] = []
        for idx, rec in enumerate(records):
            span = self._record_span(rec, T)
            if span is None:
                continue
            action_kind = str(rec.get("action_kind", "") or "").strip().lower()
            cid = self._record_class_id(rec, classes)
            if cid >= 0:
                item = {
                    "start": int(span[0]),
                    "end": int(span[1]),
                    "class_id": int(cid),
                    "label": str(classes[int(cid)]),
                    "order": int(idx),
                }
                if str(rec.get("point_type", "") or "").strip().lower() == "segment" and action_kind == "segment_lock":
                    locked_segments.append(dict(item))
                else:
                    label_spans.append(dict(item))
            old_name = str(rec.get("old_label", "") or "").strip()
            old_cid = int(classes.index(old_name)) if old_name and old_name in classes else -1
            if old_cid >= 0 and action_kind in {"label_replace", "label_remove"}:
                negative_label_spans.append(
                    {
                        "start": int(span[0]),
                        "end": int(span[1]),
                        "class_id": int(old_cid),
                        "label": str(classes[int(old_cid)]),
                        "order": int(idx),
                    }
                )
            if str(rec.get("point_type", "") or "").strip().lower() == "boundary":
                try:
                    frame = int(rec.get("boundary_frame", span[1]))
                except Exception:
                    frame = int(span[1])
                if action_kind in {"boundary_remove"}:
                    try:
                        old_frame = int(rec.get("old_boundary_frame", frame))
                    except Exception:
                        old_frame = int(frame)
                    old_frame = max(0, min(T - 1, old_frame))
                    removed_boundary_frames.append(int(old_frame))
                else:
                    frame = max(0, min(T - 1, frame))
                    boundary_frames.append(int(frame))
                    if action_kind == "boundary_move":
                        try:
                            old_frame = int(rec.get("old_boundary_frame", frame))
                        except Exception:
                            old_frame = int(frame)
                        old_frame = max(0, min(T - 1, old_frame))
                        removed_boundary_frames.append(int(old_frame))
        return {
            "records": records,
            "label_spans": label_spans,
            "negative_label_spans": negative_label_spans,
            "locked_segments": locked_segments,
            "boundary_frames": sorted(set(int(x) for x in boundary_frames if 0 <= int(x) < T)),
            "removed_boundary_frames": sorted(
                set(int(x) for x in removed_boundary_frames if 0 <= int(x) < T)
            ),
        }

    def _apply_boundary_hard_protection(
        self,
        boundary_score: np.ndarray,
        boundary_frames: Sequence[int],
        removed_boundary_frames: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        out = np.asarray(boundary_score, dtype=np.float32).copy()
        T = int(out.shape[0])
        if T <= 0:
            return out
        for raw in boundary_frames or []:
            try:
                frame = int(raw)
            except Exception:
                continue
            if not (0 <= frame < T):
                continue
            out[frame] = max(float(out[frame]), 1.0)
            if frame - 1 >= 0:
                out[frame - 1] = max(float(out[frame - 1]), 0.6)
            if frame + 1 < T:
                out[frame + 1] = max(float(out[frame + 1]), 0.6)
        for raw in removed_boundary_frames or []:
            try:
                frame = int(raw)
            except Exception:
                continue
            if not (0 <= frame < T):
                continue
            out[frame] = min(float(out[frame]), 0.05)
            if frame - 1 >= 0:
                out[frame - 1] = min(float(out[frame - 1]), 0.2)
            if frame + 1 < T:
                out[frame + 1] = min(float(out[frame + 1]), 0.2)
        return out.astype(np.float32)

    def _frame_ids_from_segments(
        self,
        segments: Sequence[Dict[str, Any]],
        label_scores: np.ndarray,
        classes: Sequence[str],
        T: int,
    ) -> np.ndarray:
        frame_ids = np.full((max(0, int(T)),), -1, dtype=np.int32)
        K = int(label_scores.shape[1]) if getattr(label_scores, "ndim", 0) == 2 else 0
        class_to_idx = {str(name): i for i, name in enumerate(classes)}
        for i, seg in enumerate(segments or []):
            s = int(seg.get("start_frame", 0))
            e = int(seg.get("end_frame", s))
            s = max(0, min(T - 1, s))
            e = max(s, min(T - 1, e))
            cid = None
            if i < int(getattr(label_scores, "shape", (0, 0))[0]) and K > 0:
                try:
                    cid = int(np.argmax(label_scores[i]))
                except Exception:
                    cid = None
            if cid is None or not (0 <= cid < len(classes)):
                cname = str(seg.get("class_name", "") or "").strip()
                if cname in class_to_idx:
                    cid = int(class_to_idx[cname])
                else:
                    try:
                        cid = int(seg.get("class_id", 0))
                    except Exception:
                        cid = 0
            cid = max(0, min(max(0, len(classes) - 1), int(cid)))
            frame_ids[s : e + 1] = int(cid)
        if frame_ids.size > 0 and int(frame_ids[0]) < 0:
            frame_ids[0] = 0
        for i in range(1, int(frame_ids.size)):
            if int(frame_ids[i]) < 0:
                frame_ids[i] = frame_ids[i - 1]
        return frame_ids

    def _rebuild_segments_from_frame_ids(
        self,
        frame_ids: np.ndarray,
        classes: Sequence[str],
        forced_cuts: Optional[Sequence[int]] = None,
        protected_ids: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
        ids = np.asarray(frame_ids, dtype=np.int32).reshape(-1)
        T = int(ids.shape[0])
        if T <= 0:
            return [], {}
        K = max(1, len(classes))
        ids = np.clip(ids, 0, K - 1)
        cuts = {int(x) for x in (forced_cuts or []) if 0 <= int(x) < T - 1}
        segments: List[Dict[str, Any]] = []
        protected_rows: Dict[int, int] = {}
        start = 0
        cur = int(ids[0])
        for frame in range(1, T + 1):
            split = (
                frame >= T
                or int(ids[frame]) != cur
                or (frame - 1) in cuts
            )
            if not split:
                continue
            row_idx = len(segments)
            segments.append(
                {
                    "start_frame": int(start),
                    "end_frame": int(frame - 1),
                    "class_id": int(cur),
                    "class_name": str(classes[int(cur)] if 0 <= int(cur) < len(classes) else f"cls_{int(cur)}"),
                }
            )
            if protected_ids is not None:
                block = np.asarray(protected_ids[start:frame], dtype=np.int32).reshape(-1)
                valid = block[block >= 0]
                if valid.size > 0:
                    uniq = np.unique(valid)
                    if uniq.size == 1 and int(uniq[0]) == int(cur):
                        protected_rows[int(row_idx)] = int(cur)
            if frame < T:
                start = int(frame)
                cur = int(ids[frame])
        return segments, protected_rows

    def _apply_hard_protection(
        self,
        *,
        features_dir: str,
        features: np.ndarray,
        boundary_score: np.ndarray,
        segments: Sequence[Dict[str, Any]],
        label_scores: np.ndarray,
        classes: Sequence[str],
        text_prior_map: Optional[Dict[str, np.ndarray]] = None,
        class_map: Optional[Dict[str, np.ndarray]] = None,
        proto_map: Optional[Dict[str, np.ndarray]] = None,
        fusion_gate: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray, np.ndarray, Dict[str, int]]:
        T = int(features.shape[0])
        protection = self._collect_hard_protection(features_dir, classes, T)
        label_spans = list(protection.get("label_spans") or [])
        negative_label_spans = list(protection.get("negative_label_spans") or [])
        locked_segments = list(protection.get("locked_segments") or [])
        boundary_frames = list(protection.get("boundary_frames") or [])
        removed_boundary_frames = list(protection.get("removed_boundary_frames") or [])
        if not label_spans and not negative_label_spans and not locked_segments and not boundary_frames and not removed_boundary_frames:
            return (
                np.asarray(boundary_score, dtype=np.float32),
                list(segments or []),
                np.asarray(label_scores, dtype=np.float32),
                self._segment_embeddings(features, segments or []),
                {
                    "records": 0,
                    "label_spans": 0,
                    "negative_label_spans": 0,
                    "locked_segments": 0,
                    "boundary_frames": 0,
                    "removed_boundary_frames": 0,
                },
            )

        protected_boundary = self._apply_boundary_hard_protection(
            boundary_score,
            boundary_frames,
            removed_boundary_frames=removed_boundary_frames,
        )
        frame_ids = self._frame_ids_from_segments(segments, label_scores, classes, T)
        protected_ids = np.full((T,), -1, dtype=np.int32)
        forced_cuts = set(int(x) for x in boundary_frames if 0 <= int(x) < T - 1)
        for item in locked_segments:
            s = int(item.get("start", 0))
            e = int(item.get("end", s))
            cid = int(item.get("class_id", 0))
            s = max(0, min(T - 1, s))
            e = max(s, min(T - 1, e))
            cid = max(0, min(max(0, len(classes) - 1), cid))
            frame_ids[s : e + 1] = int(cid)
            protected_ids[s : e + 1] = int(cid)
            if s > 0:
                forced_cuts.add(int(s - 1))
            if e < T - 1:
                forced_cuts.add(int(e))
        for item in label_spans:
            s = int(item.get("start", 0))
            e = int(item.get("end", s))
            cid = int(item.get("class_id", 0))
            s = max(0, min(T - 1, s))
            e = max(s, min(T - 1, e))
            cid = max(0, min(max(0, len(classes) - 1), cid))
            frame_ids[s : e + 1] = int(cid)
            protected_ids[s : e + 1] = int(cid)
            if s > 0:
                forced_cuts.add(int(s - 1))
            if e < T - 1:
                forced_cuts.add(int(e))

        protected_segments, protected_rows = self._rebuild_segments_from_frame_ids(
            frame_ids=frame_ids,
            classes=classes,
            forced_cuts=sorted(forced_cuts),
            protected_ids=protected_ids,
        )
        seg_embeddings = self._segment_embeddings(features, protected_segments)
        protected_scores = self._label_scores_from_segments(protected_segments, classes)
        if (text_prior_map or class_map or proto_map) and protected_scores.size > 0:
            protected_scores = self._reweight_label_scores(
                base_scores=protected_scores,
                seg_embeds=seg_embeddings,
                segment_lengths=[
                    max(
                        1,
                        int(seg.get("end_frame", seg.get("start_frame", 0)))
                        - int(seg.get("start_frame", 0))
                        + 1,
                    )
                    for seg in protected_segments
                ],
                classes=classes,
                text_prior_map=text_prior_map or {},
                class_map=class_map or {},
                proto_map=proto_map or {},
                fusion_gate=fusion_gate,
            )
        if negative_label_spans and protected_scores.size > 0:
            for item in negative_label_spans:
                s = int(item.get("start", 0))
                e = int(item.get("end", s))
                cid = int(item.get("class_id", -1))
                if not (0 <= cid < int(protected_scores.shape[1])):
                    continue
                for row_idx, seg in enumerate(protected_segments):
                    seg_s = int(seg.get("start_frame", 0))
                    seg_e = int(seg.get("end_frame", seg_s))
                    if seg_e < s or seg_s > e:
                        continue
                    row = np.asarray(protected_scores[row_idx], dtype=np.float32).reshape(-1)
                    if row.size <= 1:
                        continue
                    row[cid] = 0.0
                    den = float(np.sum(row))
                    if den <= 1e-6:
                        row[:] = 1.0 / max(1, row.size - 1)
                        row[cid] = 0.0
                        den = float(np.sum(row))
                    if den > 1e-6:
                        row = row / den
                    protected_scores[row_idx] = row.astype(np.float32)
        for row_idx, cid in protected_rows.items():
            if 0 <= int(row_idx) < int(protected_scores.shape[0]) and 0 <= int(cid) < int(protected_scores.shape[1]):
                row = np.zeros((protected_scores.shape[1],), dtype=np.float32)
                row[int(cid)] = 1.0
                protected_scores[int(row_idx)] = row
        self._attach_topk(protected_segments, protected_scores, classes, k=5)
        return (
            protected_boundary,
            protected_segments,
            protected_scores.astype(np.float32),
            seg_embeddings.astype(np.float32),
            {
                "records": int(len(protection.get("records") or [])),
                "label_spans": int(len(label_spans)),
                "negative_label_spans": int(len(negative_label_spans)),
                "locked_segments": int(len(locked_segments)),
                "boundary_frames": int(len(boundary_frames)),
                "removed_boundary_frames": int(len(removed_boundary_frames)),
            },
        )

    def _segment_class_id(self, seg: Dict[str, Any], classes: Sequence[str]) -> int:
        try:
            cid = int(seg.get("class_id", -1))
        except Exception:
            cid = -1
        if 0 <= cid < len(classes):
            return int(cid)
        name = str(seg.get("class_name", seg.get("label")) or "").strip()
        if name and name in classes:
            return int(classes.index(name))
        return 0

    def _gap_fill_row(
        self,
        *,
        gap_embed: np.ndarray,
        classes: Sequence[str],
        left_seg: Optional[Dict[str, Any]],
        right_seg: Optional[Dict[str, Any]],
        left_row: Optional[np.ndarray],
        right_row: Optional[np.ndarray],
        text_map: Optional[Dict[str, np.ndarray]],
        proto_map: Optional[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        K = max(1, len(classes))
        row = np.zeros((K,), dtype=np.float32)

        left_cid = self._segment_class_id(left_seg or {}, classes) if left_seg is not None else -1
        right_cid = self._segment_class_id(right_seg or {}, classes) if right_seg is not None else -1

        if left_seg is not None and 0 <= left_cid < K:
            conf = float(np.max(np.asarray(left_row, dtype=np.float32))) if left_row is not None and np.asarray(left_row).size else 1.0
            row[left_cid] += 0.30 + 0.20 * max(0.0, min(1.0, conf))
        if right_seg is not None and 0 <= right_cid < K:
            conf = float(np.max(np.asarray(right_row, dtype=np.float32))) if right_row is not None and np.asarray(right_row).size else 1.0
            row[right_cid] += 0.30 + 0.20 * max(0.0, min(1.0, conf))
        if left_cid >= 0 and right_cid >= 0 and left_cid == right_cid:
            row[left_cid] += 0.35

        embed = np.asarray(gap_embed, dtype=np.float32).reshape(1, -1)
        embed = self._normalize_rows(embed)

        txt_mat = self._class_matrix_from_map(classes, text_map or {}, int(embed.shape[1]))
        if txt_mat is not None:
            row += 0.35 * self._softmax_rows(embed @ txt_mat.T)[0]

        proto_mat = self._class_matrix_from_map(classes, proto_map or {}, int(embed.shape[1]))
        if proto_mat is not None:
            row += 0.25 * self._softmax_rows(embed @ proto_mat.T)[0]

        if float(row.sum()) <= 0.0:
            fallback = left_cid if left_cid >= 0 else (right_cid if right_cid >= 0 else 0)
            row[int(max(0, min(K - 1, fallback)))] = 1.0
        return self._normalize_scores(row)

    def _fill_segment_gaps(
        self,
        *,
        features: np.ndarray,
        segments: Sequence[Dict[str, Any]],
        label_scores: np.ndarray,
        classes: Sequence[str],
        text_map: Optional[Dict[str, np.ndarray]] = None,
        proto_map: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, Dict[str, int]]:
        T = int(features.shape[0])
        segs = normalize_segments(segments or [], classes)
        K = max(1, len(classes))
        if T <= 0:
            return [], np.zeros((0, K), dtype=np.float32), np.zeros((0, int(features.shape[1])), dtype=np.float32), {
                "gap_ranges": 0,
                "gap_frames": 0,
            }

        if label_scores.ndim == 2 and int(label_scores.shape[0]) == len(segs) and int(label_scores.shape[1]) == K:
            base_rows = np.asarray(label_scores, dtype=np.float32)
        else:
            base_rows = self._label_scores_from_segments(segs, classes)

        if not segs:
            full_embed = np.asarray(features.mean(axis=0), dtype=np.float32)
            row = self._gap_fill_row(
                gap_embed=full_embed,
                classes=classes,
                left_seg=None,
                right_seg=None,
                left_row=None,
                right_row=None,
                text_map=text_map,
                proto_map=proto_map,
            )
            cid = int(np.argmax(row))
            filled = [{
                "start_frame": 0,
                "end_frame": T - 1,
                "class_id": cid,
                "class_name": str(classes[cid] if 0 <= cid < len(classes) else f"cls_{cid}"),
            }]
            embeds = self._segment_embeddings(features, filled)
            return filled, row.reshape(1, -1).astype(np.float32), embeds.astype(np.float32), {
                "gap_ranges": 1,
                "gap_frames": int(T),
            }

        out_segments: List[Dict[str, Any]] = []
        out_rows: List[np.ndarray] = []
        gap_ranges = 0
        gap_frames = 0
        cursor = 0

        for idx, seg in enumerate(segs):
            s = int(seg.get("start_frame", 0))
            e = int(seg.get("end_frame", s))
            if s > cursor:
                gs, ge = int(cursor), int(s - 1)
                if ge >= gs:
                    gap_embed = np.asarray(features[gs: ge + 1].mean(axis=0), dtype=np.float32)
                    left_seg = out_segments[-1] if out_segments else None
                    left_row = out_rows[-1] if out_rows else None
                    right_seg = seg
                    right_row = base_rows[idx] if idx < int(base_rows.shape[0]) else None
                    row = self._gap_fill_row(
                        gap_embed=gap_embed,
                        classes=classes,
                        left_seg=left_seg,
                        right_seg=right_seg,
                        left_row=left_row,
                        right_row=right_row,
                        text_map=text_map,
                        proto_map=proto_map,
                    )
                    cid = int(np.argmax(row))
                    out_segments.append(
                        {
                            "start_frame": gs,
                            "end_frame": ge,
                            "class_id": cid,
                            "class_name": str(classes[cid] if 0 <= cid < len(classes) else f"cls_{cid}"),
                        }
                    )
                    out_rows.append(row.astype(np.float32))
                    gap_ranges += 1
                    gap_frames += int(ge - gs + 1)
            out_segments.append(dict(seg))
            if idx < int(base_rows.shape[0]):
                out_rows.append(np.asarray(base_rows[idx], dtype=np.float32))
            else:
                out_rows.append(self._label_scores_from_segments([seg], classes)[0])
            cursor = int(e + 1)

        if cursor <= T - 1:
            gs, ge = int(cursor), int(T - 1)
            gap_embed = np.asarray(features[gs: ge + 1].mean(axis=0), dtype=np.float32)
            left_seg = out_segments[-1] if out_segments else None
            left_row = out_rows[-1] if out_rows else None
            row = self._gap_fill_row(
                gap_embed=gap_embed,
                classes=classes,
                left_seg=left_seg,
                right_seg=None,
                left_row=left_row,
                right_row=None,
                text_map=text_map,
                proto_map=proto_map,
            )
            cid = int(np.argmax(row))
            out_segments.append(
                {
                    "start_frame": gs,
                    "end_frame": ge,
                    "class_id": cid,
                    "class_name": str(classes[cid] if 0 <= cid < len(classes) else f"cls_{cid}"),
                }
            )
            out_rows.append(row.astype(np.float32))
            gap_ranges += 1
            gap_frames += int(ge - gs + 1)

        out_scores = np.asarray(out_rows, dtype=np.float32) if out_rows else np.zeros((0, K), dtype=np.float32)
        out_embeds = self._segment_embeddings(features, out_segments)
        return out_segments, out_scores.astype(np.float32), out_embeds.astype(np.float32), {
            "gap_ranges": int(gap_ranges),
            "gap_frames": int(gap_frames),
        }

    def _online_adapter_path(self, features_dir: str) -> str:
        return os.path.join(self._runtime_dir(features_dir), "online_adapter.pt")

    def _read_online_step(self, features_dir: str) -> int:
        path = self._online_adapter_path(features_dir)
        if not os.path.isfile(path) or torch is None:
            return 0
        try:
            obj = _safe_torch_or_pickle_load(path)
            if isinstance(obj, dict):
                return int(obj.get("step", 0))
        except Exception:
            pass
        return 0

    def _load_delta_prototypes(self, features_dir: str) -> Dict[str, np.ndarray]:
        path = os.path.join(self._runtime_dir(features_dir), "model_delta.pt")
        if not os.path.isfile(path):
            return {}
        try:
            obj = _safe_torch_or_pickle_load(path)
            protos = dict((obj or {}).get("prototypes") or {})
        except Exception:
            protos = {}
        out: Dict[str, np.ndarray] = {}
        for name, vec in protos.items():
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            n = float(np.linalg.norm(arr))
            if n > 0:
                arr = arr / n
            out[str(name)] = arr
        return out

    def _load_delta_transitions(self, features_dir: str) -> Dict[str, Dict[str, float]]:
        path = os.path.join(self._runtime_dir(features_dir), "model_delta.pt")
        if not os.path.isfile(path):
            return {}
        try:
            obj = _safe_torch_or_pickle_load(path)
            raw = dict((obj or {}).get("transition_deltas") or {})
        except Exception:
            raw = {}
        out: Dict[str, Dict[str, float]] = {}
        for src, row in raw.items():
            if not isinstance(row, dict):
                continue
            clean: Dict[str, float] = {}
            for dst, val in row.items():
                try:
                    score = float(val)
                except Exception:
                    continue
                if abs(score) <= 1e-6:
                    continue
                clean[str(dst)] = score
            if clean:
                out[str(src)] = clean
        return out

    def _load_delta_prototypes_from_bundle(self, bundle: Dict[str, Any]) -> Dict[str, np.ndarray]:
        obj = bundle.get("model_delta")
        if not isinstance(obj, dict):
            return {}
        protos = dict((obj or {}).get("prototypes") or {})
        out: Dict[str, np.ndarray] = {}
        for name, vec in protos.items():
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                continue
            n = float(np.linalg.norm(arr))
            if n > 0:
                arr = arr / n
            out[str(name)] = arr
        return out

    def _load_delta_transitions_from_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        obj = bundle.get("model_delta")
        if not isinstance(obj, dict):
            return {}
        raw = dict((obj or {}).get("transition_deltas") or {})
        out: Dict[str, Dict[str, float]] = {}
        for src, row in raw.items():
            if not isinstance(row, dict):
                continue
            clean: Dict[str, float] = {}
            for dst, val in row.items():
                try:
                    score = float(val)
                except Exception:
                    continue
                if abs(score) <= 1e-6:
                    continue
                clean[str(dst)] = score
            if clean:
                out[str(src)] = clean
        return out

    def _load_text_map_from_bundle(
        self, bundle: Dict[str, Any], feat_dim: int
    ) -> Dict[str, np.ndarray]:
        obj = bundle.get("label_text_bank")
        if not isinstance(obj, dict):
            return {}
        names = [str(x).strip() for x in (obj.get("classes") or []) if str(x).strip()]
        table = np.asarray(obj.get("text_table"), dtype=np.float32) if obj.get("text_table") is not None else np.zeros((0, 0), dtype=np.float32)
        if table.ndim != 2 or table.shape[0] != len(names) or table.shape[1] != int(feat_dim):
            return {}
        table = self._normalize_rows(table)
        text_map: Dict[str, np.ndarray] = {}
        for i, name in enumerate(names):
            text_map[name] = table[i]
        return text_map

    def _load_online_side_data(self, features_dir: str, feat_dim: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "step": int(self._read_online_step(features_dir)),
            "class_map": {},
            "proto_map": self._load_delta_prototypes(features_dir),
            "transition_map": self._load_delta_transitions(features_dir),
            "fusion_gate": None,
        }
        path = self._online_adapter_path(features_dir)
        if not os.path.isfile(path) or torch is None:
            return out
        try:
            ckpt = _safe_torch_or_pickle_load(path)
        except Exception:
            return out
        if not isinstance(ckpt, dict):
            return out
        try:
            out["step"] = int(ckpt.get("step", out["step"]))
        except Exception:
            pass

        text_table = ckpt.get("text_table")
        text_classes = [str(x).strip() for x in (ckpt.get("classes") or []) if str(x).strip()]
        if text_table is None or not text_classes:
            return out
        table = np.asarray(text_table, dtype=np.float32)
        if table.ndim != 2 or table.shape[0] != len(text_classes) or table.shape[1] != int(feat_dim):
            return out
        table = self._normalize_rows(table)
        class_map: Dict[str, np.ndarray] = {}
        for i, name in enumerate(text_classes):
            class_map[name] = table[i]
        out["class_map"] = class_map
        gate_weight = ckpt.get("fusion_gate_weight")
        gate_bias = ckpt.get("fusion_gate_bias")
        if gate_weight is not None and gate_bias is not None:
            try:
                out["fusion_gate"] = {
                    "weight": np.asarray(gate_weight, dtype=np.float32),
                    "bias": np.asarray(gate_bias, dtype=np.float32).reshape(-1),
                    "sources": [str(x) for x in (ckpt.get("fusion_gate_sources") or SOURCE_NAMES)],
                    "feature_names": [str(x) for x in (ckpt.get("fusion_gate_feature_names") or [])],
                }
            except Exception:
                out["fusion_gate"] = None
        return out

    def _load_label_text_map(
        self,
        features_dir: str,
        classes: Sequence[str],
        feat_dim: int,
    ) -> Dict[str, np.ndarray]:
        if (
            not classes
            or int(feat_dim) <= 0
            or ensure_label_text_bank is None
            or load_label_text_bank_map is None
        ):
            return {}
        try:
            ensure_label_text_bank(
                features_dir,
                classes,
                int(feat_dim),
                backend=str(self.text_bank_version or "auto"),
                progress_cb=self._log,
            )
        except Exception as exc:
            self._log(f"[EAST][TEXT][WARN] Failed to ensure label text bank: {exc}")
        try:
            return load_label_text_bank_map(features_dir, classes, int(feat_dim))
        except Exception:
            return {}

    def _load_online_side_data_from_bundle(self, bundle: Dict[str, Any], feat_dim: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "step": 0,
            "text_prior_map": self._load_text_map_from_bundle(bundle, feat_dim),
            "class_map": {},
            "proto_map": self._load_delta_prototypes_from_bundle(bundle),
            "transition_map": self._load_delta_transitions_from_bundle(bundle),
            "fusion_gate": None,
        }
        ckpt = bundle.get("online_adapter")
        if not isinstance(ckpt, dict):
            return out
        try:
            out["step"] = int(ckpt.get("step", 0))
        except Exception:
            pass
        text_table = ckpt.get("text_table")
        text_classes = [str(x).strip() for x in (ckpt.get("classes") or []) if str(x).strip()]
        if text_table is None or not text_classes:
            return out
        table = np.asarray(text_table, dtype=np.float32)
        if table.ndim != 2 or table.shape[0] != len(text_classes) or table.shape[1] != int(feat_dim):
            return out
        table = self._normalize_rows(table)
        class_map: Dict[str, np.ndarray] = {}
        for i, name in enumerate(text_classes):
            class_map[name] = table[i]
        out["class_map"] = class_map
        gate_weight = ckpt.get("fusion_gate_weight")
        gate_bias = ckpt.get("fusion_gate_bias")
        if gate_weight is not None and gate_bias is not None:
            try:
                out["fusion_gate"] = {
                    "weight": np.asarray(gate_weight, dtype=np.float32),
                    "bias": np.asarray(gate_bias, dtype=np.float32).reshape(-1),
                    "sources": [str(x) for x in (ckpt.get("fusion_gate_sources") or SOURCE_NAMES)],
                    "feature_names": [str(x) for x in (ckpt.get("fusion_gate_feature_names") or [])],
                }
            except Exception:
                out["fusion_gate"] = None
        return out

    def _merge_online_side_data(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "step": int(max(int((base or {}).get("step", 0)), int((override or {}).get("step", 0)))),
            "text_prior_map": dict((base or {}).get("text_prior_map") or {}),
            "class_map": dict((base or {}).get("class_map") or {}),
            "proto_map": dict((base or {}).get("proto_map") or {}),
            "transition_map": {},
            "fusion_gate": (override or {}).get("fusion_gate") or (base or {}).get("fusion_gate"),
        }
        out["text_prior_map"].update(dict((override or {}).get("text_prior_map") or {}))
        out["class_map"].update(dict((override or {}).get("class_map") or {}))
        out["proto_map"].update(dict((override or {}).get("proto_map") or {}))
        for source in ((base or {}).get("transition_map") or {}, (override or {}).get("transition_map") or {}):
            for src, row in dict(source).items():
                if not isinstance(row, dict):
                    continue
                bucket = out["transition_map"].setdefault(str(src), {})
                for dst, val in row.items():
                    try:
                        score = float(val)
                    except Exception:
                        continue
                    bucket[str(dst)] = float(bucket.get(str(dst), 0.0) + score)
        return out

    def _apply_online_adapter(self, features_dir: str, features: np.ndarray) -> Optional[Dict[str, Any]]:
        if torch is None or OnlineInteractiveAdapter is None:
            return None
        path = self._online_adapter_path(features_dir)
        if not os.path.isfile(path):
            return None

        try:
            ckpt = _safe_torch_or_pickle_load(path)
        except Exception:
            return None
        out = self._apply_online_adapter_ckpt(
            ckpt,
            features,
            source_name=self._feature_source_name(features_dir),
        )
        if out is not None:
            out["proto_map"] = self._load_delta_prototypes(features_dir)
        return out

    def _apply_shared_online_adapter(
        self,
        bundle: Dict[str, Any],
        features: np.ndarray,
        source_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        out = self._apply_online_adapter_ckpt(
            bundle.get("online_adapter"),
            features,
            source_name=source_name,
        )
        if out is not None:
            out["proto_map"] = self._load_delta_prototypes_from_bundle(bundle)
        return out

    def _apply_online_adapter_ckpt(
        self,
        ckpt: Any,
        features: np.ndarray,
        source_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(ckpt, dict):
            return None
        if int(ckpt.get("input_dim", features.shape[1])) != int(features.shape[1]):
            return None

        try:
            hidden_ratio = float(ckpt.get("hidden_ratio", 0.5))
        except Exception:
            hidden_ratio = 0.5
        if (not np.isfinite(hidden_ratio)) or hidden_ratio <= 0.0:
            hidden_ratio = 0.5
        model = OnlineInteractiveAdapter(input_dim=int(features.shape[1]), hidden_ratio=hidden_ratio)
        try:
            model.load_state_dict(ckpt.get("model", {}), strict=False)
        except Exception:
            return None
        model.eval()

        source_name = str(source_name or ckpt.get("feature_source") or "unknown")
        feat_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
        with torch.no_grad():
            out = model(feat_tensor, source_name=source_name)
            adapted = out["features"][0].detach().cpu().numpy().astype(np.float32)
            cp_prob = torch.sigmoid(out["cp_logits"][0]).detach().cpu().numpy().astype(np.float32)

        class_map: Dict[str, np.ndarray] = {}
        text_table = ckpt.get("text_table")
        text_classes = [str(x).strip() for x in (ckpt.get("classes") or []) if str(x).strip()]
        if text_table is not None and text_classes:
            table = np.asarray(text_table, dtype=np.float32)
            if table.ndim == 2 and table.shape[0] == len(text_classes) and table.shape[1] == features.shape[1]:
                table = self._normalize_rows(table)
                for i, name in enumerate(text_classes):
                    class_map[name] = table[i]

        fusion_gate = None
        gate_weight = ckpt.get("fusion_gate_weight")
        gate_bias = ckpt.get("fusion_gate_bias")
        if gate_weight is not None and gate_bias is not None:
            try:
                fusion_gate = {
                    "weight": np.asarray(gate_weight, dtype=np.float32),
                    "bias": np.asarray(gate_bias, dtype=np.float32).reshape(-1),
                    "sources": [str(x) for x in (ckpt.get("fusion_gate_sources") or SOURCE_NAMES)],
                    "feature_names": [str(x) for x in (ckpt.get("fusion_gate_feature_names") or [])],
                }
            except Exception:
                fusion_gate = None

        return {
            "step": int(ckpt.get("step", 0)),
            "features": adapted,
            "boundary_prob": cp_prob,
            "class_map": class_map,
            "proto_map": {},
            "transition_map": {},
            "fusion_gate": fusion_gate,
        }

    def _class_matrix_from_map(self, classes: Sequence[str], vec_map: Dict[str, np.ndarray], dim: int) -> Optional[np.ndarray]:
        if not classes or not vec_map:
            return None
        rows = []
        for name in classes:
            vec = vec_map.get(str(name))
            if vec is None:
                return None
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.size != dim:
                return None
            rows.append(arr)
        if not rows:
            return None
        return self._normalize_rows(np.asarray(rows, dtype=np.float32))

    def _reweight_label_scores(
        self,
        base_scores: np.ndarray,
        seg_embeds: np.ndarray,
        segment_lengths: Sequence[int],
        classes: Sequence[str],
        text_prior_map: Optional[Dict[str, np.ndarray]],
        class_map: Optional[Dict[str, np.ndarray]],
        proto_map: Optional[Dict[str, np.ndarray]],
        fusion_gate: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        base = np.asarray(base_scores, dtype=np.float32)
        if base.ndim != 2 or base.shape[0] != seg_embeds.shape[0]:
            return base
        if not classes or seg_embeds.ndim != 2:
            return base
        k = len(classes)
        if base.shape[1] != k:
            return base

        base = np.maximum(base, 0.0)
        base = base / np.maximum(base.sum(axis=1, keepdims=True), 1e-6)
        seg = self._normalize_rows(seg_embeds)
        text_scores = None
        txt_mat = self._class_matrix_from_map(classes, text_prior_map or {}, seg.shape[1])
        if txt_mat is not None:
            text_scores = self._softmax_rows(seg @ txt_mat.T)

        class_scores = None
        class_mat = self._class_matrix_from_map(classes, class_map or {}, seg.shape[1])
        if class_mat is not None:
            class_scores = self._softmax_rows(seg @ class_mat.T)

        proto_scores = None
        proto_mat = self._class_matrix_from_map(classes, proto_map or {}, seg.shape[1])
        if proto_mat is not None:
            proto_scores = self._softmax_rows(seg @ proto_mat.T)

        source_probs = {
            "base": base,
            "text_prior": text_scores if text_scores is not None else np.tile(base.mean(axis=0, keepdims=True), (base.shape[0], 1)),
            "class_table": class_scores if class_scores is not None else np.tile(base.mean(axis=0, keepdims=True), (base.shape[0], 1)),
            "prototype": proto_scores if proto_scores is not None else np.tile(base.mean(axis=0, keepdims=True), (base.shape[0], 1)),
        }

        if (
            fusion_gate
            and apply_fusion_gate is not None
            and build_gate_features is not None
        ):
            try:
                gate_features = build_gate_features(
                    base,
                    [max(1, int(x)) for x in segment_lengths],
                    text_available=text_scores is not None,
                    class_available=class_scores is not None,
                    proto_available=proto_scores is not None,
                )
                fused, _mix = apply_fusion_gate(
                    source_probs,
                    np.asarray(fusion_gate.get("weight"), dtype=np.float32),
                    np.asarray(fusion_gate.get("bias"), dtype=np.float32),
                    gate_features,
                )
                return fused.astype(np.float32, copy=False)
            except Exception:
                pass

        fused = 0.45 * base
        total = 0.45
        if text_scores is not None:
            fused += 0.2 * text_scores
            total += 0.2
        if class_scores is not None:
            fused += 0.2 * class_scores
            total += 0.2
        if proto_scores is not None:
            fused += 0.15 * proto_scores
            total += 0.15

        fused = fused / max(total, 1e-6)
        fused = fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-6)
        return fused.astype(np.float32)

    def _transition_matrix_from_map(
        self,
        classes: Sequence[str],
        transition_map: Optional[Dict[str, Dict[str, float]]],
    ) -> Optional[np.ndarray]:
        if not classes or not transition_map:
            return None
        k = len(classes)
        class_to_idx = {str(name): i for i, name in enumerate(classes)}
        mat = np.zeros((k, k), dtype=np.float32)
        filled = False
        for src, row in dict(transition_map or {}).items():
            i = class_to_idx.get(str(src))
            if i is None or not isinstance(row, dict):
                continue
            for dst, val in row.items():
                j = class_to_idx.get(str(dst))
                if j is None:
                    continue
                try:
                    score = float(val)
                except Exception:
                    continue
                if abs(score) <= 1e-6:
                    continue
                mat[int(i), int(j)] += np.float32(score)
                filled = True
        if not filled:
            return None
        return mat

    def _apply_transition_prior(
        self,
        label_scores: np.ndarray,
        classes: Sequence[str],
        transition_map: Optional[Dict[str, Dict[str, float]]],
    ) -> np.ndarray:
        scores = np.asarray(label_scores, dtype=np.float32)
        if scores.ndim != 2 or scores.shape[0] <= 1:
            return scores
        mat = self._transition_matrix_from_map(classes, transition_map)
        if mat is None or mat.shape[0] != scores.shape[1]:
            return scores
        max_abs = float(np.max(np.abs(mat))) if mat.size > 0 else 0.0
        if max_abs <= 1e-6:
            return scores
        bias = mat / max_abs
        out = np.maximum(scores, 0.0)
        out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-6)
        try:
            alpha = float(os.environ.get("EAST_TRANSITION_ALPHA", "0.18") or 0.18)
        except Exception:
            alpha = 0.18
        alpha = max(0.0, min(0.5, alpha))
        try:
            iterations = int(os.environ.get("EAST_TRANSITION_ITERS", "1") or 1)
        except Exception:
            iterations = 1
        iterations = max(1, min(3, iterations))
        locked = np.max(out, axis=1) >= 0.999
        for _ in range(iterations):
            prev_term = np.zeros_like(out, dtype=np.float32)
            next_term = np.zeros_like(out, dtype=np.float32)
            prev_term[1:] = out[:-1] @ bias
            next_term[:-1] = out[1:] @ bias.T
            logits = np.log(np.maximum(out, 1e-6)) + alpha * (prev_term + next_term)
            updated = self._softmax_rows(logits)
            if np.any(locked):
                updated[locked] = out[locked]
            out = updated.astype(np.float32)
        out = out / np.maximum(out.sum(axis=1, keepdims=True), 1e-6)
        return out.astype(np.float32)

    def _build_boundary_score(self, features: np.ndarray) -> np.ndarray:
        T = int(features.shape[0])
        out = np.zeros((T,), dtype=np.float32)
        if T <= 1:
            return out
        d = np.diff(features, axis=0)
        energy = np.linalg.norm(d, axis=1)
        out[1:] = energy.astype(np.float32)
        mn = float(out.min())
        mx = float(out.max())
        if mx > mn:
            out = (out - mn) / (mx - mn)
        return out.astype(np.float32)

    def _resolve_video_path(
        self,
        video_path: Optional[str],
        video_id: str,
        features_dir: str,
    ) -> Optional[str]:
        if video_path:
            vp = os.path.abspath(os.path.expanduser(str(video_path)))
            if os.path.isfile(vp):
                return vp

        meta_path = os.path.join(features_dir, "meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                vp = str((meta or {}).get("video_path", "") or "").strip()
                if vp:
                    vp = os.path.abspath(os.path.expanduser(vp))
                    if os.path.isfile(vp):
                        return vp
            except Exception:
                pass

        stem = os.path.splitext(os.path.basename(str(video_id or "")))[0]
        for ext in (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"):
            for parent in (
                features_dir,
                os.path.dirname(features_dir),
                os.path.abspath(os.path.join(features_dir, "..", "..")),
            ):
                cand = os.path.join(parent, f"{stem}{ext}")
                if os.path.isfile(cand):
                    return cand
        return None

    def _boundary_from_east_detections(
        self,
        detections: Sequence[Dict[str, Any]],
        T: int,
        duration_sec: float,
    ) -> np.ndarray:
        out = np.zeros((max(0, int(T)),), dtype=np.float32)
        if T <= 1 or not detections:
            return out
        dur = float(duration_sec) if float(duration_sec) > 1e-6 else 0.0
        if dur <= 0.0:
            return out

        for row in detections:
            try:
                s = float(row.get("t_start", 0.0))
                e = float(row.get("t_end", 0.0))
                sc = float(row.get("score", 0.0))
            except Exception:
                continue
            if e <= s:
                continue
            sc = max(0.0, sc)
            sf = int(np.clip(round((s / dur) * (T - 1)), 0, T - 1))
            ef = int(np.clip(round((e / dur) * (T - 1)), 0, T - 1))
            if ef < sf:
                sf, ef = ef, sf
            out[sf] = max(out[sf], float(sc))
            out[ef] = max(out[ef], float(sc))
            span = max(1, ef - sf + 1)
            out[sf: ef + 1] += float(sc) * (0.15 / float(span))

        mn = float(out.min())
        mx = float(out.max())
        if mx > mn:
            out = (out - mn) / (mx - mn)
        return out.astype(np.float32)

    def _refine_seed_with_boundary(
        self,
        seed_segments: Sequence[Dict[str, Any]],
        boundary_score: np.ndarray,
        T: int,
    ) -> List[Dict[str, Any]]:
        if T <= 0:
            return []
        seed = normalize_segments(seed_segments, [])
        if not seed:
            return []
        seed.sort(key=lambda x: (int(x.get("start_frame", 0)), int(x.get("end_frame", 0))))
        if len(seed) == 1:
            only = dict(seed[0])
            only["start_frame"] = 0
            only["end_frame"] = T - 1
            return [only]

        scores = np.asarray(boundary_score, dtype=np.float32)
        window = max(8, int(round(T * 0.02)))

        snapped: List[int] = []
        prev_cut = -1
        for i in range(len(seed) - 1):
            nominal = int(seed[i].get("end_frame", prev_cut + 1))
            lo = max(prev_cut + 1, nominal - window)
            hi = min(T - 2, nominal + window)
            if hi < lo:
                cut = min(T - 2, max(prev_cut + 1, nominal))
            else:
                cut = int(lo + int(np.argmax(scores[lo: hi + 1])))
            cut = min(T - 2, max(prev_cut + 1, cut))
            snapped.append(cut)
            prev_cut = cut

        labels: List[Tuple[int, str]] = []
        for seg in seed:
            cid = int(seg.get("class_id", 0))
            cname = str(seg.get("class_name", "") or "").strip() or f"cls_{cid}"
            labels.append((cid, cname))

        out: List[Dict[str, Any]] = []
        start = 0
        for i, (cid, cname) in enumerate(labels):
            end = int(snapped[i]) if i < len(snapped) else (T - 1)
            end = max(start, min(T - 1, end))
            out.append(
                {
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "class_id": int(cid),
                    "class_name": str(cname),
                }
            )
            start = int(end + 1)
            if start > T - 1:
                break

        if out:
            out[-1]["end_frame"] = T - 1
        return out

    def _seed_json_candidates(self, features_dir: str) -> List[str]:
        runtime_dir = self._runtime_dir(features_dir)
        return [
            os.path.join(runtime_dir, "segments.json"),
            os.path.join(features_dir, "pred_east_segments.json"),
            os.path.join(features_dir, "pred_asot_segments.json"),
            os.path.join(features_dir, "pred_asot_full_segments.json"),
            os.path.join(features_dir, "pred_fact_segments.json"),
        ]

    def _segment_label_names(self, segments: Sequence[Dict[str, Any]]) -> List[str]:
        names: List[str] = []
        for seg in segments or []:
            name = str(seg.get("class_name", seg.get("label")) or "").strip()
            if name and name not in names:
                names.append(name)
        return names

    def _resolve_payload_classes(
        self,
        payload: Dict[str, Any],
        label_bank: Sequence[str],
    ) -> Tuple[bool, List[str]]:
        labels = normalize_label_bank(label_bank)
        payload_classes = normalize_label_bank(payload.get("classes") or [])
        segment_names = self._segment_label_names(payload.get("segments") or [])
        if labels:
            if payload_classes:
                return (payload_classes == labels, list(labels))
            compatible = all(name in labels for name in segment_names)
            return (compatible, list(labels))
        merged = payload_classes or segment_names
        return (True, merged)

    def _load_seed_payload(self, features_dir: str, label_bank: Sequence[str]) -> Optional[Dict[str, Any]]:
        labels = normalize_label_bank(label_bank)
        for path in self._seed_json_candidates(features_dir):
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                continue
            matches, classes = self._resolve_payload_classes(obj, labels)
            if not matches:
                self._log(f"[EAST] Skip seed segments with label bank mismatch: {path}")
                continue
            payload = build_segments_payload(
                obj.get("segments") or [],
                classes,
                source=str(obj.get("source") or "seed"),
            )
            if payload.get("segments"):
                self._log(f"[EAST] Loaded seed segments: {path}")
                return payload
        return None

    def _make_segments_from_boundary(
        self,
        boundary_score: np.ndarray,
        T: int,
        labels: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if T <= 0:
            return []
        if T == 1:
            name = labels[0] if labels else "action"
            return [{"start_frame": 0, "end_frame": 0, "class_id": 0, "class_name": name}]

        score = np.asarray(boundary_score, dtype=np.float32)
        thr = float(np.quantile(score[1:], 0.90)) if score.shape[0] > 1 else 1.0
        min_gap = 15
        picks: List[int] = []
        for idx in np.where(score >= thr)[0].tolist():
            if idx <= 0 or idx >= T:
                continue
            if picks and idx - picks[-1] < min_gap:
                if score[idx] > score[picks[-1]]:
                    picks[-1] = int(idx)
                continue
            picks.append(int(idx))

        seg_bounds = [0] + picks + [T]
        out: List[Dict[str, Any]] = []
        names = normalize_label_bank(labels)
        if not names:
            names = ["action"]
        for i in range(len(seg_bounds) - 1):
            s = int(seg_bounds[i])
            e = int(seg_bounds[i + 1] - 1)
            if e < s:
                continue
            cid = i % len(names)
            out.append(
                {
                    "start_frame": s,
                    "end_frame": e,
                    "class_id": int(cid),
                    "class_name": names[cid],
                }
            )
        return out

    def _segment_embeddings(self, features: np.ndarray, segments: Sequence[Dict[str, Any]]) -> np.ndarray:
        if not segments:
            return np.zeros((0, features.shape[1]), dtype=np.float32)
        embs = []
        T = int(features.shape[0])
        for seg in segments:
            s = max(0, min(T - 1, int(seg.get("start_frame", 0))))
            e = max(s, min(T - 1, int(seg.get("end_frame", s))))
            chunk = features[s: e + 1]
            if chunk.size == 0:
                vec = np.zeros((features.shape[1],), dtype=np.float32)
            else:
                vec = np.asarray(chunk.mean(axis=0), dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            embs.append(vec)
        return np.asarray(embs, dtype=np.float32)

    def _classes_from_payload(self, payload: Dict[str, Any], label_bank: Sequence[str]) -> List[str]:
        classes = normalize_label_bank(payload.get("classes") or [])
        if not classes:
            classes = normalize_label_bank(label_bank)
        if not classes:
            classes = []
            for seg in payload.get("segments", []):
                name = str(seg.get("class_name") or "").strip()
                if name and name not in classes:
                    classes.append(name)
        if not classes:
            classes = ["action"]
        return classes

    def _label_scores_from_segments(self, segments: Sequence[Dict[str, Any]], classes: Sequence[str]) -> np.ndarray:
        K = max(1, len(classes))
        N = len(segments)
        scores = np.zeros((N, K), dtype=np.float32)
        class_to_idx = {name: i for i, name in enumerate(classes)}

        for i, seg in enumerate(segments):
            row = np.zeros((K,), dtype=np.float32)
            topk = seg.get("topk")
            if isinstance(topk, list):
                for tk in topk:
                    if not isinstance(tk, dict):
                        continue
                    cid = tk.get("id")
                    cname = tk.get("name")
                    idx = None
                    if cid is not None:
                        try:
                            idx = int(cid)
                        except Exception:
                            idx = None
                    if idx is None and cname is not None:
                        idx = class_to_idx.get(str(cname))
                    if idx is None or not (0 <= idx < K):
                        continue
                    sv = tk.get("score", 1.0)
                    try:
                        row[idx] = max(row[idx], float(sv))
                    except Exception:
                        row[idx] = max(row[idx], 1.0)

            if row.sum() <= 0:
                cname = str(seg.get("class_name") or "").strip()
                cid = seg.get("class_id")
                idx = None
                if cname in class_to_idx:
                    idx = class_to_idx[cname]
                elif cid is not None:
                    try:
                        idx = int(cid)
                    except Exception:
                        idx = None
                if idx is None or not (0 <= idx < K):
                    idx = 0
                row[idx] = 1.0

            scores[i] = self._normalize_scores(row)

        return scores.astype(np.float32)

    def _attach_topk(self, segments: Sequence[Dict[str, Any]], label_scores: np.ndarray, classes: Sequence[str], k: int = 5) -> None:
        if label_scores.ndim != 2:
            return
        K = int(label_scores.shape[1])
        for i, seg in enumerate(segments):
            if i >= int(label_scores.shape[0]):
                break
            row = label_scores[i]
            order = np.argsort(row)[::-1][: min(max(1, int(k)), K)]
            topk = []
            for cid in order:
                cid = int(cid)
                topk.append(
                    {
                        "id": cid,
                        "name": classes[cid] if 0 <= cid < len(classes) else f"cls_{cid}",
                        "score": float(row[cid]),
                    }
                )
            seg["topk"] = topk

    def _frame_scores(self, segments: Sequence[Dict[str, Any]], label_scores: np.ndarray, T: int) -> np.ndarray:
        if label_scores.ndim != 2:
            return np.zeros((T, 1), dtype=np.float32)
        K = int(label_scores.shape[1])
        out = np.zeros((T, K), dtype=np.float32)
        for i, seg in enumerate(segments):
            if i >= int(label_scores.shape[0]):
                break
            s = int(seg.get("start_frame", 0))
            e = int(seg.get("end_frame", s))
            s = max(0, min(T - 1, s))
            e = max(s, min(T - 1, e))
            out[s: e + 1] = label_scores[i]
        return out

    def _transition(self, segments: Sequence[Dict[str, Any]], K: int) -> np.ndarray:
        mat = np.zeros((K, K), dtype=np.float32)
        for i in range(len(segments) - 1):
            a = int(segments[i].get("class_id", 0))
            b = int(segments[i + 1].get("class_id", 0))
            if 0 <= a < K and 0 <= b < K:
                mat[a, b] += 1.0
        return mat

    def _prototype(self, segments: Sequence[Dict[str, Any]], seg_embeds: np.ndarray, K: int) -> np.ndarray:
        if seg_embeds.ndim != 2:
            return np.zeros((K, 1), dtype=np.float32)
        D = int(seg_embeds.shape[1])
        proto = np.zeros((K, D), dtype=np.float32)
        cnt = np.zeros((K,), dtype=np.float32)
        for i, seg in enumerate(segments):
            if i >= int(seg_embeds.shape[0]):
                break
            cid = int(seg.get("class_id", 0))
            if not (0 <= cid < K):
                continue
            proto[cid] += seg_embeds[i]
            cnt[cid] += 1.0
        for cid in range(K):
            if cnt[cid] > 0:
                proto[cid] /= cnt[cid]
            norm = float(np.linalg.norm(proto[cid]))
            if norm > 0:
                proto[cid] /= norm
        return proto

    def _meta_matches(self, meta: Dict[str, Any]) -> bool:
        cached_identity = {
            "ckpt_hash": str(meta.get("ckpt_hash") or "missing"),
            "cfg_hash": str(meta.get("cfg_hash") or "missing"),
            "feature_backbone_version": str(meta.get("feature_backbone_version") or "unknown"),
        }
        current_identity = {
            "ckpt_hash": str(self.model_identity.get("ckpt_hash") or "missing"),
            "cfg_hash": str(self.model_identity.get("cfg_hash") or "missing"),
            "feature_backbone_version": str(self.model_identity.get("feature_backbone_version") or "unknown"),
        }
        if not model_identity_matches(current_identity, cached_identity):
            return False

        old_tb = str(meta.get("text_bank_version") or "disabled")
        cur_tb = str(self.text_bank_version or "disabled")
        if old_tb != cur_tb:
            return False
        if str(meta.get("category_adapter_signature") or "") != str(self._category_adapter_signature() or ""):
            return False
        return True

    def load_runtime(
        self,
        features_dir: str,
        label_bank: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        runtime_dir = self._runtime_dir(features_dir)
        meta_path = os.path.join(runtime_dir, "meta.json")
        seg_path = os.path.join(runtime_dir, "segments.json")
        boundary_path = os.path.join(runtime_dir, "boundary.npy")
        seg_embed_path = os.path.join(runtime_dir, "seg_embeds.npy")
        label_scores_path = os.path.join(runtime_dir, "label_scores.npy")

        if not (os.path.isfile(meta_path) and os.path.isfile(seg_path)):
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return None

        mismatch_key = self._runtime_mismatch_key(features_dir)
        if not self._meta_matches(meta):
            if mismatch_key not in EastRuntime._logged_meta_mismatch_keys:
                self._log("[EAST] Runtime meta mismatch, cache invalidated.")
                EastRuntime._logged_meta_mismatch_keys.add(mismatch_key)
            return None
        EastRuntime._logged_meta_mismatch_keys.discard(mismatch_key)

        try:
            with open(seg_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        matches, classes = self._resolve_payload_classes(payload, label_bank or [])
        if not matches:
            if mismatch_key not in EastRuntime._logged_meta_mismatch_keys:
                self._log("[EAST] Runtime label bank mismatch, cache invalidated.")
                EastRuntime._logged_meta_mismatch_keys.add(mismatch_key)
            return None
        payload = build_segments_payload(
            payload.get("segments") or [],
            classes,
            source=str(payload.get("source") or "EAST"),
        )

        try:
            boundary = np.load(boundary_path) if os.path.isfile(boundary_path) else None
        except Exception:
            boundary = None
        try:
            seg_embeds = np.load(seg_embed_path) if os.path.isfile(seg_embed_path) else None
        except Exception:
            seg_embeds = None
        try:
            label_scores = np.load(label_scores_path) if os.path.isfile(label_scores_path) else None
        except Exception:
            label_scores = None

        runtime = {
            "segments_json": payload,
            "boundary_score": boundary,
            "label_scores": label_scores,
            "seg_embeddings": seg_embeds,
            "meta": meta,
        }
        self._last_runtime = runtime
        return runtime

    def _read_train_step(self, runtime_dir: str) -> int:
        meta_path = os.path.join(runtime_dir, "meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return int(meta.get("train_step", 0))
            except Exception:
                pass
        return 0

    def _save_runtime_artifacts(
        self,
        features_dir: str,
        payload: Dict[str, Any],
        boundary_score: np.ndarray,
        label_scores: np.ndarray,
        seg_embeddings: np.ndarray,
        frame_scores: np.ndarray,
        prototype: np.ndarray,
        transition: np.ndarray,
        meta: Dict[str, Any],
    ) -> None:
        runtime_dir = self._runtime_dir(features_dir)
        os.makedirs(runtime_dir, exist_ok=True)

        with open(os.path.join(runtime_dir, "segments.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        np.save(os.path.join(runtime_dir, "boundary.npy"), boundary_score)
        np.save(os.path.join(runtime_dir, "seg_embeds.npy"), seg_embeddings)
        np.save(os.path.join(runtime_dir, "label_scores.npy"), label_scores)
        np.save(os.path.join(runtime_dir, "prototype.npy"), prototype)
        np.save(os.path.join(runtime_dir, "transition.npy"), transition)

        # Keep logits.pt for compatibility with downstream tools.
        logits_path = os.path.join(runtime_dir, "logits.pt")
        if torch is not None:
            torch.save(torch.from_numpy(frame_scores.astype(np.float32)), logits_path)
        else:
            with open(logits_path, "wb") as f:
                pickle.dump(frame_scores.astype(np.float32), f)

        replay_path = os.path.join(runtime_dir, "replay.pkl")
        if not os.path.isfile(replay_path):
            with open(replay_path, "wb") as f:
                pickle.dump([], f)

        delta_path = os.path.join(runtime_dir, "model_delta.pt")
        if not os.path.isfile(delta_path):
            init_delta = {
                "version": 1,
                "step": int(meta.get("train_step", 0)),
                "prototypes": {},
                "counts": {},
                "updated_at": meta.get("updated_at", ""),
            }
            if torch is not None:
                torch.save(init_delta, delta_path)
            else:
                with open(delta_path, "wb") as f:
                    pickle.dump(init_delta, f)

        meta.setdefault("runtime_dirty", False)
        with open(os.path.join(runtime_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)

    def save_runtime(self, features_dir: str) -> bool:
        # Runtime is persisted in predict(). This method keeps API symmetry.
        return bool(self._last_runtime) and os.path.isdir(self._runtime_dir(features_dir))

    def predict(
        self,
        video_id: str,
        features_dir: str,
        label_bank: Sequence[str],
        video_path: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        features_dir = os.path.abspath(os.path.expanduser(features_dir))
        if not os.path.isdir(features_dir):
            raise FileNotFoundError(f"features_dir not found: {features_dir}")

        labels = normalize_label_bank(label_bank)
        label_hash = self._label_bank_hash(labels)
        runtime_dir = self._runtime_dir(features_dir)
        bundle = self._category_adapter()
        shared_online_step = 0
        if bundle:
            try:
                shared_online_step = int(((bundle.get("online_adapter") or {}).get("step", 0)) if isinstance(bundle.get("online_adapter"), dict) else 0)
            except Exception:
                shared_online_step = 0
            display_name = self._category_adapter_display_name()
            if display_name:
                self._log(f"[EAST] Shared adapter selected: {display_name}")
        # Current default EAST runtime uses only the feature-level online adapter.
        # Detector-side vit delta / interactive-adapter loading is disabled here.
        apply_mode = "online_only"
        detector_delta_sources: List[Any] = []
        current_online_step = max(int(self._read_online_step(features_dir)), int(shared_online_step))

        cached = self.load_runtime(features_dir, label_bank=labels)
        if cached is not None:
            cmeta = cached.get("meta") or {}
            if str(cmeta.get("label_bank_hash") or "") == label_hash:
                cached_online_step = int(cmeta.get("online_adapter_step", cmeta.get("train_step", 0)))
                runtime_dirty = bool(cmeta.get("runtime_dirty", False))
                cached_mode = str(cmeta.get("online_apply_mode", "") or "").strip().lower()
                if not cached_mode:
                    cached_mode = "legacy"
                mode_match = cached_mode == apply_mode
                delta_match = bool(cmeta.get("vit_delta_enabled", False)) == bool(detector_delta_sources)
                if (not runtime_dirty) and cached_online_step == int(current_online_step) and mode_match and delta_match:
                    self._log("[EAST] Loaded cached runtime.")
                    c_boundary = cached.get("boundary_score")
                    c_label_scores = cached.get("label_scores")
                    c_seg_embeds = cached.get("seg_embeddings")
                    return (
                        cached["segments_json"],
                        np.asarray(c_boundary if c_boundary is not None else np.array([], dtype=np.float32), dtype=np.float32),
                        np.asarray(c_label_scores if c_label_scores is not None else np.array([], dtype=np.float32), dtype=np.float32),
                        np.asarray(c_seg_embeds if c_seg_embeds is not None else np.array([], dtype=np.float32), dtype=np.float32),
                    )
                self._log("[EAST] Cache stale (online update/mode change), rebuilding runtime.")

        self._log("[EAST] Building runtime...")
        self._log(f"[EAST] Online apply mode: {apply_mode}")
        features = self._load_features(features_dir)
        feature_source = self._feature_source_name(features_dir)
        shared_side = self._load_online_side_data_from_bundle(bundle, feat_dim=int(features.shape[1])) if bundle else {}
        runtime_side = self._load_online_side_data(features_dir, feat_dim=int(features.shape[1]))
        online_side = self._merge_online_side_data(shared_side, runtime_side)
        online_out: Optional[Dict[str, Any]] = None
        runtime_online_exists = os.path.isfile(self._online_adapter_path(features_dir))
        if bundle and (not runtime_online_exists):
            shared_online_out = self._apply_shared_online_adapter(bundle, features, source_name=feature_source)
            if shared_online_out is not None and np.asarray(shared_online_out.get("features")).shape == features.shape:
                features = np.asarray(shared_online_out["features"], dtype=np.float32)
                online_out = shared_online_out
                self._log(f"[EAST] Applied shared online adapter forward (step={int(shared_online_out.get('step', 0))}).")
        runtime_online_out = self._apply_online_adapter(features_dir, features)
        if runtime_online_out is not None and np.asarray(runtime_online_out.get("features")).shape == features.shape:
            online_out = runtime_online_out
            features = np.asarray(runtime_online_out["features"], dtype=np.float32)
            self._log(f"[EAST] Applied runtime online adapter forward (step={int(runtime_online_out.get('step', 0))}).")
        elif online_out is not None and np.asarray(online_out.get("features")).shape == features.shape:
            features = np.asarray(online_out["features"], dtype=np.float32)
        T = int(features.shape[0])

        seed = self._load_seed_payload(features_dir, labels)
        energy_boundary = self._build_boundary_score(features)
        boundary_score = np.asarray(energy_boundary, dtype=np.float32)

        resolved_video = self._resolve_video_path(video_path=video_path, video_id=video_id, features_dir=features_dir)
        east_boundary = None
        if self.ckpt_path and self.cfg_path and resolved_video:
            try:
                self._log(f"[EAST] Running detector inference: {os.path.basename(resolved_video)}")
                infer_out = infer_east_detections(
                    video_path=resolved_video,
                    cfg_path=self.cfg_path,
                    ckpt_path=self.ckpt_path,
                    label_bank=labels,
                    progress_cb=self.progress_cb,
                    adapter_delta_sources=detector_delta_sources,
                )
                dets = list((infer_out or {}).get("detections") or [])
                det_duration = float((infer_out or {}).get("duration", 0.0) or 0.0)
                east_boundary = self._boundary_from_east_detections(dets, T=T, duration_sec=det_duration)
                if east_boundary.size == boundary_score.size and east_boundary.max() > 0:
                    boundary_score = np.clip(0.75 * east_boundary + 0.25 * energy_boundary, 0.0, 1.0).astype(np.float32)
                    self._log(f"[EAST] Detector returned {len(dets)} proposals.")
                else:
                    self._log("[EAST] Detector produced no usable proposals; fallback to feature boundary.")
            except Exception as exc:
                self._log(f"[EAST][WARN] Detector inference failed: {exc}")
        else:
            if not (self.ckpt_path and self.cfg_path):
                self._log("[EAST] Missing ckpt/cfg; using feature-only fallback.")
            elif not resolved_video:
                self._log("[EAST] Video path unresolved; using feature-only fallback.")

        if online_out is not None:
            online_boundary = np.asarray(online_out.get("boundary_prob"), dtype=np.float32).reshape(-1)
            if online_boundary.shape[0] == boundary_score.shape[0]:
                alpha = float(os.environ.get("EAST_ONLINE_BOUNDARY_ALPHA", "0.35") or 0.35)
                alpha = max(0.0, min(1.0, alpha))
                boundary_score = np.clip((1.0 - alpha) * boundary_score + alpha * online_boundary, 0.0, 1.0).astype(np.float32)

        classes = self._classes_from_payload(seed or {}, labels)
        if seed is not None and list(seed.get("segments") or []):
            base_seed = list(seed.get("segments") or [])
            if east_boundary is not None and east_boundary.max() > 0:
                segments = self._refine_seed_with_boundary(base_seed, east_boundary, T=T)
            else:
                segments = normalize_segments(base_seed, classes)
        else:
            segments = self._make_segments_from_boundary(boundary_score, T, classes)
            seed = build_segments_payload(segments, classes or (labels or ["action"]), source="east_fallback")

        for seg in segments:
            cname = str(seg.get("class_name") or "").strip()
            if cname and cname in classes:
                seg["class_id"] = int(classes.index(cname))
            else:
                cid = int(seg.get("class_id", 0))
                seg["class_id"] = cid if 0 <= cid < len(classes) else 0
                seg["class_name"] = classes[int(seg["class_id"])]

        seg_embeddings = self._segment_embeddings(features, segments)
        label_scores = self._label_scores_from_segments(segments, classes)
        text_prior_map = self._load_label_text_map(features_dir, classes, int(features.shape[1]))
        class_map: Dict[str, np.ndarray] = {}
        proto_map = {}
        transition_map = {}
        fusion_gate = None
        if isinstance(online_side, dict):
            base_text_prior_map = dict(online_side.get("text_prior_map") or {})
            if base_text_prior_map:
                text_prior_map.update(base_text_prior_map)
            base_class_map = dict(online_side.get("class_map") or {})
            if base_class_map:
                class_map.update(base_class_map)
            proto_map = dict(online_side.get("proto_map") or {})
            transition_map = dict(online_side.get("transition_map") or {})
            fusion_gate = online_side.get("fusion_gate") or fusion_gate
        if online_out is not None:
            online_class_map = dict(online_out.get("class_map") or {})
            if online_class_map:
                class_map.update(online_class_map)
            proto_map = dict(online_out.get("proto_map") or proto_map)
            transition_map = dict(online_out.get("transition_map") or transition_map)
            fusion_gate = online_out.get("fusion_gate") or fusion_gate
        semantic_map = dict(text_prior_map)
        if class_map:
            semantic_map.update(class_map)
        if text_prior_map or class_map or proto_map:
            label_scores = self._reweight_label_scores(
                base_scores=label_scores,
                seg_embeds=seg_embeddings,
                segment_lengths=[
                    max(
                        1,
                        int(seg.get("end_frame", seg.get("start_frame", 0)))
                        - int(seg.get("start_frame", 0))
                        + 1,
                    )
                    for seg in segments
                ],
                classes=classes,
                text_prior_map=text_prior_map,
                class_map=class_map,
                proto_map=proto_map,
                fusion_gate=fusion_gate,
            )
        if transition_map:
            label_scores = self._apply_transition_prior(
                label_scores=label_scores,
                classes=classes,
                transition_map=transition_map,
            )
        boundary_score, segments, label_scores, seg_embeddings, protection_stats = self._apply_hard_protection(
            features_dir=features_dir,
            features=features,
            boundary_score=boundary_score,
            segments=segments,
            label_scores=label_scores,
            classes=classes,
            text_prior_map=text_prior_map,
            class_map=class_map,
            proto_map=proto_map,
            fusion_gate=fusion_gate,
        )
        if any(
            int(protection_stats.get(k, 0)) > 0
            for k in (
                "label_spans",
                "negative_label_spans",
                "locked_segments",
                "boundary_frames",
                "removed_boundary_frames",
            )
        ):
            self._log(
                "[EAST] Applied hard protection: "
                f"label_spans={int(protection_stats.get('label_spans', 0))}, "
                f"negative_label_spans={int(protection_stats.get('negative_label_spans', 0))}, "
                f"locked_segments={int(protection_stats.get('locked_segments', 0))}, "
                f"boundary_frames={int(protection_stats.get('boundary_frames', 0))}, "
                f"removed_boundary_frames={int(protection_stats.get('removed_boundary_frames', 0))}"
            )
        segments, label_scores, seg_embeddings, gap_fill_stats = self._fill_segment_gaps(
            features=features,
            segments=segments,
            label_scores=label_scores,
            classes=classes,
            text_map=semantic_map,
            proto_map=proto_map,
        )
        if any(int(gap_fill_stats.get(k, 0)) > 0 for k in ("gap_ranges", "gap_frames")):
            self._log(
                "[EAST] Filled uncovered gaps: "
                f"ranges={int(gap_fill_stats.get('gap_ranges', 0))}, "
                f"frames={int(gap_fill_stats.get('gap_frames', 0))}"
            )
        if transition_map and label_scores.size > 0:
            label_scores = self._apply_transition_prior(
                label_scores=label_scores,
                classes=classes,
                transition_map=transition_map,
            )
        self._attach_topk(segments, label_scores, classes, k=5)

        payload = build_segments_payload(segments, classes, source="EAST")
        ok, err = validate_segments(payload, T=T, fps=30.0, label_list=classes)
        if not ok:
            raise ValueError(f"Invalid EAST segments format: {err}")

        frame_scores = self._frame_scores(payload["segments"], label_scores, T)
        transition = self._transition(payload["segments"], K=max(1, len(classes)))
        prototype = self._prototype(payload["segments"], seg_embeddings, K=max(1, len(classes)))

        meta = {
            "runtime_version": 1,
            "video_id": str(video_id or ""),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "ckpt_path": self.model_identity.get("ckpt_path", ""),
            "cfg_path": self.model_identity.get("cfg_path", ""),
            "ckpt_hash": self.model_identity.get("ckpt_hash", "missing"),
            "cfg_hash": self.model_identity.get("cfg_hash", "missing"),
            "video_path": str(resolved_video or video_path or ""),
            "text_bank_version": self.text_bank_version,
            "feature_backbone_version": self.feature_backbone_version,
            "label_bank_hash": label_hash,
            "feature_shape": [int(features.shape[0]), int(features.shape[1])],
            "train_step": self._read_train_step(runtime_dir),
            "online_adapter_step": int(
                max(
                    int((online_out or {}).get("step", 0)),
                    int((online_side or {}).get("step", 0)),
                    int(current_online_step),
                )
            ),
            "online_apply_mode": str(apply_mode),
            "fusion_gate_enabled": bool(fusion_gate),
            "vit_delta_enabled": bool(detector_delta_sources),
            "category_adapter_path": str(self.category_adapter_path or ""),
            "category_adapter_name": str(self._category_adapter_display_name() or ""),
            "category_adapter_signature": str(self._category_adapter_signature() or ""),
            "hard_protection_records": int(protection_stats.get("records", 0)),
            "hard_protection_label_spans": int(protection_stats.get("label_spans", 0)),
            "hard_protection_negative_label_spans": int(
                protection_stats.get("negative_label_spans", 0)
            ),
            "hard_protection_locked_segments": int(
                protection_stats.get("locked_segments", 0)
            ),
            "hard_protection_boundary_frames": int(protection_stats.get("boundary_frames", 0)),
            "hard_protection_removed_boundary_frames": int(
                protection_stats.get("removed_boundary_frames", 0)
            ),
            "transition_delta_pairs": int(
                sum(
                    len(row)
                    for row in (transition_map or {}).values()
                    if isinstance(row, dict)
                )
            ),
            "gap_fill_ranges": int(gap_fill_stats.get("gap_ranges", 0)),
            "gap_fill_frames": int(gap_fill_stats.get("gap_frames", 0)),
            "runtime_dirty": False,
        }

        self._save_runtime_artifacts(
            features_dir=features_dir,
            payload=payload,
            boundary_score=boundary_score,
            label_scores=label_scores,
            seg_embeddings=seg_embeddings,
            frame_scores=frame_scores,
            prototype=prototype,
            transition=transition,
            meta=meta,
        )

        runtime = {
            "segments_json": payload,
            "boundary_score": boundary_score,
            "label_scores": label_scores,
            "seg_embeddings": seg_embeddings,
            "meta": meta,
        }
        self._last_runtime = runtime
        self._log("[EAST] Runtime saved to east_runtime/")
        return payload, boundary_score, label_scores, seg_embeddings


def _load_labels(args: argparse.Namespace) -> List[str]:
    labels: List[str] = []
    if args.labels_json and os.path.isfile(args.labels_json):
        try:
            with open(args.labels_json, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                labels = [str(x).strip() for x in obj if str(x).strip()]
            elif isinstance(obj, dict) and isinstance(obj.get("labels"), list):
                labels = [str(x).strip() for x in obj.get("labels") if str(x).strip()]
        except Exception:
            labels = []

    label_txt = str(args.labels_txt or "").strip()
    if (not label_txt) or (not os.path.isfile(label_txt)):
        label_txt = resolve_label_source(
            features_dir=str(args.features_dir or ""),
            video_path=str(args.video_path or ""),
            repo_root=_REPO_ROOT,
        )
    if not labels and label_txt and os.path.isfile(label_txt):
        labels = load_label_names(label_txt)

    return normalize_label_bank(labels)


def main() -> int:
    ap = argparse.ArgumentParser(description="EAST runtime adapter: features -> segments.json/txt")
    ap.add_argument("--features_dir", required=True, help="Directory containing features.npy")
    ap.add_argument("--video_id", default="", help="Video id for runtime meta")
    ap.add_argument("--labels_json", default="", help="JSON list or {'labels':[...]} for label bank")
    ap.add_argument("--labels_txt", default="", help="TXT label bank (one label per line)")
    ap.add_argument("--video_path", default="", help="Optional raw video path for real EAST detector inference")
    ap.add_argument("--ckpt", default="", help="EAST checkpoint path (for cache-version binding)")
    ap.add_argument("--cfg", default="", help="EAST cfg path (for cache-version binding)")
    ap.add_argument("--text_bank_version", default="disabled")
    ap.add_argument("--feature_backbone_version", default="")
    ap.add_argument("--out_prefix", default="pred_east")
    ap.add_argument("--category_adapter", default="", help="Optional shared EAST adapter asset (.pt)")
    args = ap.parse_args()

    features_dir = os.path.abspath(os.path.expanduser(args.features_dir))
    video_id = str(args.video_id or os.path.basename(features_dir.rstrip(os.sep)))
    labels = _load_labels(args)

    runtime = EastRuntime(
        ckpt_path=args.ckpt,
        cfg_path=args.cfg,
        text_bank_version=args.text_bank_version,
        feature_backbone_version=args.feature_backbone_version or None,
        progress_cb=lambda m: print(m),
        category_adapter_path=args.category_adapter or None,
    )

    payload, boundary_score, label_scores, seg_embeddings = runtime.predict(
        video_id=video_id,
        features_dir=features_dir,
        label_bank=labels,
        video_path=args.video_path or None,
    )

    txt_path, json_path = write_segments_outputs(features_dir, args.out_prefix, payload)
    print(f"[EAST] segments_json: {json_path}")
    print(f"[EAST] segments_txt : {txt_path}")
    print(f"[EAST] boundary_score shape: {tuple(boundary_score.shape)}")
    print(f"[EAST] label_scores shape : {tuple(label_scores.shape)}")
    print(f"[EAST] seg_embeddings shape: {tuple(seg_embeddings.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
