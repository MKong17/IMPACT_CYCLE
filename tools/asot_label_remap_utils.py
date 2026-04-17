from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


DEFAULT_REMAP_FILENAMES = (
    "asot_label_remap.json",
    "label_remap.json",
)


def canonicalize_label_name(name: Any) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    text = text.replace("/", "_")
    text = text.replace("-", "_")
    text = text.replace(" ", "_")
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.upper()


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def resolve_asot_label_remap_path(
    *,
    explicit_path: str = "",
    class_names_path: str = "",
    features_dir: str = "",
    repo_root: str = "",
) -> str:
    def _ancestor_dirs(path: str, *, max_levels: int) -> List[str]:
        raw = str(path or "").strip()
        if not raw:
            return []
        cur = os.path.abspath(os.path.expanduser(raw))
        out: List[str] = []
        for _ in range(max(0, int(max_levels)) + 1):
            if cur not in out:
                out.append(cur)
            parent = os.path.dirname(cur)
            if (not parent) or parent == cur:
                break
            cur = parent
        return out

    direct_candidates = [
        str(explicit_path or "").strip(),
        str(os.environ.get("ASOT_LABEL_REMAP_JSON", "") or "").strip(),
    ]
    for raw in direct_candidates:
        if not raw:
            continue
        path = os.path.abspath(os.path.expanduser(raw))
        if os.path.isfile(path):
            return path

    search_dirs: List[str] = []
    candidate_dirs: List[str] = []
    candidate_dirs.extend(
        _ancestor_dirs(os.path.dirname(str(class_names_path or "").strip()), max_levels=0)
    )
    candidate_dirs.extend(_ancestor_dirs(str(features_dir or "").strip(), max_levels=3))
    if repo_root:
        candidate_dirs.append(str(repo_root))
    for raw_dir in candidate_dirs:
        if not raw_dir:
            continue
        path = os.path.abspath(os.path.expanduser(raw_dir))
        if path not in search_dirs:
            search_dirs.append(path)

    for root in search_dirs:
        for name in DEFAULT_REMAP_FILENAMES:
            cand = os.path.join(root, name)
            if os.path.isfile(cand):
                return cand
    return ""


def load_asot_label_remap(
    remap_path: str,
    semantic_classes: Sequence[str],
    *,
    num_clusters: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    path = str(remap_path or "").strip()
    if (not path) or (not os.path.isfile(path)):
        return None

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        raw_mapping: Any = {str(idx): value for idx, value in enumerate(obj)}
    elif isinstance(obj, dict):
        raw_mapping = (
            obj.get("cluster_to_label")
            or obj.get("mapping")
            or obj.get("cluster_to_class")
            or obj.get("remap")
        )
        if raw_mapping is None:
            raw_mapping = {
                key: value
                for key, value in obj.items()
                if str(key).strip().lstrip("+-").isdigit()
            }
    else:
        raw_mapping = None

    if not isinstance(raw_mapping, (dict, list)):
        return None

    classes = _dedupe_keep_order(semantic_classes or [])
    if not classes:
        return None

    sem_lookup = {
        canonicalize_label_name(name): idx
        for idx, name in enumerate(classes)
        if canonicalize_label_name(name)
    }
    size = max(int(num_clusters or 0), len(classes))
    cluster_to_semantic_idx = [
        idx if idx < len(classes) else 0 for idx in range(max(1, int(size)))
    ]
    cluster_to_label: List[str] = [
        classes[idx] if idx < len(classes) else classes[0]
        for idx in cluster_to_semantic_idx
    ]
    applied = 0

    items = raw_mapping.items() if isinstance(raw_mapping, dict) else enumerate(list(raw_mapping))
    for raw_key, raw_value in items:
        try:
            cluster_id = int(raw_key)
        except Exception:
            continue
        if cluster_id < 0:
            continue
        while cluster_id >= len(cluster_to_semantic_idx):
            cluster_to_semantic_idx.append(0)
            cluster_to_label.append(classes[0])

        semantic_idx: Optional[int] = None
        semantic_name: str = ""
        if isinstance(raw_value, dict):
            for key in ("semantic_id", "class_id", "id"):
                if key in raw_value:
                    try:
                        semantic_idx = int(raw_value.get(key))
                    except Exception:
                        semantic_idx = None
                    if semantic_idx is not None:
                        break
            semantic_name = str(
                raw_value.get("label")
                or raw_value.get("name")
                or raw_value.get("class_name")
                or ""
            ).strip()
        else:
            semantic_name = str(raw_value or "").strip()

        if semantic_idx is None and semantic_name:
            semantic_idx = sem_lookup.get(canonicalize_label_name(semantic_name))

        if semantic_idx is None or not (0 <= int(semantic_idx) < len(classes)):
            continue
        cluster_to_semantic_idx[cluster_id] = int(semantic_idx)
        cluster_to_label[cluster_id] = classes[int(semantic_idx)]
        applied += 1

    return {
        "path": os.path.abspath(path),
        "cluster_to_semantic_idx": cluster_to_semantic_idx,
        "cluster_to_label": cluster_to_label,
        "semantic_classes": list(classes),
        "clusters_applied": int(applied),
        "raw": obj,
    }


def remap_cluster_ids_to_semantic_ids(
    cluster_ids: np.ndarray,
    cluster_to_semantic_idx: Sequence[int],
    semantic_count: int,
) -> np.ndarray:
    raw = np.asarray(cluster_ids, dtype=np.int64)
    out = np.zeros(raw.shape, dtype=np.int32)
    mapping = list(cluster_to_semantic_idx or [])
    for cid in np.unique(raw):
        cid_int = int(cid)
        if 0 <= cid_int < len(mapping):
            sid = int(mapping[cid_int])
        elif 0 <= cid_int < int(semantic_count):
            sid = cid_int
        else:
            sid = 0
        sid = max(0, min(max(0, int(semantic_count) - 1), sid))
        out[raw == cid_int] = sid
    return out


def aggregate_cluster_probs_to_semantic_probs(
    cluster_probs: np.ndarray,
    cluster_to_semantic_idx: Sequence[int],
    semantic_count: int,
) -> np.ndarray:
    probs = np.asarray(cluster_probs, dtype=np.float32)
    if probs.ndim != 2 or semantic_count <= 0:
        return probs
    out = np.zeros((int(probs.shape[0]), int(semantic_count)), dtype=np.float32)
    mapping = list(cluster_to_semantic_idx or [])
    for cid in range(int(probs.shape[1])):
        if 0 <= cid < len(mapping):
            sid = int(mapping[cid])
        elif 0 <= cid < int(semantic_count):
            sid = cid
        else:
            sid = 0
        sid = max(0, min(max(0, int(semantic_count) - 1), sid))
        out[:, sid] += probs[:, cid]
    denom = np.maximum(out.sum(axis=1, keepdims=True), 1e-6)
    out = out / denom
    return out.astype(np.float32)
