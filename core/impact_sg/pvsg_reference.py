from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _normalize_label(label: str) -> str:
    raw = str(label or "").strip().lower()
    if raw in {"adult", "child", "baby", "man", "woman", "boy", "girl", "person", "people", "human"}:
        return "person"
    if raw == "ballon":
        return "balloon"
    return raw


def _normalize_relation(rel: str) -> str:
    return str(rel or "").strip().lower().replace(" ", "_")


def _bbox_iou(a: Sequence[int], b: Sequence[int]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, aw, ah = [float(x or 0.0) for x in a[:4]]
    bx1, by1, bw, bh = [float(x or 0.0) for x in b[:4]]
    ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
    bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = max(1e-6, area_a + area_b - inter)
    return float(inter / union)


def _f1(tp: int, fp: int, fn: int) -> float:
    p = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    r = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    if p + r <= 0:
        return 0.0
    return 2.0 * p * r / (p + r)


@lru_cache(maxsize=1)
def _load_pvsg_index(pvsg_json_path: str) -> Dict[str, Dict[str, Any]]:
    with open(pvsg_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for row in list((payload or {}).get("data") or []):
        if not isinstance(row, dict):
            continue
        vid = str(row.get("video_id", "") or "").strip()
        if vid:
            out[vid] = dict(row)
    return out


def _video_id_candidates_from_graph(graph: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    metadata = dict(graph.get("metadata") or {})
    image_path = str(metadata.get("image_path", "") or "").strip()
    if image_path:
        base = os.path.splitext(os.path.basename(image_path))[0]
        # e.g. 1203_8316378691_f000210 -> 1203_8316378691
        if "_f" in base:
            out.append(base.rsplit("_f", 1)[0])
        out.append(base)
    for key in ("video_id", "source_video_id"):
        v = str(metadata.get(key, "") or "").strip()
        if v:
            out.append(v)
    dedup: List[str] = []
    seen: set[str] = set()
    for v in out:
        if v and v not in seen:
            seen.add(v)
            dedup.append(v)
    return dedup


def _video_id_candidates_from_video_path(video_path: str) -> List[str]:
    out: List[str] = []
    stem = os.path.splitext(os.path.basename(str(video_path or "").strip()))[0]
    if stem:
        out.append(stem)
    # common fallback: parent folder may store video id naming
    parent = os.path.basename(os.path.dirname(str(video_path or "").strip()))
    if parent:
        out.append(parent)
    dedup: List[str] = []
    seen: set[str] = set()
    for v in out:
        key = str(v or "").strip()
        if key and key not in seen:
            seen.add(key)
            dedup.append(key)
    return dedup


def _frame_idx_from_graph(graph: Dict[str, Any]) -> int:
    metadata = dict(graph.get("metadata") or {})
    try:
        return int(metadata.get("graph_frame_idx", 0) or 0)
    except Exception:
        return 0


def _load_panoptic_index_png(mask_png_path: str) -> Optional[Tuple[int, int, List[int]]]:
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    if not os.path.isfile(mask_png_path):
        return None
    try:
        img = Image.open(mask_png_path)
        if img.mode != "P":
            img = img.convert("P")
        w, h = img.size
        data = list(img.getdata())
        return int(w), int(h), [int(x) for x in data]
    except Exception:
        return None


def _build_gt_nodes_from_mask(
    *,
    entry: Dict[str, Any],
    frame_idx: int,
    masks_root: str,
) -> Tuple[List[Dict[str, Any]], set[int]]:
    video_id = str(entry.get("video_id", "") or "").strip()
    png_path = os.path.join(masks_root, video_id, f"{int(frame_idx):04d}.png")
    packed = _load_panoptic_index_png(png_path)
    if packed is None:
        return [], set()
    w, h, values = packed
    boxes: Dict[int, List[int]] = {}
    for i, oid in enumerate(values):
        if int(oid) <= 0:
            continue
        x = int(i % w)
        y = int(i // w)
        cur = boxes.get(int(oid))
        if cur is None:
            boxes[int(oid)] = [x, y, x, y]
        else:
            if x < cur[0]:
                cur[0] = x
            if y < cur[1]:
                cur[1] = y
            if x > cur[2]:
                cur[2] = x
            if y > cur[3]:
                cur[3] = y

    object_map: Dict[int, Dict[str, Any]] = {}
    for row in list(entry.get("objects") or []):
        if not isinstance(row, dict):
            continue
        try:
            oid = int(row.get("object_id", -1) or -1)
        except Exception:
            oid = -1
        if oid > 0:
            object_map[oid] = dict(row)

    nodes: List[Dict[str, Any]] = []
    visible: set[int] = set()
    for oid, xyxy in boxes.items():
        info = dict(object_map.get(int(oid)) or {})
        label = _normalize_label(str(info.get("category", "") or "object"))
        x1, y1, x2, y2 = xyxy
        bbox = [int(x1), int(y1), int(max(0, x2 - x1 + 1)), int(max(0, y2 - y1 + 1))]
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        nodes.append(
            {
                "entity_id": str(oid),
                "object_id": int(oid),
                "canonical_label": label,
                "bbox": bbox,
            }
        )
        visible.add(int(oid))
    return nodes, visible


def _build_gt_edges_for_frame(entry: Dict[str, Any], *, frame_idx: int, visible_ids: set[int]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rel_rows = list(entry.get("relations") or [])
    for row in rel_rows:
        if not isinstance(row, list) or len(row) < 4:
            continue
        try:
            src = int(row[0])
            dst = int(row[1])
        except Exception:
            continue
        rel = _normalize_relation(str(row[2] or ""))
        spans = list(row[3] or [])
        if src not in visible_ids or dst not in visible_ids:
            continue
        active = False
        for span in spans:
            if not isinstance(span, list) or len(span) < 2:
                continue
            try:
                st = int(span[0])
                ed = int(span[1])
            except Exception:
                continue
            if st <= int(frame_idx) <= ed:
                active = True
                break
        if not active:
            continue
        out.append({"src_id": str(src), "dst_id": str(dst), "relation": rel})
    return out


def _match_nodes(pred_nodes: List[Dict[str, Any]], gt_nodes: List[Dict[str, Any]], *, iou_thr: float) -> Dict[int, int]:
    pairs: List[Tuple[float, int, int]] = []
    for pi, p in enumerate(pred_nodes):
        pl = _normalize_label(str(p.get("canonical_label", "") or ""))
        pb = list(p.get("bbox") or [0, 0, 0, 0])
        for gi, g in enumerate(gt_nodes):
            gl = _normalize_label(str(g.get("canonical_label", "") or ""))
            if pl != gl:
                continue
            gb = list(g.get("bbox") or [0, 0, 0, 0])
            iou = _bbox_iou(pb, gb)
            if iou >= float(iou_thr):
                pairs.append((float(iou), int(pi), int(gi)))
    pairs.sort(key=lambda x: x[0], reverse=True)
    pred_used: set[int] = set()
    gt_used: set[int] = set()
    matched: Dict[int, int] = {}
    for _iou, pi, gi in pairs:
        if pi in pred_used or gi in gt_used:
            continue
        pred_used.add(pi)
        gt_used.add(gi)
        matched[pi] = gi
    return matched


def evaluate_graph_against_pvsg(
    *,
    graph: Dict[str, Any],
    pvsg_json_path: str = "/cvhci/temp/wkong/sample_videos/pvsg.json",
    masks_root: str = "/cvhci/temp/wkong/sample_videos/VidOR/masks",
    node_iou_thr: float = 0.5,
) -> Optional[Dict[str, Any]]:
    pvsg_path = os.path.abspath(os.path.expanduser(str(pvsg_json_path or "").strip()))
    masks_dir = os.path.abspath(os.path.expanduser(str(masks_root or "").strip()))
    if not os.path.isfile(pvsg_path) or not os.path.isdir(masks_dir):
        return None

    try:
        pvsg_index = _load_pvsg_index(pvsg_path)
    except Exception:
        return None
    if not pvsg_index:
        return None

    candidates = _video_id_candidates_from_graph(graph)
    entry = None
    selected_vid = ""
    for vid in candidates:
        if vid in pvsg_index:
            entry = pvsg_index[vid]
            selected_vid = vid
            break
    if not isinstance(entry, dict):
        return None

    frame_idx = _frame_idx_from_graph(graph)
    gt_nodes, visible_ids = _build_gt_nodes_from_mask(entry=entry, frame_idx=frame_idx, masks_root=masks_dir)
    if not gt_nodes:
        return {
            "video_id": selected_vid,
            "frame_idx": int(frame_idx),
            "reference_available": False,
            "reason": "mask_or_objects_missing_for_frame",
        }
    gt_edges = _build_gt_edges_for_frame(entry, frame_idx=frame_idx, visible_ids=visible_ids)

    pred_nodes = [dict(x) for x in list(graph.get("nodes") or []) if isinstance(x, dict)]
    pred_edges = [dict(x) for x in list(graph.get("edges") or []) if isinstance(x, dict)]

    matches = _match_nodes(pred_nodes, gt_nodes, iou_thr=float(node_iou_thr))
    tp_nodes = len(matches)
    fp_nodes = max(0, len(pred_nodes) - tp_nodes)
    fn_nodes = max(0, len(gt_nodes) - tp_nodes)
    target_f1 = _f1(tp_nodes, fp_nodes, fn_nodes)

    gt_triplets = {(str(e.get("src_id", "")), _normalize_relation(str(e.get("relation", ""))), str(e.get("dst_id", ""))) for e in gt_edges}
    pred_triplets: set[Tuple[str, str, str]] = set()
    for edge in pred_edges:
        src = str(edge.get("src_id", "") or "")
        dst = str(edge.get("dst_id", "") or "")
        rel = _normalize_relation(str(edge.get("relation", "") or ""))
        pi_src = -1
        pi_dst = -1
        for pi, node in enumerate(pred_nodes):
            if str(node.get("entity_id", "") or "") == src:
                pi_src = pi
            if str(node.get("entity_id", "") or "") == dst:
                pi_dst = pi
        if pi_src < 0 or pi_dst < 0:
            continue
        if pi_src not in matches or pi_dst not in matches:
            continue
        gi_src = matches[pi_src]
        gi_dst = matches[pi_dst]
        src_oid = str(gt_nodes[gi_src].get("object_id", ""))
        dst_oid = str(gt_nodes[gi_dst].get("object_id", ""))
        if src_oid and dst_oid and rel:
            pred_triplets.add((src_oid, rel, dst_oid))

    rel_tp = len(pred_triplets.intersection(gt_triplets))
    rel_fp = len(pred_triplets - gt_triplets)
    rel_fn = len(gt_triplets - pred_triplets)
    edge_f1 = _f1(rel_tp, rel_fp, rel_fn)

    return {
        "video_id": selected_vid,
        "frame_idx": int(frame_idx),
        "reference_available": True,
        "target_accuracy_gt": _clamp01(target_f1),
        "edge_accuracy_gt": _clamp01(edge_f1),
        "node_match": {
            "tp": int(tp_nodes),
            "fp": int(fp_nodes),
            "fn": int(fn_nodes),
            "gt_nodes": int(len(gt_nodes)),
            "pred_nodes": int(len(pred_nodes)),
        },
        "edge_match": {
            "tp": int(rel_tp),
            "fp": int(rel_fp),
            "fn": int(rel_fn),
            "gt_edges": int(len(gt_triplets)),
            "pred_edges": int(len(pred_triplets)),
        },
    }


def load_pvsg_video_reference(
    *,
    video_path: str,
    frame_indices: Optional[Sequence[int]] = None,
    pvsg_json_path: str = "/cvhci/temp/wkong/sample_videos/pvsg.json",
    masks_root: str = "/cvhci/temp/wkong/sample_videos/VidOR/masks",
) -> Dict[str, Any]:
    """
    Build video-level PVSG reference summary and optional frame-level GT details.

    Returns a dict:
    {
      reference_available: bool,
      video_id: str,
      object_categories: [...],
      relation_types: [...],
      relation_spans: [{src_id,dst_id,relation,start,end}, ...],
      annotated_frame_ranges: [{start,end,relations}, ...],
      per_frame: {
        "20": {"objects":[...], "edges":[...], "reference_available": bool, "reason": "..."},
        ...
      }
    }
    """
    pvsg_path = os.path.abspath(os.path.expanduser(str(pvsg_json_path or "").strip()))
    masks_dir = os.path.abspath(os.path.expanduser(str(masks_root or "").strip()))
    if not os.path.isfile(pvsg_path):
        return {
            "reference_available": False,
            "video_path": str(video_path or ""),
            "reason": f"pvsg_json_missing:{pvsg_path}",
        }
    try:
        pvsg_index = _load_pvsg_index(pvsg_path)
    except Exception as exc:
        return {
            "reference_available": False,
            "video_path": str(video_path or ""),
            "reason": f"pvsg_json_load_failed:{exc}",
        }
    if not pvsg_index:
        return {
            "reference_available": False,
            "video_path": str(video_path or ""),
            "reason": "pvsg_index_empty",
        }

    selected_vid = ""
    entry: Optional[Dict[str, Any]] = None
    for vid in _video_id_candidates_from_video_path(video_path):
        if vid in pvsg_index:
            selected_vid = vid
            entry = dict(pvsg_index[vid] or {})
            break
    if not isinstance(entry, dict):
        return {
            "reference_available": False,
            "video_path": str(video_path or ""),
            "reason": "video_id_not_found_in_pvsg",
            "video_id_candidates": _video_id_candidates_from_video_path(video_path),
        }

    object_rows = [dict(x) for x in list(entry.get("objects") or []) if isinstance(x, dict)]
    categories = sorted(
        {
            _normalize_label(str(row.get("category", "") or ""))
            for row in object_rows
            if str(row.get("category", "") or "").strip()
        }
    )
    relation_types = sorted(
        {
            _normalize_relation(str(row[2] or ""))
            for row in list(entry.get("relations") or [])
            if isinstance(row, list) and len(row) >= 3 and str(row[2] or "").strip()
        }
    )

    relation_spans: List[Dict[str, Any]] = []
    annotated_ranges_raw: List[Tuple[int, int]] = []
    for row in list(entry.get("relations") or []):
        if not isinstance(row, list) or len(row) < 4:
            continue
        try:
            src = int(row[0])
            dst = int(row[1])
        except Exception:
            continue
        rel = _normalize_relation(str(row[2] or ""))
        for span in list(row[3] or []):
            if not isinstance(span, list) or len(span) < 2:
                continue
            try:
                st = int(span[0])
                ed = int(span[1])
            except Exception:
                continue
            relation_spans.append(
                {
                    "src_id": int(src),
                    "dst_id": int(dst),
                    "relation": rel,
                    "start": int(st),
                    "end": int(ed),
                }
            )
            if ed >= st:
                annotated_ranges_raw.append((int(st), int(ed)))

    annotated_ranges_raw.sort(key=lambda x: (x[0], x[1]))
    merged_ranges: List[List[int]] = []
    for st, ed in annotated_ranges_raw:
        if not merged_ranges:
            merged_ranges.append([st, ed])
            continue
        last = merged_ranges[-1]
        if st <= int(last[1]) + 1:
            last[1] = max(int(last[1]), int(ed))
        else:
            merged_ranges.append([st, ed])
    annotated_frame_ranges = [
        {"start": int(st), "end": int(ed), "relations": int(max(0, ed - st + 1))}
        for st, ed in merged_ranges
    ]

    per_frame: Dict[str, Dict[str, Any]] = {}
    if frame_indices:
        for frame_idx in sorted({int(x) for x in list(frame_indices or []) if int(x) >= 0}):
            if not os.path.isdir(masks_dir):
                per_frame[str(frame_idx)] = {
                    "reference_available": False,
                    "reason": f"masks_root_missing:{masks_dir}",
                    "objects": [],
                    "edges": [],
                }
                continue
            gt_nodes, visible_ids = _build_gt_nodes_from_mask(entry=entry, frame_idx=frame_idx, masks_root=masks_dir)
            if not gt_nodes:
                per_frame[str(frame_idx)] = {
                    "reference_available": False,
                    "reason": "mask_or_objects_missing_for_frame",
                    "objects": [],
                    "edges": [],
                }
                continue
            gt_edges = _build_gt_edges_for_frame(entry, frame_idx=frame_idx, visible_ids=visible_ids)
            per_frame[str(frame_idx)] = {
                "reference_available": True,
                "objects": [
                    {
                        "object_id": int(node.get("object_id", -1) or -1),
                        "label": str(node.get("canonical_label", "") or ""),
                        "bbox": [int(v) for v in list(node.get("bbox") or [0, 0, 0, 0])[:4]],
                    }
                    for node in gt_nodes
                ],
                "edges": [
                    {
                        "src_id": str(edge.get("src_id", "") or ""),
                        "relation": str(edge.get("relation", "") or ""),
                        "dst_id": str(edge.get("dst_id", "") or ""),
                    }
                    for edge in gt_edges
                ],
            }

    return {
        "reference_available": True,
        "video_path": str(video_path or ""),
        "video_id": str(selected_vid),
        "pvsg_json_path": pvsg_path,
        "masks_root": masks_dir,
        "object_categories": categories,
        "relation_types": relation_types,
        "relation_spans": relation_spans,
        "annotated_frame_ranges": annotated_frame_ranges,
        "objects_total": int(len(object_rows)),
        "relations_total": int(len(list(entry.get("relations") or []))),
        "frames_requested": int(len(list(frame_indices or []))) if frame_indices is not None else 0,
        "per_frame": per_frame,
    }
