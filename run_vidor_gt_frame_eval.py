from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# Ensure local imports work when executed as a standalone script.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core.impact_sg.cycle_pipeline import run_cycle_refine
from core.impact_sg.claim_graph import graph_to_claims
from core.impact_sg.detection_io import load_detection_record
from core.impact_sg.mllm_adapters.defaults import (  # noqa: E402
    DEFAULT_CYCLE_PROVIDER,
    LOW_QUOTA_API_MAX_OUTPUT_TOKENS,
    normalize_cycle_provider,
)
from core.impact_sg.mllm_adapters.factory import build_vision_verifier  # noqa: E402
from core.impact_sg.ontology import load_ontology  # noqa: E402
from core.impact_sg.pipeline import release_backend_pool, run_build_scene_graph  # noqa: E402
from core.impact_sg.pvsg_reference import load_pvsg_video_reference  # noqa: E402


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text or ""))


def _normalize_rel(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _normalize_label(value: object) -> str:
    token = str(value or "").strip().lower()
    if token in {"adult", "child", "baby", "man", "woman", "boy", "girl", "person", "people", "human"}:
        return "person"
    if token == "ballon":
        return "balloon"
    return token


def _bbox_iou(a: Sequence[object], b: Sequence[object]) -> float:
    if len(a) < 4 or len(b) < 4:
        return 0.0
    try:
        ax, ay, aw, ah = [float(a[i] or 0.0) for i in range(4)]
        bx, by, bw, bh = [float(b[i] or 0.0) for i in range(4)]
    except Exception:
        return 0.0
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    den = area_a + area_b - inter
    if den <= 1e-6:
        return 0.0
    return float(inter / den)


def _f1(tp: int, fp: int, fn: int) -> float:
    p = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    r = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    if p + r <= 0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _annotated_frames_from_entry(entry: Dict[str, object]) -> List[int]:
    frames: Set[int] = set()
    for rel in list(entry.get("relations") or []):
        if not isinstance(rel, list) or len(rel) < 4:
            continue
        spans = list(rel[3] or [])
        for span in spans:
            if not isinstance(span, list) or len(span) < 2:
                continue
            try:
                st = int(span[0])
                ed = int(span[1])
            except Exception:
                continue
            if ed < st:
                continue
            for idx in range(st, ed + 1):
                frames.add(int(idx))
    return sorted(frames)


_DYNAMIC_VERIFY_RELATIONS = {
    "holding",
    "looking_at",
    "blowing",
    "pointing_to",
    "picking",
    "sitting_on",
}

_STATIC_VERIFY_RELATIONS = {
    "on",
    "next_to",
    "in_front_of",
    "behind",
    "inside",
}


def _sample_span_frames(start_idx: int, end_idx: int, *, sample_mode: str) -> List[int]:
    if int(end_idx) < int(start_idx):
        return []
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    mid_idx = int((start_idx + end_idx) // 2)
    if str(sample_mode or "").strip().lower() == "static":
        return [mid_idx]
    return sorted({start_idx, mid_idx, end_idx})


def _sample_verify_frames_from_entry(entry: Dict[str, object]) -> List[int]:
    frames: Set[int] = set()
    for rel in list(entry.get("relations") or []):
        if not isinstance(rel, list) or len(rel) < 4:
            continue
        relation = _normalize_rel(rel[2] if len(rel) > 2 else "")
        sample_mode = "dynamic"
        if relation in _STATIC_VERIFY_RELATIONS:
            sample_mode = "static"
        elif relation in _DYNAMIC_VERIFY_RELATIONS:
            sample_mode = "dynamic"
        spans = list(rel[3] or [])
        for span in spans:
            if not isinstance(span, list) or len(span) < 2:
                continue
            try:
                start_idx = int(span[0])
                end_idx = int(span[1])
            except Exception:
                continue
            for frame_idx in _sample_span_frames(start_idx, end_idx, sample_mode=sample_mode):
                frames.add(int(frame_idx))
    return sorted(frames)


def _limit_frames_uniform(frames: List[int], max_frames: int) -> List[int]:
    ordered = [int(x) for x in list(frames or [])]
    limit = int(max_frames or 0)
    if limit <= 0 or len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    if limit == 2:
        return [ordered[0], ordered[-1]]
    last_idx = len(ordered) - 1
    picks: List[int] = []
    for slot in range(limit):
        pos = int(round(float(slot) * float(last_idx) / float(limit - 1)))
        pos = max(0, min(last_idx, pos))
        picks.append(pos)
    deduped_positions: List[int] = []
    for pos in picks:
        if pos not in deduped_positions:
            deduped_positions.append(pos)
    if len(deduped_positions) < limit:
        for pos in range(last_idx + 1):
            if pos not in deduped_positions:
                deduped_positions.append(pos)
            if len(deduped_positions) >= limit:
                break
    deduped_positions = sorted(deduped_positions[:limit])
    return [ordered[pos] for pos in deduped_positions]


def _resolve_video_path(videos_dir: str, video_id: str) -> str:
    base = str(video_id or "").strip()
    if not base:
        return ""
    for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        candidate = os.path.join(videos_dir, base + ext)
        if os.path.isfile(candidate):
            return candidate
    # Fallback: exact name in folder.
    for name in os.listdir(videos_dir):
        stem, _ext = os.path.splitext(name)
        if stem == base:
            candidate = os.path.join(videos_dir, name)
            if os.path.isfile(candidate):
                return candidate
    return ""


def _extract_frame_to_jpg(video_path: str, frame_idx: int, out_dir: str) -> Tuple[str, int, int]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for frame extraction (cv2 import failed).") from exc
    cap = cv2.VideoCapture(video_path)
    if cap is None or not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to decode frame {frame_idx} from {video_path}")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_f{int(frame_idx):06d}.jpg")
    if not cv2.imwrite(out_path, frame):
        raise RuntimeError(f"Unable to write frame image: {out_path}")
    h, w = frame.shape[:2]
    return out_path, int(w), int(h)


def _gt_graph_from_frame_ref(video_id: str, frame_idx: int, frame_ref: Dict[str, object]) -> Dict[str, object]:
    gt_nodes: List[Dict[str, object]] = []
    gt_edges: List[Dict[str, object]] = []
    for obj in list(frame_ref.get("objects") or []):
        if not isinstance(obj, dict):
            continue
        oid = str(obj.get("object_id", "") or "").strip()
        if not oid:
            continue
        gt_nodes.append(
            {
                "entity_id": oid,
                "canonical_label": _normalize_label(obj.get("label", "")),
                "bbox": [int(v) for v in list(obj.get("bbox") or [0, 0, 0, 0])[:4]],
                "attributes": [],
            }
        )
    for edge in list(frame_ref.get("edges") or []):
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("src_id", "") or "").strip()
        dst = str(edge.get("dst_id", "") or "").strip()
        rel = _normalize_rel(edge.get("relation", ""))
        if src and dst and rel:
            gt_edges.append({"src_id": src, "dst_id": dst, "relation": rel})
    return {
        "image_id": f"{video_id}_f{int(frame_idx):06d}",
        "nodes": gt_nodes,
        "edges": gt_edges,
        "metadata": {"graph_frame_idx": int(frame_idx), "video_id": video_id},
    }


@dataclass
class NodeMatchResult:
    pred_to_gt: Dict[int, int]
    matched_ious: List[float]


def _match_nodes(pred_nodes: List[Dict[str, object]], gt_nodes: List[Dict[str, object]], iou_thr: float) -> NodeMatchResult:
    candidates: List[Tuple[float, int, int]] = []
    for pi, pred in enumerate(pred_nodes):
        plabel = _normalize_label(pred.get("canonical_label", pred.get("label", "")))
        pbbox = list(pred.get("bbox") or [0, 0, 0, 0])
        for gi, gt in enumerate(gt_nodes):
            glabel = _normalize_label(gt.get("canonical_label", gt.get("label", "")))
            if plabel != glabel:
                continue
            iou = _bbox_iou(pbbox, list(gt.get("bbox") or [0, 0, 0, 0]))
            if iou >= float(iou_thr):
                candidates.append((float(iou), int(pi), int(gi)))
    candidates.sort(key=lambda x: x[0], reverse=True)
    used_pred: Set[int] = set()
    used_gt: Set[int] = set()
    mapping: Dict[int, int] = {}
    ious: List[float] = []
    for iou, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        mapping[pi] = gi
        ious.append(float(iou))
    return NodeMatchResult(pred_to_gt=mapping, matched_ious=ious)


def _matched_label_pairs(pred_graph: Dict[str, object], gt_graph: Dict[str, object], *, iou_thr: float) -> List[str]:
    pred_nodes = [dict(x) for x in list(pred_graph.get("nodes") or []) if isinstance(x, dict)]
    gt_nodes = [dict(x) for x in list(gt_graph.get("nodes") or []) if isinstance(x, dict)]
    matches = _match_nodes(pred_nodes, gt_nodes, iou_thr=float(iou_thr))
    out: List[str] = []
    for pred_idx, gt_idx in sorted(matches.pred_to_gt.items(), key=lambda item: int(item[0])):
        pred_node = dict(pred_nodes[pred_idx] or {})
        gt_node = dict(gt_nodes[gt_idx] or {})
        pred_label = _normalize_label(pred_node.get("canonical_label", pred_node.get("label", "")))
        gt_label = _normalize_label(gt_node.get("canonical_label", gt_node.get("label", "")))
        out.append(
            f"{str(pred_node.get('entity_id', '') or '').strip() or f'pred#{pred_idx}'}:{pred_label} vs "
            f"{str(gt_node.get('entity_id', '') or '').strip() or f'gt#{gt_idx}'}:{gt_label}"
        )
    return out


def _evaluate_graph(pred_graph: Dict[str, object], gt_graph: Dict[str, object], *, iou_thr: float) -> Dict[str, object]:
    pred_nodes = [dict(x) for x in list(pred_graph.get("nodes") or []) if isinstance(x, dict)]
    gt_nodes = [dict(x) for x in list(gt_graph.get("nodes") or []) if isinstance(x, dict)]
    pred_edges = [dict(x) for x in list(pred_graph.get("edges") or []) if isinstance(x, dict)]
    gt_edges = [dict(x) for x in list(gt_graph.get("edges") or []) if isinstance(x, dict)]

    matches = _match_nodes(pred_nodes, gt_nodes, iou_thr=float(iou_thr))
    tp_nodes = len(matches.pred_to_gt)
    fp_nodes = max(0, len(pred_nodes) - tp_nodes)
    fn_nodes = max(0, len(gt_nodes) - tp_nodes)

    pred_eid_to_idx = {
        str(node.get("entity_id", "") or "").strip(): idx
        for idx, node in enumerate(pred_nodes)
        if str(node.get("entity_id", "") or "").strip()
    }
    gt_oid_by_idx = {
        idx: str(node.get("entity_id", "") or "").strip()
        for idx, node in enumerate(gt_nodes)
    }

    pred_triplets: Set[Tuple[str, str, str]] = set()
    for edge in pred_edges:
        src = str(edge.get("src_id", "") or "").strip()
        dst = str(edge.get("dst_id", "") or "").strip()
        rel = _normalize_rel(edge.get("relation", ""))
        src_pi = pred_eid_to_idx.get(src, -1)
        dst_pi = pred_eid_to_idx.get(dst, -1)
        if src_pi < 0 or dst_pi < 0:
            continue
        if src_pi not in matches.pred_to_gt or dst_pi not in matches.pred_to_gt:
            continue
        src_oid = gt_oid_by_idx.get(matches.pred_to_gt[src_pi], "")
        dst_oid = gt_oid_by_idx.get(matches.pred_to_gt[dst_pi], "")
        if src_oid and dst_oid and rel:
            pred_triplets.add((src_oid, rel, dst_oid))

    gt_triplets = {
        (
            str(edge.get("src_id", "") or "").strip(),
            _normalize_rel(edge.get("relation", "")),
            str(edge.get("dst_id", "") or "").strip(),
        )
        for edge in gt_edges
        if str(edge.get("src_id", "") or "").strip()
        and str(edge.get("dst_id", "") or "").strip()
        and _normalize_rel(edge.get("relation", ""))
    }
    rel_tp = len(pred_triplets.intersection(gt_triplets))
    rel_fp = len(pred_triplets - gt_triplets)
    rel_fn = len(gt_triplets - pred_triplets)

    gt_attrs: Set[Tuple[str, str, str]] = set()
    pred_attrs: Set[Tuple[str, str, str]] = set()
    for gi, gnode in enumerate(gt_nodes):
        oid = str(gnode.get("entity_id", "") or "").strip()
        for attr in list(gnode.get("attributes") or []):
            if not isinstance(attr, dict):
                continue
            slot = str(attr.get("slot", "") or "").strip()
            value = str(attr.get("value", "") or "").strip()
            if oid and slot and value:
                gt_attrs.add((oid, slot, value))
    for pi, gi in matches.pred_to_gt.items():
        oid = gt_oid_by_idx.get(gi, "")
        pnode = pred_nodes[pi]
        for attr in list(pnode.get("attributes") or []):
            if not isinstance(attr, dict):
                continue
            slot = str(attr.get("slot", "") or "").strip()
            value = str(attr.get("value", "") or "").strip()
            if oid and slot and value:
                pred_attrs.add((oid, slot, value))
    attr_tp = len(pred_attrs.intersection(gt_attrs))
    attr_fp = len(pred_attrs - gt_attrs)
    attr_fn = len(gt_attrs - pred_attrs)
    attr_available = len(gt_attrs) > 0

    return {
        "node": {
            "tp": int(tp_nodes),
            "fp": int(fp_nodes),
            "fn": int(fn_nodes),
            "gt_count": int(len(gt_nodes)),
            "pred_count": int(len(pred_nodes)),
            "precision": float(tp_nodes) / float(max(1, tp_nodes + fp_nodes)),
            "recall": float(tp_nodes) / float(max(1, tp_nodes + fn_nodes)),
            "f1": _f1(tp_nodes, fp_nodes, fn_nodes),
            "label_correctness": float(tp_nodes) / float(max(1, len(gt_nodes))),
            "matched_iou_mean": (
                float(sum(matches.matched_ious) / len(matches.matched_ious))
                if matches.matched_ious
                else 0.0
            ),
        },
        "relation": {
            "tp": int(rel_tp),
            "fp": int(rel_fp),
            "fn": int(rel_fn),
            "gt_count": int(len(gt_triplets)),
            "pred_count": int(len(pred_triplets)),
            "f1": _f1(rel_tp, rel_fp, rel_fn),
            "correctness": float(rel_tp) / float(max(1, len(gt_triplets))),
        },
        "attribute": {
            "available": bool(attr_available),
            "tp": int(attr_tp),
            "fp": int(attr_fp),
            "fn": int(attr_fn),
            "gt_count": int(len(gt_attrs)),
            "pred_count": int(len(pred_attrs)),
            "f1": _f1(attr_tp, attr_fp, attr_fn) if attr_available else 0.0,
        },
    }


def _evaluate_instances(
    pred_graph: Dict[str, object],
    gt_graph: Dict[str, object],
    *,
    mask_or_box_mode: str = "bbox",
    iou_thr: float = 0.5,
) -> Dict[str, object]:
    pred_nodes = [dict(x) for x in list(pred_graph.get("nodes") or []) if isinstance(x, dict)]
    gt_nodes = [dict(x) for x in list(gt_graph.get("nodes") or []) if isinstance(x, dict)]
    matches = _match_nodes(pred_nodes, gt_nodes, iou_thr=float(iou_thr))
    tp = int(len(matches.pred_to_gt))
    fp = int(max(0, len(pred_nodes) - tp))
    fn = int(max(0, len(gt_nodes) - tp))
    precision = float(tp) / float(max(1, tp + fp))
    recall = float(tp) / float(max(1, tp + fn))
    f1 = _f1(tp, fp, fn)
    matched_iou_mean = float(sum(matches.matched_ious) / len(matches.matched_ious)) if matches.matched_ious else 0.0
    return {
        "mode": str(mask_or_box_mode or "bbox"),
        "iou_thr": float(iou_thr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_count": int(len(gt_nodes)),
        "pred_count": int(len(pred_nodes)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_iou_mean": matched_iou_mean,
        "miou": matched_iou_mean,
    }


def _claim_truth_map(
    graph: Dict[str, object],
    gt_graph: Dict[str, object],
    *,
    iou_thr: float,
    claims: Optional[List[object]] = None,
) -> Dict[str, Dict[str, object]]:
    pred_nodes = [dict(x) for x in list(graph.get("nodes") or []) if isinstance(x, dict)]
    gt_nodes = [dict(x) for x in list(gt_graph.get("nodes") or []) if isinstance(x, dict)]
    gt_edges = [dict(x) for x in list(gt_graph.get("edges") or []) if isinstance(x, dict)]
    matches = _match_nodes(pred_nodes, gt_nodes, iou_thr=float(iou_thr))

    pred_eid_to_idx = {
        str(node.get("entity_id", "") or "").strip(): idx
        for idx, node in enumerate(pred_nodes)
        if str(node.get("entity_id", "") or "").strip()
    }
    gt_oid_by_idx = {
        idx: str(node.get("entity_id", "") or "").strip()
        for idx, node in enumerate(gt_nodes)
    }
    gt_label_by_oid = {
        str(node.get("entity_id", "") or "").strip(): _normalize_label(node.get("canonical_label", node.get("label", "")))
        for node in gt_nodes
        if str(node.get("entity_id", "") or "").strip()
    }
    gt_triplets = {
        (
            str(edge.get("src_id", "") or "").strip(),
            _normalize_rel(edge.get("relation", "")),
            str(edge.get("dst_id", "") or "").strip(),
        )
        for edge in gt_edges
        if str(edge.get("src_id", "") or "").strip()
        and str(edge.get("dst_id", "") or "").strip()
        and _normalize_rel(edge.get("relation", ""))
    }
    gt_attrs: Set[Tuple[str, str, str]] = set()
    for node in gt_nodes:
        oid = str(node.get("entity_id", "") or "").strip()
        for attr in list(node.get("attributes") or []):
            if not isinstance(attr, dict):
                continue
            slot = str(attr.get("slot", "") or "").strip()
            value = str(attr.get("value", "") or "").strip()
            if oid and slot and value:
                gt_attrs.add((oid, slot, value))
    has_gt_attrs = bool(gt_attrs)

    claim_rows = list(claims) if isinstance(claims, list) else list(graph_to_claims(graph))
    out: Dict[str, Dict[str, object]] = {}
    for claim in claim_rows:
        claim_id = str(claim.claim_id or "").strip()
        claim_type = str(claim.claim_type or "").strip().lower()
        subject_id = str(claim.subject_id or "").strip()
        object_id = str(claim.object_id or "").strip()
        predicate = str(claim.predicate or "").strip()
        value = str(claim.value or "").strip()
        supported = True
        is_true = False

        subject_idx = pred_eid_to_idx.get(subject_id, -1)
        object_idx = pred_eid_to_idx.get(object_id, -1)
        subject_gt_id = gt_oid_by_idx.get(matches.pred_to_gt.get(subject_idx, -1), "") if subject_idx >= 0 else ""
        object_gt_id = gt_oid_by_idx.get(matches.pred_to_gt.get(object_idx, -1), "") if object_idx >= 0 else ""

        if claim_type == "existence":
            is_true = bool(subject_gt_id)
        elif claim_type == "label":
            if subject_gt_id:
                is_true = _normalize_label(value) == str(gt_label_by_oid.get(subject_gt_id, "") or "").strip()
            else:
                is_true = False
        elif claim_type == "relation":
            triplet = (subject_gt_id, _normalize_rel(predicate), object_gt_id)
            is_true = bool(subject_gt_id and object_gt_id and triplet in gt_triplets)
        elif claim_type == "attribute":
            if not has_gt_attrs:
                supported = False
                is_true = False
            else:
                is_true = bool(subject_gt_id and (subject_gt_id, str(predicate).strip(), str(value).strip()) in gt_attrs)
        else:
            supported = False
            is_true = False

        out[claim_id] = {
            "claim_id": claim_id,
            "claim_type": claim_type,
            "supported": bool(supported),
            "is_true": bool(is_true),
        }
    return out


def _aggregate_vote_directions(votes: List[Dict[str, object]]) -> Dict[str, str]:
    buckets: Dict[str, Dict[str, float]] = {}
    for row in list(votes or []):
        if not isinstance(row, dict):
            continue
        claim_id = str(row.get("claim_id", "") or "").strip()
        vote = str(row.get("vote", "uncertain") or "uncertain").strip().lower()
        if not claim_id:
            continue
        if vote not in {"support", "conflict", "uncertain"}:
            vote = "uncertain"
        score = _safe_float(row.get("score", 0.0), 0.0)
        slot = buckets.setdefault(claim_id, {"support": 0.0, "conflict": 0.0, "uncertain": 0.0})
        slot[vote] = float(slot.get(vote, 0.0) + max(0.0, score))
    out: Dict[str, str] = {}
    for claim_id, bucket in buckets.items():
        ranked = sorted(
            [(str(k), _safe_float(v, 0.0)) for k, v in dict(bucket).items()],
            key=lambda item: (-float(item[1]), str(item[0])),
        )
        out[claim_id] = str(ranked[0][0] if ranked else "uncertain")
    return out


def _evaluate_verify_effectiveness(
    initial_graph: Dict[str, object],
    final_graph: Dict[str, object],
    gt_graph: Dict[str, object],
    cycle_result: Dict[str, object],
    *,
    iou_thr: float,
) -> Dict[str, object]:
    claim_universe = list(graph_to_claims(initial_graph) or [])
    before_truth = _claim_truth_map(initial_graph, gt_graph, iou_thr=float(iou_thr), claims=claim_universe)
    after_truth = _claim_truth_map(final_graph, gt_graph, iou_thr=float(iou_thr), claims=claim_universe)
    vote_rows = [dict(x) for x in list(cycle_result.get("votes") or []) if isinstance(x, dict)]
    vote_direction = _aggregate_vote_directions(vote_rows)

    def _count_supported(truth_map: Dict[str, Dict[str, object]]) -> Tuple[int, int, int]:
        supported_rows = [row for row in truth_map.values() if bool(row.get("supported", False))]
        true_rows = [row for row in supported_rows if bool(row.get("is_true", False))]
        false_rows = [row for row in supported_rows if not bool(row.get("is_true", False))]
        return len(supported_rows), len(true_rows), len(false_rows)

    supported_before, true_before, false_before = _count_supported(before_truth)
    supported_after, true_after, false_after = _count_supported(after_truth)

    voted_supported = 0
    direction_correct = 0
    true_support_hits = 0
    true_conflict_errors = 0
    false_conflict_hits = 0
    false_support_errors = 0
    for claim_id, direction in vote_direction.items():
        truth = dict(before_truth.get(claim_id) or {})
        if not bool(truth.get("supported", False)):
            continue
        voted_supported += 1
        is_true = bool(truth.get("is_true", False))
        if is_true and direction == "support":
            direction_correct += 1
            true_support_hits += 1
        elif (not is_true) and direction == "conflict":
            direction_correct += 1
            false_conflict_hits += 1
        elif is_true and direction == "conflict":
            true_conflict_errors += 1
        elif (not is_true) and direction == "support":
            false_support_errors += 1

    return {
        "before": {
            "supported_claims": int(supported_before),
            "true_claims": int(true_before),
            "false_claims": int(false_before),
            "claim_accuracy": float(true_before) / float(max(1, supported_before)),
        },
        "after": {
            "supported_claims": int(supported_after),
            "true_claims": int(true_after),
            "false_claims": int(false_after),
            "claim_accuracy": float(true_after) / float(max(1, supported_after)),
        },
        "vote_eval": {
            "voted_supported_claims": int(voted_supported),
            "vote_direction_accuracy": float(direction_correct) / float(max(1, voted_supported)),
            "true_claim_support_rate": float(true_support_hits) / float(max(1, true_before)),
            "false_claim_conflict_rate": float(false_conflict_hits) / float(max(1, false_before)),
            "true_claim_conflict_error_rate": float(true_conflict_errors) / float(max(1, true_before)),
            "false_claim_support_error_rate": float(false_support_errors) / float(max(1, false_before)),
        },
        "correction": {
            "false_claim_reduction": int(false_before - false_after),
            "false_claim_reduction_rate": float(false_before - false_after) / float(max(1, false_before)),
            "true_claim_delta": int(true_after - true_before),
            "correction_hit": bool(false_after < false_before),
        },
    }


def _is_invalid_probe_response(resp: Dict[str, object]) -> bool:
    schema_valid = bool(resp.get("schema_valid", True))
    is_truncated = bool(resp.get("is_truncated", False))
    is_valid_flag = resp.get("is_valid", None)
    explicitly_invalid = bool(is_valid_flag is False)
    schema_invalid_and_not_salvaged = (not schema_valid) and (is_valid_flag is not True)
    return bool(schema_invalid_and_not_salvaged or is_truncated or explicitly_invalid)


def _cycle_probe_stats(result: Dict[str, object]) -> Dict[str, object]:
    probe_rows = [dict(x) for x in list(result.get("probe_results") or []) if isinstance(x, dict)]
    votes = [dict(x) for x in list(result.get("votes") or []) if isinstance(x, dict)]
    human_queue = [dict(x) for x in list(result.get("human_queue") or []) if isinstance(x, dict)]
    summary = dict(result.get("summary") or {})

    support = 0
    conflict = 0
    uncertain = 0
    by_view_vote: Dict[str, Dict[str, int]] = {}
    for row in votes:
        view = str(row.get("view_type", "unknown") or "unknown")
        vote = str(row.get("vote", "uncertain") or "uncertain").strip().lower()
        if vote == "support":
            support += 1
        elif vote == "conflict":
            conflict += 1
        else:
            uncertain += 1
            vote = "uncertain"
        slot = by_view_vote.setdefault(view, {"support": 0, "conflict": 0, "uncertain": 0})
        slot[vote] = int(slot.get(vote, 0) + 1)

    by_view_probe: Dict[str, Dict[str, int]] = {}
    invalid_count = 0
    for row in probe_rows:
        view = str(row.get("view_type", "unknown") or "unknown")
        resp = dict(row.get("response") or {})
        invalid = _is_invalid_probe_response(resp)
        if invalid:
            invalid_count += 1
        slot = by_view_probe.setdefault(view, {"total": 0, "invalid": 0, "schema_invalid": 0, "truncated": 0})
        slot["total"] += 1
        if invalid:
            slot["invalid"] += 1
        if not bool(resp.get("schema_valid", True)):
            slot["schema_invalid"] += 1
        if bool(resp.get("is_truncated", False)):
            slot["truncated"] += 1

    caption_payload = dict(result.get("caption") or {})
    caption_feedback = dict(caption_payload.get("feedback") or {})
    caption_stats = {
        "structured": bool(caption_feedback.get("structured", False)),
        "schema_valid": bool(caption_payload.get("schema_valid", True)),
        "is_valid": bool(caption_payload.get("is_valid", True)),
        "vote_count": int(len(list(caption_payload.get("votes") or []))),
        "support_vote_count": int(caption_feedback.get("support_vote_count", 0) or 0),
        "conflict_vote_count": int(caption_feedback.get("conflict_vote_count", 0) or 0),
        "hallucinated_mentions_count": int(len(list(caption_feedback.get("hallucinated_mentions") or []))),
    }

    return {
        "votes": {
            "support": int(support),
            "conflict": int(conflict),
            "uncertain": int(uncertain),
            "total": int(len(votes)),
            "by_view_type": by_view_vote,
        },
        "probes": {
            "total": int(len(probe_rows)),
            "invalid": int(invalid_count),
            "by_view_type": by_view_probe,
        },
        "claims": {
            "accepted_claim_count": int(summary.get("accepted_claim_count", 0) or 0),
            "accepted_confirm_count": int(summary.get("accepted_confirm_count", 0) or 0),
            "accepted_correct_count": int(summary.get("accepted_correct_count", 0) or 0),
            "flagged_claim_count": int(summary.get("flagged_claim_count", 0) or 0),
            "queue_count": int(len(human_queue)),
        },
        "caption": caption_stats,
    }


def _merge_count_dict(dst: Dict[str, int], src: Dict[str, int]) -> Dict[str, int]:
    out = dict(dst)
    for key, value in dict(src or {}).items():
        out[str(key)] = int(out.get(str(key), 0) + int(value or 0))
    return out


def _build_cycle_cfg(
    base_cfg: Dict[str, object],
    *,
    provider: str,
    rounds: int,
    enable_single: Optional[bool],
    enable_multi: Optional[bool],
    enable_caption: Optional[bool],
    low_quota: bool,
) -> Dict[str, object]:
    cfg = json.loads(json.dumps(base_cfg))
    cycle = dict(cfg.get("cycle") or {})
    runtime = dict(cfg.get("runtime") or {})
    api_verifier = dict(cfg.get("api_verifier") or {})
    local_verifier = dict(cfg.get("local_verifier") or {})

    cycle["max_revision_rounds"] = max(1, int(rounds or 1))
    if enable_single is not None:
        cycle["enable_single_turn_probes"] = bool(enable_single)
    if enable_multi is not None:
        cycle["enable_multi_turn_probes"] = bool(enable_multi)
    if enable_caption is not None:
        cycle["enable_caption_probe"] = bool(enable_caption)
    if low_quota:
        cycle["enable_multi_turn_probes"] = False
        cycle["enable_caption_probe"] = False
        api_verifier["max_output_tokens"] = int(
            min(
                int(api_verifier.get("max_output_tokens", LOW_QUOTA_API_MAX_OUTPUT_TOKENS) or LOW_QUOTA_API_MAX_OUTPUT_TOKENS),
                int(LOW_QUOTA_API_MAX_OUTPUT_TOKENS),
            )
        )

    runtime["preferred_provider"] = str(provider)
    if provider == "gemini_api":
        api_verifier["enabled"] = True
        api_verifier["provider"] = "gemini"
    elif provider == "chatgpt_api":
        api_verifier["enabled"] = True
        api_verifier["provider"] = "openai"
    elif provider == "qwen25_vl":
        local_verifier["provider"] = "qwen25_vl"
    elif provider == "manual":
        runtime["preferred_provider"] = "manual"

    cfg["cycle"] = cycle
    cfg["runtime"] = runtime
    cfg["api_verifier"] = api_verifier
    cfg["local_verifier"] = local_verifier
    return cfg


def _video_summary_from_frames(rows: List[Dict[str, object]]) -> Dict[str, object]:
    frame_count = len(rows)
    build_failed = len([r for r in rows if str(r.get("status", "")).startswith("build_failed")])
    cycle_failed = len([r for r in rows if str(r.get("status", "")).startswith("cycle_failed")])
    succeeded = len([r for r in rows if str(r.get("status", "")).strip().lower() == "ok"])

    claim_acc_before = [_safe_float((((r.get("verify_eval") or {}).get("before") or {}).get("claim_accuracy", 0.0)), 0.0) for r in rows]
    claim_acc_after = [_safe_float((((r.get("verify_eval") or {}).get("after") or {}).get("claim_accuracy", 0.0)), 0.0) for r in rows]
    false_claims_before = [int((((r.get("verify_eval") or {}).get("before") or {}).get("false_claims", 0) or 0)) for r in rows]
    false_claims_after = [int((((r.get("verify_eval") or {}).get("after") or {}).get("false_claims", 0) or 0)) for r in rows]
    true_claims_before = [int((((r.get("verify_eval") or {}).get("before") or {}).get("true_claims", 0) or 0)) for r in rows]
    true_claims_after = [int((((r.get("verify_eval") or {}).get("after") or {}).get("true_claims", 0) or 0)) for r in rows]
    vote_direction_accuracy = [_safe_float((((r.get("verify_eval") or {}).get("vote_eval") or {}).get("vote_direction_accuracy", 0.0)), 0.0) for r in rows]
    true_claim_support_rate = [_safe_float((((r.get("verify_eval") or {}).get("vote_eval") or {}).get("true_claim_support_rate", 0.0)), 0.0) for r in rows]
    false_claim_conflict_rate = [_safe_float((((r.get("verify_eval") or {}).get("vote_eval") or {}).get("false_claim_conflict_rate", 0.0)), 0.0) for r in rows]
    false_claim_reduction = [int((((r.get("verify_eval") or {}).get("correction") or {}).get("false_claim_reduction", 0) or 0)) for r in rows]
    sampled_frame_count = sum(int((r.get("sampling") or {}).get("verify_frames_sampled", 0) or 0) for r in rows[:1])
    annotated_frame_count = sum(int((r.get("sampling") or {}).get("annotated_frames_total", 0) or 0) for r in rows[:1])
    correction_hit_count = sum(int(bool((((r.get("verify_eval") or {}).get("correction") or {}).get("correction_hit", False)))) for r in rows)

    probe_total = sum(int(((r.get("cycle") or {}).get("probes") or {}).get("total", 0) or 0) for r in rows)
    probe_invalid = sum(int(((r.get("cycle") or {}).get("probes") or {}).get("invalid", 0) or 0) for r in rows)
    vote_support = sum(int(((r.get("cycle") or {}).get("votes") or {}).get("support", 0) or 0) for r in rows)
    vote_conflict = sum(int(((r.get("cycle") or {}).get("votes") or {}).get("conflict", 0) or 0) for r in rows)
    vote_uncertain = sum(int(((r.get("cycle") or {}).get("votes") or {}).get("uncertain", 0) or 0) for r in rows)
    accepted_claims = sum(int(((r.get("cycle") or {}).get("claims") or {}).get("accepted_claim_count", 0) or 0) for r in rows)
    flagged_claims = sum(int(((r.get("cycle") or {}).get("claims") or {}).get("flagged_claim_count", 0) or 0) for r in rows)
    queue_count = sum(int(((r.get("cycle") or {}).get("claims") or {}).get("queue_count", 0) or 0) for r in rows)

    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / max(1, len(values)))

    return {
        "frames_total": int(frame_count),
        "frames_ok": int(succeeded),
        "frames_build_failed": int(build_failed),
        "frames_cycle_failed": int(cycle_failed),
        "verify_metrics_mean": {
            "claim_accuracy_before": _mean(claim_acc_before),
            "claim_accuracy_after": _mean(claim_acc_after),
            "claim_accuracy_delta": _mean(claim_acc_after) - _mean(claim_acc_before),
            "vote_direction_accuracy": _mean(vote_direction_accuracy),
            "true_claim_support_rate": _mean(true_claim_support_rate),
            "false_claim_conflict_rate": _mean(false_claim_conflict_rate),
            "false_claim_reduction": _mean([float(x) for x in false_claim_reduction]),
        },
        "sampling": {
            "annotated_frames_total": int(annotated_frame_count),
            "verify_frames_sampled": int(sampled_frame_count),
        },
        "claim_counts": {
            "true_claims_before": int(sum(true_claims_before)),
            "false_claims_before": int(sum(false_claims_before)),
            "true_claims_after": int(sum(true_claims_after)),
            "false_claims_after": int(sum(false_claims_after)),
        },
        "correction_hit_rate": float(correction_hit_count) / float(max(1, frame_count)),
        "cycle": {
            "probe_total": int(probe_total),
            "probe_invalid": int(probe_invalid),
            "vote_support": int(vote_support),
            "vote_conflict": int(vote_conflict),
            "vote_uncertain": int(vote_uncertain),
            "accepted_claim_count": int(accepted_claims),
            "flagged_claim_count": int(flagged_claims),
            "queue_count": int(queue_count),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset-level VidOR GT-frame evaluation using IMPACT Multi-Agent/cycle refine pipeline."
    )
    parser.add_argument(
        "--videos_dir",
        default="/cvhci/temp/wkong/sample_videos/VidOR/videos",
        help="Directory containing source videos.",
    )
    parser.add_argument(
        "--masks_dir",
        default="/cvhci/temp/wkong/sample_videos/VidOR/masks",
        help="Directory containing per-video panoptic masks.",
    )
    parser.add_argument(
        "--gt_json",
        default="/cvhci/temp/wkong/sample_videos/pvsg.json",
        help="PVSG ground-truth json path.",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_CYCLE_PROVIDER,
        choices=["gemini_api", "chatgpt_api", "qwen25_vl", "manual", "gemini", "openai", "chatgpt", "qwen"],
        help="Verifier provider.",
    )
    parser.add_argument(
        "--config",
        default="configs/impact_cycle.json",
        help="Cycle config JSON path.",
    )
    parser.add_argument(
        "--pipeline_config",
        default="configs/impact_sg_pipeline.json",
        help="Scene graph pipeline config JSON path.",
    )
    parser.add_argument(
        "--ontology",
        default="configs/impact_sg_ontology.json",
        help="Ontology JSON path.",
    )
    parser.add_argument("--rounds", type=int, default=2, help="Cycle refine rounds.")
    parser.add_argument("--enable_single", action="store_true", help="Force-enable single-turn probes.")
    parser.add_argument("--disable_single", action="store_true", help="Force-disable single-turn probes.")
    parser.add_argument("--enable_multi", action="store_true", help="Force-enable multi-turn probes.")
    parser.add_argument("--disable_multi", action="store_true", help="Force-disable multi-turn probes.")
    parser.add_argument("--enable_caption", action="store_true", help="Force-enable caption probe.")
    parser.add_argument("--disable_caption", action="store_true", help="Force-disable caption probe.")
    parser.add_argument("--low_quota", action="store_true", help="Disable multi-turn/caption and lower token budget.")
    parser.add_argument("--skip_cycle", action="store_true", help="Build-only ablation: skip cycle verify and keep pre-cycle metrics as final metrics.")
    parser.add_argument(
        "--load_detections_from",
        default="",
        help="Optional Stage-1 detection directory. When set, Stage 2 loads saved SAM detections and will not rerun SAM3.",
    )
    parser.add_argument(
        "--verification_mode",
        default="legacy",
        choices=["legacy", "none", "single_turn", "multi_turn", "full_cycle"],
        help="Optional Stage-2 verification mode override. 'legacy' keeps the existing flag-based behavior.",
    )
    parser.add_argument("--output_dir", default="outputs/vidor_eval", help="Output directory.")
    parser.add_argument("--run_name", default="", help="Optional explicit run name; otherwise a unique run tag is generated.")
    parser.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold for node matching.")
    parser.add_argument("--det_iou_thr", type=float, default=0.5, help="IoU threshold for detection / instance matching.")
    parser.add_argument("--max_videos", type=int, default=5, help="Optional cap for number of videos (default 5, 0 = no cap).")
    parser.add_argument("--max_frames_per_video", type=int, default=5, help="Optional cap per video after sampling (default 5, 0 = no cap).")
    parser.add_argument("--video_ids", nargs="*", default=None, help="Optional explicit video ids to evaluate.")
    parser.add_argument(
        "--reverse_entries",
        action="store_true",
        help="Reverse the GT entry traversal order before any ordinal slicing or max_videos truncation.",
    )
    parser.add_argument(
        "--video_ord_start",
        type=int,
        default=0,
        help="Optional 1-based inclusive start ordinal in the GT entry list (0 = disabled).",
    )
    parser.add_argument(
        "--video_ord_end",
        type=int,
        default=0,
        help="Optional 1-based inclusive end ordinal in the GT entry list (0 = disabled, requires --video_ord_start).",
    )
    parser.add_argument("--write_csv", action="store_true", help="Also write frame_level.csv and video_level.csv.")
    parser.add_argument("--save_visualizations", action="store_true", help="Save per-frame bbox visualizations for successful frames.")
    parser.add_argument("--save_plots", action="store_true", help="Save summary plots under output_dir/plots.")
    parser.add_argument("--debug_claim_trace", action="store_true", help="Print per-frame claim lifecycle trace for debugging.")
    parser.add_argument(
        "--ablation_suite",
        default="none",
        choices=["none", "paper5"],
        help="Optional bundled ablation sweep. 'paper5' runs Backbone Only, Single-turn, Single-turn + Caption, Single-turn + Multi-turn, and Full Cycle in one pass.",
    )
    return parser.parse_args()


def _apply_verification_mode_overrides(
    args: argparse.Namespace,
    *,
    enable_single: Optional[bool],
    enable_multi: Optional[bool],
    enable_caption: Optional[bool],
) -> Tuple[bool, Optional[bool], Optional[bool], Optional[bool]]:
    mode = str(args.verification_mode or "legacy").strip().lower()
    if mode == "legacy":
        return bool(args.skip_cycle), enable_single, enable_multi, enable_caption
    if mode == "none":
        return True, False, False, False
    if mode == "single_turn":
        return False, True, False, False
    if mode == "multi_turn":
        return False, True, True, False
    if mode == "full_cycle":
        return False, True, True, True
    return bool(args.skip_cycle), enable_single, enable_multi, enable_caption


def _collect_frame_csv_rows(per_video_results: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for video_id, items in per_video_results.items():
        for item in list(items or []):
            verify_eval = dict(item.get("verify_eval") or {})
            before = dict(verify_eval.get("before") or {})
            after = dict(verify_eval.get("after") or {})
            vote_eval = dict(verify_eval.get("vote_eval") or {})
            correction = dict(verify_eval.get("correction") or {})
            cycle = dict(item.get("cycle") or {})
            probes = dict(cycle.get("probes") or {})
            votes = dict(cycle.get("votes") or {})
            claims = dict(cycle.get("claims") or {})
            sampling = dict(item.get("sampling") or {})
            rows.append(
                {
                    "video_id": video_id,
                    "frame_idx": int(item.get("frame_idx", -1) or -1),
                    "status": str(item.get("status", "") or ""),
                    "elapsed_sec": _safe_float(item.get("elapsed_sec", 0.0), 0.0),
                    "annotated_frames_total": int(sampling.get("annotated_frames_total", 0) or 0),
                    "verify_frames_sampled": int(sampling.get("verify_frames_sampled", 0) or 0),
                    "supported_claims_before": int(before.get("supported_claims", 0) or 0),
                    "supported_claims_after": int(after.get("supported_claims", 0) or 0),
                    "true_claims_before": int(before.get("true_claims", 0) or 0),
                    "false_claims_before": int(before.get("false_claims", 0) or 0),
                    "true_claims_after": int(after.get("true_claims", 0) or 0),
                    "false_claims_after": int(after.get("false_claims", 0) or 0),
                    "claim_accuracy_before": _safe_float(before.get("claim_accuracy", 0.0), 0.0),
                    "claim_accuracy_after": _safe_float(after.get("claim_accuracy", 0.0), 0.0),
                    "claim_accuracy_delta": _safe_float(after.get("claim_accuracy", 0.0), 0.0) - _safe_float(before.get("claim_accuracy", 0.0), 0.0),
                    "vote_direction_accuracy": _safe_float(vote_eval.get("vote_direction_accuracy", 0.0), 0.0),
                    "true_claim_support_rate": _safe_float(vote_eval.get("true_claim_support_rate", 0.0), 0.0),
                    "false_claim_conflict_rate": _safe_float(vote_eval.get("false_claim_conflict_rate", 0.0), 0.0),
                    "true_claim_conflict_error_rate": _safe_float(vote_eval.get("true_claim_conflict_error_rate", 0.0), 0.0),
                    "false_claim_support_error_rate": _safe_float(vote_eval.get("false_claim_support_error_rate", 0.0), 0.0),
                    "false_claim_reduction": int(correction.get("false_claim_reduction", 0) or 0),
                    "false_claim_reduction_rate": _safe_float(correction.get("false_claim_reduction_rate", 0.0), 0.0),
                    "correction_hit": int(bool(correction.get("correction_hit", False))),
                    "probe_total": int(probes.get("total", 0) or 0),
                    "probe_invalid": int(probes.get("invalid", 0) or 0),
                    "vote_support": int(votes.get("support", 0) or 0),
                    "vote_conflict": int(votes.get("conflict", 0) or 0),
                    "vote_uncertain": int(votes.get("uncertain", 0) or 0),
                    "accepted_claim_count": int(claims.get("accepted_claim_count", 0) or 0),
                    "flagged_claim_count": int(claims.get("flagged_claim_count", 0) or 0),
                    "queue_count": int(claims.get("queue_count", 0) or 0),
                }
            )
    rows.sort(key=lambda r: (str(r.get("video_id", "")), int(r.get("frame_idx", -1))))
    return rows


def _write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _match_nodes_bbox_only(
    pred_nodes: List[Dict[str, object]],
    gt_nodes: List[Dict[str, object]],
    *,
    iou_thr: float,
) -> NodeMatchResult:
    candidates: List[Tuple[float, int, int]] = []
    for pi, pred in enumerate(pred_nodes):
        pbbox = list(pred.get("bbox") or [0, 0, 0, 0])
        for gi, gt in enumerate(gt_nodes):
            iou = _bbox_iou(pbbox, list(gt.get("bbox") or [0, 0, 0, 0]))
            if iou >= float(iou_thr):
                candidates.append((float(iou), int(pi), int(gi)))
    candidates.sort(key=lambda x: x[0], reverse=True)
    used_pred: Set[int] = set()
    used_gt: Set[int] = set()
    mapping: Dict[int, int] = {}
    ious: List[float] = []
    for iou, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        mapping[pi] = gi
        ious.append(float(iou))
    return NodeMatchResult(pred_to_gt=mapping, matched_ious=ious)


def _cycle_provenance_entries(node: Dict[str, object]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in list(node.get("provenance") or []):
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", "") or "").strip().lower()
        mode = str(row.get("mode", "") or "").strip().lower()
        if source == "cycle_refine" or "verifier" in source or "cycle" in mode or "verifier" in mode:
            out.append(dict(row))
    return out


def _extract_accepted_claim_sets(cycle_update: Dict[str, object]) -> Tuple[List[str], List[str], List[str]]:
    accepted = [str(x).strip() for x in list(cycle_update.get("accepted_claim_ids") or []) if str(x).strip()]
    explicit_confirm = [str(x).strip() for x in list(cycle_update.get("accepted_confirm_claim_ids") or []) if str(x).strip()]
    explicit_correct = [str(x).strip() for x in list(cycle_update.get("accepted_correct_claim_ids") or []) if str(x).strip()]
    if explicit_confirm or explicit_correct:
        return accepted, explicit_confirm, explicit_correct
    correction_ids = {
        str((row or {}).get("claim_id", "")).strip()
        for row in list(cycle_update.get("correction_applied") or [])
        if isinstance(row, dict) and str((row or {}).get("claim_id", "")).strip()
    }
    confirm = [cid for cid in accepted if cid not in correction_ids]
    correct = [cid for cid in accepted if cid in correction_ids]
    return accepted, confirm, correct


def _method_name_from_cfg(args: argparse.Namespace, cycle_cfg: Dict[str, object]) -> str:
    if bool(args.skip_cycle):
        return "Backbone Only"
    cycle = dict(cycle_cfg.get("cycle") or {})
    single = bool(cycle.get("enable_single_turn_probes", True))
    multi = bool(cycle.get("enable_multi_turn_probes", True))
    caption = bool(cycle.get("enable_caption_probe", True))
    if single and (not multi) and (not caption):
        return "Single-turn"
    if single and (not multi) and caption:
        return "Single-turn + Caption"
    if single and multi and (not caption):
        return "Single-turn + Multi-turn"
    if single and multi and caption:
        return "Full Cycle"
    if single:
        return "Verification Variant"
    return "Backbone Only"


def _build_variant_specs(
    args: argparse.Namespace,
    *,
    base_cfg: Dict[str, object],
    provider: str,
) -> List[Dict[str, object]]:
    if str(args.ablation_suite or "none").strip().lower() != "paper5":
        return [
            {
                "slug": "main",
                "method_name": _method_name_from_cfg(args, base_cfg),
                "cycle_cfg": dict(base_cfg),
                "skip_cycle": bool(args.skip_cycle),
            }
        ]
    variants: List[Tuple[str, str, bool, Optional[bool], Optional[bool], Optional[bool]]] = [
        ("backbone_only", "Backbone Only", True, False, False, False),
        ("single_turn", "Single-turn", False, True, False, False),
        ("single_turn_caption", "Single-turn + Caption", False, True, False, True),
        ("single_turn_multi", "Single-turn + Multi-turn", False, True, True, False),
        ("full_cycle", "Full Cycle", False, True, True, True),
    ]
    out: List[Dict[str, object]] = []
    for slug, method_name, skip_cycle, single, multi, caption in variants:
        cfg = _build_cycle_cfg(
            base_cfg,
            provider=provider,
            rounds=int(args.rounds),
            enable_single=single,
            enable_multi=multi,
            enable_caption=caption,
            low_quota=False,
        )
        out.append(
            {
                "slug": slug,
                "method_name": method_name,
                "cycle_cfg": cfg,
                "skip_cycle": bool(skip_cycle),
            }
        )
    return out


def _format_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{int(digits)}f}"


def _verification_utility_outputs(
    *,
    per_video_results: Dict[str, List[Dict[str, object]]],
    output_dir: str,
    run_tag: str,
    method_name: str,
    iou_thr: float,
) -> Dict[str, str]:
    rng = random.Random(7)
    ok_rows: List[Dict[str, object]] = []
    for rows in per_video_results.values():
        ok_rows.extend(
            [
                dict(row)
                for row in list(rows or [])
                if str((row or {}).get("status", "") or "").strip().lower() == "ok"
            ]
        )

    all_nodes: List[Dict[str, object]] = []
    verified_nodes: List[Dict[str, object]] = []
    unverified_nodes: List[Dict[str, object]] = []
    modified_nodes: List[Dict[str, object]] = []
    provenance_nodes: List[Dict[str, object]] = []
    accepted_confirm_rows: List[Dict[str, object]] = []
    accepted_correct_rows: List[Dict[str, object]] = []
    false_accepted_rows: List[Dict[str, object]] = []
    accepted_total = 0
    probes_total = 0
    probes_invalid = 0

    for row in ok_rows:
        frame_idx_raw = row.get("frame_idx", -1)
        frame_name = f"{_safe_int(frame_idx_raw, -1):06d}.json"
        initial_graph = dict(row.get("initial_graph") or {})
        final_graph = dict(row.get("final_graph") or {})
        gt_graph = dict(row.get("gt_graph") or {})
        cycle = dict(row.get("cycle") or {})
        probes = dict(cycle.get("probes") or {})
        probes_total += int(probes.get("total", 0) or 0)
        probes_invalid += int(probes.get("invalid", 0) or 0)

        initial_nodes = [dict(x) for x in list(initial_graph.get("nodes") or []) if isinstance(x, dict)]
        final_nodes = [dict(x) for x in list(final_graph.get("nodes") or []) if isinstance(x, dict)]
        gt_nodes = [dict(x) for x in list(gt_graph.get("nodes") or []) if isinstance(x, dict)]
        before_by_id = {str(n.get("entity_id", "") or "").strip(): dict(n) for n in initial_nodes}

        claims = list(graph_to_claims(initial_graph) or [])
        claim_map = {str(claim.claim_id or "").strip(): claim for claim in claims}
        before_truth = _claim_truth_map(initial_graph, gt_graph, iou_thr=float(iou_thr), claims=claims)
        after_truth = _claim_truth_map(final_graph, gt_graph, iou_thr=float(iou_thr), claims=claims)

        after_match = _match_nodes_bbox_only(final_nodes, gt_nodes, iou_thr=float(iou_thr))
        pred_to_gt = dict(after_match.pred_to_gt)
        matched_iou_by_pred: Dict[int, float] = {}
        for pred_idx, gt_idx in pred_to_gt.items():
            matched_iou_by_pred[int(pred_idx)] = _bbox_iou(
                list(final_nodes[pred_idx].get("bbox") or [0, 0, 0, 0]),
                list(gt_nodes[gt_idx].get("bbox") or [0, 0, 0, 0]),
            )

        cycle_update = dict((final_graph.get("metadata") or {}).get("cycle_update") or {})
        accepted_ids, accepted_confirm_ids, accepted_correct_ids = _extract_accepted_claim_sets(cycle_update)
        accepted_total += len(accepted_ids)
        correction_rows = {
            str((entry or {}).get("claim_id", "")).strip(): dict(entry)
            for entry in list(cycle_update.get("correction_applied") or [])
            if isinstance(entry, dict) and str((entry or {}).get("claim_id", "")).strip()
        }

        for pred_idx, node in enumerate(final_nodes):
            entity_id = str(node.get("entity_id", "") or "").strip()
            before_node = dict(before_by_id.get(entity_id) or {})
            gt_idx = pred_to_gt.get(int(pred_idx))
            gt_node = dict(gt_nodes[gt_idx]) if gt_idx is not None and 0 <= gt_idx < len(gt_nodes) else {}
            predicted_label = _normalize_label(node.get("canonical_label", node.get("label", "")))
            gt_label = _normalize_label(gt_node.get("canonical_label", gt_node.get("label", ""))) if gt_node else ""
            correct = bool(gt_node) and (predicted_label == gt_label)
            verified = bool(node.get("verified", False))
            cycle_before = _cycle_provenance_entries(before_node)
            cycle_after = _cycle_provenance_entries(node)
            modified = bool(
                ((not bool(before_node.get("verified", False))) and verified)
                or before_node.get("confidence") != node.get("confidence")
                or before_node.get("verify_confidence") != node.get("verify_confidence")
                or abs(_safe_float(node.get("score", 0.0), 0.0) - _safe_float(before_node.get("score", 0.0), 0.0)) > 1e-6
                or len(cycle_after) > len(cycle_before)
            )
            node_record = {
                "frame": frame_name,
                "node_id": entity_id,
                "predicted_label": predicted_label,
                "gt_label": gt_label,
                "correct": bool(correct),
                "verified": bool(verified),
                "iou": float(matched_iou_by_pred.get(int(pred_idx), 0.0)),
                "score": node.get("score"),
                "confidence": node.get("confidence"),
                "verify_confidence": node.get("verify_confidence"),
                "provenance": list(node.get("provenance") or []),
            }
            all_nodes.append(node_record)
            if verified:
                verified_nodes.append(node_record)
            else:
                unverified_nodes.append(node_record)
            if modified:
                modified_nodes.append(
                    {
                        "frame": frame_name,
                        "node_id": entity_id,
                        "before": {
                            "verified": before_node.get("verified"),
                            "score": before_node.get("score"),
                            "confidence": before_node.get("confidence"),
                            "verify_confidence": before_node.get("verify_confidence"),
                            "provenance_len": len(list(before_node.get("provenance") or [])),
                        },
                        "after": {
                            "verified": node.get("verified"),
                            "score": node.get("score"),
                            "confidence": node.get("confidence"),
                            "verify_confidence": node.get("verify_confidence"),
                            "provenance_len": len(list(node.get("provenance") or [])),
                        },
                    }
                )
            if list(node.get("provenance") or []):
                provenance_nodes.append(
                    {
                        "frame": frame_name,
                        "node_id": entity_id,
                        "label": predicted_label,
                        "provenance": list(node.get("provenance") or []),
                    }
                )

        def _accepted_truth(cid: str) -> Tuple[Optional[bool], Optional[bool]]:
            before_info = dict(before_truth.get(cid) or {})
            after_info = dict(after_truth.get(cid) or {})
            claim = claim_map.get(cid)
            before_correct = None
            after_correct = None
            if before_info.get("supported") is True:
                before_correct = bool(before_info.get("is_true"))
            if after_info.get("supported") is True:
                after_correct = bool(after_info.get("is_true"))
            if claim is not None and claim.claim_type == "label" and claim.subject_id:
                final_node = dict(next((n for n in final_nodes if str(n.get("entity_id", "") or "").strip() == str(claim.subject_id or "").strip()), {}) or {})
                if final_node:
                    gt_match_iou = 0.0
                    gt_match: Dict[str, object] = {}
                    for gt_node in gt_nodes:
                        iou = _bbox_iou(list(final_node.get("bbox") or [0, 0, 0, 0]), list(gt_node.get("bbox") or [0, 0, 0, 0]))
                        if iou > gt_match_iou:
                            gt_match_iou = iou
                            gt_match = dict(gt_node)
                    if gt_match and gt_match_iou >= float(iou_thr):
                        after_correct = (
                            _normalize_label(final_node.get("canonical_label", final_node.get("label", "")))
                            == _normalize_label(gt_match.get("canonical_label", gt_match.get("label", "")))
                        )
            return before_correct, after_correct

        for cid in accepted_ids:
            claim = claim_map.get(cid)
            before_correct, after_correct = _accepted_truth(cid)
            entry = {
                "frame": frame_name,
                "claim_id": cid,
                "subject": str(getattr(claim, "subject_id", "") or ""),
                "relation": str(getattr(claim, "predicate", "") or ""),
                "object": str(getattr(claim, "value", "") or ""),
                "before_correct": before_correct,
                "after_correct": after_correct,
            }
            if cid in accepted_confirm_ids:
                accepted_confirm_rows.append(dict(entry))
                if before_correct is False:
                    false_accepted_rows.append(dict(entry))
            if cid in accepted_correct_ids:
                patched = dict(entry)
                patched["selected_value"] = str((correction_rows.get(cid) or {}).get("selected_value", "") or "")
                accepted_correct_rows.append(patched)
                if after_correct is False:
                    false_accepted_rows.append(dict(patched))

    frames = len(ok_rows)
    nodes_total = len(all_nodes)
    verified_total = len(verified_nodes)
    verified_correct = len([row for row in verified_nodes if bool(row.get("correct"))])
    overall_correct = len([row for row in all_nodes if bool(row.get("correct"))])
    modified_total = len(modified_nodes)
    provenance_total = len(provenance_nodes)
    accepted_confirm_total = len(accepted_confirm_rows)
    accepted_correct_total = len(accepted_correct_rows)
    false_accepted_total = len(false_accepted_rows)
    unverified_total = len(unverified_nodes)
    unverified_correct = len([row for row in unverified_nodes if bool(row.get("correct"))])

    verified_precision = float(verified_correct) / float(max(1, verified_total))
    overall_accuracy = float(overall_correct) / float(max(1, nodes_total))
    structured_update_rate = float(modified_total) / float(max(1, nodes_total))
    accepted_confirm_per_frame = float(accepted_confirm_total) / float(max(1, frames))
    accepted_correct_per_frame = float(accepted_correct_total) / float(max(1, frames))
    provenance_coverage = float(provenance_total) / float(max(1, nodes_total))
    review_reduction = float(verified_total) / float(max(1, nodes_total))
    false_accepted_rate = float(false_accepted_total) / float(max(1, accepted_total))
    verified_acc = verified_precision
    unverified_acc = float(unverified_correct) / float(max(1, unverified_total))
    probes_per_frame = float(probes_total) / float(max(1, frames))
    invalid_probe_rate = float(probes_invalid) / float(max(1, probes_total))

    missing_variants = [
        name
        for name in [
            "Backbone Only",
            "Single-turn",
            "Single-turn + Caption",
            "Single-turn + Multi-turn",
            "Full Cycle",
            "Full Method",
        ]
        if name != method_name
    ]

    summary_row = {
        "run_tag": str(run_tag or ""),
        "method_name": str(method_name or ""),
        "frames": int(frames),
        "nodes_total": int(nodes_total),
        "verified_precision": float(verified_precision),
        "overall_accuracy": float(overall_accuracy),
        "structured_update_rate": float(structured_update_rate),
        "accepted_confirm_per_frame": float(accepted_confirm_per_frame),
        "accepted_correct_per_frame": float(accepted_correct_per_frame),
        "provenance_coverage": float(provenance_coverage),
        "review_reduction": float(review_reduction),
        "false_accepted_rate": float(false_accepted_rate),
        "verified_acc": float(verified_acc),
        "unverified_acc": float(unverified_acc),
        "probes_per_frame": float(probes_per_frame),
        "invalid_probe_rate": float(invalid_probe_rate),
    }

    def _sample(items: List[Dict[str, object]], count: int) -> List[Dict[str, object]]:
        if len(items) <= count:
            return [dict(x) for x in items]
        picks = rng.sample(list(range(len(items))), count)
        return [dict(items[idx]) for idx in sorted(picks)]

    evidence = {
        "verified_precision_examples": _sample(verified_nodes, 10),
        "structured_update_examples": _sample(modified_nodes, 10),
        "accepted_confirm_examples": _sample(accepted_confirm_rows, 10),
        "accepted_correct_examples": _sample(accepted_correct_rows, 10),
        "provenance_examples": _sample(provenance_nodes, 5),
        "false_accepted_examples": _sample(false_accepted_rows, 5),
    }

    json_payload = {
        "run_tag": str(run_tag or ""),
        "method_name": str(method_name or ""),
        "metrics": {
            **summary_row,
            "verified_precision_counts": {"numerator": int(verified_correct), "denominator": int(verified_total)},
            "overall_accuracy_counts": {"numerator": int(overall_correct), "denominator": int(nodes_total)},
            "structured_update_counts": {"numerator": int(modified_total), "denominator": int(nodes_total)},
            "accepted_confirm_counts": {"numerator": int(accepted_confirm_total), "denominator": int(frames)},
            "accepted_correct_counts": {"numerator": int(accepted_correct_total), "denominator": int(frames)},
            "provenance_coverage_counts": {"numerator": int(provenance_total), "denominator": int(nodes_total)},
            "review_reduction_counts": {"verified": int(verified_total), "total": int(nodes_total)},
            "false_accepted_counts": {"numerator": int(false_accepted_total), "denominator": int(accepted_total)},
            "calibration_counts": {
                "verified_correct": int(verified_correct),
                "verified_total": int(verified_total),
                "unverified_correct": int(unverified_correct),
                "unverified_total": int(unverified_total),
            },
            "probe_counts": {"total": int(probes_total), "invalid": int(probes_invalid)},
        },
        "notes": {
            "accepted_claim_reconstruction": bool(not ok_rows or not (((ok_rows[0].get("final_graph") or {}).get("metadata") or {}).get("cycle_update", {}).get("accepted_confirm_claim_ids"))),
            "missing_method_variants": missing_variants,
        },
        "evidence": evidence,
    }

    latex_lines = [
        "% Auto-generated by run_vidor_gt_frame_eval.py",
        "\\begin{tabular}{lcccccccc}",
        "\\hline",
        "Method & Verified Prec. $\\uparrow$ & Overall Acc. $\\uparrow$ & Struct. Update $\\uparrow$ & Accept (Confirm) / F $\\uparrow$ & Accept (Correct) / F $\\uparrow$ & Prov. Coverage $\\uparrow$ & Review Reduction $\\uparrow$ & False Accept $\\downarrow$ \\\\",
        "\\hline",
        (
            f"{method_name} & "
            f"{_format_float(verified_precision)} & "
            f"{_format_float(overall_accuracy)} & "
            f"{_format_float(structured_update_rate)} & "
            f"{_format_float(accepted_confirm_per_frame)} & "
            f"{_format_float(accepted_correct_per_frame)} & "
            f"{_format_float(provenance_coverage)} & "
            f"{_format_float(review_reduction)} & "
            f"{_format_float(false_accepted_rate)} \\\\"
        ),
        "\\hline",
        "\\end{tabular}",
    ]
    latex_table = "\n".join(latex_lines) + "\n"

    evidence_lines: List[str] = []
    evidence_lines.append("# Verification Utility Evidence")
    evidence_lines.append("")
    evidence_lines.append(f"- Run: `{run_tag}`")
    evidence_lines.append(f"- Method: `{method_name}`")
    evidence_lines.append(f"- Frames: `{frames}`")
    evidence_lines.append(f"- Nodes: `{nodes_total}`")
    evidence_lines.append("")
    evidence_lines.append("## Metrics")
    evidence_lines.append("")
    evidence_lines.append(f"- Verified Precision: `{_format_float(verified_precision)}` ({verified_correct}/{verified_total})")
    evidence_lines.append(f"- Overall Accuracy: `{_format_float(overall_accuracy)}` ({overall_correct}/{nodes_total})")
    evidence_lines.append(f"- Structured Update Rate: `{_format_float(structured_update_rate)}` ({modified_total}/{nodes_total})")
    evidence_lines.append(f"- Accepted Confirm / Frame: `{_format_float(accepted_confirm_per_frame)}` ({accepted_confirm_total}/{frames})")
    evidence_lines.append(f"- Accepted Correct / Frame: `{_format_float(accepted_correct_per_frame)}` ({accepted_correct_total}/{frames})")
    evidence_lines.append(f"- Provenance Coverage: `{_format_float(provenance_coverage)}` ({provenance_total}/{nodes_total})")
    evidence_lines.append(f"- Review Reduction: `{_format_float(review_reduction)}` ({verified_total}/{nodes_total})")
    evidence_lines.append(f"- False Accepted Rate: `{_format_float(false_accepted_rate)}` ({false_accepted_total}/{max(1, accepted_total)})")
    evidence_lines.append(f"- accuracy(verified=True): `{_format_float(verified_acc)}` ({verified_correct}/{verified_total})")
    evidence_lines.append(f"- accuracy(verified=False): `{_format_float(unverified_acc)}` ({unverified_correct}/{unverified_total})")
    evidence_lines.append(f"- probes/frame: `{_format_float(probes_per_frame)}` ({probes_total}/{frames})")
    evidence_lines.append(f"- invalid probe rate: `{_format_float(invalid_probe_rate)}` ({probes_invalid}/{max(1, probes_total)})")
    evidence_lines.append("")
    evidence_lines.append("## Verified Precision Evidence")
    evidence_lines.append("")
    if evidence["verified_precision_examples"]:
        for row in evidence["verified_precision_examples"]:
            evidence_lines.append(
                f"- `{row['frame']}` `{row['node_id']}` pred=`{row['predicted_label']}` gt=`{row['gt_label']}` "
                f"verified=`{row['verified']}` correct=`{row['correct']}` score=`{row['score']}` "
                f"confidence=`{row['confidence']}` verify_confidence=`{row['verify_confidence']}`"
            )
    else:
        evidence_lines.append("- No verified nodes were produced.")
    evidence_lines.append("")
    evidence_lines.append("## Structured Update Evidence")
    evidence_lines.append("")
    if evidence["structured_update_examples"]:
        for row in evidence["structured_update_examples"]:
            evidence_lines.append(
                f"- `{row['frame']}` `{row['node_id']}` before={json.dumps(row['before'], ensure_ascii=True)} "
                f"after={json.dumps(row['after'], ensure_ascii=True)}"
            )
    else:
        evidence_lines.append("- No verifier-induced node updates were observed.")
    evidence_lines.append("")
    evidence_lines.append("## Accepted Confirm Evidence")
    evidence_lines.append("")
    if evidence["accepted_confirm_examples"]:
        for row in evidence["accepted_confirm_examples"]:
            evidence_lines.append(
                f"- `{row['frame']}` `{row['claim_id']}` subject=`{row['subject']}` relation=`{row['relation']}` "
                f"object=`{row['object']}` before_correct=`{row['before_correct']}`"
            )
    else:
        evidence_lines.append("- No accepted confirm claims were produced.")
    evidence_lines.append("")
    evidence_lines.append("## Accepted Correct Evidence")
    evidence_lines.append("")
    if evidence["accepted_correct_examples"]:
        for row in evidence["accepted_correct_examples"]:
            evidence_lines.append(
                f"- `{row['frame']}` `{row['claim_id']}` subject=`{row['subject']}` relation=`{row['relation']}` "
                f"object=`{row['object']}` selected_value=`{row.get('selected_value', '')}` "
                f"before_correct=`{row['before_correct']}` after_correct=`{row['after_correct']}`"
            )
    else:
        evidence_lines.append("- No semantic corrections were accepted in this run.")
        evidence_lines.append("- No `correction_applied` records were found, so accepted claims are confirmations rather than graph-changing semantic corrections.")
    evidence_lines.append("")
    evidence_lines.append("## Provenance Evidence")
    evidence_lines.append("")
    if evidence["provenance_examples"]:
        for row in evidence["provenance_examples"]:
            evidence_lines.append(
                f"- `{row['frame']}` `{row['node_id']}` label=`{row['label']}` provenance={json.dumps(row['provenance'], ensure_ascii=True)}"
            )
    else:
        evidence_lines.append("- No provenance entries were found.")
    evidence_lines.append("")
    evidence_lines.append("## Bad Cases")
    evidence_lines.append("")
    if evidence["false_accepted_examples"]:
        for row in evidence["false_accepted_examples"]:
            evidence_lines.append(
                f"- `{row['frame']}` `{row['claim_id']}` subject=`{row['subject']}` relation=`{row['relation']}` "
                f"object=`{row['object']}` before_correct=`{row['before_correct']}` after_correct=`{row['after_correct']}`"
            )
    else:
        evidence_lines.append("- Zero false accepted cases were observed.")
    evidence_lines.append("")
    evidence_lines.append("## Interpretation")
    evidence_lines.append("")
    if verified_acc > unverified_acc:
        evidence_lines.append("- Verification improves reliability calibration.")
    else:
        evidence_lines.append("- Verification is operational as a structured updater, but current acceptance policy does not yet improve reliability calibration.")
    if accepted_correct_total <= 0:
        evidence_lines.append("- `accepted_correct_per_frame = 0` because no `correction_applied` records exist in this run.")
    if structured_update_rate > 0.0:
        evidence_lines.append("- The agent is actively modifying graph records and attaching evidence, as shown by non-zero structured updates and provenance additions.")
    if review_reduction > 0.0:
        evidence_lines.append("- Human review space is reduced by marking a subset of nodes as verified and leaving the remainder as unresolved.")
    if missing_variants:
        evidence_lines.append(f"- Missing method variants in this run: {', '.join(missing_variants)}.")
    evidence_text = "\n".join(evidence_lines) + "\n"

    csv_fields = [
        "run_tag",
        "method_name",
        "frames",
        "nodes_total",
        "verified_precision",
        "overall_accuracy",
        "structured_update_rate",
        "accepted_confirm_per_frame",
        "accepted_correct_per_frame",
        "provenance_coverage",
        "review_reduction",
        "false_accepted_rate",
        "verified_acc",
        "unverified_acc",
        "probes_per_frame",
        "invalid_probe_rate",
    ]
    csv_path = os.path.join(output_dir, "verification_utility_summary.csv")
    json_path = os.path.join(output_dir, "verification_utility_summary.json")
    md_path = os.path.join(output_dir, "verification_utility_evidence.md")
    tex_path = os.path.join(output_dir, "verification_utility_table.tex")

    _write_csv(csv_path, [summary_row], csv_fields)
    _save_json(json_path, json_payload)
    _save_text(md_path, evidence_text)
    _save_text(tex_path, latex_table)

    return {
        "csv": csv_path,
        "json": json_path,
        "evidence_md": md_path,
        "table_tex": tex_path,
        "latex_table": latex_table,
    }


def _update_verification_utility_aggregate(output_root: str) -> Dict[str, str]:
    root = os.path.abspath(str(output_root or ""))
    if not root or not os.path.isdir(root):
        return {}

    rows: List[Dict[str, object]] = []
    json_payloads: List[Dict[str, object]] = []
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        if not os.path.isdir(run_dir):
            continue
        summary_path = os.path.join(run_dir, "verification_utility_summary.json")
        if not os.path.isfile(summary_path):
            continue
        try:
            payload = dict(_load_json(summary_path) or {})
        except Exception:
            continue
        metrics = dict(payload.get("metrics") or {})
        row = {
            "run_tag": str(payload.get("run_tag", name) or name),
            "method_name": str(payload.get("method_name", "") or ""),
            "frames": int(metrics.get("frames", 0) or 0),
            "nodes_total": int(metrics.get("nodes_total", 0) or 0),
            "verified_precision": _safe_float(metrics.get("verified_precision", 0.0), 0.0),
            "overall_accuracy": _safe_float(metrics.get("overall_accuracy", 0.0), 0.0),
            "structured_update_rate": _safe_float(metrics.get("structured_update_rate", 0.0), 0.0),
            "accepted_confirm_per_frame": _safe_float(metrics.get("accepted_confirm_per_frame", 0.0), 0.0),
            "accepted_correct_per_frame": _safe_float(metrics.get("accepted_correct_per_frame", 0.0), 0.0),
            "provenance_coverage": _safe_float(metrics.get("provenance_coverage", 0.0), 0.0),
            "review_reduction": _safe_float(metrics.get("review_reduction", 0.0), 0.0),
            "false_accepted_rate": _safe_float(metrics.get("false_accepted_rate", 0.0), 0.0),
            "verified_acc": _safe_float(metrics.get("verified_acc", 0.0), 0.0),
            "unverified_acc": _safe_float(metrics.get("unverified_acc", 0.0), 0.0),
            "probes_per_frame": _safe_float(metrics.get("probes_per_frame", 0.0), 0.0),
            "invalid_probe_rate": _safe_float(metrics.get("invalid_probe_rate", 0.0), 0.0),
        }
        rows.append(row)
        json_payloads.append(payload)

    if not rows:
        return {}

    rows.sort(key=lambda row: (str(row.get("method_name", "")), str(row.get("run_tag", ""))))
    fieldnames = [
        "run_tag",
        "method_name",
        "frames",
        "nodes_total",
        "verified_precision",
        "overall_accuracy",
        "structured_update_rate",
        "accepted_confirm_per_frame",
        "accepted_correct_per_frame",
        "provenance_coverage",
        "review_reduction",
        "false_accepted_rate",
        "verified_acc",
        "unverified_acc",
        "probes_per_frame",
        "invalid_probe_rate",
    ]

    csv_path = os.path.join(root, "verification_utility_all_runs.csv")
    json_path = os.path.join(root, "verification_utility_all_runs.json")
    tex_path = os.path.join(root, "verification_utility_all_runs.tex")
    md_path = os.path.join(root, "verification_utility_all_runs.md")

    _write_csv(csv_path, rows, fieldnames)
    _save_json(
        json_path,
        {
            "runs": json_payloads,
            "rows": rows,
        },
    )

    latex_lines = [
        "% Auto-generated aggregate verification utility table",
        "\\begin{tabular}{lcccccccc}",
        "\\hline",
        "Method & Verified Prec. $\\uparrow$ & Overall Acc. $\\uparrow$ & Struct. Update $\\uparrow$ & Accept (Confirm) / F $\\uparrow$ & Accept (Correct) / F $\\uparrow$ & Prov. Coverage $\\uparrow$ & Review Reduction $\\uparrow$ & False Accept $\\downarrow$ \\\\",
        "\\hline",
    ]
    for row in rows:
        latex_lines.append(
            f"{str(row.get('method_name', '') or row.get('run_tag', ''))} & "
            f"{_format_float(_safe_float(row.get('verified_precision', 0.0), 0.0))} & "
            f"{_format_float(_safe_float(row.get('overall_accuracy', 0.0), 0.0))} & "
            f"{_format_float(_safe_float(row.get('structured_update_rate', 0.0), 0.0))} & "
            f"{_format_float(_safe_float(row.get('accepted_confirm_per_frame', 0.0), 0.0))} & "
            f"{_format_float(_safe_float(row.get('accepted_correct_per_frame', 0.0), 0.0))} & "
            f"{_format_float(_safe_float(row.get('provenance_coverage', 0.0), 0.0))} & "
            f"{_format_float(_safe_float(row.get('review_reduction', 0.0), 0.0))} & "
            f"{_format_float(_safe_float(row.get('false_accepted_rate', 0.0), 0.0))} \\\\"
        )
    latex_lines.extend(["\\hline", "\\end{tabular}"])
    latex_table = "\n".join(latex_lines) + "\n"
    _save_text(tex_path, latex_table)

    md_lines = [
        "# Aggregate Verification Utility",
        "",
        "| Run | Method | Verified Prec. | Overall Acc. | Struct. Update | Accept Confirm / F | Accept Correct / F | Prov. Coverage | Review Reduction | False Accept |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['run_tag']} | {row['method_name']} | "
            f"{_format_float(row['verified_precision'])} | "
            f"{_format_float(row['overall_accuracy'])} | "
            f"{_format_float(row['structured_update_rate'])} | "
            f"{_format_float(row['accepted_confirm_per_frame'])} | "
            f"{_format_float(row['accepted_correct_per_frame'])} | "
            f"{_format_float(row['provenance_coverage'])} | "
            f"{_format_float(row['review_reduction'])} | "
            f"{_format_float(row['false_accepted_rate'])} |"
        )
    _save_text(md_path, "\n".join(md_lines) + "\n")

    return {
        "csv": csv_path,
        "json": json_path,
        "tex": tex_path,
        "md": md_path,
        "latex_table": latex_table,
    }


def _finalize_variant_outputs(
    *,
    output_dir: str,
    output_root: str,
    run_tag: str,
    provider: str,
    verifier_meta: Dict[str, object],
    cycle_cfg_path: str,
    pipeline_cfg_path: str,
    ontology_path: str,
    videos_dir: str,
    masks_dir: str,
    gt_json_path: str,
    requested_videos: Set[str],
    args: argparse.Namespace,
    cycle_cfg: Dict[str, object],
    per_video_results: Dict[str, List[Dict[str, object]]],
    t_global: float,
    method_name: str,
    variant_skip_cycle: bool,
) -> Dict[str, object]:
    per_video_summary: Dict[str, object] = {}
    all_rows: List[Dict[str, object]] = []
    for video_id, rows in per_video_results.items():
        per_video_summary[video_id] = _video_summary_from_frames(rows)
        all_rows.extend(list(rows or []))

    overall = _video_summary_from_frames(all_rows)
    overall["videos_total"] = int(len(per_video_results))
    overall["videos_with_ok_frames"] = int(
        len(
            [
                vid
                for vid, rows in per_video_results.items()
                if len([r for r in rows if str(r.get("status", "")).strip().lower() == "ok"]) > 0
            ]
        )
    )
    overall["sampling"] = {
        "annotated_frames_total": int(sum(int(((rows[:1] or [{}])[0].get("sampling") or {}).get("annotated_frames_total", 0) or 0) for rows in per_video_results.values() if rows)),
        "verify_frames_sampled": int(sum(int(((rows[:1] or [{}])[0].get("sampling") or {}).get("verify_frames_sampled", 0) or 0) for rows in per_video_results.values() if rows)),
    }

    payload = {
        "run_meta": {
            "gt_json": gt_json_path,
            "videos_dir": videos_dir,
            "masks_dir": masks_dir,
            "output_root": output_root,
            "run_tag": run_tag,
            "run_output_dir": output_dir,
            "method_name": method_name,
            "provider": provider,
            "skip_cycle": bool(variant_skip_cycle),
            "requested_videos": sorted(requested_videos),
            "reverse_entries": bool(args.reverse_entries),
            "video_ord_start": int(args.video_ord_start or 0),
            "video_ord_end": int(args.video_ord_end or 0),
            "verifier_meta": dict(verifier_meta or {}),
            "config": cycle_cfg_path,
            "pipeline_config": pipeline_cfg_path,
            "ontology": ontology_path,
            "rounds": int(args.rounds),
            "enable_single": bool((cycle_cfg.get("cycle") or {}).get("enable_single_turn_probes", False)),
            "enable_multi": bool((cycle_cfg.get("cycle") or {}).get("enable_multi_turn_probes", False)),
            "enable_caption": bool((cycle_cfg.get("cycle") or {}).get("enable_caption_probe", False)),
            "low_quota": bool(args.low_quota),
            "iou_thr": float(args.iou_thr),
            "det_iou_thr": float(args.det_iou_thr),
            "save_visualizations": bool(args.save_visualizations),
            "save_plots": bool(args.save_plots),
            "elapsed_sec": round(time.time() - t_global, 3),
        },
        "overall_summary": overall,
        "per_video_summary": per_video_summary,
    }

    _save_json(os.path.join(output_dir, "summary.json"), payload)
    _save_json(os.path.join(output_dir, "per_video_summary.json"), per_video_summary)
    if bool(args.write_csv):
        frame_rows = _collect_frame_csv_rows(per_video_results)
        frame_fields = [
            "video_id",
            "frame_idx",
            "status",
            "elapsed_sec",
            "annotated_frames_total",
            "verify_frames_sampled",
            "supported_claims_before",
            "supported_claims_after",
            "true_claims_before",
            "false_claims_before",
            "true_claims_after",
            "false_claims_after",
            "claim_accuracy_before",
            "claim_accuracy_after",
            "claim_accuracy_delta",
            "vote_direction_accuracy",
            "true_claim_support_rate",
            "false_claim_conflict_rate",
            "true_claim_conflict_error_rate",
            "false_claim_support_error_rate",
            "false_claim_reduction",
            "false_claim_reduction_rate",
            "correction_hit",
            "probe_total",
            "probe_invalid",
            "vote_support",
            "vote_conflict",
            "vote_uncertain",
            "accepted_claim_count",
            "flagged_claim_count",
            "queue_count",
        ]
        _write_csv(os.path.join(output_dir, "frame_level.csv"), frame_rows, frame_fields)

        video_rows: List[Dict[str, object]] = []
        for video_id, stats in per_video_summary.items():
            metrics_mean = dict((stats or {}).get("verify_metrics_mean") or {})
            cycle_stats = dict((stats or {}).get("cycle") or {})
            claim_counts = dict((stats or {}).get("claim_counts") or {})
            sampling = dict((stats or {}).get("sampling") or {})
            video_rows.append(
                {
                    "video_id": video_id,
                    "frames_total": int((stats or {}).get("frames_total", 0) or 0),
                    "frames_ok": int((stats or {}).get("frames_ok", 0) or 0),
                    "frames_build_failed": int((stats or {}).get("frames_build_failed", 0) or 0),
                    "frames_cycle_failed": int((stats or {}).get("frames_cycle_failed", 0) or 0),
                    "annotated_frames_total": int(sampling.get("annotated_frames_total", 0) or 0),
                    "verify_frames_sampled": int(sampling.get("verify_frames_sampled", 0) or 0),
                    "claim_accuracy_before_mean": _safe_float(metrics_mean.get("claim_accuracy_before", 0.0), 0.0),
                    "claim_accuracy_after_mean": _safe_float(metrics_mean.get("claim_accuracy_after", 0.0), 0.0),
                    "claim_accuracy_delta_mean": _safe_float(metrics_mean.get("claim_accuracy_delta", 0.0), 0.0),
                    "vote_direction_accuracy_mean": _safe_float(metrics_mean.get("vote_direction_accuracy", 0.0), 0.0),
                    "true_claim_support_rate_mean": _safe_float(metrics_mean.get("true_claim_support_rate", 0.0), 0.0),
                    "false_claim_conflict_rate_mean": _safe_float(metrics_mean.get("false_claim_conflict_rate", 0.0), 0.0),
                    "false_claim_reduction_mean": _safe_float(metrics_mean.get("false_claim_reduction", 0.0), 0.0),
                    "correction_hit_rate": _safe_float((stats or {}).get("correction_hit_rate", 0.0), 0.0),
                    "true_claims_before": int(claim_counts.get("true_claims_before", 0) or 0),
                    "false_claims_before": int(claim_counts.get("false_claims_before", 0) or 0),
                    "true_claims_after": int(claim_counts.get("true_claims_after", 0) or 0),
                    "false_claims_after": int(claim_counts.get("false_claims_after", 0) or 0),
                    "probe_total": int(cycle_stats.get("probe_total", 0) or 0),
                    "probe_invalid": int(cycle_stats.get("probe_invalid", 0) or 0),
                    "vote_support": int(cycle_stats.get("vote_support", 0) or 0),
                    "vote_conflict": int(cycle_stats.get("vote_conflict", 0) or 0),
                    "vote_uncertain": int(cycle_stats.get("vote_uncertain", 0) or 0),
                    "accepted_claim_count": int(cycle_stats.get("accepted_claim_count", 0) or 0),
                    "flagged_claim_count": int(cycle_stats.get("flagged_claim_count", 0) or 0),
                    "queue_count": int(cycle_stats.get("queue_count", 0) or 0),
                }
            )
        video_rows.sort(key=lambda r: str(r.get("video_id", "")))
        video_fields = [
            "video_id",
            "frames_total",
            "frames_ok",
            "frames_build_failed",
            "frames_cycle_failed",
            "annotated_frames_total",
            "verify_frames_sampled",
            "claim_accuracy_before_mean",
            "claim_accuracy_after_mean",
            "claim_accuracy_delta_mean",
            "vote_direction_accuracy_mean",
            "true_claim_support_rate_mean",
            "false_claim_conflict_rate_mean",
            "false_claim_reduction_mean",
            "correction_hit_rate",
            "true_claims_before",
            "false_claims_before",
            "true_claims_after",
            "false_claims_after",
            "probe_total",
            "probe_invalid",
            "vote_support",
            "vote_conflict",
            "vote_uncertain",
            "accepted_claim_count",
            "flagged_claim_count",
            "queue_count",
        ]
        _write_csv(os.path.join(output_dir, "video_level.csv"), video_rows, video_fields)

    utility_outputs = _verification_utility_outputs(
        per_video_results=per_video_results,
        output_dir=output_dir,
        run_tag=str(run_tag or ""),
        method_name=str(method_name or ""),
        iou_thr=float(args.iou_thr),
    )
    payload["verification_utility_files"] = {
        "summary_csv": utility_outputs.get("csv", ""),
        "summary_json": utility_outputs.get("json", ""),
        "evidence_md": utility_outputs.get("evidence_md", ""),
        "table_tex": utility_outputs.get("table_tex", ""),
    }
    _save_json(os.path.join(output_dir, "summary.json"), payload)
    if bool(args.save_plots):
        try:
            plot_rows = _collect_frame_csv_rows(per_video_results)
            payload["plot_files"] = _save_summary_plots(output_dir, plot_rows)
            _save_json(os.path.join(output_dir, "summary.json"), payload)
        except Exception as exc:
            print(f"[vidor-eval] Plot generation skipped: {exc}")
    return {
        "payload": payload,
        "utility_outputs": utility_outputs,
    }


def _claim_debug_samples_from_graph(graph: Dict[str, object], *, limit: int = 3) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for claim in list(graph_to_claims(graph) or [])[: max(1, int(limit or 1))]:
        out.append(
            {
                "claim_id": str(claim.claim_id or ""),
                "claim_type": str(claim.claim_type or ""),
                "subject_id": str(claim.subject_id or ""),
                "predicate": str(claim.predicate or ""),
                "object_id": str(claim.object_id or ""),
                "value": str(claim.value or ""),
                "status": str(getattr(claim, "status", "") or ""),
            }
        )
    return out


def _claim_debug_samples_from_payload(payload: object, *, limit: int = 3) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    if not isinstance(payload, dict):
        return out
    for _, row in list(dict(payload).items())[: max(1, int(limit or 1))]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "claim_id": str(row.get("claim_id", "") or ""),
                "claim_type": str(row.get("claim_type", "") or ""),
                "subject_id": str(row.get("subject_id", "") or ""),
                "predicate": str(row.get("predicate", "") or ""),
                "object_id": str(row.get("object_id", "") or ""),
                "value": str(row.get("value", "") or ""),
                "status": str(row.get("status", "") or ""),
            }
        )
    return out


def _vote_debug_samples(votes: object, *, limit: int = 3) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in list(votes or [])[: max(1, int(limit or 1))]:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "claim_id": str(row.get("claim_id", "") or ""),
                "vote": str(row.get("vote", "") or ""),
                "score": _safe_float(row.get("score", 0.0), 0.0),
                "view_type": str(row.get("view_type", "") or ""),
                "question": str(row.get("question", "") or "")[:180],
            }
        )
    return out


def _probe_debug_samples(results: object, *, limit: int = 3) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in list(results or [])[: max(1, int(limit or 1))]:
        if not isinstance(row, dict):
            continue
        parsed = dict(row.get("parsed_response") or {})
        out.append(
            {
                "claim_id": str(row.get("claim_id", "") or row.get("target_claim_id", "") or ""),
                "view_type": str(row.get("view_type", "") or ""),
                "schema_valid": bool(row.get("schema_valid", True)),
                "question": str(row.get("question", "") or "")[:180],
                "raw_text": str(parsed.get("raw_text", row.get("raw_text", "")) or "")[:220],
                "parsed_keys": sorted([str(x) for x in parsed.keys()])[:10],
            }
        )
    return out


def _sanitize_run_token(value: object, *, default: str = "run") -> str:
    token = str(value or "").strip().lower()
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in token)
    token = "_".join([part for part in token.split("_") if part])
    return token or str(default)


def _build_run_tag(
    *,
    provider: str,
    requested_videos: Set[str],
    max_videos: int,
    max_frames_per_video: int,
    skip_cycle: bool,
    low_quota: bool,
    explicit_name: str = "",
) -> str:
    custom = _sanitize_run_token(explicit_name, default="") if str(explicit_name or "").strip() else ""
    if custom:
        return custom
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    scope = "allvideos" if int(max_videos or 0) == 0 else f"v{int(max_videos)}"
    if requested_videos:
        video_list = sorted([_sanitize_run_token(x, default="video") for x in requested_videos])
        preview = "-".join(video_list[:2])
        if len(video_list) > 2:
            preview = f"{preview}-plus{len(video_list) - 2}"
        scope = f"ids_{preview}"
    frame_cap = "fAll" if int(max_frames_per_video or 0) == 0 else f"f{int(max_frames_per_video)}"
    mode = "buildonly" if bool(skip_cycle) else "cycle"
    quota = "lowq" if bool(low_quota) else "fullq"
    return "_".join(
        [
            ts,
            _sanitize_run_token(provider, default="provider"),
            scope,
            frame_cap,
            mode,
            quota,
        ]
    )


def _draw_graph_overlay(
    image_path: str,
    graph: Dict[str, object],
    out_path: str,
    *,
    title_lines: List[str],
    box_color: Tuple[int, int, int],
) -> None:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for visualization output.") from exc
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Unable to read image for visualization: {image_path}")
    canvas = image.copy()
    for idx, node in enumerate(list(graph.get("nodes") or []), start=1):
        if not isinstance(node, dict):
            continue
        bbox = list(node.get("bbox") or [0, 0, 0, 0])
        if len(bbox) < 4:
            continue
        try:
            x, y, w, h = [int(float(bbox[i] or 0)) for i in range(4)]
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        label = _normalize_label(node.get("canonical_label", node.get("label", ""))) or "object"
        entity_id = str(node.get("entity_id", "") or "").strip()
        text = f"{idx}:{label}"
        if entity_id:
            text = f"{text} ({entity_id})"
        cv2.rectangle(canvas, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(
            canvas,
            text,
            (x, max(14, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )
    y = 18
    for line in list(title_lines or []):
        text = str(line or "").strip()
        if not text:
            continue
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        y += 18
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if not cv2.imwrite(str(out_path), canvas):
        raise RuntimeError(f"Unable to write visualization image: {out_path}")


def _save_frame_visualizations(
    *,
    frame_image_path: str,
    video_out_dir: str,
    frame_idx: int,
    initial_graph: Dict[str, object],
    final_graph: Dict[str, object],
    eval_before: Dict[str, object],
    eval_after: Dict[str, object],
    det_eval_before: Dict[str, object],
    det_eval_after: Dict[str, object],
) -> Dict[str, str]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("opencv-python is required for visualization output.") from exc
    stem = f"{int(frame_idx):06d}"
    initial_path = os.path.join(video_out_dir, f"vis_initial_{stem}.jpg")
    final_path = os.path.join(video_out_dir, f"vis_final_{stem}.jpg")
    compare_path = os.path.join(video_out_dir, f"vis_compare_{stem}.jpg")
    before_lines = [
        f"Before nodeF1={_safe_float(((eval_before.get('node') or {}).get('f1')), 0.0):.3f}",
        f"Before relF1={_safe_float(((eval_before.get('relation') or {}).get('f1')), 0.0):.3f}",
        f"Before detF1={_safe_float((det_eval_before.get('f1')), 0.0):.3f}",
        f"Before mIoU={_safe_float((det_eval_before.get('miou')), 0.0):.3f}",
        f"Before TP/FP/FN={int(det_eval_before.get('tp', 0) or 0)}/{int(det_eval_before.get('fp', 0) or 0)}/{int(det_eval_before.get('fn', 0) or 0)}",
    ]
    after_lines = [
        f"After nodeF1={_safe_float(((eval_after.get('node') or {}).get('f1')), 0.0):.3f}",
        f"After relF1={_safe_float(((eval_after.get('relation') or {}).get('f1')), 0.0):.3f}",
        f"After detF1={_safe_float((det_eval_after.get('f1')), 0.0):.3f}",
        f"After mIoU={_safe_float((det_eval_after.get('miou')), 0.0):.3f}",
        f"After TP/FP/FN={int(det_eval_after.get('tp', 0) or 0)}/{int(det_eval_after.get('fp', 0) or 0)}/{int(det_eval_after.get('fn', 0) or 0)}",
    ]
    _draw_graph_overlay(frame_image_path, initial_graph, initial_path, title_lines=before_lines, box_color=(0, 140, 255))
    _draw_graph_overlay(frame_image_path, final_graph, final_path, title_lines=after_lines, box_color=(0, 220, 0))
    left = cv2.imread(initial_path)
    right = cv2.imread(final_path)
    if left is None or right is None:
        raise RuntimeError("Unable to build compare visualization because one side failed to render.")
    if left.shape[0] != right.shape[0]:
        target_h = min(left.shape[0], right.shape[0])
        left = cv2.resize(left, (int(left.shape[1] * target_h / max(1, left.shape[0])), target_h))
        right = cv2.resize(right, (int(right.shape[1] * target_h / max(1, right.shape[0])), target_h))
    compare = cv2.hconcat([left, right])
    if not cv2.imwrite(compare_path, compare):
        raise RuntimeError(f"Unable to write compare visualization image: {compare_path}")
    return {
        "vis_initial": initial_path,
        "vis_final": final_path,
        "vis_compare": compare_path,
    }


def _save_summary_plots(output_dir: str, frame_rows: List[Dict[str, object]]) -> List[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for --save_plots.") from exc
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    ok_rows = [dict(row) for row in list(frame_rows or []) if str(row.get("status", "") or "").strip().lower() == "ok"]
    if not ok_rows:
        return []

    def _mean(key: str) -> float:
        vals = [_safe_float(row.get(key, 0.0), 0.0) for row in ok_rows]
        return float(sum(vals) / max(1, len(vals)))

    saved: List[str] = []
    paired_specs = [
        ("claim_accuracy_before", "claim_accuracy_after", "claim_accuracy_before_after.png", "Claim Accuracy Before/After"),
        ("true_claim_support_rate", "false_claim_conflict_rate", "verify_vote_quality.png", "Verify Vote Quality"),
    ]
    for before_key, after_key, filename, title in paired_specs:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.bar(["before", "after"], [_mean(before_key), _mean(after_key)], color=["#4C78A8", "#F58518"])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.set_ylabel("score")
        path = os.path.join(plot_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    deltas = [_safe_float(row.get("claim_accuracy_delta", 0.0), 0.0) for row in ok_rows]
    ax.hist(deltas, bins=min(20, max(5, len(deltas))), color="#54A24B", edgecolor="white")
    ax.set_title("Claim Accuracy Delta Histogram")
    ax.set_xlabel("delta")
    ax.set_ylabel("frame count")
    path = os.path.join(plot_dir, "claim_delta_histogram.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    false_before = sum(int(row.get("false_claims_before", 0) or 0) for row in ok_rows)
    true_before = sum(int(row.get("true_claims_before", 0) or 0) for row in ok_rows)
    false_after = sum(int(row.get("false_claims_after", 0) or 0) for row in ok_rows)
    true_after = sum(int(row.get("true_claims_after", 0) or 0) for row in ok_rows)
    ax.bar(["False before", "True before", "False after", "True after"], [false_before, true_before, false_after, true_after], color=["#E45756", "#72B7B2", "#E45756", "#72B7B2"])
    ax.set_title("Claim True/False Comparison")
    ax.set_ylabel("count")
    path = os.path.join(plot_dir, "claim_true_false_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)
    return saved


def main() -> None:
    args = _parse_args()

    gt_json_path = os.path.abspath(os.path.expanduser(str(args.gt_json)))
    videos_dir = os.path.abspath(os.path.expanduser(str(args.videos_dir)))
    masks_dir = os.path.abspath(os.path.expanduser(str(args.masks_dir)))
    output_root = os.path.abspath(os.path.expanduser(str(args.output_dir)))

    cycle_cfg_path = str(args.config)
    if not os.path.isabs(cycle_cfg_path):
        cycle_cfg_path = os.path.join(_HERE, cycle_cfg_path)
    cycle_cfg_path = os.path.abspath(os.path.expanduser(cycle_cfg_path))

    pipeline_cfg_path = str(args.pipeline_config)
    if not os.path.isabs(pipeline_cfg_path):
        pipeline_cfg_path = os.path.join(_HERE, pipeline_cfg_path)
    pipeline_cfg_path = os.path.abspath(os.path.expanduser(pipeline_cfg_path))

    ontology_path = str(args.ontology)
    if not os.path.isabs(ontology_path):
        ontology_path = os.path.join(_HERE, ontology_path)
    ontology_path = os.path.abspath(os.path.expanduser(ontology_path))

    if not os.path.isfile(gt_json_path):
        raise FileNotFoundError(f"GT json not found: {gt_json_path}")
    if not os.path.isdir(videos_dir):
        raise NotADirectoryError(f"videos_dir not found: {videos_dir}")
    if not os.path.isdir(masks_dir):
        raise NotADirectoryError(f"masks_dir not found: {masks_dir}")
    if not os.path.isfile(cycle_cfg_path):
        raise FileNotFoundError(f"cycle config not found: {cycle_cfg_path}")
    if not os.path.isfile(pipeline_cfg_path):
        raise FileNotFoundError(f"pipeline config not found: {pipeline_cfg_path}")
    if not os.path.isfile(ontology_path):
        raise FileNotFoundError(f"ontology config not found: {ontology_path}")
    detection_dir = os.path.abspath(os.path.expanduser(str(args.load_detections_from or "").strip())) if str(args.load_detections_from or "").strip() else ""
    if detection_dir and not os.path.isdir(detection_dir):
        raise NotADirectoryError(f"load_detections_from not found: {detection_dir}")

    provider = normalize_cycle_provider(args.provider, default=DEFAULT_CYCLE_PROVIDER)
    requested_videos = {str(x).strip() for x in list(args.video_ids or []) if str(x).strip()}
    enable_single: Optional[bool] = None
    enable_multi: Optional[bool] = None
    enable_caption: Optional[bool] = None
    if args.enable_single:
        enable_single = True
    if args.disable_single:
        enable_single = False
    if args.enable_multi:
        enable_multi = True
    if args.disable_multi:
        enable_multi = False
    if args.enable_caption:
        enable_caption = True
    if args.disable_caption:
        enable_caption = False
    effective_skip_cycle, enable_single, enable_multi, enable_caption = _apply_verification_mode_overrides(
        args,
        enable_single=enable_single,
        enable_multi=enable_multi,
        enable_caption=enable_caption,
    )
    args.skip_cycle = bool(effective_skip_cycle)
    run_tag = _build_run_tag(
        provider=provider,
        requested_videos=sorted(requested_videos),
        max_videos=int(args.max_videos or 0),
        max_frames_per_video=int(args.max_frames_per_video or 0),
        skip_cycle=bool(effective_skip_cycle),
        low_quota=bool(args.low_quota),
        explicit_name=str(args.run_name or ""),
    )
    output_dir = os.path.join(output_root, run_tag)
    frames_cache_dir = os.path.join(output_dir, "_frame_cache")

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_cache_dir, exist_ok=True)
    print(f"[vidor-eval] Output root: {output_root}")
    print(f"[vidor-eval] Run output dir: {output_dir}")
    if detection_dir:
        print(f"[vidor-eval] Loading fixed detections from: {detection_dir}")
    print(f"[vidor-eval] Verification mode: {str(args.verification_mode or 'legacy')}")

    print("[vidor-eval] Loading GT json...")
    gt_payload = _load_json(gt_json_path)
    entries = [dict(x) for x in list(gt_payload.get("data") or []) if isinstance(x, dict)]
    if requested_videos:
        entries = [row for row in entries if str(row.get("video_id", "") or "").strip() in requested_videos]
    if bool(args.reverse_entries):
        entries = list(reversed(entries))
        print(f"[vidor-eval] Reversed GT entry traversal order (count={len(entries)})")
    video_ord_start = int(args.video_ord_start or 0)
    video_ord_end = int(args.video_ord_end or 0)
    if video_ord_start > 0:
        if video_ord_end <= 0:
            video_ord_end = video_ord_start
        if video_ord_end < video_ord_start:
            raise ValueError("--video_ord_end must be >= --video_ord_start")
        start_idx = max(0, video_ord_start - 1)
        end_idx = min(len(entries), video_ord_end)
        entries = entries[start_idx:end_idx]
        print(
            "[vidor-eval] Selected GT entry ordinals: "
            f"{video_ord_start}-{video_ord_end} "
            f"(1-based, inclusive; actual_count={len(entries)})"
        )
    if int(args.max_videos or 0) > 0:
        entries = entries[: int(args.max_videos)]
    print(f"[vidor-eval] Videos selected: {len(entries)}")

    base_cycle_cfg: Dict[str, object] = {}
    cycle_cfg: Dict[str, object] = {}
    ontology = None
    verifier = None
    verifier_meta: Dict[str, object] = {"verifier_provider": "disabled", "verifier_model_id": ""}
    if bool(args.skip_cycle) and str(args.ablation_suite or "none").strip().lower() == "none":
        print("[vidor-eval] Cycle verify disabled (--skip_cycle): running build-only ablation.")
    else:
        print("[vidor-eval] Loading cycle config and ontology...")
        base_cycle_cfg = _load_json(cycle_cfg_path)
        cycle_cfg = _build_cycle_cfg(
            base_cfg=base_cycle_cfg,
            provider=provider,
            rounds=int(args.rounds),
            enable_single=enable_single,
            enable_multi=enable_multi,
            enable_caption=enable_caption,
            low_quota=bool(args.low_quota),
        )
        ontology = load_ontology(ontology_path)

        print(f"[vidor-eval] Building verifier provider={provider} ...")
        verifier, verifier_meta = build_vision_verifier(
            cycle_cfg,
            preferred_provider=provider,
            allow_mock_fallback=False,
        )
        print(
            "[vidor-eval] Verifier ready: "
            f"{verifier_meta.get('verifier_provider', '?')} "
            f"model={verifier_meta.get('verifier_model_id', '?')}"
        )

    variant_specs = _build_variant_specs(
        args,
        base_cfg=(base_cycle_cfg or cycle_cfg or _load_json(cycle_cfg_path)),
        provider=provider,
    )
    suite_mode = len(variant_specs) > 1
    if suite_mode:
        print("[vidor-eval] Ablation suite enabled:")
        for spec in variant_specs:
            print(
                f"[vidor-eval]   - {spec['slug']}: {spec['method_name']} "
                f"skip_cycle={int(bool(spec['skip_cycle']))}"
            )

    t_global = time.time()
    per_variant_results: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        str(spec["slug"]): {} for spec in variant_specs
    }

    try:
        for vid_idx, entry in enumerate(entries, start=1):
            video_id = str(entry.get("video_id", "") or "").strip()
            if not video_id:
                continue
            video_path = _resolve_video_path(videos_dir, video_id)
            if not video_path:
                print(f"[vidor-eval] [{vid_idx}/{len(entries)}] skip {video_id}: video file missing")
                missing_row = {
                    "video_id": video_id,
                    "status": "video_missing",
                    "error": f"missing video in {videos_dir}",
                }
                for spec in variant_specs:
                    per_variant_results[str(spec["slug"])][video_id] = [dict(missing_row)]
                continue

            all_gt_frames = _annotated_frames_from_entry(entry)
            sampled_frames = _sample_verify_frames_from_entry(entry)
            final_frames = _limit_frames_uniform(sampled_frames, int(args.max_frames_per_video or 0))
            if not final_frames:
                for spec in variant_specs:
                    per_variant_results[str(spec["slug"])][video_id] = []
                continue

            print(
                f"[vidor-eval] [{video_id}] "
                f"annotated_frames_total={len(all_gt_frames)} "
                f"sampled_frames_count={len(sampled_frames)} "
                f"final_verify_frames_count={len(final_frames)}"
            )
            print(f"[vidor-eval] [{vid_idx}/{len(entries)}] loading GT reference for video={video_id}")
            gt_ref = load_pvsg_video_reference(
                video_path=video_path,
                frame_indices=final_frames,
                pvsg_json_path=gt_json_path,
                masks_root=masks_dir,
            )
            per_frame_gt = dict(gt_ref.get("per_frame") or {})
            variant_video_out_dirs: Dict[str, str] = {}
            variant_frame_results: Dict[str, List[Dict[str, object]]] = {}
            for spec in variant_specs:
                variant_root = output_dir if not suite_mode else os.path.join(output_dir, str(spec["slug"]))
                video_out_dir = os.path.join(variant_root, video_id)
                os.makedirs(video_out_dir, exist_ok=True)
                variant_video_out_dirs[str(spec["slug"])] = video_out_dir
                variant_frame_results[str(spec["slug"])] = []
            total_frames = len(final_frames)
            for frame_pos, fidx in enumerate(final_frames, start=1):
                frame_key = str(int(fidx))
                frame_gt = dict(per_frame_gt.get(frame_key) or {})
                t0 = time.time()
                print(
                    f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                    f"frame {frame_pos}/{total_frames} start video={video_id} frame_idx={int(fidx)}"
                )
                if not bool(frame_gt.get("reference_available", False)):
                    row = {
                        "video_id": video_id,
                        "frame_idx": int(fidx),
                        "status": "gt_frame_unavailable",
                        "error": str(frame_gt.get("reason", "gt_reference_unavailable")),
                    }
                    print(
                        f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                        f"frame {frame_pos}/{total_frames} skip video={video_id} frame_idx={int(fidx)} "
                        f"status=gt_frame_unavailable reason={row['error']}"
                    )
                    for spec in variant_specs:
                        variant_frame_results[str(spec["slug"])].append(dict(row))
                        _save_json(os.path.join(variant_video_out_dirs[str(spec["slug"])], f"{int(fidx):06d}.json"), row)
                    continue

                try:
                    print(
                        f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                        f"frame {frame_pos}/{total_frames} extract start video={video_id} frame_idx={int(fidx)}"
                    )
                    frame_img_path, frame_w, frame_h = _extract_frame_to_jpg(
                        video_path=video_path,
                        frame_idx=int(fidx),
                        out_dir=os.path.join(frames_cache_dir, video_id),
                    )
                    print(
                        f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                        f"frame {frame_pos}/{total_frames} extract done video={video_id} frame_idx={int(fidx)} "
                        f"image={frame_img_path} size={int(frame_w)}x{int(frame_h)}"
                    )
                except Exception as exc:
                    row = {
                        "video_id": video_id,
                        "frame_idx": int(fidx),
                        "status": "frame_extract_failed",
                        "error": str(exc),
                    }
                    print(
                        f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                        f"frame {frame_pos}/{total_frames} extract failed video={video_id} frame_idx={int(fidx)} "
                        f"error={str(exc)}"
                    )
                    for spec in variant_specs:
                        variant_frame_results[str(spec["slug"])].append(dict(row))
                        _save_json(os.path.join(variant_video_out_dirs[str(spec["slug"])], f"{int(fidx):06d}.json"), row)
                    continue

                image_id = f"{video_id}_f{int(fidx):06d}"
                gt_graph = _gt_graph_from_frame_ref(video_id, int(fidx), frame_gt)
                loaded_detection_payload: Optional[Dict[str, object]] = None
                if detection_dir:
                    try:
                        loaded_detection_payload = load_detection_record(detection_dir, video_id, int(fidx))
                        loaded_prompt_count = len(list((loaded_detection_payload.get("prompt_results") or [])))
                        loaded_post_count = len(list((loaded_detection_payload.get("post_threshold_records") or [])))
                        print(
                            f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                            f"frame {frame_pos}/{total_frames} detections loaded video={video_id} frame_idx={int(fidx)} "
                            f"prompts={loaded_prompt_count} detections={loaded_post_count}"
                        )
                    except Exception as exc:
                        row = {
                            "video_id": video_id,
                            "frame_idx": int(fidx),
                            "status": "detection_load_failed",
                            "error": str(exc),
                            "elapsed_sec": round(time.time() - t0, 3),
                        }
                        print(
                            f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                            f"frame {frame_pos}/{total_frames} detection load failed video={video_id} frame_idx={int(fidx)} "
                            f"elapsed={row['elapsed_sec']:.3f}s error={str(exc)}"
                        )
                        for spec in variant_specs:
                            variant_frame_results[str(spec["slug"])].append(dict(row))
                            _save_json(os.path.join(variant_video_out_dirs[str(spec["slug"])], f"{int(fidx):06d}.json"), row)
                        continue

                try:
                    print(
                        f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                        f"frame {frame_pos}/{total_frames} build start video={video_id} frame_idx={int(fidx)} image_id={image_id}"
                    )
                    initial_graph = run_build_scene_graph(
                        image_id=image_id,
                        image_path=frame_img_path,
                        ontology_path=ontology_path,
                        pipeline_cfg_path=pipeline_cfg_path,
                        image_size=(int(frame_w), int(frame_h)),
                        enable_sentence_refine=False,
                        precomputed_detections=loaded_detection_payload,
                    )
                    initial_graph.setdefault("metadata", {})
                    initial_graph["metadata"]["video_id"] = video_id
                    initial_graph["metadata"]["graph_frame_idx"] = int(fidx)
                    candidate_debug = dict((initial_graph.get("metadata") or {}).get("candidate_debug") or {})
                    num_raw_detections = int(candidate_debug.get("raw_count", 0) or 0)
                    num_after_filter = int(candidate_debug.get("kept_count", 0) or 0)
                    num_in_graph = int(len(list(initial_graph.get("nodes") or [])))
                    print(
                        f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                        f"frame {frame_pos}/{total_frames} build done video={video_id} frame_idx={int(fidx)} "
                        f"nodes={len(list(initial_graph.get('nodes') or []))} "
                        f"edges={len(list(initial_graph.get('edges') or []))} "
                        f"num_raw_detections={num_raw_detections} "
                        f"num_after_filter={num_after_filter} "
                        f"num_in_graph={num_in_graph} "
                        f"loaded_detection_file={int(bool(candidate_debug.get('loaded_from_detection_file', False)))}"
                    )
                except Exception as exc:
                    row = {
                        "video_id": video_id,
                        "frame_idx": int(fidx),
                        "status": "build_failed",
                        "error": str(exc),
                        "elapsed_sec": round(time.time() - t0, 3),
                    }
                    print(
                        f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                        f"frame {frame_pos}/{total_frames} build failed video={video_id} frame_idx={int(fidx)} "
                        f"elapsed={row['elapsed_sec']:.3f}s error={str(exc)}"
                    )
                    for spec in variant_specs:
                        variant_frame_results[str(spec["slug"])].append(dict(row))
                        _save_json(os.path.join(variant_video_out_dirs[str(spec["slug"])], f"{int(fidx):06d}.json"), row)
                    continue

                eval_before = _evaluate_graph(initial_graph, gt_graph, iou_thr=float(args.iou_thr))
                matched_pairs_before = _matched_label_pairs(initial_graph, gt_graph, iou_thr=float(args.iou_thr))
                print(
                    f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                    f"frame {frame_pos}/{total_frames} eval-before video={video_id} frame_idx={int(fidx)} "
                    f"node_f1={_safe_float(((eval_before.get('node') or {}).get('f1')), 0.0):.3f} "
                    f"rel_f1={_safe_float(((eval_before.get('relation') or {}).get('f1')), 0.0):.3f} "
                    f"attr_f1={_safe_float(((eval_before.get('attribute') or {}).get('f1')), 0.0):.3f}"
                )
                print(
                    f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                    f"frame {frame_pos}/{total_frames} matched_labels_before video={video_id} frame_idx={int(fidx)} "
                    f"pairs={matched_pairs_before if matched_pairs_before else []}"
                )

                for spec in variant_specs:
                    variant_slug = str(spec["slug"])
                    variant_name = str(spec["method_name"])
                    variant_skip_cycle = bool(spec["skip_cycle"])
                    variant_cycle_cfg = dict(spec["cycle_cfg"] or {})
                    video_out_dir = variant_video_out_dirs[variant_slug]
                    if variant_skip_cycle:
                        cycle_result = {
                            "graph_after": dict(initial_graph),
                            "summary": {},
                            "metrics": {},
                            "probe_results": [],
                            "human_queue": [],
                            "votes": [],
                            "caption": {},
                        }
                        final_graph = dict(initial_graph)
                        print(
                            f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                            f"frame {frame_pos}/{total_frames} cycle skipped video={video_id} frame_idx={int(fidx)} "
                            f"variant={variant_slug} mode={str(args.verification_mode or 'legacy')}"
                        )
                    else:
                        try:
                            print(
                                f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                                f"frame {frame_pos}/{total_frames} cycle start video={video_id} frame_idx={int(fidx)} "
                                f"variant={variant_slug} mode={str(args.verification_mode or 'legacy')}"
                            )
                            cycle_result = run_cycle_refine(
                                graph=initial_graph,
                                image_path=frame_img_path,
                                verifier=verifier,
                                ontology=ontology,
                                cfg=variant_cycle_cfg,
                            )
                            final_graph = dict(cycle_result.get("graph_after") or initial_graph)
                            print(
                                f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                                f"frame {frame_pos}/{total_frames} cycle done video={video_id} frame_idx={int(fidx)} "
                                f"variant={variant_slug} probes={len(list(cycle_result.get('probe_results') or []))} "
                                f"queue={len(list(cycle_result.get('human_queue') or []))} "
                                f"graph_delta_nodes={len(list(final_graph.get('nodes') or [])) - len(list(initial_graph.get('nodes') or []))} "
                                f"graph_delta_edges={len(list(final_graph.get('edges') or [])) - len(list(initial_graph.get('edges') or []))}"
                            )
                        except Exception as exc:
                            row = {
                                "video_id": video_id,
                                "frame_idx": int(fidx),
                                "status": "cycle_failed",
                                "error": str(exc),
                                "eval": {"before": eval_before},
                                "gt_graph": gt_graph,
                                "initial_graph": initial_graph,
                                "elapsed_sec": round(time.time() - t0, 3),
                                "sampling": {
                                    "annotated_frames_total": int(len(all_gt_frames)),
                                    "verify_frames_sampled": int(len(final_frames)),
                                },
                            }
                            print(
                                f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                                f"frame {frame_pos}/{total_frames} cycle failed video={video_id} frame_idx={int(fidx)} "
                                f"variant={variant_slug} elapsed={row['elapsed_sec']:.3f}s error={str(exc)}"
                            )
                            variant_frame_results[variant_slug].append(row)
                            _save_json(os.path.join(video_out_dir, f"{int(fidx):06d}.json"), row)
                            continue

                    eval_after = _evaluate_graph(final_graph, gt_graph, iou_thr=float(args.iou_thr))
                    cycle_stats = _cycle_probe_stats(cycle_result)
                    cycle_stats["enabled"] = int(not variant_skip_cycle)
                    verify_eval = _evaluate_verify_effectiveness(
                        initial_graph,
                        final_graph,
                        gt_graph,
                        cycle_result,
                        iou_thr=float(args.iou_thr),
                    )
                    if bool(args.debug_claim_trace):
                        initial_claims = list(graph_to_claims(initial_graph) or [])
                        final_claims = list(graph_to_claims(final_graph) or [])
                        cycle_claims = dict(cycle_result.get("claims") or {})
                        cycle_votes = [dict(x) for x in list(cycle_result.get("votes") or []) if isinstance(x, dict)]
                        cycle_probes = [dict(x) for x in list(cycle_result.get("probe_results") or []) if isinstance(x, dict)]
                        print(
                            f"[TRACE][A] variant={variant_slug} video={video_id} frame_idx={int(fidx)} "
                            f"build_claims={len(initial_claims)} samples={json.dumps(_claim_debug_samples_from_graph(initial_graph), ensure_ascii=True)}"
                        )
                        print(
                            f"[TRACE][B] variant={variant_slug} video={video_id} frame_idx={int(fidx)} "
                            f"raw_probe_samples={json.dumps(_probe_debug_samples(cycle_probes), ensure_ascii=True)}"
                        )
                        print(
                            f"[TRACE][C] variant={variant_slug} video={video_id} frame_idx={int(fidx)} "
                            f"parsed_vote_count={len(cycle_votes)} vote_samples={json.dumps(_vote_debug_samples(cycle_votes), ensure_ascii=True)}"
                        )
                        print(
                            f"[TRACE][D] variant={variant_slug} video={video_id} frame_idx={int(fidx)} "
                            f"cycle_claims={len(cycle_claims)} samples={json.dumps(_claim_debug_samples_from_payload(cycle_claims), ensure_ascii=True)}"
                        )
                        print(
                            f"[TRACE][E] variant={variant_slug} video={video_id} frame_idx={int(fidx)} "
                            f"final_claims={len(final_claims)} samples={json.dumps(_claim_debug_samples_from_graph(final_graph), ensure_ascii=True)}"
                        )
                        print(
                            f"[TRACE][F] variant={variant_slug} video={video_id} frame_idx={int(fidx)} "
                            f"verify_before={json.dumps(dict(verify_eval.get('before') or {}), ensure_ascii=True)} "
                            f"verify_after={json.dumps(dict(verify_eval.get('after') or {}), ensure_ascii=True)}"
                        )

                    vis_paths: Dict[str, str] = {}
                    if bool(args.save_visualizations):
                        try:
                            vis_paths = _save_frame_visualizations(
                                frame_image_path=frame_img_path,
                                video_out_dir=video_out_dir,
                                frame_idx=int(fidx),
                                initial_graph=initial_graph,
                                final_graph=final_graph,
                                eval_before=eval_before,
                                eval_after=eval_after,
                                det_eval_before={},
                                det_eval_after={},
                            )
                        except Exception as exc:
                            print(
                                f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                                f"frame {frame_pos}/{total_frames} visualization failed video={video_id} frame_idx={int(fidx)} "
                                f"variant={variant_slug} error={str(exc)}"
                            )

                    row = {
                        "video_id": video_id,
                        "frame_idx": int(fidx),
                        "image_id": image_id,
                        "status": "ok",
                        "paths": {
                            "video_path": video_path,
                            "frame_image_path": frame_img_path,
                            "mask_root": masks_dir,
                            **vis_paths,
                        },
                        "sampling": {
                            "annotated_frames_total": int(len(all_gt_frames)),
                            "verify_frames_sampled": int(len(final_frames)),
                        },
                        "build_debug": dict(((initial_graph.get("metadata") or {}).get("candidate_debug")) or {}),
                        "detection_source": dict(loaded_detection_payload or {}),
                        "gt_graph": gt_graph,
                        "initial_graph": initial_graph,
                        "final_graph": final_graph,
                        "eval": {
                            "before": eval_before,
                            "after": eval_after,
                        },
                        "verify_eval": verify_eval,
                        "cycle": cycle_stats,
                        "summary": dict(cycle_result.get("summary") or {}),
                        "metrics": dict(cycle_result.get("metrics") or {}),
                        "votes": [dict(x) for x in list(cycle_result.get("votes") or []) if isinstance(x, dict)],
                        "probe_results": [dict(x) for x in list(cycle_result.get("probe_results") or []) if isinstance(x, dict)],
                        "human_queue": [dict(x) for x in list(cycle_result.get("human_queue") or []) if isinstance(x, dict)],
                        "cycle_result_raw": dict(cycle_result),
                        "elapsed_sec": round(time.time() - t0, 3),
                        "verification_mode": str(args.verification_mode or "legacy"),
                    }
                    print(
                        f"[vidor-eval] [{vid_idx}/{len(entries)}] "
                        f"frame {frame_pos}/{total_frames} done video={video_id} frame_idx={int(fidx)} variant={variant_slug} "
                        f"elapsed={row['elapsed_sec']:.3f}s "
                        f"claim_acc={_safe_float(((verify_eval.get('before') or {}).get('claim_accuracy')), 0.0):.3f}"
                        f"->{_safe_float(((verify_eval.get('after') or {}).get('claim_accuracy')), 0.0):.3f} "
                        f"vote_dir_acc={_safe_float(((verify_eval.get('vote_eval') or {}).get('vote_direction_accuracy')), 0.0):.3f} "
                        f"false_claim_reduction={int(((verify_eval.get('correction') or {}).get('false_claim_reduction', 0) or 0))} "
                        f"hit={int(bool((verify_eval.get('correction') or {}).get('correction_hit', False)))}"
                    )
                    variant_frame_results[variant_slug].append(row)
                    _save_json(os.path.join(video_out_dir, f"{int(fidx):06d}.json"), row)

            for spec in variant_specs:
                variant_slug = str(spec["slug"])
                per_variant_results[variant_slug][video_id] = list(variant_frame_results[variant_slug])
                ok_count = len([row for row in variant_frame_results[variant_slug] if str(row.get("status", "") or "").strip().lower() == "ok"])
                print(
                    f"[vidor-eval] [{vid_idx}/{len(entries)}] video done video={video_id} "
                    f"variant={variant_slug} ok_frames={ok_count}/{len(variant_frame_results[variant_slug])}"
                )
    finally:
        # Keep backend resource handling consistent with mainline runtime.
        release_backend_pool()
    finalize_results: Dict[str, Dict[str, object]] = {}
    for spec in variant_specs:
        variant_slug = str(spec["slug"])
        variant_root = output_dir if not suite_mode else os.path.join(output_dir, variant_slug)
        finalize_results[variant_slug] = _finalize_variant_outputs(
            output_dir=variant_root,
            output_root=output_root,
            run_tag=(f"{run_tag}/{variant_slug}" if suite_mode else str(run_tag)),
            provider=provider,
            verifier_meta=verifier_meta,
            cycle_cfg_path=cycle_cfg_path,
            pipeline_cfg_path=pipeline_cfg_path,
            ontology_path=ontology_path,
            videos_dir=videos_dir,
            masks_dir=masks_dir,
            gt_json_path=gt_json_path,
            requested_videos=requested_videos,
            args=args,
            cycle_cfg=dict(spec["cycle_cfg"] or {}),
            per_video_results=per_variant_results.get(variant_slug, {}),
            t_global=t_global,
            method_name=str(spec["method_name"]),
            variant_skip_cycle=bool(spec["skip_cycle"]),
        )

    aggregate_run_outputs = _update_verification_utility_aggregate(output_dir if suite_mode else output_root)
    if suite_mode:
        _save_json(
            os.path.join(output_dir, "ablation_manifest.json"),
            {
                "run_tag": run_tag,
                "variants": [
                    {
                        "slug": str(spec["slug"]),
                        "method_name": str(spec["method_name"]),
                        "output_dir": (output_dir if not suite_mode else os.path.join(output_dir, str(spec["slug"]))),
                        "summary_json": dict((finalize_results.get(str(spec["slug"])) or {}).get("utility_outputs") or {}).get("json", ""),
                    }
                    for spec in variant_specs
                ],
                "aggregate_outputs": aggregate_run_outputs,
            },
        )

    print(f"[vidor-eval] Outputs: {output_dir}")
    for spec in variant_specs:
        variant_slug = str(spec['slug'])
        utility_outputs = dict((finalize_results.get(variant_slug) or {}).get("utility_outputs") or {})
        print(f"[vidor-eval] Variant {variant_slug}: {spec['method_name']}")
        print(f"[vidor-eval]   Verification utility CSV: {utility_outputs.get('csv', '')}")
        print(f"[vidor-eval]   Verification utility JSON: {utility_outputs.get('json', '')}")
        print(f"[vidor-eval]   Verification utility evidence: {utility_outputs.get('evidence_md', '')}")
        print(f"[vidor-eval]   Verification utility table: {utility_outputs.get('table_tex', '')}")
    if aggregate_run_outputs:
        print(f"[vidor-eval] Aggregate utility CSV: {aggregate_run_outputs.get('csv', '')}")
        print(f"[vidor-eval] Aggregate utility JSON: {aggregate_run_outputs.get('json', '')}")
        print(f"[vidor-eval] Aggregate utility Markdown: {aggregate_run_outputs.get('md', '')}")
        print(f"[vidor-eval] Aggregate utility table: {aggregate_run_outputs.get('tex', '')}")


if __name__ == "__main__":
    main()
