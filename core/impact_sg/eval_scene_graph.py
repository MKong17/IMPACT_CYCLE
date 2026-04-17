from __future__ import annotations

from typing import Dict, List, Tuple

from .mask_ops import mask_or_bbox_iou


def _f1(tp: int, fp: int, fn: int) -> float:
    p = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    r = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    if p + r <= 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def evaluate_scene_graph(pred: Dict[str, object], gt: Dict[str, object], *, iou_threshold: float = 0.5) -> Dict[str, float]:
    pred_nodes = pred.get("nodes") or []
    gt_nodes = gt.get("nodes") or []
    pred_edges = pred.get("edges") or []
    gt_edges = gt.get("edges") or []

    matched_gt = set()
    tp_label = 0
    tp_mask = 0
    tp_bbox = 0
    attr_tp = 0
    attr_fp = 0
    attr_fn = 0

    for p in pred_nodes:
        best_idx = None
        best_iou = 0.0
        for idx, g in enumerate(gt_nodes):
            if idx in matched_gt:
                continue
            if str(p.get("canonical_label")) != str(g.get("canonical_label")):
                continue
            iou = mask_or_bbox_iou(
                p.get("mask") or {},
                g.get("mask") or {},
                bbox_a=p.get("bbox") or [0, 0, 0, 0],
                bbox_b=g.get("bbox") or [0, 0, 0, 0],
            )
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None:
            matched_gt.add(best_idx)
            tp_label += 1
            if best_iou >= iou_threshold:
                tp_mask += 1
                tp_bbox += 1

            p_attrs = {(str(a.get("slot")), str(a.get("value"))) for a in (p.get("attributes") or []) if isinstance(a, dict)}
            g_attrs = {(str(a.get("slot")), str(a.get("value"))) for a in (gt_nodes[best_idx].get("attributes") or []) if isinstance(a, dict)}
            attr_tp += len(p_attrs.intersection(g_attrs))
            attr_fp += len(p_attrs - g_attrs)
            attr_fn += len(g_attrs - p_attrs)

    fp_label = max(0, len(pred_nodes) - tp_label)
    fn_label = max(0, len(gt_nodes) - tp_label)

    pred_edge_set = {(str(e.get("src_id")), str(e.get("relation")), str(e.get("dst_id"))) for e in pred_edges}
    gt_edge_set = {(str(e.get("src_id")), str(e.get("relation")), str(e.get("dst_id"))) for e in gt_edges}
    rel_tp = len(pred_edge_set.intersection(gt_edge_set))
    rel_fp = len(pred_edge_set - gt_edge_set)
    rel_fn = len(gt_edge_set - pred_edge_set)

    ged = float(fp_label + fn_label + rel_fp + rel_fn)

    return {
        "entity_label_accuracy": float(tp_label) / float(max(1, len(gt_nodes))),
        "mask_iou_accuracy": float(tp_mask) / float(max(1, len(gt_nodes))),
        "bbox_iou_accuracy": float(tp_bbox) / float(max(1, len(gt_nodes))),
        "attribute_f1": _f1(attr_tp, attr_fp, attr_fn),
        "relation_f1": _f1(rel_tp, rel_fp, rel_fn),
        "graph_edit_distance": ged,
        "human_correction_time": 0.0,
        "edits_per_image": float(fp_label + rel_fp),
    }
