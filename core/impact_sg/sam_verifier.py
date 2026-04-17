from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _bbox(node: Dict[str, Any]) -> Tuple[int, int, int, int]:
    row = list(node.get("bbox") or [0, 0, 0, 0])
    if len(row) < 4:
        return 0, 0, 0, 0
    return int(row[0]), int(row[1]), max(0, int(row[2])), max(0, int(row[3]))


def _mask_pixels(node: Dict[str, Any]) -> List[List[int]]:
    mask = node.get("mask")
    if not isinstance(mask, dict):
        return []
    pixels = mask.get("pixels")
    if not isinstance(pixels, list):
        return []
    out: List[List[int]] = []
    for item in pixels:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                out.append([int(item[0]), int(item[1])])
            except Exception:
                continue
    return out


def verify_sam_mask(
    node: Dict[str, Any],
    *,
    frame_size: Tuple[int, int] = (1, 1),
    main_actor_area_ratio: float = 0.015,
) -> Dict[str, Any]:
    """
    Lightweight SAM plausibility checks used by STAGE validation.
    """
    fw = max(1, int(frame_size[0] or 1))
    fh = max(1, int(frame_size[1] or 1))
    frame_area = float(fw * fh)
    x, y, w, h = _bbox(node)
    bbox_area = float(max(0, w) * max(0, h))
    bbox_area_ratio = bbox_area / max(1.0, frame_area)
    pixels = _mask_pixels(node)
    mask_area = float(len(pixels))
    mask_area_ratio = mask_area / max(1.0, frame_area)

    reasons: List[str] = []
    if bbox_area <= 0:
        reasons.append("missing_bbox")
    if bbox_area_ratio < 0.0002:
        reasons.append("too_small_for_reliable_object")
    if bbox_area_ratio > 0.75:
        reasons.append("bbox_too_large_may_cover_background")
    if pixels and bbox_area > 0:
        fill_ratio = mask_area / max(1.0, bbox_area)
        if fill_ratio < 0.04:
            reasons.append("mask_box_consistency_weak")
        if fill_ratio > 1.05:
            reasons.append("mask_exceeds_bbox_area")
    if not pixels:
        reasons.append("mask_missing_or_empty")

    label = str(node.get("canonical_label", "") or "").strip().lower()
    is_person = label == "person"
    foreground_importance = 0.0
    if is_person:
        foreground_importance = min(1.0, bbox_area_ratio / max(1e-6, main_actor_area_ratio))
        if bbox_area_ratio < 0.0025:
            reasons.append("person_too_small_for_main_actor")
    else:
        foreground_importance = min(1.0, bbox_area_ratio / 0.05)

    score = 1.0
    for _ in reasons:
        score -= 0.12
    score = max(0.0, min(1.0, score))

    return {
        "mask_confidence": float(max(0.0, min(1.0, float(node.get("score", 0.0) or 0.0)))),
        "bbox_area_ratio": round(float(bbox_area_ratio), 8),
        "mask_area_ratio": round(float(mask_area_ratio), 8),
        "foreground_importance": round(float(foreground_importance), 6),
        "plausibility_score": round(float(score), 6),
        "reasons": reasons,
    }


def detect_main_actor_missing(nodes: List[Dict[str, Any]], *, area_ratio_threshold: float = 0.01) -> Dict[str, Any]:
    person_nodes = [n for n in nodes if str(n.get("canonical_label", "") or "").strip().lower() == "person"]
    if not person_nodes:
        return {
            "likely_missing_main_actor": True,
            "reason": "no_person_detected",
        }
    max_ratio = 0.0
    for node in person_nodes:
        ratio = float((node.get("mask_area_ratio", node.get("bbox_area_ratio", 0.0))) or 0.0)
        if ratio > max_ratio:
            max_ratio = ratio
    return {
        "likely_missing_main_actor": bool(max_ratio < float(area_ratio_threshold)),
        "max_person_area_ratio": float(max_ratio),
        "reason": "persons_too_small_or_far" if max_ratio < float(area_ratio_threshold) else "ok",
    }
