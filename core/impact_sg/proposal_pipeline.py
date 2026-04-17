from __future__ import annotations

from typing import Dict, List, Tuple

from .mask_ops import bbox_is_valid, mask_or_bbox_area, mask_or_bbox_iou


HUMAN_HINTS = {
    "person",
    "player",
    "football player",
    "goalkeeper",
    "referee",
    "athlete",
    "man",
    "woman",
    "human",
}
STATIC_BACKGROUND_HINTS = {
    "stairs",
    "stand",
    "stands",
    "crowd",
    "stadium",
    "ad board",
    "billboard",
    "field marking",
    "bleacher",
}


def _label_text(proposal: Dict[str, object]) -> str:
    return str(proposal.get("canonical_label", "") or proposal.get("label", "") or "").strip().lower()


def _prompt_text(proposal: Dict[str, object]) -> str:
    return str(proposal.get("prompt_used", "") or "").strip().lower()


def is_human_like(proposal: Dict[str, object]) -> bool:
    text = " ".join([_label_text(proposal), _prompt_text(proposal)]).strip().lower()
    return any(token in text for token in HUMAN_HINTS)


def is_static_background_like(proposal: Dict[str, object]) -> bool:
    text = " ".join([_label_text(proposal), _prompt_text(proposal)]).strip().lower()
    return any(token in text for token in STATIC_BACKGROUND_HINTS)


def _foreground_bias(bbox: List[int], image_wh: Tuple[int, int]) -> float:
    if not bbox_is_valid(bbox):
        return 0.0
    w_img, h_img = image_wh
    w_img = max(1, int(w_img))
    h_img = max(1, int(h_img))
    cx = float(bbox[0]) + (float(bbox[2]) / 2.0)
    cy = float(bbox[1]) + (float(bbox[3]) / 2.0)
    x_norm = cx / float(w_img)
    y_norm = cy / float(h_img)
    center_bias = max(0.0, 1.0 - min(1.0, abs(x_norm - 0.5) / 0.5))
    lower_mid_bias = max(0.0, 1.0 - min(1.0, abs(y_norm - 0.68) / 0.68))
    return (0.45 * center_bias) + (0.55 * lower_mid_bias)


def _area_ratio(mask: Dict[str, object], bbox: List[int], image_wh: Tuple[int, int]) -> float:
    w_img, h_img = image_wh
    img_area = max(1, int(w_img) * int(h_img))
    return float(mask_or_bbox_area(mask, bbox)) / float(img_area)


def _risk_low_conf(score: float, threshold: float) -> float:
    s = float(score)
    t = max(1e-6, float(threshold))
    if s >= t:
        return 0.0
    return min(1.0, (t - s) / t)


def _risk_small_thin(mask: Dict[str, object], bbox: List[int], image_wh: Tuple[int, int], small_area_ratio: float, thin_min_dim: int) -> float:
    w_img, h_img = image_wh
    img_area = max(1, int(w_img) * int(h_img))
    area = mask_or_bbox_area(mask, bbox)
    area_ratio = float(area) / float(img_area)
    bw = int(bbox[2]) if len(bbox) >= 3 else 0
    bh = int(bbox[3]) if len(bbox) >= 4 else 0
    thin = bw <= int(thin_min_dim) or bh <= int(thin_min_dim)
    small = area_ratio <= float(small_area_ratio)
    if small and thin:
        return 1.0
    if small or thin:
        return 0.6
    return 0.0


def _merge_records(base: Dict[str, object], other: Dict[str, object]) -> Dict[str, object]:
    merged = dict(base)
    merged["score"] = max(float(base.get("score", 0.0)), float(other.get("score", 0.0)))
    provenance = list(base.get("provenance") or [])
    provenance.extend(list(other.get("provenance") or []))
    merged["provenance"] = provenance
    merged_from = list(base.get("merged_from") or [])
    if not merged_from:
        merged_from.append(
            {
                "prompt_used": base.get("prompt_used", ""),
                "score": float(base.get("score", 0.0)),
            }
        )
    merged_from.append(
        {
            "prompt_used": other.get("prompt_used", ""),
            "score": float(other.get("score", 0.0)),
        }
    )
    merged["merged_from"] = merged_from
    return merged


def merge_entity_proposals(
    records: List[Dict[str, object]],
    merge_mask_iou_threshold: float,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for rec in records:
        label = str(rec.get("canonical_label", "")).strip().lower()
        mask = rec.get("mask")
        bbox = rec.get("bbox") or [0, 0, 0, 0]
        if not label:
            continue
        if not isinstance(mask, dict) and not bbox_is_valid(bbox):
            continue
        merged = False
        for idx, kept in enumerate(out):
            if str(kept.get("canonical_label", "")).strip().lower() != label:
                continue
            iou = mask_or_bbox_iou(
                mask if isinstance(mask, dict) else {},
                kept.get("mask") or {},
                bbox_a=bbox,
                bbox_b=kept.get("bbox") or [0, 0, 0, 0],
            )
            if iou >= float(merge_mask_iou_threshold):
                out[idx] = _merge_records(kept, rec)
                merged = True
                break
        if not merged:
            out.append(rec)
    return out


def score_proposal_risk(
    proposal: Dict[str, object],
    *,
    image_wh: Tuple[int, int],
    overlap_conflict: float,
    ontology_ambiguity: float,
    risk_weights: Dict[str, float],
    low_confidence_threshold: float,
    small_area_ratio: float,
    thin_min_dim: int,
) -> float:
    score = float(proposal.get("score", 0.0))
    mask = proposal.get("mask") or {}
    bbox = proposal.get("bbox") or [0, 0, 0, 0]

    low_conf = _risk_low_conf(score, threshold=low_confidence_threshold)
    small_thin = _risk_small_thin(mask, bbox, image_wh, small_area_ratio, thin_min_dim)

    w_low = float(risk_weights.get("low_confidence", 0.4))
    w_ov = float(risk_weights.get("overlap_conflict", 0.3))
    w_st = float(risk_weights.get("small_thin_prior", 0.2))
    w_amb = float(risk_weights.get("ontology_ambiguity", 0.1))

    risk = (w_low * low_conf) + (w_ov * float(overlap_conflict)) + (w_st * small_thin) + (w_amb * float(ontology_ambiguity))
    return max(0.0, min(1.0, risk))


def build_entity_proposals(
    raw_records: List[Dict[str, object]],
    *,
    merge_mask_iou_threshold: float,
    image_wh: Tuple[int, int],
    risk_weights: Dict[str, float],
    low_confidence_threshold: float,
    small_area_ratio: float,
    thin_min_dim: int,
) -> List[Dict[str, object]]:
    merged = merge_entity_proposals(raw_records, merge_mask_iou_threshold=merge_mask_iou_threshold)

    out: List[Dict[str, object]] = []
    for idx, rec in enumerate(merged):
        max_same_overlap = 0.0
        for jdx, other in enumerate(merged):
            if idx == jdx:
                continue
            if str(other.get("canonical_label", "")).strip().lower() != str(rec.get("canonical_label", "")).strip().lower():
                continue
            iou = mask_or_bbox_iou(
                rec.get("mask") or {},
                other.get("mask") or {},
                bbox_a=rec.get("bbox") or [0, 0, 0, 0],
                bbox_b=other.get("bbox") or [0, 0, 0, 0],
            )
            if iou > max_same_overlap:
                max_same_overlap = iou

        ambiguity = 1.0 if bool(rec.get("ontology_ambiguous", False)) else 0.0
        rec2 = dict(rec)
        rec2["debug_metrics"] = {
            "max_same_label_overlap": float(max_same_overlap),
            "is_human_like": bool(is_human_like(rec2)),
            "is_static_background_like": bool(is_static_background_like(rec2)),
            "foreground_bias": float(_foreground_bias(rec2.get("bbox") or [0, 0, 0, 0], image_wh)),
            "area_ratio": float(_area_ratio(rec2.get("mask") or {}, rec2.get("bbox") or [0, 0, 0, 0], image_wh)),
        }
        rec2["risk"] = score_proposal_risk(
            rec2,
            image_wh=image_wh,
            overlap_conflict=max_same_overlap,
            ontology_ambiguity=ambiguity,
            risk_weights=risk_weights,
            low_confidence_threshold=low_confidence_threshold,
            small_area_ratio=small_area_ratio,
            thin_min_dim=thin_min_dim,
        )
        out.append(rec2)
    return out
