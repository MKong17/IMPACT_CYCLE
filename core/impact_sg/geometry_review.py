from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

from .claim_graph import existence_claim_id, label_claim_id
from .cycle_types import Claim
from .mask_ops import bbox_is_valid
from .scene_graph_builder import build_scene_graph


def _safe_bbox(value: Sequence[object]) -> List[int]:
    raw = list(value[:4]) if isinstance(value, (list, tuple)) else [0, 0, 0, 0]
    out: List[int] = []
    for item in raw:
        try:
            out.append(int(round(float(item or 0))))
        except Exception:
            out.append(0)
    while len(out) < 4:
        out.append(0)
    out[0] = max(0, out[0])
    out[1] = max(0, out[1])
    out[2] = max(1, out[2])
    out[3] = max(1, out[3])
    return out[:4]


def _bbox_center(bbox: Sequence[object]) -> Tuple[float, float]:
    x, y, w, h = [float(v or 0.0) for v in list(bbox[:4])]
    return x + (w / 2.0), y + (h / 2.0)


def _move_box(bbox: Sequence[object], *, x: Optional[float] = None, y: Optional[float] = None) -> List[int]:
    src = _safe_bbox(bbox)
    out = list(src)
    if x is not None:
        out[0] = max(0, int(round(float(x))))
    if y is not None:
        out[1] = max(0, int(round(float(y))))
    return out


def _scale_box(bbox: Sequence[object], *, x: Optional[float] = None, y: Optional[float] = None, w: Optional[float] = None, h: Optional[float] = None) -> List[int]:
    src = _safe_bbox(bbox)
    out = list(src)
    if x is not None:
        out[0] = max(0, int(round(float(x))))
    if y is not None:
        out[1] = max(0, int(round(float(y))))
    if w is not None:
        out[2] = max(1, int(round(float(w))))
    if h is not None:
        out[3] = max(1, int(round(float(h))))
    return out


def _relation_present(
    *,
    subject_bbox: Sequence[object],
    object_bbox: Sequence[object],
    relation: str,
) -> bool:
    # Use a person-anchored synthetic pair so scene_graph_builder's
    # person-edge gating does not suppress spatial relation checks.
    subject = {
        "entity_id": "subject",
        "canonical_label": "person",
        "prompt_used": "subject",
        "mask": {"pixels": []},
        "bbox": _safe_bbox(subject_bbox),
        "score": 1.0,
        "attributes": [],
        "provenance": [],
        "risk": 0.0,
        "verified": False,
        "validator_flags": [],
    }
    obj = {
        "entity_id": "object",
        "canonical_label": "object",
        "prompt_used": "object",
        "mask": {"pixels": []},
        "bbox": _safe_bbox(object_bbox),
        "score": 1.0,
        "attributes": [],
        "provenance": [],
        "risk": 0.0,
        "verified": False,
        "validator_flags": [],
    }
    graph = build_scene_graph(
        image_id="geometry_probe",
        proposals=[subject, obj],
        relation_vocab={"spatial": [str(relation or "").strip()], "interaction": []},
        touching_iou_epsilon=0.02,
        pairwise_max=16,
        enable_interaction_relations=False,
    )
    for edge in graph.get("edges") or []:
        if (
            str(edge.get("src_id", "") or "") == "subject"
            and str(edge.get("dst_id", "") or "") == "object"
            and str(edge.get("relation", "") or "").strip() == str(relation or "").strip()
        ):
            return True
    return False


def _candidate_score(
    *,
    current_bbox: Sequence[object],
    candidate_bbox: Sequence[object],
    other_bbox: Sequence[object],
    relation: str,
    moving_subject: bool,
) -> float:
    current = _safe_bbox(current_bbox)
    candidate = _safe_bbox(candidate_bbox)
    other = _safe_bbox(other_bbox)
    if not bbox_is_valid(candidate):
        return 0.0
    if moving_subject:
        satisfied = _relation_present(subject_bbox=candidate, object_bbox=other, relation=relation)
    else:
        satisfied = _relation_present(subject_bbox=other, object_bbox=candidate, relation=relation)
    base = 0.9 if satisfied else 0.3
    cur_cx, cur_cy = _bbox_center(current)
    cand_cx, cand_cy = _bbox_center(candidate)
    denom = max(16.0, float(max(current[2], current[3], other[2], other[3])))
    displacement = ((cand_cx - cur_cx) ** 2 + (cand_cy - cur_cy) ** 2) ** 0.5 / denom
    penalty = min(0.35, 0.08 * float(displacement))
    return max(0.0, min(1.0, base - penalty))


def _relation_candidates(
    *,
    target_bbox: Sequence[object],
    anchor_bbox: Sequence[object],
    relation: str,
    moving_subject: bool,
) -> List[Tuple[str, List[int], str]]:
    target = _safe_bbox(target_bbox)
    anchor = _safe_bbox(anchor_bbox)
    tx, ty, tw, th = target
    ax, ay, aw, ah = anchor
    margin = max(2, int(round(min(tw, th, aw, ah) * 0.15)))
    pad = max(2, int(round(min(aw, ah) * 0.08)))
    candidates: List[Tuple[str, List[int], str]] = []

    if relation == "left_of":
        if moving_subject:
            candidates.append(("Move left of counterpart", _move_box(target, x=ax - tw - margin), "Shift the selected box left so it clearly stays left of the counterpart."))
            candidates.append(("Move left and align vertically", _move_box(target, x=ax - tw - margin, y=ay + max(0, (ah - th) / 2.0)), "Shift left and center it vertically against the counterpart."))
        else:
            candidates.append(("Move right of subject", _move_box(target, x=ax + aw + margin), "Shift the selected box right so the subject remains left of it."))
            candidates.append(("Move right and align vertically", _move_box(target, x=ax + aw + margin, y=ay + max(0, (ah - th) / 2.0)), "Shift right and center it vertically against the subject."))
    elif relation == "right_of":
        if moving_subject:
            candidates.append(("Move right of counterpart", _move_box(target, x=ax + aw + margin), "Shift the selected box right so it clearly stays right of the counterpart."))
            candidates.append(("Move right and align vertically", _move_box(target, x=ax + aw + margin, y=ay + max(0, (ah - th) / 2.0)), "Shift right and center it vertically against the counterpart."))
        else:
            candidates.append(("Move left of subject", _move_box(target, x=ax - tw - margin), "Shift the selected box left so the subject remains right of it."))
            candidates.append(("Move left and align vertically", _move_box(target, x=ax - tw - margin, y=ay + max(0, (ah - th) / 2.0)), "Shift left and center it vertically against the subject."))
    elif relation == "above":
        if moving_subject:
            candidates.append(("Move above counterpart", _move_box(target, y=ay - th - margin), "Shift the selected box above the counterpart."))
            candidates.append(("Move above and align horizontally", _move_box(target, x=ax + max(0, (aw - tw) / 2.0), y=ay - th - margin), "Shift above and center it horizontally against the counterpart."))
        else:
            candidates.append(("Move below subject", _move_box(target, y=ay + ah + margin), "Shift the selected box below so the subject remains above it."))
            candidates.append(("Move below and align horizontally", _move_box(target, x=ax + max(0, (aw - tw) / 2.0), y=ay + ah + margin), "Shift below and center it horizontally against the subject."))
    elif relation == "below":
        if moving_subject:
            candidates.append(("Move below counterpart", _move_box(target, y=ay + ah + margin), "Shift the selected box below the counterpart."))
            candidates.append(("Move below and align horizontally", _move_box(target, x=ax + max(0, (aw - tw) / 2.0), y=ay + ah + margin), "Shift below and center it horizontally against the counterpart."))
        else:
            candidates.append(("Move above subject", _move_box(target, y=ay - th - margin), "Shift the selected box above so the subject remains below it."))
            candidates.append(("Move above and align horizontally", _move_box(target, x=ax + max(0, (aw - tw) / 2.0), y=ay - th - margin), "Shift above and center it horizontally against the subject."))
    elif relation == "inside":
        if moving_subject:
            new_w = min(tw, max(1, aw - (2 * pad)))
            new_h = min(th, max(1, ah - (2 * pad)))
            candidates.append(("Place inside counterpart", _scale_box(target, x=ax + pad, y=ay + pad, w=new_w, h=new_h), "Move and shrink the selected box so it fits inside the counterpart."))
            candidates.append(("Center inside counterpart", _scale_box(target, x=ax + max(0, (aw - new_w) / 2.0), y=ay + max(0, (ah - new_h) / 2.0), w=new_w, h=new_h), "Center the selected box inside the counterpart while preserving a smaller footprint."))
        else:
            x = min(ax, tx) - pad
            y = min(ay, ty) - pad
            w = max(aw + (ax - x), tx + tw - x) + pad
            h = max(ah + (ay - y), ty + th - y) + pad
            candidates.append(("Expand around subject", _scale_box(target, x=x, y=y, w=w, h=h), "Expand the selected box so the subject fits inside it."))
    elif relation == "surrounding":
        if moving_subject:
            x = min(ax, tx) - pad
            y = min(ay, ty) - pad
            w = max(aw + (ax - x), tx + tw - x) + pad
            h = max(ah + (ay - y), ty + th - y) + pad
            candidates.append(("Expand around counterpart", _scale_box(target, x=x, y=y, w=w, h=h), "Expand the selected box so it surrounds the counterpart."))
        else:
            new_w = min(tw, max(1, aw - (2 * pad)))
            new_h = min(th, max(1, ah - (2 * pad)))
            candidates.append(("Place inside subject", _scale_box(target, x=ax + pad, y=ay + pad, w=new_w, h=new_h), "Move and shrink the selected box so it stays inside the subject."))
    elif relation == "touching":
        tcx, tcy = _bbox_center(target)
        acx, acy = _bbox_center(anchor)
        if abs(tcx - acx) >= abs(tcy - acy):
            if tcx <= acx:
                x = ax - tw
            else:
                x = ax + aw
            candidates.append(("Touch on horizontal edge", _move_box(target, x=x), "Shift the selected box to touch the counterpart along the nearest horizontal edge."))
        else:
            if tcy <= acy:
                y = ay - th
            else:
                y = ay + ah
            candidates.append(("Touch on vertical edge", _move_box(target, y=y), "Shift the selected box to touch the counterpart along the nearest vertical edge."))
        candidates.append(("Touch and center", _move_box(target, x=ax + max(0, (aw - tw) / 2.0), y=ay + max(0, (ah - th) / 2.0)), "Center the selected box on the counterpart to maximize contact."))
    elif relation in {"overlap", "intersect"}:
        candidates.append(("Center overlap", _move_box(target, x=ax + max(0, (aw - tw) / 2.0), y=ay + max(0, (ah - th) / 2.0)), "Move the selected box so it overlaps the counterpart at the center."))
        candidates.append(("Partial overlap", _move_box(target, x=ax + max(0, aw * 0.25 - tw * 0.25), y=ay + max(0, ah * 0.25 - th * 0.25)), "Move the selected box into a partial overlap that preserves both extents."))
    return candidates


def _dedupe_candidates(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen = set()
    for row in candidates:
        bbox_key = tuple(_safe_bbox(row.get("bbox") or [0, 0, 0, 0]))
        key = (str(row.get("target_node_id", "") or ""), bbox_key)
        if key in seen:
            continue
        seen.add(key)
        item = dict(row)
        item["bbox"] = list(bbox_key)
        out.append(item)
    return out


def _node_confidence(
    node: Dict[str, object],
    claims: Dict[str, Claim],
) -> float:
    node_id = str(node.get("entity_id", "") or "").strip()
    if not node_id:
        return 0.0
    base = float(node.get("score", 0.0) or 0.0)
    label_claim = claims.get(label_claim_id(node_id))
    if label_claim is not None:
        base += 0.2 * float(label_claim.support_ratio)
        base -= 0.2 * float(label_claim.conflict_ratio)
    existence_claim = claims.get(existence_claim_id(node_id))
    if existence_claim is not None:
        base += 0.1 * float(existence_claim.support_ratio)
        base -= 0.15 * float(existence_claim.conflict_ratio)
    if bool(node.get("verified", False)):
        base += 0.05
    return max(0.0, min(1.0, base))


def _choose_geometry_target(
    *,
    subject_node: Dict[str, object],
    object_node: Dict[str, object],
    claims: Dict[str, Claim],
    preferred_anchor_label: str,
) -> Tuple[str, Dict[str, object], Dict[str, object]]:
    subject_label = str(subject_node.get("canonical_label", "") or "").strip().lower()
    object_label = str(object_node.get("canonical_label", "") or "").strip().lower()
    anchor = str(preferred_anchor_label or "").strip().lower()
    if anchor and subject_label == anchor and object_label != anchor:
        return "object", object_node, subject_node
    if anchor and object_label == anchor and subject_label != anchor:
        return "subject", subject_node, object_node
    subject_conf = _node_confidence(subject_node, claims)
    object_conf = _node_confidence(object_node, claims)
    if subject_conf <= object_conf:
        return "subject", subject_node, object_node
    return "object", object_node, subject_node


def build_geometry_review_queue(
    graph: Dict[str, object],
    claims: Dict[str, Claim],
    *,
    relation_vocab: Dict[str, List[str]],
    preferred_anchor_label: str = "person",
    conflict_threshold: float = 0.60,
    max_items: int = 2,
) -> List[Dict[str, object]]:
    spatial_vocab = {str(x).strip() for x in list((relation_vocab or {}).get("spatial") or []) if str(x).strip()}
    if not spatial_vocab:
        return []
    node_by_id = {
        str(node.get("entity_id", "") or "").strip(): node
        for node in graph.get("nodes") or []
        if str(node.get("entity_id", "") or "").strip()
    }
    queue: List[Dict[str, object]] = []
    for claim in claims.values():
        relation = str(claim.predicate or "").strip()
        if claim.claim_type != "relation" or relation not in spatial_vocab:
            continue
        if float(claim.conflict_ratio) < float(conflict_threshold):
            continue
        subject_node = node_by_id.get(str(claim.subject_id or "").strip())
        object_node = node_by_id.get(str(claim.object_id or "").strip())
        if subject_node is None or object_node is None:
            continue
        subject_bbox = list(subject_node.get("bbox") or [0, 0, 0, 0])
        object_bbox = list(object_node.get("bbox") or [0, 0, 0, 0])
        if not bbox_is_valid(subject_bbox) or not bbox_is_valid(object_bbox):
            continue
        target_role, target_node, anchor_node = _choose_geometry_target(
            subject_node=subject_node,
            object_node=object_node,
            claims=claims,
            preferred_anchor_label=preferred_anchor_label,
        )
        target_bbox = list(target_node.get("bbox") or [0, 0, 0, 0])
        anchor_bbox = list(anchor_node.get("bbox") or [0, 0, 0, 0])
        moving_subject = target_role == "subject"
        candidates: List[Dict[str, object]] = [
            {
                "value": "keep_current",
                "label": "Keep current box",
                "description": "Leave the current box unchanged and keep existing geometry.",
                "target_node_id": str(target_node.get("entity_id", "") or ""),
                "bbox": _safe_bbox(target_bbox),
                "clear_mask": False,
                "score": _candidate_score(
                    current_bbox=target_bbox,
                    candidate_bbox=target_bbox,
                    other_bbox=anchor_bbox,
                    relation=relation,
                    moving_subject=moving_subject,
                ),
            }
        ]
        for idx, (label, bbox, description) in enumerate(
            _relation_candidates(
                target_bbox=target_bbox,
                anchor_bbox=anchor_bbox,
                relation=relation,
                moving_subject=moving_subject,
            ),
            start=1,
        ):
            candidates.append(
                {
                    "value": f"geom_{str(target_node.get('entity_id', '') or 'node')}_{idx}",
                    "label": label,
                    "description": description,
                    "target_node_id": str(target_node.get("entity_id", "") or ""),
                    "bbox": _safe_bbox(bbox),
                    "clear_mask": True,
                    "score": _candidate_score(
                        current_bbox=target_bbox,
                        candidate_bbox=bbox,
                        other_bbox=anchor_bbox,
                        relation=relation,
                        moving_subject=moving_subject,
                    ),
                }
            )
        candidates = _dedupe_candidates(candidates)
        ranked = sorted(candidates, key=lambda row: (-float(row.get("score", 0.0) or 0.0), str(row.get("value", "") or "")))
        current_score = next(
            (float(row.get("score", 0.0) or 0.0) for row in ranked if str(row.get("value", "") or "") == "keep_current"),
            0.0,
        )
        best_alt = next(
            (row for row in ranked if str(row.get("value", "") or "") != "keep_current"),
            None,
        )
        if best_alt is None:
            continue
        best_alt_score = float(best_alt.get("score", 0.0) or 0.0)
        if best_alt_score < max(0.62, current_score + 0.08):
            continue
        target_node_id = str(target_node.get("entity_id", "") or "")
        source_relation_claim_id = str(claim.claim_id or "")
        question = (
            f"Which bounding box best restores the spatial relation '{relation}' for {target_node_id} "
            f"relative to {str(anchor_node.get('entity_id', '') or '')}?"
        )
        queue.append(
            {
                "claim_id": f"claim_bbox_{target_node_id}_{source_relation_claim_id}",
                "claim_type": "bbox",
                "priority": min(1.0, (0.55 * float(claim.conflict_ratio)) + (0.35 * best_alt_score) + 0.10),
                "question": question,
                "subject_id": target_node_id,
                "object_id": str(anchor_node.get("entity_id", "") or ""),
                "predicate": relation,
                "value": ",".join([str(v) for v in _safe_bbox(target_bbox)]),
                "target_node_id": target_node_id,
                "target_role": target_role,
                "source_relation_claim_id": source_relation_claim_id,
                "source_relation_edge_id": str((claim.evidence_edge_ids or [""])[0] or ""),
                "question_options": [str(row.get("value", "") or "") for row in ranked],
                "resolution_options": ranked,
                "geometry_candidates": ranked,
                "suggested_value": str(best_alt.get("value", "") or ""),
                "suggested_score": best_alt_score,
                "claim_row": {
                    "claim_id": f"claim_bbox_{target_node_id}_{source_relation_claim_id}",
                    "claim_type": "bbox",
                    "subject_id": target_node_id,
                    "object_id": str(anchor_node.get("entity_id", "") or ""),
                    "predicate": "bbox",
                    "value": ",".join([str(v) for v in _safe_bbox(target_bbox)]),
                    "evidence_node_ids": [target_node_id, str(anchor_node.get("entity_id", "") or "")],
                    "provenance": [
                        {
                            "source": "cycle_geometry_review",
                            "relation_claim_id": source_relation_claim_id,
                            "relation": relation,
                        }
                    ],
                },
            }
        )
    queue.sort(key=lambda row: (-float(row.get("priority", 0.0) or 0.0), str(row.get("claim_id", "") or "")))
    return queue[: max(0, int(max_items or 0))]


def rebuild_spatial_edges(
    graph: Dict[str, object],
    *,
    relation_vocab: Dict[str, List[str]],
    touching_iou_epsilon: float = 0.02,
    pairwise_max: int = 200,
) -> Dict[str, object]:
    out = copy.deepcopy(graph or {})
    nodes = list(out.get("nodes") or [])
    proposals: List[Dict[str, object]] = []
    for node in nodes:
        proposals.append(
            {
                "entity_id": str(node.get("entity_id", "") or ""),
                "canonical_label": str(node.get("canonical_label", "") or ""),
                "prompt_used": str(node.get("prompt_used", node.get("canonical_label", "")) or ""),
                "mask": dict(node.get("mask") or {"pixels": []}),
                "bbox": list(node.get("bbox") or [0, 0, 0, 0]),
                "score": float(node.get("score", 0.0) or 0.0),
                "attributes": list(node.get("attributes") or []),
                "provenance": list(node.get("provenance") or []),
                "risk": float(node.get("risk", 0.0) or 0.0),
                "verified": bool(node.get("verified", False)),
                "validator_flags": list(node.get("validator_flags") or []),
            }
        )
    rebuilt = build_scene_graph(
        image_id=str(out.get("image_id", "") or "geometry_rebuild"),
        proposals=proposals,
        relation_vocab=relation_vocab,
        touching_iou_epsilon=float(touching_iou_epsilon),
        pairwise_max=int(pairwise_max),
        enable_interaction_relations=False,
    )
    allowed_spatial = {str(x).strip() for x in list((relation_vocab or {}).get("spatial") or []) if str(x).strip()}
    preserved_edges = [
        dict(edge)
        for edge in list(out.get("edges") or [])
        if str(edge.get("relation", "") or "").strip() not in allowed_spatial
    ]
    out["edges"] = list(rebuilt.get("edges") or []) + preserved_edges
    metadata = dict(out.get("metadata") or {})
    history = list(metadata.get("geometry_rebuild_history") or [])
    history.append(
        {
            "spatial_edge_count": len(list(rebuilt.get("edges") or [])),
            "preserved_non_spatial_edge_count": len(preserved_edges),
        }
    )
    metadata["geometry_rebuild_history"] = history
    out["metadata"] = metadata
    return out
