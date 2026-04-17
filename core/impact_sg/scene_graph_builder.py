from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Tuple

from .mask_ops import bbox_from_mask_rle, bbox_is_valid, mask_or_bbox_iou, touches_or_bbox


def _edge_id() -> str:
    return f"edge_{uuid.uuid4().hex[:10]}"


def _node_id() -> str:
    return f"ent_{uuid.uuid4().hex[:10]}"


def _center(bbox: List[int]) -> Tuple[float, float]:
    x, y, w, h = [int(v) for v in (bbox or [0, 0, 0, 0])[:4]]
    return x + (w / 2.0), y + (h / 2.0)


def _inside(a: List[int], b: List[int]) -> bool:
    ax, ay, aw, ah = [int(v) for v in (a or [0, 0, 0, 0])[:4]]
    bx, by, bw, bh = [int(v) for v in (b or [0, 0, 0, 0])[:4]]
    return ax >= bx and ay >= by and (ax + aw) <= (bx + bw) and (ay + ah) <= (by + bh)


def _resolved_bbox(mask: Dict[str, object], bbox: List[int]) -> List[int]:
    mask_bbox = bbox_from_mask_rle(mask)
    if bbox_is_valid(mask_bbox):
        return mask_bbox
    if bbox_is_valid(bbox):
        return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    return [0, 0, 0, 0]


def _is_person_node(node: Dict[str, object]) -> bool:
    return str(node.get("canonical_label", "") or "").strip().lower() == "person"


def _spatial_relation_candidates(
    node_a: Dict[str, object],
    node_b: Dict[str, object],
    touching_iou_epsilon: float,
) -> List[Tuple[str, str, str]]:
    """Return ordered candidate triples: (src_id, relation, dst_id).

    Candidates are ordered by priority so callers can pick the first allowed
    relation and keep at most one spatial edge per object pair.
    """
    rels: List[Tuple[str, str, str]] = []
    bbox_a = node_a.get("bbox") or [0, 0, 0, 0]
    bbox_b = node_b.get("bbox") or [0, 0, 0, 0]
    cx_a, cy_a = _center(bbox_a)
    cx_b, cy_b = _center(bbox_b)
    a_id = str(node_a.get("entity_id", "") or "")
    b_id = str(node_b.get("entity_id", "") or "")

    iou = mask_or_bbox_iou(
        node_a.get("mask") or {},
        node_b.get("mask") or {},
        bbox_a=bbox_a,
        bbox_b=bbox_b,
    )

    # Top priority: containment.
    if _inside(bbox_a, bbox_b):
        rels.append((a_id, "inside", b_id))
        rels.append((b_id, "surrounding", a_id))
    if _inside(bbox_b, bbox_a):
        rels.append((b_id, "inside", a_id))
        rels.append((a_id, "surrounding", b_id))

    # Medium priority: overlap/intersection/contact.
    if iou > 0.0:
        rels.append((a_id, "overlap", b_id))
        rels.append((a_id, "intersect", b_id))
    if iou <= float(touching_iou_epsilon) and touches_or_bbox(
        node_a.get("mask") or {},
        node_b.get("mask") or {},
        bbox_a=bbox_a,
        bbox_b=bbox_b,
    ):
        rels.append((a_id, "touching", b_id))

    # Lowest priority: directional relation from dominant axis only.
    dx = float(cx_b - cx_a)
    dy = float(cy_b - cy_a)
    if abs(dx) >= abs(dy):
        if dx >= 0.0:
            rels.append((a_id, "left_of", b_id))
            rels.append((b_id, "right_of", a_id))
        else:
            rels.append((b_id, "left_of", a_id))
            rels.append((a_id, "right_of", b_id))
    else:
        if dy >= 0.0:
            rels.append((a_id, "above", b_id))
            rels.append((b_id, "below", a_id))
        else:
            rels.append((b_id, "above", a_id))
            rels.append((a_id, "below", b_id))

    uniq: List[Tuple[str, str, str]] = []
    seen = set()
    for src, rel, dst in rels:
        key = (str(src), str(rel), str(dst))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq


def _pick_single_spatial_edge(
    node_a: Dict[str, object],
    node_b: Dict[str, object],
    *,
    touching_iou_epsilon: float,
    allowed_spatial: set[str],
) -> Optional[Tuple[str, str, str]]:
    for src_id, relation, dst_id in _spatial_relation_candidates(
        node_a,
        node_b,
        touching_iou_epsilon=touching_iou_epsilon,
    ):
        if relation in allowed_spatial:
            return src_id, relation, dst_id
    return None


def build_scene_graph(
    image_id: str,
    proposals: List[Dict[str, object]],
    *,
    relation_vocab: Dict[str, List[str]],
    touching_iou_epsilon: float,
    pairwise_max: int | None = None,
    enable_interaction_relations: bool = False,
    interaction_relation_hook=None,
) -> Dict[str, object]:
    nodes: List[Dict[str, object]] = []
    for p in proposals:
        mask = p.get("mask") or {"pixels": []}
        bbox = _resolved_bbox(mask, list(p.get("bbox") or [0, 0, 0, 0]))
        node = {
            "entity_id": str(p.get("entity_id") or _node_id()),
            "canonical_label": str(p.get("canonical_label", "")).strip().lower(),
            "prompt_used": p.get("prompt_used", ""),
            "mask": mask,
            "bbox": bbox,
            "score": float(p.get("score", 0.0)),
            "attributes": list(p.get("attributes") or []),
            "provenance": list(p.get("provenance") or []),
            "risk": float(p.get("risk", 0.0)),
            "verified": bool(p.get("verified", False)),
            "validator_flags": list(p.get("validator_flags") or []),
        }
        # Preserve optional LLM-side attribute payloads so downstream attribute
        # extraction can merge them into final node attributes.
        for k in ("llm_attributes", "qwen_attributes", "person_attributes"):
            if k in p:
                node[k] = p.get(k)
        nodes.append(node)

    edges: List[Dict[str, object]] = []
    allowed_spatial = set((relation_vocab or {}).get("spatial") or [])
    allowed_inter = set((relation_vocab or {}).get("interaction") or [])
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a = nodes[i]
            b = nodes[j]
            # Keep only person-person and person-other edges.
            if (not _is_person_node(a)) and (not _is_person_node(b)):
                continue
            selected = _pick_single_spatial_edge(
                a,
                b,
                touching_iou_epsilon=touching_iou_epsilon,
                allowed_spatial=allowed_spatial,
            )
            if not selected:
                continue
            src_id, rel, dst_id = selected
            src_node = a if str(a.get("entity_id", "") or "") == src_id else b
            dst_node = b if str(b.get("entity_id", "") or "") == dst_id else a
            edges.append(
                {
                    "edge_id": _edge_id(),
                    "src_id": src_id,
                    "relation": rel,
                    "dst_id": dst_id,
                    "score": 1.0,
                    "evidence": {
                        "type": "deterministic_spatial",
                        "src_bbox": list(src_node.get("bbox") or [0, 0, 0, 0]),
                        "dst_bbox": list(dst_node.get("bbox") or [0, 0, 0, 0]),
                    },
                    "validator_flags": [],
                    "risk": 0.0,
                    "verified": False,
                }
            )

    if enable_interaction_relations and callable(interaction_relation_hook):
        extra_edges = interaction_relation_hook(nodes)
        node_by_id = {str(n.get("entity_id", "") or ""): n for n in nodes}
        if isinstance(extra_edges, list):
            for edge in extra_edges:
                if not isinstance(edge, dict):
                    continue
                if str(edge.get("relation", "")) not in allowed_inter:
                    continue
                src_id = str(edge.get("src_id", "") or "")
                dst_id = str(edge.get("dst_id", "") or "")
                src_node = dict(node_by_id.get(src_id) or {})
                dst_node = dict(node_by_id.get(dst_id) or {})
                if (not _is_person_node(src_node)) and (not _is_person_node(dst_node)):
                    continue
                edge.setdefault("edge_id", _edge_id())
                edge.setdefault("validator_flags", [])
                edge.setdefault("risk", 0.0)
                edge.setdefault("verified", False)
                edges.append(edge)

    # Global cap for relation edges to avoid quadratic blowup on noisy proposals.
    if pairwise_max is not None:
        try:
            cap = max(1, int(pairwise_max))
            if len(edges) > cap:
                edges = edges[:cap]
        except Exception:
            pass

    return {
        "image_id": str(image_id),
        "nodes": nodes,
        "edges": edges,
    }
