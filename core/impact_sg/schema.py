from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .mask_ops import bbox_from_mask_rle, bbox_is_valid


@dataclass
class SchemaValidationResult:
    valid: bool
    errors: List[str]


REQUIRED_NODE_FIELDS = {
    "entity_id",
    "canonical_label",
    "prompt_used",
    "mask",
    "bbox",
    "score",
    "attributes",
    "provenance",
    "risk",
    "verified",
}

REQUIRED_EDGE_FIELDS = {
    "edge_id",
    "src_id",
    "relation",
    "dst_id",
    "score",
    "evidence",
    "validator_flags",
    "risk",
    "verified",
}


def derive_node_bbox(node: Dict[str, object]) -> List[int]:
    mask = node.get("mask") if isinstance(node, dict) else None
    if isinstance(mask, dict):
        bbox = bbox_from_mask_rle(mask)
        if bbox_is_valid(bbox):
            return bbox
    raw_bbox = node.get("bbox") if isinstance(node, dict) else None
    if bbox_is_valid(raw_bbox):
        return [int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3])]
    return [0, 0, 0, 0]


def validate_scene_graph_schema(graph: Dict[str, object]) -> SchemaValidationResult:
    errors: List[str] = []
    if not isinstance(graph, dict):
        return SchemaValidationResult(valid=False, errors=["Graph is not a dict"])

    for top in ("image_id", "nodes", "edges"):
        if top not in graph:
            errors.append(f"Missing top-level key: {top}")

    nodes = graph.get("nodes") or []
    if not isinstance(nodes, list):
        errors.append("nodes must be a list")
        nodes = []
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"nodes[{idx}] must be dict")
            continue
        missing = [k for k in REQUIRED_NODE_FIELDS if k not in node]
        if missing:
            errors.append(f"nodes[{idx}] missing fields: {missing}")

    edges = graph.get("edges") or []
    if not isinstance(edges, list):
        errors.append("edges must be a list")
        edges = []
    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"edges[{idx}] must be dict")
            continue
        missing = [k for k in REQUIRED_EDGE_FIELDS if k not in edge]
        if missing:
            errors.append(f"edges[{idx}] missing fields: {missing}")

    return SchemaValidationResult(valid=len(errors) == 0, errors=errors)
