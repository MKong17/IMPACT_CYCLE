from __future__ import annotations

from typing import Dict, List

from .mask_ops import mask_or_bbox_iou
from .schema import validate_scene_graph_schema


_IMPOSSIBLE_RELATIONS = {
    ("left_of", "left_of"),
    ("right_of", "right_of"),
    ("above", "above"),
    ("below", "below"),
}


def validate_scene_graph(graph: Dict[str, object], *, ontology, cfg: Dict[str, object]) -> Dict[str, object]:
    result = {
        "graph_flags": [],
        "node_flags": {},
        "edge_flags": {},
    }

    schema_res = validate_scene_graph_schema(graph)
    if not schema_res.valid:
        result["graph_flags"].append({"type": "schema_violation", "details": schema_res.errors})

    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []

    node_low = float(cfg.get("node_low_conf_threshold", 0.4))
    edge_low = float(cfg.get("edge_low_conf_threshold", 0.4))
    max_dup = float(cfg.get("max_duplicate_mask_iou", 0.9))
    mandatory_required = bool(cfg.get("mandatory_attributes_required", True))

    for i, node in enumerate(nodes):
        flags: List[str] = []
        nid = str(node.get("entity_id", f"node_{i}"))
        if float(node.get("score", 0.0)) < node_low:
            flags.append("low_confidence_node")

        if mandatory_required:
            label = str(node.get("canonical_label", "")).strip().lower()
            mandatory = set(ontology.mandatory_attributes_for_label(label))
            have = set()
            for att in node.get("attributes") or []:
                if isinstance(att, dict):
                    have.add(str(att.get("slot", "")).strip())
            missing = sorted(list(mandatory - have))
            if missing:
                flags.append(f"missing_mandatory_attributes:{','.join(missing)}")

        for j, other in enumerate(nodes):
            if i >= j:
                continue
            if str(other.get("canonical_label", "")).strip().lower() != str(node.get("canonical_label", "")).strip().lower():
                continue
            iou = mask_or_bbox_iou(
                node.get("mask") or {},
                other.get("mask") or {},
                bbox_a=node.get("bbox") or [0, 0, 0, 0],
                bbox_b=other.get("bbox") or [0, 0, 0, 0],
            )
            if iou >= max_dup:
                flags.append("duplicate_entity_candidate")

        if flags:
            result["node_flags"][nid] = flags

    pair_to_rels = {}
    for idx, edge in enumerate(edges):
        eid = str(edge.get("edge_id", f"edge_{idx}"))
        flags: List[str] = []
        if float(edge.get("score", 0.0)) < edge_low:
            flags.append("low_confidence_edge")

        key = (str(edge.get("src_id", "")), str(edge.get("dst_id", "")))
        rel = str(edge.get("relation", ""))
        pair_to_rels.setdefault(key, []).append(rel)

        if flags:
            result["edge_flags"][eid] = flags

    # Simple contradiction check.
    for key, rels in pair_to_rels.items():
        rel_set = set(rels)
        if "left_of" in rel_set and "right_of" in rel_set:
            result["graph_flags"].append({"type": "relation_contradiction", "pair": key, "details": ["left_of vs right_of"]})
        if "above" in rel_set and "below" in rel_set:
            result["graph_flags"].append({"type": "relation_contradiction", "pair": key, "details": ["above vs below"]})

    return result


def validate_vqa_evidence(
    qa_items: List[Dict[str, object]],
    graph: Dict[str, object],
) -> List[Dict[str, object]]:
    node_ids = {str(n.get("entity_id")) for n in (graph.get("nodes") or [])}
    edge_ids = {str(e.get("edge_id")) for e in (graph.get("edges") or [])}
    out = []
    for qa in qa_items:
        flags = []
        for nid in qa.get("evidence_node_ids") or []:
            if str(nid) not in node_ids:
                flags.append("answer_evidence_mismatch:node")
                break
        for eid in qa.get("evidence_edge_ids") or []:
            if str(eid) not in edge_ids:
                flags.append("answer_evidence_mismatch:edge")
                break
        row = dict(qa)
        row["validator_flags"] = sorted(set(list(row.get("validator_flags") or []) + flags))
        out.append(row)
    return out
