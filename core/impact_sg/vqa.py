from __future__ import annotations

import uuid
from typing import Dict, List, Tuple


def _qid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _node_name(node: Dict[str, object]) -> str:
    return str(node.get("canonical_label", "object"))


def generate_single_turn_vqa(graph: Dict[str, object], max_questions: int = 64) -> List[Dict[str, object]]:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    out: List[Dict[str, object]] = []

    for n in nodes:
        label = _node_name(n)
        out.append(
            {
                "qid": _qid("sq_exist"),
                "question": f"Is there a {label} in the image?",
                "answer": "yes",
                "answer_type": "boolean",
                "evidence_node_ids": [n.get("entity_id")],
                "evidence_edge_ids": [],
                "graph_snapshot_id": graph.get("image_id"),
                "risk": 0.1,
                "verified": False,
                "validator_flags": [],
            }
        )

    # Counting per label.
    count_map: Dict[str, List[str]] = {}
    for n in nodes:
        count_map.setdefault(_node_name(n), []).append(str(n.get("entity_id")))
    for label, ids in count_map.items():
        out.append(
            {
                "qid": _qid("sq_count"),
                "question": f"How many {label} objects are visible?",
                "answer": str(len(ids)),
                "answer_type": "count",
                "evidence_node_ids": ids,
                "evidence_edge_ids": [],
                "graph_snapshot_id": graph.get("image_id"),
                "risk": 0.15,
                "verified": False,
                "validator_flags": [],
            }
        )

    # Attribute questions.
    for n in nodes:
        attrs = n.get("attributes") or []
        for a in attrs:
            if not isinstance(a, dict):
                continue
            slot = str(a.get("slot", "")).strip()
            if not slot:
                continue
            out.append(
                {
                    "qid": _qid("sq_attr"),
                    "question": f"What is the {slot} of the {_node_name(n)}?",
                    "answer": str(a.get("value", "unknown")),
                    "answer_type": "attribute",
                    "evidence_node_ids": [n.get("entity_id")],
                    "evidence_edge_ids": [],
                    "graph_snapshot_id": graph.get("image_id"),
                    "risk": 0.2,
                    "verified": False,
                    "validator_flags": [],
                }
            )

    # Relation questions.
    for e in edges:
        rel = str(e.get("relation", ""))
        src = str(e.get("src_id", ""))
        dst = str(e.get("dst_id", ""))
        out.append(
            {
                "qid": _qid("sq_rel"),
                "question": f"What is the relation between {src} and {dst}?",
                "answer": rel,
                "answer_type": "relation",
                "evidence_node_ids": [src, dst],
                "evidence_edge_ids": [e.get("edge_id")],
                "graph_snapshot_id": graph.get("image_id"),
                "risk": 0.25,
                "verified": False,
                "validator_flags": [],
            }
        )

    max_q = max(1, int(max_questions))
    if max_q == 3 and len(out) > 3:
        node_items = [row for row in out if str(row.get("answer_type", "") or "").strip().lower() != "relation"]
        edge_items = [row for row in out if str(row.get("answer_type", "") or "").strip().lower() == "relation"]
        picked: List[Dict[str, object]] = []
        if node_items:
            picked.append(node_items.pop(0))
        if edge_items and len(picked) < 3:
            picked.append(edge_items.pop(0))
        if node_items and len(picked) < 3:
            picked.append(node_items.pop(0))
        for pool in (node_items, edge_items):
            for row in pool:
                if len(picked) >= 3:
                    break
                picked.append(row)
            if len(picked) >= 3:
                break
        return picked[:3]
    return out[:max_q]


def generate_multi_turn_vqa(graph: Dict[str, object], max_chains: int = 24) -> List[Dict[str, object]]:
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    out: List[Dict[str, object]] = []

    chain_count = 0
    for n in nodes:
        if chain_count >= max_chains:
            break
        nid = str(n.get("entity_id"))
        label = _node_name(n)
        chain_id = f"chain_{uuid.uuid4().hex[:8]}"

        # Turn 1 identify object.
        out.append(
            {
                "qid": _qid("mq_t1"),
                "chain_id": chain_id,
                "turn": 1,
                "question": f"Which object are we talking about?",
                "answer": f"{label} ({nid})",
                "answer_type": "reference",
                "evidence_node_ids": [nid],
                "evidence_edge_ids": [],
                "graph_snapshot_id": graph.get("image_id"),
                "risk": 0.15,
                "verified": False,
                "validator_flags": [],
                "dialogue_state": {"focus_node_id": nid},
            }
        )

        attrs = n.get("attributes") or []
        slot = "state"
        value = "unknown"
        if attrs and isinstance(attrs[0], dict):
            slot = str(attrs[0].get("slot", "state"))
            value = str(attrs[0].get("value", "unknown"))

        # Turn 2 attribute question on same object.
        out.append(
            {
                "qid": _qid("mq_t2"),
                "chain_id": chain_id,
                "turn": 2,
                "question": f"What is its {slot}?",
                "answer": value,
                "answer_type": "attribute",
                "evidence_node_ids": [nid],
                "evidence_edge_ids": [],
                "graph_snapshot_id": graph.get("image_id"),
                "risk": 0.2,
                "verified": False,
                "validator_flags": [],
                "dialogue_state": {"focus_node_id": nid, "last_slot": slot},
            }
        )

        # Turn 3 relation on same object if available.
        rel_edge = None
        for e in edges:
            if str(e.get("src_id")) == nid or str(e.get("dst_id")) == nid:
                rel_edge = e
                break
        if rel_edge:
            other = str(rel_edge.get("dst_id")) if str(rel_edge.get("src_id")) == nid else str(rel_edge.get("src_id"))
            out.append(
                {
                    "qid": _qid("mq_t3"),
                    "chain_id": chain_id,
                    "turn": 3,
                    "question": f"How is it related to {other}?",
                    "answer": str(rel_edge.get("relation", "unknown")),
                    "answer_type": "relation",
                    "evidence_node_ids": [nid, other],
                    "evidence_edge_ids": [rel_edge.get("edge_id")],
                    "graph_snapshot_id": graph.get("image_id"),
                    "risk": 0.25,
                    "verified": False,
                    "validator_flags": [],
                    "dialogue_state": {"focus_node_id": nid, "focus_edge_id": rel_edge.get("edge_id")},
                }
            )

        # Turn 4 comparison/count style.
        same_label = [x for x in nodes if _node_name(x) == label]
        out.append(
            {
                "qid": _qid("mq_t4"),
                "chain_id": chain_id,
                "turn": 4,
                "question": f"How many objects are the same type as it?",
                "answer": str(len(same_label)),
                "answer_type": "count",
                "evidence_node_ids": [x.get("entity_id") for x in same_label],
                "evidence_edge_ids": [],
                "graph_snapshot_id": graph.get("image_id"),
                "risk": 0.2,
                "verified": False,
                "validator_flags": [],
                "dialogue_state": {"focus_node_id": nid, "comparison_group": label},
            }
        )

        chain_count += 1

    return out
