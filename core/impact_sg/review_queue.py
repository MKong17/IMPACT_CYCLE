from __future__ import annotations

from typing import Dict, List

from .stage_validator import STAGE_A, STAGE_E, STAGE_G, STAGE_S, STAGE_T, StageValidator


def build_review_queue(graph: Dict[str, object], qa_items: List[Dict[str, object]]) -> List[Dict[str, object]]:
    queue: List[Dict[str, object]] = []

    for node in graph.get("nodes") or []:
        risk = float(node.get("risk", 0.0))
        flags = list(node.get("validator_flags") or [])
        if risk >= 0.5 or flags:
            queue.append(
                {
                    "item_type": "node",
                    "item_id": node.get("entity_id"),
                    "priority": max(risk, 0.6 if flags else 0.0),
                    "reasons": flags or ["high_risk_node"],
                    "actions": [
                        "accept",
                        "edit_label",
                        "edit_mask",
                        "re_prompt_by_category",
                        "re_prompt_by_referring_expression",
                        "mark_invalid",
                        "add_missing_node",
                    ],
                }
            )

    for edge in graph.get("edges") or []:
        risk = float(edge.get("risk", 0.0))
        flags = list(edge.get("validator_flags") or [])
        if risk >= 0.5 or flags:
            queue.append(
                {
                    "item_type": "edge",
                    "item_id": edge.get("edge_id"),
                    "priority": max(risk, 0.6 if flags else 0.0),
                    "reasons": flags or ["high_risk_edge"],
                    "actions": ["accept", "mark_invalid", "add_missing_edge"],
                }
            )

    for qa in qa_items:
        risk = float(qa.get("risk", 0.0))
        flags = list(qa.get("validator_flags") or [])
        if risk >= 0.5 or flags:
            queue.append(
                {
                    "item_type": "vqa",
                    "item_id": qa.get("qid"),
                    "priority": max(risk, 0.6 if flags else 0.0),
                    "reasons": flags or ["high_risk_vqa"],
                    "actions": ["accept", "edit_label", "mark_invalid"],
                }
            )

    queue.sort(key=lambda x: float(x.get("priority", 0.0)), reverse=True)
    return queue


def build_stage_review_queue(
    graph: Dict[str, object],
    *,
    scene_graph_bundle: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """
    STAGE-native queue for verification UI.
    Keeps legacy build_review_queue intact for backward compatibility.
    """
    validator = StageValidator()
    result = validator.validate(
        graph=dict(graph or {}),
        scene_graph_bundle=dict(scene_graph_bundle or {}),
    )
    stage_items = dict(result.get("stage_items") or {})
    queue = list(result.get("review_queue") or [])
    queue.sort(key=lambda row: float(row.get("priority", 0.0) or 0.0), reverse=True)
    return {
        "review_queue": queue,
        "stage_items": {
            STAGE_T: list(stage_items.get(STAGE_T) or []),
            STAGE_A: list(stage_items.get(STAGE_A) or []),
            STAGE_E: list(stage_items.get(STAGE_E) or []),
            STAGE_S: list(stage_items.get(STAGE_S) or []),
            STAGE_G: list(stage_items.get(STAGE_G) or []),
        },
        "module_scores": dict(result.get("module_scores") or {}),
        "conflicts": list(result.get("conflicts") or []),
        "sam_verification": dict(result.get("sam_verification") or {}),
    }
