from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

from core.impact_sg.eval_cycle import evaluate_cycle_result
from core.impact_sg.eval_scene_graph import evaluate_scene_graph
from core.impact_sg.eval_vqa import evaluate_vqa
from core.impact_sg.review_queue import build_review_queue, build_stage_review_queue
from core.validation_log import (
    apply_decision,
    export_ndjson,
    filter_by_task,
    import_ndjson,
    merge_changes,
    new_change,
    summarize,
)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp01(value: object) -> float:
    return max(0.0, min(1.0, _safe_float(value, 0.0)))


def _round4(value: object) -> float:
    return round(_safe_float(value, 0.0), 4)


def _counts_by_status(changes: Sequence[Dict[str, object]]) -> Dict[str, int]:
    counts = {"proposed": 0, "confirmed": 0, "rejected": 0}
    for row in changes:
        status = str(row.get("status", "proposed")).strip().lower() or "proposed"
        counts[status] = int(counts.get(status, 0) + 1)
    return counts


def _summarize_score_cards(cards: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not cards:
        return {"overall_score": 0.0, "cards": []}
    score_sum = 0.0
    valid = 0
    for card in cards:
        if card.get("score") is None:
            continue
        score_sum += _clamp01(card.get("score"))
        valid += 1
    overall = (score_sum / float(valid)) if valid > 0 else 0.0
    return {"overall_score": _round4(overall), "cards": list(cards)}


class UIFeatureService:
    """
    Stable facade for UI-visible validation/review/scoring features.

    The goal is to keep feature logic callable even when the visible Qt layout
    changes. Future UI code should depend on this service instead of reaching
    into task-specific widgets.
    """

    SUPPORTED_TASKS = (
        "Video Scene Graph",
        "Single-turn VQA",
        "Multi-turn VQA",
        "Video Captioning",
    )

    def list_supported_features(self) -> List[Dict[str, object]]:
        return [
            {
                "feature_id": "validation_log",
                "title": "Validation Log",
                "description": "Create, merge, import, export, confirm, and reject review changes.",
                "tasks": list(self.SUPPORTED_TASKS),
            },
            {
                "feature_id": "scene_graph_review_queue",
                "title": "Scene Graph Review Queue",
                "description": "Rank high-risk nodes, edges, and VQA items for manual review.",
                "tasks": ["Video Scene Graph", "Single-turn VQA", "Multi-turn VQA"],
            },
            {
                "feature_id": "stage_verification",
                "title": "STAGE Verification",
                "description": "Reliability scoring and priority queue for Tracks/Attributes/Edges/Dynamics/Summary.",
                "tasks": ["Video Scene Graph"],
            },
            {
                "feature_id": "scene_graph_metrics",
                "title": "Scene Graph Metrics",
                "description": "Compute scene-graph evaluation metrics and UI-ready score cards.",
                "tasks": ["Video Scene Graph"],
            },
            {
                "feature_id": "vqa_metrics",
                "title": "VQA Metrics",
                "description": "Compute VQA evaluation metrics and UI-ready score cards.",
                "tasks": ["Single-turn VQA", "Multi-turn VQA"],
            },
            {
                "feature_id": "cycle_metrics",
                "title": "Cycle Metrics",
                "description": "Compute cross-task cycle consistency and HITL efficiency metrics.",
                "tasks": ["Video Scene Graph"],
            },
        ]

    def create_validation_change(
        self,
        *,
        task_type: str,
        item_id: str,
        op: str,
        field_path: str,
        before,
        after,
        validator_id: str,
        round_idx: int,
        reason: str = "",
    ) -> Dict[str, object]:
        return new_change(
            task_type=task_type,
            item_id=item_id,
            op=op,
            field_path=field_path,
            before=before,
            after=after,
            validator_id=validator_id,
            round_idx=round_idx,
            reason=reason,
        )

    def summarize_changes(self, changes: Sequence[Dict[str, object]]) -> Dict[str, object]:
        rows = [dict(row) for row in changes if isinstance(row, dict)]
        counts = _counts_by_status(rows)
        by_task: Dict[str, Dict[str, int]] = {}
        for task_name in self.SUPPORTED_TASKS:
            task_rows = filter_by_task(rows, task_name)
            by_task[task_name] = _counts_by_status(task_rows)
        return {
            "total_changes": len(rows),
            "status_counts": counts,
            "task_counts": by_task,
            "items": [
                {
                    "change_id": str(row.get("change_id", "")),
                    "task_type": str(row.get("task_type", "")),
                    "status": str(row.get("status", "proposed")),
                    "summary": summarize(row),
                }
                for row in rows
            ],
        }

    def filter_changes(self, changes: Sequence[Dict[str, object]], task_type: str) -> List[Dict[str, object]]:
        rows = [dict(row) for row in changes if isinstance(row, dict)]
        return filter_by_task(rows, task_type)

    def merge_validation_changes(
        self,
        existing: Sequence[Dict[str, object]],
        incoming: Sequence[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        left = [dict(row) for row in existing if isinstance(row, dict)]
        right = [dict(row) for row in incoming if isinstance(row, dict)]
        return merge_changes(left, right)

    def set_change_decision(
        self,
        change: Dict[str, object],
        *,
        approved: bool,
        decision_by: str,
        reason: str = "",
    ) -> Dict[str, object]:
        return apply_decision(change, approved=approved, decision_by=decision_by, reason=reason)

    def import_validation_log(self, path: str) -> List[Dict[str, object]]:
        return import_ndjson(path)

    def export_validation_log(self, changes: Sequence[Dict[str, object]], path: str) -> None:
        rows = [dict(row) for row in changes if isinstance(row, dict)]
        export_ndjson(rows, path)

    def build_scene_graph_review_bundle(
        self,
        *,
        graph: Optional[Dict[str, object]],
        qa_items: Optional[Sequence[Dict[str, object]]] = None,
        changes: Optional[Sequence[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        graph_payload = copy.deepcopy(graph) if isinstance(graph, dict) else {}
        qa_payload = [dict(row) for row in (qa_items or []) if isinstance(row, dict)]
        change_rows = [dict(row) for row in (changes or []) if isinstance(row, dict)]
        queue = build_review_queue(graph_payload, qa_payload)
        return {
            "queue": queue,
            "queue_count": len(queue),
            "highest_priority": _round4(max([float(x.get("priority", 0.0)) for x in queue], default=0.0)),
            "validation_summary": self.summarize_changes(change_rows),
        }

    def build_stage_verification_bundle(
        self,
        *,
        graph: Optional[Dict[str, object]],
        scene_graph_bundle: Optional[Dict[str, object]] = None,
        qwen_track_result: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        return build_stage_review_queue(
            dict(graph or {}),
            scene_graph_bundle=dict(scene_graph_bundle or {}),
            qwen_track_result=dict(qwen_track_result or {}),
        )

    def evaluate_scene_graph_bundle(
        self,
        *,
        pred_graph: Dict[str, object],
        gt_graph: Dict[str, object],
        iou_threshold: float = 0.5,
    ) -> Dict[str, object]:
        metrics = evaluate_scene_graph(pred_graph or {}, gt_graph or {}, iou_threshold=float(iou_threshold))
        cards = [
            self._score_card("entity_label_accuracy", "Entity Label Accuracy", metrics.get("entity_label_accuracy")),
            self._score_card("mask_iou_accuracy", "Mask IoU Accuracy", metrics.get("mask_iou_accuracy")),
            self._score_card("bbox_iou_accuracy", "BBox IoU Accuracy", metrics.get("bbox_iou_accuracy")),
            self._score_card("attribute_f1", "Attribute F1", metrics.get("attribute_f1")),
            self._score_card("relation_f1", "Relation F1", metrics.get("relation_f1")),
            {
                "metric_id": "graph_edit_distance",
                "title": "Graph Edit Distance",
                "value": _round4(metrics.get("graph_edit_distance")),
                "score": None,
                "direction": "lower_is_better",
            },
            {
                "metric_id": "edits_per_image",
                "title": "Edits Per Image",
                "value": _round4(metrics.get("edits_per_image")),
                "score": None,
                "direction": "lower_is_better",
            },
        ]
        return {
            "task_type": "Video Scene Graph",
            "metrics": {k: _round4(v) for k, v in metrics.items()},
            "score_summary": _summarize_score_cards(cards),
        }

    def evaluate_vqa_bundle(
        self,
        *,
        pred_items: Sequence[Dict[str, object]],
        gt_items: Sequence[Dict[str, object]],
        task_type: str = "Single-turn VQA",
    ) -> Dict[str, object]:
        metrics = evaluate_vqa(
            [dict(row) for row in pred_items if isinstance(row, dict)],
            [dict(row) for row in gt_items if isinstance(row, dict)],
        )
        cards = [
            self._score_card("answer_accuracy", "Answer Accuracy", metrics.get("answer_accuracy")),
            self._score_card(
                "evidence_grounding_accuracy",
                "Evidence Grounding Accuracy",
                metrics.get("evidence_grounding_accuracy"),
            ),
            self._score_card("chain_consistency", "Chain Consistency", metrics.get("chain_consistency")),
            {
                "metric_id": "human_rewrite_rate",
                "title": "Human Rewrite Rate",
                "value": _round4(metrics.get("human_rewrite_rate")),
                "score": _round4(1.0 - _clamp01(metrics.get("human_rewrite_rate"))),
                "direction": "lower_is_better",
            },
            {
                "metric_id": "answer_evidence_mismatch_rate",
                "title": "Answer Evidence Mismatch Rate",
                "value": _round4(metrics.get("answer_evidence_mismatch_rate")),
                "score": _round4(1.0 - _clamp01(metrics.get("answer_evidence_mismatch_rate"))),
                "direction": "lower_is_better",
            },
        ]
        return {
            "task_type": str(task_type or "Single-turn VQA"),
            "metrics": {k: _round4(v) for k, v in metrics.items()},
            "score_summary": _summarize_score_cards(cards),
        }

    def evaluate_cycle_bundle(
        self,
        *,
        cycle_result: Dict[str, object],
        frames_in_session: Optional[int] = None,
    ) -> Dict[str, object]:
        metrics = evaluate_cycle_result(dict(cycle_result or {}), frames_in_session=frames_in_session)
        cards = [
            self._score_card("claim_agreement_rate", "Claim Agreement Rate", metrics.get("claim_agreement_rate")),
            {
                "metric_id": "graph_caption_contradiction_rate",
                "title": "Graph-Caption Contradiction Rate",
                "value": _round4(metrics.get("graph_caption_contradiction_rate")),
                "score": _round4(1.0 - _clamp01(metrics.get("graph_caption_contradiction_rate"))),
                "direction": "lower_is_better",
            },
            {
                "metric_id": "graph_vqa_contradiction_rate",
                "title": "Graph-VQA Contradiction Rate",
                "value": _round4(metrics.get("graph_vqa_contradiction_rate")),
                "score": _round4(1.0 - _clamp01(metrics.get("graph_vqa_contradiction_rate"))),
                "direction": "lower_is_better",
            },
            {
                "metric_id": "human_queries_per_frame",
                "title": "Human Queries Per Frame",
                "value": _round4(metrics.get("human_queries_per_frame")),
                "score": None,
                "direction": "lower_is_better",
            },
            self._score_card(
                "automatic_resolution_rate_before_human_review",
                "Automatic Resolution Rate",
                metrics.get("automatic_resolution_rate_before_human_review"),
            ),
        ]
        return {
            "task_type": "Video Scene Graph",
            "metrics": {k: _round4(v) for k, v in metrics.items()},
            "score_summary": _summarize_score_cards(cards),
        }

    @staticmethod
    def _score_card(metric_id: str, title: str, value: object) -> Dict[str, object]:
        val = _clamp01(value)
        return {
            "metric_id": str(metric_id),
            "title": str(title),
            "value": _round4(value),
            "score": _round4(val),
            "direction": "higher_is_better",
        }
