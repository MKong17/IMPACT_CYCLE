from __future__ import annotations

from typing import Dict, Iterable, List, Optional


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _claims(payload: Dict[str, object]) -> List[Dict[str, object]]:
    claims = payload.get("claims") or {}
    if isinstance(claims, dict):
        return [dict(row) for row in claims.values() if isinstance(row, dict)]
    if isinstance(claims, list):
        return [dict(row) for row in claims if isinstance(row, dict)]
    return []


def _votes(payload: Dict[str, object], *, view_types: Optional[Iterable[str]] = None) -> List[Dict[str, object]]:
    rows = [dict(row) for row in list(payload.get("votes") or []) if isinstance(row, dict)]
    if view_types is None:
        return rows
    allowed = {str(item).strip() for item in list(view_types or []) if str(item).strip()}
    return [row for row in rows if str(row.get("view_type", "") or "").strip() in allowed]


def infer_frames_in_session(payload: Dict[str, object]) -> int:
    for graph_key in ("graph_after", "graph_before"):
        graph = dict(payload.get(graph_key) or {})
        metadata = dict(graph.get("metadata") or {})
        temporal = dict(metadata.get("temporal_context") or {})
        for key in ("frames_in_session", "bundle_size"):
            value = _safe_int(temporal.get(key), 0)
            if value > 0:
                return value
        sampled = list(temporal.get("sampled_frame_indices") or [])
        if sampled:
            return max(1, len(sampled))
    return 1


def evaluate_cycle_result(
    payload: Dict[str, object],
    *,
    frames_in_session: Optional[int] = None,
) -> Dict[str, float]:
    claims = _claims(payload)
    reviewed_claims = [
        row
        for row in claims
        if (
            str(row.get("status", "") or "").strip().lower() != "unreviewed"
            or (_safe_float(row.get("support_score"), 0.0) + _safe_float(row.get("conflict_score"), 0.0)) > 0.0
        )
    ]
    resolved_claims = [
        row
        for row in reviewed_claims
        if str(row.get("status", "") or "").strip().lower() in {"supported", "conflicted"}
    ]
    uncertain_claims = [
        row
        for row in claims
        if str(row.get("status", "") or "").strip().lower() == "uncertain"
    ]
    reviewed_count = max(1, len(reviewed_claims))
    claim_count = max(1, len(claims))

    caption_votes = _votes(payload, view_types=["caption"])
    caption_conflicts = [
        row for row in caption_votes if str(row.get("vote", "") or "").strip().lower() == "conflict"
    ]

    vqa_votes = _votes(payload, view_types=["single_turn_vqa", "multi_turn_vqa"])
    vqa_conflicts = [
        row for row in vqa_votes if str(row.get("vote", "") or "").strip().lower() == "conflict"
    ]

    human_queue = [dict(row) for row in list(payload.get("human_queue") or []) if isinstance(row, dict)]
    frames = max(1, int(frames_in_session or infer_frames_in_session(payload)))

    summary = dict(payload.get("summary") or {})
    graph_after = dict(payload.get("graph_after") or {})
    cycle_update = dict((graph_after.get("metadata") or {}).get("cycle_update") or {})
    accepted_claim_count = max(
        0,
        _safe_int(summary.get("accepted_claim_count"), len(list(cycle_update.get("accepted_claim_ids") or []))),
    )

    temporal_probe_count = len(
        [
            row
            for row in list(payload.get("probe_results") or [])
            if (
                isinstance(row, dict)
                and str(row.get("view_type", "") or "").strip() == "multi_turn_vqa"
                and str(row.get("probe_family", "") or "").strip() == "temporal_consistency"
            )
        ]
    )
    multi_turn_probe_count = len(
        [
            row
            for row in list(payload.get("probe_results") or [])
            if isinstance(row, dict) and str(row.get("view_type", "") or "").strip() == "multi_turn_vqa"
        ]
    )
    margins = [
        abs(_safe_float(row.get("support_score"), 0.0) - _safe_float(row.get("conflict_score"), 0.0))
        for row in reviewed_claims
    ]
    caption_feedback = dict(((payload.get("caption") or {}).get("feedback")) or {})
    hallucinations = list(caption_feedback.get("hallucinated_mentions") or [])

    return {
        "claim_agreement_rate": float(len(resolved_claims)) / float(reviewed_count),
        "graph_caption_contradiction_rate": float(len(caption_conflicts)) / float(max(1, len(caption_votes))),
        "graph_vqa_contradiction_rate": float(len(vqa_conflicts)) / float(max(1, len(vqa_votes))),
        "human_queries_per_frame": float(len(human_queue)) / float(max(1, frames)),
        "automatic_resolution_rate_before_human_review": float(accepted_claim_count) / float(
            max(1, accepted_claim_count + len(human_queue))
        ),
        "uncertain_claim_rate": float(len(uncertain_claims)) / float(claim_count),
        "support_conflict_margin_mean": float(sum(margins)) / float(max(1, len(margins))),
        "caption_hallucination_count": float(len(hallucinations)),
        "caption_structured_feedback_rate": 1.0 if bool(caption_feedback.get("structured")) else 0.0,
        "temporal_multi_turn_share": float(temporal_probe_count) / float(max(1, multi_turn_probe_count)),
    }
