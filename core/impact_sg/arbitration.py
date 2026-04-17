from __future__ import annotations

from typing import Dict, List, Optional

from .correction_memory import common_confusions, confusion_frequency, is_verified_locked
from .cycle_types import Claim


def build_minimal_human_question(
    claim: Claim,
    *,
    correction_memory: Optional[Dict[str, object]] = None,
    correction_payload: Optional[Dict[str, object]] = None,
) -> str:
    suffix = ""
    key = claim.value if claim.claim_type == "label" else claim.predicate
    confusions = common_confusions(
        correction_memory,
        claim_type=claim.claim_type,
        canonical_value=key,
    )
    if confusions:
        suffix = " Common confusion to rule out: " + ", ".join(confusions) + "."
    options = [str(x).strip() for x in list((correction_payload or {}).get("options") or []) if str(x).strip()]
    best_value = str((correction_payload or {}).get("best_value", "") or "").strip()
    if options and claim.claim_type == "label":
        return (
            f"Which canonical label best fits object {claim.subject_id}? "
            f"Options: {', '.join(options)}."
            + (f" Current best machine suggestion: {best_value}." if best_value else "")
            + suffix
        )
    if options and claim.claim_type == "relation":
        return (
            f"Which canonical relation best describes {claim.subject_id} and {claim.object_id}? "
            f"Options: {', '.join(options)}."
            + (f" Current best machine suggestion: {best_value}." if best_value else "")
            + suffix
        )
    if claim.claim_type == "label":
        return f"Is object {claim.subject_id} correctly labeled as '{claim.value}'?{suffix}"
    if claim.claim_type == "relation":
        return f"Does relation '{claim.predicate}' hold between {claim.subject_id} and {claim.object_id}?{suffix}"
    if claim.claim_type == "attribute":
        return f"Is {claim.subject_id} '{claim.predicate}={claim.value}'?"
    if claim.claim_type == "existence":
        return f"Is object {claim.subject_id} actually visible in this frame?"
    return f"Please verify claim {claim.claim_id}."


def build_human_arbitration_queue(
    claims: Dict[str, Claim],
    *,
    max_items: int = 3,
    threshold: float = 0.45,
    correction_memory: Optional[Dict[str, object]] = None,
    correction_candidates: Optional[Dict[str, object]] = None,
    frame_idx: Optional[int] = None,
) -> List[Dict[str, object]]:
    if int(max_items or 0) <= 0:
        return []
    queue: List[Dict[str, object]] = []
    for claim in claims.values():
        uncertainty = float(claim.uncertainty_score)
        conflict = float(claim.conflict_ratio)
        scarcity_bonus = 0.15 if claim.claim_type in {"relation", "attribute"} else 0.05
        key = claim.value if claim.claim_type == "label" else claim.predicate
        confusion_total = confusion_frequency(
            correction_memory,
            claim_type=claim.claim_type,
            canonical_value=key,
        )
        memory_bonus = min(0.2, 0.03 * float(confusion_total))
        locked = is_verified_locked(
            correction_memory,
            subject_id=claim.subject_id,
            frame_idx=frame_idx,
        )
        if locked and conflict < 0.75:
            continue
        utility = (0.5 * uncertainty) + (0.35 * conflict) + scarcity_bonus + memory_bonus
        correction_payload = None
        if isinstance(correction_candidates, dict):
            payload = correction_candidates.get(claim.claim_id)
            if isinstance(payload, dict):
                correction_payload = payload
                if str(payload.get("best_value", "") or "").strip():
                    utility += min(0.12, 0.04 * float(payload.get("best_score", 0.0) or 0.0))
        if locked:
            utility = max(0.0, utility - 0.1)
        if utility < float(threshold):
            continue
        queue.append(
            {
                "claim_id": claim.claim_id,
                "claim_type": claim.claim_type,
                "priority": utility,
                "question": build_minimal_human_question(
                    claim,
                    correction_memory=correction_memory,
                    correction_payload=correction_payload,
                ),
                "subject_id": claim.subject_id,
                "object_id": claim.object_id,
                "predicate": claim.predicate,
                "value": claim.value,
                "evidence_node_ids": [str(x) for x in list(claim.evidence_node_ids or []) if str(x).strip()],
                "evidence_edge_ids": [str(x) for x in list(claim.evidence_edge_ids or []) if str(x).strip()],
                "source_graph_snapshot_id": str(claim.source_graph_snapshot_id or ""),
                "claim_row": claim.to_dict(),
                "locked": locked,
                "memory_bonus": memory_bonus,
                "question_options": list((correction_payload or {}).get("options") or []),
                "suggested_value": str((correction_payload or {}).get("best_value", "") or ""),
                "suggested_score": float((correction_payload or {}).get("best_score", 0.0) or 0.0),
                "frame_idx": int(frame_idx) if frame_idx is not None else -1,
            }
        )
    queue.sort(key=lambda x: float(x.get("priority", 0.0)), reverse=True)
    return queue[: max(0, int(max_items))]
