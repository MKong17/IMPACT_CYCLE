from __future__ import annotations

from typing import Dict, Optional

from .correction_memory import confusion_frequency, is_verified_locked
from .cycle_types import Claim


def _ensure_flags(item: Dict[str, object]) -> None:
    flags = item.get("validator_flags")
    if not isinstance(flags, list):
        item["validator_flags"] = []


def _append_unique_flag(item: Dict[str, object], flag: str) -> None:
    _ensure_flags(item)
    if flag not in item["validator_flags"]:
        item["validator_flags"].append(flag)


def _append_provenance(item: Dict[str, object], record: Dict[str, object]) -> None:
    prov = item.get("provenance")
    if not isinstance(prov, list):
        prov = []
        item["provenance"] = prov
    prov.append(record)


def _clamp_score(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def _claim_memory_key(claim: Claim) -> str:
    if claim.claim_type == "label":
        return str(claim.value or "").strip()
    if claim.claim_type == "relation":
        return str(claim.predicate or "").strip()
    return ""


def _best_correction_candidate(
    correction_candidates: Optional[Dict[str, object]],
    claim_id: str,
) -> Optional[Dict[str, object]]:
    if not isinstance(correction_candidates, dict):
        return None
    bucket = correction_candidates.get(str(claim_id or "").strip())
    if not isinstance(bucket, dict):
        return None
    best_value = str(bucket.get("best_value", "") or "").strip()
    if not best_value:
        ranked = list(bucket.get("ranked") or [])
        if ranked and isinstance(ranked[0], dict):
            best_value = str(ranked[0].get("value", "") or "").strip()
    if not best_value:
        return None
    best_score_raw = bucket.get("best_score", 0.0)
    try:
        best_score = float(best_score_raw)
    except Exception:
        best_score = 0.0
    return {
        "value": best_value,
        "score": _clamp_score(best_score, 0.0),
        "options": [str(x).strip() for x in list(bucket.get("options") or []) if str(x).strip()],
    }


def _memory_adjustment(
    claim: Claim,
    *,
    correction_memory: Optional[Dict[str, object]],
    frame_idx: Optional[int],
    auto_accept_threshold: float,
    auto_reject_threshold: float,
    finalized_weight_boost: float,
) -> Dict[str, object]:
    key = _claim_memory_key(claim)
    confusion_count = confusion_frequency(
        correction_memory,
        claim_type=claim.claim_type,
        canonical_value=key,
    )
    confusion_penalty = min(0.10, 0.015 * float(confusion_count))
    locked = is_verified_locked(
        correction_memory,
        subject_id=claim.subject_id,
        frame_idx=frame_idx,
    )
    lock_bonus = 0.0
    if locked:
        lock_bonus = min(0.08, 0.04 * max(1.0, float(finalized_weight_boost or 1.0)))
    prior_bonus = min(0.08, max(0.0, float(claim.prior_score) - 0.5) * 0.2)
    low_prior_penalty = min(0.05, max(0.0, 0.45 - float(claim.prior_score)) * 0.15)

    accept_threshold = _clamp_score(
        float(auto_accept_threshold) + confusion_penalty - (0.5 * lock_bonus),
        auto_accept_threshold,
    )
    reject_threshold = _clamp_score(
        float(auto_reject_threshold) - confusion_penalty + (0.75 * lock_bonus),
        auto_reject_threshold,
    )
    effective_support = _clamp_score(
        float(claim.support_ratio) + prior_bonus + lock_bonus - confusion_penalty,
        claim.support_ratio,
    )
    effective_conflict = _clamp_score(
        float(claim.conflict_ratio) + low_prior_penalty + confusion_penalty,
        claim.conflict_ratio,
    )

    return {
        "locked": locked,
        "confusion_count": int(confusion_count),
        "confusion_penalty": float(confusion_penalty),
        "prior_bonus": float(prior_bonus),
        "low_prior_penalty": float(low_prior_penalty),
        "lock_bonus": float(lock_bonus),
        "effective_support": float(effective_support),
        "effective_conflict": float(effective_conflict),
        "accept_threshold": float(accept_threshold),
        "reject_threshold": float(reject_threshold),
    }


def _set_attr(node: Dict[str, object], slot: str, value: str, score: float, claim_id: str) -> None:
    attrs = node.get("attributes")
    if not isinstance(attrs, list):
        attrs = []
        node["attributes"] = attrs
    for att in attrs:
        if not isinstance(att, dict):
            continue
        if str(att.get("slot", "") or "").strip() != slot:
            continue
        att["value"] = value
        att["confidence"] = max(float(att.get("confidence", 0.0) or 0.0), score)
        att["verified"] = True
        prov = att.get("provenance")
        if not isinstance(prov, list):
            prov = []
            att["provenance"] = prov
        prov.append({"source": "cycle_refine", "mode": "auto_accept_attribute", "claim_id": claim_id})
        return
    attrs.append(
        {
            "slot": slot,
            "value": value,
            "confidence": score,
            "provenance": [{"source": "cycle_refine", "mode": "auto_accept_attribute", "claim_id": claim_id}],
            "verified": True,
        }
    )


def revise_graph_from_claims(
    graph: Dict[str, object],
    claims: Dict[str, Claim],
    *,
    auto_accept_threshold: float = 0.85,
    auto_reject_threshold: float = 0.80,
    auto_drop_existence_threshold: float = 0.92,
    auto_drop_support_ceiling: float = 0.20,
    correction_memory: Optional[Dict[str, object]] = None,
    correction_candidates: Optional[Dict[str, object]] = None,
    frame_idx: Optional[int] = None,
    finalized_weight_boost: float = 1.25,
) -> Dict[str, object]:
    out = {
        "image_id": graph.get("image_id"),
        "nodes": [dict(x) for x in graph.get("nodes") or []],
        "edges": [dict(x) for x in graph.get("edges") or []],
        "validator_flags": list(graph.get("validator_flags") or []),
        "metadata": dict(graph.get("metadata") or {}),
    }
    existing_update = dict((out.get("metadata") or {}).get("cycle_update") or {})

    node_by_id = {str(n.get("entity_id")): n for n in out["nodes"]}
    edge_by_id = {str(e.get("edge_id")): e for e in out["edges"]}
    summary = {
        "accepted_claim_ids": [str(x).strip() for x in list(existing_update.get("accepted_claim_ids") or []) if str(x).strip()],
        "accepted_confirm_claim_ids": [str(x).strip() for x in list(existing_update.get("accepted_confirm_claim_ids") or []) if str(x).strip()],
        "accepted_correct_claim_ids": [str(x).strip() for x in list(existing_update.get("accepted_correct_claim_ids") or []) if str(x).strip()],
        "flagged_claim_ids": [str(x).strip() for x in list(existing_update.get("flagged_claim_ids") or []) if str(x).strip()],
        "memory_adjustments": [dict(x) for x in list(existing_update.get("memory_adjustments") or []) if isinstance(x, dict)],
        "correction_applied": [dict(x) for x in list(existing_update.get("correction_applied") or []) if isinstance(x, dict)],
        "auto_removed_node_ids": [str(x).strip() for x in list(existing_update.get("auto_removed_node_ids") or []) if str(x).strip()],
        "auto_removed_claim_ids": [str(x).strip() for x in list(existing_update.get("auto_removed_claim_ids") or []) if str(x).strip()],
    }
    auto_removed_node_ids = set()

    for claim in claims.values():
        memory_adj = _memory_adjustment(
            claim,
            correction_memory=correction_memory,
            frame_idx=frame_idx,
            auto_accept_threshold=auto_accept_threshold,
            auto_reject_threshold=auto_reject_threshold,
            finalized_weight_boost=finalized_weight_boost,
        )
        support_ratio = float(memory_adj["effective_support"])
        conflict_ratio = float(memory_adj["effective_conflict"])
        correction_choice = _best_correction_candidate(correction_candidates, claim.claim_id)
        if (
            bool(memory_adj["locked"])
            or float(memory_adj["prior_bonus"]) > 0.0
            or float(memory_adj["low_prior_penalty"]) > 0.0
            or int(memory_adj["confusion_count"]) > 0
        ):
            summary["memory_adjustments"].append(
                {
                    "claim_id": claim.claim_id,
                    "claim_type": claim.claim_type,
                    "subject_id": claim.subject_id,
                    "predicate": claim.predicate,
                    "value": claim.value,
                    **memory_adj,
                }
            )

        if claim.claim_type == "label":
            node = node_by_id.get(claim.subject_id)
            if node is None:
                continue
            node["verify_confidence"] = float(
                _clamp_score(node.get("verify_confidence", support_ratio), support_ratio) * 0.5
                + support_ratio * 0.5
            )
            if (
                correction_choice is not None
                and str(correction_choice.get("value", "") or "").strip()
                and str(correction_choice.get("value", "") or "").strip() != str(claim.value or "").strip()
                and float(correction_choice.get("score", 0.0) or 0.0) >= float(memory_adj["accept_threshold"])
                and float(correction_choice.get("score", 0.0) or 0.0) >= support_ratio
            ):
                node["canonical_label"] = str(correction_choice["value"])
                node["score"] = max(float(node.get("score", 0.0) or 0.0), float(correction_choice["score"]))
                node["confidence"] = max(float(node.get("confidence", node.get("score", 0.0)) or 0.0), float(correction_choice["score"]))
                node["verify_confidence"] = max(float(node.get("verify_confidence", 0.0) or 0.0), float(correction_choice["score"]))
                node["verified"] = True
                _append_provenance(
                    node,
                    {
                        "source": "cycle_refine",
                        "mode": "auto_correct_label",
                        "claim_id": claim.claim_id,
                        "selected_value": correction_choice["value"],
                        "options": list(correction_choice.get("options") or []),
                        "memory_adjusted": memory_adj,
                    },
                )
                summary["accepted_claim_ids"].append(claim.claim_id)
                summary["accepted_correct_claim_ids"].append(claim.claim_id)
                summary["correction_applied"].append(
                    {
                        "claim_id": claim.claim_id,
                        "claim_type": "label",
                        "selected_value": correction_choice["value"],
                        "score": float(correction_choice.get("score", 0.0) or 0.0),
                    }
                )
                continue
            if (
                support_ratio >= float(memory_adj["accept_threshold"])
                and claim.value
                and float(memory_adj.get("low_prior_penalty", 0.0) or 0.0) <= 0.0
            ):
                node["canonical_label"] = claim.value
                node["score"] = max(float(node.get("score", 0.0) or 0.0), support_ratio)
                node["confidence"] = max(float(node.get("confidence", node.get("score", 0.0)) or 0.0), support_ratio)
                node["verify_confidence"] = max(float(node.get("verify_confidence", 0.0) or 0.0), support_ratio)
                node["verified"] = True
                _append_provenance(
                    node,
                    {
                        "source": "cycle_refine",
                        "mode": "auto_accept_label",
                        "claim_id": claim.claim_id,
                        "memory_adjusted": memory_adj,
                    },
                )
                summary["accepted_claim_ids"].append(claim.claim_id)
                summary["accepted_confirm_claim_ids"].append(claim.claim_id)
            elif conflict_ratio >= float(memory_adj["reject_threshold"]):
                node["risk"] = max(float(node.get("risk", 0.0) or 0.0), conflict_ratio)
                _append_unique_flag(node, "cycle_label_conflict")
                summary["flagged_claim_ids"].append(claim.claim_id)

        elif claim.claim_type == "attribute":
            node = node_by_id.get(claim.subject_id)
            if node is None:
                continue
            node["verify_confidence"] = float(
                _clamp_score(node.get("verify_confidence", support_ratio), support_ratio) * 0.5
                + support_ratio * 0.5
            )
            if support_ratio >= float(memory_adj["accept_threshold"]) and claim.value:
                _set_attr(node, claim.predicate, claim.value, support_ratio, claim.claim_id)
                node["confidence"] = max(float(node.get("confidence", node.get("score", 0.0)) or 0.0), support_ratio)
                node["verify_confidence"] = max(float(node.get("verify_confidence", 0.0) or 0.0), support_ratio)
                node["verified"] = True
                summary["accepted_claim_ids"].append(claim.claim_id)
                summary["accepted_confirm_claim_ids"].append(claim.claim_id)
            elif conflict_ratio >= float(memory_adj["reject_threshold"]):
                node["risk"] = max(float(node.get("risk", 0.0) or 0.0), conflict_ratio)
                _append_unique_flag(node, "cycle_attribute_conflict")
                summary["flagged_claim_ids"].append(claim.claim_id)

        elif claim.claim_type == "existence":
            node = node_by_id.get(claim.subject_id)
            if node is None:
                continue
            node["verify_confidence"] = float(
                _clamp_score(node.get("verify_confidence", support_ratio), support_ratio) * 0.5
                + support_ratio * 0.5
            )
            if conflict_ratio >= float(memory_adj["reject_threshold"]):
                strong_conflict = (
                    float(conflict_ratio) >= float(auto_drop_existence_threshold)
                    and float(support_ratio) <= float(auto_drop_support_ceiling)
                    and not bool(memory_adj.get("locked", False))
                )
                if strong_conflict:
                    auto_removed_node_ids.add(str(claim.subject_id or "").strip())
                    summary["auto_removed_claim_ids"].append(claim.claim_id)
                else:
                    node["risk"] = max(float(node.get("risk", 0.0) or 0.0), conflict_ratio)
                    _append_unique_flag(node, "cycle_existence_conflict")
                    summary["flagged_claim_ids"].append(claim.claim_id)

        elif claim.claim_type == "relation":
            edge_id = claim.evidence_edge_ids[0] if claim.evidence_edge_ids else ""
            edge = edge_by_id.get(str(edge_id))
            if edge is None:
                continue
            edge["verify_confidence"] = float(
                _clamp_score(edge.get("verify_confidence", support_ratio), support_ratio) * 0.5
                + support_ratio * 0.5
            )
            if (
                correction_choice is not None
                and str(correction_choice.get("value", "") or "").strip()
                and str(correction_choice.get("value", "") or "").strip() != str(claim.predicate or "").strip()
                and float(correction_choice.get("score", 0.0) or 0.0) >= float(memory_adj["accept_threshold"])
                and float(correction_choice.get("score", 0.0) or 0.0) >= support_ratio
            ):
                edge["relation"] = str(correction_choice["value"])
                edge["verified"] = True
                edge["score"] = max(float(edge.get("score", 0.0) or 0.0), float(correction_choice.get("score", 0.0) or 0.0))
                edge["confidence"] = max(float(edge.get("confidence", edge.get("score", 0.0)) or 0.0), float(correction_choice.get("score", 0.0) or 0.0))
                edge["verify_confidence"] = max(float(edge.get("verify_confidence", 0.0) or 0.0), float(correction_choice.get("score", 0.0) or 0.0))
                evidence = edge.get("evidence")
                if not isinstance(evidence, list):
                    evidence = []
                    edge["evidence"] = evidence
                evidence.append(
                    {
                        "source": "cycle_refine",
                        "mode": "auto_correct_relation",
                        "claim_id": claim.claim_id,
                        "selected_value": correction_choice["value"],
                        "options": list(correction_choice.get("options") or []),
                        "memory_adjusted": memory_adj,
                    }
                )
                summary["accepted_claim_ids"].append(claim.claim_id)
                summary["accepted_correct_claim_ids"].append(claim.claim_id)
                summary["correction_applied"].append(
                    {
                        "claim_id": claim.claim_id,
                        "claim_type": "relation",
                        "selected_value": correction_choice["value"],
                        "score": float(correction_choice.get("score", 0.0) or 0.0),
                    }
                )
                continue
            if support_ratio >= float(memory_adj["accept_threshold"]):
                edge["verified"] = True
                edge["score"] = max(float(edge.get("score", 0.0) or 0.0), support_ratio)
                edge["confidence"] = max(float(edge.get("confidence", edge.get("score", 0.0)) or 0.0), support_ratio)
                edge["verify_confidence"] = max(float(edge.get("verify_confidence", 0.0) or 0.0), support_ratio)
                evidence = edge.get("evidence")
                if not isinstance(evidence, list):
                    evidence = []
                    edge["evidence"] = evidence
                evidence.append(
                    {
                        "source": "cycle_refine",
                        "mode": "auto_accept_relation",
                        "claim_id": claim.claim_id,
                        "memory_adjusted": memory_adj,
                    }
                )
                summary["accepted_claim_ids"].append(claim.claim_id)
                summary["accepted_confirm_claim_ids"].append(claim.claim_id)
            elif conflict_ratio >= float(memory_adj["reject_threshold"]):
                edge["risk"] = max(float(edge.get("risk", 0.0) or 0.0), conflict_ratio)
                _append_unique_flag(edge, "cycle_relation_conflict")
                summary["flagged_claim_ids"].append(claim.claim_id)

    if auto_removed_node_ids:
        out["nodes"] = [
            dict(node)
            for node in list(out.get("nodes") or [])
            if str(node.get("entity_id", "") or "").strip() not in auto_removed_node_ids
        ]
        out["edges"] = [
            dict(edge)
            for edge in list(out.get("edges") or [])
            if (
                str(edge.get("src_id", "") or "").strip() not in auto_removed_node_ids
                and str(edge.get("dst_id", "") or "").strip() not in auto_removed_node_ids
            )
        ]
    summary["accepted_claim_ids"] = sorted({str(x).strip() for x in list(summary.get("accepted_claim_ids") or []) if str(x).strip()})
    summary["accepted_confirm_claim_ids"] = sorted({str(x).strip() for x in list(summary.get("accepted_confirm_claim_ids") or []) if str(x).strip()})
    summary["accepted_correct_claim_ids"] = sorted({str(x).strip() for x in list(summary.get("accepted_correct_claim_ids") or []) if str(x).strip()})
    summary["flagged_claim_ids"] = sorted({str(x).strip() for x in list(summary.get("flagged_claim_ids") or []) if str(x).strip()})
    summary["auto_removed_claim_ids"] = sorted({str(x).strip() for x in list(summary.get("auto_removed_claim_ids") or []) if str(x).strip()})
    merged_removed_nodes = {str(x).strip() for x in list(summary.get("auto_removed_node_ids") or []) if str(x).strip()}
    merged_removed_nodes.update({str(x).strip() for x in auto_removed_node_ids if str(x).strip()})
    summary["auto_removed_node_ids"] = sorted(merged_removed_nodes)
    out["metadata"]["cycle_update"] = summary
    return out
