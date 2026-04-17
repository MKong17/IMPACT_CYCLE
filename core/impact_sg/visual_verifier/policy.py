from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def _claim_type_from_id(claim_id: str) -> str:
    token = str(claim_id or "").strip()
    if token.startswith("claim_label_"):
        return "label"
    if token.startswith("claim_exists_"):
        return "existence"
    if token.startswith("claim_attr_"):
        return "attribute"
    if token.startswith("claim_rel_"):
        return "relation"
    return "other"


DEFAULT_ROLE_POLICY: Dict[str, object] = {
    "caption_label_enabled": False,
    "single_turn": {
        "existence": 1.0,
        "label": 1.0,
        "attribute": 1.0,
        "relation": 1.0,
        "other": 0.9,
    },
    "multi_turn": {
        "temporal_consistency": 1.0,
        "binary_verification": 0.8,
        "constrained_correction": 0.8,
        "counterfactual_verification": 0.7,
        "group_relation_verification": 0.75,
        "other": 0.75,
    },
    "caption": {
        "existence": 0.7,
        "attribute": 0.7,
        "relation": 0.7,
        "label": 0.0,
        "other": 0.6,
    },
}


def _merge_policy(override: Dict[str, object] | None) -> Dict[str, object]:
    out = {
        "caption_label_enabled": bool(DEFAULT_ROLE_POLICY["caption_label_enabled"]),
        "single_turn": dict(DEFAULT_ROLE_POLICY["single_turn"]),
        "multi_turn": dict(DEFAULT_ROLE_POLICY["multi_turn"]),
        "caption": dict(DEFAULT_ROLE_POLICY["caption"]),
    }
    payload = dict(override or {})
    if "caption_label_enabled" in payload:
        out["caption_label_enabled"] = bool(payload.get("caption_label_enabled"))
    for key in ("single_turn", "multi_turn", "caption"):
        if isinstance(payload.get(key), dict):
            out[key].update(dict(payload.get(key) or {}))
    if bool(out["caption_label_enabled"]):
        out["caption"]["label"] = max(0.0, float(out["caption"].get("label", 0.0) or 0.0))
    return out


def _weight_for_vote(vote: Dict[str, object], policy: Dict[str, object]) -> float:
    view_type = str(vote.get("view_type", "") or "").strip().lower()
    claim_type = _claim_type_from_id(str(vote.get("claim_id", "") or ""))
    probe_family = str(vote.get("probe_family", "") or "").strip().lower() or "other"

    if view_type == "single_turn_vqa":
        table = dict(policy.get("single_turn") or {})
        return max(0.0, float(table.get(claim_type, table.get("other", 0.9)) or 0.9))
    if view_type == "multi_turn_vqa":
        table = dict(policy.get("multi_turn") or {})
        return max(0.0, float(table.get(probe_family, table.get("other", 0.75)) or 0.75))
    if view_type == "caption":
        table = dict(policy.get("caption") or {})
        return max(0.0, float(table.get(claim_type, table.get("other", 0.6)) or 0.6))
    return 1.0


def apply_role_policy(
    votes: Iterable[Dict[str, object]],
    *,
    policy_override: Dict[str, object] | None = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    payload = dict(policy_override or {})
    if not payload:
        out: List[Dict[str, object]] = []
        for vote in list(votes or []):
            row = dict(vote or {})
            try:
                base_score = float(row.get("score", 0.0) or 0.0)
            except Exception:
                base_score = 0.0
            row["base_score"] = base_score
            row["weight"] = 1.0
            row["role_policy_applied"] = False
            out.append(row)
        return out, {"policy": {}, "vote_count": len(out), "weighted_vote_count": len(out)}

    policy = _merge_policy(payload)
    out: List[Dict[str, object]] = []
    for vote in list(votes or []):
        row = dict(vote or {})
        try:
            base_score = float(row.get("score", 0.0) or 0.0)
        except Exception:
            base_score = 0.0
        weight = _weight_for_vote(row, policy)
        weighted_score = max(0.0, min(1.0, base_score * weight))
        row["base_score"] = base_score
        row["weight"] = weight
        row["score"] = weighted_score
        row["role_policy_applied"] = True
        out.append(row)

    report = {
        "policy": policy,
        "vote_count": len(out),
        "weighted_vote_count": len([row for row in out if float(row.get("weight", 0.0) or 0.0) > 0.0]),
    }
    return out, report
