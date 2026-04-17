from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


VERDICT_SUPPORTED = "supported"
VERDICT_WEAK = "weak"
VERDICT_CONFLICTING = "conflicting"
VERDICT_UNSUPPORTED = "unsupported"


@dataclass
class VerificationTarget:
    target_type: str
    target_id: str
    payload: Dict[str, Any]
    image_path: str = ""


@dataclass
class VerificationResult:
    target_type: str
    target_id: str
    support_score: float
    conflict_score: float
    verdict: str
    reasons: List[str]
    suggested_action: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_type": str(self.target_type),
            "target_id": str(self.target_id),
            "support_score": float(max(0.0, min(1.0, self.support_score))),
            "conflict_score": float(max(0.0, min(1.0, self.conflict_score))),
            "verdict": str(self.verdict),
            "reasons": [str(x) for x in list(self.reasons or [])],
            "suggested_action": str(self.suggested_action),
            "metadata": dict(self.metadata or {}),
        }


class BaseVerifier:
    """
    Extensible verifier API:
    - future VLM adapters can subclass this and call external models.
    - current implementation below degrades gracefully to metadata-only checks.
    """

    def verify(self, target: VerificationTarget) -> VerificationResult:
        raise NotImplementedError


class HeuristicVerifier(BaseVerifier):
    def verify(self, target: VerificationTarget) -> VerificationResult:
        payload = dict(target.payload or {})
        support = float(payload.get("estimated_support", payload.get("support_score", 0.5)) or 0.5)
        conflict = float(payload.get("estimated_conflict", payload.get("conflict_score", 0.0)) or 0.0)
        support = max(0.0, min(1.0, support))
        conflict = max(0.0, min(1.0, conflict))

        reasons: List[str] = []
        if not str(target.image_path or "").strip():
            reasons.append("metadata_only_verification")
        if support < 0.45:
            reasons.append("weak_support_from_available_evidence")
        if conflict > 0.55:
            reasons.append("cross_path_conflict_detected")
        if not reasons:
            reasons.append("no_strong_issue_detected")

        verdict = VERDICT_SUPPORTED
        action = "keep"
        if conflict > 0.7:
            verdict = VERDICT_CONFLICTING
            action = "review"
        elif support < 0.3:
            verdict = VERDICT_UNSUPPORTED
            action = "remove"
        elif support < 0.55 or conflict > 0.4:
            verdict = VERDICT_WEAK
            action = "review"

        return VerificationResult(
            target_type=str(target.target_type),
            target_id=str(target.target_id),
            support_score=support,
            conflict_score=conflict,
            verdict=verdict,
            reasons=reasons,
            suggested_action=action,
            metadata={"degraded_to_metadata_only": not bool(str(target.image_path or "").strip())},
        )


def verification_json_schema() -> Dict[str, Any]:
    # JSON schema for downstream adapters and UI contract.
    return {
        "type": "object",
        "required": [
            "target_type",
            "target_id",
            "support_score",
            "conflict_score",
            "verdict",
            "reasons",
            "suggested_action",
        ],
        "properties": {
            "target_type": {"type": "string", "enum": ["track", "attribute", "edge", "dynamic", "summary", "segmentation"]},
            "target_id": {"type": "string"},
            "support_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "conflict_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "verdict": {"type": "string", "enum": [VERDICT_SUPPORTED, VERDICT_WEAK, VERDICT_CONFLICTING, VERDICT_UNSUPPORTED]},
            "reasons": {"type": "array", "items": {"type": "string"}},
            "suggested_action": {
                "type": "string",
                "enum": ["keep", "review", "relabel", "remove", "split", "merge", "rewrite"],
            },
            "metadata": {"type": "object"},
        },
    }
