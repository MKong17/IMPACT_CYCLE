from __future__ import annotations

from typing import Dict, Iterable

from .cycle_types import Claim


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


def _refresh_status(claim: Claim) -> None:
    if claim.support_score <= 0.0 and claim.conflict_score <= 0.0:
        claim.status = "unreviewed"
        claim.uncertainty_score = max(claim.uncertainty_score, 1.0)
        return
    if claim.support_ratio >= 0.65 and claim.support_score > claim.conflict_score:
        claim.status = "supported"
    elif claim.conflict_ratio >= 0.65 and claim.conflict_score > claim.support_score:
        claim.status = "conflicted"
    else:
        claim.status = "uncertain"


def aggregate_claim_scores(
    claims: Dict[str, Claim],
    votes: Iterable[Dict[str, object]],
) -> Dict[str, Claim]:
    for vote in votes:
        cid = str(vote.get("claim_id", "") or "").strip()
        if cid not in claims:
            continue
        score = _clamp_score(vote.get("score", 0.0), 0.0)
        decision = str(vote.get("vote", "") or "").strip().lower()
        row = claims[cid]
        if decision == "support":
            row.support_score = min(1.0, row.support_score + score)
        elif decision == "conflict":
            row.conflict_score = min(1.0, row.conflict_score + score)
        else:
            row.uncertainty_score = max(row.uncertainty_score, min(1.0, 0.5 + (0.5 * score)))

        total = row.support_score + row.conflict_score
        if total > 0.0:
            disagreement = abs(row.support_score - row.conflict_score) / (total + 1e-6)
            row.uncertainty_score = min(row.uncertainty_score, 1.0 - disagreement)
        _refresh_status(row)

    for row in claims.values():
        _refresh_status(row)
    return claims
