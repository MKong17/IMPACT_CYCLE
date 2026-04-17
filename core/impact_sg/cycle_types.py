from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


def _safe_score(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


@dataclass
class Claim:
    claim_id: str
    claim_type: str
    subject_id: str
    predicate: str
    object_id: str = ""
    value: str = ""
    source_graph_snapshot_id: str = ""
    evidence_node_ids: List[str] = field(default_factory=list)
    evidence_edge_ids: List[str] = field(default_factory=list)
    provenance: List[Dict[str, object]] = field(default_factory=list)
    prior_score: float = 0.5
    support_score: float = 0.0
    conflict_score: float = 0.0
    uncertainty_score: float = 1.0
    status: str = "proposed"

    def __post_init__(self) -> None:
        self.prior_score = _safe_score(self.prior_score, 0.5)
        self.support_score = _safe_score(self.support_score, 0.0)
        self.conflict_score = _safe_score(self.conflict_score, 0.0)
        self.uncertainty_score = _safe_score(self.uncertainty_score, 1.0)

    @property
    def support_ratio(self) -> float:
        denom = self.support_score + self.conflict_score + 1e-6
        return self.support_score / denom

    @property
    def conflict_ratio(self) -> float:
        denom = self.support_score + self.conflict_score + 1e-6
        return self.conflict_score / denom

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class EvidenceView:
    view_id: str
    view_type: str
    payload: Dict[str, object]
    derived_claim_ids: List[str] = field(default_factory=list)
    confidence: float = 0.5
    provenance: List[Dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = _safe_score(self.confidence, 0.5)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class BeliefState:
    graph: Dict[str, object]
    claims: Dict[str, Claim] = field(default_factory=dict)
    views: Dict[str, EvidenceView] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "graph": dict(self.graph),
            "claims": {k: v.to_dict() for k, v in self.claims.items()},
            "views": {k: v.to_dict() for k, v in self.views.items()},
            "metadata": dict(self.metadata),
        }
