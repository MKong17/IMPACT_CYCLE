from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class CorrectionSession:
    kind: str
    meta: Dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=_utc_now_iso)
    steps: int = 0


class CorrectionBuffer:
    def __init__(self) -> None:
        self.active: Optional[CorrectionSession] = None
        self.history: List[Dict[str, Any]] = []

    def begin(
        self,
        kind: str,
        *,
        meta: Optional[Dict[str, Any]] = None,
        replace: bool = True,
    ) -> CorrectionSession:
        if self.active is not None and not replace:
            return self.active
        self.active = CorrectionSession(
            kind=str(kind or "unknown").strip() or "unknown",
            meta=dict(meta or {}),
        )
        return self.active

    def note_step(self, count: int = 1) -> None:
        if self.active is None:
            return
        try:
            delta = int(count)
        except Exception:
            delta = 1
        if delta <= 0:
            delta = 1
        self.active.steps += delta

    def commit(
        self,
        *,
        records: Optional[List[Dict[str, Any]]] = None,
        meta_update: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.active or CorrectionSession(kind="implicit")
        if meta_update:
            session.meta.update(dict(meta_update))
        summary = {
            "kind": session.kind,
            "meta": dict(session.meta),
            "started_at": session.started_at,
            "committed_at": _utc_now_iso(),
            "steps": int(session.steps),
            "records": list(records or []),
            "changed": bool(records),
        }
        self.history.append(summary)
        self.active = None
        return summary

    def discard(self, *, reason: str = "") -> Optional[Dict[str, Any]]:
        if self.active is None:
            return None
        summary = {
            "kind": self.active.kind,
            "meta": dict(self.active.meta),
            "started_at": self.active.started_at,
            "discarded_at": _utc_now_iso(),
            "steps": int(self.active.steps),
            "reason": str(reason or "").strip(),
            "changed": False,
        }
        self.history.append(summary)
        self.active = None
        return summary
