from __future__ import annotations

import copy
import datetime as dt
import json
import uuid
from typing import Dict, List, Optional


def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def new_change(
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
    return {
        "change_id": f"chg_{uuid.uuid4().hex[:12]}",
        "task_type": str(task_type or "").strip(),
        "item_id": str(item_id or "").strip(),
        "op": str(op or "update_field").strip(),
        "field_path": str(field_path or "").strip(),
        "before": copy.deepcopy(before),
        "after": copy.deepcopy(after),
        "validator_id": str(validator_id or "").strip(),
        "round": int(round_idx),
        "timestamp": now_iso(),
        "reason": str(reason or "").strip(),
        "status": "proposed",
        "decision_by": "",
        "decision_timestamp": "",
        "decision_reason": "",
    }


def apply_decision(change: Dict[str, object], *, approved: bool, decision_by: str, reason: str = "") -> Dict[str, object]:
    out = dict(change)
    out["status"] = "confirmed" if approved else "rejected"
    out["decision_by"] = str(decision_by or "").strip()
    out["decision_timestamp"] = now_iso()
    out["decision_reason"] = str(reason or "").strip()
    return out


def export_ndjson(changes: List[Dict[str, object]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in changes:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def import_ndjson(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def merge_changes(existing: List[Dict[str, object]], incoming: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = {str(x.get("change_id", "")): i for i, x in enumerate(existing)}
    out = [dict(x) for x in existing]
    for row in incoming:
        cid = str(row.get("change_id", "")).strip()
        if cid and cid in seen:
            out[seen[cid]] = dict(row)
        else:
            out.append(dict(row))
    return out


def filter_by_task(changes: List[Dict[str, object]], task_type: str) -> List[Dict[str, object]]:
    t = str(task_type or "").strip()
    return [c for c in changes if str(c.get("task_type", "")).strip() == t]


def summarize(change: Dict[str, object]) -> str:
    status = str(change.get("status", "proposed"))
    item_id = str(change.get("item_id", ""))
    field_path = str(change.get("field_path", ""))
    op = str(change.get("op", "update"))
    validator = str(change.get("validator_id", ""))
    return f"[{status}] {op} {item_id} :: {field_path} (by {validator})"
