from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _heuristic_value(slot: str, canonical_label: str) -> str:
    slot = str(slot or "").strip().lower()
    label = str(canonical_label or "").strip().lower()
    defaults = {
        "color": "unknown",
        "material": "unknown",
        "size": "medium",
        "state": "visible",
        "countability": "countable",
        "affordance": "usable",
    }
    if label == "person" and slot == "countability":
        return "countable"
    return defaults.get(slot, "unknown")


def _norm_slot(text: object) -> str:
    return str(text or "").strip().lower()


def _norm_value(text: object) -> str:
    return str(text or "").strip()


def _extract_llm_like_attrs(node: Dict[str, object]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    # 1) Existing attributes on node (may already come from upstream LLM backend).
    for row in list(node.get("attributes") or []):
        if not isinstance(row, dict):
            continue
        slot = _norm_slot(row.get("slot", ""))
        if not slot:
            continue
        out.append(dict(row))

    # 2) Optional side channels used by some backends.
    for key in ("llm_attributes", "qwen_attributes", "person_attributes"):
        payload = node.get(key)
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    slot = _norm_slot(row.get("slot", row.get("name", "")))
                    if not slot:
                        continue
                    value = _norm_value(row.get("value", row.get("text", "")))
                    out.append(
                        {
                            "slot": slot,
                            "value": value,
                            "confidence": float(row.get("confidence", row.get("score", 0.0)) or 0.0),
                            "provenance": str(row.get("provenance", "llm")) or "llm",
                            "verified": bool(row.get("verified", False)),
                        }
                    )
        elif isinstance(payload, dict):
            for k, v in payload.items():
                slot = _norm_slot(k)
                if not slot:
                    continue
                out.append(
                    {
                        "slot": slot,
                        "value": _norm_value(v),
                        "confidence": 0.0,
                        "provenance": "llm",
                        "verified": False,
                    }
                )
    return out


def _merge_attr_rows(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    """
    Keep best row per slot.
    Preference: non-empty value > higher confidence.
    """
    out: Dict[str, Dict[str, object]] = {}
    for row in rows:
        slot = _norm_slot(row.get("slot", ""))
        if not slot:
            continue
        value = _norm_value(row.get("value", ""))
        conf = float(row.get("confidence", 0.0) or 0.0)
        normalized = {
            "slot": slot,
            "value": value,
            "confidence": conf,
            "provenance": str(row.get("provenance", "llm") or "llm"),
            "verified": bool(row.get("verified", False)),
        }
        prev = out.get(slot)
        if prev is None:
            out[slot] = normalized
            continue
        prev_value = _norm_value(prev.get("value", ""))
        prev_conf = float(prev.get("confidence", 0.0) or 0.0)
        score_new: Tuple[int, float] = (1 if value else 0, conf)
        score_prev: Tuple[int, float] = (1 if prev_value else 0, prev_conf)
        if score_new > score_prev:
            out[slot] = normalized
    return out


def extract_attributes_for_nodes(
    nodes: List[Dict[str, object]],
    *,
    ontology,
    default_confidence: float,
    allow_affordance: bool,
) -> List[Dict[str, object]]:
    # Build one unified slot schema for all nodes in this graph/frame.
    # Requirement: every node must share identical attribute categories.
    unified_slots: List[str] = []
    seen_slots: set[str] = set()
    prepared: List[Tuple[Dict[str, object], Dict[str, Dict[str, object]], List[str]]] = []

    for node in nodes:
        n = dict(node)
        label = str(n.get("canonical_label", "")).strip().lower()
        allowed_slots = [str(x).strip().lower() for x in list(ontology.attribute_slots_for_label(label) or []) if str(x).strip()]
        if not allow_affordance:
            allowed_slots = [x for x in allowed_slots if x != "affordance"]
        llm_rows = _extract_llm_like_attrs(n)
        merged = _merge_attr_rows(llm_rows)

        local_slots = list(allowed_slots)
        for slot in merged.keys():
            if slot not in local_slots:
                local_slots.append(slot)
        for slot in local_slots:
            if slot and slot not in seen_slots:
                seen_slots.add(slot)
                unified_slots.append(slot)
        prepared.append((n, merged, allowed_slots))

    out: List[Dict[str, object]] = []
    for n, merged_llm, allowed_slots in prepared:
        label = str(n.get("canonical_label", "")).strip().lower()
        attrs: List[Dict[str, object]] = []
        allowed_set = set(allowed_slots)
        for slot in unified_slots:
            llm_attr = dict(merged_llm.get(slot) or {})
            if llm_attr:
                attrs.append(
                    {
                        "slot": slot,
                        "value": _norm_value(llm_attr.get("value", "")),
                        "confidence": float(llm_attr.get("confidence", default_confidence) or 0.0),
                        "provenance": str(llm_attr.get("provenance", "llm") or "llm"),
                        "verified": bool(llm_attr.get("verified", False)),
                    }
                )
                continue

            # Missing/undetected attributes are kept empty by design.
            if slot in allowed_set:
                attrs.append(
                    {
                        "slot": slot,
                        "value": "",
                        "confidence": 0.0,
                        "provenance": "missing",
                        "verified": False,
                    }
                )
            else:
                attrs.append(
                    {
                        "slot": slot,
                        "value": "",
                        "confidence": 0.0,
                        "provenance": "not_applicable",
                        "verified": False,
                    }
                )
        n["attributes"] = attrs
        out.append(n)
    return out
