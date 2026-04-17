from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List


def _node_label(node: Dict[str, Any], default: str = "object") -> str:
    return str(node.get("canonical_label", node.get("label", default)) or default).strip()


def _graph_frame_idx(graph: Dict[str, Any]) -> int:
    def _parse_from_image_id(text: str) -> int:
        s = str(text or "")
        m = re.search(r"_f(\d+)", s)
        if not m:
            return 0
        try:
            return int(m.group(1))
        except Exception:
            return 0

    meta = dict(graph.get("metadata") or {})
    img_idx = _parse_from_image_id(str(graph.get("image_id", "") or ""))
    try:
        if "graph_frame_idx" in meta:
            meta_idx = int(meta.get("graph_frame_idx", 0) or 0)
            if meta_idx == 0 and img_idx > 0:
                return int(img_idx)
            return meta_idx
    except Exception:
        pass
    try:
        if "frame_idx" in graph:
            return int(graph.get("frame_idx", 0) or 0)
    except Exception:
        pass
    if img_idx > 0:
        return int(img_idx)
    return 0


def _chunk(seq: List[Any], n: int) -> List[List[Any]]:
    size = max(1, int(n))
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _safe_json_extract(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    starts = [idx for idx, ch in enumerate(raw) if ch in "[{"]
    for idx in starts:
        for end in range(len(raw), idx + 1, -1):
            frag = raw[idx:end].strip()
            if not frag:
                continue
            try:
                return json.loads(frag)
            except Exception:
                continue
    return None


def _frame_digest(graph: Dict[str, Any]) -> str:
    nodes = [dict(x) for x in list(graph.get("nodes") or []) if isinstance(x, dict)]
    edges = [dict(x) for x in list(graph.get("edges") or []) if isinstance(x, dict)]
    label_counts: Dict[str, int] = {}
    for node in nodes:
        label = _node_label(node, "object").lower() or "object"
        label_counts[label] = int(label_counts.get(label, 0) + 1)
    top_labels = sorted(label_counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:4]
    rel_counts: Dict[str, int] = {}
    for edge in edges:
        rel = str(edge.get("relation", "") or "").strip().lower()
        if rel:
            rel_counts[rel] = int(rel_counts.get(rel, 0) + 1)
    top_rels = sorted(rel_counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:3]
    frame_idx = int(_graph_frame_idx(graph) or 0)
    labels = ", ".join([f"{k}:{v}" for k, v in top_labels]) if top_labels else "none"
    rels = ", ".join([f"{k}:{v}" for k, v in top_rels]) if top_rels else "none"
    return f"frame={frame_idx} labels=[{labels}] rels=[{rels}]"


def _person_nodes(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for node in list(graph.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        if _node_label(node, "").lower() != "person":
            continue
        out.append(dict(node))
    return out


def _bbox_area(node: Dict[str, Any]) -> float:
    bbox = list(node.get("bbox") or [0, 0, 0, 0])
    if len(bbox) < 4:
        return 0.0
    try:
        w = max(0.0, float(bbox[2] or 0.0))
        h = max(0.0, float(bbox[3] or 0.0))
    except Exception:
        return 0.0
    return float(w * h)


def _focus_person_nodes(graph: Dict[str, Any], *, top_k: int = 3) -> List[Dict[str, Any]]:
    people = _person_nodes(graph)
    people.sort(key=lambda n: _bbox_area(n), reverse=True)
    return [dict(x) for x in people[: max(1, int(top_k))]]


def _object_nodes(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for node in list(graph.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        if _node_label(node, "").lower() == "person":
            continue
        out.append(dict(node))
    return out


def _focus_object_nodes(graph: Dict[str, Any], *, top_k: int = 8) -> List[Dict[str, Any]]:
    objs = _object_nodes(graph)
    objs.sort(key=lambda n: _bbox_area(n), reverse=True)
    return [dict(x) for x in objs[: max(1, int(top_k))]]


def _single_person_prompt(*, frame_idx: int, entity_id: str, label: str) -> str:
    return (
        "You are a visual attribute extractor for one PERSON.\n"
        "Input image is a cropped patch for exactly one person instance.\n"
        "Return ONLY JSON object with keys: state, emotion, apparel, action.\n"
        "Never return empty values; use \"unknown\" if uncertain.\n"
        "Example: {\"state\":\"standing\",\"emotion\":\"neutral\",\"apparel\":\"white shirt\",\"action\":\"holding box\"}\n"
        f"frame={int(frame_idx)} entity_id={entity_id} label={label}\n"
        "JSON:"
    )


def _single_object_prompt(*, frame_idx: int, entity_id: str, label: str) -> str:
    return (
        "You are a visual attribute extractor for one NON-PERSON object.\n"
        "Input image is a cropped patch for exactly one object instance.\n"
        "Return ONLY JSON object with keys: color, size, shape, category.\n"
        "Never return empty values; use \"unknown\" if uncertain.\n"
        "Example: {\"color\":\"red\",\"size\":\"small\",\"shape\":\"round\",\"category\":\"ball\"}\n"
        f"frame={int(frame_idx)} entity_id={entity_id} label={label}\n"
        "JSON:"
    )


def _crop_patch_image(
    *,
    image_path: str,
    bbox: List[Any],
    patch_id: str,
    expand_ratio: float = 0.08,
) -> str:
    src = str(image_path or "").strip()
    if not src or not os.path.isfile(src):
        return src
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return src
    try:
        arr = list(bbox or [0, 0, 0, 0])
        if len(arr) < 4:
            return src
        x = float(arr[0] or 0.0)
        y = float(arr[1] or 0.0)
        w = float(arr[2] or 0.0)
        h = float(arr[3] or 0.0)
        if w <= 1.0 or h <= 1.0:
            return src
        img = Image.open(src).convert("RGB")
        iw, ih = img.size
        ex = float(expand_ratio) * w
        ey = float(expand_ratio) * h
        x0 = max(0, int(round(x - ex)))
        y0 = max(0, int(round(y - ey)))
        x1 = min(iw, int(round(x + w + ex)))
        y1 = min(ih, int(round(y + h + ey)))
        if x1 - x0 <= 2 or y1 - y0 <= 2:
            return src
        crop = img.crop((x0, y0, x1, y1))
        out_dir = os.path.join("/tmp", "impact_qwen_attr_patches")
        os.makedirs(out_dir, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(patch_id or "patch"))[:120]
        out_path = os.path.join(out_dir, f"{safe}.jpg")
        crop.save(out_path, format="JPEG", quality=95)
        return out_path
    except Exception:
        return src


def _parse_single_slots(raw: str, slot_aliases: Dict[str, List[str]]) -> Dict[str, str]:
    text = str(raw or "").strip()
    out: Dict[str, str] = {}
    if not text:
        return {k: "unknown" for k in slot_aliases.keys()}

    parsed = _safe_json_extract(text)
    candidate: Dict[str, Any] = {}
    if isinstance(parsed, dict):
        attrs = parsed.get("attributes")
        if isinstance(attrs, dict):
            candidate = dict(attrs)
        else:
            candidate = dict(parsed)
    elif isinstance(parsed, list):
        for row in parsed:
            if isinstance(row, dict):
                attrs = row.get("attributes")
                if isinstance(attrs, dict):
                    candidate = dict(attrs)
                else:
                    candidate = dict(row)
                break

    for slot, aliases in slot_aliases.items():
        val = ""
        for key in [slot, *list(aliases or [])]:
            if key in candidate:
                val = str(candidate.get(key, "") or "").strip()
                if val:
                    break
            lk = str(key).strip().lower()
            for ck, cv in candidate.items():
                if str(ck or "").strip().lower() == lk:
                    val = str(cv or "").strip()
                    if val:
                        break
            if val:
                break
        if not val:
            pat = r"\b(?:%s)\b\s*[:=]\s*([^\n,;|]+)" % "|".join([re.escape(slot), *[re.escape(x) for x in aliases]])
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                val = str(m.group(1) or "").strip()
        out[slot] = val if val else "unknown"
    return out


def _person_prompt(graph: Dict[str, Any]) -> str:
    frame_idx = int(_graph_frame_idx(graph) or 0)
    all_people = _person_nodes(graph)
    people = _focus_person_nodes(graph, top_k=3)
    lines: List[str] = []
    for node in people:
        eid = str(node.get("entity_id", "") or "")
        bbox = list(node.get("bbox") or [0, 0, 0, 0])
        score = float(node.get("score", 0.0) or 0.0)
        area = _bbox_area(node)
        lines.append(f"- entity_id={eid} bbox={bbox[:4]} area={area:.1f} score={score:.3f}")
    roster = "\n".join(lines) if lines else "- none"
    return (
        "You are a visual attribute extractor for PERSON nodes.\n"
        "Focus ONLY on the TOP-3 persons by bbox area in this frame.\n"
        "For each selected person entity_id, output JSON list rows with fixed slots:\n"
        "state, emotion, apparel, action.\n"
        "Rules:\n"
        "1) Return ONLY a JSON array, no markdown, no explanation.\n"
        "2) Use exactly the selected entity_id values.\n"
        "3) Never output empty string. If uncertain, use \"unknown\".\n"
        "Schema:\n"
        "[{\"entity_id\":\"...\",\"attributes\":{\"state\":\"unknown\",\"emotion\":\"unknown\",\"apparel\":\"unknown\",\"action\":\"unknown\"}}]\n\n"
        f"frame={frame_idx}\n"
        f"total_person_nodes={len(all_people)} selected_topk={len(people)}\n"
        f"persons:\n{roster}\n\n"
        "JSON:"
    )


def _object_prompt(graph: Dict[str, Any]) -> str:
    frame_idx = int(_graph_frame_idx(graph) or 0)
    all_objs = _object_nodes(graph)
    objs = _focus_object_nodes(graph, top_k=8)
    lines: List[str] = []
    for node in objs:
        eid = str(node.get("entity_id", "") or "")
        label = _node_label(node, "object").lower()
        bbox = list(node.get("bbox") or [0, 0, 0, 0])
        score = float(node.get("score", 0.0) or 0.0)
        area = _bbox_area(node)
        lines.append(f"- entity_id={eid} label={label} bbox={bbox[:4]} area={area:.1f} score={score:.3f}")
    roster = "\n".join(lines) if lines else "- none"
    return (
        "You are a visual attribute extractor for NON-PERSON object nodes.\n"
        "Focus ONLY on the TOP-8 non-person objects by bbox area in this frame.\n"
        "For each selected object entity_id, output JSON list rows with fixed slots:\n"
        "color, size, shape, category.\n"
        "Rules:\n"
        "1) Return ONLY a JSON array, no markdown, no explanation.\n"
        "2) Use exactly the selected entity_id values.\n"
        "3) Never output empty string. If uncertain, use \"unknown\".\n"
        "Schema:\n"
        "[{\"entity_id\":\"...\",\"attributes\":{\"color\":\"unknown\",\"size\":\"unknown\",\"shape\":\"unknown\",\"category\":\"unknown\"}}]\n\n"
        f"frame={frame_idx}\n"
        f"total_object_nodes={len(all_objs)} selected_topk={len(objs)}\n"
        f"objects:\n{roster}\n\n"
        "JSON:"
    )


def _fallback_person_attrs(graph: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for node in _person_nodes(graph):
        eid = str(node.get("entity_id", "") or "")
        if not eid:
            continue
        out[eid] = {"state": "unknown", "emotion": "unknown", "apparel": "unknown", "action": "unknown"}
    return out


def _fallback_object_attrs(graph: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for node in _object_nodes(graph):
        eid = str(node.get("entity_id", "") or "")
        if not eid:
            continue
        out[eid] = {"color": "unknown", "size": "unknown", "shape": "unknown", "category": "unknown"}
    return out


def _parse_person_attr_response(obj: Any, *, expected_entity_ids: List[str] | None = None) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    expected = [str(x or "").strip() for x in list(expected_entity_ids or []) if str(x or "").strip()]
    expected_set = set(expected)
    unmatched_slots: List[Dict[str, str]] = []

    def _pick_slot(attrs: Dict[str, Any], key: str, *aliases: str) -> str:
        keys = [key, *aliases]
        for k in keys:
            if k in attrs:
                return str(attrs.get(k, "") or "").strip()
            lk = str(k).strip().lower()
            for ak, av in attrs.items():
                if str(ak or "").strip().lower() == lk:
                    return str(av or "").strip()
        return ""

    def _slots_from(attrs: Dict[str, Any]) -> Dict[str, str]:
        return {
            "state": _pick_slot(attrs, "state", "status"),
            "apparel": _pick_slot(attrs, "apparel", "clothing", "cloth", "outfit"),
            "action": _pick_slot(attrs, "action", "activity", "verb"),
            "emotion": _pick_slot(attrs, "emotion", "mood", "feeling"),
        }

    def _norm_eid(row: Dict[str, Any]) -> str:
        for k in ("entity_id", "id", "node_id", "track_id", "person_id"):
            v = str(row.get(k, "") or "").strip()
            if v:
                return v
        return ""

    rows: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        rows = [dict(x) for x in obj if isinstance(x, dict)]
    elif isinstance(obj, dict):
        if isinstance(obj.get("persons"), list):
            rows = [dict(x) for x in list(obj.get("persons") or []) if isinstance(x, dict)]
        elif isinstance(obj.get("results"), list):
            rows = [dict(x) for x in list(obj.get("results") or []) if isinstance(x, dict)]
        else:
            # Mapping style: {"person_1": {"state": "...", ...}, ...}
            for k, v in obj.items():
                if not isinstance(v, dict):
                    continue
                row = dict(v)
                row.setdefault("entity_id", str(k or ""))
                rows.append(row)
    for row in rows:
        attrs_raw = row.get("attributes")
        attrs: Dict[str, Any] = {}
        if isinstance(attrs_raw, dict):
            attrs = dict(attrs_raw)
        elif isinstance(row.get("slots"), dict):
            attrs = dict(row.get("slots") or {})
        else:
            # Flat style row: {"entity_id":"...","state":"...","pose":"..."}
            attrs = dict(row)
        slots = _slots_from(attrs)
        eid = _norm_eid(row)
        if eid and ((not expected_set) or (eid in expected_set)):
            out[eid] = slots
        elif any(str(v or "").strip() for v in slots.values()):
            unmatched_slots.append(slots)

    # If model didn't echo entity_id correctly, map by order to expected ids.
    if expected:
        remaining = [eid for eid in expected if eid not in out]
        for slots in unmatched_slots:
            if not remaining:
                break
            out[remaining.pop(0)] = dict(slots)
    return out


def _parse_person_attr_from_text(raw: str, *, expected_entity_ids: List[str] | None = None) -> Dict[str, Dict[str, str]]:
    text = str(raw or "").strip()
    if not text:
        return {}
    parsed = _safe_json_extract(text)
    out = _parse_person_attr_response(parsed, expected_entity_ids=expected_entity_ids)
    if out:
        return out
    # Fallback parse for loose formats.
    slot_pattern = re.compile(
        r"entity_id\s*[:=]\s*([A-Za-z0-9_\-:.]+).*?"
        r"state\s*[:=]\s*([^,\n;|]+).*?"
        r"emotion\s*[:=]\s*([^,\n;|]+).*?"
        r"apparel\s*[:=]\s*([^,\n;|]+).*?"
        r"action\s*[:=]\s*([^,\n;|]+)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for m in slot_pattern.finditer(text):
        eid = str(m.group(1) or "").strip()
        if not eid:
            continue
        out[eid] = {
            "state": str(m.group(2) or "").strip(),
            "emotion": str(m.group(3) or "").strip(),
            "apparel": str(m.group(4) or "").strip(),
            "action": str(m.group(5) or "").strip(),
        }
    if out:
        return out

    # Last-resort fallback: parse generic "slot: value" block and assign by order.
    if expected_entity_ids:
        slot_re = {
            "state": re.compile(r"\bstate\b\s*[:=]\s*([^\n,;|]+)", re.IGNORECASE),
            "emotion": re.compile(r"\b(emotion|mood|feeling)\b\s*[:=]\s*([^\n,;|]+)", re.IGNORECASE),
            "apparel": re.compile(r"\b(apparel|clothing|outfit)\b\s*[:=]\s*([^\n,;|]+)", re.IGNORECASE),
            "action": re.compile(r"\b(action|activity)\b\s*[:=]\s*([^\n,;|]+)", re.IGNORECASE),
        }
        slots: Dict[str, str] = {"state": "unknown", "emotion": "unknown", "apparel": "unknown", "action": "unknown"}
        for slot, rx in slot_re.items():
            m = rx.search(text)
            if not m:
                continue
            val = m.group(2) if slot in {"apparel", "action", "emotion"} and len(m.groups()) >= 2 else m.group(1)
            slots[slot] = str(val or "").strip()
        if any(str(v or "").strip() for v in slots.values()):
            out[str(expected_entity_ids[0])] = slots
    return out


def _parse_object_attr_response(obj: Any, *, expected_entity_ids: List[str] | None = None) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    expected = [str(x or "").strip() for x in list(expected_entity_ids or []) if str(x or "").strip()]
    expected_set = set(expected)
    unmatched_slots: List[Dict[str, str]] = []

    def _pick_slot(attrs: Dict[str, Any], key: str, *aliases: str) -> str:
        keys = [key, *aliases]
        for k in keys:
            if k in attrs:
                return str(attrs.get(k, "") or "").strip()
            lk = str(k).strip().lower()
            for ak, av in attrs.items():
                if str(ak or "").strip().lower() == lk:
                    return str(av or "").strip()
        return ""

    def _slots_from(attrs: Dict[str, Any]) -> Dict[str, str]:
        return {
            "color": _pick_slot(attrs, "color", "colour"),
            "size": _pick_slot(attrs, "size", "scale"),
            "shape": _pick_slot(attrs, "shape", "form"),
            "category": _pick_slot(attrs, "category", "type", "class", "label"),
        }

    def _norm_eid(row: Dict[str, Any]) -> str:
        for k in ("entity_id", "id", "node_id", "track_id", "object_id"):
            v = str(row.get(k, "") or "").strip()
            if v:
                return v
        return ""

    rows: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        rows = [dict(x) for x in obj if isinstance(x, dict)]
    elif isinstance(obj, dict):
        if isinstance(obj.get("objects"), list):
            rows = [dict(x) for x in list(obj.get("objects") or []) if isinstance(x, dict)]
        elif isinstance(obj.get("results"), list):
            rows = [dict(x) for x in list(obj.get("results") or []) if isinstance(x, dict)]
        else:
            for k, v in obj.items():
                if not isinstance(v, dict):
                    continue
                row = dict(v)
                row.setdefault("entity_id", str(k or ""))
                rows.append(row)
    for row in rows:
        attrs_raw = row.get("attributes")
        attrs: Dict[str, Any] = {}
        if isinstance(attrs_raw, dict):
            attrs = dict(attrs_raw)
        elif isinstance(row.get("slots"), dict):
            attrs = dict(row.get("slots") or {})
        else:
            attrs = dict(row)
        slots = _slots_from(attrs)
        eid = _norm_eid(row)
        if eid and ((not expected_set) or (eid in expected_set)):
            out[eid] = slots
        elif any(str(v or "").strip() for v in slots.values()):
            unmatched_slots.append(slots)
    if expected:
        remaining = [eid for eid in expected if eid not in out]
        for slots in unmatched_slots:
            if not remaining:
                break
            out[remaining.pop(0)] = dict(slots)
    return out


def _parse_object_attr_from_text(raw: str, *, expected_entity_ids: List[str] | None = None) -> Dict[str, Dict[str, str]]:
    text = str(raw or "").strip()
    if not text:
        return {}
    parsed = _safe_json_extract(text)
    out = _parse_object_attr_response(parsed, expected_entity_ids=expected_entity_ids)
    if out:
        return out
    slot_pattern = re.compile(
        r"entity_id\s*[:=]\s*([A-Za-z0-9_\-:.]+).*?"
        r"color\s*[:=]\s*([^,\n;|]+).*?"
        r"size\s*[:=]\s*([^,\n;|]+).*?"
        r"shape\s*[:=]\s*([^,\n;|]+).*?"
        r"category\s*[:=]\s*([^,\n;|]+)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for m in slot_pattern.finditer(text):
        eid = str(m.group(1) or "").strip()
        if not eid:
            continue
        out[eid] = {
            "color": str(m.group(2) or "").strip(),
            "size": str(m.group(3) or "").strip(),
            "shape": str(m.group(4) or "").strip(),
            "category": str(m.group(5) or "").strip(),
        }
    return out


def _upsert_person_attrs(graph: Dict[str, Any], person_attr_map: Dict[str, Dict[str, str]]) -> None:
    nodes = list(graph.get("nodes") or [])
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if _node_label(node, "").lower() != "person":
            continue
        eid = str(node.get("entity_id", "") or "")
        slots = dict(person_attr_map.get(eid) or {})
        if not slots:
            slots = {"state": "unknown", "emotion": "unknown", "apparel": "unknown", "action": "unknown"}
        def _mk(slot: str) -> Dict[str, Any]:
            value = str(slots.get(slot, "") or "").strip()
            if not value:
                value = "unknown"
            return {
                "slot": slot,
                "value": value,
                "confidence": 0.8 if value != "unknown" else 0.2,
                "provenance": "qwen" if value != "unknown" else "qwen_unknown",
                "verified": False,
            }

        llm_attrs = [
            _mk("state"),
            _mk("emotion"),
            _mk("apparel"),
            _mk("action"),
        ]
        node["llm_person_attributes"] = llm_attrs
        attrs = [dict(a) for a in list(node.get("attributes") or []) if isinstance(a, dict)]
        slot_index = {str(a.get("slot", "") or "").strip().lower(): i for i, a in enumerate(attrs)}
        for row in llm_attrs:
            slot = str(row.get("slot", "") or "").strip().lower()
            if slot in slot_index:
                idx = int(slot_index[slot])
                attrs[idx]["value"] = str(row.get("value", "") or "")
                attrs[idx]["confidence"] = float(row.get("confidence", attrs[idx].get("confidence", 0.0)) or 0.0)
                attrs[idx]["provenance"] = str(row.get("provenance", attrs[idx].get("provenance", "qwen")) or "qwen")
            else:
                attrs.append(dict(row))
        node["attributes"] = attrs


def _upsert_object_attrs(graph: Dict[str, Any], object_attr_map: Dict[str, Dict[str, str]]) -> None:
    nodes = list(graph.get("nodes") or [])
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if _node_label(node, "").lower() == "person":
            continue
        eid = str(node.get("entity_id", "") or "")
        slots = dict(object_attr_map.get(eid) or {})
        if not slots:
            slots = {"color": "unknown", "size": "unknown", "shape": "unknown", "category": "unknown"}

        def _mk(slot: str) -> Dict[str, Any]:
            value = str(slots.get(slot, "") or "").strip()
            if not value:
                value = "unknown"
            return {
                "slot": slot,
                "value": value,
                "confidence": 0.8 if value != "unknown" else 0.2,
                "provenance": "qwen" if value != "unknown" else "qwen_unknown",
                "verified": False,
            }

        llm_attrs = [
            _mk("color"),
            _mk("size"),
            _mk("shape"),
            _mk("category"),
        ]
        node["llm_object_attributes"] = llm_attrs
        attrs = [dict(a) for a in list(node.get("attributes") or []) if isinstance(a, dict)]
        slot_index = {str(a.get("slot", "") or "").strip().lower(): i for i, a in enumerate(attrs)}
        for row in llm_attrs:
            slot = str(row.get("slot", "") or "").strip().lower()
            if slot in slot_index:
                idx = int(slot_index[slot])
                attrs[idx]["value"] = str(row.get("value", "") or "")
                attrs[idx]["confidence"] = float(row.get("confidence", attrs[idx].get("confidence", 0.0)) or 0.0)
                attrs[idx]["provenance"] = str(row.get("provenance", attrs[idx].get("provenance", "qwen")) or "qwen")
            else:
                attrs.append(dict(row))
        node["attributes"] = attrs


def _normalize_graph_attribute_schema(graph: Dict[str, Any]) -> None:
    nodes = [dict(n) for n in list(graph.get("nodes") or []) if isinstance(n, dict)]
    if not nodes:
        graph["nodes"] = nodes
        return

    for node in nodes:
        label = _node_label(node, "").lower()
        target_slots = (
            ["state", "emotion", "apparel", "action"]
            if label == "person"
            else ["color", "size", "shape", "category"]
        )
        attrs = [dict(a) for a in list(node.get("attributes") or []) if isinstance(a, dict)]
        slot_to_row: Dict[str, Dict[str, Any]] = {}
        for row in attrs:
            slot = str(row.get("slot", "") or "").strip().lower()
            if not slot:
                continue
            slot_to_row[slot] = dict(row)
        normalized: List[Dict[str, Any]] = []
        for slot in target_slots:
            row = dict(slot_to_row.get(slot) or {})
            if row:
                row["slot"] = slot
                value = str(row.get("value", "") or "").strip()
                if not value:
                    value = "unknown"
                    row["confidence"] = max(0.2, float(row.get("confidence", 0.0) or 0.0))
                    prev_prov = str(row.get("provenance", "") or "").strip().lower()
                    if prev_prov in {"qwen", "qwen_empty", "missing", ""}:
                        row["provenance"] = "qwen_unknown"
                else:
                    row["confidence"] = float(row.get("confidence", 0.0) or 0.0)
                    row["provenance"] = str(row.get("provenance", "llm") or "llm")
                row["value"] = value
                row["verified"] = bool(row.get("verified", False))
            else:
                row = {
                    "slot": slot,
                    "value": "unknown",
                    "confidence": 0.2,
                    "provenance": "missing",
                    "verified": False,
                }
            normalized.append(row)
        node["attributes"] = normalized
    graph["nodes"] = nodes


class QwenTextRunner:
    def __init__(self, model_path: str, *, require_cuda: bool = False, device_hint: str = ""):
        try:
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("transformers + torch are required for LLM summaries.") from exc
        self._torch = torch
        self._require_cuda = bool(require_cuda)
        self._device_hint = str(device_hint or "").strip().lower()
        self._backend = "text"
        self._tokenizer = None
        self._processor = None
        self._image_cls = None
        cuda_available = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
        print(
            "[LLM-SUMMARY][DEVICE] "
            f"hint={self._device_hint or 'auto'} "
            f"cuda_available={cuda_available} "
            f"device_count={cuda_count} "
            f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
        )
        if self._require_cuda and (not cuda_available):
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False in qwen env.")
        model_cfg_path = os.path.join(model_path, "config.json")
        arch_text = ""
        try:
            with open(model_cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            archs = list(cfg.get("architectures") or [])
            arch_text = " ".join([str(x) for x in archs]).lower()
        except Exception:
            arch_text = ""
        prefers_vl = ("_vl" in arch_text) or ("vl" in arch_text) or ("-vl" in str(model_path).lower())

        if prefers_vl:
            try:
                from PIL import Image  # type: ignore
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore

                self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    device_map="auto",
                )
                self._image_cls = Image
                self._backend = "vl"
            except Exception as exc:
                err = str(exc or "")
                if (
                    "Qwen2_5_VLForConditionalGeneration" in err
                    or "qwen2_5_vl" in err.lower()
                    or "does not recognize this architecture" in err.lower()
                ):
                    raise RuntimeError(
                        "Qwen2.5-VL requires newer transformers in the qwen env. "
                        "Please run: conda run -n qwen pip install -U 'transformers>=4.51.0' "
                        "'accelerate>=0.33.0' 'qwen-vl-utils'"
                    ) from exc
                print(f"[LLM-SUMMARY][WARN] VL init failed, fallback to text backend: {exc}")

        if self._backend != "vl":
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            self._backend = "text"
            print("[LLM-SUMMARY][WARN] Loaded text-only Qwen backend; image input is not available for attributes.")

        self._device = next(self._model.parameters()).device
        print(f"[LLM-SUMMARY][DEVICE] selected={self._device} backend={self._backend}")
        if self._require_cuda and getattr(self._device, "type", "") != "cuda":
            raise RuntimeError(f"CUDA requested but model loaded on device={self._device}.")

    def summarize(self, prompt: str, *, max_new_tokens: int = 96, temperature: float = 0.2, image_path: str = "") -> str:
        if self._backend == "vl" and self._processor is not None and self._image_cls is not None:
            abs_image = os.path.abspath(os.path.expanduser(str(image_path or "").strip())) if image_path else ""
            has_image = bool(abs_image and os.path.isfile(abs_image))
            content: List[Dict[str, str]] = []
            if has_image:
                content.append({"type": "image", "image": abs_image})
            content.append({"type": "text", "text": str(prompt or "")})
            messages = [{"role": "user", "content": content}]
            prompt_text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if has_image:
                image = self._image_cls.open(abs_image).convert("RGB")
                inputs = self._processor(
                    text=[prompt_text],
                    images=[image],
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = self._processor(
                    text=[prompt_text],
                    padding=True,
                    return_tensors="pt",
                )
            inputs = inputs.to(self._device)
            with self._torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max(16, int(max_new_tokens)),
                    do_sample=bool(float(temperature) > 0.0),
                    temperature=max(0.0, float(temperature)),
                )
            trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            text = self._processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return str(text or "").strip()

        tok = None
        # Prefer chat template for instruct models (Qwen2.5-Instruct).
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                chat_text = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": str(prompt or "")}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                tok = self._tokenizer(chat_text, return_tensors="pt")
            except Exception:
                tok = None
        if tok is None:
            tok = self._tokenizer(prompt, return_tensors="pt")
        input_ids = tok["input_ids"].to(self._device)
        attn = tok.get("attention_mask")
        if attn is not None:
            attn = attn.to(self._device)
        with self._torch.no_grad():
            out_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max(16, int(max_new_tokens)),
                do_sample=bool(float(temperature) > 0.0),
                temperature=max(0.0, float(temperature)),
            )
        gen_ids = out_ids[0][input_ids.shape[-1] :]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        return str(text or "").strip()


def _looks_like_hf_model_dir(path: str) -> bool:
    p = os.path.abspath(os.path.expanduser(str(path or "").strip()))
    if not p or (not os.path.isdir(p)):
        return False
    return os.path.isfile(os.path.join(p, "config.json"))


def _fallback_summary(chunk_graphs: List[Dict[str, Any]]) -> str:
    if not chunk_graphs:
        return "No scene content."
    labels: Dict[str, int] = {}
    for graph in chunk_graphs:
        for node in list(graph.get("nodes") or []):
            if not isinstance(node, dict):
                continue
            label = str(node.get("canonical_label", "object") or "object").strip().lower() or "object"
            labels[label] = int(labels.get(label, 0) + 1)
    top = sorted(labels.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:3]
    label_text = ", ".join([f"{k} x{v}" for k, v in top]) if top else "objects"
    return f"This segment mainly shows {label_text} with stable scene composition."


def _make_prompt(chunk_graphs: List[Dict[str, Any]]) -> str:
    lines = [_frame_digest(graph) for graph in chunk_graphs]
    table = "\n".join(lines)
    return (
        "You are a video understanding assistant.\n"
        "Given N-frame scene-graph digests, produce ONE concise global semantic summary sentence.\n"
        "Focus on dominant actors, actions, and spatial context continuity.\n"
        "Do not output JSON.\n\n"
        f"{table}\n\n"
        "Summary:"
    )


def _compact_scene_graph_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    src = dict(bundle or {})
    out: Dict[str, Any] = {
        "type": str(src.get("type", "scene_graph_sequence") or "scene_graph_sequence"),
        "version": int(src.get("version", 1) or 1),
        "video_path": str(src.get("video_path", "") or ""),
        "video_name": str(src.get("video_name", "") or ""),
        "source_fps": float(src.get("source_fps", 0.0) or 0.0),
        "sampling_fps": float(src.get("sampling_fps", 0.0) or 0.0),
        "sampled_frame_indices": [int(x) for x in list(src.get("sampled_frame_indices") or [])],
    }
    if isinstance(src.get("validation"), dict):
        out["validation"] = dict(src.get("validation") or {})

    compact_graphs: List[Dict[str, Any]] = []
    for graph in list(src.get("graphs") or []):
        if not isinstance(graph, dict):
            continue
        meta = dict(graph.get("metadata") or {})
        cmeta: Dict[str, Any] = {
            "graph_frame_idx": int(meta.get("graph_frame_idx", graph.get("frame_idx", 0)) or 0),
            "graph_time_sec": float(meta.get("graph_time_sec", 0.0) or 0.0),
            "image_path": str(meta.get("image_path", "") or graph.get("image_path", "") or ""),
            "global_summary": str(meta.get("global_summary", "") or ""),
            "global_semantic_summary": str(meta.get("global_semantic_summary", "") or ""),
            "stage_summary_statements": [str(x) for x in list(meta.get("stage_summary_statements") or []) if str(x or "").strip()],
        }

        cnodes: List[Dict[str, Any]] = []
        for node in list(graph.get("nodes") or []):
            if not isinstance(node, dict):
                continue
            attrs = [dict(a) for a in list(node.get("attributes") or []) if isinstance(a, dict)]
            canonical = _node_label(node, "")
            cnodes.append(
                {
                    "entity_id": str(node.get("entity_id", "") or ""),
                    "canonical_label": str(canonical or ""),
                    "label": str(node.get("label", canonical) or canonical or ""),
                    "bbox": list(node.get("bbox") or [0, 0, 0, 0])[:4],
                    "score": float(node.get("score", node.get("confidence", 0.0)) or 0.0),
                    "attributes": attrs,
                }
            )

        cedges: List[Dict[str, Any]] = []
        for edge in list(graph.get("edges") or []):
            if not isinstance(edge, dict):
                continue
            cedges.append(
                {
                    "src_id": str(edge.get("src_id", "") or ""),
                    "relation": str(edge.get("relation", "") or ""),
                    "dst_id": str(edge.get("dst_id", "") or ""),
                    "score": float(edge.get("score", edge.get("confidence", 0.0)) or 0.0),
                }
            )

        cgraph: Dict[str, Any] = {
            "image_id": str(graph.get("image_id", "") or ""),
            "metadata": cmeta,
            "nodes": cnodes,
            "edges": cedges,
        }
        if isinstance(graph.get("validation"), dict):
            cgraph["validation"] = dict(graph.get("validation") or {})
        compact_graphs.append(cgraph)

    out["graphs"] = compact_graphs

    person_rows = [dict(x) for x in list(src.get("llm_person_attributes") or []) if isinstance(x, dict)]
    out["llm_person_attributes"] = [
        {
            "frame_idx": int(r.get("frame_idx", 0) or 0),
            "num_person_nodes": int(r.get("num_person_nodes", 0) or 0),
            "num_person_with_nonempty_attrs": int(r.get("num_person_with_nonempty_attrs", 0) or 0),
            "source": str(r.get("source", "") or ""),
        }
        for r in person_rows
    ]

    object_rows = [dict(x) for x in list(src.get("llm_object_attributes") or []) if isinstance(x, dict)]
    out["llm_object_attributes"] = [
        {
            "frame_idx": int(r.get("frame_idx", 0) or 0),
            "num_object_nodes": int(r.get("num_object_nodes", 0) or 0),
            "num_object_with_nonempty_attrs": int(r.get("num_object_with_nonempty_attrs", 0) or 0),
            "source": str(r.get("source", "") or ""),
        }
        for r in object_rows
    ]

    out["llm_batch_summaries"] = [dict(x) for x in list(src.get("llm_batch_summaries") or []) if isinstance(x, dict)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate LLM global semantic summary every N frames for scene graph bundle.")
    ap.add_argument("--bundle_json", required=True, help="Input scene graph bundle JSON.")
    ap.add_argument("--model_path", required=True, help="Local Qwen model path.")
    ap.add_argument("--batch_size", type=int, default=3, help="Frames per summary batch.")
    ap.add_argument("--person_attr", type=int, default=1, help="Enable per-frame person attribute extraction (1/0).")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--out", default="", help="Output bundle JSON path (default: overwrite input).")
    ap.add_argument("--device_hint", default="", help="Device hint, e.g. cuda:1.")
    ap.add_argument("--require_cuda", type=int, default=0, help="Fail if CUDA is unavailable (1/0).")
    args = ap.parse_args()

    in_path = os.path.abspath(os.path.expanduser(str(args.bundle_json)))
    if not os.path.isfile(in_path):
        raise SystemExit(f"[LLM-SUMMARY][ERROR] bundle not found: {in_path}")
    out_path = str(args.out or "").strip()
    if not out_path:
        out_path = in_path
    out_path = os.path.abspath(os.path.expanduser(out_path))

    with open(in_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    if not isinstance(bundle, dict):
        raise SystemExit("[LLM-SUMMARY][ERROR] bundle JSON must be an object.")
    graphs = [g for g in list(bundle.get("graphs") or []) if isinstance(g, dict)]
    if not graphs:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=True, indent=2)
        print(f"[LLM-SUMMARY][WARN] no graphs found, wrote unchanged bundle: {out_path}")
        return 0

    model_path = str(args.model_path or "").strip()
    runner: QwenTextRunner | None = None
    if not _looks_like_hf_model_dir(model_path):
        print(
            f"[LLM-SUMMARY][WARN] invalid model_path={model_path} (missing config.json). "
            "Using fallback summaries/attributes only."
        )
    else:
        try:
            runner = QwenTextRunner(
                model_path=model_path,
                require_cuda=bool(int(args.require_cuda)),
                device_hint=str(args.device_hint or ""),
            )
        except Exception as exc:
            if bool(int(args.require_cuda)):
                raise SystemExit(f"[LLM-SUMMARY][ERROR] {exc}")
            print(
                f"[LLM-SUMMARY][WARN] failed to load model from {model_path}: {exc}. "
                "Using fallback summaries/attributes only."
            )
            runner = None

    # Step 1: per-frame person/object attributes (resume-aware).
    do_person_attr = bool(int(args.person_attr))
    existing_person_rows = [dict(x) for x in list(bundle.get("llm_person_attributes") or []) if isinstance(x, dict)]
    person_done: Dict[int, Dict[str, Any]] = {}
    for row in existing_person_rows:
        try:
            fidx = int(row.get("frame_idx", -1) or -1)
        except Exception:
            fidx = -1
        if fidx >= 0:
            person_done[fidx] = dict(row)
    person_rows: List[Dict[str, Any]] = list(person_done.values())
    existing_object_rows = [dict(x) for x in list(bundle.get("llm_object_attributes") or []) if isinstance(x, dict)]
    object_done: Dict[int, Dict[str, Any]] = {}
    for row in existing_object_rows:
        try:
            fidx = int(row.get("frame_idx", -1) or -1)
        except Exception:
            fidx = -1
        if fidx >= 0:
            object_done[fidx] = dict(row)
    object_rows: List[Dict[str, Any]] = list(object_done.values())

    if do_person_attr:
        total_frames = len(graphs)
        for fi, graph in enumerate(graphs, start=1):
            frame_idx = int(_graph_frame_idx(graph) or 0)
            people = _person_nodes(graph)
            image_path = str((graph.get("metadata") or {}).get("image_path", "") or "").strip()
            prev_person = dict(person_done.get(frame_idx) or {})
            person_need_refresh = (
                frame_idx not in person_done
                or str(prev_person.get("source", "") or "").strip().lower() in {"fallback", "fallback_no_model"}
                or int(prev_person.get("num_person_with_nonempty_attrs", 0) or 0) <= 0
            )
            if people and person_need_refresh:
                focused_people = _focus_person_nodes(graph, top_k=3)
                attr_map: Dict[str, Dict[str, str]] = {}
                raw_preview_rows: List[str] = []
                if runner is not None:
                    for node in focused_people:
                        eid = str(node.get("entity_id", "") or "").strip()
                        if not eid:
                            continue
                        patch = _crop_patch_image(
                            image_path=image_path,
                            bbox=list(node.get("bbox") or [0, 0, 0, 0]),
                            patch_id=f"person_{frame_idx}_{eid}",
                        )
                        prompt = _single_person_prompt(
                            frame_idx=frame_idx,
                            entity_id=eid,
                            label=_node_label(node, "person"),
                        )
                        raw = ""
                        try:
                            raw = runner.summarize(
                                prompt,
                                max_new_tokens=max(96, int(args.max_new_tokens)),
                                temperature=float(args.temperature),
                                image_path=patch or image_path,
                            )
                        except Exception:
                            raw = ""
                        slots = _parse_single_slots(
                            raw,
                            slot_aliases={
                                "state": ["status"],
                                "emotion": ["mood", "feeling"],
                                "apparel": ["clothing", "outfit"],
                                "action": ["activity", "verb"],
                            },
                        )
                        attr_map[eid] = dict(slots)
                        raw_preview_rows.append(f"{eid}:{str(raw or '')[:80]}")
                if not attr_map:
                    attr_map = _fallback_person_attrs(graph)
                non_empty_count = 0
                for slots in attr_map.values():
                    if any(str(v or "").strip() and str(v or "").strip().lower() != "unknown" for v in dict(slots or {}).values()):
                        non_empty_count += 1
                _upsert_person_attrs(graph, attr_map)
                person_done[int(frame_idx)] = {
                    "frame_idx": int(frame_idx),
                    "num_person_nodes": int(len(people)),
                    "num_person_nodes_topk": int(len(focused_people)),
                    "num_person_with_attrs": int(len(attr_map)),
                    "num_person_with_nonempty_attrs": int(non_empty_count),
                    "source": "qwen" if non_empty_count > 0 else ("fallback" if runner is not None else "fallback_no_model"),
                    "raw_preview": " | ".join(raw_preview_rows)[:240],
                }
                person_rows = list(person_done.values())
                print(f"[LLM-ATTR][PERSON] frame {fi}/{total_frames} frame_idx={frame_idx} persons={len(people)}")

            objs = _object_nodes(graph)
            prev_obj = dict(object_done.get(frame_idx) or {})
            object_need_refresh = (
                frame_idx not in object_done
                or str(prev_obj.get("source", "") or "").strip().lower() in {"fallback", "fallback_no_model"}
                or int(prev_obj.get("num_object_with_nonempty_attrs", 0) or 0) <= 0
            )
            if objs and object_need_refresh:
                focused_objs = _focus_object_nodes(graph, top_k=8)
                obj_map: Dict[str, Dict[str, str]] = {}
                raw_obj_rows: List[str] = []
                if runner is not None:
                    for node in focused_objs:
                        eid = str(node.get("entity_id", "") or "").strip()
                        if not eid:
                            continue
                        patch = _crop_patch_image(
                            image_path=image_path,
                            bbox=list(node.get("bbox") or [0, 0, 0, 0]),
                            patch_id=f"object_{frame_idx}_{eid}",
                        )
                        prompt_obj = _single_object_prompt(
                            frame_idx=frame_idx,
                            entity_id=eid,
                            label=_node_label(node, "object"),
                        )
                        raw_obj = ""
                        try:
                            raw_obj = runner.summarize(
                                prompt_obj,
                                max_new_tokens=max(96, int(args.max_new_tokens)),
                                temperature=float(args.temperature),
                                image_path=patch or image_path,
                            )
                        except Exception:
                            raw_obj = ""
                        slots = _parse_single_slots(
                            raw_obj,
                            slot_aliases={
                                "color": ["colour"],
                                "size": ["scale"],
                                "shape": ["form"],
                                "category": ["type", "class", "label"],
                            },
                        )
                        obj_map[eid] = dict(slots)
                        raw_obj_rows.append(f"{eid}:{str(raw_obj or '')[:80]}")
                if not obj_map:
                    obj_map = _fallback_object_attrs(graph)
                non_empty_obj = 0
                for slots in obj_map.values():
                    if any(str(v or "").strip() and str(v or "").strip().lower() != "unknown" for v in dict(slots or {}).values()):
                        non_empty_obj += 1
                _upsert_object_attrs(graph, obj_map)
                object_done[int(frame_idx)] = {
                    "frame_idx": int(frame_idx),
                    "num_object_nodes": int(len(objs)),
                    "num_object_nodes_topk": int(len(focused_objs)),
                    "num_object_with_attrs": int(len(obj_map)),
                    "num_object_with_nonempty_attrs": int(non_empty_obj),
                    "source": "qwen" if non_empty_obj > 0 else ("fallback" if runner is not None else "fallback_no_model"),
                    "raw_preview": " | ".join(raw_obj_rows)[:240],
                }
                object_rows = list(object_done.values())
                print(f"[LLM-ATTR][OBJECT] frame {fi}/{total_frames} frame_idx={frame_idx} objects={len(objs)}")

            _normalize_graph_attribute_schema(graph)

    # Step 2: global summaries every 5 frames (resume-aware).
    chunks = _chunk(graphs, int(args.batch_size))
    existing_batch_rows = [dict(x) for x in list(bundle.get("llm_batch_summaries") or []) if isinstance(x, dict)]
    batch_done: Dict[tuple[int, int], Dict[str, Any]] = {}
    for row in existing_batch_rows:
        try:
            key = (int(row.get("start_frame", -1) or -1), int(row.get("end_frame", -1) or -1))
        except Exception:
            key = (-1, -1)
        if key[0] >= 0 and key[1] >= 0:
            batch_done[key] = dict(row)
    batch_rows: List[Dict[str, Any]] = list(batch_done.values())
    for i, chunk_graphs in enumerate(chunks, start=1):
        start_frame = int(_graph_frame_idx(chunk_graphs[0]) or 0)
        end_frame = int(_graph_frame_idx(chunk_graphs[-1]) or 0)
        chunk_key = (int(start_frame), int(end_frame))
        if chunk_key in batch_done:
            continue
        prompt = _make_prompt(chunk_graphs)
        text = ""
        if runner is not None:
            try:
                text = runner.summarize(
                    prompt,
                    max_new_tokens=int(args.max_new_tokens),
                    temperature=float(args.temperature),
                )
            except Exception:
                text = ""
        if not text:
            text = _fallback_summary(chunk_graphs)
        text = " ".join(str(text).replace("\n", " ").split()).strip()
        for graph in chunk_graphs:
            meta = dict(graph.get("metadata") or {})
            meta["global_summary"] = text
            meta["global_semantic_summary"] = text
            meta["stage_summary_statements"] = [text]
            graph["metadata"] = meta
            _normalize_graph_attribute_schema(graph)
        batch_done[chunk_key] = {
            "batch_index": int(i),
            "start_frame": int(start_frame),
            "end_frame": int(end_frame),
            "num_frames": int(len(chunk_graphs)),
            "summary": text,
        }
        batch_rows = list(batch_done.values())
        print(f"[LLM-SUMMARY] batch {i}/{len(chunks)} start={start_frame} end={end_frame}")

    bundle["graphs"] = graphs
    bundle["llm_person_attributes"] = sorted(person_rows, key=lambda r: int(r.get("frame_idx", 0) or 0))
    bundle["llm_object_attributes"] = sorted(object_rows, key=lambda r: int(r.get("frame_idx", 0) or 0))
    bundle["llm_batch_summaries"] = sorted(
        batch_rows,
        key=lambda r: (int(r.get("start_frame", 0) or 0), int(r.get("end_frame", 0) or 0)),
    )
    bundle = _compact_scene_graph_bundle(bundle)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=True, indent=2)
    print(f"[LLM-SUMMARY][OK] output={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
