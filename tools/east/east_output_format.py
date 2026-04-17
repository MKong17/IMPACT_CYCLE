from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def normalize_label_bank(label_bank: Optional[Iterable[Any]]) -> List[str]:
    out: List[str] = []
    if not label_bank:
        return out
    for item in label_bank:
        name = str(item).strip() if item is not None else ""
        if not name:
            continue
        if name not in out:
            out.append(name)
    return out


def _int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, float) and not float(v).is_integer():
        return None
    try:
        return int(v)
    except Exception:
        return None


def normalize_segments(raw_segments: Sequence[Dict[str, Any]], classes: Sequence[str]) -> List[Dict[str, Any]]:
    cls = list(classes or [])
    out: List[Dict[str, Any]] = []
    for seg in raw_segments or []:
        sf = _int_or_none(seg.get("start_frame", seg.get("start")))
        ef = _int_or_none(seg.get("end_frame", seg.get("end")))
        if sf is None or ef is None or ef < sf:
            continue
        cid = _int_or_none(seg.get("class_id"))
        cname = seg.get("class_name", seg.get("label"))
        if cname is None and cid is not None and 0 <= cid < len(cls):
            cname = cls[cid]
        cname = str(cname).strip() if cname is not None else ""
        if cid is None and cname and cname in cls:
            cid = cls.index(cname)
        if cid is None:
            cid = 0
        if not cname:
            cname = cls[cid] if 0 <= cid < len(cls) else f"cls_{cid}"

        item: Dict[str, Any] = {
            "start_frame": int(sf),
            "end_frame": int(ef),
            "class_id": int(cid),
            "class_name": cname,
        }
        topk = seg.get("topk")
        if isinstance(topk, list) and topk:
            cleaned_topk = []
            for tk in topk:
                if not isinstance(tk, dict):
                    continue
                tid = _int_or_none(tk.get("id"))
                tname = tk.get("name")
                if tname is None and tid is not None and 0 <= tid < len(cls):
                    tname = cls[tid]
                if tname is None:
                    continue
                score = tk.get("score")
                try:
                    score = float(score) if score is not None else None
                except Exception:
                    score = None
                cleaned_topk.append({
                    "id": int(tid) if tid is not None else (cls.index(tname) if tname in cls else -1),
                    "name": str(tname),
                    "score": score,
                })
            if cleaned_topk:
                item["topk"] = cleaned_topk
        out.append(item)

    out.sort(key=lambda x: (int(x.get("start_frame", 0)), int(x.get("end_frame", 0))))
    return out


def build_segments_payload(
    segments: Sequence[Dict[str, Any]],
    classes: Sequence[str],
    source: str = "EAST",
) -> Dict[str, Any]:
    cls = normalize_label_bank(classes)
    norm_segments = normalize_segments(list(segments or []), cls)
    return {
        "source": str(source),
        "classes": cls,
        "segments": norm_segments,
    }


def validate_segments(
    segments_json: Dict[str, Any],
    T: int,
    fps: float,
    label_list: Sequence[str],
) -> Tuple[bool, str]:
    _ = fps  # reserved for future second/frame conversions
    if not isinstance(segments_json, dict):
        return False, "segments_json must be a dict"

    segs = segments_json.get("segments")
    if not isinstance(segs, list) or not segs:
        return False, "segments_json.segments must be a non-empty list"

    label_set = set(normalize_label_bank(label_list))
    cls = normalize_label_bank(segments_json.get("classes") or [])
    label_set.update(cls)

    if T <= 0:
        return False, "feature length T must be > 0"

    prev_end = -1
    prev_start = -1
    for idx, seg in enumerate(segs):
        if not isinstance(seg, dict):
            return False, f"segment[{idx}] must be a dict"

        s_raw = seg.get("start_frame", seg.get("start"))
        e_raw = seg.get("end_frame", seg.get("end"))
        if isinstance(s_raw, float) and not float(s_raw).is_integer():
            return False, f"segment[{idx}] start is not a frame index (float detected)"
        if isinstance(e_raw, float) and not float(e_raw).is_integer():
            return False, f"segment[{idx}] end is not a frame index (float detected)"

        s = _int_or_none(s_raw)
        e = _int_or_none(e_raw)
        if s is None or e is None:
            return False, f"segment[{idx}] start/end must be integer frame indices"
        if s < 0 or e < 0:
            return False, f"segment[{idx}] has negative frame index"
        if s > e:
            return False, f"segment[{idx}] violates closed interval rule: start > end"
        if e >= T:
            return False, f"segment[{idx}] end {e} out of range for T={T}"

        if idx > 0 and (s < prev_start or s <= prev_end):
            return False, f"segment[{idx}] overlaps or is unsorted"
        prev_start = s
        prev_end = e

        cname = seg.get("class_name", seg.get("label"))
        cid = _int_or_none(seg.get("class_id"))
        if cname is None and cid is None:
            return False, f"segment[{idx}] missing both class_name and class_id"
        if cname is not None:
            cname = str(cname).strip()
            if not cname:
                return False, f"segment[{idx}] has empty class_name"
            if label_set and cname not in label_set:
                return False, f"segment[{idx}] class_name '{cname}' not in label bank"
        if cid is not None and cid < 0:
            return False, f"segment[{idx}] class_id must be >= 0"

    return True, "ok"


def write_segments_outputs(features_dir: str, out_prefix: str, payload: Dict[str, Any]) -> Tuple[str, str]:
    os.makedirs(features_dir, exist_ok=True)
    json_path = os.path.join(features_dir, f"{out_prefix}_segments.json")
    txt_path = os.path.join(features_dir, f"{out_prefix}_segments.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in payload.get("segments", []):
            f.write(f"{int(seg['start_frame'])}\n")
            f.write(f"{int(seg['end_frame'])}\n")
            f.write(f"{seg['class_name']}\n\n")

    return txt_path, json_path
