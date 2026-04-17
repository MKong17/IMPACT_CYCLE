"""Convert legacy action-seg JSON to current native JSON schema.

Supported legacy segment fields:
- label id: action_label | label_id | id
- label name: label | label_name | action_name | name
- start/end: start_frame/end_frame | f_start/f_end | start/end

Output segment fields (required by current loader):
- action_label (int)
- start_frame (int)
- end_frame (int)

Optional fields are preserved when present:
- entity
- phase
- anomaly_type (list)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object.")
    return data


def _parse_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _parse_span(seg: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    candidates = [
        ("start_frame", "end_frame"),
        ("f_start", "f_end"),
        ("start", "end"),
    ]
    for sk, ek in candidates:
        sv = seg.get(sk)
        ev = seg.get(ek)
        if sv is None and ev is None:
            continue
        s = _parse_int(sv if sv is not None else ev)
        e = _parse_int(ev if ev is not None else sv)
        if s is None or e is None:
            continue
        if e < s:
            s, e = e, s
        return s, e
    return None


def _normalize_labels(data: Dict[str, Any]) -> Tuple[Dict[int, str], Dict[str, int], List[Dict[str, Any]]]:
    raw = data.get("labels")
    if not isinstance(raw, list):
        raw = data.get("action_labels")
    if not isinstance(raw, list):
        raw = []

    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}
    labels_out: List[Dict[str, Any]] = []

    for item in raw:
        if not isinstance(item, dict):
            continue
        lid = _parse_int(item.get("id"))
        if lid is None or lid < 0:
            continue
        name = str(item.get("name", f"Label_{lid}")).strip() or f"Label_{lid}"
        color = item.get("color")
        if not color:
            color = "Gray"
        if lid not in id_to_name:
            id_to_name[lid] = name
            name_to_id[name] = lid
            labels_out.append({"id": lid, "name": name, "color": color})

    labels_out.sort(key=lambda x: int(x["id"]))
    return id_to_name, name_to_id, labels_out


def _ensure_label(
    lid: Optional[int],
    lname: Optional[str],
    id_to_name: Dict[int, str],
    name_to_id: Dict[str, int],
    labels_out: List[Dict[str, Any]],
) -> Optional[int]:
    if lid is not None and lid >= 0:
        if lid not in id_to_name:
            name = (lname or "").strip() or f"Label_{lid}"
            id_to_name[lid] = name
            name_to_id[name] = lid
            labels_out.append({"id": lid, "name": name, "color": "Gray"})
        return lid

    if lname:
        name = lname.strip()
        if not name:
            return None
        if name in name_to_id:
            return name_to_id[name]
        next_id = max(id_to_name.keys(), default=-1) + 1
        id_to_name[next_id] = name
        name_to_id[name] = next_id
        labels_out.append({"id": next_id, "name": name, "color": "Gray"})
        return next_id
    return None


def _normalize_segments(
    segs: Any,
    id_to_name: Dict[int, str],
    name_to_id: Dict[str, int],
    labels_out: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not isinstance(segs, list):
        return [], ["'segments' must be a list"]

    out: List[Dict[str, Any]] = []
    issues: List[str] = []

    for idx, seg in enumerate(segs):
        if not isinstance(seg, dict):
            issues.append(f"segment[{idx}] is not an object")
            continue

        span = _parse_span(seg)
        if span is None:
            issues.append(f"segment[{idx}] missing valid start/end")
            continue
        s, e = span

        lid = _parse_int(seg.get("action_label"))
        if lid is None:
            lid = _parse_int(seg.get("label_id"))
        if lid is None:
            lid = _parse_int(seg.get("id"))

        lname = None
        for key in ("label", "label_name", "action_name", "name"):
            raw = seg.get(key)
            if raw is None:
                continue
            name = str(raw).strip()
            if name:
                lname = name
                break

        lid = _ensure_label(lid, lname, id_to_name, name_to_id, labels_out)
        if lid is None:
            issues.append(f"segment[{idx}] missing label id/name")
            continue

        item: Dict[str, Any] = {
            "action_label": int(lid),
            "start_frame": int(s),
            "end_frame": int(e),
        }

        entity = seg.get("entity")
        if entity not in (None, ""):
            item["entity"] = str(entity)
        phase = seg.get("phase")
        if phase not in (None, ""):
            item["phase"] = str(phase)
        anomaly_type = seg.get("anomaly_type")
        if isinstance(anomaly_type, list):
            item["anomaly_type"] = anomaly_type

        out.append(item)

    return out, issues


def convert_payload(data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    id_to_name, name_to_id, labels_out = _normalize_labels(data)
    segs_out, issues = _normalize_segments(
        data.get("segments"), id_to_name, name_to_id, labels_out
    )
    labels_out.sort(key=lambda x: int(x["id"]))

    out: Dict[str, Any] = {
        "video_id": data.get("video_id", ""),
        "view": data.get("view", ""),
        "view_start": _parse_int(data.get("view_start")) or 0,
        "view_end": _parse_int(data.get("view_end")) or 0,
        "labels": labels_out,
        "segments": segs_out,
    }
    if isinstance(data.get("meta_data"), dict):
        out["meta_data"] = data["meta_data"]
    if isinstance(data.get("anomaly_types"), list):
        out["anomaly_types"] = data["anomaly_types"]
    if isinstance(data.get("verbs"), list):
        out["verbs"] = data["verbs"]
    if isinstance(data.get("nouns"), list):
        out["nouns"] = data["nouns"]
    return out, issues


def _iter_inputs(inp: Path) -> List[Path]:
    if inp.is_dir():
        return sorted([p for p in inp.rglob("*.json") if p.is_file()])
    return [inp]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Legacy json file or folder")
    ap.add_argument(
        "--out_dir",
        default="",
        help="Output folder (default: next to input file)",
    )
    ap.add_argument(
        "--suffix",
        default="_native",
        help="Output filename suffix before .json",
    )
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input does not exist: {inp}")

    files = _iter_inputs(inp)
    if not files:
        raise SystemExit("No JSON files found.")

    out_dir_arg = Path(args.out_dir) if args.out_dir else None
    converted = 0

    for fp in files:
        try:
            data = _load_json(fp)
            out, issues = convert_payload(data)
            if not out["segments"]:
                raise ValueError("No valid segments after conversion.")

            out_dir = out_dir_arg if out_dir_arg else fp.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{fp.stem}{args.suffix}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

            report_path = out_dir / f"{fp.stem}{args.suffix}.report.json"
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "source": str(fp),
                        "target": str(out_path),
                        "segments_in": len(data.get("segments", []))
                        if isinstance(data.get("segments"), list)
                        else 0,
                        "segments_out": len(out["segments"]),
                        "issues": issues,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            converted += 1
            print(f"[OK] {fp} -> {out_path}")
        except Exception as ex:
            print(f"[ERROR] {fp}: {ex}")

    print(f"Done. Converted {converted}/{len(files)} file(s).")


if __name__ == "__main__":
    main()
