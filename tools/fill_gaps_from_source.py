#!/usr/bin/env python3
"""
Fill timeline gaps in fine annotation files using source timeline files.

Input pair:
- gaps file:   <gaps_root>/<view>/<subject>/<name>_final_fine.json
- source file: <source_root>/<view>/<subject>/<name>.json

Safety rules:
1) Only fill uncovered intervals (gaps) for entity in {"left", "right"}.
2) Never edit existing segments in the gaps file.
3) Keep all non-segment fields unchanged.
4) Write detailed reports: which files had gaps and which labels were inserted.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


ANOMALY_FALLBACK = [
    "error_temporal",
    "error_spatial",
    "error_handling",
    "error_wrong_part",
    "error_wrong_tool",
    "error_procedural",
]


def normalize_name(name: str) -> str:
    return " ".join(str(name).strip().split())


def build_rename_patch() -> Dict[str, str]:
    return {
        "refine_drive_shaft": "adjust_drive_shaft",
        "refine_bearing_plate": "adjust_bearing_plate",
        "refine_lever": "align_lever",
        "refine_anti_vibration_handle": "adjust_anti_vibration_handle",
        "refine_nut": "align_nut",
        "refine_M6_nut": "align_M6_nut",
        "refine_screw": "align_screw",
        "attach_screw_to_phillips_screwdriver": "attach_screw",
    }


KNOWN_VERB_PREFIXES = [
    "pick_up",
    "hand_tighten",
    "hand_loosen",
    "tighten",
    "loosen",
    "remove",
    "insert",
    "mount",
    "attach",
    "detach",
    "dismount",
    "extract",
    "place",
    "store",
    "transfer",
    "hold",
    "flip",
    "thread",
    "adjust",
    "align",
    "seat",
]


def infer_verb_noun(action_name: str) -> Tuple[Optional[str], Optional[str]]:
    action_name = normalize_name(action_name)
    if action_name == "null":
        return None, None
    for v in sorted(KNOWN_VERB_PREFIXES, key=len, reverse=True):
        pref = v + "_"
        if action_name.startswith(pref):
            noun = action_name[len(pref) :]
            return v, (noun if noun else None)
    if "_" in action_name:
        v, n = action_name.split("_", 1)
        return v, (n if n else None)
    return action_name, None


@dataclass(frozen=True)
class FineTuple:
    action_label: int
    verb: int
    noun: int
    phase: str
    anomaly_type: Tuple[int, ...]


@dataclass
class Span:
    start: int
    end: int


@dataclass
class SmoothRun:
    start: int
    end: int
    fine: FineTuple

    @property
    def length(self) -> int:
        return int(self.end - self.start + 1)


def coalesce_smooth_runs(runs: List[SmoothRun]) -> List[SmoothRun]:
    if not runs:
        return []
    runs = sorted(runs, key=lambda r: (r.start, r.end))
    out: List[SmoothRun] = [runs[0]]
    for cur in runs[1:]:
        prev = out[-1]
        if cur.fine == prev.fine and cur.start <= prev.end + 1:
            prev.end = max(prev.end, cur.end)
        else:
            out.append(cur)
    return out


def runs_from_segments(
    segments: List[Dict[str, Any]],
    entity: str,
    anom_len: int,
) -> List[SmoothRun]:
    out: List[SmoothRun] = []
    for seg in sort_segments_by_entity(segments, entity):
        s0, s1 = segment_span(seg)
        if s1 < s0:
            continue
        out.append(SmoothRun(start=s0, end=s1, fine=fine_tuple_from_seg(seg, anom_len)))
    return coalesce_smooth_runs(out)


def runs_to_segments(
    runs: List[SmoothRun],
    entity: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in runs:
        out.append(
            {
                "action_label": int(r.fine.action_label),
                "verb": int(r.fine.verb),
                "noun": int(r.fine.noun),
                "start_frame": int(r.start),
                "end_frame": int(r.end),
                "phase": str(r.fine.phase),
                "anomaly_type": [int(x) for x in r.fine.anomaly_type],
                "entity": entity,
            }
        )
    return out


def fine_label_text(
    ft: FineTuple,
    fine_id2name: Dict[int, str],
    anomaly_names: List[str],
) -> str:
    if ft.phase == "anomaly":
        names = [
            anomaly_names[i]
            for i, v in enumerate(ft.anomaly_type)
            if i < len(anomaly_names) and int(v) == 1
        ]
        return "anomaly:" + ("+".join(names) if names else "unknown")
    if ft.phase == "recovery":
        return "recovery"
    return fine_id2name.get(int(ft.action_label), f"id_{ft.action_label}")


def try_expand_short_run(
    runs: List[SmoothRun],
    idx: int,
    min_len: int,
) -> Tuple[bool, int]:
    cur = runs[idx]
    need = int(min_len - cur.length)
    if need <= 0:
        return False, 0

    left_idx = idx - 1 if idx > 0 else None
    right_idx = idx + 1 if idx + 1 < len(runs) else None
    if left_idx is None and right_idx is None:
        return False, 0

    # Strict donor policy: donors that are >= min_len must stay >= min_len.
    # This guarantees expansion itself does not create new < min_len runs.
    if left_idx is None:
        left_avail = 0
    else:
        left = runs[left_idx]
        left_floor = min_len if left.length >= min_len else 1
        left_avail = max(0, left.length - left_floor)
    if right_idx is None:
        right_avail = 0
    else:
        right = runs[right_idx]
        right_floor = min_len if right.length >= min_len else 1
        right_avail = max(0, right.length - right_floor)

    if left_avail + right_avail < need:
        return False, 0

    if left_idx is None:
        want_left, want_right = 0, need
    elif right_idx is None:
        want_left, want_right = need, 0
    else:
        want_left = need // 2
        want_right = need - want_left

    take_left = min(want_left, left_avail)
    take_right = min(want_right, right_avail)
    remain = need - take_left - take_right
    if remain > 0 and left_idx is not None:
        add = min(remain, left_avail - take_left)
        take_left += add
        remain -= add
    if remain > 0 and right_idx is not None:
        add = min(remain, right_avail - take_right)
        take_right += add
        remain -= add
    if remain > 0:
        return False, 0

    if take_left > 0 and left_idx is not None:
        runs[left_idx].end -= take_left
        cur.start -= take_left
    if take_right > 0 and right_idx is not None:
        runs[right_idx].start += take_right
        cur.end += take_right
    return True, need


def choose_merge_target(
    runs: List[SmoothRun],
    idx: int,
    min_len: int,
) -> Optional[FineTuple]:
    cur = runs[idx]
    left_idx = idx - 1 if idx > 0 else None
    right_idx = idx + 1 if idx + 1 < len(runs) else None
    if left_idx is None and right_idx is None:
        return None
    candidates: List[Tuple[str, SmoothRun]] = []
    if left_idx is not None:
        candidates.append(("left", runs[left_idx]))
    if right_idx is not None:
        candidates.append(("right", runs[right_idx]))
    if not candidates:
        return None

    def score(side: str, r: SmoothRun) -> Tuple[int, int, int]:
        s = 0
        if r.fine == cur.fine:
            s += 100000
        if r.fine.phase == cur.fine.phase:
            s += 1000
        if r.length >= min_len:
            s += 100
        s += int(r.length)
        # Stable tie-breaker: prefer left then right when all else equal.
        side_bias = 1 if side == "left" else 0
        return (s, int(r.length), side_bias)

    best_side, best_run = max(candidates, key=lambda x: score(x[0], x[1]))
    _ = best_side
    return best_run.fine


def smooth_short_runs(
    runs: List[SmoothRun],
    min_len: int,
    expand_from: int,
    max_passes: int,
) -> Tuple[List[SmoothRun], Dict[str, Any]]:
    runs = coalesce_smooth_runs(runs)
    stats: Dict[str, Any] = {
        "short_before": sum(1 for r in runs if r.length < min_len),
        "short_after": 0,
        "expanded_runs": 0,
        "expanded_frames": 0,
        "merged_runs": 0,
        "changed": False,
    }
    if not runs:
        return runs, stats

    for _ in range(max_passes):
        changed = False
        i = 0
        while i < len(runs):
            cur = runs[i]
            clen = cur.length
            if clen >= min_len:
                i += 1
                continue

            did = False
            if expand_from <= clen < min_len:
                ok, gain = try_expand_short_run(runs, i, min_len=min_len)
                if ok:
                    stats["expanded_runs"] += 1
                    stats["expanded_frames"] += int(gain)
                    runs = coalesce_smooth_runs([r for r in runs if r.end >= r.start])
                    changed = True
                    did = True
                    i = max(0, i - 1)

            if not did:
                tgt = choose_merge_target(runs, i, min_len=min_len)
                if tgt is not None and tgt != runs[i].fine:
                    runs[i].fine = tgt
                    stats["merged_runs"] += 1
                    runs = coalesce_smooth_runs(runs)
                    changed = True
                    i = max(0, i - 1)
                else:
                    i += 1

        if not changed:
            break

    stats["short_after"] = sum(1 for r in runs if r.length < min_len)
    stats["changed"] = bool(stats["expanded_runs"] > 0 or stats["merged_runs"] > 0)
    return runs, stats


def frame_fine_map(runs: List[SmoothRun]) -> Dict[int, FineTuple]:
    out: Dict[int, FineTuple] = {}
    for r in runs:
        for f in range(int(r.start), int(r.end) + 1):
            out[f] = r.fine
    return out


def canonical_entity(seg: Dict[str, Any]) -> Optional[str]:
    e = seg.get("entity")
    if e is None:
        return None
    txt = str(e).strip().lower()
    return txt if txt else None


def segment_span(seg: Dict[str, Any]) -> Tuple[int, int]:
    return int(seg.get("start_frame", 0)), int(seg.get("end_frame", 0))


def overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    s = max(a0, b0)
    e = min(a1, b1)
    return max(0, e - s + 1)


def anomaly_names_from_fine(fine_json: Dict[str, Any]) -> List[str]:
    arr = fine_json.get("anomaly_types")
    if isinstance(arr, list) and arr:
        out = []
        for item in arr:
            if isinstance(item, dict) and item.get("name"):
                out.append(str(item["name"]))
        if out:
            return out
    return list(ANOMALY_FALLBACK)


def phase_name(value: Any) -> str:
    txt = str(value or "").strip().lower()
    if txt in ("normal", "anomaly", "recovery"):
        return txt
    return "normal"


def fine_tuple_from_seg(seg: Dict[str, Any], anom_len: int) -> FineTuple:
    arr = seg.get("anomaly_type")
    vec: List[int] = []
    if isinstance(arr, list):
        for x in arr[:anom_len]:
            try:
                vec.append(1 if int(x) else 0)
            except Exception:
                vec.append(0)
    while len(vec) < anom_len:
        vec.append(0)
    return FineTuple(
        action_label=int(seg.get("action_label", 0)),
        verb=int(seg.get("verb", -1)),
        noun=int(seg.get("noun", -1)),
        phase=phase_name(seg.get("phase")),
        anomaly_type=tuple(vec),
    )


def sort_segments_by_entity(
    segments: Iterable[Dict[str, Any]], entity: str
) -> List[Dict[str, Any]]:
    out = [s for s in segments if canonical_entity(s) == entity]
    out.sort(key=lambda x: (int(x.get("start_frame", 0)), int(x.get("end_frame", 0))))
    return out


def gap_spans_for_entity(
    segments: List[Dict[str, Any]],
    view_start: int,
    view_end: int,
) -> List[Span]:
    if not segments:
        return [Span(view_start, view_end)] if view_end >= view_start else []
    gaps: List[Span] = []
    first_s = int(segments[0]["start_frame"])
    if first_s > view_start:
        gaps.append(Span(view_start, first_s - 1))
    for i in range(1, len(segments)):
        prev_e = int(segments[i - 1]["end_frame"])
        cur_s = int(segments[i]["start_frame"])
        if cur_s > prev_e + 1:
            gaps.append(Span(prev_e + 1, cur_s - 1))
    last_e = int(segments[-1]["end_frame"])
    if last_e < view_end:
        gaps.append(Span(last_e + 1, view_end))
    return [g for g in gaps if g.end >= g.start]


def spans_total_len(spans: Iterable[Span]) -> int:
    return sum(max(0, s.end - s.start + 1) for s in spans)


def merged_spans(spans: Iterable[Span]) -> List[Span]:
    data = sorted(
        [Span(int(s.start), int(s.end)) for s in spans if int(s.end) >= int(s.start)],
        key=lambda x: (x.start, x.end),
    )
    if not data:
        return []
    out: List[Span] = [data[0]]
    for cur in data[1:]:
        prev = out[-1]
        if cur.start <= prev.end + 1:
            prev.end = max(prev.end, cur.end)
        else:
            out.append(cur)
    return out


def expand_clip_spans(
    spans: Iterable[Span],
    pad: int,
    lo: int,
    hi: int,
) -> List[Span]:
    out: List[Span] = []
    p = max(0, int(pad))
    for s in spans:
        a = max(int(lo), int(s.start) - p)
        b = min(int(hi), int(s.end) + p)
        if b >= a:
            out.append(Span(a, b))
    return merged_spans(out)


def frame_set_from_spans(spans: Iterable[Span]) -> set[int]:
    out: set[int] = set()
    for s in spans:
        for f in range(int(s.start), int(s.end) + 1):
            out.add(int(f))
    return out


def source_label_maps(
    src_json: Dict[str, Any]
) -> Tuple[Dict[int, str], Dict[str, int]]:
    id2name: Dict[int, str] = {}
    name2id: Dict[str, int] = {}
    for item in src_json.get("labels", []):
        if not isinstance(item, dict):
            continue
        try:
            i = int(item.get("id"))
        except Exception:
            continue
        n = normalize_name(item.get("name", ""))
        if not n:
            continue
        id2name[i] = n
        if n not in name2id:
            name2id[n] = i
    return id2name, name2id


def fine_label_maps(fine_json: Dict[str, Any]) -> Tuple[Dict[int, str], Dict[str, int]]:
    id2name: Dict[int, str] = {}
    name2id: Dict[str, int] = {}
    for item in fine_json.get("action_labels", []):
        if not isinstance(item, dict):
            continue
        try:
            i = int(item.get("id"))
        except Exception:
            continue
        n = normalize_name(item.get("name", ""))
        if not n:
            continue
        id2name[i] = n
        if n not in name2id:
            name2id[n] = i
    return id2name, name2id


def fine_vocab_maps(fine_json: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[str, int]]:
    verb2id: Dict[str, int] = {}
    noun2id: Dict[str, int] = {}
    for x in fine_json.get("verbs", []):
        if (
            isinstance(x, dict)
            and x.get("name") is not None
            and x.get("id") is not None
        ):
            verb2id[str(x["name"])] = int(x["id"])
    for x in fine_json.get("nouns", []):
        if (
            isinstance(x, dict)
            and x.get("name") is not None
            and x.get("id") is not None
        ):
            noun2id[str(x["name"])] = int(x["id"])
    return verb2id, noun2id


def build_local_map_for_entity(
    src_segments: List[Dict[str, Any]],
    fine_segments: List[Dict[str, Any]],
    anom_len: int,
) -> Dict[int, Tuple[FineTuple, float]]:
    votes: DefaultDict[int, Counter] = defaultdict(Counter)
    src_segments = sorted(
        src_segments, key=lambda s: (int(s["start_frame"]), int(s["end_frame"]))
    )
    fine_segments = sorted(
        fine_segments, key=lambda s: (int(s["start_frame"]), int(s["end_frame"]))
    )
    i = 0
    j = 0
    while i < len(src_segments) and j < len(fine_segments):
        s = src_segments[i]
        f = fine_segments[j]
        s0, s1 = segment_span(s)
        f0, f1 = segment_span(f)
        ov = overlap_len(s0, s1, f0, f1)
        if ov > 0:
            sid = int(s.get("action_label", -1))
            votes[sid][fine_tuple_from_seg(f, anom_len)] += ov
        if s1 <= f1:
            i += 1
        if f1 <= s1:
            j += 1

    out: Dict[int, Tuple[FineTuple, float]] = {}
    for sid, cnt in votes.items():
        total = sum(cnt.values())
        if total <= 0:
            continue
        best_tuple, best_frames = cnt.most_common(1)[0]
        out[sid] = (best_tuple, float(best_frames) / float(total))
    return out


def build_global_name_map(
    gaps_files: List[Path],
    gaps_root: Path,
    source_root: Path,
) -> Dict[str, Tuple[FineTuple, float]]:
    votes: DefaultDict[str, Counter] = defaultdict(Counter)
    for fine_path in gaps_files:
        rel = fine_path.relative_to(gaps_root)
        src_rel = Path(str(rel).replace("_final_fine.json", ".json"))
        src_path = source_root / src_rel
        if not src_path.exists():
            continue
        try:
            fine = json.loads(fine_path.read_text(encoding="utf-8"))
            src = json.loads(src_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        src_id2name, _ = source_label_maps(src)
        anom_len = len(anomaly_names_from_fine(fine))
        for entity in ("left", "right"):
            src_segs = sort_segments_by_entity(src.get("segments", []), entity)
            fine_segs = sort_segments_by_entity(fine.get("segments", []), entity)
            if not src_segs or not fine_segs:
                continue
            i = 0
            j = 0
            while i < len(src_segs) and j < len(fine_segs):
                s = src_segs[i]
                f = fine_segs[j]
                s0, s1 = segment_span(s)
                f0, f1 = segment_span(f)
                ov = overlap_len(s0, s1, f0, f1)
                if ov > 0:
                    sid = int(s.get("action_label", -1))
                    sname = src_id2name.get(sid)
                    if sname:
                        votes[sname][fine_tuple_from_seg(f, anom_len)] += ov
                if s1 <= f1:
                    i += 1
                if f1 <= s1:
                    j += 1
    out: Dict[str, Tuple[FineTuple, float]] = {}
    for sname, cnt in votes.items():
        total = sum(cnt.values())
        if total <= 0:
            continue
        best_tuple, best_frames = cnt.most_common(1)[0]
        out[sname] = (best_tuple, float(best_frames) / float(total))
    return out


def anomaly_vec_for_name(name: str, anomaly_names: List[str]) -> List[int]:
    vec = [0] * len(anomaly_names)
    low = name.strip().lower()
    if low in anomaly_names:
        vec[anomaly_names.index(low)] = 1
    elif low in ANOMALY_FALLBACK and low in anomaly_names:
        vec[anomaly_names.index(low)] = 1
    return vec


def fallback_map(
    src_name: str,
    fine_name2id: Dict[str, int],
    fine_id2name: Dict[int, str],
    verb2id: Dict[str, int],
    noun2id: Dict[str, int],
    anomaly_names: List[str],
    rename_patch: Dict[str, str],
) -> FineTuple:
    src_name = normalize_name(src_name)
    if src_name.startswith("error_"):
        return FineTuple(
            action_label=0,
            verb=-1,
            noun=-1,
            phase="anomaly",
            anomaly_type=tuple(anomaly_vec_for_name(src_name, anomaly_names)),
        )
    if src_name == "recovery":
        return FineTuple(
            action_label=0,
            verb=-1,
            noun=-1,
            phase="recovery",
            anomaly_type=tuple([0] * len(anomaly_names)),
        )
    if src_name == "transfer_tool":
        return FineTuple(
            action_label=0,
            verb=-1,
            noun=-1,
            phase="normal",
            anomaly_type=tuple([0] * len(anomaly_names)),
        )

    mapped = src_name
    if mapped not in fine_name2id and mapped in rename_patch:
        mapped = rename_patch[mapped]
    aid = int(fine_name2id.get(mapped, 0))
    label_name = fine_id2name.get(aid, "null")
    v, n = infer_verb_noun(label_name)
    vid = int(verb2id.get(v, -1)) if v else -1
    nid = int(noun2id.get(n, -1)) if n else -1
    return FineTuple(
        action_label=aid,
        verb=vid,
        noun=nid,
        phase="normal",
        anomaly_type=tuple([0] * len(anomaly_names)),
    )


def resolve_fine_tuple(
    src_id: int,
    src_name: str,
    local_map: Dict[int, Tuple[FineTuple, float]],
    global_name_map: Dict[str, Tuple[FineTuple, float]],
    fine_name2id: Dict[str, int],
    fine_id2name: Dict[int, str],
    verb2id: Dict[str, int],
    noun2id: Dict[str, int],
    anomaly_names: List[str],
    rename_patch: Dict[str, str],
    min_conf: float,
) -> Tuple[FineTuple, str]:
    # Keep special labels deterministic to match convert_fine_labels.py behavior.
    # They must not be overridden by learned local/global mappings.
    if src_name.startswith("error_") or src_name in {"recovery", "transfer_tool"}:
        return (
            fallback_map(
                src_name=src_name,
                fine_name2id=fine_name2id,
                fine_id2name=fine_id2name,
                verb2id=verb2id,
                noun2id=noun2id,
                anomaly_names=anomaly_names,
                rename_patch=rename_patch,
            ),
            "fallback",
        )
    if src_id in local_map:
        tup, conf = local_map[src_id]
        if conf >= min_conf:
            return tup, "local"
    if src_name in global_name_map:
        tup, conf = global_name_map[src_name]
        if conf >= min_conf:
            return tup, "global"
    return (
        fallback_map(
            src_name=src_name,
            fine_name2id=fine_name2id,
            fine_id2name=fine_id2name,
            verb2id=verb2id,
            noun2id=noun2id,
            anomaly_names=anomaly_names,
            rename_patch=rename_patch,
        ),
        "fallback",
    )


def source_segments_overlapping_span(
    src_segments: List[Dict[str, Any]],
    span: Span,
) -> List[Tuple[Dict[str, Any], int, int]]:
    out: List[Tuple[Dict[str, Any], int, int]] = []
    for seg in src_segments:
        s0, s1 = segment_span(seg)
        ov0 = max(s0, span.start)
        ov1 = min(s1, span.end)
        if ov1 >= ov0:
            out.append((seg, ov0, ov1))
    return out


def canonical_seg_tuple(seg: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        seg.get("entity"),
        int(seg.get("start_frame", 0)),
        int(seg.get("end_frame", 0)),
        int(seg.get("action_label", 0)),
        int(seg.get("verb", -1)),
        int(seg.get("noun", -1)),
        str(seg.get("phase", "")),
        tuple(int(x) for x in (seg.get("anomaly_type") or [])),
    )


def entity_frame_bounds(
    segments: List[Dict[str, Any]], entity: str
) -> Optional[Tuple[int, int]]:
    track = sort_segments_by_entity(segments, entity)
    if not track:
        return None
    s0 = min(int(s.get("start_frame", 0)) for s in track)
    s1 = max(int(s.get("end_frame", 0)) for s in track)
    return (s0, s1)


def phase_bounds_align_with_actions(
    segments: List[Dict[str, Any]], entity: str
) -> bool:
    track = sort_segments_by_entity(segments, entity)
    if not track:
        return True

    action_starts = {int(s.get("start_frame", 0)) for s in track}
    action_ends = {int(s.get("end_frame", 0)) for s in track}

    phase_runs: List[Tuple[int, int]] = []
    cur_phase = str(track[0].get("phase", ""))
    run_start = int(track[0].get("start_frame", 0))
    prev_end = int(track[0].get("end_frame", 0))
    for seg in track[1:]:
        s0 = int(seg.get("start_frame", 0))
        s1 = int(seg.get("end_frame", 0))
        ph = str(seg.get("phase", ""))
        if ph == cur_phase and s0 == prev_end + 1:
            prev_end = s1
            continue
        phase_runs.append((run_start, prev_end))
        cur_phase = ph
        run_start = s0
        prev_end = s1
    phase_runs.append((run_start, prev_end))

    for ps, pe in phase_runs:
        if ps not in action_starts or pe not in action_ends:
            return False
    return True


def runs_from_frame_map_dense(frame_map: Dict[int, FineTuple]) -> List[SmoothRun]:
    if not frame_map:
        return []
    frames = sorted(int(x) for x in frame_map.keys())
    out: List[SmoothRun] = []
    cur_start = frames[0]
    cur_end = frames[0]
    cur_tuple = frame_map[frames[0]]
    for f in frames[1:]:
        ft = frame_map[f]
        if f == cur_end + 1 and ft == cur_tuple:
            cur_end = f
            continue
        out.append(SmoothRun(start=cur_start, end=cur_end, fine=cur_tuple))
        cur_start = f
        cur_end = f
        cur_tuple = ft
    out.append(SmoothRun(start=cur_start, end=cur_end, fine=cur_tuple))
    return out


def process_one_file(
    fine_path: Path,
    src_path: Path,
    out_path: Path,
    global_name_map: Dict[str, Tuple[FineTuple, float]],
    min_conf: float,
    apply_write: bool,
    optimize_short: bool,
    short_min_len: int,
    short_expand_from: int,
    short_max_passes: int,
    short_max_change_ratio: float,
    short_boundary_pad: int,
) -> Optional[Dict[str, Any]]:
    try:
        fine = json.loads(fine_path.read_text(encoding="utf-8"))
        src = json.loads(src_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    src_id2name, _ = source_label_maps(src)
    fine_id2name, fine_name2id = fine_label_maps(fine)
    verb2id, noun2id = fine_vocab_maps(fine)
    anomaly_names = anomaly_names_from_fine(fine)
    rename_patch = build_rename_patch()

    original_segments = fine.get("segments", [])
    if not isinstance(original_segments, list):
        return None

    # Use original fine timeline as the primary temporal bound to avoid output
    # exceeding the clipped final length. Fall back to source only if needed.
    fine_starts: List[int] = []
    fine_ends: List[int] = []
    src_starts: List[int] = []
    src_ends: List[int] = []
    for seg in list(original_segments):
        if not isinstance(seg, dict):
            continue
        ent = canonical_entity(seg)
        if ent not in {"left", "right"}:
            continue
        try:
            s0, s1 = segment_span(seg)
        except Exception:
            continue
        fine_starts.append(int(s0))
        fine_ends.append(int(s1))
    for seg in list(src.get("segments", [])):
        if not isinstance(seg, dict):
            continue
        ent = canonical_entity(seg)
        if ent not in {"left", "right"}:
            continue
        try:
            s0, s1 = segment_span(seg)
        except Exception:
            continue
        src_starts.append(int(s0))
        src_ends.append(int(s1))

    if "view_start" in fine:
        view_start = int(fine.get("view_start", 0))
    elif fine_starts:
        view_start = min(fine_starts)
    elif src_starts:
        view_start = min(src_starts)
    else:
        view_start = 0
    if "view_end" in fine:
        view_end = int(fine.get("view_end", view_start))
    elif fine_ends:
        view_end = max(fine_ends)
    elif src_ends:
        view_end = max(src_ends)
    else:
        view_end = view_start
    if view_end < view_start:
        view_end = view_start

    inserted: List[Dict[str, Any]] = []
    file_report: Dict[str, Any] = {
        "file": str(fine_path),
        "source_file": str(src_path),
        "changed": False,
        "gap_changed": False,
        "short_changed": False,
        "source_missing": False,
        "parse_error": False,
        "write_mode": "no_write_preview",
        "entities": {},
        "inserted_segments": 0,
        "inserted_labels_source": {},
        "inserted_labels_target": {},
        "mapping_mode_counts": {"local": 0, "global": 0, "fallback": 0},
        "short_label_corrections": {},
        "short_label_changed_frames": 0,
    }

    for entity in ("left", "right"):
        fine_track = sort_segments_by_entity(original_segments, entity)
        src_track = sort_segments_by_entity(src.get("segments", []), entity)
        # Fill each entity on the same global clip timeline to avoid per-entity
        # leading/trailing blanks (e.g., one hand starts later than the other).
        # The global range is still bounded by original final timeline.
        entity_view_start = view_start
        entity_view_end = view_end
        gaps_before = gap_spans_for_entity(
            fine_track,
            view_start=entity_view_start,
            view_end=entity_view_end,
        )
        ent_report = {
            "gaps_before_count": len(gaps_before),
            "gaps_before_frames": spans_total_len(gaps_before),
            "gaps_before_spans": [[int(g.start), int(g.end)] for g in gaps_before],
            "gaps_filled_count": 0,
            "gaps_filled_frames": 0,
            "gaps_after_count": len(gaps_before),
            "gaps_after_frames": spans_total_len(gaps_before),
            "inserted_segments": 0,
            "short_before": 0,
            "short_after": 0,
            "short_expanded_runs": 0,
            "short_expanded_frames": 0,
            "short_merged_runs": 0,
            "short_changed_frames": 0,
            "short_changed_frames_inside_window": 0,
            "short_changed_frames_outside_window": 0,
            "short_mutable_window_frames": 0,
            "short_coverage_before": 0,
            "short_coverage_after": 0,
            "short_label_corrections": {},
            "short_guard_reverted": False,
            "short_guard_reason": "",
        }
        if not fine_track or not src_track or not gaps_before:
            file_report["entities"][entity] = ent_report
            continue

        local_map = build_local_map_for_entity(
            src_segments=src_track,
            fine_segments=fine_track,
            anom_len=len(anomaly_names),
        )

        for gap in gaps_before:
            pieces = source_segments_overlapping_span(src_track, gap)
            if not pieces:
                continue
            gap_inserted_frames = 0
            for src_seg, ps, pe in pieces:
                sid = int(src_seg.get("action_label", -1))
                sname = src_id2name.get(sid, f"__unknown_{sid}")
                ft, mode = resolve_fine_tuple(
                    src_id=sid,
                    src_name=sname,
                    local_map=local_map,
                    global_name_map=global_name_map,
                    fine_name2id=fine_name2id,
                    fine_id2name=fine_id2name,
                    verb2id=verb2id,
                    noun2id=noun2id,
                    anomaly_names=anomaly_names,
                    rename_patch=rename_patch,
                    min_conf=min_conf,
                )
                new_seg = {
                    "action_label": int(ft.action_label),
                    "verb": int(ft.verb),
                    "noun": int(ft.noun),
                    "start_frame": int(ps),
                    "end_frame": int(pe),
                    "phase": ft.phase,
                    "anomaly_type": [int(x) for x in ft.anomaly_type],
                    "entity": entity,
                }
                inserted.append(new_seg)
                nframes = int(pe - ps + 1)
                gap_inserted_frames += nframes
                ent_report["inserted_segments"] += 1
                file_report["mapping_mode_counts"][mode] += 1
                file_report["inserted_labels_source"][sname] = (
                    int(file_report["inserted_labels_source"].get(sname, 0)) + nframes
                )
                tname = fine_id2name.get(int(ft.action_label), f"id_{ft.action_label}")
                file_report["inserted_labels_target"][tname] = (
                    int(file_report["inserted_labels_target"].get(tname, 0)) + nframes
                )
            if gap_inserted_frames > 0:
                ent_report["gaps_filled_count"] += 1
                ent_report["gaps_filled_frames"] += gap_inserted_frames

        if ent_report["inserted_segments"] > 0:
            merged_track = fine_track + [
                x for x in inserted if canonical_entity(x) == entity
            ]
            merged_track = sorted(
                merged_track, key=lambda s: (int(s["start_frame"]), int(s["end_frame"]))
            )
            after = gap_spans_for_entity(
                merged_track,
                view_start=entity_view_start,
                view_end=entity_view_end,
            )
            ent_report["gaps_after_count"] = len(after)
            ent_report["gaps_after_frames"] = spans_total_len(after)

        file_report["entities"][entity] = ent_report

    out_segments = list(original_segments) + inserted
    gap_changed = len(inserted) > 0
    short_changed = False

    if optimize_short:
        non_lr_segments = [
            s for s in out_segments if canonical_entity(s) not in {"left", "right"}
        ]
        lr_segments: List[Dict[str, Any]] = []
        correction_counter: Counter = Counter()

        for entity in ("left", "right"):
            runs_before = runs_from_segments(
                segments=out_segments,
                entity=entity,
                anom_len=len(anomaly_names),
            )
            before_map = frame_fine_map(runs_before)
            runs_after, st = smooth_short_runs(
                runs=runs_before,
                min_len=short_min_len,
                expand_from=short_expand_from,
                max_passes=short_max_passes,
            )
            after_map = frame_fine_map(runs_after)

            ent = file_report["entities"].setdefault(
                entity,
                {
                    "gaps_before_count": 0,
                    "gaps_before_frames": 0,
                    "gaps_filled_count": 0,
                    "gaps_filled_frames": 0,
                    "gaps_after_count": 0,
                    "gaps_after_frames": 0,
                    "inserted_segments": 0,
                    "short_before": 0,
                    "short_after": 0,
                    "short_expanded_runs": 0,
                    "short_expanded_frames": 0,
                    "short_merged_runs": 0,
                    "short_changed_frames": 0,
                    "short_changed_frames_inside_window": 0,
                    "short_changed_frames_outside_window": 0,
                    "short_mutable_window_frames": 0,
                    "short_coverage_before": 0,
                    "short_coverage_after": 0,
                    "short_label_corrections": {},
                },
            )
            ent["short_before"] = int(st["short_before"])
            ent["short_after"] = int(st["short_after"])
            ent["short_expanded_runs"] = int(st["expanded_runs"])
            ent["short_expanded_frames"] = int(st["expanded_frames"])
            ent["short_merged_runs"] = int(st["merged_runs"])
            ent["short_coverage_before"] = int(len(before_map))
            ent["short_coverage_after"] = int(len(after_map))

            if len(before_map) != len(after_map):
                raise RuntimeError(
                    f"Short optimization changed frame coverage in {fine_path} entity={entity}"
                )

            raw_gap_spans = [
                Span(int(x[0]), int(x[1]))
                for x in ent.get("gaps_before_spans", [])
                if isinstance(x, list) and len(x) == 2
            ]
            mutable_spans = expand_clip_spans(
                spans=raw_gap_spans,
                pad=int(short_boundary_pad),
                lo=int(view_start),
                hi=int(view_end),
            )
            mutable_frames = frame_set_from_spans(mutable_spans)
            ent["short_mutable_window_frames"] = int(len(mutable_frames))

            # Protect non-gap regions: keep labels unchanged outside
            # (expanded) original gap windows.
            all_frames = set(before_map.keys()) | set(after_map.keys())
            outside_change_candidates = 0
            for f in all_frames:
                b = before_map.get(f)
                a = after_map.get(f)
                if b != a and int(f) not in mutable_frames:
                    outside_change_candidates += 1
                    after_map[int(f)] = b
            if outside_change_candidates > 0:
                runs_after = runs_from_frame_map_dense(after_map)
                after_map = frame_fine_map(runs_after)

            if set(before_map.keys()) != set(after_map.keys()):
                raise RuntimeError(
                    f"Short optimization changed frame key set in {fine_path} entity={entity}"
                )

            local_changes: Counter = Counter()
            local_changes_inside = 0
            local_changes_outside = 0
            for f in sorted(set(before_map.keys()) | set(after_map.keys())):
                b = before_map.get(f)
                a = after_map.get(f)
                if b != a:
                    btxt = (
                        fine_label_text(b, fine_id2name, anomaly_names)
                        if b is not None
                        else "<none>"
                    )
                    atxt = (
                        fine_label_text(a, fine_id2name, anomaly_names)
                        if a is not None
                        else "<none>"
                    )
                    local_changes[(btxt, atxt)] += 1
                    if int(f) in mutable_frames:
                        local_changes_inside += 1
                    else:
                        local_changes_outside += 1
            changed_frames = int(sum(local_changes.values()))
            coverage = int(max(1, len(before_map)))
            changed_ratio = float(changed_frames) / float(coverage)

            # Recompute short-count after protected rollback.
            short_before_real = sum(1 for r in runs_before if r.length < short_min_len)
            short_after_real = sum(1 for r in runs_after if r.length < short_min_len)
            st["short_before"] = int(short_before_real)
            st["short_after"] = int(short_after_real)
            st["changed"] = bool(changed_frames > 0)

            # Guard 1: avoid chain modifications that are too broad.
            # Guard 2: never allow optimization that increases the count of short runs.
            # Guard 3: never allow any change outside protected mutable windows.
            guard_reason = ""
            if short_max_change_ratio >= 0 and changed_ratio > float(
                short_max_change_ratio
            ):
                guard_reason = f"changed_ratio={changed_ratio:.4f} > short_max_change_ratio={short_max_change_ratio:.4f}"
            elif int(st.get("short_after", 0)) > int(st.get("short_before", 0)):
                guard_reason = f"short_after={st.get('short_after')} > short_before={st.get('short_before')}"
            elif local_changes_outside > 0:
                guard_reason = f"non_gap_changed_frames={local_changes_outside} (outside mutable windows)"

            if guard_reason:
                runs_after = runs_before
                after_map = before_map
                local_changes = Counter()
                changed_frames = 0
                local_changes_inside = 0
                local_changes_outside = 0
                st["short_after"] = st["short_before"]
                st["expanded_runs"] = 0
                st["expanded_frames"] = 0
                st["merged_runs"] = 0
                st["changed"] = False
                ent["short_guard_reverted"] = True
                ent["short_guard_reason"] = guard_reason

            ent["short_changed_frames"] = changed_frames
            ent["short_changed_frames_inside_window"] = int(local_changes_inside)
            ent["short_changed_frames_outside_window"] = int(local_changes_outside)
            ent["short_label_corrections"] = {
                f"{k[0]} -> {k[1]}": int(v) for k, v in local_changes.most_common()
            }
            correction_counter.update(local_changes)

            if st["changed"] or changed_frames > 0:
                short_changed = True

            lr_segments.extend(runs_to_segments(runs_after, entity))

        out_segments = non_lr_segments + lr_segments
        out_segments = sorted(
            out_segments,
            key=lambda s: (
                str(s.get("entity", "")),
                int(s.get("start_frame", 0)),
                int(s.get("end_frame", 0)),
            ),
        )
        file_report["short_label_corrections"] = {
            f"{k[0]} -> {k[1]}": int(v) for k, v in correction_counter.most_common()
        }
        file_report["short_label_changed_frames"] = int(
            sum(correction_counter.values())
        )

    file_report["inserted_segments"] = len(inserted)
    file_report["gap_changed"] = bool(gap_changed)
    file_report["short_changed"] = bool(short_changed)
    file_report["changed"] = bool(gap_changed or short_changed)

    # Invariants:
    # 1) Left/right output must stay within original final clip timeline.
    # 2) Left/right must fully cover global clip timeline (no temporal holes).
    # 3) Every phase-run boundary must coincide with an action boundary.
    for entity in ("left", "right"):
        out_bounds = entity_frame_bounds(out_segments, entity)
        if out_bounds is not None and (
            int(out_bounds[0]) < int(view_start) or int(out_bounds[1]) > int(view_end)
        ):
            raise RuntimeError(
                f"Entity bounds exceed clip timeline in {fine_path} entity={entity} "
                f"clip=[{view_start},{view_end}] out={out_bounds}"
            )
        out_track = sort_segments_by_entity(out_segments, entity)
        if out_track:
            rem = gap_spans_for_entity(
                out_track,
                view_start=view_start,
                view_end=view_end,
            )
            if rem:
                raise RuntimeError(
                    f"Entity still has gaps in clip timeline in {fine_path} "
                    f"entity={entity} rem={[(x.start, x.end) for x in rem[:5]]}"
                )
        else:
            raise RuntimeError(
                f"Entity track missing after repair in {fine_path} entity={entity}"
            )
        if not phase_bounds_align_with_actions(out_segments, entity):
            raise RuntimeError(
                f"Phase/action boundary misalignment in {fine_path} entity={entity}"
            )

    if not file_report["changed"]:
        if apply_write:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(fine, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            file_report["write_mode"] = "unchanged_passthrough"
        return file_report

    if not optimize_short:
        # Safety: existing segments stay untouched by design when only gap fill is enabled.
        old_sig = Counter(
            canonical_seg_tuple(s) for s in original_segments if isinstance(s, dict)
        )
        out_sig = Counter(
            canonical_seg_tuple(s) for s in out_segments if isinstance(s, dict)
        )
        for k, v in old_sig.items():
            if out_sig[k] < v:
                raise RuntimeError(
                    f"Existing segment altered unexpectedly in {fine_path}"
                )

    out_json = copy.deepcopy(fine)
    out_json["segments"] = out_segments
    if apply_write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        file_report["write_mode"] = "changed_repaired"
    else:
        file_report["write_mode"] = "no_write_preview_changed"

    return file_report


def write_text_summary(report: Dict[str, Any], out_path: Path) -> None:
    lines: List[str] = []
    lines.append(f"gaps_root: {report['gaps_root']}")
    lines.append(f"source_root: {report['source_root']}")
    lines.append(f"out_root: {report['out_root']}")
    lines.append(f"apply: {report['apply']}")
    lines.append(f"optimize_short: {report['optimize_short']}")
    lines.append(f"short_min_len: {report['short_min_len']}")
    lines.append(f"short_expand_from: {report['short_expand_from']}")
    lines.append(f"short_max_passes: {report['short_max_passes']}")
    lines.append(f"short_max_change_ratio: {report['short_max_change_ratio']}")
    lines.append(f"short_boundary_pad: {report['short_boundary_pad']}")
    lines.append(f"input_files_total: {report['input_files_total']}")
    lines.append(f"processed_files: {report['processed_files']}")
    lines.append(f"files_written: {report['files_written']}")
    lines.append(f"files_unchanged_written: {report['files_unchanged_written']}")
    lines.append(f"files_missing_source: {report['files_missing_source']}")
    lines.append(
        f"files_parse_error_passthrough: {report['files_parse_error_passthrough']}"
    )
    lines.append(f"files_with_gaps: {report['files_with_gaps']}")
    lines.append(f"files_changed: {report['files_changed']}")
    lines.append(f"gap_changed_files: {report['gap_changed_files']}")
    lines.append(f"short_changed_files: {report['short_changed_files']}")
    lines.append(f"inserted_segments_total: {report['inserted_segments_total']}")
    lines.append(
        f"short_label_changed_frames_total: {report['short_label_changed_frames_total']}"
    )
    lines.append("")
    lines.append("All processed files:")
    for item in report["files"]:
        mode = item.get("write_mode", "unknown")
        lines.append(f"- [{mode}] {item.get('file', '')}")
    lines.append("")
    lines.append("Missing source files:")
    for item in report.get("missing_source_files", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Parse error passthrough files:")
    for item in report.get("parse_error_files", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Changed files:")
    for item in report["changed_files"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Inserted source labels (frame count):")
    for k, v in report["inserted_labels_source"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Inserted target labels (frame count):")
    for k, v in report["inserted_labels_target"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Short label corrections (frame count):")
    for k, v in report["short_label_corrections"].items():
        lines.append(f"- {k}: {v}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps_root", type=str, required=True)
    ap.add_argument("--source_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--min_conf", type=float, default=0.60)
    ap.add_argument(
        "--optimize_short",
        action="store_true",
        help="Enable post-processing for short segments (< short_min_len) on left/right tracks.",
    )
    ap.add_argument("--short_min_len", type=int, default=15)
    ap.add_argument("--short_expand_from", type=int, default=13)
    ap.add_argument("--short_max_passes", type=int, default=20)
    ap.add_argument(
        "--short_max_change_ratio",
        type=float,
        default=0.08,
        help="If short-optimization changes more than this ratio of frames in one entity, revert that entity optimization. Use negative to disable.",
    )
    ap.add_argument(
        "--short_boundary_pad",
        type=int,
        default=-1,
        help="Only allow short optimization changes within [gap_start-pad, gap_end+pad]. Negative means auto (short_min_len-1).",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Write repaired files to out_root. Without this flag, only reports are generated.",
    )
    ap.add_argument("--report_json", type=str, default="gap_fill_report.json")
    ap.add_argument("--report_txt", type=str, default="gap_fill_report.txt")
    args = ap.parse_args()

    gaps_root = Path(args.gaps_root)
    source_root = Path(args.source_root)
    out_root = Path(args.out_root)

    fine_files = sorted(gaps_root.rglob("*_final_fine.json"))
    print(f"[INFO] fine files found: {len(fine_files)}")
    print("[INFO] building global name mapping...")
    global_name_map = build_global_name_map(
        gaps_files=fine_files,
        gaps_root=gaps_root,
        source_root=source_root,
    )
    print(f"[INFO] global source names mapped: {len(global_name_map)}")

    reports: List[Dict[str, Any]] = []
    changed_files: List[str] = []
    inserted_src_counter: Counter = Counter()
    inserted_tgt_counter: Counter = Counter()
    short_label_correction_counter: Counter = Counter()
    inserted_total = 0
    short_label_changed_frames_total = 0
    files_with_gaps = 0
    gap_changed_files = 0
    short_changed_files = 0
    files_written = 0
    files_unchanged_written = 0
    files_missing_source = 0
    files_parse_error_passthrough = 0
    missing_source_files: List[str] = []
    parse_error_files: List[str] = []

    if args.short_min_len <= 1:
        raise ValueError("--short_min_len must be >= 2")
    if not (1 <= args.short_expand_from <= args.short_min_len):
        raise ValueError("--short_expand_from must be in [1, short_min_len]")
    if args.short_max_passes < 1:
        raise ValueError("--short_max_passes must be >= 1")
    if args.short_max_change_ratio > 1.0:
        raise ValueError(
            "--short_max_change_ratio must be <= 1.0 (or negative to disable)"
        )
    short_boundary_pad = int(args.short_boundary_pad)
    if short_boundary_pad < 0:
        short_boundary_pad = int(args.short_min_len) - 1

    for fine_path in fine_files:
        rel = fine_path.relative_to(gaps_root)
        src_rel = Path(str(rel).replace("_final_fine.json", ".json"))
        src_path = source_root / src_rel
        out_path = out_root / rel
        if not src_path.exists():
            if bool(args.apply):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    fine_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
                files_written += 1
                files_unchanged_written += 1
            files_missing_source += 1
            missing_source_files.append(str(fine_path))
            reports.append(
                {
                    "file": str(fine_path),
                    "source_file": str(src_path),
                    "changed": False,
                    "gap_changed": False,
                    "short_changed": False,
                    "source_missing": True,
                    "parse_error": False,
                    "write_mode": (
                        "source_missing_passthrough"
                        if bool(args.apply)
                        else "source_missing_no_write"
                    ),
                    "entities": {},
                    "inserted_segments": 0,
                    "inserted_labels_source": {},
                    "inserted_labels_target": {},
                    "mapping_mode_counts": {"local": 0, "global": 0, "fallback": 0},
                    "short_label_corrections": {},
                    "short_label_changed_frames": 0,
                }
            )
            print(f"[MISS] {fine_path} source missing: {src_path}")
            continue
        rep = process_one_file(
            fine_path=fine_path,
            src_path=src_path,
            out_path=out_path,
            global_name_map=global_name_map,
            min_conf=float(args.min_conf),
            apply_write=bool(args.apply),
            optimize_short=bool(args.optimize_short),
            short_min_len=int(args.short_min_len),
            short_expand_from=int(args.short_expand_from),
            short_max_passes=int(args.short_max_passes),
            short_max_change_ratio=float(args.short_max_change_ratio),
            short_boundary_pad=int(short_boundary_pad),
        )
        if rep is None:
            if bool(args.apply):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    fine_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
                files_written += 1
                files_unchanged_written += 1
            files_parse_error_passthrough += 1
            parse_error_files.append(str(fine_path))
            reports.append(
                {
                    "file": str(fine_path),
                    "source_file": str(src_path),
                    "changed": False,
                    "gap_changed": False,
                    "short_changed": False,
                    "source_missing": False,
                    "parse_error": True,
                    "write_mode": (
                        "parse_error_passthrough"
                        if bool(args.apply)
                        else "parse_error_no_write"
                    ),
                    "entities": {},
                    "inserted_segments": 0,
                    "inserted_labels_source": {},
                    "inserted_labels_target": {},
                    "mapping_mode_counts": {"local": 0, "global": 0, "fallback": 0},
                    "short_label_corrections": {},
                    "short_label_changed_frames": 0,
                }
            )
            print(f"[ERR ] {fine_path} parse error, passthrough")
            continue
        reports.append(rep)
        if bool(args.apply):
            files_written += 1
            if not rep.get("changed", False):
                files_unchanged_written += 1
        has_gap = any(x["gaps_before_count"] > 0 for x in rep["entities"].values())
        if has_gap:
            files_with_gaps += 1
        if rep.get("gap_changed"):
            gap_changed_files += 1
        if rep.get("short_changed"):
            short_changed_files += 1
        short_label_changed_frames_total += int(
            rep.get("short_label_changed_frames", 0)
        )
        short_label_correction_counter.update(rep.get("short_label_corrections", {}))
        if rep["changed"]:
            changed_files.append(rep["file"])
            inserted_total += int(rep["inserted_segments"])
            inserted_src_counter.update(rep["inserted_labels_source"])
            inserted_tgt_counter.update(rep["inserted_labels_target"])
            print(
                f"[FIX] {fine_path} inserted={rep['inserted_segments']} "
                f"gap_changed={rep.get('gap_changed', False)} short_changed={rep.get('short_changed', False)}"
            )
        else:
            print(f"[SKIP] {fine_path} no changes")

    summary = {
        "gaps_root": str(gaps_root),
        "source_root": str(source_root),
        "out_root": str(out_root),
        "apply": bool(args.apply),
        "min_conf": float(args.min_conf),
        "optimize_short": bool(args.optimize_short),
        "short_min_len": int(args.short_min_len),
        "short_expand_from": int(args.short_expand_from),
        "short_max_passes": int(args.short_max_passes),
        "short_max_change_ratio": float(args.short_max_change_ratio),
        "short_boundary_pad": int(short_boundary_pad),
        "input_files_total": len(fine_files),
        "processed_files": len(reports),
        "files_written": int(files_written),
        "files_unchanged_written": int(files_unchanged_written),
        "files_missing_source": int(files_missing_source),
        "files_parse_error_passthrough": int(files_parse_error_passthrough),
        "files_with_gaps": int(files_with_gaps),
        "files_changed": len(changed_files),
        "gap_changed_files": int(gap_changed_files),
        "short_changed_files": int(short_changed_files),
        "inserted_segments_total": int(inserted_total),
        "short_label_changed_frames_total": int(short_label_changed_frames_total),
        "missing_source_files": missing_source_files,
        "parse_error_files": parse_error_files,
        "changed_files": changed_files,
        "inserted_labels_source": dict(inserted_src_counter.most_common()),
        "inserted_labels_target": dict(inserted_tgt_counter.most_common()),
        "short_label_corrections": dict(short_label_correction_counter.most_common()),
        "files": reports,
    }

    out_root.mkdir(parents=True, exist_ok=True)
    report_json_path = out_root / args.report_json
    report_txt_path = out_root / args.report_txt
    report_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_text_summary(summary, report_txt_path)

    print(f"[DONE] processed={len(reports)} changed={len(changed_files)}")
    print(f"[DONE] report_json={report_json_path}")
    print(f"[DONE] report_txt={report_txt_path}")


if __name__ == "__main__":
    main()
