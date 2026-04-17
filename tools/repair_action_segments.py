#!/usr/bin/env python3
"""
Repair action-segmentation JSON files by:
1) Closing internal gaps per entity track.
2) Smoothing short segments with a minimum duration rule.

Rules (default):
- len >= 15: keep.
- len in [13, 14]: expand from neighbors to 15 when possible.
- len < 13: merge into neighbor labels.

Typical usage:
python tools/repair_action_segments.py ^
  --input "new_label_val/new label_val" ^
  --out_dir "new_label_val/repaired" ^
  --min_len 15 ^
  --expand_from 13
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Run:
    start: int
    end: int
    label: Any
    entity: Optional[str]

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class TrackStats:
    gaps_before: int = 0
    gap_frames_before: int = 0
    gaps_closed: int = 0
    gap_frames_closed: int = 0
    overlaps_resolved_frames: int = 0
    fully_covered_segments_dropped: int = 0
    short_lt_min_before: int = 0
    short_lt_expand_before: int = 0
    expanded_runs: int = 0
    expanded_frames: int = 0
    merged_runs: int = 0
    short_lt_min_after: int = 0
    gaps_after: int = 0
    gap_frames_after: int = 0


def gather_json_inputs(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted([p for p in path.rglob("*.json") if p.is_file()])


def count_gaps(runs: List[Run]) -> Tuple[int, int]:
    gaps = 0
    gap_frames = 0
    for i in range(1, len(runs)):
        gap = runs[i].start - runs[i - 1].end - 1
        if gap > 0:
            gaps += 1
            gap_frames += gap
    return gaps, gap_frames


def coalesce_runs(runs: List[Run]) -> List[Run]:
    if not runs:
        return []
    runs = sorted(runs, key=lambda r: (r.start, r.end))
    out = [runs[0]]
    for cur in runs[1:]:
        prev = out[-1]
        if cur.label == prev.label and cur.start <= prev.end + 1:
            prev.end = max(prev.end, cur.end)
        else:
            out.append(cur)
    return out


def normalize_non_overlap(runs: List[Run], stats: TrackStats) -> List[Run]:
    if not runs:
        return []
    runs = sorted(runs, key=lambda r: (r.start, r.end))
    out: List[Run] = []
    for r in runs:
        if r.end < r.start:
            continue
        if not out:
            out.append(r)
            continue
        prev = out[-1]
        if r.start <= prev.end:
            overlap = prev.end - r.start + 1
            stats.overlaps_resolved_frames += max(0, overlap)
            new_start = prev.end + 1
            if new_start > r.end:
                stats.fully_covered_segments_dropped += 1
                continue
            r.start = new_start
        out.append(r)
    return out


def close_internal_gaps(runs: List[Run], stats: TrackStats) -> None:
    for i in range(1, len(runs)):
        left = runs[i - 1]
        right = runs[i]
        gap = right.start - left.end - 1
        if gap <= 0:
            continue
        # Midpoint split to avoid bias.
        left_add = gap // 2
        right_add = gap - left_add
        left.end += left_add
        right.start -= right_add
        stats.gaps_closed += 1
        stats.gap_frames_closed += gap


def try_expand_run(
    runs: List[Run],
    idx: int,
    min_len: int,
    stats: TrackStats,
) -> bool:
    cur = runs[idx]
    need = min_len - cur.length
    if need <= 0:
        return False

    left_idx = idx - 1 if idx > 0 else None
    right_idx = idx + 1 if idx + 1 < len(runs) else None
    if left_idx is None and right_idx is None:
        return False

    left_avail = runs[left_idx].length - 1 if left_idx is not None else 0
    right_avail = runs[right_idx].length - 1 if right_idx is not None else 0
    left_avail = max(0, left_avail)
    right_avail = max(0, right_avail)

    if left_avail + right_avail < need:
        return False

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
        return False

    if take_left > 0 and left_idx is not None:
        runs[left_idx].end -= take_left
        cur.start -= take_left
    if take_right > 0 and right_idx is not None:
        runs[right_idx].start += take_right
        cur.end += take_right

    stats.expanded_runs += 1
    stats.expanded_frames += need
    return True


def merge_short_run(runs: List[Run], idx: int, stats: TrackStats) -> bool:
    if not runs:
        return False
    left_idx = idx - 1 if idx > 0 else None
    right_idx = idx + 1 if idx + 1 < len(runs) else None
    if left_idx is None and right_idx is None:
        return False

    target_label: Any
    if left_idx is not None and right_idx is not None:
        left = runs[left_idx]
        right = runs[right_idx]
        if left.label == right.label:
            target_label = left.label
        elif left.length >= right.length:
            target_label = left.label
        else:
            target_label = right.label
    elif left_idx is not None:
        target_label = runs[left_idx].label
    else:
        target_label = runs[right_idx].label

    runs[idx].label = target_label
    stats.merged_runs += 1
    return True


def smooth_runs(
    runs: List[Run],
    min_len: int,
    expand_from: int,
    max_passes: int,
    stats: TrackStats,
) -> List[Run]:
    runs = coalesce_runs(runs)
    stats.short_lt_min_before = sum(1 for r in runs if r.length < min_len)
    stats.short_lt_expand_before = sum(1 for r in runs if r.length < expand_from)

    for _ in range(max_passes):
        changed = False
        i = 0
        while i < len(runs):
            cur_len = runs[i].length
            if cur_len >= min_len:
                i += 1
                continue

            did = False
            if expand_from <= cur_len < min_len:
                did = try_expand_run(runs, i, min_len=min_len, stats=stats)
                if did:
                    changed = True
                    runs = coalesce_runs(runs)
                    i = max(0, i - 1)
                    continue

            did = merge_short_run(runs, i, stats=stats)
            if did:
                changed = True
                runs = coalesce_runs(runs)
                i = max(0, i - 1)
                continue
            i += 1
        if not changed:
            break

    stats.short_lt_min_after = sum(1 for r in runs if r.length < min_len)
    return runs


def process_entity_runs(
    runs: List[Run],
    min_len: int,
    expand_from: int,
    max_passes: int,
) -> Tuple[List[Run], TrackStats]:
    stats = TrackStats()
    runs = normalize_non_overlap(runs, stats)
    before_gaps, before_gap_frames = count_gaps(runs)
    stats.gaps_before = before_gaps
    stats.gap_frames_before = before_gap_frames
    close_internal_gaps(runs, stats)
    runs = smooth_runs(
        runs=runs,
        min_len=min_len,
        expand_from=expand_from,
        max_passes=max_passes,
        stats=stats,
    )
    after_gaps, after_gap_frames = count_gaps(runs)
    stats.gaps_after = after_gaps
    stats.gap_frames_after = after_gap_frames
    return runs, stats


def process_one_file(
    in_path: Path,
    out_path: Path,
    min_len: int,
    expand_from: int,
    max_passes: int,
) -> Optional[Dict[str, Any]]:
    try:
        raw = json.loads(in_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    segs = raw.get("segments")
    if not isinstance(segs, list):
        return None

    entity_groups: Dict[Optional[str], List[Run]] = {}
    for seg in segs:
        if not isinstance(seg, dict):
            continue
        if (
            "start_frame" not in seg
            or "end_frame" not in seg
            or "action_label" not in seg
        ):
            continue
        try:
            start = int(seg.get("start_frame"))
            end = int(seg.get("end_frame"))
        except Exception:
            continue
        entity = seg.get("entity")
        entity_key = None if entity is None else str(entity)
        run = Run(
            start=start,
            end=end,
            label=seg.get("action_label"),
            entity=entity_key,
        )
        entity_groups.setdefault(entity_key, []).append(run)

    repaired_runs: List[Run] = []
    per_entity_report: Dict[str, Any] = {}
    for entity_key, runs in entity_groups.items():
        runs_out, st = process_entity_runs(
            runs=runs,
            min_len=min_len,
            expand_from=expand_from,
            max_passes=max_passes,
        )
        repaired_runs.extend(runs_out)
        report_key = entity_key if entity_key is not None else "<none>"
        per_entity_report[report_key] = {
            "segments_in": len(runs),
            "segments_out": len(runs_out),
            "gaps_before": st.gaps_before,
            "gap_frames_before": st.gap_frames_before,
            "gaps_closed": st.gaps_closed,
            "gap_frames_closed": st.gap_frames_closed,
            "gaps_after": st.gaps_after,
            "gap_frames_after": st.gap_frames_after,
            "short_lt_min_before": st.short_lt_min_before,
            "short_lt_expand_before": st.short_lt_expand_before,
            "expanded_runs": st.expanded_runs,
            "expanded_frames": st.expanded_frames,
            "merged_runs": st.merged_runs,
            "short_lt_min_after": st.short_lt_min_after,
            "overlaps_resolved_frames": st.overlaps_resolved_frames,
            "fully_covered_segments_dropped": st.fully_covered_segments_dropped,
        }

    repaired_runs.sort(
        key=lambda r: (r.start, r.end, "" if r.entity is None else r.entity)
    )
    out_segments: List[Dict[str, Any]] = []
    for r in repaired_runs:
        row: Dict[str, Any] = {
            "action_label": r.label,
            "start_frame": int(r.start),
            "end_frame": int(r.end),
        }
        if r.entity is not None:
            row["entity"] = r.entity
        out_segments.append(row)

    out_data = dict(raw)
    out_data["segments"] = out_segments
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "file": str(in_path),
        "out_file": str(out_path),
        "segments_in": len(segs),
        "segments_out": len(out_segments),
        "entities": per_entity_report,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", required=True, type=str, help="Input JSON file or directory"
    )
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--min_len", type=int, default=15, help="Minimum segment length")
    ap.add_argument(
        "--expand_from",
        type=int,
        default=13,
        help="Runs in [expand_from, min_len-1] try expansion first; shorter ones merge",
    )
    ap.add_argument(
        "--max_passes", type=int, default=12, help="Maximum smoothing passes"
    )
    ap.add_argument(
        "--report_json",
        type=str,
        default="repair_report.json",
        help="Report filename under out_dir",
    )
    args = ap.parse_args()

    if args.min_len <= 1:
        raise ValueError("--min_len must be >= 2")
    if not (1 <= args.expand_from <= args.min_len):
        raise ValueError("--expand_from must be in [1, min_len]")

    in_root = Path(args.input)
    out_root = Path(args.out_dir)
    inputs = gather_json_inputs(in_root)

    reports: List[Dict[str, Any]] = []
    skipped = 0
    for in_path in inputs:
        # Skip non-annotation json patterns if any.
        lname = in_path.name.lower()
        if lname.endswith("_mapping_report.json"):
            skipped += 1
            continue

        rel = in_path.name if in_root.is_file() else in_path.relative_to(in_root)
        out_path = out_root / rel
        report = process_one_file(
            in_path=in_path,
            out_path=out_path,
            min_len=args.min_len,
            expand_from=args.expand_from,
            max_passes=args.max_passes,
        )
        if report is None:
            skipped += 1
            continue
        reports.append(report)
        print(f"[OK] {in_path} -> {out_path}")

    report_path = out_root / args.report_json
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "input": str(in_root),
        "out_dir": str(out_root),
        "min_len": args.min_len,
        "expand_from": args.expand_from,
        "max_passes": args.max_passes,
        "processed_files": len(reports),
        "skipped_files": skipped,
        "files": reports,
    }
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] processed={len(reports)} skipped={skipped}")
    print(f"[DONE] report={report_path}")


if __name__ == "__main__":
    main()
