from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def majority_smooth(arr: np.ndarray, k: int) -> np.ndarray:
    """Simple majority filter over a 1D label sequence."""
    if k <= 1:
        return arr
    k = k if k % 2 == 1 else k + 1
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    out = np.empty_like(arr)
    for i in range(len(arr)):
        win = padded[i : i + k]
        vals, cnts = np.unique(win, return_counts=True)
        out[i] = vals[np.argmax(cnts)]
    return out


def segments_from_labels(labels: np.ndarray) -> List[Tuple[int, int, int]]:
    """Convert per-frame labels into contiguous (start, end, class_id) runs."""
    if labels.size == 0:
        return []
    segs: List[Tuple[int, int, int]] = []
    start = 0
    cur = int(labels[0])
    for idx in range(1, len(labels)):
        val = int(labels[idx])
        if val != cur:
            segs.append((start, idx - 1, cur))
            start = idx
            cur = val
    segs.append((start, len(labels) - 1, cur))
    return segs


def apply_min_seg_len(
    labels: np.ndarray, probs: np.ndarray, min_len: int
) -> np.ndarray:
    """
    Merge short segments, then re-assign segment labels using mean probabilities.
    """
    if min_len <= 1 or labels.size == 0:
        return labels

    segs = segments_from_labels(labels)
    i = 0
    while i < len(segs):
        s, e, _lab = segs[i]
        seg_len = e - s + 1
        if seg_len >= min_len or len(segs) == 1:
            i += 1
            continue
        if i == 0:
            target = 1
        elif i == len(segs) - 1:
            target = i - 1
        else:
            left_label = segs[i - 1][2]
            right_label = segs[i + 1][2]
            seg_probs = probs[s : e + 1]
            left_score = (
                float(seg_probs[:, left_label].mean()) if seg_probs.size else 0.0
            )
            right_score = (
                float(seg_probs[:, right_label].mean()) if seg_probs.size else 0.0
            )
            target = i - 1 if left_score >= right_score else i + 1
        if target < i:
            ls, _le, llab = segs[target]
            segs[target] = (ls, e, llab)
            segs.pop(i)
            i = max(target - 1, 0)
        else:
            _rs, re, rlab = segs[target]
            segs[target] = (s, re, rlab)
            segs.pop(i)
            i = max(target - 1, 0)

    new_segs: List[Tuple[int, int, int]] = []
    for s, e, _ in segs:
        seg_probs = probs[s : e + 1]
        if seg_probs.size:
            new_label = int(seg_probs.mean(axis=0).argmax())
        else:
            new_label = int(labels[s])
        if new_segs and new_segs[-1][2] == new_label and new_segs[-1][1] + 1 >= s:
            new_segs[-1] = (new_segs[-1][0], e, new_label)
        else:
            new_segs.append((s, e, new_label))

    out = labels.copy()
    for s, e, lab in new_segs:
        out[s : e + 1] = lab
    return out


def merge_segments_by_label(
    per_frame: np.ndarray,
    class_names: Sequence[str],
    fallback_prefix: str = "cls_",
    max_len: Optional[int] = None,
) -> List[Dict[str, int | str]]:
    """
    Convert per-frame labels into export segments.

    max_len:
      - None: split only when class changes.
      - int: force split when segment reaches max_len frames.
    """
    total = len(per_frame)
    if total == 0:
        return []

    names = list(class_names)
    segs: List[Dict[str, int | str]] = []
    start = 0
    cur = int(per_frame[0])

    for idx in range(1, total):
        need_split = int(per_frame[idx]) != cur
        too_long = max_len is not None and (idx - start) >= int(max_len)
        if need_split or too_long:
            segs.append(
                {
                    "start_frame": int(start),
                    "end_frame": int(idx - 1),
                    "class_id": int(cur),
                    "class_name": (
                        names[cur] if cur < len(names) else f"{fallback_prefix}{cur}"
                    ),
                }
            )
            start = idx
            cur = int(per_frame[idx])

    segs.append(
        {
            "start_frame": int(start),
            "end_frame": int(total - 1),
            "class_id": int(cur),
            "class_name": names[cur] if cur < len(names) else f"{fallback_prefix}{cur}",
        }
    )
    return segs
