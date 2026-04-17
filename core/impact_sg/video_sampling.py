from __future__ import annotations

from typing import List


def sample_frame_indices(
    frame_count: int,
    source_fps: float,
    sampling_fps: float,
    *,
    include_last_frame: bool = True,
) -> List[int]:
    total = max(0, int(frame_count))
    if total <= 0:
        return []
    if total == 1:
        return [0]

    src = max(0.1, float(source_fps or 0.0))
    sample = max(0.1, float(sampling_fps or 0.0))
    if sample >= src:
        return list(range(total))

    interval_frames = max(1.0, src / sample)
    out: List[int] = []
    pos = 0.0
    while True:
        frame_idx = int(round(pos))
        if frame_idx >= total:
            break
        if not out or out[-1] != frame_idx:
            out.append(frame_idx)
        pos += interval_frames

    if include_last_frame and out[-1] != total - 1:
        out.append(total - 1)
    return out
