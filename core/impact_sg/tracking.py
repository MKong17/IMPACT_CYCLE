from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


def _removed() -> RuntimeError:
    return RuntimeError(
        "Tracking logic has been removed from this project."
    )


def clip_bbox(bbox: Sequence[float], frame_w: int, frame_h: int) -> List[int]:
    raise _removed()


def bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    raise _removed()


def center_distance(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> float:
    raise _removed()


def smooth_bbox(prev_bbox: Sequence[float], new_bbox: Sequence[float], *, alpha: float = 0.8) -> List[int]:
    raise _removed()


def crop_patch(frame_bgr, bbox: Sequence[float], *, max_side: int = 96):
    raise _removed()


def update_template(old_template, new_template, *, alpha: float = 0.25):
    raise _removed()


def template_track_bbox(
    frame_bgr,
    template_gray,
    prev_bbox: Sequence[float],
    *,
    search_radius: int = 72,
    min_response: float = 0.35,
):
    raise _removed()


def greedy_track_match(
    tracks: Dict[str, Dict[str, object]],
    detections: List[Dict[str, object]],
    *,
    current_frame_idx: Optional[int] = None,
    short_gap_frames: int = 24,
    long_gap_frames: int = 120,
    max_track_gap_frames: int = 240,
    min_match_score_short: float = 0.2,
    min_match_score_long: float = 0.35,
    max_center_distance_factor_short: float = 1.8,
    max_center_distance_factor_long: float = 1.2,
) -> List[Tuple[str, int, float]]:
    raise _removed()
