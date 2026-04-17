from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

MaskRLE = Dict[str, object]


def bbox_from_mask_rle(mask: MaskRLE) -> List[int]:
    """
    Derive bbox from mask coordinates; never treat bbox as primary truth.
    Expects mask as {"pixels": [[x, y], ...]}.
    """
    pixels = mask.get("pixels") if isinstance(mask, dict) else None
    if not isinstance(pixels, list) or not pixels:
        return [0, 0, 0, 0]
    xs: List[int] = []
    ys: List[int] = []
    for item in pixels:
        if not (isinstance(item, list) or isinstance(item, tuple)) or len(item) != 2:
            continue
        try:
            xs.append(int(item[0]))
            ys.append(int(item[1]))
        except Exception:
            continue
    if not xs or not ys:
        return [0, 0, 0, 0]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return [x_min, y_min, max(0, x_max - x_min + 1), max(0, y_max - y_min + 1)]


def bbox_is_valid(bbox: List[int] | Tuple[int, ...] | None) -> bool:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    try:
        return int(bbox[2]) > 0 and int(bbox[3]) > 0
    except Exception:
        return False


def bbox_area(bbox: List[int] | Tuple[int, ...] | None) -> int:
    if not bbox_is_valid(bbox):
        return 0
    return max(0, int(bbox[2])) * max(0, int(bbox[3]))


def bbox_iou(
    bbox_a: List[int] | Tuple[int, ...] | None,
    bbox_b: List[int] | Tuple[int, ...] | None,
) -> float:
    if not bbox_is_valid(bbox_a) or not bbox_is_valid(bbox_b):
        return 0.0
    ax, ay, aw, ah = [int(v) for v in (bbox_a or [0, 0, 0, 0])[:4]]
    bx, by, bw, bh = [int(v) for v in (bbox_b or [0, 0, 0, 0])[:4]]
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = bbox_area(bbox_a) + bbox_area(bbox_b) - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _has_mask_pixels(mask: MaskRLE) -> bool:
    pixels = mask.get("pixels") if isinstance(mask, dict) else None
    return isinstance(pixels, list) and len(pixels) > 0


def _to_pixel_set(mask: MaskRLE) -> set:
    pixels = mask.get("pixels") if isinstance(mask, dict) else None
    out = set()
    if not isinstance(pixels, list):
        return out
    for item in pixels:
        if not (isinstance(item, list) or isinstance(item, tuple)) or len(item) != 2:
            continue
        try:
            out.add((int(item[0]), int(item[1])))
        except Exception:
            continue
    return out


def mask_iou(mask_a: MaskRLE, mask_b: MaskRLE) -> float:
    pa = _to_pixel_set(mask_a)
    pb = _to_pixel_set(mask_b)
    if not pa and not pb:
        return 1.0
    if not pa or not pb:
        return 0.0
    inter = len(pa.intersection(pb))
    union = len(pa.union(pb))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def mask_area(mask: MaskRLE) -> int:
    return len(_to_pixel_set(mask))


def mask_or_bbox_area(mask: MaskRLE, bbox: List[int] | Tuple[int, ...] | None = None) -> int:
    if _has_mask_pixels(mask):
        return mask_area(mask)
    return bbox_area(bbox)


def mask_centroid(mask: MaskRLE) -> Tuple[float, float]:
    pixels = list(_to_pixel_set(mask))
    if not pixels:
        return 0.0, 0.0
    sx = 0.0
    sy = 0.0
    for x, y in pixels:
        sx += float(x)
        sy += float(y)
    n = float(len(pixels))
    return sx / n, sy / n


def touches(mask_a: MaskRLE, mask_b: MaskRLE) -> bool:
    pa = _to_pixel_set(mask_a)
    pb = _to_pixel_set(mask_b)
    if not pa or not pb:
        return False
    # 8-neighborhood touching criterion.
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    for x, y in pa:
        for dx, dy in neigh:
            if (x + dx, y + dy) in pb:
                return True
    return False


def mask_or_bbox_iou(
    mask_a: MaskRLE,
    mask_b: MaskRLE,
    *,
    bbox_a: List[int] | Tuple[int, ...] | None = None,
    bbox_b: List[int] | Tuple[int, ...] | None = None,
) -> float:
    if _has_mask_pixels(mask_a) and _has_mask_pixels(mask_b):
        return mask_iou(mask_a, mask_b)
    if bbox_a is None and _has_mask_pixels(mask_a):
        bbox_a = bbox_from_mask_rle(mask_a)
    if bbox_b is None and _has_mask_pixels(mask_b):
        bbox_b = bbox_from_mask_rle(mask_b)
    return bbox_iou(bbox_a, bbox_b)


def bbox_touches(
    bbox_a: List[int] | Tuple[int, ...] | None,
    bbox_b: List[int] | Tuple[int, ...] | None,
    *,
    tolerance: int = 1,
) -> bool:
    if not bbox_is_valid(bbox_a) or not bbox_is_valid(bbox_b):
        return False
    ax, ay, aw, ah = [int(v) for v in (bbox_a or [0, 0, 0, 0])[:4]]
    bx, by, bw, bh = [int(v) for v in (bbox_b or [0, 0, 0, 0])[:4]]
    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh
    gap_x = max(ax - bx2, bx - ax2, 0)
    gap_y = max(ay - by2, by - ay2, 0)
    return gap_x <= int(tolerance) and gap_y <= int(tolerance)


def touches_or_bbox(
    mask_a: MaskRLE,
    mask_b: MaskRLE,
    *,
    bbox_a: List[int] | Tuple[int, ...] | None = None,
    bbox_b: List[int] | Tuple[int, ...] | None = None,
    tolerance: int = 1,
) -> bool:
    if _has_mask_pixels(mask_a) and _has_mask_pixels(mask_b):
        return touches(mask_a, mask_b)
    if bbox_a is None and _has_mask_pixels(mask_a):
        bbox_a = bbox_from_mask_rle(mask_a)
    if bbox_b is None and _has_mask_pixels(mask_b):
        bbox_b = bbox_from_mask_rle(mask_b)
    return bbox_touches(bbox_a, bbox_b, tolerance=tolerance)
