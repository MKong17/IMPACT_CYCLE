from typing import List, Callable, Optional, Tuple
import bisect
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QFontMetrics
from PyQt5.QtWidgets import (
    QWidget,
    QScrollArea,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QCheckBox,
    QSizePolicy,
    QToolButton,
    QApplication,
)
from core.models import AnnotationStore, LabelDef
from utils.constants import (
    color_from_key,
    ROW_HEIGHT,
    EDGE_TOLERANCE_PX,
    DEFAULT_VIEW_SPAN,
    SNAP_RADIUS_FRAMES,
    CURRENT_FRAME_SNAP_RADIUS_FRAMES,
    PREFER_FORWARD,
    MIN_VIEW_SPAN,
    EDGE_SNAP_FRAMES,
    EXTRA_LABEL_NAME,
    EXTRA_ALIASES,
    is_extra_label,
)
import weakref

try:
    import sip  # type: ignore
except Exception:
    try:
        from PyQt5 import sip  # type: ignore
    except Exception:
        sip = None
from dataclasses import dataclass


# ---- helper: contiguous runs ----
def frames_to_runs(frames: List[int]) -> List[Tuple[int, int]]:
    if not frames:
        return []
    frames = sorted(frames)
    runs, s, e = [], frames[0], frames[0]
    for f in frames[1:]:
        if f == e + 1:
            e = f
        else:
            runs.append((s, e))
            s = e = f
    runs.append((s, e))
    return runs


def _safe_qt_call(ref, method: str, *args):
    obj = ref()
    if obj is None:
        return
    if sip is not None:
        try:
            if sip.isdeleted(obj):
                return
        except Exception:
            pass
    try:
        getattr(obj, method)(*args)
    except Exception:
        pass


class BaseTimelineRow(QWidget):
    """Shared geometry + grid/marker drawing for timeline rows."""

    def __init__(
        self,
        get_frame_count: Callable[[], int],
        get_view_start: Callable[[], int],
        get_view_span: Callable[[], int],
        get_fps: Callable[[], int],
        get_gutter: Callable[[], int],
        parent=None,
    ):
        super().__init__(parent)
        self.get_fc = get_frame_count
        self.get_vs = get_view_start
        self.get_span = get_view_span
        self.get_fps = get_fps
        self.get_gutter = get_gutter
        self._hover_frame: Optional[int] = None
        self.current_frame: Optional[int] = None
        self._row_dragging = False
        self._row_drag_active = False
        self._row_drag_start = None
        self._timeline_ref = None
        self._edit_mask_spans: Optional[List[Tuple[int, int]]] = None

    def frame_to_x(self, f: int) -> int:
        g = self.get_gutter()
        span = max(1, self.get_span())
        avail = max(1, self.width() - g)
        return g + int((f - self.get_vs()) * avail / span)

    def x_to_frame(self, x: int) -> int:
        g = self.get_gutter()
        span = max(1, self.get_span())
        avail = max(1, self.width() - g)
        x_adj = max(0, min(avail, x - g))
        return int(round(self.get_vs() + x_adj * span / avail))

    def set_current_frame(self, f: Optional[int]):
        self.current_frame = f
        self.update()

    @staticmethod
    def _normalize_spans(
        spans: Optional[List[Tuple[int, int]]],
    ) -> Optional[List[Tuple[int, int]]]:
        if spans is None:
            return None
        cleaned = []
        for seg in spans:
            try:
                s = int(seg[0])
                e = int(seg[1])
            except Exception:
                continue
            if e < s:
                s, e = e, s
            cleaned.append((s, e))
        if not cleaned:
            return []
        cleaned.sort(key=lambda x: x[0])
        merged = []
        cur_s, cur_e = cleaned[0]
        for s, e in cleaned[1:]:
            if s <= cur_e + 1:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    def set_edit_mask_spans(self, spans: Optional[List[Tuple[int, int]]]) -> None:
        self._edit_mask_spans = self._normalize_spans(spans)
        self.update()

    def _frame_in_edit_mask(self, frame: int) -> bool:
        spans = self._edit_mask_spans
        if spans is None:
            return True
        for s, e in spans:
            if int(s) <= int(frame) <= int(e):
                return True
            if int(frame) < int(s):
                break
        return False

    def _interval_in_edit_mask(self, start: int, end: int) -> bool:
        spans = self._edit_mask_spans
        if spans is None:
            return True
        if end < start:
            start, end = end, start
        for s, e in spans:
            if int(s) <= int(start) <= int(e):
                return int(end) <= int(e)
            if int(start) < int(s):
                break
        return False

    def _non_editable_visible_runs(self, start: int, end: int) -> List[Tuple[int, int]]:
        spans = self._edit_mask_spans
        if spans is None:
            return []
        if end < start:
            start, end = end, start
        blocked = []
        cursor = int(start)
        for s, e in spans:
            s = int(s)
            e = int(e)
            if e < cursor:
                continue
            if s > int(end):
                break
            if cursor < s:
                blocked.append((cursor, min(int(end), s - 1)))
            cursor = max(cursor, e + 1)
            if cursor > int(end):
                break
        if cursor <= int(end):
            blocked.append((cursor, int(end)))
        return blocked

    def _draw_non_editable_overlay(self, p: QPainter, start: int, end: int) -> None:
        blocked = self._non_editable_visible_runs(start, end)
        if not blocked:
            return
        base_fill = QColor(110, 110, 110, 70)
        hatch_a = QBrush(QColor(75, 75, 75, 150), Qt.BDiagPattern)
        hatch_b = QBrush(QColor(75, 75, 75, 120), Qt.FDiagPattern)
        edge_pen = QPen(QColor(95, 95, 95, 165), 1)
        for s, e in blocked:
            x1 = self.frame_to_x(int(s))
            x2 = self.frame_to_x(int(e) + 1)
            rect = QRect(x1, 0, max(1, x2 - x1), self.height())
            p.fillRect(rect, base_fill)
            p.setBrush(hatch_a)
            p.setPen(Qt.NoPen)
            p.drawRect(rect)
            p.setBrush(hatch_b)
            p.setPen(Qt.NoPen)
            p.drawRect(rect)
            p.setBrush(Qt.NoBrush)
            p.setPen(edge_pen)
            p.drawRect(rect)

    def _draw_time_grid(self, p: QPainter, start: int, end: int, fps: int):
        g = self.get_gutter()
        span = self.get_span()
        # gutter separator line
        p.setPen(QPen(QColor(200, 200, 200)))
        p.drawLine(g, 0, g, self.height())

        def step_px(step_frames: int) -> float:
            avail = max(1, self.width() - g)
            return step_frames * avail / max(1, span)

        s_minor = max(1, int(round(fps * 0.1)))
        if step_px(s_minor) >= 4:
            p.setPen(QPen(QColor(210, 210, 210)))
            first = (start // s_minor) * s_minor
            for f in range(first, end + s_minor, s_minor):
                x = self.frame_to_x(f)
                p.drawLine(x, 12, x, self.height() - 6)

        s_mid = max(1, int(round(fps * 0.5)))
        if step_px(s_mid) >= 6:
            p.setPen(QPen(QColor(190, 190, 190)))
            first = (start // s_mid) * s_mid
            for f in range(first, end + s_mid, s_mid):
                x = self.frame_to_x(f)
                p.drawLine(x, 8, x, self.height() - 4)

        s_major = fps
        p.setPen(QPen(QColor(160, 160, 160)))
        first = (start // s_major) * s_major
        for f in range(first, end + s_major, s_major):
            x = self.frame_to_x(f)
            p.drawLine(x, 4, x, self.height() - 2)
            if step_px(s_major) >= 60:
                sec = f // fps
                p.setPen(QPen(QColor(90, 90, 90)))
                p.setFont(QFont("Arial", 8))
                txt = f"{sec}s"
                w = p.fontMetrics().width(txt)
                p.drawText(x - w // 2, 12, txt)
                p.setPen(QPen(QColor(160, 160, 160)))

    def _draw_gutter_title(self, p: QPainter, text: str):
        p.setPen(QPen(QColor(60, 60, 60)))
        p.setFont(QFont("Arial", 9))
        p.drawText(6, self.height() - 8, text)

    def _draw_current_frame_marker(self, p: QPainter, start: int, end: int):
        if self.current_frame is None or not (start <= self.current_frame <= end):
            return
        x = self.frame_to_x(self.current_frame)
        p.setPen(QPen(QColor(220, 0, 0), 2))
        p.drawLine(x, 0, x, self.height())

    def _draw_hover_marker(
        self, p: QPainter, start: int, end: int, fps: int, text: str
    ):
        if self._hover_frame is None or not (start <= self._hover_frame <= end):
            return
        x = self.frame_to_x(self._hover_frame)
        p.setPen(QPen(QColor(50, 120, 255, 180), 1, Qt.DashLine))
        p.drawLine(x, 0, x, self.height())
        p.setFont(QFont("Arial", 8))
        w = p.fontMetrics().width(text) + 8
        h = p.fontMetrics().height() + 6
        rx = x + 6
        if rx + w > self.width():
            rx = x - w - 6
        bubble = QRect(rx, 2, w, h)
        p.setBrush(QColor(255, 255, 255, 220))
        p.setPen(QPen(QColor(80, 80, 80)))
        p.drawRoundedRect(bubble, 3, 3)
        p.drawText(bubble.adjusted(4, 2, -4, -2), Qt.AlignLeft | Qt.AlignVCenter, text)


class TimelineRow(BaseTimelineRow):
    hoverFrame = pyqtSignal(int)
    changed = pyqtSignal()

    def __init__(
        self,
        label: LabelDef,
        store: AnnotationStore,
        get_frame_count: Callable[[], int],
        get_view_start: Callable[[], int],
        get_view_span: Callable[[], int],
        get_fps: Callable[[], int],
        get_gutter: Callable[[], int],
        title_prefix: str = "",
        parent=None,
    ):
        super().__init__(
            get_frame_count, get_view_start, get_view_span, get_fps, get_gutter, parent
        )
        self.label = label
        self.store = store
        self.title_prefix = title_prefix

        self.setMouseTracking(True)
        self.setMinimumHeight(ROW_HEIGHT)

        # drag state
        self._dragging = False
        self._mode: Optional[str] = None  # "create"|"move"|"resize_left"|"resize_right"
        self._preview_interval: Optional[Tuple[int, int]] = None
        self._active_interval: Optional[Tuple[int, int]] = None
        self._grab_offset_frames: int = 0

        # hover overlay
        self.current_hit: bool = False  # highlight when current frame falls in this row

        # create-mode anchor (fix start; only drag end)
        self._create_anchor: Optional[int] = None

        # optional snap-to-segment boundaries
        self._snap_segments: List[Tuple[int, int]] = []
        self._snap_starts: List[int] = []
        self._snap_ends: List[int] = []
        self._snap_end_set: set = set()
        self._snap_soft = False
        self._snap_radius = SNAP_RADIUS_FRAMES
        self._current_snap_radius = CURRENT_FRAME_SNAP_RADIUS_FRAMES
        self._frame_snap_radius = SNAP_RADIUS_FRAMES
        self._edge_snap_frames = EDGE_SNAP_FRAMES
        self._row_dragging = False
        self._row_drag_active = False
        self._row_drag_start = None
        self._timeline_ref = None

        # search highlight
        self.highlighted: bool = False
        self.delete_handler = None
        self.split_handler = None
        self._segment_cuts: List[int] = []

    def _gutter_px(self) -> int:
        """Left text gutter so bars don't cover the label name."""
        fm = self.fontMetrics()
        try:
            text_w = fm.horizontalAdvance(self.label.name)  # PyQt5 newer
        except AttributeError:
            text_w = fm.width(self.label.name)  # fallback
        return max(80, text_w + 16)  # minimum 80px; text width + padding

    def sizeHint(self):
        return QSize(800, ROW_HEIGHT)

    # painting
    def paintEvent(self, e):
        p = QPainter(self)
        bg = QColor(240, 240, 240)
        if self.highlighted:
            bg = QColor(255, 250, 230)
        if self.current_hit:
            bg = QColor(245, 250, 255)
        p.fillRect(self.rect(), bg)

        start = self.get_vs()
        span = self.get_span()
        end = start + span
        fps = max(1, self.get_fps())

        self._draw_time_grid(p, start, end, fps)
        self._draw_gutter_title(p, f"{self.title_prefix}{self.label.name}")

        # committed intervals
        color = color_from_key(self.label.color_name)
        fill = QBrush(color.lighter(100))
        border = QPen(color.darker(130), 2)
        for s, e_ in self._label_runs():
            if e_ < start or s > end:
                continue
            s_vis = max(s, start)
            e_vis = min(e_, end)
            x1 = self.frame_to_x(s_vis)
            x2 = self.frame_to_x(e_vis + 1)
            rect = QRect(x1, 6, max(4, x2 - x1), self.height() - 12)
            p.setBrush(fill)
            p.setPen(border)
            p.drawRoundedRect(rect, 4, 4)
            h = rect.height()
            handle_w = 6
            p.fillRect(
                QRect(rect.left() - handle_w // 2, rect.top(), handle_w, h),
                color.darker(120),
            )
            p.fillRect(
                QRect(rect.right() - handle_w // 2, rect.top(), handle_w, h),
                color.darker(120),
            )

        # preview interval
        if self._preview_interval is not None:
            s, e_ = self._preview_interval
            if not (e_ < start or s > end):
                s_vis = max(s, start)
                e_vis = min(e_, end)
                x1 = self.frame_to_x(s_vis)
                x2 = self.frame_to_x(e_vis + 1)
                rect = QRect(x1, 8, max(4, x2 - x1), self.height() - 16)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(color.darker(150), 2, Qt.DashLine))
                p.drawRoundedRect(rect, 4, 4)
        if self._segment_cuts:
            cuts = [c for c in self._segment_cuts if start <= int(c) <= end]
            if cuts:
                p.setPen(QPen(QColor(80, 80, 80, 140), 1, Qt.DotLine))
                for c in cuts:
                    x = self.frame_to_x(int(c))
                    p.drawLine(x, 0, x, self.height())

        self._draw_non_editable_overlay(p, start, end)
        self._draw_current_frame_marker(p, start, end)
        label = (
            f"F {self._hover_frame} | {self._hover_frame / fps:.2f}s"
            if self._hover_frame is not None
            else ""
        )
        self._draw_hover_marker(p, start, end, fps, label)

    # hit tests / utils
    def _label_runs(self) -> List[Tuple[int, int]]:
        runs = frames_to_runs(self.store.frames_of(self.label.name))
        if not runs or not self._segment_cuts:
            return runs
        cut_set = {int(c) for c in self._segment_cuts if c is not None}
        split_runs = []
        for s, e in runs:
            split_points = sorted(c for c in cut_set if int(s) < c <= int(e))
            seg_start = int(s)
            for cut in split_points:
                split_runs.append((seg_start, int(cut) - 1))
                seg_start = int(cut)
            split_runs.append((seg_start, int(e)))
        return split_runs

    def _hit_interval(self, x: int) -> Tuple[Optional[Tuple[int, int]], str]:
        for s, e in self._label_runs():
            x1 = self.frame_to_x(s)
            x2 = self.frame_to_x(e + 1)
            if x1 - EDGE_TOLERANCE_PX <= x <= x2 + EDGE_TOLERANCE_PX:
                if abs(x - x1) <= EDGE_TOLERANCE_PX:
                    return (s, e), "left"
                if abs(x - x2) <= EDGE_TOLERANCE_PX:
                    return (s, e), "right"
                if x1 < x < x2:
                    return (s, e), "center"
        return None, "none"

    def set_highlighted(self, on: bool):
        if self.highlighted != on:
            self.highlighted = on
            self.update()

    def set_delete_handler(self, handler):
        self.delete_handler = handler

    def set_split_handler(self, handler):
        self.split_handler = handler

    def set_segment_cuts(self, cuts: List[int]) -> None:
        cleaned = []
        for c in cuts or []:
            try:
                cleaned.append(int(c))
            except Exception:
                continue
        self._segment_cuts = sorted(set(cleaned))

    def set_current_snap_radius(self, radius: int) -> None:
        try:
            self._current_snap_radius = max(0, int(radius))
        except Exception:
            self._current_snap_radius = CURRENT_FRAME_SNAP_RADIUS_FRAMES

    def set_frame_snap_radius(self, radius: int) -> None:
        try:
            self._frame_snap_radius = max(0, int(radius))
        except Exception:
            self._frame_snap_radius = SNAP_RADIUS_FRAMES

    def set_edge_snap_frames(self, radius: int) -> None:
        try:
            self._edge_snap_frames = max(0, int(radius))
        except Exception:
            self._edge_snap_frames = EDGE_SNAP_FRAMES

    def set_segment_snap_radius(self, radius: int) -> None:
        try:
            self._snap_radius = max(0, int(radius))
        except Exception:
            self._snap_radius = SNAP_RADIUS_FRAMES

    def set_current_hit(self, on: bool):
        if self.current_hit != on:
            self.current_hit = on
            self.update()

    def _snap_unlabeled(self, target: int) -> Optional[int]:
        fc = self.get_fc()
        target = max(0, min(target, fc - 1))
        f = self.store.nearest_unlabeled(
            target, self._frame_snap_radius, prefer_forward=PREFER_FORWARD
        )
        if f is None:
            return None
        return max(0, min(f, fc - 1))

    def _snap_to_current(self, start: int, end: int) -> Tuple[int, int]:
        """Snap endpoints to current frame within a smaller playhead radius."""
        cf = self.current_frame
        if cf is None:
            return start, end
        if abs(start - cf) <= self._current_snap_radius:
            start = cf
        if abs(end - cf) <= self._current_snap_radius:
            end = cf
        return start, end

    def set_snap_segments(self, segments: List[Tuple[int, int]]) -> None:
        cleaned = []
        for seg in segments or []:
            try:
                s = int(seg[0])
                e = int(seg[1])
            except Exception:
                continue
            if e < s:
                s, e = e, s
            cleaned.append((s, e))
        cleaned.sort(key=lambda x: x[0])
        self._snap_segments = cleaned
        self._snap_starts = [s for s, _ in cleaned]
        self._snap_ends = sorted({e for _s, e in cleaned})
        self._snap_end_set = set(self._snap_ends)

    def _segment_bounds_for_frame(self, frame: int) -> Optional[Tuple[int, int]]:
        if not self._snap_segments:
            return None
        idx = bisect.bisect_right(self._snap_starts, int(frame)) - 1
        if idx < 0 or idx >= len(self._snap_segments):
            return None
        s, e = self._snap_segments[idx]
        if s <= frame <= e:
            return s, e
        return None

    def _nearest_in_list(self, values: List[int], frame: int) -> Optional[int]:
        if not values:
            return None
        frame = int(frame)
        idx = bisect.bisect_left(values, frame)
        candidates = []
        if idx < len(values):
            candidates.append(values[idx])
        if idx > 0:
            candidates.append(values[idx - 1])
        if not candidates:
            return None
        return min(candidates, key=lambda v: abs(v - frame))

    def _snap_to_segment_start(self, frame: int) -> int:
        if not self._snap_starts:
            return frame
        seg = self._segment_bounds_for_frame(frame)
        if seg:
            if not self._snap_soft:
                return seg[0]
            if abs(frame - seg[0]) <= self._snap_radius:
                return seg[0]
        nearest = self._nearest_in_list(self._snap_starts, frame)
        if nearest is None:
            return frame
        if self._snap_soft and abs(frame - nearest) > self._snap_radius:
            return frame
        return nearest

    def _snap_to_segment_end(self, frame: int) -> int:
        if not self._snap_ends:
            return frame
        seg = self._segment_bounds_for_frame(frame)
        if seg:
            if not self._snap_soft:
                return seg[1]
            if abs(frame - seg[1]) <= self._snap_radius:
                return seg[1]
        nearest = self._nearest_in_list(self._snap_ends, frame)
        if nearest is None:
            return frame
        if self._snap_soft and abs(frame - nearest) > self._snap_radius:
            return frame
        return nearest

    def _snap_move_start(self, cand_start: int, length: int) -> Optional[int]:
        if not self._snap_starts:
            return cand_start
        if length < 0:
            return None
        best = None
        best_dist = None
        for s in self._snap_starts:
            if (s + length) not in self._snap_end_set:
                continue
            dist = abs(s - cand_start)
            if best is None or dist < best_dist:
                best = s
                best_dist = dist
        if best is None:
            return None
        if self._snap_soft and best_dist is not None and best_dist > self._snap_radius:
            return cand_start
        return best

    # --- Snap helpers: only-left boundary snap to e+1 ---
    def _is_occ_here(self, f: int) -> bool:
        """Optional: if store.is_occupied supports row/entity queries, pass the current row; otherwise fall back to global."""
        try:
            return self.store.is_occupied(f, row=getattr(self, "row_key", None))
        except TypeError:
            return self.store.is_occupied(f)

    def _snap_edge_after_label_left(self, target: int) -> int:
        """
        Search left within EDGE_SNAP_FRAMES for the occupied->free boundary and return the next frame (e+1).
        Return -1 when no boundary is found (no snap).
        """
        fc = max(1, self.get_fc())
        t = max(0, min(target, fc - 1))
        for d in range(0, self._edge_snap_frames + 1):
            cand = t - d
            if (
                cand >= 1
                and (not self._is_occ_here(cand))
                and self._is_occ_here(cand - 1)
            ):
                return cand
        return -1

    def _interval_clamped_free(self, a: int, b: int) -> Optional[Tuple[int, int]]:
        fc = self.get_fc()
        a = max(0, min(a, fc - 1))
        b = max(0, min(b, fc - 1))
        if a > b:
            a, b = b, a
        # Manual trim cuts are virtual split markers and should not hard-block
        # later drag edits; only real occupied frames from other labels do.
        end = b
        for f in range(a, b + 1):
            if self.store.is_occupied(f) and self.store.label_at(f) != self.label.name:
                end = f - 1
                break
        if end < a:
            return None
        return (a, end)

    # mouse
    def mouseMoveEvent(self, e):
        g = self.get_gutter()
        if not self._dragging and e.x() < g:
            # Only update cursor style inside gutter; skip preview/seek
            self._hover_frame = None
            self.hoverFrame.emit(-1)
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return
        f = self.x_to_frame(e.x())
        self._hover_frame = f
        self.hoverFrame.emit(f)
        self.setToolTip(f"Frame {f}")

        if not self._dragging:
            interval, where = self._hit_interval(e.x())
            if where in ("left", "right"):
                self.setCursor(Qt.SizeHorCursor)
            elif where == "center":
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            self.update()
            return

        # --- Dragging modes ---
        if self._mode == "create":
            # Start fixed at _create_anchor; drag end only; disable edge snap to avoid sticking at e+1
            start = (
                self._create_anchor
                if self._create_anchor is not None
                else (self._preview_interval[0] if self._preview_interval else f)
            )
            if self._snap_segments:
                start = self._snap_to_segment_start(start)
                end_cand = self._snap_to_segment_end(f)
            else:
                end_cand = self._snap_unlabeled(f) or f
            cand = (min(start, end_cand), max(start, end_cand))
            if not self._snap_segments:
                cand = self._snap_to_current(cand[0], cand[1])
            self._preview_interval = self._interval_clamped_free(*cand)
            if self._preview_interval is not None and not self._interval_in_edit_mask(
                self._preview_interval[0], self._preview_interval[1]
            ):
                self._preview_interval = None
            self.changed.emit()
            self.update()

        elif self._mode == "resize_left" and self._active_interval:
            old_s, old_e = self._active_interval
            if self._snap_segments:
                cand = min(f, old_e) if f >= old_s else f
                new_s = self._snap_to_segment_start(cand)
                if new_s > old_e:
                    new_s = old_e
                self._preview_interval = self._interval_clamped_free(
                    min(new_s, old_e), old_e
                )
            elif f >= old_s:
                # Move right => shorten; clamp to min(f, old_e)
                new_s = min(f, old_e)
                new_s, old_e = self._snap_to_current(new_s, old_e)
                self._preview_interval = self._interval_clamped_free(new_s, old_e)
            else:
                # Extend left => allow edge snap (search left for e+1); otherwise fallback to nearest unlabeled
                new_s = self._snap_edge_after_label_left(f)
                if new_s < 0:
                    new_s = self._snap_unlabeled(f)
                if new_s is not None:
                    new_s, old_e = self._snap_to_current(new_s, old_e)
                self._preview_interval = (
                    None
                    if new_s is None
                    else self._interval_clamped_free(min(new_s, old_e), old_e)
                )
            if self._preview_interval is not None and not self._interval_in_edit_mask(
                self._preview_interval[0], self._preview_interval[1]
            ):
                self._preview_interval = None
            self.update()

        elif self._mode == "resize_right" and self._active_interval:
            old_s, old_e = self._active_interval
            if self._snap_segments:
                cand = max(old_s, f) if f <= old_e else f
                new_e = self._snap_to_segment_end(cand)
                if new_e < old_s:
                    new_e = old_s
                new_s = old_s
                self._preview_interval = self._interval_clamped_free(new_s, new_e)
            elif f <= old_e:
                # Shorten
                new_e = max(old_s, f)
                new_s, new_e = self._snap_to_current(old_s, new_e)
                self._preview_interval = self._interval_clamped_free(new_s, new_e)
            else:
                # Extend right => disable edge snap; only use nearest unlabeled
                new_e = self._snap_unlabeled(f) or f
                new_s, new_e = self._snap_to_current(old_s, max(old_s, new_e))
                self._preview_interval = self._interval_clamped_free(new_s, new_e)
            if self._preview_interval is not None and not self._interval_in_edit_mask(
                self._preview_interval[0], self._preview_interval[1]
            ):
                self._preview_interval = None
            self.update()

        elif self._mode == "move" and self._active_interval:
            old_s, old_e = self._active_interval
            length = old_e - old_s
            target_s = f - self._grab_offset_frames
            cand_s = max(0, min(target_s, self.get_fc() - 1 - length))
            if self._snap_segments:
                snapped_s = self._snap_move_start(cand_s, length)
                if snapped_s is None:
                    self._preview_interval = None
                else:
                    cand_s = max(0, min(snapped_s, self.get_fc() - 1 - length))
                    cand_e = cand_s + length
                    self._preview_interval = self._interval_clamped_free(cand_s, cand_e)
            else:
                cand_e = cand_s + length
                self._preview_interval = self._interval_clamped_free(cand_s, cand_e)
            if self._preview_interval is not None and not self._interval_in_edit_mask(
                self._preview_interval[0], self._preview_interval[1]
            ):
                self._preview_interval = None
            self.update()

    def leaveEvent(self, e):
        self._hover_frame = None
        self.hoverFrame.emit(-1)
        self.update()
        return super().leaveEvent(e)

    def mousePressEvent(self, e):
        if (
            getattr(self, "on_extra_boundary", None)
            and getattr(self, "is_extra_mode", lambda: False)()
            and is_extra_label(self.label.name)
        ):
            try:
                self.on_extra_boundary(self.x_to_frame(e.x()))
            except Exception:
                pass
            return
        if e.button() != Qt.LeftButton:
            return
        g = self.get_gutter()
        if e.x() < g:
            return
        if (e.modifiers() & Qt.ControlModifier) and callable(self.split_handler):
            frame = self.x_to_frame(e.x())
            if not self._frame_in_edit_mask(frame):
                return
            try:
                handled = bool(self.split_handler(frame, self))
            except Exception:
                handled = False
            if handled:
                return
        f = self.x_to_frame(e.x())
        interval, where = self._hit_interval(e.x())
        if interval:
            if not self._interval_in_edit_mask(interval[0], interval[1]):
                return
            # start transaction
            if hasattr(self.store, "begin_txn"):
                self.store.begin_txn()

            self._dragging = True
            self._active_interval = interval
            if where == "left":
                self._mode = "resize_left"
                self.setCursor(Qt.SizeHorCursor)
            elif where == "right":
                self._mode = "resize_right"
                self.setCursor(Qt.SizeHorCursor)
            else:
                self._mode = "move"
                self.setCursor(Qt.ClosedHandCursor)
                self._grab_offset_frames = max(0, f - interval[0])
            self._preview_interval = interval
            self.update()
            return

        if not self._frame_in_edit_mask(f):
            return
        # Create in empty space: snap start to segment boundary when requested
        if self._snap_segments:
            s = self._snap_to_segment_start(f)
        else:
            s = self._snap_edge_after_label_left(f)
            if s < 0:
                s = self._snap_unlabeled(f)
        if s is None:
            return
        if not self._frame_in_edit_mask(s):
            return

        # start transaction
        if hasattr(self.store, "begin_txn"):
            self.store.begin_txn()

        self._dragging = True
        self._mode = "create"
        self._active_interval = None
        self._create_anchor = s  # lock start
        self._preview_interval = (s, s)
        self.setCursor(Qt.CrossCursor)
        self.update()

    def mouseReleaseEvent(self, e):
        if not self._dragging:
            return
        self.setCursor(Qt.ArrowCursor)

        if self._preview_interval is not None:
            s, e_ = self._preview_interval
            if not self._interval_in_edit_mask(s, e_):
                self._preview_interval = None
        if self._preview_interval is not None:
            s, e_ = self._preview_interval
            # 1) add frames needed for the new interval
            for f in range(s, e_ + 1):
                if (not self.store.is_occupied(f)) or (
                    self.store.label_at(f) == self.label.name
                ):
                    self.store.add(self.label.name, f)

            # 2) remove frames from the old interval that are outside the new interval
            if self._active_interval is not None:
                old_s, old_e = self._active_interval
                # left part trimmed away
                for f in range(old_s, min(s, old_e + 1)):
                    if self.store.label_at(f) == self.label.name:
                        self.store.remove_at(f)
                # right part trimmed away
                for f in range(max(e_ + 1, old_s), old_e + 1):
                    if self.store.label_at(f) == self.label.name:
                        self.store.remove_at(f)

        if hasattr(self.store, "end_txn"):
            self.store.end_txn()

        # reset drag state
        self._dragging = False
        self._mode = None
        self._active_interval = None
        self._preview_interval = None
        self._create_anchor = None
        self.changed.emit()
        self.update()

    def contextMenuEvent(self, e):
        interval, _ = self._hit_interval(e.x())
        if interval:
            s, e_ = interval
            if not self._interval_in_edit_mask(s, e_):
                return
            if callable(self.delete_handler):
                handled = bool(self.delete_handler(s, e_, self.label.name, self))
                if handled:
                    return
            if hasattr(self.store, "begin_txn"):
                self.store.begin_txn()
            bulk_remove = getattr(self.store, "remove_range", None)
            if callable(bulk_remove):
                bulk_remove(self.label.name, s, e_)
            else:
                for f in range(s, e_ + 1):
                    if self.store.label_at(f) == self.label.name:
                        self.store.remove_at(f)
            if hasattr(self.store, "end_txn"):
                self.store.end_txn()
            self.changed.emit()
            self.update()


@dataclass
class TranscriptSegment:
    start_frame: int
    end_frame: int
    text: str
    lang: str  # 'en' | 'de' | 'zh' | 'other'


@dataclass
class TranscriptTrack:
    name: str  # Row title, e.g. "ASR: EN"
    segments: list  # List[TranscriptSegment]


class SubtitleRow(BaseTimelineRow):
    """Read-only subtitle track: draw text blocks; double-click to jump; does not participate in exclusivity."""

    hoverFrame = pyqtSignal(int)  # mirrors TimelineRow signal
    changed = pyqtSignal()  # placeholder; never emitted

    def __init__(
        self,
        track: TranscriptTrack,
        get_frame_count,
        get_view_start,
        get_view_span,
        get_fps,
        get_gutter,
        parent=None,
    ):
        super().__init__(
            get_frame_count, get_view_start, get_view_span, get_fps, get_gutter, parent
        )
        self.track = track
        self.setMinimumHeight(ROW_HEIGHT)

    def sizeHint(self):
        return QSize(800, ROW_HEIGHT)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(245, 245, 245))
        # gutter title
        self._draw_gutter_title(p, self.track.name)

        start = self.get_vs()
        end = start + self.get_span()
        # render segments
        for seg in self.track.segments:
            s, e_ = seg.start_frame, seg.end_frame
            if e_ < start or s > end:
                continue
            s_vis = max(s, start)
            e_vis = min(e_, end)
            x1 = self.frame_to_x(s_vis)
            x2 = self.frame_to_x(e_vis + 1)
            rect = QRect(x1, 6, max(4, x2 - x1), self.height() - 12)
            # color by language (soft fill)
            base = {
                "en": QColor(66, 133, 244),
                "de": QColor(52, 168, 83),
                "zh": QColor(244, 180, 0),
            }.get(seg.lang, QColor(156, 39, 176))
            fill = QBrush(QColor(base.red(), base.green(), base.blue(), 60))
            border = QPen(QColor(base.red(), base.green(), base.blue(), 160), 1)
            p.setBrush(fill)
            p.setPen(border)
            p.drawRoundedRect(rect, 3, 3)
            # text
            p.setPen(QPen(QColor(40, 40, 40)))
            p.setFont(QFont("Arial", 8))
            metrics = p.fontMetrics()
            text = seg.text.replace("\n", " ").strip()
            # simple elide
            elided = metrics.elidedText(text, Qt.ElideRight, rect.width() - 8)
            p.drawText(
                rect.adjusted(4, 2, -4, -2), Qt.AlignLeft | Qt.AlignVCenter, elided
            )

        # current frame line
        self._draw_non_editable_overlay(p, start, end)
        self._draw_current_frame_marker(p, start, end)

    def mouseDoubleClickEvent(self, e):
        # double-click subtitle to preview/jump to start
        x = e.x()
        f = self.x_to_frame(x)
        # seek to the start of the clicked segment
        for seg in self.track.segments:
            if seg.start_frame <= f <= seg.end_frame:
                self.hoverFrame.emit(seg.start_frame)
                break


class CombinedTimelineRow(BaseTimelineRow):
    """
    Single-row, read-only view that paints the active label for every frame using the label's color.
    """

    hoverFrame = pyqtSignal(int)
    labelClicked = pyqtSignal(str, int)
    segmentSelected = pyqtSignal(int, int, object)
    changed = pyqtSignal()  # unused for now (view-only)

    def __init__(
        self,
        labels: List[LabelDef],
        row_sources: list,
        get_frame_count: Callable[[], int],
        get_view_start: Callable[[], int],
        get_view_span: Callable[[], int],
        get_fps: Callable[[], int],
        get_gutter: Callable[[], int],
        title: str = "Timeline",
        show_label_text: bool = True,
        extra_cuts: Optional[List[int]] = None,
        segment_cuts: Optional[List[int]] = None,
        editable: bool = False,
        show_extra_overlay: bool = True,
        parent=None,
    ):
        super().__init__(
            get_frame_count, get_view_start, get_view_span, get_fps, get_gutter, parent
        )
        self.labels = labels
        self.row_sources = row_sources or []
        self.title = title
        self.show_label_text = bool(show_label_text)
        self.editable = bool(editable)
        self.show_extra_overlay = bool(show_extra_overlay)

        self.setMouseTracking(True)
        self.setMinimumHeight(44)
        self.setMaximumHeight(44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.highlight_labels: set = set()
        self._current_hits: set = set()
        self._current_hit: bool = False
        self._flash_frame: Optional[int] = None
        self._dragging = False
        self._mode: Optional[str] = None  # "create"|"resize_left"|"resize_right"
        self._preview_interval: Optional[Tuple[int, int]] = None
        self._active_interval: Optional[Tuple[int, int]] = None
        self._active_label: Optional[str] = None
        self._create_anchor: Optional[int] = None
        self._selected_interval: Optional[Tuple[int, int]] = None
        self._selected_label: Optional[str] = None
        self.interaction_points: list = []
        self.active_interaction_id = None
        self.delete_handler = None
        self.split_handler = None
        self._selection_scope = "segment"
        self._snap_soft = False
        self._snap_radius = SNAP_RADIUS_FRAMES
        self._current_snap_radius = CURRENT_FRAME_SNAP_RADIUS_FRAMES
        self._frame_snap_radius = SNAP_RADIUS_FRAMES
        self._edge_snap_frames = EDGE_SNAP_FRAMES
        self._snap_segments: List[Tuple[int, int]] = []
        self._snap_starts: List[int] = []
        self._snap_ends: List[int] = []
        self._snap_end_set: set = set()

        # build color map once; rebuilt when the widget is reconstructed
        self._color_map = {lb.name: color_from_key(lb.color_name) for lb in labels}
        # share color across alias names for interaction/Extra
        for alias in EXTRA_ALIASES:
            for name, col in list(self._color_map.items()):
                if is_extra_label(name):
                    self._color_map[alias] = col
        # prefer non-interaction labels over interaction when multiple stores carry the same frame
        self._label_sources = []
        extras = []
        self._extra_store = None
        self._extra_cuts = list(extra_cuts or [])
        self._segment_cuts = list(segment_cuts or [])
        seen = set()
        for lb, st, _ in self.row_sources:
            if lb.name in seen:
                continue
            seen.add(lb.name)
            (extras if is_extra_label(lb.name) else self._label_sources).append(
                (lb.name, st)
            )
            if is_extra_label(lb.name):
                self._extra_store = st
        self._label_sources.extend(extras)
        self._label_to_store = {}
        for lb, st, _prefix in self.row_sources:
            if lb.name not in self._label_to_store:
                self._label_to_store[lb.name] = st

    def sizeHint(self):
        return QSize(800, 44)

    def set_interaction_points(self, points, active_id=None):
        self.interaction_points = list(points or [])
        self.active_interaction_id = active_id
        self.update()

    def set_delete_handler(self, handler):
        self.delete_handler = handler

    def set_split_handler(self, handler):
        self.split_handler = handler

    def set_current_snap_radius(self, radius: int) -> None:
        try:
            self._current_snap_radius = max(0, int(radius))
        except Exception:
            self._current_snap_radius = CURRENT_FRAME_SNAP_RADIUS_FRAMES

    def set_frame_snap_radius(self, radius: int) -> None:
        try:
            self._frame_snap_radius = max(0, int(radius))
        except Exception:
            self._frame_snap_radius = SNAP_RADIUS_FRAMES

    def set_edge_snap_frames(self, radius: int) -> None:
        try:
            self._edge_snap_frames = max(0, int(radius))
        except Exception:
            self._edge_snap_frames = EDGE_SNAP_FRAMES

    def set_segment_snap_radius(self, radius: int) -> None:
        try:
            self._snap_radius = max(0, int(radius))
        except Exception:
            self._snap_radius = SNAP_RADIUS_FRAMES

    def _row_drag_enabled(self) -> bool:
        tl = getattr(self, "_timeline_ref", None)
        if tl is None:
            return False
        if not callable(getattr(tl, "_combined_reorder_handler", None)):
            return False
        rows = getattr(tl, "_combined_rows", []) or []
        if len(rows) <= 1:
            return False
        return getattr(tl, "layout_mode", "") == "combined"

    def _finish_row_drag(self, global_pos):
        tl = getattr(self, "_timeline_ref", None)
        if tl is None:
            return
        try:
            tl._handle_combined_row_drop(self, global_pos)
        except Exception:
            pass

    def _color_for_label(self, name: Optional[str]) -> QColor:
        if name is None:
            return QColor(190, 190, 190)
        return self._color_map.get(name, QColor(120, 120, 120))

    def _label_at(self, frame: int) -> Optional[str]:
        for name, st in self._label_sources:
            try:
                if st.label_at(frame) == name:
                    return name
            except Exception:
                continue
        return None

    def _extra_frames(self) -> List[int]:
        frames = set()
        if self._extra_store is None:
            return []
        for alias in EXTRA_ALIASES:
            try:
                frames.update(self._extra_store.frames_of(alias))
            except Exception:
                continue
        return sorted(frames)

    def _extra_runs(self, start: int, end: int) -> List[Tuple[int, int]]:
        """Optional overlay strip for interaction boundaries."""
        if self._extra_store is None:
            return []
        frames = self._extra_frames()
        runs = AnnotationStore.frames_to_runs(frames) if frames else []
        return [
            (max(s, start), min(e, end)) for s, e in runs if not (e < start or s > end)
        ]

    def _extra_boundaries(self, start: int, end: int) -> List[int]:
        bounds = set()
        for s, e in self._extra_runs(start, end):
            bounds.add(s)
            bounds.add(e + 1)
        for c in self._extra_cuts:
            if start <= c <= end:
                bounds.add(c)
        return sorted(bounds)

    def _status_color(self, status: str) -> QColor:
        mapping = {
            "PENDING": QColor("#f79009"),
            "ACTIVE": QColor("#175cd3"),
            "RESOLVED": QColor("#12b76a"),
        }
        return mapping.get(status, QColor("#667085"))

    def _label_runs(self, start: int, end: int) -> List[Tuple[int, int, Optional[str]]]:
        fc = max(1, self.get_fc()) if callable(self.get_fc) else None
        if fc is not None:
            start = max(0, min(start, fc - 1))
            end = max(0, min(end, fc - 1))
        runs = []
        if end < start:
            return runs
        cut_set = {
            int(c)
            for c in (self._segment_cuts or [])
            if c is not None and start < int(c) <= end
        }
        cur = self._label_at(start)
        s = start
        for f in range(start + 1, end + 1):
            lb = self._label_at(f)
            if f in cut_set or lb != cur:
                runs.append((s, f - 1, cur))
                s, cur = f, lb
        runs.append((s, end, cur))
        return runs

    def _segment_at(self, frame: int) -> Tuple[int, int, Optional[str]]:
        fc = max(1, self.get_fc())
        f = max(0, min(frame, fc - 1))
        cut_set = {int(c) for c in (self._segment_cuts or []) if c is not None}
        lb = self._label_at(f)
        s = f
        while s > 0 and self._label_at(s - 1) == lb and s not in cut_set:
            s -= 1
        e = f
        while e < fc - 1 and self._label_at(e + 1) == lb and (e + 1) not in cut_set:
            e += 1
        return s, e, lb

    def _store_for_label(self, name: Optional[str]):
        if not name:
            return None
        return self._label_to_store.get(name)

    def _is_occupied(self, frame: int) -> bool:
        return self._label_at(frame) is not None

    def _snap_unlabeled(self, target: int) -> Optional[int]:
        fc = max(1, self.get_fc())
        target = max(0, min(target, fc - 1))
        for d in range(0, self._frame_snap_radius + 1):
            f = target + d
            if f < fc and not self._is_occupied(f):
                return f
        for d in range(1, self._frame_snap_radius + 1):
            f = target - d
            if f >= 0 and not self._is_occupied(f):
                return f
        return None

    def _snap_to_current(self, start: int, end: int) -> Tuple[int, int]:
        cf = self.current_frame
        if cf is None:
            return start, end
        if abs(start - cf) <= self._current_snap_radius:
            start = cf
        if abs(end - cf) <= self._current_snap_radius:
            end = cf
        return start, end

    def set_snap_segments(self, segments: List[Tuple[int, int]]) -> None:
        cleaned = []
        for seg in segments or []:
            try:
                s = int(seg[0])
                e = int(seg[1])
            except Exception:
                continue
            if e < s:
                s, e = e, s
            cleaned.append((s, e))
        cleaned.sort(key=lambda x: x[0])
        self._snap_segments = cleaned
        self._snap_starts = [s for s, _ in cleaned]
        self._snap_ends = sorted({e for _s, e in cleaned})
        self._snap_end_set = set(self._snap_ends)

    def _segment_bounds_for_frame(self, frame: int) -> Optional[Tuple[int, int]]:
        if not self._snap_segments:
            return None
        idx = bisect.bisect_right(self._snap_starts, int(frame)) - 1
        if idx < 0 or idx >= len(self._snap_segments):
            return None
        s, e = self._snap_segments[idx]
        if s <= frame <= e:
            return s, e
        return None

    def _nearest_in_list(self, values: List[int], frame: int) -> Optional[int]:
        if not values:
            return None
        frame = int(frame)
        idx = bisect.bisect_left(values, frame)
        candidates = []
        if idx < len(values):
            candidates.append(values[idx])
        if idx > 0:
            candidates.append(values[idx - 1])
        if not candidates:
            return None
        return min(candidates, key=lambda v: abs(v - frame))

    def _snap_to_segment_start(self, frame: int) -> int:
        if not self._snap_starts:
            return frame
        seg = self._segment_bounds_for_frame(frame)
        if seg:
            if not self._snap_soft:
                return seg[0]
            if abs(frame - seg[0]) <= self._snap_radius:
                return seg[0]
        nearest = self._nearest_in_list(self._snap_starts, frame)
        if nearest is None:
            return frame
        if self._snap_soft and abs(frame - nearest) > self._snap_radius:
            return frame
        return nearest

    def _snap_to_segment_end(self, frame: int) -> int:
        if not self._snap_ends:
            return frame
        seg = self._segment_bounds_for_frame(frame)
        if seg:
            if not self._snap_soft:
                return seg[1]
            if abs(frame - seg[1]) <= self._snap_radius:
                return seg[1]
        nearest = self._nearest_in_list(self._snap_ends, frame)
        if nearest is None:
            return frame
        if self._snap_soft and abs(frame - nearest) > self._snap_radius:
            return frame
        return nearest

    def _snap_move_start(self, cand_start: int, length: int) -> Optional[int]:
        if not self._snap_starts:
            return cand_start
        if length < 0:
            return None
        best = None
        best_dist = None
        for s in self._snap_starts:
            if (s + length) not in self._snap_end_set:
                continue
            dist = abs(s - cand_start)
            if best is None or dist < best_dist:
                best = s
                best_dist = dist
        if best is None:
            return None
        if self._snap_soft and best_dist is not None and best_dist > self._snap_radius:
            return cand_start
        return best

    def _snap_edge_after_label_left(self, target: int) -> int:
        fc = max(1, self.get_fc())
        t = max(0, min(target, fc - 1))
        for d in range(0, self._edge_snap_frames + 1):
            cand = t - d
            if (
                cand >= 1
                and (not self._is_occupied(cand))
                and self._is_occupied(cand - 1)
            ):
                return cand
        return -1

    def _interval_clamped_free(
        self, a: int, b: int, allow_label: Optional[str]
    ) -> Optional[Tuple[int, int]]:
        fc = max(1, self.get_fc())
        a = max(0, min(a, fc - 1))
        b = max(0, min(b, fc - 1))
        if a > b:
            a, b = b, a
        # Keep trim cuts as visual/selection boundaries only.
        # If a labeled segment is being edited, allow continuous dragging.
        if self.editable and allow_label:
            return (a, b)
        end = b
        for f in range(a, b + 1):
            cur = self._label_at(f)
            if cur is None:
                continue
            if allow_label and cur == allow_label:
                continue
            end = f - 1
            break
        if end < a:
            return None
        return (a, end)

    def set_editable(self, on: bool):
        self.editable = bool(on)
        self.setCursor(Qt.ArrowCursor)

    def apply_label_to_selection(self, new_label: str) -> bool:
        if not self._selected_interval:
            return False
        if not self._interval_in_edit_mask(
            self._selected_interval[0], self._selected_interval[1]
        ):
            return False
        if not new_label:
            return False
        if self._selected_label == new_label:
            return False
        new_store = self._store_for_label(new_label)
        if new_store is None:
            return False
        s, e = self._selected_interval
        touched_stores = []
        seen_ids = set()
        for f in range(s, e + 1):
            cur = self._label_at(f)
            if cur:
                st = self._store_for_label(cur)
                if st and id(st) not in seen_ids:
                    seen_ids.add(id(st))
                    touched_stores.append(st)
        if id(new_store) not in seen_ids:
            seen_ids.add(id(new_store))
            touched_stores.append(new_store)
        for st in touched_stores:
            try:
                st.begin_txn()
            except Exception:
                pass
        for f in range(s, e + 1):
            cur = self._label_at(f)
            if cur == new_label:
                continue
            if cur:
                st = self._store_for_label(cur)
                if st:
                    st.remove_at(f)
            new_store.add(new_label, f)
        for st in touched_stores:
            try:
                st.end_txn()
            except Exception:
                pass
        self._selected_label = new_label
        self.changed.emit()
        self.update()
        return True

    def _hit_edge(self, x: int) -> Optional[Tuple[Tuple[int, int], Optional[str], str]]:
        def check(interval: Tuple[int, int], label: Optional[str]):
            x1 = self.frame_to_x(interval[0])
            x2 = self.frame_to_x(interval[1] + 1)
            if abs(x - x1) <= EDGE_TOLERANCE_PX:
                return interval, label, "left"
            if abs(x - x2) <= EDGE_TOLERANCE_PX:
                return interval, label, "right"
            return None

        if self._selected_interval is not None:
            hit = check(self._selected_interval, self._selected_label)
            if hit:
                return hit
        f = self.x_to_frame(x)
        seg = self._segment_at(f)
        if seg and seg[2] is not None:
            return check((seg[0], seg[1]), seg[2])
        return None

    def set_current_frame(self, f: Optional[int]):
        self.current_frame = f
        self._update_current_hit()
        self.update()

    def set_current_hits(self, names):
        self._current_hits = set(names or [])
        self._update_current_hit()
        self.update()

    def set_highlight_labels(self, names):
        self.highlight_labels = set(names or [])
        self.update()

    def set_boundary_flash(self, frame: Optional[int]):
        self._flash_frame = None if frame is None else int(frame)
        if frame is not None:
            ref = weakref.ref(self)
            QTimer.singleShot(
                800, lambda: _safe_qt_call(ref, "set_boundary_flash", None)
            )
        self.update()

    def flash_labels(self, names):
        base = set(self.highlight_labels)
        flash = set(names or [])
        self.set_highlight_labels(base | flash)
        ref = weakref.ref(self)
        QTimer.singleShot(220, lambda: _safe_qt_call(ref, "set_highlight_labels", base))

    def _update_current_hit(self):
        if self.current_frame is None or not self._current_hits:
            self._current_hit = False
            return
        lb = self._label_at(self.current_frame)
        self._current_hit = lb in self._current_hits

    def paintEvent(self, e):
        p = QPainter(self)
        meta = getattr(self, "_group_meta", None)
        is_phase_row = bool(isinstance(meta, dict) and meta.get("row_type") == "phase")
        bg = QColor(240, 240, 240)
        if is_phase_row:
            bg = QColor(232, 238, 248)
        if self._current_hit:
            bg = QColor(245, 250, 255) if not is_phase_row else QColor(225, 240, 252)
        p.fillRect(self.rect(), bg)

        start = self.get_vs()
        span = self.get_span()
        end = start + span
        fps = max(1, self.get_fps())

        self._draw_time_grid(p, start, end, fps)
        self._draw_gutter_title(p, self.title)

        # draw label runs
        runs = self._label_runs(start, end)
        for s, e_, lb in runs:
            if lb is None:
                continue
            s_vis = max(s, start)
            e_vis = min(e_, end)
            x1 = self.frame_to_x(s_vis)
            x2 = self.frame_to_x(e_vis + 1)
            rect = QRect(x1, 6, max(4, x2 - x1), self.height() - 12)
            base_col = self._color_for_label(lb)
            fill_col = QColor(base_col)
            if is_phase_row:
                fill_col = fill_col.lighter(115)
                fill_col.setAlpha(170)
            if lb in self.highlight_labels:
                fill_col = fill_col.lighter(130)
            p.setBrush(QBrush(fill_col.lighter(100)))
            border_col = base_col.darker(170) if is_phase_row else base_col.darker(140)
            p.setPen(QPen(border_col, 2 if lb in self.highlight_labels else 1))
            p.drawRoundedRect(rect, 4, 4)
            if lb and self.show_label_text:
                p.setPen(QPen(QColor(40, 40, 40)))
                p.setFont(QFont("Arial", 8))
                text = str(lb)
                elided = p.fontMetrics().elidedText(
                    text, Qt.ElideRight, rect.width() - 8
                )
                p.drawText(
                    rect.adjusted(4, 2, -4, -2), Qt.AlignLeft | Qt.AlignVCenter, elided
                )

        # selection highlight
        if self._selected_interval is not None:
            s, e_ = self._selected_interval
            s_vis = max(s, start)
            e_vis = min(e_, end)
            if s_vis <= e_vis:
                x1 = self.frame_to_x(s_vis)
                x2 = self.frame_to_x(e_vis + 1)
                rect = QRect(x1, 4, max(4, x2 - x1), self.height() - 8)
                sel_col = self._color_for_label(self._selected_label)
                pen = QPen(sel_col.darker(150), 3)
                if self._selected_label is None:
                    pen.setStyle(Qt.DashLine)
                p.setBrush(Qt.NoBrush)
                p.setPen(pen)
                p.drawRoundedRect(rect, 6, 6)

        # preview interval while dragging
        if self._preview_interval is not None:
            s, e_ = self._preview_interval
            s_vis = max(s, start)
            e_vis = min(e_, end)
            if s_vis <= e_vis:
                x1 = self.frame_to_x(s_vis)
                x2 = self.frame_to_x(e_vis + 1)
                rect = QRect(x1, 6, max(4, x2 - x1), self.height() - 12)
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(70, 90, 120), 2, Qt.DashLine))
                p.drawRoundedRect(rect, 4, 4)

        # manual segment cuts overlay (split markers)
        if self._segment_cuts:
            cuts = [c for c in self._segment_cuts if start <= int(c) <= end]
            if cuts:
                p.setPen(QPen(QColor(80, 80, 80, 140), 1, Qt.DotLine))
                for c in cuts:
                    x = self.frame_to_x(int(c))
                    p.drawLine(x, 0, x, self.height())

        # assisted interaction overlays (uncertain boundaries/labels)
        if self.interaction_points:
            active_id = self.active_interaction_id
            # label-level uncertainty bars near the top
            for pt in self.interaction_points:
                if pt.get("type") != "label":
                    continue
                s = int(pt.get("start", 0))
                e_ = int(pt.get("end", s))
                if e_ < start or s > end:
                    continue
                s_vis = max(s, start)
                e_vis = min(e_, end)
                x1 = self.frame_to_x(s_vis)
                x2 = self.frame_to_x(e_vis + 1)
                bar_h = 6
                rect = QRect(x1, 2, max(2, x2 - x1), bar_h)
                col = self._status_color(pt.get("status"))
                fill = QColor(col.red(), col.green(), col.blue(), 70)
                p.fillRect(rect, fill)
                pen = QPen(col, 2 if pt.get("id") == active_id else 1, Qt.DashLine)
                p.setPen(pen)
                p.drawRect(rect)

            # boundary markers with status colors
            for pt in self.interaction_points:
                if pt.get("type") != "boundary":
                    continue
                frame = int(pt.get("frame", -1))
                if frame < start or frame > end:
                    continue
                x = self.frame_to_x(frame)
                col = self._status_color(pt.get("status"))
                line_pen = QPen(
                    QColor(col.red(), col.green(), col.blue(), 140), 1, Qt.DashLine
                )
                tick_pen = QPen(col, 3 if pt.get("id") == active_id else 2)
                p.setPen(line_pen)
                p.drawLine(x, 0, x, self.height())
                p.setPen(tick_pen)
                p.drawLine(x, 0, x, 14)

        # interaction overlay strip (bottom bar) to visualize boundaries even in combined mode
        extra_runs = self._extra_runs(start, end) if self.show_extra_overlay else []
        if extra_runs:
            bar_h = 14
            y0 = self.height() - bar_h - 2
            extra_color = (
                self._color_map.get(EXTRA_LABEL_NAME)
                or self._color_map.get("Extra")
                or QColor(100, 100, 100)
            )
            for s, e_ in extra_runs:
                s_vis = max(s, start)
                e_vis = min(e_, end)
                x1 = self.frame_to_x(s_vis)
                x2 = self.frame_to_x(e_vis + 1)
                rect = QRect(x1, y0, max(2, x2 - x1), bar_h)
                p.fillRect(rect, extra_color.lighter(110))
                p.setPen(QPen(extra_color.darker(130)))
                p.drawRect(rect)
            # boundary ticks on top of the extra bar
            ticks = self._extra_boundaries(start, end)
            if ticks:
                boundary_col = QColor(255, 0, 180)  # fixed magenta for clarity
                tick_pen = QPen(boundary_col, 3)
                line_pen = QPen(
                    QColor(
                        boundary_col.red(),
                        boundary_col.green(),
                        boundary_col.blue(),
                        140,
                    ),
                    1,
                )
                for b in ticks:
                    x = self.frame_to_x(b)
                    p.setPen(tick_pen)
                    p.drawLine(x, y0, x, y0 + bar_h)
                    # extend faint line through the timeline for visibility
                    p.setPen(line_pen)
                    p.drawLine(x, 0, x, self.height())

        self._draw_non_editable_overlay(p, start, end)
        self._draw_current_frame_marker(p, start, end)
        if self._hover_frame is not None:
            label = self._label_at(self._hover_frame) or "Unlabeled"
            txt = f"{label} | F {self._hover_frame} | {self._hover_frame / fps:.2f}s"
        else:
            txt = ""
        self._draw_hover_marker(p, start, end, fps, txt)

    def mouseMoveEvent(self, e):
        if self._row_dragging:
            if self._row_drag_start is None:
                self._row_drag_start = e.pos()
            if not self._row_drag_active:
                dist = (e.pos() - self._row_drag_start).manhattanLength()
                if dist >= QApplication.startDragDistance():
                    self._row_drag_active = True
                    self.setCursor(Qt.ClosedHandCursor)
            return
        g = self.get_gutter()
        if e.x() < g:
            self._hover_frame = None
            self.hoverFrame.emit(-1)
            self.setCursor(Qt.ArrowCursor)
            self.update()
            return
        f = self.x_to_frame(e.x())
        self._hover_frame = f
        self.setCursor(Qt.ArrowCursor)
        self.hoverFrame.emit(f)
        self.update()
        if not self.editable:
            return

        if not self._dragging:
            hit = self._hit_edge(e.x())
            if hit:
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            return

        if self._mode == "create":
            start = (
                self._create_anchor
                if self._create_anchor is not None
                else (self._preview_interval[0] if self._preview_interval else f)
            )
            if self._snap_segments:
                start = self._snap_to_segment_start(start)
                end_cand = self._snap_to_segment_end(f)
            else:
                end_cand = self._snap_unlabeled(f) or f
            cand = (min(start, end_cand), max(start, end_cand))
            if not self._snap_segments:
                cand = self._snap_to_current(cand[0], cand[1])
            self._preview_interval = self._interval_clamped_free(cand[0], cand[1], None)
            if self._preview_interval is not None and not self._interval_in_edit_mask(
                self._preview_interval[0], self._preview_interval[1]
            ):
                self._preview_interval = None
            self.update()
            return

        if self._mode == "resize_left" and self._active_interval:
            old_s, old_e = self._active_interval
            if self._snap_segments:
                cand = min(f, old_e) if f >= old_s else f
                new_s = self._snap_to_segment_start(cand)
                if new_s > old_e:
                    new_s = old_e
                self._preview_interval = self._interval_clamped_free(
                    min(new_s, old_e), old_e, self._active_label
                )
            elif f >= old_s:
                new_s = min(f, old_e)
                if not (self.editable and self._active_label):
                    new_s, old_e = self._snap_to_current(new_s, old_e)
                self._preview_interval = self._interval_clamped_free(
                    new_s, old_e, self._active_label
                )
            else:
                new_s = self._snap_edge_after_label_left(f)
                if new_s < 0:
                    new_s = self._snap_unlabeled(f)
                if new_s is None and self.editable and self._active_label:
                    new_s = f
                if new_s is not None:
                    if not (self.editable and self._active_label):
                        new_s, old_e = self._snap_to_current(new_s, old_e)
                self._preview_interval = (
                    None
                    if new_s is None
                    else self._interval_clamped_free(
                        min(new_s, old_e), old_e, self._active_label
                    )
                )
            if self._preview_interval is not None and not self._interval_in_edit_mask(
                self._preview_interval[0], self._preview_interval[1]
            ):
                self._preview_interval = None
            self.update()
            return

        if self._mode == "resize_right" and self._active_interval:
            old_s, old_e = self._active_interval
            if self._snap_segments:
                cand = max(old_s, f) if f <= old_e else f
                new_e = self._snap_to_segment_end(cand)
                if new_e < old_s:
                    new_e = old_s
                self._preview_interval = self._interval_clamped_free(
                    old_s, new_e, self._active_label
                )
            elif f <= old_e:
                new_e = max(old_s, f)
                if self.editable and self._active_label:
                    new_s = old_s
                else:
                    new_s, new_e = self._snap_to_current(old_s, new_e)
                self._preview_interval = self._interval_clamped_free(
                    new_s, new_e, self._active_label
                )
            else:
                # While resizing a labeled segment, allow dragging through
                # neighboring labels to move the boundary continuously.
                if self.editable and self._active_label:
                    new_e = f
                else:
                    new_e = self._snap_unlabeled(f) or f
                cand_e = max(old_s, new_e)
                if self.editable and self._active_label:
                    new_s, new_e = old_s, cand_e
                else:
                    new_s, new_e = self._snap_to_current(old_s, cand_e)
                self._preview_interval = self._interval_clamped_free(
                    new_s, new_e, self._active_label
                )
            if self._preview_interval is not None and not self._interval_in_edit_mask(
                self._preview_interval[0], self._preview_interval[1]
            ):
                self._preview_interval = None
            self.update()

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        g = self.get_gutter()
        if self._row_drag_enabled() and e.x() < g:
            self._row_dragging = True
            self._row_drag_active = False
            self._row_drag_start = e.pos()
            self.setCursor(Qt.OpenHandCursor)
            return
        if e.x() < g:
            return
        if (e.modifiers() & Qt.ControlModifier) and callable(self.split_handler):
            frame = self.x_to_frame(e.x())
            if not self._frame_in_edit_mask(frame):
                return
            try:
                handled = bool(self.split_handler(frame, self))
            except Exception:
                handled = False
            if handled:
                return
        if not self.editable:
            f = self.x_to_frame(e.x())
            lb = self._label_at(f)
            if lb:
                self.labelClicked.emit(lb, f)
            return
        hit = self._hit_edge(e.x())
        if hit:
            interval, label, where = hit
            if not self._interval_in_edit_mask(interval[0], interval[1]):
                return
            self._dragging = True
            self._active_interval = interval
            self._active_label = label
            self._mode = "resize_left" if where == "left" else "resize_right"
            self._preview_interval = interval
            st = self._store_for_label(label)
            if st is not None:
                try:
                    st.begin_txn()
                except Exception:
                    pass
            self.setCursor(Qt.SizeHorCursor)
            self.update()
            return

        f = self.x_to_frame(e.x())
        if not self._frame_in_edit_mask(f):
            return
        if self._label_at(f) is None:
            if self._snap_segments:
                s = self._snap_to_segment_start(f)
            else:
                s = self._snap_edge_after_label_left(f)
                if s < 0:
                    s = self._snap_unlabeled(f)
            if s is None:
                return
            if not self._frame_in_edit_mask(s):
                return
            self._dragging = True
            self._mode = "create"
            self._active_interval = None
            self._active_label = None
            self._create_anchor = s
            self._preview_interval = (s, s)
            self.setCursor(Qt.CrossCursor)
            self.update()

    def mouseDoubleClickEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        g = self.get_gutter()
        if e.x() < g:
            self._row_dragging = False
            self._row_drag_active = False
            self._row_drag_start = None
            self.setCursor(Qt.ArrowCursor)
            fc = max(1, self.get_fc())
            self._selected_interval = (0, fc - 1)
            self._selected_label = None
            self._selection_scope = "all"
            self.segmentSelected.emit(0, fc - 1, None)
            self.update()
            return
        f = self.x_to_frame(e.x())
        s, e_, lb = self._segment_at(f)
        self._selected_interval = (s, e_)
        self._selected_label = lb
        self._selection_scope = "segment"
        self.segmentSelected.emit(s, e_, lb)
        if lb:
            self.labelClicked.emit(lb, f)
        self.update()

    def leaveEvent(self, e):
        self._hover_frame = None
        self.hoverFrame.emit(-1)
        self.update()
        return super().leaveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._row_dragging:
            if self._row_drag_active:
                self._finish_row_drag(e.globalPos())
            self._row_dragging = False
            self._row_drag_active = False
            self._row_drag_start = None
            self.setCursor(Qt.ArrowCursor)
            return
        if not self._dragging:
            return
        self.setCursor(Qt.ArrowCursor)
        if self._preview_interval is not None:
            s, e_ = self._preview_interval
            if not self._interval_in_edit_mask(s, e_):
                self._preview_interval = None
        if self._preview_interval is not None:
            s, e_ = self._preview_interval
            if self._active_label is None:
                self._selected_interval = (s, e_)
                self._selected_label = None
                self.segmentSelected.emit(s, e_, None)
            else:
                st = self._store_for_label(self._active_label)
                if st is not None:
                    old_s = old_e = None
                    left_fill_label = None
                    right_fill_label = None
                    no_gap_fill = False
                    for f in range(s, e_ + 1):
                        cur = self._label_at(f)
                        if cur is None:
                            st.add(self._active_label, f)
                        elif cur != self._active_label:
                            st.remove_at(f)
                            st.add(self._active_label, f)
                    if self._active_interval is not None:
                        old_s, old_e = self._active_interval
                        meta = getattr(self, "_group_meta", None)
                        if isinstance(meta, dict) and bool(meta.get("psr_no_gap_fill")):
                            no_gap_fill = True
                            try:
                                old_s_i = int(old_s)
                                old_e_i = int(old_e)
                            except Exception:
                                old_s_i = old_e_i = None
                            if old_s_i is not None and old_s_i > 0:
                                left_fill_label = self._label_at(old_s_i - 1)
                            if old_e_i is not None:
                                fc = max(1, self.get_fc())
                                if old_e_i < fc - 1:
                                    right_fill_label = self._label_at(old_e_i + 1)
                            default_fill = meta.get("psr_default_label")
                            if left_fill_label is None and default_fill:
                                left_fill_label = default_fill
                            if right_fill_label is None and default_fill:
                                right_fill_label = default_fill

                        # Left trimmed span (start moved right): optionally fill to
                        # avoid gaps in PSR no-gap mode.
                        for f in range(old_s, min(s, old_e + 1)):
                            if self._label_at(f) != self._active_label:
                                continue
                            fill = (
                                left_fill_label
                                if no_gap_fill and self._mode == "resize_left"
                                else None
                            )
                            if fill and fill != self._active_label:
                                st.remove_at(f)
                                st.add(fill, f)
                            else:
                                st.remove_at(f)

                        # Right trimmed span (end moved left): optionally fill to
                        # shift the adjacent boundary in PSR no-gap mode.
                        for f in range(max(e_ + 1, old_s), old_e + 1):
                            if self._label_at(f) != self._active_label:
                                continue
                            fill = (
                                right_fill_label
                                if no_gap_fill and self._mode == "resize_right"
                                else None
                            )
                            if fill and fill != self._active_label:
                                st.remove_at(f)
                                st.add(fill, f)
                            else:
                                st.remove_at(f)
                    try:
                        st.end_txn()
                    except Exception:
                        pass
                self._selected_interval = (s, e_)
                self._selected_label = self._active_label
                self.segmentSelected.emit(s, e_, self._active_label)
                self.changed.emit()
        else:
            if self._active_label:
                st = self._store_for_label(self._active_label)
                if st is not None:
                    try:
                        st.end_txn()
                    except Exception:
                        pass

        self._dragging = False
        self._mode = None
        self._active_interval = None
        self._preview_interval = None
        self._create_anchor = None
        self._active_label = None
        self.update()

    def contextMenuEvent(self, e):
        g = self.get_gutter()
        if e.x() < g:
            return
        f = self.x_to_frame(e.x())
        s, e_, lb = self._segment_at(f)
        if not lb:
            return
        if not self._interval_in_edit_mask(s, e_):
            return
        if callable(self.delete_handler):
            handled = bool(self.delete_handler(s, e_, lb, self))
            if handled:
                self._selected_interval = None
                self._selected_label = None
                self.update()
                return
        if not self.editable:
            return
        st = self._store_for_label(lb)
        if st is None:
            return
        try:
            st.begin_txn()
        except Exception:
            pass
        bulk_remove = getattr(st, "remove_range", None)
        if callable(bulk_remove):
            bulk_remove(lb, s, e_)
        else:
            for fr in range(s, e_ + 1):
                if self._label_at(fr) == lb:
                    st.remove_at(fr)
        try:
            st.end_txn()
        except Exception:
            pass
        self._selected_interval = None
        self._selected_label = None
        self.changed.emit()
        self.update()


class TimelineArea(QWidget):
    hoverFrame = pyqtSignal(int)
    changed = pyqtSignal()
    viewPanned = pyqtSignal()  # emitted when user moves view/zoom
    labelClicked = pyqtSignal(str, int)
    segmentSelected = pyqtSignal(int, int, object)
    gapPrevRequested = pyqtSignal()
    gapNextRequested = pyqtSignal()

    def __init__(
        self,
        labels: List[LabelDef],
        store: AnnotationStore,
        get_frame_count: Callable[[], int],
        get_fps: Callable[[], int],
        on_extra_boundary: Optional[Callable[[int], None]] = None,
        is_extra_mode: Optional[Callable[[], bool]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.labels = labels
        self.store = store
        self.get_fc = get_frame_count
        self.get_fps = get_fps
        self._extra_boundary_cb = on_extra_boundary
        self._is_extra_mode = is_extra_mode or (lambda: False)

        self.view_start = 0
        self.view_span = DEFAULT_VIEW_SPAN
        self._gutter_px = 80
        self._row_sources = None  # List[Tuple[LabelDef, AnnotationStore, str]]
        self.highlight_labels = set()  # label names to highlight
        self.current_frame: Optional[int] = None
        self._current_hits = set()
        self._block_view_signal = False
        self.layout_mode = "combined"  # "combined" | "per_label"
        self._extra_cuts: List[int] = []
        self._segment_cuts: List[int] = []
        self._snap_segments: List[Tuple[int, int]] = []
        self._current_frame_snap_radius = CURRENT_FRAME_SNAP_RADIUS_FRAMES
        self._frame_snap_radius = SNAP_RADIUS_FRAMES
        self._edge_snap_frames = EDGE_SNAP_FRAMES
        self._segment_snap_radius = SNAP_RADIUS_FRAMES
        self._center_single_row = False
        self._combined_show_text = True
        self._combined_editable = False
        self._combined_groups = None
        self._tail_combined_groups = None
        self._combined_rows = []
        self._active_combined_row = None
        self._row_delete_handler = None
        self._row_split_handler = None
        self._row_segment_cuts_provider = None
        self._row_edit_mask_provider = None
        self._combined_delete_handler = None
        self._combined_split_handler = None
        self._combined_reorder_handler = None
        self._interaction_points = []
        self._active_interaction_id = None

        root = QVBoxLayout(self)

        # scrollable rows
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.container = QWidget()
        self.container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vbox = QVBoxLayout(self.container)
        self.vbox.setContentsMargins(0, 0, 0, 0)
        self.vbox.setSpacing(4)
        self._default_vbox_spacing = 4
        self.scroll.setWidget(self.container)
        # set a modest minimum height so single-row mode stays visible
        self.scroll.setMinimumHeight(64)
        root.addWidget(self.scroll, 1)

        # view controls: start + span
        row = QHBoxLayout()
        self.chk_layout = QCheckBox("Single timeline", self)
        self.chk_layout.setChecked(True)
        self.chk_layout.setToolTip(
            "Show all labels on one track (toggle off for per-label editing)"
        )
        self.chk_layout.toggled.connect(self._on_layout_mode_toggled)
        row.addWidget(self.chk_layout, 0)
        self.chk_action_lock = QCheckBox("Lock to segment", self)
        self.chk_action_lock.setToolTip("Snap state boundaries to segments")
        self.chk_action_lock.setVisible(False)
        row.addWidget(self.chk_action_lock, 0)
        row.addSpacing(8)
        row.addWidget(QLabel("View start:"))
        self.slider_view = QSlider(Qt.Horizontal, self)
        self.slider_view.valueChanged.connect(self._on_view_start_changed)
        row.addWidget(self.slider_view, 2)

        row.addSpacing(8)
        row.addWidget(QLabel("View span:"))
        self.slider_span = QSlider(Qt.Horizontal, self)
        self.slider_span.valueChanged.connect(self._on_view_span_changed)
        row.addWidget(self.slider_span, 3)

        row.addSpacing(8)
        row.addStretch(1)
        self.lbl_gap = QLabel("Gaps: n/a", self)
        self.lbl_gap.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_gap.setStyleSheet("color: #667085;")
        row.addWidget(self.lbl_gap, 0)
        self.btn_gap_prev = QToolButton(self)
        self.btn_gap_prev.setText("<")
        self.btn_gap_prev.setToolTip("Previous gap")
        self.btn_gap_prev.clicked.connect(self.gapPrevRequested.emit)
        row.addWidget(self.btn_gap_prev, 0)
        self.btn_gap_next = QToolButton(self)
        self.btn_gap_next.setText(">")
        self.btn_gap_next.setToolTip("Next gap")
        self.btn_gap_next.clicked.connect(self.gapNextRequested.emit)
        row.addWidget(self.btn_gap_next, 0)

        root.addLayout(row)

        self.rows: List[QWidget] = []
        self.rebuild_rows()

        self.subtitle_tracks: list = []  # List[TranscriptTrack]

    def get_view_start(self):
        return self.view_start

    def get_view_span(self):
        return self.view_span

    def _compute_gutter_px(self):
        # Compute max label text width across all labels, then add padding
        fm = QFontMetrics(QFont("Arial", 9))
        max_w = 0
        for lb in self.labels:
            try:
                w = fm.horizontalAdvance(lb.name)
            except AttributeError:
                w = fm.width(lb.name)
            if w > max_w:
                max_w = w
        self._gutter_px = max(80, max_w + 16)  # min 80px, or longest text + padding

    def _compute_gutter_px_from_sources(self):
        from PyQt5.QtGui import QFont, QFontMetrics

        fm = QFontMetrics(QFont("Arial", 9))
        max_w = 0
        if getattr(self, "layout_mode", "") == "combined":
            if self._combined_groups:
                titles = []
                for g in self._combined_groups:
                    if isinstance(g, (list, tuple)) and len(g) >= 1:
                        title = g[0] if g[0] else "Timeline"
                    else:
                        title = "Timeline"
                    titles.append(str(title))
            else:
                titles = ["Timeline"]
        else:
            if self._row_sources:
                titles = [
                    f"{prefix}{lb.name}" for (lb, _store, prefix) in self._row_sources
                ]
            else:
                titles = [lb.name for lb in self.labels]
            if self._tail_combined_groups:
                for g in self._tail_combined_groups:
                    if isinstance(g, (list, tuple)) and len(g) >= 1:
                        title = g[0] if g[0] else "Timeline"
                    else:
                        title = "Timeline"
                    titles.append(str(title))
        for t in titles:
            try:
                w = fm.horizontalAdvance(t)
            except AttributeError:
                w = fm.width(t)
            max_w = max(max_w, w)
        self._gutter_px = max(80, max_w + 16)

    def get_gutter(self) -> int:
        return self._gutter_px

    def set_row_sources(self, row_sources):
        """row_sources: list of (label_def, store, title_prefix)"""
        self._row_sources = row_sources
        self.rebuild_rows()

    def set_center_single_row(self, on: bool):
        self._center_single_row = bool(on)
        self.rebuild_rows()

    def set_combined_label_text(self, show: bool):
        self._combined_show_text = bool(show)
        self.rebuild_rows()

    def set_combined_groups(self, groups):
        """Define grouped combined rows: list of (title, row_sources)."""
        self._combined_groups = groups or None
        self.rebuild_rows()

    def set_tail_combined_groups(self, groups):
        """Define combined rows appended after per-label rows."""
        self._tail_combined_groups = groups or None
        if self.layout_mode != "combined":
            self.rebuild_rows()

    def set_gap_summary(self, text: str, tooltip: str = "", has_gaps: bool = False):
        if not getattr(self, "lbl_gap", None):
            return
        self.lbl_gap.setText(text)
        self.lbl_gap.setToolTip(tooltip or "")
        if has_gaps:
            self.lbl_gap.setStyleSheet("color: #b42318; font-weight: 600;")
        else:
            self.lbl_gap.setStyleSheet("color: #667085;")

    def set_combined_editable(self, on: bool):
        self._combined_editable = bool(on)
        if self.layout_mode == "combined" and self._combined_rows:
            for row in self._combined_rows:
                try:
                    meta = getattr(row, "_group_meta", None)
                    if isinstance(meta, dict) and "editable" in meta:
                        row.set_editable(bool(meta["editable"]))
                    else:
                        row.set_editable(self._combined_editable)
                except Exception:
                    pass
        self.rebuild_rows()

    def set_row_delete_handler(self, handler):
        self._row_delete_handler = handler
        if self.layout_mode != "combined":
            for row in self.rows:
                try:
                    row.set_delete_handler(handler)
                except Exception:
                    pass

    def set_row_split_handler(self, handler):
        self._row_split_handler = handler
        if self.layout_mode != "combined":
            for row in self.rows:
                try:
                    row.set_split_handler(handler)
                except Exception:
                    pass

    def _apply_snap_tuning_to_row(self, row) -> None:
        if row is None:
            return
        try:
            row.set_current_snap_radius(self._current_frame_snap_radius)
        except Exception:
            pass
        try:
            row.set_frame_snap_radius(self._frame_snap_radius)
        except Exception:
            pass
        try:
            row.set_edge_snap_frames(self._edge_snap_frames)
        except Exception:
            pass
        try:
            row.set_segment_snap_radius(self._segment_snap_radius)
        except Exception:
            pass

    def set_snap_tuning(
        self,
        current_frame_radius: Optional[int] = None,
        frame_snap_radius: Optional[int] = None,
        edge_snap_frames: Optional[int] = None,
        segment_snap_radius: Optional[int] = None,
        refresh: bool = True,
    ) -> None:
        changed = False
        if current_frame_radius is not None:
            try:
                val = max(0, int(current_frame_radius))
            except Exception:
                val = CURRENT_FRAME_SNAP_RADIUS_FRAMES
            if val != self._current_frame_snap_radius:
                self._current_frame_snap_radius = val
                changed = True
        if frame_snap_radius is not None:
            try:
                val = max(0, int(frame_snap_radius))
            except Exception:
                val = SNAP_RADIUS_FRAMES
            if val != self._frame_snap_radius:
                self._frame_snap_radius = val
                changed = True
        if edge_snap_frames is not None:
            try:
                val = max(0, int(edge_snap_frames))
            except Exception:
                val = EDGE_SNAP_FRAMES
            if val != self._edge_snap_frames:
                self._edge_snap_frames = val
                changed = True
        if segment_snap_radius is not None:
            try:
                val = max(0, int(segment_snap_radius))
            except Exception:
                val = SNAP_RADIUS_FRAMES
            if val != self._segment_snap_radius:
                self._segment_snap_radius = val
                changed = True
        if not changed and not refresh:
            return
        for row in self.rows:
            self._apply_snap_tuning_to_row(row)
        if refresh:
            self.refresh_all_rows()

    def set_row_segment_cuts_provider(self, provider):
        self._row_segment_cuts_provider = provider if callable(provider) else None
        self.apply_row_segment_cuts()

    def apply_row_segment_cuts(self):
        provider = self._row_segment_cuts_provider
        for row in self.rows:
            if not hasattr(row, "set_segment_cuts"):
                continue
            cuts = []
            if callable(provider):
                try:
                    cuts = list(provider(row) or [])
                except Exception:
                    cuts = []
            try:
                row.set_segment_cuts(cuts)
            except Exception:
                pass

    def set_row_edit_mask_provider(self, provider):
        self._row_edit_mask_provider = provider if callable(provider) else None
        self.apply_row_edit_masks()

    def apply_row_edit_masks(self):
        provider = self._row_edit_mask_provider
        for row in self.rows:
            if not hasattr(row, "set_edit_mask_spans"):
                continue
            spans = None
            if callable(provider):
                try:
                    spans = provider(row)
                except Exception:
                    spans = None
            try:
                row.set_edit_mask_spans(spans)
            except Exception:
                pass

    def set_combined_delete_handler(self, handler):
        self._combined_delete_handler = handler
        if self.layout_mode == "combined":
            for row in self._combined_rows:
                try:
                    row.set_delete_handler(handler)
                except Exception:
                    pass

    def set_combined_split_handler(self, handler):
        self._combined_split_handler = handler
        if self.layout_mode == "combined":
            for row in self._combined_rows:
                try:
                    row.set_split_handler(handler)
                except Exception:
                    pass

    def set_combined_reorder_handler(self, handler):
        self._combined_reorder_handler = handler

    def _combined_row_at_global(self, global_pos):
        for row in self._combined_rows:
            try:
                top_left = row.mapToGlobal(row.rect().topLeft())
                rect = QRect(top_left, row.size())
                if rect.contains(global_pos):
                    return row
            except Exception:
                continue
        return None

    def _handle_combined_row_drop(self, src_row, global_pos):
        if not callable(self._combined_reorder_handler):
            return
        target = self._combined_row_at_global(global_pos)
        if target is None or target is src_row:
            return
        try:
            self._combined_reorder_handler(
                getattr(src_row, "title", None), getattr(target, "title", None)
            )
        except Exception:
            pass

    def apply_combined_label(self, name: str) -> bool:
        if self.layout_mode != "combined":
            return False
        row = self._active_combined_row or (
            self._combined_rows[0] if self._combined_rows else None
        )
        if row is None or not getattr(row, "editable", False):
            return False
        try:
            return bool(row.apply_label_to_selection(name))
        except Exception:
            return False

    def set_subtitle_tracks(self, tracks: list):
        self.subtitle_tracks = tracks or []
        self.rebuild_rows()

    def set_layout_mode(self, mode: str):
        target = "combined" if mode == "combined" else "per_label"
        self.layout_mode = target
        try:
            self.chk_layout.blockSignals(True)
            self.chk_layout.setChecked(target == "combined")
            self.chk_layout.blockSignals(False)
        except Exception:
            pass
        if target == "combined":
            self.fit_full_view()
        self.rebuild_rows()

    def _on_layout_mode_toggled(self, on: bool):
        self.layout_mode = "combined" if on else "per_label"
        self.rebuild_rows()

    def set_highlight_labels(self, names):
        self.highlight_labels = set(names or [])
        for row in self.rows:
            if hasattr(row, "set_highlight_labels"):
                try:
                    row.set_highlight_labels(self.highlight_labels)
                except Exception:
                    pass
                continue
            if hasattr(row, "label") and hasattr(row, "set_highlighted"):
                try:
                    row.set_highlighted(row.label.name in self.highlight_labels)
                except Exception:
                    pass
        self.refresh_all_rows()

    def flash_boundary_marker(self, frame: int):
        for row in self.rows:
            if hasattr(row, "set_boundary_flash"):
                try:
                    row.set_boundary_flash(frame)
                except Exception:
                    pass
        self.refresh_all_rows()

    def set_extra_cuts(self, cuts: List[int]):
        self._extra_cuts = list(cuts or [])
        for row in self.rows:
            if hasattr(row, "_extra_cuts"):
                try:
                    row._extra_cuts = list(self._extra_cuts)
                except Exception:
                    pass
        self.refresh_all_rows()

    def set_segment_cuts(self, cuts: List[int]):
        self._segment_cuts = list(cuts or [])
        for row in self.rows:
            if hasattr(row, "_segment_cuts"):
                try:
                    meta = getattr(row, "_group_meta", None)
                    if (
                        isinstance(meta, dict)
                        and meta.get("show_segment_cuts") is False
                    ):
                        row._segment_cuts = []
                    elif (
                        isinstance(meta, dict) and meta.get("segment_cuts") is not None
                    ):
                        row._segment_cuts = list(meta.get("segment_cuts") or [])
                    else:
                        row._segment_cuts = list(self._segment_cuts)
                except Exception:
                    pass
        self.refresh_all_rows()

    def set_snap_segments(self, segments: List[Tuple[int, int]]):
        self._snap_segments = list(segments or [])
        for row in self.rows:
            if hasattr(row, "set_snap_segments"):
                try:
                    row.set_snap_segments(self._snap_segments)
                except Exception:
                    pass
        self.refresh_all_rows()

    def set_interaction_points(self, points, active_id=None):
        self._interaction_points = list(points or [])
        self._active_interaction_id = active_id
        for row in self.rows:
            if hasattr(row, "set_interaction_points"):
                try:
                    row.set_interaction_points(self._interaction_points, active_id)
                except Exception:
                    pass
        self.refresh_all_rows()

    def set_current_hits(self, names):
        hits = set(names or [])
        self._current_hits = hits
        for row in self.rows:
            if hasattr(row, "set_current_hits"):
                try:
                    row.set_current_hits(hits)
                except Exception:
                    pass
                continue
            if hasattr(row, "label") and hasattr(row, "set_current_hit"):
                try:
                    row.set_current_hit(row.label.name in hits)
                except Exception:
                    pass

    def _ensure_visible(self, frame: int):
        """Ensure frame is within view; recenters if outside."""
        fc = max(1, self.get_fc())
        span = max(1, self.view_span)
        start = self.view_start
        end = start + span - 1
        if frame < start or frame > end:
            new_start = max(0, min(fc - span, frame - span // 2))
            self._block_view_signal = True
            self.slider_view.blockSignals(True)
            self.slider_view.setValue(new_start)
            self.slider_view.blockSignals(False)
            self._block_view_signal = False
            self.view_start = new_start
            for r in self.rows:
                r.update()

    def set_current_frame(self, frame: int, follow: bool = False):
        self.current_frame = max(0, int(frame))
        if follow:
            self._ensure_visible(self.current_frame)
        for row in self.rows:
            if hasattr(row, "set_current_frame"):
                try:
                    row.set_current_frame(self.current_frame)
                except Exception:
                    pass

    def _on_combined_segment_selected(self, row):
        self._active_combined_row = row

    def _emit_segment_selected(self, start: int, end: int, label, row):
        self._on_combined_segment_selected(row)
        self.segmentSelected.emit(start, end, label)

    def rebuild_rows(self):
        # clear old layout items (rows, subtitles, spacers)
        while self.vbox.count():
            item = self.vbox.takeAt(0)
            w = item.widget() if item else None
            if w is not None:
                try:
                    w.hide()
                except Exception:
                    pass
                try:
                    w.deleteLater()
                except Exception:
                    pass
            elif item is not None and item.layout() is not None:
                try:
                    item.layout().setParent(None)
                except Exception:
                    pass
        self.rows.clear()
        self._combined_rows = []
        self._active_combined_row = None
        self._compute_gutter_px_from_sources()

        # row sources: default to global store for each label when not provided
        sources = self._row_sources or [(lb, self.store, "") for lb in self.labels]

        # build rows
        def _add_combined_row(group):
            if isinstance(group, (list, tuple)) and len(group) >= 2:
                title = group[0]
                group_sources = group[1]
                meta = group[2] if len(group) >= 3 else None
            else:
                title = "Timeline"
                group_sources = sources
                meta = None
            row_height = None
            labels_for_row = self.labels
            show_label_text = getattr(self, "_combined_show_text", True)
            editable = getattr(self, "_combined_editable", False)
            show_extra_overlay = True
            segment_cuts = getattr(self, "_segment_cuts", [])
            if isinstance(meta, dict):
                row_height = meta.get("row_height")
                if meta.get("labels"):
                    labels_for_row = meta.get("labels") or self.labels
                if "show_label_text" in meta:
                    show_label_text = bool(meta["show_label_text"])
                if "editable" in meta:
                    editable = bool(meta["editable"])
                if "show_extra_overlay" in meta:
                    show_extra_overlay = bool(meta["show_extra_overlay"])
                if "segment_cuts" in meta:
                    segment_cuts = list(meta["segment_cuts"] or [])
                if "show_segment_cuts" in meta and not meta["show_segment_cuts"]:
                    segment_cuts = []
            row = CombinedTimelineRow(
                labels_for_row,
                group_sources,
                self.get_fc,
                self.get_view_start,
                self.get_view_span,
                self.get_fps,
                self.get_gutter,
                title=title,
                show_label_text=show_label_text,
                extra_cuts=getattr(self, "_extra_cuts", []),
                segment_cuts=segment_cuts,
                editable=editable,
                show_extra_overlay=show_extra_overlay,
            )
            if row_height is not None:
                try:
                    rh = int(row_height)
                    if rh > 0:
                        row.setMinimumHeight(rh)
                        row.setMaximumHeight(rh)
                except Exception:
                    pass
            self._apply_snap_tuning_to_row(row)
            row._timeline_ref = self
            if meta is not None:
                try:
                    row._group_meta = meta
                except Exception:
                    pass
                try:
                    if isinstance(meta, dict):
                        snap_mode = meta.get("snap_mode")
                        if snap_mode == "soft":
                            row._snap_soft = True
                        if "snap_radius" in meta:
                            row.set_segment_snap_radius(
                                int(meta.get("snap_radius", SNAP_RADIUS_FRAMES))
                            )
                except Exception:
                    pass
            try:
                row.set_delete_handler(self._combined_delete_handler)
            except Exception:
                pass
            try:
                row.set_split_handler(self._combined_split_handler)
            except Exception:
                pass
            row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.hoverFrame.connect(self.hoverFrame.emit)
            row.labelClicked.connect(self.labelClicked.emit)
            row.segmentSelected.connect(
                lambda _s, _e, _lb, r=row: self._emit_segment_selected(_s, _e, _lb, r)
            )
            row.changed.connect(self.changed.emit)
            row.set_highlight_labels(self.highlight_labels)
            row.set_current_frame(self.current_frame)
            row.set_current_hits(self._current_hits)
            try:
                row.set_interaction_points(
                    getattr(self, "_interaction_points", []),
                    getattr(self, "_active_interaction_id", None),
                )
            except Exception:
                pass
            try:
                if isinstance(meta, dict) and meta.get("snap_segments") is not None:
                    row.set_snap_segments(meta.get("snap_segments") or [])
                else:
                    row.set_snap_segments(getattr(self, "_snap_segments", []))
            except Exception:
                pass
            self.vbox.addWidget(row)
            self.rows.append(row)
            self._combined_rows.append(row)

        if self.layout_mode == "combined":
            groups = self._combined_groups or [("Timeline", sources)]
            center_block = bool(self._center_single_row) and not getattr(
                self, "subtitle_tracks", []
            )
            if center_block:
                self.vbox.setSpacing(0)
            else:
                self.vbox.setSpacing(self._default_vbox_spacing)
            if center_block:
                self.vbox.addStretch(1)
            for group in groups:
                _add_combined_row(group)
                if center_block:
                    self.vbox.addStretch(1)
        else:
            self.vbox.setSpacing(self._default_vbox_spacing)
            for lb, st, prefix in sources:
                row = TimelineRow(
                    lb,  # LabelDef
                    st,
                    self.get_fc,
                    self.get_view_start,
                    self.get_view_span,
                    self.get_fps,
                    self.get_gutter,
                    title_prefix=prefix,  # prefix shown on the left
                )
                self._apply_snap_tuning_to_row(row)
                row.hoverFrame.connect(self.hoverFrame.emit)
                row.changed.connect(self.changed.emit)
                row.set_highlighted(lb.name in self.highlight_labels)
                row.set_current_frame(self.current_frame)
                row.set_current_hit(lb.name in self._current_hits)
                if hasattr(row, "set_interaction_points"):
                    try:
                        row.set_interaction_points(
                            getattr(self, "_interaction_points", []),
                            getattr(self, "_active_interaction_id", None),
                        )
                    except Exception:
                        pass
                try:
                    row.set_snap_segments(getattr(self, "_snap_segments", []))
                except Exception:
                    pass
                try:
                    row.set_delete_handler(self._row_delete_handler)
                except Exception:
                    pass
                try:
                    row.set_split_handler(self._row_split_handler)
                except Exception:
                    pass
                try:
                    provider = self._row_segment_cuts_provider
                    cuts = list(provider(row) or []) if callable(provider) else []
                    row.set_segment_cuts(cuts)
                except Exception:
                    pass
                self.vbox.addWidget(row)
                self.rows.append(row)
            if self._tail_combined_groups:
                for group in self._tail_combined_groups:
                    _add_combined_row(group)

        for tr in getattr(self, "subtitle_tracks", []):
            row = SubtitleRow(
                tr,
                self.get_fc,
                self.get_view_start,
                self.get_view_span,
                self.get_fps,
                self.get_gutter,
            )
            row.hoverFrame.connect(self.hoverFrame.emit)
            row.set_current_frame(self.current_frame)
            self.vbox.addWidget(row)
            self.rows.append(row)

        # avoid auto-scrolling down in combined layout
        if self.layout_mode == "combined":
            if not center_block:
                self.vbox.addSpacing(0)
        else:
            self.vbox.addStretch(1)
        self._init_sliders()
        try:
            sb = self.scroll.verticalScrollBar()
            sb.blockSignals(True)
            sb.setValue(0)
            sb.blockSignals(False)
        except Exception:
            pass
        self.apply_row_segment_cuts()
        self.apply_row_edit_masks()
        self.update()

    def _row_by_name(self, name: str):
        if self.layout_mode == "combined":
            return self.rows[0] if self.rows else None
        for r in self.rows:
            if getattr(r, "label", None) and r.label.name == name:
                return r
        return None

    def flash_label(self, name: str):
        """Scroll to the label row, make it visible, and blink highlight twice."""
        if self.layout_mode == "combined":
            row = self.rows[0] if self.rows else None
            if row is None:
                return
            row_ref = weakref.ref(row)
            try:
                sb = self.scroll.verticalScrollBar()
                sb.setValue(0)
            except Exception:
                pass
            try:
                _safe_qt_call(row_ref, "flash_labels", [name])
            except Exception:
                pass
            return
        row = self._row_by_name(name)
        if row is None:
            return
        row_ref = weakref.ref(row)
        # scroll into view (roughly center it)
        try:
            sb = self.scroll.verticalScrollBar()
            y = row.pos().y()
            h = row.height()
            target = max(0, y + h // 2 - self.scroll.viewport().height() // 2)
            sb.setValue(target)
        except Exception:
            pass

        base_on = row.label.name in self.highlight_labels

        def set_state(on: bool):
            _safe_qt_call(row_ref, "set_highlighted", base_on or on)

        # blink twice
        set_state(True)
        QTimer.singleShot(220, row, lambda: set_state(False))
        QTimer.singleShot(440, row, lambda: set_state(True))
        QTimer.singleShot(660, row, lambda: set_state(base_on))

    def refresh_all_rows(self):
        for r in getattr(self, "rows", []):
            try:
                r.update()
            except Exception:
                pass
        try:
            self.container.update()
        except Exception:
            pass
        self.update()

    def focus_combined_title(self, title: str):
        if self.layout_mode != "combined" or not title:
            return
        for row in self._combined_rows:
            if getattr(row, "title", "") == title:
                self._active_combined_row = row
                try:
                    sb = self.scroll.verticalScrollBar()
                    y = row.pos().y()
                    h = row.height()
                    target = max(0, y + h // 2 - self.scroll.viewport().height() // 2)
                    sb.setValue(target)
                except Exception:
                    pass
                break

    def fit_full_view(self):
        """Fit view to full frame count (helpful for single-timeline scale)."""
        fc = max(1, self.get_fc())
        self.view_start = 0
        self.view_span = fc
        self._init_sliders()
        self.refresh_all_rows()

    # ----- sliders -----
    def _init_sliders(self):
        fc = max(1, self.get_fc())
        # span slider: 0..100 -> MIN_VIEW_SPAN..fc
        self._block_view_signal = True
        self.slider_span.blockSignals(True)
        self.slider_span.setMinimum(0)
        self.slider_span.setMaximum(100)

        # linear mapping: val=0 -> min, val=100 -> full
        def span_to_val(span):
            span = max(MIN_VIEW_SPAN, min(span, fc))
            return int(round(100 * (span - MIN_VIEW_SPAN) / max(1, fc - MIN_VIEW_SPAN)))

        # default span if unset: medium range
        if self.view_span is None:
            self.view_span = min(fc, max(MIN_VIEW_SPAN, fc // 5))
        self.slider_span.setValue(span_to_val(self.view_span))
        self.slider_span.blockSignals(False)

        self._refresh_view_slider()
        self._block_view_signal = False

    def _refresh_view_slider(self):
        fc = max(1, self.get_fc())
        max_start = max(0, fc - self.view_span)
        self.slider_view.blockSignals(True)
        self.slider_view.setMinimum(0)
        self.slider_view.setMaximum(max_start)
        self.view_start = min(self.view_start, max_start)
        self.slider_view.setValue(self.view_start)
        self.slider_view.blockSignals(False)

        for r in self.rows:
            r.update()

    def _on_view_start_changed(self, v: int):
        self.view_start = v
        for r in self.rows:
            r.update()
        if not self._block_view_signal:
            self.viewPanned.emit()

    def _on_view_span_changed(self, val: int):
        # val 0..100 -> span MIN_VIEW_SPAN..full
        fc = max(1, self.get_fc())
        new_span = int(round(MIN_VIEW_SPAN + (fc - MIN_VIEW_SPAN) * val / 100.0))
        new_span = max(MIN_VIEW_SPAN, min(new_span, fc))
        # keep view_start within bounds
        if self.view_start + new_span > fc:
            self.view_start = max(0, fc - new_span)
        self.view_span = new_span
        self._refresh_view_slider()
        if not self._block_view_signal:
            self.viewPanned.emit()
