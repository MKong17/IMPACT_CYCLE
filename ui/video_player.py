import cv2
from PyQt5.QtCore import Qt, QTimer, QRect, QRectF
from PyQt5.QtGui import QImage, QPainter, QColor, QFont, QPen, QPainterPath
from PyQt5.QtWidgets import (
    QLabel,
    QToolButton,
    QRubberBand,
    QInputDialog,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QCompleter,
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl


class VideoPlayer(QLabel):
    """
    Zoomable/pannable video viewer:
      - Ctrl + wheel: zoom around cursor (0.2x ~ 8x)
      - Left drag: pan
      - Top-right center button: re-center (keep current zoom)
      - Double-click: fit to window (zoom=1, re-center)
    Backward-compatible API: load(file), seek(frame), play(), pause(), stop(), set_crop(start,end)
    """

    def __init__(self, status_cb=None, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(360)
        self.setContentsMargins(0, 0, 0, 0)
        self.setScaledContents(False)

        # --- playback state ---
        self.cap = None
        self.frame_rate = 30
        self.frame_count = 0
        self.current_frame = 0
        self.crop_start = 0
        self.crop_end = 0
        self.playback_speed = 1.0
        self.is_playing = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.status_cb = status_cb  # func(str)
        self.on_playback_state_changed = None

        # ---- audio player ----
        self.audio_player = QMediaPlayer(self)
        self._audio_enabled = False
        self._audio_path = None
        self._audio_offset_ms = 0  # alignment offset (+ delays audio, - leads audio)
        self._audio_error_notified = False
        self.audio_player.setVolume(100)
        try:
            self.audio_player.error.connect(self._on_audio_error)
        except Exception:
            pass

        # --- frame cache ---
        self._last_qimage = None
        self._last_frame_bgr = None
        self._frame_w = 0
        self._frame_h = 0

        # --- view transform: fit_scale x zoom + pan ---
        self._zoom = 1.0
        self._min_zoom = 0.2
        self._max_zoom = 8.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_active = False
        self._drag_last = None

        # --- top-right recenter button ---
        self.btn_center = QToolButton(self)
        self.btn_center.setText("Center")
        self.btn_center.setToolTip("Re-center (keep zoom)")
        self.btn_center.setCursor(Qt.PointingHandCursor)
        self.btn_center.setStyleSheet(
            """
            QToolButton {
                background: #ffffffDD; border: 1px solid #c8cdd3;
                border-radius: 10px; padding: 2px 6px; font-weight: 600;
            }
            QToolButton:hover { background: #f3f6f9DD; }
        """
        )
        self.btn_center.clicked.connect(self._recenter_keep_zoom)
        self.btn_center.hide()  # hide when no frame is loaded

        # --- magnifier (select to zoom) ---
        self.magnifier_enabled = False
        self._mag_selecting = False
        self._mag_origin = None
        self._rubber = QRubberBand(QRubberBand.Rectangle, self)
        self._rubber.hide()
        self.setMouseTracking(True)

        # overlay state
        self.overlay_enabled = False
        self.overlay_labels = []
        self._audio_speed_supported = True
        self.overlay_boxes = []
        self.overlay_relations = []
        self._overlay_relation_hit_regions = []
        self.overlay_bars = []
        # box editing state
        self.edit_enabled = False
        self.edit_boxes = []
        self.edit_callback = None
        self.edit_ctrl_pick_callback = None
        self.edit_relation_pick_callback = None
        self.edit_relation_edit_callback = None
        self.edit_label_resolver = None
        self.edit_label_suggestions = []
        self._edit_active = None  # (box dict, mode)
        self._edit_anchor = None  # (x, y) image coords at press
        self._edit_add_active = False
        self._edit_add_origin = None
        # click callback
        self.on_click_frame = None
        self._drag_moved = False
        # back-pointer to main window (for Extra mode)
        self.main_window = None

    # ========== basic I/O ==========
    def load(self, path: str) -> bool:
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap = None
            return False

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_rate = int(round(fps)) if fps > 0 else 30
        self.crop_start = 0
        self.crop_end = max(0, self.frame_count - 1)
        self.current_frame = self.crop_start

        # fetch first frame
        self.seek(self.current_frame)
        self._status(f"Loaded: {self.frame_count} frames @ {self.frame_rate} FPS")
        self.btn_center.show()
        return True

    def set_crop(self, start: int, end: int):
        start = max(0, min(start, self.frame_count - 1))
        end = max(start, min(end, self.frame_count - 1))
        self.crop_start, self.crop_end = start, end
        self.seek(self.crop_start)
        self._status(
            f"Cropped to [{self.crop_start}, {self.crop_end}] (absolute frames)"
        )

    def seek(self, frame: int, preview_only=False):
        if not self.cap:
            return
        frame = max(self.crop_start, min(frame, self.crop_end))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, img = self.cap.read()
        if not ret:
            return
        self._last_frame_bgr = img.copy()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self._last_qimage = qimg
        self._frame_w, self._frame_h = w, h

        if not preview_only:
            self.current_frame = frame
        self._ensure_pan_clamped()
        self.update()
        if not preview_only:
            self._sync_audio_to_current_frame()

    def play(self):
        if self.cap and not self.is_playing:
            self.is_playing = True
            self._start_timer()
            if (
                self._audio_enabled
                and self._audio_path
                and not self._audio_error_notified
            ):
                self._sync_audio_to_current_frame()
                self.audio_player.play()
            self._notify_playback_state()

    def _start_timer(self):
        interval = max(
            1, int(round(1000.0 / max(1.0, self.frame_rate * self.playback_speed)))
        )
        self.timer.start(interval)

    def set_playback_speed(self, speed: float):
        # clamp to 0.25x ~ 4x
        speed = max(0.25, min(float(speed), 4.0))
        target_speed = speed
        # Some devices do not support audio rate change; if unsupported, force 1x to keep A/V in sync
        if hasattr(self.audio_player, "setPlaybackRate"):
            if speed == 1.0:
                try:
                    self.audio_player.setPlaybackRate(1.0)
                    self._audio_speed_supported = True
                except Exception:
                    self._audio_speed_supported = False
            elif self._audio_speed_supported:
                try:
                    self.audio_player.setPlaybackRate(speed)
                except Exception:
                    self._audio_speed_supported = False
                    self._status(
                        "Audio device does not support speed change; locked audio to 1x to keep A/V in sync."
                    )
            if not self._audio_speed_supported and speed != 1.0:
                target_speed = 1.0
        else:
            if speed != 1.0:
                target_speed = 1.0

        self.playback_speed = target_speed
        if self.is_playing:
            self._start_timer()

    def pause(self):
        if self.cap and self.is_playing:
            self.is_playing = False
            self.timer.stop()
            if self._audio_enabled:
                self.audio_player.pause()
            self._notify_playback_state()

    def stop(self):
        if self.cap:
            self.pause()
            self.seek(self.crop_start)
            if self._audio_enabled:
                self.audio_player.stop()
                self._sync_audio_to_current_frame()

    def _notify_playback_state(self):
        cb = getattr(self, "on_playback_state_changed", None)
        if callable(cb):
            try:
                cb(bool(self.is_playing))
            except Exception:
                pass

    def _on_timer(self):
        if not self.cap:
            return
        if self.current_frame >= self.crop_end:
            self.stop()
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return
        self.current_frame += 1
        self._last_frame_bgr = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        self._last_qimage = qimg
        self._frame_w, self._frame_h = w, h
        self._ensure_pan_clamped()
        self.update()

        if hasattr(self, "on_frame_advanced") and callable(self.on_frame_advanced):
            self.on_frame_advanced(self.current_frame)

        # Extra mode: keep extending current span and finalize at end
        if self.main_window and getattr(self.main_window, "extra_mode", False):
            try:
                if getattr(self.main_window, "extra_waiting", False):
                    self.main_window._append_extra_progress(self.current_frame)
                if self.current_frame >= self.crop_end and getattr(
                    self.main_window, "extra_waiting", False
                ):
                    self.main_window._finalize_extra_segment(self.current_frame)
                    self.main_window.exit_extra_mode()
            except Exception:
                pass

    def attach_audio_from_video(self, media_path: str) -> bool:
        """Try to mount the container's audio track. Return True on success."""
        try:
            self.audio_player.setMedia(QMediaContent(QUrl.fromLocalFile(media_path)))
            self._audio_enabled = True
            self._audio_error_notified = False
            self._audio_path = media_path
            self._sync_audio_to_current_frame()
            return True
        except Exception:
            return False

    def attach_audio_file(self, audio_path: str):
        """Attach an external audio file (wav/mp3/m4a/etc)."""
        self.audio_player.setMedia(QMediaContent(QUrl.fromLocalFile(audio_path)))
        self._audio_enabled = True
        self._audio_error_notified = False
        self._audio_path = audio_path
        self._sync_audio_to_current_frame()

    def get_current_frame_bgr(self):
        """Return the last rendered frame in BGR (OpenCV order)."""
        if self._last_frame_bgr is None:
            return None
        return self._last_frame_bgr.copy()

    def release_media(self):
        """Release video/audio resources to avoid decoder/file-handle buildup."""
        try:
            self.pause()
        except Exception:
            pass
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.frame_count = 0
        self.current_frame = 0
        self.crop_start = 0
        self.crop_end = 0
        self._last_qimage = None
        self._last_frame_bgr = None
        self._frame_w = 0
        self._frame_h = 0
        try:
            self.audio_player.stop()
            self.audio_player.setMedia(QMediaContent())
        except Exception:
            pass
        self._audio_path = None
        self._audio_error_notified = False
        self.update()

    def closeEvent(self, event):
        try:
            self.release_media()
        except Exception:
            pass
        return super().closeEvent(event)

    def set_audio_enabled(self, on: bool):
        self._audio_enabled = bool(on)
        if not on:
            try:
                self.audio_player.stop()
            except Exception:
                pass
            self._audio_error_notified = False

    def set_audio_offset_ms(self, ms: int):
        """Set audio alignment offset: +ms delays audio, -ms leads audio."""
        self._audio_offset_ms = int(ms)
        self._sync_audio_to_current_frame()

    def _sync_audio_to_current_frame(self):
        if (
            not self._audio_enabled
            or not self._audio_path
            or self._audio_error_notified
        ):
            return
        ms = int(
            round(1000.0 * (self.current_frame / max(1.0, float(self.frame_rate))))
        )
        crop_ms0 = int(
            round(1000.0 * (self.crop_start / max(1.0, float(self.frame_rate))))
        )
        target_ms = max(crop_ms0, ms + int(self._audio_offset_ms))
        self.audio_player.setPosition(max(0, target_ms))

    # ========== view geometry ==========
    def _fit_scale(self) -> float:
        if not (self._frame_w and self._frame_h):
            return 1.0
        lw, lh = max(1, self.width()), max(1, self.height())
        return min(lw / self._frame_w, lh / self._frame_h)

    def _fit_offset(self):
        """Where the image top-left should be when zoom=1 and pan=0 (center-fit)."""
        s = self._fit_scale()
        lw, lh = self.width(), self.height()
        draw_w, draw_h = self._frame_w * s, self._frame_h * s
        return ((lw - draw_w) * 0.5, (lh - draw_h) * 0.5)

    def _draw_rect(self):
        """Target draw rect (x, y, w, h) under current zoom+pan."""
        s = self._fit_scale() * self._zoom
        ox, oy = self._fit_offset()
        x = ox + self._pan_x
        y = oy + self._pan_y
        w = self._frame_w * s
        h = self._frame_h * s
        return x, y, w, h

    def _ensure_pan_clamped(self):
        """Keep the image from being dragged fully out of view; recenter when smaller than viewport."""
        if not (self._frame_w and self._frame_h):
            return
        lw, lh = self.width(), self.height()
        ox, oy = self._fit_offset()
        x, y, w, h = self._draw_rect()

        # X direction
        if w <= lw:
            # smaller than viewport: always center horizontally
            self._pan_x = (lw - w) * 0.5 - ox
        else:
            min_x = lw - w
            max_x = 0.0
            self._pan_x = max(min_x - ox, min(max_x - ox, self._pan_x))

        # Y direction
        if h <= lh:
            self._pan_y = (lh - h) * 0.5 - oy
        else:
            min_y = lh - h
            max_y = 0.0
            self._pan_y = max(min_y - oy, min(max_y - oy, self._pan_y))

    def _zoom_to_selection(self, sel_rect: QRect):
        """Zoom to the selected viewport rectangle."""
        if self._last_qimage is None or sel_rect.width() <= 0 or sel_rect.height() <= 0:
            return
        # intersection with current drawn rect
        x, y, w, h = self._draw_rect()
        draw_rect = QRect(
            int(round(x)), int(round(y)), max(1, int(round(w))), max(1, int(round(h)))
        )
        vis = sel_rect.intersected(draw_rect)
        if vis.width() <= 0 or vis.height() <= 0:
            return

        s_fit = self._fit_scale()
        # convert selection (viewport coords) to image coords
        ix1 = (vis.left() - x - self._pan_x) / (s_fit * self._zoom)
        iy1 = (vis.top() - y - self._pan_y) / (s_fit * self._zoom)
        ix2 = (vis.right() - x - self._pan_x) / (s_fit * self._zoom)
        iy2 = (vis.bottom() - y - self._pan_y) / (s_fit * self._zoom)
        iw = max(1e-3, abs(ix2 - ix1))
        ih = max(1e-3, abs(iy2 - iy1))

        # target zoom so selection fits viewport
        target_zoom = min(self.width() / (iw * s_fit), self.height() / (ih * s_fit))
        target_zoom = max(self._min_zoom, min(self._max_zoom, target_zoom))

        # center selection in viewport
        cx = min(self._frame_w, max(0.0, min(ix1, ix2) + iw * 0.5))
        cy = min(self._frame_h, max(0.0, min(iy1, iy2) + ih * 0.5))
        self._zoom = target_zoom
        ox, oy = self._fit_offset()
        self._pan_x = (self.width() * 0.5) - ox - cx * s_fit * self._zoom
        self._pan_y = (self.height() * 0.5) - oy - cy * s_fit * self._zoom
        self._ensure_pan_clamped()
        self.update()

    def _recenter_keep_zoom(self):
        """Re-center without changing zoom."""
        if not (self._frame_w and self._frame_h):
            return
        lw, lh = self.width(), self.height()
        s = self._fit_scale() * self._zoom
        ox, oy = self._fit_offset()
        draw_w, draw_h = self._frame_w * s, self._frame_h * s
        self._pan_x = (lw - draw_w) * 0.5 - ox
        self._pan_y = (lh - draw_h) * 0.5 - oy
        self.update()

    # ========== interaction ==========
    def wheelEvent(self, e):
        if self._last_qimage is None:
            return super().wheelEvent(e)
        if e.modifiers() & Qt.ControlModifier:
            # zoom around cursor to keep the pixel under cursor fixed
            steps = e.angleDelta().y() / 120.0  # one notch = 120
            if steps == 0:
                return
            old_zoom = self._zoom
            new_zoom = max(
                self._min_zoom, min(self._max_zoom, old_zoom * (1.15**steps))
            )
            if abs(new_zoom - old_zoom) < 1e-6:
                return

            s_fit = self._fit_scale()
            ox, oy = self._fit_offset()
            cx, cy = e.pos().x(), e.pos().y()

            # image coordinates at cursor position (old zoom)
            xi = (cx - (ox + self._pan_x)) / (s_fit * old_zoom)
            yi = (cy - (oy + self._pan_y)) / (s_fit * old_zoom)

            # apply new zoom and back-calc pan to keep anchor fixed
            self._zoom = new_zoom
            self._pan_x = cx - ox - xi * (s_fit * self._zoom)
            self._pan_y = cy - oy - yi * (s_fit * self._zoom)
            self._ensure_pan_clamped()
            self.update()
        else:
            # non-Ctrl wheel falls back to default handling
            return super().wheelEvent(e)

    def resizeEvent(self, e):
        # recompute clamp + place top-right button
        self._ensure_pan_clamped()
        self._place_center_button()
        return super().resizeEvent(e)

    def _place_center_button(self):
        m = 8
        sz = self.btn_center.sizeHint()
        self.btn_center.move(self.width() - sz.width() - m, m)

    # ========== compatibility shims ==========
    def _status(self, msg: str):
        if self.status_cb:
            self.status_cb(msg)

    def set_magnifier_enabled(self, on: bool):
        """Toggle select-to-zoom mode on this view."""
        self.magnifier_enabled = bool(on)
        self._mag_selecting = False
        self._mag_origin = None
        self._rubber.hide()
        # disable pan-drag cursor when turning off magnifier
        if not self.magnifier_enabled and self._drag_active:
            self._drag_active = False
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def set_overlay_enabled(self, on: bool):
        self.overlay_enabled = bool(on)
        self.update()

    def set_overlay_labels(self, labels):
        self.overlay_labels = list(labels or [])
        if self.overlay_enabled:
            self.update()

    def set_overlay_boxes(self, boxes):
        """boxes: list of dicts with x1,y1,x2,y2,label,color"""
        self.overlay_boxes = list(boxes or [])
        self.update()

    def set_overlay_relations(self, rels):
        """rels: list of dicts with x1,y1,x2,y2,label,color representing connectors"""
        self.overlay_relations = list(rels or [])
        self._overlay_relation_hit_regions = []
        self.update()

    def set_overlay_bars(self, bars):
        """bars: list of dicts with color/height/alpha or colors list."""
        self.overlay_bars = list(bars or [])
        if self.overlay_enabled:
            self.update()

    # ========== editable boxes ==========
    def set_edit_context(
        self,
        boxes,
        on_change=None,
        on_ctrl_object_pick=None,
        on_relation_pick=None,
        on_relation_edit=None,
        label_resolver=None,
        label_suggestions=None,
        auto_label_fetcher=None,
    ):
        """Enable editing for the given boxes (list of dict with id/x1/y1/x2/y2/label).
        label_resolver(name:str)->(label, class_id, known:bool) maps ids/names to labels.
        auto_label_fetcher()->(label, class_id) can prefill a new box label.
        """
        self.edit_boxes = list(boxes or [])
        self.edit_enabled = bool(on_change)
        self.edit_callback = on_change
        self.edit_ctrl_pick_callback = on_ctrl_object_pick
        self.edit_relation_pick_callback = on_relation_pick
        self.edit_relation_edit_callback = on_relation_edit
        self.edit_label_resolver = label_resolver
        self.edit_label_suggestions = list(label_suggestions or [])
        self.edit_auto_label_fetcher = auto_label_fetcher
        self._edit_active = None
        self._edit_anchor = None
        self._edit_add_active = False
        self._edit_add_origin = None
        self.update()

    # ========== painting ==========
    def paintEvent(self, e):
        # QLabel paints the background; we draw the frame on top
        super().paintEvent(e)
        if self._last_qimage is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        viewport_rect = QRectF(self.rect().adjusted(1, 1, -1, -1))
        clip_path = QPainterPath()
        clip_path.addRoundedRect(viewport_rect, 18, 18)
        p.setClipPath(clip_path)
        x, y, w, h = self._draw_rect()
        self._last_draw_rect = (x, y, w, h)
        src_w = max(1, self._frame_w)
        src_h = max(1, self._frame_h)
        sx = float(w) / float(src_w)
        sy = float(h) / float(src_h)
        p.drawImage(
            QRect(
                int(round(x)),
                int(round(y)),
                max(1, int(round(w))),
                max(1, int(round(h))),
            ),
            self._last_qimage,
        )
        # draw overlay boxes
        if self.overlay_boxes:
            for box in self.overlay_boxes:
                try:
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                except Exception:
                    continue
                lbl = box.get("label", "")
                col = box.get("color") or "#ff4d4f"
                try:
                    color = QColor(col)
                except Exception:
                    color = QColor("#ff4d4f")
                # scale and offset to drawn image space
                dx1 = int(round(x + x1 * sx))
                dy1 = int(round(y + y1 * sy))
                dx2 = int(round(x + x2 * sx))
                dy2 = int(round(y + y2 * sy))
                pen = QColor(color)
                pen.setAlpha(220)
                width = 4 if box.get("thick") else 2
                p.setPen(QPen(pen, width))
                p.setBrush(Qt.NoBrush)
                p.drawRect(dx1, dy1, max(1, dx2 - dx1), max(1, dy2 - dy1))
                if lbl:
                    font = QFont("Arial", 10)
                    font.setBold(True)
                    p.setFont(font)
                    metrics = p.fontMetrics()
                    try:
                        tw = metrics.horizontalAdvance(lbl) + 6
                    except AttributeError:
                        tw = metrics.width(lbl) + 6
                    th = metrics.height() + 4
                    px = dx1
                    py = max(int(round(y)), dy1 - th)
                    bg = QColor(color)
                    bg.setAlpha(140)
                    p.fillRect(px, py, tw, th, bg)
                    p.setPen(QColor(255, 255, 255))
                    p.drawText(px + 3, py + th - 4, lbl)
                if self.edit_enabled:
                    handle = 6
                    cx = (dx1 + dx2) // 2
                    cy = (dy1 + dy2) // 2
                    points = [
                        (dx1, dy1),
                        (cx, dy1),
                        (dx2, dy1),
                        (dx1, cy),
                        (dx2, cy),
                        (dx1, dy2),
                        (cx, dy2),
                        (dx2, dy2),
                    ]
                    p.setBrush(color)
                    p.setPen(Qt.NoPen)
                    for hx, hy in points:
                        p.drawRect(
                            int(hx - handle / 2), int(hy - handle / 2), handle, handle
                        )
        if getattr(self, "overlay_relations", None):
            self._overlay_relation_hit_regions = []
            for rel in self.overlay_relations:
                try:
                    x1, y1, x2, y2 = rel["x1"], rel["y1"], rel["x2"], rel["y2"]
                except Exception:
                    continue
                lbl = rel.get("label", "")
                col = rel.get("color") or "#1677ff"
                try:
                    color = QColor(col)
                except Exception:
                    color = QColor("#1677ff")
                try:
                    rel_alpha = int(rel.get("alpha", 170) or 170)
                except Exception:
                    rel_alpha = 170
                rel_alpha = max(0, min(255, rel_alpha))
                color.setAlpha(rel_alpha)
                try:
                    line_width = float(rel.get("line_width", 3.0) or 3.0)
                except Exception:
                    line_width = 3.0
                line_width = max(0.8, float(line_width))
                pen = QPen(color, line_width, Qt.SolidLine)
                p.setPen(pen)
                dx1 = int(round(x + x1 * sx))
                dy1 = int(round(y + y1 * sy))
                dx2 = int(round(x + x2 * sx))
                dy2 = int(round(y + y2 * sy))
                p.drawLine(dx1, dy1, dx2, dy2)
                if lbl:
                    try:
                        font_size = int(rel.get("font_size", 10) or 10)
                    except Exception:
                        font_size = 10
                    font_size = max(6, min(24, font_size))
                    font = QFont("Arial", font_size)
                    font.setBold(bool(rel.get("font_bold", True)))
                    p.setFont(font)
                    metrics = p.fontMetrics()
                    try:
                        tw = metrics.horizontalAdvance(lbl) + 6
                    except AttributeError:
                        tw = metrics.width(lbl) + 6
                    th = metrics.height() + 4
                    cx = int((dx1 + dx2) / 2)
                    cy = int((dy1 + dy2) / 2)
                    px = cx - tw // 2
                    py = cy - th // 2
                    bg = QColor(color)
                    try:
                        label_alpha = int(rel.get("label_alpha", 200) or 200)
                    except Exception:
                        label_alpha = 200
                    bg.setAlpha(max(0, min(255, label_alpha)))
                    p.fillRect(px, py, tw, th, bg)
                    edge_id = str(rel.get("edge_id", "") or "").strip()
                    if edge_id:
                        self._overlay_relation_hit_regions.append((edge_id, QRect(px, py, tw, th)))
                    text_color = rel.get("text_color")
                    if text_color:
                        try:
                            p.setPen(QColor(str(text_color)))
                        except Exception:
                            p.setPen(QColor(255, 255, 255))
                    else:
                        p.setPen(QColor(255, 255, 255))
                    p.drawText(px + 3, py + th - 4, lbl)
        if self.overlay_enabled and getattr(self, "overlay_bars", None):
            bars = self.overlay_bars or []
            if bars:
                margin = 6
                spacing = 2
                total_h = 0
                heights = []
                for bar in bars:
                    try:
                        hbar = int(bar.get("height", 6))
                    except Exception:
                        hbar = 6
                    hbar = max(2, hbar)
                    heights.append(hbar)
                    total_h += hbar
                if len(bars) > 1:
                    total_h += spacing * (len(bars) - 1)
                start_y = int(round(y + h - margin - total_h))
                cur_y = start_y
                for idx, bar in enumerate(bars):
                    hbar = heights[idx] if idx < len(heights) else 6
                    colors = bar.get("colors")
                    color = bar.get("color")
                    alpha = bar.get("alpha", 200)
                    label = bar.get("label") if isinstance(bar, dict) else ""
                    label_color = (
                        bar.get("label_color") if isinstance(bar, dict) else None
                    )
                    px = int(round(x + margin))
                    pw = max(1, int(round(w - margin * 2)))
                    cols = []
                    if colors:
                        try:
                            cols = list(colors)
                        except Exception:
                            cols = []
                    if cols:
                        stripe_h = max(1, hbar // len(cols))
                        y0 = cur_y
                        for cidx, col in enumerate(cols):
                            hstripe = (
                                hbar - (stripe_h * cidx)
                                if cidx == len(cols) - 1
                                else stripe_h
                            )
                            try:
                                qcol = QColor(col)
                            except Exception:
                                qcol = QColor("#999999")
                            qcol.setAlpha(alpha if isinstance(alpha, int) else 200)
                            p.fillRect(px, y0, pw, hstripe, qcol)
                            y0 += hstripe
                    elif color:
                        try:
                            qcol = QColor(color)
                        except Exception:
                            qcol = QColor("#999999")
                        qcol.setAlpha(alpha if isinstance(alpha, int) else 200)
                        p.fillRect(px, cur_y, pw, hbar, qcol)
                    draw_border = True
                    try:
                        alpha_val = int(alpha)
                    except Exception:
                        alpha_val = 200
                    if alpha_val <= 0 and not cols and not label:
                        draw_border = False
                    if draw_border:
                        p.setPen(QPen(QColor(0, 0, 0, 120), 1))
                        p.drawRect(px, cur_y, pw, hbar)
                    if label:
                        try:
                            p.save()
                            p.setClipRect(px, cur_y, pw, hbar)
                            font = QFont("Arial", 8)
                            font.setBold(False)
                            p.setFont(font)
                            metrics = p.fontMetrics()
                            max_w = max(4, pw - 4)
                            try:
                                display = metrics.elidedText(
                                    label, Qt.ElideRight, max_w
                                )
                            except Exception:
                                display = label
                            try:
                                tw = metrics.horizontalAdvance(display)
                            except AttributeError:
                                tw = metrics.width(display)
                            tx = px + max(2, int((pw - tw) / 2))
                            ty = cur_y + int(
                                (hbar + metrics.ascent() - metrics.descent()) / 2
                            )
                            p.setPen(QColor(0, 0, 0, 160))
                            p.drawText(tx + 1, ty + 1, display)
                            pen_col = (
                                QColor(label_color)
                                if label_color
                                else QColor(255, 255, 255)
                            )
                            p.setPen(pen_col)
                            p.drawText(tx, ty, display)
                        except Exception:
                            pass
                        finally:
                            try:
                                p.restore()
                            except Exception:
                                pass
                    cur_y += hbar + spacing
        if self.overlay_enabled and self.overlay_labels:
            try:
                p.setRenderHint(QPainter.TextAntialiasing, True)
            except Exception:
                pass
            font = QFont("Arial", 12)
            font.setBold(True)
            p.setFont(font)
            padding = 8
            prepared = []
            for item in self.overlay_labels:
                if isinstance(item, tuple) and len(item) >= 1:
                    text = str(item[0]).strip()
                    col = item[1] if len(item) > 1 else None
                else:
                    text = str(item).strip()
                    col = None
                if not text:
                    continue
                prepared.append((text, col))
            if prepared:
                metrics = p.fontMetrics()
                widths = []
                for text, col in prepared:
                    try:
                        widths.append(metrics.horizontalAdvance(text))
                    except AttributeError:
                        widths.append(metrics.width(text))
                box_w = max(widths or [0]) + padding * 2
                box_h = metrics.height() * len(prepared) + padding * 2
                rect = QRect(int(round(x)) + 8, int(round(y)) + 8, box_w, box_h)
                # darker, more opaque background + subtle border for readability
                p.fillRect(rect, QColor(0, 0, 0, 200))
                p.setPen(QPen(QColor(255, 255, 255, 180)))
                p.drawRect(rect)
                y_offset = rect.top() + padding + metrics.ascent()
                for text, col in prepared:
                    if col:
                        pen_col = QColor(col)
                        pen_col.setAlpha(255)
                        p.setPen(pen_col)
                    else:
                        p.setPen(QColor("#ffd666"))  # bright yellow for contrast
                    p.drawText(rect.left() + padding, y_offset, text)
                    y_offset += metrics.height()
        p.end()

    # ========== box editing interactions ==========
    def _screen_to_image(self, px, py):
        if (
            not hasattr(self, "_last_draw_rect")
            or self._frame_w <= 0
            or self._frame_h <= 0
        ):
            return None
        x, y, w, h = self._last_draw_rect
        if w <= 0 or h <= 0:
            return None
        sx = float(self._frame_w) / float(w)
        sy = float(self._frame_h) / float(h)
        ix = (px - x) * sx
        iy = (py - y) * sy
        return ix, iy

    def _hit_box(self, ix, iy):
        if not self.edit_boxes:
            return None, None
        handle_size = 10
        for box in reversed(self.edit_boxes):
            x1, y1, x2, y2 = (
                box.get("x1", 0),
                box.get("y1", 0),
                box.get("x2", 0),
                box.get("y2", 0),
            )
            # determine handle hits
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            handles = {
                "tl": (x1, y1),
                "t": (cx, y1),
                "tr": (x2, y1),
                "l": (x1, cy),
                "r": (x2, cy),
                "bl": (x1, y2),
                "b": (cx, y2),
                "br": (x2, y2),
            }
            for name, (hx, hy) in handles.items():
                if abs(ix - hx) <= handle_size and abs(iy - hy) <= handle_size:
                    return box, name
            # fallback to body hit
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                return box, "move"
        return None, None

    @staticmethod
    def _cursor_for_mode(mode: str):
        if mode in ("l", "r"):
            return Qt.SizeHorCursor
        if mode in ("t", "b"):
            return Qt.SizeVerCursor
        if mode in ("tl", "br"):
            return Qt.SizeFDiagCursor
        if mode in ("tr", "bl"):
            return Qt.SizeBDiagCursor
        if mode == "move":
            return Qt.SizeAllCursor
        return Qt.ArrowCursor

    def _resolve_label_input(self, dlg):
        raw_txt = dlg.line.text().strip()
        if not raw_txt:
            return None
        resolved_label, resolved_cid, known = dlg.resolved_value()
        lbl_txt = raw_txt
        cid_val = resolved_cid
        if known and resolved_label:
            lbl_txt = resolved_label
        elif raw_txt.isdigit():
            try:
                cid_val = int(raw_txt)
            except Exception:
                cid_val = None
            manual_label, ok_name = QInputDialog.getText(
                self, "New label", f"Label for id {cid_val}:"
            )
            if not ok_name or not manual_label.strip():
                return None
            lbl_txt = manual_label.strip()
        return lbl_txt, cid_val

    def mousePressEvent(self, e):
        # --- Extra mode: single-click boundary ---
        if (
            self.main_window
            and getattr(self.main_window, "extra_mode", False)
            and e.button() == Qt.LeftButton
            and self._last_qimage is not None
        ):
            mw = self.main_window
            frame = self.current_frame
            try:
                mw._split_extra_at(frame)
                e.accept()
                return
            except Exception:
                pass

        # --- relation select on image overlay ---
        if self.edit_enabled and e.button() == Qt.LeftButton:
            for edge_id, rect in list(self._overlay_relation_hit_regions or []):
                if rect.contains(e.pos()):
                    if callable(self.edit_relation_pick_callback):
                        try:
                            self.edit_relation_pick_callback(str(edge_id or ""))
                        except Exception:
                            pass
                    e.accept()
                    return

        # --- add new box (HOI) ---
        if (
            self.edit_enabled
            and e.button() == Qt.LeftButton
            and (e.modifiers() & Qt.ControlModifier)
            and self._last_qimage is not None
        ):
            mapped = self._screen_to_image(e.x(), e.y())
            if mapped:
                bx, _ = self._hit_box(mapped[0], mapped[1])
                if bx and callable(self.edit_ctrl_pick_callback):
                    try:
                        self.edit_ctrl_pick_callback(str(bx.get("id", "") or ""))
                    except Exception:
                        pass
                    e.accept()
                    return
            # Ctrl+drag on empty area keeps "add new box" behavior.
            self._edit_add_active = True
            self._edit_add_origin = e.pos()
            self._rubber.setGeometry(QRect(self._edit_add_origin, e.pos()).normalized())
            self._rubber.show()
            e.accept()
            return

        # --- delete box (HOI) ---
        if self.edit_enabled and e.button() == Qt.RightButton:
            mapped = self._screen_to_image(e.x(), e.y())
            if mapped:
                bx, _ = self._hit_box(mapped[0], mapped[1])
                if bx and callable(self.edit_callback):
                    payload = dict(bx)
                    payload["_action"] = "delete"
                    try:
                        self.edit_callback(bx.get("id"), payload)
                    except Exception:
                        pass
                    self.update()
                    e.accept()
                    return

        # --- box editing (HOI) ---
        if self.edit_enabled and e.button() == Qt.LeftButton:
            pos = e.pos()
            mapped = self._screen_to_image(pos.x(), pos.y())
            if mapped:
                bx, mode = self._hit_box(mapped[0], mapped[1])
                if bx:
                    self._edit_active = {"box": bx, "mode": mode}
                    self._edit_anchor = mapped
                    e.accept()
                    return

        # --- magnifier selection ---
        if (
            self.magnifier_enabled
            and e.button() == Qt.LeftButton
            and self._last_qimage is not None
        ):
            self._mag_selecting = True
            self._mag_origin = e.pos()
            self._rubber.setGeometry(QRect(self._mag_origin, e.pos()).normalized())
            self._rubber.show()
            e.accept()
            return

        # --- pan/drag with left button ---
        if e.button() == Qt.LeftButton and self._last_qimage is not None:
            self._drag_active = True
            self._drag_last = e.pos()
            self._drag_moved = False
            self.setCursor(Qt.ClosedHandCursor)
            e.accept()
            return

        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if (
            self.edit_enabled
            and self._edit_add_active
            and (e.buttons() & Qt.LeftButton)
        ):
            if self._edit_add_origin is not None:
                self._rubber.setGeometry(
                    QRect(self._edit_add_origin, e.pos()).normalized()
                )
                e.accept()
                return

        if (
            self.magnifier_enabled
            and self._mag_selecting
            and self._mag_origin is not None
        ):
            self._rubber.setGeometry(QRect(self._mag_origin, e.pos()).normalized())
            e.accept()
            return

        if self.edit_enabled and self._edit_active and e.buttons() & Qt.LeftButton:
            mapped = self._screen_to_image(e.x(), e.y())
            if not mapped:
                return
            bx = self._edit_active["box"]
            mode = self._edit_active["mode"]
            ix, iy = mapped
            x1, y1, x2, y2 = (
                bx.get("x1", 0),
                bx.get("y1", 0),
                bx.get("x2", 0),
                bx.get("y2", 0),
            )
            if mode == "move":
                ox, oy = self._edit_anchor
                dx, dy = ix - ox, iy - oy
                x1 += dx
                x2 += dx
                y1 += dy
                y2 += dy
                self._edit_anchor = (ix, iy)
            else:
                if "l" in mode:
                    x1 = min(x2 - 1, ix)
                if "r" in mode:
                    x2 = max(x1 + 1, ix)
                if "t" in mode:
                    y1 = min(y2 - 1, iy)
                if "b" in mode:
                    y2 = max(y1 + 1, iy)
            x1 = max(0, min(x1, self._frame_w - 1))
            x2 = max(1, min(x2, self._frame_w))
            y1 = max(0, min(y1, self._frame_h - 1))
            y2 = max(1, min(y2, self._frame_h))
            bx.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            self.update()
            e.accept()
            return
        elif self.edit_enabled:
            mapped = self._screen_to_image(e.x(), e.y())
            if mapped:
                _, mode = self._hit_box(mapped[0], mapped[1])
                self.setCursor(self._cursor_for_mode(mode))
            else:
                self.setCursor(Qt.ArrowCursor)

        if self._drag_active and self._drag_last is not None:
            delta = e.pos() - self._drag_last
            self._drag_last = e.pos()
            self._pan_x += float(delta.x())
            self._pan_y += float(delta.y())
            if delta.x() or delta.y():
                self._drag_moved = True
            self._ensure_pan_clamped()
            self.update()
            return

        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self.edit_enabled and self._edit_add_active and e.button() == Qt.LeftButton:
            rect = QRect(self._edit_add_origin or e.pos(), e.pos()).normalized()
            self._edit_add_active = False
            self._edit_add_origin = None
            self._rubber.hide()
            if (
                rect.width() >= 4
                and rect.height() >= 4
                and self._last_qimage is not None
            ):
                p1 = self._screen_to_image(rect.left(), rect.top())
                p2 = self._screen_to_image(rect.right(), rect.bottom())
                if p1 and p2:
                    x1 = max(0.0, min(p1[0], p2[0]))
                    y1 = max(0.0, min(p1[1], p2[1]))
                    x2 = min(float(self._frame_w), max(p1[0], p2[0]))
                    y2 = min(float(self._frame_h), max(p1[1], p2[1]))
                    if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                        resolved = None
                        if callable(getattr(self, "edit_auto_label_fetcher", None)):
                            try:
                                resolved = self.edit_auto_label_fetcher()
                            except Exception:
                                resolved = None
                        if not resolved:
                            dlg = self._rename_dialog("", title="New box label")
                            if dlg.exec_() == QDialog.Accepted:
                                resolved = self._resolve_label_input(dlg)
                        if resolved and callable(self.edit_callback):
                            lbl_txt, cid_val = resolved
                            payload = {
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "label": lbl_txt,
                                "class_id": cid_val,
                                "frame": self.current_frame,
                                "_action": "add",
                            }
                            try:
                                self.edit_callback(None, payload)
                            except Exception:
                                pass
                            self.update()
            e.accept()
            return

        if self.magnifier_enabled and self._mag_selecting:
            self._mag_selecting = False
            self._rubber.hide()
            rect = QRect(self._mag_origin or e.pos(), e.pos()).normalized()
            self._mag_origin = None
            if (
                rect.width() >= 8
                and rect.height() >= 8
                and self._last_qimage is not None
            ):
                self._zoom_to_selection(rect)
            e.accept()
            return

        if self.edit_enabled and self._edit_active and e.button() == Qt.LeftButton:
            bx = self._edit_active.get("box")
            self._edit_active = None
            self._edit_anchor = None
            if callable(self.edit_callback) and bx:
                try:
                    self.edit_callback(bx.get("id"), dict(bx))
                except Exception:
                    pass
            e.accept()
            return

        if e.button() == Qt.LeftButton and self._drag_active:
            self._drag_active = False
            self._drag_last = None
            self.setCursor(Qt.ArrowCursor)
            if (not self._drag_moved) and callable(
                getattr(self, "on_click_frame", None)
            ):
                try:
                    self.on_click_frame(self.current_frame)
                except Exception:
                    pass
            e.accept()
            return

        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self.edit_enabled and e.button() == Qt.LeftButton:
            for edge_id, rect in list(self._overlay_relation_hit_regions or []):
                if rect.contains(e.pos()):
                    if callable(self.edit_relation_edit_callback):
                        try:
                            self.edit_relation_edit_callback(str(edge_id or ""))
                        except Exception:
                            pass
                    e.accept()
                    return
        if self.edit_enabled and e.button() == Qt.LeftButton:
            mapped = self._screen_to_image(e.x(), e.y())
            if mapped:
                bx, _ = self._hit_box(mapped[0], mapped[1])
                if bx:
                    dlg = self._rename_dialog(str(bx.get("label", "")))
                    if dlg.exec_() == QDialog.Accepted:
                        resolved = self._resolve_label_input(dlg)
                        if not resolved:
                            e.accept()
                            return
                        lbl_txt, cid_val = resolved
                        if cid_val is not None:
                            bx["class_id"] = cid_val
                        bx["label"] = lbl_txt
                        if callable(self.edit_callback):
                            try:
                                self.edit_callback(bx.get("id"), dict(bx))
                            except Exception:
                                pass
                        self.update()
                        e.accept()
                        return

        if e.button() == Qt.LeftButton and self._last_qimage is not None:
            self._zoom = 1.0
            self._pan_x = self._pan_y = 0.0
            self.update()
            e.accept()
            return

        super().mouseDoubleClickEvent(e)

    def _rename_dialog(self, current: str, title: str = "Rename box"):
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setStyleSheet(
            "QDialog { background: #ffffff; color: #111111; }"
            "QLabel { color: #111111; }"
            "QLineEdit { background: #ffffff; color: #111111; border: 1px solid #c9ced6; padding: 4px; }"
            "QPushButton { background: #ffffff; color: #111111; border: 1px solid #c9ced6; padding: 5px 10px; }"
            "QPushButton:hover { background: #f5f7fa; }"
        )
        v = QVBoxLayout(dlg)
        self_line = QLineEdit(current, dlg)
        if self.edit_label_suggestions:
            completer = QCompleter(self.edit_label_suggestions, dlg)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setFilterMode(Qt.MatchContains)
            self_line.setCompleter(completer)
        hint = QLabel("", dlg)
        hint.setStyleSheet("color: #555;")

        def update_hint(txt):
            lbl = ""
            cid = None
            known = False
            if callable(self.edit_label_resolver):
                try:
                    res = self.edit_label_resolver(txt)
                    if isinstance(res, tuple):
                        if len(res) >= 1:
                            lbl = res[0] or ""
                        if len(res) >= 2:
                            cid = res[1]
                        if len(res) >= 3:
                            known = bool(res[2])
                    else:
                        lbl = str(res)
                except Exception:
                    lbl = ""
            if known:
                hint.setText(f"-> {lbl} (id={cid})" if cid is not None else f"-> {lbl}")
            elif txt and txt.isdigit():
                hint.setText(f"New id {txt} (not in class list)")
            elif lbl and (lbl != txt):
                hint.setText(f"-> {lbl}")
            else:
                hint.setText("Enter class id or name; new ids will be added")

        self_line.textChanged.connect(update_hint)
        update_hint(current)
        v.addWidget(QLabel("Enter class id or name:"))
        v.addWidget(self_line)
        v.addWidget(hint)
        btns = QHBoxLayout()
        ok = QPushButton("OK", dlg)
        cancel = QPushButton("Cancel", dlg)
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        v.addLayout(btns)
        dlg.line = self_line

        def _resolved_tuple():
            txt = self_line.text().strip()
            if callable(self.edit_label_resolver):
                try:
                    res = self.edit_label_resolver(txt)
                    if isinstance(res, tuple):
                        if len(res) == 2:
                            return res[0], res[1], True
                        if len(res) >= 3:
                            return res[0], res[1], bool(res[2])
                        return res[0], None, True
                    return str(res), None, True
                except Exception:
                    pass
            return "", None, False

        dlg.resolved_value = _resolved_tuple
        return dlg

    # ========== audio errors ==========

    def _on_audio_error(self, err):
        """Handle audio errors gracefully: notify once and disable audio to avoid repeated popups."""
        if self._audio_error_notified:
            return
        self._audio_error_notified = True
        try:
            self.audio_player.stop()
        except Exception:
            pass
        self._audio_enabled = False
        try:
            es = self.audio_player.errorString()
        except Exception:
            es = str(err)
        self._status(f"Audio disabled (error): {es}")
