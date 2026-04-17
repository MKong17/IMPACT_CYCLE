from typing import Callable, Optional

from PyQt5.QtCore import QMimeData, Qt
from PyQt5.QtGui import QDrag
from PyQt5.QtWidgets import QApplication, QComboBox, QLabel, QWidget


_VIEW_REORDER_MIME = "application/x-cvhci-view-index"


class _NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        # Prevent accidental action changes when scrolling over the combo box.
        event.ignore()


def _view_drag_index_from_mime(mime) -> Optional[int]:
    if mime is None or not mime.hasFormat(_VIEW_REORDER_MIME):
        return None
    try:
        raw = bytes(mime.data(_VIEW_REORDER_MIME)).decode("ascii", errors="ignore")
        return int(raw.strip())
    except Exception:
        return None


class _ViewReorderHandle(QLabel):
    def __init__(
        self,
        view_idx: int,
        on_drop: Optional[Callable[[int, int], None]] = None,
        parent=None,
    ):
        super().__init__("::", parent)
        self._view_idx = int(view_idx)
        self._on_drop = on_drop
        self._drag_start = None
        self.setFixedWidth(18)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.OpenHandCursor)
        self.setAcceptDrops(True)
        self.setToolTip("Drag to reorder view position")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start is None:
            super().mouseMoveEvent(event)
            return
        if not (event.buttons() & Qt.LeftButton):
            super().mouseMoveEvent(event)
            return
        if (
            event.pos() - self._drag_start
        ).manhattanLength() < QApplication.startDragDistance():
            event.accept()
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(_VIEW_REORDER_MIME, str(self._view_idx).encode("ascii"))
        drag.setMimeData(mime)
        drag.exec_(Qt.MoveAction)
        self._drag_start = None
        self.setCursor(Qt.OpenHandCursor)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_start = None
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def _accept_drag(self, event) -> bool:
        src_idx = _view_drag_index_from_mime(event.mimeData())
        if src_idx is None or int(src_idx) == int(self._view_idx):
            event.ignore()
            return False
        event.acceptProposedAction()
        return True

    def dragEnterEvent(self, event):
        self._accept_drag(event)

    def dragMoveEvent(self, event):
        self._accept_drag(event)

    def dropEvent(self, event):
        src_idx = _view_drag_index_from_mime(event.mimeData())
        if src_idx is None or int(src_idx) == int(self._view_idx):
            event.ignore()
            return
        if callable(self._on_drop):
            try:
                self._on_drop(int(src_idx), int(self._view_idx))
            except Exception:
                pass
        event.acceptProposedAction()


class _ViewDropPanel(QWidget):
    def __init__(
        self,
        view_idx: int,
        on_drop: Optional[Callable[[int, int], None]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._view_idx = int(view_idx)
        self._on_drop = on_drop
        self.setAcceptDrops(True)
        self._drop_active = False
        self.setObjectName("viewDropPanel")
        self._refresh_drop_style()

    def _refresh_drop_style(self) -> None:
        if self._drop_active:
            self.setStyleSheet(
                "#viewDropPanel { border: 2px dashed #2f80ed; border-radius: 4px; }"
            )
        else:
            self.setStyleSheet("#viewDropPanel { border: none; }")

    def _set_drop_active(self, on: bool) -> None:
        on = bool(on)
        if on == self._drop_active:
            return
        self._drop_active = on
        self._refresh_drop_style()

    def _accept_drag(self, event) -> bool:
        src_idx = _view_drag_index_from_mime(event.mimeData())
        if src_idx is None or int(src_idx) == int(self._view_idx):
            event.ignore()
            self._set_drop_active(False)
            return False
        event.acceptProposedAction()
        self._set_drop_active(True)
        return True

    def dragEnterEvent(self, event):
        self._accept_drag(event)

    def dragMoveEvent(self, event):
        self._accept_drag(event)

    def dragLeaveEvent(self, event):
        self._set_drop_active(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self._set_drop_active(False)
        src_idx = _view_drag_index_from_mime(event.mimeData())
        if src_idx is None or int(src_idx) == int(self._view_idx):
            event.ignore()
            return
        if callable(self._on_drop):
            try:
                self._on_drop(int(src_idx), int(self._view_idx))
            except Exception:
                pass
        event.acceptProposedAction()
