from PyQt5.QtWidgets import QWidget, QVBoxLayout


class TaskHost(QWidget):
    """Simple container used to re-parent a shared task widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._body = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._layout = layout

    def set_body(self, widget):
        if widget is self._body:
            return
        if self._body is not None:
            try:
                self._layout.removeWidget(self._body)
            except Exception:
                pass
        self._body = widget
        if widget is None:
            return
        old_parent = widget.parent()
        if old_parent is not None and old_parent is not self:
            try:
                old_layout = getattr(old_parent, "_layout", None)
                if old_layout is not None:
                    old_layout.removeWidget(widget)
            except Exception:
                pass
            try:
                if getattr(old_parent, "_body", None) is widget:
                    old_parent._body = None
            except Exception:
                pass
        widget.setParent(self)
        self._layout.addWidget(widget)
