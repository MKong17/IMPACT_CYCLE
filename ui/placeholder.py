from typing import Callable, List, Optional

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QComboBox,
    QSizePolicy,
)
from PyQt5.QtCore import Qt


class PlaceholderPane(QWidget):
    def __init__(
        self,
        title: str,
        message: str,
        parent=None,
        tasks: Optional[List[str]] = None,
        on_switch_task: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(parent)
        self._task_items = list(tasks or [])
        self._on_switch_task = on_switch_task
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        if self._task_items:
            controls = QHBoxLayout()
            lbl_task = QLabel("Task:")
            lbl_task.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            controls.addWidget(lbl_task)
            self.combo_task = QComboBox()
            self.combo_task.addItems(self._task_items)
            if title in self._task_items:
                self.combo_task.setCurrentText(title)
            self.combo_task.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            self.combo_task.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            self.combo_task.currentTextChanged.connect(self._emit_task_changed)
            controls.addWidget(self.combo_task)
            controls.addStretch(1)
            layout.addLayout(controls)
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        lbl_msg = QLabel(message)
        lbl_msg.setWordWrap(True)
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_msg)
        layout.addStretch(1)

    def _emit_task_changed(self, text: str):
        if callable(self._on_switch_task):
            self._on_switch_task(text)

    def set_task(self, text: str):
        if not getattr(self, "combo_task", None):
            return
        try:
            self.combo_task.blockSignals(True)
            self.combo_task.setCurrentText(text)
            self.combo_task.blockSignals(False)
        except Exception:
            pass
