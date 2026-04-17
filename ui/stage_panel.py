from __future__ import annotations

from typing import Dict, List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class StageVerificationPanel(QWidget):
    actionRequested = pyqtSignal(dict, str)  # item, action

    STAGES = ["Tracks", "Attributes", "Edges", "Semantic Dynamics", "Global Summary"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tables: Dict[str, QTableWidget] = {}
        self._rows: Dict[str, List[Dict[str, object]]] = {k: [] for k in self.STAGES}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        title = QLabel("STAGE Verification")
        title.setStyleSheet("font-weight: 700;")
        root.addWidget(title)

        self.tabs = QTabWidget()
        for stage in self.STAGES:
            table = QTableWidget(0, 5)
            table.setHorizontalHeaderLabels(["Item", "Reliability", "Status", "Reason", "Suggested"])
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.setSelectionMode(QAbstractItemView.SingleSelection)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.horizontalHeader().setStretchLastSection(True)
            table.verticalHeader().setVisible(False)
            host = QWidget()
            host_l = QVBoxLayout(host)
            host_l.setContentsMargins(0, 0, 0, 0)
            host_l.addWidget(table)
            self.tabs.addTab(host, stage)
            self._tables[stage] = table
        root.addWidget(self.tabs, 1)

        btn_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply Suggested")
        self.btn_apply.clicked.connect(lambda: self._emit_action("suggested"))
        self.btn_unknown = QPushButton("Set Unknown")
        self.btn_unknown.clicked.connect(lambda: self._emit_action("unknown"))
        self.btn_relabel = QPushButton("Relabel")
        self.btn_relabel.clicked.connect(lambda: self._emit_action("relabel"))
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(lambda: self._emit_action("remove"))
        for btn in (self.btn_apply, self.btn_unknown, self.btn_relabel, self.btn_remove):
            btn_row.addWidget(btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

    @staticmethod
    def _bg_for_warning(warning: str) -> QColor:
        key = str(warning or "").strip().lower()
        if key == "red":
            return QColor(254, 226, 226)
        if key == "purple":
            return QColor(243, 232, 255)
        if key == "yellow":
            return QColor(254, 249, 195)
        return QColor(255, 255, 255)

    def set_stage_items(self, stage_items: Dict[str, List[Dict[str, object]]]) -> None:
        for stage in self.STAGES:
            rows = [dict(x) for x in list(stage_items.get(stage) or []) if isinstance(x, dict)]
            self._rows[stage] = rows
            table = self._tables[stage]
            table.setRowCount(len(rows))
            for i, row in enumerate(rows):
                reason = "; ".join([str(x) for x in list(row.get("reasons") or [])[:2]])
                values = [
                    str(row.get("label", row.get("target_id", ""))),
                    f"{float(row.get('reliability', 0.0) or 0.0):.3f}",
                    str(row.get("status", "")),
                    reason,
                    str(row.get("suggested_action", "")),
                ]
                bg = self._bg_for_warning(str(row.get("warning", "")))
                for col, text in enumerate(values):
                    item = QTableWidgetItem(text)
                    item.setBackground(bg)
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    table.setItem(i, col, item)
            table.resizeRowsToContents()

    def _selected_item(self) -> Dict[str, object]:
        stage = self.tabs.tabText(self.tabs.currentIndex())
        table = self._tables.get(stage)
        if table is None:
            return {}
        sel = table.selectionModel()
        if sel is None:
            return {}
        rows = sel.selectedRows()
        if not rows:
            return {}
        idx = int(rows[0].row())
        stage_rows = list(self._rows.get(stage) or [])
        if idx < 0 or idx >= len(stage_rows):
            return {}
        return dict(stage_rows[idx])

    def _emit_action(self, action: str) -> None:
        item = self._selected_item()
        if not item:
            return
        self.actionRequested.emit(item, str(action))
