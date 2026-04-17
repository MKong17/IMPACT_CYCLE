from __future__ import annotations

from typing import Dict, List

from PyQt5.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class ConflictInspectorPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        self.title = QLabel("Conflict Inspector")
        self.title.setStyleSheet("font-weight: 700;")
        root.addWidget(self.title)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Type", "Target", "Status", "Reason"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        root.addWidget(self.table, 1)

    def set_conflicts(self, rows: List[Dict[str, object]]) -> None:
        items = [dict(x) for x in list(rows or []) if isinstance(x, dict)]
        self.table.setRowCount(len(items))
        for i, row in enumerate(items):
            reason = "; ".join([str(x) for x in list(row.get("reasons") or [])[:2]])
            vals = [
                str(row.get("target_type", "")),
                str(row.get("target_id", "")),
                str(row.get("status", "")),
                reason,
            ]
            for c, v in enumerate(vals):
                self.table.setItem(i, c, QTableWidgetItem(v))
        self.table.resizeRowsToContents()
