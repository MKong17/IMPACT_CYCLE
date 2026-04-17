from __future__ import annotations

from typing import Dict, List

from PyQt5.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class ReviewQueuePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        title = QLabel("Review Queue (Priority-Sorted)")
        title.setStyleSheet("font-weight: 700;")
        root.addWidget(title)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Priority", "Reliability", "Type", "Target", "Reason"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        root.addWidget(self.table, 1)

    def set_queue(self, rows: List[Dict[str, object]]) -> None:
        items = [dict(x) for x in list(rows or []) if isinstance(x, dict)]
        items.sort(key=lambda row: float(row.get("priority", 0.0) or 0.0), reverse=True)
        self.table.setRowCount(len(items))
        for i, row in enumerate(items):
            reason = "; ".join([str(x) for x in list(row.get("reasons") or [])[:2]])
            vals = [
                f"{float(row.get('priority', 0.0) or 0.0):.3f}",
                f"{float(row.get('reliability', 0.0) or 0.0):.3f}",
                str(row.get("target_type", "")),
                str(row.get("target_id", "")),
                reason,
            ]
            for c, v in enumerate(vals):
                self.table.setItem(i, c, QTableWidgetItem(v))
        self.table.resizeRowsToContents()
