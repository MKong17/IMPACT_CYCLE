from typing import Dict, List, Any, Optional
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QWidget,
    QSlider,
    QAbstractItemView,
    QHeaderView,
)
from PyQt5.QtCore import Qt


def _normalize_key(text: str) -> str:
    if text is None:
        return ""
    cleaned = str(text).strip().lower()
    cleaned = cleaned.replace("-", "_").replace(" ", "_")
    return cleaned


class PSRRulesDialog(QDialog):
    def __init__(
        self,
        parent=None,
        labels: Optional[List[str]] = None,
        components: Optional[List[Dict[str, Any]]] = None,
        rules: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit PSR/ASR/ASD Rules")
        self.setMinimumSize(520, 420)
        self._labels = labels or []
        self._components = components or []
        self._rules = rules or {}
        self._slider_by_row: Dict[int, QSlider] = {}
        self._combo_lists_by_row: Dict[int, List[QComboBox]] = {}
        self._component_layout_by_row: Dict[int, QVBoxLayout] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        hint = QLabel("State: drag left = -1, center = 0, right = 1.")
        hint.setStyleSheet("color: #666666;")
        root.addWidget(hint)

        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["Label", "Components", "State"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setColumnWidth(0, 220)
        self.table.setColumnWidth(1, 160)
        self.table.setColumnWidth(2, 120)
        try:
            self.table.verticalHeader().setSectionResizeMode(
                QHeaderView.ResizeToContents
            )
        except Exception:
            pass
        root.addWidget(self.table, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_ok = QPushButton("OK")
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_ok)
        root.addLayout(btn_row)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self.accept)

        self._populate()

    def _populate(self):
        comps = list(self._components or [])
        comps_sorted = sorted(
            comps, key=lambda c: (c.get("id") is None, c.get("id", 0))
        )
        rule_map = {str(k): v for k, v in (self._rules or {}).items()}
        self.table.setRowCount(len(self._labels))
        for row, label in enumerate(self._labels):
            item = QTableWidgetItem(label)
            item.setFlags(item.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, 0, item)

            comp_cell, comp_layout = self._make_component_cell()
            self.table.setCellWidget(row, 1, comp_cell)
            self._component_layout_by_row[row] = comp_layout
            combo_list = []
            self._combo_lists_by_row[row] = combo_list
            self._add_component_row(
                comp_layout,
                combo_list,
                comps_sorted,
                selected_id=None,
                selected_name=None,
            )

            slider_cell, slider = self._make_state_slider()
            self.table.setCellWidget(row, 2, slider_cell)
            self._slider_by_row[row] = slider

            rule = rule_map.get(label)
            if rule:
                comps = rule.get("components") or []
                if comps:
                    # replace default with rule components
                    for i in range(len(combo_list) - 1, -1, -1):
                        self._remove_component_row(combo_list, i)
                    for comp_entry in comps:
                        self._add_component_row(
                            comp_layout,
                            combo_list,
                            comps_sorted,
                            selected_id=comp_entry.get("component_id"),
                            selected_name=comp_entry.get("component"),
                        )
                state_val = rule.get("state")
                if state_val is None and comps:
                    state_val = comps[0].get("state")
                if state_val is None:
                    state_val = -1 if "error" in _normalize_key(label) else 0
                slider.setValue(self._clamp_state(state_val))
            else:
                default_state = -1 if "error" in _normalize_key(label) else 0
                slider.setValue(self._clamp_state(default_state))

    def _make_state_slider(self):
        cell = QWidget(self.table)
        layout = QHBoxLayout(cell)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(6)
        slider = QSlider(Qt.Horizontal, cell)
        slider.setMinimum(-1)
        slider.setMaximum(1)
        slider.setSingleStep(1)
        slider.setPageStep(1)
        slider.setTickInterval(1)
        slider.setTickPosition(QSlider.TicksBelow)
        label = QLabel("0", cell)
        label.setMinimumWidth(16)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(slider, 1)
        layout.addWidget(label, 0)

        def _on_change(val):
            label.setText(str(int(val)))

        slider.valueChanged.connect(_on_change)
        slider.setValue(0)
        return cell, slider

    def _make_component_cell(self):
        cell = QWidget(self.table)
        layout = QVBoxLayout(cell)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)
        return cell, layout

    def _add_component_row(
        self, layout, combo_list, comps_sorted, selected_id=None, selected_name=None
    ):
        row = QWidget(self.table)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        combo = QComboBox(row)
        combo.addItem("None", None)
        for comp in comps_sorted:
            cid = comp.get("id")
            name = comp.get("name", "")
            combo.addItem(f"{name}", cid)
        if selected_id is not None:
            idx = combo.findData(selected_id)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        if combo.currentIndex() == 0 and selected_name:
            for i in range(1, combo.count()):
                if _normalize_key(combo.itemText(i)) == _normalize_key(selected_name):
                    combo.setCurrentIndex(i)
                    break
        btn_remove = QToolButton(row)
        btn_remove.setText("×")
        btn_remove.setToolTip("Remove component")
        btn_remove.setFixedWidth(22)
        btn_remove.clicked.connect(
            lambda: self._remove_component_row(combo_list, combo_list.index(combo))
        )
        row_layout.addWidget(combo, 1)
        row_layout.addWidget(btn_remove, 0)

        btn_add = self._take_plus_button(layout)
        layout.addWidget(row)
        combo_list.append(combo)

        # add "+" button on the last row
        if btn_add is None:
            btn_add = self._build_plus_button(layout, combo_list, comps_sorted)
        layout.addWidget(btn_add)
        self._resize_rows_to_contents()

    def _build_plus_button(self, layout, combo_list, comps_sorted):
        btn_add = QToolButton(self.table)
        btn_add.setText("+")
        btn_add.setToolTip("Add component")
        btn_add.setProperty("psr_add", True)
        btn_add.clicked.connect(
            lambda: self._add_component_row(layout, combo_list, comps_sorted)
        )
        return btn_add

    def _take_plus_button(self, layout):
        for i in range(layout.count() - 1, -1, -1):
            item = layout.itemAt(i)
            if not item:
                continue
            w = item.widget()
            if isinstance(w, QToolButton) and w.property("psr_add"):
                layout.removeWidget(w)
                return w
        return None

    def _remove_component_row(self, combo_list, idx: int):
        if idx < 0 or idx >= len(combo_list):
            return
        combo = combo_list.pop(idx)
        row = combo.parentWidget()
        if row is not None:
            row.setParent(None)
        self._resize_rows_to_contents()

    def _resize_rows_to_contents(self) -> None:
        try:
            self.table.resizeRowsToContents()
        except Exception:
            pass

    def _clamp_state(self, value: Any) -> int:
        try:
            val = int(value)
        except Exception:
            val = 0
        if val > 1:
            return 1
        if val < -1:
            return -1
        return val

    def get_rules(self) -> Dict[str, Dict[str, Any]]:
        rules: Dict[str, Dict[str, Any]] = {}
        for row, label in enumerate(self._labels):
            combos = self._combo_lists_by_row.get(row, [])
            slider = self._slider_by_row.get(row)
            if slider is None:
                continue
            state = self._clamp_state(slider.value())
            comps = []
            for combo in combos:
                comp_id = combo.currentData()
                if comp_id is None:
                    continue
                comp_name = combo.currentText()
                comps.append({"component": comp_name, "component_id": comp_id})
            if not comps:
                continue
            rules[label] = {"components": comps, "state": state}
        return rules
