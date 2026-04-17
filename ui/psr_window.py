from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QSizePolicy,
    QComboBox,
    QToolButton,
    QCheckBox,
    QGridLayout,
    QShortcut,
    QHeaderView,
    QGroupBox,
    QButtonGroup,
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QKeySequence, QIcon, QPainter, QPen, QColor, QPixmap
from ui.task_host import TaskHost
from utils.psr_models import (
    load_psr_model_registry,
    enabled_psr_models,
    normalize_psr_model_type,
)
from utils.shortcut_settings import (
    load_shortcut_bindings,
    default_shortcut_bindings,
    shortcut_value,
    set_shortcut_key,
)

class ComponentTable(QTableWidget):
    def __init__(
        self, rows, columns, on_reorder=None, on_drop_finished=None, parent=None
    ):
        super().__init__(rows, columns, parent)
        self._on_reorder = on_reorder
        self._on_drop_finished = on_drop_finished
        self._drag_active = False
        self._drag_source_rows = []
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDragDropOverwriteMode(False)
        self.setDefaultDropAction(Qt.CopyAction)
        self.setDropIndicatorShown(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)

    def startDrag(self, supportedActions):
        self._drag_active = True
        self._drag_source_rows = sorted({idx.row() for idx in self.selectedIndexes()})
        if not self._drag_source_rows:
            cur = self.currentRow()
            if cur >= 0:
                self._drag_source_rows = [cur]
        try:
            # Use copy action for internal drag transport to prevent Qt from
            # auto-removing source rows; we apply move semantics ourselves.
            super().startDrag(Qt.CopyAction)
        finally:
            # Ensure drag state is eventually cleared even when drop is cancelled.
            self._drag_active = False
            self._drag_source_rows = []

    def dropEvent(self, event):
        self._drag_active = True
        try:
            if not callable(self._on_reorder):
                super().dropEvent(event)
                return
            source_rows = list(self._drag_source_rows or [])
            if not source_rows:
                cur = self.currentRow()
                if cur >= 0:
                    source_rows = [cur]
            if not source_rows:
                event.ignore()
                return
            row_count = self.rowCount()
            source_rows = [r for r in source_rows if 0 <= r < row_count]
            if not source_rows:
                event.ignore()
                return
            drop_row = self.indexAt(event.pos()).row()
            drop_pos = self.dropIndicatorPosition()
            if drop_row < 0:
                drop_row = row_count
            elif drop_pos in (
                QAbstractItemView.BelowItem,
                QAbstractItemView.OnItem,
            ):
                drop_row += 1
            elif drop_pos == QAbstractItemView.OnViewport:
                drop_row = row_count
            drop_row -= sum(1 for r in source_rows if r < drop_row)
            if drop_row < 0:
                drop_row = 0
            max_dest = row_count - len(source_rows)
            if max_dest < 0:
                max_dest = 0
            if drop_row > max_dest:
                drop_row = max_dest
            # Forward source/destination indices and let the panel apply
            # canonical insertion-order rebalancing.
            self._on_reorder(source_rows, drop_row)
            event.setDropAction(Qt.CopyAction)
            event.accept()
        finally:
            self._drag_active = False
            self._drag_source_rows = []
            if callable(self._on_drop_finished):
                QTimer.singleShot(0, self._on_drop_finished)

    def is_drag_active(self) -> bool:
        return bool(self._drag_active)


class PSRWindow(QWidget):
    """Dedicated PSR/ASR/ASD container with a state side panel."""

    def _create_section_box(self, title: str):
        box = QGroupBox(title, self.side_panel)
        box.setStyleSheet("QGroupBox { font-weight: 600; }")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(8)
        return box, layout

    def _create_compact_button(
        self,
        text: str,
        tooltip: str = "",
        *,
        checkable: bool = False,
        min_width: int = 84,
        min_height: int = 34,
    ):
        btn = QPushButton(text)
        btn.setCheckable(bool(checkable))
        if tooltip:
            btn.setToolTip(tooltip)
        btn.setMinimumWidth(int(min_width))
        btn.setMinimumHeight(int(min_height))
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return btn

    def _icon_only_tooltip(self, title: str, detail: str, shortcut: str = "") -> str:
        parts = [str(title).strip(), str(detail).strip()]
        if shortcut:
            parts.append(f"Shortcut: {shortcut}")
        return "\n".join(part for part in parts if part)

    def _make_scope_icon(self, kind: str) -> QIcon:
        size = 18
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        base_fill = QColor("#d0d5dd")
        base_pen = QPen(QColor("#98a2b3"), 1.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        accent_fill = QColor("#7dd3fc")
        accent_pen = QPen(QColor("#0284c7"), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

        def draw_block(x: int, y: int, w: int, h: int, *, active: bool = False) -> None:
            painter.setPen(accent_pen if active else base_pen)
            painter.setBrush(accent_fill if active else base_fill)
            painter.drawRoundedRect(x, y, w, h, 2.0, 2.0)

        if kind == "segment":
            for idx, x in enumerate((1, 7, 13)):
                draw_block(x, 6, 4, 6, active=(idx == 1))
        elif kind == "from_here":
            draw_block(1, 6, 4, 6, active=False)
            draw_block(7, 6, 4, 6, active=True)
            draw_block(13, 6, 4, 6, active=True)
            painter.setPen(accent_pen)
            painter.drawLine(11, 9, 16, 9)
            painter.drawLine(14, 7, 16, 9)
            painter.drawLine(14, 11, 16, 9)
        elif kind == "split":
            painter.setPen(base_pen)
            painter.setBrush(base_fill)
            painter.drawRoundedRect(1, 6, 16, 6, 2.5, 2.5)
            painter.setPen(QPen(QColor("#0284c7"), 1.8, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(9, 3, 9, 15)
        elif kind == "merge":
            draw_block(1, 6, 6, 6, active=False)
            draw_block(11, 6, 6, 6, active=False)
            painter.setPen(accent_pen)
            painter.drawLine(6, 9, 12, 9)
            painter.drawLine(10, 7, 12, 9)
            painter.drawLine(10, 11, 12, 9)
        painter.end()
        return QIcon(pixmap)

    def _create_symbol_button(
        self,
        kind: str,
        tooltip: str,
        *,
        checkable: bool = False,
        fixed_size: int = 42,
    ):
        btn = QToolButton(self.side_panel)
        btn.setCheckable(bool(checkable))
        btn.setToolTip(tooltip)
        btn.setStatusTip(tooltip.replace("\n", " | "))
        btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        btn.setIcon(self._make_scope_icon(kind))
        btn.setIconSize(QSize(18, 18))
        btn.setFixedSize(fixed_size, fixed_size)
        btn.setStyleSheet(
            """
            QToolButton {
                background: #ffffff;
                border: 1px solid #d0d5dd;
                border-radius: 10px;
                padding: 0px;
            }
            QToolButton:hover {
                background: #f8fafc;
                border-color: #98a2b3;
            }
            QToolButton:pressed {
                background: #eef2f6;
            }
            QToolButton:checked {
                background: #e0f2fe;
                border-color: #38bdf8;
            }
            QToolButton:disabled {
                background: #f2f4f7;
                border-color: #e4e7ec;
            }
            """
        )
        return btn

    def _create_action_button(
        self, text: str, tooltip: str = "", *, visible: bool = True
    ):
        btn = QPushButton(text)
        if tooltip:
            btn.setToolTip(tooltip)
        btn.setVisible(bool(visible))
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return btn

    def _normalize_model_specs(self, models):
        specs = []
        seen = set()
        for raw in list(models or []):
            if not isinstance(raw, dict):
                continue
            model_id = str(raw.get("id") or "").strip()
            if not model_id:
                continue
            key = model_id.upper()
            if key in seen:
                continue
            seen.add(key)
            specs.append(
                {
                    "id": model_id,
                    "display_name": str(
                        raw.get("display_name") or raw.get("name") or model_id
                    ).strip()
                    or model_id,
                    "description": str(raw.get("description") or "").strip(),
                    "enabled": bool(raw.get("enabled", True)),
                }
            )
        return specs or load_psr_model_registry()

    def _clear_layout_widgets(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            try:
                self.model_button_group.removeButton(widget)
            except Exception:
                pass
            widget.deleteLater()

    def _enabled_model_specs(self):
        return enabled_psr_models(self._model_specs)

    def _refresh_model_selector(self) -> None:
        specs = self._enabled_model_specs()
        known_ids = [str(spec.get("id") or "").strip() for spec in specs]
        use_buttons = len(known_ids) <= 2 and self._model_type in known_ids

        self._clear_layout_widgets(self.model_buttons_layout)
        self._model_buttons = {}
        button_style = """
            QPushButton { padding: 4px 10px; }
            QPushButton:checked {
                background: #e0f2fe;
                border: 1px solid #38bdf8;
                color: #075985;
                font-weight: 600;
            }
        """
        for spec in specs:
            model_id = str(spec.get("id") or "").strip()
            display_name = str(spec.get("display_name") or model_id).strip() or model_id
            detail = str(spec.get("description") or "").strip()
            tooltip = display_name if not detail else f"{display_name}\n{detail}"
            btn = self._create_compact_button(
                display_name,
                tooltip,
                checkable=True,
                min_width=98,
                min_height=32,
            )
            btn.setStyleSheet(button_style)
            btn.setProperty("model_type", model_id)
            self.model_button_group.addButton(btn)
            self._model_buttons[model_id] = btn
            self.model_buttons_layout.addWidget(btn)
        self.model_buttons_layout.addStretch(1)

        self.combo_model_type.blockSignals(True)
        self.combo_model_type.clear()
        for spec in specs:
            model_id = str(spec.get("id") or "").strip()
            display_name = str(spec.get("display_name") or model_id).strip() or model_id
            self.combo_model_type.addItem(display_name, model_id)
        if self._model_type and self._model_type not in known_ids:
            self.combo_model_type.addItem(f"Unknown: {self._model_type}", self._model_type)
        idx = self.combo_model_type.findData(self._model_type)
        if idx < 0 and self.combo_model_type.count() > 0:
            idx = 0
            self._model_type = str(self.combo_model_type.itemData(0) or "").strip()
        self.combo_model_type.setCurrentIndex(idx)
        self.combo_model_type.blockSignals(False)

        self.model_buttons_host.setVisible(use_buttons)
        self.combo_model_type.setVisible(not use_buttons)
        for model_id, btn in self._model_buttons.items():
            try:
                prev_block = btn.blockSignals(True)
                btn.setChecked(model_id == self._model_type)
                btn.blockSignals(prev_block)
            except Exception:
                pass

    def _add_grid_widgets(self, grid, placements):
        for widget, row, col, rowspan, colspan in placements:
            grid.addWidget(widget, row, col, rowspan, colspan)

    def _connect_button(self, button, callback, *, disable_if_missing: bool = False):
        if callable(callback):
            button.clicked.connect(callback)
            return
        if disable_if_missing:
            button.setEnabled(False)

    def _create_collapsible_section(self, title: str):
        toggle = QToolButton(self.side_panel)
        toggle.setText(title)
        toggle.setCheckable(True)
        toggle.setChecked(False)
        toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.RightArrow)
        body = QWidget(self.side_panel)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(6)
        body.setVisible(False)
        return toggle, body, layout

    def _set_collapsible_section_visible(self, toggle, body, on: bool) -> None:
        try:
            body.setVisible(bool(on))
            toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        except Exception:
            pass

    def __init__(
        self,
        parent=None,
        on_activate=None,
        on_load_components=None,
        on_save_components=None,
        on_load_rules=None,
        on_edit_rules=None,
        on_export_rules=None,
        on_apply_rules=None,
        on_learn_rules=None,
        on_batch_convert=None,
        on_state_changed=None,
        on_reset_segment=None,
        on_invert_segment=None,
        on_merge_identical=None,
        on_undo=None,
        on_redo=None,
        on_select_from_here=None,
        on_select_segment=None,
        on_split_at_playhead=None,
        on_model_type_changed=None,
        available_models=None,
        initial_model_type="",
    ):
        super().__init__(parent)
        self._on_activate = on_activate
        self._on_load_components = on_load_components
        self._on_save_components = on_save_components
        self._on_load_rules = on_load_rules
        self._on_edit_rules = on_edit_rules
        self._on_export_rules = on_export_rules
        self._on_apply_rules = on_apply_rules
        self._on_learn_rules = on_learn_rules
        self._on_batch_convert = on_batch_convert
        self._on_state_changed = on_state_changed
        self._on_reset_segment = on_reset_segment
        self._on_invert_segment = on_invert_segment
        self._on_merge_identical = on_merge_identical
        self._on_undo = on_undo
        self._on_redo = on_redo
        self._on_select_from_here = on_select_from_here
        self._on_select_segment = on_select_segment
        self._on_split_at_playhead = on_split_at_playhead
        self._on_model_type_changed = on_model_type_changed
        self._shortcut_bindings = load_shortcut_bindings()
        self._shortcut_defaults = default_shortcut_bindings()
        self._component_snapshot = []
        self._state_combos = []
        self._block_state_signals = False
        self._block_component_updates = False
        self._splitter_initialized = False
        self._all_components = []
        self._all_states = []
        self._delta_component_ids = []
        self._ui_component_order = []
        self._segment_scope = "segment"
        self._model_specs = self._normalize_model_specs(available_models)
        self._model_type = normalize_psr_model_type(
            initial_model_type,
            self._model_specs,
            allow_unknown=True,
        )
        self._state_choices = [
            ("Not installed", 0),
            ("Installed", 1),
            ("Error", -1),
        ]
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)
        root.addWidget(self.splitter)

        self.host = TaskHost(self)
        self.splitter.addWidget(self.host)

        self.side_panel = QWidget(self)
        self.side_panel.setMinimumWidth(240)
        self.side_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.splitter.addWidget(self.side_panel)
        self.splitter.setStretchFactor(0, 5)
        self.splitter.setStretchFactor(1, 2)

        side = QVBoxLayout(self.side_panel)
        side.setContentsMargins(10, 10, 10, 10)
        side.setSpacing(8)

        title = QLabel("Assembly State Editor")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title.setStyleSheet("font-weight: 600;")
        side.addWidget(title)

        self.lbl_meta = QLabel("Components: 0 | Rules: 0 | Transitions: 0")
        self.lbl_meta.setStyleSheet("color: #666666;")
        side.addWidget(self.lbl_meta)
        self.lbl_diag = QLabel("")
        self.lbl_diag.setStyleSheet("color: #cc3300;")
        side.addWidget(self.lbl_diag)

        model_row = QHBoxLayout()
        self.lbl_model_type = QLabel("Model")
        self.lbl_model_type.setStyleSheet("color: #666666;")
        model_row.addWidget(self.lbl_model_type)
        self.model_button_group = QButtonGroup(self)
        self.model_button_group.setExclusive(True)
        self._model_buttons = {}
        self.model_buttons_host = QWidget(self.side_panel)
        self.model_buttons_layout = QHBoxLayout(self.model_buttons_host)
        self.model_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.model_buttons_layout.setSpacing(6)
        model_row.addWidget(self.model_buttons_host, 1)
        self.combo_model_type = QComboBox(self.side_panel)
        self.combo_model_type.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo_model_type.currentIndexChanged.connect(self._on_model_combo_changed)
        model_row.addWidget(self.combo_model_type, 1)
        side.addLayout(model_row)

        current_box, current_layout = self._create_section_box("Current Segment")
        self.lbl_state_summary = QLabel("")
        self.lbl_state_summary.setStyleSheet("color: #475467;")
        self.lbl_state_summary.setWordWrap(True)
        self.lbl_state_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        current_layout.addWidget(self.lbl_state_summary)
        self.chk_show_all_components = QCheckBox("Show all components")
        self.chk_show_all_components.setChecked(True)
        self.chk_show_all_components.setToolTip(
            "Off: edit only components changed in this segment."
        )
        current_layout.addWidget(self.chk_show_all_components)

        scope_row = QGridLayout()
        scope_row.setContentsMargins(0, 0, 0, 0)
        scope_row.setHorizontalSpacing(4)
        scope_row.setVerticalSpacing(6)
        for col in range(4):
            scope_row.setColumnStretch(col, 1)
        for name, icon_kind, tooltip, checkable, col in (
            (
                "btn_scope_segment",
                "segment",
                self._icon_only_tooltip(
                    "This Segment",
                    "Edit scope: selected segment only",
                    "Ctrl+Shift+S",
                ),
                True,
                0,
            ),
            (
                "btn_scope_from_here",
                "from_here",
                self._icon_only_tooltip(
                    "From Here",
                    "Edit scope: selected segment and all later segments",
                    "Ctrl+Shift+F",
                ),
                True,
                1,
            ),
            (
                "btn_split_playhead",
                "split",
                self._icon_only_tooltip(
                    "Split",
                    "Split selected state segment at current playhead frame",
                    "Ctrl+K",
                ),
                False,
                2,
            ),
            (
                "btn_merge_identical",
                "merge",
                self._icon_only_tooltip(
                    "Merge",
                    "Merge adjacent identical segments",
                    "Ctrl+M",
                ),
                False,
                3,
            ),
        ):
            btn = self._create_symbol_button(
                icon_kind,
                tooltip,
                checkable=checkable,
            )
            setattr(self, name, btn)
            scope_row.addWidget(btn, 0, col)
        current_layout.addLayout(scope_row)

        for name, text, tooltip, visible in (
            ("btn_load_components", "Load components", "", False),
            ("btn_save_components", "Save components", "", False),
            ("btn_load_rules", "Load rules", "", True),
            ("btn_edit_rules", "Edit rules", "", True),
            ("btn_apply_rules", "Apply rules", "", True),
            ("btn_export_rules", "Export rules", "", True),
            ("btn_learn_rules", "Learn rules from edits", "", True),
            ("btn_batch_convert", "Batch convert dataset", "", True),
        ):
            setattr(
                self,
                name,
                self._create_action_button(text, tooltip, visible=visible),
            )

        self.btn_toggle_rules, self.rules_body, rules_layout = self._create_collapsible_section(
            "Rules"
        )
        controls = QGridLayout()
        controls.setHorizontalSpacing(8)
        controls.setVerticalSpacing(6)
        self._add_grid_widgets(
            controls,
            [
                (self.btn_load_components, 0, 0, 1, 1),
                (self.btn_save_components, 0, 1, 1, 1),
                (self.btn_load_rules, 1, 0, 1, 1),
                (self.btn_edit_rules, 1, 1, 1, 1),
                (self.btn_apply_rules, 2, 0, 1, 1),
                (self.btn_export_rules, 2, 1, 1, 1),
            ],
        )
        controls.setColumnStretch(0, 1)
        controls.setColumnStretch(1, 1)
        rules_layout.addLayout(controls)

        self.btn_toggle_advanced, self.advanced_body, advanced_layout = (
            self._create_collapsible_section("Advanced")
        )
        advanced_layout.addWidget(self.btn_learn_rules)
        advanced_layout.addWidget(self.btn_batch_convert)
        current_layout.addWidget(self.btn_toggle_rules)
        current_layout.addWidget(self.rules_body)
        current_layout.addWidget(self.btn_toggle_advanced)
        current_layout.addWidget(self.advanced_body)

        table_header = QHBoxLayout()
        self.lbl_table_hint = QLabel("")
        self.lbl_table_hint.setStyleSheet("color: #666666;")
        self.lbl_table_hint.setVisible(False)
        table_header.addWidget(self.lbl_table_hint)
        table_header.addStretch(1)
        self.btn_reset_segment = QToolButton()
        self.btn_reset_segment.setText("🔁")
        self.btn_reset_segment.setToolTip(
            "Reset selected state segment to rule-derived state"
        )
        self.btn_reset_segment.setFixedWidth(28)
        self.btn_invert_segment = QToolButton()
        self.btn_invert_segment.setText("⇄")
        self.btn_invert_segment.setToolTip(
            "Invert state for the selected segment (Installed <-> Not installed)"
        )
        self.btn_invert_segment.setFixedWidth(28)
        table_header.addWidget(self.btn_reset_segment)
        table_header.addWidget(self.btn_invert_segment)
        current_layout.addLayout(table_header)

        self.table = ComponentTable(
            0,
            3,
            on_reorder=self._on_components_reordered,
            on_drop_finished=self._on_components_drop_finished,
        )
        self.table.setHorizontalHeaderLabels(["ID", "Component", "State"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setColumnWidth(0, 42)
        self.table.setColumnWidth(1, 160)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        # UI-only reordering: row order affects display convenience only.
        # Host-side component ids/order remain fixed for export consistency.
        self.table.setDragEnabled(True)
        self.table.setAcceptDrops(True)
        self.table.setDragDropMode(QAbstractItemView.DragDrop)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        current_layout.addWidget(self.table, 1)

        side.addWidget(current_box, 1)

        self._connect_button(self.btn_load_components, self._on_load_components)
        self._connect_button(self.btn_save_components, self._on_save_components)
        self._connect_button(self.btn_load_rules, self._on_load_rules)
        self._connect_button(self.btn_edit_rules, self._on_edit_rules)
        self._connect_button(self.btn_export_rules, self._on_export_rules)
        if callable(self._on_apply_rules):
            self.btn_apply_rules.clicked.connect(self._on_apply_rules)
        self._connect_button(self.btn_reset_segment, self._on_reset_segment)
        self._connect_button(self.btn_invert_segment, self._on_invert_segment)
        self._connect_button(
            self.btn_split_playhead,
            self._on_split_clicked if callable(self._on_split_at_playhead) else None,
            disable_if_missing=True,
        )
        self._connect_button(
            self.btn_merge_identical,
            self._on_merge_identical,
            disable_if_missing=True,
        )
        self._connect_button(
            self.btn_scope_segment,
            self._on_scope_segment_clicked if callable(self._on_select_segment) else None,
            disable_if_missing=True,
        )
        self._connect_button(
            self.btn_scope_from_here,
            self._on_scope_from_here_clicked
            if callable(self._on_select_from_here)
            else None,
            disable_if_missing=True,
        )
        self._connect_button(self.btn_learn_rules, self._on_learn_rules)
        self._connect_button(self.btn_batch_convert, self._on_batch_convert)
        self.chk_show_all_components.toggled.connect(self._render_component_table)
        self.chk_show_all_components.toggled.connect(self._update_table_drag_enabled)
        self.btn_toggle_rules.toggled.connect(self._set_rules_visible)
        self.btn_toggle_advanced.toggled.connect(self._set_advanced_visible)
        self.model_button_group.buttonClicked.connect(self._on_model_button_clicked)
        self._set_rules_visible(False)
        self._set_advanced_visible(False)
        self.set_available_models(self._model_specs, emit=False)
        self.set_model_type(initial_model_type, emit=False)

        if callable(self._on_undo):
            sc_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
            sc_undo.setContext(Qt.ApplicationShortcut)
            sc_undo.setEnabled(False)
            sc_undo.activated.connect(self._on_undo)
            self._sc_undo = sc_undo
        if callable(self._on_redo):
            sc_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
            sc_redo.setContext(Qt.ApplicationShortcut)
            sc_redo.setEnabled(False)
            sc_redo.activated.connect(self._on_redo)
            self._sc_redo = sc_redo
        if callable(self._on_split_at_playhead):
            sc_split = QShortcut(QKeySequence("Ctrl+K"), self)
            sc_split.setContext(Qt.ApplicationShortcut)
            sc_split.setEnabled(False)
            sc_split.activated.connect(self._on_split_clicked)
            self._sc_split = sc_split
        if callable(self._on_select_segment):
            sc_scope_segment = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
            sc_scope_segment.setContext(Qt.ApplicationShortcut)
            sc_scope_segment.setEnabled(False)
            sc_scope_segment.activated.connect(self._on_scope_segment_clicked)
            self._sc_scope_segment = sc_scope_segment
        if callable(self._on_select_from_here):
            sc_scope_from_here = QShortcut(QKeySequence("Ctrl+Shift+F"), self)
            sc_scope_from_here.setContext(Qt.ApplicationShortcut)
            sc_scope_from_here.setEnabled(False)
            sc_scope_from_here.activated.connect(self._on_scope_from_here_clicked)
            self._sc_scope_from_here = sc_scope_from_here
        if callable(self._on_reset_segment):
            sc_reset = QShortcut(QKeySequence("Ctrl+Backspace"), self)
            sc_reset.setContext(Qt.ApplicationShortcut)
            sc_reset.setEnabled(False)
            sc_reset.activated.connect(self._on_reset_segment)
            self._sc_reset = sc_reset
        if callable(self._on_invert_segment):
            sc_invert = QShortcut(QKeySequence("Ctrl+I"), self)
            sc_invert.setContext(Qt.ApplicationShortcut)
            sc_invert.setEnabled(False)
            sc_invert.activated.connect(self._on_invert_segment)
            self._sc_invert = sc_invert
        if callable(self._on_merge_identical):
            sc_merge = QShortcut(QKeySequence("Ctrl+M"), self)
            sc_merge.setContext(Qt.ApplicationShortcut)
            sc_merge.setEnabled(False)
            sc_merge.activated.connect(self._on_merge_identical)
            self._sc_merge = sc_merge
        self._refresh_scope_buttons()
        self._update_table_drag_enabled()
        self.apply_shortcut_settings(self._shortcut_bindings)
        QTimer.singleShot(0, self._init_splitter_sizes)

    def _init_splitter_sizes(self):
        total = max(1, int(self.splitter.width() or self.width() or 0))
        if total <= 1:
            return
        side_target = max(240, min(380, int(total * 0.27)))
        side_target = min(side_target, max(220, total - 180))
        side_min = max(180, min(side_target, int(total * 0.18)))
        side_max = max(side_target, int(total * 0.38))
        side_max = min(side_max, max(side_target, total - 140))
        sizes = self.splitter.sizes() if hasattr(self.splitter, "sizes") else []
        reset = not self._splitter_initialized or len(sizes) < 2 or sum(sizes) <= 0
        if not reset:
            try:
                side_now = int(sizes[1])
            except Exception:
                side_now = side_target
            if side_now < side_min or side_now > side_max:
                reset = True
        if reset:
            self.splitter.setSizes([max(1, total - side_target), side_target])
        self._splitter_initialized = True

    def _set_shortcut_key(self, shortcut, sid: str, default_key: str) -> None:
        key = shortcut_value(
            self._shortcut_bindings,
            self._shortcut_defaults,
            sid,
            default_key,
        )
        set_shortcut_key(shortcut, key, default_key)

    def _normalize_model_type(self, value: str) -> str:
        return normalize_psr_model_type(value, self._model_specs, allow_unknown=True)

    def _set_rules_visible(self, on: bool) -> None:
        self._set_collapsible_section_visible(
            self.btn_toggle_rules, self.rules_body, bool(on)
        )

    def _set_advanced_visible(self, on: bool) -> None:
        self._set_collapsible_section_visible(
            self.btn_toggle_advanced, self.advanced_body, bool(on)
        )

    def _on_model_button_clicked(self, button) -> None:
        if button is None:
            return
        model = button.property("model_type") or self._model_type
        self._model_type = self._normalize_model_type(model)
        self._refresh_model_selector()
        if callable(self._on_model_type_changed):
            self._on_model_type_changed(self._model_type)

    def _on_model_combo_changed(self, _index: int) -> None:
        model = self.combo_model_type.currentData()
        if model in (None, ""):
            return
        model = self._normalize_model_type(model)
        if model == self._model_type:
            return
        self._model_type = model
        self._refresh_model_selector()
        if callable(self._on_model_type_changed):
            self._on_model_type_changed(self._model_type)

    def model_type(self) -> str:
        return str(self._model_type)

    def set_available_models(self, models, emit: bool = False) -> None:
        current = str(self._model_type or "").strip()
        self._model_specs = self._normalize_model_specs(models)
        self._model_type = self._normalize_model_type(current)
        self._refresh_model_selector()
        if emit and callable(self._on_model_type_changed):
            self._on_model_type_changed(self._model_type)

    def set_model_type(self, value: str, emit: bool = False) -> None:
        model = self._normalize_model_type(value)
        self._model_type = model
        self._refresh_model_selector()
        if emit and callable(self._on_model_type_changed):
            self._on_model_type_changed(self._model_type)

    def _iter_shortcuts(self):
        return (
            getattr(self, "_sc_undo", None),
            getattr(self, "_sc_redo", None),
            getattr(self, "_sc_split", None),
            getattr(self, "_sc_scope_segment", None),
            getattr(self, "_sc_scope_from_here", None),
            getattr(self, "_sc_reset", None),
            getattr(self, "_sc_invert", None),
            getattr(self, "_sc_merge", None),
        )

    def _refresh_scope_buttons(self) -> None:
        scope = str(self._segment_scope or "segment")
        is_from_here = scope == "from_here"
        self.btn_scope_segment.setChecked(not is_from_here)
        self.btn_scope_from_here.setChecked(is_from_here)
        self._update_table_drag_enabled()

    def _update_table_drag_enabled(self):
        # Reordering in a filtered subset is ambiguous; allow drag only when
        # full component list is shown.
        allow = bool(self.chk_show_all_components.isChecked()) or str(
            self._segment_scope or "segment"
        ) != "segment"
        try:
            self.table.setDragEnabled(bool(allow))
            self.table.setAcceptDrops(bool(allow))
            self.table.setDragDropMode(
                QAbstractItemView.DragDrop
                if allow
                else QAbstractItemView.NoDragDrop
            )
        except Exception:
            pass

    def _on_scope_segment_clicked(self):
        self._segment_scope = "segment"
        self._refresh_scope_buttons()
        if callable(self._on_select_segment):
            self._on_select_segment()

    def _on_scope_from_here_clicked(self):
        self._segment_scope = "from_here"
        self._refresh_scope_buttons()
        if callable(self._on_select_from_here):
            self._on_select_from_here()

    def _on_split_clicked(self):
        if callable(self._on_split_at_playhead):
            self._on_split_at_playhead()

    def apply_shortcut_settings(self, bindings=None):
        self._shortcut_bindings = (
            load_shortcut_bindings() if bindings is None else dict(bindings)
        )
        self._shortcut_defaults = default_shortcut_bindings()
        self._set_shortcut_key(getattr(self, "_sc_undo", None), "psr.undo", "Ctrl+Z")
        self._set_shortcut_key(getattr(self, "_sc_redo", None), "psr.redo", "Ctrl+Y")
        self._set_shortcut_key(
            getattr(self, "_sc_split", None), "psr.split_at_playhead", "Ctrl+K"
        )
        self._set_shortcut_key(
            getattr(self, "_sc_scope_segment", None),
            "psr.scope_segment",
            "Ctrl+Shift+S",
        )
        self._set_shortcut_key(
            getattr(self, "_sc_scope_from_here", None),
            "psr.scope_from_here",
            "Ctrl+Shift+F",
        )
        self._set_shortcut_key(
            getattr(self, "_sc_reset", None), "psr.reset_segment", "Ctrl+Backspace"
        )
        self._set_shortcut_key(
            getattr(self, "_sc_invert", None), "psr.invert_segment", "Ctrl+I"
        )
        self._set_shortcut_key(
            getattr(self, "_sc_merge", None), "psr.merge_identical", "Ctrl+M"
        )

    def set_body(self, widget):
        self.host.set_body(widget)

    def activate(self):
        if callable(self._on_activate):
            self._on_activate()

    def showEvent(self, event):
        super().showEvent(event)
        self._init_splitter_sizes()
        QTimer.singleShot(0, self._init_splitter_sizes)
        for sc in self._iter_shortcuts():
            if sc is not None:
                try:
                    sc.setEnabled(True)
                except Exception:
                    pass

    def hideEvent(self, event):
        super().hideEvent(event)
        for sc in self._iter_shortcuts():
            if sc is not None:
                try:
                    sc.setEnabled(False)
                except Exception:
                    pass

    def _emit_state_changed(self, component_id, state):
        if self._block_state_signals:
            return
        if callable(self._on_state_changed):
            self._on_state_changed(component_id, state)

    def _state_combo_style(self, state_val: int) -> str:
        # Requested color scheme:
        # Not installed -> green, Error -> yellow, Installed -> red.
        if int(state_val) == 0:
            bg = "#E9F7EF"
            border = "#76B688"
        elif int(state_val) == -1:
            bg = "#FFF4CC"
            border = "#D8B24C"
        else:
            bg = "#FDEAEA"
            border = "#D88989"
        return (
            "QComboBox {"
            f"background-color: {bg};"
            f"border: 1px solid {border};"
            "border-radius: 3px;"
            "padding: 1px 4px;"
            "}"
        )

    def _ordered_component_pairs(self, comps, states):
        pairs = []
        for idx, comp in enumerate(comps or []):
            val = 0
            if idx < len(states or []):
                try:
                    val = int(states[idx])
                except Exception:
                    val = 0
            cid = comp.get("id", idx)
            pairs.append((str(cid), comp, val))
        if not pairs:
            return []
        if not self._ui_component_order:
            self._ui_component_order = [cid for cid, _comp, _val in pairs]
            return pairs
        rank = {cid: i for i, cid in enumerate(self._ui_component_order)}
        return sorted(
            pairs, key=lambda item: (rank.get(item[0], len(rank)), str(item[0]))
        )

    def _normalize_ui_component_order(self, incoming_ids):
        ids = [str(cid) for cid in (incoming_ids or [])]
        incoming_set = set(ids)
        seen = set()
        keep = []
        for cid in (self._ui_component_order or []):
            if cid in incoming_set and cid not in seen:
                keep.append(cid)
                seen.add(cid)
        extras = []
        for cid in ids:
            if cid not in seen:
                extras.append(cid)
                seen.add(cid)
        self._ui_component_order = keep + extras

    def _rebalance_ui_order(self, visible_ids):
        all_ids = [
            str(comp.get("id", idx))
            for idx, comp in enumerate(self._all_components or [])
        ]
        known = set(all_ids)
        seen = set()
        ordered_visible = []
        for cid in (visible_ids or []):
            scid = str(cid)
            if scid not in known or scid in seen:
                continue
            ordered_visible.append(scid)
            seen.add(scid)
        base = [
            str(cid)
            for cid in (
                self._ui_component_order
                if self._ui_component_order
                else all_ids
            )
        ]
        tail = [cid for cid in base if cid in known and cid not in seen]
        self._ui_component_order = ordered_visible + tail
        self._normalize_ui_component_order(all_ids)

    def _visible_component_rows(self):
        comps = list(self._all_components or [])
        states = list(self._all_states or [])
        pairs = self._ordered_component_pairs(comps, states)
        if self.chk_show_all_components.isChecked() or self._segment_scope != "segment":
            return [comp for _cid, comp, _val in pairs], [val for _cid, _comp, val in pairs]
        delta_set = {str(cid) for cid in (self._delta_component_ids or [])}
        if not delta_set:
            return [], []
        vis_comps = []
        vis_states = []
        for cid, comp, val in pairs:
            if str(cid) not in delta_set:
                continue
            vis_comps.append(comp)
            vis_states.append(val)
        return vis_comps, vis_states

    def _refresh_table_hint(self, visible_count: int) -> None:
        if self._segment_scope == "segment" and not self.chk_show_all_components.isChecked():
            if visible_count > 0:
                self.lbl_table_hint.setText(
                    f"Delta components ({visible_count})  |  changed vs previous segment"
                )
            else:
                self.lbl_table_hint.setText(
                    "Delta components: none  |  enable 'Show all components' to force edits"
                )
            self.lbl_table_hint.setVisible(True)
            return
        self.lbl_table_hint.setText("")
        self.lbl_table_hint.setVisible(False)

    def _render_component_table(self):
        try:
            if self.table.is_drag_active():
                return
        except Exception:
            pass
        self._block_state_signals = True
        comps, states = self._visible_component_rows()
        snapshot = [(c.get("id", idx), c.get("name", "")) for idx, c in enumerate(comps)]
        rebuild = snapshot != self._component_snapshot
        if rebuild:
            self._block_component_updates = True
            self.table.blockSignals(True)
            self._component_snapshot = snapshot
            self._state_combos = []
            self.table.setRowCount(len(comps))
            for row, comp in enumerate(comps):
                cid = comp.get("id", row)
                name = comp.get("name", "")
                item_id = QTableWidgetItem(str(cid))
                item_name = QTableWidgetItem(str(name))
                item_id.setFlags(item_id.flags() & ~Qt.ItemIsEditable)
                item_name.setFlags(item_name.flags() & ~Qt.ItemIsEditable)
                item_id.setTextAlignment(Qt.AlignCenter)
                item_id.setData(Qt.UserRole, cid)
                item_name.setData(Qt.UserRole, cid)
                self.table.setItem(row, 0, item_id)
                self.table.setItem(row, 1, item_name)
                combo = QComboBox(self.table)
                combo.blockSignals(True)
                for text, val in self._state_choices:
                    combo.addItem(text, val)
                combo.currentIndexChanged.connect(
                    lambda _idx, cid=cid, cb=combo: self._emit_state_changed(
                        cid, cb.currentData()
                    )
                )
                self.table.setCellWidget(row, 2, combo)
                self._state_combos.append(combo)
            self.table.blockSignals(False)
            self._block_component_updates = False

        for row, _comp in enumerate(comps):
            val = 0
            if states and row < len(states):
                try:
                    val = int(states[row])
                except Exception:
                    val = 0
            if row < len(self._state_combos):
                combo = self._state_combos[row]
                idx = combo.findData(val)
                if idx < 0:
                    idx = combo.findData(0)
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.setStyleSheet(self._state_combo_style(val))
                combo.blockSignals(False)
        self._refresh_table_hint(len(comps))
        self._block_state_signals = False

    def update_component_states(
        self,
        components,
        states,
        rules_count=0,
        source="",
        diagnostics=None,
        delta_component_ids=None,
        segment_scope="segment",
        state_summary="",
    ):
        self._all_components = list(components or [])
        self._all_states = list(states or [])
        self._delta_component_ids = list(delta_component_ids or [])
        incoming_ids = [
            str(comp.get("id", idx))
            for idx, comp in enumerate(self._all_components or [])
        ]
        self._normalize_ui_component_order(incoming_ids)
        self._segment_scope = str(segment_scope or "segment")
        self._refresh_scope_buttons()
        self.lbl_state_summary.setText(str(state_summary or ""))
        try:
            if self.table.is_drag_active():
                return
            else:
                self._render_component_table()
        except Exception:
            self._render_component_table()
        src = f" ({source})" if source else ""
        events = 0
        unmapped = 0
        mismatch = 0
        flow = ""
        initial_state = None
        if diagnostics:
            events = diagnostics.get("events", 0)
            unmapped = diagnostics.get("unmapped", 0)
            mismatch = diagnostics.get("rule_mismatch", 0)
            flow = str(diagnostics.get("flow", "") or "")
            initial_state = diagnostics.get("initial_state")
        init_txt = ""
        try:
            iv = int(initial_state)
            if iv == 1:
                init_txt = "Installed"
            elif iv == 0:
                init_txt = "Not installed"
            elif iv == -1:
                init_txt = "Error"
        except Exception:
            init_txt = ""
        flow_txt = ""
        if flow:
            flow_txt = f" ({flow})"
        init_meta = ""
        if init_txt:
            init_meta = f" | Init: {init_txt}{flow_txt}"
        self.lbl_meta.setText(
            f"Components: {len(self._all_components)}{src} | Rules: {rules_count} | Transitions: {events}{init_meta}"
        )
        if unmapped or mismatch:
            self.lbl_diag.setText(
                f"Unmapped labels: {unmapped} | Rule mismatch: {mismatch}"
            )
        else:
            self.lbl_diag.setText("")

    def _on_components_reordered(self, source_rows, dest_row):
        if self._block_component_updates:
            return
        # Preferred path: already-reordered visible ids from table drop.
        if source_rows and isinstance(source_rows[0], str):
            new_visible = [str(x) for x in source_rows if str(x)]
            if not new_visible:
                return
            self._rebalance_ui_order(new_visible)
            self._component_snapshot = []
            self._render_component_table()
            return
        try:
            dest_row = int(dest_row)
        except Exception:
            return
        source_rows = sorted({int(r) for r in (source_rows or []) if r is not None})
        if not source_rows:
            return
        current = [str(cid) for cid, _name in (self._component_snapshot or [])]
        if not current:
            pairs = self._ordered_component_pairs(
                self._all_components or [], self._all_states or []
            )
            current = [str(cid) for cid, _comp, _val in pairs]
        if not current:
            return
        row_count = len(current)
        source_rows = [r for r in source_rows if 0 <= r < row_count]
        if not source_rows:
            return
        if dest_row < 0:
            dest_row = 0
        if dest_row > row_count:
            dest_row = row_count
        moved = [current[r] for r in source_rows]
        source_set = set(source_rows)
        remaining = [cid for idx, cid in enumerate(current) if idx not in source_set]
        dest_row -= sum(1 for r in source_rows if r < dest_row)
        if dest_row < 0:
            dest_row = 0
        if dest_row > len(remaining):
            dest_row = len(remaining)
        new_visible = remaining[:dest_row] + moved + remaining[dest_row:]
        if new_visible == current:
            return
        self._rebalance_ui_order(new_visible)
        self._render_component_table()

    def _on_components_drop_finished(self):
        # Always re-sync table from canonical component/state data after a drag.
        # This prevents Qt internal-move artifacts (e.g., a row disappearing)
        # from persisting in the UI.
        all_ids = [
            str(comp.get("id", idx))
            for idx, comp in enumerate(self._all_components or [])
        ]
        self._normalize_ui_component_order(all_ids)
        self._component_snapshot = []
        self._render_component_table()

    def handle_state_segment_split(self, host, frame, row) -> bool:
        if host is None or not callable(getattr(host, "_is_psr_task", None)):
            return False
        if not host._is_psr_task():
            return False
        try:
            frame = int(frame)
        except Exception:
            return False
        if getattr(host, "_psr_snap_to_action_segments", False):
            starts = []
            ends = []
            snapper = getattr(host, "_psr_action_segment_starts_for_snap", None)
            if callable(snapper):
                starts = snapper() or []
            ends = list(getattr(host, "_psr_action_segment_ends", []) or [])
            candidates = []
            try:
                fc = max(1, host._get_frame_count())
            except Exception:
                fc = None
            for s in starts:
                try:
                    candidates.append(int(s))
                except Exception:
                    continue
            for e in ends:
                try:
                    cand = int(e) + 1
                except Exception:
                    continue
                if fc is None or (0 <= cand <= fc - 1):
                    candidates.append(cand)
            if candidates:
                frame = min(candidates, key=lambda v: (abs(v - frame), v))
        comp_id = host._psr_component_id_from_row(row)
        added_boundary = False
        refreshed = False
        for ev in host._psr_manual_events:
            try:
                if int(ev.get("frame", -1)) != frame:
                    continue
            except Exception:
                continue
            if (
                ev.get("force_boundary")
                or str(ev.get("label")) == host._psr_boundary_label
            ):
                added_boundary = True
                break
        if not added_boundary:
            host._psr_push_undo("segment_split")
            host._psr_manual_events.append(
                {
                    "frame": int(frame),
                    "component_id": None,
                    "state": None,
                    "label": host._psr_boundary_label,
                    "force_boundary": True,
                }
            )
            host._psr_mark_dirty()
            host._psr_refresh_state_timeline(force=True)
            added_boundary = True
            refreshed = True
        label = None
        seg_end = frame
        row_ref = None
        if not refreshed:
            row_ref = row
        timeline = getattr(host, "timeline", None)
        if row_ref is None and timeline is not None:
            row_ref = getattr(timeline, "_active_combined_row", None)
        if row_ref is None and timeline is not None:
            rows = getattr(timeline, "rows", []) or []
            if rows:
                row_ref = rows[0]
        if row_ref is not None:
            try:
                label = row_ref._label_at(frame)
            except Exception:
                label = None
            try:
                _s, _e, _lb = row_ref._segment_at(frame)
                seg_end = int(_e)
            except Exception:
                seg_end = frame
        host._psr_selected_segment = {
            "start": frame,
            "end": seg_end,
            "label": label,
            "component_id": comp_id,
            "scope": "segment",
            "manual_split": True,
        }
        if refreshed and hasattr(host, "_psr_record_validation_entry"):
            host._psr_record_validation_entry(
                "split",
                comp_id,
                frame,
                frame,
                str(label) if label is not None else "Unlabeled",
                str(label) if label is not None else "Unlabeled",
                note="boundary split",
            )
        if row_ref is not None:
            try:
                row_ref._selected_interval = (frame, seg_end)
                row_ref._selected_label = label
                row_ref._selection_scope = "segment"
                row_ref.update()
            except Exception:
                pass
        if getattr(host, "views", None):
            try:
                host._sync_views_to_frame(frame, preview_only=False)
            except Exception:
                pass
        if timeline is not None:
            try:
                timeline.set_current_frame(frame, follow=True)
            except Exception:
                pass
        host._psr_update_component_panel(frame)
        host._log("psr_state_split", frame=frame, component_id=comp_id)
        return True
