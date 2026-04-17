from typing import List, Callable, Optional, Dict, Tuple
import re
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QComboBox,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QColorDialog,
    QSpinBox,
    QStyledItemDelegate,
    QAbstractItemView,
    QLabel,
    QSplitter,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from core.models import LabelDef
from utils.constants import PRESET_COLORS, make_color_icon, color_from_key


class PhraseLabelDelegate(QStyledItemDelegate):
    """
    Inline editor that edits the full phrase label while the list shows only the object part.
    """

    def __init__(self, panel: "LabelPanel"):
        super().__init__(panel)
        self.panel = panel

    def createEditor(self, parent, option, index):
        return QLineEdit(parent)

    def setEditorData(self, editor, index):
        item = self.panel.obj_list.item(index.row())
        name = item.data(Qt.UserRole) if item is not None else ""
        editor.setText(name or "")
        editor.selectAll()

    def setModelData(self, editor, model, index):
        item = self.panel.obj_list.item(index.row())
        old_name = item.data(Qt.UserRole) if item is not None else ""
        new_name = editor.text().strip()
        if not new_name or new_name == old_name:
            return
        if any(lb.name == new_name for lb in self.panel.labels if lb.name != old_name):
            QMessageBox.information(
                self.panel, "Info", f"Label name '{new_name}' already exists."
            )
            return
        self.panel._rename_label_by_name(old_name, new_name)


class LabelPanel(QWidget):
    def __init__(
        self,
        labels: List[LabelDef],
        on_add: Callable[[LabelDef], None],
        on_remove_idx: Callable[[int], None],
        on_rename: Callable[[str, str], None],
        on_search_matches: Callable[[List[str]], None] = None,
        on_select_idx: Callable[[int], None] = None,
        parent=None,
        manage_storage: bool = True,
    ):
        super().__init__(parent)
        self.labels = labels
        self.on_add = on_add
        self.on_remove_idx = on_remove_idx
        self.on_rename = on_rename
        self.on_search_matches = on_search_matches
        self.on_select_idx = on_select_idx
        self.manage_storage = manage_storage

        self._candidate_order: List[str] = []
        self._candidate_conf: Dict[str, Optional[float]] = {}
        self._candidate_only: bool = False
        self._other_token = "__OTHER_LABEL__"
        self._empty_object_label = "(none)"
        self._last_matches: List[str] = []

        self._verb_items: Dict[str, List[dict]] = {}
        self._verb_order: List[str] = []
        self._visible_verbs: List[str] = []
        self._visible_labels: List[str] = []
        self._selected_verb: Optional[str] = None
        self._selected_label_name: Optional[str] = None
        self._item_fg = QColor(32, 32, 32)

        root = QVBoxLayout(self)

        search_row = QHBoxLayout()
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("Search labels")
        search_row.addWidget(self.search_edit)
        root.addLayout(search_row)

        row = QHBoxLayout()
        self.edit = QLineEdit(self)
        self.edit.setPlaceholderText("New label name")
        self.id_spin = QSpinBox(self)
        self.id_spin.setMinimum(0)
        self.id_spin.setMaximum(10**6)
        self.id_spin.setPrefix("id:")
        self.combo = QComboBox(self)
        for name, qcol in PRESET_COLORS.items():
            self.combo.addItem(name)
            idx = self.combo.count() - 1
            self.combo.setItemIcon(idx, make_color_icon(qcol))
        self.combo.addItem("Custom")

        self.btn_add = QPushButton("Add")
        row.addWidget(self.edit, 2)
        row.addWidget(self.id_spin, 0)
        row.addWidget(self.combo, 1)
        row.addWidget(self.btn_add, 0)
        root.addLayout(row)

        self._verb_only = False
        self._no_split_tokens = {"null", "error", "finish"}
        # Keep multi-token verbs grouped in the Verb list.
        self._compound_verbs = {
            "pick_up",
            "hand_tighten",
            "hand_loosen",
            "hand_spin",
        }
        self.split = QSplitter(Qt.Horizontal, self)
        self.verb_col = QWidget(self)
        verb_layout = QVBoxLayout(self.verb_col)
        verb_layout.setContentsMargins(0, 0, 0, 0)
        verb_layout.setSpacing(4)
        verb_layout.addWidget(QLabel("Verb"))
        self.verb_list = QListWidget(self)
        self.verb_list.setSelectionMode(QAbstractItemView.SingleSelection)
        verb_layout.addWidget(self.verb_list, 1)

        self.obj_col = QWidget(self)
        obj_layout = QVBoxLayout(self.obj_col)
        obj_layout.setContentsMargins(0, 0, 0, 0)
        obj_layout.setSpacing(4)
        obj_layout.addWidget(QLabel("Object"))
        self.obj_list = QListWidget(self)
        self.obj_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.obj_list.setItemDelegate(PhraseLabelDelegate(self))
        self.obj_list.setEditTriggers(QAbstractItemView.DoubleClicked)
        obj_layout.addWidget(self.obj_list, 1)

        self.split.addWidget(self.verb_col)
        self.split.addWidget(self.obj_col)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 2)
        root.addWidget(self.split, 1)

        self.btn_del = QPushButton("Remove Selected")
        root.addWidget(self.btn_del)

        self.btn_add.clicked.connect(self._add)
        self.btn_del.clicked.connect(self._del)
        self.combo.activated[int].connect(self._maybe_pick_custom)
        self.search_edit.textChanged.connect(self._on_search_text_changed)
        self.verb_list.currentRowChanged.connect(self._on_verb_selected)
        self.obj_list.currentRowChanged.connect(self._on_object_selected)

        self.refresh()

    def set_compound_verbs(self, verbs: List[str], refresh: bool = True):
        cleaned = set()
        for raw in verbs or []:
            name = str(raw or "").strip()
            if "_" in name:
                cleaned.add(name)
        if not cleaned:
            cleaned = {
                "pick_up",
                "hand_tighten",
                "hand_loosen",
                "hand_spin",
            }
        if cleaned == self._compound_verbs:
            return
        self._compound_verbs = set(cleaned)
        if refresh:
            self.refresh()

    def set_verb_only(self, on: bool):
        """Toggle verb-only mode by hiding the object column."""
        self._verb_only = bool(on)
        if self._verb_only:
            self.obj_col.hide()
            self.split.setSizes([1, 0])
        else:
            self.obj_col.show()
            self.split.setSizes([1, 2])

    def _is_no_split_label(self, text: str) -> bool:
        if not text:
            return False
        norm = re.sub(r"[^0-9a-z]+", " ", str(text).lower()).strip()
        if not norm:
            return False
        first = norm.split(None, 1)[0]
        return first in self._no_split_tokens

    def _split_label(self, name: str) -> Tuple[str, str]:
        raw = (name or "").strip()
        if not raw:
            return "", ""
        # Preserve underscore-based compound verbs such as pick_up / hand_tighten.
        raw = re.sub(r"^\s*\d+\s*[:\-_.)]*\s*", "", raw).strip()
        if "_" in raw:
            parts = [p.strip() for p in raw.split("_") if p.strip()]
            if parts:
                if len(parts) >= 2:
                    compound = f"{parts[0]}_{parts[1]}"
                    if compound in self._compound_verbs:
                        verb = compound
                        obj = "_".join(parts[2:]).strip()
                        return verb, obj
                return parts[0], "_".join(parts[1:]).strip()
        text = raw
        # Strip leading numeric id and normalize separators for display.
        text = text.replace("_", " ")
        text = re.sub(r"(?<!\s)\(", " (", text)
        text = " ".join(text.split())
        if not text:
            return "", ""
        if self._is_no_split_label(text):
            return text, ""
        parts = text.split(None, 1)
        verb = parts[0]
        obj = parts[1] if len(parts) > 1 else ""
        return verb, obj

    def _label_names_for_display(self) -> List[str]:
        if self._candidate_only and self._candidate_order:
            name_set = {lb.name for lb in self.labels}
            names = [name for name in self._candidate_order if name in name_set]
            if names:
                return names
        return [lb.name for lb in self.labels]

    def _notify_selection(self):
        if not callable(self.on_select_idx):
            return
        if self._selected_label_name:
            idx = self.index_of_label(self._selected_label_name)
        else:
            idx = -1
        self.on_select_idx(idx)

    def index_of_label(self, name: str) -> int:
        for i, lb in enumerate(self.labels):
            if lb.name == name:
                return i
        return -1

    def current_label_name(self) -> str:
        return self._selected_label_name or ""

    def select_label_by_name(self, name: str) -> bool:
        if not name:
            return False
        if self.index_of_label(name) < 0:
            return False
        verb, _ = self._split_label(name)
        self._selected_verb = verb
        self._selected_label_name = name
        self.refresh()
        self._notify_selection()
        return True

    def refresh(self):
        name_to_label = {lb.name: lb for lb in self.labels}
        label_names = self._label_names_for_display()
        search_text = self.search_edit.text().strip().lower()
        verb_items: Dict[str, List[dict]] = {}
        verb_order: List[str] = []
        matches: List[str] = []

        for name in label_names:
            lb = name_to_label.get(name)
            if lb is None:
                continue
            verb, obj = self._split_label(name)
            if not verb:
                continue
            if not obj and self._is_no_split_label(name):
                obj_display = verb
            else:
                obj_display = obj if obj else self._empty_object_label
            match = (
                True
                if not search_text
                else (
                    search_text in name.lower()
                    or search_text in verb.lower()
                    or search_text in obj.lower()
                )
            )
            if match:
                matches.append(name)
            item = {
                "label": name,
                "verb": verb,
                "object": obj,
                "object_display": obj_display,
                "id": lb.id,
                "color_name": lb.color_name,
                "conf": self._candidate_conf.get(name),
                "candidate": name in self._candidate_conf,
                "match": match,
            }
            if verb not in verb_items:
                verb_items[verb] = []
                verb_order.append(verb)
            verb_items[verb].append(item)

        self._verb_items = verb_items
        self._verb_order = verb_order

        if matches != self._last_matches:
            self._last_matches = matches
            if callable(self.on_search_matches):
                self.on_search_matches(matches)

        visible_verbs: List[str] = []
        verb_match: Dict[str, bool] = {}
        for verb in verb_order:
            items = verb_items.get(verb, [])
            has_match = any(it["match"] for it in items) if search_text else True
            verb_match[verb] = has_match
            if has_match:
                visible_verbs.append(verb)
        self._visible_verbs = visible_verbs

        if self._selected_label_name and self._selected_label_name not in name_to_label:
            self._selected_label_name = None
        if self._selected_label_name:
            self._selected_verb, _ = self._split_label(self._selected_label_name)
        if not self._selected_label_name and self._selected_verb not in visible_verbs:
            self._selected_verb = visible_verbs[0] if visible_verbs else None

        self.verb_list.blockSignals(True)
        self.verb_list.clear()
        for verb in visible_verbs:
            item = QListWidgetItem(verb)
            item.setData(Qt.UserRole, verb)
            base_bg = (
                QColor(240, 249, 255) if self._candidate_only else QColor(Qt.white)
            )
            if search_text and verb_match.get(verb):
                item.setBackground(QColor(255, 248, 196))
            else:
                item.setBackground(base_bg)
            item.setForeground(self._item_fg)
            if self._candidate_only:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.verb_list.addItem(item)
        if self._candidate_only and visible_verbs:
            other = QListWidgetItem("Other label...")
            other.setData(Qt.UserRole, self._other_token)
            font = other.font()
            font.setItalic(True)
            other.setFont(font)
            other.setForeground(self._item_fg)
            self.verb_list.addItem(other)

        if self._selected_verb and self._selected_verb in visible_verbs:
            self.verb_list.setCurrentRow(visible_verbs.index(self._selected_verb))
        else:
            self.verb_list.setCurrentRow(-1)
        self.verb_list.blockSignals(False)

        self._refresh_object_list()

    def _refresh_object_list(self):
        search_text = self.search_edit.text().strip().lower()
        self.obj_list.blockSignals(True)
        self.obj_list.clear()
        self._visible_labels = []

        if not self._selected_verb:
            self.obj_list.blockSignals(False)
            return
        items = self._verb_items.get(self._selected_verb, [])
        for item in items:
            if search_text and not item["match"]:
                continue
            label_name = item["label"]
            conf = item["conf"]
            suffix = f" ({conf:.2f})" if conf is not None else ""
            text = f"{item['object_display']}{suffix}  [id={item['id']}]"
            it = QListWidgetItem(text)
            it.setData(Qt.UserRole, label_name)
            it.setToolTip(label_name)
            it.setIcon(make_color_icon(color_from_key(item["color_name"])))
            it.setFlags(it.flags() | Qt.ItemIsEditable)
            base_bg = QColor(240, 249, 255) if item["candidate"] else QColor(Qt.white)
            if search_text:
                it.setBackground(QColor(255, 248, 196))
            else:
                it.setBackground(base_bg)
            it.setForeground(self._item_fg)
            self.obj_list.addItem(it)
            self._visible_labels.append(label_name)

        if self._selected_label_name in self._visible_labels:
            self.obj_list.setCurrentRow(
                self._visible_labels.index(self._selected_label_name)
            )
        else:
            self.obj_list.setCurrentRow(-1)
        self.obj_list.blockSignals(False)

    def _rename_label_by_name(self, old_name: str, new_name: str):
        idx = self.index_of_label(old_name)
        if idx < 0:
            return
        self.labels[idx].name = new_name
        if callable(self.on_rename):
            self.on_rename(old_name, new_name)
        self._selected_label_name = (
            new_name
            if self._selected_label_name == old_name
            else self._selected_label_name
        )
        if self._selected_label_name:
            self._selected_verb, _ = self._split_label(self._selected_label_name)
        self.refresh()

    def _maybe_pick_custom(self, idx: int):
        if self.combo.itemText(idx).startswith("Custom"):
            col = QColorDialog.getColor(parent=self, title="Pick a color")
            if col.isValid():
                key = f"custom:{col.name()}"
                self.combo.insertItem(idx, key)
                self.combo.setItemIcon(idx, make_color_icon(col))
                self.combo.setCurrentIndex(idx)

    def _add(self):
        name = self.edit.text().strip()
        if not name:
            QMessageBox.information(self, "Info", "Please input label name.")
            return
        color_key = self.combo.currentText()
        if color_key == "Custom":
            QMessageBox.information(
                self, "Info", "Please pick a color or choose a preset."
            )
            return
        lbid = self.id_spin.value()
        if any(x.id == lbid for x in self.labels):
            QMessageBox.information(self, "Info", f"Label id {lbid} already exists.")
            return
        if any(x.name == name for x in self.labels):
            QMessageBox.information(
                self, "Info", f"Label name '{name}' already exists."
            )
            return
        lb = LabelDef(name=name, color_name=color_key, id=lbid)
        if self.manage_storage:
            self.labels.append(lb)
        self.on_add(lb)
        self.edit.clear()
        self.refresh()

    def _del(self):
        if not self._selected_label_name:
            QMessageBox.information(self, "Info", "Please select a label to remove.")
            return
        idx = self.index_of_label(self._selected_label_name)
        if idx < 0:
            return
        self.on_remove_idx(idx)
        if self.manage_storage:
            self.labels.pop(idx)
        self._selected_label_name = None
        self.refresh()
        self._notify_selection()

    def _on_search_text_changed(self):
        self.refresh()

    def _on_verb_selected(self, row: int):
        item = self.verb_list.item(row) if row >= 0 else None
        if item is None:
            self._selected_verb = None
            self._selected_label_name = None
            self._refresh_object_list()
            self._notify_selection()
            return
        verb = item.data(Qt.UserRole) or ""
        if verb == self._other_token:
            self._selected_verb = None
            self._selected_label_name = None
            self.clear_candidate_priority()
            self._notify_selection()
            return
        prev_label = self._selected_label_name
        self._selected_verb = verb
        if self._selected_label_name:
            sel_verb, _ = self._split_label(self._selected_label_name)
            if sel_verb != verb:
                self._selected_label_name = None
        self._refresh_object_list()
        if prev_label and self._selected_label_name is None:
            self._notify_selection()

    def _on_object_selected(self, row: int):
        item = self.obj_list.item(row) if row >= 0 else None
        if item is None:
            self._selected_label_name = None
            self._notify_selection()
            return
        name = item.data(Qt.UserRole) or ""
        self._selected_label_name = name if name else None
        self._notify_selection()

    def set_candidate_priority(self, candidates: List[tuple]):
        if not candidates:
            self.clear_candidate_priority()
            return
        names = [c[0] for c in candidates if c and c[0]]
        self._candidate_order = names
        self._candidate_conf = {name: conf for name, conf in candidates if name}
        self._candidate_only = True
        self._selected_label_name = None
        self._selected_verb = None
        self.refresh()

    def clear_candidate_priority(self):
        self._candidate_order = []
        self._candidate_conf = {}
        self._candidate_only = False
        self.refresh()
