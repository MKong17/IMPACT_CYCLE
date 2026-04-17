# ui/entities_panel.py
from typing import List, Callable, Optional, Set
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QSpinBox,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QStyledItemDelegate,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt
from core.models import EntityDef


class NameOnlyDelegate(QStyledItemDelegate):
    def __init__(self, panel: "EntitiesPanel"):
        super().__init__(panel)
        self.panel = panel

    def createEditor(self, parent, option, index):
        return QLineEdit(parent)

    def setEditorData(self, editor, index):
        row = index.row()
        if 0 <= row < len(self.panel.entities):
            editor.setText(self.panel.entities[row].name)
            editor.selectAll()

    def setModelData(self, editor, model, index):
        row = index.row()
        if not (0 <= row < len(self.panel.entities)):
            return
        old = self.panel.entities[row].name
        new = editor.text().strip()
        if not new or new == old:
            return
        if any(i != row and e.name == new for i, e in enumerate(self.panel.entities)):
            QMessageBox.information(
                self.panel, "Info", f"Entity name '{new}' already exists."
            )
            return
        self.panel.entities[row].name = new
        item = self.panel.list.item(row)
        item.setText(f"{new}  [id={self.panel.entities[row].id}]")
        if callable(self.panel.on_rename):
            self.panel.on_rename(old, new)


class EntitiesPanel(QWidget):
    """
    Checklist + add/delete + inline rename.
    Check state applies to the current label in applicability mode; in visibility mode,
    check state reflects which entities are visible.
    Callbacks:
      - on_add(EntityDef)
      - on_remove_idx(int)
      - on_rename(old_name: str, new_name: str)
      - on_applicability_changed(label_name: str, selected_entities: List[str])
      - on_visibility_changed(selected_entities: List[str])
    """

    def __init__(
        self,
        entities: List[EntityDef],
        on_add: Callable[[EntityDef], None],
        on_remove_idx: Callable[[int], None],
        on_rename: Callable[[str, str], None],
        on_applicability_changed: Callable[[str, List[str]], None],
        on_visibility_changed: Optional[Callable[[List[str]], None]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.entities = entities
        self.on_add = on_add
        self.on_remove_idx = on_remove_idx
        self.on_rename = on_rename
        self.on_applicability_changed = on_applicability_changed
        self.on_visibility_changed = on_visibility_changed

        self.current_label: Optional[str] = None
        self.mode = "applicability"

        root = QVBoxLayout(self)

        row = QHBoxLayout()
        self.edit = QLineEdit(self)
        self.edit.setPlaceholderText("New entity name")
        self.id_spin = QSpinBox(self)
        self.id_spin.setMinimum(0)
        self.id_spin.setMaximum(10**6)
        self.id_spin.setPrefix("id:")
        self.btn_add = QPushButton("Add")
        row.addWidget(self.edit, 2)
        row.addWidget(self.id_spin, 0)
        row.addWidget(self.btn_add, 0)
        root.addLayout(row)

        self.list = QListWidget(self)
        self.list.setItemDelegate(NameOnlyDelegate(self))
        self.list.setEditTriggers(QAbstractItemView.DoubleClicked)
        self.list.itemChanged.connect(self._item_changed)
        root.addWidget(self.list, 1)

        self.btn_del = QPushButton("Remove Selected")
        self.btn_del.clicked.connect(self._del)
        root.addWidget(self.btn_del)

        self.btn_add.clicked.connect(self._add)

        self.refresh()

    def refresh(self, checked_names: Optional[Set[str]] = None):
        """Rebuild list; checked_names corresponds to current label selection."""
        self.list.blockSignals(True)
        self.list.clear()
        for e in self.entities:
            it = QListWidgetItem(f"{e.name}  [id={e.id}]")
            it.setFlags(
                it.flags()
                | Qt.ItemIsUserCheckable
                | Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
                | Qt.ItemIsEditable
            )
            checked = checked_names is not None and e.name in checked_names
            it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.list.addItem(it)
        self.list.blockSignals(False)

    def set_mode(self, mode: str, checked_names: Optional[Set[str]] = None):
        mode = mode if mode in ("applicability", "visibility") else "applicability"
        self.mode = mode
        self.refresh(checked_names or set())

    def set_current_label(
        self, label_name: Optional[str], checked_names: Optional[Set[str]]
    ):
        """Called when label selection changes; updates current label and checks."""
        self.current_label = label_name
        if self.mode != "applicability":
            return
        self.refresh(checked_names or set())

    def _selected_names(self) -> List[str]:
        names = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.Checked:
                names.append(self.entities[i].name)
        return names

    def _add(self):
        name = self.edit.text().strip()
        if not name:
            QMessageBox.information(self, "Info", "Please input entity name.")
            return
        eid = self.id_spin.value()
        if any(x.id == eid for x in self.entities):
            QMessageBox.information(self, "Info", f"Entity id {eid} already exists.")
            return
        if any(x.name == name for x in self.entities):
            QMessageBox.information(self, "Info", f"Entity '{name}' already exists.")
            return
        self.entities.append(EntityDef(name=name, id=eid))
        self.edit.clear()
        # keep current label check state
        cur_checked = set(self._selected_names())
        self.refresh(cur_checked)
        if callable(self.on_add):
            self.on_add(self.entities[-1])

    def _del(self):
        row = self.list.currentRow()
        if row < 0 or row >= len(self.entities):
            return
        if callable(self.on_remove_idx):
            self.on_remove_idx(row)
        self.entities.pop(row)
        # refresh while keeping current checks
        cur_checked = set(self._selected_names())
        self.refresh(cur_checked)

    def _item_changed(self, _):
        if self.mode == "visibility":
            if callable(self.on_visibility_changed):
                self.on_visibility_changed(self._selected_names())
            return
        if not self.current_label or not callable(self.on_applicability_changed):
            return
        self.on_applicability_changed(self.current_label, self._selected_names())
