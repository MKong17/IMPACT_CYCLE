"""Ontology (Entity & Relation) editor UI component."""

import json
import os
from typing import Dict, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class OntologyEditor(QWidget):
    """Edit entities and relations in the ontology.
    
    Signals:
        ontology_changed: Emitted when ontology is modified, with signature (ontology_dict)
    """

    ontology_changed = pyqtSignal(dict)

    def __init__(self, ontology_path: str, parent=None):
        """
        Args:
            ontology_path: Path to impact_sg_ontology.json
            parent: Parent widget
        """
        super().__init__(parent)
        self._ontology_path = ontology_path
        self._ontology: Dict = {}
        self._load_ontology()
        self._build_ui()

    def _load_ontology(self) -> None:
        """Load ontology from JSON file."""
        if not os.path.isfile(self._ontology_path):
            # 创建默认 ontology
            self._ontology = {
                "canonical_entities": [],
                "relation_vocabulary": {
                    "spatial": [],
                    "interaction": [],
                },
                "question_types": [
                    "existence",
                    "counting",
                    "attribute_query",
                    "spatial_relation",
                    "interaction_relation",
                ],
            }
            return

        try:
            with open(self._ontology_path, "r", encoding="utf-8") as f:
                self._ontology = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load ontology: {e}")

    def get_ontology(self) -> Dict:
        """Return current ontology dict."""
        return self._ontology

    def _build_ui(self) -> None:
        """Build the editor UI."""
        layout = QVBoxLayout(self)

        tabs = QTabWidget()

        # Entity 编辑标签页
        self._build_entity_tab(tabs)

        # Relation 编辑标签页
        self._build_relation_tab(tabs)

        # 导入导出标签页
        self._build_io_tab(tabs)

        layout.addWidget(tabs)

        # 底部操作按钮
        bottom = QHBoxLayout()
        self.btn_save_to_file = QPushButton("Save Ontology to File")
        self.btn_save_to_file.clicked.connect(self._save_ontology_to_file)
        self.btn_reset = QPushButton("Reset to Default")
        self.btn_reset.clicked.connect(self._reset_ontology)
        bottom.addWidget(self.btn_save_to_file)
        bottom.addWidget(self.btn_reset)
        bottom.addStretch(1)
        layout.addLayout(bottom)

    def _build_entity_tab(self, tabs: QTabWidget) -> None:
        """Entity 编辑标签页。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Entities 表格
        info = QLabel("Manage canonical entities. Each entity can have synonyms and attribute slots.")
        layout.addWidget(info)

        self.entity_table = QTableWidget(0, 4)
        self.entity_table.setHorizontalHeaderLabels(["Label", "Synonyms (comma-separated)", "Mandatory Attrs", ""])
        self.entity_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.entity_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.entity_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.entity_table.horizontalHeader().setStretchLastSection(True)
        self.entity_table.setColumnWidth(0, 180)
        self.entity_table.setColumnWidth(1, 320)
        self.entity_table.setColumnWidth(2, 220)
        layout.addWidget(self.entity_table, 1)

        # 控制按钮
        controls = QHBoxLayout()
        self.btn_add_entity = QPushButton("Add Entity")
        self.btn_add_entity.clicked.connect(self._add_entity)
        self.btn_remove_entity = QPushButton("Remove Selected")
        self.btn_remove_entity.clicked.connect(self._remove_entity)
        controls.addWidget(self.btn_add_entity)
        controls.addWidget(self.btn_remove_entity)
        controls.addStretch(1)
        layout.addLayout(controls)

        # Entity 详情编辑
        detail_group = QGroupBox("Edit Entity Details")
        detail_form = QFormLayout(detail_group)
        self.entity_label_input = QLineEdit()
        self.entity_synonyms_input = QLineEdit()
        self.entity_attrs_input = QLineEdit()
        self.entity_mandatory_input = QLineEdit()
        detail_form.addRow("Label:", self.entity_label_input)
        detail_form.addRow("Synonyms (comma-sep):", self.entity_synonyms_input)
        detail_form.addRow("Attribute Slots (comma-sep):", self.entity_attrs_input)
        detail_form.addRow("Mandatory Attrs (comma-sep):", self.entity_mandatory_input)
        self.btn_update_entity = QPushButton("Update Entity")
        self.btn_update_entity.clicked.connect(self._update_entity)
        self.btn_apply_person_template = QPushButton("Apply Person Attribute Template")
        self.btn_apply_person_template.clicked.connect(self._apply_person_attribute_template)
        detail_form.addRow("", self.btn_update_entity)
        detail_form.addRow("", self.btn_apply_person_template)
        layout.addWidget(detail_group)

        tabs.addTab(panel, "Entities")
        self.entity_table.itemSelectionChanged.connect(self._load_entity_to_editor)
        self._render_entities()

    def _render_entities(self) -> None:
        """Render entities into table."""
        entities = self._ontology.get("canonical_entities", [])
        self.entity_table.setRowCount(len(entities))

        for i, e in enumerate(entities):
            label = str(e.get("label", ""))
            synonyms = ", ".join(e.get("synonyms", []))
            mandatory = ", ".join(e.get("mandatory_attributes", []))

            self.entity_table.setItem(i, 0, QTableWidgetItem(label))
            self.entity_table.setItem(i, 1, QTableWidgetItem(synonyms))
            self.entity_table.setItem(i, 2, QTableWidgetItem(mandatory))

    def _load_entity_to_editor(self) -> None:
        """Load selected entity to detail editor."""
        sel = self.entity_table.selectionModel()
        if not sel or not sel.hasSelection():
            return

        row = sel.selectedRows()[0].row()
        entities = self._ontology.get("canonical_entities", [])
        if row < 0 or row >= len(entities):
            return

        e = entities[row]
        self.entity_label_input.setText(str(e.get("label", "")))
        self.entity_synonyms_input.setText(", ".join(e.get("synonyms", [])))
        self.entity_attrs_input.setText(", ".join(e.get("attribute_slots", [])))
        self.entity_mandatory_input.setText(", ".join(e.get("mandatory_attributes", [])))

    def _add_entity(self) -> None:
        """Add new entity."""
        entities = self._ontology.get("canonical_entities", [])
        new_entity = {
            "label": f"entity_{len(entities)}",
            "synonyms": [],
            "attribute_slots": [],
            "mandatory_attributes": [],
        }
        entities.append(new_entity)
        self._render_entities()
        self.ontology_changed.emit(self._ontology)

    def _remove_entity(self) -> None:
        """Remove selected entity."""
        sel = self.entity_table.selectionModel()
        if not sel or not sel.hasSelection():
            QMessageBox.information(self, "No Selection", "Select an entity to remove.")
            return

        row = sel.selectedRows()[0].row()
        entities = self._ontology.get("canonical_entities", [])
        if row < 0 or row >= len(entities):
            return

        entities.pop(row)
        self._render_entities()
        self.ontology_changed.emit(self._ontology)

    def _update_entity(self) -> None:
        """Update selected entity with editor values."""
        sel = self.entity_table.selectionModel()
        if not sel or not sel.hasSelection():
            QMessageBox.information(self, "No Selection", "Select an entity to update.")
            return

        row = sel.selectedRows()[0].row()
        entities = self._ontology.get("canonical_entities", [])
        if row < 0 or row >= len(entities):
            return

        label = self.entity_label_input.text().strip()
        if not label:
            QMessageBox.warning(self, "Invalid Input", "Entity label cannot be empty.")
            return

        entities[row]["label"] = label
        entities[row]["synonyms"] = [s.strip() for s in self.entity_synonyms_input.text().split(",") if s.strip()]
        entities[row]["attribute_slots"] = [
            a.strip() for a in self.entity_attrs_input.text().split(",") if a.strip()
        ]
        entities[row]["mandatory_attributes"] = [
            a.strip() for a in self.entity_mandatory_input.text().split(",") if a.strip()
        ]

        self._render_entities()
        self.ontology_changed.emit(self._ontology)

    def _build_relation_tab(self, tabs: QTabWidget) -> None:
        """关系 (Relation) 编辑标签页."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        info = QLabel(
            "Manage relations vocabulary. Relations are grouped by type (spatial/interaction).\n"
            "Enable/disable relations to customize the scene graph generation."
        )
        layout.addWidget(info)

        splitter = QSplitter(Qt.Horizontal)

        # 关系类型列表
        type_group = QGroupBox("Relation Types")
        type_layout = QVBoxLayout(type_group)
        self.relation_type_list = QListWidget()
        self.relation_type_list.itemSelectionChanged.connect(self._load_relation_type)
        type_layout.addWidget(self.relation_type_list)
        splitter.addWidget(type_group)

        # 关系列表和编辑
        rel_group = QGroupBox("Relations in Selected Type")
        rel_layout = QVBoxLayout(rel_group)
        self.relation_table = QTableWidget(0, 2)
        self.relation_table.setHorizontalHeaderLabels(["Relation Name", "Enabled"])
        self.relation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.relation_table.horizontalHeader().setStretchLastSection(True)
        self.relation_table.setColumnWidth(0, 260)
        self.relation_table.setColumnWidth(1, 120)
        rel_layout.addWidget(self.relation_table)

        controls = QHBoxLayout()
        self.btn_add_relation = QPushButton("Add Relation")
        self.btn_add_relation.clicked.connect(self._add_relation)
        self.btn_remove_relation = QPushButton("Remove Selected")
        self.btn_remove_relation.clicked.connect(self._remove_relation)
        controls.addWidget(self.btn_add_relation)
        controls.addWidget(self.btn_remove_relation)
        controls.addStretch(1)
        rel_layout.addLayout(controls)

        rel_edit_group = QGroupBox("Add New Relation")
        rel_edit_form = QFormLayout(rel_edit_group)
        self.relation_name_input = QLineEdit()
        rel_edit_form.addRow("Relation Name:", self.relation_name_input)
        self.btn_add_new_relation = QPushButton("Add")
        self.btn_add_new_relation.clicked.connect(self._add_new_relation_to_type)
        rel_edit_form.addRow("", self.btn_add_new_relation)
        rel_layout.addWidget(rel_edit_group)

        splitter.addWidget(rel_group)
        layout.addWidget(splitter)

        tabs.addTab(panel, "Relations")
        self._render_relation_types()

    def _render_relation_types(self) -> None:
        """Render relation types to list."""
        rel_vocab = self._ontology.get("relation_vocabulary", {})
        self.relation_type_list.clear()

        for rel_type in rel_vocab.keys():
            self.relation_type_list.addItem(QListWidgetItem(rel_type))

    def _load_relation_type(self) -> None:
        """Load relations for selected type."""
        sel = self.relation_type_list.selectedItems()
        if not sel:
            self.relation_table.setRowCount(0)
            return

        rel_type = str(sel[0].text())
        rel_vocab = self._ontology.get("relation_vocabulary", {})
        relations = rel_vocab.get(rel_type, [])

        self.relation_table.setRowCount(len(relations))
        for i, rel in enumerate(relations):
            self.relation_table.setItem(i, 0, QTableWidgetItem(str(rel)))
            self.relation_table.setItem(i, 1, QTableWidgetItem("✓"))

        self._current_relation_type = rel_type

    def _add_new_relation_to_type(self) -> None:
        """Add new relation to selected type."""
        if not hasattr(self, "_current_relation_type"):
            QMessageBox.information(self, "No Type", "Select a relation type first.")
            return

        rel_name = self.relation_name_input.text().strip()
        if not rel_name:
            QMessageBox.warning(self, "Invalid Input", "Relation name cannot be empty.")
            return

        rel_vocab = self._ontology.get("relation_vocabulary", {})
        rel_type = self._current_relation_type
        if rel_type not in rel_vocab:
            rel_vocab[rel_type] = []

        if rel_name not in rel_vocab[rel_type]:
            rel_vocab[rel_type].append(rel_name)
            self.relation_name_input.clear()
            self._load_relation_type()
            self.ontology_changed.emit(self._ontology)

    def _add_relation(self) -> None:
        """从下拉列表添加新的关系类型."""
        if not hasattr(self, "_current_relation_type"):
            QMessageBox.information(self, "No Type", "Select a relation type first.")
            return

        text, ok = self._get_text_input("Add Relation Type", "Enter new relation type name:")
        if not ok or not text:
            return

        text = text.strip()
        rel_vocab = self._ontology.get("relation_vocabulary", {})
        if text not in rel_vocab:
            rel_vocab[text] = []
            self._render_relation_types()
            self.ontology_changed.emit(self._ontology)

    def _remove_relation(self) -> None:
        """Remove selected relation from current type."""
        if not hasattr(self, "_current_relation_type"):
            return

        sel = self.relation_table.selectionModel()
        if not sel or not sel.hasSelection():
            QMessageBox.information(self, "No Selection", "Select a relation to remove.")
            return

        row = sel.selectedRows()[0].row()
        rel_vocab = self._ontology.get("relation_vocabulary", {})
        rel_type = self._current_relation_type
        relations = rel_vocab.get(rel_type, [])

        if row < 0 or row >= len(relations):
            return

        relations.pop(row)
        self._load_relation_type()
        self.ontology_changed.emit(self._ontology)

    def _build_io_tab(self, tabs: QTabWidget) -> None:
        """Import/Export 标签页."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        info = QLabel("Import custom ontology from JSON or export current configuration.")
        layout.addWidget(info)

        # 预览区
        preview_group = QGroupBox("Current Ontology JSON")
        preview_form = QFormLayout(preview_group)
        self.ontology_json_preview = QTextEdit()
        self.ontology_json_preview.setReadOnly(True)
        preview_form.addRow(self.ontology_json_preview)
        layout.addWidget(preview_group)

        # 控制按钮
        controls = QHBoxLayout()
        self.btn_import_ontology = QPushButton("Import from JSON")
        self.btn_import_ontology.clicked.connect(self._import_ontology_from_file)
        self.btn_export_ontology = QPushButton("Export to JSON")
        self.btn_export_ontology.clicked.connect(self._export_ontology_to_file)
        self.btn_refresh_preview = QPushButton("Refresh Preview")
        self.btn_refresh_preview.clicked.connect(self._refresh_ontology_preview)
        controls.addWidget(self.btn_import_ontology)
        controls.addWidget(self.btn_export_ontology)
        controls.addWidget(self.btn_refresh_preview)
        controls.addStretch(1)
        layout.addLayout(controls)

        tabs.addTab(panel, "Import/Export")
        self._refresh_ontology_preview()

    def _refresh_ontology_preview(self) -> None:
        """Refresh JSON preview."""
        json_str = json.dumps(self._ontology, ensure_ascii=False, indent=2)
        self.ontology_json_preview.setPlainText(json_str)

    def _import_ontology_from_file(self) -> None:
        """Import ontology from JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Ontology",
            os.path.dirname(self._ontology_path),
            "JSON Files (*.json)",
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                new_ontology = json.load(f)
            self._ontology = new_ontology
            self._render_entities()
            self._render_relation_types()
            self._refresh_ontology_preview()
            self.ontology_changed.emit(self._ontology)
            QMessageBox.information(self, "Success", f"Imported ontology from {path}")
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Failed to import: {e}")

    def _export_ontology_to_file(self) -> None:
        """Export ontology to JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Ontology",
            os.path.dirname(self._ontology_path),
            "JSON Files (*.json)",
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._ontology, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "Success", f"Exported ontology to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export: {e}")

    def _save_ontology_to_file(self) -> None:
        """Save current ontology to original file."""
        try:
            os.makedirs(os.path.dirname(self._ontology_path), exist_ok=True)
            with open(self._ontology_path, "w", encoding="utf-8") as f:
                json.dump(self._ontology, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "Saved", f"Ontology saved to {self._ontology_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Failed to save: {e}")

    def _reset_ontology(self) -> None:
        """Reset to default ontology."""
        reply = QMessageBox.question(
            self,
            "Reset Ontology",
            "Reset to default? This will discard all changes.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._load_ontology()
        self._render_entities()
        self._render_relation_types()
        self._refresh_ontology_preview()
        self.ontology_changed.emit(self._ontology)

    def _get_text_input(self, title: str, prompt: str) -> tuple:
        """Simple text input dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(prompt))
        input_box = QLineEdit()
        layout.addWidget(input_box)
        buttons = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        if dialog.exec_() == QDialog.Accepted:
            return input_box.text(), True
        return "", False

    def _apply_person_attribute_template(self) -> None:
        """Quick-fill common person attributes for demographic/appearance annotations."""
        self.entity_attrs_input.setText("gender, age_group, height_level, body_type, clothing_color, state")
        self.entity_mandatory_input.setText("gender, age_group")
