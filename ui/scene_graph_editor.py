"""Interactive scene graph editor with image visualization and bbox editing."""

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRect, QSize, QPoint, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QMouseEvent
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QComboBox,
)

# Low-confidence threshold for edge highlighting
EDGE_LOW_CONF_THRESHOLD = 0.5

# Predefined relation vocabulary for edge annotation — spatial relations first
RELATION_VOCABULARY: List[str] = [
    # ── Spatial: coarse direction ──────────────────────────────────────────
    "left of", "right of", "above", "below", "in front of", "behind",
    "beside", "next to", "near", "between", "across from",
    # ── Spatial: fine-grained / relative ──────────────────────────────────
    "to the left of", "to the right of",
    "directly above", "directly below",
    "diagonally above-left", "diagonally above-right",
    "diagonally below-left", "diagonally below-right",
    "facing toward", "facing away from",
    # ── Spatial: containment / contact ────────────────────────────────────
    "on", "in", "at", "under", "over", "on top of",
    "inside", "outside", "surrounding",
    "touching", "overlapping", "adjacent to",
    "attached to", "connected to", "hanging from",
    # ── Spatial: distance ─────────────────────────────────────────────────
    "close to", "far from",
    # ── Action / interaction ───────────────────────────────────────────────
    "holding", "wearing", "carrying", "using",
    "eating", "riding", "watching", "looking at",
    "walking toward", "walking away from",
    # ── Semantic / ownership ──────────────────────────────────────────────
    "part of", "made of", "has", "contains", "belongs to",
    "other",
]


class BBoxDrawWidget(QWidget):
    """Widget for drawing and editing bboxes on image."""

    bbox_changed = pyqtSignal(str, QRect)  # entity_id, rect
    entity_selected = pyqtSignal(str)  # entity_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._image_path = ""
        self._scale_factor = 1.0
        self._bboxes: Dict[str, Dict] = {}  # {entity_id: {"bbox": [x, y, w, h], "label": str, "color": ...}}
        self._selected_entity_id: Optional[str] = None
        self._dragging_corner: Optional[Tuple[str, int]] = None  # (entity_id, corner_idx)
        self._dragging_offset = (0, 0)

        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)

    def load_image(self, image_path: str) -> None:
        """Load image from path."""
        if not os.path.isfile(image_path):
            raise ValueError(f"Image not found: {image_path}")

        self._image_path = image_path
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qt_image)

        # Auto-fit widget size
        self.setMinimumSize(min(w, 800), min(h, 600))
        self.update()

    def add_bbox(self, entity_id: str, label: str, bbox: List[int], color: Tuple[int, int, int] = (0, 255, 0)) -> None:
        """Add bbox to display.

        Args:
            entity_id: Unique entity identifier
            label: Entity label/name
            bbox: [x, y, w, h] bbox coordinates
            color: (R, G, B) color
        """
        self._bboxes[entity_id] = {
            "label": label,
            "bbox": bbox,  # [x, y, w, h]
            "color": color,
        }
        self.update()

    def remove_bbox(self, entity_id: str) -> None:
        """Remove bbox from display."""
        if entity_id in self._bboxes:
            del self._bboxes[entity_id]
            if self._selected_entity_id == entity_id:
                self._selected_entity_id = None
        self.update()

    def bring_to_top(self, entity_id: str) -> None:
        """Bring entity bbox to top rendering layer so it appears on top of others."""
        if entity_id not in self._bboxes:
            return
        # Re-insert at end of ordered dict so it's drawn last (on top)
        data = self._bboxes.pop(entity_id)
        self._bboxes[entity_id] = data
        self.update()

    def select_entity(self, entity_id: str) -> None:
        """Select entity bbox and bring it to top layer."""
        if entity_id in self._bboxes:
            self._selected_entity_id = entity_id
            self.bring_to_top(entity_id)
            self.update()

    def get_selected_entity(self) -> Optional[str]:
        """Get currently selected entity ID."""
        return self._selected_entity_id

    def get_bbox(self, entity_id: str) -> Optional[List[int]]:
        """Get bbox for entity."""
        if entity_id in self._bboxes:
            return self._bboxes[entity_id]["bbox"]
        return None

    def set_bbox(self, entity_id: str, bbox: List[int]) -> None:
        """Update bbox coordinates."""
        if entity_id in self._bboxes:
            self._bboxes[entity_id]["bbox"] = bbox
            self.update()

    def update_label(self, entity_id: str, label: str) -> None:
        """Update display label for entity."""
        if entity_id in self._bboxes:
            self._bboxes[entity_id]["label"] = label
            self.update()

    def paintEvent(self, event):
        """Paint image and bboxes."""
        painter = QPainter(self)

        if self._pixmap:
            # Draw image (scaled to fit widget)
            target_rect = self.rect()
            painter.drawPixmap(target_rect, self._pixmap)

            # Calculate scale
            pix_w, pix_h = self._pixmap.width(), self._pixmap.height()
            widget_w, widget_h = self.width(), self.height()
            self._scale_factor = min(widget_w / pix_w, widget_h / pix_h)
        else:
            # No image — gray background
            painter.fillRect(self.rect(), QColor(80, 80, 80))
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawText(self.rect(), Qt.AlignCenter, "(No image loaded)")
            self._scale_factor = 1.0

        # Draw bboxes
        for entity_id, data in self._bboxes.items():
            bbox = data["bbox"]  # [x, y, w, h]
            label = data["label"]
            color = data["color"]

            x, y, w, h = bbox
            scaled_x = int(x * self._scale_factor)
            scaled_y = int(y * self._scale_factor)
            scaled_w = int(w * self._scale_factor)
            scaled_h = int(h * self._scale_factor)

            # Determine color
            is_selected = entity_id == self._selected_entity_id
            if is_selected:
                pen_color = QColor(0, 255, 0)  # Green for selected
                pen_width = 3
            else:
                r, g, b = color
                pen_color = QColor(r, g, b)
                pen_width = 2

            pen = QPen(pen_color, pen_width)
            painter.setPen(pen)

            # Draw rectangle
            painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)

            # Draw label
            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(scaled_x + 2, scaled_y + 15, f"{label} ({entity_id})")

            # Draw handles if selected
            if is_selected:
                handle_size = 6
                handle_color = QColor(0, 255, 0)
                painter.fillRect(
                    scaled_x - handle_size // 2, scaled_y - handle_size // 2,
                    handle_size, handle_size, handle_color
                )
                painter.fillRect(
                    scaled_x + scaled_w - handle_size // 2, scaled_y - handle_size // 2,
                    handle_size, handle_size, handle_color
                )
                painter.fillRect(
                    scaled_x - handle_size // 2, scaled_y + scaled_h - handle_size // 2,
                    handle_size, handle_size, handle_color
                )
                painter.fillRect(
                    scaled_x + scaled_w - handle_size // 2, scaled_y + scaled_h - handle_size // 2,
                    handle_size, handle_size, handle_color
                )

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for selection and dragging."""
        pix_w = self._pixmap.width() if self._pixmap else 0
        pix_h = self._pixmap.height() if self._pixmap else 0
        widget_w, widget_h = self.width(), self.height()
        self._scale_factor = min(widget_w / pix_w, widget_h / pix_h) if pix_w > 0 and pix_h > 0 else 1.0

        x, y = event.x(), event.y()

        # Check if clicking on a bbox handle
        for entity_id, data in self._bboxes.items():
            bbox = data["bbox"]
            ex, ey, ew, eh = bbox
            scaled_ex = int(ex * self._scale_factor)
            scaled_ey = int(ey * self._scale_factor)
            scaled_ew = int(ew * self._scale_factor)
            scaled_eh = int(eh * self._scale_factor)

            handle_size = 12
            # Top-left
            if (scaled_ex - handle_size // 2 <= x <= scaled_ex + handle_size // 2 and
                    scaled_ey - handle_size // 2 <= y <= scaled_ey + handle_size // 2):
                self._selected_entity_id = entity_id
                self._dragging_corner = (entity_id, 0)  # top-left
                self._dragging_offset = (x - scaled_ex, y - scaled_ey)
                self.bring_to_top(entity_id)
                self.entity_selected.emit(entity_id)
                self.update()
                return

            # Top-right
            if (scaled_ex + scaled_ew - handle_size // 2 <= x <= scaled_ex + scaled_ew + handle_size // 2 and
                    scaled_ey - handle_size // 2 <= y <= scaled_ey + handle_size // 2):
                self._selected_entity_id = entity_id
                self._dragging_corner = (entity_id, 1)  # top-right
                self._dragging_offset = (x - (scaled_ex + scaled_ew), y - scaled_ey)
                self.bring_to_top(entity_id)
                self.entity_selected.emit(entity_id)
                self.update()
                return

            # Bottom-left
            if (scaled_ex - handle_size // 2 <= x <= scaled_ex + handle_size // 2 and
                    scaled_ey + scaled_eh - handle_size // 2 <= y <= scaled_ey + scaled_eh + handle_size // 2):
                self._selected_entity_id = entity_id
                self._dragging_corner = (entity_id, 2)  # bottom-left
                self._dragging_offset = (x - scaled_ex, y - (scaled_ey + scaled_eh))
                self.bring_to_top(entity_id)
                self.entity_selected.emit(entity_id)
                self.update()
                return

            # Bottom-right
            if (scaled_ex + scaled_ew - handle_size // 2 <= x <= scaled_ex + scaled_ew + handle_size // 2 and
                    scaled_ey + scaled_eh - handle_size // 2 <= y <= scaled_ey + scaled_eh + handle_size // 2):
                self._selected_entity_id = entity_id
                self._dragging_corner = (entity_id, 3)  # bottom-right
                self._dragging_offset = (x - (scaled_ex + scaled_ew), y - (scaled_ey + scaled_eh))
                self.bring_to_top(entity_id)
                self.entity_selected.emit(entity_id)
                self.update()
                return

        # Check if clicking on bbox itself
        for entity_id, data in self._bboxes.items():
            bbox = data["bbox"]
            ex, ey, ew, eh = bbox
            scaled_ex = int(ex * self._scale_factor)
            scaled_ey = int(ey * self._scale_factor)
            scaled_ew = int(ew * self._scale_factor)
            scaled_eh = int(eh * self._scale_factor)

            if scaled_ex <= x <= scaled_ex + scaled_ew and scaled_ey <= y <= scaled_ey + scaled_eh:
                self._selected_entity_id = entity_id
                self._dragging_corner = (entity_id, 4)  # Body (move)
                self._dragging_offset = (x - scaled_ex, y - scaled_ey)
                self.bring_to_top(entity_id)
                self.entity_selected.emit(entity_id)
                self.update()
                return

        # No bbox selected
        self._selected_entity_id = None
        self._dragging_corner = None
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for bbox resizing/moving."""
        if not self._dragging_corner or not self._pixmap:
            return

        entity_id, corner = self._dragging_corner
        if entity_id not in self._bboxes:
            return

        pix_w, pix_h = self._pixmap.width(), self._pixmap.height()
        widget_w, widget_h = self.width(), self.height()
        self._scale_factor = min(widget_w / pix_w, widget_h / pix_h) if pix_w > 0 and pix_h > 0 else 1.0

        x, y = event.x(), event.y()
        bbox = self._bboxes[entity_id]["bbox"]
        bx, by, bw, bh = bbox

        if corner == 0:  # top-left
            new_x = int((x - self._dragging_offset[0]) / self._scale_factor)
            new_y = int((y - self._dragging_offset[1]) / self._scale_factor)
            new_x = max(0, min(new_x, bx + bw - 10))
            new_y = max(0, min(new_y, by + bh - 10))
            self._bboxes[entity_id]["bbox"] = [new_x, new_y, bw - (new_x - bx), bh - (new_y - by)]
        elif corner == 1:  # top-right
            new_x = int((x - self._dragging_offset[0]) / self._scale_factor)
            new_y = int((y - self._dragging_offset[1]) / self._scale_factor)
            new_x = max(bx + 10, new_x)
            new_y = max(0, min(new_y, by + bh - 10))
            self._bboxes[entity_id]["bbox"] = [bx, new_y, new_x - bx, bh - (new_y - by)]
        elif corner == 2:  # bottom-left
            new_x = int((x - self._dragging_offset[0]) / self._scale_factor)
            new_y = int((y - self._dragging_offset[1]) / self._scale_factor)
            new_x = max(0, min(new_x, bx + bw - 10))
            new_y = max(by + 10, new_y)
            self._bboxes[entity_id]["bbox"] = [new_x, by, bw - (new_x - bx), new_y - by]
        elif corner == 3:  # bottom-right
            new_x = int((x - self._dragging_offset[0]) / self._scale_factor)
            new_y = int((y - self._dragging_offset[1]) / self._scale_factor)
            new_x = max(bx + 10, new_x)
            new_y = max(by + 10, new_y)
            self._bboxes[entity_id]["bbox"] = [bx, by, new_x - bx, new_y - by]
        elif corner == 4:  # body (move)
            new_x = int((x - self._dragging_offset[0]) / self._scale_factor)
            new_y = int((y - self._dragging_offset[1]) / self._scale_factor)
            new_x = max(0, min(new_x, pix_w - bw))
            new_y = max(0, min(new_y, pix_h - bh))
            self._bboxes[entity_id]["bbox"] = [new_x, new_y, bw, bh]

        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if self._dragging_corner and self._selected_entity_id:
            entity_id = self._selected_entity_id
            bbox = self._bboxes[entity_id]["bbox"]
            self.bbox_changed.emit(entity_id, QRect(bbox[0], bbox[1], bbox[2], bbox[3]))
        self._dragging_corner = None


class SceneGraphAnnotationEditor(QWidget):
    """Interactive scene graph editor widget."""

    graph_changed = pyqtSignal(dict)  # Emitted when graph is modified

    def __init__(self, parent=None):
        super().__init__(parent)
        self._graph: Dict[str, object] = {}
        self._image_path = ""
        self._undo_stack: List[Dict[str, object]] = []
        self._redo_stack: List[Dict[str, object]] = []
        self._suppress_item_changed = False  # Guard against recursive itemChanged
        self._editing_edge_row: Optional[int] = None  # Row index of edge being edited
        self._build_ui()
        # Accept keyboard focus so Ctrl+Z works when widget is focused
        self.setFocusPolicy(Qt.StrongFocus)

    def _build_ui(self) -> None:
        """Build the UI."""
        layout = QVBoxLayout(self)

        # Main content: image on left, controls on right
        splitter = QSplitter(Qt.Horizontal)

        # Left: Image with bboxes
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        scroll = QScrollArea()
        self.bbox_widget = BBoxDrawWidget()
        self.bbox_widget.entity_selected.connect(self._on_entity_selected_in_image)
        self.bbox_widget.bbox_changed.connect(self._on_bbox_dragged)
        scroll.setWidget(self.bbox_widget)
        scroll.setWidgetResizable(True)
        left_layout.addWidget(scroll, 1)

        splitter.addWidget(left_widget)

        # Right: Entity and relation controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Vertical splitter for entity and relation panels
        right_splitter = QSplitter(Qt.Vertical)

        # ── Entity list ──────────────────────────────────────────────────────
        entity_group = QGroupBox("Entity Annotations")
        entity_form = QVBoxLayout(entity_group)

        # 3 columns: ID (read-only), Label (editable), Bbox (read-only)
        self.entity_table = QTableWidget(0, 3)
        self.entity_table.setHorizontalHeaderLabels(["ID", "Label", "Bbox [x,y,w,h]"])
        self.entity_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.entity_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.entity_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.entity_table.itemSelectionChanged.connect(self._on_entity_selected)
        self.entity_table.itemChanged.connect(self._on_label_edited)
        entity_form.addWidget(self.entity_table)

        controls = QHBoxLayout()
        self.btn_add_entity = QPushButton("Add Entity")
        self.btn_add_entity.clicked.connect(self._add_entity)
        self.btn_remove_entity = QPushButton("Remove Selected")
        self.btn_remove_entity.clicked.connect(self._remove_entity)
        controls.addWidget(self.btn_add_entity)
        controls.addWidget(self.btn_remove_entity)
        entity_form.addLayout(controls)

        right_splitter.addWidget(entity_group)

        # ── Relation editor ──────────────────────────────────────────────────
        relation_group = QGroupBox("Relation (Edge) Editor")
        relation_form = QVBoxLayout(relation_group)

        # Source / target entity pickers
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("From:"))
        self.combo_src = QComboBox()
        self.combo_src.setMinimumWidth(80)
        src_row.addWidget(self.combo_src, 1)

        src_row.addWidget(QLabel("To:"))
        self.combo_dst = QComboBox()
        self.combo_dst.setMinimumWidth(80)
        src_row.addWidget(self.combo_dst, 1)
        relation_form.addLayout(src_row)

        # Relation dropdown
        rel_row = QHBoxLayout()
        rel_row.addWidget(QLabel("Relation:"))
        self.combo_relation = QComboBox()
        self.combo_relation.setEditable(True)  # allow custom text too
        self.combo_relation.addItems(RELATION_VOCABULARY)
        rel_row.addWidget(self.combo_relation, 1)
        relation_form.addLayout(rel_row)

        # Add / update / remove edge buttons
        edge_btn_row = QHBoxLayout()
        self.btn_add_edge = QPushButton("Add Edge")
        self.btn_add_edge.clicked.connect(self._add_edge)
        self.btn_update_edge = QPushButton("Update Selected")
        self.btn_update_edge.setEnabled(False)
        self.btn_update_edge.clicked.connect(self._update_edge)
        self.btn_remove_edge = QPushButton("Remove Selected")
        self.btn_remove_edge.clicked.connect(self._remove_edge)
        edge_btn_row.addWidget(self.btn_add_edge)
        edge_btn_row.addWidget(self.btn_update_edge)
        edge_btn_row.addWidget(self.btn_remove_edge)
        relation_form.addLayout(edge_btn_row)

        # Edge list table — columns: From label, To label, Relation, Conf
        # Entity IDs stored as Qt.UserRole in col 0/1; raw relation in col 2 UserRole.
        # Low-confidence rows are highlighted in orange.
        self.edge_table = QTableWidget(0, 4)
        self.edge_table.setHorizontalHeaderLabels(["From", "To", "Relation", "Conf"])
        self.edge_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.edge_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.edge_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.edge_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.edge_table.itemSelectionChanged.connect(self._on_edge_selected)
        relation_form.addWidget(self.edge_table, 1)

        right_splitter.addWidget(relation_group)

        right_splitter.setSizes([280, 220])
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)

        right_layout.addWidget(right_splitter, 1)

        # Undo/Redo buttons (fixed at bottom)
        undo_redo = QHBoxLayout()
        self.btn_undo = QPushButton("Undo  (Ctrl+Z)")
        self.btn_undo.clicked.connect(self._undo)
        self.btn_redo = QPushButton("Redo  (Ctrl+Y)")
        self.btn_redo.clicked.connect(self._redo)
        undo_redo.addWidget(self.btn_undo)
        undo_redo.addWidget(self.btn_redo)
        undo_redo.addStretch(1)
        right_layout.addLayout(undo_redo)

        splitter.addWidget(right_widget)
        splitter.setSizes([500, 420])
        layout.addWidget(splitter, 1)

    # ── keyboard ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        """Handle Ctrl+Z (undo) and Ctrl+Y (redo) shortcuts."""
        mods = event.modifiers()
        key = event.key()
        if mods & Qt.ControlModifier:
            if key == Qt.Key_Z:
                self._undo()
                return
            if key == Qt.Key_Y:
                self._redo()
                return
        super().keyPressEvent(event)

    # ── load ──────────────────────────────────────────────────────────────────

    def load_image_and_graph(self, image_path: str, graph: Dict[str, object]) -> None:
        """Load image and scene graph for editing.

        If *image_path* is missing or fails to load the editor still opens and
        all entity/edge controls are fully functional (annotation without image).
        """
        self._image_path = image_path
        self._graph = json.loads(json.dumps(graph))  # Deep copy
        self._undo_stack = []
        self._redo_stack = []

        # Attempt image load — continue even on failure
        if image_path and os.path.isfile(image_path):
            try:
                self.bbox_widget.load_image(image_path)
            except Exception as e:
                QMessageBox.warning(self, "Image Load Failed",
                                    f"Image could not be loaded:\n{e}\n\n"
                                    "Annotation controls are still available.")
        else:
            # No image — bbox widget shows gray placeholder
            self.bbox_widget._pixmap = None
            self.bbox_widget.update()

        self._render_entities()
        self._render_edges()
        self._update_relation_combos()

    # ── render ────────────────────────────────────────────────────────────────

    def _render_entities(self) -> None:
        """Render entities as rows in table and bboxes on image."""
        self._suppress_item_changed = True
        nodes = self._graph.get("nodes", [])

        # Rebuild table
        self.entity_table.setRowCount(0)
        self.entity_table.setRowCount(len(nodes))

        # Clear bboxes
        self.bbox_widget._bboxes.clear()

        for i, node in enumerate(nodes):
            entity_id = str(node.get("entity_id", ""))
            label = str(node.get("canonical_label", ""))
            bbox_data = list(node.get("bbox") or [10, 10, 50, 50])
            if len(bbox_data) < 4:
                bbox_data = [10, 10, 50, 50]
            bbox_str = f"[{bbox_data[0]}, {bbox_data[1]}, {bbox_data[2]}, {bbox_data[3]}]"

            # ID — not editable
            id_item = QTableWidgetItem(entity_id)
            id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
            self.entity_table.setItem(i, 0, id_item)

            # Label — editable inline
            label_item = QTableWidgetItem(label)
            self.entity_table.setItem(i, 1, label_item)

            # Bbox — not editable
            bbox_item = QTableWidgetItem(bbox_str)
            bbox_item.setFlags(bbox_item.flags() & ~Qt.ItemIsEditable)
            self.entity_table.setItem(i, 2, bbox_item)

            # Add entity to image widget
            color = (int(255 * i / max(len(nodes), 1)), 100, 150)
            self.bbox_widget.add_bbox(entity_id, label, bbox_data, color)

        # Re-bring previously selected entity to top layer so it remains editable
        sel_id = self.bbox_widget._selected_entity_id
        if sel_id and sel_id in self.bbox_widget._bboxes:
            self.bbox_widget.bring_to_top(sel_id)
        self.bbox_widget.update()
        self._suppress_item_changed = False

    def _node_label_map(self) -> Dict[str, str]:
        """Return {entity_id: canonical_label} for quick lookup."""
        return {
            str(n.get("entity_id", "") or ""): str(n.get("canonical_label", "") or "")
            for n in (self._graph.get("nodes") or [])
            if isinstance(n, dict)
        }

    def _render_edges(self) -> None:
        """Render edges in the edge list table (labels shown, IDs stored as UserRole).

        Low-confidence edges (conf < EDGE_LOW_CONF_THRESHOLD) are highlighted in
        orange and also get a 'low_confidence' validator flag written back to the graph.
        """
        edges = self._graph.get("edges", [])
        label_map = self._node_label_map()
        self._editing_edge_row = None
        self.btn_update_edge.setEnabled(False)
        self.edge_table.setRowCount(0)
        self.edge_table.setRowCount(len(edges))

        low_conf_bg = QColor(255, 180, 60)   # orange — low confidence
        normal_bg = QColor(255, 255, 255, 0) # transparent (default)

        for i, edge in enumerate(edges):
            src_id = str(edge.get("src_id", "") or "")
            dst_id = str(edge.get("dst_id", "") or "")
            relation = str(edge.get("relation", "") or "")
            conf = float(edge.get("confidence", edge.get("score", 1.0)) or 1.0)
            src_label = label_map.get(src_id, src_id) or src_id
            dst_label = label_map.get(dst_id, dst_id) or dst_id

            is_low_conf = conf < EDGE_LOW_CONF_THRESHOLD
            # Write low_confidence flag back to the graph data (non-destructive)
            flags: List[str] = list(edge.get("validator_flags") or [])
            if is_low_conf and "low_confidence" not in flags:
                flags.append("low_confidence")
                edge["validator_flags"] = flags
            elif not is_low_conf and "low_confidence" in flags:
                flags = [f for f in flags if f != "low_confidence"]
                edge["validator_flags"] = flags

            src_item = QTableWidgetItem(f"{src_label} [{src_id}]")
            src_item.setData(Qt.UserRole, src_id)
            dst_item = QTableWidgetItem(f"{dst_label} [{dst_id}]")
            dst_item.setData(Qt.UserRole, dst_id)
            rel_item = QTableWidgetItem(relation)
            rel_item.setData(Qt.UserRole, relation)
            conf_item = QTableWidgetItem(f"{conf:.2f}")
            conf_item.setData(Qt.UserRole, conf)
            conf_item.setFlags(conf_item.flags() & ~Qt.ItemIsEditable)

            if is_low_conf:
                for item in (src_item, dst_item, rel_item, conf_item):
                    item.setBackground(low_conf_bg)
                    item.setToolTip(f"Low confidence ({conf:.2f}) — consider verifying or removing this edge")
            else:
                for item in (src_item, dst_item, rel_item, conf_item):
                    item.setBackground(normal_bg)

            self.edge_table.setItem(i, 0, src_item)
            self.edge_table.setItem(i, 1, dst_item)
            self.edge_table.setItem(i, 2, rel_item)
            self.edge_table.setItem(i, 3, conf_item)

    def _update_relation_combos(self) -> None:
        """Refresh entity dropdowns in the relation editor."""
        nodes = self._graph.get("nodes", [])
        labels = [f"{n.get('entity_id','')} ({n.get('canonical_label','')})" for n in nodes]

        self.combo_src.blockSignals(True)
        self.combo_dst.blockSignals(True)

        prev_src = self.combo_src.currentText()
        prev_dst = self.combo_dst.currentText()

        self.combo_src.clear()
        self.combo_dst.clear()
        self.combo_src.addItems(labels)
        self.combo_dst.addItems(labels)

        # Restore previous selection if still valid
        src_idx = self.combo_src.findText(prev_src)
        dst_idx = self.combo_dst.findText(prev_dst)
        if src_idx >= 0:
            self.combo_src.setCurrentIndex(src_idx)
        if dst_idx >= 0:
            self.combo_dst.setCurrentIndex(dst_idx)

        self.combo_src.blockSignals(False)
        self.combo_dst.blockSignals(False)

    # ── entity table signals ──────────────────────────────────────────────────

    def _on_entity_selected(self) -> None:
        """Handle entity selection in table — sync bbox widget and bring to top."""
        sel = self.entity_table.selectionModel()
        if not sel or not sel.hasSelection():
            return

        row = sel.selectedRows()[0].row()
        id_item = self.entity_table.item(row, 0)
        if id_item is None:
            return
        entity_id = id_item.text()
        # Bring bbox to top layer for easier resize
        self.bbox_widget.select_entity(entity_id)

    def _on_entity_selected_in_image(self, entity_id: str) -> None:
        """Handle entity selection in image widget — sync table."""
        for i in range(self.entity_table.rowCount()):
            item = self.entity_table.item(i, 0)
            if item and item.text() == entity_id:
                self.entity_table.selectRow(i)
                break

    def _on_label_edited(self, item: QTableWidgetItem) -> None:
        """Handle inline label editing in the entity table."""
        if self._suppress_item_changed:
            return
        if item.column() != 1:  # Only care about label column
            return

        row = item.row()
        id_item = self.entity_table.item(row, 0)
        if id_item is None:
            return
        entity_id = id_item.text()
        new_label = item.text().strip()
        if not new_label:
            return

        # Save undo state
        self._undo_stack.append(json.loads(json.dumps(self._graph)))
        self._redo_stack.clear()

        # Update graph
        for node in self._graph.get("nodes", []):
            if str(node.get("entity_id", "")) == entity_id:
                node["canonical_label"] = new_label
                break

        # Update bbox widget label display
        self.bbox_widget.update_label(entity_id, new_label)
        self.graph_changed.emit(self._graph)

    def _on_bbox_dragged(self, entity_id: str, rect: QRect) -> None:
        """Handle bbox drag-resize from image widget — sync graph data."""
        self._undo_stack.append(json.loads(json.dumps(self._graph)))
        self._redo_stack.clear()

        new_bbox = [rect.x(), rect.y(), rect.width(), rect.height()]
        for node in self._graph.get("nodes", []):
            if str(node.get("entity_id", "")) == entity_id:
                node["bbox"] = new_bbox
                break

        # Update bbox string in table
        self._suppress_item_changed = True
        for row in range(self.entity_table.rowCount()):
            id_item = self.entity_table.item(row, 0)
            if id_item and id_item.text() == entity_id:
                bbox_str = f"[{new_bbox[0]}, {new_bbox[1]}, {new_bbox[2]}, {new_bbox[3]}]"
                self.entity_table.item(row, 2).setText(bbox_str)
                break
        self._suppress_item_changed = False

        self.graph_changed.emit(self._graph)

    # ── entity add / remove ───────────────────────────────────────────────────

    def _add_entity(self) -> None:
        """Add new entity annotation."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Entity")
        dlg_layout = QVBoxLayout(dialog)

        label_input = QLineEdit()
        label_input.setPlaceholderText("Entity label (e.g., 'cup', 'person')")
        dlg_layout.addWidget(label_input)

        buttons = QHBoxLayout()
        ok_btn = QPushButton("Add")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        dlg_layout.addLayout(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return

        label = label_input.text().strip()
        if not label:
            QMessageBox.warning(self, "Invalid Input", "Label cannot be empty.")
            return

        # Save undo state
        self._undo_stack.append(json.loads(json.dumps(self._graph)))
        self._redo_stack.clear()

        # Add entity — confidence 1.0 for manual annotations, default bbox
        new_entity = {
            "entity_id": f"ent_manual_{len(self._graph.get('nodes', []))}",
            "canonical_label": label,
            "confidence": 1.0,
            "bbox": [10, 10, 100, 100],
            "validator_flags": ["manual_annotation"],
        }
        self._graph.setdefault("nodes", []).append(new_entity)
        self._render_entities()
        self._render_edges()
        self._update_relation_combos()
        self.graph_changed.emit(self._graph)

    def _remove_entity(self) -> None:
        """Remove selected entity."""
        sel = self.entity_table.selectionModel()
        if not sel or not sel.hasSelection():
            QMessageBox.information(self, "No Selection", "Select an entity to remove.")
            return

        row = sel.selectedRows()[0].row()
        id_item = self.entity_table.item(row, 0)
        if id_item is None:
            return
        entity_id = id_item.text()

        # Save undo state
        self._undo_stack.append(json.loads(json.dumps(self._graph)))
        self._redo_stack.clear()

        # Remove entity and related edges
        nodes = self._graph.get("nodes", [])
        nodes[:] = [n for n in nodes if str(n.get("entity_id", "")) != entity_id]
        edges = self._graph.get("edges", [])
        edges[:] = [e for e in edges
                    if str(e.get("src_id", "")) != entity_id and str(e.get("dst_id", "")) != entity_id]

        self.bbox_widget.remove_bbox(entity_id)
        self._render_entities()
        self._render_edges()
        self._update_relation_combos()
        self.graph_changed.emit(self._graph)

    # ── edge selection → pre-fill combos ─────────────────────────────────────

    def _on_edge_selected(self) -> None:
        """When a row is selected in the edge table, pre-fill the combos for editing."""
        sel = self.edge_table.selectionModel()
        if not sel or not sel.hasSelection():
            self._editing_edge_row = None
            self.btn_update_edge.setEnabled(False)
            return

        row = sel.selectedRows()[0].row()
        src_item = self.edge_table.item(row, 0)
        dst_item = self.edge_table.item(row, 1)
        rel_item = self.edge_table.item(row, 2)
        if src_item is None:
            return

        src_id = src_item.data(Qt.UserRole) or src_item.text()
        dst_id = (dst_item.data(Qt.UserRole) or dst_item.text()) if dst_item else ""
        relation = (rel_item.data(Qt.UserRole) or rel_item.text()) if rel_item else ""

        nodes = self._graph.get("nodes", [])
        node_ids = [str(n.get("entity_id", "")) for n in nodes]

        # Pre-fill Source combo
        if src_id in node_ids:
            self.combo_src.setCurrentIndex(node_ids.index(src_id))
        # Pre-fill Destination combo
        if dst_id in node_ids:
            self.combo_dst.setCurrentIndex(node_ids.index(dst_id))
        # Pre-fill Relation combo
        rel_idx = self.combo_relation.findText(relation)
        if rel_idx >= 0:
            self.combo_relation.setCurrentIndex(rel_idx)
        else:
            self.combo_relation.setEditText(relation)

        self._editing_edge_row = row
        self.btn_update_edge.setEnabled(True)

    # ── edge add / update / remove ────────────────────────────────────────────

    def _resolve_src_dst(self) -> Optional[tuple]:
        """Resolve (src_id, dst_id, relation) from combos. Returns None on error."""
        nodes = self._graph.get("nodes", [])
        if len(nodes) < 2:
            QMessageBox.information(self, "Not Enough Entities",
                                    "At least two entities are required to add a relation edge.")
            return None
        src_idx = self.combo_src.currentIndex()
        dst_idx = self.combo_dst.currentIndex()
        relation = self.combo_relation.currentText().strip()
        if src_idx < 0 or dst_idx < 0 or src_idx == dst_idx:
            QMessageBox.warning(self, "Invalid Selection",
                                "Please select different source and target entities.")
            return None
        if not relation:
            QMessageBox.warning(self, "Empty Relation", "Please specify a relation.")
            return None
        src_id = str(nodes[src_idx].get("entity_id", ""))
        dst_id = str(nodes[dst_idx].get("entity_id", ""))
        return src_id, dst_id, relation

    def _check_reverse_duplicate(self, src_id: str, dst_id: str, exclude_row: Optional[int] = None) -> bool:
        """Warn if the reverse edge (dst→src) already exists.

        Returns True if the user chose to proceed anyway, False to cancel.
        """
        edges = self._graph.get("edges", [])
        for i, edge in enumerate(edges):
            if i == exclude_row:
                continue
            if (str(edge.get("src_id", "")) == dst_id and
                    str(edge.get("dst_id", "")) == src_id):
                label_map = self._node_label_map()
                src_lbl = label_map.get(dst_id, dst_id)
                dst_lbl = label_map.get(src_id, src_id)
                existing_rel = str(edge.get("relation", "") or "")
                reply = QMessageBox.question(
                    self,
                    "Reverse Edge Exists",
                    f"The reverse edge  \"{src_lbl} → {dst_lbl}\"  "
                    f"(relation: '{existing_rel}') already exists.\n\n"
                    f"Adding the opposite direction is usually redundant for spatial relations.\n"
                    f"Add anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                return reply == QMessageBox.Yes
        return True  # No reverse edge found — proceed

    def _add_edge(self) -> None:
        """Add a new relation edge between two entities."""
        resolved = self._resolve_src_dst()
        if resolved is None:
            return
        src_id, dst_id, relation = resolved

        # Block exact duplicate (same direction, same relation)
        for edge in (self._graph.get("edges") or []):
            if (str(edge.get("src_id", "")) == src_id and
                    str(edge.get("dst_id", "")) == dst_id and
                    str(edge.get("relation", "")) == relation):
                QMessageBox.information(self, "Duplicate Edge",
                                        "This exact edge already exists.")
                return

        # Warn about reverse duplicate (B→A when A→B being added)
        if not self._check_reverse_duplicate(src_id, dst_id):
            return

        # Save undo state
        self._undo_stack.append(json.loads(json.dumps(self._graph)))
        self._redo_stack.clear()

        edge_id = f"edge_manual_{len(self._graph.get('edges', []))}"
        new_edge = {
            "edge_id": edge_id,
            "src_id": src_id,
            "dst_id": dst_id,
            "relation": relation,
            "confidence": 1.0,
            "validator_flags": ["manual_annotation"],
        }
        self._graph.setdefault("edges", []).append(new_edge)
        self._render_edges()
        self.graph_changed.emit(self._graph)

    def _update_edge(self) -> None:
        """Update the currently selected edge with the combo values."""
        if self._editing_edge_row is None:
            return
        resolved = self._resolve_src_dst()
        if resolved is None:
            return
        src_id, dst_id, relation = resolved

        # Warn about reverse duplicate (excluding self)
        if not self._check_reverse_duplicate(src_id, dst_id, exclude_row=self._editing_edge_row):
            return

        # Save undo state
        self._undo_stack.append(json.loads(json.dumps(self._graph)))
        self._redo_stack.clear()

        edges = self._graph.get("edges", [])
        if self._editing_edge_row < len(edges):
            edge = edges[self._editing_edge_row]
            edge["src_id"] = src_id
            edge["dst_id"] = dst_id
            edge["relation"] = relation

        self._render_edges()
        self.graph_changed.emit(self._graph)

    def _remove_edge(self) -> None:
        """Remove selected edge from edge table."""
        sel = self.edge_table.selectionModel()
        if not sel or not sel.hasSelection():
            QMessageBox.information(self, "No Selection", "Select an edge row to remove.")
            return

        row = sel.selectedRows()[0].row()
        src_item = self.edge_table.item(row, 0)
        if src_item is None:
            return

        # Save undo state
        self._undo_stack.append(json.loads(json.dumps(self._graph)))
        self._redo_stack.clear()

        edges = self._graph.get("edges", [])
        if row < len(edges):
            del edges[row]

        self._render_edges()
        self.graph_changed.emit(self._graph)

    # ── undo / redo ───────────────────────────────────────────────────────────

    def _undo(self) -> None:
        """Undo last change."""
        if not self._undo_stack:
            return

        self._redo_stack.append(json.loads(json.dumps(self._graph)))
        self._graph = self._undo_stack.pop()
        self._render_entities()
        self._render_edges()
        self._update_relation_combos()
        self.graph_changed.emit(self._graph)

    def _redo(self) -> None:
        """Redo last undone change."""
        if not self._redo_stack:
            return

        self._undo_stack.append(json.loads(json.dumps(self._graph)))
        self._graph = self._redo_stack.pop()
        self._render_entities()
        self._render_edges()
        self._update_relation_combos()
        self.graph_changed.emit(self._graph)

    def get_graph(self) -> Dict[str, object]:
        """Get current edited graph."""
        return self._graph
