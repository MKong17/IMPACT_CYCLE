"""
Scene Graph Image Overlay - Visualize SGGraph nodes and edges on the extracted frame image.
Supports node/edge selection, highlight, and direct image-based interaction.
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal
from PyQt5.QtGui import (
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QBrush,
    QFont,
    QImage,
    QPolygon,
)
from PyQt5.QtWidgets import (
    QLabel,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QInputDialog,
)

# Relation vocabulary shared with scene_graph_editor
try:
    from .scene_graph_editor import RELATION_VOCABULARY
except ImportError:
    RELATION_VOCABULARY: List[str] = [
        "on", "in", "at", "near", "next to", "beside",
        "above", "below", "under", "over",
        "behind", "in front of", "left of", "right of", "between",
        "holding", "wearing", "carrying", "using", "touching",
        "eating", "riding", "watching",
        "part of", "made of", "has", "contains",
        "attached to", "connected to", "hanging from",
        "looking at", "facing", "belongs to",
        "other",
    ]


class SceneGraphImageOverlay(QLabel):
    """
    Display scene graph overlaid on the extracted frame image.
    Shows nodes (bboxes) and edges (relation lines) with labels.
    Supports selection and editing via double-click, right-click.
    """

    node_selected = pyqtSignal(str)  # entity_id
    edge_selected = pyqtSignal(str)  # edge_id
    node_edited = pyqtSignal(str, str, float)  # entity_id, new_label, new_score
    node_relation_edited = pyqtSignal(str, str)  # edge_id, new_relation

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_pixmap: Optional[QPixmap] = None
        self._graph_data: Dict = {}
        self._nodes: Dict[str, Dict] = {}  # entity_id -> node data
        self._edges: List[Dict] = []        # edge list
        self._image_width = 0
        self._image_height = 0
        
        # Selection state
        self._selected_node_id: Optional[str] = None
        self._selected_edge_id: Optional[str] = None
        
        # Node bbox cache (for hit testing)
        self._node_rects: Dict[str, QRect] = {}  # entity_id -> screen rect
        self._edge_label_rects: Dict[str, QRect] = {}  # edge_id -> label rect
        
        # Rendering parameters
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._offset_x = 0
        self._offset_y = 0
        
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self.setCursor(Qt.CrossCursor)
        
        # Enable interactions
        self.setMouseTracking(True)
        self.setStyleSheet("border: 1px solid #ccc;")

    @staticmethod
    def node_confidence(node: Dict) -> float:
        """Unified confidence accessor for overlay rendering/editing."""
        try:
            return float(node.get("score", node.get("confidence", 0.0)) or 0.0)
        except Exception:
            return 0.0

    def load_image_and_graph(self, image_path: str, graph_data: Dict) -> None:
        """Load the frame image and scene graph data."""
        try:
            # Load image
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                raise ValueError(f"Unable to load image: {image_path}")
            
            # Convert BGR to RGB and load as QPixmap
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = 3 * w
            
            qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self._base_pixmap = QPixmap.fromImage(qt_image)
            self._image_width = w
            self._image_height = h
            
            # Store graph data
            self._graph_data = graph_data
            self._nodes = {}
            self._edges = []
            
            # Index nodes
            for node in graph_data.get("nodes") or []:
                entity_id = str(node.get("entity_id", ""))
                self._nodes[entity_id] = node
            
            # Store edges
            self._edges = list(graph_data.get("edges") or [])
            
            # Reset selection
            self._selected_node_id = None
            self._selected_edge_id = None
            self._node_rects.clear()
            self._edge_label_rects.clear()
            
            # Render and display
            self._render_overlay()
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Failed to load image:\n{e}")

    def _render_overlay(self) -> None:
        """Render the image with scene graph overlay."""
        if self._base_pixmap is None:
            return
        
        # Create a copy for drawing
        overlay_pixmap = self._base_pixmap.copy()
        
        # Calculate scaling to fit label
        label_w = self.width() if self.width() > 0 else self._image_width
        label_h = self.height() if self.height() > 0 else self._image_height
        
        if label_w <= 0 or label_h <= 0:
            self.setPixmap(overlay_pixmap)
            return
        
        # Fit to label while keeping aspect ratio
        img_aspect = self._image_width / max(1, self._image_height)
        label_aspect = label_w / max(1, label_h)
        
        if img_aspect > label_aspect:
            # Image wider
            display_w = label_w
            display_h = int(label_w / img_aspect)
        else:
            # Image taller
            display_h = label_h
            display_w = int(label_h * img_aspect)
        
        # Scale pixmap
        scaled_pixmap = self._base_pixmap.scaledToWidth(display_w, Qt.SmoothTransformation)
        display_h = scaled_pixmap.height()
        
        # Calculate offset to center in label
        self._offset_x = (label_w - display_w) / 2
        self._offset_y = (label_h - display_h) / 2
        
        # Calculate scale factors
        self._scale_x = display_w / self._image_width
        self._scale_y = display_h / self._image_height
        
        # Create final pixmap with margin
        final_pixmap = QPixmap(label_w, label_h)
        final_pixmap.fill(QColor(245, 245, 250))
        
        painter = QPainter(final_pixmap)
        painter.drawPixmap(int(self._offset_x), int(self._offset_y), scaled_pixmap)
        
        # Draw scene graph overlays
        self._draw_edges(painter)
        self._draw_nodes(painter)
        
        painter.end()
        
        self.setPixmap(final_pixmap)
        self.update()

    def _draw_nodes(self, painter: QPainter) -> None:
        """Draw node bboxes and labels."""
        self._node_rects.clear()
        
        for entity_id, node in self._nodes.items():
            bbox = node.get("bbox", [0, 0, 0, 0])
            if not bbox or len(bbox) < 4:
                continue
            
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            
            # Convert to screen coordinates
            screen_x = int(x * self._scale_x + self._offset_x)
            screen_y = int(y * self._scale_y + self._offset_y)
            screen_w = int(w * self._scale_x)
            screen_h = int(h * self._scale_y)
            
            rect = QRect(screen_x, screen_y, screen_w, screen_h)
            self._node_rects[entity_id] = rect
            
            # Draw bbox
            is_selected = (entity_id == self._selected_node_id)
            color = QColor(0, 120, 255) if is_selected else QColor(100, 150, 200)
            pen = QPen(color, 2 if is_selected else 1)
            painter.setPen(pen)
            painter.drawRect(rect)
            
            # Draw label and confidence
            label = str(node.get("canonical_label", "unknown"))
            score = self.node_confidence(node)
            
            font = painter.font()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)
            
            # Draw text background
            text = f"{label} ({score:.2f})"
            metric = painter.fontMetrics()
            text_width = metric.width(text)
            text_height = metric.height()
            
            text_bg = QRect(screen_x, screen_y - text_height - 4, text_width + 4, text_height + 2)
            painter.fillRect(text_bg, QColor(0, 0, 0, 180))
            
            # Draw text
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(screen_x + 2, screen_y - 4, text)

    def _draw_edges(self, painter: QPainter) -> None:
        """Draw relation edges between nodes."""
        self._edge_label_rects.clear()
        for edge in self._edges:
            src_id = str(edge.get("src_id", ""))
            dst_id = str(edge.get("dst_id", ""))
            edge_id = str(edge.get("edge_id", ""))
            relation = str(edge.get("relation", "unknown"))
            
            if src_id not in self._nodes or dst_id not in self._nodes:
                continue
            
            src_node = self._nodes[src_id]
            dst_node = self._nodes[dst_id]
            
            # Get center of each bbox
            src_bbox = src_node.get("bbox", [0, 0, 0, 0])
            dst_bbox = dst_node.get("bbox", [0, 0, 0, 0])
            
            src_x = (src_bbox[0] + src_bbox[2] / 2) * self._scale_x + self._offset_x
            src_y = (src_bbox[1] + src_bbox[3] / 2) * self._scale_y + self._offset_y
            
            dst_x = (dst_bbox[0] + dst_bbox[2] / 2) * self._scale_x + self._offset_x
            dst_y = (dst_bbox[1] + dst_bbox[3] / 2) * self._scale_y + self._offset_y
            
            # Draw line
            is_selected = (edge_id == self._selected_edge_id)
            color = QColor(255, 100, 0) if is_selected else QColor(100, 200, 100)
            pen = QPen(color, 2 if is_selected else 1)
            painter.setPen(pen)
            painter.drawLine(int(src_x), int(src_y), int(dst_x), int(dst_y))
            
            # Draw arrowhead at destination
            angle = math.atan2(dst_y - src_y, dst_x - src_x)
            arrow_size = 10
            
            p1_x = dst_x - arrow_size * math.cos(angle - math.pi / 6)
            p1_y = dst_y - arrow_size * math.sin(angle - math.pi / 6)
            p2_x = dst_x - arrow_size * math.cos(angle + math.pi / 6)
            p2_y = dst_y - arrow_size * math.sin(angle + math.pi / 6)
            
            painter.setBrush(QBrush(color))
            painter.drawPolygon(QPolygon([
                QPoint(int(dst_x), int(dst_y)),
                QPoint(int(p1_x), int(p1_y)),
                QPoint(int(p2_x), int(p2_y)),
            ]))
            
            # Draw relation label at midpoint
            mid_x = (src_x + dst_x) / 2
            mid_y = (src_y + dst_y) / 2
            
            font = painter.font()
            font.setPointSize(7)
            painter.setFont(font)
            
            metric = painter.fontMetrics()
            text_width = metric.width(relation)
            text_height = metric.height()
            
            text_bg = QRect(
                int(mid_x - text_width / 2 - 2),
                int(mid_y - text_height / 2 - 1),
                text_width + 4,
                text_height + 2,
            )
            self._edge_label_rects[edge_id] = text_bg
            painter.fillRect(text_bg, QColor(255, 255, 255, 220))
            
            painter.setPen(QPen(QColor(0, 0, 0)))
            painter.drawText(
                int(mid_x - text_width / 2),
                int(mid_y + text_height / 3),
                relation,
            )

    def mousePressEvent(self, event):
        """Handle mouse click for selection."""
        pos = event.pos()
        
        # Check node hit
        for entity_id, rect in self._node_rects.items():
            if rect.contains(pos):
                self._selected_node_id = entity_id
                self._selected_edge_id = None
                self._render_overlay()
                self.node_selected.emit(entity_id)
                return
        
        # Check edge label hit
        for edge_id, rect in self._edge_label_rects.items():
            if rect.contains(pos):
                self._selected_edge_id = edge_id
                self._selected_node_id = None
                self._render_overlay()
                self.edge_selected.emit(edge_id)
                return
        
        # Deselect
        self._selected_node_id = None
        self._selected_edge_id = None
        self._render_overlay()

    def mouseDoubleClickEvent(self, event):
        """Handle double-click for editing."""
        pos = event.pos()
        
        # Check node double-click
        for entity_id, rect in self._node_rects.items():
            if rect.contains(pos):
                node = self._nodes.get(entity_id) or {}
                current_label = str(node.get("canonical_label", "unknown"))
                current_score = self.node_confidence(node)

                new_label, ok = QInputDialog.getText(self, "Edit Node Label", "Label:", text=current_label)
                if not ok:
                    return
                new_label = new_label.strip()
                if not new_label:
                    return

                new_score, ok = QInputDialog.getDouble(
                    self,
                    "Edit Node Score",
                    "Score:",
                    value=current_score,
                    min=0.0,
                    max=1.0,
                    decimals=4,
                )
                if not ok:
                    return

                node["canonical_label"] = new_label
                node["score"] = float(new_score)
                node["confidence"] = float(new_score)
                self._selected_node_id = entity_id
                self._selected_edge_id = None
                self._render_overlay()
                self.node_selected.emit(entity_id)
                self.node_edited.emit(entity_id, new_label, float(new_score))
                return

        # Double-click on edge label to edit relation via dropdown
        for edge in self._edges:
            edge_id = str(edge.get("edge_id", ""))
            rect = self._edge_label_rects.get(edge_id)
            if rect is not None and rect.contains(pos):
                current_rel = str(edge.get("relation", ""))
                # Build item list: predefined vocabulary + current if custom
                items = list(RELATION_VOCABULARY)
                current_idx = 0
                if current_rel and current_rel not in items:
                    items.insert(0, current_rel)
                    current_idx = 0
                elif current_rel in items:
                    current_idx = items.index(current_rel)
                new_rel, ok = QInputDialog.getItem(
                    self,
                    "Edit Relation",
                    f"Relation for edge {edge_id}:",
                    items,
                    current=current_idx,
                    editable=True,
                )
                if ok:
                    new_rel = new_rel.strip()
                    if new_rel:
                        edge["relation"] = new_rel
                        self._selected_edge_id = edge_id
                        self._selected_node_id = None
                        self._render_overlay()
                        self.node_relation_edited.emit(edge_id, new_rel)
                return

    def resizeEvent(self, event):
        """Handle resize - re-render with new dimensions."""
        super().resizeEvent(event)
        if self._base_pixmap is not None:
            self._render_overlay()

    def set_selected_node(self, entity_id: Optional[str]) -> None:
        """Set selected node from external source (sync with graph editor)."""
        self._selected_node_id = entity_id
        self._selected_edge_id = None
        self._render_overlay()

    def set_selected_edge(self, edge_id: Optional[str]) -> None:
        """Set selected edge from external source."""
        self._selected_edge_id = edge_id
        self._selected_node_id = None
        self._render_overlay()

    def clear(self) -> None:
        """Clear the overlay."""
        self._base_pixmap = None
        self._graph_data = {}
        self._nodes.clear()
        self._edges.clear()
        self._node_rects.clear()
        self._edge_label_rects.clear()
        self._selected_node_id = None
        self._selected_edge_id = None
        self.setPixmap(QPixmap())
