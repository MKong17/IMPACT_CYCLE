"""
Interactive Scene Graph Visualizer with draggable nodes and edges.
Supports force-directed layout, node/edge selection, and property editing.
"""

import json
import math
import random
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPointF, QRectF, QObject
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QBrush, QPalette
from PyQt5.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsItem,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsTextItem,
    QGraphicsPathItem,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox,
    QMenu,
)
from PyQt5.QtGui import QPolygonF, QPainterPath


class GraphNodeEvents(QObject):
    """Signal emitter for graph node events."""
    node_selected = pyqtSignal(str)  # entity_id
    node_deleted = pyqtSignal(str)   # entity_id
    properties_changed = pyqtSignal(str, dict)  # entity_id, updated_properties


class GraphEdgeEvents(QObject):
    """Signal emitter for graph edge events."""
    edge_selected = pyqtSignal(str)  # edge_id
    edge_deleted = pyqtSignal(str)   # edge_id


class GraphNode(QGraphicsItem):
    """Represents a node in the scene graph with dragging and selection support."""

    def __init__(self, entity_id: str, label: str, confidence: float, events: GraphNodeEvents, parent=None):
        super().__init__(parent)
        self.entity_id = str(entity_id)
        self.label = str(label)
        self.confidence = float(confidence)
        self._is_selected = False
        self._radius = 30
        self._dragging = False
        self._edges: List["GraphEdge"] = []
        self._events = events
        
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

    def boundingRect(self) -> QRectF:
        r = self._radius + 2
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, painter: QPainter, option, widget=None):
        r = self._radius
        
        # Draw circle
        if self._is_selected:
            painter.setPen(QPen(QColor(0, 120, 255), 3))
            painter.setBrush(QBrush(QColor(200, 230, 255)))
        else:
            painter.setPen(QPen(QColor(100, 100, 100), 2))
            painter.setBrush(QBrush(QColor(220, 240, 255)))
        
        painter.drawEllipse(-r, -r, 2 * r, 2 * r)
        
        # Draw label
        painter.setPen(QPen(QColor(0, 0, 0)))
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        
        # Draw entity label (truncated)
        label_text = self.label[:10] if len(self.label) > 10 else self.label
        metric = painter.fontMetrics()
        text_width = metric.width(label_text)
        painter.drawText(-text_width // 2, -5, label_text)
        
        # Draw confidence score below
        score_text = f"{self.confidence:.2f}"
        font.setPointSize(7)
        painter.setFont(font)
        metric = painter.fontMetrics()
        text_width = metric.width(score_text)
        painter.drawText(-text_width // 2, 8, score_text)

    def mousePressEvent(self, event):
        self._dragging = True
        self._is_selected = True
        self.update()
        if self._events:
            self._events.node_selected.emit(self.entity_id)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        # Update all connected edges
        for edge in self._edges:
            edge.adjust()

    def mouseReleaseEvent(self, event):
        self._dragging = False
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        """Right-click menu for node operations."""
        menu = QMenu()
        delete_action = menu.addAction("Delete Node")
        
        action = menu.exec_(event.screenPos())
        if action == delete_action:
            if self._events:
                self._events.node_deleted.emit(self.entity_id)

    def set_selected(self, selected: bool):
        self._is_selected = selected
        self.update()

    def add_edge(self, edge: "GraphEdge"):
        if edge not in self._edges:
            self._edges.append(edge)

    def remove_edge(self, edge: "GraphEdge"):
        if edge in self._edges:
            self._edges.remove(edge)


class GraphEdge(QGraphicsItem):
    """Represents an edge (relation) between two nodes."""

    def __init__(self, edge_id: str, src_node: GraphNode, relation: str, dst_node: GraphNode, events: GraphEdgeEvents, score: float = 1.0):
        super().__init__()
        self.edge_id = str(edge_id)
        self.src_node = src_node
        self.dst_node = dst_node
        self.relation = str(relation)
        self.score = float(score)
        self._is_selected = False
        self._events = events
        
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(5)
        
        src_node.add_edge(self)
        dst_node.add_edge(self)
        self.adjust()

    def boundingRect(self) -> QRectF:
        if not self.src_node or not self.dst_node:
            return QRectF()
        
        x1, y1 = self.src_node.pos().x(), self.src_node.pos().y()
        x2, y2 = self.dst_node.pos().x(), self.dst_node.pos().y()
        
        pad = 20
        return QRectF(
            min(x1, x2) - pad,
            min(y1, y2) - pad,
            abs(x2 - x1) + 2 * pad,
            abs(y2 - y1) + 2 * pad,
        )

    def paint(self, painter: QPainter, option, widget=None):
        if not self.src_node or not self.dst_node:
            return
        
        # Get positions
        x1, y1 = self.src_node.pos().x(), self.src_node.pos().y()
        x2, y2 = self.dst_node.pos().x(), self.dst_node.pos().y()
        
        # Draw line
        line_color = QColor(0, 120, 255) if self._is_selected else QColor(100, 150, 200)
        line_width = 2 if self._is_selected else 1
        painter.setPen(QPen(line_color, line_width))
        painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_size = 15
        
        # Arrow tip at destination
        p1_x = x2 - arrow_size * math.cos(angle - math.pi / 6)
        p1_y = y2 - arrow_size * math.sin(angle - math.pi / 6)
        p2_x = x2 - arrow_size * math.cos(angle + math.pi / 6)
        p2_y = y2 - arrow_size * math.sin(angle + math.pi / 6)
        
        painter.setBrush(QBrush(line_color))
        points = QPolygonF([QPointF(x2, y2), QPointF(p1_x, p1_y), QPointF(p2_x, p2_y)])
        painter.drawPolygon(points)
        
        # Draw relation label at midpoint
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        painter.setPen(QPen(QColor(0, 0, 0)))
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        
        rel_text = self.relation[:15] if len(self.relation) > 15 else self.relation
        metric = painter.fontMetrics()
        text_width = metric.width(rel_text)
        painter.drawText(int(mid_x - text_width // 2), int(mid_y - 5), rel_text)

    def adjust(self):
        """Update bounding rect when nodes move."""
        self.prepareGeometryChange()

    def mousePressEvent(self, event):
        self._is_selected = True
        self.update()
        if self._events:
            self._events.edge_selected.emit(self.edge_id)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool):
        self._is_selected = selected
        self.update()

    def contextMenuEvent(self, event):
        """Right-click menu for edge operations."""
        menu = QMenu()
        delete_action = menu.addAction("Delete Edge")
        
        action = menu.exec_(event.screenPos())
        if action == delete_action:
            if self._events:
                self._events.edge_deleted.emit(self.edge_id)


class SceneGraphVisualizer(QWidget):
    """
    Interactive scene graph editor with force-directed layout.
    Supports dragging nodes, selecting items, and synchronized editing.
    """

    node_selected = pyqtSignal(str)      # entity_id
    edge_selected = pyqtSignal(str)      # edge_id
    graph_changed = pyqtSignal()          # Any change to graph

    def __init__(self, parent=None):
        super().__init__(parent)
        self._graph_data: Dict = {}
        self._nodes_by_id: Dict[str, GraphNode] = {}
        self._edges_by_id: Dict[str, GraphEdge] = {}
        self._layout_running = False
        self._velocities: Dict[str, Tuple[float, float]] = {}
        
        # Event emitters
        self._node_events = GraphNodeEvents()
        self._edge_events = GraphEdgeEvents()
        
        # Connect event signals
        self._node_events.node_selected.connect(self._on_node_selected)
        self._node_events.node_deleted.connect(self._on_node_deleted)
        self._edge_events.edge_selected.connect(self._on_edge_selected)
        self._edge_events.edge_deleted.connect(self._on_edge_deleted)
        
        # Physics parameters for force-directed layout
        self._repulsion_strength = 500
        self._attraction_strength = 0.05  # Reduced from 0.1 for smoother convergence
        self._damping = 0.95  # Increased from 0.85 for faster settling
        self._max_velocity = 10
        self._convergence_threshold = 0.08  # Tighter convergence (was 0.5)
        
        self._build_ui()
        self._setup_layout_timer()

    def _build_ui(self):
        """Build the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Top toolbar
        toolbar = QHBoxLayout()
        
        toolbar.addWidget(QLabel("Layout:"))
        self.combo_layout = QComboBox()
        self.combo_layout.addItems(["Circular (Fixed)", "Grid (Fixed)", "Force-Directed (Animated)"])
        self.combo_layout.setCurrentIndex(0)  # Default to Circular
        self.combo_layout.currentIndexChanged.connect(self._on_layout_changed)
        toolbar.addWidget(self.combo_layout)
        
        self.btn_reset_layout = QPushButton("Recalculate Layout")
        self.btn_reset_layout.clicked.connect(self._reset_layout)
        toolbar.addWidget(self.btn_reset_layout)
        
        self.btn_fit_view = QPushButton("Fit to View")
        self.btn_fit_view.clicked.connect(self._fit_to_view)
        toolbar.addWidget(self.btn_fit_view)
        
        toolbar.addStretch(1)
        
        layout.addLayout(toolbar)
        
        # Graphics view
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setBackgroundBrush(QBrush(QColor(245, 245, 250)))
        layout.addWidget(self.view, 1)
        
        # Status bar
        status = QHBoxLayout()
        self.label_status = QLabel("No graph loaded")
        status.addWidget(self.label_status)
        layout.addLayout(status)

    def _setup_layout_timer(self):
        """Setup timer for force-directed layout animation."""
        self._layout_timer = QTimer()
        self._layout_timer.timeout.connect(self._step_force_layout)
        self._layout_timer.setInterval(30)  # 30ms per frame

    def load_graph(self, graph_data: Dict) -> None:
        """Load a scene graph for visualization."""
        self._graph_data = graph_data
        self._clear_visualization()
        
        nodes = list(graph_data.get("nodes") or [])
        edges = list(graph_data.get("edges") or [])
        
        # Create nodes
        for node in nodes:
            entity_id = str(node.get("entity_id", ""))
            label = str(node.get("canonical_label", "unknown"))
            confidence = float(node.get("score", 0.0))
            
            gnode = GraphNode(entity_id, label, confidence, self._node_events)
            
            # Temporary position (will be updated by layout)
            gnode.setPos(0, 0)
            self._nodes_by_id[entity_id] = gnode
            self._velocities[entity_id] = (0.0, 0.0)
            self.scene.addItem(gnode)
        
        # Create edges
        for edge in edges:
            edge_id = str(edge.get("edge_id", ""))
            src_id = str(edge.get("src_id", ""))
            dst_id = str(edge.get("dst_id", ""))
            relation = str(edge.get("relation", "unknown"))
            score = float(edge.get("score", 1.0))
            
            if src_id not in self._nodes_by_id or dst_id not in self._nodes_by_id:
                continue
            
            src_node = self._nodes_by_id[src_id]
            dst_node = self._nodes_by_id[dst_id]
            
            gedge = GraphEdge(edge_id, src_node, relation, dst_node, self._edge_events, score)
            
            self._edges_by_id[edge_id] = gedge
            self.scene.addItem(gedge)
        
        self._update_status()
        
        # Apply default layout (Circular - fast, static, no animation)
        self._apply_circular_layout()
        self._fit_to_view()

    def _clear_visualization(self):
        """Clear all nodes and edges from the scene."""
        self.scene.clear()
        self._nodes_by_id.clear()
        self._edges_by_id.clear()
        self._velocities.clear()
        if self._layout_timer.isActive():
            self._layout_timer.stop()

    def _on_node_selected(self, entity_id: str):
        """Handle node selection."""
        # Deselect all others
        for nid, node in self._nodes_by_id.items():
            node.set_selected(nid == entity_id)
        
        for eid, edge in self._edges_by_id.items():
            edge.set_selected(False)
        
        self.node_selected.emit(entity_id)

    def _on_edge_selected(self, edge_id: str):
        """Handle edge selection."""
        # Deselect all nodes
        for node in self._nodes_by_id.values():
            node.set_selected(False)
        
        # Deselect other edges
        for eid, edge in self._edges_by_id.items():
            edge.set_selected(eid == edge_id)
        
        self.edge_selected.emit(edge_id)

    def _on_node_deleted(self, entity_id: str):
        """Handle node deletion."""
        if entity_id not in self._nodes_by_id:
            return
        
        node = self._nodes_by_id[entity_id]
        
        # Remove all connected edges
        edges_to_remove = []
        for edge_id, edge in list(self._edges_by_id.items()):
            if edge.src_node == node or edge.dst_node == node:
                edges_to_remove.append(edge_id)
        
        for edge_id in edges_to_remove:
            del self._edges_by_id[edge_id]
        
        # Remove node
        self.scene.removeItem(node)
        del self._nodes_by_id[entity_id]
        del self._velocities[entity_id]
        
        self._update_status()
        self.graph_changed.emit()

    def _on_edge_deleted(self, edge_id: str):
        """Handle edge deletion."""
        if edge_id not in self._edges_by_id:
            return
        
        edge = self._edges_by_id[edge_id]
        self.scene.removeItem(edge)
        del self._edges_by_id[edge_id]
        
        self._update_status()
        self.graph_changed.emit()

    def _on_layout_changed(self, idx: int):
        """Handle layout type change."""
        layout_type = str(self.combo_layout.currentText())
        
        # Stop any running animation
        if self._layout_timer.isActive():
            self._layout_timer.stop()
        
        if "Circular" in layout_type:
            self._apply_circular_layout()
        elif "Grid" in layout_type:
            self._apply_grid_layout()
        elif "Force-Directed" in layout_type:
            # Only animate force-directed if user explicitly requests it
            self._reset_layout()

    def _reset_layout(self):
        """Reset layout with force-directed algorithm."""
        if self._layout_timer.isActive():
            self._layout_timer.stop()
        
        # Start from circular layout, then apply forces
        self._apply_circular_layout()
        
        # Add small random perturbation to break symmetry
        for entity_id, node in self._nodes_by_id.items():
            dx = random.uniform(-10, 10)
            dy = random.uniform(-10, 10)
            node.setPos(node.pos().x() + dx, node.pos().y() + dy)
            self._velocities[entity_id] = (0.0, 0.0)
        
        # Start physics simulation
        self._layout_timer.start()

    def _step_force_layout(self):
        """Perform one step of force-directed layout."""
        if not self._nodes_by_id:
            self._layout_timer.stop()
            return
        
        num_nodes = len(self._nodes_by_id)
        
        # For graphs with >30 nodes, disable repulsive force (O(n²) is too expensive)
        # and only use attractive forces
        enable_repulsion = num_nodes <= 30
        
        # Apply repulsive forces (all pairs) - only for small graphs
        if enable_repulsion:
            for id1, node1 in self._nodes_by_id.items():
                force_x, force_y = 0.0, 0.0
                
                for id2, node2 in self._nodes_by_id.items():
                    if id1 == id2:
                        continue
                    
                    dx = node1.pos().x() - node2.pos().x()
                    dy = node1.pos().y() - node2.pos().y()
                    dist_sq = dx * dx + dy * dy + 1.0
                    dist = math.sqrt(dist_sq)
                    
                    # Repulsive force
                    if dist > 0:
                        force = self._repulsion_strength / dist_sq
                        force_x += (dx / dist) * force
                        force_y += (dy / dist) * force
                
                # Apply attractive forces (connected edges)
                if hasattr(node1, '_edges'):
                    for edge in node1._edges:
                        other = edge.dst_node if edge.src_node == node1 else edge.src_node
                        dx = other.pos().x() - node1.pos().x()
                        dy = other.pos().y() - node1.pos().y()
                        dist = math.sqrt(dx * dx + dy * dy + 1.0)
                        
                        # Attractive force (softer: logarithmic)
                        if dist > 0:
                            force = self._attraction_strength * math.log(dist + 1.0)
                            force_x += (dx / dist) * force
                            force_y += (dy / dist) * force
                
                # Update velocity and position
                vx, vy = self._velocities.get(id1, (0.0, 0.0))
                vx = (vx + force_x) * self._damping
                vy = (vy + force_y) * self._damping
                
                # Clamp velocity
                speed = math.sqrt(vx * vx + vy * vy)
                if speed > self._max_velocity:
                    vx = (vx / speed) * self._max_velocity
                    vy = (vy / speed) * self._max_velocity
                
                self._velocities[id1] = (vx, vy)
                node1.setPos(node1.pos().x() + vx, node1.pos().y() + vy)
        else:
            # For large graphs, only apply weak attractive forces between edges
            for id1, node1 in self._nodes_by_id.items():
                force_x, force_y = 0.0, 0.0
                
                if hasattr(node1, '_edges'):
                    for edge in node1._edges:
                        other = edge.dst_node if edge.src_node == node1 else edge.src_node
                        dx = other.pos().x() - node1.pos().x()
                        dy = other.pos().y() - node1.pos().y()
                        dist = math.sqrt(dx * dx + dy * dy + 1.0)
                        
                        # Very weak attractive force
                        if dist > 0:
                            force = self._attraction_strength * 0.5 * math.log(dist + 1.0)
                            force_x += (dx / dist) * force
                            force_y += (dy / dist) * force
                
                vx, vy = self._velocities.get(id1, (0.0, 0.0))
                vx = (vx + force_x) * self._damping
                vy = (vy + force_y) * self._damping
                
                speed = math.sqrt(vx * vx + vy * vy)
                if speed > self._max_velocity * 0.3:  # Lower max velocity for large graphs
                    vx = (vx / speed) * (self._max_velocity * 0.3)
                    vy = (vy / speed) * (self._max_velocity * 0.3)
                
                self._velocities[id1] = (vx, vy)
                node1.setPos(node1.pos().x() + vx, node1.pos().y() + vy)
        
        # Stop after convergence
        max_speed = max(math.sqrt(vx * vx + vy * vy) for vx, vy in self._velocities.values())
        if max_speed < self._convergence_threshold:
            self._layout_timer.stop()

    def _apply_grid_layout(self):
        """Arrange nodes in a grid with adaptive spacing."""
        self._layout_timer.stop()
        
        node_list = list(self._nodes_by_id.values())
        if not node_list:
            return
        
        num_nodes = len(node_list)
        # For large graphs, use more columns to keep grid compact
        cols = max(4, int(math.ceil(math.sqrt(num_nodes * 1.2))))
        rows = (num_nodes + cols - 1) // cols
        
        # Adaptive spacing based on node count
        base_spacing = 120
        spacing = base_spacing + (num_nodes // 10) * 10
        
        for i, node in enumerate(node_list):
            col = i % cols
            row = i // cols
            x = col * spacing - (cols - 1) * spacing / 2
            y = row * spacing - (rows - 1) * spacing / 2
            node.setPos(x, y)
            self._velocities[node.entity_id] = (0.0, 0.0)

    def _apply_circular_layout(self):
        """Arrange nodes in a circle with adaptive radius."""
        self._layout_timer.stop()
        
        node_list = list(self._nodes_by_id.values())
        if not node_list:
            return
        
        # Adaptive radius based on node count
        num_nodes = len(node_list)
        base_radius = 150
        # Scale radius by sqrt(num_nodes) to accommodate large graphs
        radius = base_radius + (num_nodes ** 0.5) * 20
        
        angle_step = 2 * math.pi / num_nodes
        
        for i, node in enumerate(node_list):
            angle = i * angle_step
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            node.setPos(x, y)
            self._velocities[node.entity_id] = (0.0, 0.0)

    def _fit_to_view(self):
        """Fit all items to the current view."""
        if not self._nodes_by_id:
            return
        
        # Get bounding rect of all items
        rect = self.scene.itemsBoundingRect()
        if rect.isEmpty():
            return
        
        # Add padding
        padding = 50
        rect.adjust(-padding, -padding, padding, padding)
        
        # Fit to view with some margin
        self.view.fitInView(rect, Qt.KeepAspectRatio)
        self.view.scale(0.95, 0.95)  # Leave 5% margin

    def _update_status(self):
        """Update status label."""
        num_nodes = len(self._nodes_by_id)
        num_edges = len(self._edges_by_id)
        self.label_status.setText(f"Nodes: {num_nodes} | Edges: {num_edges}")

    def get_current_graph(self) -> Dict:
        """Export current graph state with updated positions."""
        graph = json.loads(json.dumps(self._graph_data))
        
        # Update node positions
        nodes = list(graph.get("nodes") or [])
        for i, node in enumerate(nodes):
            entity_id = str(node.get("entity_id", ""))
            if entity_id in self._nodes_by_id:
                gnode = self._nodes_by_id[entity_id]
                # Store position (optional, for future use)
                node["_visual_pos"] = [gnode.pos().x(), gnode.pos().y()]
        
        return graph

    def sync_from_table(self, graph_data: Dict):
        """Sync visualization from updated table data."""
        # Update properties for existing nodes
        nodes = list(graph_data.get("nodes") or [])
        for node in nodes:
            entity_id = str(node.get("entity_id", ""))
            if entity_id in self._nodes_by_id:
                gnode = self._nodes_by_id[entity_id]
                gnode.label = str(node.get("canonical_label", gnode.label))
                gnode.confidence = float(node.get("score", gnode.confidence))
                gnode.update()

    def set_selected_node(self, entity_id: Optional[str]) -> None:
        """Set selected node by entity_id (for external sync)."""
        if entity_id is None:
            for node in self._nodes_by_id.values():
                node.set_selected(False)
            return

        if entity_id in self._nodes_by_id:
            for nid, node in self._nodes_by_id.items():
                node.set_selected(nid == entity_id)
            for edge in self._edges_by_id.values():
                edge.set_selected(False)

    def set_selected_edge(self, edge_id: Optional[str]) -> None:
        """Set selected edge by edge_id (for external sync)."""
        if edge_id is None:
            for edge in self._edges_by_id.values():
                edge.set_selected(False)
            return

        if edge_id in self._edges_by_id:
            for node in self._nodes_by_id.values():
                node.set_selected(False)
            for eid, edge in self._edges_by_id.items():
                edge.set_selected(eid == edge_id)
