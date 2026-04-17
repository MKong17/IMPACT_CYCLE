from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

import cv2

from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.impact_sg.pipeline import run_build_scene_graph, run_generate_vqa


class ImpactSGPanel(QWidget):
    """Minimal visual panel for IMPACT-SG build + VQA generation."""

    def __init__(
        self,
        parent=None,
        tasks: Optional[List[str]] = None,
        on_switch_task: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(parent)
        self._task_items = list(tasks or [])
        self._on_switch_task = on_switch_task
        self._repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self._default_ontology = os.path.join(self._repo_root, "configs", "impact_sg_ontology.json")
        self._default_pipeline = os.path.join(self._repo_root, "configs", "impact_sg_pipeline.json")
        self._cache_frame_dir = os.path.join(self._repo_root, ".cache", "impact_sg", "frames")
        self._current_graph: Optional[Dict[str, object]] = None
        self._current_vqa: Optional[Dict[str, object]] = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        if self._task_items:
            task_row = QHBoxLayout()
            task_row.addWidget(QLabel("Task"))
            self.combo_task = QComboBox()
            self.combo_task.addItems(self._task_items)
            self.combo_task.currentTextChanged.connect(self._emit_task_changed)
            task_row.addWidget(self.combo_task)
            task_row.addStretch(1)
            root.addLayout(task_row)

        cfg_box = QGroupBox("IMPACT-SG Inputs")
        cfg_grid = QGridLayout(cfg_box)

        self.video_path_edit = QLineEdit()
        self.frame_idx_spin = QSpinBox()
        self.frame_idx_spin.setRange(0, 1000000)
        self.frame_idx_spin.setValue(0)
        self.image_path_edit = QLineEdit()
        self.image_id_edit = QLineEdit("IMG_001")
        self.ontology_edit = QLineEdit(self._default_ontology)
        self.pipeline_edit = QLineEdit(self._default_pipeline)

        self.w_spin = QSpinBox()
        self.h_spin = QSpinBox()
        for spin in (self.w_spin, self.h_spin):
            spin.setRange(1, 10000)
        self.w_spin.setValue(1280)
        self.h_spin.setValue(720)

        btn_browse_video = QPushButton("Browse Video")
        btn_browse_video.clicked.connect(self._pick_video)
        btn_extract_frame = QPushButton("Extract Frame -> Image")
        btn_extract_frame.clicked.connect(self._extract_frame_from_video)
        btn_browse_image = QPushButton("Browse Image")
        btn_browse_image.clicked.connect(self._pick_image)
        btn_browse_ontology = QPushButton("Browse Ontology")
        btn_browse_ontology.clicked.connect(lambda: self._pick_file(self.ontology_edit, "JSON Files (*.json)"))
        btn_browse_pipeline = QPushButton("Browse Pipeline")
        btn_browse_pipeline.clicked.connect(lambda: self._pick_file(self.pipeline_edit, "JSON Files (*.json)"))

        cfg_grid.addWidget(QLabel("Video"), 0, 0)
        cfg_grid.addWidget(self.video_path_edit, 0, 1)
        cfg_grid.addWidget(btn_browse_video, 0, 2)

        cfg_grid.addWidget(QLabel("Frame Index"), 1, 0)
        cfg_grid.addWidget(self.frame_idx_spin, 1, 1)
        cfg_grid.addWidget(btn_extract_frame, 1, 2)

        cfg_grid.addWidget(QLabel("Image"), 2, 0)
        cfg_grid.addWidget(self.image_path_edit, 2, 1)
        cfg_grid.addWidget(btn_browse_image, 2, 2)

        cfg_grid.addWidget(QLabel("Image ID"), 3, 0)
        cfg_grid.addWidget(self.image_id_edit, 3, 1)

        cfg_grid.addWidget(QLabel("Width"), 4, 0)
        cfg_grid.addWidget(self.w_spin, 4, 1)
        cfg_grid.addWidget(QLabel("Height"), 4, 2)
        cfg_grid.addWidget(self.h_spin, 4, 3)

        cfg_grid.addWidget(QLabel("Ontology"), 5, 0)
        cfg_grid.addWidget(self.ontology_edit, 5, 1)
        cfg_grid.addWidget(btn_browse_ontology, 5, 2)

        cfg_grid.addWidget(QLabel("Pipeline Config"), 6, 0)
        cfg_grid.addWidget(self.pipeline_edit, 6, 1)
        cfg_grid.addWidget(btn_browse_pipeline, 6, 2)

        root.addWidget(cfg_box)

        action_row = QHBoxLayout()
        btn_build = QPushButton("Build Scene Graph")
        btn_build.clicked.connect(self._on_build_graph)
        btn_load_graph = QPushButton("Load Graph JSON")
        btn_load_graph.clicked.connect(self._on_load_graph)
        btn_save_graph = QPushButton("Save Graph JSON")
        btn_save_graph.clicked.connect(self._on_save_graph)
        btn_generate_vqa = QPushButton("Generate VQA")
        btn_generate_vqa.clicked.connect(self._on_generate_vqa)
        btn_save_vqa = QPushButton("Save VQA JSON")
        btn_save_vqa.clicked.connect(self._on_save_vqa)

        action_row.addWidget(btn_build)
        action_row.addWidget(btn_load_graph)
        action_row.addWidget(btn_save_graph)
        action_row.addWidget(btn_generate_vqa)
        action_row.addWidget(btn_save_vqa)
        root.addLayout(action_row)

        self.summary_label = QLabel("Ready. Build or load a scene graph to start.")
        root.addWidget(self.summary_label)

        self.output_edit = QTextEdit()
        self.output_edit.setReadOnly(True)
        root.addWidget(self.output_edit)

    def _pick_file(self, target_edit: QLineEdit, filter_text: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select File", self._repo_root, filter_text)
        if path:
            target_edit.setText(path)

    def _pick_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            self._repo_root,
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not path:
            return
        self.image_path_edit.setText(path)
        base = os.path.splitext(os.path.basename(path))[0]
        if base:
            self.image_id_edit.setText(base)

        img = QImage(path)
        if not img.isNull():
            self.w_spin.setValue(max(1, img.width()))
            self.h_spin.setValue(max(1, img.height()))

    def _pick_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            self._repo_root,
            "Videos (*.mp4 *.avi *.mov *.mkv *.m4v)",
        )
        if path:
            self.video_path_edit.setText(path)

    def _extract_frame_from_video(self) -> None:
        video_path = self.video_path_edit.text().strip()
        if not os.path.isfile(video_path):
            QMessageBox.warning(self, "Missing File", "Please choose a valid video path.")
            return

        frame_idx = int(self.frame_idx_spin.value())
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Open Failed", "Unable to open video file.")
            return

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0:
                frame_idx = max(0, min(frame_idx, total - 1))
                self.frame_idx_spin.setValue(frame_idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
        finally:
            cap.release()

        if not ok or frame is None:
            QMessageBox.critical(self, "Read Failed", f"Unable to read frame {frame_idx}.")
            return

        os.makedirs(self._cache_frame_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_name = f"{stem}_f{frame_idx:06d}.jpg"
        out_path = os.path.join(self._cache_frame_dir, out_name)
        ok_write = cv2.imwrite(out_path, frame)
        if not ok_write:
            QMessageBox.critical(self, "Write Failed", "Unable to save extracted frame image.")
            return

        h, w = frame.shape[:2]
        self.image_path_edit.setText(out_path)
        self.image_id_edit.setText(f"{stem}_f{frame_idx:06d}")
        self.w_spin.setValue(int(w))
        self.h_spin.setValue(int(h))
        self.summary_label.setText(f"Extracted frame {frame_idx} from video. Ready to build scene graph.")

    def _emit_task_changed(self, text: str) -> None:
        if callable(self._on_switch_task):
            self._on_switch_task(text)

    def set_task(self, text: str) -> None:
        if not getattr(self, "combo_task", None):
            return
        try:
            self.combo_task.blockSignals(True)
            self.combo_task.setCurrentText(text)
            self.combo_task.blockSignals(False)
        except Exception:
            pass

    def _validate_input_paths(self) -> bool:
        image_path = self.image_path_edit.text().strip()
        ontology_path = self.ontology_edit.text().strip()
        pipeline_path = self.pipeline_edit.text().strip()

        if not os.path.isfile(image_path):
            QMessageBox.warning(self, "Missing File", "Please choose a valid image path.")
            return False
        if not os.path.isfile(ontology_path):
            QMessageBox.warning(self, "Missing File", "Please choose a valid ontology JSON file.")
            return False
        if not os.path.isfile(pipeline_path):
            QMessageBox.warning(self, "Missing File", "Please choose a valid pipeline config JSON file.")
            return False
        return True

    def _on_build_graph(self) -> None:
        if not self._validate_input_paths():
            return
        try:
            graph = run_build_scene_graph(
                image_id=self.image_id_edit.text().strip() or "IMG_001",
                image_path=self.image_path_edit.text().strip(),
                ontology_path=self.ontology_edit.text().strip(),
                pipeline_cfg_path=self.pipeline_edit.text().strip(),
                image_size=(int(self.w_spin.value()), int(self.h_spin.value())),
                enable_sentence_refine=False,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Build Failed", f"Failed to build scene graph:\n{exc}")
            return

        self._current_graph = graph
        self._current_vqa = None
        self._render_graph_summary(graph)
        self.output_edit.setPlainText(json.dumps(graph, ensure_ascii=True, indent=2))

    def _on_load_graph(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Scene Graph", self._repo_root, "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                graph = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", f"Failed to load JSON:\n{exc}")
            return

        self._current_graph = graph
        self._current_vqa = None
        self._render_graph_summary(graph)
        self.output_edit.setPlainText(json.dumps(graph, ensure_ascii=True, indent=2))

    def _on_save_graph(self) -> None:
        if not self._current_graph:
            QMessageBox.information(self, "No Data", "No scene graph available to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Scene Graph", self._repo_root, "JSON Files (*.json)")
        if not path:
            return
        self._save_json(path, self._current_graph)

    def _on_generate_vqa(self) -> None:
        if not self._current_graph:
            QMessageBox.information(self, "No Graph", "Build or load a scene graph first.")
            return

        pipeline_path = self.pipeline_edit.text().strip()
        if not os.path.isfile(pipeline_path):
            QMessageBox.warning(self, "Missing File", "Please choose a valid pipeline config JSON file.")
            return

        try:
            vqa = run_generate_vqa(self._current_graph, pipeline_path)
        except Exception as exc:
            QMessageBox.critical(self, "Generate Failed", f"Failed to generate VQA:\n{exc}")
            return

        self._current_vqa = vqa
        all_items = vqa.get("all") or []
        review_items = vqa.get("review_queue") or []
        self.summary_label.setText(
            f"VQA generated. all_items={len(all_items)} review_queue={len(review_items)}"
        )
        self.output_edit.setPlainText(json.dumps(vqa, ensure_ascii=True, indent=2))

    def _on_save_vqa(self) -> None:
        if not self._current_vqa:
            QMessageBox.information(self, "No Data", "No VQA data available to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save VQA", self._repo_root, "JSON Files (*.json)")
        if not path:
            return
        self._save_json(path, self._current_vqa)

    def _save_json(self, path: str, payload: Dict[str, object]) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Failed to save JSON:\n{exc}")
            return
        QMessageBox.information(self, "Saved", f"Saved JSON to:\n{path}")

    def _render_graph_summary(self, graph: Dict[str, object]) -> None:
        nodes = graph.get("nodes") or []
        edges = graph.get("edges") or []
        image_id = graph.get("image_id") or "N/A"
        self.summary_label.setText(f"Scene graph ready. image_id={image_id} nodes={len(nodes)} edges={len(edges)}")
