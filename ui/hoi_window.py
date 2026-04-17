from typing import List, Dict, Optional
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QToolButton,
    QSpinBox,
    QSlider,
    QCheckBox,
    QShortcut,
    QMenu,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QSplitter,
    QListView,
    QSizePolicy,
    QInputDialog,
    QFrame,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialogButtonBox,
    QRadioButton,
)
from ui.mixins import FrameControlMixin
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QStyle
from PyQt5.QtGui import QKeySequence, QColor
import copy
import json
import hashlib
from ui.video_player import VideoPlayer
from ui.label_panel import LabelPanel
from core.models import LabelDef
from utils.constants import PRESET_COLORS, color_from_key
from utils.shortcut_settings import (
    load_shortcut_bindings,
    default_shortcut_bindings,
    shortcut_value,
    set_shortcut_key,
)
from utils.op_logger import OperationLogger
from ui.hoi_timeline import HOITimeline
from ui.widgets import ToggleSwitch, ClickToggleList
import os
import cv2
import re
from datetime import datetime


class HOIWindow(FrameControlMixin, QWidget):
    """
    HandOI/HOI detection annotator:
    - Single video.
    - Load YOLO bbox text (per-line: frame class xc yc w h [conf]; normalized if <=1).
    - Overlay bboxes on video; show current-frame boxes in a list.
    - Define verbs via LabelPanel; select two boxes + a verb to create/delete relations.
    """

    def __init__(
        self,
        parent=None,
        on_close=None,
        on_switch_task=None,
        tasks: List[str] = None,
        logger: OperationLogger = None,
    ):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setWindowTitle("HandOI / HOI Detection")
        self.resize(1100, 720)
        self._on_close = on_close
        self._on_switch_task = on_switch_task
        self.op_logger = logger or OperationLogger(False)
        self._shortcut_bindings = load_shortcut_bindings()
        self._shortcut_defaults = default_shortcut_bindings()

        self.player = VideoPlayer()
        self.player.on_frame_advanced = self._on_frame_advanced
        self.player.on_playback_state_changed = self._on_player_playback_state_changed

        # data
        self.bboxes: Dict[int, List[dict]] = {}  # frame -> list of boxes
        self.box_id_counter = 1
        self.verbs: List[LabelDef] = []
        self.current_verb_idx = -1
        self.events: List[dict] = []  # Store all events (internal event structure)
        self.event_id_counter = 0  # Event ID counter
        self._reset_event_draft()  # Initialize event draft structure
        self.class_map: Dict[int, str] = {}
        self.labels_dir: str = ""
        self.start_offset = 0
        self.end_frame = None  # optional clamp
        self.raw_boxes: List[dict] = []  # keep raw frame indices for re-mapping
        self.current_hands = {"Left_hand": None, "Right_hand": None}
        self.selected_hand_label = None  # "Left_hand"|"Right_hand"|None
        self.selected_event_id = None
        self._required_loaded = False
        self.video_path = ""
        # YOLO current-frame detection
        self.yolo_model = None
        self.yolo_weights_path = ""
        self.yolo_conf = 0.25
        self.yolo_iou = 0.45
        self.yolo_existing_policy = None  # "append" or "replace"
        self.mp_hands = None
        self.mp_hands_error = None
        self.mp_hands_max = 2
        self.mp_hands_conf = 0.5
        self.mp_hands_track_conf = 0.5
        self.mp_hands_swap = False
        self._hoi_undo_stack = []
        self._hoi_redo_stack = []
        self._undo_block = False
        self._undo_limit = 50
        self.validation_enabled = False
        self.validation_session_active = False
        self.validation_started_at = None
        self.validation_modified = False
        self.validation_change_count = 0
        self.validator_name = ""
        self.validation_summary_enabled = True
        self.validation_op_logger = OperationLogger(self.op_logger.enabled)
        self._validation_highlights = {}
        self._hoi_saved_signature = None

        # controls (reuse toolbar look/logic)
        controls = QHBoxLayout()
        # task switch for consistency / return
        lbl_task = QLabel("Task:")
        lbl_task.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls.addWidget(lbl_task)
        self.combo_task = QComboBox()
        items = tasks or ["Action Segmentation", "HandOI / HOI Detection"]
        self.combo_task.addItems(items)
        self.combo_task.setCurrentText("HandOI / HOI Detection")
        self.combo_task.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_task.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.combo_task.currentTextChanged.connect(self._on_task_combo_changed)
        controls.addWidget(self.combo_task)
        controls.addSpacing(8)

        # consolidated file menu
        self.file_menu = QMenu(self)
        self.file_menu.addAction("Load Video...", self._load_video)
        self.file_menu.addSeparator()
        self.file_menu.addAction("Import Instrument List...", self._import_instruments)
        self.file_menu.addAction("Import Target List...", self._import_targets)
        self.file_menu.addSeparator()
        self.file_menu.addAction("Load Class Map (data.yaml)...", self._load_yaml)
        self.file_menu.addAction("Import YOLO Boxes...", self._load_bboxes)
        self.file_menu.addSeparator()
        self.file_menu.addAction("Load YOLO Model...", self._load_yolo_model)
        self.file_menu.addAction(
            "Detect Current Frame", self._detect_current_frame_combined
        )
        self.file_menu.addAction(
            "Detect Selected Action", self._detect_selected_action
        )
        self.file_menu.addAction("Detect All Actions", self._detect_all_actions)
        self.file_menu.addAction("Load Hands XML...", self._load_hands_xml)
        self.file_menu.addAction("Import Verb List...", self._load_verbs_txt)
        self.file_menu.addAction(
            "Load HOI Annotations...", self._load_annotations_json
        )
        self.file_menu.addSeparator()
        self.file_menu.addAction("Save HOI Annotations...", self._save_annotations_json)
        self.file_menu.addAction("Export Hands XML...", self._save_hands_xml)
        self.btn_file_menu = QToolButton()
        self.btn_file_menu.setText("Files")
        self.btn_file_menu.setPopupMode(QToolButton.InstantPopup)
        self.btn_file_menu.setMenu(self.file_menu)
        self.btn_rew = QToolButton()
        self.btn_rew.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.btn_play = QToolButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setToolTip("Play")
        self.btn_stop = QToolButton()
        self.btn_stop.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.btn_ff = QToolButton()
        self.btn_ff.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        for b in (
            self.btn_rew,
            self.btn_play,
            self.btn_stop,
            self.btn_ff,
        ):
            b.setToolButtonStyle(Qt.ToolButtonIconOnly)
            b.setIconSize(QSize(22, 22))
        self.spin_jump = QSpinBox()
        self.spin_jump.setMinimum(0)
        self.spin_jump.setMaximum(0)
        self.spin_jump.setKeyboardTracking(False)
        self.btn_jump = QPushButton("Go")

        self.btn_rew.clicked.connect(lambda: self._seek_relative(-10))
        self.btn_ff.clicked.connect(lambda: self._seek_relative(+10))
        self.btn_play.clicked.connect(self._toggle_play_pause)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_jump.clicked.connect(self._jump_to_spin)

        self.combo_verb = QComboBox()
        self.combo_verb.setMinimumWidth(120)
        self.combo_verb.setEditable(True)
        # order mirrors the Action Segmentation toolbar for consistency
        controls.addWidget(self.btn_rew)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_ff)
        controls.addSpacing(12)
        label_width = 48
        lbl_jump = QLabel("Jump")
        lbl_jump.setFixedWidth(label_width)
        lbl_jump.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        controls.addWidget(lbl_jump)
        controls.addWidget(self.spin_jump)
        controls.addWidget(self.btn_jump)
        controls.addSpacing(12)
        # compact start/end controls inline
        lbl_start = QLabel("Start")
        lbl_start.setFixedWidth(label_width)
        lbl_start.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        controls.addWidget(lbl_start)
        self.spin_start_offset = QSpinBox()
        self.spin_start_offset.setMinimum(0)
        self.spin_start_offset.valueChanged.connect(self._on_offset_changed)
        controls.addWidget(self.spin_start_offset)
        lbl_end = QLabel("End")
        lbl_end.setFixedWidth(label_width)
        lbl_end.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        controls.addWidget(lbl_end)
        self.spin_end_frame = QSpinBox()
        self.spin_end_frame.setMinimum(0)
        self.spin_end_frame.setMaximum(10**9)
        self.spin_end_frame.valueChanged.connect(self._on_offset_changed)
        controls.addWidget(self.spin_end_frame)
        controls.addSpacing(12)
        controls.addWidget(self.btn_file_menu)
        self.btn_detect = QToolButton()
        self.btn_detect.setText("Detect Frame")
        self.btn_detect.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.btn_detect.setToolTip(
            "Detect objects + hands on current frame (Ctrl+Shift+D)"
        )
        self.btn_detect.clicked.connect(self._detect_current_frame_combined)
        controls.addWidget(self.btn_detect)
        self.btn_detect_action = QToolButton()
        self.btn_detect_action.setText("Detect Action")
        self.btn_detect_action.setToolTip(
            "Detect assigned objects on the selected action's start, onset, and end frames."
        )
        self.btn_detect_action.clicked.connect(self._detect_selected_action)
        controls.addWidget(self.btn_detect_action)
        self.btn_detect_all = QToolButton()
        self.btn_detect_all.setText("Detect All")
        self.btn_detect_all.setToolTip(
            "Detect assigned objects across every action keyframe."
        )
        self.btn_detect_all.clicked.connect(self._detect_all_actions)
        controls.addWidget(self.btn_detect_all)
        self.chk_swap_hands = QCheckBox("Auto swap L/R")
        self.chk_swap_hands.setToolTip(
            "Swap Left/Right labels for MediaPipe detections (mirrored video). Affects new detections only."
        )
        self.chk_swap_hands.toggled.connect(
            lambda on: setattr(self, "mp_hands_swap", on)
        )
        controls.addWidget(self.chk_swap_hands)
        self.chk_edit_boxes = QCheckBox("Edit boxes")
        self.chk_edit_boxes.toggled.connect(self._on_edit_boxes_toggled)
        controls.addWidget(self.chk_edit_boxes)
        controls.addSpacing(6)
        controls.addWidget(QLabel("Auto label"))
        self.rad_draw_none = QRadioButton("Manual")
        self.rad_draw_inst = QRadioButton("Instrument")
        self.rad_draw_target = QRadioButton("Target")
        self.rad_draw_none.setChecked(True)
        self.rad_draw_none.setToolTip("Draw boxes and enter labels manually.")
        self.rad_draw_inst.setToolTip(
            "New boxes inherit the selected hand's instrument label."
        )
        self.rad_draw_target.setToolTip(
            "New boxes inherit the selected hand's target label."
        )
        controls.addWidget(self.rad_draw_none)
        controls.addWidget(self.rad_draw_inst)
        controls.addWidget(self.rad_draw_target)
        for widget in (self.rad_draw_none, self.rad_draw_inst, self.rad_draw_target):
            widget.setEnabled(False)
        controls.addSpacing(8)
        self.sep_edit_validation = QFrame()
        self.sep_edit_validation.setFrameShape(QFrame.VLine)
        self.sep_edit_validation.setFrameShadow(QFrame.Sunken)
        self.sep_edit_validation.setLineWidth(1)
        self.sep_edit_validation.setFixedHeight(18)
        controls.addWidget(self.sep_edit_validation)
        controls.addSpacing(8)
        self.lbl_validation = QLabel("Validation")
        self.lbl_validation.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        controls.addWidget(self.lbl_validation)
        self.btn_validation = ToggleSwitch(self)
        self.btn_validation.setToolTip("Toggle validation on/off")
        self.btn_validation.toggled.connect(self._on_validation_toggled)
        controls.addWidget(self.btn_validation)
        controls.addStretch(1)

        # video row (match main layout: video on left, panels on right), controls below
        video_row = QHBoxLayout()
        video_row.setContentsMargins(0, 0, 0, 0)
        video_row.setSpacing(8)
        video_row.addWidget(self.player, 3)

        right_col = QVBoxLayout()
        self.list_objects = QListWidget()
        self.list_objects.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_objects.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_objects.customContextMenuRequested.connect(self._on_object_list_menu)
        self.list_objects.show()
        self.list_objects.itemSelectionChanged.connect(self._on_object_selection)
        self.list_objects.setEnabled(False)
        right_col.addWidget(QLabel("Objects (current frame):"))
        right_col.addWidget(self.list_objects, 2)

        self.group_anomaly = QGroupBox("Hand Anomaly Label")
        self.anomaly_labels = [
            "Visibility / Occlusion",
            "Handling Anomaly",
            "Tool Anomaly",
            "Part Mismatch",
            "Spatial Anomaly",
            "Temporal Anomaly",
            "Quality / Outcome-visible Defect",
            "Completion / Finish Missing",
            "Detach / Fall-out",
            "Recovery / Rework",
            "Normal",
        ]
        self.anomaly_rules = {}
        self._init_anomaly_rules()
        self.anomaly_list = ClickToggleList()
        self.anomaly_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.anomaly_list.setFlow(QListView.LeftToRight)
        self.anomaly_list.setWrapping(True)
        self.anomaly_list.setResizeMode(QListView.Adjust)
        self.anomaly_list.setSpacing(10)
        self.anomaly_list.setStyleSheet(
            "QListWidget::item { padding: 3px 10px; }"
            "QListWidget::indicator { width: 14px; height: 14px; }"
        )
        self.anomaly_list.itemChanged.connect(self._on_anomaly_item_changed)
        self.anomaly_list.setWordWrap(True)
        self.anomaly_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._anomaly_block = False
        for name in self.anomaly_labels:
            self._add_anomaly_item(name, checked=(name.strip().lower() == "normal"))

        self.anomaly_edit = QLineEdit(self.group_anomaly)
        self.anomaly_edit.setPlaceholderText("New label")
        self.btn_anomaly_add = QPushButton("Add")
        self.btn_anomaly_remove = QPushButton("Remove")
        self.btn_anomaly_rules = QPushButton("Rules")
        self.btn_anomaly_add.clicked.connect(self._add_anomaly_label)
        self.btn_anomaly_remove.clicked.connect(self._remove_anomaly_label)
        self.btn_anomaly_rules.clicked.connect(self._edit_anomaly_rules)

        anomaly_layout = QVBoxLayout()
        anomaly_layout.addWidget(self.anomaly_list)
        row = QHBoxLayout()
        row.addWidget(self.anomaly_edit, 1)
        row.addWidget(self.btn_anomaly_add)
        row.addWidget(self.btn_anomaly_remove)
        row.addWidget(self.btn_anomaly_rules)
        anomaly_layout.addLayout(row)
        self.group_anomaly.setLayout(anomaly_layout)

        hand_row = QHBoxLayout()
        hand_row.addWidget(QLabel("Hands:"))
        self.chk_left = QCheckBox("L")
        self.chk_left.setToolTip("Left hand")
        self.chk_right = QCheckBox("R")
        self.chk_right.setToolTip("Right hand")
        self.chk_left.toggled.connect(lambda on: self._select_hand("Left_hand", on))
        self.chk_right.toggled.connect(lambda on: self._select_hand("Right_hand", on))
        hand_row.addWidget(self.chk_left)
        hand_row.addWidget(self.chk_right)

        self.btn_swap_draft = QPushButton("Swap")
        self.btn_swap_draft.setToolTip(
            "Swap Left/Right hand boxes on the current frame."
        )
        self.btn_swap_draft.setStyleSheet("color: #111;")
        self.btn_swap_draft.clicked.connect(self._swap_frame_hands)

        hand_row.addSpacing(10)
        hand_row.addWidget(self.btn_swap_draft)

        hand_row.addStretch(1)

        # Initialize Verbs panel
        self.label_panel = LabelPanel(
            self.verbs,
            on_add=self._on_verb_added,
            on_remove_idx=self._on_verb_removed,
            on_rename=self._on_verb_renamed,
            on_search_matches=None,
            on_select_idx=self._on_verb_selected,
            manage_storage=False,
        )

        # Modify placeholder text: Label -> Verb
        for le in self.label_panel.findChildren(QLineEdit):
            ph = le.placeholderText().lower()
            if "search" in ph:
                le.setPlaceholderText("Search verb")
            elif "new" in ph:
                le.setPlaceholderText("New verb name")
        self._init_verb_color_combo()
        self.label_panel.set_verb_only(True)

        self.group_library = QGroupBox("Entity Library")
        lib_layout = QFormLayout()

        self.combo_instrument = QComboBox()
        self.combo_target = QComboBox()
        self.combo_instrument.addItem("None", None)
        self.combo_target.addItem("None", None)

        self._enable_combo_search(
            self.combo_instrument, placeholder="Search instrument..."
        )
        self._enable_combo_search(self.combo_target, placeholder="Search target...")

        lib_layout.addRow("Instrument:", self.combo_instrument)
        lib_layout.addRow("Target:", self.combo_target)

        self.group_library.setLayout(lib_layout)

        right_panel = QWidget()
        right_panel.setLayout(right_col)
        right_panel.setMaximumWidth(450)
        video_row.addWidget(right_panel, 1)

        root = QVBoxLayout(self)

        top_block = QWidget(self)
        top_layout = QVBoxLayout(top_block)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(4)
        top_layout.addLayout(video_row, 1)
        top_layout.addLayout(controls)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider_changed)
        top_layout.addWidget(self.slider)
        self.hoi_timeline = HOITimeline(
            get_segments_for_hand=self._hoi_segments_for_hand,
            get_color_for_verb=self._hoi_color_for_verb,
            on_select=self._on_hoi_timeline_select,
            on_update=self._on_hoi_timeline_update,
            on_create=self._on_hoi_timeline_create,
            on_delete=self._on_hoi_timeline_delete,
            on_hover=self._on_hoi_timeline_hover,
            get_frame_count=lambda: int(self.player.frame_count or 0),
            get_fps=lambda: int(self.player.frame_rate or 30),
            get_title_for_hand=self._hoi_title_for_hand,
            parent=self,
        )
        self.lbl_incomplete = QLabel("Incomplete: n/a")
        self.lbl_incomplete.setToolTip("No HandOI completeness check available yet.")
        self.btn_incomplete_prev = QToolButton()
        self.btn_incomplete_prev.setText("<")
        self.btn_incomplete_prev.setToolTip("Previous incomplete segment (A)")
        self.btn_incomplete_prev.clicked.connect(lambda: self._jump_incomplete(-1))
        self.btn_incomplete_next = QToolButton()
        self.btn_incomplete_next.setText(">")
        self.btn_incomplete_next.setToolTip("Next incomplete segment (D)")
        self.btn_incomplete_next.clicked.connect(lambda: self._jump_incomplete(+1))
        self.btn_incomplete_prev.setEnabled(False)
        self.btn_incomplete_next.setEnabled(False)
        self._incomplete_issues = []
        self._incomplete_idx = -1

        timeline_container = QWidget()
        timeline_layout = QVBoxLayout(timeline_container)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(4)
        timeline_layout.addWidget(self.hoi_timeline, 1)
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addStretch(1)
        footer.addWidget(self.lbl_incomplete)
        footer.addSpacing(6)
        footer.addWidget(self.btn_incomplete_prev)
        footer.addWidget(self.btn_incomplete_next)
        timeline_layout.addLayout(footer)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        hand_widget = QWidget()
        hand_layout = QHBoxLayout(hand_widget)
        hand_layout.setContentsMargins(0, 0, 0, 0)
        hand_layout.addLayout(hand_row)

        verb_container = QWidget()
        verb_layout = QVBoxLayout(verb_container)
        verb_layout.setContentsMargins(0, 0, 0, 0)
        verb_layout.setSpacing(2)

        self.combo_verb.setPlaceholderText("Select or type verb...")
        verb_layout.addWidget(self.combo_verb)

        verb_layout.addWidget(self.label_panel)

        internal_list = self.label_panel.findChild(QListWidget)
        if internal_list:
            try:
                internal_list.itemDoubleClicked.disconnect()
            except Exception:
                pass
            internal_list.itemDoubleClicked.connect(self._on_verb_double_clicked)

        self.left_panel_split = QSplitter(Qt.Vertical, self)
        self.left_panel_split.setChildrenCollapsible(False)
        self.left_panel_split.setHandleWidth(6)

        self.left_panel_split.addWidget(self.group_library)
        self.left_panel_split.addWidget(hand_widget)
        self.left_panel_split.addWidget(verb_container)

        self.left_panel_split.setStretchFactor(0, 0)
        self.left_panel_split.setStretchFactor(1, 0)
        self.left_panel_split.setStretchFactor(2, 1)
        self.left_panel_split.setSizes([140, 60, 420])

        left_layout.addWidget(self.left_panel_split)
        left_panel.setMinimumWidth(180)

        self.hoi_bottom_split = QSplitter(Qt.Horizontal, self)
        self.hoi_bottom_split.setChildrenCollapsible(False)
        self.hoi_bottom_split.setHandleWidth(6)
        self.timeline_split = QSplitter(Qt.Vertical, self)
        self.timeline_split.setChildrenCollapsible(False)
        self.timeline_split.setHandleWidth(6)
        self.timeline_split.addWidget(self.group_anomaly)
        self.timeline_split.addWidget(timeline_container)
        self.timeline_split.setStretchFactor(0, 2)
        self.timeline_split.setStretchFactor(1, 3)
        self.timeline_split.setSizes([260, 360])

        self.hoi_bottom_split.addWidget(left_panel)
        self.hoi_bottom_split.addWidget(self.timeline_split)
        self.hoi_bottom_split.setStretchFactor(0, 0)
        self.hoi_bottom_split.setStretchFactor(1, 1)
        self.hoi_bottom_split.setSizes([260, 940])
        self.root_split = QSplitter(Qt.Vertical, self)
        self.root_split.setChildrenCollapsible(False)
        self.root_split.setHandleWidth(6)
        self.root_split.addWidget(top_block)
        self.root_split.addWidget(self.hoi_bottom_split)
        self.root_split.setStretchFactor(0, 2)
        self.root_split.setStretchFactor(1, 3)
        self.root_split.setSizes([540, 360])
        root.addWidget(self.root_split, 1)
        self.chk_filter_current = None

        # shortcuts (only within HOI window)
        self.sc_left = QShortcut(
            QKeySequence(Qt.Key_Left), self, activated=lambda: self._seek_relative(-1)
        )
        self.sc_left.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_right = QShortcut(
            QKeySequence(Qt.Key_Right), self, activated=lambda: self._seek_relative(+1)
        )
        self.sc_right.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_up = QShortcut(
            QKeySequence(Qt.Key_Up), self, activated=lambda: self._seek_seconds(-1)
        )
        self.sc_up.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_down = QShortcut(
            QKeySequence(Qt.Key_Down), self, activated=lambda: self._seek_seconds(+1)
        )
        self.sc_down.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_toggle_play = QShortcut(
            QKeySequence(Qt.Key_Space), self, activated=self._toggle_play_pause
        )
        self.sc_toggle_play.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_pause = QShortcut(QKeySequence("K"), self, activated=self._pause)
        self.sc_pause.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_detect = QShortcut(
            QKeySequence("Ctrl+Shift+D"),
            self,
            activated=self._detect_current_frame_combined,
        )
        self.sc_detect.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_jump_start = QShortcut(
            QKeySequence("Z"), self, activated=self._jump_to_selected_start
        )
        self.sc_jump_start.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_jump_onset = QShortcut(
            QKeySequence("X"), self, activated=self._jump_to_selected_onset
        )
        self.sc_jump_onset.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_jump_end = QShortcut(
            QKeySequence("C"), self, activated=self._jump_to_selected_end
        )
        self.sc_jump_end.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_inc_prev = QShortcut(
            QKeySequence("A"), self, activated=lambda: self._jump_incomplete(-1)
        )
        self.sc_inc_prev.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_inc_next = QShortcut(
            QKeySequence("D"), self, activated=lambda: self._jump_incomplete(+1)
        )
        self.sc_inc_next.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_undo = QShortcut(QKeySequence.Undo, self, activated=self._hoi_undo)
        self.sc_undo.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_redo = QShortcut(QKeySequence.Redo, self, activated=self._hoi_redo)
        self.sc_redo.setContext(Qt.WidgetWithChildrenShortcut)
        self.apply_shortcut_settings(self._shortcut_bindings)

        # 3. Safely handle list_objects signals
        try:
            self.list_objects.itemSelectionChanged.disconnect()
        except (TypeError, RuntimeError):
            pass

        # 4. Connect new logic
        self.list_objects.itemSelectionChanged.connect(self._on_object_selection)
        # --- New: Global Object ID Registry ---
        # Structure: { "screwdriver_1": 0, "gear_1": 1, ... }
        self.global_object_map: Dict[str, int] = {}
        self.object_id_counter = 0

        self.entity_library = {}
        self.id_to_category = {}

        self.combo_verb.currentTextChanged.connect(self._on_hoi_meta_changed)
        self.combo_instrument.currentIndexChanged.connect(self._on_hoi_meta_changed)
        self.combo_target.currentIndexChanged.connect(self._on_hoi_meta_changed)
        self._mark_hoi_saved()

        self._update_verb_combo()
        self._update_play_pause_button()

    def _set_shortcut_key(
        self, shortcut: Optional[QShortcut], sid: str, default_key: str
    ) -> None:
        set_shortcut_key(
            shortcut,
            shortcut_value(
                self._shortcut_bindings,
                self._shortcut_defaults,
                sid,
                default_key,
            ),
            default_key,
        )

    def apply_shortcut_settings(
        self, bindings: Optional[Dict[str, str]] = None
    ) -> None:
        self._shortcut_bindings = (
            load_shortcut_bindings() if bindings is None else dict(bindings)
        )
        self._shortcut_defaults = default_shortcut_bindings()
        self._set_shortcut_key(getattr(self, "sc_left", None), "hoi.step_prev", "Left")
        self._set_shortcut_key(
            getattr(self, "sc_right", None), "hoi.step_next", "Right"
        )
        self._set_shortcut_key(
            getattr(self, "sc_up", None), "hoi.seek_prev_second", "Up"
        )
        self._set_shortcut_key(
            getattr(self, "sc_down", None), "hoi.seek_next_second", "Down"
        )
        self._set_shortcut_key(
            getattr(self, "sc_toggle_play", None), "hoi.play_pause", "Space"
        )
        self._set_shortcut_key(getattr(self, "sc_pause", None), "hoi.pause", "K")
        self._set_shortcut_key(
            getattr(self, "sc_detect", None), "hoi.detect", "Ctrl+Shift+D"
        )
        self._set_shortcut_key(getattr(self, "sc_undo", None), "hoi.undo", "Ctrl+Z")
        self._set_shortcut_key(getattr(self, "sc_redo", None), "hoi.redo", "Ctrl+Y")

    # ---------- data helpers ----------
    def _enable_combo_search(self, combo: QComboBox, placeholder: str = "Search..."):
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if combo.lineEdit() is not None:
            combo.lineEdit().setPlaceholderText(placeholder)
        completer = combo.completer()
        if completer is not None:
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            try:
                completer.setFilterMode(Qt.MatchContains)
            except Exception:
                pass

    def _init_verb_color_combo(self):
        combo = self.label_panel.combo
        if combo.findText("Auto") == -1:
            combo.insertItem(0, "Auto")
        combo.setCurrentText("Auto")
        combo.currentTextChanged.connect(self._on_verb_color_changed)

    def _on_verb_color_changed(self, color_key: str):
        if color_key == "Custom":
            return
        selected = self.label_panel.current_label_name()
        if not selected:
            return
        if self.label_panel.edit.text().strip():
            return
        idx = self.label_panel.index_of_label(selected)
        if idx < 0:
            return
        self.verbs[idx].color_name = color_key
        self.label_panel.refresh()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.refresh()

    def _update_verb_combo(self):
        """[Restore] Populate the verb combobox from self.verbs list."""
        current = self.combo_verb.currentText()
        self.combo_verb.blockSignals(True)
        self.combo_verb.clear()

        sorted_verbs = sorted([v.name for v in self.verbs])
        self.combo_verb.addItems(sorted_verbs)

        idx = self.combo_verb.findText(current)
        if idx >= 0:
            self.combo_verb.setCurrentIndex(idx)

        self.combo_verb.blockSignals(False)

    def _hoi_color_for_verb(self, verb: str) -> Optional[QColor]:
        for v in self.verbs:
            if v.name == verb:
                color_name = v.color_name or ""
                if color_name.lower() == "auto":
                    return None
                return color_from_key(color_name)
        return None

    def _find_event_by_id(self, event_id: int) -> Optional[dict]:
        for ev in self.events:
            if ev.get("event_id") == event_id:
                return ev
        return None

    def _compute_event_frames(self, event: dict) -> tuple:
        times = []
        for hand in ("Left_hand", "Right_hand"):
            h = event.get("hoi_data", {}).get(hand, {})
            for key in (
                "interaction_start",
                "functional_contact_onset",
                "interaction_end",
            ):
                val = h.get(key)
                if val is not None:
                    try:
                        times.append(int(val))
                    except Exception:
                        continue
        if not times:
            return None, None
        return min(times), max(times)

    def _sync_event_frames(self, event: dict):
        ws, we = self._compute_event_frames(event)
        event["frames"] = [ws, we]

    def _apply_draft_to_selected_event(self):
        if self.selected_event_id is None:
            return
        ev = self._find_event_by_id(self.selected_event_id)
        if not ev:
            return
        ev["hoi_data"] = {
            "Left_hand": dict(self.event_draft.get("Left_hand", {})),
            "Right_hand": dict(self.event_draft.get("Right_hand", {})),
        }
        self._sync_event_frames(ev)

    def _apply_selected_event(self):
        if self.selected_event_id is None:
            return
        self._save_ui_to_hand_draft(self.selected_hand_label)
        self._apply_draft_to_selected_event()
        self._refresh_events()
        self._update_hoi_titles()
        self.hoi_timeline.refresh()

    def _set_selected_event(self, event_id: int, hand_key: Optional[str] = None):
        if self.selected_event_id == event_id and hand_key == self.selected_hand_label:
            return
        if self.selected_event_id is not None:
            self._save_ui_to_hand_draft(self.selected_hand_label)
            self._apply_draft_to_selected_event()

        ev = self._find_event_by_id(event_id)
        if not ev:
            return
        self._sync_event_frames(ev)
        self.selected_event_id = event_id
        self.event_draft = {
            "Left_hand": dict(ev.get("hoi_data", {}).get("Left_hand", {})),
            "Right_hand": dict(ev.get("hoi_data", {}).get("Right_hand", {})),
        }

        if hand_key not in ("Left_hand", "Right_hand"):
            hand_key = (
                "Left_hand"
                if ev.get("hoi_data", {}).get("Left_hand", {}).get("interaction_start")
                is not None
                else "Right_hand"
            )
        self.selected_hand_label = hand_key

        self.chk_left.blockSignals(True)
        self.chk_right.blockSignals(True)
        self.chk_left.setChecked(hand_key == "Left_hand")
        self.chk_right.setChecked(hand_key == "Right_hand")
        self.chk_left.blockSignals(False)
        self.chk_right.blockSignals(False)

        self._load_hand_draft_to_ui(hand_key)
        self._update_status_label()
        self.list_objects.setEnabled(True)

        self.hoi_timeline.set_selected(event_id, hand_key)
        self._update_overlay(self.player.current_frame)
        self._update_hoi_titles()

    def _hoi_segments_for_hand(self, hand_key: str) -> List[Dict]:
        segs = []
        for ev in self.events:
            h = ev.get("hoi_data", {}).get(hand_key, {})
            s = h.get("interaction_start")
            e = h.get("interaction_end")
            if s is None or e is None:
                continue
            onset = h.get("functional_contact_onset", s)
            verb = h.get("verb", "")
            base_color = self._hoi_color_for_verb(verb)
            segs.append(
                {
                    "event_id": ev.get("event_id"),
                    "start": int(s),
                    "end": int(e),
                    "onset": int(onset) if onset is not None else int(s),
                    "verb": verb,
                    "color": base_color,
                    "auto_color": base_color is None,
                }
            )
        segs = sorted(segs, key=lambda x: (x["start"], x.get("event_id", -1)))
        palette = list(PRESET_COLORS.values()) or [QColor(120, 120, 120)]
        prev_color = None
        for idx, seg in enumerate(segs):
            auto_color = bool(seg.get("auto_color", False))
            color = seg.get("color")
            if auto_color:
                start_idx = seg.get("event_id")
                if start_idx is None:
                    start_idx = idx
                try:
                    start_idx = abs(int(start_idx))
                except Exception:
                    start_idx = idx
                for offset in range(len(palette)):
                    cand = palette[(start_idx + offset) % len(palette)]
                    if prev_color is None or cand.name() != prev_color.name():
                        color = cand
                        break
            elif color is None:
                color = QColor(120, 120, 120)
            seg["color"] = color
            prev_color = color
        return segs

    def _hand_has_segment(self, ev: dict, hand_key: str) -> bool:
        h = ev.get("hoi_data", {}).get(hand_key, {})
        s = h.get("interaction_start")
        e = h.get("interaction_end")
        return s is not None and e is not None

    def _on_hoi_timeline_delete(self, event_id: int, hand_key: str):
        ev = self._find_event_by_id(event_id)
        if not ev:
            return
        self._push_undo()
        ev.get("hoi_data", {})[hand_key] = self._blank_hand_data()
        self._sync_event_frames(ev)

        if not self._hand_has_segment(ev, "Left_hand") and not self._hand_has_segment(
            ev, "Right_hand"
        ):
            if ev in self.events:
                self.events.remove(ev)
            if self.selected_event_id == event_id:
                self.selected_event_id = None
                self.selected_hand_label = None
                self._reset_event_draft()
                if getattr(self, "hoi_timeline", None):
                    self.hoi_timeline.set_selected(None, None)
        else:
            if (
                self.selected_event_id == event_id
                and self.selected_hand_label == hand_key
            ):
                other = "Right_hand" if hand_key == "Left_hand" else "Left_hand"
                if self._hand_has_segment(ev, other):
                    self._set_selected_event(event_id, other)
                else:
                    self.selected_hand_label = None
                    self._reset_event_draft()

        self._update_status_label()
        self._refresh_events()
        self._update_hoi_titles()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.refresh()
        self._log("hoi_segment_delete", event_id=event_id, hand=hand_key)

    def _hoi_title_for_hand(self, hand_key: str) -> str:
        base = "L" if hand_key == "Left_hand" else "R"
        if self.selected_event_id is None:
            return base
        ev = self._find_event_by_id(self.selected_event_id)
        if not ev:
            return base
        h_data = {}
        if isinstance(getattr(self, "event_draft", None), dict):
            h_data = self.event_draft.get(hand_key, {}) or {}
        if not h_data:
            h_data = ev.get("hoi_data", {}).get(hand_key, {}) or {}
        has_data = any(
            [
                h_data.get("interaction_start") is not None,
                h_data.get("interaction_end") is not None,
                h_data.get("functional_contact_onset") is not None,
                bool(h_data.get("verb")),
                h_data.get("instrument_object_id") is not None,
                h_data.get("target_object_id") is not None,
            ]
        )
        check = "\u2713" if has_data else "\u25a1"
        verb = h_data.get("verb") or "_"
        tool = self._object_name_for_id(
            h_data.get("instrument_object_id"), default_for_none="_"
        )
        target = self._object_name_for_id(
            h_data.get("target_object_id"), default_for_none="_"
        )
        return f"{base} {check} | Verb: {verb}, Tool: {tool}, Target: {target}"

    def _update_hoi_titles(self):
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.update_titles()

    def _current_hand_meta(self) -> dict:
        return {
            "verb": self.combo_verb.currentText(),
            "instrument_object_id": self.combo_instrument.currentData(),
            "target_object_id": self.combo_target.currentData(),
            "anomaly_label": self._selected_anomaly_label(),
        }

    def _blank_hand_data(self) -> dict:
        return {
            "verb": "",
            "instrument_object_id": None,
            "target_object_id": None,
            "interaction_start": None,
            "functional_contact_onset": None,
            "interaction_end": None,
            "anomaly_label": "Normal",
        }

    def _on_hoi_timeline_select(self, event_id: int, hand_key: str):
        self._set_selected_event(event_id, hand_key)

    def _on_hoi_timeline_update(
        self, event_id: int, hand_key: str, start: int, end: int, onset: int
    ):
        ev = self._find_event_by_id(event_id)
        if not ev:
            return
        self._push_undo()
        h = ev.get("hoi_data", {}).setdefault(hand_key, self._blank_hand_data())
        h["interaction_start"] = int(start)
        h["interaction_end"] = int(end)
        h["functional_contact_onset"] = int(onset)
        if h.get("anomaly_label") in (None, ""):
            h["anomaly_label"] = "Normal"
        self._sync_event_frames(ev)
        if self.selected_event_id == event_id:
            self.event_draft[hand_key] = dict(h)
            self._update_status_label()
        self._refresh_events()
        self.hoi_timeline.refresh()
        self._log(
            "hoi_event_edit_frames",
            event_id=event_id,
            hand=hand_key,
            start=start,
            onset=onset,
            end=end,
        )

    def _on_hoi_timeline_create(
        self, hand_key: str, start: int, end: int, onset: int
    ) -> Optional[int]:
        if not self.player.cap:
            QMessageBox.information(
                self, "Info", "Load a video before creating HandOI segments."
            )
            return None
        self._push_undo()
        new_event = {
            "event_id": self.event_id_counter,
            "frames": [start, end],
            "hoi_data": {
                "Left_hand": self._blank_hand_data(),
                "Right_hand": self._blank_hand_data(),
            },
        }
        hand_data = new_event["hoi_data"][hand_key]
        hand_data.update(self._current_hand_meta())
        hand_data["interaction_start"] = int(start)
        hand_data["interaction_end"] = int(end)
        hand_data["functional_contact_onset"] = int(onset)
        if hand_data.get("anomaly_label") in (None, ""):
            hand_data["anomaly_label"] = "Normal"
        self._sync_event_frames(new_event)
        self.events.append(new_event)
        self.event_id_counter += 1
        self._refresh_events()
        self.hoi_timeline.refresh()
        self._set_selected_event(new_event["event_id"], hand_key)
        self._log(
            "hoi_event_create",
            event_id=new_event["event_id"],
            hand=hand_key,
            start=start,
            onset=onset,
            end=end,
        )
        return new_event["event_id"]

    def _jump_to_selected_start(self):
        self._jump_to_selected_keyframe("interaction_start")

    def _jump_to_selected_onset(self):
        self._jump_to_selected_keyframe("functional_contact_onset")

    def _jump_to_selected_end(self):
        self._jump_to_selected_keyframe("interaction_end")

    def _jump_to_selected_keyframe(self, key: str):
        if self.selected_event_id is None or not self.selected_hand_label:
            return
        ev = self._find_event_by_id(self.selected_event_id)
        if not ev:
            return
        hand_data = ev.get("hoi_data", {}).get(self.selected_hand_label, {})
        frame = hand_data.get(key)
        if frame is None:
            return
        try:
            self.player.seek(int(frame))
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._update_overlay(self.player.current_frame)
        except Exception:
            pass

    def _on_hoi_timeline_hover(self, frame: int):
        if not self.player.cap:
            return
        if getattr(self.player, "is_playing", False):
            return
        if frame is None or frame < 0:
            target = int(getattr(self.player, "current_frame", 0))
            try:
                self.player.seek(target, preview_only=True)
            except Exception:
                return
            self._refresh_boxes_for_frame(target)
            self._update_overlay(target)
            return
        try:
            target = max(self.player.crop_start, min(frame, self.player.crop_end))
        except Exception:
            target = int(frame)
        self.player.seek(target, preview_only=True)
        self._refresh_boxes_for_frame(target)
        self._update_overlay(target)

    def _on_hoi_meta_changed(self):
        if not self.selected_hand_label:
            return

        self._save_ui_to_hand_draft(self.selected_hand_label)
        if self.selected_event_id is not None:
            self._apply_draft_to_selected_event()
            self._refresh_events()
            self._update_hoi_titles()
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.refresh()
        self._update_status_label()

    def _after_events_loaded(self):
        self.selected_event_id = None
        self.selected_hand_label = None
        self._reset_event_draft()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_selected(None, None)
            self._update_hoi_titles()
            self.hoi_timeline.refresh()

    def _on_verb_added(self, lb):
        """lb may be a LabelDef (from LabelPanel) or a name string."""
        missing = self._missing_requirements(for_relation=False)
        if missing:
            QMessageBox.information(
                self,
                "Info",
                f"Please load required data before adding verbs: {', '.join(missing)}",
            )
            return
        max_id = max([v.id for v in self.verbs], default=-1)
        if isinstance(lb, LabelDef):
            name = lb.name
        else:
            name = str(lb)
        color_name = (
            getattr(lb, "color_name", None) if isinstance(lb, LabelDef) else None
        )
        if not color_name:
            color_name = "Auto"
        new_lb = LabelDef(name=name, id=max_id + 1, color_name=color_name)
        self.verbs.append(new_lb)
        self._renumber_verbs()
        self._update_verb_combo()
        self._log("hoi_verb_add", name=new_lb.name, id=new_lb.id)

    def _on_verb_removed(self, idx: int):
        removed = None
        if 0 <= idx < len(self.verbs):
            removed = self.verbs.pop(idx)
        self._renumber_verbs()
        self.current_verb_idx = min(self.current_verb_idx, len(self.verbs) - 1)
        self._update_verb_combo()
        if removed:
            self._log("hoi_verb_remove", name=removed.name, id=removed.id)

    def _on_verb_renamed(self, idx: int, new: str):
        old = None
        if 0 <= idx < len(self.verbs):
            old = self.verbs[idx].name
            self.verbs[idx].name = new
        self._renumber_verbs()
        self._update_verb_combo()
        if old is not None:
            self._replace_verb_in_hoi_data(self.event_draft, old, new)
            for ev in self.events:
                self._replace_verb_in_hoi_data(ev.get("hoi_data", {}), old, new)
            self._refresh_events()
            self._update_hoi_titles()
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.refresh()
            self._log("hoi_verb_rename", old=old, new=new)

    def _on_verb_selected(self, idx: int):
        """[Modified] Clicking the list just syncs the combo for convenience, no auto-save."""
        pass

    def _on_verb_double_clicked(self, item: QListWidgetItem):
        """[New] Double-clicking the list updates the combo box (and triggers save)."""
        verb_name = item.text()

        self.combo_verb.setCurrentText(verb_name)

    def _renumber_verbs(self):
        """Ensure ids are monotonically increasing and new ids append after max."""
        cur_max = 0
        for v in self.verbs:
            cur_max = max(cur_max, getattr(v, "id", 0))
        # ensure unique increasing ids
        seen = set()
        for v in self.verbs:
            if v.id in seen:
                cur_max += 1
                v.id = cur_max
            seen.add(v.id)
        # advance next id in UI
        try:
            self.label_panel.id_spin.setValue(cur_max + 1)
        except Exception:
            pass

    def _normalize_hand_label(self, label: str) -> Optional[str]:
        if label is None:
            return None
        norm = str(label).strip().lower().replace(" ", "_")
        if norm in ("left_hand", "left", "l_hand"):
            return "Left_hand"
        if norm in ("right_hand", "right", "r_hand"):
            return "Right_hand"
        return None

    def _is_hand_label(self, label: str) -> bool:
        return self._normalize_hand_label(label) in ("Left_hand", "Right_hand")

    def _color_for_index(self, idx: int) -> str:
        """Pick a non-gray preset color cycling through the palette."""
        palette = [k for k in PRESET_COLORS.keys() if k.lower() != "gray"] or list(
            PRESET_COLORS.keys()
        )
        if not palette:
            return "gray"
        return palette[idx % len(palette)]

    def _verb_color(self, name: str) -> str:
        """Return a QColor-friendly string for a verb name."""
        for v in self.verbs:
            if v.name == name:
                return color_from_key(getattr(v, "color_name", "gray")).name()
        return "#f08c00"

    def _missing_requirements(self, for_relation: bool = False) -> List[str]:
        """
        New check chain: only checks Video and Hands.
        Does not check classes or object bboxes.
        """
        missing = []
        if not self.player.cap:
            missing.append("video")

        # Only check for hands data when creating a relation
        if for_relation:
            # Check if any hand data exists
            has_hand = any(
                b
                for b in self.raw_boxes
                if b.get("label") in ("Left_hand", "Right_hand")
            )
            if not has_hand:
                missing.append("hand bboxes (XML)")

        return missing

    # ---------- undo/redo ----------
    def _snapshot_state(self) -> dict:
        """Capture a deep copy of event + bbox state for undo/redo."""
        return {
            "events": copy.deepcopy(self.events),
            "event_id_counter": self.event_id_counter,
            "event_draft": copy.deepcopy(self.event_draft),
            "selected_hand_label": self.selected_hand_label,
            "selected_event_id": self.selected_event_id,
            "raw_boxes": copy.deepcopy(self.raw_boxes),
            "box_id_counter": self.box_id_counter,
            "global_object_map": copy.deepcopy(self.global_object_map),
            "id_to_category": copy.deepcopy(self.id_to_category),
            "object_id_counter": self.object_id_counter,
            "class_map": copy.deepcopy(self.class_map),
            "instrument_selected": self.combo_instrument.currentData(),
            "target_selected": self.combo_target.currentData(),
            "current_frame": int(getattr(self.player, "current_frame", 0)),
        }

    def _restore_state(self, state: dict):
        """Restore a snapshot created by _snapshot_state."""
        self.events = copy.deepcopy(state.get("events", []))
        self.event_id_counter = state.get("event_id_counter", 0)
        self.event_draft = copy.deepcopy(state.get("event_draft", {}))
        self.selected_hand_label = state.get("selected_hand_label", None)
        self.selected_event_id = state.get("selected_event_id", None)
        self.raw_boxes = copy.deepcopy(state.get("raw_boxes", []))
        self.box_id_counter = state.get("box_id_counter", 1)
        self.global_object_map = copy.deepcopy(state.get("global_object_map", {}))
        self.id_to_category = copy.deepcopy(state.get("id_to_category", {}))
        self.object_id_counter = state.get("object_id_counter", 0)
        self.class_map = copy.deepcopy(state.get("class_map", {}))

        self._rebuild_bboxes_from_raw()

        inst_sel = state.get("instrument_selected")
        tgt_sel = state.get("target_selected")
        self._rebuild_object_combos(inst_sel, tgt_sel)

        frame = state.get("current_frame", self.player.current_frame)
        if self.player.cap:
            self.player.seek(frame)
        self._refresh_boxes_for_frame(frame)
        self._set_frame_controls(frame)

        self.chk_left.blockSignals(True)
        self.chk_right.blockSignals(True)
        self.chk_left.setChecked(self.selected_hand_label == "Left_hand")
        self.chk_right.setChecked(self.selected_hand_label == "Right_hand")
        self.chk_left.blockSignals(False)
        self.chk_right.blockSignals(False)

        if self.selected_hand_label:
            self._load_hand_draft_to_ui(self.selected_hand_label)
        else:
            self._update_status_label()
        self.list_objects.setEnabled(self.selected_hand_label is not None)
        for ev in self.events:
            self._sync_event_frames(ev)
        self._refresh_events()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_selected(
                self.selected_event_id, self.selected_hand_label
            )
            self.hoi_timeline.set_current_frame(frame)
            self.hoi_timeline.refresh()

    def _push_undo(self):
        if self._undo_block:
            return
        if self.validation_enabled:
            self.validation_modified = True
            self.validation_change_count += 1
        self._hoi_undo_stack.append(self._snapshot_state())
        if len(self._hoi_undo_stack) > self._undo_limit:
            self._hoi_undo_stack.pop(0)
        self._hoi_redo_stack.clear()

    def _hoi_undo(self):
        if not self._hoi_undo_stack:
            return
        self._undo_block = True
        try:
            current = self._snapshot_state()
            state = self._hoi_undo_stack.pop()
            self._hoi_redo_stack.append(current)
            self._restore_state(state)
        finally:
            self._undo_block = False

    def _hoi_redo(self):
        if not self._hoi_redo_stack:
            return
        self._undo_block = True
        try:
            current = self._snapshot_state()
            state = self._hoi_redo_stack.pop()
            self._hoi_undo_stack.append(current)
            self._restore_state(state)
        finally:
            self._undo_block = False

    def _clear_undo_history(self):
        self._hoi_undo_stack.clear()
        self._hoi_redo_stack.clear()

    def set_oplog_enabled(self, enabled: bool) -> None:
        self.set_logging_policy(
            bool(enabled), bool(getattr(self, "validation_summary_enabled", True))
        )

    def set_logging_policy(
        self, oplog_enabled: bool, validation_summary_enabled: bool
    ) -> None:
        if getattr(self, "op_logger", None) is not None:
            self.op_logger.enabled = bool(oplog_enabled)
        if getattr(self, "validation_op_logger", None) is not None:
            self.validation_op_logger.enabled = bool(oplog_enabled)
        self.validation_summary_enabled = bool(validation_summary_enabled)

    def _flush_ops_log_safely(self, logger, log_path: str, context: str) -> None:
        if logger is None or not log_path:
            return
        should_write = bool(getattr(logger, "enabled", False))
        try:
            logger.flush(log_path)
            if should_write:
                print(f"[LOG] {log_path}")
        except Exception as ex:
            print(f"[LOG][ERROR] {context} ops log write failed: {log_path} ({ex})")
            QMessageBox.warning(
                self,
                "Operation log",
                f"Operation log write failed, but annotation save already succeeded.\n\nTarget: {log_path}\n\n{ex}",
            )

    def _log(self, event: str, **fields):
        # Keep the log focused on user-driven actions.
        auto_events = {"frame_advanced"}
        if event in auto_events:
            return
        if getattr(self, "op_logger", None):
            try:
                frame = getattr(self.player, "current_frame", None)
            except Exception:
                frame = None
            if frame is not None:
                fields.setdefault("frame", frame)
            if self.validation_enabled and self.validator_name:
                fields.setdefault("validator", self.validator_name)
            self.op_logger.log(event, **fields)
        if self.validation_enabled and getattr(self, "validation_op_logger", None):
            self.validation_op_logger.log(event, **fields)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Left:
            self._seek_relative(-1)
            event.accept()
            return
        if key == Qt.Key_Right:
            self._seek_relative(+1)
            event.accept()
            return
        if key == Qt.Key_Up:
            self._seek_seconds(-1)
            event.accept()
            return
        if key == Qt.Key_Down:
            self._seek_seconds(+1)
            event.accept()
            return
        return super().keyPressEvent(event)

    # ---------- IO ----------
    @staticmethod
    def _normalized_video_path(path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        try:
            return os.path.normcase(os.path.abspath(raw))
        except Exception:
            return os.path.normcase(raw)

    def _hoi_has_annotation_data(self) -> bool:
        return bool(self.events or self.raw_boxes)

    def _apply_video_session(
        self,
        path: str,
        start: int = None,
        end: int = None,
        frame: int = None,
        log_event: str = "hoi_load_video",
    ) -> bool:
        target_path = str(path or "").strip()
        if not target_path:
            return False
        target_norm = self._normalized_video_path(target_path)
        current_norm = self._normalized_video_path(self.video_path)
        reuse_loaded = bool(self.player.cap) and bool(current_norm) and current_norm == target_norm
        if not reuse_loaded:
            if not self.player.load(target_path):
                return False

        total_frames = int(getattr(self.player, "frame_count", 0) or 0)
        if total_frames <= 0:
            return False
        start_frame = 0 if start is None else int(start)
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = total_frames - 1 if end is None else int(end)
        end_frame = max(start_frame, min(end_frame, total_frames - 1))
        current_frame = start_frame if frame is None else int(frame)
        current_frame = max(start_frame, min(current_frame, end_frame))

        self.player.set_crop(start_frame, end_frame)
        if int(getattr(self.player, "current_frame", start_frame)) != current_frame:
            self.player.seek(current_frame)

        self.video_path = target_path
        self.start_offset = start_frame
        self.end_frame = end_frame

        self.spin_start_offset.blockSignals(True)
        self.spin_start_offset.setMaximum(max(0, total_frames - 1))
        self.spin_start_offset.setValue(start_frame)
        self.spin_start_offset.blockSignals(False)

        self.spin_end_frame.blockSignals(True)
        self.spin_end_frame.setMaximum(max(0, total_frames - 1))
        self.spin_end_frame.setValue(end_frame)
        self.spin_end_frame.blockSignals(False)

        self.spin_jump.setMinimum(start_frame)
        self.spin_jump.setMaximum(max(start_frame, end_frame))
        self.spin_jump.setValue(current_frame)

        self.slider.blockSignals(True)
        self.slider.setMinimum(start_frame)
        self.slider.setMaximum(end_frame)
        self.slider.setValue(current_frame)
        self.slider.blockSignals(False)

        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(current_frame)
        self._set_frame_controls(current_frame)

        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_frame_count(self.player.frame_count)
            self.hoi_timeline.set_current_frame(current_frame)
            self.hoi_timeline.refresh()

        self._update_play_pause_button()
        if log_event:
            self._log(
                log_event,
                path=target_path,
                start=start_frame,
                end=end_frame,
                frame=current_frame,
                frames=self.player.frame_count,
            )
        return True

    def inherit_action_video_session(self, session: dict) -> bool:
        if not isinstance(session, dict):
            return False
        path = str(session.get("path") or "").strip()
        if not path:
            return False
        start = int(session.get("start", 0) or 0)
        end = int(session.get("end", start) or start)
        frame = int(session.get("frame", start) or start)
        target_norm = self._normalized_video_path(path)
        current_norm = self._normalized_video_path(self.video_path)
        has_annotations = self._hoi_has_annotation_data()

        if has_annotations:
            if current_norm and current_norm == target_norm and self.player.cap:
                target_frame = max(
                    int(getattr(self.player, "crop_start", 0)),
                    min(frame, int(getattr(self.player, "crop_end", frame))),
                )
                self.player.seek(target_frame)
                self._refresh_boxes_for_frame(target_frame)
                self._set_frame_controls(target_frame)
                if getattr(self, "hoi_timeline", None):
                    self.hoi_timeline.set_current_frame(target_frame)
                    self.hoi_timeline.refresh()
                self._update_play_pause_button()
                self._log("hoi_sync_frame_from_action", path=path, frame=target_frame)
                return True
            return False

        ok = self._apply_video_session(
            path,
            start=start,
            end=end,
            frame=frame,
            log_event="hoi_inherit_action_video",
        )
        if ok:
            self._mark_hoi_saved()
        return ok

    def _load_video(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Load Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if not fp:
            return
        if not self._apply_video_session(fp, log_event="hoi_load_video"):
            QMessageBox.warning(self, "Error", "Failed to load video.")
            return

    def _load_bboxes(self):
        """
        Phase 1 Smart Loader (Corrected):
        Saves filtered YOLO boxes into self.raw_boxes to prevent overwriting.
        """
        # --- 1. Dependency Check ---
        if not self.player.cap:
            QMessageBox.warning(
                self, "Missing prerequisite", "Please load a video first (resolution needed)."
            )
            return
        if not self.global_object_map:
            QMessageBox.warning(
                self,
                "Missing prerequisite",
                "Please import Instrument/Target lists first.",
            )
            return
        if not self.class_map:
            QMessageBox.warning(
                self,
                "Missing prerequisite",
                "Please load a class map first (data.yaml).",
            )
            return

        dir_path = QFileDialog.getExistingDirectory(
            self, "Select YOLO labels directory"
        )
        if not dir_path:
            return

        # --- 2. Build Smart Map ---
        category_to_default_id = self._build_category_to_default_id()

        if not category_to_default_id:
            QMessageBox.warning(
                self,
                "Error",
                "Cannot build the default object map. Check the Instrument/Target lists.",
            )
            return

        # --- 3. Parse Loop ---
        loaded_count = 0
        w, h = self.player._frame_w, self.player._frame_h
        new_raw_boxes = []

        import os

        files = sorted([f for f in os.listdir(dir_path) if f.endswith(".txt")])

        for fname in files:
            # Get raw frame index (without offset)
            frame_idx = self._frame_from_filename(fname)

            path = os.path.join(dir_path, fname)
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                    except Exception:
                        continue

                    cls_name = self._norm_category(self.class_map.get(cls_id, ""))
                    if not cls_name:
                        continue

                    # Match ID
                    matched_uid = category_to_default_id.get(cls_name)

                    if matched_uid is not None:
                        # Calc absolute coordinates
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h

                        # Construct raw_box dict
                        # Must include orig_frame for _rebuild to calculate offset correctly
                        new_raw_boxes.append(
                            {
                                "id": matched_uid,  # Forced merge to global ID
                                "orig_frame": frame_idx,
                                "label": self.class_map.get(cls_id, ""),
                                "class_id": cls_id,
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                            }
                        )
                        loaded_count += 1

        # --- 4. Update Memory ---
        # Append new data to raw_boxes
        # Strategy: Keep hands (Left_hand/Right_hand), clear other old objects, add new objects
        has_existing_objects = any(
            b
            for b in self.raw_boxes
            if b.get("label") not in ("Left_hand", "Right_hand")
        )
        if new_raw_boxes or has_existing_objects:
            self._push_undo()
        self.raw_boxes = [
            b for b in self.raw_boxes if b.get("label") in ("Left_hand", "Right_hand")
        ]
        self.raw_boxes.extend(new_raw_boxes)

        # Rebuild bboxes index
        self._rebuild_bboxes_from_raw()

        self._refresh_boxes_for_frame(self.player.current_frame)
        self._log("hoi_load_bboxes", path=dir_path, count=loaded_count)
        QMessageBox.information(
            self,
            "Done",
            f"Loaded {loaded_count} detection boxes.\nOld objects cleared, hands preserved.",
        )
        self._mark_hoi_saved()

    def _load_yaml(self):
        """Load data.yaml to get class_id -> class_name mapping"""
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Load Class Map (data.yaml)",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not fp:
            return
        try:
            self.class_map = self._parse_yaml_names(fp)
            QMessageBox.information(
                self, "Loaded", f"Loaded {len(self.class_map)} classes from yaml."
            )
            self._log("hoi_load_classes", path=fp, count=len(self.class_map))
        except Exception as ex:
            QMessageBox.warning(self, "Error", f"Failed to parse data.yaml:\n{ex}")

    def _load_verbs_txt(self):
        """Import verb list txt with duplication check"""
        fp, _ = QFileDialog.getOpenFileName(
            self, "Import Verb List", "", "Text Files (*.txt);;All Files (*)"
        )
        if not fp:
            return
        try:
            lines = [
                ln.strip()
                for ln in open(fp, "r", encoding="utf-8").read().splitlines()
                if ln.strip()
            ]

            # Calc max ID for new assignment
            max_id = max([v.id for v in self.verbs], default=-1)
            added_count = 0

            for idx, ln in enumerate(lines):
                parts = ln.split()
                if not parts:
                    continue
                name = parts[0]

                # Check for duplicates (case-insensitive)
                if any(v.name.lower() == name.lower() for v in self.verbs):
                    continue

                max_id += 1
                color_name = "Auto"
                self.verbs.append(LabelDef(name=name, id=max_id, color_name=color_name))
                added_count += 1

            self._renumber_verbs()
            self._update_verb_combo()
            self.label_panel.refresh()

            if added_count > 0:
                QMessageBox.information(
                    self,
                    "Loaded",
                    f"Successfully imported {added_count} new verbs (duplicates skipped).",
                )
            else:
                QMessageBox.information(
                    self, "Info", "No new verbs imported (all exist)."
                )
            self._log("hoi_load_verbs", path=fp, added=added_count)

        except Exception as ex:
            QMessageBox.warning(self, "Error", f"Failed to load verbs:\n{ex}")

    def _load_yolo_model(self):
        """Load Ultralytics YOLOv11 .pt weights for current-frame detection."""
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load YOLO Model", "", "YOLO Weights (*.pt);;All Files (*)"
        )
        if not fp:
            return
        try:
            from ultralytics import YOLO
        except Exception as ex:
            QMessageBox.warning(
                self, "Missing package", f"Ultralytics is not available:\n{ex}"
            )
            return
        try:
            self.yolo_model = YOLO(fp)
            self.yolo_weights_path = fp
            QMessageBox.information(
                self, "Loaded", f"YOLO model loaded:\n{os.path.basename(fp)}"
            )
            self._log("hoi_load_yolo_model", path=fp)
        except Exception as ex:
            QMessageBox.warning(self, "Error", f"Failed to load model:\n{ex}")

    def _ask_existing_policy(self):
        """Ask how to handle existing boxes on the current frame."""
        box = QMessageBox(self)
        box.setWindowTitle("Existing boxes")
        box.setText(
            "This frame already has object boxes. How do you want to apply YOLO results?"
        )
        append_btn = box.addButton("Append", QMessageBox.AcceptRole)
        replace_btn = box.addButton("Replace", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        remember = QCheckBox("Remember my choice")
        box.setCheckBox(remember)
        box.exec_()
        clicked = box.clickedButton()
        policy = None
        if clicked == append_btn:
            policy = "append"
        elif clicked == replace_btn:
            policy = "replace"
        if policy and remember.isChecked():
            self.yolo_existing_policy = policy
        return policy

    def _detect_current_frame_yolo(
        self,
        frame_idx=None,
        include_ids=None,
        override_policy=None,
        push_undo=True,
    ):
        """Run YOLOv11 detection on one frame and optionally filter to selected object ids."""
        if not self.player.cap:
            QMessageBox.warning(self, "Missing prerequisite", "Please load a video first.")
            return
        if not self.class_map:
            QMessageBox.warning(
                self, "Missing prerequisite", "Please load a class map first (data.yaml)."
            )
            return
        if not self.global_object_map:
            QMessageBox.warning(
                self,
                "Missing prerequisite",
                "Please import Instrument/Target lists first.",
            )
            return
        if self.yolo_model is None:
            self._load_yolo_model()
            if self.yolo_model is None:
                return

        frame_bgr = self.player.get_current_frame_bgr()
        if frame_bgr is None:
            QMessageBox.warning(self, "Error", "No video frame available.")
            return

        if frame_idx is None:
            frame_idx = int(self.player.current_frame)
        existing = [
            b
            for b in self.bboxes.get(frame_idx, [])
            if not self._is_hand_label(b.get("label"))
        ]
        replace_existing = False
        if existing:
            policy = override_policy or self.yolo_existing_policy or self._ask_existing_policy()
            if not policy:
                return
            replace_existing = policy == "replace"

        # Build category -> default object id mapping (normalized)
        category_to_default_id = self._build_category_to_default_id()

        if not category_to_default_id:
            QMessageBox.warning(
                self,
                "Error",
                "Cannot map detections without Instrument/Target lists.",
            )
            return

        # Try GPU first, then CPU
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"

        try:
            results = self.yolo_model.predict(
                source=frame_bgr,
                conf=self.yolo_conf,
                iou=self.yolo_iou,
                device=device,
                verbose=False,
            )
        except Exception as ex:
            if device == "cuda":
                try:
                    results = self.yolo_model.predict(
                        source=frame_bgr,
                        conf=self.yolo_conf,
                        iou=self.yolo_iou,
                        device="cpu",
                        verbose=False,
                    )
                except Exception as ex2:
                    QMessageBox.warning(self, "Error", f"YOLO inference failed:\n{ex2}")
                    return
            else:
                QMessageBox.warning(self, "Error", f"YOLO inference failed:\n{ex}")
                return

        if not results:
            QMessageBox.information(self, "Info", "No detections found.")
            return

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            QMessageBox.information(self, "Info", "No detections found.")
            return

        try:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            QMessageBox.warning(self, "Error", "Failed to parse YOLO outputs.")
            return

        w = self.player._frame_w
        h = self.player._frame_h
        added = 0
        skipped = 0
        new_boxes = []
        for i in range(len(xyxy)):
            cls_id = int(cls_ids[i])
            class_name = self.class_map.get(cls_id)
            if class_name is None:
                class_name = self.class_map.get(str(cls_id))
            if class_name is None and hasattr(self.yolo_model, "names"):
                names = self.yolo_model.names
                if isinstance(names, dict):
                    class_name = names.get(cls_id)
                else:
                    try:
                        class_name = names[int(cls_id)]
                    except Exception:
                        class_name = None
            if not class_name:
                skipped += 1
                continue
            norm_name = self._norm_category(class_name)
            if norm_name in ("left_hand", "right_hand"):
                skipped += 1
                continue
            uid = category_to_default_id.get(norm_name)
            if include_ids is not None and uid not in set(include_ids or []):
                matched_uid = None
                for req_id in include_ids or []:
                    req_names = [
                        name
                        for name, val in self.global_object_map.items()
                        if val == req_id
                    ]
                    if any(self._norm_category(name) == norm_name for name in req_names):
                        matched_uid = req_id
                        break
                uid = matched_uid
            if uid is None:
                skipped += 1
                continue
            if include_ids is not None and uid not in set(include_ids or []):
                skipped += 1
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            x1 = max(0.0, min(float(w), x1))
            y1 = max(0.0, min(float(h), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))
            new_boxes.append(
                {
                    "id": uid,
                    "orig_frame": frame_idx - self.start_offset,
                    "label": str(class_name),
                    "class_id": cls_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
            added += 1
        if not new_boxes and not (replace_existing and existing):
            return

        if push_undo:
            self._push_undo()
        if replace_existing and existing:
            keep = []
            for rb in self.raw_boxes:
                tgt = rb.get("orig_frame", 0) + self.start_offset
                if tgt == frame_idx and not self._is_hand_label(rb.get("label")):
                    continue
                keep.append(rb)
            self.raw_boxes = keep
        self.raw_boxes.extend(new_boxes)
        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(frame_idx)
        self._log(
            "hoi_detect_yolo",
            frame=frame_idx,
            added=added,
            skipped=skipped,
            replaced=bool(replace_existing),
        )
        # No dialog for YOLO detection; visualization and list update are enough.

    def _ensure_mp_hands(self):
        """Initialize MediaPipe Hands if available."""
        if self.mp_hands_error:
            return None
        if self.mp_hands is not None:
            return self.mp_hands
        try:
            import mediapipe as mp
        except Exception as ex:
            self.mp_hands_error = str(ex)
            QMessageBox.warning(
                self, "Missing package", f"MediaPipe is not available:\n{ex}"
            )
            return None
        if not hasattr(mp, "solutions"):
            try:
                import mediapipe.solutions as mp_solutions

                mp.solutions = mp_solutions
            except Exception as ex:
                self.mp_hands_error = str(ex)
                QMessageBox.warning(
                    self,
                    "MediaPipe Error",
                    f"MediaPipe solutions are unavailable:\n{ex}",
                )
                return None
        try:
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=self.mp_hands_max,
                min_detection_confidence=self.mp_hands_conf,
                min_tracking_confidence=self.mp_hands_track_conf,
            )
            return self.mp_hands
        except Exception as ex:
            self.mp_hands_error = str(ex)
            QMessageBox.warning(
                self, "Error", f"Failed to initialize MediaPipe Hands:\n{ex}"
            )
            return None

    def _detect_current_frame_hands(self, frame_bgr, frame_idx=None, push_undo=True):
        """Run MediaPipe Hands on current frame and replace hand boxes for that frame."""
        mp_hands = self._ensure_mp_hands()
        if mp_hands is None:
            return 0
        if frame_bgr is None:
            return 0
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            results = mp_hands.process(rgb)
        except Exception as ex:
            if not self.mp_hands_error:
                self.mp_hands_error = str(ex)
                QMessageBox.warning(
                    self, "MediaPipe Error", f"Hand detection failed:\n{ex}"
                )
            return 0
        if not results or not results.multi_hand_landmarks:
            return 0

        handedness = []
        if results.multi_handedness:
            for hd in results.multi_handedness:
                try:
                    handedness.append(hd.classification[0].label)
                except Exception:
                    handedness.append(None)
        while len(handedness) < len(results.multi_hand_landmarks):
            handedness.append(None)

        boxes = []
        used = set()
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            xs = [lm.x for lm in hand_lms.landmark]
            ys = [lm.y for lm in hand_lms.landmark]
            x1 = max(0.0, min(float(w), min(xs) * w))
            y1 = max(0.0, min(float(h), min(ys) * h))
            x2 = max(0.0, min(float(w), max(xs) * w))
            y2 = max(0.0, min(float(h), max(ys) * h))

            label = None
            if idx < len(handedness) and handedness[idx]:
                if str(handedness[idx]).lower().startswith("left"):
                    label = "Left_hand"
                elif str(handedness[idx]).lower().startswith("right"):
                    label = "Right_hand"
            if label is None:
                label = "Left_hand" if "Left_hand" not in used else "Right_hand"
            if label in used:
                other = "Right_hand" if label == "Left_hand" else "Left_hand"
                if other not in used:
                    label = other
                else:
                    continue
            if self.mp_hands_swap and label in ("Left_hand", "Right_hand"):
                label = "Right_hand" if label == "Left_hand" else "Left_hand"
            if label in used:
                other = "Right_hand" if label == "Left_hand" else "Left_hand"
                if other not in used:
                    label = other
                else:
                    continue
            if label in used and "Left_hand" in used and "Right_hand" in used:
                continue
            used.add(label)
            boxes.append((label, x1, y1, x2, y2))

        detected_norm = {
            self._normalize_hand_label(b[0])
            for b in boxes
            if self._normalize_hand_label(b[0])
        }

        if not boxes:
            return 0

        if frame_idx is None:
            frame_idx = int(self.player.current_frame)
        if push_undo:
            self._push_undo()
        keep = []
        for rb in self.raw_boxes:
            tgt = rb.get("orig_frame", 0) + self.start_offset
            if tgt == frame_idx:
                norm = self._normalize_hand_label(rb.get("label"))
                if norm and norm in detected_norm:
                    continue
            keep.append(rb)
        self.raw_boxes = keep

        for label, x1, y1, x2, y2 in boxes:
            self.raw_boxes.append(
                {
                    "id": self.box_id_counter,
                    "orig_frame": frame_idx - self.start_offset,
                    "label": label,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
            self.box_id_counter += 1

        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(frame_idx)
        self._log("hoi_detect_hands", frame=frame_idx, count=len(boxes))
        return len(boxes)

    def _detect_selected_action(self):
        if self.selected_event_id is None:
            QMessageBox.information(
                self,
                "HOI Detection",
                "Select an action first, then use Detect Action.",
            )
            return
        self._detect_action_active_items(self.selected_event_id)

    def _detect_action_active_items(self, event_id: int):
        ev = self._find_event_by_id(event_id)
        if not ev:
            return
        keyframes = set()
        active_ids = set()
        for hand_key in ("Left_hand", "Right_hand"):
            hand_data = ev.get("hoi_data", {}).get(hand_key, {}) or {}
            for key in (
                "interaction_start",
                "functional_contact_onset",
                "interaction_end",
            ):
                val = hand_data.get(key)
                if val is not None:
                    keyframes.add(int(val))
            inst_id = hand_data.get("instrument_object_id")
            if inst_id is not None:
                active_ids.add(inst_id)
            target_id = hand_data.get("target_object_id")
            if target_id is not None:
                active_ids.add(target_id)

        if not keyframes:
            QMessageBox.warning(
                self,
                "HOI Detection",
                "The selected action does not have start, onset, or end frames yet.",
            )
            return

        include_ids = set(active_ids)
        if not include_ids:
            reply = QMessageBox.question(
                self,
                "HOI Detection",
                "This action has no assigned instrument or target.\nDetect all object categories on its keyframes instead?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            include_ids = None

        orig_frame = int(self.player.current_frame)
        batch_policy = "append"
        for fidx in keyframes:
            existing = [
                b
                for b in self.bboxes.get(int(fidx), [])
                if not self._is_hand_label(b.get("label"))
            ]
            if existing:
                batch_policy = self.yolo_existing_policy or self._ask_existing_policy()
                if not batch_policy:
                    return
                break

        self._push_undo()
        detected_frames = 0
        try:
            for fidx in sorted(keyframes):
                self.player.seek(int(fidx))
                self._detect_current_frame_yolo(
                    frame_idx=int(fidx),
                    include_ids=include_ids,
                    override_policy=batch_policy,
                    push_undo=False,
                )
                detected_frames += 1
        finally:
            self.player.seek(orig_frame)
            self._refresh_boxes_for_frame(orig_frame)
            self._update_overlay(orig_frame)
        self._log(
            "hoi_detect_action_keyframes",
            event_id=event_id,
            frames=detected_frames,
            filtered=bool(include_ids),
        )

    def _detect_all_actions(self):
        if not self.events:
            QMessageBox.information(self, "HOI Detection", "No actions are available.")
            return

        frame_to_active_ids = {}
        for ev in self.events:
            for hand_key in ("Left_hand", "Right_hand"):
                hand_data = ev.get("hoi_data", {}).get(hand_key, {}) or {}
                keyframes = set()
                for key in (
                    "interaction_start",
                    "functional_contact_onset",
                    "interaction_end",
                ):
                    val = hand_data.get(key)
                    if val is not None:
                        keyframes.add(int(val))
                if not keyframes:
                    continue
                active_ids = set()
                inst_id = hand_data.get("instrument_object_id")
                if inst_id is not None:
                    active_ids.add(inst_id)
                target_id = hand_data.get("target_object_id")
                if target_id is not None:
                    active_ids.add(target_id)
                for fidx in keyframes:
                    frame_to_active_ids.setdefault(int(fidx), set()).update(active_ids)

        if not frame_to_active_ids:
            QMessageBox.information(
                self,
                "HOI Detection",
                "No action keyframes with assigned instrument or target were found.",
            )
            return

        orig_frame = int(self.player.current_frame)
        batch_policy = "append"
        for fidx in frame_to_active_ids.keys():
            existing = [
                b
                for b in self.bboxes.get(int(fidx), [])
                if not self._is_hand_label(b.get("label"))
            ]
            if existing:
                batch_policy = self.yolo_existing_policy or self._ask_existing_policy()
                if not batch_policy:
                    return
                break

        self._push_undo()
        detected_frames = 0
        try:
            for fidx, active_ids in sorted(frame_to_active_ids.items()):
                self.player.seek(int(fidx))
                self._detect_current_frame_yolo(
                    frame_idx=int(fidx),
                    include_ids=active_ids if active_ids else None,
                    override_policy=batch_policy,
                    push_undo=False,
                )
                detected_frames += 1
        finally:
            self.player.seek(orig_frame)
            self._refresh_boxes_for_frame(orig_frame)
            self._update_overlay(orig_frame)
        self._log("hoi_detect_all_action_keyframes", frames=detected_frames)

    def _detect_current_frame_combined(self):
        """Run YOLO object detection and MediaPipe hand detection on the current frame."""
        if not self.player.cap:
            QMessageBox.warning(self, "Missing prerequisite", "Please load a video first.")
            return
        frame_bgr = self.player.get_current_frame_bgr()
        if frame_bgr is None:
            QMessageBox.warning(self, "Error", "No video frame available.")
            return

        # Objects (YOLO)
        if self.class_map and self.global_object_map:
            self._detect_current_frame_yolo()
        else:
            QMessageBox.information(
                self,
                "Info",
                "Object detection skipped. Load a class map and Instrument/Target lists first.",
            )

        # Hands (MediaPipe)
        self._detect_current_frame_hands(frame_bgr)

    def _save_annotations_json(self):
        """Save HOI annotations with optional integrity checks."""
        try:
            # persist any in-progress edits before validation/export
            if self.selected_event_id is not None:
                self._save_ui_to_hand_draft(self.selected_hand_label)
                self._apply_draft_to_selected_event()

            # 1. Basic check
            if not self.events and not self.raw_boxes:
                QMessageBox.information(self, "Info", "No HOI data to save.")
                return

            # 2. Generate filename
            base = os.path.splitext(os.path.basename(self.video_path or "annotation"))[
                0
            ]
            default_name = f"{base}_hoi.json"
            fp, _ = QFileDialog.getSaveFileName(
                self,
                "Save HOI Annotations",
                default_name,
                "JSON Files (*.json);;All Files (*)",
            )
            if not fp:
                return

            validation_error = None

            for i, event in enumerate(self.events):
                event_id = event["event_id"]
                for hand_key in ["Left_hand", "Right_hand"]:
                    h_data = event["hoi_data"][hand_key]

                    has_intent = (
                        bool(h_data.get("verb"))
                        or h_data.get("interaction_start") is not None
                    )

                    if has_intent:
                        errs = self._validate_integrity(h_data)
                        if errs:
                            err = errs[0]
                            validation_error = (
                                f"Event {event_id} ({hand_key}):\n{err['msg']}"
                            )
                            break
                if validation_error:
                    break

            ok_incomplete, _ = self._check_incomplete_hoi(context="save")
            if not ok_incomplete:
                return

            # -----------------------------------------------------------
            # Toggle Block
            # -----------------------------------------------------------
            # if validation_error:
            #     reply = QMessageBox.question(
            #         self, "Integrity Warning",
            #         f"{validation_error}\n\n(Object bbox might be missing at this frame)\nForce save anyway?",
            #         QMessageBox.Yes | QMessageBox.No
            #     )
            #     if reply == QMessageBox.No:
            #         self.player.seek(error_frame)
            #         self._refresh_boxes_for_frame(error_frame)
            #         return
            # -----------------------------------------------------------

            # 4. Build payload
            payload = self._build_payload_v2()

            # 5. Write file
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            self._log(
                "hoi_save_annotations",
                path=fp,
                events=len(self.events),
                boxes=len(self.raw_boxes),
            )
            log_base = os.path.splitext(fp)[0]
            if log_base:
                self._flush_ops_log_safely(
                    getattr(self, "op_logger", None),
                    log_base + ".ops.log.csv",
                    context="HOI save",
                )
            if (
                log_base
                and getattr(self, "validation_op_logger", None)
                and self.validation_op_logger.enabled
                and self.validation_session_active
            ):
                self._flush_ops_log_safely(
                    self.validation_op_logger,
                    log_base + ".validation.ops.log.csv",
                    context="HOI validation save",
                )
            self._save_validation_summary(fp, payload)
            self._mark_hoi_saved()
            QMessageBox.information(
                self, "Saved", f"Successfully saved to:\n{os.path.basename(fp)}"
            )

        except Exception as ex:
            import traceback

            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Save failed:\n{str(ex)}")

    def _collect_incomplete_hoi(self) -> List[dict]:
        issues = []
        for ev in self.events:
            ev_id = ev.get("event_id")
            for hand_key in ("Left_hand", "Right_hand"):
                h_data = ev.get("hoi_data", {}).get(hand_key, {}) or {}
                start = h_data.get("interaction_start")
                end = h_data.get("interaction_end")
                onset = h_data.get("functional_contact_onset")
                has_segment = start is not None or end is not None or onset is not None
                has_meta = (
                    bool(h_data.get("verb"))
                    or h_data.get("instrument_object_id") is not None
                    or h_data.get("target_object_id") is not None
                )
                if not (has_segment or has_meta):
                    continue
                labels = self._parse_anomaly_labels(h_data.get("anomaly_label"))
                allow_missing_bbox = self._anomaly_rule_allows(
                    labels, "allow_missing_bbox"
                )
                allow_missing_verb = self._anomaly_rule_allows(
                    labels, "allow_missing_verb"
                )
                missing = []
                if start is None or end is None:
                    missing.append("start/end")
                if onset is None:
                    missing.append("onset")
                if not h_data.get("verb") and not allow_missing_verb:
                    missing.append("verb")
                if h_data.get("instrument_object_id") is None:
                    missing.append("instrument")
                if h_data.get("target_object_id") is None:
                    missing.append("target")
                if not allow_missing_bbox:
                    bbox_errors = self._validate_integrity(h_data)
                    if bbox_errors:
                        missing.append("bbox")
                if missing:
                    frame = (
                        onset
                        if onset is not None
                        else (
                            start
                            if start is not None
                            else end if end is not None else None
                        )
                    )
                    issues.append(
                        {
                            "event_id": ev_id,
                            "hand": hand_key,
                            "missing": missing,
                            "frame": frame,
                        }
                    )
        return issues

    def _check_incomplete_hoi(self, context: str = "save") -> tuple:
        issues = self._collect_incomplete_hoi()
        if not issues:
            return True, False
        preview = []
        for entry in issues[:4]:
            hand = "L" if entry["hand"] == "Left_hand" else "R"
            missing = ", ".join(entry["missing"])
            preview.append(f"Event {entry['event_id']} {hand}: {missing}")
        if len(issues) > 4:
            preview.append(f"... and {len(issues) - 4} more")
        msg = (
            "Incomplete HandOI segments detected.\n"
            "Missing fields: start/end, onset, verb, instrument, target, or bbox.\n\n"
            + "\n".join(preview)
            + "\n\nContinue anyway?"
        )
        reply = QMessageBox.question(
            self, "Incomplete annotations", msg, QMessageBox.Yes | QMessageBox.No
        )
        proceed = reply == QMessageBox.Yes
        self._log(
            "hoi_incomplete_check",
            context=context,
            has_issues=True,
            proceed=proceed,
            count=len(issues),
        )
        return proceed, True

    def _update_incomplete_indicator(self):
        issues = self._collect_incomplete_hoi()
        self._incomplete_issues = issues
        if not issues:
            self.lbl_incomplete.setText("Incomplete: none")
            self.lbl_incomplete.setToolTip("No incomplete HandOI segments detected.")
            self.btn_incomplete_prev.setEnabled(False)
            self.btn_incomplete_next.setEnabled(False)
            self._incomplete_idx = -1
            return
        self._incomplete_idx = min(
            self._incomplete_idx if self._incomplete_idx >= 0 else 0, len(issues) - 1
        )
        self.lbl_incomplete.setText(f"Incomplete: {len(issues)}")
        tooltip = []
        for entry in issues[:6]:
            hand = "L" if entry["hand"] == "Left_hand" else "R"
            missing = ", ".join(entry["missing"])
            tooltip.append(f"Event {entry['event_id']} {hand}: {missing}")
        if len(issues) > 6:
            tooltip.append("...")
        self.lbl_incomplete.setToolTip("\n".join(tooltip))
        self.btn_incomplete_prev.setEnabled(True)
        self.btn_incomplete_next.setEnabled(True)

    def _jump_incomplete(self, direction: int):
        if not self._incomplete_issues:
            return
        self._incomplete_idx = (self._incomplete_idx + direction) % len(
            self._incomplete_issues
        )
        issue = self._incomplete_issues[self._incomplete_idx]
        frame = issue.get("frame")
        if frame is None:
            frame = int(getattr(self.player, "current_frame", 0))
        if self.player.cap:
            self.player.seek(frame)
        self._refresh_boxes_for_frame(frame)
        self._set_frame_controls(frame)
        event_id = issue.get("event_id")
        hand = issue.get("hand")
        if event_id is not None and hand in ("Left_hand", "Right_hand"):
            self._set_selected_event(event_id, hand)
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_current_frame(frame)
            self.hoi_timeline.refresh()
        self._log(
            "hoi_incomplete_jump",
            direction=direction,
            target=frame,
            event_id=event_id,
            hand=hand,
        )

    def _save_validation_summary(self, annotations_path: str, payload: dict):
        if not bool(getattr(self, "validation_summary_enabled", True)):
            return
        if not self.validation_session_active:
            return
        summary = {
            "version": "HOI_VALIDATION_V1",
            "editor": self.validator_name,
            "validation_started_at": self.validation_started_at,
            "validation_saved_at": datetime.now().isoformat(timespec="seconds"),
            "video_path": self.video_path,
            "annotation_path": annotations_path,
            "events": len(self.events),
            "boxes": len(self.raw_boxes),
            "modified": bool(self.validation_modified),
            "change_count": int(self.validation_change_count),
            "final_state": payload,
        }
        log_path = os.path.splitext(annotations_path)[0] + ".validation.json"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=True)
            print(f"[LOG] {log_path}")
            self._log("hoi_validation_summary_saved", path=log_path)
        except Exception as ex:
            print(f"[LOG][ERROR] HOI validation summary write failed: {log_path} ({ex})")
            QMessageBox.warning(
                self,
                "Validation log",
                f"Failed to write validation log:\n{log_path}\n\n{ex}",
            )

    def _init_anomaly_rules(self):
        self.anomaly_rules = {}
        for label in self.anomaly_labels:
            self.anomaly_rules[label] = {
                "allow_missing_bbox": False,
                "allow_missing_verb": False,
            }
        if "Visibility / Occlusion" in self.anomaly_rules:
            self.anomaly_rules["Visibility / Occlusion"]["allow_missing_bbox"] = True
            self.anomaly_rules["Visibility / Occlusion"]["allow_missing_verb"] = True

    def _ensure_anomaly_rules(self):
        for label in self.anomaly_labels:
            if label not in self.anomaly_rules:
                self.anomaly_rules[label] = {
                    "allow_missing_bbox": False,
                    "allow_missing_verb": False,
                }
        for label in list(self.anomaly_rules.keys()):
            if label not in self.anomaly_labels:
                del self.anomaly_rules[label]

    def _normalize_anomaly_label(self, label: str) -> str:
        if not label:
            return "Normal"
        text = str(label).strip()
        if text.lower() == "correct":
            return "Normal"
        return text

    def _parse_anomaly_labels(self, label_str: str) -> List[str]:
        if not label_str:
            return ["Normal"]
        raw = [
            self._normalize_anomaly_label(t.strip())
            for t in str(label_str).split(",")
            if t.strip()
        ]
        if not raw:
            return ["Normal"]
        if "Normal" in raw and len(raw) > 1:
            raw = [t for t in raw if t != "Normal"]
        # preserve order, de-dupe
        seen = set()
        out = []
        for t in raw:
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out

    def _anomaly_rule_allows(self, labels: List[str], key: str) -> bool:
        self._ensure_anomaly_rules()
        for label in labels:
            for rule_key, rule in self.anomaly_rules.items():
                if rule_key.strip().lower() == label.strip().lower():
                    if rule.get(key):
                        return True
        return False

    def _edit_anomaly_rules(self):
        self._ensure_anomaly_rules()
        dialog = QDialog(self)
        dialog.setWindowTitle("Anomaly Rules")
        dialog.resize(480, 320)
        layout = QVBoxLayout(dialog)
        table = QTableWidget(dialog)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(
            ["Anomaly label", "Allow missing bbox", "Allow missing verb"]
        )
        table.verticalHeader().setVisible(False)
        table.setRowCount(len(self.anomaly_labels))
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for row_idx, label in enumerate(self.anomaly_labels):
            name_item = QTableWidgetItem(label)
            table.setItem(row_idx, 0, name_item)
            rule = self.anomaly_rules.get(label, {})
            bbox_item = QTableWidgetItem("")
            bbox_item.setFlags(
                bbox_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled
            )
            bbox_item.setCheckState(
                Qt.Checked if rule.get("allow_missing_bbox") else Qt.Unchecked
            )
            table.setItem(row_idx, 1, bbox_item)
            verb_item = QTableWidgetItem("")
            verb_item.setFlags(
                verb_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled
            )
            verb_item.setCheckState(
                Qt.Checked if rule.get("allow_missing_verb") else Qt.Unchecked
            )
            table.setItem(row_idx, 2, verb_item)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        layout.addWidget(table)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        for row_idx, label in enumerate(self.anomaly_labels):
            bbox_item = table.item(row_idx, 1)
            verb_item = table.item(row_idx, 2)
            self.anomaly_rules[label] = {
                "allow_missing_bbox": (
                    bbox_item.checkState() == Qt.Checked if bbox_item else False
                ),
                "allow_missing_verb": (
                    verb_item.checkState() == Qt.Checked if verb_item else False
                ),
            }
        self._log("hoi_anomaly_rules_update", rules=self.anomaly_rules)

    def _load_annotations_json(self):
        """
        Load HOI annotations.
        Supports unified per-hand event format and legacy window format.
        """
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load HOI annotations", "", "JSON Files (*.json);;All Files (*)"
        )
        if not fp:
            return
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as ex:
            QMessageBox.warning(self, "Error", f"Failed to read JSON:\n{ex}")
            return
        if not isinstance(data, dict):
            QMessageBox.warning(self, "Error", "Invalid JSON: root must be an object.")
            return
        self._clear_undo_history()

        if isinstance(data, dict) and "tracks" in data and "hoi_events" in data:
            self._load_annotations_v2(data)
            self._log(
                "hoi_load_annotations",
                path=fp,
                events=len(self.events),
                boxes=len(self.raw_boxes),
            )
            self._mark_hoi_saved()
            return

        # --- 1. Restore Meta Data (Object Lookup) ---
        meta = data.get("meta_data", {})
        if not isinstance(meta, dict):
            meta = {}
        obj_lookup = meta.get("object_lookup", {})
        if not isinstance(obj_lookup, dict):
            obj_lookup = {}

        self.global_object_map.clear()
        self.object_id_counter = 0
        self.combo_instrument.clear()
        self.combo_target.clear()
        self.combo_instrument.addItem("None", None)
        self.combo_target.addItem("None", None)

        # Restore objects by ID
        sorted_ids = []
        for k in obj_lookup.keys():
            try:
                sorted_ids.append(int(k))
            except Exception:
                continue
        sorted_ids.sort()
        for uid in sorted_ids:
            info = obj_lookup.get(str(uid), {})
            name = info.get("name", f"obj_{uid}")
            category = info.get("category", "unknown")

            self.global_object_map[name] = uid
            self.id_to_category[name] = category

            if uid >= self.object_id_counter:
                self.object_id_counter = uid + 1

            display_text = f"[{uid}] {name}"
            self.combo_instrument.addItem(display_text, uid)
            self.combo_target.addItem(display_text, uid)

        # --- 2. Restore Action Lookup ---
        action_lookup = meta.get("action_lookup", {})
        if isinstance(action_lookup, dict) and action_lookup:
            self.verbs.clear()
            sorted_vids = []
            for k in action_lookup.keys():
                try:
                    sorted_vids.append(int(k))
                except Exception:
                    continue
            sorted_vids.sort()
            for vid in sorted_vids:
                vname = action_lookup[str(vid)]
                color = self._color_for_index(vid)
                self.verbs.append(LabelDef(name=vname, id=vid, color_name=color))

            self._renumber_verbs()
            self._update_verb_combo()
            self.label_panel.refresh()

        # --- 3. Restore Events and BBoxes ---
        self.events.clear()
        self.event_id_counter = 0

        # Map for deduplication of restored boxes
        restored_boxes_map = {}

        loaded_windows = data.get("windows", [])

        for w_item in loaded_windows:
            # A. Basic Event Info (legacy window_id)
            event_id = w_item.get("window_id", self.event_id_counter)
            frames = w_item.get("frames", [0, 0])
            anomalies = w_item.get("anomaly_labels", {})
            anomaly_label = "Normal"
            if isinstance(anomalies, dict):
                for k, v in anomalies.items():
                    try:
                        if int(v) == 1:
                            anomaly_label = self._normalize_anomaly_label(str(k))
                            break
                    except Exception:
                        continue

            if event_id >= self.event_id_counter:
                self.event_id_counter = event_id + 1

            hoi_data = {
                "Left_hand": {
                    "verb": "",
                    "instrument_object_id": None,
                    "target_object_id": None,
                    "interaction_start": None,
                    "functional_contact_onset": None,
                    "interaction_end": None,
                    "anomaly_label": anomaly_label,
                },
                "Right_hand": {
                    "verb": "",
                    "instrument_object_id": None,
                    "target_object_id": None,
                    "interaction_start": None,
                    "functional_contact_onset": None,
                    "interaction_end": None,
                    "anomaly_label": anomaly_label,
                },
            }

            # B. Parse Hands
            hands_list = w_item.get("hands", [])
            for h in hands_list:
                hid_str = h.get("hand_id", "").lower()

                internal_key = None
                box_label = None

                if "left" in hid_str:
                    internal_key = "Left_hand"
                    box_label = "Left_hand"
                elif "right" in hid_str:
                    internal_key = "Right_hand"
                    box_label = "Right_hand"

                if internal_key:
                    hoi_entry = h.get("hoi", {})
                    hoi_data[internal_key].update(
                        {
                            "verb": hoi_entry.get("verb", ""),
                            "instrument_object_id": hoi_entry.get(
                                "instrument_object_id"
                            ),
                            "target_object_id": hoi_entry.get("target_object_id"),
                            "interaction_start": hoi_entry.get("interaction_start"),
                            "functional_contact_onset": hoi_entry.get(
                                "functional_contact_onset"
                            ),
                            "interaction_end": hoi_entry.get("interaction_end"),
                        }
                    )

                # Restore Hand BBox Track
                if box_label:
                    tracks = h.get("bbox_track", {})
                    for f_str, coords in tracks.items():
                        f_idx = int(f_str)
                        key = (f_idx, box_label)
                        if key not in restored_boxes_map:
                            restored_boxes_map[key] = {
                                "id": self.box_id_counter,
                                "orig_frame": f_idx,
                                "label": box_label,
                                "x1": coords[0],
                                "y1": coords[1],
                                "x2": coords[2],
                                "y2": coords[3],
                            }
                            self.box_id_counter += 1

            # C. Parse Objects
            objs_list = w_item.get("objects", [])
            for obj in objs_list:
                uid = obj.get("object_id")
                obj_name = ""
                for k, v in self.global_object_map.items():
                    if v == uid:
                        obj_name = k
                        break

                tracks = obj.get("bbox_track", {})
                for f_str, coords in tracks.items():
                    f_idx = int(f_str)
                    key = (f_idx, uid)
                    if key not in restored_boxes_map:
                        restored_boxes_map[key] = {
                            "id": uid,
                            "orig_frame": f_idx,
                            "label": obj_name,
                            "x1": coords[0],
                            "y1": coords[1],
                            "x2": coords[2],
                            "y2": coords[3],
                        }

            # D. Build Event object
            new_event = {"event_id": event_id, "frames": frames, "hoi_data": hoi_data}
            self.events.append(new_event)

        # --- 4. Write restored BBoxes to memory ---
        if restored_boxes_map:
            self.raw_boxes.extend(restored_boxes_map.values())
            self._rebuild_bboxes_from_raw()
            self._refresh_boxes_for_frame(self.player.current_frame)

        # --- 5. Refresh UI ---
        self._ensure_verbs_cover_events()
        self._refresh_events()
        self._after_events_loaded()

        video_name = meta.get("video_name", "unknown")
        box_count = len(restored_boxes_map)
        QMessageBox.information(
            self,
            "Loaded",
            f"Successfully loaded {len(self.events)} events.\n"
            f"Restored {box_count} keyframe BBoxes.\n"
            f"Video source: {video_name}",
        )
        self._log(
            "hoi_load_annotations_legacy",
            path=fp,
            events=len(self.events),
            boxes=box_count,
        )
        self._mark_hoi_saved()

    def _load_annotations_v2(self, data: dict):
        """Load unified per-hand event format."""
        self._clear_undo_history()
        # Reset state
        self.events.clear()
        self.event_id_counter = 0
        self.raw_boxes = []
        self.bboxes = {}
        self._clear_relation_highlight()
        self.video_path = data.get("video_path", "") or self.video_path

        # Restore libraries
        self.global_object_map.clear()
        self.id_to_category.clear()
        self.object_id_counter = 0
        self.combo_instrument.clear()
        self.combo_target.clear()
        self.combo_instrument.addItem("None", None)
        self.combo_target.addItem("None", None)

        obj_lib = data.get("object_library", {})
        obj_class_map = {}
        max_obj_id = -1
        for id_str in sorted(obj_lib.keys(), key=lambda x: int(x)):
            try:
                uid = int(id_str)
            except Exception:
                continue
            info = obj_lib.get(id_str, {}) or {}
            label = (
                info.get("label")
                or info.get("name")
                or info.get("category")
                or f"obj_{uid}"
            )
            class_id = info.get("class_id")
            if class_id is not None:
                try:
                    class_id = int(class_id)
                except Exception:
                    pass
                obj_class_map[uid] = class_id
                if (
                    label
                    and class_id not in self.class_map
                    and str(class_id) not in self.class_map
                ):
                    self.class_map[class_id] = label
            self.global_object_map[label] = uid
            self.id_to_category[label] = label
            max_obj_id = max(max_obj_id, uid)
            display_text = f"[{uid}] {label}"
            self.combo_instrument.addItem(display_text, uid)
            self.combo_target.addItem(display_text, uid)
        self.object_id_counter = max_obj_id + 1
        self.box_id_counter = max(self.object_id_counter + 1, 1)

        if "verb_library" in data:
            verb_lib = data.get("verb_library", {}) or {}
            self.verbs.clear()
            for vid_str in sorted(verb_lib.keys(), key=lambda x: int(x)):
                try:
                    vid = int(vid_str)
                except Exception:
                    continue
                name = verb_lib.get(vid_str, "")
                color = self._color_for_index(vid)
                self.verbs.append(LabelDef(name=name, id=vid, color_name=color))
            self._renumber_verbs()
            self._update_verb_combo()
            self.label_panel.refresh()
        rules = data.get("anomaly_rules", {})
        if isinstance(rules, dict) and rules:
            self.anomaly_rules = {}
            for label, rule in rules.items():
                if not isinstance(rule, dict):
                    continue
                self.anomaly_rules[str(label)] = {
                    "allow_missing_bbox": bool(rule.get("allow_missing_bbox")),
                    "allow_missing_verb": bool(rule.get("allow_missing_verb")),
                }
        self._ensure_anomaly_rules()

        # Tracks -> raw boxes
        tracks = data.get("tracks", {})
        track_obj_id = {}

        def _label_for_id(uid):
            for name, id_val in self.global_object_map.items():
                if id_val == uid:
                    return name
            return None

        for track_id, info in tracks.items():
            info = info or {}
            category = str(info.get("category", "")).strip()
            obj_id = info.get("object_id", None)
            track_class_id = info.get("class_id")
            if track_class_id is None and obj_id is not None:
                track_class_id = obj_class_map.get(obj_id)
            track_obj_id[track_id] = obj_id
            boxes = info.get("boxes", []) or []

            tid_lower = str(track_id).lower()
            cat_lower = category.lower()
            is_left = (cat_lower == "left_hand") or (
                "left" in tid_lower and "hand" in tid_lower
            )
            is_right = (cat_lower == "right_hand") or (
                "right" in tid_lower and "hand" in tid_lower
            )

            if is_left:
                label = "Left_hand"
            elif is_right:
                label = "Right_hand"
            else:
                label = _label_for_id(obj_id) or category or f"obj_{obj_id}"

            for entry in boxes:
                frame = entry.get("frame", None)
                bbox = entry.get("bbox", None)
                if frame is None or not bbox or len(bbox) != 4:
                    continue
                try:
                    f_idx = int(frame)
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                except Exception:
                    continue
                box_class_id = entry.get("class_id", track_class_id)
                if box_class_id is not None:
                    try:
                        box_class_id = int(box_class_id)
                    except Exception:
                        pass
                if is_left or is_right or obj_id is None:
                    bid = self.box_id_counter
                    self.box_id_counter += 1
                else:
                    bid = obj_id
                if not (is_left or is_right) and box_class_id is not None and label:
                    if (
                        box_class_id not in self.class_map
                        and str(box_class_id) not in self.class_map
                    ):
                        self.class_map[box_class_id] = label
                new_rb = {
                    "id": bid,
                    "orig_frame": f_idx - self.start_offset,
                    "label": label,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
                if not (is_left or is_right) and box_class_id is not None:
                    new_rb["class_id"] = box_class_id
                self.raw_boxes.append(new_rb)

        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(self.player.current_frame)

        # Events -> internal event entries
        events = data.get("hoi_events", {})

        def _event_to_entry(side_key: str, event: dict):
            s = event.get("start_frame")
            o = event.get("contact_onset_frame")
            e = event.get("end_frame")
            links = event.get("links", {}) or {}
            verb = event.get("verb") or ""
            anomaly = self._normalize_anomaly_label(event.get("anomaly_label"))
            target_tid = links.get("target_track_id")
            tool_tid = links.get("tool_track_id")
            target_id = track_obj_id.get(target_tid) if target_tid else None
            tool_id = track_obj_id.get(tool_tid) if tool_tid else None

            def _safe_int(val, fallback):
                try:
                    return int(val)
                except Exception:
                    return fallback

            start = _safe_int(s, 0)
            end = _safe_int(e, start)
            if end < start:
                start, end = end, start

            empty = {
                "verb": "",
                "instrument_object_id": None,
                "target_object_id": None,
                "interaction_start": None,
                "functional_contact_onset": None,
                "interaction_end": None,
                "anomaly_label": "Normal",
            }
            hoi_data = {"Left_hand": dict(empty), "Right_hand": dict(empty)}
            hand_key = "Left_hand" if side_key == "left_hand" else "Right_hand"
            hoi_data[hand_key] = {
                "verb": verb,
                "instrument_object_id": tool_id,
                "target_object_id": target_id,
                "interaction_start": s,
                "functional_contact_onset": o,
                "interaction_end": e,
                "anomaly_label": anomaly,
            }
            return {
                "event_id": self.event_id_counter,
                "frames": [start, end],
                "hoi_data": hoi_data,
            }

        for side_key in ["left_hand", "right_hand"]:
            for event in events.get(side_key, []) or []:
                self.events.append(_event_to_entry(side_key, event))
                self.event_id_counter += 1

        self._ensure_verbs_cover_events()
        self._refresh_events()
        self._after_events_loaded()

        video_id = data.get("video_id", "")
        QMessageBox.information(
            self,
            "Loaded",
            f"Loaded {len(self.events)} events.\nVideo ID: {video_id or 'unknown'}",
        )

    def _parse_yolo(self, path: str, w: int, h: int) -> List[dict]:
        """
        Supports:
        - Single txt with lines: frame cls xc yc w h [conf]
        - Or directory of per-frame txt (frame_XXXX.txt) with lines: cls xc yc w h [conf]
          coords assumed normalized if <=1, else treated as absolute center/size in px.
        """
        data: List[dict] = []
        if os.path.isdir(path):
            files = sorted([f for f in os.listdir(path) if f.lower().endswith(".txt")])
            for fname in files:
                frame_id = self._frame_from_filename(fname)
                full = os.path.join(path, fname)
                self._parse_yolo_file(full, w, h, data, frame_override=frame_id)
        else:
            self._parse_yolo_file(path, w, h, data, frame_override=None)
        return data

    def _frame_from_filename(self, fname: str) -> int:
        """Extract frame number from filename (e.g., frame_000123.txt -> 123)"""
        name = os.path.splitext(fname)[0]
        # expect frame_000123
        digits = "".join(ch for ch in name if ch.isdigit())
        try:
            return int(digits)
        except Exception:
            return 0

    def _parse_yolo_file(
        self, path: str, w: int, h: int, data: List[dict], frame_override: int = None
    ):
        """Parse individual YOLO text file."""
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) < 5:
                    continue
                idx = 0
                frame = None
                has_frame_token = len(parts) >= 6 and self._looks_number(parts[0])
                if has_frame_token:
                    try:
                        frame = int(float(parts[0]))
                        idx = 1
                    except Exception:
                        frame = None
                        idx = 0
                if frame is None:
                    frame = frame_override if frame_override is not None else 0
                cls = parts[idx]
                coords = parts[idx + 1 : idx + 5]
                if len(coords) < 4:
                    continue
                xc, yc, bw, bh = map(float, coords[:4])
                normed = max(xc, yc, bw, bh) <= 1.0
                if normed:
                    xc *= w
                    yc *= h
                    bw *= w
                    bh *= h
                x1 = max(0.0, xc - bw * 0.5)
                y1 = max(0.0, yc - bh * 0.5)
                x2 = min(float(w), xc + bw * 0.5)
                y2 = min(float(h), yc + bh * 0.5)
                box = {
                    "id": self.box_id_counter,
                    "orig_frame": frame,
                    "label": self._cls_name(cls),
                    "class_id": self._cls_id(cls),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
                self.box_id_counter += 1
                data.append(box)

    @staticmethod
    def _looks_number(txt: str) -> bool:
        try:
            float(txt)
            return True
        except Exception:
            return False

    def _cls_name(self, cls_token):
        try:
            cid = int(float(cls_token))
            if cid in self.class_map:
                return self.class_map[cid]
        except Exception:
            pass
        return str(cls_token)

    def _cls_id(self, cls_token):
        try:
            return int(float(cls_token))
        except Exception:
            return None

    def _build_category_to_default_id(self) -> dict:
        category_to_default_id = {}
        for obj_name, uid in self.global_object_map.items():
            category = self._norm_category(self.id_to_category.get(obj_name, obj_name))
            if not category:
                continue
            if (
                category not in category_to_default_id
                or uid < category_to_default_id[category]
            ):
                category_to_default_id[category] = uid
        return category_to_default_id

    def _norm_category(self, text: str) -> str:
        if not text:
            return ""
        cleaned = str(text).strip().lower()
        cleaned = cleaned.replace("-", "_").replace(" ", "_")
        cleaned = re.sub(r"_+", "_", cleaned)
        return cleaned.strip("_")

    def _parse_yaml_names(self, path: str) -> Dict[int, str]:
        """
        Minimal parser for data.yaml names mapping. Supports both:
        names: {0: 'label', 1: 'label2'} and flat mappings.
        """
        out: Dict = {}
        in_names = False
        pending_lines = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                pending_lines.append(line)
                if line.startswith("names"):
                    in_names = True
                    continue
                if in_names:
                    if ":" in line:
                        key_txt, val_txt = line.split(":", 1)
                        key_txt = key_txt.strip()
                        val_txt = val_txt.strip().strip("'\"")
                        try:
                            idx = int(key_txt)
                            out[idx] = val_txt
                        except Exception:
                            continue
                    else:
                        in_names = False
        if not out:
            for line in pending_lines:
                if ":" not in line:
                    continue
                key_txt, val_txt = line.split(":", 1)
                key_txt = key_txt.strip()
                val_txt = val_txt.strip().strip("'\"")
                if not key_txt.isdigit():
                    continue
                try:
                    idx = int(key_txt)
                    out[idx] = val_txt
                except Exception:
                    continue
        return out

    def _cid_from_label(self, label: str):
        """Return class id for a label name; auto-append if missing."""
        if label is None:
            return None
        for cid, name in self.class_map.items():
            if name == label:
                return cid
        if self.class_map:
            numeric_ids = []
            for k in self.class_map.keys():
                try:
                    numeric_ids.append(int(k))
                except Exception:
                    continue
            new_id = (max(numeric_ids) + 1) if numeric_ids else 0
            self.class_map[new_id] = label
            return new_id
        return None

    def _verb_id_by_name(self, name: str):
        for v in self.verbs:
            if v.name == name:
                return v.id
        return None

    def _replace_verb_in_hoi_data(self, hoi_data: dict, old: str, new: str) -> None:
        """Update verb string in a hoi_data dict for both hands."""
        if not hoi_data or not old or old == new:
            return
        for hand_key in ("Left_hand", "Right_hand"):
            h = hoi_data.get(hand_key, {})
            if h.get("verb") == old:
                h["verb"] = new

    def _collect_event_verbs(self) -> List[str]:
        """Return unique verb strings referenced by any hand in current events."""
        seen = []
        for ev in self.events:
            hoi_data = ev.get("hoi_data", {}) or {}
            for hand_key in ("Left_hand", "Right_hand"):
                verb = (hoi_data.get(hand_key, {}) or {}).get("verb") or ""
                if verb and verb not in seen:
                    seen.append(verb)
        return seen

    def _ensure_verbs_cover_events(self) -> None:
        """
        Ensure the verb list includes every verb referenced in events.
        Used when loading legacy files or events with verbs but no library.
        """
        name_to_id = {v.name: v.id for v in self.verbs}
        cur_max = max(name_to_id.values(), default=-1)
        added = False
        for verb in self._collect_event_verbs():
            if verb in name_to_id:
                continue
            cur_max += 1
            name_to_id[verb] = cur_max
            self.verbs.append(
                LabelDef(
                    name=verb, id=cur_max, color_name=self._color_for_index(cur_max)
                )
            )
            added = True
        if added:
            self._renumber_verbs()
            self._update_verb_combo()
            self.label_panel.refresh()

    def _build_payload_v2(self) -> dict:
        """[Fix] Build payload with STRICT filtering for empty hands."""

        self.events.sort(key=lambda x: x.get("frames", [0, 0])[0])

        base = os.path.splitext(os.path.basename(self.video_path or "annotation"))[0]

        name_to_id = {v.name: v.id for v in self.verbs}
        cur_max_vid = max(name_to_id.values(), default=-1)
        for verb_name in self._collect_event_verbs():
            if verb_name in name_to_id:
                continue
            cur_max_vid += 1
            name_to_id[verb_name] = cur_max_vid
        verb_library = {
            str(v_id): name
            for name, v_id in sorted(name_to_id.items(), key=lambda x: x[1])
        }

        object_library = {}
        for name, uid in sorted(self.global_object_map.items(), key=lambda x: x[1]):
            entry = {"label": name, "category": name}
            cid = self._class_id_for_object(uid, name)
            if cid is not None:
                entry["class_id"] = cid
            object_library[str(uid)] = entry

        if not object_library:
            for rb in self.raw_boxes:
                label = rb.get("label")
                if label in ("Left_hand", "Right_hand"):
                    continue
                uid = rb.get("id")
                if uid is None or str(uid) in object_library:
                    continue
                entry = {"label": str(label), "category": str(label)}
                cid = self._class_id_for_object(uid, label)
                if cid is not None:
                    entry["class_id"] = cid
                object_library[str(uid)] = entry

        def label_for_id(uid):
            for name, id_val in self.global_object_map.items():
                if id_val == uid:
                    return name
            return None

        # --------------------------------------

        # Build per-hand events
        events = {"left_hand": [], "right_hand": []}
        counts = {"Left_hand": 0, "Right_hand": 0}

        for i, event in enumerate(self.events):
            global_start, global_end = event.get("frames", [0, 0])

            for hand_key, side_key in [
                ("Left_hand", "left_hand"),
                ("Right_hand", "right_hand"),
            ]:
                h_data = event.get("hoi_data", {}).get(hand_key, {})

                verb = h_data.get("verb", "")
                target = h_data.get("target_object_id")
                instr = h_data.get("instrument_object_id")
                s = h_data.get("interaction_start")
                o = h_data.get("functional_contact_onset")
                e = h_data.get("interaction_end")
                anom = h_data.get("anomaly_label", "Normal")

                has_verb = bool(verb and verb.strip())
                has_objects = (target is not None) or (instr is not None)
                has_timestamps = (s is not None) or (o is not None) or (e is not None)
                is_abnormal = anom != "Normal"

                has_info = has_verb or has_objects or has_timestamps or is_abnormal

                if not has_info:
                    continue

                counts[hand_key] += 1
                prefix = "L" if hand_key == "Left_hand" else "R"
                event_id = f"{prefix}_{counts[hand_key]:03d}"

                final_start = s if s is not None else global_start
                final_onset = o if o is not None else global_start
                final_end = e if e is not None else global_end

                labels = self._parse_anomaly_labels(anom)
                allow_missing_verb = self._anomaly_rule_allows(
                    labels, "allow_missing_verb"
                )

                event_entry = {
                    "event_id": event_id,
                    "start_frame": final_start,
                    "contact_onset_frame": final_onset,
                    "end_frame": final_end,
                    "anomaly_label": anom,
                }

                if has_verb:
                    event_entry["verb"] = verb
                elif not allow_missing_verb:
                    event_entry["verb"] = ""

                interaction = {}
                tool_label = label_for_id(instr)
                target_label = label_for_id(target)
                if tool_label:
                    interaction["tool"] = tool_label
                if target_label:
                    interaction["target"] = target_label
                event_entry["interaction"] = interaction

                links = {
                    "subject_track_id": (
                        "T_LHAND" if hand_key == "Left_hand" else "T_RHAND"
                    ),
                    "tool_track_id": f"T_OBJ_{instr}" if instr is not None else None,
                    "target_track_id": (
                        f"T_OBJ_{target}" if target is not None else None
                    ),
                }
                event_entry["links"] = links

                events[side_key].append(event_entry)

        tracks = {}

        def collect_hand_boxes(label):
            out = []
            for rb in self.raw_boxes:
                if rb.get("label") == label:
                    out.append(
                        {
                            "frame": rb["orig_frame"],
                            "bbox": [rb["x1"], rb["y1"], rb["x2"], rb["y2"]],
                        }
                    )
            out.sort(key=lambda x: x["frame"])
            return out

        tracks["T_LHAND"] = {
            "category": "left_hand",
            "object_id": None,
            "boxes": collect_hand_boxes("Left_hand"),
        }
        tracks["T_RHAND"] = {
            "category": "right_hand",
            "object_id": None,
            "boxes": collect_hand_boxes("Right_hand"),
        }

        for uid_str, info in object_library.items():
            uid = int(uid_str)
            label = info["label"]
            cat = info["category"]
            cid = info.get("class_id")
            b_list = []
            for rb in self.raw_boxes:
                if rb.get("id") == uid:
                    ent = {
                        "frame": rb["orig_frame"],
                        "bbox": [rb["x1"], rb["y1"], rb["x2"], rb["y2"]],
                    }
                    if cid is not None:
                        ent["class_id"] = cid
                    b_list.append(ent)
            b_list.sort(key=lambda x: x["frame"])
            tracks[f"T_OBJ_{uid}"] = {
                "category": cat,
                "object_id": uid,
                "boxes": b_list,
            }
            if cid is not None:
                tracks[f"T_OBJ_{uid}"]["class_id"] = cid

        return {
            "version": "HOI-1.0-ActionSeg",
            "video_id": base,
            "video_path": self.video_path or "",
            "fps": self.player.frame_rate,
            "frame_size": [self.player._frame_w, self.player._frame_h],
            "frame_count": self.player.frame_count,
            "bbox_mode": "xyxy",
            "bbox_normalized": False,
            "object_library": object_library,
            "verb_library": verb_library,
            "tracks": tracks,
            "hoi_events": events,
        }

    # ---------- UI refresh ----------
    def _sync_slider(self):
        # hooks are driven by VideoPlayer's callbacks; no extra slider here
        pass

    def _set_frame_controls(self, frame: int):
        super()._set_frame_controls(frame)
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_current_frame(frame)

    def _on_frame_advanced(self, frame: int):
        self._refresh_boxes_for_frame(frame)
        self._set_frame_controls(frame)

    def _on_player_playback_state_changed(self, _playing: bool):
        self._update_play_pause_button()

    def _update_play_pause_button(self):
        playing = bool(getattr(self.player, "is_playing", False))
        self.btn_play.setIcon(
            self.style().standardIcon(
                QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay
            )
        )
        self.btn_play.setToolTip("Pause" if playing else "Play")

    def _refresh_boxes_for_frame(self, frame: int, skip_events: bool = False):
        """
        Refresh logic: prepare boxes, update list, update player overlay, refresh relations.
        """
        self._update_overlay(frame)
        highlights = getattr(self, "_validation_highlights", {}) or {}
        highlight_ids = highlights.get("by_id", {})
        highlight_labels = highlights.get("by_label", {})
        boxes = self.bboxes.get(frame, [])
        self.current_hands = {"Left_hand": None, "Right_hand": None}

        self.list_objects.blockSignals(True)
        self.list_objects.clear()

        display_boxes = []

        for b in boxes:
            lbl = str(b.get("label"))
            norm_hand = self._normalize_hand_label(lbl)
            if norm_hand:
                self.current_hands[norm_hand] = b
                hand_short = "L" if norm_hand == "Left_hand" else "R"
                item_txt = f"[Hand] {hand_short}"
                draw_label = hand_short
                color = "#ff00ff"
            else:
                uid = b.get("id")
                name = self._object_name_for_id(uid, fallback=f"ID_{uid}")
                if name.startswith("ID_") and b.get("label"):
                    name = str(b.get("label"))
                cid = b.get("class_id")
                if cid is None:
                    cid = self._class_id_for_label(b.get("label") or name)
                cid_txt = str(cid) if cid is not None else "_"
                item_txt = f"[obj:{uid} | cls:{cid_txt}] {name}"
                draw_label = item_txt
                color = self._class_color(cid)

            item = QListWidgetItem(item_txt)
            item.setData(Qt.UserRole, b)
            self.list_objects.addItem(item)
            thick = False
            if self.validation_enabled:
                if b.get("id") in highlight_ids:
                    color = highlight_ids[b.get("id")]
                    thick = True
                elif norm_hand and norm_hand in highlight_labels:
                    color = highlight_labels[norm_hand]
                    thick = True

            display_boxes.append(
                {
                    "x1": b["x1"],
                    "y1": b["y1"],
                    "x2": b["x2"],
                    "y2": b["y2"],
                    "label": draw_label,
                    "color": color,
                    "thick": thick,
                }
            )

        self.list_objects.blockSignals(False)

        if hasattr(self.player, "set_overlay_boxes"):
            self.player.set_overlay_boxes(display_boxes)
        elif hasattr(self.player, "set_boxes"):
            self.player.set_boxes(display_boxes)
        else:
            print("Warning: VideoPlayer missing set_overlay_boxes method")

        if hasattr(self.player, "set_edit_context"):
            if self.chk_edit_boxes.isChecked():
                edit_boxes = []
                for b in boxes:
                    edit_boxes.append(
                        {
                            "id": b.get("id"),
                            "frame": frame,
                            "orig_frame": b.get("orig_frame"),
                            "x1": b.get("x1"),
                            "y1": b.get("y1"),
                            "x2": b.get("x2"),
                            "y2": b.get("y2"),
                            "label": b.get("label"),
                            "class_id": b.get("class_id"),
                        }
                    )
                self.player.set_edit_context(
                    edit_boxes,
                    on_change=self._on_box_edited,
                    label_resolver=self._resolve_label_and_id,
                    label_suggestions=self._label_suggestions(),
                    auto_label_fetcher=self._get_auto_draw_label,
                )
            else:
                self.player.set_edit_context(
                    [],
                    on_change=None,
                    label_resolver=None,
                    auto_label_fetcher=None,
                )

        if not skip_events:
            self._refresh_events(refresh_boxes=False)

    def _get_auto_draw_label(self):
        if not self.selected_hand_label:
            return None
        hand_data = self.event_draft.get(self.selected_hand_label) or {}
        target_obj_id = None
        if getattr(self, "rad_draw_inst", None) and self.rad_draw_inst.isChecked():
            target_obj_id = hand_data.get("instrument_object_id")
        elif getattr(self, "rad_draw_target", None) and self.rad_draw_target.isChecked():
            target_obj_id = hand_data.get("target_object_id")
        if target_obj_id is None:
            return None
        for name, uid in self.global_object_map.items():
            if uid == target_obj_id:
                return name, target_obj_id
        return None

    def _refresh_events(self, refresh_boxes: bool = True):
        """[Step 3 Fix] Refresh list: show detailed info including Instrument for error checking."""
        frame = self.player.current_frame
        for ev in self.events:
            self._sync_event_frames(ev)
        self._update_overlay(frame)
        if refresh_boxes and self.validation_enabled:
            self._refresh_boxes_for_frame(frame, skip_events=True)
        self._update_incomplete_indicator()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.refresh()

    def _update_overlay(self, frame: int):
        """
        [Step 2 Fix] Draw Hand -> Instrument -> Target relation lines on video.
        Adapted to self.events structure.
        """
        hand_color = "#3b82f6"
        instrument_color = "#f59e0b"
        target_color = "#22c55e"
        display_rels = []

        # 1. Valid HOIs from committed events
        for ev in self.events:
            for hand_key in ["Left_hand", "Right_hand"]:
                h_data = ev["hoi_data"][hand_key]
                s = h_data.get("interaction_start")
                e = h_data.get("interaction_end")

                if s is not None and e is not None and s <= frame <= e:
                    draw_item = dict(h_data)
                    draw_item["hand_id"] = hand_key
                    display_rels.append(draw_item)

        # 2. Preview from current Draft
        if hasattr(self, "event_draft"):
            for hand_key in ["Left_hand", "Right_hand"]:
                h_data = self.event_draft[hand_key]
                if h_data.get("target_object_id"):
                    draw_item = dict(h_data)
                    draw_item["hand_id"] = hand_key
                    display_rels.append(draw_item)

        overlay_data = []
        highlight_ids = {}
        highlight_labels = {}
        current_boxes = {b["id"]: b for b in self.bboxes.get(frame, [])}

        for r in display_rels:
            hand_label = r.get("hand_id")
            hand_box = None
            for b in current_boxes.values():
                if self._normalize_hand_label(b.get("label")) == hand_label:
                    hand_box = b
                    break

            if not hand_box:
                continue

            inst_id = r.get("instrument_object_id")
            target_id = r.get("target_object_id")
            verb = r.get("verb", "unknown")

            # Case A: Only Target
            if inst_id is None and target_id is not None:
                if target_id in current_boxes:
                    overlay_data.append(
                        {
                            "box_a": hand_box,
                            "box_b": current_boxes[target_id],
                            "label": verb,
                            "color": target_color,
                        }
                    )
                    if self.validation_enabled:
                        highlight_labels[hand_label] = hand_color
                        highlight_ids[target_id] = target_color

            # Case B: With Instrument
            elif inst_id is not None:
                if inst_id in current_boxes:
                    # Line 1: Hand -> Instrument
                    inst_box = current_boxes[inst_id]
                    overlay_data.append(
                        {
                            "box_a": hand_box,
                            "box_b": inst_box,
                            "label": "",
                            "color": instrument_color,
                        }
                    )
                    if self.validation_enabled:
                        highlight_labels[hand_label] = hand_color
                        highlight_ids[inst_id] = instrument_color
                    # Line 2: Instrument -> Target
                    if target_id is not None and target_id in current_boxes:
                        overlay_data.append(
                            {
                                "box_a": inst_box,
                                "box_b": current_boxes[target_id],
                                "label": verb,
                                "color": target_color,
                            }
                        )
                        if self.validation_enabled:
                            highlight_ids[target_id] = target_color

        if self.validation_enabled:
            self._validation_highlights = {
                "by_id": highlight_ids,
                "by_label": highlight_labels,
            }
        else:
            self._validation_highlights = {}

        overlay_lines = []
        for rel in overlay_data:
            try:
                box_a = rel["box_a"]
                box_b = rel["box_b"]
                x1 = (float(box_a["x1"]) + float(box_a["x2"])) / 2.0
                y1 = (float(box_a["y1"]) + float(box_a["y2"])) / 2.0
                x2 = (float(box_b["x1"]) + float(box_b["x2"])) / 2.0
                y2 = (float(box_b["y1"]) + float(box_b["y2"])) / 2.0
            except Exception:
                continue
            overlay_lines.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": rel.get("label", ""),
                    "color": rel.get("color", ""),
                }
            )
        self.player.set_overlay_relations(overlay_lines)

    def _stop(self):
        try:
            self.player.stop()
        except Exception:
            pass
        self._update_play_pause_button()
        self._set_frame_controls(self.player.current_frame)
        self._refresh_boxes_for_frame(self.player.current_frame)
        self._log("hoi_stop")

    def _toggle_play_pause(self):
        if getattr(self.player, "is_playing", False):
            self._pause()
        else:
            self._play()

    def _play(self):
        try:
            self.player.play()
        except Exception:
            pass
        self._update_play_pause_button()
        self._log("hoi_play")

    def _pause(self):
        try:
            self.player.pause()
        except Exception:
            pass
        self._update_play_pause_button()
        self._log("hoi_pause")

    def _seek_relative(self, delta: int):
        if not self.player.cap:
            return
        target = max(
            self.player.crop_start,
            min(self.player.crop_end, self.player.current_frame + delta),
        )
        self.player.seek(target)
        self._refresh_boxes_for_frame(target)
        self._set_frame_controls(target)
        self._log("hoi_seek_relative", delta=delta, target=target)

    def _seek_seconds(self, direction: int):
        """Seek by +/-1 second based on video FPS."""
        if not self.player.cap:
            return
        fps = max(1, int(round(self.player.frame_rate or 30)))
        self._seek_relative(direction * fps)

    def _jump_to_spin(self):
        frame = int(self.spin_jump.value())
        if self.player.cap:
            self.player.seek(frame)
            self._refresh_boxes_for_frame(frame)
            self.slider.blockSignals(True)
            self.slider.setValue(frame)
            self.slider.blockSignals(False)
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.set_current_frame(frame)
            self._log("hoi_jump_to_frame", target=frame)

    def _on_slider_changed(self, val: int):
        if self.player.cap:
            self.player.seek(val)
            self._refresh_boxes_for_frame(val)
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.set_current_frame(val)
            self._log("hoi_seek_slider", target=val)

    def _on_offset_changed(self):
        self.start_offset = int(self.spin_start_offset.value())
        end_val = int(self.spin_end_frame.value())
        self.end_frame = end_val if end_val >= self.start_offset else None
        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(self.player.current_frame)
        self._log("hoi_crop_set", start=self.start_offset, end=self.end_frame)

    def _on_edit_boxes_toggled(self, on: bool):
        for widget in (
            getattr(self, "rad_draw_none", None),
            getattr(self, "rad_draw_inst", None),
            getattr(self, "rad_draw_target", None),
        ):
            if widget is not None:
                widget.setEnabled(bool(on))
        self._refresh_boxes_for_frame(self.player.current_frame)
        self._log("hoi_edit_boxes_toggle", on=bool(on))

    def _set_validation_ui_state(self, on: bool):
        if getattr(self, "lbl_validation", None):
            if on:
                self.lbl_validation.setStyleSheet("color: #12b76a; font-weight: 600;")
            else:
                self.lbl_validation.setStyleSheet("")
        if getattr(self, "btn_validation", None):
            if on:
                self.btn_validation.setToolTip("Return to annotation mode")
            else:
                self.btn_validation.setToolTip("Toggle validation on/off")

    def _on_validation_toggled(self, on: bool):
        if on:
            name, ok = QInputDialog.getText(self, "Validation", "Editor name:")
            if not ok or not name.strip():
                self.btn_validation.blockSignals(True)
                self.btn_validation.setChecked(False)
                self.btn_validation.blockSignals(False)
                self._set_validation_ui_state(False)
                self._log("hoi_validation_cancel")
                return
            self.validator_name = name.strip()
            self.validation_enabled = True
            self.validation_session_active = True
            self.validation_started_at = datetime.now().isoformat(timespec="seconds")
            self.validation_modified = False
            self.validation_change_count = 0
            if getattr(self, "validation_op_logger", None):
                self.validation_op_logger.enabled = self.op_logger.enabled
                self.validation_op_logger.clear()
            self._set_validation_ui_state(True)
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._log("hoi_validation_on", validator=self.validator_name)
        else:
            self.validation_enabled = False
            self._set_validation_ui_state(False)
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._log("hoi_validation_off")

    def _on_task_combo_changed(self, text: str):
        lower = (text or "").lower()
        if ("handoi" in lower) or ("hoi" in lower):
            return
        if callable(self._on_switch_task):
            self._on_switch_task(text)

    def _hoi_state_signature(self) -> str:
        payload = {
            "events": self.events,
            "raw_boxes": self.raw_boxes,
            "event_draft": getattr(self, "event_draft", None),
            "object_map": self.global_object_map,
            "class_map": self.class_map,
            "start_offset": self.start_offset,
            "end_frame": self.end_frame,
        }
        try:
            blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        except Exception:
            blob = str(
                (
                    len(self.events),
                    len(self.raw_boxes),
                    self.start_offset,
                    self.end_frame,
                )
            )
        return hashlib.md5(blob.encode("utf-8")).hexdigest()

    def _mark_hoi_saved(self) -> None:
        self._hoi_saved_signature = self._hoi_state_signature()

    def _hoi_has_unsaved_changes(self) -> bool:
        sig = self._hoi_state_signature()
        if self._hoi_saved_signature is None:
            has_data = bool(
                self.events or self.raw_boxes or getattr(self, "event_draft", None)
            )
            return has_data
        return sig != self._hoi_saved_signature

    def _confirm_task_switch(self, _target: str = "") -> bool:
        if not self._hoi_has_unsaved_changes():
            return True
        msg = "HandOI has unsaved changes.\nSwitch tasks anyway?"
        reply = QMessageBox.question(
            self,
            "Unsaved HandOI changes",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _on_object_selection(self):
        if not self.selected_hand_label:
            self._clear_relation_highlight()
            self._update_overlay(self.player.current_frame)
            return

        items = self.list_objects.selectedItems()
        if not items:
            return
        obj_box = items[0].data(Qt.UserRole)
        if not obj_box:
            return
        if str(obj_box.get("label")) in ("Left_hand", "Right_hand"):
            return
        obj_id = obj_box["id"]

        hand_data = self.event_draft.get(self.selected_hand_label)
        if not hand_data:
            return
        prev_inst = hand_data["instrument_object_id"]
        prev_tar = hand_data["target_object_id"]

        if hand_data["instrument_object_id"] is None:
            hand_data["instrument_object_id"] = obj_id
        elif hand_data["target_object_id"] is None:
            if obj_id == hand_data["instrument_object_id"]:
                hand_data["instrument_object_id"] = None
                hand_data["target_object_id"] = obj_id
            else:
                hand_data["target_object_id"] = obj_id

        self._load_hand_draft_to_ui(self.selected_hand_label)
        self._update_overlay(self.player.current_frame)
        self._apply_draft_to_selected_event()
        self._refresh_events()
        self._update_hoi_titles()
        self.hoi_timeline.refresh()
        if (
            prev_inst != hand_data["instrument_object_id"]
            or prev_tar != hand_data["target_object_id"]
        ):
            self._log(
                "hoi_select_object",
                hand=self.selected_hand_label,
                instrument=hand_data["instrument_object_id"],
                target=hand_data["target_object_id"],
            )

    def _select_hand(self, hand_label: str, on: bool):
        """[Step 2] Switch hand context: save old data -> switch -> load new data."""

        # 1. Mutex Logic & UI State Management
        if on:
            # Save previous hand state
            if self.selected_hand_label and self.selected_hand_label != hand_label:
                self._save_ui_to_hand_draft(self.selected_hand_label)
                self._apply_draft_to_selected_event()

            # Set new state
            self.selected_hand_label = hand_label

            # Enforce Mutex UI
            if hand_label == "Left_hand":
                self.chk_right.blockSignals(True)
                self.chk_right.setChecked(False)
                self.chk_right.blockSignals(False)
            else:
                self.chk_left.blockSignals(True)
                self.chk_left.setChecked(False)
                self.chk_left.blockSignals(False)

            # Load new hand data to UI
            self._load_hand_draft_to_ui(hand_label)
            self._log("hoi_select_hand", hand=hand_label, on=True)
        else:
            # Uncheck current hand
            if self.selected_hand_label == hand_label:
                self._save_ui_to_hand_draft(hand_label)  # Save final state
                self._apply_draft_to_selected_event()
                self.selected_hand_label = None
                self._update_status_label()
                self._log("hoi_select_hand", hand=hand_label, on=False)

        self.list_objects.setEnabled(self.selected_hand_label is not None)
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_selected(
                self.selected_event_id, self.selected_hand_label
            )
            self._update_hoi_titles()
        self._update_overlay(self.player.current_frame)

    def _clear_relation_highlight(self):
        self.selected_event_id = None
        self.selected_hand_label = None
        self._reset_event_draft()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_selected(None, None)
            self._update_hoi_titles()
        self._update_overlay(self.player.current_frame)

    def _on_box_edited(self, box_id: int, new_box: dict):
        """Apply edits to the raw_boxes by id and refresh views."""
        action = new_box.get("_action")
        target_frame = new_box.get("frame")
        target_orig = new_box.get("orig_frame")
        if target_orig is None and target_frame is not None:
            try:
                target_orig = int(target_frame) - int(self.start_offset)
            except Exception:
                target_orig = None

        def _matches(rb):
            if box_id is not None and rb.get("id") != box_id:
                return False
            if target_orig is not None and rb.get("orig_frame") != target_orig:
                return False
            return True

        if action == "delete":
            if any(_matches(rb) for rb in self.raw_boxes):
                self._push_undo()
                self.raw_boxes = [rb for rb in self.raw_boxes if not _matches(rb)]
                self._rebuild_bboxes_from_raw()
                self._refresh_boxes_for_frame(self.player.current_frame)
                self._log(
                    "hoi_box_delete", box_id=box_id, frame=self.player.current_frame
                )
            return

        label_txt = new_box.get("label")
        new_class_id = new_box.get("class_id", None)
        if label_txt is not None:
            lbl_res, cid_res, _ = self._resolve_label_and_id(label_txt)
            if lbl_res:
                label_txt = lbl_res
            if cid_res is not None:
                new_class_id = cid_res

        if action == "add":
            if not label_txt:
                return

            obj_id = None
            if self.global_object_map and self.id_to_category:
                norm_label = self._norm_category(label_txt)
                if norm_label:
                    category_to_default_id = self._build_category_to_default_id()
                    obj_id = category_to_default_id.get(norm_label)

            if target_orig is None:
                try:
                    target_orig = int(self.player.current_frame) - int(
                        self.start_offset
                    )
                except Exception:
                    target_orig = 0

            is_new_object = False
            if obj_id is None:
                obj_id = self.object_id_counter
                is_new_object = True

            self._push_undo()
            if new_class_id is not None and label_txt:
                if (
                    new_class_id not in self.class_map
                    and str(new_class_id) not in self.class_map
                ):
                    self.class_map[new_class_id] = label_txt
            if is_new_object:
                self.object_id_counter += 1
                self._register_object_entry(obj_id, label_txt)

            new_rb = {
                "id": obj_id,
                "orig_frame": target_orig,
                "label": label_txt,
                "x1": new_box.get("x1"),
                "y1": new_box.get("y1"),
                "x2": new_box.get("x2"),
                "y2": new_box.get("y2"),
            }
            if new_class_id is not None:
                new_rb["class_id"] = new_class_id

            self.raw_boxes.append(new_rb)
            self._rebuild_bboxes_from_raw()
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._log(
                "hoi_box_add",
                box_id=obj_id,
                class_id=new_class_id,
                label=label_txt,
                frame=self.player.current_frame,
            )
            return

        changed = False
        if any(_matches(rb) for rb in self.raw_boxes):
            self._push_undo()
        if new_class_id is not None and label_txt:
            if (
                new_class_id not in self.class_map
                and str(new_class_id) not in self.class_map
            ):
                self.class_map[new_class_id] = label_txt
        for rb in self.raw_boxes:
            if _matches(rb):
                for key in ("x1", "y1", "x2", "y2"):
                    if key in new_box and new_box[key] is not None:
                        rb[key] = new_box[key]
                if label_txt is not None:
                    rb["label"] = label_txt
                if new_class_id is not None:
                    rb["class_id"] = new_class_id
                changed = True
        if changed:
            self._rebuild_bboxes_from_raw()
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._log(
                "hoi_box_edit",
                box_id=box_id,
                class_id=new_class_id,
                label=label_txt,
                frame=self.player.current_frame,
            )

    def _resolve_label_and_id(self, text: str):
        """
        Resolve text to (label, class_id, known_flag).
        Matches numeric ids or class names.
        """
        if text is None:
            return "", None, False
        stripped = str(text).strip()
        try:
            cid = int(stripped)
            if cid in self.class_map:
                return self.class_map[cid], cid, True
            if str(cid) in self.class_map:
                return self.class_map[str(cid)], cid, True
            for rb in self.raw_boxes:
                if rb.get("class_id") == cid and rb.get("label"):
                    return str(rb.get("label")), cid, True
            return stripped, cid, False
        except Exception:
            pass
        lowered = stripped.lower()
        for k, v in self.class_map.items():
            try:
                if str(v).lower() == lowered:
                    try:
                        cid_val = int(k)
                    except Exception:
                        cid_val = k
                    return v, cid_val, True
            except Exception:
                continue
        for rb in self.raw_boxes:
            if str(rb.get("label", "")).lower() == lowered:
                return rb.get("label"), rb.get("class_id"), True
        return stripped, None, False

    def _class_id_for_label(self, label: str):
        if not label:
            return None
        target = str(label).lower().strip()
        for key, val in self.class_map.items():
            try:
                if str(val).lower().strip() == target:
                    try:
                        return int(key)
                    except Exception:
                        return key
            except Exception:
                continue
        return None

    def _class_id_for_object(self, uid, label=None):
        for rb in self.raw_boxes:
            if rb.get("id") == uid and rb.get("label") not in (
                "Left_hand",
                "Right_hand",
            ):
                cid = rb.get("class_id")
                if cid is not None:
                    return cid
        if label:
            return self._class_id_for_label(label)
        return None

    def _class_color(self, class_id):
        if class_id is None:
            return "#00ffff"
        try:
            idx = abs(int(class_id))
        except Exception:
            idx = abs(hash(str(class_id)))
        color_key = self._color_for_index(idx)
        return color_from_key(color_key).name()

    def _object_name_for_id(
        self, uid, default_for_none: str = "_", fallback: str = None
    ) -> str:
        if uid is None:
            return default_for_none
        for name, id_val in self.global_object_map.items():
            if id_val == uid:
                return name
        if fallback is not None:
            return fallback
        return str(uid)

    def _register_object_entry(self, uid, label: str):
        """Ensure a new object id is represented in the registry and combos."""
        if uid is None:
            return
        for name, id_val in self.global_object_map.items():
            if id_val == uid:
                return
        base = (label or f"obj_{uid}").strip()
        if not base:
            base = f"obj_{uid}"
        name = base
        if name in self.global_object_map:
            idx = 1
            while f"{base}_{idx}" in self.global_object_map:
                idx += 1
            name = f"{base}_{idx}"
        self.global_object_map[name] = uid
        self.id_to_category[name] = base
        display_text = f"[{uid}] {name}"
        if self.combo_instrument.findData(uid) == -1:
            self.combo_instrument.addItem(display_text, uid)
        if self.combo_target.findData(uid) == -1:
            self.combo_target.addItem(display_text, uid)
        if uid >= self.object_id_counter:
            self.object_id_counter = uid + 1

    def _rebuild_object_combos(self, inst_selected=None, tgt_selected=None):
        """Rebuild instrument/target combos from the current object library."""
        self.combo_instrument.blockSignals(True)
        self.combo_target.blockSignals(True)
        self.combo_instrument.clear()
        self.combo_target.clear()
        self.combo_instrument.addItem("None", None)
        self.combo_target.addItem("None", None)
        for name, uid in sorted(self.global_object_map.items(), key=lambda x: x[1]):
            display_text = f"[{uid}] {name}"
            self.combo_instrument.addItem(display_text, uid)
            self.combo_target.addItem(display_text, uid)
        if inst_selected is not None:
            idx = self.combo_instrument.findData(inst_selected)
            if idx >= 0:
                self.combo_instrument.setCurrentIndex(idx)
        if tgt_selected is not None:
            idx = self.combo_target.findData(tgt_selected)
            if idx >= 0:
                self.combo_target.setCurrentIndex(idx)
        self.combo_instrument.blockSignals(False)
        self.combo_target.blockSignals(False)

    def _label_suggestions(self):
        suggestions = []
        for cid, name in self.class_map.items():
            suggestions.append(str(cid))
            if name:
                suggestions.append(str(name))
        seen = set()
        ordered = []
        for item in suggestions:
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def closeEvent(self, e):
        ok_incomplete, _ = self._check_incomplete_hoi(context="close")
        if not ok_incomplete:
            e.ignore()
            return
        if callable(self._on_close):
            try:
                self._on_close()
            except Exception:
                pass
        return super().closeEvent(e)

    def _rebuild_bboxes_from_raw(self):
        self.bboxes = {}
        for b in self.raw_boxes:
            target_frame = b.get("orig_frame", 0) + self.start_offset
            if self.end_frame is not None and target_frame > self.end_frame:
                continue
            mapped = dict(b)
            mapped["frame"] = target_frame
            self.bboxes.setdefault(target_frame, []).append(mapped)

    def _load_hands_xml(self):
        """[Fix] Load hand bbox XML with Auto-Alignment for clipped videos."""
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load hand bbox XML", "", "XML Files (*.xml);;All Files (*)"
        )
        if not fp:
            return

        if not self.player.cap:
            QMessageBox.information(self, "Info", "Please load a video first.")
            return

        try:
            parsed = self._parse_cvat_xml(fp)

            if not parsed:
                QMessageBox.warning(self, "Parse Error", "No valid hand boxes found.")
                return

            min_xml_frame = min(p["orig_frame"] for p in parsed)
            max_xml_frame = max(p["orig_frame"] for p in parsed)
            video_frame_count = self.player.frame_count
            offset_needed = 0

            if min_xml_frame > 0:
                msg = (
                    f"XML data starts at frame {min_xml_frame} (End: {max_xml_frame}).\n"
                    f"Video length is {video_frame_count} frames.\n\n"
                    f"Do you want to AUTO-ALIGN the data?\n"
                    f"(Shift frames by -{min_xml_frame}, so Frame {min_xml_frame} becomes Frame 0)"
                )

                reply = QMessageBox.question(
                    self, "Auto Align?", msg, QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    offset_needed = min_xml_frame
                    for p in parsed:
                        p["orig_frame"] -= offset_needed

                    max_xml_frame -= offset_needed
                    print(
                        f"DEBUG: Applied offset -{offset_needed}. New range: 0 to {max_xml_frame}"
                    )

            final_max_frame = max_xml_frame + self.start_offset

            if final_max_frame >= video_frame_count:
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"Hand bbox frames exceed video length!\n\n"
                    f"Max Data Frame: {final_max_frame}\n"
                    f"Video Length: {video_frame_count}\n\n"
                    "If you chose 'No' to alignment, try loading again and verify offsets.",
                )

        except Exception as ex:
            import traceback

            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to parse XML:\n{ex}")
            return

        self._push_undo()

        self.raw_boxes = [
            b for b in self.raw_boxes if not self._is_hand_label(b.get("label"))
        ]

        self.raw_boxes.extend(parsed)
        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(self.player.current_frame)

        success_msg = (
            f"Loaded {len(parsed)} hand boxes.\nOriginal Start: {min_xml_frame}"
        )
        if offset_needed > 0:
            success_msg += f"\nAuto-Aligned: shifted by -{offset_needed} frames."

        QMessageBox.information(self, "Success", success_msg)
        self._log(
            "hoi_load_hands_xml",
            path=fp,
            count=len(parsed),
            offset_shifted=offset_needed,
        )
        self._mark_hoi_saved()

    def _parse_cvat_xml(self, path: str) -> List[dict]:
        """
        [Updated] Robust parser for CVAT XML.
        - Supports 'CVAT for Images' (your current format).
        - Supports 'CVAT for Video' (legacy format).
        - Auto-corrects labels: "left"->"Left_hand", "right"->"Right_hand".
        - Extracts frame numbers from filenames like "...-00135.jpg".
        """
        import xml.etree.ElementTree as ET
        import re

        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception as e:
            raise ValueError(f"Invalid XML file: {e}")

        out = []

        def normalize_label(l_str):
            if not l_str:
                return None
            l_lower = str(l_str).lower().strip()
            if l_lower in ["left", "left_hand", "l_hand", "left hand"]:
                return "Left_hand"
            if l_lower in ["right", "right_hand", "r_hand", "right hand"]:
                return "Right_hand"
            return None

        def extract_frame_from_name(name_str):
            matches = re.findall(r"\d+", name_str)
            if matches:
                return int(matches[-1])
            return 0

        images = root.findall("image")
        if images:
            print(
                f"Debug: Found {len(images)} image tags. Parsing 'CVAT for Images' format..."
            )
            for img in images:
                name = img.attrib.get("name", "")
                img_id = img.attrib.get("id")

                try:
                    frame = extract_frame_from_name(name)
                except Exception:
                    frame = int(img_id) if img_id else 0

                for box in img.findall("box"):
                    raw_label = box.attrib.get("label")
                    label = normalize_label(raw_label)

                    if label is None:
                        continue

                    try:
                        out.append(
                            {
                                "id": self.box_id_counter,
                                "orig_frame": frame,
                                "label": label,
                                "x1": float(box.attrib.get("xtl")),
                                "y1": float(box.attrib.get("ytl")),
                                "x2": float(box.attrib.get("xbr")),
                                "y2": float(box.attrib.get("ybr")),
                            }
                        )
                        self.box_id_counter += 1
                    except (ValueError, TypeError):
                        continue

            if out:
                return out

        tracks = root.findall("track")
        if tracks:
            print(
                f"Debug: Found {len(tracks)} track tags. Parsing 'CVAT for Video' format..."
            )
            for track in tracks:
                raw_label = track.attrib.get("label")
                label = normalize_label(raw_label)
                if label is None:
                    continue

                for box in track.findall("box"):
                    if box.attrib.get("outside") == "1":
                        continue
                    try:
                        out.append(
                            {
                                "id": self.box_id_counter,
                                "orig_frame": int(box.attrib.get("frame")),
                                "label": label,
                                "x1": float(box.attrib.get("xtl")),
                                "y1": float(box.attrib.get("ytl")),
                                "x2": float(box.attrib.get("xbr")),
                                "y2": float(box.attrib.get("ybr")),
                            }
                        )
                        self.box_id_counter += 1
                    except Exception:
                        continue

        return out

    def _validate_integrity(self, relation: dict) -> list:
        """
        Check BBox existence for Start/Onset/End frames in a Relation.
        Returns error list: [{'msg': '...', 'frame': 123}, ...]
        """
        errors = []

        check_points = {
            "Start": relation.get("interaction_start"),
            "Onset": relation.get("functional_contact_onset"),
            "End": relation.get("interaction_end"),
        }

        targets = [
            ("Instrument", relation.get("instrument_object_id")),
            ("Target", relation.get("target_object_id")),
        ]

        for time_label, frame in check_points.items():
            if frame is None:
                continue

            frame_boxes = self.bboxes.get(frame, [])
            existing_ids = set(b.get("id") for b in frame_boxes)

            for role, uid in targets:
                if uid is None:
                    continue

                if uid not in existing_ids:
                    name = self._object_name_for_id(
                        uid, default_for_none="", fallback=""
                    )

                    errors.append(
                        {
                            "msg": f"Frame {frame} ({time_label}) missing {role}: [{uid}] {name}",
                            "frame": frame,
                        }
                    )
        return errors

    def _add_anomaly_item(self, name: str, checked: bool = False):
        item = QListWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.anomaly_list.addItem(item)

    def _find_anomaly_item(self, name: str):
        for i in range(self.anomaly_list.count()):
            it = self.anomaly_list.item(i)
            if it.text().strip().lower() == name.strip().lower():
                return it
        return None

    def _set_anomaly_checked(self, name: str, checked: bool):
        it = self._find_anomaly_item(name)
        if not it:
            return
        it.setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _any_anomaly_checked(self) -> bool:
        for i in range(self.anomaly_list.count()):
            if self.anomaly_list.item(i).checkState() == Qt.Checked:
                return True
        return False

    def _selected_anomaly_label(self) -> str:
        """Return checked anomaly labels as a comma-separated string."""
        selected = []
        for i in range(self.anomaly_list.count()):
            it = self.anomaly_list.item(i)
            if it.checkState() == Qt.Checked:
                selected.append(it.text())
        if not selected:
            return "Normal"
        if "Normal" in selected and len(selected) > 1:
            selected = [t for t in selected if t != "Normal"]
        return ", ".join(selected)

    def _set_selected_anomaly_label(self, label_str: str):
        """Parse stored label(s) and select all matching items."""
        target_keys = [t.strip().lower() for t in self._parse_anomaly_labels(label_str)]
        if not target_keys:
            target_keys = ["normal"]
        self._anomaly_block = True
        try:
            found_any = False
            for i in range(self.anomaly_list.count()):
                it = self.anomaly_list.item(i)
                txt = it.text().strip().lower()
                if txt in target_keys:
                    it.setCheckState(Qt.Checked)
                    found_any = True
                else:
                    it.setCheckState(Qt.Unchecked)
            if not found_any:
                self._set_anomaly_checked("Normal", True)
        finally:
            self._anomaly_block = False

    def _on_anomaly_item_changed(self, item: QListWidgetItem):
        """
        Support multi-select anomaly labels with Normal as the exclusive fallback.
        """
        if self._anomaly_block:
            return

        self._anomaly_block = True
        try:
            is_checked = item.checkState() == Qt.Checked
            curr_text = item.text().strip()
            if is_checked:
                if curr_text.lower() == "normal":
                    for i in range(self.anomaly_list.count()):
                        it = self.anomaly_list.item(i)
                        if it is not item:
                            it.setCheckState(Qt.Unchecked)
                else:
                    for i in range(self.anomaly_list.count()):
                        it = self.anomaly_list.item(i)
                        if it.text().strip().lower() == "normal":
                            it.setCheckState(Qt.Unchecked)

            if not self._any_anomaly_checked():
                self._set_anomaly_checked("Normal", True)

            if (
                self.selected_hand_label
                and self.selected_hand_label in self.event_draft
            ):
                self.event_draft[self.selected_hand_label][
                    "anomaly_label"
                ] = self._selected_anomaly_label()
                self._update_status_label()
                if self.selected_event_id is not None:
                    self._apply_draft_to_selected_event()
                    self._refresh_events()
                    if getattr(self, "hoi_timeline", None):
                        self.hoi_timeline.refresh()

        finally:
            self._anomaly_block = False

    def _add_anomaly_label(self):
        name = self.anomaly_edit.text().strip()
        if not name:
            QMessageBox.information(self, "Info", "Please input a label name.")
            return
        if self._find_anomaly_item(name):
            QMessageBox.information(self, "Info", f"Label '{name}' already exists.")
            return
        self._add_anomaly_item(name, checked=False)
        self.anomaly_labels.append(name)
        self._ensure_anomaly_rules()
        self.anomaly_edit.clear()
        self._log("hoi_anomaly_add", name=name)

    def _remove_anomaly_label(self):
        checked_items = []
        for i in range(self.anomaly_list.count()):
            it = self.anomaly_list.item(i)
            if it.checkState() == Qt.Checked:
                checked_items.append(it)
        if not checked_items:
            return
        item = checked_items[0]
        name = item.text().strip().lower()
        if name == "normal":
            QMessageBox.information(
                self, "Info", "The 'Normal' label cannot be removed."
            )
            return
        row = self.anomaly_list.row(item)
        self.anomaly_list.takeItem(row)
        for label in list(self.anomaly_labels):
            if label.strip().lower() == name:
                self.anomaly_labels.remove(label)
                break
        self._ensure_anomaly_rules()
        if not self._any_anomaly_checked():
            self._set_selected_anomaly_label("Normal")
        if self.selected_hand_label and self.selected_hand_label in self.event_draft:
            self.event_draft[self.selected_hand_label][
                "anomaly_label"
            ] = self._selected_anomaly_label()
            self._log(
                "hoi_anomaly_select",
                hand=self.selected_hand_label,
                label=self.event_draft[self.selected_hand_label]["anomaly_label"],
            )
        self._log("hoi_anomaly_remove", name=name)

    def _import_instruments(self):
        """Import instrument list from txt."""
        fp, _ = QFileDialog.getOpenFileName(
            self, "Import Instrument List", "", "Text Files (*.txt)"
        )
        if not fp:
            return
        self.combo_instrument.clear()
        self.combo_instrument.addItem("None", None)
        loaded = self._load_entities_into_combo(fp, self.combo_instrument)
        self._log("hoi_load_instruments", path=fp, count=loaded)

    def _import_targets(self):
        """Import target list from txt."""
        fp, _ = QFileDialog.getOpenFileName(
            self, "Import Target List", "", "Text Files (*.txt)"
        )
        if not fp:
            return
        self.combo_target.clear()
        self.combo_target.addItem("None", None)
        loaded = self._load_entities_into_combo(fp, self.combo_target)
        self._log("hoi_load_targets", path=fp, count=loaded)

    def _load_entities_into_combo(self, fp, combo):
        """
        Universal entity loader: reads TXT, assigns global IDs, populates combo box.
        """
        try:
            loaded_count = 0
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) != 2:
                        continue
                    category, count = parts[0].strip(), int(parts[1].strip())

                    for i in range(1, count + 1):
                        entity_name = f"{category}_{i}"

                        if entity_name in self.global_object_map:
                            uid = self.global_object_map[entity_name]
                        else:
                            uid = self.object_id_counter
                            self.global_object_map[entity_name] = uid
                            self.object_id_counter += 1

                        self.id_to_category[entity_name] = category

                        display_text = f"[{uid}] {entity_name}"

                        if combo.findData(uid) == -1:
                            combo.addItem(display_text, uid)
                            loaded_count += 1

            QMessageBox.information(
                self,
                "Success",
                f"Loaded {loaded_count} entities to {combo.objectName() or 'list'}",
            )
            return loaded_count
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed: {e}")
            return 0

    def _on_object_list_menu(self, pos):
        """Triggered when right-clicking the Objects list."""
        item = self.list_objects.itemAt(pos)
        if not item:
            return

        box_data = item.data(Qt.UserRole)
        if not box_data:
            return

        if str(box_data.get("label")) in ("Left_hand", "Right_hand"):
            return

        menu = QMenu(self)

        action_change = menu.addAction("Change ID / Propagate...")
        action_change.triggered.connect(lambda: self._dialog_change_id(item))

        action_delete = menu.addAction("Delete Box")
        action_delete.triggered.connect(lambda: self._delete_box(item))

        menu.exec_(self.list_objects.mapToGlobal(pos))

    def _dialog_change_id(self, item):
        """[Fix] Dialog to change ID to ANY target (removes category restriction)."""
        box = item.data(Qt.UserRole)
        current_uid = box.get("id")

        curr_name = self._object_name_for_id(current_uid, fallback="Unknown")

        candidates = []
        for name, uid in self.global_object_map.items():
            candidates.append((name, uid))

        candidates.sort(key=lambda x: x[1])

        if not candidates:
            QMessageBox.warning(self, "Error", "Entity Library is empty.")
            return

        from PyQt5.QtWidgets import (
            QDialog,
            QVBoxLayout,
            QComboBox,
            QCheckBox,
            QDialogButtonBox,
        )

        d = QDialog(self)
        d.setWindowTitle(f"Change ID for {curr_name}")
        d.resize(400, 150)
        layout = QVBoxLayout(d)

        layout.addWidget(QLabel(f"Current: [{current_uid}] {curr_name}"))
        layout.addWidget(QLabel("Change to:"))

        combo = QComboBox()
        current_idx = 0
        for i, (name, uid) in enumerate(candidates):
            combo.addItem(f"[{uid}] {name}", uid)
            if uid == current_uid:
                current_idx = i
        combo.setCurrentIndex(current_idx)
        layout.addWidget(combo)

        chk_propagate = QCheckBox("Apply to subsequent frames (Auto-Track)")
        chk_propagate.setChecked(True)
        chk_propagate.setToolTip(
            "Use IoU overlap to apply changes to subsequent frames."
        )
        layout.addWidget(chk_propagate)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(d.accept)
        btns.rejected.connect(d.reject)
        layout.addWidget(btns)

        if d.exec_() == QDialog.Accepted:
            new_uid = combo.currentData()
            if new_uid == current_uid:
                return

            new_label = next(
                (n for n, u in self.global_object_map.items() if u == new_uid), ""
            )

            self._push_undo()

            self._update_raw_box_id(
                self.player.current_frame, box, current_uid, new_uid, new_label
            )

            box["id"] = new_uid
            box["label"] = new_label

            count = 1
            if chk_propagate.isChecked():
                count += self._propagate_id(box, new_uid, new_label)

            self._rebuild_bboxes_from_raw()
            self._refresh_boxes_for_frame(self.player.current_frame)

            self._log(
                "hoi_box_change_id",
                old_id=current_uid,
                new_id=new_uid,
                new_label=new_label,
                propagated=bool(chk_propagate.isChecked()),
                count=count,
            )
            QMessageBox.information(
                self,
                "Updated",
                f"Updated ID to [{new_uid}] {new_label}\nApplied to {count} frames.",
            )

    def _propagate_id(self, source_box, new_uid, new_label=None):
        """
        [Fix] Propagate ID change based on IoU only (ignoring category mismatch).
        Allows correcting misclassified objects across frames.
        """
        start_frame = source_box["frame"]

        curr_frame = start_frame
        curr_box_geo = [
            source_box["x1"],
            source_box["y1"],
            source_box["x2"],
            source_box["y2"],
        ]
        modified_count = 0

        while True:
            curr_frame += 1
            if curr_frame >= self.player.frame_count:
                break

            next_boxes = self.bboxes.get(curr_frame, [])
            if not next_boxes:
                break

            best_match = None
            max_iou = 0.0

            for nb in next_boxes:
                if str(nb.get("label")) in ("Left_hand", "Right_hand"):
                    continue

                def get_iou(boxA, boxB):
                    xA = max(boxA[0], boxB["x1"])
                    yA = max(boxA[1], boxB["y1"])
                    xB = min(boxA[2], boxB["x2"])
                    yB = min(boxA[3], boxB["y2"])
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                    boxBArea = (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])
                    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

                iou = get_iou(curr_box_geo, nb)

                if iou > 0.3:
                    if iou > max_iou:
                        max_iou = iou
                        best_match = nb

            if best_match:
                old_id = best_match.get("id")

                best_match["id"] = new_uid
                if new_label:
                    best_match["label"] = new_label

                self._update_raw_box_id(
                    curr_frame, best_match, old_id, new_uid, new_label
                )

                curr_box_geo = [
                    best_match["x1"],
                    best_match["y1"],
                    best_match["x2"],
                    best_match["y2"],
                ]
                modified_count += 1
            else:
                break

        return modified_count

    @staticmethod
    def _coords_close(a: dict, b: dict, tol: float = 1.0) -> bool:
        for key in ("x1", "y1", "x2", "y2"):
            if abs(float(a.get(key, 0.0)) - float(b.get(key, 0.0))) > tol:
                return False
        return True

    def _update_raw_box_id(
        self, frame: int, box: dict, old_id: int, new_id: int, new_label: str = None
    ):
        """[Fix] Update raw box ID and LABEL/CLASS to keep consistency."""
        target_frame = int(frame) - self.start_offset
        for rb in self.raw_boxes:
            if rb.get("orig_frame") != target_frame:
                continue
            if old_id is not None and rb.get("id") != old_id:
                continue
            if not self._coords_close(rb, box):
                continue

            rb["id"] = new_id
            if new_label:
                rb["label"] = new_label
                cid = self._class_id_for_label(new_label)
                if cid is not None:
                    rb["class_id"] = cid

    def _delete_box(self, item):
        """Delete specific box from the current frame."""
        box = item.data(Qt.UserRole)
        frame = self.player.current_frame

        if frame in self.bboxes and box in self.bboxes[frame]:
            self._push_undo()
            target_frame = int(frame) - self.start_offset
            self.raw_boxes = [
                rb
                for rb in self.raw_boxes
                if not (
                    rb.get("orig_frame") == target_frame
                    and rb.get("id") == box.get("id")
                    and rb.get("label") == box.get("label")
                    and self._coords_close(rb, box)
                )
            ]
            self._rebuild_bboxes_from_raw()
            self._refresh_boxes_for_frame(frame)
            self._log("hoi_box_delete", box_id=box.get("id"), frame=frame)

    def _save_hands_xml(self):
        """Export current hand BBoxes (including modifications) to CVAT XML format."""
        if not self.raw_boxes:
            QMessageBox.information(self, "Info", "No hand data to save.")
            return

        default_name = "hands_modified.xml"
        fp, _ = QFileDialog.getSaveFileName(
            self, "Save Hands XML", default_name, "XML Files (*.xml)"
        )
        if not fp:
            return

        try:
            import xml.etree.ElementTree as ET

            root = ET.Element("annotations")
            ET.SubElement(root, "meta")

            tracks = {"Left_hand": [], "Right_hand": []}

            for b in self.raw_boxes:
                lbl = b.get("label")
                if lbl in tracks:
                    tracks[lbl].append(b)

            for label, boxes in tracks.items():
                if not boxes:
                    continue

                boxes.sort(key=lambda x: x["orig_frame"])

                track_node = ET.SubElement(
                    root, "track", id=str(self.box_id_counter), label=label
                )
                self.box_id_counter += 1

                for b in boxes:
                    ET.SubElement(
                        track_node,
                        "box",
                        frame=str(b["orig_frame"]),
                        xtl=f"{b['x1']:.2f}",
                        ytl=f"{b['y1']:.2f}",
                        xbr=f"{b['x2']:.2f}",
                        ybr=f"{b['y2']:.2f}",
                        outside="0",
                        occluded="0",
                        keyframe="1",
                    )

            tree = ET.ElementTree(root)
            tree.write(fp, encoding="utf-8", xml_declaration=True)
            self._log("hoi_save_hands_xml", path=fp, count=len(self.raw_boxes))
            self._mark_hoi_saved()
            QMessageBox.information(self, "Saved", f"Hand XML saved to:\n{fp}")

        except Exception as ex:
            QMessageBox.critical(self, "Error", f"Failed to save XML:\n{ex}")

    def _reset_event_draft(self):
        """[Step 1] Initialize/reset event draft structure."""
        self.event_draft = {
            "Left_hand": {
                "verb": "",
                "instrument_object_id": None,
                "target_object_id": None,
                "interaction_start": None,
                "functional_contact_onset": None,
                "interaction_end": None,
                "anomaly_label": "Normal",
            },
            "Right_hand": {
                "verb": "",
                "instrument_object_id": None,
                "target_object_id": None,
                "interaction_start": None,
                "functional_contact_onset": None,
                "interaction_end": None,
                "anomaly_label": "Normal",
            },
        }
        if hasattr(self, "lbl_event_status"):
            self.lbl_event_status.setText("No event selected.")
        if hasattr(self, "anomaly_list"):
            self._set_selected_anomaly_label("Normal")

    def _save_ui_to_hand_draft(self, hand_label: str):
        """[Restore] Save UI to draft using COMBO BOX."""
        if not hand_label or hand_label not in self.event_draft:
            return

        verb = self.combo_verb.currentText()

        inst_id = self.combo_instrument.currentData()
        target_id = self.combo_target.currentData()
        anomaly = self._selected_anomaly_label()

        hand_data = self.event_draft[hand_label]
        hand_data["verb"] = verb
        hand_data["instrument_object_id"] = inst_id
        hand_data["target_object_id"] = target_id
        hand_data["anomaly_label"] = anomaly

    def _swap_frame_hands(self):
        """Swap Left/Right hand boxes on the current frame and persist the labels."""
        if not self.player.cap:
            QMessageBox.information(
                self, "Info", "Load a video before swapping frame hands."
            )
            return
        frame_idx = int(self.player.current_frame)
        target_orig = frame_idx - int(self.start_offset)
        hand_boxes = [
            rb
            for rb in self.raw_boxes
            if rb.get("orig_frame") == target_orig
            and rb.get("label") in ("Left_hand", "Right_hand")
        ]
        if not hand_boxes:
            QMessageBox.information(
                self, "Info", "No hand boxes found on the current frame."
            )
            return

        self._push_undo()
        for rb in self.raw_boxes:
            if rb.get("orig_frame") != target_orig:
                continue
            lbl = rb.get("label")
            if lbl == "Left_hand":
                rb["label"] = "Right_hand"
            elif lbl == "Right_hand":
                rb["label"] = "Left_hand"

        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(frame_idx)
        self._log("hoi_frame_swap_hands", frame=frame_idx)
        QMessageBox.information(
            self, "Swapped", "Swapped Left/Right hand boxes on this frame."
        )

    def _load_hand_draft_to_ui(self, hand_label: str):
        """[Restore] Load draft to UI using COMBO BOX."""
        if not hand_label or hand_label not in self.event_draft:
            return

        hand_data = self.event_draft[hand_label]

        self.combo_verb.blockSignals(True)
        self.combo_instrument.blockSignals(True)
        self.combo_target.blockSignals(True)

        try:
            verb = hand_data.get("verb", "")

            idx = self.combo_verb.findText(verb)
            if idx >= 0:
                self.combo_verb.setCurrentIndex(idx)
            else:
                self.combo_verb.setCurrentText(verb)

            inst_id = hand_data.get("instrument_object_id")
            idx_inst = self.combo_instrument.findData(inst_id)
            if idx_inst >= 0:
                self.combo_instrument.setCurrentIndex(idx_inst)
            else:
                self.combo_instrument.setCurrentIndex(0)

            target_id = hand_data.get("target_object_id")
            idx_tar = self.combo_target.findData(target_id)
            if idx_tar >= 0:
                self.combo_target.setCurrentIndex(idx_tar)
            else:
                self.combo_target.setCurrentIndex(0)

        finally:
            self.combo_verb.blockSignals(False)
            self.combo_instrument.blockSignals(False)
            self.combo_target.blockSignals(False)

        anomaly = hand_data.get("anomaly_label", "correct")
        self._set_selected_anomaly_label(anomaly)

        self._update_status_label()

    def _update_status_label(self):
        """Refresh status bar showing the selected HandOI segment and hand details."""
        if not hasattr(self, "lbl_event_status"):
            return
        if self.selected_event_id is None:
            self.lbl_event_status.setText("No HandOI segment selected.")
            return

        hand = self.selected_hand_label
        if hand:
            h_data = self.event_draft.get(hand, {})
            s = h_data.get("interaction_start")
            o = h_data.get("functional_contact_onset")
            e = h_data.get("interaction_end")
            s_txt = str(s) if s is not None else "_"
            o_txt = str(o) if o is not None else "_"
            e_txt = str(e) if e is not None else "_"
            anom = h_data.get("anomaly_label", "Normal")
            verb = h_data.get("verb") or "_"
            inst_id = h_data.get("instrument_object_id")
            target_id = h_data.get("target_object_id")
            inst_name = "_"
            target_name = "_"
            for name, id_val in self.global_object_map.items():
                if inst_id is not None and id_val == inst_id:
                    inst_name = name
                if target_id is not None and id_val == target_id:
                    target_name = name
            hand_short = "L" if hand == "Left_hand" else "R"
            hand_info = f"{hand_short}: [{s_txt}-{o_txt}-{e_txt}] | Anom: {anom} | Verb: {verb} | Tool: {inst_name} | Target: {target_name}"
        else:
            hand_info = "Select L/R to edit this HandOI segment"

        self.lbl_event_status.setText(hand_info)
