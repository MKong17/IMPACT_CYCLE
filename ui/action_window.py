from typing import List, Dict, Tuple, Optional, Any, Set, Callable, Iterable
from collections import Counter
from bisect import bisect_left, bisect_right
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QSpinBox,
    QSlider,
    QFileDialog,
    QMessageBox,
    QInputDialog,
    QShortcut,
    QSplitter,
    QWidget,
    QVBoxLayout,
    QToolButton,
    QStyle,
    QProgressDialog,
    QApplication,
    QFrame,
    QScrollArea,
    QListWidget,
    QListWidgetItem,
    QToolTip,
    QSizePolicy,
    QDialog,
    QDialogButtonBox,
    QCheckBox,
    QFormLayout,
    QDoubleSpinBox,
    QGroupBox,
    QButtonGroup,
    QAbstractItemView,
    QListView,
    QMenu,
    QTabWidget,
    QKeySequenceEdit,
    QPlainTextEdit,
    QLineEdit,
)
from PyQt5.QtCore import (
    Qt,
    QTimer,
    QSize,
    QThread,
    QEvent,
)
from PyQt5.QtGui import QKeySequence, QCursor, QColor
from ui.video_player import VideoPlayer
from ui.label_panel import LabelPanel
from ui.timeline import TimelineArea
from ui.entities_panel import EntitiesPanel
from ui.action_workers import (
    ASOTInferWorker,
    ASOTRemapBuildWorker,
    EASTInferWorker,
    EASTMaskedRefreshWorker,
    EASTOnlineAdapterTrainWorker,
    EASTOnlineUpdateWorker,
    EASTBatchWorker,
    FactBatchWorker,
    FeatureExtractWorker,
    load_feature_extractor_module,
)
from ui.action_view_widgets import (
    _NoWheelComboBox,
    _ViewDropPanel,
    _ViewReorderHandle,
)
from ui.mixins import FrameControlMixin
from core.models import LabelDef, AnnotationStore, EntityDef
from utils.constants import (
    PRESET_COLORS,
    color_from_key,
    EXTRA_LABEL_NAME,
    EXTRA_ALIASES,
    is_extra_label,
    SNAP_RADIUS_FRAMES,
    EDGE_SNAP_FRAMES,
    CURRENT_FRAME_SNAP_RADIUS_FRAMES,
)
from utils.config import ALGO_CONFIG
from utils.psr_models import (
    load_psr_model_registry,
    enabled_psr_models,
    default_psr_model_id,
    normalize_psr_model_type,
    psr_model_display_name,
)
from utils.optional_deps import (
    MissingOptionalDependency,
    format_missing_dependency_message,
    ensure_optional_modules,
)
from utils.feature_env import load_feature_env_defaults
from utils.adapters import ADAPTERS
from utils.shortcut_settings import (
    shortcut_definitions_by_section,
    default_shortcut_bindings,
    load_shortcut_bindings,
    load_logging_policy,
    save_shortcut_bindings,
    save_logging_policy,
    conflict_messages,
    detect_scope_conflicts,
    shortcut_value,
    set_shortcut_key,
)
from utils.east_ui_state import load_east_ui_state, save_east_ui_state
from utils.default_action_label_templates import get_default_action_label_template
from tools.label_utils import (
    DEFAULT_VERB_PREFIXES,
    infer_verb_noun as infer_label_verb_noun,
    load_label_names,
    resolve_label_source,
)
from core.psr_state import (
    load_components as psr_load_components,
    load_rules as psr_load_rules,
    derive_events as psr_derive_events,
    build_state_sequence as psr_build_state_sequence,
    build_state_runs as psr_build_state_runs,
)
from core.action_corrections import CorrectionBuffer
from core.asr_worker import (
    probe_audio_stream,
    extract_wav_16k_mono_verbose,
    ensure_cached_wav_16k_mono_verbose,
)
from core.east_shared_assets import (
    consolidate_east_shared_adapter,
    export_east_runtime_report,
    export_runtime_as_shared_adapter,
    inspect_east_runtime_assets,
    inspect_east_shared_adapter_bundle,
    load_east_runtime_confusion_map,
)
from ui.timeline import TranscriptSegment, TranscriptTrack
from ui.psr_rules_dialog import PSRRulesDialog
from ui.widgets import ToggleSwitch, ClickToggleList
import copy
import bisect
from datetime import datetime
import glob
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import tempfile

try:
    import sip  # type: ignore
except Exception:
    sip = None
import numpy as np
from utils.op_logger import OperationLogger

# ----- Fine mode extras (phase/anomaly/verb-noun vocab) -----
PHASE_LABEL_DEFS = [
    ("normal", "Green", 0),
    ("anomaly", "Red", 1),
    ("recovery", "Blue", 2),
]

DEFAULT_ANOMALY_TYPES = [
    "error_temporal",
    "error_spatial",
    "error_handling",
    "error_wrong_part",
    "error_wrong_tool",
    "error_procedural",
]

# Fine mode: tweak combined row heights for clearer grouping
FINE_ACTION_ROW_HEIGHT = 44
FINE_PHASE_ROW_HEIGHT = 30

# Verb prefix helpers for label -> (verb, noun) inference
KNOWN_VERB_PREFIXES = list(DEFAULT_VERB_PREFIXES)

FEATURE_VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".m4v")

# Fixed HAS component catalog used by PSR/ASR/ASD state inference.
HAS_COMPONENT_CATALOG = [
    (0, "anti_vibration_handle"),
    (1, "gearbox_housing"),
    (2, "drive_shaft"),
    (3, "bevel_gear"),
    (4, "adapter_plate"),
    (5, "bearing_plate"),
    (6, "screw_lever"),
    (7, "screw_adaptor_topleft"),
    (8, "screw_adaptor_lowright"),
    (9, "bearing_screw_topleft"),
    (10, "bearing_screw_lowright"),
    (11, "M4_nut_plate_topleft"),
    (12, "M4_nut_plate_lowright"),
    (13, "spring"),
    (14, "lever"),
    (15, "washer"),
    (16, "M6_nut"),
]

# ---- helper: read 3-line txt segments ----
def load_segments_txt(txt_path):
    segs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f]
    buf = []
    for ln in lines:
        if ln == "":
            if len(buf) >= 3:
                s, e, name = int(buf[0]), int(buf[1]), buf[2]
                segs.append({"start": s, "end": e, "label": name})
            buf = []
        else:
            buf.append(ln)
    if len(buf) >= 3:
        s, e, name = int(buf[0]), int(buf[1]), buf[2]
        segs.append({"start": s, "end": e, "label": name})
    return segs

class ActionWindow(FrameControlMixin, QWidget):
    def __init__(
        self,
        logger: OperationLogger = None,
        on_switch_task=None,
        tasks: Optional[List[str]] = None,
        on_shortcuts_updated=None,
        on_logging_policy_updated=None,
        on_oplog_updated=None,
    ):
        super().__init__()
        self._app_title = "CV:HCI Video Annotation Tools"
        self._on_switch_task = on_switch_task
        self._on_shortcuts_updated = on_shortcuts_updated
        self._on_logging_policy_updated = on_logging_policy_updated
        self._on_oplog_updated = on_oplog_updated
        self._task_items = list(tasks or [])
        # Ensure the main window stays resizable across platforms.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(800, 600)
        self._root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self._disable_qt_audio = os.environ.get("DISABLE_QT_AUDIO", "1") == "1"
        self._shortcut_bindings = load_shortcut_bindings()
        self._shortcut_defaults = default_shortcut_bindings()

        # data
        self.labels: List[LabelDef] = []
        self.store = AnnotationStore()
        self.prelabel_store = AnnotationStore()
        self._prelabel_source: str = ""
        self.extra_store = (
            AnnotationStore()
        )  # dedicated store for interaction/extra to avoid collisions
        self.hoi_window = None
        self.validation_enabled = False
        self.validator_name: str = ""
        self.validation_modifications = []  # list of dict entries
        self.validation_errors = []  # list of dict entries
        self._validation_forced_layout = False
        self._validation_prev_combined_text = True
        self._validation_prev_center_single = False
        self._validation_overlay_mode = "both"
        self.review_items = []  # imported modifications for review
        self.review_idx: int = -1
        self.views = (
            []
        )  # list of dict: {"name": str, "player": VideoPlayer, "start": int, "end": int, "path": str, "store": AnnotationStore, "prelabel_store": AnnotationStore, "entity_stores": Dict[str, AnnotationStore], "phase_stores": Dict[str, AnnotationStore], "anomaly_type_stores": Dict[str, Dict[str, AnnotationStore]], "dirty": bool, "widget": QWidget, "name_combo": QComboBox}
        self.active_view_idx = 0
        self._sync_edit_view_indices: Set[int] = {0}
        self._timeline_auto_follow = (
            True  # keep timeline centered on current frame unless user pans
        )
        self.asr_segments = []  # List[TranscriptSegment]
        self.asr_lang = "auto"
        self._transcript_current_index = -1
        self._transcript_result_cache: Dict[str, List[TranscriptSegment]] = {}
        self.currentEastFeatureDir: Optional[str] = None
        self._feature_followup_task: str = ""
        self._feature_followup_backbone: str = ""
        self._east_ckpt: str = ""
        self._east_cfg: str = ""
        self._east_text_bank_version: str = str(
            os.environ.get("EAST_TEXT_BANK_BACKEND")
            or os.environ.get("EAST_TEXT_BANK_VERSION", "auto")
            or "auto"
        )
        self._east_category_adapter_path: str = ""
        self._east_masked_refresh_context_override: int = 0
        self._action_label_bank_source: str = ""
        self._correction_buffer = CorrectionBuffer()
        self._confirmed_correction_records: List[Dict[str, Any]] = []
        self._pending_finalized_records: List[Dict[str, Any]] = []
        self._east_online_update_thread = None
        self._east_online_update_worker = None
        self._east_online_update_running = False
        self._east_online_update_pending = False
        self._east_online_update_pending_dir = ""
        self._east_online_update_timer = QTimer(self)
        self._east_online_update_timer.setSingleShot(True)
        self._east_online_update_timer.setInterval(600)
        self._east_online_update_timer.timeout.connect(
            self._flush_east_online_update
        )
        self._east_online_adapter_thread = None
        self._east_online_adapter_worker = None
        self._east_online_adapter_running = False
        self._east_online_adapter_pending = False
        self._east_online_adapter_pending_dir = ""
        self._east_online_adapter_timer = QTimer(self)
        self._east_online_adapter_timer.setSingleShot(True)
        self._east_online_adapter_timer.setInterval(900)
        self._east_online_adapter_timer.timeout.connect(
            self._flush_east_online_adapter_update
        )
        self._east_masked_refresh_thread = None
        self._east_masked_refresh_worker = None
        self._east_masked_refresh_running = False
        self._east_masked_refresh_pending = False
        self._east_masked_refresh_pending_dir = ""
        self._last_feature_error_message = ""
        self._east_masked_refresh_timer = QTimer(self)
        self._east_masked_refresh_timer.setSingleShot(True)
        self._east_masked_refresh_timer.setInterval(1200)
        self._east_masked_refresh_timer.timeout.connect(
            self._flush_east_masked_refresh
        )
        self._east_skip_next_masked_refresh = False
        self._east_batch_thread = None
        self._east_batch_worker = None
        self._east_batch_progress = None
        self._east_batch_done = 0
        self._east_batch_current = 0
        self._east_batch_total = None
        self._asot_remap_thread = None
        self._asot_remap_worker = None
        self._asot_remap_progress = None
        self.op_logger = logger or OperationLogger(False)
        logging_policy = load_logging_policy(
            default_ops_enabled=bool(getattr(self.op_logger, "enabled", False)),
            default_validation_summary_enabled=True,
        )
        self._validation_summary_enabled = bool(
            logging_policy.get("validation_summary_enabled", True)
        )
        self._validation_comment_prompt_enabled = bool(
            logging_policy.get("validation_comment_prompt_enabled", True)
        )
        self.current_annotation_path: Optional[str] = None
        self.currentFeatureDir: Optional[str] = None
        self.extra_mode = False
        self.extra_label: Optional[LabelDef] = None
        self.extra_start_frame = 0
        self.extra_last_frame: Optional[int] = None
        self.extra_cuts: List[int] = []  # segment boundary start frames (absolute)
        self._extra_txn_stores: List[AnnotationStore] = []
        self._extra_force_follow = (
            False  # keep view centered during fast auto-fill to reduce jitter
        )
        self._freeze = False
        self._extra_overlay_timer = None
        self._hover_preview_pending_frame: Optional[int] = None
        self._hover_preview_last_frame: Optional[int] = None
        self._hover_preview_last_targets: Dict[int, int] = {}
        self._hover_preview_timer = QTimer(self)
        self._hover_preview_timer.setSingleShot(True)
        # Slightly throttle hover preview seeks to reduce decode churn on low-end devices.
        self._hover_preview_timer.setInterval(40)
        self._hover_preview_timer.timeout.connect(self._flush_timeline_hover_preview)
        # interaction (manual + assisted)
        self.interaction_mode: Optional[str] = None  # "manual" | "assisted"
        self._interaction_cfg = {
            "boundary": {
                "window_size": 10,
                "frame_step": 1,
                "uncertainty_range": (0.4, 0.55),
            },
            "label": {"top_k": 5, "min_confidence": 0.6, "diff_eps": 0.05},
        }
        self._algo_cfg = (
            copy.deepcopy(ALGO_CONFIG)
            if isinstance(ALGO_CONFIG, dict)
            else {
                "timeline_snap": {
                    "playhead_radius": CURRENT_FRAME_SNAP_RADIUS_FRAMES,
                    "empty_space_radius": SNAP_RADIUS_FRAMES,
                    "edge_search_radius": EDGE_SNAP_FRAMES,
                    "segment_soft_radius": SNAP_RADIUS_FRAMES,
                    "phase_soft_radius": 8,
                    "hover_preview_multi": True,
                    "hover_preview_align": "absolute",
                },
                "boundary_snap": {"enabled": True, "window_size": 15},
                "segment_embedding": {"trim_ratio": 0.1},
                "topk": {"enabled": True, "k": 5, "uncertainty_margin": 0.25},
                "assisted": {"boundary_min_gap": 15},
            }
        )
        self._psr_model_specs = load_psr_model_registry()
        self._ensure_algo_cfg_defaults()
        self._boundary_snap_cache: Dict[str, Any] = {}
        self._fine_prev_timeline_tooltip = None
        self._auto_boundary_candidates: List[int] = []
        self._auto_boundary_source: str = ""
        self._segment_embedding_cache: Dict[Tuple[int, int, str], np.ndarray] = {}
        self._label_prototypes: Dict[str, np.ndarray] = {}
        self._label_proto_counts: Dict[str, int] = {}
        self._knn_memory: List[Tuple[np.ndarray, str]] = []
        self._forced_segment: Optional[Dict[str, Any]] = None
        self.assisted_points: List[dict] = []  # list of interaction points with status
        self.assisted_active_idx: int = -1
        self._assisted_review_done = False
        self._assisted_loop_timer: Optional[QTimer] = None
        self._assisted_loop_range: Optional[Tuple[int, int]] = None
        self._assisted_candidates: Dict[Any, Any] = {}  # per-segment label candidates
        self._assisted_source_segments: List[dict] = []
        self._has_auto_segments: bool = (
            False  # set after running ASOT/FACT or loading their results
        )
        self.current_video_id = ""
        self.current_video_name = ""
        self._psr_asr_asd_invisible_label = "Unobservable"
        self._psr_boundary_label = "__boundary__"
        self.psr_components: List[Dict[str, Any]] = [
            {"id": int(cid), "name": name} for cid, name in HAS_COMPONENT_CATALOG
        ]
        self.psr_component_source = "fixed_catalog"
        self.psr_rules: Dict[str, Dict[str, Any]] = {}
        self.psr_rules_path = ""
        self._psr_cache_dirty = True
        self._psr_events_cache: List[Dict[str, Any]] = []
        self._psr_state_seq: List[Dict[str, Any]] = []
        self._psr_state_frames: List[int] = []
        self._psr_diag = {"events": 0, "unmapped": 0, "rule_mismatch": 0}
        self._psr_manual_events: List[Dict[str, Any]] = []
        self._psr_state_dirty = True
        self._psr_single_timeline = False
        self._psr_prev_timeline_labels = None
        self._psr_prev_timeline_layout = None
        self._psr_prev_combined_editable = None
        self._psr_prev_timeline_tooltip = None
        self._psr_state_label_map = {1: "Installed", 0: "Not installed", -1: "Error"}
        self._psr_state_label_defs = [
            LabelDef(name="Installed", color_name="Green", id=1),
            LabelDef(name="Not installed", color_name="Gray", id=0),
            LabelDef(name="Error", color_name="Red", id=-1),
        ]
        self._psr_detected_flow: str = "assemble"
        self._psr_detected_initial_state: int = 0
        self._psr_model_type: str = self._psr_normalize_model_type(
            self._psr_cfg().get("model_type", self._psr_default_model_type())
        )
        self._psr_state_store_combined: Optional[AnnotationStore] = None
        self._psr_state_stores: Dict[Any, AnnotationStore] = {}
        self._psr_combined_label_states: Dict[str, List[int]] = {}
        self._psr_state_color_cache: Dict[Tuple[int, ...], str] = {}
        self._psr_segment_cuts: List[int] = []
        self._psr_gap_spans_combined: List[Tuple[int, int]] = []
        self._psr_gap_spans_by_comp: Dict[str, List[Tuple[int, int]]] = {}
        self._psr_action_segment_starts: List[int] = []
        self._psr_action_segment_ends: List[int] = []
        self._psr_action_segments: List[Tuple[int, int]] = []
        self._psr_snap_to_action_segments: bool = False
        self._psr_selected_segment: Optional[Dict[str, Any]] = None
        self._psr_timeline_changed_bound = False
        self._psr_timeline_change_deferred = False
        self._psr_sync_apply_in_progress = False
        self._psr_undo_stack: List[Dict[str, Any]] = []
        self._psr_redo_stack: List[Dict[str, Any]] = []
        self._psr_undo_block = False
        self._psr_undo_limit = 50
        self._active_entity_name: Optional[str] = None

        # create an initial player placeholder for timeline hooks
        self.player = VideoPlayer(status_cb=self._set_status)
        try:
            self.player.set_playback_speed(1.0)
        except Exception:
            pass
        self.player.on_frame_advanced = self._on_player_frame_advanced
        self.player.on_playback_state_changed = self._on_player_playback_state_changed
        self.player.main_window = self

        root = QVBoxLayout(self)

        # --- Controls row (scrollable for small screens) ---
        ctrl_container = QWidget(self)
        ctrl_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ctrl = QHBoxLayout(ctrl_container)
        ctrl.setContentsMargins(0, 0, 0, 0)

        ctrl.addWidget(QLabel("Task:"))
        self.combo_task = QComboBox()
        items = self._task_items or [
            "Action Segmentation",
            "HandOI / HOI Detection",
            "Assembly State (PSR/ASR/ASD)",
            "Single-turn VQA",
            "Multi-turn VQA",
            "Video Captioning",
        ]
        self.combo_task.addItems(items)
        self.combo_task.currentTextChanged.connect(self._emit_task_changed)
        ctrl.addWidget(self.combo_task)
        ctrl.addSpacing(8)
        self._fit_combo_to_contents(self.combo_task, min_width=180)

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
            self._shape_btn(b)
            b.setToolButtonStyle(Qt.ToolButtonIconOnly)
            b.setIconSize(QSize(22, 22))
            ctrl.addWidget(b)

        ctrl.addSpacing(12)
        self.lbl_jump = QLabel("Jump to frame:")
        ctrl.addWidget(self.lbl_jump)
        self.spin_jump = QSpinBox()
        self.spin_jump.setMinimum(0)
        self.spin_jump.setMaximum(0)
        self.spin_jump.setKeyboardTracking(False)
        self.btn_jump = QPushButton("Go")
        ctrl.addWidget(self.spin_jump)
        ctrl.addWidget(self.btn_jump)

        ctrl.addSpacing(12)
        self.combo_actions = _NoWheelComboBox()
        self._action_groups = [
            (
                "Session",
                [
                    "Open Session...",
                    "Load Video...",
                    "Crop by frames...",
                    "Import JSON (selected views)...",
                    "Export JSON...",
                    "Export JSON (selected views to folders)...",
                    "Export to Seed Dataset...",
                ],
            ),
            (
                "Labels",
                [
                    "Import label map (TXT)...",
                    "Export label map (TXT)...",
                ],
            ),
            (
                "Model",
                [
                    "ASOT: Build Label Remap...",
                    "Batch Pre-label...",
                    "EAST Setup...",
                    "EAST: Finalize Current Video",
                    "EAST: Inspect Runtime Assets",
                    "EAST: Export Runtime Report...",
                    "EAST: Export Shared Adapter...",
                    "EAST: Select Shared Adapter...",
                    "EAST: Clear Shared Adapter",
                    "EAST: Consolidate Shared Adapter...",
                ],
            ),
            (
                "Assembly State",
                [
                    "Assembly State: Load Components...",
                    "Assembly State: Save Components...",
                    "Assembly State: Load Rules...",
                    "Assembly State: Load State JSON...",
                    "Assembly State: Export State JSON...",
                ],
            ),
            (
                "Review",
                [
                    "Import Review Log...",
                    "Open Review Panel",
                ],
            ),
            (
                "Transcript",
                [
                    "Transcript: Open Workspace",
                    "Transcript Audio: Attach External Audio...",
                    "Transcript Audio: Set Audio Offset (ms)...",
                    "Transcript: Quick Generate / Import...",
                ],
            ),
        ]
        self._action_items = [item for _, items in self._action_groups for item in items]
        self._action_section_headers: Set[str] = set()
        self.combo_actions.addItem("Choose action...")
        self._refresh_east_backend_defaults()
        self._restore_east_ui_state()
        self._apply_psr_action_dropdown(False)
        ctrl.addWidget(self.combo_actions)
        ctrl.addSpacing(12)
        self._fit_combo_to_contents(self.combo_actions, min_width=220)
        self.lbl_mode = QLabel("Mode:")
        ctrl.addWidget(self.lbl_mode)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Coarse", "Fine"])
        ctrl.addWidget(self.combo_mode)
        self._fit_combo_to_contents(self.combo_mode, min_width=90)

        ctrl.addSpacing(8)
        self.lbl_speed = QLabel("Speed:")
        ctrl.addWidget(self.lbl_speed)
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.25x", "0.5x", "1x", "1.5x", "2x"])
        self.combo_speed.setCurrentText("1x")
        ctrl.addWidget(self.combo_speed)
        self._fit_combo_to_contents(self.combo_speed, min_width=90)
        ctrl.addSpacing(6)
        self.btn_settings = QToolButton()
        self._shape_btn(self.btn_settings)
        self.btn_settings.setText("⚙")
        self.btn_settings.setToolTip("Settings (Ctrl+,)")
        self.btn_settings.clicked.connect(self._open_settings_dialog)
        ctrl.addWidget(self.btn_settings)

        ctrl.addSpacing(12)
        self.btn_auto_label = QPushButton("EAST Refine")
        self.btn_auto_label.setToolTip("EAST refinement for the current video")
        self.btn_auto_label.clicked.connect(self.on_click_auto_label)
        ctrl.addWidget(self.btn_auto_label)
        self.btn_auto_label_asot = QPushButton("ASOT Pre-label")
        self.btn_auto_label_asot.setToolTip("ASOT pre-labeling for the current video")
        self.btn_auto_label_asot.clicked.connect(self.on_click_auto_label_asot)
        ctrl.addWidget(self.btn_auto_label_asot)

        # Magnifier toggle
        ctrl.addSpacing(12)
        self.btn_mag = QToolButton()
        self._shape_btn(self.btn_mag)
        self.btn_mag.setText("🔍")
        self.btn_mag.setCheckable(True)
        ctrl.addWidget(self.btn_mag)

        ctrl.addSpacing(8)
        self.lbl_validation = QLabel("Validation")
        ctrl.addWidget(self.lbl_validation, 0)
        self.btn_validation = ToggleSwitch(self)
        self.btn_validation.setToolTip("Toggle validation on/off")
        ctrl.addWidget(self.btn_validation, 0)
        ctrl.addSpacing(6)
        self.lbl_overlay = QLabel("Overlay:")
        ctrl.addWidget(self.lbl_overlay, 0)
        self.combo_overlay = QComboBox()
        self.combo_overlay.setToolTip("On-video validation overlay")
        self.combo_overlay.currentIndexChanged.connect(
            self._on_validation_overlay_changed
        )
        ctrl.addWidget(self.combo_overlay, 0)
        self._fit_combo_to_contents(self.combo_overlay, min_width=150)

        ctrl.addSpacing(8)
        self.btn_extra = QToolButton()
        self._shape_btn(self.btn_extra)
        self.btn_extra.setText("Manual Segmentation")
        self.btn_extra.setCheckable(True)
        self.btn_extra.setToolTip("Manual global segmentation mode")
        self.btn_extra.clicked.connect(self.on_extra_clicked)
        self.btn_extra.setVisible(False)
        self.btn_assisted = QToolButton()
        self._shape_btn(self.btn_assisted)
        self.btn_assisted.setText("Assisted Review")
        self.btn_assisted.setCheckable(True)
        self.btn_assisted.setToolTip("Review model predictions via interaction points")
        self.btn_assisted.clicked.connect(self.on_assisted_clicked)
        self.btn_assisted.setVisible(False)

        ctrl.addWidget(self.btn_extra)
        ctrl.addWidget(self.btn_assisted)
        ctrl.addSpacing(8)
        self.lbl_interaction = QLabel("Interaction:")
        ctrl.addWidget(self.lbl_interaction)
        self.combo_interaction = QComboBox()
        self.combo_interaction.addItems(
            [
                "Interaction...",
                "Manual Segmentation (toggle)",
                "Assisted Review (toggle)",
                "Exit Interaction",
            ]
        )
        self.combo_interaction.setToolTip("Switch interaction modes")
        self.combo_interaction.activated[int].connect(self._on_interaction_selected)
        ctrl.addWidget(self.combo_interaction)
        self._fit_combo_to_contents(self.combo_interaction, min_width=180)
        ctrl.addSpacing(6)
        self.lbl_interaction_status = QLabel("Interaction: idle")
        self.lbl_interaction_status.setStyleSheet("color: #667085;")
        ctrl.addWidget(self.lbl_interaction_status)

        ctrl.addStretch(1)

        # playback slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider_changed)

        # --- TOP PANE: video + (right) review info, controls, playback slider ---
        topPane = QWidget(self)
        topLayout = QVBoxLayout(topPane)
        topLayout.setContentsMargins(0, 0, 0, 0)
        topLayout.setSpacing(6)

        self.ctrl_scroll = QScrollArea(self)
        self.ctrl_scroll.setWidgetResizable(True)
        self.ctrl_scroll.setFrameShape(QFrame.NoFrame)
        self.ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.ctrl_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ctrl_scroll.setWidget(ctrl_container)
        self.ctrl_scroll.setContentsMargins(0, 0, 0, 0)
        self.ctrl_scroll.setStyleSheet(
            """
            QScrollArea { background: transparent; }
            QScrollBar:horizontal { height: 10px; margin: 0px; }
            QScrollBar::handle:horizontal { background: #cbd5e1; border-radius: 5px; min-width: 24px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
        """
        )

        # Top row: video + review + ASR inside a splitter for resizable/hideable ASR
        video_split = QSplitter(Qt.Horizontal, self)
        video_split.setChildrenCollapsible(False)
        try:
            video_split.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass

        # left: video views (scrollable)
        self.video_grid_container = QWidget(self)
        self.video_grid_inner = QWidget(self.video_grid_container)
        self.video_stack_layout = QVBoxLayout(self.video_grid_inner)
        self.video_stack_layout.setContentsMargins(0, 0, 0, 0)
        self.video_stack_layout.setSpacing(0)

        self.video_scroll = QScrollArea(self)
        self.video_scroll.setWidgetResizable(True)
        self.video_scroll.setWidget(self.video_grid_inner)
        self.video_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.video_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # view toolbar (always visible)
        view_toolbar = QHBoxLayout()
        view_toolbar.setContentsMargins(0, 0, 0, 0)
        view_toolbar.setSpacing(6)
        view_toolbar.addWidget(QLabel("Views"))
        self.btn_add_view = QPushButton("+")
        self.btn_add_view.setFixedWidth(28)
        self.btn_add_view.setToolTip("Add view")
        self.btn_add_view.clicked.connect(self._on_add_view)
        view_toolbar.addStretch(1)
        view_toolbar.addWidget(self.btn_add_view)

        left_video_panel = QWidget(self)
        left_v = QVBoxLayout(left_video_panel)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(6)
        left_v.addLayout(view_toolbar)
        left_v.addWidget(self.video_scroll, 1)
        video_split.addWidget(left_video_panel)

        # middle: review queue
        self.review_panel = QWidget(self)
        self.review_panel.setMinimumWidth(220)
        rv = QVBoxLayout(self.review_panel)
        rv.setContentsMargins(8, 8, 8, 8)
        rv.setSpacing(6)
        review_head = QHBoxLayout()
        review_head.setContentsMargins(0, 0, 0, 0)
        review_head.setSpacing(6)
        self.lbl_review_title = QLabel("Review Queue")
        self.lbl_review_title.setStyleSheet("font-weight: 600;")
        review_head.addWidget(self.lbl_review_title)
        review_head.addStretch(1)
        self.lbl_review_progress = QLabel("0 / 0")
        self.lbl_review_progress.setStyleSheet("color: #475467; font-weight: 600;")
        review_head.addWidget(self.lbl_review_progress)
        self.btn_hide_review = QToolButton(self.review_panel)
        self.btn_hide_review.setText("Hide")
        self.btn_hide_review.clicked.connect(lambda: self._set_review_panel_visible(False))
        review_head.addWidget(self.btn_hide_review)
        self.lbl_review_info = QLabel("")
        self.lbl_review_info.setWordWrap(True)
        btn_row = QHBoxLayout()
        self.btn_accept = QPushButton("Accept Change")
        self.btn_reject = QPushButton("Reject Change")
        self.btn_accept.clicked.connect(lambda: self._decide_current_change(True))
        self.btn_reject.clicked.connect(lambda: self._decide_current_change(False))
        btn_row.addWidget(self.btn_accept)
        btn_row.addWidget(self.btn_reject)
        rv.addLayout(review_head)
        rv.addWidget(self.lbl_review_info, 1)
        rv.addLayout(btn_row)
        self.review_panel.hide()
        video_split.addWidget(self.review_panel)

        # right: transcript workspace side panel (resizable/hideable)
        self.asr_panel = QWidget(self)
        self.asr_panel.setMinimumWidth(0)  # allow full collapse when hidden
        asr_layout = QVBoxLayout(self.asr_panel)
        asr_layout.setContentsMargins(8, 8, 8, 8)
        asr_layout.setSpacing(6)
        header_asr = QHBoxLayout()
        self.lbl_asr_title = QLabel("Transcript Workspace")
        header_asr.addWidget(self.lbl_asr_title)
        header_asr.addStretch(1)
        self.btn_toggle_asr_panel = QToolButton(self)
        self.btn_toggle_asr_panel.setText("Hide")
        self.btn_toggle_asr_panel.setCheckable(True)
        self.btn_toggle_asr_panel.setChecked(True)
        self.btn_toggle_asr_panel.toggled.connect(self._toggle_asr_panel)
        header_asr.addWidget(self.btn_toggle_asr_panel)
        asr_layout.addLayout(header_asr)
        self.lbl_asr_hint = QLabel("Transcript cues for timing and segmentation.")
        self.lbl_asr_hint.setWordWrap(True)
        self.lbl_asr_hint.setStyleSheet("color: #667085;")
        asr_layout.addWidget(self.lbl_asr_hint)
        self.lbl_asr_summary = QLabel("No transcript loaded.")
        self.lbl_asr_summary.setWordWrap(True)
        self.lbl_asr_summary.setStyleSheet("color: #475467;")
        asr_layout.addWidget(self.lbl_asr_summary)
        self.transcript_focus_card = QFrame(self.asr_panel)
        self.transcript_focus_card.setObjectName("transcriptFocusCard")
        self.transcript_focus_card.setStyleSheet(
            """
            QFrame#transcriptFocusCard {
                background: #f8fafc;
                border: 1px solid #d0d5dd;
                border-radius: 10px;
            }
            QLabel#transcriptFocusTitle {
                color: #344054;
                font-weight: 600;
            }
            QLabel#transcriptFocusTime {
                color: #667085;
            }
            QLabel#transcriptFocusText {
                color: #101828;
                font-size: 15px;
                font-weight: 600;
            }
            QLabel#transcriptFocusMeta {
                color: #475467;
            }
            """
        )
        focus_layout = QVBoxLayout(self.transcript_focus_card)
        focus_layout.setContentsMargins(10, 10, 10, 10)
        focus_layout.setSpacing(4)
        focus_head = QHBoxLayout()
        focus_head.setContentsMargins(0, 0, 0, 0)
        focus_head.setSpacing(6)
        self.lbl_transcript_focus_title = QLabel("Current Cue")
        self.lbl_transcript_focus_title.setObjectName("transcriptFocusTitle")
        focus_head.addWidget(self.lbl_transcript_focus_title)
        focus_head.addStretch(1)
        self.lbl_transcript_focus_time = QLabel("--")
        self.lbl_transcript_focus_time.setObjectName("transcriptFocusTime")
        focus_head.addWidget(self.lbl_transcript_focus_time)
        focus_layout.addLayout(focus_head)
        self.lbl_transcript_focus_text = QLabel("Generate or import a transcript.")
        self.lbl_transcript_focus_text.setObjectName("transcriptFocusText")
        self.lbl_transcript_focus_text.setWordWrap(True)
        focus_layout.addWidget(self.lbl_transcript_focus_text)
        self.lbl_transcript_focus_meta = QLabel("Follows the playhead.")
        self.lbl_transcript_focus_meta.setObjectName("transcriptFocusMeta")
        self.lbl_transcript_focus_meta.setWordWrap(True)
        focus_layout.addWidget(self.lbl_transcript_focus_meta)
        asr_layout.addWidget(self.transcript_focus_card)
        nav_row = QHBoxLayout()
        nav_row.setContentsMargins(0, 0, 0, 0)
        nav_row.setSpacing(6)
        self.btn_transcript_prev = QToolButton(self.asr_panel)
        self.btn_transcript_prev.setText("◀")
        self.btn_transcript_prev.setToolTip("Previous cue")
        self.btn_transcript_prev.clicked.connect(self._jump_to_prev_transcript)
        self.btn_transcript_prev.setEnabled(False)
        nav_row.addWidget(self.btn_transcript_prev)
        self.btn_transcript_focus = QToolButton(self.asr_panel)
        self.btn_transcript_focus.setText("◎")
        self.btn_transcript_focus.setToolTip("Cue at playhead")
        self.btn_transcript_focus.clicked.connect(self._jump_to_playhead_transcript)
        self.btn_transcript_focus.setEnabled(False)
        nav_row.addWidget(self.btn_transcript_focus)
        self.btn_transcript_next = QToolButton(self.asr_panel)
        self.btn_transcript_next.setText("▶")
        self.btn_transcript_next.setToolTip("Next cue")
        self.btn_transcript_next.clicked.connect(self._jump_to_next_transcript)
        self.btn_transcript_next.setEnabled(False)
        nav_row.addWidget(self.btn_transcript_next)
        asr_layout.addLayout(nav_row)
        asr_btn_row = QHBoxLayout()
        asr_btn_row.setContentsMargins(0, 0, 0, 0)
        asr_btn_row.setSpacing(6)
        self.btn_transcript_generate = QPushButton("Generate / Import")
        self.btn_transcript_generate.clicked.connect(
            self._on_transcript_generate_clicked
        )
        asr_btn_row.addWidget(self.btn_transcript_generate)
        self.btn_transcript_apply = QPushButton("Apply to Label")
        self.btn_transcript_apply.clicked.connect(self._apply_loaded_transcript_to_label)
        self.btn_transcript_apply.setEnabled(False)
        asr_btn_row.addWidget(self.btn_transcript_apply)
        self.btn_transcript_clear = QPushButton("Clear")
        self.btn_transcript_clear.clicked.connect(self._clear_transcript_workspace)
        self.btn_transcript_clear.setEnabled(False)
        asr_btn_row.addWidget(self.btn_transcript_clear)
        asr_layout.addLayout(asr_btn_row)
        self.lbl_asr_list_title = QLabel("Cues")
        self.lbl_asr_list_title.setStyleSheet("color: #344054; font-weight: 600;")
        asr_layout.addWidget(self.lbl_asr_list_title)
        self.list_asr = QListWidget(self.asr_panel)
        self.list_asr.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_asr.setAlternatingRowColors(False)
        self.list_asr.setSpacing(6)
        self.list_asr.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list_asr.setStyleSheet(
            """
            QListWidget {
                background: transparent;
                border: none;
                padding: 0px;
            }
            QListWidget::item {
                padding: 0px;
                margin: 0px;
                border: none;
            }
            QListWidget::item:selected {
                background: transparent;
                color: inherit;
            }
            """
        )
        self.list_asr.itemDoubleClicked.connect(self._on_asr_item_activated)
        asr_layout.addWidget(self.list_asr, 1)
        video_split.addWidget(self.asr_panel)

        # stub shown when transcript workspace is hidden (so user can re-open it)
        self.asr_stub = QWidget(self)
        stub_v = QVBoxLayout(self.asr_stub)
        stub_v.setContentsMargins(2, 4, 2, 4)
        stub_v.setSpacing(2)
        self.btn_show_asr = QToolButton(self.asr_stub)
        self.btn_show_asr.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
        self.btn_show_asr.setToolTip("Show Transcript Workspace")
        self.btn_show_asr.setFixedWidth(24)
        self.asr_stub.setMinimumWidth(26)
        self.asr_stub.setMaximumWidth(36)
        self.btn_show_asr.clicked.connect(lambda: self._toggle_asr_panel(True))
        stub_v.addStretch(1)
        stub_v.addWidget(self.btn_show_asr, alignment=Qt.AlignCenter)
        stub_v.addStretch(1)
        self.asr_stub.setVisible(False)
        video_split.addWidget(self.asr_stub)

        video_split.setStretchFactor(0, 3)  # video area
        video_split.setStretchFactor(1, 1)  # review area
        video_split.setStretchFactor(2, 1)  # ASR area
        video_split.setStretchFactor(3, 0)  # stub

        # keep a handle to restore transcript workspace width after hide/show
        self._asr_last_size = 320
        self.video_split = video_split
        self._psr_prev_asr_visible = None
        self._psr_prev_left_sizes = None

        topLayout.addWidget(video_split, 1)
        topLayout.addWidget(self.slider, 0)
        topLayout.addWidget(self.ctrl_scroll, 0)
        # start with ASR panel hidden
        try:
            self.btn_toggle_asr_panel.blockSignals(True)
            self.btn_toggle_asr_panel.setChecked(False)
            self.btn_toggle_asr_panel.setText("Show")
            self.btn_toggle_asr_panel.blockSignals(False)
        except Exception:
            pass
        self._set_review_panel_visible(False)
        QTimer.singleShot(0, lambda: self._toggle_asr_panel(False))

        # Fine vocab caches are needed before the label panel is created because
        # compound-verb hints are synchronized into the panel during init.
        self.fine_verbs: List[Dict[str, Any]] = []
        self.fine_nouns: List[Dict[str, Any]] = []

        # --- BOTTOM PANE: annotation area (label + timeline) ---
        self.panel = LabelPanel(
            self.labels,
            on_add=self._on_label_added,
            on_remove_idx=self._on_label_removed,
            on_rename=self._on_label_renamed,
            on_search_matches=self._on_label_search_matches,
            on_select_idx=self._on_label_selected,
        )
        self._sync_label_panel_verb_hints(refresh=False)

        # --- mode/entities state ---
        self.mode = "Coarse"  # or "Fine"
        self.label_entity_map: Dict[str, set] = {}
        self.current_label_idx: int = -1
        self.entities: List[EntityDef] = []
        self.entity_stores: Dict[str, AnnotationStore] = {}
        self.visible_entities: List[str] = []
        # ---- fine mode phase/anomaly stores ----
        self.phase_mode_enabled = False
        self.phase_labels: List[LabelDef] = [
            LabelDef(name=name, color_name=color, id=pid)
            for (name, color, pid) in PHASE_LABEL_DEFS
        ]
        self.phase_stores: Dict[str, AnnotationStore] = {}
        self._phase_active_label = (
            self.phase_labels[0].name if self.phase_labels else "normal"
        )
        self._phase_selected = None  # dict: {entity,start,end,label}
        self.anomaly_types: List[Dict[str, Any]] = []
        self.anomaly_type_stores: Dict[str, Dict[str, AnnotationStore]] = {}
        self._anomaly_block = False
        self._phase_block = False
        self.timeline = TimelineArea(
            self.labels,
            self.store,
            get_frame_count=self._get_frame_count,
            get_fps=self._get_fps,
        )
        try:
            self.timeline.set_combined_editable(True)
        except Exception:
            pass
        self.timeline.set_extra_cuts(self.extra_cuts)
        self.timeline.hoverFrame.connect(self._on_timeline_hover_frame)
        self.timeline.viewPanned.connect(
            lambda: setattr(self, "_timeline_auto_follow", False)
        )
        self.timeline.labelClicked.connect(self._on_timeline_label_clicked)
        self.timeline.segmentSelected.connect(self._on_timeline_segment_selected)
        self.timeline.gapPrevRequested.connect(lambda: self._goto_gap(-1))
        self.timeline.gapNextRequested.connect(lambda: self._goto_gap(+1))
        try:
            self.timeline.set_combined_reorder_handler(self._on_entity_row_reordered)
        except Exception:
            pass
        self._apply_timeline_snap_settings(refresh=False)

        # EntitiesPanel
        self.entities_panel = EntitiesPanel(
            self.entities,
            on_add=self._on_entity_added,
            on_remove_idx=self._on_entity_removed,
            on_rename=self._on_entity_renamed,
            on_applicability_changed=self._on_entity_applicability_changed,
            on_visibility_changed=self._on_entity_visibility_changed,
        )

        # Phase panel (Fine mode)
        self.phase_panel = QGroupBox("Phase", self)
        phase_layout = QVBoxLayout(self.phase_panel)
        phase_layout.setContentsMargins(8, 8, 8, 8)
        phase_layout.setSpacing(6)
        try:
            self.phase_panel.setLayout(phase_layout)
        except Exception:
            pass
        self.chk_phase_mode = QCheckBox("Enable phase", self.phase_panel)
        self.chk_phase_mode.setToolTip("Toggle phase axis for fine annotations")
        self.chk_phase_mode.toggled.connect(self._on_phase_mode_toggled)
        phase_layout.addWidget(self.chk_phase_mode, 0)

        btn_row = QHBoxLayout()
        self.phase_button_group = QButtonGroup(self.phase_panel)
        self.phase_button_group.setExclusive(True)
        self._phase_buttons: Dict[str, QToolButton] = {}
        for lb in self.phase_labels:
            btn = QToolButton(self.phase_panel)
            btn.setText(lb.name)
            btn.setCheckable(True)
            btn.setProperty("phase_label", lb.name)
            try:
                col = color_from_key(lb.color_name).name()
                btn.setStyleSheet(f"color: {col};")
            except Exception:
                pass
            btn_row.addWidget(btn)
            self.phase_button_group.addButton(btn)
            self._phase_buttons[lb.name] = btn
        self.phase_button_group.buttonClicked.connect(self._on_phase_label_clicked)
        phase_layout.addLayout(btn_row)
        # default selection
        if self.phase_labels and self._phase_buttons.get(self._phase_active_label):
            try:
                self._phase_buttons[self._phase_active_label].setChecked(True)
            except Exception:
                pass

        self.anomaly_group = QGroupBox("Anomaly types", self.phase_panel)
        self.anomaly_layout = QVBoxLayout(self.anomaly_group)
        self.anomaly_layout.setContentsMargins(6, 6, 6, 6)
        self.anomaly_layout.setSpacing(4)
        self.anomaly_list = ClickToggleList(self.anomaly_group)
        self.anomaly_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.anomaly_list.setFlow(QListView.LeftToRight)
        self.anomaly_list.setWrapping(True)
        self.anomaly_list.setResizeMode(QListView.Adjust)
        self.anomaly_list.setSpacing(4)
        self.anomaly_list.setStyleSheet(
            "QListWidget::item { padding: 2px 8px; }"
            "QListWidget::indicator { width: 14px; height: 14px; }"
        )
        self.anomaly_list.itemChanged.connect(self._on_anomaly_item_changed)
        self.anomaly_list.setWordWrap(True)
        self.anomaly_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.anomaly_layout.addWidget(self.anomaly_list)
        try:
            self.anomaly_group.setLayout(self.anomaly_layout)
        except Exception:
            pass
        try:
            self.anomaly_group.setMinimumHeight(90)
        except Exception:
            pass
        phase_layout.addWidget(self.anomaly_group, 1)
        self._ensure_anomaly_types()
        self._rebuild_anomaly_type_panel()
        self._set_anomaly_panel_enabled(False, clear=True)

        # Mode switch
        self.combo_mode.currentTextChanged.connect(self._on_mode_changed)
        self.btn_mag.toggled.connect(self._on_magnifier_toggled)

        splitter_ann = QSplitter(Qt.Horizontal, self)
        self.splitter_ann = splitter_ann
        splitter_ann.setChildrenCollapsible(False)

        # LEFT composite panel (entities + labels) with adjustable splitter
        self.entities_panel.setVisible(False)
        self.phase_panel.setVisible(False)
        self.panel.setMinimumWidth(140)
        left_split = QSplitter(Qt.Vertical, self)
        left_split.setChildrenCollapsible(False)
        left_split.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        left_split.addWidget(self.entities_panel)
        left_split.addWidget(self.phase_panel)
        left_split.addWidget(self.panel)
        left_split.setStretchFactor(0, 0)
        left_split.setStretchFactor(1, 0)
        left_split.setStretchFactor(2, 1)
        left_split.setSizes([90, 110, 380])
        self.left_splitter = left_split

        left_scroll = QScrollArea(self)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setWidget(left_split)
        left_scroll.setContentsMargins(0, 0, 0, 0)
        left_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        left_scroll.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Expanding)
        self.left_scroll = left_scroll

        splitter_ann.addWidget(left_scroll)
        splitter_ann.addWidget(self.timeline)

        splitter_ann.setStretchFactor(0, 0)
        splitter_ann.setStretchFactor(1, 1)
        splitter_ann.setSizes([260, 940])

        # --- MAIN VERTICAL SPLITTER: top / bottom ---
        splitter_main = QSplitter(Qt.Vertical, self)
        self.splitter_main = splitter_main
        self._splitter_init_done = False
        splitter_main.setChildrenCollapsible(False)
        splitter_main.setHandleWidth(8)
        splitter_main.setOpaqueResize(True)
        try:
            topPane.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass
        try:
            splitter_ann.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        except Exception:
            pass
        splitter_main.addWidget(topPane)
        splitter_main.addWidget(splitter_ann)
        splitter_main.setStretchFactor(0, 6)
        splitter_main.setStretchFactor(1, 4)

        root.addWidget(splitter_main, 1)
        QTimer.singleShot(0, self._init_main_splitter_sizes)

        # status
        self.lbl_status = QLabel("Ready. Choose 'Load Video...' from the dropdown.")
        root.addWidget(self.lbl_status)

        # wire controls
        try:
            self.btn_play.clicked.disconnect()
        except Exception:
            pass
        self.btn_play.clicked.connect(self._toggle_play_pause)
        try:
            self.btn_stop.clicked.disconnect()
        except Exception:
            pass
        self.btn_stop.clicked.connect(self._stop_all)
        self.btn_ff.clicked.connect(lambda: self._seek_relative(+10))
        self.btn_rew.clicked.connect(lambda: self._seek_relative(-10))
        self.btn_jump.clicked.connect(self._jump_to_spin)
        try:
            self.spin_jump.lineEdit().returnPressed.connect(self._jump_to_spin)
        except Exception:
            pass
        self.combo_actions.activated[int].connect(self._on_action_selected)
        self.combo_speed.currentTextChanged.connect(self._on_speed_changed)
        self.btn_validation.toggled.connect(self._on_validation_toggled)
        self._update_validation_overlay_controls()
        self._update_play_pause_button()
        # review navigation shortcuts
        self.sc_review_next = QShortcut(
            QKeySequence(Qt.Key_Right), self, activated=self._goto_next_review
        )
        self.sc_review_next.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_review_prev = QShortcut(
            QKeySequence(Qt.Key_Left), self, activated=self._goto_prev_review
        )
        self.sc_review_prev.setContext(Qt.WidgetWithChildrenShortcut)
        # assisted interaction shortcuts (enabled only in assisted mode)
        self.sc_assist_left = QShortcut(
            QKeySequence(Qt.Key_Left),
            self,
            activated=lambda: self._assist_nudge_boundary(-1),
        )
        self.sc_assist_left.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_assist_left.setEnabled(False)
        self.sc_assist_right = QShortcut(
            QKeySequence(Qt.Key_Right),
            self,
            activated=lambda: self._assist_nudge_boundary(+1),
        )
        self.sc_assist_right.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_assist_right.setEnabled(False)
        self.sc_assist_confirm = QShortcut(
            QKeySequence("S"), self, activated=self._confirm_active_boundary
        )
        self.sc_assist_confirm.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_assist_confirm.setEnabled(False)
        self.sc_assist_down = QShortcut(
            QKeySequence(Qt.Key_Down), self, activated=self._confirm_active_boundary
        )
        self.sc_assist_down.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_assist_down.setEnabled(False)
        self.sc_assist_next = QShortcut(
            QKeySequence("N"), self, activated=lambda: self._shift_assisted_idx(+1)
        )
        self.sc_assist_next.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_assist_next.setEnabled(False)
        self.sc_assist_prev = QShortcut(
            QKeySequence("P"), self, activated=lambda: self._shift_assisted_idx(-1)
        )
        self.sc_assist_prev.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_assist_prev.setEnabled(False)
        self.sc_assist_skip = QShortcut(
            QKeySequence("X"), self, activated=self._skip_active_assisted_point
        )
        self.sc_assist_skip.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_assist_skip.setEnabled(False)
        self.sc_assist_merge = QShortcut(
            QKeySequence(Qt.Key_Backspace), self, activated=self._merge_active_boundary
        )
        self.sc_assist_merge.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_assist_merge.setEnabled(False)
        self.sc_assist_merge_del = QShortcut(
            QKeySequence(Qt.Key_Delete), self, activated=self._merge_active_boundary
        )
        self.sc_assist_merge_del.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_assist_merge_del.setEnabled(False)

        self._init_primary_view()
        self._apply_default_label_template(self.mode, reason="init")

        self._set_controls_enabled(False)
        self._rebuild_timeline_sources()

        self._dirty = False  # save state
        self._suspend_store_changed = False
        self.timeline.changed.connect(self._on_store_changed)  # changed
        self._update_gap_indicator()

        self._undo_stack = []  # List[(store, deltas)] or batch history payloads
        self._redo_stack = []
        self._undo_limit = 200
        self._redo_limit = 200
        self._store_change_preview_limit = 32
        self._segment_embedding_cache_limit = 2048
        self._knn_memory_limit = 4096
        self._validation_log_limit = 20000

        self.video_path = None

        # Shortcuts
        self.sc_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.sc_undo.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_undo.activated.connect(self._undo)
        self.sc_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.sc_redo.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_redo.activated.connect(self._redo)
        self.sc_toggle_play = QShortcut(
            QKeySequence(Qt.Key_Space), self, activated=self._toggle_play_pause
        )
        self.sc_toggle_play.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_step_prev = QShortcut(
            QKeySequence("A"), self, activated=lambda: self._step_frames(-1)
        )
        self.sc_step_prev.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_step_next = QShortcut(
            QKeySequence("D"), self, activated=lambda: self._step_frames(+1)
        )
        self.sc_step_next.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_step_prev_fast = QShortcut(
            QKeySequence("Shift+A"), self, activated=lambda: self._step_frames(-10)
        )
        self.sc_step_prev_fast.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_step_next_fast = QShortcut(
            QKeySequence("Shift+D"), self, activated=lambda: self._step_frames(+10)
        )
        self.sc_step_next_fast.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_jump_start = QShortcut(
            QKeySequence(Qt.Key_Home),
            self,
            activated=lambda: self._jump_to_bound("start"),
        )
        self.sc_jump_start.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_jump_end = QShortcut(
            QKeySequence(Qt.Key_End), self, activated=lambda: self._jump_to_bound("end")
        )
        self.sc_jump_end.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_seek_back_1s = QShortcut(
            QKeySequence("J"), self, activated=lambda: self._seek_relative(-1)
        )
        self.sc_seek_back_1s.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_pause_key = QShortcut(
            QKeySequence("K"), self, activated=self._pause_all
        )
        self.sc_pause_key.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_seek_fwd_1s = QShortcut(
            QKeySequence("L"), self, activated=lambda: self._seek_relative(+1)
        )
        self.sc_seek_fwd_1s.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_uncertainty_margin = QShortcut(
            QKeySequence("Ctrl+Shift+U"),
            self,
            activated=self._prompt_uncertainty_margin,
        )
        self.sc_uncertainty_margin.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_settings = QShortcut(
            QKeySequence("Ctrl+,"), self, activated=self._open_settings_dialog
        )
        self.sc_settings.setContext(Qt.WidgetWithChildrenShortcut)
        self.apply_shortcut_settings(self._shortcut_bindings)
        self.installEventFilter(self)
        self.psr_embedded = None

    # ----- helpers -----
    def _shape_btn(self, b):
        b.setFixedWidth(40)
        b.setFixedHeight(36)
        b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet(
            """
                QToolButton {
                    background: #f6f7f9;
                    border: 1px solid #d0d5dd;
                    border-radius: 6px;
                    padding: 4px;
                }
                QToolButton:hover { background: #eef2f7; }
                QToolButton:pressed { background: #e3e8ef; }
                QToolButton:disabled { color: #9aa4b2; }
            """
        )

    def _fit_combo_to_contents(self, combo: QComboBox, min_width: int = 140) -> None:
        try:
            fm = combo.fontMetrics()
            max_w = 0
            for i in range(combo.count()):
                text = combo.itemText(i)
                try:
                    w = fm.horizontalAdvance(text)
                except AttributeError:
                    w = fm.width(text)
                max_w = max(max_w, w)
            width = max(min_width, max_w + 40)
            combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            combo.setMinimumWidth(width)
            view = combo.view()
            if view is not None:
                view.setMinimumWidth(max(width, max_w + 60))
        except Exception:
            pass

    def set_logging_policy(
        self, oplog_enabled: bool, validation_summary_enabled: bool
    ) -> None:
        if getattr(self, "op_logger", None) is not None:
            self.op_logger.enabled = bool(oplog_enabled)
        self._validation_summary_enabled = bool(validation_summary_enabled)

    def _set_shortcut_key(
        self, shortcut: Optional[QShortcut], sid: str, default_key: str
    ):
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

    def _apply_action_shortcuts_from_settings(self):
        # Action / review / assisted shortcuts
        self._set_shortcut_key(
            getattr(self, "sc_review_next", None),
            "action.review_next",
            "Right",
        )
        self._set_shortcut_key(
            getattr(self, "sc_review_prev", None),
            "action.review_prev",
            "Left",
        )
        self._set_shortcut_key(
            getattr(self, "sc_assist_left", None),
            "action.assist_nudge_left",
            "Left",
        )
        self._set_shortcut_key(
            getattr(self, "sc_assist_right", None),
            "action.assist_nudge_right",
            "Right",
        )
        self._set_shortcut_key(
            getattr(self, "sc_assist_confirm", None),
            "action.assist_confirm",
            "S",
        )
        self._set_shortcut_key(
            getattr(self, "sc_assist_down", None),
            "action.assist_confirm_down",
            "Down",
        )
        self._set_shortcut_key(
            getattr(self, "sc_assist_next", None),
            "action.assist_next",
            "N",
        )
        self._set_shortcut_key(
            getattr(self, "sc_assist_prev", None),
            "action.assist_prev",
            "P",
        )
        self._set_shortcut_key(
            getattr(self, "sc_assist_skip", None),
            "action.assist_skip",
            "X",
        )
        self._set_shortcut_key(
            getattr(self, "sc_assist_merge", None),
            "action.assist_merge",
            "Backspace",
        )
        self._set_shortcut_key(
            getattr(self, "sc_assist_merge_del", None),
            "action.assist_merge_delete",
            "Delete",
        )
        self._set_shortcut_key(getattr(self, "sc_undo", None), "action.undo", "Ctrl+Z")
        self._set_shortcut_key(getattr(self, "sc_redo", None), "action.redo", "Ctrl+Y")
        self._set_shortcut_key(
            getattr(self, "sc_toggle_play", None),
            "action.play_pause",
            "Space",
        )
        self._set_shortcut_key(
            getattr(self, "sc_step_prev", None),
            "action.step_prev",
            "A",
        )
        self._set_shortcut_key(
            getattr(self, "sc_step_next", None),
            "action.step_next",
            "D",
        )
        self._set_shortcut_key(
            getattr(self, "sc_step_prev_fast", None),
            "action.step_prev_fast",
            "Shift+A",
        )
        self._set_shortcut_key(
            getattr(self, "sc_step_next_fast", None),
            "action.step_next_fast",
            "Shift+D",
        )
        self._set_shortcut_key(
            getattr(self, "sc_jump_start", None),
            "action.jump_start",
            "Home",
        )
        self._set_shortcut_key(
            getattr(self, "sc_jump_end", None),
            "action.jump_end",
            "End",
        )
        self._set_shortcut_key(
            getattr(self, "sc_seek_back_1s", None),
            "action.seek_back_1s",
            "J",
        )
        self._set_shortcut_key(
            getattr(self, "sc_pause_key", None),
            "action.pause",
            "K",
        )
        self._set_shortcut_key(
            getattr(self, "sc_seek_fwd_1s", None),
            "action.seek_fwd_1s",
            "L",
        )
        self._set_shortcut_key(
            getattr(self, "sc_uncertainty_margin", None),
            "action.adjust_uncertainty",
            "Ctrl+Shift+U",
        )
        self._set_shortcut_key(
            getattr(self, "sc_settings", None),
            "action.open_settings",
            "Ctrl+,",
        )

    def apply_shortcut_settings(self, bindings: Optional[Dict[str, Any]] = None):
        self._shortcut_bindings = (
            load_shortcut_bindings() if bindings is None else dict(bindings)
        )
        self._shortcut_defaults = default_shortcut_bindings()
        self._apply_action_shortcuts_from_settings()

    def _set_primary_undo_shortcuts_enabled(self, enabled: bool) -> None:
        for sc in (getattr(self, "sc_undo", None), getattr(self, "sc_redo", None)):
            if sc is not None:
                try:
                    sc.setEnabled(bool(enabled))
                except Exception:
                    pass

    def _push_undo_entry(
        self,
        store: AnnotationStore,
        deltas: List[Tuple],
    ) -> None:
        if not deltas:
            return
        self._push_undo_item((store, deltas))

    def _push_undo_item(self, item: Any) -> None:
        if item is None:
            return
        self._undo_stack.append(item)
        limit = max(1, int(getattr(self, "_undo_limit", 200)))
        if len(self._undo_stack) > limit:
            del self._undo_stack[: len(self._undo_stack) - limit]

    def _push_undo_batch(
        self,
        batches: List[Tuple[AnnotationStore, List[Tuple]]],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        normalized = []
        for st, ds in batches or []:
            if st is None or not ds:
                continue
            normalized.append((st, list(ds)))
        if not normalized:
            return
        payload: Dict[str, Any] = {"kind": "batch", "entries": normalized}
        if isinstance(meta, dict):
            payload["meta"] = dict(meta)
        self._push_undo_item(payload)

    def _push_undo_composite(
        self,
        batches: List[Tuple[AnnotationStore, List[Tuple]]],
        trim_ops: List[Dict[str, Any]],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        normalized = []
        for st, ds in batches or []:
            if st is None or not ds:
                continue
            normalized.append((st, list(ds)))
        trim_norm = [dict(op) for op in (trim_ops or []) if isinstance(op, dict)]
        if not normalized and not trim_norm:
            return
        payload: Dict[str, Any] = {
            "kind": "composite",
            "entries": normalized,
            "trim_ops": trim_norm,
        }
        if isinstance(meta, dict):
            payload["meta"] = dict(meta)
        self._push_undo_item(payload)

    def _push_redo_item(self, item: Any) -> None:
        if item is None:
            return
        self._redo_stack.append(item)
        limit = max(1, int(getattr(self, "_redo_limit", 200)))
        if len(self._redo_stack) > limit:
            del self._redo_stack[: len(self._redo_stack) - limit]

    def _history_item_frame_count(self, item: Any) -> int:
        if isinstance(item, dict) and item.get("kind") == "trim_cuts":
            return len(item.get("ops") or [])
        if isinstance(item, dict) and item.get("kind") == "composite":
            total = len(item.get("trim_ops") or [])
            for _st, ds in item.get("entries") or []:
                total += self._delta_frame_count(ds or [])
            return total
        if isinstance(item, dict) and item.get("kind") == "batch":
            total = 0
            for _st, ds in item.get("entries") or []:
                total += self._delta_frame_count(ds or [])
            return total
        if isinstance(item, tuple) and len(item) == 2:
            _st, ds = item
            return self._delta_frame_count(ds or [])
        return 0

    def _apply_history_item(self, item: Any, reverse: bool = False) -> bool:
        method_name = "apply_deltas_reverse" if reverse else "apply_deltas"
        if isinstance(item, dict) and item.get("kind") == "trim_cuts":
            ops = list(item.get("ops") or [])
            if reverse:
                ops = list(reversed(ops))
            changed = False
            for op in ops:
                changed = bool(self._apply_trim_cut_op(op, reverse=reverse)) or changed
            return changed
        if isinstance(item, dict) and item.get("kind") == "composite":
            entries = list(item.get("entries") or [])
            trim_ops = list(item.get("trim_ops") or [])
            changed = False
            if reverse:
                for op in reversed(trim_ops):
                    changed = (
                        bool(self._apply_trim_cut_op(op, reverse=True)) or changed
                    )
                for st, ds in reversed(entries):
                    fn = getattr(st, "apply_deltas_reverse", None)
                    if callable(fn) and ds:
                        fn(ds)
                        changed = True
            else:
                for st, ds in entries:
                    fn = getattr(st, "apply_deltas", None)
                    if callable(fn) and ds:
                        fn(ds)
                        changed = True
                for op in trim_ops:
                    changed = (
                        bool(self._apply_trim_cut_op(op, reverse=False)) or changed
                    )
            return changed
        if isinstance(item, dict) and item.get("kind") == "batch":
            entries = list(item.get("entries") or [])
            if reverse:
                entries = list(reversed(entries))
            changed = False
            for st, ds in entries:
                fn = getattr(st, method_name, None)
                if callable(fn) and ds:
                    fn(ds)
                    changed = True
            return changed
        if isinstance(item, tuple) and len(item) == 2:
            st, ds = item
            fn = getattr(st, method_name, None)
            if callable(fn):
                fn(ds)
                return True
        return False

    def _append_validation_modifications(self, entries: List[Dict[str, Any]]) -> None:
        if not entries:
            return
        self.validation_modifications.extend(entries)
        limit = max(1, int(getattr(self, "_validation_log_limit", 20000)))
        if len(self.validation_modifications) > limit:
            del self.validation_modifications[
                : len(self.validation_modifications) - limit
            ]

    @staticmethod
    def _delta_to_span(delta: Tuple) -> Tuple[int, int, Optional[str], Optional[str]]:
        if len(delta) >= 4:
            s, e, old_label, new_label = delta[:4]
            s = int(s)
            e = int(e)
            if e < s:
                s, e = e, s
            return s, e, old_label, new_label
        frame, old_label, new_label = delta[:3]
        f = int(frame)
        return f, f, old_label, new_label

    @classmethod
    def _iter_delta_spans(cls, deltas: List[Tuple]):
        for d in deltas or []:
            try:
                yield cls._delta_to_span(d)
            except Exception:
                continue

    @classmethod
    def _delta_frame_count(cls, deltas: List[Tuple]) -> int:
        total = 0
        for s, e, _old, _new in cls._iter_delta_spans(deltas):
            total += max(0, int(e) - int(s) + 1)
        return total

    @staticmethod
    def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not spans:
            return []
        spans = sorted(
            (int(s), int(e)) if int(s) <= int(e) else (int(e), int(s)) for s, e in spans
        )
        merged: List[Tuple[int, int]] = []
        cur_s, cur_e = spans[0]
        for s, e in spans[1:]:
            if s <= cur_e + 1:
                cur_e = max(cur_e, e)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    def _summarize_deltas(self, deltas, max_chars: int = 240) -> str:
        if not deltas:
            return ""
        preview_limit = max(1, int(getattr(self, "_store_change_preview_limit", 32)))
        parts = []
        shown = 0
        for s, e, old, new in self._iter_delta_spans(deltas):
            span_txt = f"{s}" if s == e else f"{s}-{e}"
            parts.append(f"{span_txt}:{old or '-'}->{new or '-'}")
            shown += max(0, int(e) - int(s) + 1)
            if len(parts) >= preview_limit:
                break
        total = self._delta_frame_count(deltas)
        if total > shown:
            parts.append(f"... +{total - shown} more")
        summary = "; ".join(parts)
        if len(summary) > max_chars:
            summary = summary[: max(0, max_chars - 3)] + "..."
        return summary

    def _clone_store(self, store: AnnotationStore) -> AnnotationStore:
        new = AnnotationStore()
        new.frame_to_label = dict(store.frame_to_label)
        new.label_to_frames = {k: list(v) for k, v in store.label_to_frames.items()}
        return new

    def _rebuild_store_index(self, store: Optional[AnnotationStore]) -> None:
        if store is None:
            return
        label_to_frames: Dict[str, List[int]] = {}
        try:
            items = sorted(
                (int(frame), str(label))
                for frame, label in getattr(store, "frame_to_label", {}).items()
                if label
            )
        except Exception:
            items = []
        for frame, label in items:
            label_to_frames.setdefault(label, []).append(int(frame))
        store.label_to_frames = label_to_frames

    def _replace_store_frames_in_span(
        self,
        store: Optional[AnnotationStore],
        start: int,
        end: int,
        frame_map: Dict[int, str],
    ) -> None:
        if store is None:
            return
        start = int(start)
        end = int(end)
        if end < start:
            start, end = end, start
        new_map: Dict[int, str] = {}
        try:
            for frame, label in getattr(store, "frame_to_label", {}).items():
                frame_i = int(frame)
                if frame_i < start or frame_i > end:
                    new_map[frame_i] = str(label)
        except Exception:
            new_map = {}
        for frame, label in (frame_map or {}).items():
            frame_i = int(frame)
            if start <= frame_i <= end and label:
                new_map[frame_i] = str(label)
        store.frame_to_label = dict(sorted(new_map.items()))
        self._rebuild_store_index(store)

    def _patch_store_frames_in_spans(
        self,
        store: Optional[AnnotationStore],
        spans: List[Tuple[int, int]],
        frame_map: Dict[int, str],
    ) -> None:
        if store is None or not spans:
            return
        merged = self._merge_spans(spans)
        new_map: Dict[int, str] = {}
        try:
            for frame, label in getattr(store, "frame_to_label", {}).items():
                frame_i = int(frame)
                if any(int(s) <= frame_i <= int(e) for s, e in merged):
                    continue
                new_map[frame_i] = str(label)
        except Exception:
            new_map = {}
        for frame, label in (frame_map or {}).items():
            frame_i = int(frame)
            if label and any(int(s) <= frame_i <= int(e) for s, e in merged):
                new_map[frame_i] = str(label)
        store.frame_to_label = dict(sorted(new_map.items()))
        self._rebuild_store_index(store)

    def _replace_cut_set_in_span(
        self,
        cut_set: Optional[Set[int]],
        start: int,
        end: int,
        new_cuts: Iterable[int],
    ) -> None:
        if cut_set is None:
            return
        start = int(start)
        end = int(end)
        if end < start:
            start, end = end, start
        keep = {int(c) for c in (cut_set or set()) if int(c) < start or int(c) > end}
        keep.update(int(c) for c in (new_cuts or []) if start <= int(c) <= end)
        cut_set.clear()
        cut_set.update(keep)

    def _replace_cut_set_excluding_positions(
        self,
        cut_set: Optional[Set[int]],
        start: int,
        end: int,
        *,
        protected_positions: Optional[Set[int]] = None,
        new_cuts: Optional[Iterable[int]] = None,
    ) -> None:
        if cut_set is None:
            return
        start = int(start)
        end = int(end)
        if end < start:
            start, end = end, start
        protected = {int(c) for c in (protected_positions or set())}
        keep = {
            int(c)
            for c in (cut_set or set())
            if int(c) < start or int(c) > end or int(c) in protected
        }
        keep.update(
            int(c)
            for c in (new_cuts or [])
            if start <= int(c) <= end and int(c) not in protected
        )
        cut_set.clear()
        cut_set.update(keep)

    def _clone_entity_stores(
        self, stores: Dict[str, AnnotationStore]
    ) -> Dict[str, AnnotationStore]:
        cloned = {name: self._clone_store(st) for name, st in (stores or {}).items()}
        for ent in self.entities:
            cloned.setdefault(ent.name, AnnotationStore())
        return cloned

    def _clone_phase_stores(
        self, stores: Dict[str, AnnotationStore]
    ) -> Dict[str, AnnotationStore]:
        cloned = {name: self._clone_store(st) for name, st in (stores or {}).items()}
        for ent in self.entities:
            cloned.setdefault(ent.name, AnnotationStore())
        return cloned

    def _clone_anomaly_type_stores(
        self, stores: Dict[str, Dict[str, AnnotationStore]]
    ) -> Dict[str, Dict[str, AnnotationStore]]:
        cloned: Dict[str, Dict[str, AnnotationStore]] = {}
        names = self._anomaly_type_names()
        for ename, type_map in (stores or {}).items():
            new_map: Dict[str, AnnotationStore] = {}
            for tname, st in (type_map or {}).items():
                new_map[tname] = self._clone_store(st)
            for tname in names:
                new_map.setdefault(tname, AnnotationStore())
            cloned[ename] = new_map
        for ent in self.entities:
            cloned.setdefault(ent.name, {tname: AnnotationStore() for tname in names})
        return cloned

    def _psr_empty_view_state(self, single_timeline: Optional[bool] = None) -> Dict[str, Any]:
        if single_timeline is None:
            single_timeline = bool(getattr(self, "_psr_single_timeline", False))
        return {
            "manual_events": [],
            "gap_spans_combined": [],
            "gap_spans_by_comp": {},
            "selected_segment": None,
            "undo_stack": [],
            "redo_stack": [],
            "single_timeline": bool(single_timeline),
        }

    def _psr_clone_view_state(
        self, state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        src = state or {}
        return {
            "manual_events": copy.deepcopy(src.get("manual_events", [])),
            "gap_spans_combined": copy.deepcopy(src.get("gap_spans_combined", [])),
            "gap_spans_by_comp": copy.deepcopy(src.get("gap_spans_by_comp", {})),
            "selected_segment": copy.deepcopy(src.get("selected_segment")),
            "undo_stack": copy.deepcopy(src.get("undo_stack", [])),
            "redo_stack": copy.deepcopy(src.get("redo_stack", [])),
            "single_timeline": bool(
                src.get("single_timeline", getattr(self, "_psr_single_timeline", False))
            ),
        }

    def _psr_capture_active_view_state(self) -> Dict[str, Any]:
        return self._psr_clone_view_state(
            {
                "manual_events": self._psr_manual_events,
                "gap_spans_combined": self._psr_gap_spans_combined,
                "gap_spans_by_comp": self._psr_gap_spans_by_comp,
                "selected_segment": self._psr_selected_segment,
                "undo_stack": self._psr_undo_stack,
                "redo_stack": self._psr_redo_stack,
                "single_timeline": self._psr_single_timeline,
            }
        )

    def _psr_reset_runtime_state(self) -> None:
        self._psr_segment_cuts = []
        self._psr_state_store_combined = None
        self._psr_state_stores = {}
        self._psr_combined_label_states = {}
        self._psr_state_color_cache = {}
        self._psr_events_cache = []
        self._psr_state_seq = []
        self._psr_state_frames = []
        self._psr_diag = {"events": 0, "unmapped": 0, "rule_mismatch": 0}
        self._psr_action_segment_starts = []
        self._psr_action_segment_ends = []
        self._psr_action_segments = []
        self._psr_detected_flow = "assemble"
        self._psr_detected_initial_state = 0
        self._psr_cache_dirty = True
        self._psr_state_dirty = True
        self._psr_timeline_change_deferred = False

    def _psr_active_view_is_bound(self) -> bool:
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return False
        view = self.views[self.active_view_idx]
        return view.get("player") is getattr(self, "player", None)

    def _psr_ensure_view_state(
        self, view: Optional[Dict[str, Any]], clone_from_active: bool = False
    ) -> Dict[str, Any]:
        if view is None:
            return self._psr_empty_view_state()
        state = view.get("psr_state")
        if isinstance(state, dict):
            cloned = self._psr_clone_view_state(state)
        elif clone_from_active and self._psr_active_view_is_bound():
            cloned = self._psr_capture_active_view_state()
        else:
            cloned = self._psr_empty_view_state()
        view["psr_state"] = cloned
        return cloned

    def _psr_store_active_view_state(self) -> None:
        if not self._psr_active_view_is_bound():
            return
        self.views[self.active_view_idx]["psr_state"] = (
            self._psr_capture_active_view_state()
        )

    def _psr_set_single_timeline_preference(
        self,
        on: bool,
        target_indices: Optional[Iterable[int]] = None,
    ) -> None:
        on = bool(on)
        self._psr_single_timeline = on
        if not self.views:
            return
        indices = (
            range(len(self.views))
            if target_indices is None
            else [int(idx) for idx in target_indices]
        )
        active_idx = int(self.active_view_idx)
        active_snapshot = None
        if self._psr_active_view_is_bound():
            active_snapshot = self._psr_capture_active_view_state()
            active_snapshot["single_timeline"] = on
        for idx in indices:
            if not (0 <= idx < len(self.views)):
                continue
            if idx == active_idx and active_snapshot is not None:
                self.views[idx]["psr_state"] = self._psr_clone_view_state(active_snapshot)
                continue
            state = self._psr_ensure_view_state(self.views[idx])
            state["single_timeline"] = on
            self.views[idx]["psr_state"] = self._psr_clone_view_state(state)

    def _psr_load_view_state(self, view: Optional[Dict[str, Any]]) -> None:
        state = self._psr_ensure_view_state(view)
        self._psr_reset_runtime_state()
        self._psr_manual_events = copy.deepcopy(state.get("manual_events", []))
        self._psr_gap_spans_combined = copy.deepcopy(
            state.get("gap_spans_combined", [])
        )
        self._psr_gap_spans_by_comp = copy.deepcopy(
            state.get("gap_spans_by_comp", {})
        )
        self._psr_selected_segment = copy.deepcopy(state.get("selected_segment"))
        self._psr_undo_stack = copy.deepcopy(state.get("undo_stack", []))
        self._psr_redo_stack = copy.deepcopy(state.get("redo_stack", []))
        self._psr_single_timeline = bool(
            state.get("single_timeline", self._psr_single_timeline)
        )

    def _psr_sync_target_indices(self) -> List[int]:
        if (
            not self._is_psr_task()
            or bool(getattr(self, "_psr_sync_apply_in_progress", False))
        ):
            return []
        indices = self._effective_sync_edit_indices()
        return list(indices) if len(indices) > 1 else []

    def _psr_sync_context(
        self,
        frame: Any = None,
        component_id: Any = None,
        scope: Any = "segment",
    ) -> Dict[str, Any]:
        try:
            frame_val = int(
                getattr(self.player, "current_frame", 0) if frame is None else frame
            )
        except Exception:
            frame_val = int(getattr(self.player, "current_frame", 0))
        return {
            "frame": int(frame_val),
            "component_id": component_id,
            "scope": self._psr_normalize_selection_scope(scope),
        }

    def _psr_current_sync_context(self) -> Dict[str, Any]:
        seg = self._psr_sync_selected_segment()
        row = (
            getattr(self.timeline, "_active_combined_row", None)
            if getattr(self, "timeline", None)
            else None
        )
        try:
            frame = int(getattr(self.player, "current_frame", 0))
        except Exception:
            frame = 0
        if isinstance(seg, dict):
            try:
                frame = int(seg.get("start", frame))
            except Exception:
                pass
        return self._psr_sync_context(
            frame=frame,
            component_id=(
                seg.get("component_id")
                if isinstance(seg, dict)
                else self._psr_component_id_from_row(row)
            ),
            scope=(seg.get("scope", "segment") if isinstance(seg, dict) else "segment"),
        )

    def _psr_row_for_sync_context(self, context: Optional[Dict[str, Any]]):
        comp_id = context.get("component_id") if isinstance(context, dict) else None
        row = self._psr_row_for_component(comp_id)
        if row is None and getattr(self, "timeline", None) is not None:
            row = getattr(self.timeline, "_active_combined_row", None)
        if row is None and getattr(self, "timeline", None) is not None:
            rows = getattr(self.timeline, "_combined_rows", []) or []
            if rows:
                row = rows[0]
        return row

    def _psr_segment_from_sync_context(
        self, context: Optional[Dict[str, Any]]
    ) -> Tuple[int, int, Any, Any]:
        row = self._psr_row_for_sync_context(context)
        try:
            frame = int(context.get("frame", getattr(self.player, "current_frame", 0)))
        except Exception:
            frame = int(getattr(self.player, "current_frame", 0))
        if row is None:
            return int(frame), int(frame), None, None
        try:
            start, end, label = row._segment_at(int(frame))
        except Exception:
            start = end = int(frame)
            label = None
        return int(start), int(end), label, row

    def _psr_apply_sync_context(self, context: Optional[Dict[str, Any]]) -> None:
        if not self._is_psr_task() or not context:
            return
        try:
            frame = int(context.get("frame", getattr(self.player, "current_frame", 0)))
        except Exception:
            frame = int(getattr(self.player, "current_frame", 0))
        scope = self._psr_normalize_selection_scope(context.get("scope", "segment"))
        comp_id = context.get("component_id")
        try:
            self.player.seek(int(frame), preview_only=False)
        except Exception:
            pass
        try:
            self.timeline.set_current_frame(int(frame), follow=True)
        except Exception:
            pass
        row = self._psr_row_for_component(comp_id)
        if row is None and getattr(self, "timeline", None) is not None:
            rows = getattr(self.timeline, "_combined_rows", []) or []
            if rows:
                row = rows[0]
        if row is None:
            return
        if scope == "all":
            try:
                fc = max(1, self._get_frame_count())
            except Exception:
                fc = 1
            try:
                row._selected_interval = (0, fc - 1)
                row._selected_label = None
                row._selection_scope = "all"
                row.update()
            except Exception:
                pass
            try:
                self.timeline._active_combined_row = row
            except Exception:
                pass
            self._psr_set_selected_segment(
                0, fc - 1, None, row=row, scope="all", component_id=comp_id
            )
            return
        try:
            s, e, lb = row._segment_at(int(frame))
        except Exception:
            s = e = int(frame)
            lb = None
        try:
            row._selected_interval = (int(s), int(e))
            row._selected_label = lb
            row._selection_scope = scope
            row.update()
        except Exception:
            pass
        try:
            self.timeline._active_combined_row = row
        except Exception:
            pass
        self._psr_set_selected_segment(
            int(s),
            int(e),
            lb,
            row=row,
            scope=scope,
            component_id=comp_id,
        )

    def _psr_run_across_selected_views(
        self,
        runner: Callable[[], Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        indices = self._psr_sync_target_indices()
        if not indices:
            return runner()
        restore_idx = int(self.active_view_idx)
        selected_before = set(self._sync_edit_view_indices or set())
        if context is None:
            context = self._psr_current_sync_context()
        result = None
        self._psr_store_active_view_state()
        self._psr_sync_apply_in_progress = True
        try:
            for idx in indices:
                if idx != self.active_view_idx:
                    self._set_primary_view(int(idx))
                self._psr_apply_sync_context(context)
                result = runner()
        finally:
            self._psr_sync_apply_in_progress = False
            if restore_idx != self.active_view_idx:
                self._set_primary_view(restore_idx)
            self._sync_edit_view_indices = set(int(i) for i in selected_before)
            self._update_view_highlight()
            self._apply_sync_edit_masks()
        return result

    @staticmethod
    def _empty_trim_state() -> Dict[str, Any]:
        return {
            "store": set(),
            "entity": {},
            "phase": {},
        }

    def _clone_trim_state(self, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        src = state or {}
        out = self._empty_trim_state()
        out["store"] = {int(c) for c in (src.get("store") or set())}
        for kind in ("entity", "phase"):
            bucket = src.get(kind) or {}
            out[kind] = {
                str(name): {int(c) for c in (cuts or set())}
                for name, cuts in bucket.items()
            }
        return out

    def _ensure_view_trim_state(self, view: Dict[str, Any]) -> None:
        if view is None:
            return
        if "trim_cuts" not in view:
            view["trim_cuts"] = self._empty_trim_state()
        else:
            view["trim_cuts"] = self._clone_trim_state(view.get("trim_cuts"))

    def _trim_cut_set_for_view(
        self, view: Dict[str, Any], descriptor: Dict[str, Any], create: bool = False
    ) -> Set[int]:
        if view is None or descriptor is None:
            return set()
        self._ensure_view_trim_state(view)
        cuts = view.get("trim_cuts") or self._empty_trim_state()
        kind = str(descriptor.get("kind") or "")
        if kind == "store":
            return cuts.setdefault("store", set())
        if kind in ("entity", "phase"):
            key = str(descriptor.get("entity") or "")
            bucket = cuts.setdefault(kind, {})
            if create:
                return bucket.setdefault(key, set())
            return bucket.get(key, set())
        return set()

    def _segment_bounds_with_cuts(
        self, st: Optional[AnnotationStore], frame: int, cuts: Optional[Set[int]] = None
    ) -> Tuple[int, int, Optional[str]]:
        if st is None:
            return int(frame), int(frame), None
        fc = max(1, self._get_frame_count())
        f = int(frame)
        f = max(0, min(f, fc - 1))
        lb = st.label_at(f)
        cut_set = {int(c) for c in (cuts or set()) if c is not None}
        s = f
        while s > 0 and st.label_at(s - 1) == lb and s not in cut_set:
            s -= 1
        e = f
        while e < fc - 1 and st.label_at(e + 1) == lb and (e + 1) not in cut_set:
            e += 1
        return int(s), int(e), lb

    def _active_trim_cuts_for_descriptor(self, descriptor: Dict[str, Any]) -> List[int]:
        if not (self.views and 0 <= self.active_view_idx < len(self.views)):
            return []
        cuts = self._trim_cut_set_for_view(
            self.views[self.active_view_idx], descriptor, create=False
        )
        return sorted(int(c) for c in (cuts or set()))

    def _baseline_trim_cut_set_for_view(
        self, view: Dict[str, Any], descriptor: Dict[str, Any], create: bool = False
    ) -> Set[int]:
        if view is None or descriptor is None:
            return set()
        cuts = view.get("baseline_trim_cuts")
        if not isinstance(cuts, dict):
            cuts = self._empty_trim_state()
            view["baseline_trim_cuts"] = cuts
        kind = str(descriptor.get("kind") or "")
        if kind == "store":
            return cuts.setdefault("store", set())
        if kind in ("entity", "phase"):
            key = str(descriptor.get("entity") or "")
            bucket = cuts.setdefault(kind, {})
            if create:
                return bucket.setdefault(key, set())
            return bucket.get(key, set())
        return set()

    def _active_baseline_trim_cuts_for_descriptor(
        self, descriptor: Dict[str, Any]
    ) -> List[int]:
        if not (self.views and 0 <= self.active_view_idx < len(self.views)):
            return []
        cuts = self._baseline_trim_cut_set_for_view(
            self.views[self.active_view_idx], descriptor, create=False
        )
        return sorted(int(c) for c in (cuts or set()))

    def _correction_features_dir(self) -> Optional[str]:
        for raw in (
            getattr(self, "currentEastFeatureDir", None),
            self._default_east_features_dir_for_video(),
        ):
            path = str(raw or "").strip()
            if not path:
                continue
            path = os.path.abspath(path)
            if os.path.isdir(path):
                return path
        return None

    def _active_label_baseline_store(self) -> Optional[AnnotationStore]:
        if self.views and 0 <= self.active_view_idx < len(self.views):
            baseline = self.views[self.active_view_idx].get("prelabel_store")
            if baseline is not None:
                return baseline
        return getattr(self, "prelabel_store", None)

    def _active_label_baseline_source(self) -> str:
        if self.views and 0 <= self.active_view_idx < len(self.views):
            text = str(self.views[self.active_view_idx].get("prelabel_source", "") or "").strip()
            if text:
                return text
        return str(getattr(self, "_prelabel_source", "") or "").strip()

    def _set_active_label_baseline_source(self, source: str) -> None:
        text = str(source or "").strip().upper()
        self._prelabel_source = text
        if self.views and 0 <= self.active_view_idx < len(self.views):
            self.views[self.active_view_idx]["prelabel_source"] = text

    def _east_runtime_meta_path(self, features_dir: Optional[str]) -> str:
        path = os.path.abspath(os.path.expanduser(str(features_dir or "").strip()))
        if not path:
            return ""
        return os.path.join(path, "east_runtime", "meta.json")

    def _load_east_runtime_meta(self, features_dir: Optional[str]) -> Dict[str, Any]:
        meta_path = self._east_runtime_meta_path(features_dir)
        if not meta_path or not os.path.isfile(meta_path):
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _update_east_runtime_meta(self, features_dir: Optional[str], **fields: Any) -> bool:
        meta_path = self._east_runtime_meta_path(features_dir)
        if not meta_path:
            return False
        try:
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            meta = self._load_east_runtime_meta(features_dir)
            meta.update(fields)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=True, indent=2, default=str)
            return True
        except Exception:
            return False

    def _east_runtime_debug_context(
        self,
        report: Optional[Dict[str, Any]] = None,
        *,
        features_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        active_features_dir = str(features_dir or "") or str(
            (report or {}).get("features_dir") or ""
        )
        active_features_dir = (
            os.path.abspath(os.path.expanduser(active_features_dir.strip()))
            if active_features_dir.strip()
            else ""
        )
        runtime_ok = bool((report or {}).get("ok", False))
        runtime_meta = dict((report or {}).get("runtime_meta") or {})
        if not runtime_meta:
            runtime_meta = self._load_east_runtime_meta(active_features_dir)
        baseline_source = str(self._active_label_baseline_source() or "").strip().upper()
        try:
            confirmed_count = int(
                len(self._rebuild_confirmed_correction_records_for_active_view())
            )
        except Exception:
            confirmed_count = int(
                len(getattr(self, "_confirmed_correction_records", []) or [])
            )
        features_ready = bool(
            active_features_dir
            and os.path.isfile(os.path.join(active_features_dir, "features.npy"))
        )
        bootstrap_source = str(
            runtime_meta.get("bootstrap_source") or baseline_source or ""
        ).strip().upper()
        bootstrap_pending = bool(runtime_meta.get("bootstrap_pending", False))
        bootstrap_completed = bool(runtime_meta.get("bootstrap_completed", False))
        runtime_origin = str(runtime_meta.get("runtime_origin") or "").strip().lower()
        if runtime_ok:
            if runtime_origin == "bootstrapped" or bootstrap_source or bootstrap_pending or bootstrap_completed:
                runtime_source = "bootstrapped"
                source_name = bootstrap_source or baseline_source or "CONFIRMED_SUPERVISION"
                runtime_source_label = f"{source_name} bootstrapped to EAST"
            else:
                runtime_source = "native_east"
                runtime_source_label = "Native EAST runtime"
        else:
            runtime_source = "not_initialized"
            runtime_source_label = (
                "EAST features ready; runtime not initialized"
                if features_ready
                else "EAST not initialized"
            )
        return {
            "runtime_available": bool(runtime_ok),
            "runtime_status_reason": str((report or {}).get("reason") or ""),
            "baseline_source": baseline_source,
            "current_confirmed_record_count": int(confirmed_count),
            "features_ready": bool(features_ready),
            "runtime_source": runtime_source,
            "runtime_source_label": runtime_source_label,
            "bootstrap_source": bootstrap_source,
            "bootstrap_pending": bool(bootstrap_pending),
            "bootstrap_completed": bool(bootstrap_completed),
            "bootstrap_confirmed_count": int(
                runtime_meta.get(
                    "bootstrap_confirmed_count",
                    confirmed_count if confirmed_count > 0 else 0,
                )
                or 0
            ),
            "bootstrap_initialized_at": str(
                runtime_meta.get("bootstrap_initialized_at") or ""
            ),
            "bootstrap_refresh_deferred": bool(
                runtime_meta.get("bootstrap_refresh_deferred", False)
            ),
        }

    def _east_online_updates_allowed(self) -> bool:
        return True

    def _active_explicit_confirm_records(self, create: bool = False) -> List[Dict[str, Any]]:
        if not (self.views and 0 <= self.active_view_idx < len(self.views)):
            return []
        view = self.views[self.active_view_idx]
        rows = view.get("confirmed_accept_records")
        if isinstance(rows, list):
            return rows
        if create:
            rows = []
            view["confirmed_accept_records"] = rows
            return rows
        return []

    def _confirm_record_key(self, rec: Dict[str, Any]) -> Tuple[Any, ...]:
        point_type = str(rec.get("point_type", "") or "").strip().lower()
        action_kind = str(rec.get("action_kind", "") or "").strip().lower()
        if point_type == "boundary":
            try:
                frame = int(rec.get("boundary_frame", rec.get("feedback_start", 0)))
            except Exception:
                frame = 0
            return (point_type, action_kind, int(frame))
        try:
            start = int(rec.get("feedback_start", 0))
            end = int(rec.get("feedback_end", start))
        except Exception:
            start = end = 0
        label = str(rec.get("label", "") or "").strip()
        return (point_type, action_kind, int(start), int(end), label)

    def _store_explicit_confirm_record(self, rec: Dict[str, Any]) -> None:
        if not isinstance(rec, dict):
            return
        rows = self._active_explicit_confirm_records(create=True)
        key = self._confirm_record_key(rec)
        clean = dict(rec)
        for idx, old in enumerate(list(rows)):
            if self._confirm_record_key(old) == key:
                rows[idx] = clean
                return
        rows.append(clean)

    def _build_accept_record_for_point(
        self, pt: Optional[Dict[str, Any]], *, boundary_frame: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(pt, dict) or not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return None
        view = self.views[self.active_view_idx]
        view_name = self._effective_view_name(view, idx=self.active_view_idx)
        point_type = str(pt.get("type", "") or "").strip().lower()
        if point_type == "label":
            try:
                start = int(pt.get("start", 0))
                end = int(pt.get("end", start))
            except Exception:
                return None
            label = str(pt.get("label", "") or "").strip()
            if not label:
                return None
            if end < start:
                start, end = end, start
            return {
                "schema_version": 2,
                "point_type": "label",
                "action_kind": "label_accept",
                "segment_action": "segment_accept",
                "feedback_start": int(start),
                "feedback_end": int(end),
                "label": label,
                "old_label": "",
                "hard_negative_labels": [],
                "view_name": view_name,
                "source": "confirmed_accept_v1",
                "confirmed_kind": "accepted",
            }
        if point_type == "boundary":
            try:
                frame = int(
                    boundary_frame
                    if boundary_frame is not None
                    else pt.get("frame", 0)
                )
            except Exception:
                return None
            left_label = str((pt.get("left") or {}).get("label", "") or "").strip()
            right_label = str((pt.get("right") or {}).get("label", "") or "").strip()
            return {
                "schema_version": 2,
                "point_type": "boundary",
                "action_kind": "boundary_accept",
                "segment_action": "segment_accept",
                "feedback_start": int(frame),
                "feedback_end": int(frame),
                "boundary_frame": int(frame),
                "left_label": left_label,
                "right_label": right_label,
                "view_name": view_name,
                "source": "confirmed_accept_v1",
                "confirmed_kind": "accepted",
            }
        return None

    def _correction_boundary_match_radius(self) -> int:
        try:
            radius = int(self._interaction_cfg.get("boundary", {}).get("window_size", 10))
        except Exception:
            radius = 10
        return max(1, radius)

    def _segments_for_correction_store(
        self,
        store: Optional[AnnotationStore],
        *,
        start: int,
        end: int,
        cut_frames: Optional[Iterable[int]] = None,
    ) -> List[Dict[str, Any]]:
        if store is None:
            return []
        try:
            frames = sorted(
                int(f)
                for f in getattr(store, "frame_to_label", {}).keys()
                if int(start) <= int(f) <= int(end)
            )
        except Exception:
            frames = []
        if not frames:
            return []
        cut_set = {int(c) for c in (cut_frames or []) if start <= int(c) <= end}
        segments: List[Dict[str, Any]] = []
        s = frames[0]
        prev = frames[0]
        cur = store.label_at(prev)
        for f in frames[1:]:
            lb = store.label_at(f)
            split = (f != prev + 1) or (lb != cur) or (f in cut_set)
            if split:
                if cur and not is_extra_label(str(cur)):
                    segments.append(
                        {
                            "start": int(s),
                            "end": int(prev),
                            "label": str(cur),
                        }
                    )
                s = f
                cur = lb
            prev = f
        if cur and not is_extra_label(str(cur)):
            segments.append(
                {
                    "start": int(s),
                    "end": int(prev),
                    "label": str(cur),
                }
            )
        return segments

    def _assisted_candidates_for_span(
        self, start: int, end: int
    ) -> List[Tuple[str, Optional[float]]]:
        try:
            start = int(start)
            end = int(end)
        except Exception:
            return []
        if end < start:
            start, end = end, start
        best_score = -1.0
        best_items: List[Tuple[str, Optional[float]]] = []
        for key, raw_items in dict(getattr(self, "_assisted_candidates", {}) or {}).items():
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            try:
                ks = int(key[0])
                ke = int(key[1])
            except Exception:
                continue
            overlap = min(end, ke) - max(start, ks) + 1
            if overlap <= 0:
                continue
            union = max(end, ke) - min(start, ks) + 1
            score = float(overlap) / float(max(1, union))
            if score < best_score:
                continue
            items: List[Tuple[str, Optional[float]]] = []
            for item in raw_items or []:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("label")
                    raw_score = item.get("score")
                elif isinstance(item, (tuple, list)) and item:
                    name = item[0]
                    raw_score = item[1] if len(item) > 1 else None
                else:
                    continue
                label = str(name or "").strip()
                if not label:
                    continue
                try:
                    score_val = None if raw_score is None else float(raw_score)
                except Exception:
                    score_val = None
                items.append((label, score_val))
            if not items:
                continue
            best_score = score
            best_items = items
        return list(best_items)

    def _hard_negative_labels_for_span(
        self,
        start: int,
        end: int,
        *,
        new_label: str = "",
        old_label: str = "",
    ) -> List[str]:
        negatives: List[str] = []
        target = str(new_label or "").strip()
        previous = str(old_label or "").strip()
        if previous and previous != target and previous not in negatives:
            negatives.append(previous)
        for name, _score in self._assisted_candidates_for_span(start, end):
            label = str(name or "").strip()
            if not label or label == target or label in negatives:
                continue
            negatives.append(label)
        for label in self._confusion_memory_negatives_for_label(target):
            if not label or label == target or label in negatives:
                continue
            negatives.append(label)
        return negatives[:8]

    def _confusion_memory_negatives_for_label(
        self,
        label_name: str,
    ) -> List[str]:
        target = str(label_name or "").strip()
        if not target:
            return []
        features_dir = self._correction_features_dir()
        if not features_dir:
            return []
        try:
            confusion_map = load_east_runtime_confusion_map(features_dir)
        except Exception:
            return []
        row = dict(confusion_map.get(target) or {})
        if not row:
            return []
        try:
            limit = int(os.environ.get("EAST_CONFUSION_MEMORY_TOPK", "3") or 3)
        except Exception:
            limit = 3
        limit = max(0, min(8, int(limit)))
        ordered = sorted(
            (
                (str(name).strip(), float(score))
                for name, score in row.items()
                if str(name).strip() and str(name).strip() != target
            ),
            key=lambda item: (-float(item[1]), str(item[0])),
        )
        return [name for name, _score in ordered[:limit]]

    def _store_diff_records_for_active_view(self) -> List[Dict[str, Any]]:
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return []
        view = self.views[self.active_view_idx]
        current_store = view.get("store") or self.store
        baseline_store = self._active_label_baseline_store()
        if current_store is None:
            return []
        start = int(view.get("start", 0))
        end = int(view.get("end", start))
        if end < start:
            start, end = end, start
        changed_frames: List[Tuple[int, Optional[str], Optional[str]]] = []
        current_keys = set()
        baseline_keys = set()
        try:
            current_keys = {
                int(f)
                for f in getattr(current_store, "frame_to_label", {}).keys()
                if start <= int(f) <= end
            }
        except Exception:
            current_keys = set()
        try:
            baseline_keys = {
                int(f)
                for f in getattr(baseline_store, "frame_to_label", {}).keys()
                if start <= int(f) <= end
            }
        except Exception:
            baseline_keys = set()
        split_points = set(self._active_trim_cuts_for_descriptor({"kind": "store"}) or [])
        split_points.update(
            set(self._active_baseline_trim_cuts_for_descriptor({"kind": "store"}) or [])
        )
        for frame in sorted(current_keys | baseline_keys):
            cur = current_store.label_at(frame) if current_store else None
            old = baseline_store.label_at(frame) if baseline_store else None
            if is_extra_label(str(cur or "")):
                cur = None
            if is_extra_label(str(old or "")):
                old = None
            if cur == old:
                continue
            changed_frames.append((int(frame), old, cur))
        if not changed_frames:
            return []
        runs: List[Tuple[int, int, Optional[str], Optional[str]]] = []
        run_start = changed_frames[0][0]
        prev_frame = changed_frames[0][0]
        run_old = changed_frames[0][1]
        run_new = changed_frames[0][2]
        for frame, old_label, new_label in changed_frames[1:]:
            if (
                frame == prev_frame + 1
                and old_label == run_old
                and new_label == run_new
                and frame not in split_points
            ):
                prev_frame = frame
                continue
            runs.append((run_start, prev_frame, run_old, run_new))
            run_start = prev_frame = frame
            run_old = old_label
            run_new = new_label
        runs.append((run_start, prev_frame, run_old, run_new))
        view_name = self._effective_view_name(view, idx=self.active_view_idx)
        records: List[Dict[str, Any]] = []
        for s, e, old_label, new_label in runs:
            old_name = str(old_label or "").strip()
            new_name = str(new_label or "").strip()
            if new_name and not old_name:
                action_kind = "label_assign"
                segment_action = "segment_create"
            elif old_name and not new_name:
                action_kind = "label_remove"
                segment_action = "segment_delete"
            elif old_name and new_name and old_name != new_name:
                action_kind = "label_replace"
                segment_action = "segment_relabel"
            else:
                continue
            hard_negative_labels = self._hard_negative_labels_for_span(
                int(s),
                int(e),
                new_label=new_name,
                old_label=old_name,
            )
            records.append(
                {
                    "schema_version": 2,
                    "point_type": "label",
                    "action_kind": action_kind,
                    "segment_action": segment_action,
                    "feedback_start": int(s),
                    "feedback_end": int(e),
                    "label": new_name,
                    "old_label": old_name,
                    "hard_negative_labels": list(hard_negative_labels),
                    "view_name": view_name,
                    "source": "confirmed_correction_v2",
                    "confirmed_kind": "corrected",
                }
            )
        return records

    def _trim_cut_diff_records_for_active_view(self) -> List[Dict[str, Any]]:
        descriptor = {"kind": "store"}
        current = set(self._active_trim_cuts_for_descriptor(descriptor) or [])
        baseline = set(self._active_baseline_trim_cuts_for_descriptor(descriptor) or [])
        added = sorted(int(f) for f in (current - baseline))
        removed = sorted(int(f) for f in (baseline - current))
        if (
            (not added and not removed)
            or not self.views
            or not (0 <= self.active_view_idx < len(self.views))
        ):
            return []
        view = self.views[self.active_view_idx]
        view_name = self._effective_view_name(view, idx=self.active_view_idx)
        radius = self._correction_boundary_match_radius()
        remaining_removed = list(removed)
        records: List[Dict[str, Any]] = []

        for frame in added:
            best_idx = -1
            best_dist = None
            for idx, old_frame in enumerate(remaining_removed):
                dist = abs(int(frame) - int(old_frame))
                if dist > radius:
                    continue
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx >= 0:
                old_frame = int(remaining_removed.pop(best_idx))
                records.append(
                    {
                        "schema_version": 2,
                        "point_type": "boundary",
                        "action_kind": "boundary_move",
                        "feedback_start": int(min(old_frame, frame)),
                        "feedback_end": int(max(old_frame, frame)),
                        "boundary_frame": int(frame),
                        "old_boundary_frame": int(old_frame),
                        "delta_frames": int(frame - old_frame),
                        "view_name": view_name,
                        "source": "confirmed_correction_v2",
                        "confirmed_kind": "corrected",
                    }
                )
            else:
                records.append(
                    {
                        "schema_version": 2,
                        "point_type": "boundary",
                        "action_kind": "boundary_add",
                        "segment_action": "segment_split",
                        "feedback_start": int(frame),
                        "feedback_end": int(frame),
                        "boundary_frame": int(frame),
                        "view_name": view_name,
                        "source": "confirmed_correction_v2",
                        "confirmed_kind": "corrected",
                    }
                )

        for frame in remaining_removed:
            records.append(
                {
                    "schema_version": 2,
                    "point_type": "boundary",
                    "action_kind": "boundary_remove",
                    "segment_action": "segment_merge",
                    "feedback_start": int(frame),
                    "feedback_end": int(frame),
                    "old_boundary_frame": int(frame),
                    "view_name": view_name,
                    "source": "confirmed_correction_v2",
                    "confirmed_kind": "corrected",
                }
            )
        return records

    def _transition_diff_records_for_active_view(self) -> List[Dict[str, Any]]:
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return []
        view = self.views[self.active_view_idx]
        current_store = view.get("store") or self.store
        baseline_store = self._active_label_baseline_store()
        if current_store is None:
            return []
        start = int(view.get("start", 0))
        end = int(view.get("end", start))
        if end < start:
            start, end = end, start
        current_segments = self._segments_for_correction_store(
            current_store,
            start=start,
            end=end,
            cut_frames=self._active_trim_cuts_for_descriptor({"kind": "store"}),
        )
        baseline_segments = self._segments_for_correction_store(
            baseline_store,
            start=start,
            end=end,
            cut_frames=self._active_baseline_trim_cuts_for_descriptor({"kind": "store"}),
        )
        current_pairs = Counter(
            (str(current_segments[i]["label"]), str(current_segments[i + 1]["label"]))
            for i in range(len(current_segments) - 1)
            if current_segments[i].get("label")
            and current_segments[i + 1].get("label")
            and str(current_segments[i]["label"]) != str(current_segments[i + 1]["label"])
        )
        baseline_pairs = Counter(
            (str(baseline_segments[i]["label"]), str(baseline_segments[i + 1]["label"]))
            for i in range(len(baseline_segments) - 1)
            if baseline_segments[i].get("label")
            and baseline_segments[i + 1].get("label")
            and str(baseline_segments[i]["label"]) != str(baseline_segments[i + 1]["label"])
        )
        if not current_pairs and not baseline_pairs:
            return []
        view_name = self._effective_view_name(view, idx=self.active_view_idx)
        records: List[Dict[str, Any]] = []
        for pair in sorted(set(current_pairs.keys()) | set(baseline_pairs.keys())):
            cur_count = int(current_pairs.get(pair, 0))
            base_count = int(baseline_pairs.get(pair, 0))
            delta = cur_count - base_count
            if delta > 0:
                records.append(
                    {
                        "schema_version": 2,
                        "point_type": "transition",
                        "action_kind": "transition_add",
                        "from_label": str(pair[0]),
                        "to_label": str(pair[1]),
                        "count": int(delta),
                        "view_name": view_name,
                        "source": "confirmed_correction_v2",
                        "confirmed_kind": "corrected",
                    }
                )
            elif delta < 0:
                records.append(
                    {
                        "schema_version": 2,
                        "point_type": "transition",
                        "action_kind": "transition_remove",
                        "from_label": str(pair[0]),
                        "to_label": str(pair[1]),
                        "count": int(abs(delta)),
                        "view_name": view_name,
                        "source": "confirmed_correction_v2",
                        "confirmed_kind": "corrected",
                    }
                )
        return records

    def _segment_lock_records_for_active_view(self) -> List[Dict[str, Any]]:
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return []
        view = self.views[self.active_view_idx]
        current_store = view.get("store") or self.store
        if current_store is None:
            return []
        start = int(view.get("start", 0))
        end = int(view.get("end", start))
        if end < start:
            start, end = end, start
        current_segments = self._segments_for_correction_store(
            current_store,
            start=start,
            end=end,
            cut_frames=self._active_trim_cuts_for_descriptor({"kind": "store"}),
        )
        if not current_segments:
            return []

        affected_spans: List[Tuple[int, int]] = []
        for rec in self._store_diff_records_for_active_view():
            try:
                s = int(rec.get("feedback_start", 0))
                e = int(rec.get("feedback_end", s))
            except Exception:
                continue
            if e < s:
                s, e = e, s
            affected_spans.append((int(s), int(e)))

        if not affected_spans:
            for rec in self._active_explicit_confirm_records(create=False):
                if str(rec.get("point_type", "") or "").strip().lower() != "label":
                    continue
                try:
                    s = int(rec.get("feedback_start", 0))
                    e = int(rec.get("feedback_end", s))
                except Exception:
                    continue
                if e < s:
                    s, e = e, s
                affected_spans.append((int(s), int(e)))
        if not affected_spans:
            return []

        merged_spans: List[Tuple[int, int]] = []
        for s, e in sorted(affected_spans):
            s = max(int(start), int(s))
            e = min(int(end), int(e))
            if e < s:
                continue
            if not merged_spans or s > merged_spans[-1][1] + 1:
                merged_spans.append((s, e))
            else:
                prev_s, prev_e = merged_spans[-1]
                merged_spans[-1] = (prev_s, max(prev_e, e))

        locked_segments: List[Dict[str, Any]] = []
        seen = set()
        for seg in current_segments:
            seg_s = int(seg.get("start", 0))
            seg_e = int(seg.get("end", seg_s))
            seg_label = str(seg.get("label", "") or "").strip()
            if not seg_label:
                continue
            overlaps = any(not (seg_e < s or seg_s > e) for s, e in merged_spans)
            if not overlaps:
                continue
            key = (seg_s, seg_e, seg_label)
            if key in seen:
                continue
            seen.add(key)
            locked_segments.append(
                {
                    "schema_version": 2,
                    "point_type": "segment",
                    "action_kind": "segment_lock",
                    "segment_action": "segment_lock",
                    "feedback_start": int(seg_s),
                    "feedback_end": int(seg_e),
                    "label": seg_label,
                    "view_name": self._effective_view_name(view, idx=self.active_view_idx),
                    "source": "confirmed_correction_v2",
                    "confirmed_kind": "corrected",
                }
            )
        return locked_segments

    def _rebuild_confirmed_correction_records_for_active_view(self) -> List[Dict[str, Any]]:
        records = []
        label_diff_records = self._store_diff_records_for_active_view()
        boundary_diff_records = self._trim_cut_diff_records_for_active_view()
        records.extend(label_diff_records)
        records.extend(boundary_diff_records)
        records.extend(self._transition_diff_records_for_active_view())
        records.extend(self._segment_lock_records_for_active_view())
        accept_records = list(self._active_explicit_confirm_records(create=False) or [])
        if accept_records:
            label_diff_spans = {
                (
                    int(rec.get("feedback_start", 0)),
                    int(rec.get("feedback_end", rec.get("feedback_start", 0))),
                )
                for rec in label_diff_records
                if str(rec.get("point_type", "") or "").strip().lower() == "label"
            }
            boundary_diff_frames = set()
            for rec in boundary_diff_records:
                try:
                    if rec.get("boundary_frame") is not None:
                        boundary_diff_frames.add(int(rec.get("boundary_frame")))
                except Exception:
                    pass
                try:
                    if rec.get("old_boundary_frame") is not None:
                        boundary_diff_frames.add(int(rec.get("old_boundary_frame")))
                except Exception:
                    pass
            filtered_accepts: List[Dict[str, Any]] = []
            for rec in accept_records:
                point_type = str(rec.get("point_type", "") or "").strip().lower()
                if point_type == "label":
                    try:
                        span = (
                            int(rec.get("feedback_start", 0)),
                            int(rec.get("feedback_end", rec.get("feedback_start", 0))),
                        )
                    except Exception:
                        continue
                    if span in label_diff_spans:
                        continue
                elif point_type == "boundary":
                    try:
                        frame = int(rec.get("boundary_frame", rec.get("feedback_start", 0)))
                    except Exception:
                        continue
                    if frame in boundary_diff_frames:
                        continue
                filtered_accepts.append(dict(rec))
            records.extend(filtered_accepts)
        records.sort(
            key=lambda row: (
                int(row.get("feedback_start", 0) or 0),
                0 if str(row.get("point_type", "") or "") == "boundary" else 1,
                str(row.get("action_kind", "") or ""),
                str(row.get("label", "") or ""),
            )
        )
        self._confirmed_correction_records = list(records)
        return list(records)

    def _write_correction_record_buffer(
        self, records: List[Dict[str, Any]], *, force: bool = False
    ) -> bool:
        if not force and not self._east_online_updates_allowed():
            return False
        features_dir = self._correction_features_dir()
        if not features_dir:
            return False
        runtime_dir = os.path.join(features_dir, "east_runtime")
        try:
            os.makedirs(runtime_dir, exist_ok=True)
            path = os.path.join(runtime_dir, "record_buffer.pkl")
            with open(path, "wb") as f:
                pickle.dump(list(records or []), f)
            return True
        except Exception:
            return False

    def _write_finalized_record_buffer(
        self, records: List[Dict[str, Any]], *, force: bool = False
    ) -> bool:
        if not force and not self._east_online_updates_allowed():
            return False
        features_dir = self._correction_features_dir()
        if not features_dir:
            return False
        runtime_dir = os.path.join(features_dir, "east_runtime")
        try:
            os.makedirs(runtime_dir, exist_ok=True)
            path = os.path.join(runtime_dir, "finalized_record_buffer.pkl")
            with open(path, "wb") as f:
                pickle.dump(list(records or []), f)
            return True
        except Exception:
            return False

    def _invalidate_east_video_finalization(
        self, features_dir: Optional[str] = None
    ) -> None:
        path = str(features_dir or self._correction_features_dir() or "").strip()
        if not path:
            return
        runtime_dir = os.path.join(os.path.abspath(path), "east_runtime")
        finalized_path = os.path.join(runtime_dir, "finalized_record_buffer.pkl")
        try:
            if os.path.isfile(finalized_path):
                os.remove(finalized_path)
        except Exception:
            pass
        self._update_east_runtime_meta(
            path,
            video_finalized=False,
            finalized_record_count=0,
            finalized_at="",
        )

    def _build_finalized_supervision_records_for_active_view(self) -> List[Dict[str, Any]]:
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return []
        view = self.views[self.active_view_idx]
        current_store = view.get("store") or self.store
        if current_store is None:
            return []
        start = int(view.get("start", 0))
        end = int(view.get("end", start))
        if end < start:
            start, end = end, start
        cut_frames = self._active_trim_cuts_for_descriptor({"kind": "store"})
        current_segments = self._segments_for_correction_store(
            current_store,
            start=start,
            end=end,
            cut_frames=cut_frames,
        )
        if not current_segments:
            return []
        view_name = self._effective_view_name(view, idx=self.active_view_idx)
        records: List[Dict[str, Any]] = []
        for seg in current_segments:
            seg_s = int(seg.get("start", 0))
            seg_e = int(seg.get("end", seg_s))
            seg_label = str(seg.get("label", "") or "").strip()
            if not seg_label:
                continue
            records.append(
                {
                    "schema_version": 2,
                    "point_type": "label",
                    "action_kind": "label_finalize",
                    "segment_action": "segment_finalize",
                    "feedback_start": int(seg_s),
                    "feedback_end": int(seg_e),
                    "label": seg_label,
                    "old_label": "",
                    "hard_negative_labels": list(
                        self._hard_negative_labels_for_span(
                            int(seg_s),
                            int(seg_e),
                            new_label=seg_label,
                            old_label="",
                        )
                    ),
                    "view_name": view_name,
                    "source": "finalized_supervision_v1",
                    "confirmed_kind": "finalized",
                }
            )
            records.append(
                {
                    "schema_version": 2,
                    "point_type": "segment",
                    "action_kind": "segment_lock",
                    "segment_action": "segment_lock",
                    "feedback_start": int(seg_s),
                    "feedback_end": int(seg_e),
                    "label": seg_label,
                    "view_name": view_name,
                    "source": "finalized_supervision_v1",
                    "confirmed_kind": "finalized",
                }
            )
        seg_by_start = {int(seg.get("start", 0)): seg for seg in current_segments}
        for frame in sorted(int(f) for f in cut_frames if start <= int(f) <= end):
            left_label = ""
            right_label = ""
            for seg in current_segments:
                seg_s = int(seg.get("start", 0))
                seg_e = int(seg.get("end", seg_s))
                if seg_s <= frame - 1 <= seg_e:
                    left_label = str(seg.get("label", "") or "").strip()
                if seg_s <= frame <= seg_e or seg_s == frame:
                    right_label = str(seg.get("label", "") or "").strip()
                    if right_label:
                        break
            records.append(
                {
                    "schema_version": 2,
                    "point_type": "boundary",
                    "action_kind": "boundary_finalize",
                    "segment_action": "segment_finalize",
                    "feedback_start": int(frame),
                    "feedback_end": int(frame),
                    "boundary_frame": int(frame),
                    "left_label": left_label,
                    "right_label": right_label,
                    "view_name": view_name,
                    "source": "finalized_supervision_v1",
                    "confirmed_kind": "finalized",
                }
            )
        for idx in range(len(current_segments) - 1):
            left = current_segments[idx]
            right = current_segments[idx + 1]
            src = str(left.get("label", "") or "").strip()
            dst = str(right.get("label", "") or "").strip()
            if not src or not dst or src == dst:
                continue
            records.append(
                {
                    "schema_version": 2,
                    "point_type": "transition",
                    "action_kind": "transition_add",
                    "from_label": src,
                    "to_label": dst,
                    "count": 1,
                    "view_name": view_name,
                    "source": "finalized_supervision_v1",
                    "confirmed_kind": "finalized",
                }
            )
        records.sort(
            key=lambda row: (
                int(row.get("feedback_start", 0) or 0),
                0 if str(row.get("point_type", "") or "") == "boundary" else 1,
                str(row.get("action_kind", "") or ""),
                str(row.get("label", "") or ""),
            )
        )
        return records

    def _finalize_current_video_for_east(self) -> None:
        if not getattr(self, "video_path", None):
            QMessageBox.information(
                self, "Finalize Current Video", "Load a video first."
            )
            return
        records = self._build_finalized_supervision_records_for_active_view()
        if not records:
            QMessageBox.information(
                self,
                "Finalize Current Video",
                "There are no labeled segments to finalize for the current video.",
            )
            return
        feat_dir = self._ensure_east_features_for_current_video()
        if not feat_dir:
            return
        feat_dir = os.path.abspath(feat_dir)
        self.currentEastFeatureDir = feat_dir
        self._pending_finalized_records = list(records)
        self._update_east_runtime_meta(
            feat_dir,
            video_finalized=False,
            finalized_record_count=int(len(records)),
            finalized_pending=True,
        )
        if os.path.isfile(os.path.join(feat_dir, "features.npy")):
            persisted = self._write_finalized_record_buffer(records, force=True)
            if not persisted:
                QMessageBox.warning(
                    self,
                    "Finalize Current Video",
                    "Failed to persist finalized supervision for the current video.",
                )
                return
            self._update_east_runtime_meta(
                feat_dir,
                video_finalized=True,
                finalized_pending=False,
                finalized_record_count=int(len(records)),
                finalized_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            )
            self._set_status(
                "Current video finalized for EAST offline supervision. Updating runtime..."
            )
            self._schedule_east_online_update(feat_dir)
            return
        if not self._ensure_east_runtime_ready(
            feature_name="Finalize Current Video",
            unavailable_status=(
                "Finalizing the current video requires a full EAST-main checkout and the EAST runtime dependencies."
            ),
            show_dialog=True,
        ):
            return
        if not self._ensure_east_model_paths():
            return
        self._set_status(
            "Building EAST backbone features before finalizing the current video..."
        )
        self._start_feature_extraction(
            feat_dir,
            next_task="east_finalize",
            backbone="east_backbone",
        )

    def _begin_correction_session(self, kind: str, **meta) -> None:
        self._correction_buffer.begin(kind, meta=meta, replace=True)

    def _note_correction_step(self, count: int = 1) -> None:
        self._correction_buffer.note_step(count)
        if int(count or 0) > 0:
            self._invalidate_east_video_finalization()

    def _commit_correction_session(self, **meta_update) -> Dict[str, Any]:
        records = self._rebuild_confirmed_correction_records_for_active_view()
        features_dir = self._correction_features_dir()
        persisted = self._write_correction_record_buffer(records)
        if meta_update:
            meta_update = dict(meta_update)
        else:
            meta_update = {}
        meta_update["persisted"] = bool(persisted)
        summary = self._correction_buffer.commit(records=records, meta_update=meta_update)
        if persisted and features_dir and self._east_online_updates_allowed():
            self._schedule_east_online_update(features_dir)
        elif records and self._east_online_updates_allowed():
            self._maybe_bootstrap_east_runtime_from_confirmed_supervision(records)
        return summary

    def _discard_correction_session(self, reason: str = "") -> None:
        self._correction_buffer.discard(reason=reason)

    def _maybe_bootstrap_east_runtime_from_confirmed_supervision(
        self, records: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        rows = list(records or self._confirmed_correction_records or [])
        if not rows or not getattr(self, "video_path", None):
            return False
        bootstrap_source = str(
            self._active_label_baseline_source() or "CONFIRMED_SUPERVISION"
        ).strip().upper()
        existing = str(getattr(self, "currentEastFeatureDir", "") or "").strip()
        if existing and os.path.isfile(os.path.join(existing, "features.npy")):
            return False
        feat_dir = self._ensure_east_features_for_current_video()
        if not feat_dir:
            return False
        feat_dir = os.path.abspath(feat_dir)
        if os.path.isfile(os.path.join(feat_dir, "features.npy")):
            self.currentEastFeatureDir = feat_dir
            self._update_east_runtime_meta(
                feat_dir,
                runtime_origin="bootstrapped",
                bootstrap_source=bootstrap_source,
                bootstrap_pending=False,
                bootstrap_completed=False,
                bootstrap_confirmed_count=int(len(rows)),
                bootstrap_initialized_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                bootstrap_refresh_deferred=True,
            )
            if not self._write_correction_record_buffer(rows, force=True):
                return False
            self._update_east_runtime_meta(
                feat_dir,
                runtime_origin="bootstrapped",
                bootstrap_source=bootstrap_source,
                bootstrap_pending=False,
                bootstrap_completed=True,
                bootstrap_confirmed_count=int(len(rows)),
                bootstrap_refresh_deferred=True,
            )
            self._east_skip_next_masked_refresh = True
            self._set_status(
                "Initialized EAST runtime from confirmed supervision. Starting online update..."
            )
            self._schedule_east_online_update(feat_dir)
            return True
        if not self._ensure_feature_extractor_available(show_dialog=True):
            return False
        if not self._ensure_east_model_paths():
            return False
        self._update_east_runtime_meta(
            feat_dir,
            runtime_origin="bootstrapped",
            bootstrap_source=bootstrap_source,
            bootstrap_pending=True,
            bootstrap_completed=False,
            bootstrap_confirmed_count=int(len(rows)),
            bootstrap_initialized_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            bootstrap_refresh_deferred=True,
        )
        self._east_skip_next_masked_refresh = True
        self._set_status(
            "Building EAST backbone features from confirmed supervision in background..."
        )
        self._start_feature_extraction(feat_dir, next_task="east_bootstrap", backbone="east_backbone")
        return True

    def _schedule_east_online_update(self, features_dir: Optional[str] = None) -> None:
        path = str(features_dir or self._correction_features_dir() or "").strip()
        if not path:
            return
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            return
        self._east_online_update_pending = True
        self._east_online_update_pending_dir = path
        if self._east_online_update_running:
            return
        try:
            self._east_online_update_timer.start()
        except Exception:
            self._flush_east_online_update()

    def _flush_east_online_update(self) -> None:
        if self._east_online_update_running or not self._east_online_update_pending:
            return
        features_dir = str(self._east_online_update_pending_dir or "").strip()
        self._east_online_update_pending = False
        self._east_online_update_pending_dir = ""
        if not features_dir:
            return
        self._start_east_online_update(features_dir)

    def _start_east_online_update(self, features_dir: str) -> None:
        path = os.path.abspath(os.path.expanduser(str(features_dir or "")))
        if not path or not os.path.isdir(path):
            return
        if self._east_online_update_running:
            self._east_online_update_pending = True
            self._east_online_update_pending_dir = path
            return
        self._east_online_update_running = True
        self._east_online_update_thread = QThread(self)
        self._east_online_update_worker = EASTOnlineUpdateWorker(
            features_dir=path,
            trim_ratio=self._segment_embedding_cfg(),
        )
        self._east_online_update_worker.moveToThread(self._east_online_update_thread)
        self._east_online_update_thread.started.connect(self._east_online_update_worker.run)
        self._east_online_update_worker.progress.connect(self._set_status)
        self._east_online_update_worker.done.connect(self._on_east_online_update_done)
        self._east_online_update_worker.done.connect(self._east_online_update_thread.quit)
        self._east_online_update_thread.finished.connect(
            self._east_online_update_worker.deleteLater
        )
        self._east_online_update_thread.finished.connect(
            self._east_online_update_thread.deleteLater
        )
        self._east_online_update_thread.start()

    def _on_east_online_update_done(self, stats: Any) -> None:
        self._east_online_update_running = False
        if self._east_online_update_timer.isActive():
            try:
                self._east_online_update_timer.stop()
            except Exception:
                pass
        data = stats if isinstance(stats, dict) else {}
        if bool(data.get("ok", False)):
            if bool(data.get("changed", False)):
                record_count = int(data.get("record_count", 0) or 0)
                proto_count = int(data.get("prototype_count", 0) or 0)
                transition_count = int(data.get("transition_delta_count", 0) or 0)
                if record_count > 0:
                    self._set_status(
                        "EAST online update ready in background "
                        f"({record_count} confirmed corrections, {proto_count} prototypes, {transition_count} transition pairs)."
                    )
                else:
                    self._set_status(
                        "EAST online update cleared stale correction state for the current runtime."
                    )
        else:
            err = str(data.get("error", "") or "").strip()
            if err:
                self._set_status(f"EAST online update skipped: {err}")
        self._east_online_update_worker = None
        self._east_online_update_thread = None
        if bool(data.get("ok", False)):
            self._schedule_east_online_adapter_update(
                str(data.get("features_dir") or self._correction_features_dir() or "")
            )
        if self._east_online_update_pending and self._east_online_update_pending_dir:
            self._east_online_update_timer.start()

    def _schedule_east_online_adapter_update(self, features_dir: Optional[str] = None) -> None:
        path = str(features_dir or self._correction_features_dir() or "").strip()
        if not path:
            return
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            return
        self._east_online_adapter_pending = True
        self._east_online_adapter_pending_dir = path
        if self._east_online_adapter_running:
            return
        try:
            self._east_online_adapter_timer.start()
        except Exception:
            self._flush_east_online_adapter_update()

    def _flush_east_online_adapter_update(self) -> None:
        if self._east_online_adapter_running or not self._east_online_adapter_pending:
            return
        features_dir = str(self._east_online_adapter_pending_dir or "").strip()
        self._east_online_adapter_pending = False
        self._east_online_adapter_pending_dir = ""
        if not features_dir:
            return
        self._start_east_online_adapter_update(features_dir)

    def _start_east_online_adapter_update(self, features_dir: str) -> None:
        path = os.path.abspath(os.path.expanduser(str(features_dir or "")))
        if not path or not os.path.isdir(path):
            return
        if self._east_online_adapter_running:
            self._east_online_adapter_pending = True
            self._east_online_adapter_pending_dir = path
            return
        self._east_online_adapter_running = True
        self._east_online_adapter_thread = QThread(self)
        self._east_online_adapter_worker = EASTOnlineAdapterTrainWorker(path)
        self._east_online_adapter_worker.moveToThread(self._east_online_adapter_thread)
        self._east_online_adapter_thread.started.connect(self._east_online_adapter_worker.run)
        self._east_online_adapter_worker.progress.connect(self._set_status)
        self._east_online_adapter_worker.done.connect(self._on_east_online_adapter_done)
        self._east_online_adapter_worker.done.connect(self._east_online_adapter_thread.quit)
        self._east_online_adapter_thread.finished.connect(
            self._east_online_adapter_worker.deleteLater
        )
        self._east_online_adapter_thread.finished.connect(
            self._east_online_adapter_thread.deleteLater
        )
        self._east_online_adapter_thread.start()

    def _schedule_east_masked_refresh(self, features_dir: Optional[str] = None) -> None:
        path = str(features_dir or self._correction_features_dir() or "").strip()
        if not path:
            return
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            return
        self._east_masked_refresh_pending = True
        self._east_masked_refresh_pending_dir = path
        if self._east_masked_refresh_running:
            return
        try:
            self._east_masked_refresh_timer.start()
        except Exception:
            self._flush_east_masked_refresh()

    def _flush_east_masked_refresh(self) -> None:
        if self._east_masked_refresh_running or not self._east_masked_refresh_pending:
            return
        features_dir = str(self._east_masked_refresh_pending_dir or "").strip()
        self._east_masked_refresh_pending = False
        self._east_masked_refresh_pending_dir = ""
        if not features_dir:
            return
        ok, reason = self._masked_refresh_safe_state(features_dir)
        if not ok:
            if reason in {
                "interaction_busy",
                "east_refine_running",
                "online_update_running",
            }:
                self._east_masked_refresh_pending = True
                self._east_masked_refresh_pending_dir = features_dir
                self._east_masked_refresh_timer.start()
            return
        self._start_east_masked_refresh(features_dir)

    def _start_east_masked_refresh(self, features_dir: str) -> None:
        path = os.path.abspath(os.path.expanduser(str(features_dir or "")))
        if not path or not os.path.isdir(path):
            return
        ok, _reason = self._masked_refresh_safe_state(path)
        if not ok:
            return
        if not self._ensure_east_model_paths():
            return
        windows = self._masked_refresh_windows(path)
        if not windows:
            return
        label_txt = self._prepare_east_labels_file(path)
        if not label_txt:
            return
        meta = self._load_feature_meta(path)
        feature_backbone_version = str(
            meta.get("backbone")
            or meta.get("source")
            or meta.get("feature_backbone_version")
            or "east_backbone"
        ).strip()
        video_id = (
            str(getattr(self, "current_video_id", "") or "").strip()
            or os.path.splitext(os.path.basename(str(getattr(self, "video_path", "") or "")))[0]
            or os.path.basename(os.path.abspath(path))
        )
        log_path = os.path.join(path, "pred_east_masked_refresh.log")
        self._east_masked_refresh_running = True
        self._east_masked_refresh_thread = QThread(self)
        self._east_masked_refresh_worker = EASTMaskedRefreshWorker(
            features_dir=path,
            video_id=video_id,
            windows=windows,
            video_path=str(getattr(self, "video_path", "") or ""),
            labels_txt=label_txt,
            ckpt=self._east_ckpt,
            cfg=self._east_cfg,
            text_bank_version=self._east_text_bank_version,
            feature_backbone_version=feature_backbone_version,
            category_adapter=self._effective_east_shared_adapter_path(),
            out_prefix="pred_east_masked_refresh",
            log_path=log_path,
        )
        self._east_masked_refresh_worker.moveToThread(self._east_masked_refresh_thread)
        self._east_masked_refresh_thread.started.connect(self._east_masked_refresh_worker.run)
        self._east_masked_refresh_worker.done.connect(self._on_east_masked_refresh_done)
        self._east_masked_refresh_worker.done.connect(self._east_masked_refresh_thread.quit)
        self._east_masked_refresh_thread.finished.connect(
            self._east_masked_refresh_worker.deleteLater
        )
        self._east_masked_refresh_thread.finished.connect(
            self._east_masked_refresh_thread.deleteLater
        )
        self._east_masked_refresh_thread.start()

    def _on_east_masked_refresh_done(self, payload: Any) -> None:
        self._east_masked_refresh_running = False
        if self._east_masked_refresh_timer.isActive():
            try:
                self._east_masked_refresh_timer.stop()
            except Exception:
                pass
        data = payload if isinstance(payload, dict) else {}
        try:
            applied = self._apply_east_predictions_masked_local(data)
        except Exception:
            applied = False
        if applied:
            applied_count = int(data.get("applied_window_count", 0) or 0)
            partial = bool(data.get("partial", False))
            msg = "Updated unlocked regions with the latest EAST refinement."
            if applied_count > 0:
                msg = (
                    f"Updated unlocked regions with the latest EAST refinement "
                    f"({applied_count} local window{'s' if applied_count != 1 else ''})."
                )
            if partial:
                msg = msg[:-1] + " Partial refresh only."
            self._set_status(msg)
        else:
            err = str(data.get("error", "") or "").strip()
            if err:
                self._set_status(f"EAST local refresh skipped: {err}")
        self._east_masked_refresh_worker = None
        self._east_masked_refresh_thread = None
        if self._east_masked_refresh_pending and self._east_masked_refresh_pending_dir:
            self._east_masked_refresh_timer.start()

    def _on_east_online_adapter_done(self, stats: Any) -> None:
        self._east_online_adapter_running = False
        if self._east_online_adapter_timer.isActive():
            try:
                self._east_online_adapter_timer.stop()
            except Exception:
                pass
        skip_refresh = bool(getattr(self, "_east_skip_next_masked_refresh", False))
        self._east_skip_next_masked_refresh = False
        data = stats if isinstance(stats, dict) else {}
        if bool(data.get("ok", False)):
            if bool(data.get("changed", False)):
                span_count = int(data.get("positive_span_count", 0) or 0)
                boundary_count = int(data.get("boundary_target_count", 0) or 0)
                class_count = int(data.get("class_count", 0) or 0)
                self._set_status(
                    "EAST online adapter ready in background "
                    f"({span_count} positive spans, {boundary_count} boundary targets, {class_count} classes)."
                )
                if not skip_refresh:
                    self._schedule_east_masked_refresh(
                        str(data.get("features_dir") or self._correction_features_dir() or "")
                    )
        else:
            err = str(data.get("error", "") or "").strip()
            if err:
                self._set_status(f"EAST online adapter skipped: {err}")
        self._east_online_adapter_worker = None
        self._east_online_adapter_thread = None
        if self._east_online_adapter_pending and self._east_online_adapter_pending_dir:
            self._east_online_adapter_timer.start()

    def _restore_east_ui_state(self) -> None:
        try:
            state = load_east_ui_state()
        except Exception:
            state = {}
        ckpt = os.path.abspath(
            os.path.expanduser(str(state.get("east_ckpt") or "").strip())
        )
        cfg = os.path.abspath(
            os.path.expanduser(str(state.get("east_cfg") or "").strip())
        )
        backend = str(state.get("text_bank_backend") or "").strip().lower()
        try:
            refresh_context = int(state.get("masked_refresh_context_frames", 0) or 0)
        except Exception:
            refresh_context = 0
        if ckpt and os.path.isfile(ckpt):
            self._east_ckpt = ckpt
        if cfg and os.path.isfile(cfg):
            self._east_cfg = cfg
        if backend:
            self._east_text_bank_version = backend
        self._east_masked_refresh_context_override = max(0, int(refresh_context))
        path = os.path.abspath(
            os.path.expanduser(str(state.get("shared_adapter_path") or "").strip())
        )
        if path and os.path.isfile(path):
            info = inspect_east_shared_adapter_bundle(path)
            if bool(info.get("ok", False)):
                self._east_category_adapter_path = path
                return
        self._east_category_adapter_path = ""
        try:
            self._persist_east_ui_state()
        except Exception:
            pass

    def _persist_east_ui_state(self) -> None:
        path = self._effective_east_shared_adapter_path()
        ok, path_or_err = save_east_ui_state(
            {
                "shared_adapter_path": path,
                "east_ckpt": str(self._east_ckpt or "").strip(),
                "east_cfg": str(self._east_cfg or "").strip(),
                "text_bank_backend": str(self._east_text_bank_version or "").strip(),
                "masked_refresh_context_frames": int(
                    getattr(self, "_east_masked_refresh_context_override", 0) or 0
                ),
            }
        )
        if not ok:
            print(f"[EAST][WARN] Failed to save UI state: {path_or_err}")

    def _effective_east_shared_adapter_path(self) -> str:
        path = os.path.abspath(
            os.path.expanduser(str(self._east_category_adapter_path or "").strip())
        )
        if path and os.path.isfile(path):
            return path
        return ""

    def _selected_east_shared_adapter_info(self) -> Dict[str, Any]:
        shared_path = self._effective_east_shared_adapter_path()
        if not shared_path:
            return {}
        try:
            info = inspect_east_shared_adapter_bundle(shared_path)
        except Exception:
            info = {}
        return info if bool(info.get("ok", False)) else {}

    def _east_refine_launch_summary(self) -> str:
        shared_info = self._selected_east_shared_adapter_info()
        if not shared_info:
            return "Starting EAST refinement."
        name = str(shared_info.get("display_name") or os.path.basename(str(self._east_category_adapter_path or "").strip()) or "shared adapter")
        return f"Starting EAST refinement (shared adapter: {name})."

    def _current_query_policy_snapshot(self) -> Dict[str, Any]:
        cfg = self._assisted_cfg()
        points = list(getattr(self, "_assisted_points", []) or [])
        label_scores: List[float] = []
        boundary_scores: List[float] = []
        buckets: Dict[str, int] = {}
        for pt in points:
            if not isinstance(pt, dict):
                continue
            try:
                score = float(pt.get("query_score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            ptype = str(pt.get("type", "") or "").strip().lower()
            if ptype == "label":
                label_scores.append(score)
            elif ptype == "boundary":
                boundary_scores.append(score)
            bucket = int(max(0, min(10, round(score * 10.0))))
            buckets[str(bucket)] = int(buckets.get(str(bucket), 0) + 1)

        def _avg(rows: List[float]) -> float:
            if not rows:
                return 0.0
            return float(sum(rows) / max(1, len(rows)))

        return {
            "sort_mode": str(self._assisted_sort_mode() or "query_score"),
            "label_query_min": float(self._assisted_query_threshold("label")),
            "boundary_query_min": float(self._assisted_query_threshold("boundary")),
            "queue_count": int(len(points)),
            "label_point_count": int(len(label_scores)),
            "boundary_point_count": int(len(boundary_scores)),
            "label_query_avg": _avg(label_scores),
            "boundary_query_avg": _avg(boundary_scores),
            "label_query_max": float(max(label_scores) if label_scores else 0.0),
            "boundary_query_max": float(max(boundary_scores) if boundary_scores else 0.0),
            "priority_buckets": dict(sorted(buckets.items(), key=lambda row: int(row[0]), reverse=True)),
            "confusion_weight": float(cfg.get("query_confusion_weight", 0.0) or 0.0),
            "disagreement_weight": float(cfg.get("query_disagreement_weight", 0.0) or 0.0),
            "boundary_uncertainty_weight": float(
                cfg.get("boundary_query_uncertainty_weight", 0.0) or 0.0
            ),
            "boundary_energy_weight": float(
                cfg.get("boundary_query_energy_weight", 0.0) or 0.0
            ),
        }

    def _masked_refresh_context_frames(self) -> int:
        raw = int(
            getattr(self, "_east_masked_refresh_context_override", 0) or 0
        )
        if raw <= 0:
            raw = (
                self.algo.get("east_masked_refresh_context_frames")
                if isinstance(getattr(self, "algo", None), dict)
                else None
            )
        if raw in (None, ""):
            raw = os.environ.get("EAST_MASKED_REFRESH_CONTEXT_FRAMES", "48")
        try:
            value = int(raw)
        except Exception:
            value = 48
        return max(8, min(512, int(value)))

    def _feature_sequence_length(self, features_dir: str) -> int:
        path = os.path.join(str(features_dir or ""), "features.npy")
        if not os.path.isfile(path):
            return 0
        try:
            arr = np.load(path, mmap_mode="r")
            if getattr(arr, "ndim", 0) != 2:
                return 0
            return int(arr.shape[0])
        except Exception:
            return 0

    def _feature_index_range_for_frame_span(
        self,
        frame_map: List[int],
        frame_start: int,
        frame_end: int,
    ) -> Optional[Tuple[int, int]]:
        if not frame_map:
            return None
        start = int(frame_start)
        end = int(frame_end)
        if end < start:
            start, end = end, start
        left = bisect_left(frame_map, start)
        right = bisect_right(frame_map, end) - 1
        if 0 <= left <= right < len(frame_map):
            return int(left), int(right)
        if left >= len(frame_map):
            left = len(frame_map) - 1
        if right < 0:
            right = 0
        left = max(0, min(int(left), len(frame_map) - 1))
        right = max(0, min(int(right), len(frame_map) - 1))
        if right < left:
            right = left
        return int(left), int(right)

    def _masked_refresh_windows(self, features_dir: str) -> List[Dict[str, Any]]:
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return []
        view = self.views[self.active_view_idx]
        view_start = int(view.get("start", 0))
        view_end = int(view.get("end", view_start))
        if view_end < view_start:
            view_start, view_end = view_end, view_start

        locked_segments = self._locked_segments_for_active_view()
        locked_spans = self._merge_spans(
            [(int(seg.get("start", 0)), int(seg.get("end", 0))) for seg in locked_segments]
        )
        protected_boundaries: List[int] = []
        for rec in self._rebuild_confirmed_correction_records_for_active_view():
            if str(rec.get("point_type", "") or "").strip().lower() != "boundary":
                continue
            action_kind = str(rec.get("action_kind", "") or "").strip().lower()
            if action_kind not in {"boundary_add", "boundary_move"}:
                continue
            try:
                frame = int(rec.get("boundary_frame", rec.get("feedback_start", 0)))
            except Exception:
                continue
            if int(view_start) < int(frame) <= int(view_end):
                protected_boundaries.append(int(frame))
        protected_boundaries = sorted(set(protected_boundaries))
        unlocked_spans: List[Tuple[int, int]] = []
        cursor = int(view_start)
        for lock_s, lock_e in locked_spans:
            lock_s = max(int(view_start), int(lock_s))
            lock_e = min(int(view_end), int(lock_e))
            if cursor < lock_s:
                unlocked_spans.append((int(cursor), int(lock_s - 1)))
            cursor = max(int(cursor), int(lock_e + 1))
        if cursor <= int(view_end):
            unlocked_spans.append((int(cursor), int(view_end)))
        if protected_boundaries and unlocked_spans:
            split_spans: List[Tuple[int, int]] = []
            for span_s, span_e in unlocked_spans:
                seg_cursor = int(span_s)
                for frame in protected_boundaries:
                    if not (int(span_s) < int(frame) <= int(span_e)):
                        continue
                    if seg_cursor <= int(frame) - 1:
                        split_spans.append((int(seg_cursor), int(frame) - 1))
                    seg_cursor = max(int(seg_cursor), int(frame))
                if seg_cursor <= int(span_e):
                    split_spans.append((int(seg_cursor), int(span_e)))
            unlocked_spans = split_spans
        if not unlocked_spans:
            return []

        seq_len = self._feature_sequence_length(features_dir)
        if seq_len <= 0:
            return []
        frame_map = self._feature_frame_map(features_dir, seq_len)
        if not frame_map:
            return []

        ctx = self._masked_refresh_context_frames()
        planned: List[Dict[str, Any]] = []
        for span_s, span_e in unlocked_spans:
            win_s = max(int(view_start), int(span_s) - int(ctx))
            win_e = min(int(view_end), int(span_e) + int(ctx))
            feat_span = self._feature_index_range_for_frame_span(frame_map, win_s, win_e)
            if feat_span is None:
                continue
            planned.append(
                {
                    "feature_start": int(feat_span[0]),
                    "feature_end": int(feat_span[1]),
                    "focus_spans": [(int(span_s), int(span_e))],
                }
            )
        if not planned:
            return []

        merged: List[Dict[str, Any]] = []
        for item in sorted(planned, key=lambda row: (int(row["feature_start"]), int(row["feature_end"]))):
            if not merged:
                merged.append(
                    {
                        "feature_start": int(item["feature_start"]),
                        "feature_end": int(item["feature_end"]),
                        "focus_spans": list(item.get("focus_spans") or []),
                    }
                )
                continue
            prev = merged[-1]
            if int(item["feature_start"]) <= int(prev["feature_end"]) + 1:
                prev["feature_end"] = max(int(prev["feature_end"]), int(item["feature_end"]))
                prev["focus_spans"] = list(prev.get("focus_spans") or []) + list(item.get("focus_spans") or [])
            else:
                merged.append(
                    {
                        "feature_start": int(item["feature_start"]),
                        "feature_end": int(item["feature_end"]),
                        "focus_spans": list(item.get("focus_spans") or []),
                    }
                )

        windows: List[Dict[str, Any]] = []
        for item in merged:
            fs = int(item.get("feature_start", 0))
            fe = int(item.get("feature_end", fs))
            if fe < fs or fs < 0 or fe >= len(frame_map):
                continue
            focus_spans = self._merge_spans(list(item.get("focus_spans") or []))
            if not focus_spans:
                continue
            windows.append(
                {
                    "feature_start": int(fs),
                    "feature_end": int(fe),
                    "focus_spans": [(int(a), int(b)) for a, b in focus_spans],
                    "frame_indices": [int(x) for x in frame_map[fs : fe + 1]],
                }
            )
        return windows

    def _east_assets_features_dir(self) -> Optional[str]:
        return self._correction_features_dir()

    def _current_east_runtime_report(self) -> Optional[Dict[str, Any]]:
        features_dir = self._east_assets_features_dir()
        if not features_dir:
            QMessageBox.information(
                self,
                "EAST Runtime",
                "No EAST feature directory is available for the current video.",
            )
            return None
        report = inspect_east_runtime_assets(features_dir)
        if not bool(report.get("ok", False)):
            reason = str(report.get("reason", "runtime_unavailable") or "runtime_unavailable")
            QMessageBox.information(
                self,
                "EAST Runtime",
                f"EAST runtime assets are not available yet ({reason}).",
            )
            return None
        merged = dict(report)
        merged.update(self._east_runtime_debug_context(report, features_dir=features_dir))
        return merged

    def _east_runtime_debug_report(self) -> Dict[str, Any]:
        features_dir = self._east_assets_features_dir()
        if features_dir:
            report = inspect_east_runtime_assets(features_dir)
        else:
            fallback_dir = str(self._default_east_features_dir_for_video() or "").strip()
            report = {
                "ok": False,
                "reason": "features_dir_missing",
                "features_dir": os.path.abspath(fallback_dir) if fallback_dir else "",
                "runtime_dir": (
                    os.path.join(os.path.abspath(fallback_dir), "east_runtime")
                    if fallback_dir
                    else ""
                ),
            }
        merged = dict(report or {})
        merged.update(self._east_runtime_debug_context(merged, features_dir=features_dir))
        return merged

    def _inspect_east_runtime_assets(self) -> None:
        report = self._east_runtime_debug_report()
        shared_info = self._selected_east_shared_adapter_info()
        query_info = self._current_query_policy_snapshot()
        dlg = QDialog(self)
        dlg.setWindowTitle("EAST Runtime Debug")
        dlg.setMinimumSize(720, 560)
        dlg.resize(860, 680)
        outer = QVBoxLayout(dlg)
        tabs = QTabWidget(dlg)
        outer.addWidget(tabs, 1)

        summary_scroll = QScrollArea(dlg)
        summary_scroll.setWidgetResizable(True)
        summary_scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget(summary_scroll)
        root = QVBoxLayout(content)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        summary_scroll.setWidget(content)
        tabs.addTab(summary_scroll, "Summary")

        def _add_section(title: str, rows: List[tuple]) -> None:
            box = QGroupBox(title, content)
            form = QFormLayout(box)
            form.setContentsMargins(10, 10, 10, 10)
            form.setSpacing(8)
            for label, value in rows:
                lbl = QLabel(str(value), box)
                lbl.setWordWrap(True)
                lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
                form.addRow(str(label), lbl)
            root.addWidget(box)

        _add_section(
            "Runtime",
            [
                ("Runtime status", report.get("runtime_source_label") or "-"),
                ("Runtime ready", "Yes" if report.get("runtime_available") else "No"),
                ("Baseline source", report.get("baseline_source") or "-"),
                ("Video ID", report.get("video_id") or "-"),
                ("Features dir", report.get("features_dir") or "-"),
                ("Runtime dir", report.get("runtime_dir") or "-"),
                ("Segments", int(report.get("segment_count", 0) or 0)),
                ("Classes", int(report.get("class_count", 0) or 0)),
                ("Confirmed corrections", int(report.get("record_count", 0) or 0)),
                ("Runtime dirty", "Yes" if report.get("runtime_dirty") else "No"),
            ],
        )
        _add_section(
            "Bootstrap State",
            [
                ("Features ready", "Yes" if report.get("features_ready") else "No"),
                ("Bootstrap source", report.get("bootstrap_source") or "-"),
                ("Bootstrap pending", "Yes" if report.get("bootstrap_pending") else "No"),
                (
                    "Bootstrap completed",
                    "Yes" if report.get("bootstrap_completed") else "No",
                ),
                (
                    "Bootstrap confirmed records",
                    int(report.get("bootstrap_confirmed_count", 0) or 0),
                ),
                (
                    "Current confirmed records",
                    int(report.get("current_confirmed_record_count", 0) or 0),
                ),
                (
                    "Bootstrap initialized at",
                    report.get("bootstrap_initialized_at") or "-",
                ),
                (
                    "Refresh deferred once",
                    "Yes" if report.get("bootstrap_refresh_deferred") else "No",
                ),
                ("Status reason", report.get("runtime_status_reason") or "-"),
            ],
        )
        _add_section(
            "Learning State",
            [
                ("Locked segments", int(report.get("locked_segment_count", 0) or 0)),
                (
                    "Label text bank",
                    (
                        "Yes"
                        if report.get("has_label_text_bank")
                        else "No"
                    )
                    + (
                        f" ({str(report.get('label_text_bank_backend') or '').strip()})"
                        if str(report.get("label_text_bank_backend") or "").strip()
                        else ""
                    ),
                ),
                (
                    "Online adapter",
                    f"{'Yes' if report.get('has_online_adapter') else 'No'} "
                    f"(step {int(report.get('online_step', 0) or 0)})",
                ),
                (
                    "Label spans",
                    f"+{int(report.get('online_positive_span_count', 0) or 0)} / "
                    f"-{int(report.get('online_negative_span_count', 0) or 0)}",
                ),
                (
                    "Ranking pairs",
                    int(report.get("online_ranking_pair_count", 0) or 0),
                ),
                (
                    "Boundary targets",
                    int(report.get("online_boundary_supervision_count", 0) or 0),
                ),
                (
                    "Model delta",
                    f"{'Yes' if report.get('has_model_delta') else 'No'} "
                    f"(step {int(report.get('model_delta_step', 0) or 0)})",
                ),
                ("Prototypes", int(report.get("prototype_count", 0) or 0)),
                ("Transition pairs", int(report.get("transition_delta_pairs", 0) or 0)),
                ("Confusion pairs", int(report.get("confusion_delta_pairs", 0) or 0)),
            ],
        )
        _add_section(
            "Query Policy",
            [
                ("Sort mode", query_info.get("sort_mode") or "query_score"),
                (
                    "Thresholds",
                    f"Boundary >= {float(query_info.get('boundary_query_min', 0.0) or 0.0):.2f}, "
                    f"Label >= {float(query_info.get('label_query_min', 0.0) or 0.0):.2f}",
                ),
                ("Queued review points", int(query_info.get("queue_count", 0) or 0)),
                ("Boundary points", int(query_info.get("boundary_point_count", 0) or 0)),
                ("Label points", int(query_info.get("label_point_count", 0) or 0)),
                (
                    "Avg query score",
                    f"Boundary {float(query_info.get('boundary_query_avg', 0.0) or 0.0):.2f}, "
                    f"Label {float(query_info.get('label_query_avg', 0.0) or 0.0):.2f}",
                ),
                (
                    "Max query score",
                    f"Boundary {float(query_info.get('boundary_query_max', 0.0) or 0.0):.2f}, "
                    f"Label {float(query_info.get('label_query_max', 0.0) or 0.0):.2f}",
                ),
                (
                    "Priority buckets",
                    ", ".join(
                        f"{key}:{int(val)}"
                        for key, val in (query_info.get("priority_buckets") or {}).items()
                    )
                    or "-",
                ),
            ],
        )
        if shared_info:
            _add_section(
                "Selected Shared Adapter",
                [
                    ("Name", shared_info.get("display_name") or "-"),
                    ("Members", int(shared_info.get("member_count", 0) or 0)),
                    (
                        "Label text bank",
                        (
                            "Yes"
                            if shared_info.get("has_label_text_bank")
                            else "No"
                        )
                        + (
                            f" ({str(shared_info.get('label_text_bank_backend') or '').strip()})"
                            if str(shared_info.get("label_text_bank_backend") or "").strip()
                            else ""
                        ),
                    ),
                    ("Online adapter", "Yes" if shared_info.get("has_online_adapter") else "No"),
                    ("Online step", int(shared_info.get("online_step", 0) or 0)),
                    ("Ranking pairs", int(shared_info.get("online_ranking_pair_count", 0) or 0)),
                    ("Prototypes", int(shared_info.get("prototype_count", 0) or 0)),
                    ("Transition pairs", int(shared_info.get("transition_delta_pairs", 0) or 0)),
                    ("Confusion pairs", int(shared_info.get("confusion_delta_pairs", 0) or 0)),
                ],
            )

        action_counts = dict(report.get("action_counts") or {})
        if action_counts:
            box_actions = QGroupBox("Confirmed Action Counts", content)
            form_actions = QFormLayout(box_actions)
            form_actions.setContentsMargins(10, 10, 10, 10)
            for key in sorted(action_counts.keys()):
                form_actions.addRow(str(key), QLabel(str(int(action_counts.get(key, 0) or 0)), box_actions))
            root.addWidget(box_actions)

        root.addStretch(1)

        raw_page = QWidget(dlg)
        raw_layout = QVBoxLayout(raw_page)
        raw_layout.setContentsMargins(0, 0, 0, 0)
        raw_text = QPlainTextEdit(raw_page)
        raw_text.setReadOnly(True)
        raw_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        detail_payload = {
            "runtime": report,
            "selected_shared_adapter": shared_info,
            "query_policy": query_info,
            "advanced": {
                "runtime_files": {
                    "has_boundary_score": bool(report.get("has_boundary_score")),
                    "has_seg_embeddings": bool(report.get("has_seg_embeddings")),
                    "has_label_scores": bool(report.get("has_label_scores")),
                    "has_transition_matrix": bool(report.get("has_transition_matrix")),
                    "has_runtime_prototype": bool(report.get("has_runtime_prototype")),
                    "text_table_shape": report.get("text_table_shape") or (0, 0),
                },
                "shared_adapter_path": str(self._east_category_adapter_path or ""),
                "action_counts": action_counts,
            },
        }
        try:
            raw_text.setPlainText(
                json.dumps(
                    detail_payload,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )
        except Exception:
            raw_text.setPlainText(str(detail_payload))
        raw_layout.addWidget(raw_text)
        tabs.addTab(raw_page, "Raw JSON")

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=dlg)
        btn_export_report = buttons.addButton("Export Runtime Report...", QDialogButtonBox.ActionRole)
        btn_export_shared = buttons.addButton("Export Shared Adapter...", QDialogButtonBox.ActionRole)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        btn_export_report.clicked.connect(self._export_east_runtime_report)
        btn_export_shared.clicked.connect(self._export_current_east_shared_adapter)
        outer.addWidget(buttons)
        dlg.exec_()

    def _export_east_runtime_report(self) -> None:
        report = self._current_east_runtime_report()
        if not report:
            return
        default_name = (
            str(report.get("video_id") or "east_runtime") + ".runtime_report.json"
        )
        fp, _ = QFileDialog.getSaveFileName(
            self,
            "Export EAST Runtime Report",
            os.path.join(os.getcwd(), default_name),
            "JSON Files (*.json)",
        )
        if not fp:
            return
        export_east_runtime_report(str(report.get("features_dir") or ""), fp)
        shared_path = self._effective_east_shared_adapter_path()
        query_info = self._current_query_policy_snapshot()
        if shared_path:
            try:
                shared_info = inspect_east_shared_adapter_bundle(shared_path)
            except Exception:
                shared_info = {}
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    payload["selected_shared_adapter_path"] = shared_path
                    payload["selected_shared_adapter"] = shared_info
                    payload["query_policy"] = query_info
                    with open(fp, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            except Exception:
                pass
        elif query_info:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    payload["query_policy"] = query_info
                    with open(fp, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            except Exception:
                pass
        self._set_status(f"Exported EAST runtime report: {os.path.basename(fp)}")

    def _export_current_east_shared_adapter(self) -> None:
        report = self._current_east_runtime_report()
        if not report:
            return
        if not bool(report.get("has_online_adapter")) and not bool(report.get("has_model_delta")):
            QMessageBox.information(
                self,
                "EAST Shared Adapter",
                "The current runtime does not contain an online adapter or model delta yet.",
            )
            return
        default_name = str(report.get("video_id") or "east_shared_adapter") + ".pt"
        fp, _ = QFileDialog.getSaveFileName(
            self,
            "Export EAST Shared Adapter",
            os.path.join(os.getcwd(), default_name),
            "PyTorch Files (*.pt *.pth)",
        )
        if not fp:
            return
        category, ok = QInputDialog.getText(
            self,
            "Shared Adapter Category",
            "Category name (optional):",
            text="action_segmentation",
        )
        if not ok:
            return
        label_bank = list(report.get("classes") or [])
        out = export_runtime_as_shared_adapter(
            str(report.get("features_dir") or ""),
            fp,
            category=str(category or "").strip(),
            label_bank=label_bank,
        )
        self._set_status(f"Exported EAST shared adapter: {os.path.basename(out)}")

    def _select_east_shared_adapter(self) -> None:
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Select EAST Shared Adapter",
            "",
            "PyTorch Files (*.pt *.pth)",
        )
        if not fp:
            return
        info = inspect_east_shared_adapter_bundle(fp)
        if not bool(info.get("ok", False)):
            QMessageBox.warning(
                self,
                "EAST Shared Adapter",
                "The selected file is not a valid EAST shared adapter bundle.",
            )
            return
        self._east_category_adapter_path = fp
        self._persist_east_ui_state()
        self._set_status(
            f"Selected EAST shared adapter: {info.get('display_name') or os.path.basename(fp)}"
        )

    def _clear_east_shared_adapter(self) -> None:
        if not str(self._east_category_adapter_path or "").strip():
            self._set_status("No EAST shared adapter is currently selected.")
            return
        self._east_category_adapter_path = ""
        self._persist_east_ui_state()
        self._set_status("Cleared selected EAST shared adapter.")

    def _open_east_setup_dialog(self) -> None:
        self._refresh_east_backend_defaults()
        dlg = QDialog(self)
        dlg.setWindowTitle("EAST Setup")
        dlg.setMinimumWidth(640)
        outer = QVBoxLayout(dlg)

        def _path_row(initial_text: str, browse_title: str, filt: str) -> Tuple[QWidget, QLineEdit, QPushButton]:
            row = QWidget(dlg)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            edit = QLineEdit(row)
            edit.setText(str(initial_text or "").strip())
            btn = QPushButton("Browse...", row)
            btn.clicked.connect(
                lambda: self._browse_east_setup_path(edit, browse_title, filt)
            )
            layout.addWidget(edit, 1)
            layout.addWidget(btn, 0)
            return row, edit, btn

        group_model = QGroupBox("Model Assets", dlg)
        form_model = QFormLayout(group_model)
        row_ckpt, edit_ckpt, _ = _path_row(
            str(self._east_ckpt or "").strip(),
            "Choose EAST checkpoint",
            "PyTorch Models (*.pth *.pt *.ckpt)",
        )
        row_cfg, edit_cfg, _ = _path_row(
            str(self._east_cfg or "").strip(),
            "Choose EAST config",
            "Config Files (*.py *.yaml *.yml)",
        )
        form_model.addRow("Checkpoint", row_ckpt)
        form_model.addRow("Config", row_cfg)
        outer.addWidget(group_model)

        group_alignment = QGroupBox("Online Alignment", dlg)
        form_alignment = QFormLayout(group_alignment)
        combo_backend = QComboBox(group_alignment)
        for title, value in self._east_text_bank_backend_items():
            combo_backend.addItem(title, value)
        idx_backend = combo_backend.findData(
            str(self._east_text_bank_version or "auto").strip().lower() or "auto"
        )
        combo_backend.setCurrentIndex(idx_backend if idx_backend >= 0 else 0)
        row_shared = QWidget(group_alignment)
        row_shared_l = QHBoxLayout(row_shared)
        row_shared_l.setContentsMargins(0, 0, 0, 0)
        row_shared_l.setSpacing(6)
        edit_shared = QLineEdit(row_shared)
        edit_shared.setReadOnly(True)
        edit_shared.setText(str(self._effective_east_shared_adapter_path() or "").strip())
        btn_shared_select = QPushButton("Select...", row_shared)
        btn_shared_clear = QPushButton("Clear", row_shared)

        def _select_shared() -> None:
            fp, _ = QFileDialog.getOpenFileName(
                self,
                "Select EAST Shared Adapter",
                "",
                "PyTorch Files (*.pt *.pth)",
            )
            if not fp:
                return
            info = inspect_east_shared_adapter_bundle(fp)
            if not bool(info.get("ok", False)):
                QMessageBox.warning(
                    dlg,
                    "EAST Shared Adapter",
                    "The selected file is not a valid EAST shared adapter bundle.",
                )
                return
            edit_shared.setText(os.path.abspath(fp))

        btn_shared_select.clicked.connect(_select_shared)
        btn_shared_clear.clicked.connect(lambda: edit_shared.setText(""))
        row_shared_l.addWidget(edit_shared, 1)
        row_shared_l.addWidget(btn_shared_select, 0)
        row_shared_l.addWidget(btn_shared_clear, 0)
        form_alignment.addRow("Label text backend", combo_backend)
        form_alignment.addRow("Shared adapter", row_shared)
        outer.addWidget(group_alignment)

        group_refresh = QGroupBox("Refresh", dlg)
        form_refresh = QFormLayout(group_refresh)
        sp_context = QSpinBox(group_refresh)
        sp_context.setRange(8, 512)
        sp_context.setValue(int(self._masked_refresh_context_frames()))
        form_refresh.addRow("Masked refresh context (frames)", sp_context)
        outer.addWidget(group_refresh)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=dlg,
        )
        btn_reset = buttons.addButton(
            "Reset to defaults",
            QDialogButtonBox.ResetRole,
        )

        def _reset_defaults() -> None:
            try:
                load_feature_env_defaults(repo_root=self._root_dir)
            except Exception:
                pass
            edit_ckpt.setText(str(os.environ.get("EAST_CKPT", "") or "").strip())
            edit_cfg.setText(str(os.environ.get("EAST_CFG", "") or "").strip())
            idx = combo_backend.findData("auto")
            combo_backend.setCurrentIndex(idx if idx >= 0 else 0)
            sp_context.setValue(48)
            edit_shared.setText("")

        btn_reset.clicked.connect(_reset_defaults)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        outer.addWidget(buttons)

        while True:
            if dlg.exec_() != QDialog.Accepted:
                return
            ckpt = os.path.abspath(os.path.expanduser(str(edit_ckpt.text() or "").strip()))
            cfg = os.path.abspath(os.path.expanduser(str(edit_cfg.text() or "").strip()))
            shared = os.path.abspath(os.path.expanduser(str(edit_shared.text() or "").strip()))
            if ckpt and not os.path.isfile(ckpt):
                QMessageBox.warning(dlg, "Invalid checkpoint", "Selected EAST checkpoint does not exist.")
                continue
            if cfg and not os.path.isfile(cfg):
                QMessageBox.warning(dlg, "Invalid config", "Selected EAST config does not exist.")
                continue
            if shared:
                info = inspect_east_shared_adapter_bundle(shared)
                if not bool(info.get("ok", False)):
                    QMessageBox.warning(
                        dlg,
                        "Invalid shared adapter",
                        "Selected shared adapter is not a valid EAST shared adapter bundle.",
                    )
                    continue
            self._east_ckpt = ckpt
            self._east_cfg = cfg
            self._east_text_bank_version = str(
                combo_backend.currentData() or "auto"
            ).strip() or "auto"
            self._east_masked_refresh_context_override = int(sp_context.value())
            self._east_category_adapter_path = shared
            self._persist_east_ui_state()
            self._set_status("Updated EAST setup.")
            return

    def _browse_east_setup_path(self, line_edit: QLineEdit, title: str, filt: str) -> None:
        start_dir = ""
        try:
            existing = os.path.abspath(os.path.expanduser(str(line_edit.text() or "").strip()))
            if existing:
                start_dir = existing if os.path.isdir(existing) else os.path.dirname(existing)
        except Exception:
            start_dir = ""
        path, _ = QFileDialog.getOpenFileName(self, title, start_dir, filt)
        if path:
            line_edit.setText(os.path.abspath(path))

    def _guess_open_session_annotation_path(self, video_path: str) -> str:
        path = os.path.abspath(os.path.expanduser(str(video_path or "").strip()))
        if not path or not os.path.isfile(path):
            return ""
        video_dir = os.path.dirname(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        direct_candidates = [
            os.path.join(video_dir, f"{stem}.json"),
            os.path.join(video_dir, f"{stem}_annotations.json"),
            os.path.join(video_dir, f"{stem}_annotation.json"),
            os.path.join(video_dir, f"{stem}_segments.json"),
            os.path.join(video_dir, f"{stem}_labels.json"),
            os.path.join(video_dir, f"{stem}_coarse.json"),
            os.path.join(video_dir, f"{stem}_fine.json"),
        ]
        for cand in direct_candidates:
            if os.path.isfile(cand):
                return cand
        try:
            matches = []
            for name in sorted(os.listdir(video_dir)):
                low = str(name or "").lower()
                if not low.endswith(".json"):
                    continue
                if any(
                    token in low
                    for token in (
                        "validation",
                        "review",
                        "runtime_report",
                        "state",
                        "components",
                        "rules",
                        "shared_adapter",
                    )
                ):
                    continue
                if stem.lower() in low:
                    matches.append(os.path.join(video_dir, name))
            if matches:
                return matches[0]
        except Exception:
            pass
        return ""

    def _east_setup_summary_text(self) -> str:
        ckpt = str(getattr(self, "_east_ckpt", "") or "").strip()
        cfg = str(getattr(self, "_east_cfg", "") or "").strip()
        shared = str(self._effective_east_shared_adapter_path() or "").strip()
        backend = str(getattr(self, "_east_text_bank_version", "auto") or "auto").strip()
        ckpt_name = os.path.basename(ckpt) if ckpt else "Not set"
        cfg_name = os.path.basename(cfg) if cfg else "Not set"
        shared_name = os.path.basename(shared) if shared else "None"
        return (
            f"Checkpoint: {ckpt_name}\n"
            f"Config: {cfg_name}\n"
            f"Label text backend: {backend}\n"
            f"Shared adapter: {shared_name}"
        )

    def _open_session_wizard(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Open Session")
        dlg.setMinimumWidth(760)
        outer = QVBoxLayout(dlg)

        intro = QLabel(
            "Load a video and optionally import nearby labels and annotations in one flow."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        def _path_row(
            initial_text: str,
            browse_title: str,
            filt: str,
            *,
            on_browse: Optional[Callable[[], None]] = None,
        ) -> Tuple[QWidget, QLineEdit]:
            row = QWidget(dlg)
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            edit = QLineEdit(row)
            edit.setText(str(initial_text or "").strip())
            btn_browse = QPushButton("Browse...", row)

            def _browse() -> None:
                self._browse_east_setup_path(edit, browse_title, filt)
                if on_browse is not None:
                    on_browse()

            btn_browse.clicked.connect(_browse)
            layout.addWidget(edit, 1)
            layout.addWidget(btn_browse, 0)
            return row, edit

        group_assets = QGroupBox("Session Assets", dlg)
        form_assets = QFormLayout(group_assets)
        row_labels, edit_labels = _path_row(
            str(self._current_label_bank_source_path() or "").strip(),
            "Choose label map (TXT)",
            "Text Files (*.txt)",
        )
        row_json, edit_json = _path_row(
            str(getattr(self, "current_annotation_path", "") or "").strip(),
            "Choose annotations JSON",
            "JSON Files (*.json)",
        )
        cb_labels = QCheckBox("Import labels from TXT", group_assets)
        cb_json = QCheckBox("Import annotations JSON", group_assets)
        btn_detect = QPushButton("Autofill Nearby Files", group_assets)

        def _autofill_from_video(force: bool = False) -> None:
            video_fp = os.path.abspath(os.path.expanduser(str(edit_video.text() or "").strip()))
            if not video_fp or not os.path.isfile(video_fp):
                return
            label_fp = resolve_label_source(
                video_path=video_fp,
                repo_root=self._root_dir,
            )
            ann_fp = self._guess_open_session_annotation_path(video_fp)
            if force or not str(edit_labels.text() or "").strip():
                edit_labels.setText(str(label_fp or ""))
            if force or not str(edit_json.text() or "").strip():
                edit_json.setText(str(ann_fp or ""))
            cb_labels.setChecked(bool(str(edit_labels.text() or "").strip()))
            cb_json.setChecked(bool(str(edit_json.text() or "").strip()))

        row_video, edit_video = _path_row(
            str(getattr(self, "video_path", "") or "").strip(),
            "Choose Video",
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
            on_browse=lambda: _autofill_from_video(force=True),
        )
        form_assets.addRow("Video", row_video)
        form_assets.addRow("", btn_detect)
        form_assets.addRow("Labels", row_labels)
        form_assets.addRow("", cb_labels)
        form_assets.addRow("Annotations", row_json)
        form_assets.addRow("", cb_json)
        btn_detect.clicked.connect(lambda: _autofill_from_video(force=True))
        outer.addWidget(group_assets)

        group_model = QGroupBox("Current EAST Setup", dlg)
        model_layout = QVBoxLayout(group_model)
        lbl_model = QLabel(group_model)
        lbl_model.setWordWrap(True)
        lbl_model.setText(self._east_setup_summary_text())
        btn_model = QPushButton("Configure EAST...", group_model)
        btn_model.clicked.connect(self._open_east_setup_dialog)
        btn_model.clicked.connect(lambda: lbl_model.setText(self._east_setup_summary_text()))
        model_layout.addWidget(lbl_model)
        model_layout.addWidget(btn_model, 0, Qt.AlignLeft)
        outer.addWidget(group_model)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=dlg,
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        outer.addWidget(buttons)

        _autofill_from_video(force=False)

        if dlg.exec_() != QDialog.Accepted:
            return

        video_fp = os.path.abspath(os.path.expanduser(str(edit_video.text() or "").strip()))
        label_fp = os.path.abspath(os.path.expanduser(str(edit_labels.text() or "").strip()))
        ann_fp = os.path.abspath(os.path.expanduser(str(edit_json.text() or "").strip()))
        if not video_fp or not os.path.isfile(video_fp):
            QMessageBox.warning(self, "Open Session", "Choose a valid video file.")
            return
        if cb_labels.isChecked() and label_fp and not os.path.isfile(label_fp):
            QMessageBox.warning(self, "Open Session", "The selected label map does not exist.")
            return
        if cb_json.isChecked() and ann_fp and not os.path.isfile(ann_fp):
            QMessageBox.warning(self, "Open Session", "The selected annotation JSON does not exist.")
            return
        if not self._prompt_save_if_dirty(context="opening a new session"):
            return
        if not self._load_primary_video(video_fp):
            return
        if cb_labels.isChecked() and label_fp:
            self._import_label_map_txt(label_fp)
        if cb_json.isChecked() and ann_fp:
            self._load_json_annotations(path=ann_fp)
        self._set_status(f"Opened session: {os.path.basename(video_fp)}")

    def _consolidate_east_shared_adapter(self) -> None:
        root_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Root Folder Containing EAST Runtime Assets",
            os.getcwd(),
        )
        if not root_dir:
            return
        default_name = "east_shared_consolidated.pt"
        fp, _ = QFileDialog.getSaveFileName(
            self,
            "Save Consolidated EAST Shared Adapter",
            os.path.join(os.getcwd(), default_name),
            "PyTorch Files (*.pt *.pth)",
        )
        if not fp:
            return
        category, ok = QInputDialog.getText(
            self,
            "Consolidated Adapter Category",
            "Category name (optional):",
            text="action_segmentation",
        )
        if not ok:
            return
        result = consolidate_east_shared_adapter(
            root_dir,
            fp,
            category=str(category or "").strip(),
        )
        if not bool(result.get("ok", False)):
            QMessageBox.information(
                self,
                "EAST Shared Adapter",
                f"Consolidation skipped: {result.get('reason') or 'unknown error'}.",
            )
            return
        self._east_category_adapter_path = str(result.get("output_path") or fp)
        self._persist_east_ui_state()
        self._set_status(
            "Consolidated EAST shared adapter ready "
            f"({int(result.get('runtime_count', 0) or 0)} runtimes, "
            f"{int(result.get('label_count', 0) or 0)} labels, "
            f"online={'yes' if result.get('has_online_adapter') else 'no'}, "
            f"delta={'yes' if result.get('has_model_delta') else 'no'})."
        )

    def _apply_trim_cut_op(self, op: Dict[str, Any], reverse: bool = False) -> bool:
        if not isinstance(op, dict):
            return False
        if not self.views:
            return False
        try:
            idx = int(op.get("view_idx", -1))
        except Exception:
            idx = -1
        if not (0 <= idx < len(self.views)):
            return False
        descriptor = op.get("descriptor") or {}
        if not self._syncable_descriptor(descriptor):
            return False
        try:
            frame = int(op.get("frame"))
        except Exception:
            return False
        kind = str(op.get("op") or "")
        if reverse:
            kind = "remove" if kind == "add" else ("add" if kind == "remove" else kind)
        view = self.views[idx]
        cuts = self._trim_cut_set_for_view(view, descriptor, create=True)
        if kind == "add":
            cuts.add(frame)
            return True
        if kind == "remove":
            if frame in cuts:
                cuts.discard(frame)
                return True
            return False
        return False

    def _cleanup_trim_cuts_for_spans(
        self,
        view_idx: int,
        descriptor: Optional[Dict[str, Any]],
        spans: List[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        if not self.views or not self._syncable_descriptor(descriptor):
            return []
        if not (0 <= int(view_idx) < len(self.views)):
            return []
        kind = str((descriptor or {}).get("kind") or "")
        if kind not in ("store", "entity", "phase"):
            return []
        merged = self._merge_spans(spans or [])
        if not merged:
            return []
        view = self.views[int(view_idx)]
        st = self._store_for_view_descriptor(view, descriptor or {})
        if st is None:
            return []
        cuts = self._trim_cut_set_for_view(view, descriptor or {}, create=False)
        if not cuts:
            return []
        fc = max(1, self._get_frame_count())
        to_remove: List[int] = []
        for cut in sorted({int(c) for c in cuts if c is not None}):
            if cut <= 0 or cut >= fc:
                continue
            touched = False
            for s, e in merged:
                if int(s) - 1 <= cut <= int(e) + 1:
                    touched = True
                    break
            if not touched:
                continue
            left = st.label_at(int(cut) - 1) if int(cut) - 1 >= 0 else None
            right = st.label_at(int(cut))
            if left == right:
                to_remove.append(int(cut))
        if not to_remove:
            return []
        ops: List[Dict[str, Any]] = []
        for cut in to_remove:
            cuts.discard(int(cut))
            ops.append(
                {
                    "view_idx": int(view_idx),
                    "descriptor": dict(descriptor or {}),
                    "frame": int(cut),
                    "op": "remove",
                }
            )
        return ops

    def _view_has_annotations(self, view: dict) -> bool:
        if not view:
            return False
        st = view.get("store")
        if st and st.frame_to_label:
            return True
        for ent_store in view.get("entity_stores", {}).values():
            if ent_store and ent_store.frame_to_label:
                return True
        return False

    def _target_views_for_annotation_load(self) -> List[dict]:
        if not self.views:
            return []
        targets = [self.views[self.active_view_idx]]
        for i, v in enumerate(self.views):
            if i == self.active_view_idx:
                continue
            if not self._view_has_annotations(v):
                targets.append(v)
        return targets

    def _normalize_sync_edit_selection(self) -> Set[int]:
        if not self.views:
            self._sync_edit_view_indices = set()
            return set()
        valid = {
            int(i)
            for i in (self._sync_edit_view_indices or set())
            if 0 <= int(i) < len(self.views)
        }
        if 0 <= self.active_view_idx < len(self.views):
            valid.add(int(self.active_view_idx))
        self._sync_edit_view_indices = valid
        return set(valid)

    def _effective_sync_edit_indices(self) -> List[int]:
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return []
        selected = self._normalize_sync_edit_selection()
        return sorted(selected)

    def _multiview_sync_active(self) -> bool:
        return len(self._effective_sync_edit_indices()) > 1

    def _playback_sync_indices(self) -> List[int]:
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return []
        selected = self._normalize_sync_edit_selection()
        if len(selected) > 1:
            return sorted(int(i) for i in selected)
        return [int(self.active_view_idx)]

    def _on_view_clicked(self, idx: int, event=None) -> None:
        if not (0 <= idx < len(self.views)):
            return
        mods = 0
        try:
            if event is not None:
                mods = int(event.modifiers())
        except Exception:
            mods = 0
        ctrl_down = bool(mods & int(Qt.ControlModifier))
        if ctrl_down:
            selected = self._normalize_sync_edit_selection()
            # Explicit exit from multi-select: Ctrl+click current active view.
            if idx == self.active_view_idx and len(selected) > 1:
                selected = {int(idx)}
            elif idx in selected and idx != self.active_view_idx:
                selected.discard(idx)
            else:
                selected.add(idx)
            self._sync_edit_view_indices = selected
            self._update_view_highlight()
            self._apply_sync_edit_masks()
            return
        # In multi-select mode, a plain click switches primary view without
        # collapsing the sync selection. This avoids accidental single-view exit.
        if not ctrl_down and self._multiview_sync_active():
            selected_before = set(self._normalize_sync_edit_selection())
            if idx != self.active_view_idx:
                self._set_primary_view(idx)
                if idx in selected_before:
                    self._sync_edit_view_indices = selected_before
                else:
                    self._sync_edit_view_indices = set(selected_before) | {int(idx)}
            else:
                self._sync_edit_view_indices = selected_before
            self._update_view_highlight()
            self._apply_sync_edit_masks()
            return
        self._sync_edit_view_indices = {int(idx)}
        self._set_primary_view(idx)

    def _attach_view_click(self, player: VideoPlayer, idx: int):
        try:
            orig = player.mousePressEvent
        except Exception:
            orig = None

        def handler(e, i=idx, orig_evt=orig):
            self._on_view_clicked(i, e)
            if callable(orig_evt):
                try:
                    orig_evt(e)
                except Exception:
                    pass

        player.mousePressEvent = handler
        try:
            player.on_click_frame = lambda frame, i=idx: self._on_player_clicked(
                i, frame
            )
        except Exception:
            pass

    def _build_splitter_layout(self, total: int) -> QWidget:
        if total <= 0:
            return None
        views = self.views[:total]

        # helper to create splitter with given widgets vertically
        def vert_split(widgets):
            if len(widgets) == 1:
                return widgets[0]
            sp = QSplitter(Qt.Vertical)
            for w in widgets:
                sp.addWidget(w)
            for i in range(len(widgets)):
                sp.setStretchFactor(i, 1)
            return sp

        panels = [self._build_view_panel(v, i) for i, v in enumerate(views)]

        if total == 1:
            return panels[0]
        if total == 2:
            sp = QSplitter(Qt.Horizontal)
            sp.addWidget(panels[0])
            sp.addWidget(panels[1])
            sp.setStretchFactor(0, 1)
            sp.setStretchFactor(1, 1)
            return sp
        if total in (3, 4):
            left = vert_split([panels[0], panels[1]]) if total >= 2 else panels[0]
            right_widgets = []
            if total >= 3:
                right_widgets.append(panels[2])
            if total == 4:
                right_widgets.append(panels[3])
            right = vert_split(right_widgets) if right_widgets else None
            sp = QSplitter(Qt.Horizontal)
            sp.addWidget(left)
            if right:
                sp.addWidget(right)
                sp.setStretchFactor(1, 1)
            sp.setStretchFactor(0, 1)
            return sp
        # total >=5 -> 3 columns: (0,1) left; (2,3) middle; (4) right
        left = vert_split([panels[0], panels[1]])
        mid = vert_split([panels[2], panels[3]])
        right = panels[4]
        sp = QSplitter(Qt.Horizontal)
        sp.addWidget(left)
        sp.addWidget(mid)
        sp.addWidget(right)
        sp.setStretchFactor(0, 1)
        sp.setStretchFactor(1, 1)
        sp.setStretchFactor(2, 1)
        return sp

    def _on_view_panel_dropped(self, src_idx: int, dst_idx: int) -> None:
        self._reorder_view_position(src_idx, dst_idx)

    def _reorder_view_position(self, src_idx: int, dst_idx: int) -> None:
        if not (0 <= src_idx < len(self.views)) or not (0 <= dst_idx < len(self.views)):
            return
        if src_idx == dst_idx:
            return
        self._psr_store_active_view_state()

        active_id = None
        if 0 <= self.active_view_idx < len(self.views):
            active_id = id(self.views[self.active_view_idx])
        selected_ids = {
            id(self.views[i])
            for i in self._normalize_sync_edit_selection()
            if 0 <= i < len(self.views)
        }

        src_name = self._effective_view_name(self.views[src_idx], idx=src_idx)
        dst_name = self._effective_view_name(self.views[dst_idx], idx=dst_idx)

        moving = self.views.pop(src_idx)
        insert_at = int(dst_idx)
        insert_at = max(0, min(insert_at, len(self.views)))
        self.views.insert(insert_at, moving)

        new_active_idx = 0
        new_selected = set()
        for i, vw in enumerate(self.views):
            vid = id(vw)
            if active_id is not None and vid == active_id:
                new_active_idx = i
            if vid in selected_ids:
                new_selected.add(i)
        if not new_selected:
            new_selected = {new_active_idx}

        self._sync_edit_view_indices = set(int(i) for i in new_selected)
        self._rebuild_view_widgets()
        self._set_primary_view(new_active_idx)
        self._set_status(f"Reordered view: {src_name}")
        self._log(
            "view_reorder",
            src=src_name,
            dst=dst_name,
            from_idx=int(src_idx),
            to_idx=int(insert_at),
        )

    def _close_view(self, idx: int):
        if not (0 <= idx < len(self.views)):
            return
        if len(self.views) == 1:
            QMessageBox.information(self, "Info", "Cannot close the last view.")
            return
        self._psr_store_active_view_state()
        vw = self.views[idx]
        view_name = vw.get("name", "")
        if vw.get("dirty"):
            ret = QMessageBox.question(
                self,
                "Unsaved changes",
                "This view has unsaved changes. Close anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return
        player = vw.get("player")
        if player is not None:
            try:
                player.release_media()
            except Exception:
                pass
        self.views.pop(idx)
        if self.active_view_idx >= len(self.views):
            self.active_view_idx = len(self.views) - 1
        self._rebuild_view_widgets()
        self._set_primary_view(self.active_view_idx)
        self._log("close_view", view=view_name, remaining=len(self.views))

    def _on_player_playback_state_changed(self, _playing: bool):
        self._update_play_pause_button()

    def _update_play_pause_button(self):
        btn = getattr(self, "btn_play", None)
        if btn is None:
            return
        playing = False
        if self.views and (0 <= self.active_view_idx < len(self.views)):
            try:
                playing = bool(self.views[self.active_view_idx]["player"].is_playing)
            except Exception:
                playing = False
        btn.setIcon(
            self.style().standardIcon(
                QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay
            )
        )
        btn.setToolTip("Pause" if playing else "Play")

    def _play_all(self):
        if not self.views:
            return
        frame = self.views[self.active_view_idx]["player"].current_frame
        sync_indices = set(self._playback_sync_indices())
        sync_all = len(sync_indices) > 1
        # align and pause all (only when multi-view sync edit is active)
        for i, vw in enumerate(self.views):
            try:
                if sync_all and i in sync_indices:
                    vw["player"].seek(frame, preview_only=False)
                if i != self.active_view_idx:
                    vw["player"].pause()
            except Exception:
                pass
            try:
                vw["player"].set_audio_enabled(i == self.active_view_idx)
            except Exception:
                pass
        # Only active view drives timing; others are mirrored only in sync mode.
        try:
            self.views[self.active_view_idx]["player"].play()
        except Exception:
            pass
        self._update_play_pause_button()
        self._log(
            "play", view=self.views[self.active_view_idx].get("name", ""), frame=frame
        )

    def _pause_all(self):
        frame = None
        if self.views:
            try:
                frame = self.views[self.active_view_idx]["player"].current_frame
            except Exception:
                frame = None
        for vw in self.views:
            try:
                vw["player"].pause()
            except Exception:
                pass
        self._update_play_pause_button()
        self._log(
            "pause",
            view=self.views[self.active_view_idx].get("name", "") if self.views else "",
            frame=frame,
        )

    def _stop_all(self):
        for vw in self.views:
            try:
                vw["player"].stop()
            except Exception:
                pass
        self._update_play_pause_button()
        # sync UI to active view start frame
        if self.views:
            p = self.views[self.active_view_idx]["player"]
            frame = p.current_frame
            self._set_frame_controls(frame)
            self._update_overlay_for_frame(frame)
            self._log(
                "stop",
                view=self.views[self.active_view_idx].get("name", ""),
                frame=frame,
            )

    # ----- multi-view helpers -----
    def _default_view_names(self):
        return ["Top", "Front", "Left", "Right", "Ego"]

    def _effective_view_name(self, view: dict, idx: Optional[int] = None) -> str:
        if not isinstance(view, dict):
            return "view"
        name = str(view.get("name") or "").strip()
        combo = view.get("name_combo")
        if combo is not None:
            try:
                combo_name = str(combo.currentText() or "").strip()
            except Exception:
                combo_name = ""
            if combo_name and combo_name != "Custom...":
                name = combo_name
        if not name or name.lower() == "main":
            defaults = self._default_view_names()
            if defaults:
                if idx is None:
                    try:
                        idx = self.views.index(view)
                    except Exception:
                        idx = 0
                idx = max(0, int(idx or 0))
                name = defaults[idx % len(defaults)]
            else:
                name = "view"
        if view.get("name") != name:
            view["name"] = name
        return name

    def _ensure_custom_last(self, combo: QComboBox):
        idx = combo.findText("Custom...")
        if idx >= 0:
            text = combo.itemText(idx)
            icon = combo.itemIcon(idx)
            combo.removeItem(idx)
            combo.addItem(icon, text)
        else:
            combo.addItem("Custom...")

    def _unique_view_name(self, base: str, skip_idx: int = None) -> str:
        base = base.strip() or "View"
        existing = {vw["name"] for i, vw in enumerate(self.views) if i != skip_idx}
        if base not in existing:
            return base
        k = 2
        while f"{base}_{k}" in existing:
            k += 1
        return f"{base}_{k}"

    def _init_primary_view(self, reuse_primary: bool = False):
        # create initial empty view/panel with its own store
        if reuse_primary and self.views:
            return
        vp = self.player if reuse_primary else VideoPlayer(status_cb=self._set_status)
        try:
            vp.set_playback_speed(1.0)
        except Exception:
            pass
        try:
            vp.main_window = self
        except Exception:
            pass
        vp.on_playback_state_changed = self._on_player_playback_state_changed
        defaults = self._default_view_names()
        view = {
            "name": defaults[0] if defaults else "view",
            "player": vp,
            "start": 0,
            "end": 0,
            "path": None,
            "widget": None,
            "name_combo": None,
            "store": AnnotationStore(),
            "prelabel_store": AnnotationStore(),
            "prelabel_source": "",
            "confirmed_accept_records": [],
            "entity_stores": self._clone_entity_stores(self.entity_stores),
            "phase_stores": self._clone_phase_stores(getattr(self, "phase_stores", {})),
            "anomaly_type_stores": self._clone_anomaly_type_stores(
                getattr(self, "anomaly_type_stores", {})
            ),
            "psr_state": self._psr_empty_view_state(),
            "trim_cuts": self._empty_trim_state(),
            "baseline_trim_cuts": self._empty_trim_state(),
            "dirty": False,
        }
        self.views = [view]
        self.active_view_idx = 0
        self._sync_edit_view_indices = {0}
        self.player = vp
        self.store = view["store"]
        self.prelabel_store = view["prelabel_store"]
        self._prelabel_source = str(view.get("prelabel_source", "") or "")
        self.entity_stores = view["entity_stores"]
        self.phase_stores = view.get("phase_stores", {})
        self.anomaly_type_stores = view.get("anomaly_type_stores", {})
        self._psr_load_view_state(view)
        self._rebuild_view_widgets()

    def _rebuild_view_widgets(self):
        # clear layout
        while self.video_stack_layout.count():
            item = self.video_stack_layout.takeAt(0)
            w = item.widget()
            if w:
                try:
                    w.hide()
                except Exception:
                    pass
                try:
                    w.deleteLater()
                except Exception:
                    pass
        max_views = min(len(self.views), 5)
        layout_widget = self._build_splitter_layout(max_views)
        if layout_widget:
            self.video_stack_layout.addWidget(layout_widget)

    def _build_view_panel(self, view, idx):
        panel = _ViewDropPanel(idx, self._on_view_panel_dropped, self)
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        combo = QComboBox(panel)
        combo.setEditable(False)
        for item in self._default_view_names():
            combo.addItem(item)
        self._ensure_custom_last(combo)
        effective_name = self._effective_view_name(view, idx=idx)
        if effective_name and combo.findText(effective_name) < 0:
            combo.insertItem(max(0, combo.count() - 1), effective_name)
        combo.setCurrentText(effective_name or combo.itemText(0))
        view["name_combo"] = combo
        combo.activated[str].connect(
            lambda text, i=idx: self._on_view_name_selected(i, text)
        )
        combo.setStyleSheet("QComboBox { text-align: center; padding-right: 26px; }")
        btn_close = QPushButton("X", panel)
        btn_close.setFixedWidth(24)
        btn_close.setFixedHeight(24)
        btn_close.setToolTip("Close view")
        btn_close.clicked.connect(lambda _, i=idx: self._close_view(i))
        # overlay close button on top-right of the frame
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(0)
        drag_handle = _ViewReorderHandle(idx, self._on_view_panel_dropped, panel)
        header.addWidget(drag_handle, 0, Qt.AlignLeft | Qt.AlignVCenter)
        header.addWidget(combo, 1)
        header.addWidget(btn_close, 0, Qt.AlignRight | Qt.AlignTop)
        v.addLayout(header)
        frame = QFrame(panel)
        frame.setFrameShape(QFrame.Box)
        frame.setLineWidth(2)
        frame.setLayout(QVBoxLayout())
        frame.layout().setContentsMargins(0, 0, 0, 0)
        frame.layout().addWidget(view["player"])
        view["frame"] = frame
        frame.mousePressEvent = lambda e, i=idx: self._on_view_clicked(i, e)
        self._attach_view_click(view["player"], idx)
        v.addWidget(frame, 1)
        self._update_view_highlight()
        self._log("add_view", view=view.get("name", ""), idx=idx)
        return panel

    def _set_view_name(self, idx: int, name: str):
        if not (0 <= idx < len(self.views)):
            return
        name = self._unique_view_name(name, skip_idx=idx)
        self.views[idx]["name"] = name
        combo = self.views[idx].get("name_combo")
        if combo:
            combo.blockSignals(True)
            if combo.findText(name) < 0:
                combo.addItem(name)
            self._ensure_custom_last(combo)
            combo.setCurrentText(name)
            combo.blockSignals(False)

    def _on_view_name_selected(self, idx: int, text: str):
        if text == "Custom...":
            name, ok = QInputDialog.getText(self, "Custom view name", "Name this view:")
            if not ok or not name.strip():
                # revert to current stored name
                self._set_view_name(idx, self.views[idx]["name"])
                return
            self._set_view_name(idx, name.strip())
        else:
            self._set_view_name(idx, text)

    def _set_primary_view(self, idx: int):
        if not (0 <= idx < len(self.views)):
            return
        self._psr_store_active_view_state()
        self._hover_preview_pending_frame = None
        self._hover_preview_last_targets = {}
        try:
            self._hover_preview_timer.stop()
        except Exception:
            pass
        self.active_view_idx = idx
        self._ensure_view_trim_state(self.views[idx])
        self._normalize_sync_edit_selection()
        if not self._sync_edit_view_indices:
            self._sync_edit_view_indices = {int(idx)}
        else:
            self._sync_edit_view_indices.add(int(idx))
        self.player = self.views[idx]["player"]
        self.store = self.views[idx]["store"]
        if "prelabel_store" not in self.views[idx]:
            self.views[idx]["prelabel_store"] = AnnotationStore()
        if "prelabel_source" not in self.views[idx]:
            self.views[idx]["prelabel_source"] = ""
        if "confirmed_accept_records" not in self.views[idx]:
            self.views[idx]["confirmed_accept_records"] = []
        self.prelabel_store = self.views[idx]["prelabel_store"]
        self._prelabel_source = str(self.views[idx].get("prelabel_source", "") or "")
        if "entity_stores" not in self.views[idx]:
            self.views[idx]["entity_stores"] = self._clone_entity_stores(
                self.entity_stores
            )
        self.entity_stores = self.views[idx]["entity_stores"]
        if "phase_stores" not in self.views[idx]:
            self.views[idx]["phase_stores"] = self._clone_phase_stores(
                self.phase_stores
            )
        if "anomaly_type_stores" not in self.views[idx]:
            self.views[idx]["anomaly_type_stores"] = self._clone_anomaly_type_stores(
                self.anomaly_type_stores
            )
        self.phase_stores = self.views[idx].get("phase_stores", {})
        self.anomaly_type_stores = self.views[idx].get("anomaly_type_stores", {})
        self._psr_load_view_state(self.views[idx])
        self._phase_selected = None
        self._sync_anomaly_panel()
        if (
            self._active_entity_name
            and self._active_entity_name not in self.entity_stores
        ):
            self._active_entity_name = None
        for i, vw in enumerate(self.views):
            vw["player"].on_frame_advanced = (
                self._on_player_frame_advanced if i == idx else None
            )
            vw["player"].on_playback_state_changed = (
                self._on_player_playback_state_changed
            )
            try:
                vw["player"].set_audio_enabled(i == idx)
            except Exception:
                pass
        self._apply_magnifier_state()
        self._sync_controls_with_primary()
        self._rebuild_timeline_sources()
        self._apply_sync_edit_masks()
        self._update_view_highlight()
        try:
            frame = self.views[self.active_view_idx]["player"].current_frame
        except Exception:
            frame = 0
        self._timeline_auto_follow = True
        self.timeline.set_current_frame(frame, follow=True)
        self.timeline.set_current_hits(self._hit_names_for_frame(frame))
        self._update_play_pause_button()
        self._log(
            "set_primary_view",
            view=self.views[self.active_view_idx].get("name", ""),
            frame=frame,
        )

    def export_active_view_video_session(self) -> Optional[Dict[str, Any]]:
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return None
        view = self.views[self.active_view_idx] or {}
        player = view.get("player") or getattr(self, "player", None)
        if player is None or not getattr(player, "cap", None):
            return None
        path = str(view.get("path") or getattr(self, "video_path", "") or "").strip()
        if not path:
            return None
        start = int(view.get("start", getattr(player, "crop_start", 0)) or 0)
        end = int(view.get("end", getattr(player, "crop_end", start)) or start)
        frame = int(getattr(player, "current_frame", start) or start)
        frame = max(start, min(frame, end))
        return {
            "path": path,
            "start": start,
            "end": end,
            "frame": frame,
            "view_name": str(view.get("name", "") or ""),
        }

    def _update_view_highlight(self):
        selected = set(self._normalize_sync_edit_selection())
        for i, vw in enumerate(self.views):
            frame = vw.get("frame")
            if not frame:
                continue
            if i == self.active_view_idx:
                frame.setStyleSheet("QFrame { border: 2px solid #2f80ed; }")
            elif i in selected:
                frame.setStyleSheet("QFrame { border: 2px solid #12b76a; }")
            else:
                frame.setStyleSheet("QFrame { border: 1px solid #d0d5dd; }")

    def _sync_controls_with_primary(self):
        pv = self.views[self.active_view_idx]
        p = pv["player"]
        self.slider.setMinimum(p.crop_start)
        self.slider.setMaximum(p.crop_end)
        self.slider.setValue(p.current_frame)
        self.spin_jump.setMinimum(p.crop_start)
        self.spin_jump.setMaximum(p.crop_end)
        self.spin_jump.setValue(p.current_frame)
        self._update_overlay_for_frame(p.current_frame)
        self.timeline.set_current_frame(
            p.current_frame, follow=self._timeline_auto_follow
        )
        self.timeline.set_current_hits(self._hit_names_for_frame(p.current_frame))
        self._update_transcript_workspace_for_frame(p.current_frame)
        self._update_gap_indicator()
        self._log(
            "sync_controls",
            view=self.views[self.active_view_idx].get("name", ""),
            frame=p.current_frame,
        )

    def _sync_views_to_frame(self, frame: int, preview_only: bool = False):
        if not self.views:
            return
        pv = self.views[self.active_view_idx]
        # update controls
        if not preview_only:
            self._set_frame_controls(frame)
        try:
            pv["player"].seek(frame, preview_only=preview_only)
        except Exception:
            pass
        # timeline follow/highlight
        self.timeline.set_current_frame(
            frame, follow=self._timeline_auto_follow and not preview_only
        )
        self._update_transcript_workspace_for_frame(frame)

    def _sync_other_views(self, frame: int, force: bool = False):
        if not self.views:
            return
        sync_indices = set(self._playback_sync_indices())
        if len(sync_indices) <= 1:
            return
        if not force:
            try:
                if not self.views[self.active_view_idx]["player"].is_playing:
                    return
            except Exception:
                return
        base = self.views[self.active_view_idx]
        offset = frame - base.get("start", 0)
        for i, vw in enumerate(self.views):
            if i == self.active_view_idx or i not in sync_indices:
                continue
            target = vw.get("start", 0) + offset
            target = max(
                vw.get("start", 0), min(target, vw.get("end", vw.get("start", 0)))
            )
            try:
                vw["player"].seek(target, preview_only=False)
            except Exception:
                pass

    def _toggle_play_pause(self):
        if not self.views:
            return
        active_player = self.views[self.active_view_idx]["player"]
        if active_player.is_playing:
            self._pause_all()
        else:
            self._play_all()

    def _step_frames(self, delta: int):
        if not self.views:
            return
        if self.interaction_mode == "assisted":
            pt = self._active_assisted_point()
            if pt and pt.get("type") == "boundary" and abs(int(delta)) == 1:
                self._adjust_active_boundary(int(delta))
                return
        p = self.views[self.active_view_idx]["player"]
        target = max(p.crop_start, min(p.current_frame + delta, p.crop_end))
        self._sync_views_to_frame(target, preview_only=False)
        self._sync_other_views(target)
        self._update_overlay_for_frame(target)
        self._log("step_frame", delta=delta, target=target)

    def _jump_to_bound(self, which: str):
        if not self.views:
            return
        p = self.views[self.active_view_idx]["player"]
        if which == "start":
            frame = p.crop_start
        else:
            frame = p.crop_end
        self._sync_views_to_frame(frame, preview_only=False)
        self._sync_other_views(frame)
        self._update_overlay_for_frame(frame)
        self._log("jump_to_bound", which=which, frame=frame)

    def _primary_span_len(self) -> int:
        if not self.views:
            return 0
        v = self.views[self.active_view_idx]
        return max(0, v.get("end", 0) - v.get("start", 0) + 1)

    def _load_primary_video(self, path: str) -> bool:
        if not self.views:
            self._init_primary_view()
        # clear extra views
        primary_view = self.views[0]
        for extra in self.views[1:]:
            player = extra.get("player")
            if player is not None:
                try:
                    player.release_media()
                except Exception:
                    pass
        self.views = [primary_view]
        self.active_view_idx = 0
        self.player = primary_view["player"]

        # load into primary player
        if not self.player.load(path):
            QMessageBox.warning(self, "Error", "Failed to load video.")
            return False
        total_frames = self.player.frame_count
        s, ok1 = QInputDialog.getInt(
            self,
            "Start frame",
            f"Start (0..{total_frames - 1}):",
            value=0,
            min=0,
            max=max(0, total_frames - 1),
        )
        if not ok1:
            return False
        e, ok2 = QInputDialog.getInt(
            self,
            "End frame",
            f"End (>=start..{total_frames - 1}):",
            value=total_frames - 1,
            min=s,
            max=max(0, total_frames - 1),
        )
        if not ok2:
            return False
        self.player.set_crop(s, e)
        self.video_path = path
        primary_view.update({"path": path, "start": s, "end": e})
        self._warn_if_crop_conflicts(primary_view["store"], s, e)
        # audio attach (optional, default off to avoid missing GStreamer plugins)
        if self._disable_qt_audio:
            try:
                self.player.set_audio_enabled(False)
            except Exception:
                pass
            self._set_status("Audio disabled (missing plugins); video loaded.")
        else:
            if self.player.attach_audio_from_video(path):
                self.player.set_audio_enabled(True)
                self._set_status("Audio attached from container.")
            else:
                has, _ = probe_audio_stream(path)
                if has:
                    wav = os.path.join(
                        tempfile.gettempdir(), f"_cache_audio_16k_{abs(hash(path))}.wav"
                    )
                    ok, elog = extract_wav_16k_mono_verbose(path, wav)
                    if ok:
                        self.player.attach_audio_file(wav)
                        self.player.set_audio_enabled(True)
                        self._set_status("Audio extracted & attached.")
                    else:
                        self._set_status(
                            "Video has audio, but extract failed; you may attach external audio."
                        )
                else:
                    self._set_status(
                        "No audio track in video. You can attach an external audio file."
                    )
        self.player.stop()
        self._reset_all_state_for_new_video(keep_views=True)
        self._after_video_loaded()
        # rebuild view widgets and sync controls
        self._rebuild_view_widgets()
        self._set_primary_view(0)
        self._log("load_video", path=path, start=s, end=e, frames=total_frames)
        return True

    def _on_add_view(self):
        if not self.views or not self.views[0]["player"].cap:
            QMessageBox.information(self, "Info", "Load the primary video first.")
            return
        if len(self.views) >= 5:
            QMessageBox.information(self, "Info", "Maximum 5 views supported.")
            return
        self._psr_store_active_view_state()
        path, _ = QFileDialog.getOpenFileName(
            self, "Load additional view", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return
        vp = VideoPlayer(status_cb=self._set_status)
        try:
            vp.main_window = self
        except Exception:
            pass
        vp.on_playback_state_changed = self._on_player_playback_state_changed
        if not vp.load(path):
            QMessageBox.warning(self, "Error", "Failed to load video.")
            return
        total_frames = vp.frame_count
        # ask start/end
        s, ok1 = QInputDialog.getInt(
            self,
            "Start frame",
            f"Start (0..{total_frames - 1}):",
            value=0,
            min=0,
            max=max(0, total_frames - 1),
        )
        if not ok1:
            return
        e, ok2 = QInputDialog.getInt(
            self,
            "End frame",
            f"End (>=start..{total_frames - 1}):",
            value=total_frames - 1,
            min=s,
            max=max(0, total_frames - 1),
        )
        if not ok2:
            return
        vp.set_crop(s, e)
        span = e - s + 1
        primary_span = self._primary_span_len()
        if primary_span and span != primary_span:
            QMessageBox.warning(
                self,
                "Error",
                f"Span mismatch. Expected {primary_span} frames to match primary view.",
            )
            vp.setParent(None)
            return
        # pick name
        defaults = self._default_view_names()
        suggested = (
            defaults[len(self.views) % len(defaults)]
            if defaults
            else f"View{len(self.views)+1}"
        )
        name, ok3 = QInputDialog.getText(
            self, "View name", "Name this view:", text=suggested
        )
        if not ok3 or not name.strip():
            name = suggested
        name = self._unique_view_name(name)
        # clone annotation store from active view
        base_store = self.views[self.active_view_idx]["store"]
        new_store = self._clone_store(base_store)
        base_entities = self.views[self.active_view_idx].get("entity_stores", {})
        new_entities = self._clone_entity_stores(base_entities)
        base_prelabel = self.views[self.active_view_idx].get("prelabel_store")
        new_prelabel = (
            self._clone_store(base_prelabel) if base_prelabel else AnnotationStore()
        )
        base_prelabel_source = str(
            self.views[self.active_view_idx].get("prelabel_source", "") or ""
        ).strip()
        base_phase = self.views[self.active_view_idx].get("phase_stores", {})
        new_phase = self._clone_phase_stores(base_phase)
        base_anom = self.views[self.active_view_idx].get("anomaly_type_stores", {})
        new_anom = self._clone_anomaly_type_stores(base_anom)
        base_psr_state = self._psr_clone_view_state(
            self.views[self.active_view_idx].get("psr_state")
        )
        base_trim = self.views[self.active_view_idx].get("trim_cuts")
        new_trim = self._clone_trim_state(base_trim)
        base_baseline_trim = self.views[self.active_view_idx].get("baseline_trim_cuts")
        new_baseline_trim = self._clone_trim_state(base_baseline_trim)
        view = {
            "name": name.strip(),
            "player": vp,
            "start": s,
            "end": e,
            "path": path,
            "widget": None,
            "name_combo": None,
            "store": new_store,
            "prelabel_store": new_prelabel,
            "prelabel_source": base_prelabel_source,
            "confirmed_accept_records": [
                dict(row)
                for row in (self.views[self.active_view_idx].get("confirmed_accept_records") or [])
                if isinstance(row, dict)
            ],
            "entity_stores": new_entities,
            "phase_stores": new_phase,
            "anomaly_type_stores": new_anom,
            "psr_state": base_psr_state,
            "trim_cuts": new_trim,
            "baseline_trim_cuts": new_baseline_trim,
            "dirty": False,
            "stretch": 1,
        }
        self._warn_if_crop_conflicts(new_store, s, e)
        # mute extra audio
        try:
            vp.set_audio_enabled(False)
        except Exception:
            pass
        self.views.append(view)
        self._rebuild_view_widgets()
        self._set_primary_view(self.active_view_idx)
        self._log(
            "add_view_source", view=name, path=path, start=s, end=e, frames=total_frames
        )

    def _set_controls_enabled(self, on: bool):
        for w in (
            self.btn_rew,
            self.btn_play,
            self.btn_stop,
            self.btn_ff,
            self.spin_jump,
            self.btn_jump,
            self.slider,
            self.btn_mag,
            self.combo_speed,
            self.btn_validation,
            self.btn_extra,
            self.btn_assisted,
            getattr(self, "combo_interaction", None),
        ):
            if w is None:
                continue
            w.setEnabled(on)
        # keep these accessible even before a video is loaded
        try:
            self.combo_actions.setEnabled(True)
        except Exception:
            pass
        try:
            self.btn_auto_label.setEnabled(True)
        except Exception:
            pass
        if not on:
            try:
                self.btn_mag.setChecked(False)
                self._apply_magnifier_state()
            except Exception:
                pass

    def _apply_magnifier_state(self):
        """Apply magnifier toggle to the active view; disable on others."""
        state = bool(self.btn_mag.isChecked())
        for i, vw in enumerate(self.views):
            try:
                vw["player"].set_magnifier_enabled(
                    state if i == self.active_view_idx else False
                )
            except Exception:
                pass

    def _on_magnifier_toggled(self, on: bool):
        self._apply_magnifier_state()
        self._log("magnifier_toggle", on=bool(on))

    def _set_status(self, s: str):
        self.lbl_status.setText(s)

    def _open_progress_dialog(
        self, title: str, label: str, total: Optional[int] = None
    ) -> QProgressDialog:
        max_val = int(total) if total and int(total) > 0 else 0
        dlg = QProgressDialog(label, "", 0, max_val, self)
        dlg.setWindowTitle(title)
        dlg.setCancelButton(None)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumDuration(0)
        dlg.show()
        return dlg

    def _close_progress_dialog(self, dlg: Optional[QProgressDialog]):
        if not dlg:
            return
        try:
            dlg.reset()
            dlg.close()
        except Exception:
            pass

    def _on_feat_progress(self, done: int, total: int):
        dlg = getattr(self, "_feat_progress", None)
        if not dlg:
            return
        if total and dlg.maximum() != total:
            dlg.setRange(0, total)
        if total:
            dlg.setValue(min(done, total))
            dlg.setLabelText(f"Extracting features... {done}/{total}")
        else:
            dlg.setRange(0, 0)
            dlg.setLabelText(f"Extracting features... {done}")

    def _ensure_feature_extractor_available(self, show_dialog: bool = True) -> bool:
        try:
            load_feature_extractor_module()
            return True
        except MissingOptionalDependency as ex:
            if show_dialog:
                QMessageBox.information(
                    self,
                    "Missing dependency",
                    format_missing_dependency_message(ex),
                )
            self._set_status(
                "Feature extraction is unavailable until the optional dependencies are installed."
            )
            return False
        except Exception as ex:
            if show_dialog:
                QMessageBox.warning(
                    self,
                    "Feature extraction unavailable",
                    f"Failed to initialize feature extraction:\n{ex}",
                )
            self._set_status("Feature extraction failed to initialize.")
            return False

    def _ensure_python_modules_available(
        self,
        modules: Iterable[str],
        *,
        feature_name: str,
        install_hint: str = "",
        unavailable_status: str = "",
        show_dialog: bool = True,
    ) -> bool:
        try:
            ensure_optional_modules(
                modules,
                feature_name=feature_name,
                install_hint=install_hint,
            )
            return True
        except MissingOptionalDependency as ex:
            if show_dialog:
                QMessageBox.information(
                    self,
                    "Missing dependency",
                    format_missing_dependency_message(ex),
                )
            self._set_status(
                unavailable_status
                or (
                    f"{feature_name} is unavailable until the optional dependencies are installed."
                )
            )
            return False
        except Exception as ex:
            if show_dialog:
                QMessageBox.warning(
                    self,
                    f"{feature_name} unavailable",
                    f"Failed to initialize {feature_name.lower()}:\n{ex}",
                )
            self._set_status(f"{feature_name} failed to initialize.")
            return False

    def _ensure_east_runtime_ready(
        self,
        *,
        feature_name: str = "EAST refinement",
        unavailable_status: str = "",
        show_dialog: bool = True,
    ) -> bool:
        if not self._ensure_python_modules_available(
            ("torch", "mmengine"),
            feature_name=feature_name,
            install_hint=(
                "Install the optional EAST runtime dependencies, for example: "
                "pip install torch mmengine"
            ),
            unavailable_status=(
                unavailable_status
                or (
                    f"{feature_name} is unavailable until the EAST runtime dependencies are installed."
                )
            ),
            show_dialog=show_dialog,
        ):
            return False
        try:
            from tools.east.east_model_infer import (
                _ensure_east_importable,
                resolve_east_repo_root,
            )

            east_root = resolve_east_repo_root()
            _ensure_east_importable(east_root)
            return True
        except Exception as ex:
            message = (
                f"{feature_name} requires a complete EAST runtime checkout.\n\n"
                f"Details: {ex}\n\n"
                "Expected one of these setups:\n"
                "1. Set EAST_REPO_ROOT to a full EAST-main checkout that contains opentad/ and the nms shim.\n"
                "2. Place a full EAST-main checkout at external/EAST-main.\n\n"
                "A folder that only contains configs/ and pretrained/ is not enough."
            )
            if show_dialog:
                QMessageBox.warning(self, f"{feature_name} unavailable", message)
            self._set_status(
                unavailable_status
                or (
                    f"{feature_name} is unavailable until a full EAST-main checkout is configured."
                )
            )
            return False

    def _on_feat_progress_message(self, line: str):
        text = str(line or "").strip()
        if text:
            self._set_status(text)
        if "[FEATS][ERROR]" in text:
            detail = text.split("[FEATS][ERROR]", 1)[-1].strip()
            self._last_feature_error_message = detail or text

    def _count_videos_in_dir(self, video_dir: str) -> int:
        try:
            return len(
                [
                    name
                    for name in os.listdir(video_dir)
                    if name.lower().endswith(FEATURE_VIDEO_EXTS)
                ]
            )
        except Exception:
            return 0

    def _on_fact_batch_progress(self, line: str):
        self._set_status(line)
        dlg = getattr(self, "_fact_batch_progress", None)
        if not dlg:
            return
        m = re.search(r"\((\d+)/(\d+)\)", line)
        if m:
            current = int(m.group(1))
            total = int(m.group(2))
            self._fact_batch_current = current
            self._fact_batch_total = total
            done = int(getattr(self, "_fact_batch_done", 0) or 0)
            if current > 1:
                done = max(done, current - 1)
                self._fact_batch_done = done
        if line.startswith("[INFO] Found"):
            m = re.search(r"Found\s+(\d+)", line)
            if m:
                self._fact_batch_total = int(m.group(1))
        if line.startswith("[OK]") and "feats" in line:
            self._fact_batch_done = int(getattr(self, "_fact_batch_done", 0)) + 1
        if line.startswith("[WARN] no features extracted for"):
            self._fact_batch_done = int(getattr(self, "_fact_batch_done", 0)) + 1
        if line.startswith("[OK] FACT batch inference done"):
            if getattr(self, "_fact_batch_total", None):
                self._fact_batch_done = int(self._fact_batch_total)

        total = int(getattr(self, "_fact_batch_total", 0) or 0)
        done = int(getattr(self, "_fact_batch_done", 0) or 0)
        current = int(getattr(self, "_fact_batch_current", 0) or 0)
        if total > 0:
            if dlg.maximum() != total:
                dlg.setRange(0, total)
            dlg.setValue(min(done, total))
            label = f"FACT batch labeling... {min(done, total)}/{total}"
            if current > 0 and done < total:
                label += f" (processing {min(current, total)}/{total})"
            dlg.setLabelText(label)
        else:
            dlg.setRange(0, 0)
            label = "FACT batch labeling..."
            if current > 0:
                label += f" (processing {current})"
            dlg.setLabelText(label)

    def _set_interaction_status(self, s: str):
        try:
            self.lbl_interaction_status.setText(s)
        except Exception:
            pass

    def _get_frame_count(self) -> int:
        return self.player.frame_count

    def _get_fps(self) -> int:
        return max(1, self.player.frame_rate or 30)

    def _view_idx_by_name(self, name: str) -> int:
        for i, vw in enumerate(self.views):
            if vw.get("name") == name:
                return i
        return -1

    def _default_asot_ckpt(self) -> str:
        """Find a reasonable default ASOT checkpoint: prefer newest under lightning_logs, then wandb ckpts, else bundled."""
        search_dirs = [
            os.path.abspath(
                os.path.join(
                    self._root_dir, "external", "action_seg_ot", "lightning_logs"
                )
            ),
            os.path.abspath(
                os.path.join(self._root_dir, "external", "action_seg_ot", "wandb")
            ),
        ]
        newest = ""
        newest_mtime = -1
        for sd in search_dirs:
            for p in glob.glob(os.path.join(sd, "**", "*.ckpt"), recursive=True):
                try:
                    mtime = os.path.getmtime(p)
                    if mtime > newest_mtime:
                        newest_mtime = mtime
                        newest = p
                except Exception:
                    continue
        if newest:
            return newest
        # fallback to the bundled wandb ckpt
        bundled = os.path.join(
            self._root_dir,
            "external",
            "action_seg_ot",
            "wandb",
            "video_ssl",
            "s8ijjapy",
            "checkpoints",
            "epoch=79-step=7360.ckpt",
        )
        return bundled if os.path.isfile(bundled) else ""

    def _default_features_dir_for_video(self) -> Optional[str]:
        if not getattr(self, "video_path", None):
            return None
        base = os.path.splitext(os.path.basename(self.video_path))[0]
        return os.path.join(os.path.dirname(self.video_path), f"{base}_features")

    def _ensure_features_for_current_video(self) -> Optional[str]:
        """
        Ensure features.npy exists for the currently loaded video.
        Returns the features directory path or None on failure.
        """
        feat_dir = self._default_features_dir_for_video()
        if not feat_dir:
            return None
        os.makedirs(feat_dir, exist_ok=True)
        feat_path = os.path.join(feat_dir, "features.npy")
        if os.path.isfile(feat_path):
            return feat_dir
        # Defer extraction to a background worker; caller will start the thread.
        return feat_dir

    def _default_east_features_dir_for_video(self) -> Optional[str]:
        if not getattr(self, "video_path", None):
            return None
        base = os.path.splitext(os.path.basename(self.video_path))[0]
        return os.path.join(os.path.dirname(self.video_path), f"{base}_features_east")

    def _ensure_east_features_for_current_video(self) -> Optional[str]:
        feat_dir = self._default_east_features_dir_for_video()
        if not feat_dir:
            return None
        os.makedirs(feat_dir, exist_ok=True)
        return feat_dir

    def _east_label_bank(self) -> List[str]:
        labels: List[str] = []
        for label in getattr(self, "labels", []) or []:
            name = str(getattr(label, "name", "") or "").strip()
            if name and name not in labels:
                labels.append(name)
        return labels

    def _current_label_bank_source_path(self) -> str:
        path = os.path.abspath(
            os.path.expanduser(str(getattr(self, "_action_label_bank_source", "") or "").strip())
        )
        if not path or not os.path.isfile(path):
            return ""
        try:
            current = list(self._east_label_bank())
            source_names = load_label_names(path)
        except Exception:
            return ""
        if current and source_names and list(source_names) == current:
            return path
        return ""

    def _write_label_bank_txt(
        self,
        target_dir: str,
        *,
        file_name: str,
        labels: Optional[List[str]] = None,
    ) -> Optional[str]:
        names = [str(x).strip() for x in (labels or self._east_label_bank()) if str(x).strip()]
        if not names:
            return None
        try:
            os.makedirs(target_dir, exist_ok=True)
            path = os.path.join(target_dir, file_name)
            with open(path, "w", encoding="utf-8") as f:
                for name in names:
                    f.write(f"{name}\n")
            return path
        except Exception:
            return None

    def _resolve_action_label_bank_path(
        self,
        target_dir: str,
        *,
        generated_file_name: str,
        dialog_title: str,
        missing_title: str,
        missing_text: str,
    ) -> Optional[str]:
        source_path = self._current_label_bank_source_path()
        if source_path:
            return source_path
        generated = self._write_label_bank_txt(
            target_dir,
            file_name=generated_file_name,
        )
        if generated:
            return generated
        label_source = resolve_label_source(
            features_dir=target_dir,
            video_path=str(getattr(self, "video_path", "") or ""),
            repo_root=self._root_dir,
        )
        if label_source and os.path.isfile(label_source):
            return label_source
        label_txt, _ = QFileDialog.getOpenFileName(
            self,
            dialog_title,
            "",
            "Text Files (*.txt)",
        )
        if label_txt:
            return label_txt
        QMessageBox.warning(self, missing_title, missing_text)
        return None

    def _east_text_bank_backend_items(self) -> List[Tuple[str, str]]:
        return [
            ("Auto", "auto"),
            ("MobileCLIP / CLIP", "mobileclip"),
            ("SentenceTransformer", "sentence_transformers"),
            ("Lexical fallback", "hashed_lexical"),
        ]

    def _refresh_east_backend_defaults(self) -> None:
        try:
            load_feature_env_defaults(repo_root=self._root_dir)
        except Exception:
            pass
        env_ckpt = str(os.environ.get("EAST_CKPT", "") or "").strip()
        env_cfg = str(os.environ.get("EAST_CFG", "") or "").strip()
        env_text_bank = str(
            os.environ.get("EAST_TEXT_BANK_BACKEND")
            or os.environ.get("EAST_TEXT_BANK_VERSION", "auto")
            or "auto"
        ).strip()
        if env_ckpt and (
            (not str(self._east_ckpt or "").strip()) or (not os.path.isfile(self._east_ckpt))
        ):
            self._east_ckpt = env_ckpt
        if env_cfg and (
            (not str(self._east_cfg or "").strip()) or (not os.path.isfile(self._east_cfg))
        ):
            self._east_cfg = env_cfg
        if env_text_bank:
            self._east_text_bank_version = env_text_bank

    def _east_feature_meta_backbone(self, features_dir: str) -> str:
        meta = self._load_feature_meta(features_dir)
        candidates = [
            meta.get("source"),
            meta.get("backbone"),
            meta.get("feature_backbone_version"),
        ]
        for item in candidates:
            text = str(item or "").strip()
            if text:
                return text
        if meta.get("east_cfg") or meta.get("east_ckpt"):
            return "east_backbone"
        return ""

    def _east_features_compatible(self, features_dir: str) -> bool:
        tag = self._east_feature_meta_backbone(features_dir).strip().lower()
        if not tag:
            return False
        return any(
            token in tag
            for token in (
                "east",
                "videomae",
                "video_mae",
            )
        )

    def _ensure_east_model_paths(self) -> bool:
        self._refresh_east_backend_defaults()
        ckpt = str(self._east_ckpt or "").strip()
        cfg = str(self._east_cfg or "").strip()
        if not ckpt or not os.path.isfile(ckpt):
            ckpt, _ = QFileDialog.getOpenFileName(
                self, "Choose EAST checkpoint", "", "PyTorch Models (*.pth *.pt)"
            )
            if not ckpt:
                return False
            self._east_ckpt = ckpt
        if not cfg or not os.path.isfile(cfg):
            cfg, _ = QFileDialog.getOpenFileName(
                self, "Choose EAST config", "", "Config Files (*.py *.yaml *.yml)"
            )
            if not cfg:
                return False
            self._east_cfg = cfg
        try:
            self._persist_east_ui_state()
        except Exception:
            pass
        return True

    def _prepare_east_labels_file(self, features_dir: str) -> Optional[str]:
        return self._resolve_action_label_bank_path(
            features_dir,
            generated_file_name="east_labels.txt",
            dialog_title="Choose label bank (TXT)",
            missing_title="Missing labels",
            missing_text="EAST refinement requires a label bank or current action labels.",
        )

    def _feature_layout_hint(self, features_dir: str) -> str:
        """
        Infer feature layout for ASOT:
        - Returns 'BDT' if shape looks like (D, T) with D=2048.
        - Returns 'BTD' otherwise (default T x D).
        """
        feat_path = os.path.join(features_dir, "features.npy")
        try:
            arr = np.load(feat_path, mmap_mode="r")
            if arr.ndim == 2:
                h, w = arr.shape
                if h == 2048 and w != 2048:
                    return "BDT"
                if w == 2048 and h != 2048:
                    return "BTD"
        except Exception:
            pass
        return "BTD"

    def _feature_frame_count(self, features_dir: str) -> Optional[int]:
        feat_path = os.path.join(features_dir, "features.npy")
        meta = self._load_feature_meta(features_dir)
        picked = meta.get("picked_indices")
        if isinstance(picked, list) and picked:
            return len(picked)
        try:
            arr = np.load(feat_path, mmap_mode="r")
            if arr.ndim == 2:
                h, w = arr.shape
                if h == 2048 and w != 2048:
                    return int(w)
                if w == 2048 and h != 2048:
                    return int(h)
                return int(max(h, w))
        except Exception:
            return None
        return None

    def _emit_task_changed(self, text: str):
        if callable(self._on_switch_task):
            self._on_switch_task(text)

    def set_task(self, text: str):
        if not getattr(self, "combo_task", None):
            return
        try:
            self.combo_task.blockSignals(True)
            self.combo_task.setCurrentText(text)
            self.combo_task.blockSignals(False)
        except Exception:
            pass

    def set_psr_panel(self, panel):
        self.psr_embedded = panel
        if panel is None:
            return
        try:
            if callable(getattr(panel, "set_available_models", None)):
                panel.set_available_models(self._psr_model_specs, emit=False)
        except Exception:
            pass
        try:
            if callable(getattr(panel, "set_model_type", None)):
                panel.set_model_type(self._psr_model_type, emit=False)
        except Exception:
            pass

    def set_review_shortcuts_enabled(self, enabled: bool):
        for sc in (
            getattr(self, "sc_review_next", None),
            getattr(self, "sc_review_prev", None),
        ):
            if sc is not None:
                try:
                    sc.setEnabled(bool(enabled))
                except Exception:
                    pass

    def enter_psr_mode(self):
        self._on_psr_asr_asd_activated()
        self._set_primary_undo_shortcuts_enabled(False)
        self._apply_psr_controls(True)
        self._set_status("Assembly state annotation mode ready.")

    def exit_psr_mode(self):
        self._apply_psr_controls(False)
        self._set_primary_undo_shortcuts_enabled(True)
        try:
            self.player.pause()
        except Exception:
            pass
        if getattr(self, "timeline", None):
            try:
                self.timeline.set_highlight_labels([])
            except Exception:
                pass
        self._update_gap_indicator()

    def _apply_psr_controls(self, is_psr: bool):
        widgets = [
            getattr(self, "btn_auto_label", None),
            getattr(self, "btn_auto_label_asot", None),
            getattr(self, "btn_mag", None),
            getattr(self, "btn_extra", None),
            getattr(self, "btn_assisted", None),
            getattr(self, "lbl_interaction", None),
            getattr(self, "combo_interaction", None),
            getattr(self, "lbl_interaction_status", None),
        ]
        for w in widgets:
            if w is not None:
                w.setVisible(not is_psr)
        if getattr(self, "lbl_mode", None):
            self.lbl_mode.setVisible(not is_psr)
        if getattr(self, "combo_mode", None):
            if is_psr:
                try:
                    self.combo_mode.setCurrentText("Coarse")
                except Exception:
                    pass
            self.combo_mode.setVisible(not is_psr)
        self._apply_psr_action_dropdown(is_psr)
        self._apply_psr_asr_panel(is_psr)
        self._apply_psr_left_panel(is_psr)
        self._apply_psr_timeline(is_psr)
        self._update_phase_panel_visibility()

    def _apply_psr_action_dropdown(self, is_psr: bool) -> None:
        combo = getattr(self, "combo_actions", None)
        if combo is None:
            return
        try:
            combo.hidePopup()
        except Exception:
            pass
        if is_psr:
            hidden = {
                "Export JSON...",
                "Export JSON (selected views to folders)...",
                "Export to Seed Dataset...",
                "Import label map (TXT)...",
                "Export label map (TXT)...",
                "ASOT: Build Label Remap...",
                "Batch Pre-label...",
                "EAST Setup...",
                "EAST: Finalize Current Video",
                "EAST: Inspect Runtime Assets",
                "EAST: Export Runtime Report...",
                "EAST: Export Shared Adapter...",
                "EAST: Select Shared Adapter...",
                "EAST: Clear Shared Adapter",
                "EAST: Consolidate Shared Adapter...",
                "Transcript: Open Workspace",
                "Transcript Audio: Attach External Audio...",
                "Transcript Audio: Set Audio Offset (ms)...",
                "Transcript: Quick Generate / Import...",
            }
        else:
            hidden = {
                "Assembly State: Load Components...",
                "Assembly State: Save Components...",
                "Assembly State: Load Rules...",
                "Assembly State: Load State JSON...",
                "Assembly State: Export State JSON...",
            }
        items = ["Choose action..."]
        headers: Set[str] = set()
        for title, group_items in list(getattr(self, "_action_groups", [])):
            visible_items = [item for item in group_items if item not in hidden]
            if not visible_items:
                continue
            header = f"[{title}]"
            headers.add(header)
            items.append(header)
            items.extend(visible_items)
        current_items = [combo.itemText(i) for i in range(combo.count())]
        if current_items == items and headers == set(
            getattr(self, "_action_section_headers", set())
        ):
            return
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        self._action_section_headers = set(headers)
        model = combo.model()
        for i in range(combo.count()):
            item_text = combo.itemText(i)
            try:
                model_item = model.item(i)
            except Exception:
                model_item = None
            if model_item is None:
                continue
            if item_text in headers:
                model_item.setEnabled(False)
                model_item.setForeground(QColor("#667085"))
            else:
                model_item.setEnabled(True)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)
        try:
            combo.hidePopup()
        except Exception:
            pass

    def _apply_psr_asr_panel(self, is_psr: bool):
        if not getattr(self, "asr_panel", None):
            return
        if is_psr:
            if self._psr_prev_asr_visible is None:
                self._psr_prev_asr_visible = bool(self.asr_panel.isVisible())
            try:
                self._toggle_asr_panel(False)
            except Exception:
                pass
            for w in (
                self.asr_panel,
                self.asr_stub,
                self.btn_toggle_asr_panel,
                self.lbl_asr_title,
            ):
                if w is not None:
                    w.setVisible(False)
        else:
            if self._psr_prev_asr_visible is None:
                restore = True
            else:
                restore = self._psr_prev_asr_visible
                self._psr_prev_asr_visible = None
            for w in (self.btn_toggle_asr_panel, self.lbl_asr_title):
                if w is not None:
                    w.setVisible(True)
            try:
                self._toggle_asr_panel(bool(restore))
            except Exception:
                pass

    def _apply_psr_left_panel(self, is_psr: bool):
        splitter = getattr(self, "splitter_ann", None)
        left = getattr(self, "left_scroll", None) or getattr(
            self, "left_splitter", None
        )
        if splitter is None or left is None:
            return
        if is_psr:
            if self._psr_prev_left_sizes is None:
                self._psr_prev_left_sizes = splitter.sizes()
            left.setVisible(False)
            try:
                splitter.setSizes([0, max(1, splitter.sizes()[1])])
            except Exception:
                pass
        else:
            left.setVisible(True)
            if self._psr_prev_left_sizes:
                try:
                    splitter.setSizes(self._psr_prev_left_sizes)
                except Exception:
                    pass
            self._psr_prev_left_sizes = None

    def _apply_psr_timeline(self, is_psr: bool):
        if not getattr(self, "timeline", None):
            return
        if is_psr:
            if self._psr_prev_timeline_layout is None:
                self._psr_prev_timeline_layout = getattr(
                    self.timeline, "layout_mode", "combined"
                )
            if self._psr_prev_timeline_labels is None:
                self._psr_prev_timeline_labels = getattr(self.timeline, "labels", None)
            if self._psr_prev_combined_editable is None:
                self._psr_prev_combined_editable = getattr(
                    self.timeline, "_combined_editable", None
                )
            self._psr_bind_timeline_toggle(True)
            self._psr_apply_timeline_editability()
            self._psr_bind_timeline_changed(True)
            self._psr_refresh_state_timeline(force=True)
        else:
            self._psr_bind_timeline_toggle(False)
            self._psr_bind_timeline_changed(False)
            try:
                self.timeline.set_combined_delete_handler(None)
            except Exception:
                pass
            try:
                self.timeline.set_segment_cuts([])
            except Exception:
                pass
            if self._psr_prev_timeline_labels is not None:
                try:
                    self.timeline.labels = self._psr_prev_timeline_labels
                except Exception:
                    pass
                self._psr_prev_timeline_labels = None
            if self._psr_prev_combined_editable is not None:
                try:
                    self.timeline.set_combined_editable(
                        bool(self._psr_prev_combined_editable)
                    )
                except Exception:
                    pass
                self._psr_prev_combined_editable = None
            if self._psr_prev_timeline_layout:
                try:
                    self.timeline.set_layout_mode(self._psr_prev_timeline_layout)
                except Exception:
                    pass
                self._psr_prev_timeline_layout = None
        self._rebuild_timeline_sources()

    def _psr_bind_timeline_changed(self, enable: bool):
        if not getattr(self, "timeline", None):
            return
        if enable and not self._psr_timeline_changed_bound:
            try:
                self.timeline.changed.disconnect(self._on_store_changed)
            except Exception:
                pass
            try:
                self.timeline.changed.connect(self._psr_on_state_timeline_changed)
            except Exception:
                pass
            self._psr_timeline_changed_bound = True
        elif not enable and self._psr_timeline_changed_bound:
            try:
                self.timeline.changed.disconnect(self._psr_on_state_timeline_changed)
            except Exception:
                pass
            try:
                self.timeline.changed.connect(self._on_store_changed)
            except Exception:
                pass
            self._psr_timeline_changed_bound = False

    def _psr_bind_timeline_toggle(self, enable: bool):
        chk = getattr(self.timeline, "chk_layout", None)
        if chk is None:
            return
        try:
            chk.toggled.disconnect()
        except Exception:
            pass
        lock_chk = getattr(self.timeline, "chk_action_lock", None)
        if lock_chk is not None:
            try:
                lock_chk.toggled.disconnect()
            except Exception:
                pass
            lock_chk.setVisible(bool(enable))
        if enable:
            if self._psr_prev_timeline_tooltip is None:
                self._psr_prev_timeline_tooltip = chk.toolTip()
            chk.setToolTip("Toggle the PSR state timeline layout.")
            chk.toggled.connect(self._on_psr_single_timeline_toggled)
            chk.blockSignals(True)
            chk.setChecked(bool(self._psr_single_timeline))
            chk.blockSignals(False)
            if lock_chk is not None:
                lock_chk.blockSignals(True)
                lock_chk.setChecked(bool(self._psr_snap_to_action_segments))
                lock_chk.blockSignals(False)
                lock_chk.toggled.connect(self._psr_set_action_segment_lock)
        else:
            if self._psr_prev_timeline_tooltip is not None:
                chk.setToolTip(self._psr_prev_timeline_tooltip)
                self._psr_prev_timeline_tooltip = None
            chk.toggled.connect(self.timeline._on_layout_mode_toggled)

    def _bind_fine_timeline_toggle(self, enable: bool) -> None:
        chk = getattr(self.timeline, "chk_layout", None)
        if chk is None:
            return
        try:
            chk.toggled.disconnect()
        except Exception:
            pass
        if enable:
            if self._fine_prev_timeline_tooltip is None:
                self._fine_prev_timeline_tooltip = chk.toolTip()
            chk.setToolTip(
                "Single timeline (combined per-entity; uncheck for per-label)."
            )
            chk.toggled.connect(self._on_fine_single_axis_toggled)
            chk.blockSignals(True)
            chk.setChecked(getattr(self.timeline, "layout_mode", "") == "combined")
            chk.blockSignals(False)
            try:
                chk.setEnabled(True)
            except Exception:
                pass
        else:
            if self._fine_prev_timeline_tooltip is not None:
                chk.setToolTip(self._fine_prev_timeline_tooltip)
                self._fine_prev_timeline_tooltip = None
            chk.toggled.connect(self.timeline._on_layout_mode_toggled)
            chk.blockSignals(True)
            chk.setChecked(getattr(self.timeline, "layout_mode", "") == "combined")
            chk.blockSignals(False)

    def _on_fine_single_axis_toggled(self, on: bool) -> None:
        if not getattr(self, "timeline", None):
            return
        self.timeline.layout_mode = "combined" if on else "per_label"
        self._rebuild_timeline_sources()

    def _on_psr_single_timeline_toggled(self, on: bool):
        self._psr_push_undo("toggle_timeline_layout")
        self._psr_set_single_timeline_preference(on)
        self._psr_state_dirty = True
        self._psr_apply_timeline_editability()
        self._psr_refresh_state_timeline(force=True)

    # ---- PSR lifecycle and cache ----
    def _on_psr_asr_asd_activated(self):
        self._ensure_psr_asr_asd_invisible_label()
        if not self.psr_components:
            self._psr_auto_components_from_labels()
        self._psr_mark_dirty()
        self._psr_update_component_panel()

    def _ensure_psr_asr_asd_invisible_label(self, refresh: bool = True):
        name = self._psr_asr_asd_invisible_label
        if any(lb.name == name for lb in self.labels):
            return
        next_id = max([lb.id for lb in self.labels], default=-1) + 1
        self.labels.append(LabelDef(name=name, color_name="Gray", id=next_id))
        if refresh:
            try:
                self.panel.refresh()
            except Exception:
                pass
            try:
                self._rebuild_timeline_sources()
            except Exception:
                pass

    def _is_psr_task(self) -> bool:
        try:
            text = self.combo_task.currentText() or ""
        except Exception:
            text = ""
        lower = text.lower()
        return ("psr/asr/asd" in lower) or ("assembly state" in lower)

    def _psr_mark_dirty(self):
        self._psr_cache_dirty = True
        self._psr_state_dirty = True

    def _psr_set_action_segment_lock(self, enabled: bool) -> None:
        self._psr_snap_to_action_segments = bool(enabled)
        lock_chk = (
            getattr(self.timeline, "chk_action_lock", None)
            if getattr(self, "timeline", None)
            else None
        )
        if lock_chk is not None:
            lock_chk.blockSignals(True)
            lock_chk.setChecked(bool(self._psr_snap_to_action_segments))
            lock_chk.blockSignals(False)
        if self._psr_snap_to_action_segments:
            self._psr_push_undo("lock_to_segment")
            self._psr_snap_manual_events_to_action_segments()
        self._psr_mark_dirty()
        self._psr_refresh_state_timeline(force=True)
        self._psr_update_component_panel()

    def _psr_action_segment_starts_for_snap(self) -> List[int]:
        if self._psr_cache_dirty or not self._psr_action_segment_starts:
            segments = self._psr_collect_segments()
            starts = set()
            ends = set()
            bounds = []
            for seg in segments:
                try:
                    start = int(seg.get("start", 0))
                    end = int(seg.get("end", start))
                except Exception:
                    continue
                if end < start:
                    start, end = end, start
                starts.add(start)
                ends.add(end)
                bounds.append((start, end))
            self._psr_action_segment_starts = sorted(starts)
            self._psr_action_segment_ends = sorted(ends)
            self._psr_action_segments = sorted(bounds, key=lambda x: x[0])
        return list(self._psr_action_segment_starts)

    def _psr_action_segment_for_frame(
        self, frame: int
    ) -> Optional[Tuple[int, int]]:
        try:
            frame = int(frame)
        except Exception:
            return None
        # Ensure cached action bounds are up to date.
        self._psr_action_segment_starts_for_snap()
        for seg in self._psr_action_segments or []:
            try:
                s = int(seg[0])
                e = int(seg[1])
            except Exception:
                continue
            if s <= frame <= e:
                return (s, e)
            if s > frame:
                return (s, e)
        return None

    def _psr_snap_frame_to_action_start(self, frame: int) -> int:
        try:
            frame = int(frame)
        except Exception:
            frame = 0
        starts = self._psr_action_segment_starts_for_snap()
        if not starts:
            return frame
        idx = bisect.bisect_left(starts, frame)
        if idx <= 0:
            return starts[0]
        if idx >= len(starts):
            return starts[-1]
        prev_start = starts[idx - 1]
        next_start = starts[idx]
        if abs(frame - prev_start) <= abs(next_start - frame):
            return prev_start
        return next_start

    def _psr_snap_manual_events_to_action_segments(self) -> None:
        if not self._psr_snap_to_action_segments:
            return
        starts = self._psr_action_segment_starts_for_snap()
        if not starts or not self._psr_manual_events:
            return
        snapped = {}
        for ev in self._psr_manual_events:
            try:
                orig_frame = int(ev.get("frame", 0))
            except Exception:
                orig_frame = 0
            snap_frame = self._psr_snap_frame_to_action_start(orig_frame)
            is_boundary = bool(
                ev.get("force_boundary")
                or str(ev.get("label")) == self._psr_boundary_label
            )
            comp_key = "__boundary__" if is_boundary else str(ev.get("component_id"))
            key = (snap_frame, comp_key)
            prev = snapped.get(key)
            if prev is None or orig_frame >= prev[0]:
                updated = dict(ev)
                updated["frame"] = int(snap_frame)
                snapped[key] = (orig_frame, updated)
        self._psr_manual_events = [
            item[1]
            for item in sorted(
                snapped.values(),
                key=lambda x: (x[1].get("frame", 0), str(x[1].get("component_id"))),
            )
        ]

    # ---- PSR undo / redo snapshot model ----
    def _psr_snapshot_from_view_state(
        self, state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        state = self._psr_clone_view_state(state)
        return {
            "manual_events": copy.deepcopy(state.get("manual_events", [])),
            "components": copy.deepcopy(self.psr_components),
            "rules": copy.deepcopy(self.psr_rules),
            "component_source": self.psr_component_source,
            "rules_path": self.psr_rules_path,
            "single_timeline": bool(
                state.get("single_timeline", self._psr_single_timeline)
            ),
            "state_color_cache": {},
            "combined_label_states": {},
            "gap_spans_combined": copy.deepcopy(
                state.get("gap_spans_combined", [])
            ),
            "gap_spans_by_comp": copy.deepcopy(state.get("gap_spans_by_comp", {})),
            "selected_segment": copy.deepcopy(state.get("selected_segment")),
        }

    def _psr_snapshot(self) -> Dict[str, Any]:
        snap = self._psr_snapshot_from_view_state(
            {
                "manual_events": self._psr_manual_events,
                "gap_spans_combined": self._psr_gap_spans_combined,
                "gap_spans_by_comp": self._psr_gap_spans_by_comp,
                "selected_segment": self._psr_selected_segment,
                "single_timeline": self._psr_single_timeline,
            }
        )
        snap["state_color_cache"] = copy.deepcopy(self._psr_state_color_cache)
        snap["combined_label_states"] = copy.deepcopy(self._psr_combined_label_states)
        return snap

    def _psr_restore_snapshot(self, snap: Dict[str, Any]) -> None:
        if not snap:
            return
        self._psr_undo_block = True
        try:
            self._psr_manual_events = copy.deepcopy(snap.get("manual_events", []))
            self.psr_components = copy.deepcopy(snap.get("components", []))
            self.psr_rules = copy.deepcopy(snap.get("rules", {}))
            self.psr_component_source = snap.get("component_source", "")
            self.psr_rules_path = snap.get("rules_path", "")
            self._psr_single_timeline = bool(
                snap.get("single_timeline", self._psr_single_timeline)
            )
            self._psr_state_color_cache = copy.deepcopy(
                snap.get("state_color_cache", {})
            )
            self._psr_combined_label_states = copy.deepcopy(
                snap.get("combined_label_states", {})
            )
            self._psr_gap_spans_combined = copy.deepcopy(
                snap.get("gap_spans_combined", [])
            )
            self._psr_gap_spans_by_comp = copy.deepcopy(
                snap.get("gap_spans_by_comp", {})
            )
            self._psr_selected_segment = copy.deepcopy(snap.get("selected_segment"))
            self._psr_mark_dirty()
            self._psr_refresh_state_timeline(force=True)
            frame = None
            if self._psr_selected_segment:
                try:
                    frame = int(self._psr_selected_segment.get("start", 0))
                except Exception:
                    frame = None
            self._psr_update_component_panel(frame)
        finally:
            self._psr_undo_block = False

    def _psr_push_undo(self, reason: str = "") -> None:
        if self._psr_undo_block or not self._is_psr_task():
            return
        snap = self._psr_snapshot()
        self._psr_undo_stack.append(snap)
        if len(self._psr_undo_stack) > self._psr_undo_limit:
            self._psr_undo_stack.pop(0)
        self._psr_redo_stack.clear()
        if reason:
            self._log("psr_undo_push", reason=reason)

    def _psr_undo(self) -> None:
        if not self._is_psr_task():
            return
        sync_indices = []
        if not self._psr_sync_apply_in_progress:
            sync_indices = self._psr_sync_target_indices()
        if sync_indices:
            return self._psr_run_across_selected_views(
                self._psr_undo,
                context=self._psr_current_sync_context(),
            )
        if not self._psr_undo_stack:
            return
        current = self._psr_snapshot()
        snap = self._psr_undo_stack.pop()
        self._psr_redo_stack.append(current)
        self._psr_restore_snapshot(snap)
        self._log("psr_undo", remaining=len(self._psr_undo_stack))

    def _psr_redo(self) -> None:
        if not self._is_psr_task():
            return
        sync_indices = []
        if not self._psr_sync_apply_in_progress:
            sync_indices = self._psr_sync_target_indices()
        if sync_indices:
            return self._psr_run_across_selected_views(
                self._psr_redo,
                context=self._psr_current_sync_context(),
            )
        if not self._psr_redo_stack:
            return
        current = self._psr_snapshot()
        snap = self._psr_redo_stack.pop()
        self._psr_undo_stack.append(current)
        self._psr_restore_snapshot(snap)
        self._log("psr_redo", remaining=len(self._psr_redo_stack))

    def _psr_clear_undo(self) -> None:
        self._psr_undo_stack.clear()
        self._psr_redo_stack.clear()

    def _psr_fixed_components(self) -> List[Dict[str, Any]]:
        return [{"id": int(cid), "name": name} for cid, name in HAS_COMPONENT_CATALOG]

    def _psr_component_snapshot(self, comps: Optional[List[Dict[str, Any]]]) -> List[Tuple[int, str]]:
        snap: List[Tuple[int, str]] = []
        for idx, comp in enumerate(comps or []):
            try:
                cid = int(comp.get("id", idx))
            except Exception:
                cid = idx
            name = str(comp.get("name", "")).strip()
            snap.append((cid, name))
        return snap

    @staticmethod
    def _psr_component_name_key(value: Any) -> str:
        key = str(value or "").strip().lower()
        key = key.replace("-", "_").replace(" ", "_")
        key = re.sub(r"_+", "_", key).strip("_")
        # Backward compatibility for legacy bevel-screw naming.
        key = key.replace("screw_bevel_", "bearing_screw_")
        key = key.replace("bevel_screw_", "bearing_screw_")
        key = key.replace("screw_bearing_", "bearing_screw_")
        # Backward compatibility for legacy screw-plate naming.
        key = key.replace("screw_plate_", "screw_adaptor_")
        key = key.replace("plate_screw_", "screw_adaptor_")
        return key

    def _psr_components_match_fixed(self, comps: Optional[List[Dict[str, Any]]]) -> bool:
        return self._psr_component_snapshot(comps) == self._psr_component_snapshot(
            self._psr_fixed_components()
        )

    def _psr_apply_fixed_components(self) -> None:
        self.psr_components = self._psr_fixed_components()
        self.psr_component_source = "fixed_catalog"

    def _psr_cfg(self) -> Dict[str, Any]:
        self._ensure_algo_cfg_defaults()
        cfg = {}
        try:
            cfg = dict(getattr(self, "_algo_cfg", {}).get("psr", {}))
        except Exception:
            cfg = {}
        return cfg

    def _psr_reload_model_registry(self) -> None:
        self._psr_model_specs = load_psr_model_registry()

    def _psr_default_model_type(self) -> str:
        return default_psr_model_id(getattr(self, "_psr_model_specs", None))

    def _psr_normalize_model_type(
        self, value: Any, allow_unknown: bool = True
    ) -> str:
        return normalize_psr_model_type(
            value,
            getattr(self, "_psr_model_specs", None),
            allow_unknown=allow_unknown,
        )

    def _psr_model_display_name(self, value: Any) -> str:
        return psr_model_display_name(value, getattr(self, "_psr_model_specs", None))

    def _populate_psr_model_combo(
        self, combo: QComboBox, current_model: Any
    ) -> str:
        model = self._psr_normalize_model_type(current_model)
        specs = enabled_psr_models(getattr(self, "_psr_model_specs", None))
        combo.blockSignals(True)
        combo.clear()
        for spec in specs:
            model_id = str(spec.get("id") or "").strip()
            if not model_id:
                continue
            display_name = str(spec.get("display_name") or model_id).strip() or model_id
            combo.addItem(display_name, model_id)
        if model and combo.findData(model) < 0:
            combo.addItem(f"Unknown: {model}", model)
        idx = combo.findData(model)
        if idx < 0 and combo.count() > 0:
            idx = 0
            model = str(combo.itemData(0) or self._psr_default_model_type()).strip()
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)
        return model

    @staticmethod
    def _psr_workflow_from_initial_state(
        initial_state: Any, fallback: str = "assemble"
    ) -> str:
        try:
            iv = int(initial_state)
        except Exception:
            iv = None
        if iv == 1:
            return "disassemble"
        if iv == 0:
            return "assemble"
        return str(fallback or "assemble")

    def _psr_on_model_type_changed(self, model_type: str) -> None:
        model = self._psr_normalize_model_type(model_type)
        changed = model != str(
            getattr(self, "_psr_model_type", self._psr_default_model_type())
        )
        self._psr_model_type = model
        self._ensure_algo_cfg_defaults()
        self._algo_cfg.setdefault("psr", {})["model_type"] = model
        if changed:
            self._set_status(f"PSR model type: {self._psr_model_display_name(model)}")
            self._log("psr_model_type", model_type=model)

    def _psr_initial_state_policy(self) -> str:
        policy = str(self._psr_cfg().get("initial_state_policy", "auto")).strip().lower()
        if policy not in {"auto", "installed", "not_installed"}:
            policy = "auto"
        return policy

    def _psr_no_gap_timeline_enabled(self) -> bool:
        return bool(self._psr_cfg().get("no_gap_timeline", True))

    def _psr_auto_carry_next_enabled(self) -> bool:
        return bool(self._psr_cfg().get("auto_carry_next_on_edit", True))

    def _psr_initial_state_value(
        self, segments: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        policy = self._psr_initial_state_policy()
        if policy == "installed":
            self._psr_detected_initial_state = 1
            self._psr_detected_flow = self._psr_workflow_from_initial_state(1)
            return 1
        if policy == "not_installed":
            self._psr_detected_initial_state = 0
            self._psr_detected_flow = self._psr_workflow_from_initial_state(0)
            return 0
        # Auto no longer tries to classify assemble/disassemble due high
        # ambiguity. Default to assemble baseline (all not installed).
        _ = segments
        self._psr_detected_initial_state = 0
        self._psr_detected_flow = self._psr_workflow_from_initial_state(0)
        return 0

    def _psr_initial_state_vector(
        self, components: Optional[List[Dict[str, Any]]] = None, segments=None
    ) -> List[int]:
        comps = list(components or self.psr_components or [])
        init_val = int(self._psr_initial_state_value(segments=segments))
        return [init_val] * len(comps)

    def _psr_state_label_name(self, state_val: int) -> str:
        return self._psr_state_label_map.get(int(state_val), self._psr_state_label_map.get(0, "Not installed"))

    def _psr_has_state_event_at(
        self,
        frame: int,
        component_id: Any,
        events: Optional[List[Dict[str, Any]]] = None,
        include_auto_carried: bool = True,
    ) -> bool:
        try:
            frame = int(frame)
        except Exception:
            return False
        comp_key = str(component_id)
        source = events if events is not None else self._psr_events_cache
        for ev in source or []:
            if str(ev.get("component_id")) != comp_key:
                continue
            if (not include_auto_carried) and bool(ev.get("auto_carried")):
                continue
            try:
                fr = int(ev.get("frame", -1))
            except Exception:
                continue
            if fr == frame:
                return True
        return False

    def _psr_timeline_drag_in_progress(self) -> bool:
        tl = getattr(self, "timeline", None)
        if tl is None:
            return False
        active = getattr(tl, "_active_combined_row", None)
        if active is not None and bool(getattr(active, "_dragging", False)):
            return True
        for row in (getattr(tl, "_combined_rows", []) or []):
            if bool(getattr(row, "_dragging", False)):
                return True
        return False

    def _psr_flush_deferred_timeline_change(self) -> None:
        self._psr_timeline_change_deferred = False
        self._psr_on_state_timeline_changed()

    def _psr_auto_components_from_labels(self):
        # PSR/ASR/ASD uses a fixed HAS component catalog.
        self._psr_apply_fixed_components()

    def _psr_collect_segments(self) -> List[Dict[str, Any]]:
        if not self.views:
            return []
        segments = []
        if self.mode == "Fine":
            view = self.views[self.active_view_idx]
            stores = view.get("entity_stores", {}) or self.entity_stores
            phase_map = view.get("phase_stores", {}) or self.phase_stores
            anom_map = view.get("anomaly_type_stores", {}) or self.anomaly_type_stores
            for ename, st in stores.items():
                pstore = phase_map.get(ename)
                for seg in self._segments_from_store_for_interaction(st):
                    seg["entity"] = ename
                    try:
                        s = int(seg.get("start", 0))
                        e = int(seg.get("end", s))
                    except Exception:
                        s = e = 0
                    if e < s:
                        s, e = e, s
                    if self.phase_mode_enabled:
                        phase_label = self._phase_label_for_span(pstore, s, e)
                        if phase_label:
                            seg["phase"] = phase_label
                        anom_vec = self._anomaly_vector_for_span(
                            ename, s, e, stores=anom_map
                        )
                        if any(anom_vec):
                            seg["anomaly_type"] = anom_vec
                    segments.append(seg)
        else:
            segments = self._segments_from_store_for_interaction(self.store)
        # Strip invalid labels
        clean = []
        for seg in segments:
            lb = seg.get("label")
            if not lb or is_extra_label(str(lb)):
                continue
            clean.append(seg)
        return clean

    def _psr_recompute_cache(self):
        if not self._psr_cache_dirty:
            return
        if not self._psr_components_match_fixed(self.psr_components):
            self._psr_auto_components_from_labels()
        segments = self._psr_collect_segments()
        derived_events = psr_derive_events(
            segments,
            self.psr_components,
            self.psr_rules,
            ignore_labels=[self._psr_asr_asd_invisible_label],
        )
        self._psr_filter_manual_events()
        action_starts = set()
        action_ends = set()
        action_bounds = []
        for seg in segments:
            try:
                start = int(seg.get("start", 0))
                end = int(seg.get("end", start))
            except Exception:
                continue
            if end < start:
                start, end = end, start
            action_starts.add(start)
            action_ends.add(end)
            action_bounds.append((start, end))
        self._psr_action_segment_starts = sorted(action_starts)
        self._psr_action_segment_ends = sorted(action_ends)
        self._psr_action_segments = sorted(action_bounds, key=lambda x: x[0])
        cuts = set()
        for ev in self._psr_manual_events:
            if (
                ev.get("force_boundary")
                or str(ev.get("label")) == self._psr_boundary_label
            ):
                try:
                    cuts.add(int(ev.get("frame", -1)))
                except Exception:
                    continue
        if cuts:
            self._psr_segment_cuts = sorted(c for c in cuts if c >= 0)
        else:
            self._psr_segment_cuts = []
        self._psr_events_cache = self._psr_merge_events(
            derived_events, self._psr_manual_events
        )
        initial_vec = self._psr_initial_state_vector(
            components=self.psr_components, segments=segments
        )
        self._psr_state_seq = psr_build_state_sequence(
            self._psr_events_cache,
            self.psr_components,
            initial_state=initial_vec,
        )
        self._psr_state_frames = [int(x.get("frame", 0)) for x in self._psr_state_seq]
        segment_labels = {str(seg.get("label")) for seg in segments if seg.get("label")}
        event_labels = {
            str(ev.get("label")) for ev in self._psr_events_cache if ev.get("label")
        }
        unmapped = sorted(
            label_name
            for label_name in segment_labels
            if label_name not in event_labels
            and label_name != self._psr_asr_asd_invisible_label
        )
        rule_mismatch = []
        for label, rule in (self.psr_rules or {}).items():
            matches = False
            for comp in rule.get("components") or []:
                comp_id = comp.get("component_id")
                comp_name = comp.get("component")
                if comp_id is not None:
                    if any(
                        str(c.get("id")) == str(comp_id) for c in self.psr_components
                    ):
                        matches = True
                        break
                if comp_name:
                    key = self._psr_component_name_key(comp_name)
                    if any(
                        self._psr_component_name_key(c.get("name", "")) == key
                        for c in self.psr_components
                    ):
                        matches = True
                        break
            if not matches:
                rule_mismatch.append(label)
        diag_initial = int(self._psr_detected_initial_state)
        try:
            view_start, _view_end = self._psr_view_range()
            init_vec = list(self._psr_state_for_frame(int(view_start)))
            if init_vec and all(int(v) == int(init_vec[0]) for v in init_vec):
                val = int(init_vec[0])
                if val in (-1, 0, 1):
                    diag_initial = val
        except Exception:
            pass
        self._psr_detected_initial_state = int(diag_initial)
        self._psr_detected_flow = self._psr_workflow_from_initial_state(
            self._psr_detected_initial_state, fallback=self._psr_detected_flow
        )
        self._psr_diag = {
            "events": len(self._psr_events_cache),
            "unmapped": len(unmapped),
            "rule_mismatch": len(rule_mismatch),
            "flow": self._psr_detected_flow,
            "initial_state": int(self._psr_detected_initial_state),
        }
        self._psr_cache_dirty = False

    def _psr_state_for_frame(self, frame: int) -> List[int]:
        if not self.psr_components:
            return []
        initial_vec = self._psr_initial_state_vector(components=self.psr_components)
        if not self._psr_state_seq:
            return list(initial_vec)
        idx = bisect.bisect_right(self._psr_state_frames, int(frame)) - 1
        if idx < 0:
            return list(initial_vec)
        return list(self._psr_state_seq[idx].get("state", []))

    def _psr_filter_manual_events(self):
        if not self._psr_manual_events:
            return
        id_map = {str(c.get("id")): c.get("id") for c in (self.psr_components or [])}
        cleaned: List[Dict[str, Any]] = []
        for ev in self._psr_manual_events:
            comp_id = ev.get("component_id")
            key = str(comp_id)
            try:
                frame = int(ev.get("frame", 0))
            except Exception:
                continue
            if (
                ev.get("force_boundary")
                or str(ev.get("label")) == self._psr_boundary_label
            ):
                cleaned.append(
                    {
                        "frame": frame,
                        "component_id": None,
                        "state": None,
                        "label": self._psr_boundary_label,
                        "force_boundary": True,
                    }
                )
                continue
            if key not in id_map:
                continue
            try:
                state = int(ev.get("state", 0))
            except Exception:
                state = 0
            if state not in (-1, 0, 1):
                state = 0
            sticky = bool(ev.get("sticky"))
            cleaned.append(
                {
                    "frame": frame,
                    "component_id": id_map[key],
                    "state": state,
                    "label": ev.get("label"),
                    "sticky": sticky,
                    "auto_carried": bool(ev.get("auto_carried")),
                }
            )
        self._psr_manual_events = cleaned

    def _psr_merge_events(
        self, derived: List[Dict[str, Any]], manual: List[Dict[str, Any]]
    ):
        overrides: Dict[Tuple[int, str], Dict[str, Any]] = {}
        sticky_starts: Dict[str, set] = {}
        boundaries: List[int] = []
        for ev in manual:
            try:
                frame = int(ev.get("frame", 0))
            except Exception:
                frame = 0
            comp_key = str(ev.get("component_id"))
            overrides[(frame, comp_key)] = ev
            if ev.get("sticky") and ev.get("component_id") is not None:
                sticky_starts.setdefault(comp_key, set()).add(int(frame))
            if (
                ev.get("force_boundary")
                or str(ev.get("label")) == self._psr_boundary_label
            ):
                boundaries.append(frame)
        boundaries = sorted(set(boundaries))
        sticky_windows: Dict[str, List[Tuple[int, Optional[int]]]] = {}
        for comp_key, starts in sticky_starts.items():
            runs: List[Tuple[int, Optional[int]]] = []
            sorted_starts = sorted(int(s) for s in starts)
            for idx, start in enumerate(sorted_starts):
                stop_sticky = (
                    int(sorted_starts[idx + 1])
                    if idx + 1 < len(sorted_starts)
                    else None
                )
                stop_boundary = None
                if boundaries:
                    bidx = bisect.bisect_right(boundaries, int(start))
                    if bidx < len(boundaries):
                        stop_boundary = int(boundaries[bidx])
                # Keep sticky span semantics purely manual:
                # next sticky start and explicit split boundaries define stop points.
                if stop_sticky is not None:
                    stop = int(stop_sticky)
                    if stop_boundary is not None:
                        stop = min(int(stop), int(stop_boundary))
                elif stop_boundary is not None:
                    stop = int(stop_boundary)
                else:
                    stop = None
                runs.append((int(start), stop))
            sticky_windows[comp_key] = runs

        def covered_by_sticky(frame: int, comp_key: str) -> bool:
            windows = sticky_windows.get(comp_key) or []
            for start, stop in windows:
                if frame < start:
                    break
                if stop is None:
                    return True
                if start <= frame < stop:
                    return True
            return False

        merged: List[Dict[str, Any]] = []
        for ev in derived:
            try:
                frame = int(ev.get("frame", 0))
            except Exception:
                frame = 0
            comp_key = str(ev.get("component_id"))
            if (frame, comp_key) in overrides:
                continue
            if covered_by_sticky(frame, comp_key):
                continue
            merged.append(ev)
        merged.extend(overrides.values())
        merged.sort(key=lambda x: (int(x.get("frame", 0)), str(x.get("component_id"))))
        return merged

    def _psr_view_range(self) -> Tuple[int, int]:
        start = 0
        end = max(0, self._get_frame_count() - 1)
        if self.views and 0 <= self.active_view_idx < len(self.views):
            view = self.views[self.active_view_idx]
            try:
                start = int(view.get("start", start))
            except Exception:
                start = start
            try:
                end = int(view.get("end", end))
            except Exception:
                end = end
        end = max(start, end)
        return start, end

    def _psr_effective_segment_cuts(self) -> List[int]:
        view_start, view_end = self._psr_view_range()
        cuts: Set[int] = set()
        for cut in self._psr_segment_cuts or []:
            try:
                fr = int(cut)
            except Exception:
                continue
            if view_start < fr <= view_end:
                cuts.add(fr)
        return sorted(cuts)

    def _psr_gap_spans_from_store(
        self, store: Optional[AnnotationStore]
    ) -> List[Tuple[int, int]]:
        if store is None:
            return []
        start, end = self._psr_view_range()
        if end < start:
            return []
        frames = sorted(
            f
            for f in store.frame_to_label.keys()
            if isinstance(f, int) and start <= f <= end
        )
        if not frames:
            return [(start, end)] if start <= end else []
        runs = AnnotationStore.frames_to_runs(frames)
        gaps: List[Tuple[int, int]] = []
        cur = start
        for s, e in runs:
            if s > cur:
                gaps.append((cur, s - 1))
            cur = e + 1
        if cur <= end:
            gaps.append((cur, end))
        return gaps

    def _psr_apply_gap_spans_to_store(
        self, store: Optional[AnnotationStore], gaps: List[Tuple[int, int]]
    ) -> None:
        if store is None or not gaps:
            return
        for s, e in gaps:
            try:
                s = int(s)
                e = int(e)
            except Exception:
                continue
            if e < s:
                continue
            for f in range(s, e + 1):
                store.remove_at(f)

    def _psr_remove_gap_range(
        self, gaps: List[Tuple[int, int]], start: int, end: int
    ) -> List[Tuple[int, int]]:
        cleaned: List[Tuple[int, int]] = []
        for s, e in gaps:
            try:
                s = int(s)
                e = int(e)
            except Exception:
                continue
            if end < s or start > e:
                cleaned.append((s, e))
                continue
            if start > s:
                cleaned.append((s, start - 1))
            if end < e:
                cleaned.append((end + 1, e))
        return cleaned

    def _psr_clear_gap_spans(
        self, start: int, end: int, component_id: Optional[Any]
    ) -> None:
        try:
            start = int(start)
            end = int(end)
        except Exception:
            return
        if end < start:
            start, end = end, start
        if component_id is None:
            if not self._psr_gap_spans_combined:
                return
            self._psr_gap_spans_combined = self._psr_remove_gap_range(
                self._psr_gap_spans_combined, start, end
            )
            return
        key = str(component_id)
        gaps = self._psr_gap_spans_by_comp.get(key, [])
        if not gaps:
            return
        updated = self._psr_remove_gap_range(gaps, start, end)
        if updated:
            self._psr_gap_spans_by_comp[key] = updated
        else:
            self._psr_gap_spans_by_comp.pop(key, None)

    def _psr_run_start_for_frame(self, frame: int) -> int:
        start, _end = self._psr_view_range()
        if not self._psr_state_seq:
            return start
        idx = bisect.bisect_right(self._psr_state_frames, int(frame)) - 1
        if idx < 0:
            return start
        try:
            return int(self._psr_state_seq[idx].get("frame", start))
        except Exception:
            return start

    def _psr_build_state_runs(self) -> List[Dict[str, Any]]:
        if not self.psr_components:
            return []
        start, end = self._psr_view_range()
        initial_vec = self._psr_initial_state_vector(components=self.psr_components)
        runs = psr_build_state_runs(
            self._psr_events_cache,
            self.psr_components,
            start,
            end,
            initial_state=initial_vec,
        )
        if not runs and end >= start:
            runs = [
                {
                    "start_frame": start,
                    "end_frame": end,
                    "state": list(initial_vec),
                }
            ]
        return runs

    def _psr_find_adjacent_identical_segments(
        self, include_non_removable: bool = False
    ) -> List[Tuple[int, int, int, int]]:
        if not self.psr_components:
            return []
        self._psr_recompute_cache()
        runs = self._psr_build_state_runs()
        if not runs:
            return []
        cuts = sorted({int(c) for c in (self._psr_segment_cuts or []) if c is not None})
        gaps = []
        for g in self._psr_gap_spans_combined or []:
            try:
                gs = int(g[0])
                ge = int(g[1])
            except Exception:
                continue
            if ge < gs:
                gs, ge = ge, gs
            gaps.append((gs, ge))
        gaps.sort(key=lambda x: x[0])

        def split_by_gaps(seg_start: int, seg_end: int) -> List[Tuple[int, int]]:
            if seg_end < seg_start:
                return []
            parts = []
            cur = seg_start
            for gs, ge in gaps:
                if ge < cur:
                    continue
                if gs > seg_end:
                    break
                if gs > cur:
                    parts.append((cur, gs - 1))
                cur = ge + 1
                if cur > seg_end:
                    break
            if cur <= seg_end:
                parts.append((cur, seg_end))
            return parts

        segments: List[Tuple[int, int, Tuple[int, ...]]] = []
        for run in runs:
            try:
                start = int(run.get("start_frame", 0))
                end = int(run.get("end_frame", start))
            except Exception:
                continue
            if end < start:
                continue
            state_vec = tuple(int(v) for v in (run.get("state") or []))
            split_points = [c for c in cuts if start < c <= end]
            segment_starts = [start] + split_points
            for idx, seg_start in enumerate(segment_starts):
                seg_end = (
                    end
                    if idx == len(segment_starts) - 1
                    else (segment_starts[idx + 1] - 1)
                )
                if seg_end < seg_start:
                    continue
                for part_start, part_end in split_by_gaps(seg_start, seg_end):
                    segments.append((part_start, part_end, state_vec))
        segments.sort(key=lambda x: x[0])
        duplicates: List[Tuple[int, int, int, int]] = []
        prev = None
        for seg in segments:
            if prev and prev[2] == seg[2] and prev[1] + 1 == seg[0]:
                duplicates.append((prev[0], prev[1], seg[0], seg[1]))
            prev = seg
        if include_non_removable or not duplicates:
            return duplicates
        manual_frames: Set[int] = set()
        for ev in self._psr_manual_events or []:
            is_boundary = bool(
                ev.get("force_boundary")
                or str(ev.get("label")) == self._psr_boundary_label
            )
            if not is_boundary:
                continue
            try:
                fr = int(ev.get("frame", -1))
            except Exception:
                continue
            if fr >= 0:
                manual_frames.add(fr)
        return [dup for dup in duplicates if int(dup[2]) in manual_frames]

    @staticmethod
    def _psr_boundary_start_frames(
        duplicates: List[Tuple[int, int, int, int]]
    ) -> Set[int]:
        frames: Set[int] = set()
        for _a, _b, c, _d in duplicates or []:
            try:
                frames.add(int(c))
            except Exception:
                continue
        return frames

    def _psr_merge_adjacent_identical_segments(self) -> None:
        if not self._is_psr_task():
            return
        sync_indices = []
        if not self._psr_sync_apply_in_progress:
            sync_indices = self._psr_sync_target_indices()
        if sync_indices:
            return self._psr_run_across_selected_views(
                self._psr_merge_adjacent_identical_segments,
                context=self._psr_current_sync_context(),
            )
        duplicates = self._psr_find_adjacent_identical_segments()
        if not duplicates:
            if not self._psr_sync_apply_in_progress:
                QMessageBox.information(
                    self,
                    "Info",
                    "No adjacent identical state segments found for merge.",
                )
            return
        boundary_frames = self._psr_boundary_start_frames(duplicates)
        if not boundary_frames:
            return
        kept = []
        removed = 0
        for ev in self._psr_manual_events:
            is_boundary = bool(
                ev.get("force_boundary")
                or str(ev.get("label")) == self._psr_boundary_label
            )
            if not is_boundary:
                kept.append(ev)
                continue
            try:
                fr = int(ev.get("frame", -1))
            except Exception:
                fr = -1
            if fr in boundary_frames:
                removed += 1
                continue
            kept.append(ev)
        if removed <= 0:
            if not self._psr_sync_apply_in_progress:
                QMessageBox.information(
                    self,
                    "Info",
                    "No removable manual boundaries were found in identical segments.",
                )
            return
        self._psr_push_undo("segment_merge_identical")
        self._psr_manual_events = kept
        self._psr_mark_dirty()
        self._psr_refresh_state_timeline(force=True)
        self._psr_update_component_panel()
        self._log(
            "psr_merge_identical_segments",
            removed_boundaries=removed,
            merged_pairs=len(duplicates),
        )

    def _psr_build_component_stores(self, runs: List[Dict[str, Any]]):
        stores: Dict[Any, AnnotationStore] = {}
        comps = list(self.psr_components or [])
        if not comps:
            return stores
        cuts = self._psr_effective_segment_cuts()
        default_state = self._psr_initial_state_value()
        for comp in comps:
            stores[comp.get("id")] = AnnotationStore()
        for run in runs:
            try:
                start = int(run.get("start_frame", 0))
                end = int(run.get("end_frame", start))
            except Exception:
                continue
            state_vec = run.get("state", [])
            split_points = [c for c in cuts if start < c <= end]
            segment_starts = [start] + split_points
            for seg_idx, seg_start in enumerate(segment_starts):
                seg_end = (
                    end
                    if seg_idx == len(segment_starts) - 1
                    else (segment_starts[seg_idx + 1] - 1)
                )
                if seg_end < seg_start:
                    continue
                for idx, comp in enumerate(comps):
                    label = self._psr_state_label_name(default_state)
                    if idx < len(state_vec):
                        try:
                            label = self._psr_state_label_map.get(
                                int(state_vec[idx]), label
                            )
                        except Exception:
                            pass
                    store = stores.get(comp.get("id"))
                    if store is None:
                        continue
                    for f in range(seg_start, seg_end + 1):
                        store.frame_to_label[f] = label
                        store.label_to_frames.setdefault(label, []).append(f)
        for comp in comps:
            comp_id = comp.get("id")
            gaps = self._psr_gap_spans_by_comp.get(str(comp_id), [])
            if gaps:
                self._psr_apply_gap_spans_to_store(stores.get(comp_id), gaps)
        return stores

    def _psr_build_combined_store(self, runs: List[Dict[str, Any]]):
        palette = [k for k in PRESET_COLORS.keys() if k.lower() != "gray"] or list(
            PRESET_COLORS.keys()
        )
        label_defs: List[LabelDef] = []
        label_map: Dict[Tuple[int, ...], LabelDef] = {}
        label_states: Dict[str, List[int]] = {}
        store = AnnotationStore()
        color_cache = self._psr_state_color_cache
        next_id = 1
        cuts = self._psr_effective_segment_cuts()
        for run in runs:
            state_vec = tuple(int(v) for v in (run.get("state") or []))
            color_key = color_cache.get(state_vec)
            if not color_key:
                color_key = palette[(len(color_cache) % len(palette))]
                color_cache[state_vec] = color_key
            try:
                start = int(run.get("start_frame", 0))
                end = int(run.get("end_frame", start))
            except Exception:
                continue
            split_points = [c for c in cuts if start < c <= end]
            segment_starts = [start] + split_points
            for idx, seg_start in enumerate(segment_starts):
                seg_end = (
                    end
                    if idx == len(segment_starts) - 1
                    else (segment_starts[idx + 1] - 1)
                )
                if seg_end < seg_start:
                    continue
                if idx == 0:
                    label_def = label_map.get(state_vec)
                    if label_def is None:
                        label_def = LabelDef(
                            name=f"State {next_id}", color_name=color_key, id=next_id
                        )
                        label_defs.append(label_def)
                        label_map[state_vec] = label_def
                        label_states[label_def.name] = list(state_vec)
                        next_id += 1
                else:
                    label_def = LabelDef(
                        name=f"State {next_id}", color_name=color_key, id=next_id
                    )
                    label_defs.append(label_def)
                    label_states[label_def.name] = list(state_vec)
                    next_id += 1
                label = label_def.name
                for f in range(seg_start, seg_end + 1):
                    store.frame_to_label[f] = label
                    store.label_to_frames.setdefault(label, []).append(f)
        if self._psr_gap_spans_combined:
            self._psr_apply_gap_spans_to_store(store, self._psr_gap_spans_combined)
        self._psr_combined_label_states = label_states
        return label_defs, store

    def _psr_refresh_state_timeline(self, force: bool = False):
        if not self._is_psr_task():
            return
        if not getattr(self, "timeline", None):
            return
        if not force and not (self._psr_state_dirty or self._psr_cache_dirty):
            return
        prev_layout = getattr(self.timeline, "layout_mode", None)
        prev_start = getattr(self.timeline, "view_start", 0)
        prev_span = getattr(self.timeline, "view_span", None)
        self._psr_recompute_cache()
        if self._psr_no_gap_timeline_enabled():
            self._psr_gap_spans_combined = []
            self._psr_gap_spans_by_comp = {}
        try:
            seg_cuts = self._psr_effective_segment_cuts()
            self.timeline.set_segment_cuts(seg_cuts)
        except Exception:
            pass
        try:
            snap_segments = (
                self._psr_action_segments if self._psr_snap_to_action_segments else []
            )
            self.timeline.set_snap_segments(snap_segments)
        except Exception:
            pass
        runs = self._psr_build_state_runs()
        if not runs:
            self._psr_state_dirty = False
            return
        if self._psr_single_timeline or not self.psr_components:
            label_defs, store = self._psr_build_combined_store(runs)
            self._psr_state_store_combined = store
            self._psr_state_stores = {}
            default_label = self._psr_state_label_name(self._psr_initial_state_value())
            try:
                self.timeline.labels = label_defs
            except Exception:
                pass
            row_sources = [(lb, store, "") for lb in label_defs]
            try:
                self.timeline.set_row_sources(row_sources)
            except Exception:
                pass
            try:
                self.timeline.set_combined_groups(
                    [
                        (
                            "State",
                            row_sources,
                            {
                                "psr_no_gap_fill": bool(
                                    self._psr_no_gap_timeline_enabled()
                                ),
                                "psr_default_label": default_label,
                            },
                        )
                    ]
                )
            except Exception:
                pass
            try:
                if getattr(self.timeline, "layout_mode", "") != "combined":
                    self.timeline.set_layout_mode("combined")
            except Exception:
                pass
            try:
                self.timeline.set_combined_label_text(True)
            except Exception:
                pass
            try:
                self.timeline.set_center_single_row(True)
            except Exception:
                pass
        else:
            stores = self._psr_build_component_stores(runs)
            self._psr_state_store_combined = None
            self._psr_state_stores = stores
            default_label = self._psr_state_label_name(self._psr_initial_state_value())
            try:
                self.timeline.labels = list(self._psr_state_label_defs)
            except Exception:
                pass
            groups = []
            for comp in self.psr_components:
                comp_id = comp.get("id")
                title = str(comp.get("name", comp_id))
                store = stores.get(comp_id, AnnotationStore())
                group_sources = [(lb, store, "") for lb in self._psr_state_label_defs]
                groups.append(
                    (
                        title,
                        group_sources,
                        {
                            "psr_component_id": comp_id,
                            "psr_component_name": title,
                            "psr_no_gap_fill": bool(self._psr_no_gap_timeline_enabled()),
                            "psr_default_label": default_label,
                        },
                    )
                )
            try:
                self.timeline.set_combined_groups(groups)
            except Exception:
                pass
            try:
                if getattr(self.timeline, "layout_mode", "") != "combined":
                    self.timeline.set_layout_mode("combined")
            except Exception:
                pass
            try:
                self.timeline.set_combined_label_text(True)
            except Exception:
                pass
            try:
                self.timeline.set_center_single_row(False)
            except Exception:
                pass
        self._psr_apply_timeline_editability()
        if prev_layout == "combined" and prev_span is not None:
            try:
                self.timeline.view_span = prev_span
                self.timeline.view_start = prev_start
                self.timeline._init_sliders()
            except Exception:
                pass
        try:
            self._psr_restore_selected_segment_on_timeline()
        except Exception:
            pass
        self._psr_state_dirty = False

    def _psr_restore_selected_segment_on_timeline(self) -> None:
        seg = getattr(self, "_psr_selected_segment", None)
        if not isinstance(seg, dict):
            return
        tl = getattr(self, "timeline", None)
        scope = self._psr_normalize_selection_scope(seg.get("scope", "segment"))
        comp_id = seg.get("component_id")
        target_row = self._psr_row_for_component(comp_id)
        if target_row is None:
            return
        try:
            start = int(seg.get("start", 0))
        except Exception:
            start = 0
        try:
            end = int(seg.get("end", start))
        except Exception:
            end = start
        label = seg.get("label")
        try:
            if scope in {"segment", "from_here"}:
                rs, re, rl = target_row._segment_at(start)
                start, end, label = int(rs), int(re), rl
            elif scope == "all":
                fc = max(1, self._get_frame_count())
                start, end, label = 0, fc - 1, None
        except Exception:
            pass
        try:
            target_row._selected_interval = (int(start), int(end))
            target_row._selected_label = label
            target_row._selection_scope = scope
            target_row.update()
        except Exception:
            pass
        try:
            tl._active_combined_row = target_row
        except Exception:
            pass
        self._psr_set_selected_segment(
            start, end, label, row=target_row, scope=scope, component_id=comp_id
        )

    def _psr_on_state_changed(self, component_id, state):
        if not self._is_psr_task():
            return
        if not self.views:
            return
        try:
            state_val = int(state)
        except Exception:
            return
        if state_val not in (-1, 0, 1):
            return
        sync_indices = []
        if not self._psr_sync_apply_in_progress:
            sync_indices = self._psr_sync_target_indices()
        if sync_indices:
            context = self._psr_current_sync_context()
            return self._psr_run_across_selected_views(
                lambda: self._psr_on_state_changed(component_id, state_val),
                context=context,
            )
        id_map = {str(c.get("id")): c.get("id") for c in (self.psr_components or [])}
        key = str(component_id)
        if key not in id_map:
            return
        comp_id = id_map[key]
        self._psr_recompute_cache()
        try:
            cur_frame = int(getattr(self.player, "current_frame", 0))
        except Exception:
            cur_frame = 0
        seg = self._psr_sync_selected_segment()
        scope = seg.get("scope") if seg else None
        gap_match = False
        restore_frame = None
        restore_state = None
        if scope == "all":
            self._psr_apply_state_to_all_segments(comp_id, state_val)
            return
        if scope == "from_here":
            seg_start = int(seg.get("start", cur_frame)) if seg else cur_frame
            self._psr_apply_state_from_here(comp_id, state_val, seg_start)
            return
        if scope == "segment" and seg:
            seg_start = int(seg.get("start", cur_frame))
            seg_end = int(seg.get("end", seg_start))
            target_frame = seg_start
            if seg.get("label") is None:
                seg_comp = seg.get("component_id")
                if seg_comp is None or str(seg_comp) == str(comp_id):
                    gap_match = True
                    self._psr_clear_gap_spans(seg_start, seg_end, seg_comp)
            try:
                fc = max(1, self._get_frame_count())
            except Exception:
                fc = None
            next_frame = int(seg_end) + 1
            if fc is None or next_frame <= fc - 1:
                restore_frame = int(next_frame)
        else:
            target_frame = self._psr_run_start_for_frame(cur_frame)
        comp_idx = None
        for idx, comp in enumerate(self.psr_components or []):
            if str(comp.get("id")) == str(comp_id):
                comp_idx = idx
                break
        old_label = None
        if comp_idx is not None and not gap_match:
            current_state = self._psr_state_for_frame(target_frame)
            if (
                comp_idx < len(current_state)
                and int(current_state[comp_idx]) == state_val
            ):
                return
            if comp_idx < len(current_state):
                old_label = self._psr_state_label(current_state[comp_idx])
        elif gap_match:
            old_label = "Gap"
        if scope in (None, "segment"):
            start = int(seg.get("start", target_frame)) if seg else int(target_frame)
            end = int(seg.get("end", start)) if seg else int(target_frame)
            self._psr_record_validation_entry(
                "state_change",
                comp_id,
                start,
                end,
                old_label,
                self._psr_state_label(state_val),
            )

        if restore_frame is not None and comp_idx is not None:
            next_state_vec = self._psr_state_for_frame(int(restore_frame))
            if comp_idx < len(next_state_vec):
                restore_state = self._psr_state_value(
                    next_state_vec[comp_idx], fallback=0
                )
        self._psr_push_undo("state_change")
        # Use non-sticky events + explicit tail restore to avoid unexpected
        # segmentation bursts after panel edits.
        self._psr_set_manual_event(target_frame, comp_id, state_val, sticky=False)
        if (
            restore_frame is not None
            and restore_state is not None
            and int(restore_state) != int(state_val)
        ):
            self._psr_set_manual_event(
                int(restore_frame),
                comp_id,
                int(restore_state),
                sticky=False,
                auto_carried=True,
            )
        self._psr_mark_dirty()
        self._psr_refresh_state_timeline(force=True)
        self._psr_update_component_panel(target_frame)
        self._log(
            "psr_state_change",
            component_id=comp_id,
            frame=target_frame,
            state=state_val,
        )

    def _psr_apply_timeline_editability(self) -> None:
        try:
            self.timeline.set_combined_editable(True)
        except Exception:
            pass
        try:
            self.timeline.set_combined_delete_handler(self._psr_on_state_segment_delete)
        except Exception:
            pass
        try:
            self.timeline.set_combined_split_handler(self._psr_on_state_segment_split)
        except Exception:
            pass

    def _psr_label_to_state(self, label: Optional[str]) -> Optional[int]:
        if label is None:
            return None
        key = str(label).strip().lower()
        for state_val, name in self._psr_state_label_map.items():
            if str(name).strip().lower() == key:
                return int(state_val)
        return None

    def _psr_manual_events_from_store(
        self, component_id: Any, store: AnnotationStore
    ) -> List[Dict[str, Any]]:
        if not store:
            return []
        no_gap = self._psr_no_gap_timeline_enabled()
        runs: List[Tuple[int, int, Any]] = []
        if no_gap:
            view_start, view_end = self._psr_view_range()
            if view_end < view_start:
                return []
            fallback_label = self._psr_state_label_name(self._psr_initial_state_value())
            s = int(view_start)
            cur = store.label_at(s) or fallback_label
            prev_label = cur
            prev_frame = s
            for f in range(s + 1, int(view_end) + 1):
                lb = store.label_at(f)
                if lb is None:
                    lb = prev_label
                else:
                    prev_label = lb
                if lb != cur:
                    runs.append((s, prev_frame, cur))
                    s, cur = f, lb
                prev_frame = f
            runs.append((s, prev_frame, cur))
        else:
            frames = sorted(store.frame_to_label.keys())
            if not frames:
                return []
            s = frames[0]
            cur = store.label_at(s)
            prev = s
            for f in frames[1:]:
                lb = store.label_at(f)
                if lb != cur or f != prev + 1:
                    runs.append((s, prev, cur))
                    s, cur = f, lb
                prev = f
            runs.append((s, prev, cur))
        events = []
        for start, _end, lb in runs:
            state_val = self._psr_label_to_state(lb)
            if state_val is None:
                continue
            events.append(
                {
                    "frame": int(start),
                    "component_id": component_id,
                    "state": int(state_val),
                    "label": "manual",
                    "sticky": True,
                }
            )
        return events

    def _psr_manual_events_from_combined_store(
        self, store: AnnotationStore
    ) -> List[Dict[str, Any]]:
        if not store:
            return []
        comps = list(self.psr_components or [])
        label_states = getattr(self, "_psr_combined_label_states", {}) or {}
        no_gap = self._psr_no_gap_timeline_enabled()
        runs: List[Tuple[int, int, Any]] = []
        if no_gap:
            view_start, view_end = self._psr_view_range()
            if view_end < view_start:
                return []
            s = int(view_start)
            cur = store.label_at(s)
            if cur is None:
                first_frame = None
                for fr in sorted(store.frame_to_label.keys()):
                    if isinstance(fr, int) and view_start <= fr <= view_end:
                        first_frame = int(fr)
                        break
                cur = store.label_at(first_frame) if first_frame is not None else None
            prev_label = cur
            prev_frame = s
            for f in range(s + 1, int(view_end) + 1):
                lb = store.label_at(f)
                if lb is None:
                    lb = prev_label
                else:
                    prev_label = lb
                if lb != cur:
                    runs.append((s, prev_frame, cur))
                    s, cur = f, lb
                prev_frame = f
            runs.append((s, prev_frame, cur))
        else:
            frames = sorted(store.frame_to_label.keys())
            if not frames:
                return []
            s = frames[0]
            cur = store.label_at(s)
            prev = s
            for f in frames[1:]:
                lb = store.label_at(f)
                if lb != cur or f != prev + 1:
                    runs.append((s, prev, cur))
                    s, cur = f, lb
                prev = f
            runs.append((s, prev, cur))
        events = []
        for start, _end, lb in runs:
            state_vec = label_states.get(lb)
            if not state_vec:
                continue
            for idx, comp in enumerate(comps):
                val = state_vec[idx] if idx < len(state_vec) else 0
                if val not in (-1, 0, 1):
                    val = 0
                events.append(
                    {
                        "frame": int(start),
                        "component_id": comp.get("id"),
                        "state": int(val),
                        "label": "manual",
                        "sticky": True,
                    }
                )
        return events

    def _psr_on_state_timeline_changed(self) -> None:
        if not self._is_psr_task():
            return
        if self._psr_timeline_drag_in_progress():
            if not self._psr_timeline_change_deferred:
                self._psr_timeline_change_deferred = True
                QTimer.singleShot(0, self._psr_flush_deferred_timeline_change)
            return
        if self._psr_single_timeline:
            store = self._psr_state_store_combined
            if not store:
                return
            consume = getattr(store, "consume_last_deltas", None)
            if not callable(consume):
                return
            deltas = consume()
            if not deltas:
                return
            self._psr_record_validation_from_deltas(
                None, deltas, action="timeline_edit"
            )
            self._psr_push_undo("timeline_edit")
            self._psr_manual_events = self._psr_manual_events_from_combined_store(store)
            if self._psr_no_gap_timeline_enabled():
                self._psr_gap_spans_combined = []
            else:
                self._psr_gap_spans_combined = self._psr_gap_spans_from_store(store)
            self._psr_snap_manual_events_to_action_segments()
            self._log(
                "psr_state_timeline_edit",
                component_id=None,
                changes=self._delta_frame_count(deltas),
            )
            self._psr_mark_dirty()
            self._psr_refresh_state_timeline(force=True)
            self._psr_update_component_panel()
            return
        changed = False
        for comp_id, store in list(self._psr_state_stores.items()):
            if not store:
                continue
            consume = getattr(store, "consume_last_deltas", None)
            if not callable(consume):
                continue
            deltas = consume()
            if not deltas:
                continue
            if not changed:
                self._psr_push_undo("timeline_edit")
            changed = True
            self._psr_record_validation_from_deltas(
                comp_id, deltas, action="timeline_edit"
            )
            self._psr_manual_events = [
                ev
                for ev in self._psr_manual_events
                if str(ev.get("component_id")) != str(comp_id)
            ]
            self._psr_manual_events.extend(
                self._psr_manual_events_from_store(comp_id, store)
            )
            if self._psr_no_gap_timeline_enabled():
                self._psr_gap_spans_by_comp.pop(str(comp_id), None)
            else:
                self._psr_gap_spans_by_comp[str(comp_id)] = (
                    self._psr_gap_spans_from_store(store)
                )
            self._log(
                "psr_state_timeline_edit",
                component_id=comp_id,
                changes=self._delta_frame_count(deltas),
            )
        if changed:
            self._psr_snap_manual_events_to_action_segments()
            self._psr_mark_dirty()
            self._psr_refresh_state_timeline(force=True)
            self._psr_update_component_panel()
        return

    def _psr_apply_state_from_here(
        self, component_id: Any, state_val: int, start_frame: int
    ) -> None:
        if component_id is None:
            return
        self._psr_recompute_cache()
        runs = self._psr_build_state_runs()
        if not runs:
            return
        self._psr_push_undo("state_change")
        comp_idx = None
        for idx, comp in enumerate(self.psr_components or []):
            if str(comp.get("id")) == str(component_id):
                comp_idx = idx
                break
        if comp_idx is not None and self.validation_enabled:
            for run in runs:
                try:
                    frame = int(run.get("start_frame", 0))
                    end = int(run.get("end_frame", frame))
                except Exception:
                    continue
                if frame < int(start_frame):
                    continue
                state_vec = run.get("state", [])
                if comp_idx >= len(state_vec):
                    continue
                old_val = self._psr_state_value(state_vec[comp_idx], fallback=0)
                if int(old_val) == int(state_val):
                    continue
                self._psr_record_validation_entry(
                    "apply_from",
                    component_id,
                    frame,
                    end,
                    self._psr_state_label(old_val),
                    self._psr_state_label(state_val),
                )
        self._psr_manual_events = [
            ev
            for ev in self._psr_manual_events
            if not (
                str(ev.get("component_id")) == str(component_id)
                and int(ev.get("frame", -1)) >= int(start_frame)
            )
        ]
        for run in runs:
            try:
                frame = int(run.get("start_frame", 0))
            except Exception:
                continue
            if frame < int(start_frame):
                continue
            self._psr_set_manual_event(frame, component_id, state_val)
        self._psr_mark_dirty()
        self._psr_refresh_state_timeline(force=True)
        self._psr_update_component_panel()
        self._log(
            "psr_state_apply_from",
            component_id=component_id,
            frame=start_frame,
            state=state_val,
        )

    def _psr_apply_state_to_all_segments(
        self, component_id: Any, state_val: int
    ) -> None:
        if component_id is None:
            return
        self._psr_push_undo("state_change")
        self._psr_manual_events = [
            ev
            for ev in self._psr_manual_events
            if str(ev.get("component_id")) != str(component_id)
        ]
        self._psr_mark_dirty()
        self._psr_recompute_cache()
        runs = self._psr_build_state_runs()
        if not runs:
            return
        comp_idx = None
        for idx, comp in enumerate(self.psr_components or []):
            if str(comp.get("id")) == str(component_id):
                comp_idx = idx
                break
        if comp_idx is not None and self.validation_enabled:
            for run in runs:
                try:
                    frame = int(run.get("start_frame", 0))
                    end = int(run.get("end_frame", frame))
                except Exception:
                    continue
                state_vec = run.get("state", [])
                if comp_idx >= len(state_vec):
                    continue
                old_val = self._psr_state_value(state_vec[comp_idx], fallback=0)
                if int(old_val) == int(state_val):
                    continue
                self._psr_record_validation_entry(
                    "apply_all",
                    component_id,
                    frame,
                    end,
                    self._psr_state_label(old_val),
                    self._psr_state_label(state_val),
                )
        for run in runs:
            try:
                frame = int(run.get("start_frame", 0))
            except Exception:
                continue
            self._psr_set_manual_event(frame, component_id, state_val)
        self._psr_mark_dirty()
        self._psr_refresh_state_timeline(force=True)
        self._psr_update_component_panel()
        self._log(
            "psr_state_apply_all",
            component_id=component_id,
            state=state_val,
            runs=len(runs),
        )

    def _psr_component_id_from_row(self, row) -> Optional[Any]:
        if row is None:
            return None
        meta = getattr(row, "_group_meta", None)
        if isinstance(meta, dict) and "psr_component_id" in meta:
            return meta.get("psr_component_id")
        return None

    def _psr_row_for_component(self, component_id: Optional[Any]):
        tl = getattr(self, "timeline", None)
        rows = getattr(tl, "_combined_rows", None) if tl is not None else None
        if not rows:
            return None
        if component_id is None:
            return getattr(tl, "_active_combined_row", None) or rows[0]
        key = str(component_id)
        for row in rows:
            meta = getattr(row, "_group_meta", None)
            row_comp = meta.get("psr_component_id") if isinstance(meta, dict) else None
            if str(row_comp) == key:
                return row
        return None

    @staticmethod
    def _psr_normalize_selection_scope(scope: Any) -> str:
        key = str(scope or "segment")
        return key if key in {"all", "segment", "from_here"} else "segment"

    def _psr_build_selected_segment(
        self,
        start: Any,
        end: Any,
        label: Any,
        row=None,
        scope: Any = "segment",
        component_id: Any = None,
    ) -> Dict[str, Any]:
        try:
            start_i = int(start)
        except Exception:
            start_i = 0
        try:
            end_i = int(end)
        except Exception:
            end_i = start_i
        if end_i < start_i:
            start_i, end_i = end_i, start_i
        if component_id is None:
            component_id = self._psr_component_id_from_row(row)
        return {
            "start": int(start_i),
            "end": int(end_i),
            "label": label,
            "component_id": component_id,
            "scope": self._psr_normalize_selection_scope(scope),
        }

    def _psr_set_selected_segment(
        self,
        start: Any,
        end: Any,
        label: Any,
        row=None,
        scope: Any = "segment",
        component_id: Any = None,
    ) -> Dict[str, Any]:
        seg = self._psr_build_selected_segment(
            start,
            end,
            label,
            row=row,
            scope=scope,
            component_id=component_id,
        )
        self._psr_selected_segment = seg
        return seg

    def _psr_remove_manual_events(
        self, frame: int, component_id: Optional[Any] = None
    ) -> int:
        before = len(self._psr_manual_events)
        if component_id is None:
            self._psr_manual_events = [
                ev
                for ev in self._psr_manual_events
                if int(ev.get("frame", -1)) != int(frame)
            ]
        else:
            key = str(component_id)
            self._psr_manual_events = [
                ev
                for ev in self._psr_manual_events
                if not (
                    int(ev.get("frame", -1)) == int(frame)
                    and str(ev.get("component_id")) == key
                )
            ]
        return before - len(self._psr_manual_events)

    def _psr_has_manual_in_range(
        self,
        start: int,
        end: int,
        component_id: Optional[Any],
        events: Optional[List[Dict[str, Any]]] = None,
        include_auto_carried: bool = True,
    ) -> bool:
        try:
            start = int(start)
            end = int(end)
        except Exception:
            return False
        if end < start:
            start, end = end, start
        evs = events if events is not None else self._psr_manual_events
        target_key = None if component_id is None else str(component_id)
        for ev in evs:
            if (
                ev.get("force_boundary")
                or str(ev.get("label")) == self._psr_boundary_label
            ):
                continue
            if (not include_auto_carried) and bool(ev.get("auto_carried")):
                continue
            try:
                fr = int(ev.get("frame", -1))
            except Exception:
                continue
            if target_key is not None and str(ev.get("component_id")) != target_key:
                continue
            if start <= fr <= end:
                return True
            if ev.get("sticky") and fr < start:
                return True
        return False

    def _psr_has_boundary_at(
        self, frame: int, events: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        try:
            frame = int(frame)
        except Exception:
            return False
        evs = events if events is not None else self._psr_manual_events
        for ev in evs:
            if not (
                ev.get("force_boundary")
                or str(ev.get("label")) == self._psr_boundary_label
            ):
                continue
            try:
                fr = int(ev.get("frame", -1))
            except Exception:
                continue
            if fr == frame:
                return True
        return False

    def _psr_add_boundary_if_missing(self, frame: int) -> None:
        try:
            frame = int(frame)
        except Exception:
            return
        if self._psr_has_boundary_at(frame):
            return
        self._psr_manual_events.append(
            {
                "frame": frame,
                "component_id": None,
                "state": None,
                "label": self._psr_boundary_label,
                "force_boundary": True,
            }
        )

    def _psr_set_manual_event(
        self,
        frame: int,
        component_id: Any,
        state_val: int,
        sticky: bool = True,
        auto_carried: bool = False,
    ) -> None:
        orig_frame = frame
        if self._psr_snap_to_action_segments:
            frame = self._psr_snap_frame_to_action_start(frame)
        if orig_frame != frame:
            self._psr_remove_manual_events(orig_frame, component_id)
        self._psr_set_manual_event_no_snap(
            frame,
            component_id,
            state_val,
            sticky=sticky,
            auto_carried=auto_carried,
        )

    def _psr_set_manual_event_no_snap(
        self,
        frame: int,
        component_id: Any,
        state_val: int,
        sticky: bool = True,
        auto_carried: bool = False,
    ) -> None:
        try:
            frame = int(frame)
        except Exception:
            frame = 0
        self._psr_remove_manual_events(frame, component_id)
        self._psr_manual_events.append(
            {
                "frame": int(frame),
                "component_id": component_id,
                "state": int(state_val),
                "label": "manual",
                "sticky": bool(sticky),
                "auto_carried": bool(auto_carried),
            }
        )

    def _psr_on_state_segment_delete(self, start: int, end: int, label, row) -> bool:
        if not self._is_psr_task():
            return False
        sync_indices = []
        if not self._psr_sync_apply_in_progress:
            sync_indices = self._psr_sync_target_indices()
        if sync_indices:
            context = self._psr_sync_context(
                frame=start,
                component_id=self._psr_component_id_from_row(row),
                scope="segment",
            )
            return bool(
                self._psr_run_across_selected_views(
                    lambda: self._psr_on_state_segment_delete(
                        *self._psr_segment_from_sync_context(context)
                    ),
                    context=context,
                )
            )
        try:
            start = int(start)
            end = int(end)
        except Exception:
            return False
        comp_id = self._psr_component_id_from_row(row)
        next_seg = None
        prev_seg = None
        if row is not None:
            try:
                fc = max(1, self._get_frame_count())
            except Exception:
                fc = None
            try:
                if fc is None or end + 1 <= fc - 1:
                    next_seg = row._segment_at(end + 1)
            except Exception:
                next_seg = None
            try:
                if start > 0:
                    prev_seg = row._segment_at(start - 1)
            except Exception:
                prev_seg = None
        if not next_seg and not prev_seg:
            return False
        merge_label = None
        boundary_frame = None
        range_start = int(start)
        range_end = int(end)
        if next_seg and next_seg[2] is not None:
            merge_dir = "next"
            merge_label = next_seg[2]
            boundary_frame = int(next_seg[0])
        elif prev_seg and prev_seg[2] is not None:
            merge_dir = "prev"
            merge_label = prev_seg[2]
            boundary_frame = int(start)
        else:
            return False
        self._psr_push_undo("segment_delete")
        self._psr_recompute_cache()
        event_frames_by_comp: Dict[str, set] = {}
        for ev in self._psr_events_cache or []:
            try:
                fr = int(ev.get("frame", -1))
            except Exception:
                continue
            if fr < range_start or fr > range_end:
                continue
            comp_key = str(ev.get("component_id"))
            event_frames_by_comp.setdefault(comp_key, set()).add(fr)
        removed = 0
        extra_frames = set()
        if boundary_frame is not None and (
            boundary_frame < range_start or boundary_frame > range_end
        ):
            extra_frames.add(int(boundary_frame))

        def add_frame(frameset, fr):
            try:
                fr = int(fr)
            except Exception:
                return
            if fr < range_start or fr > range_end:
                return
            frameset.add(int(fr))

        if comp_id is None:
            state_vec = (
                self._psr_combined_label_states.get(str(merge_label))
                or self._psr_combined_label_states.get(str(label))
                or []
            )
            if not state_vec:
                ref_frame = (
                    boundary_frame
                    if merge_dir == "next" and boundary_frame is not None
                    else range_start
                )
                state_vec = self._psr_state_for_frame(ref_frame)
            comp_ids = [c.get("id") for c in (self.psr_components or [])]
            comp_keys = {str(cid) for cid in comp_ids}
            kept = []
            for ev in self._psr_manual_events:
                try:
                    fr = int(ev.get("frame", -1))
                except Exception:
                    fr = -1
                in_span = (range_start <= fr <= range_end) or (fr in extra_frames)
                is_boundary = (
                    ev.get("force_boundary")
                    or str(ev.get("label")) == self._psr_boundary_label
                )
                if is_boundary and in_span:
                    removed += 1
                    continue
                if in_span and str(ev.get("component_id")) in comp_keys:
                    removed += 1
                    continue
                kept.append(ev)
            self._psr_manual_events = kept
            for j, comp in enumerate(self.psr_components or []):
                state_val = state_vec[j] if j < len(state_vec) else 0
                comp_key = str(comp.get("id"))
                frames = set()
                add_frame(frames, range_start)
                for fr in event_frames_by_comp.get(comp_key, set()):
                    add_frame(frames, fr)
                for fr in sorted(frames):
                    self._psr_set_manual_event_no_snap(
                        fr, comp.get("id"), state_val, sticky=False
                    )
        else:
            state_val = self._psr_label_to_state(merge_label)
            if state_val is None:
                state_val = self._psr_label_to_state(label)
            if state_val is None:
                state_val = 0
            comp_key = str(comp_id)
            kept = []
            for ev in self._psr_manual_events:
                try:
                    fr = int(ev.get("frame", -1))
                except Exception:
                    fr = -1
                in_span = (range_start <= fr <= range_end) or (fr in extra_frames)
                is_boundary = (
                    ev.get("force_boundary")
                    or str(ev.get("label")) == self._psr_boundary_label
                )
                if is_boundary and in_span:
                    removed += 1
                    continue
                if in_span and str(ev.get("component_id")) == comp_key:
                    removed += 1
                    continue
                kept.append(ev)
            self._psr_manual_events = kept
            frames = set()
            add_frame(frames, range_start)
            for fr in event_frames_by_comp.get(comp_key, set()):
                add_frame(frames, fr)
            for fr in sorted(frames):
                self._psr_set_manual_event_no_snap(fr, comp_id, state_val, sticky=False)
        self._psr_record_validation_entry(
            "merge",
            comp_id,
            range_start,
            range_end,
            str(label) if label is not None else "Unlabeled",
            str(merge_label) if merge_label is not None else "Unlabeled",
        )
        self._psr_mark_dirty()
        self._psr_refresh_state_timeline(force=True)
        self._psr_update_component_panel(range_start)
        self._log(
            "psr_state_delete",
            frame=range_start,
            component_id=comp_id,
            removed=removed,
            merge_dir=merge_dir,
            source="right_click",
        )
        return True

    def _psr_on_state_segment_split(self, frame: int, row) -> bool:
        if not self._is_psr_task():
            return False
        sync_indices = []
        if not self._psr_sync_apply_in_progress:
            sync_indices = self._psr_sync_target_indices()
        if sync_indices:
            context = self._psr_sync_context(
                frame=frame,
                component_id=self._psr_component_id_from_row(row),
                scope="segment",
            )
            return bool(
                self._psr_run_across_selected_views(
                    lambda: self._psr_on_state_segment_split(
                        int(context.get("frame", frame)),
                        self._psr_row_for_sync_context(context),
                    ),
                    context=context,
                )
            )
        panel = getattr(self, "psr_embedded", None)
        handler = getattr(panel, "handle_state_segment_split", None)
        if callable(handler):
            return handler(self, frame, row)
        return False

    def _psr_reset_selected_segment(self) -> None:
        if not self._is_psr_task():
            return
        sync_indices = []
        if not self._psr_sync_apply_in_progress:
            sync_indices = self._psr_sync_target_indices()
        if sync_indices:
            return self._psr_run_across_selected_views(
                self._psr_reset_selected_segment,
                context=self._psr_current_sync_context(),
            )
        seg = self._psr_sync_selected_segment()
        if not seg:
            if not self._psr_sync_apply_in_progress:
                QMessageBox.information(
                    self, "Info", "Select a state segment on the timeline first."
                )
            return
        self._psr_push_undo("segment_reset")
        scope = seg.get("scope")
        comp_id = seg.get("component_id")
        if scope == "all":
            if comp_id is None:
                removed = len(self._psr_manual_events)
                self._psr_manual_events = []
            else:
                removed = self._psr_remove_manual_events(0, comp_id)
        elif scope == "from_here":
            start_frame = int(seg.get("start", 0))
            if comp_id is None:
                removed = len(self._psr_manual_events)
                self._psr_manual_events = [
                    ev
                    for ev in self._psr_manual_events
                    if int(ev.get("frame", -1)) < start_frame
                ]
            else:
                key = str(comp_id)
                before = len(self._psr_manual_events)
                self._psr_manual_events = [
                    ev
                    for ev in self._psr_manual_events
                    if not (
                        str(ev.get("component_id")) == key
                        and int(ev.get("frame", -1)) >= start_frame
                    )
                ]
                removed = before - len(self._psr_manual_events)
        else:
            target_frame = int(seg.get("start", 0))
            removed = self._psr_remove_manual_events(target_frame, comp_id)
        range_start = int(seg.get("start", 0))
        range_end = int(seg.get("end", range_start))
        if scope == "all":
            range_start, range_end = self._psr_view_range()
        elif scope == "from_here":
            _view_start, view_end = self._psr_view_range()
            range_end = view_end
        self._psr_record_validation_entry(
            "reset",
            comp_id,
            range_start,
            range_end,
            "manual",
            "derived",
        )
        self._psr_mark_dirty()
        self._psr_refresh_state_timeline(force=True)
        self._psr_update_component_panel()
        self._log(
            "psr_state_reset",
            frame=int(seg.get("start", 0)),
            component_id=comp_id,
            removed=removed,
            source="button",
        )

    def _psr_invert_selected_segment(self) -> None:
        if not self._is_psr_task():
            return
        sync_indices = []
        if not self._psr_sync_apply_in_progress:
            sync_indices = self._psr_sync_target_indices()
        if sync_indices:
            return self._psr_run_across_selected_views(
                self._psr_invert_selected_segment,
                context=self._psr_current_sync_context(),
            )
        seg = self._psr_sync_selected_segment()
        if not seg:
            if not self._psr_sync_apply_in_progress:
                QMessageBox.information(
                    self, "Info", "Select a state segment on the timeline first."
                )
            return
        self._psr_mark_dirty()
        scope = seg.get("scope")
        comp_id = seg.get("component_id")
        self._psr_recompute_cache()
        comps = list(self.psr_components or [])
        changes = []
        pending_boundary_frame = None
        if scope == "all":
            runs = self._psr_build_state_runs()
            for run in runs:
                try:
                    frame = int(run.get("start_frame", 0))
                    end_frame = int(run.get("end_frame", frame))
                except Exception:
                    continue
                state_vec = run.get("state", [])
                if comp_id is None:
                    for idx, comp in enumerate(comps):
                        if idx >= len(state_vec):
                            continue
                        cur = int(state_vec[idx])
                        if cur == -1:
                            continue
                        new_state = 1 if cur == 0 else 0
                        changes.append((frame, comp.get("id"), new_state, True))
                        self._psr_record_validation_entry(
                            "invert",
                            comp.get("id"),
                            frame,
                            end_frame,
                            self._psr_state_label(cur),
                            self._psr_state_label(new_state),
                        )
                else:
                    comp_idx = None
                    for idx, comp in enumerate(comps):
                        if str(comp.get("id")) == str(comp_id):
                            comp_idx = idx
                            break
                    if comp_idx is not None and comp_idx < len(state_vec):
                        cur = int(state_vec[comp_idx])
                        if cur != -1:
                            new_state = 1 if cur == 0 else 0
                            changes.append((frame, comp_id, new_state, True))
                            self._psr_record_validation_entry(
                                "invert",
                                comp_id,
                                frame,
                                end_frame,
                                self._psr_state_label(cur),
                                self._psr_state_label(new_state),
                            )
        elif scope == "from_here":
            start_frame = int(seg.get("start", 0))
            runs = self._psr_build_state_runs()
            for run in runs:
                try:
                    frame = int(run.get("start_frame", 0))
                    end_frame = int(run.get("end_frame", frame))
                except Exception:
                    continue
                if frame < start_frame:
                    continue
                state_vec = run.get("state", [])
                if comp_id is None:
                    for idx, comp in enumerate(comps):
                        if idx >= len(state_vec):
                            continue
                        cur = int(state_vec[idx])
                        if cur == -1:
                            continue
                        new_state = 1 if cur == 0 else 0
                        changes.append((frame, comp.get("id"), new_state, True))
                        self._psr_record_validation_entry(
                            "invert",
                            comp.get("id"),
                            frame,
                            end_frame,
                            self._psr_state_label(cur),
                            self._psr_state_label(new_state),
                        )
                else:
                    comp_idx = None
                    for idx, comp in enumerate(comps):
                        if str(comp.get("id")) == str(comp_id):
                            comp_idx = idx
                            break
                    if comp_idx is not None and comp_idx < len(state_vec):
                        cur = int(state_vec[comp_idx])
                        if cur != -1:
                            new_state = 1 if cur == 0 else 0
                            changes.append((frame, comp_id, new_state, True))
                            self._psr_record_validation_entry(
                                "invert",
                                comp_id,
                                frame,
                                end_frame,
                                self._psr_state_label(cur),
                                self._psr_state_label(new_state),
                            )
        else:
            target_frame = int(seg.get("start", 0))
            seg_end = int(seg.get("end", target_frame))
            state_vec = self._psr_state_for_frame(target_frame)
            manual_snapshot = list(self._psr_manual_events)
            next_start = None
            next_end = None
            row = (
                getattr(self.timeline, "_active_combined_row", None)
                if getattr(self, "timeline", None)
                else None
            )
            if row is not None:
                try:
                    fc = max(1, self._get_frame_count())
                except Exception:
                    fc = None
                try:
                    if fc is None or seg_end + 1 <= fc - 1:
                        ns, ne, nl = row._segment_at(seg_end + 1)
                        if nl is not None:
                            next_start = int(ns)
                            next_end = int(ne)
                except Exception:
                    pass
            if next_start is None:
                try:
                    fc = max(1, self._get_frame_count())
                except Exception:
                    fc = None
                if fc is None or seg_end + 1 <= fc - 1:
                    next_seg = self._psr_action_segment_for_frame(seg_end + 1)
                    if next_seg is not None:
                        next_start = int(next_seg[0])
                        next_end = int(next_seg[1])
            seen = set()
            auto_next_applied = False
            snapped_target = target_frame
            snapped_next = next_start
            if self._psr_snap_to_action_segments:
                try:
                    snapped_target = self._psr_snap_frame_to_action_start(
                        snapped_target
                    )
                except Exception:
                    snapped_target = target_frame
                if snapped_next is not None:
                    try:
                        snapped_next = self._psr_snap_frame_to_action_start(
                            snapped_next
                        )
                    except Exception:
                        snapped_next = next_start

            def add_change(frame, cid, new_state, sticky, auto_carried=False):
                key = (int(frame), str(cid))
                if key in seen:
                    return
                seen.add(key)
                changes.append(
                    (
                        int(frame),
                        cid,
                        int(new_state),
                        bool(sticky),
                        bool(auto_carried),
                    )
                )

            next_state_after = None
            if next_start is not None:
                next_state_after = list(self._psr_state_for_frame(next_start))
            if comp_id is None:
                for idx, comp in enumerate(comps):
                    if idx >= len(state_vec):
                        continue
                    cur = int(state_vec[idx])
                    if cur == -1:
                        continue
                    new_state = 1 if cur == 0 else 0
                    add_change(snapped_target, comp.get("id"), new_state, True)
                    self._psr_record_validation_entry(
                        "invert",
                        comp.get("id"),
                        target_frame,
                        seg_end,
                        self._psr_state_label(cur),
                        self._psr_state_label(new_state),
                    )
                    if (
                        next_start is not None
                        and next_end is not None
                        and next_state_after is not None
                        and not self._psr_has_manual_in_range(
                            next_start,
                            next_end,
                            comp.get("id"),
                            manual_snapshot,
                            include_auto_carried=False,
                        )
                        and not self._psr_has_state_event_at(
                            next_start,
                            comp.get("id"),
                            events=self._psr_events_cache,
                            include_auto_carried=False,
                        )
                    ):
                        next_val = (
                            next_state_after[idx] if idx < len(next_state_after) else 0
                        )
                        if int(next_val) != int(new_state):
                            add_change(
                                snapped_next,
                                comp.get("id"),
                                new_state,
                                True,
                                auto_carried=True,
                            )
                            next_state_after[idx] = int(new_state)
                            auto_next_applied = True
                            self._psr_record_validation_entry(
                                "invert",
                                comp.get("id"),
                                next_start,
                                next_end,
                                self._psr_state_label(next_val),
                                self._psr_state_label(new_state),
                                note="auto-propagate",
                            )
            else:
                comp_idx = None
                for idx, comp in enumerate(comps):
                    if str(comp.get("id")) == str(comp_id):
                        comp_idx = idx
                        break
                if comp_idx is not None and comp_idx < len(state_vec):
                    cur = int(state_vec[comp_idx])
                    if cur != -1:
                        new_state = 1 if cur == 0 else 0
                        add_change(snapped_target, comp_id, new_state, True)
                        self._psr_record_validation_entry(
                            "invert",
                            comp_id,
                            target_frame,
                            seg_end,
                            self._psr_state_label(cur),
                            self._psr_state_label(new_state),
                        )
                        if (
                            next_start is not None
                            and next_end is not None
                            and next_state_after is not None
                            and not self._psr_has_manual_in_range(
                                next_start,
                                next_end,
                                comp_id,
                                manual_snapshot,
                                include_auto_carried=False,
                            )
                            and not self._psr_has_state_event_at(
                                next_start,
                                comp_id,
                                events=self._psr_events_cache,
                                include_auto_carried=False,
                            )
                        ):
                            next_val = (
                                next_state_after[comp_idx]
                                if comp_idx < len(next_state_after)
                                else 0
                            )
                            if int(next_val) != int(new_state):
                                add_change(
                                    snapped_next,
                                    comp_id,
                                    new_state,
                                    True,
                                    auto_carried=True,
                                )
                                next_state_after[comp_idx] = int(new_state)
                                auto_next_applied = True
                                self._psr_record_validation_entry(
                                    "invert",
                                    comp_id,
                                    next_start,
                                    next_end,
                                    self._psr_state_label(next_val),
                                    self._psr_state_label(new_state),
                                    note="auto-propagate",
                                )
            if auto_next_applied and snapped_next is not None:
                if not self._psr_has_boundary_at(snapped_next, manual_snapshot):
                    pending_boundary_frame = int(snapped_next)
            # always cap the current segment so sticky edits do not leak forward
            try:
                fc = max(1, self._get_frame_count())
            except Exception:
                fc = None
            boundary_after_seg = seg_end + 1
            if fc is None or boundary_after_seg <= fc - 1:
                pending_boundary_frame = pending_boundary_frame or boundary_after_seg

        if not changes:
            if not self._psr_sync_apply_in_progress:
                QMessageBox.information(
                    self, "Info", "Nothing to invert for the selected segment."
                )
            return

        self._psr_push_undo("segment_invert")
        if pending_boundary_frame is not None:
            self._psr_add_boundary_if_missing(int(pending_boundary_frame))
        for change in changes:
            try:
                frame, cid, new_state, sticky = change[:4]
            except Exception:
                continue
            auto_carried = bool(change[4]) if len(change) > 4 else False
            self._psr_set_manual_event(
                frame,
                cid,
                new_state,
                sticky=sticky,
                auto_carried=auto_carried,
            )

        self._psr_mark_dirty()
        self._psr_refresh_state_timeline(force=True)
        self._psr_update_component_panel(int(seg.get("start", 0)))
        self._log(
            "psr_state_invert",
            frame=int(seg.get("start", 0)),
            component_id=comp_id,
            count=len(changes),
        )

    def _psr_select_from_here(self) -> None:
        if not self._is_psr_task():
            return
        seg = self._psr_sync_selected_segment()
        if not seg:
            if not self._psr_sync_apply_in_progress:
                QMessageBox.information(
                    self, "Info", "Select a state segment on the timeline first."
                )
            return
        self._psr_selected_segment["scope"] = "from_here"
        row = self._psr_row_for_component(self._psr_selected_segment.get("component_id"))
        if row is None:
            row = getattr(getattr(self, "timeline", None), "_active_combined_row", None)
        if row is not None:
            try:
                row._selection_scope = "from_here"
                row.update()
            except Exception:
                pass
        self._psr_update_component_panel(force=True)
        self._log(
            "psr_select_from_here",
            frame=int(self._psr_selected_segment.get("start", 0)),
            component_id=self._psr_selected_segment.get("component_id"),
        )

    def _psr_select_segment_only(self) -> None:
        if not self._is_psr_task():
            return
        seg = self._psr_sync_selected_segment()
        if not seg:
            if not self._psr_sync_apply_in_progress:
                QMessageBox.information(
                    self, "Info", "Select a state segment on the timeline first."
                )
            return
        self._psr_selected_segment["scope"] = "segment"
        row = self._psr_row_for_component(self._psr_selected_segment.get("component_id"))
        if row is None:
            row = getattr(getattr(self, "timeline", None), "_active_combined_row", None)
        if row is not None:
            try:
                row._selection_scope = "segment"
                row.update()
            except Exception:
                pass
        self._psr_update_component_panel(force=True)
        self._log(
            "psr_select_segment",
            frame=int(self._psr_selected_segment.get("start", 0)),
            component_id=self._psr_selected_segment.get("component_id"),
        )

    def _psr_split_at_playhead(self) -> None:
        if not self._is_psr_task():
            return
        timeline = getattr(self, "timeline", None)
        row = getattr(timeline, "_active_combined_row", None) if timeline is not None else None
        if row is None and timeline is not None:
            rows = getattr(timeline, "rows", []) or []
            if rows:
                row = rows[0]
        if row is None:
            self._set_status("No editable PSR state row is available for split.")
            return
        try:
            frame = int(getattr(self.player, "current_frame", 0))
        except Exception:
            frame = 0
        if not self._psr_on_state_segment_split(frame, row):
            self._set_status("Unable to split state segment at the current frame.")

    # ---- PSR selection and right panel sync ----
    def _psr_sync_selected_segment(self) -> Optional[Dict[str, Any]]:
        if not self._is_psr_task():
            self._psr_selected_segment = None
            return None
        row = (
            getattr(self.timeline, "_active_combined_row", None)
            if getattr(self, "timeline", None)
            else None
        )
        interval = getattr(row, "_selected_interval", None) if row is not None else None
        if not interval:
            cached = self._psr_selected_segment
            if isinstance(cached, dict):
                scope = self._psr_normalize_selection_scope(
                    cached.get("scope", "segment")
                )
                comp_id = cached.get("component_id")
                target_row = self._psr_row_for_component(comp_id) or row
                try:
                    start = int(cached.get("start", 0))
                except Exception:
                    start = 0
                try:
                    end = int(cached.get("end", start))
                except Exception:
                    end = start
                label = cached.get("label")
                if target_row is not None and scope in {"segment", "from_here"}:
                    try:
                        rs, re, rl = target_row._segment_at(start)
                        start, end, label = int(rs), int(re), rl
                    except Exception:
                        pass
                    try:
                        target_row._selected_interval = (int(start), int(end))
                        target_row._selected_label = label
                        target_row._selection_scope = scope
                        target_row.update()
                    except Exception:
                        pass
                    try:
                        self.timeline._active_combined_row = target_row
                    except Exception:
                        pass
                return self._psr_set_selected_segment(
                    start,
                    end,
                    label,
                    row=target_row,
                    scope=scope,
                    component_id=comp_id,
                )
            if row is not None:
                try:
                    frame = int(getattr(self.player, "current_frame", 0))
                except Exception:
                    frame = 0
                try:
                    s, e, lb = row._segment_at(frame)
                    return self._psr_set_selected_segment(
                        int(s), int(e), lb, row=row, scope="segment"
                    )
                except Exception:
                    pass
            self._psr_selected_segment = None
            return None
        try:
            start = int(interval[0])
            end = int(interval[1])
        except Exception:
            self._psr_selected_segment = None
            return None
        label = getattr(row, "_selected_label", None)
        scope = (
            getattr(row, "_selection_scope", "segment")
            if row is not None
            else "segment"
        )
        if (
            self._psr_selected_segment
            and self._psr_selected_segment.get("scope") == "from_here"
        ):
            try:
                if int(self._psr_selected_segment.get("start", -1)) == start:
                    scope = "from_here"
            except Exception:
                pass
        return self._psr_set_selected_segment(
            start, end, label, row=row, scope=scope
        )

    def _psr_panel_frame(self, frame: Optional[int], force: bool = False) -> int:
        if not force:
            seg = getattr(self, "_psr_selected_segment", None)
            if isinstance(seg, dict) and seg.get("scope") in {"segment", "from_here"}:
                try:
                    return int(seg.get("start", 0))
                except Exception:
                    pass
        if frame is None:
            return int(getattr(self.player, "current_frame", 0))
        try:
            return int(frame)
        except Exception:
            return int(getattr(self.player, "current_frame", 0))

    def _psr_state_summary_text(
        self,
        components: List[Dict[str, Any]],
        state: List[int],
        scope: str,
        seg_start: int,
        seg_end: int,
        delta_count: int,
    ) -> str:
        installed: List[str] = []
        not_installed: List[str] = []
        errors: List[str] = []
        for idx, comp in enumerate(components or []):
            try:
                value = int(state[idx]) if idx < len(state) else 0
            except Exception:
                value = 0
            name = str(comp.get("name", comp.get("id", idx)))
            if value == 1:
                installed.append(name)
            elif value == -1:
                errors.append(name)
            else:
                not_installed.append(name)
        if scope == "segment":
            scope_txt = f"Segment [{int(seg_start)}-{int(seg_end)}], delta={int(delta_count)}"
        elif scope == "from_here":
            scope_txt = f"From frame {int(seg_start)}"
        elif scope == "all":
            scope_txt = "All segments"
        else:
            scope_txt = f"Frame {int(seg_start)}"
        return (
            f"{scope_txt}\n"
            f"Installed: {len(installed)} | Not installed: {len(not_installed)} | Error: {len(errors)}"
        )

    def _psr_panel_state_context(self, frame: int) -> Dict[str, Any]:
        comps = list(self.psr_components or [])
        cur_state = list(self._psr_state_for_frame(frame))
        if len(cur_state) < len(comps):
            cur_state.extend([0] * (len(comps) - len(cur_state)))
        seg = self._psr_sync_selected_segment()
        scope = "segment"
        seg_start = int(frame)
        seg_end = int(frame)
        if isinstance(seg, dict):
            scope = self._psr_normalize_selection_scope(seg.get("scope", "segment"))
            try:
                seg_start = int(seg.get("start", frame))
            except Exception:
                seg_start = int(frame)
            try:
                seg_end = int(seg.get("end", seg_start))
            except Exception:
                seg_end = seg_start
        view_start, _view_end = self._psr_view_range()
        if scope == "segment":
            if seg_start <= view_start:
                prev_state = list(self._psr_initial_state_vector(components=comps))
            else:
                prev_state = list(self._psr_state_for_frame(seg_start - 1))
        else:
            prev_state = list(cur_state)
        if len(prev_state) < len(comps):
            prev_state.extend([0] * (len(comps) - len(prev_state)))
        delta_ids: List[Any] = []
        if scope == "segment":
            for idx, comp in enumerate(comps):
                cur_val = int(cur_state[idx]) if idx < len(cur_state) else 0
                prev_val = int(prev_state[idx]) if idx < len(prev_state) else 0
                if cur_val != prev_val:
                    delta_ids.append(comp.get("id"))
        summary_text = self._psr_state_summary_text(
            comps,
            cur_state,
            scope=scope,
            seg_start=seg_start,
            seg_end=seg_end,
            delta_count=len(delta_ids),
        )
        return {
            "components": comps,
            "state": cur_state,
            "delta_component_ids": delta_ids,
            "scope": scope,
            "summary_text": summary_text,
        }

    def _psr_update_component_panel(
        self, frame: Optional[int] = None, force: bool = False
    ):
        if not self._is_psr_task():
            return
        if not getattr(self, "psr_embedded", None):
            return
        # Avoid rebuilding PSR timeline rows while user is dragging a boundary.
        if not force and self._psr_timeline_drag_in_progress():
            return
        frame = self._psr_panel_frame(frame, force=force)
        self._psr_refresh_state_timeline()
        ctx = self._psr_panel_state_context(frame)
        try:
            self.psr_embedded.update_component_states(
                ctx.get("components", self.psr_components),
                ctx.get("state", []),
                rules_count=len(self.psr_rules),
                source=self.psr_component_source,
                diagnostics=self._psr_diag,
                delta_component_ids=ctx.get("delta_component_ids"),
                segment_scope=ctx.get("scope", "segment"),
                state_summary=ctx.get("summary_text", ""),
            )
        except Exception:
            pass

    def _psr_import_state_from_asr_payload(
        self, data: Dict[str, Any], target_view_idx: int
    ) -> Tuple[Dict[str, Any], int, int]:
        if not self.psr_components:
            self._psr_auto_components_from_labels()
        target_components = list(self.psr_components or [])
        if not target_components:
            raise ValueError("No ASR components available.")

        seq_raw = data.get("state_sequence")
        changes_raw = data.get("state_changes")
        if not isinstance(seq_raw, list) and not isinstance(changes_raw, list):
            raise ValueError(
                "ASR JSON must contain 'state_sequence' or 'state_changes'."
            )

        def _to_int(value: Any, fallback: Optional[int] = None) -> Optional[int]:
            try:
                return int(value)
            except Exception:
                return fallback

        def _norm_key(value: Any) -> str:
            txt = self._psr_component_name_key(value)
            txt = txt.replace("_", " ")
            txt = re.sub(r"\s+", " ", txt).strip()
            return txt

        target_id_by_str = {str(c.get("id")): c.get("id") for c in target_components}
        target_name_to_id = {
            _norm_key(c.get("name", c.get("id"))): c.get("id")
            for c in target_components
        }
        source_components = []
        raw_components = data.get("components")
        if not isinstance(raw_components, list):
            raw_components = data.get("component_order")
        if isinstance(raw_components, list):
            for idx, item in enumerate(raw_components):
                if isinstance(item, dict):
                    source_components.append(
                        {
                            "id": item.get("id", idx),
                            "name": str(item.get("name", "")),
                        }
                    )
                else:
                    source_components.append({"id": idx, "name": str(item)})

        state_seq_points: List[Tuple[int, List[Any]]] = []
        if isinstance(seq_raw, list):
            for item in seq_raw:
                if not isinstance(item, dict):
                    continue
                frame = _to_int(item.get("frame"), None)
                if frame is None:
                    continue
                state_vec = item.get("state")
                if not isinstance(state_vec, list):
                    state_vec = []
                state_seq_points.append((max(0, int(frame)), list(state_vec)))

        meta_in = data.get("meta_data") if isinstance(data.get("meta_data"), dict) else {}
        initial_vec_raw = data.get("initial_state_vector")
        initial_state_src = data.get("initial_state")
        if initial_state_src in (None, ""):
            initial_state_src = meta_in.get("initial_state")
        initial_state_val = self._psr_state_value(initial_state_src, fallback=0)
        source_width = max(
            len(source_components),
            max((len(v) for _f, v in state_seq_points), default=0),
            len(initial_vec_raw) if isinstance(initial_vec_raw, list) else 0,
            1,
        )
        idx_to_target: Dict[int, Any] = {}
        used_targets: Set[str] = set()
        for idx in range(source_width):
            source_id = source_components[idx]["id"] if idx < len(source_components) else idx
            source_name = (
                source_components[idx]["name"] if idx < len(source_components) else ""
            )
            target_id = None
            if source_id is not None and str(source_id) in target_id_by_str:
                target_id = target_id_by_str[str(source_id)]
            if target_id is None and source_name:
                target_id = target_name_to_id.get(_norm_key(source_name))
            if target_id is None and idx < len(target_components):
                target_id = target_components[idx].get("id")
            if target_id is None or str(target_id) in used_targets:
                continue
            used_targets.add(str(target_id))
            idx_to_target[idx] = target_id
        if not idx_to_target:
            raise ValueError("Failed to map ASR components to current component catalog.")

        if not (0 <= int(target_view_idx) < len(self.views)):
            raise ValueError("Target view index out of range.")
        view = self.views[int(target_view_idx)]
        try:
            view_start = int(view.get("start", 0))
        except Exception:
            view_start = 0
        try:
            view_end = int(view.get("end", view_start))
        except Exception:
            view_end = view_start
        if view_end < view_start:
            view_end = view_start

        imported_events: List[Dict[str, Any]] = []
        if state_seq_points:
            dedup_points: Dict[int, List[Any]] = {}
            for frame, vec in state_seq_points:
                dedup_points[int(frame)] = list(vec)
            points = sorted(dedup_points.items(), key=lambda x: int(x[0]))
            initial_vec: List[int] = [int(initial_state_val)] * source_width
            if isinstance(initial_vec_raw, list):
                for idx, raw in enumerate(initial_vec_raw[:source_width]):
                    initial_vec[idx] = self._psr_state_value(
                        raw, fallback=initial_vec[idx]
                    )
            if not points or int(points[0][0]) != 0:
                points.insert(0, (0, list(initial_vec)))
            prev_by_target: Dict[str, int] = {}
            for rel_frame, raw_state in points:
                abs_frame = int(view_start) + int(rel_frame)
                if abs_frame < int(view_start) or abs_frame > int(view_end):
                    continue
                for src_idx, target_id in idx_to_target.items():
                    fallback_state = (
                        initial_vec[src_idx]
                        if src_idx < len(initial_vec)
                        else int(initial_state_val)
                    )
                    prev_state = prev_by_target.get(str(target_id), fallback_state)
                    if src_idx < len(raw_state):
                        state_val = self._psr_state_value(
                            raw_state[src_idx], fallback=prev_state
                        )
                    else:
                        state_val = prev_state
                    if str(target_id) not in prev_by_target or int(prev_state) != int(
                        state_val
                    ):
                        imported_events.append(
                            {
                                "frame": int(abs_frame),
                                "component_id": target_id,
                                "state": int(state_val),
                                "sticky": True,
                            }
                        )
                    prev_by_target[str(target_id)] = int(state_val)
        else:
            for item in changes_raw:
                if not isinstance(item, dict):
                    continue
                rel_frame = _to_int(item.get("frame"), None)
                if rel_frame is None:
                    continue
                abs_frame = int(view_start) + int(rel_frame)
                if abs_frame < int(view_start) or abs_frame > int(view_end):
                    continue
                target_id = None
                comp_key = str(item.get("component_id"))
                if comp_key in target_id_by_str:
                    target_id = target_id_by_str[comp_key]
                else:
                    idx_guess = _to_int(item.get("component_id"), None)
                    if idx_guess is not None and idx_guess in idx_to_target:
                        target_id = idx_to_target[idx_guess]
                if target_id is None:
                    continue
                state_val = self._psr_state_value(item.get("state"), fallback=0)
                imported_events.append(
                    {
                        "frame": int(abs_frame),
                        "component_id": target_id,
                        "state": int(state_val),
                        "sticky": True,
                    }
                )

        if not imported_events:
            raise ValueError(
                "No ASR state events in file overlap with the target view range."
            )

        dedup_events: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for ev in imported_events:
            key = (int(ev.get("frame", 0)), str(ev.get("component_id")))
            dedup_events[key] = ev
        events_final = sorted(
            dedup_events.values(),
            key=lambda ev: (int(ev.get("frame", 0)), str(ev.get("component_id"))),
        )
        focus_frame = int(events_final[0].get("frame", view_start))
        state = self._psr_empty_view_state()
        state["manual_events"] = copy.deepcopy(events_final)
        return state, len(used_targets), focus_frame

    def _psr_focus_loaded_view_state(self, focus_frame: int) -> None:
        self._psr_mark_dirty()
        self._psr_refresh_state_timeline(force=True)
        self._psr_selected_segment = None
        timeline = getattr(self, "timeline", None)
        row = (
            getattr(timeline, "_active_combined_row", None) if timeline is not None else None
        )
        if row is None and timeline is not None:
            rows = getattr(timeline, "rows", []) or []
            if rows:
                row = rows[0]
        if row is not None:
            try:
                s, e, lb = row._segment_at(int(focus_frame))
                row._selected_interval = (int(s), int(e))
                row._selected_label = lb
                row._selection_scope = "segment"
                row.update()
                try:
                    timeline._active_combined_row = row
                except Exception:
                    pass
                self._psr_set_selected_segment(
                    int(s), int(e), lb, row=row, scope="segment"
                )
            except Exception:
                self._psr_selected_segment = None
        self._psr_update_component_panel(int(focus_frame), force=True)

    def _load_psr_components(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Load components",
            "",
            "Component Files (*.json *.yaml *.yml *.txt *.csv);;All Files (*)",
        )
        if not fp:
            return
        try:
            comps = psr_load_components(fp)
        except Exception as ex:
            QMessageBox.warning(
                self, "Error", f"Failed to parse components file:\n{ex}"
            )
            return
        if not comps:
            QMessageBox.information(self, "Info", "No components found in the file.")
            return
        loaded_matches_fixed = self._psr_components_match_fixed(comps)
        if not loaded_matches_fixed:
            QMessageBox.information(
                self,
                "Fixed components",
                "PSR components are fixed to the HAS catalog. "
                "The loaded file is ignored for inference.",
            )
        if not self._psr_components_match_fixed(self.psr_components):
            self._psr_push_undo("load_components")
        self._psr_apply_fixed_components()
        self._psr_mark_dirty()
        self._psr_update_component_panel()
        if loaded_matches_fixed:
            self._set_status(f"Verified fixed component catalog from {os.path.basename(fp)}")
        else:
            self._set_status("Using fixed HAS component catalog")

    def _save_psr_components(self):
        if not self.psr_components:
            QMessageBox.information(self, "Info", "No components to save.")
            return
        default_name = "psr_components.json"
        fp, _ = QFileDialog.getSaveFileName(
            self,
            "Save components",
            default_name,
            "JSON Files (*.json);;CSV Files (*.csv);;Text Files (*.txt);;All Files (*)",
        )
        if not fp:
            return
        comps = [
            {"id": c.get("id"), "name": c.get("name", "")}
            for c in (self.psr_components or [])
        ]
        ext = os.path.splitext(fp)[1].lower()
        try:
            if ext in (".txt", ".csv"):
                delimiter = "," if ext == ".csv" else "\t"
                with open(fp, "w", encoding="utf-8") as f:
                    for comp in comps:
                        cid = comp.get("id")
                        name = comp.get("name", "")
                        if cid is None:
                            cid = ""
                        f.write(f"{cid}{delimiter}{name}\n")
            else:
                payload = {"components": comps}
                with open(fp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
            self.psr_component_source = os.path.basename(fp)
            self._psr_update_component_panel()
            self._set_status(f"Saved {len(comps)} components")
            self._log("save_psr_components", path=fp, count=len(comps))
        except Exception as ex:
            QMessageBox.warning(self, "Error", f"Failed to write components:\n{ex}")

    def _load_psr_rules(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Load rules",
            "",
            "JSON/YAML Files (*.json *.yaml *.yml);;All Files (*)",
        )
        if not fp:
            return
        try:
            rules = psr_load_rules(fp)
        except Exception as ex:
            QMessageBox.warning(self, "Error", f"Failed to parse rules file:\n{ex}")
            return
        if not rules:
            QMessageBox.information(self, "Info", "No rules found in the file.")
            return
        self._psr_push_undo("load_rules")
        self.psr_rules = rules
        self.psr_rules_path = fp
        self._psr_mark_dirty()
        self._psr_update_component_panel()
        self._set_status(f"Loaded {len(rules)} rules")

    def _load_psr_asr_json(self):
        if not self._is_psr_task():
            QMessageBox.information(
                self,
                "Info",
                "Switch to the Assembly State task before loading state JSON.",
            )
            return
        if not self.views:
            QMessageBox.information(self, "Info", "Load a video first.")
            return
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Load Assembly State JSON",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not fp:
            return

        try:
            with open(fp, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception as ex:
            QMessageBox.warning(
                self, "Error", f"Failed to read assembly-state JSON:\n{ex}"
            )
            return
        if not isinstance(data, dict):
            QMessageBox.warning(
                self,
                "Error",
                "Invalid assembly-state JSON root (expected object).",
            )
            return
        meta_in = data.get("meta_data") if isinstance(data.get("meta_data"), dict) else {}
        model_type_in = data.get("model_type")
        if model_type_in in (None, ""):
            model_type_in = meta_in.get("model_type")
        if model_type_in not in (None, ""):
            model_norm = self._psr_normalize_model_type(model_type_in)
            self._psr_model_type = model_norm
            self._ensure_algo_cfg_defaults()
            self._algo_cfg.setdefault("psr", {})["model_type"] = model_norm
            panel = getattr(self, "psr_embedded", None)
            if panel is not None and callable(getattr(panel, "set_model_type", None)):
                try:
                    panel.set_model_type(model_norm, emit=False)
                except Exception:
                    pass
        indices = self._selected_view_indices_for_json_io()
        if not indices:
            QMessageBox.information(self, "Info", "No view selected.")
            return
        self._psr_store_active_view_state()

        loaded_views: List[str] = []
        active_focus_frame = None
        total_events = 0
        components_mapped = 0
        first_error = ""
        for idx in indices:
            if not (0 <= int(idx) < len(self.views)):
                continue
            view = self.views[int(idx)]
            view_name = self._effective_view_name(view, idx=int(idx))
            current_state = self._psr_ensure_view_state(view)
            try:
                new_state, mapped_count, focus_frame = self._psr_import_state_from_asr_payload(
                    data, int(idx)
                )
            except Exception as ex:
                if not first_error:
                    first_error = f"{view_name}: {ex}"
                continue
            undo_stack = list(current_state.get("undo_stack") or [])
            undo_stack.append(self._psr_snapshot_from_view_state(current_state))
            if len(undo_stack) > self._psr_undo_limit:
                undo_stack = undo_stack[-self._psr_undo_limit :]
            new_state["single_timeline"] = bool(
                current_state.get("single_timeline", self._psr_single_timeline)
            )
            new_state["undo_stack"] = undo_stack
            new_state["redo_stack"] = []
            view["psr_state"] = self._psr_clone_view_state(new_state)
            loaded_views.append(view_name)
            total_events += len(new_state.get("manual_events", []))
            components_mapped = max(int(components_mapped), int(mapped_count))
            if int(idx) == int(self.active_view_idx):
                active_focus_frame = int(focus_frame)

        if not loaded_views:
            QMessageBox.warning(
                self,
                "Error",
                first_error or "Failed to load ASR JSON for the selected view(s).",
            )
            return

        self._psr_load_view_state(self.views[self.active_view_idx])
        if active_focus_frame is None:
            try:
                active_focus_frame = int(getattr(self.player, "current_frame", 0))
            except Exception:
                active_focus_frame = 0
        self._psr_focus_loaded_view_state(int(active_focus_frame))
        shown = ",".join(loaded_views[:4])
        if len(loaded_views) > 4:
            shown = f"{shown},+{len(loaded_views)-4}"
        self._set_status(
            f"Loaded assembly-state JSON for {len(loaded_views)} view(s): {os.path.basename(fp)}"
        )
        self._log(
            "load_psr_asr_json",
            path=fp,
            events=int(total_events),
            components_mapped=int(components_mapped),
            views=shown,
            count=len(loaded_views),
        )

    def _open_psr_rules_editor(self):
        if not self.psr_components:
            self._psr_auto_components_from_labels()
        labels = [
            lb.name
            for lb in self.labels
            if not is_extra_label(lb.name)
            and lb.name != self._psr_asr_asd_invisible_label
        ]
        if not labels:
            QMessageBox.information(self, "Info", "No labels available to build rules.")
            return
        if not self.psr_components:
            QMessageBox.information(
                self, "Info", "No components available. Load a component file first."
            )
            return
        dlg = PSRRulesDialog(
            self, labels=labels, components=self.psr_components, rules=self.psr_rules
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        self._psr_push_undo("edit_rules")
        self.psr_rules = dlg.get_rules()
        self.psr_rules_path = ""
        self._psr_mark_dirty()
        self._psr_update_component_panel()
        self._set_status(f"Rules updated: {len(self.psr_rules)} entries")

    def _apply_psr_rules(self):
        if not self.psr_components:
            self._psr_auto_components_from_labels()
        self._psr_mark_dirty()
        self._psr_recompute_cache()
        self._psr_update_component_panel()
        diag = self._psr_diag or {}
        self._set_status(
            f"Rules applied: {diag.get('events', 0)} events, "
            f"{diag.get('unmapped', 0)} unmapped labels, "
            f"{diag.get('rule_mismatch', 0)} rule mismatches"
        )

    def _psr_state_value(self, value: Any, fallback: int = 0) -> int:
        try:
            num = int(value)
            if num in (-1, 0, 1):
                return num
        except Exception:
            pass
        txt = str(value).strip().lower()
        if txt in {"-1", "error", "incorrect", "wrong"}:
            return -1
        if txt in {"0", "not installed", "not_installed", "missing"}:
            return 0
        if txt in {"1", "installed", "correct"}:
            return 1
        return fallback

    def _psr_component_name(self, component_id: Optional[Any]) -> str:
        if component_id is None:
            return "combined"
        for comp in self.psr_components or []:
            if str(comp.get("id")) == str(component_id):
                return str(comp.get("name", comp.get("id")))
        return str(component_id)

    def _psr_state_label(self, state_val: Optional[Any]) -> str:
        if state_val is None:
            return "Unlabeled"
        try:
            val = int(state_val)
        except Exception:
            return str(state_val)
        return self._psr_state_label_map.get(val, str(val))

    def _psr_record_validation_entry(
        self,
        action: str,
        component_id: Optional[Any],
        start: Optional[int],
        end: Optional[int],
        old_label: Optional[str],
        new_label: Optional[str],
        note: str = "",
    ) -> None:
        if not self.validation_enabled:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        view_name = ""
        if 0 <= self.active_view_idx < len(self.views):
            view_name = self.views[self.active_view_idx].get("name", "")
        entry = {
            "ts": ts,
            "editor": self.validator_name or "unknown",
            "kind": "psr",
            "action": action,
            "component_id": component_id,
            "component": self._psr_component_name(component_id),
            "view": view_name,
            "start": int(start) if start is not None else None,
            "end": int(end) if end is not None else None,
            "old": old_label,
            "new": new_label,
        }
        if note:
            entry["note"] = note
        self._append_validation_modifications([entry])

    def _psr_record_validation_from_deltas(
        self,
        component_id: Optional[Any],
        deltas,
        action: str = "timeline_edit",
    ) -> None:
        if not self.validation_enabled or not deltas:
            return
        buckets: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        for s, e, old_label, new_label in self._iter_delta_spans(deltas):
            old_name = old_label if old_label is not None else "Unlabeled"
            new_name = new_label if new_label is not None else "Unlabeled"
            buckets.setdefault((str(old_name), str(new_name)), []).append(
                (int(s), int(e))
            )
        for (old_name, new_name), spans in buckets.items():
            for s, e in self._merge_spans(spans):
                self._psr_record_validation_entry(
                    action,
                    component_id,
                    s,
                    e,
                    old_name,
                    new_name,
                )

    def _psr_learn_rules_from_edits(self):
        if not self._is_psr_task():
            return
        if not self._psr_manual_events:
            QMessageBox.information(
                self, "Info", "No manual state edits were recorded for this video."
            )
            return
        if not self.psr_components:
            self._psr_auto_components_from_labels()
        segments = self._psr_collect_segments()
        if not segments:
            QMessageBox.information(
                self, "Info", "No segments available to infer rules."
            )
            return

        end_to_labels: Dict[int, List[str]] = {}
        for seg in segments:
            try:
                end = int(seg.get("end", -1))
            except Exception:
                continue
            label = seg.get("label")
            if label:
                end_to_labels.setdefault(end, []).append(str(label))

        latest: Dict[Tuple[str, str], int] = {}
        unmapped = 0
        ambiguous = 0
        for ev in self._psr_manual_events:
            try:
                frame = int(ev.get("frame", -1))
            except Exception:
                continue
            labels = end_to_labels.get(frame, [])
            if not labels:
                unmapped += 1
                continue
            if len(labels) > 1:
                ambiguous += 1
                continue
            label = labels[0]
            comp_id = str(ev.get("component_id"))
            state = self._psr_state_value(ev.get("state"), fallback=0)
            latest[(label, comp_id)] = state

        if not latest:
            QMessageBox.information(
                self, "Info", "No rule candidates found from manual edits."
            )
            return

        comp_lookup = {
            str(c.get("id")): c.get("name", "") for c in (self.psr_components or [])
        }
        updated = 0
        skipped = 0
        pushed = False
        for (label, comp_id), new_state in latest.items():
            rule = self.psr_rules.get(label)
            if rule is None:
                rule = {"components": [], "state": None}
                self.psr_rules[label] = rule
            comps = list(rule.get("components") or [])
            found = None
            for item in comps:
                if str(item.get("component_id")) == comp_id:
                    found = item
                    break
                if (
                    comp_lookup.get(comp_id)
                    and str(item.get("component")).strip().lower()
                    == str(comp_lookup[comp_id]).strip().lower()
                ):
                    found = item
                    break
            old_state = None
            if found is not None:
                old_state = self._psr_state_value(
                    found.get("state"), fallback=new_state
                )
                if old_state == new_state:
                    skipped += 1
                    continue
            comp_name = comp_lookup.get(comp_id, "")
            old_txt = (
                self._psr_state_label_map.get(old_state, "Unknown")
                if old_state is not None
                else "None"
            )
            new_txt = self._psr_state_label_map.get(new_state, "Unknown")
            msg = (
                f"Update rule?\n\n"
                f"Label: {label}\n"
                f"Component: {comp_id} {comp_name}\n"
                f"State: {old_txt} -> {new_txt}"
            )
            if (
                QMessageBox.question(self, "Confirm rule update", msg)
                != QMessageBox.Yes
            ):
                skipped += 1
                continue
            if not pushed:
                self._psr_push_undo("learn_rules")
                pushed = True
            if found is None:
                comps.append(
                    {
                        "component_id": (
                            int(comp_id) if str(comp_id).isdigit() else comp_id
                        ),
                        "component": comp_name or comp_id,
                        "state": int(new_state),
                    }
                )
            else:
                found["component_id"] = (
                    int(comp_id) if str(comp_id).isdigit() else comp_id
                )
                if comp_name:
                    found["component"] = comp_name
                found["state"] = int(new_state)
            rule["components"] = comps
            updated += 1

        self.psr_rules_path = ""
        self._psr_mark_dirty()
        self._psr_update_component_panel()
        summary = f"Rules updated: {updated}. Skipped: {skipped}."
        if unmapped or ambiguous:
            summary += f" Unmapped edits: {unmapped}. Ambiguous edits: {ambiguous}."
        QMessageBox.information(self, "Rule learning summary", summary)
        self._log(
            "psr_learn_rules",
            updated=updated,
            skipped=skipped,
            unmapped=unmapped,
            ambiguous=ambiguous,
        )

    def _psr_parse_action_json(
        self, path: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as ex:
            return None, f"Failed to read JSON: {ex}"
        if not isinstance(data, dict):
            return None, "Invalid JSON root (expected object)."
        segs = data.get("segments")
        if not isinstance(segs, list):
            return None, "Missing 'segments' list."
        if not any(isinstance(s, dict) and "action_label" in s for s in segs):
            return None, "Segments missing 'action_label'."
        labels_data = data.get("labels") or data.get("action_labels") or []
        id2label = {}
        if isinstance(labels_data, list) and labels_data:
            for item in labels_data:
                if not isinstance(item, dict):
                    continue
                try:
                    lid = int(item.get("id"))
                except Exception:
                    continue
                name = item.get("name", f"Label_{lid}")
                id2label[lid] = str(name)
        else:
            for s in segs:
                if not isinstance(s, dict):
                    continue
                try:
                    lid = int(s.get("action_label", -1))
                except Exception:
                    continue
                if lid >= 0 and lid not in id2label:
                    id2label[lid] = f"Label_{lid}"

        meta_data = data.get("meta_data")
        if not isinstance(meta_data, dict):
            meta_data = {}

        def _to_int_safe(value: Any, fallback: Any = 0):
            try:
                return int(value)
            except Exception:
                return fallback

        view_start = _to_int_safe(
            data.get("view_start", meta_data.get("view_start", 0)), 0
        )
        view_end_raw = data.get("view_end", meta_data.get("view_end", None))
        view_end = (
            _to_int_safe(view_end_raw, None) if view_end_raw is not None else None
        )

        fps_raw = data.get("fps", meta_data.get("fps", 0.0))
        try:
            fps_val = float(fps_raw or 0.0)
        except Exception:
            fps_val = 0.0

        resolution_raw = meta_data.get("resolution", {})
        if not isinstance(resolution_raw, dict):
            resolution_raw = {}
        meta_width = _to_int_safe(resolution_raw.get("width", 0), 0)
        meta_height = _to_int_safe(resolution_raw.get("height", 0), 0)
        src_num_frames = _to_int_safe(
            data.get("frame_count", meta_data.get("num_frames", 0)), 0
        )

        def _has_anomaly_mark(value: Any) -> bool:
            if isinstance(value, list):
                for item in value:
                    try:
                        if int(item) != 0:
                            return True
                    except Exception:
                        if bool(item):
                            return True
                return False
            if isinstance(value, dict):
                for item in value.values():
                    try:
                        if int(item) != 0:
                            return True
                    except Exception:
                        if bool(item):
                            return True
                return False
            if value is None:
                return False
            try:
                return int(value) != 0
            except Exception:
                return bool(str(value).strip())

        has_entity = any(
            isinstance(s, dict) and s.get("entity") not in (None, "") for s in segs
        )
        has_phase_data = any(
            isinstance(s, dict)
            and (
                str(s.get("phase") or "").strip() != ""
                or _has_anomaly_mark(s.get("anomaly_type"))
            )
            for s in segs
        )
        mode = "Fine" if (has_entity or has_phase_data) else "Coarse"
        parsed_segments = []
        max_abs_end = None
        for s in segs:
            if not isinstance(s, dict):
                continue
            try:
                start = int(s.get("start_frame", 0))
                end = int(s.get("end_frame", start))
                lid = int(s.get("action_label"))
            except Exception:
                continue
            label = id2label.get(lid, f"Label_{lid}")
            abs_start = start + view_start
            abs_end = end + view_start
            parsed_segments.append(
                {
                    "start": abs_start,
                    "end": abs_end,
                    "label": label,
                    "entity": s.get("entity"),
                    "phase": s.get("phase"),
                    "anomaly_type": s.get("anomaly_type"),
                }
            )
            if max_abs_end is None or abs_end > max_abs_end:
                max_abs_end = abs_end
        if not parsed_segments:
            return None, "No valid segments found."
        if view_end is None and max_abs_end is not None:
            view_end = max_abs_end
        view_span = (
            max(0, int(view_end) - int(view_start) + 1)
            if view_end is not None
            else max(0, src_num_frames)
        )
        parsed_meta = {
            "fps": float(fps_val),
            "resolution": {"width": int(meta_width), "height": int(meta_height)},
            "num_frames": int(view_span),
            "view_start": int(view_start),
            "view_end": int(view_end) if view_end is not None else None,
        }
        if src_num_frames > 0:
            parsed_meta["video_num_frames"] = int(src_num_frames)
        workflow_val = data.get("workflow")
        initial_state_val = data.get("initial_state")
        if isinstance(meta_data, dict):
            if workflow_val in (None, ""):
                workflow_val = meta_data.get("workflow")
            if initial_state_val in (None, ""):
                initial_state_val = meta_data.get("initial_state")
        model_type_val = None
        if isinstance(meta_data, dict):
            model_type_val = meta_data.get("model_type")
        if model_type_val in (None, ""):
            model_type_val = data.get("model_type")
        info = {
            "segments": parsed_segments,
            "labels": list(id2label.values()),
            "view_start": view_start,
            "view_end": int(view_end) if view_end is not None else None,
            "mode": mode,
            "video_id": data.get("video_id") or data.get("video_name") or "",
            "video_name": data.get("video_name") or data.get("video_path") or "",
            "fps": float(fps_val),
            "workflow": workflow_val,
            "initial_state": initial_state_val,
            "model_type": self._psr_normalize_model_type(model_type_val),
            "frame_count": int(src_num_frames) if src_num_frames > 0 else int(view_span),
            "meta_data": parsed_meta,
        }
        return info, None

    def _psr_canonical_video_id(self, video_id: Any, video_name: Any = "") -> str:
        def _stem(value: Any) -> str:
            txt = str(value or "").strip()
            if not txt:
                return ""
            base = os.path.basename(txt.replace("\\", "/"))
            return os.path.splitext(base)[0]

        base = (
            _stem(video_id)
            or _stem(video_name)
            or _stem(getattr(self, "video_path", ""))
            or "video"
        )
        out = str(base)
        # Strip view/crop suffixes so multi-view exports share one video id.
        for _ in range(6):
            prev = out
            out = re.sub(
                r"(?i)[_-](?:clip|clipped|crop|cropped|trim|trimmed)$", "", out
            )
            out = re.sub(
                r"(?i)[_-](?:front|back|left|right|top|bottom|main|center|centre)$",
                "",
                out,
            )
            out = out.rstrip("_-")
            if out == prev:
                break
        return out or base or "video"

    def _psr_build_export_payload(
        self,
        kind: str,
        events: List[Dict[str, Any]],
        components: List[Dict[str, Any]],
        meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        def _safe_int(value: Any, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return default

        try:
            view_start = int(meta.get("view_start", 0))
        except Exception:
            view_start = 0
        view_end_raw = meta.get("view_end", None)
        if view_end_raw is None:
            view_end = view_start
        else:
            try:
                view_end = int(view_end_raw)
            except Exception:
                view_end = view_start
        video_id = self._psr_canonical_video_id(
            meta.get("video_id"), meta.get("video_name")
        )
        fps = float(meta.get("fps", 0.0) or 0.0)

        meta_data_in = meta.get("meta_data")
        if not isinstance(meta_data_in, dict):
            meta_data_in = {}
        resolution_in = meta_data_in.get("resolution", {})
        if not isinstance(resolution_in, dict):
            resolution_in = {}
        width = _safe_int(resolution_in.get("width", meta.get("width", 0)), 0)
        height = _safe_int(resolution_in.get("height", meta.get("height", 0)), 0)
        full_frame_count = _safe_int(
            meta.get(
                "frame_count",
                meta_data_in.get("video_num_frames", meta_data_in.get("num_frames", 0)),
            ),
            0,
        )
        view_num_frames = (
            max(0, int(view_end) - int(view_start) + 1) if view_end >= view_start else 0
        )
        if view_num_frames <= 0:
            view_num_frames = _safe_int(meta_data_in.get("num_frames", 0), 0)
        if fps <= 0.0:
            try:
                fps = float(meta_data_in.get("fps", 0.0) or 0.0)
            except Exception:
                fps = 0.0
        # Export explicit timeline scope and video-level sizing to avoid
        # ambiguity when consumers reconstruct the final tail segment.
        meta_data_out = {
            "fps": float(fps),
            "resolution": {"width": int(width), "height": int(height)},
            "num_frames": int(view_num_frames),
            "view_start": int(view_start),
            "view_end": int(view_end),
        }
        if full_frame_count > 0:
            meta_data_out["video_num_frames"] = int(full_frame_count)
        init_state = self._psr_state_value(
            meta.get("psr_initial_state", self._psr_detected_initial_state),
            fallback=0,
        )
        if init_state not in (-1, 0, 1):
            init_state = 0
        model_type = self._psr_normalize_model_type(
            meta.get("model_type") or meta_data_in.get("model_type") or self._psr_model_type
        )
        initial_vec = [int(init_state)] * len(components)

        workflow_raw = str(meta.get("workflow") or meta_data_in.get("workflow") or "").strip().lower()
        if int(init_state) in (0, 1):
            workflow = self._psr_workflow_from_initial_state(int(init_state))
        elif workflow_raw.startswith("dis"):
            workflow = "disassemble"
        elif workflow_raw.startswith("ass"):
            workflow = "assemble"
        else:
            workflow = "assemble"
        meta_data_out["workflow"] = workflow
        meta_data_out["initial_state"] = int(init_state)
        meta_data_out["initial_state_label"] = (
            "Installed"
            if int(init_state) == 1
            else "Not installed"
            if int(init_state) == 0
            else "Error"
        )
        meta_data_out["model_type"] = model_type

        canonical_id_map: Dict[str, Any] = {}
        for comp in components:
            comp_id = comp.get("id")
            if comp_id is None:
                continue
            canonical_id_map[str(comp_id)] = comp_id

        norm_events: List[Dict[str, Any]] = []
        for ev in events or []:
            comp_key = str(ev.get("component_id"))
            if comp_key not in canonical_id_map:
                continue
            try:
                fr = int(ev.get("frame", 0))
            except Exception:
                continue
            norm_events.append(
                {
                    "frame": fr,
                    "component_id": canonical_id_map[comp_key],
                    "state": self._psr_state_value(ev.get("state"), fallback=0),
                    "label": ev.get("label"),
                }
            )
        norm_events.sort(
            key=lambda item: (int(item.get("frame", 0)), str(item.get("component_id")))
        )

        def _normalize_state_vector(
            value: Any, fallback_vector: List[int]
        ) -> List[int]:
            n = len(components)
            out = list(fallback_vector[:n])
            if len(out) < n:
                out.extend([int(init_state)] * (n - len(out)))
            if isinstance(value, list):
                for idx, raw in enumerate(value[:n]):
                    out[idx] = self._psr_state_value(raw, fallback=out[idx])
            return out[:n]

        rel_events = []
        for ev in norm_events:
            rel_events.append(
                {
                    "frame": max(0, int(ev.get("frame", 0)) - view_start),
                    "component_id": ev.get("component_id"),
                    "state": self._psr_state_value(ev.get("state"), fallback=0),
                }
            )
        # ASR export is intentionally view-agnostic: timeline scope is encoded
        # via view_start/view_end and meta_data, without a separate view label.
        base = {
            "task": "ASR",
            "version": "1.0",
            "video_id": video_id,
            "fps": fps,
            "view_start": view_start,
            "view_end": view_end,
            "meta_data": meta_data_out,
            "frame_count": int(full_frame_count)
            if full_frame_count > 0
            else int(view_num_frames),
            "workflow": workflow,
            "initial_state": int(init_state),
            "initial_state_vector": list(initial_vec),
            "components": [
                {"id": c.get("id"), "name": c.get("name")} for c in components
            ],
        }
        seq = psr_build_state_sequence(
            norm_events, components, initial_state=initial_vec
        )
        rel_seq_raw = []
        for it in seq:
            rel_seq_raw.append(
                {
                    "frame": max(
                        0, _safe_int(it.get("frame", view_start), view_start) - view_start
                    ),
                    "state": it.get("state", None),
                }
            )
        rel_seq_raw.sort(key=lambda item: int(item.get("frame", 0)))
        rel_seq = []
        prev_state = list(initial_vec)
        for item in rel_seq_raw:
            frame = _safe_int(item.get("frame", 0), 0)
            state_vec = _normalize_state_vector(item.get("state"), prev_state)
            packed = {"frame": max(0, frame), "state": state_vec}
            if rel_seq and rel_seq[-1]["frame"] == packed["frame"]:
                rel_seq[-1] = packed
            else:
                rel_seq.append(packed)
            prev_state = list(state_vec)
        if not rel_seq or int(rel_seq[0].get("frame", 0)) != 0:
            rel_seq.insert(0, {"frame": 0, "state": list(initial_vec)})
        else:
            rel_seq[0]["state"] = _normalize_state_vector(
                rel_seq[0].get("state"), initial_vec
            )

        # Export initial state from frame-0 state when it is explicit,
        # so manual edits at the first segment are reflected in output.
        init_vec_export = _normalize_state_vector(
            rel_seq[0].get("state"), initial_vec
        )
        base["initial_state_vector"] = list(init_vec_export)
        if init_vec_export and all(v == init_vec_export[0] for v in init_vec_export):
            try:
                init_uniform = int(init_vec_export[0])
            except Exception:
                init_uniform = int(base.get("initial_state", init_state))
            if init_uniform in (-1, 0, 1):
                base["initial_state"] = init_uniform
                if init_uniform in (0, 1):
                    wf = self._psr_workflow_from_initial_state(init_uniform, fallback=workflow)
                    base["workflow"] = wf
                    meta_data_out["workflow"] = wf
                meta_data_out["initial_state"] = init_uniform
                meta_data_out["initial_state_label"] = (
                    "Installed"
                    if init_uniform == 1
                    else "Not installed"
                    if init_uniform == 0
                    else "Error"
                )
        base["state_sequence"] = rel_seq
        # Keep sparse transition events for downstream step derivation/debugging.
        # The canonical timeline state remains state_sequence.
        base["state_changes"] = rel_events
        return base

    def _psr_batch_convert_dataset(self):
        if not self.psr_rules:
            QMessageBox.information(
                self, "Info", "Load or learn rules before batch conversion."
            )
            return
        if not self.psr_components:
            self._psr_auto_components_from_labels()
        if not self.psr_components:
            QMessageBox.information(
                self, "Info", "No components available for batch conversion."
            )
            return
        src_dir = QFileDialog.getExistingDirectory(self, "Select input folder")
        if not src_dir:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not out_dir:
            return

        files = []
        for root, _dirs, names in os.walk(src_dir):
            for name in names:
                if name.lower().endswith(".json"):
                    files.append(os.path.join(root, name))
        if not files:
            QMessageBox.information(
                self, "Info", "No JSON files found in the selected folder."
            )
            return

        parsed = []
        errors = []
        mode_counts = {"Coarse": 0, "Fine": 0}
        for path in files:
            info, err = self._psr_parse_action_json(path)
            if err:
                errors.append((path, err))
                continue
            mode_counts[info["mode"]] += 1
            parsed.append((path, info))

        if not parsed:
            msg = "No valid Action Segmentation JSON files were found."
            if errors:
                msg += f"\nErrors: {len(errors)}"
            QMessageBox.information(self, "Info", msg)
            return

        target_mode = (
            "Coarse"
            if mode_counts["Fine"] == 0
            else "Fine" if mode_counts["Coarse"] == 0 else None
        )
        if target_mode is None:
            box = QMessageBox(self)
            box.setWindowTitle("Select conversion mode")
            box.setText(
                f"Mixed annotation modes detected.\nCoarse: {mode_counts['Coarse']} files\nFine: {mode_counts['Fine']} files\n\n"
                "Choose which mode to convert. The other mode will be skipped."
            )
            btn_coarse = box.addButton("Coarse", QMessageBox.AcceptRole)
            btn_fine = box.addButton("Fine", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.exec_()
            if box.clickedButton() == btn_coarse:
                target_mode = "Coarse"
            elif box.clickedButton() == btn_fine:
                target_mode = "Fine"
            else:
                return

        converted = 0
        skipped = 0
        for path, info in parsed:
            if info["mode"] != target_mode:
                skipped += 1
                continue
            segments = info["segments"]
            events = psr_derive_events(
                segments,
                self.psr_components,
                self.psr_rules,
                ignore_labels=[self._psr_asr_asd_invisible_label],
            )
            parsed_init = info.get("initial_state")
            try:
                parsed_init = int(parsed_init)
            except Exception:
                parsed_init = None
            if parsed_init not in (-1, 0, 1):
                parsed_init = None
            if parsed_init is None:
                workflow = str(info.get("workflow") or "").strip().lower()
                if workflow in {"disassemble", "disassembly", "dismantle", "teardown"}:
                    parsed_init = 1
                elif workflow in {"assemble", "assembly", "build"}:
                    parsed_init = 0
            if parsed_init is None:
                parsed_init = self._psr_initial_state_value(segments=segments)
            meta = {
                "view_start": info.get("view_start", 0),
                "view_end": info.get("view_end", None),
                "video_id": info.get("video_id")
                or os.path.splitext(os.path.basename(path))[0],
                "video_name": info.get("video_name") or os.path.basename(path),
                "fps": info.get("fps", 0.0),
                "psr_initial_state": parsed_init,
                "frame_count": info.get("frame_count", 0),
                "model_type": info.get("model_type", self._psr_model_type),
                "meta_data": info.get("meta_data", {}),
            }
            base_rel = os.path.splitext(os.path.relpath(path, src_dir))[0]
            out_base = os.path.join(out_dir, base_rel)
            out_dir_path = os.path.dirname(out_base)
            try:
                os.makedirs(out_dir_path, exist_ok=True)
            except Exception:
                pass
            asr_payload = self._psr_build_export_payload(
                "ASR", events, self.psr_components, meta
            )
            try:
                with open(out_base + "_asr.json", "w", encoding="utf-8") as f:
                    json.dump(asr_payload, f, ensure_ascii=False, indent=2)
                converted += 1
            except Exception as ex:
                errors.append((path, f"Export failed: {ex}"))

        msg = f"Converted: {converted}. Skipped: {skipped}. Errors: {len(errors)}."
        if errors:
            preview = "\n".join(
                f"{os.path.basename(p)}: {err}" for p, err in errors[:5]
            )
            msg += f"\n\nExamples:\n{preview}"
        QMessageBox.information(self, "Batch conversion summary", msg)
        self._log(
            "psr_batch_convert",
            converted=converted,
            skipped=skipped,
            errors=len(errors),
            mode=target_mode,
        )

    def _export_psr_rules(self):
        if not self.psr_rules:
            QMessageBox.information(self, "Info", "No rules to export.")
            return
        default_name = "psr_rules.json"
        fp, _ = QFileDialog.getSaveFileName(
            self, "Export rules", default_name, "JSON Files (*.json)"
        )
        if not fp:
            return
        labels = [lb.name for lb in self.labels]
        rules_list = []
        used = set()
        for label in labels:
            rule = self.psr_rules.get(label)
            if not rule:
                continue
            used.add(label)
            rules_list.append(
                {
                    "label": label,
                    "components": rule.get("components", []),
                    "state": rule.get("state"),
                }
            )
        for label, rule in self.psr_rules.items():
            if label in used:
                continue
            rules_list.append(
                {
                    "label": label,
                    "components": rule.get("components", []),
                    "state": rule.get("state"),
                }
            )
        payload = {"rules": rules_list}
        try:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._set_status(f"Exported rules: {os.path.basename(fp)}")
            self._log("export_psr_rules", path=fp, count=len(rules_list))
        except Exception as ex:
            QMessageBox.warning(self, "Error", f"Failed to write rules:\n{ex}")

    def _psr_confirm_merge_before_export(
        self, duplicates: List[Tuple[int, int, int, int]]
    ) -> bool:
        if not duplicates:
            return True
        preview = "\n".join(f"{a}-{b} -> {c}-{d}" for a, b, c, d in duplicates[:5])
        box = QMessageBox(self)
        box.setWindowTitle("Check segments")
        box.setIcon(QMessageBox.Warning)
        box.setText(
            f"Adjacent state segments with identical states were found ({len(duplicates)}).\n"
            "You can merge them automatically before export."
        )
        box.setInformativeText(f"Examples:\n{preview}")
        btn_merge = box.addButton("Merge and export", QMessageBox.AcceptRole)
        btn_keep = box.addButton("Export without merge", QMessageBox.AcceptRole)
        btn_cancel = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(btn_merge)
        box.exec_()
        clicked = box.clickedButton()
        if clicked == btn_cancel:
            return False
        if clicked == btn_merge:
            self._psr_merge_adjacent_identical_segments()
            self._psr_recompute_cache()
        return True

    def _export_psr_asr_asd(self, kind: str):
        if not getattr(self, "video_path", None) or not self.views:
            QMessageBox.information(self, "Info", "Load a video first.")
            return
        if not self.psr_components:
            self._psr_auto_components_from_labels()
        self._psr_recompute_cache()
        if not self.psr_components:
            QMessageBox.information(self, "Info", "No components available for export.")
            return
        view = self.views[self.active_view_idx]
        view_start = int(view.get("start", 0))
        view_end = int(view.get("end", view_start))
        duplicates = self._psr_find_adjacent_identical_segments()
        if not self._psr_confirm_merge_before_export(duplicates):
            return
        events = list(self._psr_events_cache or [])
        if not events:
            QMessageBox.information(
                self,
                "Info",
                "No component events were derived from the current labels. Export will contain default states only.",
            )
        init_state = self._psr_state_value(self._psr_detected_initial_state, fallback=0)
        if init_state not in (-1, 0, 1):
            init_state = 0
        try:
            view_meta = self._meta_data_for_view(view)
        except Exception:
            view_meta = {}
        player = view.get("player") if isinstance(view, dict) else None
        frame_count = int(getattr(player, "frame_count", 0) or 0)
        meta = {
            "view_start": view_start,
            "view_end": view_end,
            "video_id": self.current_video_id or "",
            "video_name": self.current_video_name or "",
            "fps": float(getattr(self.player, "frame_rate", 0.0) or 0.0),
            "workflow": self._psr_detected_flow,
            "psr_initial_state": init_state,
            "frame_count": frame_count,
            "model_type": self._psr_model_type,
            "meta_data": view_meta,
        }
        kind_upper = str(kind or "").strip().upper()
        if kind_upper != "ASR":
            QMessageBox.information(
                self, "Info", "Only assembly-state export is enabled in this build."
            )
            return
        payload = self._psr_build_export_payload(
            "ASR", events, self.psr_components, meta
        )
        suffix = "asr"
        export_label = "ASR"

        default_name = f"{self.current_video_id or 'video'}_{suffix}.json"
        fp, _ = QFileDialog.getSaveFileName(
            self, f"Export {export_label}", default_name, "JSON Files (*.json)"
        )
        if not fp:
            return
        try:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._set_status(f"Exported {export_label}: {os.path.basename(fp)}")
            events_count = len(payload.get("state_changes", []))
            self._log(
                "export_psr_asr_asd",
                kind="ASR",
                path=fp,
                events=events_count,
            )
            self._save_validation_log_safely(fp, self.current_video_id or "", [fp])
            self._flush_ops_log_safely(
                fp + ".ops.log.csv", context="export ASR"
            )
        except Exception as ex:
            QMessageBox.warning(self, "Error", f"Failed to write file:\n{ex}")

    def _log(self, event: str, **fields):
        # Filter out automatic/system events to keep the log focused on user interactions.
        auto_events = {"frame_advanced", "sync_controls"}
        if event in auto_events:
            return
        if getattr(self, "op_logger", None):
            try:
                frame = getattr(self.player, "current_frame", None)
            except Exception:
                frame = None
            if frame is not None:
                fields.setdefault("frame", frame)
            self.op_logger.log(event, **fields)

    def _ensure_anomaly_list_widget(self) -> None:
        ag = getattr(self, "anomaly_group", None)
        if ag is None:
            return
        layout = getattr(self, "anomaly_layout", None)
        layout_deleted = False
        if layout is None:
            layout_deleted = True
        else:
            if sip is not None:
                try:
                    if bool(sip.isdeleted(layout)):
                        layout_deleted = True
                except Exception:
                    pass
            if not layout_deleted:
                try:
                    if not bool(layout):
                        layout_deleted = True
                except Exception:
                    pass
        if layout_deleted:
            self.anomaly_layout = QVBoxLayout(ag)
            self.anomaly_layout.setContentsMargins(6, 6, 6, 6)
            self.anomaly_layout.setSpacing(4)
            try:
                ag.setLayout(self.anomaly_layout)
            except Exception:
                pass
        if not getattr(self, "anomaly_layout", None):
            return
        al = getattr(self, "anomaly_list", None)
        deleted = False
        if al is None:
            deleted = True
        else:
            try:
                _ = al.count()
            except Exception:
                deleted = True
            if not deleted and sip is not None:
                try:
                    if bool(sip.isdeleted(al)):
                        deleted = True
                except Exception:
                    pass
        if deleted:
            try:
                while self.anomaly_layout.count():
                    item = self.anomaly_layout.takeAt(0)
                    w = item.widget() if item else None
                    if w is not None:
                        try:
                            w.hide()
                        except Exception:
                            pass
                        try:
                            w.deleteLater()
                        except Exception:
                            pass
            except Exception:
                pass
            self.anomaly_list = ClickToggleList(self.anomaly_group)
            self.anomaly_list.setSelectionMode(QAbstractItemView.NoSelection)
            self.anomaly_list.setFlow(QListView.LeftToRight)
            self.anomaly_list.setWrapping(True)
            self.anomaly_list.setResizeMode(QListView.Adjust)
            self.anomaly_list.setSpacing(4)
            self.anomaly_list.setStyleSheet(
                "QListWidget::item { padding: 2px 8px; }"
                "QListWidget::indicator { width: 14px; height: 14px; }"
            )
            self.anomaly_list.itemChanged.connect(self._on_anomaly_item_changed)
            self.anomaly_list.setWordWrap(True)
            self.anomaly_list.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Expanding
            )
            self.anomaly_layout.addWidget(self.anomaly_list)

    # ----- fine mode helpers (phase/anomaly/verb-noun) -----
    def _ensure_anomaly_types(
        self, types: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        if types is None:
            if not self.anomaly_types:
                self.anomaly_types = [
                    {"id": i, "name": n} for i, n in enumerate(DEFAULT_ANOMALY_TYPES)
                ]
        else:
            cleaned = []
            for idx, item in enumerate(types):
                if isinstance(item, dict):
                    name = str(item.get("name", "")).strip()
                    tid = item.get("id", idx)
                else:
                    name = str(item).strip()
                    tid = idx
                if not name:
                    continue
                try:
                    tid = int(tid)
                except Exception:
                    tid = idx
                cleaned.append({"id": tid, "name": name})
            if not cleaned:
                cleaned = [
                    {"id": i, "name": n} for i, n in enumerate(DEFAULT_ANOMALY_TYPES)
                ]
            self.anomaly_types = cleaned
        self.anomaly_types = sorted(
            self.anomaly_types, key=lambda x: int(x.get("id", 0))
        )
        names = self._anomaly_type_names()
        for ent_map in self.anomaly_type_stores.values():
            for tname in names:
                ent_map.setdefault(tname, AnnotationStore())
        for vw in getattr(self, "views", []):
            astores = vw.get("anomaly_type_stores")
            if not isinstance(astores, dict):
                continue
            for ent_map in astores.values():
                for tname in names:
                    ent_map.setdefault(tname, AnnotationStore())

    def _anomaly_type_names(self) -> List[str]:
        return [
            t.get("name")
            for t in sorted(self.anomaly_types, key=lambda x: int(x.get("id", 0)))
            if t.get("name")
        ]

    def _ensure_phase_store_for_entity(
        self, ename: str, stores: Optional[Dict[str, AnnotationStore]] = None
    ) -> AnnotationStore:
        target = stores if stores is not None else self.phase_stores
        return target.setdefault(ename, AnnotationStore())

    def _ensure_anomaly_type_stores_for_entity(
        self, ename: str, stores: Optional[Dict[str, Dict[str, AnnotationStore]]] = None
    ) -> Dict[str, AnnotationStore]:
        target = stores if stores is not None else self.anomaly_type_stores
        ent_map = target.setdefault(ename, {})
        for name in self._anomaly_type_names():
            ent_map.setdefault(name, AnnotationStore())
        return ent_map

    @staticmethod
    def _span_has_label(frames: List[int], start: int, end: int) -> bool:
        if not frames:
            return False
        i = bisect.bisect_left(frames, start)
        return i < len(frames) and frames[i] <= end

    def _phase_label_for_span(
        self, store: Optional[AnnotationStore], start: int, end: int
    ) -> str:
        if not store:
            return ""
        best = ""
        best_count = 0
        for lb in self.phase_labels:
            frames = store.frames_of(lb.name)
            if not frames:
                continue
            i0 = bisect.bisect_left(frames, start)
            i1 = bisect.bisect_right(frames, end)
            cnt = max(0, i1 - i0)
            if cnt > best_count:
                best_count = cnt
                best = lb.name
        return best if best_count > 0 else ""

    def _anomaly_vector_for_span(
        self,
        ename: str,
        start: int,
        end: int,
        stores: Optional[Dict[str, Dict[str, AnnotationStore]]] = None,
    ) -> List[int]:
        names = self._anomaly_type_names()
        if not names:
            return []
        vec = []
        root = stores if stores is not None else self.anomaly_type_stores
        stores = root.get(ename, {})
        for name in names:
            st = stores.get(name)
            if not st:
                vec.append(0)
                continue
            frames = st.frames_of(name)
            vec.append(1 if self._span_has_label(frames, start, end) else 0)
        return vec

    def _phase_anomaly_segment_cuts(self, ename: str) -> List[int]:
        if not self.phase_mode_enabled:
            return []
        pstore = self.phase_stores.get(ename)
        if not pstore:
            return []
        try:
            frames = pstore.frames_of("anomaly")
        except Exception:
            return []
        if not frames:
            return []
        anom_set = set(frames)
        if not anom_set:
            return []
        names = self._anomaly_type_names()
        ent_map = self.anomaly_type_stores.get(ename, {})
        sig: Dict[int, int] = {}
        for idx, name in enumerate(names):
            st = ent_map.get(name)
            if not st:
                continue
            try:
                tframes = st.frames_of(name)
            except Exception:
                continue
            for f in tframes:
                if f in anom_set:
                    sig[f] = sig.get(f, 0) | (1 << idx)
        frames_sorted = sorted(anom_set)
        cuts: List[int] = []
        prev_frame = frames_sorted[0]
        prev_sig = sig.get(prev_frame, 0)
        for f in frames_sorted[1:]:
            if f != prev_frame + 1:
                prev_sig = sig.get(f, 0)
                prev_frame = f
                continue
            cur_sig = sig.get(f, 0)
            if cur_sig != prev_sig:
                cuts.append(f)
            prev_sig = cur_sig
            prev_frame = f
        return cuts

    def _update_phase_segment_cuts_for_entity(self, ename: str) -> None:
        tl = getattr(self, "timeline", None)
        if tl is None:
            return
        cuts = self._phase_anomaly_segment_cuts(ename)
        rows = getattr(tl, "_combined_rows", []) or []
        for row in rows:
            meta = getattr(row, "_group_meta", None)
            if not isinstance(meta, dict):
                continue
            if meta.get("row_type") != "phase":
                continue
            if meta.get("entity") != ename:
                continue
            try:
                meta["segment_cuts"] = list(cuts)
            except Exception:
                pass
            try:
                row._segment_cuts = list(cuts)
            except Exception:
                pass
            try:
                row.update()
            except Exception:
                pass

    def _segments_from_store(
        self, store: AnnotationStore
    ) -> List[Tuple[int, int, str]]:
        if not store or not store.frame_to_label:
            return []
        frames = sorted(store.frame_to_label.keys())
        if not frames:
            return []
        segs = []
        s = frames[0]
        prev = frames[0]
        cur = store.frame_to_label.get(prev)
        for f in frames[1:]:
            lb = store.frame_to_label.get(f)
            if f == prev + 1 and lb == cur:
                prev = f
                continue
            segs.append((s, prev, cur))
            s = f
            prev = f
            cur = lb
        segs.append((s, prev, cur))
        return segs

    def _infer_verb_noun(
        self, label_name: str, verb_candidates: Optional[List[str]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        name = (label_name or "").strip()
        if not name or name.lower() == "null" or is_extra_label(name):
            return None, None
        verbs = [str(v or "").strip() for v in (verb_candidates or []) if str(v or "").strip()]
        if "hand_spin" not in verbs:
            verbs.append("hand_spin")
        return infer_label_verb_noun(name, verbs or list(KNOWN_VERB_PREFIXES))

    def _normalize_fine_vocab_items(self, items: List[Any]) -> List[Dict[str, Any]]:
        out = []
        for idx, item in enumerate(items or []):
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                vid = item.get("id", idx)
            else:
                name = str(item).strip()
                vid = idx
            if not name:
                continue
            try:
                vid = int(vid)
            except Exception:
                vid = idx
            out.append({"id": vid, "name": name})
        return out

    def _fine_verb_candidates(self) -> List[str]:
        names: List[str] = []
        for item in self._normalize_fine_vocab_items(self.fine_verbs):
            name = str(item.get("name", "")).strip()
            if name and name not in names:
                names.append(name)
        for name in KNOWN_VERB_PREFIXES:
            text = str(name or "").strip()
            if text and text not in names:
                names.append(text)
        return names

    def _sync_label_panel_verb_hints(self, refresh: bool = True) -> None:
        panel = getattr(self, "panel", None)
        if panel is None or not hasattr(panel, "set_compound_verbs"):
            return
        compound = [name for name in self._fine_verb_candidates() if "_" in str(name or "")]
        try:
            panel.set_compound_verbs(compound, refresh=refresh)
        except Exception:
            pass

    def _refresh_fine_label_decomposition(
        self,
        *,
        refresh_panel: bool = False,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        verbs = self._normalize_fine_vocab_items(self.fine_verbs)
        nouns = self._normalize_fine_vocab_items(self.fine_nouns)
        verb_map = {str(v["name"]): int(v["id"]) for v in verbs if v.get("name")}
        noun_map = {str(n["name"]): int(n["id"]) for n in nouns if n.get("name")}
        max_vid = max(verb_map.values(), default=-1)

        candidate_verbs = list(verb_map.keys())
        for name in KNOWN_VERB_PREFIXES:
            text = str(name or "").strip()
            if text and text not in candidate_verbs:
                candidate_verbs.append(text)

        derived_verbs = set()
        derived_nouns = set()
        for lb in self.labels:
            name = str(getattr(lb, "name", "") or "").strip()
            if not name or is_extra_label(name):
                continue
            v, n = self._infer_verb_noun(name, candidate_verbs)
            if v:
                derived_verbs.add(str(v))
            if n:
                derived_nouns.add(str(n))
        for v in sorted(derived_verbs):
            if v not in verb_map:
                max_vid += 1
                verb_map[v] = int(max_vid)
                verbs.append({"id": int(max_vid), "name": v})

        rebuilt_nouns: List[Dict[str, Any]] = []
        next_nid = max(noun_map.values(), default=-1) + 1
        for n in sorted(derived_nouns):
            if n in noun_map:
                rebuilt_nouns.append({"id": int(noun_map[n]), "name": n})
            else:
                rebuilt_nouns.append({"id": int(next_nid), "name": n})
                next_nid += 1

        self.fine_verbs = sorted(verbs, key=lambda x: (int(x.get("id", 0)), str(x.get("name", ""))))
        self.fine_nouns = sorted(rebuilt_nouns, key=lambda x: (int(x.get("id", 0)), str(x.get("name", ""))))
        self._sync_label_panel_verb_hints(refresh=refresh_panel)
        return (
            {str(v["name"]): int(v["id"]) for v in self.fine_verbs if v.get("name")},
            {str(n["name"]): int(n["id"]) for n in self.fine_nouns if n.get("name")},
        )

    def _ensure_fine_vocab(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        return self._refresh_fine_label_decomposition(refresh_panel=False)

    def _add_anomaly_item(
        self, name: str, checked: bool = False
    ) -> Optional[QListWidgetItem]:
        if getattr(self, "anomaly_list", None) is None:
            return None
        label = str(name).strip()
        if not label:
            return None
        it = QListWidgetItem(label)
        it.setFlags(it.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        it.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        try:
            self.anomaly_list.addItem(it)
        except Exception:
            return None
        return it

    def _set_anomaly_panel_enabled(self, on: bool, clear: bool = False) -> None:
        if not getattr(self, "anomaly_group", None):
            return
        self._ensure_anomaly_list_widget()
        try:
            self.anomaly_group.setEnabled(True)
        except Exception:
            pass
        if getattr(self, "anomaly_list", None) is not None:
            try:
                self.anomaly_list.setEnabled(bool(on))
            except Exception:
                pass
        if clear:
            self._anomaly_block = True
            if getattr(self, "anomaly_list", None) is not None:
                for i in range(self.anomaly_list.count()):
                    it = self.anomaly_list.item(i)
                    if it is None:
                        continue
                    try:
                        it.setCheckState(Qt.Unchecked)
                    except Exception:
                        pass
            self._anomaly_block = False

    def _rebuild_anomaly_type_panel(self) -> None:
        self._ensure_anomaly_list_widget()
        if getattr(self, "anomaly_list", None) is None:
            return
        names = self._anomaly_type_names()
        if not names:
            self._ensure_anomaly_types()
            names = self._anomaly_type_names()
        if not names:
            self.anomaly_types = [
                {"id": i, "name": n} for i, n in enumerate(DEFAULT_ANOMALY_TYPES)
            ]
            names = list(DEFAULT_ANOMALY_TYPES)
        try:
            self.anomaly_list.blockSignals(True)
        except Exception:
            pass
        try:
            self.anomaly_list.clear()
        except Exception:
            pass
        for name in names:
            self._add_anomaly_item(name, checked=False)
        try:
            self.anomaly_list.blockSignals(False)
        except Exception:
            pass
        try:
            self.anomaly_group.adjustSize()
        except Exception:
            pass

    def _refresh_anomaly_type_panel(self, force: bool = False) -> None:
        self._ensure_anomaly_list_widget()
        if getattr(self, "anomaly_list", None) is None:
            return
        needs_rebuild = force or self.anomaly_list.count() == 0
        if not needs_rebuild:
            return
        self._ensure_anomaly_types()
        self._rebuild_anomaly_type_panel()
        try:
            if self.anomaly_group.layout() is not None:
                self.anomaly_group.layout().activate()
            self.anomaly_group.adjustSize()
            self.anomaly_group.updateGeometry()
            if getattr(self, "phase_panel", None):
                self.phase_panel.updateGeometry()
        except Exception:
            pass

    def _sync_anomaly_panel(self) -> None:
        self._ensure_anomaly_list_widget()
        if not self.phase_mode_enabled:
            self._set_anomaly_panel_enabled(False, clear=True)
            return
        sel = self._phase_selected or {}
        if sel.get("label") != "anomaly" or not sel.get("entity"):
            self._set_anomaly_panel_enabled(False, clear=True)
            return
        ename = sel.get("entity")
        start = int(sel.get("start", 0))
        end = int(sel.get("end", start))
        selected = set()
        stores = self.anomaly_type_stores.get(ename, {})
        if getattr(self, "anomaly_list", None) is not None:
            for i in range(self.anomaly_list.count()):
                it = self.anomaly_list.item(i)
                if it is None:
                    continue
                name = it.text()
                st = stores.get(name)
                if not st:
                    continue
                frames = st.frames_of(name)
                if self._span_has_label(frames, start, end):
                    selected.add(name)
        self._anomaly_block = True
        if getattr(self, "anomaly_list", None) is not None:
            for i in range(self.anomaly_list.count()):
                it = self.anomaly_list.item(i)
                if it is None:
                    continue
                name = it.text()
                try:
                    it.setCheckState(Qt.Checked if name in selected else Qt.Unchecked)
                except Exception:
                    pass
        self._anomaly_block = False
        self._set_anomaly_panel_enabled(True, clear=False)

    def _on_anomaly_item_changed(self, item: QListWidgetItem) -> None:
        if self._anomaly_block or item is None:
            return
        name = item.text()
        checked = item.checkState() == Qt.Checked
        self._on_anomaly_type_toggled(str(name), checked)

    def _on_anomaly_type_toggled(self, name: str, checked: bool) -> None:
        if self._anomaly_block:
            return
        if not self.phase_mode_enabled:
            return
        sel = self._phase_selected or {}
        if sel.get("label") != "anomaly":
            return
        ename = sel.get("entity")
        if not ename:
            return
        try:
            start = int(sel.get("start", 0))
            end = int(sel.get("end", start))
        except Exception:
            return
        ent_map = self._ensure_anomaly_type_stores_for_entity(ename)
        st = ent_map.setdefault(name, AnnotationStore())
        self._suspend_store_changed = True
        try:
            try:
                st.begin_txn()
            except Exception:
                pass
            for f in range(start, end + 1):
                cur = st.label_at(f)
                if checked:
                    if cur != name:
                        if cur:
                            st.remove_at(f)
                        st.add(name, f)
                else:
                    if cur == name:
                        st.remove_at(f)
            try:
                st.end_txn()
            except Exception:
                pass
        finally:
            self._suspend_store_changed = False
        self._on_store_changed(prompt_validation_comment=False)
        self._sync_anomaly_panel()
        self._update_phase_segment_cuts_for_entity(ename)

    def _clear_anomaly_types_for_span(self, ename: str, start: int, end: int) -> None:
        stores = self.anomaly_type_stores.get(ename, {})
        for name, st in stores.items():
            try:
                st.begin_txn()
            except Exception:
                pass
            for f in range(start, end + 1):
                if st.label_at(f) == name:
                    st.remove_at(f)
            try:
                st.end_txn()
            except Exception:
                pass

    def _on_phase_label_clicked(self, btn: QToolButton) -> None:
        label = btn.property("phase_label") if btn is not None else None
        if not label:
            return
        self._phase_active_label = str(label)
        self._apply_phase_label_to_selection(self._phase_active_label)

    def _apply_phase_label_to_selection(self, label: str) -> None:
        if not self.phase_mode_enabled or not getattr(self, "timeline", None):
            return
        row = getattr(self.timeline, "_active_combined_row", None)
        meta = getattr(row, "_group_meta", {}) if row is not None else {}
        if not row or meta.get("row_type") != "phase":
            return
        if not getattr(row, "_selected_interval", None):
            return
        try:
            start, end = row._selected_interval
        except Exception:
            return
        ok = False
        ename = meta.get("entity")
        self._suspend_store_changed = True
        try:
            try:
                ok = bool(row.apply_label_to_selection(label))
            except Exception:
                ok = False
            if ok and ename and label != "anomaly":
                self._clear_anomaly_types_for_span(ename, int(start), int(end))
        finally:
            self._suspend_store_changed = False
        if not ok:
            return
        self._on_store_changed(prompt_validation_comment=False)
        if ename:
            self._phase_selected = {
                "entity": ename,
                "start": int(start),
                "end": int(end),
                "label": label,
            }
            self._sync_anomaly_panel()
            self._update_phase_segment_cuts_for_entity(ename)
        self._dirty = True

    def _on_phase_mode_toggled(self, on: bool) -> None:
        self.phase_mode_enabled = bool(on)
        self._update_phase_panel_visibility()
        self._rebuild_timeline_sources()
        if not self.phase_mode_enabled:
            self._phase_selected = None
            self._sync_anomaly_panel()

    def _update_phase_panel_visibility(self) -> None:
        visible = self.mode == "Fine" and self._is_action_task()
        if getattr(self, "timeline", None):
            if visible:
                if not self._is_psr_task():
                    self._bind_fine_timeline_toggle(True)
            else:
                if not self._is_psr_task():
                    self._bind_fine_timeline_toggle(False)
        if getattr(self, "phase_panel", None):
            try:
                self.phase_panel.setVisible(visible)
            except Exception:
                pass
        if getattr(self, "chk_phase_mode", None):
            try:
                self.chk_phase_mode.setEnabled(visible)
                self.chk_phase_mode.setChecked(
                    bool(self.phase_mode_enabled) if visible else False
                )
            except Exception:
                pass
        if visible:
            try:
                self._refresh_anomaly_type_panel(force=True)
            except Exception:
                pass
        enable_controls = bool(visible and self.phase_mode_enabled)
        for btn in getattr(self, "_phase_buttons", {}).values():
            try:
                btn.setEnabled(enable_controls)
            except Exception:
                pass
        if enable_controls:
            self._sync_anomaly_panel()
        else:
            self._set_anomaly_panel_enabled(False, clear=True)
        try:
            self._update_left_splitter_sizes()
        except Exception:
            pass
        try:
            self._update_validation_overlay_controls()
        except Exception:
            pass

    # ----- gap checking (Action Segmentation) -----
    def _is_action_task(self) -> bool:
        try:
            text = self.combo_task.currentText() or ""
        except Exception:
            text = ""
        lower = text.lower()
        return ("action" in lower) and ("segment" in lower)

    def _gaps_for_store(
        self, store: AnnotationStore, start: int, end: int
    ) -> List[Tuple[int, int]]:
        """Return list of [start,end] gaps (inclusive) with no labels between start/end."""
        if not store or end < start:
            return []
        labeled = [int(f) for f in store.frame_to_label.keys() if start <= f <= end]
        if not labeled:
            return [(start, end)]
        labeled.sort()
        gaps = []
        cursor = start
        for f in labeled:
            if f > cursor:
                gaps.append((cursor, f - 1))
            cursor = max(cursor, f + 1)
        if cursor <= end:
            gaps.append((cursor, end))
        return gaps

    def _gap_axes_for_view(self, view: dict) -> List[Tuple[str, List[Tuple[int, int]]]]:
        """Return list of (axis_name, gaps) for a view. Fine mode reports per-entity gaps."""
        if not view:
            return []
        p = view.get("player")
        if not p or not getattr(p, "cap", None):
            return []
        start = int(view.get("start", 0))
        end = int(view.get("end", start))
        if end < start:
            return []
        if self.mode == "Fine":
            entity_names = [e.name for e in self.entities]
            view_stores = view.get("entity_stores", {})
            for ename in sorted(view_stores.keys()):
                if ename not in entity_names:
                    entity_names.append(ename)
            axes = []
            if not entity_names:
                st = view.get("store")
                return [("GLOBAL", self._gaps_for_store(st, start, end))]
            for ename in entity_names:
                st = view_stores.get(ename)
                gaps = self._gaps_for_store(st, start, end)
                axes.append((ename, gaps))
            return axes
        st_primary = view.get("store")
        return [("GLOBAL", self._gaps_for_store(st_primary, start, end))]

    def _format_gap_list(self, gaps: List[Tuple[int, int]]) -> str:
        parts = []
        for s, e in gaps:
            parts.append(f"{s}-{e}" if s != e else f"{s}")
        return ", ".join(parts)

    def _collect_unlabeled_gap_lines(self) -> List[str]:
        if not self._is_action_task():
            return []
        if not getattr(self, "views", None):
            return []
        if not any(
            v.get("player") and getattr(v["player"], "cap", None) for v in self.views
        ):
            return []
        lines = []
        for v in self.views:
            vname = v.get("name", "view")
            for axis, gaps in self._gap_axes_for_view(v):
                if gaps:
                    if self.mode == "Fine" and axis:
                        lines.append(f"{vname}/{axis}: {self._format_gap_list(gaps)}")
                    else:
                        lines.append(f"{vname}: {self._format_gap_list(gaps)}")
        return lines

    def _check_unlabeled_gaps(self, context: str = "save") -> Tuple[bool, bool]:
        """Warn about unlabeled gaps; return (proceed, had_gaps)."""
        lines = self._collect_unlabeled_gap_lines()
        if not lines:
            return True, False
        msg = (
            "Detected unlabeled frame ranges (gaps) in Action Segmentation:\n"
            + "\n".join(lines)
            + "\nContinue anyway?"
        )
        ret = QMessageBox.warning(
            self,
            "Unlabeled frames",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        self._log("gap_check", context=context, has_gaps=True, views=len(lines))
        return ret == QMessageBox.Yes, True

    def _update_gap_indicator(self):
        if not getattr(self, "timeline", None):
            return
        if not self._is_action_task():
            try:
                self.timeline.set_gap_summary("Gaps: n/a", "", False)
            except Exception:
                pass
            return
        if not getattr(self, "views", None):
            try:
                self.timeline.set_gap_summary("Gaps: n/a", "", False)
            except Exception:
                pass
            return
        if not any(
            v.get("player") and getattr(v["player"], "cap", None) for v in self.views
        ):
            try:
                self.timeline.set_gap_summary("Gaps: n/a", "", False)
            except Exception:
                pass
            return

        lines = []
        counts = []
        total_ranges = 0
        for v in self.views:
            vname = v.get("name", "view")
            for axis, gaps in self._gap_axes_for_view(v):
                if not gaps:
                    continue
                total_ranges += len(gaps)
                if self.mode == "Fine" and axis:
                    counts.append(f"{vname}/{axis}({len(gaps)})")
                    lines.append(f"{vname}/{axis}: {self._format_gap_list(gaps)}")
                else:
                    counts.append(f"{vname}({len(gaps)})")
                    lines.append(f"{vname}: {self._format_gap_list(gaps)}")

        if total_ranges == 0:
            try:
                self.timeline.set_gap_summary(
                    "Gaps: none", "No unlabeled frames detected.", False
                )
            except Exception:
                pass
            return

        summary = ", ".join(counts)
        if len(summary) > 60:
            summary = f"{len(counts)} views / {total_ranges} ranges"
        tooltip = "Unlabeled frames:\n" + "\n".join(lines)
        try:
            self.timeline.set_gap_summary(f"Gaps: {summary}", tooltip, True)
        except Exception:
            pass

    def _goto_gap(self, direction: int):
        if not self._is_action_task():
            return
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return
        view = self.views[self.active_view_idx]
        axes = self._gap_axes_for_view(view)
        entries = []
        for axis, gaps in axes:
            for s, e in gaps:
                entries.append({"start": s, "end": e, "axis": axis})
        if not entries:
            self._set_status("No gaps in the active view.")
            return
        entries.sort(key=lambda g: (g["start"], g.get("axis") or ""))
        try:
            frame = int(view["player"].current_frame)
        except Exception:
            frame = 0

        target = None
        target_axis = None
        if direction >= 0:
            for g in entries:
                if g["start"] > frame:
                    target = g["start"]
                    target_axis = g.get("axis")
                    break
        else:
            for g in reversed(entries):
                if g["end"] < frame:
                    target = g["start"]
                    target_axis = g.get("axis")
                    break

        if target is None:
            if direction >= 0:
                self._set_status("Already past the last gap.")
            else:
                self._set_status("Already before the first gap.")
            return

        self._sync_views_to_frame(target, preview_only=False)
        self._sync_other_views(target)
        self._update_overlay_for_frame(target)
        if target_axis and target_axis != "GLOBAL":
            try:
                self.timeline.focus_combined_title(target_axis)
            except Exception:
                pass
        self._log("gap_jump", direction=direction, target=target)

    @staticmethod
    def _norm_span(start, end):
        try:
            s = int(start)
        except Exception:
            s = 0
        try:
            e = int(end)
        except Exception:
            e = s
        if e < s:
            s, e = e, s
        span_txt = f"{s}" if s == e else f"{s}-{e}"
        return s, e, span_txt

    def _labels_at_frame_for(
        self,
        store: AnnotationStore,
        frame: int,
        allowed_entities: Optional[Set[str]] = None,
        entity_stores: Optional[Dict[str, AnnotationStore]] = None,
    ):
        hits = []
        name_to_color = {lb.name: color_from_key(lb.color_name) for lb in self.labels}
        if self.mode == "Coarse":
            lb = store.label_at(frame)
            if lb:
                hits.append((lb, name_to_color.get(lb)))
        else:
            stores = entity_stores if entity_stores is not None else self.entity_stores
            for ename, st in sorted(stores.items()):
                if allowed_entities is not None and ename not in allowed_entities:
                    continue
                lb = st.label_at(frame)
                if lb:
                    hits.append((f"[{ename}] {lb}", name_to_color.get(lb)))
        return hits

    @staticmethod
    def _hand_role_for_entity(name: str) -> Optional[str]:
        if not name:
            return None
        key = re.sub(r"[\s_\-]+", "", str(name).lower())
        if key in {"l", "lh", "lhand", "lefthand"} or "left" in key:
            return "left"
        if key in {"r", "rh", "rhand", "righthand"} or "right" in key:
            return "right"
        return None

    def _select_left_right_entities(
        self, names: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        left = None
        right = None
        rest = []
        for name in names:
            role = self._hand_role_for_entity(name)
            if role == "left" and left is None:
                left = name
            elif role == "right" and right is None:
                right = name
            else:
                rest.append(name)
        if left is None and rest:
            left = rest.pop(0)
        if right is None and rest:
            right = rest.pop(0)
        return left, right

    def _phase_anomaly_bars_for_entity(
        self, view: dict, ename: Optional[str], frame: int, bar_height: int
    ) -> List[Dict[str, Any]]:
        # return a single bar per entity (placeholder if missing) to keep left/right alignment
        empty = [
            {"color": QColor(0, 0, 0), "height": bar_height, "alpha": 0, "label": ""}
        ]
        if not ename:
            return empty
        pstore = (view.get("phase_stores", {}) or {}).get(ename)
        if not pstore:
            return empty
        phase_label = pstore.label_at(frame)
        if not phase_label:
            return empty
        phase_colors = {
            lb.name: color_from_key(lb.color_name) for lb in self.phase_labels
        }
        phase_color = phase_colors.get(
            str(phase_label).lower(), phase_colors.get(phase_label)
        ) or color_from_key("Gray")
        label_text = ""
        label_color = "#ffffff"
        alpha = 170
        if str(phase_label).lower() == "anomaly":
            type_map = (view.get("anomaly_type_stores", {}) or {}).get(ename, {})
            active_types = []
            for name in self._anomaly_type_names():
                st = type_map.get(name)
                if st is None:
                    continue
                try:
                    if st.label_at(frame):
                        active_types.append(name)
                except Exception:
                    continue
            if active_types:
                label_text = ",".join(
                    [self._short_anomaly_name(n) for n in active_types]
                )
        return [
            {
                "color": phase_color,
                "height": bar_height,
                "alpha": alpha,
                "label": label_text,
                "label_color": label_color,
            }
        ]

    def _overlay_phase_anomaly_bars_for_view(
        self,
        frame: int,
        view: dict,
        allowed_entities: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if self.mode != "Fine" or not self.phase_mode_enabled:
            return []
        bar_height = 14
        names = list((view.get("phase_stores", {}) or {}).keys())
        if allowed_entities:
            names = [n for n in names if n in allowed_entities]
        if not names:
            return []
        left, right = self._select_left_right_entities(names)
        bars: List[Dict[str, Any]] = []
        bars.extend(self._phase_anomaly_bars_for_entity(view, left, frame, bar_height))
        bars.extend(self._phase_anomaly_bars_for_entity(view, right, frame, bar_height))
        return bars

    @staticmethod
    def _short_anomaly_name(name: str) -> str:
        raw = str(name or "").strip()
        if not raw:
            return ""
        low = raw.lower()
        if low.startswith("error_"):
            low = low[len("error_") :]
        low = low.replace("-", "_").replace(" ", "_")
        mapping = {
            "temporal": "T",
            "spatial": "S",
            "handling": "H",
            "wrong_part": "WP",
            "wrong_tool": "WT",
            "procedural": "P",
        }
        if low in mapping:
            return mapping[low]
        parts = [p for p in re.split(r"[^a-z0-9]+", low) if p]
        if not parts:
            return low[:4]
        if len(parts) == 1:
            return parts[0][:4]
        initials = "".join(p[0] for p in parts if p)
        if len(initials) >= 3:
            return initials[:4]
        return (parts[0][:2] + parts[1][:2])[:4]

    @staticmethod
    def _prefix_overlay_labels(labels, prefix: str):
        if not labels:
            return []
        return [(f"{prefix}{text}", col) for (text, col) in labels]

    def _hit_names_for_frame(self, frame: int):
        """Return a set of label names that cover the frame, including entity-stripped variants."""
        names = [name for name, _ in self._labels_at_frame_for(self.store, frame)]
        hits = set(names)
        for n in names:
            if n.startswith("[") and "]" in n:
                tail = n.split("]", 1)[1].strip()
                if tail:
                    hits.add(tail)
        return hits

    def _overlay_entity_filter(self) -> Optional[Set[str]]:
        if self.mode != "Fine" or not self.validation_enabled:
            return None
        return set(self.visible_entities or [])

    def _update_overlay_for_frame(self, frame: int):
        if not getattr(self, "player", None):
            return
        multi = (
            self.validation_enabled or getattr(self, "review_panel", None).isVisible()
        )
        allowed_entities = self._overlay_entity_filter()
        for i, vw in enumerate(self.views):
            overlay_mode = self._validation_overlay_mode or "action"
            labels = []
            bars = []
            if overlay_mode in ("action", "both"):
                action_labels = self._labels_at_frame_for(
                    vw["store"], frame, allowed_entities, vw.get("entity_stores")
                )
                if overlay_mode == "both":
                    action_labels = self._prefix_overlay_labels(action_labels, "A ")
                labels.extend(action_labels)
            if overlay_mode in ("phase", "both"):
                bars = self._overlay_phase_anomaly_bars_for_view(
                    frame, vw, allowed_entities
                )
            try:
                vw["player"].set_overlay_labels(labels if multi else [])
                vw["player"].set_overlay_bars(bars if multi else [])
                vw["player"].set_overlay_enabled(multi and bool(labels or bars))
            except Exception:
                pass

    def _update_overlay_for_view(self, view_idx: int, frame: int):
        if not self.views or not (0 <= view_idx < len(self.views)):
            return
        multi = (
            self.validation_enabled or getattr(self, "review_panel", None).isVisible()
        )
        vw = self.views[view_idx]
        overlay_mode = self._validation_overlay_mode or "action"
        labels = []
        bars = []
        if overlay_mode in ("action", "both"):
            action_labels = self._labels_at_frame_for(
                vw["store"],
                frame,
                self._overlay_entity_filter(),
                vw.get("entity_stores"),
            )
            if overlay_mode == "both":
                action_labels = self._prefix_overlay_labels(action_labels, "A ")
            labels.extend(action_labels)
        if overlay_mode in ("phase", "both"):
            bars = self._overlay_phase_anomaly_bars_for_view(
                frame, vw, self._overlay_entity_filter()
            )
        try:
            vw["player"].set_overlay_labels(labels if multi else [])
            vw["player"].set_overlay_bars(bars if multi else [])
            vw["player"].set_overlay_enabled(multi and bool(labels or bars))
        except Exception:
            pass

    def _auto_color_key_for_id(self, lid: int) -> str:
        # assign preset color by id (skip Gray as default when possible)
        palette_keys = [k for k in PRESET_COLORS.keys() if k.lower() != "gray"]
        if not palette_keys:
            return "Gray"
        return palette_keys[lid % len(palette_keys)]

    @staticmethod
    def _cfg_int(value: Any, default: int, lo: int = 0, hi: int = 10000) -> int:
        try:
            out = int(value)
        except Exception:
            out = int(default)
        return max(int(lo), min(int(hi), out))

    @staticmethod
    def _cfg_float(
        value: Any,
        default: float,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
    ) -> float:
        try:
            out = float(value)
        except Exception:
            out = float(default)
        if lo is not None:
            out = max(float(lo), out)
        if hi is not None:
            out = min(float(hi), out)
        return out

    def _ensure_algo_cfg_defaults(self) -> None:
        if not isinstance(getattr(self, "_algo_cfg", None), dict):
            self._algo_cfg = {}
        cfg = self._algo_cfg
        timeline_cfg = cfg.setdefault("timeline_snap", {})
        if not isinstance(timeline_cfg, dict):
            timeline_cfg = {}
            cfg["timeline_snap"] = timeline_cfg
        timeline_cfg.setdefault("playhead_radius", CURRENT_FRAME_SNAP_RADIUS_FRAMES)
        timeline_cfg.setdefault("empty_space_radius", SNAP_RADIUS_FRAMES)
        timeline_cfg.setdefault("edge_search_radius", EDGE_SNAP_FRAMES)
        timeline_cfg.setdefault("segment_soft_radius", SNAP_RADIUS_FRAMES)
        timeline_cfg.setdefault("phase_soft_radius", 8)
        timeline_cfg.setdefault("hover_preview_multi", True)
        timeline_cfg.setdefault("hover_preview_align", "absolute")

        boundary_cfg = cfg.setdefault("boundary_snap", {})
        if not isinstance(boundary_cfg, dict):
            boundary_cfg = {}
            cfg["boundary_snap"] = boundary_cfg
        boundary_cfg.setdefault("enabled", True)
        boundary_cfg.setdefault("window_size", 15)

        seg_cfg = cfg.setdefault("segment_embedding", {})
        if not isinstance(seg_cfg, dict):
            seg_cfg = {}
            cfg["segment_embedding"] = seg_cfg
        seg_cfg.setdefault("trim_ratio", 0.1)

        topk_cfg = cfg.setdefault("topk", {})
        if not isinstance(topk_cfg, dict):
            topk_cfg = {}
            cfg["topk"] = topk_cfg
        topk_cfg.setdefault("enabled", True)
        topk_cfg.setdefault("k", 5)
        if "uncertainty_margin" not in topk_cfg:
            topk_cfg["uncertainty_margin"] = 0.25

        assist_cfg = cfg.setdefault("assisted", {})
        if not isinstance(assist_cfg, dict):
            assist_cfg = {}
            cfg["assisted"] = assist_cfg
        assist_cfg.setdefault("boundary_min_gap", 15)
        assist_cfg.setdefault("sort_by", "query_score")
        assist_cfg.setdefault("label_query_min", 0.35)
        assist_cfg.setdefault("boundary_query_min", 0.30)
        assist_cfg.setdefault("label_uncertainty_weight", 0.55)
        assist_cfg.setdefault("label_disagreement_weight", 0.20)
        assist_cfg.setdefault("label_confusion_weight", 0.25)
        assist_cfg.setdefault("boundary_uncertainty_weight", 0.55)
        assist_cfg.setdefault("boundary_confusion_weight", 0.20)
        assist_cfg.setdefault("boundary_energy_weight", 0.10)
        assist_cfg.setdefault("boundary_merge_weight", 0.15)

        psr_cfg = cfg.setdefault("psr", {})
        if not isinstance(psr_cfg, dict):
            psr_cfg = {}
            cfg["psr"] = psr_cfg
        # auto|installed|not_installed
        psr_cfg.setdefault("initial_state_policy", "auto")
        psr_cfg.setdefault("no_gap_timeline", True)
        psr_cfg.setdefault("auto_carry_next_on_edit", True)
        psr_cfg.setdefault("model_type", self._psr_default_model_type())

    def _timeline_snap_cfg(self) -> Dict[str, int]:
        self._ensure_algo_cfg_defaults()
        cfg = {}
        try:
            cfg = dict(self._algo_cfg.get("timeline_snap", {}))
        except Exception:
            cfg = {}
        return {
            "playhead_radius": self._cfg_int(
                cfg.get("playhead_radius", CURRENT_FRAME_SNAP_RADIUS_FRAMES),
                CURRENT_FRAME_SNAP_RADIUS_FRAMES,
                lo=0,
                hi=120,
            ),
            "empty_space_radius": self._cfg_int(
                cfg.get("empty_space_radius", SNAP_RADIUS_FRAMES),
                SNAP_RADIUS_FRAMES,
                lo=0,
                hi=120,
            ),
            "edge_search_radius": self._cfg_int(
                cfg.get("edge_search_radius", EDGE_SNAP_FRAMES),
                EDGE_SNAP_FRAMES,
                lo=0,
                hi=60,
            ),
            "segment_soft_radius": self._cfg_int(
                cfg.get("segment_soft_radius", SNAP_RADIUS_FRAMES),
                SNAP_RADIUS_FRAMES,
                lo=0,
                hi=120,
            ),
            "phase_soft_radius": self._cfg_int(
                cfg.get("phase_soft_radius", 8), 8, lo=0, hi=120
            ),
        }

    def _timeline_hover_preview_cfg(self) -> Dict[str, Any]:
        self._ensure_algo_cfg_defaults()
        cfg = {}
        try:
            cfg = dict(self._algo_cfg.get("timeline_snap", {}))
        except Exception:
            cfg = {}
        align = str(cfg.get("hover_preview_align", "absolute") or "").strip().lower()
        if align not in ("absolute", "offset"):
            align = "absolute"
        return {
            "enabled_multi": bool(cfg.get("hover_preview_multi", True)),
            "align": align,
        }

    def _apply_timeline_snap_settings(self, refresh: bool = False) -> None:
        if not getattr(self, "timeline", None):
            return
        cfg = self._timeline_snap_cfg()
        try:
            self.timeline.set_snap_tuning(
                current_frame_radius=cfg.get("playhead_radius"),
                frame_snap_radius=cfg.get("empty_space_radius"),
                edge_snap_frames=cfg.get("edge_search_radius"),
                segment_snap_radius=cfg.get("segment_soft_radius"),
                refresh=bool(refresh),
            )
        except Exception:
            pass

    # ----- Boundary snap helpers -----
    def _boundary_snap_cfg(self) -> Tuple[bool, int]:
        self._ensure_algo_cfg_defaults()
        cfg = {}
        try:
            cfg = dict(getattr(self, "_algo_cfg", {}).get("boundary_snap", {}))
        except Exception:
            cfg = {}
        enabled = cfg.get("enabled", True)
        window = cfg.get("window_size", 15)
        try:
            window = max(1, int(window))
        except Exception:
            window = 15
        return bool(enabled), window

    def _resolve_features_dir_for_snap(self) -> Optional[str]:
        features_dir = getattr(self, "currentFeatureDir", None)
        if not features_dir or not os.path.isdir(features_dir):
            guess = self._default_features_dir_for_video()
            if guess and os.path.isdir(guess):
                features_dir = guess
        if not features_dir:
            return None
        if not os.path.isfile(os.path.join(features_dir, "features.npy")):
            return None
        return features_dir

    def _boundary_features_dir_candidates(self) -> List[str]:
        candidates: List[str] = []
        for raw in (
            getattr(self, "currentEastFeatureDir", None),
            self._default_east_features_dir_for_video(),
            getattr(self, "currentFeatureDir", None),
            self._default_features_dir_for_video(),
        ):
            path = str(raw or "").strip()
            if not path:
                continue
            path = os.path.abspath(path)
            if path in candidates:
                continue
            if not os.path.isdir(path):
                continue
            if not os.path.isfile(os.path.join(path, "features.npy")):
                continue
            candidates.append(path)
        return candidates

    def _ensure_boundary_snap_cache(self, features_dir: str) -> Dict[str, Any]:
        cache = getattr(self, "_boundary_snap_cache", None)
        if not isinstance(cache, dict):
            cache = {}
        if cache.get("features_dir") != features_dir:
            cache = {"features_dir": features_dir}
            self._boundary_snap_cache = cache
        return cache

    def _load_feature_meta(self, features_dir: str) -> Dict[str, Any]:
        cache = self._ensure_boundary_snap_cache(features_dir)
        if "meta" in cache:
            return cache.get("meta") or {}
        meta = {}
        meta_path = os.path.join(features_dir, "meta.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        cache["meta"] = meta
        return meta

    def _feature_frame_map(self, features_dir: str, seq_len: int) -> List[int]:
        meta = self._load_feature_meta(features_dir)
        picked = meta.get("picked_indices")
        if isinstance(picked, list) and len(picked) == seq_len:
            return [int(x) for x in picked]
        stride = meta.get("frame_stride")
        if stride is not None:
            try:
                stride = int(stride)
                return [i * stride for i in range(seq_len)]
            except Exception:
                pass
        return list(range(seq_len))

    def _find_boundary_logits_path(self, features_dir: str) -> Optional[str]:
        for name in (
            "frame_logits.npy",
            "logits.npy",
            "pred_asot_logits.npy",
            "pred_asot_full_logits.npy",
            "pred_fact_logits.npy",
        ):
            path = os.path.join(features_dir, name)
            if os.path.isfile(path):
                return path
        return None

    def _get_boundary_series(self) -> Optional[Dict[str, Any]]:
        for features_dir in self._boundary_features_dir_candidates():
            cache = self._ensure_boundary_snap_cache(features_dir)

            boundary_path = os.path.join(features_dir, "east_runtime", "boundary.npy")
            if os.path.isfile(boundary_path):
                if cache.get("boundary_path") != boundary_path:
                    try:
                        cache["boundary_score"] = np.load(boundary_path, mmap_mode="r")
                    except Exception:
                        cache["boundary_score"] = None
                    cache["boundary_path"] = boundary_path
                boundary = cache.get("boundary_score")
                if isinstance(boundary, np.ndarray) and boundary.ndim >= 1:
                    seq_len = int(boundary.shape[0])
                    frame_map = self._feature_frame_map(features_dir, seq_len)
                    return {
                        "source": "east_boundary",
                        "data": boundary,
                        "frame_map": frame_map,
                        "length": seq_len,
                        "features_dir": features_dir,
                    }

            logits_path = self._find_boundary_logits_path(features_dir)
            if logits_path:
                if cache.get("logits_path") != logits_path:
                    try:
                        cache["logits"] = np.load(logits_path, mmap_mode="r")
                    except Exception:
                        cache["logits"] = None
                    cache["logits_path"] = logits_path
                logits = cache.get("logits")
                if isinstance(logits, np.ndarray) and logits.ndim >= 2:
                    seq_len = int(logits.shape[0])
                    frame_map = self._feature_frame_map(features_dir, seq_len)
                    return {
                        "source": "logits",
                        "data": logits,
                        "frame_map": frame_map,
                        "length": seq_len,
                        "features_dir": features_dir,
                    }

            if cache.get("features") is None:
                feat_path = os.path.join(features_dir, "features.npy")
                try:
                    cache["features"] = np.load(feat_path, mmap_mode="r")
                except Exception:
                    cache["features"] = None
                feat = cache.get("features")
                if isinstance(feat, np.ndarray) and feat.ndim == 2:
                    h, w = feat.shape
                    if h == 2048 and w != 2048:
                        cache["features_layout"] = "BDT"
                    elif w == 2048 and h != 2048:
                        cache["features_layout"] = "BTD"
                    else:
                        cache["features_layout"] = "BTD"

            feat = cache.get("features")
            if not isinstance(feat, np.ndarray) or feat.ndim != 2:
                continue
            layout = cache.get("features_layout", "BTD")
            seq_len = int(feat.shape[1] if layout == "BDT" else feat.shape[0])
            frame_map = self._feature_frame_map(features_dir, seq_len)
            return {
                "source": "features",
                "data": feat,
                "frame_map": frame_map,
                "length": seq_len,
                "layout": layout,
                "features_dir": features_dir,
            }
        return None

    def _series_vector(self, series: Dict[str, Any], idx: int) -> Optional[np.ndarray]:
        data = series.get("data")
        if data is None:
            return None
        if series.get("source") == "features" and series.get("layout") == "BDT":
            return data[:, idx]
        return data[idx]

    def _boundary_energy_at(self, series: Dict[str, Any], idx: int) -> Optional[float]:
        if idx <= 0:
            return None
        data_len = int(series.get("length") or 0)
        if idx >= data_len:
            return None
        if series.get("source") == "east_boundary":
            data = series.get("data")
            if data is None:
                return None
            try:
                val = float(np.asarray(data[idx]).reshape(-1)[0])
            except Exception:
                return None
            if not np.isfinite(val):
                return 0.0
            return float(np.clip(val, 0.0, 1.0))
        v0 = self._series_vector(series, idx - 1)
        v1 = self._series_vector(series, idx)
        if v0 is None or v1 is None:
            return None
        if series.get("source") == "logits":
            val = float(np.linalg.norm(v1 - v0))
        else:
            denom = float(np.linalg.norm(v0) * np.linalg.norm(v1))
            if denom <= 0:
                return 0.0
            val = float(1.0 - (np.dot(v0, v1) / denom))
        if not np.isfinite(val):
            return 0.0
        return val

    def _boundary_energy_at_frame(self, frame: int) -> Optional[float]:
        series = self._get_boundary_series()
        if not series:
            return None
        frame_map = series.get("frame_map") or []
        if not frame_map:
            return None
        try:
            frame = int(frame)
        except Exception:
            return None
        idx = bisect.bisect_left(frame_map, frame)
        if idx >= len(frame_map):
            idx = len(frame_map) - 1
        if idx > 0:
            prev = idx - 1
            if abs(frame_map[prev] - frame) <= abs(frame_map[idx] - frame):
                idx = prev
        return self._boundary_energy_at(series, idx)

    def _boundary_review_score_at_frame(self, frame: int) -> Optional[float]:
        series = self._get_boundary_series()
        if not series:
            return None
        if series.get("source") != "east_boundary":
            return None
        frame_map = series.get("frame_map") or []
        if not frame_map:
            return None
        try:
            frame = int(frame)
        except Exception:
            return None
        idx = bisect.bisect_left(frame_map, frame)
        if idx >= len(frame_map):
            idx = len(frame_map) - 1
        if idx > 0:
            prev = idx - 1
            if abs(frame_map[prev] - frame) <= abs(frame_map[idx] - frame):
                idx = prev
        return self._boundary_energy_at(series, idx)

    def _throttle_boundary_points(self, points: List[dict]) -> List[dict]:
        gap = self._assisted_boundary_min_gap()
        if gap <= 0 or len(points) <= 1:
            return points
        scored = []
        for pt in points:
            try:
                frame = int(pt.get("frame", 0))
            except Exception:
                frame = 0
            try:
                score = float(pt.get("query_score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            if score <= 0.0:
                energy = self._boundary_energy_at_frame(frame)
                score = 0.0 if energy is None else float(energy)
            scored.append((frame, score, pt))
        scored.sort(key=lambda x: x[0])

        def pick_cluster(cluster):
            for item in cluster:
                if item[2].get("status") == "ACTIVE":
                    return item[2]
            return max(cluster, key=lambda x: x[1])[2]

        kept = []
        cluster = [scored[0]]
        for item in scored[1:]:
            if item[0] - cluster[-1][0] <= gap:
                cluster.append(item)
            else:
                kept.append(pick_cluster(cluster))
                cluster = [item]
        if cluster:
            kept.append(pick_cluster(cluster))
        return kept

    def _snap_boundary_frame(
        self, frame: int, lo: Optional[int] = None, hi: Optional[int] = None
    ) -> int:
        enabled, window = self._boundary_snap_cfg()
        if not enabled:
            return int(frame)
        try:
            frame = int(frame)
        except Exception:
            return frame
        if lo is not None:
            frame = max(int(lo), frame)
        if hi is not None:
            frame = min(int(hi), frame)
        win_start = frame - window
        win_end = frame + window
        if lo is not None:
            win_start = max(win_start, int(lo))
        if hi is not None:
            win_end = min(win_end, int(hi))
        if win_end < win_start:
            return frame

        series = self._get_boundary_series()
        if series and series.get("source") != "east_boundary":
            candidates = [
                c
                for c in (self._auto_boundary_candidates or [])
                if win_start <= c <= win_end
            ]
            if candidates:
                return int(min(candidates, key=lambda c: (abs(c - frame), c)))
        if not series:
            candidates = [
                c
                for c in (self._auto_boundary_candidates or [])
                if win_start <= c <= win_end
            ]
            if candidates:
                return int(min(candidates, key=lambda c: (abs(c - frame), c)))
            return frame
        frame_map = series.get("frame_map") or []
        data_len = int(series.get("length") or 0)
        if not frame_map or data_len <= 1:
            return frame
        if len(frame_map) > data_len:
            frame_map = frame_map[:data_len]

        start_idx = bisect.bisect_left(frame_map, win_start)
        end_idx = bisect.bisect_right(frame_map, win_end)
        if end_idx <= start_idx:
            return frame

        best = None
        for idx in range(max(1, start_idx), end_idx):
            energy = self._boundary_energy_at(series, idx)
            if energy is None:
                continue
            f = int(frame_map[idx])
            score = (energy, -abs(f - frame))
            if best is None or score > best[0]:
                best = (score, f)
        if best is None:
            candidates = [
                c
                for c in (self._auto_boundary_candidates or [])
                if win_start <= c <= win_end
            ]
            if candidates:
                return int(min(candidates, key=lambda c: (abs(c - frame), c)))
            return frame
        return int(best[1])

    def _move_active_boundary_to(self, new_frame: int) -> bool:
        pt = self._active_assisted_point()
        if not pt or pt.get("type") != "boundary":
            return False
        try:
            new_frame = int(new_frame)
        except Exception:
            return False
        old_frame = int(pt.get("frame", 0))
        left = pt.get("left", {})
        right = pt.get("right", {})
        min_frame = int(left.get("start", 0)) + 1
        max_frame = int(right.get("end", right.get("start", old_frame)))
        new_frame = max(min_frame, min(new_frame, max_frame))
        if new_frame == old_frame:
            return False
        try:
            self.store.begin_txn()
        except Exception:
            pass
        if new_frame < old_frame:
            rng = range(new_frame, old_frame)
            target_label = right.get("label")
        else:
            rng = range(old_frame, new_frame)
            target_label = left.get("label")
        for f in rng:
            cur = self.store.label_at(f)
            if cur and cur != target_label:
                self.store.remove_at(f)
            if target_label:
                self.store.add(target_label, f)
        try:
            self.store.end_txn()
        except Exception:
            pass
        self._note_correction_step()
        self._dirty = True
        self._build_assisted_points_from_store(
            preserve_status=True, active_hint=("boundary", new_frame)
        )
        self._update_assisted_visuals()
        return True

    # ----- Segment embedding helpers -----
    def _segment_embedding_cfg(self) -> float:
        self._ensure_algo_cfg_defaults()
        cfg = {}
        try:
            cfg = dict(getattr(self, "_algo_cfg", {}).get("segment_embedding", {}))
        except Exception:
            cfg = {}
        trim_ratio = cfg.get("trim_ratio", 0.1)
        try:
            trim_ratio = float(trim_ratio)
        except Exception:
            trim_ratio = 0.1
        if trim_ratio < 0:
            trim_ratio = 0.0
        return trim_ratio

    def _topk_cfg(self) -> Dict[str, Any]:
        self._ensure_algo_cfg_defaults()
        cfg = {}
        try:
            cfg = dict(getattr(self, "_algo_cfg", {}).get("topk", {}))
        except Exception:
            cfg = {}
        return cfg

    def _assisted_cfg(self) -> Dict[str, Any]:
        self._ensure_algo_cfg_defaults()
        cfg = {}
        try:
            cfg = dict(getattr(self, "_algo_cfg", {}).get("assisted", {}))
        except Exception:
            cfg = {}
        return cfg

    def _assisted_boundary_min_gap(self) -> int:
        cfg = self._assisted_cfg()
        try:
            gap = int(cfg.get("boundary_min_gap", 0))
        except Exception:
            gap = 0
        return max(0, gap)

    def _topk_enabled(self) -> bool:
        cfg = self._topk_cfg()
        return bool(cfg.get("enabled", True))

    def _topk_k(self) -> int:
        cfg = self._topk_cfg()
        try:
            k = int(cfg.get("k", self._interaction_cfg["label"]["top_k"]))
        except Exception:
            k = int(self._interaction_cfg["label"]["top_k"])
        return max(1, k)

    def _topk_uncertainty_margin(self) -> Optional[float]:
        cfg = self._topk_cfg()
        val = cfg.get("uncertainty_margin", None)
        if val is None:
            return None
        try:
            margin = float(val)
        except Exception:
            return None
        return margin

    def _assisted_cfg(self) -> Dict[str, Any]:
        self._ensure_algo_cfg_defaults()
        cfg = dict(getattr(self, "_algo_cfg", {}).get("assisted", {}))
        return cfg if isinstance(cfg, dict) else {}

    def _assisted_sort_mode(self) -> str:
        mode = str(self._assisted_cfg().get("sort_by", "query_score") or "query_score").strip().lower()
        return mode or "query_score"

    def _assisted_query_threshold(self, point_type: str) -> float:
        cfg = self._assisted_cfg()
        key = "boundary_query_min" if str(point_type or "").strip().lower() == "boundary" else "label_query_min"
        default = 0.30 if key == "boundary_query_min" else 0.35
        return self._cfg_float(cfg.get(key, default), default, lo=0.0, hi=1.0)

    def _current_runtime_confusion_map(self) -> Dict[str, Dict[str, float]]:
        features_dir = self._correction_features_dir()
        if not features_dir:
            return {}
        path = os.path.join(os.path.abspath(features_dir), "east_runtime", "model_delta.pt")
        try:
            stamp = float(os.path.getmtime(path)) if os.path.isfile(path) else -1.0
        except Exception:
            stamp = -1.0
        cache = getattr(self, "_assisted_confusion_cache", None)
        if (
            isinstance(cache, dict)
            and cache.get("path") == path
            and float(cache.get("stamp", -2.0) or -2.0) == float(stamp)
        ):
            return dict(cache.get("data") or {})
        try:
            data = load_east_runtime_confusion_map(features_dir)
        except Exception:
            data = {}
        self._assisted_confusion_cache = {
            "path": path,
            "stamp": float(stamp),
            "data": dict(data or {}),
        }
        return dict(data or {})

    def _label_support_count(self, label_name: str) -> float:
        label = str(label_name or "").strip()
        if not label:
            return 0.0
        try:
            local = float(self._label_proto_counts.get(label, 0) or 0.0)
        except Exception:
            local = 0.0
        return max(0.0, float(local))

    def _label_scarcity_score(self, label_name: str) -> float:
        count = self._label_support_count(label_name)
        if count <= 0.0:
            return 1.0
        try:
            score = 1.0 - min(1.0, float(np.log1p(count) / np.log(9.0)))
        except Exception:
            score = 0.0
        return max(0.0, min(1.0, float(score)))

    def _label_training_utility_score(
        self,
        current_label: str,
        candidates: List[Tuple[str, Optional[float]]],
        *,
        source: str = "model",
    ) -> float:
        current = str(current_label or "").strip()
        candidate_names = [str(name or "").strip() for name, _score in list(candidates or [])[:4] if str(name or "").strip()]
        top = candidate_names[0] if candidate_names else ""
        scarcity_labels = [name for name in [current, top] if name]
        scarcity = max((self._label_scarcity_score(name) for name in scarcity_labels), default=0.0)
        alt_count = len([name for name in candidate_names if name and name != current])
        hard_negative_potential = max(0.0, min(1.0, float(alt_count) / 3.0))
        confusion_gain = self._label_confusion_query_score(current, candidates) if current and candidates else 0.0
        source_bonus = 0.15 if str(source or "").strip().lower() == "embedding" else 0.0
        score = 0.45 * scarcity + 0.30 * hard_negative_potential + 0.20 * confusion_gain + source_bonus
        return max(0.0, min(1.0, float(score)))

    def _boundary_training_utility_score(
        self,
        left_label: str,
        right_label: str,
    ) -> float:
        left = str(left_label or "").strip()
        right = str(right_label or "").strip()
        scarcity = max(self._label_scarcity_score(left), self._label_scarcity_score(right))
        cross_label = 0.0
        if left and right:
            cross_label = 0.55 if left == right else 0.85
        confusion_gain = self._boundary_confusion_query_score(left, right)
        score = 0.50 * scarcity + 0.30 * confusion_gain + 0.20 * cross_label
        return max(0.0, min(1.0, float(score)))

    def _confusion_lookup_score(self, positive_label: str, mistaken_label: str) -> float:
        pos = str(positive_label or "").strip()
        neg = str(mistaken_label or "").strip()
        if not pos or not neg or pos == neg:
            return 0.0
        row = dict(self._current_runtime_confusion_map().get(pos) or {})
        if not row:
            return 0.0
        try:
            value = float(row.get(neg, 0.0) or 0.0)
        except Exception:
            value = 0.0
        if value <= 0.0:
            return 0.0
        max_val = 0.0
        for raw in row.values():
            try:
                max_val = max(max_val, float(raw))
            except Exception:
                continue
        if max_val <= 1e-6:
            return 0.0
        return max(0.0, min(1.0, float(value / max_val)))

    def _label_confusion_query_score(
        self,
        current_label: str,
        candidates: List[Tuple[str, Optional[float]]],
    ) -> float:
        current = str(current_label or "").strip()
        if not current or not candidates:
            return 0.0
        best = 0.0
        for name, _score in list(candidates)[:3]:
            cand = str(name or "").strip()
            if not cand or cand == current:
                continue
            best = max(
                best,
                self._confusion_lookup_score(cand, current),
                0.6 * self._confusion_lookup_score(current, cand),
            )
        return max(0.0, min(1.0, float(best)))

    def _boundary_confusion_query_score(
        self,
        left_label: str,
        right_label: str,
    ) -> float:
        left = str(left_label or "").strip()
        right = str(right_label or "").strip()
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        return max(
            self._confusion_lookup_score(left, right),
            self._confusion_lookup_score(right, left),
        )

    def _normalize_boundary_energy_for_query(self, frame: int) -> float:
        energy = self._boundary_energy_at_frame(int(frame))
        if energy is None:
            return 0.0
        val = float(max(0.0, energy))
        return max(0.0, min(1.0, val))

    def _boundary_uncertainty_query_score(self, raw_score: Optional[float]) -> float:
        if raw_score is None:
            return 0.45
        try:
            val = float(raw_score)
        except Exception:
            return 0.0
        lo, hi = self._interaction_cfg["boundary"].get("uncertainty_range", (0.4, 0.55))
        try:
            lo_f = float(lo)
            hi_f = float(hi)
        except Exception:
            lo_f, hi_f = 0.4, 0.55
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        center = (lo_f + hi_f) * 0.5
        radius = max(0.05, abs(hi_f - lo_f) * 0.5)
        score = 1.0 - abs(val - center) / radius
        return max(0.0, min(1.0, float(score)))

    def _boundary_query_score(
        self,
        *,
        frame: int,
        left_label: str,
        right_label: str,
        raw_score: Optional[float],
    ) -> Tuple[float, Dict[str, float]]:
        cfg = self._assisted_cfg()
        uncertainty = self._boundary_uncertainty_query_score(raw_score)
        confusion = self._boundary_confusion_query_score(left_label, right_label)
        energy = self._normalize_boundary_energy_for_query(frame)
        merge_bonus = 1.0 if str(left_label or "").strip() == str(right_label or "").strip() and str(left_label or "").strip() else 0.0
        utility = self._boundary_training_utility_score(left_label, right_label)
        wu = self._cfg_float(cfg.get("boundary_uncertainty_weight", 0.55), 0.55, lo=0.0, hi=1.0)
        wc = self._cfg_float(cfg.get("boundary_confusion_weight", 0.20), 0.20, lo=0.0, hi=1.0)
        we = self._cfg_float(cfg.get("boundary_energy_weight", 0.10), 0.10, lo=0.0, hi=1.0)
        wm = self._cfg_float(cfg.get("boundary_merge_weight", 0.15), 0.15, lo=0.0, hi=1.0)
        wt = self._cfg_float(cfg.get("boundary_training_utility_weight", 0.15), 0.15, lo=0.0, hi=1.0)
        score = float(wu * uncertainty + wc * confusion + we * energy + wm * merge_bonus + wt * utility)
        score = max(0.0, min(1.0, score))
        return score, {
            "uncertainty": float(uncertainty),
            "confusion": float(confusion),
            "energy": float(energy),
            "merge_bonus": float(merge_bonus),
            "training_utility": float(utility),
        }

    def _label_query_score(
        self,
        seg: dict,
        candidates: List[Tuple[str, Optional[float]]],
        *,
        source: str = "model",
    ) -> Tuple[float, Dict[str, float]]:
        current_label = str(seg.get("label", "") or "").strip()
        top_label = str(candidates[0][0] or "").strip() if candidates else ""
        top_conf = candidates[0][1] if candidates else None
        second_conf = candidates[1][1] if len(candidates) > 1 else None
        uncertainty = 0.0
        if source == "embedding":
            margin_thr = self._topk_uncertainty_margin()
            margin_ref = float(margin_thr) if margin_thr is not None else 0.25
            margin_ref = max(0.05, float(margin_ref))
            if top_conf is None:
                uncertainty = 0.55
            else:
                try:
                    top_val = float(top_conf)
                except Exception:
                    top_val = 0.0
                low_conf = max(0.0, min(1.0, 1.0 - ((top_val + 1.0) * 0.5)))
                if second_conf is None:
                    uncertainty = low_conf
                else:
                    try:
                        margin = float(top_conf) - float(second_conf)
                    except Exception:
                        margin = 0.0
                    uncertainty = max(low_conf, max(0.0, min(1.0, 1.0 - margin / margin_ref)))
        else:
            cfg = self._interaction_cfg["label"]
            if top_conf is None and second_conf is None:
                uncertainty = 0.0
            else:
                low_conf = 0.0
                if top_conf is not None:
                    try:
                        min_conf = float(cfg.get("min_confidence", 0.6))
                        top_val = float(top_conf)
                        low_conf = max(0.0, min(1.0, (min_conf - top_val) / max(min_conf, 1e-6)))
                    except Exception:
                        low_conf = 0.0
                margin_uncert = 0.0
                if top_conf is not None and second_conf is not None:
                    try:
                        diff_eps = float(cfg.get("diff_eps", 0.05))
                        margin = float(top_conf) - float(second_conf)
                        margin_uncert = max(0.0, min(1.0, 1.0 - margin / max(diff_eps, 1e-3)))
                    except Exception:
                        margin_uncert = 0.0
                uncertainty = max(low_conf, margin_uncert)
        disagreement = 1.0 if current_label and top_label and current_label != top_label else 0.0
        confusion = self._label_confusion_query_score(current_label, candidates)
        utility = self._label_training_utility_score(current_label, candidates, source=source)
        cfg = self._assisted_cfg()
        wu = self._cfg_float(cfg.get("label_uncertainty_weight", 0.55), 0.55, lo=0.0, hi=1.0)
        wd = self._cfg_float(cfg.get("label_disagreement_weight", 0.20), 0.20, lo=0.0, hi=1.0)
        wc = self._cfg_float(cfg.get("label_confusion_weight", 0.25), 0.25, lo=0.0, hi=1.0)
        wt = self._cfg_float(cfg.get("label_training_utility_weight", 0.15), 0.15, lo=0.0, hi=1.0)
        score = float(wu * uncertainty + wd * disagreement + wc * confusion + wt * utility)
        score = max(0.0, min(1.0, score))
        return score, {
            "uncertainty": float(uncertainty),
            "disagreement": float(disagreement),
            "confusion": float(confusion),
            "training_utility": float(utility),
        }

    def _update_label_prototype(
        self, label: Optional[str], emb: Optional[np.ndarray]
    ) -> None:
        if not label or emb is None:
            return
        vec = np.asarray(emb, dtype=np.float32)
        if vec.ndim != 1:
            vec = vec.reshape(-1)
        if vec.size == 0:
            return
        count = int(self._label_proto_counts.get(label, 0))
        proto = self._label_prototypes.get(label)
        if proto is None or count <= 0:
            new_proto = vec
            count = 1
        else:
            if proto.shape != vec.shape:
                return
            new_proto = (proto * count + vec) / (count + 1)
            count += 1
        norm = float(np.linalg.norm(new_proto))
        if norm > 0:
            new_proto = new_proto / norm
        self._label_prototypes[label] = new_proto
        self._label_proto_counts[label] = count
        self._knn_memory.append((vec, label))
        limit = max(1, int(getattr(self, "_knn_memory_limit", 4096)))
        if len(self._knn_memory) > limit:
            self._knn_memory = self._knn_memory[-limit:]

    def _prototype_scores_for_embedding(
        self, emb: np.ndarray
    ) -> List[Tuple[str, float]]:
        scores: List[Tuple[str, float]] = []
        if emb is None:
            return scores
        vec = np.asarray(emb, dtype=np.float32)
        if vec.ndim != 1 or vec.size == 0:
            return scores
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        for label, proto in self._label_prototypes.items():
            if proto is None:
                continue
            if proto.shape != vec.shape:
                continue
            score = float(np.dot(vec, proto))
            scores.append((label, score))
        return scores

    def _embedding_topk_candidates(
        self, seg: dict, top_k: int
    ) -> Optional[List[Tuple[str, Optional[float]]]]:
        if not self._topk_enabled():
            return None
        if not self._label_prototypes:
            return None
        try:
            s = int(seg.get("start", 0))
            e = int(seg.get("end", s))
        except Exception:
            return None
        emb = self._segment_embedding_for_span(s, e)
        if emb is None:
            return None
        scores = self._prototype_scores_for_embedding(emb)
        if not scores:
            return None
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [(name, float(score)) for name, score in scores[: max(1, top_k)]]
        if not top:
            return None
        return top

    def _get_feature_series_for_embedding(self) -> Optional[Dict[str, Any]]:
        features_dir = self._resolve_features_dir_for_snap()
        if not features_dir:
            return None
        cache = self._ensure_boundary_snap_cache(features_dir)
        feat = cache.get("features")
        if not isinstance(feat, np.ndarray):
            feat_path = os.path.join(features_dir, "features.npy")
            try:
                feat = np.load(feat_path, mmap_mode="r")
            except Exception:
                feat = None
            cache["features"] = feat
            if isinstance(feat, np.ndarray) and feat.ndim == 2:
                h, w = feat.shape
                if h == 2048 and w != 2048:
                    cache["features_layout"] = "BDT"
                elif w == 2048 and h != 2048:
                    cache["features_layout"] = "BTD"
                else:
                    cache["features_layout"] = "BTD"
        if not isinstance(feat, np.ndarray) or feat.ndim != 2:
            return None
        layout = cache.get("features_layout", "BTD")
        seq_len = int(feat.shape[1] if layout == "BDT" else feat.shape[0])
        frame_map = self._feature_frame_map(features_dir, seq_len)
        return {
            "data": feat,
            "layout": layout,
            "frame_map": frame_map,
            "length": seq_len,
        }

    def _segment_embedding_for_span(self, start: int, end: int) -> Optional[np.ndarray]:
        try:
            s = int(start)
            e = int(end)
        except Exception:
            return None
        if e < s:
            s, e = e, s
        length = e - s + 1
        if length <= 0:
            return None
        trim_ratio = self._segment_embedding_cfg()
        delta = int(trim_ratio * length)
        s2 = s + delta
        e2 = e - delta
        if e2 < s2:
            s2, e2 = s, e
        series = self._get_feature_series_for_embedding()
        if not series:
            return None
        frame_map = series.get("frame_map") or []
        if not frame_map:
            return None
        data_len = int(series.get("length") or 0)
        if data_len <= 0:
            return None
        if len(frame_map) > data_len:
            frame_map = frame_map[:data_len]

        idx_start = bisect.bisect_left(frame_map, s2)
        idx_end = bisect.bisect_right(frame_map, e2) - 1
        if idx_start > idx_end:
            idx = max(0, min(idx_start, len(frame_map) - 1))
            idx_start = idx_end = idx
        idx_slice = slice(idx_start, idx_end + 1)

        feat = series.get("data")
        layout = series.get("layout", "BTD")
        if layout == "BDT":
            seg = feat[:, idx_slice]
            if seg.size == 0:
                return None
            g = np.mean(seg, axis=1)
        else:
            seg = feat[idx_slice]
            if seg.size == 0:
                return None
            g = np.mean(seg, axis=0)
        g = np.asarray(g, dtype=np.float32)
        norm = float(np.linalg.norm(g))
        if norm > 0:
            g = g / norm
        return g

    def _cache_segment_embedding(
        self, start: int, end: int, label: Optional[str], embedding: np.ndarray
    ) -> None:
        if embedding is None:
            return
        try:
            key = (int(start), int(end), str(label or ""))
        except Exception:
            return
        self._segment_embedding_cache[key] = embedding
        limit = max(1, int(getattr(self, "_segment_embedding_cache_limit", 2048)))
        overflow = len(self._segment_embedding_cache) - limit
        while overflow > 0 and self._segment_embedding_cache:
            try:
                oldest_key = next(iter(self._segment_embedding_cache))
                self._segment_embedding_cache.pop(oldest_key, None)
            except Exception:
                break
            overflow -= 1
        for seg in self._assisted_source_segments or []:
            try:
                if (
                    int(seg.get("start", -1)) == key[0]
                    and int(seg.get("end", -1)) == key[1]
                ):
                    if seg.get("label") == key[2]:
                        seg["embedding"] = embedding
            except Exception:
                continue

    def _get_or_compute_segment_embedding(
        self, start: int, end: int, label: Optional[str]
    ) -> Optional[np.ndarray]:
        try:
            key = (int(start), int(end), str(label or ""))
        except Exception:
            key = None
        if key and key in self._segment_embedding_cache:
            return self._segment_embedding_cache.get(key)
        emb = self._segment_embedding_for_span(start, end)
        if emb is not None:
            self._cache_segment_embedding(start, end, label, emb)
        return emb

    # ----- Assisted Review (model-guided review) -----
    def _active_assisted_point(self):
        if 0 <= self.assisted_active_idx < len(self.assisted_points):
            return self.assisted_points[self.assisted_active_idx]
        return None

    def _segments_from_store_for_interaction(
        self, store: Optional[AnnotationStore] = None
    ) -> List[dict]:
        """Convert a store into ordered segments (start, end, label) within the active view span."""
        if not self.views:
            return []
        view = self.views[self.active_view_idx]
        start = int(view.get("start", 0))
        end = int(view.get("end", start))
        target = store or self.store
        if not target:
            return []
        frames = sorted(f for f in target.frame_to_label.keys() if start <= f <= end)
        if not frames:
            return []
        segments = []
        s = frames[0]
        cur = target.label_at(s)
        prev = s
        for f in frames[1:]:
            lb = target.label_at(f)
            if lb != cur or f != prev + 1:
                segments.append({"start": s, "end": prev, "label": cur})
                s, cur = f, lb
            prev = f
        segments.append({"start": s, "end": prev, "label": cur})
        return segments

    def _label_candidates_with_source(
        self, seg: dict
    ) -> Tuple[List[Tuple[str, Optional[float]]], str]:
        """Return top-k candidates with source tag: 'embedding' or 'model'."""
        top_k = self._topk_k()
        candidates: List[Tuple[str, Optional[float]]] = []
        source = "model"

        emb_candidates = self._embedding_topk_candidates(seg, top_k)
        if emb_candidates:
            candidates = list(emb_candidates)
            source = "embedding"
        else:
            key = (seg.get("start"), seg.get("end"))
            raw = self._assisted_candidates.get(key) or []
            for item in raw:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("label")
                    score = item.get("score")
                elif isinstance(item, (list, tuple)) and item:
                    name = item[0]
                    score = item[1] if len(item) > 1 else None
                else:
                    continue
                if name:
                    candidates.append((name, score))
            if not candidates and seg.get("label"):
                candidates.append((seg.get("label"), None))

        seg_label = seg.get("label")
        if (
            seg_label
            and seg_label not in {c[0] for c in candidates}
            and len(candidates) < top_k
        ):
            candidates.append((seg_label, None))
        # fill remaining slots with existing labels (de-emphasized) to keep them accessible
        if len(candidates) < top_k:
            existing = {c[0] for c in candidates}
            for lb in self.labels:
                if is_extra_label(lb.name) or lb.name in existing:
                    continue
                candidates.append((lb.name, None))
                if len(candidates) >= top_k:
                    break
        return candidates[:top_k], source

    def _label_candidates_for_segment(
        self, seg: dict
    ) -> List[Tuple[str, Optional[float]]]:
        """Return top-k candidates for a segment; fallback to predicted label only when scores are missing."""
        candidates, _source = self._label_candidates_with_source(seg)
        return candidates

    def _is_label_uncertain(
        self, candidates: List[Tuple[str, Optional[float]]], source: str = "model"
    ) -> bool:
        if not candidates:
            return False
        if source == "embedding":
            margin_thr = self._topk_uncertainty_margin()
            if margin_thr is None:
                return True
            top_conf = candidates[0][1] if candidates else None
            second_conf = candidates[1][1] if len(candidates) > 1 else None
            if top_conf is None:
                return True
            if second_conf is None:
                return False
            try:
                margin = float(top_conf) - float(second_conf)
            except Exception:
                return True
            return not (margin > float(margin_thr))
        cfg = self._interaction_cfg["label"]
        top_conf = candidates[0][1]
        second_conf = candidates[1][1] if len(candidates) > 1 else None
        if top_conf is None and second_conf is None:
            # 无置信度时默认不标为不确定，避免过多点
            return False
        try:
            if top_conf is not None and float(top_conf) < float(cfg["min_confidence"]):
                return True
        except Exception:
            pass
        if top_conf is not None and second_conf is not None:
            try:
                if abs(float(top_conf) - float(second_conf)) <= float(
                    cfg.get("diff_eps", 0.05)
                ):
                    return True
            except Exception:
                pass
        return False

    def _next_pending_interaction(
        self,
        points: Optional[List[dict]] = None,
        start_idx: int = 0,
        forward: bool = True,
        wrap: bool = False,
        type_filter: Optional[str] = None,
    ) -> int:
        pts = points if points is not None else self.assisted_points
        if not pts:
            return -1
        total = len(pts)
        if total <= 0:
            return -1
        if start_idx < 0:
            start_idx = 0
        if start_idx >= total:
            if not wrap:
                return -1
            start_idx = 0 if forward else (total - 1)

        if forward:
            for idx in range(start_idx, total):
                if type_filter and pts[idx].get("type") != type_filter:
                    continue
                if pts[idx].get("status") == "PENDING":
                    return idx
            if wrap:
                for idx in range(0, start_idx):
                    if type_filter and pts[idx].get("type") != type_filter:
                        continue
                    if pts[idx].get("status") == "PENDING":
                        return idx
        else:
            for idx in range(start_idx, -1, -1):
                if type_filter and pts[idx].get("type") != type_filter:
                    continue
                if pts[idx].get("status") == "PENDING":
                    return idx
            if wrap:
                for idx in range(total - 1, start_idx, -1):
                    if type_filter and pts[idx].get("type") != type_filter:
                        continue
                    if pts[idx].get("status") == "PENDING":
                        return idx
        return -1

    def _interaction_point_pos(self, pt: Optional[dict]) -> int:
        if not pt:
            return 0
        if pt.get("type") == "boundary":
            try:
                return int(pt.get("frame", 0))
            except Exception:
                return 0
        try:
            return int(pt.get("start", 0))
        except Exception:
            return 0

    def _next_pending_after_frame(
        self, frame: int, prefer_type: Optional[str] = None, wrap: bool = True
    ) -> int:
        try:
            frame = int(frame)
        except Exception:
            frame = 0
        start_idx = len(self.assisted_points)
        for idx, pt in enumerate(self.assisted_points):
            if self._interaction_point_pos(pt) > frame:
                start_idx = idx
                break
        target = -1
        if prefer_type:
            target = self._next_pending_interaction(
                start_idx=start_idx, forward=True, wrap=wrap, type_filter=prefer_type
            )
        if target < 0:
            target = self._next_pending_interaction(
                start_idx=start_idx, forward=True, wrap=wrap
            )
        return target

    def _update_assisted_playback_for_active(self):
        pt = self._active_assisted_point()
        self._assisted_loop_range = None
        if not pt:
            return
        if pt.get("type") == "label":
            start = int(pt.get("start", getattr(self.player, "current_frame", 0)))
            try:
                self._sync_views_to_frame(start, preview_only=False)
                self.timeline.set_current_frame(start, follow=True)
                self._pause_all()
            except Exception:
                pass
            return
        if pt.get("type") != "boundary":
            return
        window = max(1, int(self._interaction_cfg["boundary"]["window_size"]))
        frame = int(pt.get("frame", getattr(self.player, "current_frame", 0)))
        crop_start = getattr(self.player, "crop_start", 0)
        crop_end = getattr(self.player, "crop_end", frame)
        start = max(crop_start, frame - window)
        end = min(crop_end, frame + window)
        if end < start:
            end = start
        self._assisted_loop_range = (start, end)
        try:
            self._sync_views_to_frame(start, preview_only=False)
            self._timeline_auto_follow = True
            self._play_all()
        except Exception:
            pass

    def _update_assisted_visuals(self):
        active = self._active_assisted_point()
        active_id = active.get("id") if active else None
        try:
            self.timeline.set_interaction_points(
                self.assisted_points, active_id=active_id
            )
        except Exception:
            pass
        if active and active.get("type") == "label":
            cands = active.get("candidates") or []
            try:
                self.panel.set_candidate_priority(cands)
            except Exception:
                pass
            label_name = active.get("label")
            if label_name:
                try:
                    self.timeline.set_highlight_labels([label_name])
                except Exception:
                    pass
        else:
            try:
                self.panel.clear_candidate_priority()
            except Exception:
                pass
            try:
                self.timeline.set_highlight_labels([])
            except Exception:
                pass
        resolved = sum(1 for p in self.assisted_points if p.get("status") == "RESOLVED")
        total = len(self.assisted_points)
        summary = (
            "Assisted: no points"
            if total == 0
            else f"Assisted: {resolved}/{total} resolved"
        )
        if self._assisted_review_done and total:
            summary += " • review complete"
        elif active:
            summary += f" • active {active.get('type')}"
        self._set_interaction_status(summary)

    def _set_active_assisted_idx(self, idx: int):
        prev = self.assisted_active_idx
        if (
            prev != idx
            and getattr(self._correction_buffer, "active", None) is not None
            and str(getattr(self._correction_buffer.active, "kind", "") or "").startswith(
                "assisted_"
            )
        ):
            self._discard_correction_session("assisted_point_switched")
        if 0 <= prev < len(self.assisted_points):
            if self.assisted_points[prev].get("status") == "ACTIVE":
                self.assisted_points[prev]["status"] = "PENDING"
        if 0 <= idx < len(self.assisted_points):
            # allow re-activating resolved points for manual review
            self.assisted_points[idx]["status"] = "ACTIVE"
            self.assisted_active_idx = idx
            pt = self.assisted_points[idx]
            if pt.get("type") == "boundary":
                self._begin_correction_session(
                    "assisted_boundary",
                    point_id=int(pt.get("id", idx)),
                    frame=int(pt.get("frame", 0)),
                )
            elif pt.get("type") == "label":
                self._begin_correction_session(
                    "assisted_label",
                    point_id=int(pt.get("id", idx)),
                    start=int(pt.get("start", 0)),
                    end=int(pt.get("end", pt.get("start", 0))),
                    label=str(pt.get("label", "") or ""),
                )
        else:
            self.assisted_active_idx = -1
        self._update_assisted_playback_for_active()
        self._update_assisted_visuals()

    def _activate_assisted_at_frame(self, frame: int) -> bool:
        """Activate the interaction point that covers/nearest to the given frame."""
        if self.interaction_mode != "assisted" or not self.assisted_points:
            return False
        # prefer label points covering frame
        for idx, pt in enumerate(self.assisted_points):
            if pt.get("type") == "label" and int(pt.get("start", 0)) <= frame <= int(
                pt.get("end", 0)
            ):
                self._set_active_assisted_idx(idx)
                return True
        # then nearest boundary
        best = None
        window = max(1, int(self._interaction_cfg["boundary"]["window_size"]))
        for idx, pt in enumerate(self.assisted_points):
            if pt.get("type") != "boundary":
                continue
            diff = abs(int(pt.get("frame", frame)) - frame)
            if diff <= window and (best is None or diff < best[0]):
                best = (diff, idx)
        if best is not None:
            self._set_active_assisted_idx(best[1])
            return True
        return False

    def _shift_assisted_idx(self, delta: int):
        """Move active interaction point without forcing prior confirmation."""
        if not self.assisted_points:
            return
        forward = delta >= 0
        total = len(self.assisted_points)
        active = self._active_assisted_point()
        prefer_type = active.get("type") if active else None
        if self.assisted_active_idx < 0:
            start_idx = 0 if forward else (total - 1)
        else:
            start_idx = self.assisted_active_idx + (1 if forward else -1)
            if start_idx < 0 or start_idx >= total:
                start_idx = 0 if forward else (total - 1)

        has_pending = any(pt.get("status") == "PENDING" for pt in self.assisted_points)
        if has_pending:
            target = -1
            if prefer_type:
                target = self._next_pending_interaction(
                    start_idx=start_idx,
                    forward=forward,
                    wrap=True,
                    type_filter=prefer_type,
                )
            if target < 0:
                target = self._next_pending_interaction(
                    start_idx=start_idx, forward=forward, wrap=True
                )
        else:
            if self.assisted_active_idx < 0:
                target = 0 if forward else (total - 1)
            else:
                target = (self.assisted_active_idx + (1 if forward else -1)) % total
        if target >= 0:
            self._set_active_assisted_idx(target)

    def _skip_active_assisted_point(self):
        """Mark current point as resolved without edits and jump to next."""
        if self.interaction_mode != "assisted":
            return
        if self.assisted_active_idx < 0:
            return
        pt = self._active_assisted_point()
        accept_rec = self._build_accept_record_for_point(pt)
        if accept_rec is not None:
            self._store_explicit_confirm_record(accept_rec)
            self._note_correction_step()
            self._commit_correction_session(
                point_type=str(pt.get("type", "") or "").strip().lower(),
                mode="accept",
            )
        else:
            self._discard_correction_session("assisted_skip")
        self._resolve_assisted_point(self.assisted_active_idx)

    def _merge_active_boundary(self):
        """Merge boundary: apply左侧标签覆盖右段，视为已解决。"""
        pt = self._active_assisted_point()
        if (
            self.interaction_mode != "assisted"
            or not pt
            or pt.get("type") != "boundary"
        ):
            return
        ref_frame = int(pt.get("frame", 0))
        left = pt.get("left", {})
        right = pt.get("right", {})
        left_label = left.get("label")
        if not left_label:
            return
        start = int(right.get("start", pt.get("frame", 0)))
        end = int(right.get("end", start))
        try:
            self.store.begin_txn()
        except Exception:
            pass
        for f in range(start, end + 1):
            cur = self.store.label_at(f)
            if cur and cur != left_label:
                self.store.remove_at(f)
            self.store.add(left_label, f)
        try:
            self.store.end_txn()
        except Exception:
            pass
        self._note_correction_step()
        self._dirty = True
        self._commit_correction_session(
            point_type="boundary",
            boundary_frame=int(ref_frame),
            mode="merge",
        )
        # rebuild interaction points with hint to avoid losing place
        self._build_assisted_points_from_store(preserve_status=True, active_hint=None)
        next_idx = self._next_pending_after_frame(ref_frame, prefer_type="boundary")
        self._set_active_assisted_idx(next_idx)

    def _build_assisted_points_from_store(
        self,
        preserve_status: bool = True,
        active_hint: Optional[Tuple[str, int]] = None,
    ) -> bool:
        segments = self._segments_from_store_for_interaction()
        self._assisted_source_segments = segments
        if not segments:
            self.assisted_points = []
            self.assisted_active_idx = -1
            return False
        old_boundaries = (
            [p for p in self.assisted_points if p.get("type") == "boundary"]
            if preserve_status
            else []
        )
        old_labels = (
            [p for p in self.assisted_points if p.get("type") == "label"]
            if preserve_status
            else []
        )
        boundary_points = []
        label_points = []
        active_idx = -1
        # boundary points
        for i in range(1, len(segments)):
            left = segments[i - 1]
            right = segments[i]
            boundary_frame = max(
                int(right.get("start", 0)), int(left.get("end", 0)) + 1
            )
            status = "PENDING"
            raw_score = self._boundary_review_score_at_frame(boundary_frame)
            query_score, query_terms = self._boundary_query_score(
                frame=boundary_frame,
                left_label=str(left.get("label", "") or ""),
                right_label=str(right.get("label", "") or ""),
                raw_score=raw_score,
            )
            if query_score < self._assisted_query_threshold("boundary"):
                continue
            matched = None
            for j, ob in enumerate(list(old_boundaries)):
                if ob.get("left", {}).get("label") == left.get("label") and ob.get(
                    "right", {}
                ).get("label") == right.get("label"):
                    matched = ob
                    old_boundaries.pop(j)
                    break
            if matched:
                status = matched.get("status", "PENDING")
            pt = {
                "type": "boundary",
                "frame": boundary_frame,
                "left": dict(left),
                "right": dict(right),
                "status": status,
                "score": raw_score,
                "query_score": float(query_score),
                "query_terms": dict(query_terms),
                "id": 0,
            }
            boundary_points.append(pt)
        boundary_points = self._throttle_boundary_points(boundary_points)
        # label-uncertainty points
        for seg in segments:
            candidates, source = self._label_candidates_with_source(seg)
            query_score, query_terms = self._label_query_score(
                seg,
                candidates,
                source=source,
            )
            if query_score < self._assisted_query_threshold("label"):
                continue
            status = "PENDING"
            matched = None
            for j, ol in enumerate(list(old_labels)):
                if ol.get("label") == seg.get("label"):
                    o_s = int(ol.get("start", 0))
                    o_e = int(ol.get("end", o_s))
                    if not (o_e < seg.get("start", 0) or o_s > seg.get("end", 0)):
                        matched = ol
                        old_labels.pop(j)
                        break
            if matched:
                status = matched.get("status", "PENDING")
            pt = {
                "type": "label",
                "start": int(seg.get("start", 0)),
                "end": int(seg.get("end", seg.get("start", 0))),
                "label": seg.get("label"),
                "candidates": candidates,
                "status": status,
                "score": candidates[0][1] if candidates else None,
                "query_score": float(query_score),
                "query_terms": dict(query_terms),
                "candidate_source": str(source),
                "id": 0,
            }
            label_points.append(pt)

        points = boundary_points + label_points

        def _pt_key(pt):
            if pt.get("type") == "boundary":
                try:
                    pos = int(pt.get("frame", 0))
                except Exception:
                    pos = 0
                rank = 0
            else:
                try:
                    pos = int(pt.get("start", 0))
                except Exception:
                    pos = 0
                rank = 1
            if self._assisted_sort_mode() == "query_score":
                try:
                    query_bucket = int(max(0, min(10, round(float(pt.get("query_score", 0.0) or 0.0) * 10.0))))
                except Exception:
                    query_bucket = 0
                return (-query_bucket, pos, rank)
            return (pos, rank)

        points.sort(key=_pt_key)
        active_idx = -1
        for idx, pt in enumerate(points):
            pt["id"] = idx
            if pt.get("status") == "ACTIVE":
                active_idx = idx

        if active_hint and active_idx < 0:
            kind, frame = active_hint
            if kind == "boundary":
                for i, pt in enumerate(points):
                    if pt.get("type") == "boundary" and abs(
                        int(pt.get("frame", 0)) - int(frame)
                    ) <= max(1, int(self._interaction_cfg["boundary"]["frame_step"])):
                        active_idx = i
                        break
            elif kind == "label":
                for i, pt in enumerate(points):
                    if pt.get("type") == "label" and pt.get("start") <= frame <= pt.get(
                        "end"
                    ):
                        active_idx = i
                        break

        self.assisted_points = points
        self.assisted_active_idx = -1
        if active_idx < 0:
            active_idx = self._next_pending_interaction(
                points=points, type_filter="boundary"
            )
            if active_idx < 0:
                active_idx = self._next_pending_interaction(points=points)
        self._set_active_assisted_idx(active_idx)
        self._assisted_review_done = bool(points) and all(
            p.get("status") == "RESOLVED" for p in points
        )
        return True

    def _resolve_assisted_point(self, idx: int):
        if not (0 <= idx < len(self.assisted_points)):
            return
        pt = self.assisted_points[idx]
        if pt.get("type") == "label":
            try:
                start = int(pt.get("start", 0))
                end = int(pt.get("end", start))
            except Exception:
                start = end = 0
            label = pt.get("label")
            emb = self._get_or_compute_segment_embedding(start, end, label)
            if emb is not None:
                pt["embedding"] = emb
                self._update_label_prototype(label, emb)
        self.assisted_points[idx]["status"] = "RESOLVED"
        if idx == self.assisted_active_idx:
            self.assisted_active_idx = -1
            self._assisted_loop_range = None
        if all(p.get("status") == "RESOLVED" for p in self.assisted_points):
            self._assisted_review_done = True
            self._set_status(
                "Assisted review complete. All interaction points resolved."
            )
        prefer_type = pt.get("type")
        next_idx = -1
        if prefer_type:
            next_idx = self._next_pending_interaction(
                start_idx=idx + 1, forward=True, wrap=True, type_filter=prefer_type
            )
        if next_idx < 0:
            next_idx = self._next_pending_interaction(
                start_idx=idx + 1, forward=True, wrap=True
            )
        self._set_active_assisted_idx(next_idx)

    def _adjust_active_boundary(self, direction: int):
        pt = self._active_assisted_point()
        if not pt or pt.get("type") != "boundary":
            return
        step = max(1, int(self._interaction_cfg["boundary"]["frame_step"]))
        delta = step if direction > 0 else -step
        old_frame = int(pt.get("frame", 0))
        self._move_active_boundary_to(old_frame + delta)

    def _assist_nudge_boundary(self, direction: int):
        if self.interaction_mode != "assisted":
            return
        self._adjust_active_boundary(direction)

    def _confirm_active_boundary(self):
        pt = self._active_assisted_point()
        if not pt or pt.get("type") != "boundary":
            return
        try:
            frame = int(pt.get("frame", 0))
        except Exception:
            frame = 0
        left = pt.get("left", {})
        right = pt.get("right", {})
        min_frame = int(left.get("start", 0)) + 1
        max_frame = int(right.get("end", right.get("start", frame)))
        snapped = self._snap_boundary_frame(frame, lo=min_frame, hi=max_frame)
        if snapped != frame:
            self._move_active_boundary_to(snapped)
        accept_rec = self._build_accept_record_for_point(pt, boundary_frame=int(snapped))
        if accept_rec is not None:
            self._store_explicit_confirm_record(accept_rec)
        self._commit_correction_session(
            point_type="boundary",
            boundary_frame=int(snapped),
        )
        idx = self.assisted_active_idx
        self._resolve_assisted_point(idx)
        self._assisted_loop_range = None
        try:
            self._pause_all()
        except Exception:
            pass

    def _apply_label_range(
        self, store: AnnotationStore, start: int, end: int, label_name: str
    ) -> None:
        """Apply label to [start, end] in a single store."""
        if not store or not label_name:
            return
        try:
            store.begin_txn()
        except Exception:
            pass
        for f in range(start, end + 1):
            cur = store.label_at(f)
            if cur and cur != label_name:
                store.remove_at(f)
            store.add(label_name, f)
        try:
            store.end_txn()
        except Exception:
            pass

    def _target_entity_stores(
        self, label_name: str
    ) -> List[Tuple[str, AnnotationStore]]:
        """Return [(entity_name, store)] targets for Fine mode labeling."""
        targets = []
        if self.mode != "Fine" or is_extra_label(label_name):
            return targets
        # Prefer the active timeline row (entity) when a combined row is selected.
        active_entity = None
        try:
            row = getattr(self.timeline, "_active_combined_row", None)
            title = getattr(row, "title", None) if row is not None else None
            if title and title in self.entity_stores:
                active_entity = title
        except Exception:
            active_entity = None
        if not active_entity:
            active_entity = self._active_entity_name
        if active_entity:
            st = self.entity_stores.setdefault(active_entity, AnnotationStore())
            return [(active_entity, st)]
        selected = set(self.label_entity_map.get(label_name, set()) or [])
        if not selected:
            selected = set(self.visible_entities or [])
        if not selected and active_entity:
            selected = {active_entity}
        if not selected and len(self.visible_entities) == 1:
            selected = {self.visible_entities[0]}
        if not selected and active_entity:
            selected = {active_entity}
        if not selected and len(self.entity_stores) == 1:
            selected = set(self.entity_stores.keys())
        for ename in sorted(selected):
            st = self.entity_stores.setdefault(ename, AnnotationStore())
            targets.append((ename, st))
        return targets

    def _apply_assisted_label_choice(self, label_name: str) -> bool:
        pt = self._active_assisted_point()
        if not pt or pt.get("type") != "label":
            return False
        if not label_name:
            return False
        try:
            pt["label"] = label_name
        except Exception:
            pass
        start = int(pt.get("start", 0))
        end = int(pt.get("end", start))
        targets = self._target_entity_stores(label_name)
        if self.mode == "Fine" and not targets:
            QMessageBox.information(
                self,
                "Choose entity",
                "Select an entity row or enable one entity for this label before editing.",
            )
            return False
        if targets:
            for _ename, st in targets:
                self._apply_label_range(st, start, end, label_name)
        else:
            self._apply_label_range(self.store, start, end, label_name)
        if str(pt.get("label", "") or "").strip() == str(label_name or "").strip():
            accept_rec = self._build_accept_record_for_point(pt)
            if accept_rec is not None:
                self._store_explicit_confirm_record(accept_rec)
        self._dirty = True
        self._note_correction_step()
        self._commit_correction_session(
            point_type="label",
            start=int(start),
            end=int(end),
            label=str(label_name),
        )
        self._resolve_assisted_point(self.assisted_active_idx)
        self._update_assisted_visuals()
        return True

    def _apply_label_to_segment(self, start: int, end: int, label_name: str) -> bool:
        if not label_name:
            return False
        try:
            s = int(start)
            e = int(end)
        except Exception:
            return False
        if e < s:
            s, e = e, s
        self._begin_correction_session(
            "direct_label",
            start=int(s),
            end=int(e),
            label=str(label_name),
        )
        targets = self._target_entity_stores(label_name)
        if self.mode == "Fine" and not targets:
            QMessageBox.information(
                self,
                "Choose entity",
                "Select an entity row or enable one entity for this label before editing.",
            )
            self._discard_correction_session("missing_entity")
            return False
        if targets:
            for _ename, st in targets:
                self._apply_label_range(st, s, e, label_name)
        else:
            self._apply_label_range(self.store, s, e, label_name)
        self._dirty = True
        self._on_store_changed()
        self._note_correction_step()
        self._commit_correction_session(
            point_type="label",
            start=int(s),
            end=int(e),
            label=str(label_name),
        )
        return True

    def _prompt_uncertainty_margin(self):
        cfg = self._topk_cfg()
        cur = cfg.get("uncertainty_margin", None)
        cur_txt = "" if cur is None else str(cur)
        text, ok = QInputDialog.getText(
            self,
            "Top-K uncertainty margin",
            "Set margin (number). Use 'none' to disable:",
            text=cur_txt,
        )
        if not ok:
            return
        raw = (text or "").strip()
        if not raw:
            return
        if raw.lower() in ("none", "off", "disable", "disabled"):
            self._algo_cfg.setdefault("topk", {})["uncertainty_margin"] = None
            self._set_status("Top-K margin disabled: always show label points.")
            self._log("topk_margin_set", value=None)
        else:
            try:
                val = float(raw)
            except Exception:
                QMessageBox.information(
                    self, "Top-K margin", "Invalid number. Example: 0.25"
                )
                return
            self._algo_cfg.setdefault("topk", {})["uncertainty_margin"] = val
            self._set_status(f"Top-K margin set to {val:.3f}")
            self._log("topk_margin_set", value=val)

        if self.interaction_mode == "assisted":
            active = self._active_assisted_point()
            hint = None
            if active:
                if active.get("type") == "boundary":
                    try:
                        hint = ("boundary", int(active.get("frame", 0)))
                    except Exception:
                        hint = None
                elif active.get("type") == "label":
                    try:
                        hint = ("label", int(active.get("start", 0)))
                    except Exception:
                        hint = None
            try:
                self._build_assisted_points_from_store(
                    preserve_status=True, active_hint=hint
                )
                self._update_assisted_visuals()
            except Exception:
                pass

    def _open_settings_dialog(self):
        self._psr_reload_model_registry()
        if getattr(self, "psr_embedded", None) is not None and callable(
            getattr(self.psr_embedded, "set_available_models", None)
        ):
            try:
                self.psr_embedded.set_available_models(self._psr_model_specs, emit=False)
            except Exception:
                pass
        self._ensure_algo_cfg_defaults()
        timeline_cfg = self._timeline_snap_cfg()
        boundary_cfg = dict(self._algo_cfg.get("boundary_snap", {}))
        seg_cfg = dict(self._algo_cfg.get("segment_embedding", {}))
        topk_cfg = dict(self._algo_cfg.get("topk", {}))
        assist_cfg = dict(self._algo_cfg.get("assisted", {}))
        psr_cfg = dict(self._algo_cfg.get("psr", {}))

        dlg = QDialog(self)
        dlg.setWindowTitle("Settings")
        dlg.setMinimumSize(520, 420)
        dlg.resize(640, 560)
        dlg.setSizeGripEnabled(True)
        outer = QVBoxLayout(dlg)
        scroll = QScrollArea(dlg)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_content = QWidget(scroll)
        root = QVBoxLayout(scroll_content)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        scroll.setWidget(scroll_content)
        outer.addWidget(scroll, 1)

        group_timeline = QGroupBox("Timeline Editing & Snapping", dlg)
        form_timeline = QFormLayout(group_timeline)
        sp_playhead = QSpinBox(group_timeline)
        sp_playhead.setRange(0, 120)
        sp_playhead.setValue(int(timeline_cfg.get("playhead_radius", 6)))
        sp_empty = QSpinBox(group_timeline)
        sp_empty.setRange(0, 120)
        sp_empty.setValue(int(timeline_cfg.get("empty_space_radius", 10)))
        sp_edge = QSpinBox(group_timeline)
        sp_edge.setRange(0, 60)
        sp_edge.setValue(int(timeline_cfg.get("edge_search_radius", 5)))
        sp_segment = QSpinBox(group_timeline)
        sp_segment.setRange(0, 120)
        sp_segment.setValue(int(timeline_cfg.get("segment_soft_radius", 10)))
        sp_phase = QSpinBox(group_timeline)
        sp_phase.setRange(0, 120)
        sp_phase.setValue(int(timeline_cfg.get("phase_soft_radius", 8)))
        hover_cfg = self._timeline_hover_preview_cfg()
        cb_hover_multi = QCheckBox(group_timeline)
        cb_hover_multi.setChecked(bool(hover_cfg.get("enabled_multi", True)))
        combo_hover_align = QComboBox(group_timeline)
        combo_hover_align.addItem("Absolute frame", "absolute")
        combo_hover_align.addItem("Relative to active view", "offset")
        idx_hover_align = combo_hover_align.findData(
            str(hover_cfg.get("align", "absolute") or "absolute").lower()
        )
        combo_hover_align.setCurrentIndex(
            idx_hover_align if idx_hover_align >= 0 else 0
        )
        combo_hover_align.setEnabled(cb_hover_multi.isChecked())
        cb_hover_multi.toggled.connect(combo_hover_align.setEnabled)
        form_timeline.addRow("Playhead snap radius (frames)", sp_playhead)
        form_timeline.addRow("Empty-space snap radius (frames)", sp_empty)
        form_timeline.addRow("Edge search radius (frames)", sp_edge)
        form_timeline.addRow("Segment soft snap radius (frames)", sp_segment)
        form_timeline.addRow("Phase soft snap radius (frames)", sp_phase)
        form_timeline.addRow("Hover preview selected views", cb_hover_multi)
        form_timeline.addRow("Hover preview alignment", combo_hover_align)
        root.addWidget(group_timeline)

        group_assisted = QGroupBox("Boundary Assistance", dlg)
        form_assisted = QFormLayout(group_assisted)
        cb_snap = QCheckBox(group_assisted)
        cb_snap.setChecked(bool(boundary_cfg.get("enabled", True)))
        sp_snap = QSpinBox(group_assisted)
        sp_snap.setRange(1, 300)
        sp_snap.setValue(self._cfg_int(boundary_cfg.get("window_size", 15), 15, 1, 300))
        sp_gap = QSpinBox(group_assisted)
        sp_gap.setRange(0, 300)
        sp_gap.setValue(
            self._cfg_int(assist_cfg.get("boundary_min_gap", 15), 15, 0, 300)
        )
        form_assisted.addRow("Boundary snap enabled", cb_snap)
        form_assisted.addRow("Boundary snap window (frames)", sp_snap)
        form_assisted.addRow("Boundary minimum gap (frames)", sp_gap)
        root.addWidget(group_assisted)

        group_psr = QGroupBox("Assembly State Model", dlg)
        form_psr = QFormLayout(group_psr)
        combo_psr_model = QComboBox(group_psr)
        psr_model = self._populate_psr_model_combo(
            combo_psr_model,
            psr_cfg.get("model_type", self._psr_model_type),
        )
        combo_psr_init = QComboBox(group_psr)
        combo_psr_init.addItem(
            "Auto (default Assemble baseline; use invert for Disassemble)",
            "auto",
        )
        combo_psr_init.addItem("Force all installed at start", "installed")
        combo_psr_init.addItem("Force all not installed at start", "not_installed")
        psr_policy = str(psr_cfg.get("initial_state_policy", "auto") or "auto").lower()
        idx_psr_policy = combo_psr_init.findData(psr_policy)
        combo_psr_init.setCurrentIndex(idx_psr_policy if idx_psr_policy >= 0 else 0)
        cb_psr_no_gap = QCheckBox(group_psr)
        cb_psr_no_gap.setChecked(bool(psr_cfg.get("no_gap_timeline", True)))
        cb_psr_auto_carry = QCheckBox(group_psr)
        cb_psr_auto_carry.setChecked(
            bool(psr_cfg.get("auto_carry_next_on_edit", True))
        )
        form_psr.addRow("State model", combo_psr_model)
        form_psr.addRow("Initial state policy", combo_psr_init)
        form_psr.addRow("Disallow empty state spans", cb_psr_no_gap)
        form_psr.addRow("Auto-carry next segment on edit", cb_psr_auto_carry)
        root.addWidget(group_psr)

        group_suggest = QGroupBox("Prediction Suggestions", dlg)
        form_suggest = QFormLayout(group_suggest)
        cb_topk = QCheckBox(group_suggest)
        cb_topk.setChecked(bool(topk_cfg.get("enabled", True)))
        sp_topk = QSpinBox(group_suggest)
        sp_topk.setRange(1, 50)
        sp_topk.setValue(self._cfg_int(topk_cfg.get("k", 5), 5, 1, 50))
        cb_margin = QCheckBox(group_suggest)
        margin_val = topk_cfg.get("uncertainty_margin", 0.25)
        margin_enabled = margin_val is not None
        cb_margin.setChecked(bool(margin_enabled))
        sp_margin = QDoubleSpinBox(group_suggest)
        sp_margin.setDecimals(3)
        sp_margin.setSingleStep(0.01)
        sp_margin.setRange(0.0, 1.0)
        sp_margin.setValue(float(margin_val) if margin_val is not None else 0.25)
        sp_margin.setEnabled(margin_enabled)
        cb_margin.toggled.connect(sp_margin.setEnabled)
        form_suggest.addRow("Top-K enabled", cb_topk)
        form_suggest.addRow("Top-K size", sp_topk)
        form_suggest.addRow("Uncertainty gate enabled", cb_margin)
        form_suggest.addRow("Uncertainty margin", sp_margin)
        root.addWidget(group_suggest)

        group_segment = QGroupBox("Segment Embedding", dlg)
        form_segment = QFormLayout(group_segment)
        sp_trim = QDoubleSpinBox(group_segment)
        sp_trim.setDecimals(3)
        sp_trim.setSingleStep(0.01)
        sp_trim.setRange(0.0, 0.49)
        sp_trim.setValue(
            self._cfg_float(seg_cfg.get("trim_ratio", 0.1), 0.1, lo=0.0, hi=0.49)
        )
        form_segment.addRow("Segment trim ratio", sp_trim)
        root.addWidget(group_segment)

        group_logging = QGroupBox("Logging", dlg)
        form_logging = QFormLayout(group_logging)
        cb_oplog = QCheckBox(group_logging)
        cb_oplog.setChecked(bool(getattr(self.op_logger, "enabled", False)))
        cb_validation_summary = QCheckBox(group_logging)
        cb_validation_summary.setChecked(bool(self._validation_summary_enabled))
        cb_validation_comment_prompt = QCheckBox(group_logging)
        cb_validation_comment_prompt.setChecked(
            bool(self._validation_comment_prompt_enabled)
        )
        form_logging.addRow("Enable operation logs (.ops.log.csv)", cb_oplog)
        form_logging.addRow(
            "Enable validation summary logs (.validation.*)", cb_validation_summary
        )
        form_logging.addRow(
            "Prompt validation comment per edit", cb_validation_comment_prompt
        )
        root.addWidget(group_logging)

        # Keyboard controls (game-style rebinding per action)
        group_keys = QGroupBox("Keyboard Controls", dlg)
        key_root = QVBoxLayout(group_keys)
        lbl_keys = QLabel(
            "Click a key box and press a new shortcut. Changes are saved automatically.",
            group_keys,
        )
        lbl_keys.setStyleSheet("color: #667085;")
        key_root.addWidget(lbl_keys)
        key_tabs = QTabWidget(group_keys)
        key_editors: Dict[str, QKeySequenceEdit] = {}
        key_meta: Dict[str, Dict[str, str]] = {}
        section_defs = shortcut_definitions_by_section()
        for section, items in section_defs.items():
            page = QWidget(key_tabs)
            form_keys = QFormLayout(page)
            for item in items:
                sid = str(item.get("id") or "")
                key_meta[sid] = {
                    "label": str(item.get("label") or sid),
                }
                default_key = str(item.get("default") or "")
                current_key = str(self._shortcut_bindings.get(sid, default_key) or "")
                editor = QKeySequenceEdit(page)
                editor.setKeySequence(QKeySequence(current_key))
                row = QWidget(page)
                row_l = QHBoxLayout(row)
                row_l.setContentsMargins(0, 0, 0, 0)
                row_l.setSpacing(6)
                row_l.addWidget(editor, 1)
                btn_clear = QPushButton("Clear", row)
                btn_clear.setFixedWidth(64)
                btn_clear.clicked.connect(
                    lambda _=False, e=editor: e.setKeySequence(QKeySequence())
                )
                row_l.addWidget(btn_clear, 0)
                form_keys.addRow(str(item.get("label") or sid), row)
                key_editors[sid] = editor
            key_tabs.addTab(page, section)
        key_root.addWidget(key_tabs)
        lbl_key_conflicts = QLabel(group_keys)
        lbl_key_conflicts.setWordWrap(True)
        lbl_key_conflicts.setStyleSheet("color: #b42318;")
        key_root.addWidget(lbl_key_conflicts)
        root.addWidget(group_keys, 1)
        root.addStretch(1)

        def _collect_shortcut_bindings_from_editors() -> Dict[str, str]:
            bindings = dict(self._shortcut_defaults)
            for sid, editor in key_editors.items():
                try:
                    seq = (
                        editor.keySequence().toString(QKeySequence.PortableText).strip()
                    )
                except Exception:
                    seq = ""
                bindings[sid] = seq
            return bindings

        def _refresh_shortcut_conflict_ui() -> List[str]:
            bindings = _collect_shortcut_bindings_from_editors()
            for editor in key_editors.values():
                editor.setStyleSheet("")

            conflicts = detect_scope_conflicts(bindings)
            conflict_ids = set()
            lines = []
            for scope, items in conflicts.items():
                for key, ids in items:
                    conflict_ids.update(ids)
                    labels = [key_meta.get(sid, {}).get("label", sid) for sid in ids]
                    lines.append(f"[{scope}] {key}: {', '.join(labels)}")
            for sid in conflict_ids:
                editor = key_editors.get(sid)
                if editor:
                    editor.setStyleSheet(
                        "QKeySequenceEdit { border: 1px solid #d92d20; "
                        "background: #fef3f2; }"
                    )
            if lines:
                preview = lines[:8]
                extra = len(lines) - len(preview)
                if extra > 0:
                    preview.append(f"... +{extra} more")
                lbl_key_conflicts.setText(
                    "Conflicts in current scope:\n" + "\n".join(preview)
                )
            else:
                lbl_key_conflicts.setText("No shortcut conflicts.")
                lbl_key_conflicts.setStyleSheet("color: #027a48;")
            if lines:
                lbl_key_conflicts.setStyleSheet("color: #b42318;")
            return lines

        for editor in key_editors.values():
            editor.keySequenceChanged.connect(
                lambda _seq, _refresh=_refresh_shortcut_conflict_ui: _refresh()
            )
        _refresh_shortcut_conflict_ui()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg
        )
        btn_reset = buttons.addButton("Reset to defaults", QDialogButtonBox.ResetRole)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        def _reset_defaults():
            default_timeline = {
                "playhead_radius": CURRENT_FRAME_SNAP_RADIUS_FRAMES,
                "empty_space_radius": SNAP_RADIUS_FRAMES,
                "edge_search_radius": EDGE_SNAP_FRAMES,
                "segment_soft_radius": SNAP_RADIUS_FRAMES,
                "phase_soft_radius": 8,
                "hover_preview_multi": True,
                "hover_preview_align": "absolute",
            }
            if isinstance(ALGO_CONFIG, dict):
                default_timeline.update(dict((ALGO_CONFIG.get("timeline_snap") or {})))
            sp_playhead.setValue(
                self._cfg_int(
                    default_timeline.get("playhead_radius"),
                    CURRENT_FRAME_SNAP_RADIUS_FRAMES,
                    0,
                    120,
                )
            )
            sp_empty.setValue(
                self._cfg_int(
                    default_timeline.get("empty_space_radius"),
                    SNAP_RADIUS_FRAMES,
                    0,
                    120,
                )
            )
            sp_edge.setValue(
                self._cfg_int(
                    default_timeline.get("edge_search_radius"),
                    EDGE_SNAP_FRAMES,
                    0,
                    60,
                )
            )
            sp_segment.setValue(
                self._cfg_int(
                    default_timeline.get("segment_soft_radius"),
                    SNAP_RADIUS_FRAMES,
                    0,
                    120,
                )
            )
            sp_phase.setValue(
                self._cfg_int(default_timeline.get("phase_soft_radius"), 8, 0, 120)
            )
            cb_hover_multi.setChecked(
                bool(default_timeline.get("hover_preview_multi", True))
            )
            idx_align = combo_hover_align.findData(
                str(default_timeline.get("hover_preview_align", "absolute")).lower()
            )
            combo_hover_align.setCurrentIndex(idx_align if idx_align >= 0 else 0)
            cb_snap.setChecked(True)
            sp_snap.setValue(15)
            sp_gap.setValue(15)
            idx_psr_model_default = combo_psr_model.findData(
                self._psr_default_model_type()
            )
            combo_psr_model.setCurrentIndex(
                idx_psr_model_default if idx_psr_model_default >= 0 else 0
            )
            idx_psr_auto = combo_psr_init.findData("auto")
            combo_psr_init.setCurrentIndex(idx_psr_auto if idx_psr_auto >= 0 else 0)
            cb_psr_no_gap.setChecked(True)
            cb_psr_auto_carry.setChecked(True)
            cb_topk.setChecked(True)
            sp_topk.setValue(5)
            cb_margin.setChecked(True)
            sp_margin.setValue(0.25)
            sp_trim.setValue(0.1)
            cb_oplog.setChecked(False)
            cb_validation_summary.setChecked(True)
            cb_validation_comment_prompt.setChecked(True)
            for sid, editor in key_editors.items():
                default_key = str(self._shortcut_defaults.get(sid, "") or "")
                editor.setKeySequence(QKeySequence(default_key))
            _refresh_shortcut_conflict_ui()

        btn_reset.clicked.connect(_reset_defaults)
        outer.addWidget(buttons)

        if dlg.exec_() != QDialog.Accepted:
            return

        new_shortcuts = _collect_shortcut_bindings_from_editors()
        conflicts = conflict_messages(new_shortcuts)
        if conflicts:
            preview = "\n".join(conflicts[:10])
            if len(conflicts) > 10:
                preview += f"\n... +{len(conflicts) - 10} more"
            ret = QMessageBox.question(
                self,
                "Shortcut conflicts",
                "Some shortcuts overlap in the same control scope:\n\n"
                + preview
                + "\n\nSave anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return
        ok_shortcut_save, shortcut_path_or_err = save_shortcut_bindings(new_shortcuts)
        if ok_shortcut_save:
            self.apply_shortcut_settings(new_shortcuts)
            if callable(self._on_shortcuts_updated):
                try:
                    self._on_shortcuts_updated(new_shortcuts)
                except Exception:
                    pass
        else:
            QMessageBox.warning(
                self,
                "Shortcut save failed",
                f"Failed to save shortcut settings:\n{shortcut_path_or_err}",
            )

        timeline_section = self._algo_cfg.setdefault("timeline_snap", {})
        timeline_section["playhead_radius"] = self._cfg_int(
            sp_playhead.value(), 6, 0, 120
        )
        timeline_section["empty_space_radius"] = self._cfg_int(
            sp_empty.value(), 10, 0, 120
        )
        timeline_section["edge_search_radius"] = self._cfg_int(
            sp_edge.value(), 5, 0, 60
        )
        timeline_section["segment_soft_radius"] = self._cfg_int(
            sp_segment.value(), 10, 0, 120
        )
        timeline_section["phase_soft_radius"] = self._cfg_int(
            sp_phase.value(), 8, 0, 120
        )
        timeline_section["hover_preview_multi"] = bool(cb_hover_multi.isChecked())
        hover_align = str(combo_hover_align.currentData() or "absolute").lower()
        if hover_align not in ("absolute", "offset"):
            hover_align = "absolute"
        timeline_section["hover_preview_align"] = hover_align

        self._algo_cfg.setdefault("boundary_snap", {})["enabled"] = cb_snap.isChecked()
        self._algo_cfg["boundary_snap"]["window_size"] = self._cfg_int(
            sp_snap.value(), 15, 1, 300
        )
        self._algo_cfg.setdefault("segment_embedding", {})["trim_ratio"] = (
            self._cfg_float(sp_trim.value(), 0.1, lo=0.0, hi=0.49)
        )
        self._algo_cfg.setdefault("topk", {})["enabled"] = cb_topk.isChecked()
        self._algo_cfg["topk"]["k"] = self._cfg_int(sp_topk.value(), 5, 1, 50)
        self._algo_cfg["topk"]["uncertainty_margin"] = (
            self._cfg_float(sp_margin.value(), 0.25, lo=0.0, hi=1.0)
            if cb_margin.isChecked()
            else None
        )
        self._algo_cfg.setdefault("assisted", {})["boundary_min_gap"] = self._cfg_int(
            sp_gap.value(), 15, 0, 300
        )
        psr_section = self._algo_cfg.setdefault("psr", {})
        psr_policy = str(combo_psr_init.currentData() or "auto").strip().lower()
        if psr_policy not in {"auto", "installed", "not_installed"}:
            psr_policy = "auto"
        psr_model_type = self._psr_normalize_model_type(combo_psr_model.currentData())
        psr_section["model_type"] = psr_model_type
        psr_section["initial_state_policy"] = psr_policy
        psr_section["no_gap_timeline"] = bool(cb_psr_no_gap.isChecked())
        psr_section["auto_carry_next_on_edit"] = bool(cb_psr_auto_carry.isChecked())
        self._psr_model_type = psr_model_type
        if getattr(self, "psr_embedded", None) is not None:
            try:
                if callable(getattr(self.psr_embedded, "set_available_models", None)):
                    self.psr_embedded.set_available_models(
                        self._psr_model_specs, emit=False
                    )
            except Exception:
                pass
            try:
                if callable(getattr(self.psr_embedded, "set_model_type", None)):
                    self.psr_embedded.set_model_type(psr_model_type, emit=False)
            except Exception:
                pass

        oplog_enabled = bool(cb_oplog.isChecked())
        validation_summary_enabled = bool(cb_validation_summary.isChecked())
        self._validation_comment_prompt_enabled = bool(
            cb_validation_comment_prompt.isChecked()
        )
        self.set_logging_policy(oplog_enabled, validation_summary_enabled)
        if callable(self._on_logging_policy_updated):
            try:
                self._on_logging_policy_updated(
                    oplog_enabled, validation_summary_enabled
                )
            except Exception:
                pass
        elif callable(self._on_oplog_updated):
            # Backward compatibility with older callback signature.
            try:
                self._on_oplog_updated(oplog_enabled)
            except Exception:
                pass
        else:
            try:
                ok_save, path_or_err = save_logging_policy(
                    {
                        "ops_csv_enabled": oplog_enabled,
                        "validation_summary_enabled": validation_summary_enabled,
                        "validation_comment_prompt_enabled": bool(
                            self._validation_comment_prompt_enabled
                        ),
                    }
                )
                if not ok_save:
                    print(f"[LOG][ERROR] Failed to save logging policy: {path_or_err}")
            except Exception:
                pass

        self._ensure_algo_cfg_defaults()
        self._rebuild_timeline_sources()
        if ok_shortcut_save:
            self._set_status(
                f"Settings updated. Shortcut config: {shortcut_path_or_err}"
            )
        else:
            self._set_status("Settings updated. Shortcut config was not saved.")
        self._log(
            "settings_update",
            timeline_playhead=timeline_section.get("playhead_radius"),
            timeline_empty=timeline_section.get("empty_space_radius"),
            timeline_edge=timeline_section.get("edge_search_radius"),
            timeline_segment=timeline_section.get("segment_soft_radius"),
            timeline_phase=timeline_section.get("phase_soft_radius"),
            timeline_hover_multi=timeline_section.get("hover_preview_multi"),
            timeline_hover_align=timeline_section.get("hover_preview_align"),
            boundary_snap_enabled=cb_snap.isChecked(),
            boundary_snap_window=self._algo_cfg["boundary_snap"]["window_size"],
            segment_trim=self._algo_cfg["segment_embedding"]["trim_ratio"],
            topk_enabled=cb_topk.isChecked(),
            topk_k=self._algo_cfg["topk"]["k"],
            topk_margin=self._algo_cfg["topk"]["uncertainty_margin"],
            assisted_min_gap=self._algo_cfg["assisted"]["boundary_min_gap"],
            psr_model_type=self._algo_cfg["psr"].get("model_type"),
            psr_initial_policy=self._algo_cfg["psr"].get("initial_state_policy"),
            psr_no_gap=self._algo_cfg["psr"].get("no_gap_timeline"),
            psr_auto_carry=self._algo_cfg["psr"].get("auto_carry_next_on_edit"),
            oplog_enabled=oplog_enabled,
            validation_summary_enabled=validation_summary_enabled,
            validation_comment_prompt_enabled=self._validation_comment_prompt_enabled,
            shortcuts_saved=bool(ok_shortcut_save),
            shortcuts_path=(shortcut_path_or_err if ok_shortcut_save else ""),
            shortcut_conflicts=len(conflicts),
        )
        if self._is_psr_task():
            try:
                self._psr_mark_dirty()
                self._psr_refresh_state_timeline(force=True)
                self._psr_update_component_panel()
            except Exception:
                pass

        if self.interaction_mode == "assisted":
            active = self._active_assisted_point()
            hint = None
            if active:
                if active.get("type") == "boundary":
                    try:
                        hint = ("boundary", int(active.get("frame", 0)))
                    except Exception:
                        hint = None
                elif active.get("type") == "label":
                    try:
                        hint = ("label", int(active.get("start", 0)))
                    except Exception:
                        hint = None
            try:
                self._build_assisted_points_from_store(
                    preserve_status=True, active_hint=hint
                )
                self._update_assisted_visuals()
            except Exception:
                pass

    def _enter_assisted_mode(self):
        if self.extra_mode:
            QMessageBox.information(
                self,
                "Interaction",
                "Exit Manual Segmentation before starting Assisted Review.",
            )
            try:
                self.btn_assisted.setChecked(False)
            except Exception:
                pass
            return
        if not self._has_auto_segments:
            QMessageBox.information(
                self,
                "Assisted Review",
                "Run automatic segmentation (EAST / ASOT / FACT) before entering Assisted Review.",
            )
            try:
                self.btn_assisted.setChecked(False)
            except Exception:
                pass
            return
        ok = self._build_assisted_points_from_store(preserve_status=False)
        if not ok or not self.assisted_points:
            QMessageBox.information(
                self,
                "Assisted Review",
                "No review points were found. Run auto-segmentation and try again.",
            )
            try:
                self.btn_assisted.setChecked(False)
            except Exception:
                pass
            return
        self.interaction_mode = "assisted"
        self._assisted_review_done = False
        try:
            self.btn_extra.setEnabled(False)
        except Exception:
            pass
        self._set_status("Assisted Review mode: review uncertain boundaries and labels.")
        self._set_interaction_status("Assisted: starting review")
        try:
            self.sc_review_next.setEnabled(False)
            self.sc_review_prev.setEnabled(False)
        except Exception:
            pass
        for sc in (
            getattr(self, "sc_assist_left", None),
            getattr(self, "sc_assist_right", None),
            getattr(self, "sc_assist_confirm", None),
            getattr(self, "sc_assist_down", None),
            getattr(self, "sc_assist_next", None),
            getattr(self, "sc_assist_prev", None),
            getattr(self, "sc_assist_skip", None),
            getattr(self, "sc_assist_merge", None),
            getattr(self, "sc_assist_merge_del", None),
        ):
            if sc:
                try:
                    sc.setEnabled(True)
                except Exception:
                    pass
        if self.assisted_active_idx < 0:
            self._set_active_assisted_idx(self._next_pending_interaction())

    def _exit_assisted_mode(self):
        self._discard_correction_session("assisted_mode_exit")
        self.interaction_mode = None
        self.assisted_points = []
        self.assisted_active_idx = -1
        self._assisted_review_done = False
        self._assisted_loop_range = None
        self._forced_segment = None
        try:
            self.timeline.set_interaction_points([])
        except Exception:
            pass
        try:
            self.timeline.set_highlight_labels([])
        except Exception:
            pass
        try:
            self.panel.clear_candidate_priority()
        except Exception:
            pass
        try:
            self.btn_assisted.setChecked(False)
            self.btn_extra.setEnabled(True)
        except Exception:
            pass
        try:
            self.sc_review_next.setEnabled(True)
            self.sc_review_prev.setEnabled(True)
        except Exception:
            pass
        for sc in (
            getattr(self, "sc_assist_left", None),
            getattr(self, "sc_assist_right", None),
            getattr(self, "sc_assist_confirm", None),
            getattr(self, "sc_assist_down", None),
            getattr(self, "sc_assist_next", None),
            getattr(self, "sc_assist_prev", None),
            getattr(self, "sc_assist_skip", None),
            getattr(self, "sc_assist_merge", None),
            getattr(self, "sc_assist_merge_del", None),
        ):
            if sc:
                try:
                    sc.setEnabled(False)
                except Exception:
                    pass
        self._set_interaction_status("Interaction: idle")

    def on_assisted_clicked(self):
        if self.btn_assisted.isChecked():
            self._enter_assisted_mode()
        else:
            self._exit_assisted_mode()

    # ----- Interaction (legacy Extra) helpers -----
    def _record_extra_cut(self, frame: int):
        try:
            f = max(0, int(frame))
        except Exception:
            return
        cuts = set(self.extra_cuts or [])
        cuts.add(f)
        self.extra_cuts = sorted(cuts)
        try:
            self.timeline.set_extra_cuts(self.extra_cuts)
            self.timeline.set_segment_cuts(self.extra_cuts)
        except Exception:
            pass
        try:
            self._align_labels_to_manual_segments()
        except Exception:
            pass

    def _extra_frames(self) -> List[int]:
        frames = set()
        for name in EXTRA_ALIASES:
            try:
                frames.update(self.extra_store.frames_of(name))
            except Exception:
                continue
        return sorted(frames)

    def _extra_runs(self) -> List[Tuple[int, int]]:
        """Get contiguous interaction spans; prefer recorded boundary cuts, fallback to runs from the store."""
        if self.extra_cuts:
            cuts = sorted(set(self.extra_cuts))
            frames = self._extra_frames()
            last_frame = max(frames) if frames else None
            if last_frame is None and self.extra_last_frame is not None:
                last_frame = self.extra_last_frame
            if last_frame is None:
                try:
                    last_frame = getattr(self.player, "crop_end", None)
                except Exception:
                    last_frame = None
            if last_frame is None:
                last_frame = cuts[-1]
            else:
                last_frame = max(last_frame, cuts[-1])
            if last_frame is None:
                return []
            runs = []
            for idx, start in enumerate(cuts):
                end = cuts[idx + 1] - 1 if idx + 1 < len(cuts) else last_frame
                if end < start:
                    continue
                runs.append((start, end))
            return runs
        frames = self._extra_frames()
        return AnnotationStore.frames_to_runs(frames)

    def _sync_extra_cuts_from_store(self):
        runs = AnnotationStore.frames_to_runs(self._extra_frames())
        self.extra_cuts = [s for s, _ in runs]
        self.extra_last_frame = None
        try:
            self.timeline.set_extra_cuts(self.extra_cuts)
            self.timeline.set_segment_cuts(self.extra_cuts)
        except Exception:
            pass

    def _flash_extra_overlay(self, frame: int):
        """Show a brief on-video overlay indicating the boundary frame."""
        try:
            player = self.views[self.active_view_idx]["player"]
        except Exception:
            player = getattr(self, "player", None)
        if not player:
            return
        # cancel prior timer
        if self._extra_overlay_timer is not None:
            try:
                self._extra_overlay_timer.stop()
            except Exception:
                pass
            self._extra_overlay_timer = None
        # remember old overlay state
        old_enabled = getattr(player, "overlay_enabled", False)
        old_labels = list(getattr(player, "overlay_labels", []))
        player.set_overlay_enabled(True)
        player.set_overlay_labels(
            [(f"{EXTRA_LABEL_NAME} boundary", "#ff66cc"), (f"Frame {frame}", "#ffffff")]
        )
        try:
            player.update()
        except Exception:
            pass
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(
            lambda: self._clear_extra_overlay(player, old_enabled, old_labels)
        )
        timer.start(900)
        self._extra_overlay_timer = timer

    def _clear_extra_overlay(self, player, old_enabled, old_labels):
        try:
            player.set_overlay_labels(old_labels if old_enabled else [])
            player.set_overlay_enabled(old_enabled)
            player.update()
        except Exception:
            pass
        self._extra_overlay_timer = None

    def _begin_extra_span(self, frame: int):
        label = self._ensure_extra_label()
        stores = self._extra_target_stores() or []
        if not stores:
            QMessageBox.information(
                self,
                EXTRA_LABEL_NAME,
                "No target rows available (check mode/entities).",
            )
            return
        self.extra_label = label
        self.extra_start_frame = frame
        self.extra_last_frame = frame - 1
        self._extra_txn_stores = stores
        for st in stores:
            if hasattr(st, "begin_txn"):
                st.begin_txn()
        self._record_extra_cut(frame)
        self._append_extra_progress(frame)

    def _append_extra_progress(self, frame: int):
        if not (self.extra_mode and self.extra_label):
            return
        if frame is None:
            return
        # ensure interaction row exists (e.g., after importing legacy JSON)
        try:
            has_row = any(
                getattr(r, "label", None) and r.label.name == self.extra_label.name
                for r in getattr(self.timeline, "rows", [])
            )
        except Exception:
            has_row = True
        if not has_row:
            try:
                self._rebuild_timeline_sources()
            except Exception:
                pass
        end = max(frame, self.extra_start_frame)
        start = (
            (self.extra_last_frame + 1)
            if self.extra_last_frame is not None
            else self.extra_start_frame
        )
        if end < start:
            return
        for st in self._extra_txn_stores or []:
            for f in range(start, end + 1):
                st.add(self.extra_label.name, f)
        self.extra_last_frame = end
        # keep timeline centered on current progress and highlight interaction row
        try:
            self.timeline.set_current_frame(
                end, follow=getattr(self, "_timeline_auto_follow", False)
            )
        except Exception:
            pass
        try:
            self.timeline.set_highlight_labels([self.extra_label.name])
        except Exception:
            pass
        try:
            self.timeline.refresh_all_rows()
            # explicit repaint to avoid stale cache in some cases
            self.timeline.update()
        except Exception:
            try:
                self.timeline.update()
            except Exception:
                pass

    def _finalize_extra_segment(self, end_frame: Optional[int], quiet: bool = False):
        if end_frame is not None:
            self._append_extra_progress(end_frame)
        for st in self._extra_txn_stores or []:
            if hasattr(st, "end_txn"):
                st.end_txn()
        self.extra_last_frame = None
        self._on_store_changed()
        try:
            self.timeline.refresh_all_rows()
        except Exception:
            self.timeline.update()
        self._save_extra_sidecar_if_possible()

    def _split_extra_at(self, frame: int):
        """Treat a click as a boundary: close current interaction span and immediately start a new one."""
        if not self.extra_mode:
            return
        try:
            frame = int(frame)
        except Exception:
            return
        crop_end = getattr(self.player, "crop_end", frame)
        crop_start = getattr(self.player, "crop_start", 0)
        min_frame = crop_start
        try:
            if self.extra_start_frame is not None:
                min_frame = max(min_frame, int(self.extra_start_frame))
        except Exception:
            pass
        frame = self._snap_boundary_frame(frame, lo=min_frame, hi=crop_end)
        # close current run at frame-1, then start fresh at frame
        end_frame = max(crop_start, min(frame - 1, crop_end))
        if end_frame >= crop_start:
            self._finalize_extra_segment(end_frame, quiet=True)
        if frame <= crop_end:
            self._begin_extra_span(frame)
            self._append_extra_progress(frame)
            self._note_correction_step()
            try:
                self.timeline.set_current_frame(frame, follow=True)
                self.timeline.flash_label(
                    self.extra_label.name if self.extra_label else EXTRA_LABEL_NAME
                )
                self.timeline.flash_boundary_marker(frame)
            except Exception:
                pass
            msg = f"{EXTRA_LABEL_NAME} boundary @ frame {frame}"
            self._set_status(msg)
            try:
                QToolTip.showText(QCursor.pos(), msg, self)
            except Exception:
                pass
            self._flash_extra_overlay(frame)

    def _ensure_extra_label(self) -> LabelDef:
        for lb in self.labels:
            if is_extra_label(lb.name):
                if lb.name != EXTRA_LABEL_NAME:
                    old = lb.name
                    lb.name = EXTRA_LABEL_NAME
                    try:
                        self.store.rename_label(old, lb.name)
                    except Exception:
                        pass
                    try:
                        self.extra_store.rename_label(old, lb.name)
                    except Exception:
                        pass
                    if old in self.label_entity_map:
                        self.label_entity_map[lb.name] = self.label_entity_map.pop(old)
                return lb
        try:
            max_id = max(int(getattr(lb, "id", i)) for i, lb in enumerate(self.labels))
        except Exception:
            max_id = len(self.labels)
        next_id = max_id + 1
        col = self._auto_color_key_for_id(next_id)
        extra = LabelDef(name=EXTRA_LABEL_NAME, color_name=col, id=next_id)
        self.labels.append(extra)
        try:
            self.panel.refresh()
        except Exception:
            pass
        self._rebuild_timeline_sources()
        self._dirty = True
        return extra

    def _extra_sidecar_path(self) -> Optional[str]:
        if not self.current_annotation_path:
            return None
        base, ext = os.path.splitext(self.current_annotation_path)
        return f"{base}_extra{ext or '.json'}"

    def _save_extra_sidecar_if_possible(self):
        path = self._extra_sidecar_path()
        if not path:
            return
        extra = None
        for lb in self.labels:
            if is_extra_label(lb.name):
                extra = lb
                break
        if not extra:
            return
        runs = self._extra_runs()
        segments = []
        for s, e in runs:
            segments.append(
                {
                    "start_frame": int(s),
                    "end_frame": int(e),
                    "action_label": int(extra.id),
                }
            )
        payload = {
            "video_id": os.path.splitext(os.path.basename(path))[0],
            "labels": [
                {
                    "id": int(extra.id),
                    "name": EXTRA_LABEL_NAME,
                    "color": extra.color_name,
                }
            ],
            "segments": segments,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as ex:
            print(f"[EXTRA] Failed to save sidecar: {ex}")

    def _load_extra_sidecar(self, main_path: str):
        base, ext = os.path.splitext(main_path)
        path = f"{base}_extra{ext or '.json'}"
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as ex:
            print(f"[EXTRA] Failed to load sidecar: {ex}")
            return
        if not isinstance(data, dict):
            print("[EXTRA] Invalid sidecar root (expected object).")
            return
        extra = self._ensure_extra_label()
        id2label = {}
        for item in data.get("labels", []):
            if not isinstance(item, dict):
                continue
            try:
                lid = int(item.get("id"))
            except Exception:
                continue
            id2label[lid] = item.get("name", "")

        segs = data.get("segments", [])
        if not isinstance(segs, list):
            print("[EXTRA] Invalid sidecar: 'segments' must be a list.")
            return
        self.extra_cuts = []
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            try:
                s = int(seg.get("start_frame", 0))
                e = int(seg.get("end_frame", s))
                lid = int(seg.get("action_label"))
            except Exception:
                continue
            name = id2label.get(lid, None)
            if not is_extra_label(name):
                continue
            for f in range(s, e + 1):
                self.extra_store.add(extra.name, f)
            self._record_extra_cut(s)

    def _extra_target_stores(self) -> List[AnnotationStore]:
        return [self.extra_store]

    def _align_labels_to_manual_segments(self) -> bool:
        """Assign a single label to each manual segment using majority labels from prelabel store."""
        runs = self._extra_runs()
        if not runs:
            return False
        source_store = (
            self.prelabel_store
            if getattr(self.prelabel_store, "frame_to_label", None)
            else self.store
        )
        if not getattr(source_store, "frame_to_label", None):
            self._set_status("No prelabel labels to align with manual segments.")
            return False
        changed = 0
        try:
            self.store.begin_txn()
        except Exception:
            pass
        for s, e in runs:
            try:
                s = int(s)
                e = int(e)
            except Exception:
                continue
            if e < s:
                s, e = e, s
            counts = {}
            for f in range(s, e + 1):
                lb = source_store.label_at(f)
                if not lb or is_extra_label(lb):
                    continue
                counts[lb] = counts.get(lb, 0) + 1
            if not counts:
                continue
            target_label = max(counts.items(), key=lambda x: x[1])[0]
            for f in range(s, e + 1):
                cur = self.store.label_at(f)
                if cur == target_label:
                    continue
                if cur is not None:
                    self.store.remove_at(f)
                self.store.add(target_label, f)
                changed += 1
        try:
            self.store.end_txn()
        except Exception:
            pass
        if changed > 0:
            self._on_store_changed()
            self._set_status(
                f"Aligned labels to manual segments ({len(runs)} segments)."
            )
            return True
        return False

    def enter_extra_mode(self):
        if self.interaction_mode == "assisted":
            QMessageBox.information(
                self,
                "Interaction",
                "Exit Assisted Review before starting Manual Segmentation.",
            )
            try:
                self.btn_extra.setChecked(False)
            except Exception:
                pass
            return
        self.extra_label = self._ensure_extra_label()
        self._begin_correction_session(
            "manual_segmentation",
            start=int(getattr(self.player, "current_frame", 0)),
        )
        self.extra_mode = True
        self.interaction_mode = "manual"
        self.extra_last_frame = None
        self._extra_txn_stores = []
        self._extra_force_follow = True
        if not self.extra_cuts:
            self._sync_extra_cuts_from_store()
        start_frame = getattr(self.player, "current_frame", 0)
        self._begin_extra_span(start_frame)
        self._freeze = True
        try:
            self.btn_extra.setChecked(True)
        except Exception:
            pass
        try:
            self._play_all()
        except Exception:
            pass
        self._set_status(
            f"{EXTRA_LABEL_NAME} mode: auto-filling current span; click to add boundaries."
        )
        self._set_interaction_status("Manual Segmentation: active")

    def exit_extra_mode(self):
        # finalize any ongoing segment before exiting
        end_frame = getattr(
            self.player, "crop_end", getattr(self.player, "current_frame", None)
        )
        self._finalize_extra_segment(end_frame)
        # align labels to manual segments using existing predictions
        try:
            self._align_labels_to_manual_segments()
        except Exception:
            pass
        self._commit_correction_session(point_type="manual_segmentation")
        self.extra_mode = False
        self._extra_force_follow = False
        self.extra_label = None
        self.extra_last_frame = None
        self._extra_txn_stores = []
        self._freeze = False
        self.interaction_mode = None
        try:
            self.btn_extra.setChecked(False)
        except Exception:
            pass
        self._set_status(f"{EXTRA_LABEL_NAME} mode off.")
        self._set_interaction_status("Interaction: idle")

    def freeze_ui(self, on: bool):
        self._freeze = bool(on)

    def eventFilter(self, obj, event):
        if getattr(self, "_freeze", False) and self.extra_mode:
            allowed = []
            for vw in self.views:
                try:
                    allowed.append(vw.get("player"))
                except Exception:
                    continue
            allowed.append(getattr(self, "btn_extra", None))
            allowed.append(getattr(self, "combo_interaction", None))
            if event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
                if obj not in allowed:
                    return True
        return super().eventFilter(obj, event)

    def on_extra_clicked(self):
        if self.interaction_mode == "assisted":
            QMessageBox.information(
                self,
                "Interaction",
                "Assisted Review is active. Please exit it before switching to Manual Segmentation.",
            )
            try:
                self.btn_extra.setChecked(False)
            except Exception:
                pass
            return
        if not self.extra_mode:
            self.enter_extra_mode()
        else:
            self.exit_extra_mode()

    def _load_prediction_output(
        self, txt_path: str, json_path: Optional[str]
    ) -> Tuple[List[str], List[Dict[str, Any]], Dict[Tuple[int, int], List[Tuple[str, Optional[float]]]]]:
        classes: List[str] = []
        segs: List[Dict[str, Any]] = []
        topk_map: Dict[Tuple[int, int], List[Tuple[str, Optional[float]]]] = {}
        if json_path and os.path.isfile(json_path):
            try:
                obj = json.load(open(json_path, "r", encoding="utf-8"))
                classes = [str(x) for x in (obj.get("classes") or []) if str(x or "").strip()]
                raw = obj.get("segments") or []
                for s in raw:
                    try:
                        sf = int(s.get("start_frame", 0))
                        ef = int(s.get("end_frame", 0))
                        cid = s.get("class_id", 0)
                        cname = s.get("class_name")
                        segs.append(
                            {
                                "start_frame": sf,
                                "end_frame": ef,
                                "class_id": int(cid),
                                "class_name": cname,
                            }
                        )
                        topk = []
                        for tk in s.get("topk", []):
                            name = tk.get("name", cname)
                            score = tk.get("score")
                            if (
                                name is None
                                and "id" in tk
                                and int(tk["id"]) < len(classes)
                            ):
                                name = classes[int(tk["id"])]
                            if name:
                                topk.append(
                                    (str(name), score if score is None else float(score))
                                )
                        if topk:
                            topk_map[(sf, ef)] = topk
                    except Exception:
                        continue
            except Exception:
                classes = []
                segs = []
                topk_map = {}

        if not segs:
            try:
                for seg in load_segments_txt(txt_path):
                    label_name = str(seg.get("label", "") or "").strip()
                    segs.append(
                        {
                            "start_frame": int(seg.get("start", 0)),
                            "end_frame": int(seg.get("end", seg.get("start", 0))),
                            "class_name": label_name,
                        }
                    )
                    if label_name and label_name not in classes:
                        classes.append(label_name)
            except Exception:
                segs = []
        return classes, segs, topk_map

    def _ensure_action_labels_for_names(self, class_names: Iterable[str]) -> None:
        existing = {str(lb.name): int(getattr(lb, "id", idx)) for idx, lb in enumerate(self.labels)}
        next_id = max(existing.values(), default=-1) + 1
        changed = False
        for raw in class_names or []:
            name = str(raw or "").strip()
            if not name or name in existing:
                continue
            self.labels.append(
                LabelDef(
                    name=name,
                    color_name=self._auto_color_key_for_id(next_id),
                    id=int(next_id),
                )
            )
            existing[name] = int(next_id)
            next_id += 1
            changed = True
        if changed:
            self._refresh_fine_label_decomposition(refresh_panel=False)
            try:
                self.panel.refresh()
            except Exception:
                pass

    def _prediction_segments_for_active_view(
        self,
        segs: List[Dict[str, Any]],
        classes: List[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[int, str], List[int]]:
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return [], {}, []
        view = self.views[self.active_view_idx]
        view_start = int(view.get("start", 0))
        view_end = int(view.get("end", view_start))
        pred_segments: List[Dict[str, Any]] = []
        pred_map: Dict[int, str] = {}
        boundary_candidates: List[int] = []
        id_to_name = {
            int(getattr(lb, "id", idx)): str(lb.name) for idx, lb in enumerate(self.labels)
        }
        try:
            sorted_segs = sorted(segs, key=lambda s: int(s.get("start_frame", 0)))
        except Exception:
            sorted_segs = list(segs or [])
        for idx, seg in enumerate(sorted_segs):
            try:
                s = int(seg.get("start_frame", 0)) + view_start
                e = int(seg.get("end_frame", seg.get("start_frame", 0))) + view_start
            except Exception:
                continue
            if view_end >= view_start:
                if e < view_start or s > view_end:
                    continue
                s = max(s, view_start)
                e = min(e, view_end)
            cid = seg.get("class_id", None)
            name = seg.get("class_name")
            if cid is not None and int(cid) in id_to_name:
                name = id_to_name[int(cid)]
            elif cid is not None and 0 <= int(cid) < len(classes):
                name = str(classes[int(cid)])
            name = str(name or "").strip()
            if not name:
                continue
            pred_segments.append({"start": int(s), "end": int(e), "label": name})
            for frame in range(int(s), int(e) + 1):
                pred_map[int(frame)] = name
            if idx > 0:
                boundary_candidates.append(int(s))
        return pred_segments, pred_map, sorted(set(boundary_candidates))

    def _locked_segments_for_active_view(self) -> List[Dict[str, Any]]:
        records = self._rebuild_confirmed_correction_records_for_active_view()
        locked: List[Dict[str, Any]] = []
        seen = set()
        for rec in records:
            if str(rec.get("action_kind", "") or "").strip().lower() != "segment_lock":
                continue
            try:
                s = int(rec.get("feedback_start", 0))
                e = int(rec.get("feedback_end", s))
            except Exception:
                continue
            if e < s:
                s, e = e, s
            label = str(rec.get("label", "") or "").strip()
            key = (int(s), int(e), label)
            if not label or key in seen:
                continue
            seen.add(key)
            locked.append({"start": int(s), "end": int(e), "label": label})
        return sorted(locked, key=lambda seg: (int(seg["start"]), int(seg["end"])))

    def _subtract_locked_spans(
        self,
        start: int,
        end: int,
        locked_spans: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        fragments = [(int(start), int(end))]
        for lock_s, lock_e in locked_spans:
            next_parts: List[Tuple[int, int]] = []
            for frag_s, frag_e in fragments:
                if frag_e < lock_s or frag_s > lock_e:
                    next_parts.append((frag_s, frag_e))
                    continue
                if frag_s < lock_s:
                    next_parts.append((frag_s, lock_s - 1))
                if frag_e > lock_e:
                    next_parts.append((lock_e + 1, frag_e))
            fragments = next_parts
            if not fragments:
                break
        return [(int(s), int(e)) for s, e in fragments if int(e) >= int(s)]

    def _masked_refresh_safe_state(self, features_dir: str) -> Tuple[bool, str]:
        path = os.path.abspath(os.path.expanduser(str(features_dir or "").strip()))
        current_dir = os.path.abspath(
            os.path.expanduser(str(self._east_assets_features_dir() or "").strip())
        )
        if not path or not os.path.isdir(path):
            return False, "invalid_features_dir"
        if not current_dir or path != current_dir:
            return False, "stale_runtime"
        if not self._is_action_task() or self._is_psr_task():
            return False, "inactive_task"
        if str(getattr(self, "mode", "") or "") != "Coarse":
            return False, "non_coarse_mode"
        if bool(self.interaction_mode) or bool(self.extra_mode):
            return False, "interaction_busy"
        if bool(getattr(self, "_infer_thread_east", None)) and self._infer_thread_east.isRunning():
            return False, "east_refine_running"
        if self._east_online_update_running or self._east_online_adapter_running:
            return False, "online_update_running"
        if any(getattr(st, "frame_to_label", None) for st in (self.entity_stores or {}).values()):
            return False, "entity_annotations_present"
        if any(getattr(st, "frame_to_label", None) for st in (self.phase_stores or {}).values()):
            return False, "phase_annotations_present"
        return True, ""

    # ----- auto labeling results -----
    def _apply_autolabel_predictions(
        self, txt_path: str, json_path: Optional[str], model_name: str = "FACT"
    ):
        if not txt_path or not os.path.isfile(txt_path):
            QMessageBox.warning(
                self, "Auto label", f"{model_name} prediction file not found."
            )
            return
        self._assisted_candidates = {}
        self._auto_boundary_candidates = []
        self._auto_boundary_source = ""

        classes, segs, topk_map = self._load_prediction_output(txt_path, json_path)
        # cache top-k candidates for assisted label review
        self._assisted_candidates = topk_map

        if not segs:
            QMessageBox.warning(
                self, "Auto label", f"No segments found in {model_name} output."
            )
            return

        view = self.views[self.active_view_idx] if self.views else {}
        view_start = int(view.get("start", 0))
        view_end = int(view.get("end", view_start))
        boundary_candidates = []
        try:
            sorted_segs = sorted(segs, key=lambda s: int(s.get("start_frame", 0)))
        except Exception:
            sorted_segs = segs
        for seg in sorted_segs[1:]:
            try:
                boundary_candidates.append(int(seg.get("start_frame", 0)) + view_start)
            except Exception:
                continue
        if self.views and 0 <= self.active_view_idx < len(self.views):
            try:
                current_cuts = self._trim_cut_set_for_view(
                    self.views[self.active_view_idx],
                    {"kind": "store"},
                    create=True,
                )
                current_cuts.clear()
                baseline_cuts = self._baseline_trim_cut_set_for_view(
                    self.views[self.active_view_idx],
                    {"kind": "store"},
                    create=True,
                )
                baseline_cuts.clear()
                baseline_cuts.update(int(x) for x in boundary_candidates)
            except Exception:
                pass
        baseline_source = str(model_name or "").strip().upper()
        self._confirmed_correction_records = []
        if self.views and 0 <= self.active_view_idx < len(self.views):
            self.views[self.active_view_idx]["confirmed_accept_records"] = []
        self._write_correction_record_buffer([], force=True)
        self._correction_buffer = CorrectionBuffer()
        self._schedule_east_online_update(self._correction_features_dir())
        if "ASOT" in str(model_name).upper():
            self._auto_boundary_candidates = sorted(set(boundary_candidates))
        else:
            self._auto_boundary_candidates = []
        self._auto_boundary_source = model_name

        # ensure label defs contain predicted classes
        name_to_id = {
            lb.name: int(getattr(lb, "id", idx)) for idx, lb in enumerate(self.labels)
        }
        id_to_name = {
            int(getattr(lb, "id", idx)): lb.name for idx, lb in enumerate(self.labels)
        }
        for idx, cname in enumerate(classes):
            if cname not in name_to_id:
                self.labels.append(
                    LabelDef(
                        name=str(cname),
                        color_name=self._auto_color_key_for_id(idx),
                        id=idx,
                    )
                )
                name_to_id[str(cname)] = idx
                id_to_name[idx] = str(cname)

        # Build per-frame prediction map (within current view span)
        pred_map = {}
        for seg in segs:
            try:
                s = int(seg.get("start_frame", 0)) + view_start
                e = int(seg.get("end_frame", seg.get("start_frame", 0))) + view_start
            except Exception:
                continue
            if view_end >= view_start:
                if e < view_start or s > view_end:
                    continue
                e = min(e, view_end)
            cid = seg.get("class_id", None)
            name = seg.get("class_name")
            if cid is not None and cid in id_to_name:
                name = id_to_name[cid]
            if not name:
                name = f"Label_{int(cid) if cid is not None else len(id_to_name)}"
                if name not in name_to_id:
                    nid = int(cid) if cid is not None else len(name_to_id)
                    self.labels.append(
                        LabelDef(
                            name=name,
                            color_name=self._auto_color_key_for_id(nid),
                            id=nid,
                        )
                    )
                    name_to_id[name] = nid
                    id_to_name[nid] = name
            for f in range(s, e + 1):
                pred_map[f] = name

        # Cache prelabel predictions separately for alignment with manual segments.
        try:
            prelabel = AnnotationStore()
            for f, name in pred_map.items():
                prelabel.add(name, f)
            self.prelabel_store = prelabel
            self._prelabel_source = baseline_source
            if self.views and 0 <= self.active_view_idx < len(self.views):
                self.views[self.active_view_idx]["prelabel_store"] = prelabel
                self.views[self.active_view_idx]["prelabel_source"] = baseline_source
        except Exception:
            pass

        extra_frames = set(self._extra_frames())
        apply_scope = "all"
        if extra_frames:
            ret = QMessageBox.question(
                self,
                model_name,
                (
                    f"Existing {EXTRA_LABEL_NAME} spans were detected.\n"
                    f"Apply {model_name} predictions only inside those spans?\n"
                    "Choose No to replace annotations across the full video."
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            apply_scope = "extra" if ret == QMessageBox.Yes else "all"

        if extra_frames and apply_scope == "extra":
            # Apply predictions only on interaction spans; preserve existing labels elsewhere.
            applied = 0
            for f, name in pred_map.items():
                if f not in extra_frames:
                    continue
                old = self.store.label_at(f)
                if old and old != name:
                    self.store.remove_at(f)
                if (not old) or (old != name):
                    self.store.add(name, f)
                    applied += 1
            self._set_status(
                f"Applied {model_name} predictions to {EXTRA_LABEL_NAME} spans ({applied} frames)."
            )
        else:
            # Fallback: replace current annotations
            if self.store.frame_to_label:
                ret = QMessageBox.question(
                    self,
                    "Replace annotations",
                    f"This will replace current annotations with {model_name} predictions. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if ret != QMessageBox.Yes:
                    return

            self.store.frame_to_label.clear()
            self.store.label_to_frames.clear()
            for st in self.entity_stores.values():
                st.frame_to_label.clear()
                st.label_to_frames.clear()
            self.entity_stores.clear()
            self.label_entity_map.clear()
            self.entities.clear()
            self.visible_entities.clear()
            self.current_label_idx = -1
            try:
                self.entities_panel.set_current_label(None, set())
                self.entities_panel.setVisible(False)
            except Exception:
                pass
            self.mode = "Coarse"
            try:
                self.combo_mode.setCurrentText("Coarse")
            except Exception:
                pass

            # rebuild labels from classes list (keep interaction if present)
            keep_extra = any(is_extra_label(lb.name) for lb in self.labels)
            self.labels = []
            if keep_extra:
                self.labels.append(
                    LabelDef(
                        name=EXTRA_LABEL_NAME,
                        color_name=self._auto_color_key_for_id(0),
                        id=0,
                    )
                )
            id2name = {}
            for idx, cname in enumerate(classes):
                name = str(cname)
                self.labels.append(
                    LabelDef(
                        name=name,
                        color_name=self._auto_color_key_for_id(
                            idx + (1 if keep_extra else 0)
                        ),
                        id=idx,
                    )
                )
                id2name[idx] = name

            for f, name in pred_map.items():
                self.store.add(name, f)
            self.extra_store = AnnotationStore()
            self.extra_cuts = []
            result_name = {
                "EAST": "EAST refinement",
                "ASOT": "ASOT pre-labeling",
                "FACT": "FACT labeling",
            }.get(str(model_name or "").upper(), str(model_name or "Model"))
            self._set_status(
                f"Loaded {result_name} results from {os.path.basename(txt_path)}"
            )

        self.panel.refresh()
        self._rebuild_timeline_sources()
        try:
            self.timeline.refresh_all_rows()
        except Exception:
            self.timeline.update()
        self._dirty = True
        self._has_auto_segments = True

    def _apply_east_predictions_masked(
        self, txt_path: str, json_path: Optional[str]
    ) -> bool:
        if not txt_path or not os.path.isfile(txt_path):
            return False
        features_dir = os.path.abspath(os.path.dirname(txt_path))
        ok, _reason = self._masked_refresh_safe_state(features_dir)
        if not ok:
            return False
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return False
        view = self.views[self.active_view_idx]
        current_store = view.get("store") or self.store
        if current_store is None:
            return False
        baseline_store = self._active_label_baseline_store()
        if baseline_store is None:
            baseline_store = AnnotationStore()
            view["prelabel_store"] = baseline_store
            self.prelabel_store = baseline_store
        view["prelabel_source"] = "EAST"
        self._prelabel_source = "EAST"

        classes, segs, topk_map = self._load_prediction_output(txt_path, json_path)
        if not segs:
            return False
        self._ensure_action_labels_for_names(classes)
        pred_segments, pred_map, boundary_candidates = self._prediction_segments_for_active_view(
            segs, classes
        )
        if not pred_segments:
            return False

        view_start = int(view.get("start", 0))
        view_end = int(view.get("end", view_start))
        if view_end < view_start:
            view_start, view_end = view_end, view_start

        locked_segments = self._locked_segments_for_active_view()
        locked_spans = self._merge_spans(
            [(int(seg.get("start", 0)), int(seg.get("end", 0))) for seg in locked_segments]
        )
        final_segments: List[Dict[str, Any]] = []
        for seg in pred_segments:
            label = str(seg.get("label", "") or "").strip()
            if not label:
                continue
            for frag_s, frag_e in self._subtract_locked_spans(
                int(seg.get("start", 0)),
                int(seg.get("end", 0)),
                locked_spans,
            ):
                final_segments.append(
                    {"start": int(frag_s), "end": int(frag_e), "label": label}
                )
        for seg in locked_segments:
            label = str(seg.get("label", "") or "").strip()
            if not label:
                continue
            final_segments.append(
                {
                    "start": int(seg.get("start", 0)),
                    "end": int(seg.get("end", seg.get("start", 0))),
                    "label": label,
                }
            )
        final_segments = sorted(
            [
                {
                    "start": max(view_start, int(seg.get("start", 0))),
                    "end": min(view_end, int(seg.get("end", seg.get("start", 0)))),
                    "label": str(seg.get("label", "") or "").strip(),
                }
                for seg in final_segments
                if str(seg.get("label", "") or "").strip()
            ],
            key=lambda seg: (int(seg["start"]), int(seg["end"]), str(seg["label"])),
        )
        final_segments = [seg for seg in final_segments if int(seg["end"]) >= int(seg["start"])]
        if not final_segments:
            return False

        final_frame_map: Dict[int, str] = {}
        for seg in final_segments:
            for frame in range(int(seg["start"]), int(seg["end"]) + 1):
                final_frame_map[int(frame)] = str(seg["label"])

        unlocked_spans: List[Tuple[int, int]] = []
        cursor = int(view_start)
        for lock_s, lock_e in locked_spans:
            lock_s = max(int(view_start), int(lock_s))
            lock_e = min(int(view_end), int(lock_e))
            if cursor < lock_s:
                unlocked_spans.append((int(cursor), int(lock_s - 1)))
            cursor = max(int(cursor), int(lock_e + 1))
        if cursor <= int(view_end):
            unlocked_spans.append((int(cursor), int(view_end)))

        self._replace_store_frames_in_span(
            current_store,
            int(view_start),
            int(view_end),
            final_frame_map,
        )
        self._patch_store_frames_in_spans(
            baseline_store,
            unlocked_spans,
            pred_map,
        )

        current_cuts = self._trim_cut_set_for_view(view, {"kind": "store"}, create=True)
        final_cut_positions = {
            int(seg.get("start", 0))
            for seg in final_segments
            if int(seg.get("start", 0)) > int(view_start)
        }
        self._replace_cut_set_in_span(
            current_cuts,
            int(view_start),
            int(view_end),
            final_cut_positions,
        )

        baseline_cuts = self._baseline_trim_cut_set_for_view(
            view, {"kind": "store"}, create=True
        )
        protected_cuts = set()
        for seg in locked_segments:
            s = int(seg.get("start", 0))
            e = int(seg.get("end", s))
            if s > int(view_start):
                protected_cuts.add(int(s))
            if e < int(view_end):
                protected_cuts.add(int(e + 1))
        self._replace_cut_set_excluding_positions(
            baseline_cuts,
            int(view_start),
            int(view_end),
            protected_positions=protected_cuts,
            new_cuts=boundary_candidates,
        )

        self.currentEastFeatureDir = features_dir
        self._assisted_candidates = dict(topk_map or {})
        self._auto_boundary_source = "EAST"
        self._dirty = True
        self._has_auto_segments = True
        self.panel.refresh()
        self._rebuild_timeline_sources()
        try:
            self.timeline.refresh_all_rows()
        except Exception:
            self.timeline.update()
        self._update_gap_indicator()
        self._rebuild_confirmed_correction_records_for_active_view()
        self._write_correction_record_buffer(self._confirmed_correction_records)
        return True

    def _patch_assisted_candidates_in_spans(
        self,
        spans: List[Tuple[int, int]],
        topk_map: Dict[Tuple[int, int], List[Tuple[str, Optional[float]]]],
    ) -> None:
        merged = self._merge_spans(list(spans or []))
        current = dict(getattr(self, "_assisted_candidates", {}) or {})
        kept: Dict[Tuple[int, int], List[Tuple[str, Optional[float]]]] = {}
        for key, rows in current.items():
            try:
                s = int(key[0])
                e = int(key[1])
            except Exception:
                continue
            if any(not (e < ms or s > me) for ms, me in merged):
                continue
            clean_rows: List[Tuple[str, Optional[float]]] = []
            for name, score in (rows or []):
                clean_rows.append((str(name), None if score is None else float(score)))
            kept[(int(s), int(e))] = clean_rows
        for key, rows in (topk_map or {}).items():
            try:
                s = int(key[0])
                e = int(key[1])
            except Exception:
                continue
            if not any(not (e < ms or s > me) for ms, me in merged):
                continue
            clean_rows = []
            for name, score in (rows or []):
                clean_rows.append((str(name), None if score is None else float(score)))
            kept[(int(s), int(e))] = clean_rows
        self._assisted_candidates = kept

    def _patch_auto_boundary_candidates_in_spans(
        self,
        spans: List[Tuple[int, int]],
        candidates: Iterable[int],
    ) -> None:
        merged = self._merge_spans(list(spans or []))
        keep = [
            int(frame)
            for frame in (getattr(self, "_auto_boundary_candidates", []) or [])
            if not any(int(ms) <= int(frame) <= int(me) for ms, me in merged)
        ]
        keep.extend(
            int(frame)
            for frame in (candidates or [])
            if any(int(ms) <= int(frame) <= int(me) for ms, me in merged)
        )
        self._auto_boundary_candidates = sorted(set(int(x) for x in keep))

    def _apply_east_predictions_masked_local(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False
        features_dir = os.path.abspath(
            os.path.expanduser(str(payload.get("features_dir") or "").strip())
        )
        ok, _reason = self._masked_refresh_safe_state(features_dir)
        if not ok:
            return False
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return False
        view = self.views[self.active_view_idx]
        current_store = view.get("store") or self.store
        if current_store is None:
            return False
        baseline_store = self._active_label_baseline_store()
        if baseline_store is None:
            baseline_store = AnnotationStore()
            view["prelabel_store"] = baseline_store
            self.prelabel_store = baseline_store
        view["prelabel_source"] = "EAST"
        self._prelabel_source = "EAST"

        windows = [dict(item) for item in (payload.get("windows") or []) if isinstance(item, dict)]
        pred_segments: List[Dict[str, Any]] = []
        classes: List[str] = []
        topk_map: Dict[Tuple[int, int], List[Tuple[str, Optional[float]]]] = {}
        boundary_candidates: List[int] = []
        covered_spans: List[Tuple[int, int]] = []
        for item in windows:
            if not bool(item.get("ok", False)):
                continue
            for name in item.get("classes") or []:
                label = str(name or "").strip()
                if label and label not in classes:
                    classes.append(label)
            for seg in item.get("segments") or []:
                try:
                    s = int(seg.get("start", 0))
                    e = int(seg.get("end", s))
                    label = str(seg.get("label", "") or "").strip()
                except Exception:
                    continue
                if not label or e < s:
                    continue
                pred_segments.append({"start": int(s), "end": int(e), "label": label})
            for raw in item.get("boundary_candidates") or []:
                try:
                    boundary_candidates.append(int(raw))
                except Exception:
                    continue
            for row in item.get("topk_map") or []:
                if not isinstance(row, dict):
                    continue
                try:
                    s = int(row.get("start", 0))
                    e = int(row.get("end", s))
                except Exception:
                    continue
                items: List[Tuple[str, Optional[float]]] = []
                for sub in row.get("items") or []:
                    if not isinstance(sub, dict):
                        continue
                    name = str(sub.get("name", "") or "").strip()
                    if not name:
                        continue
                    score = sub.get("score")
                    try:
                        score = float(score) if score is not None else None
                    except Exception:
                        score = None
                    items.append((name, score))
                if items:
                    topk_map[(int(s), int(e))] = items
            for span in item.get("focus_spans") or []:
                try:
                    covered_spans.append((int(span[0]), int(span[1])))
                except Exception:
                    continue

        covered_spans = self._merge_spans(covered_spans)
        if not pred_segments or not covered_spans:
            return False
        self._ensure_action_labels_for_names(classes)

        pred_map: Dict[int, str] = {}
        for seg in pred_segments:
            for frame in range(int(seg["start"]), int(seg["end"]) + 1):
                pred_map[int(frame)] = str(seg["label"])

        self._patch_store_frames_in_spans(current_store, covered_spans, pred_map)
        self._patch_store_frames_in_spans(baseline_store, covered_spans, pred_map)

        current_cuts = self._trim_cut_set_for_view(view, {"kind": "store"}, create=True)
        baseline_cuts = self._baseline_trim_cut_set_for_view(
            view, {"kind": "store"}, create=True
        )
        for span_s, span_e in covered_spans:
            local_cuts = [
                int(frame)
                for frame in boundary_candidates
                if int(span_s) <= int(frame) <= int(span_e)
            ]
            self._replace_cut_set_in_span(current_cuts, int(span_s), int(span_e), local_cuts)
            self._replace_cut_set_in_span(baseline_cuts, int(span_s), int(span_e), local_cuts)

        self.currentEastFeatureDir = features_dir
        self._patch_assisted_candidates_in_spans(covered_spans, topk_map)
        self._patch_auto_boundary_candidates_in_spans(covered_spans, boundary_candidates)
        self._auto_boundary_source = "EAST"
        self._dirty = True
        self._has_auto_segments = True
        self.panel.refresh()
        self._rebuild_timeline_sources()
        try:
            self.timeline.refresh_all_rows()
        except Exception:
            self.timeline.update()
        self._update_gap_indicator()
        try:
            self._build_assisted_points_from_store(preserve_status=True, active_hint=None)
        except Exception:
            pass
        self._rebuild_confirmed_correction_records_for_active_view()
        self._write_correction_record_buffer(self._confirmed_correction_records)
        return True

    def _on_infer_done(self, txt_path: str, json_path: str, model_name: str = "FACT"):
        run_name = {
            "EAST": "EAST refinement",
            "ASOT": "ASOT pre-labeling",
            "FACT": "FACT labeling",
        }.get(str(model_name or "").upper(), f"{model_name} run")
        if not txt_path:
            log_hint = getattr(self, "_last_autolabel_log_path", "")
            if log_hint:
                self._set_status(f"{run_name} failed; see log: {log_hint}")
            else:
                self._set_status(f"{run_name} failed; check logs.")
            return
        self._apply_autolabel_predictions(txt_path, json_path, model_name=model_name)

    def _on_fact_batch_done(self, ok: bool, output_dir: str):
        self._close_progress_dialog(getattr(self, "_fact_batch_progress", None))
        self._fact_batch_progress = None
        if ok:
            self._set_status(f"FACT batch labeling done. Outputs in {output_dir}")
        else:
            log_hint = os.path.join(output_dir, "pred_fact_batch.log")
            if os.path.isfile(log_hint):
                self._set_status(f"FACT batch labeling failed; see log: {log_hint}")
            else:
                self._set_status("FACT batch labeling failed.")

    def _on_east_batch_progress(self, line: str):
        self._set_status(line)
        dlg = getattr(self, "_east_batch_progress", None)
        if not dlg:
            return
        m = re.search(r"\((\d+)/(\d+)\)", line)
        if m:
            current = int(m.group(1))
            total = int(m.group(2))
            self._east_batch_current = current
            self._east_batch_total = total
            done = int(getattr(self, "_east_batch_done", 0) or 0)
            if current > 1:
                done = max(done, current - 1)
                self._east_batch_done = done
        if line.startswith("[INFO] Found"):
            m = re.search(r"Found\s+(\d+)", line)
            if m:
                self._east_batch_total = int(m.group(1))
        if line.startswith("[OK] EAST inference done:") or line.startswith("[WARN] failed "):
            self._east_batch_done = int(getattr(self, "_east_batch_done", 0) or 0) + 1
        if line.startswith("[OK] EAST batch inference done"):
            if getattr(self, "_east_batch_total", None):
                self._east_batch_done = int(self._east_batch_total)

        total = int(getattr(self, "_east_batch_total", 0) or 0)
        done = int(getattr(self, "_east_batch_done", 0) or 0)
        current = int(getattr(self, "_east_batch_current", 0) or 0)
        if total > 0:
            if dlg.maximum() != total:
                dlg.setRange(0, total)
            dlg.setValue(min(done, total))
            label = f"Batch pre-labeling (EAST)... {min(done, total)}/{total}"
            if current > 0 and done < total:
                label += f" (processing {min(current, total)}/{total})"
            dlg.setLabelText(label)
        else:
            dlg.setRange(0, 0)
            label = "Batch pre-labeling (EAST)..."
            if current > 0:
                label += f" (processing {current})"
            dlg.setLabelText(label)

    def _on_east_batch_done(self, ok: bool, output_dir: str):
        self._close_progress_dialog(getattr(self, "_east_batch_progress", None))
        self._east_batch_progress = None
        if ok:
            self._set_status(f"Batch pre-labeling done. Outputs in {output_dir}")
        else:
            log_hint = os.path.join(output_dir, "pred_east_batch.log")
            if os.path.isfile(log_hint):
                self._set_status(f"Batch pre-labeling failed; see log: {log_hint}")
            else:
                self._set_status("Batch pre-labeling failed.")

    def _on_asot_remap_build_progress(self, line: str):
        self._set_status(line)
        dlg = getattr(self, "_asot_remap_progress", None)
        if dlg:
            dlg.setLabelText(
                "Building ASOT label remap...\n" + str(line or "").strip()
            )

    def _on_asot_remap_build_done(self, ok: bool, output_json: str):
        self._close_progress_dialog(getattr(self, "_asot_remap_progress", None))
        self._asot_remap_progress = None
        if ok:
            self._set_status(f"ASOT label remap ready: {output_json}")
        else:
            log_hint = os.path.join(
                os.path.dirname(str(output_json or "")),
                "asot_label_remap_build.log",
            )
            if os.path.isfile(log_hint):
                self._set_status(f"ASOT label remap build failed; see log: {log_hint}")
            else:
                self._set_status("ASOT label remap build failed.")

    def on_click_batch_prelabel(self):
        # Keep the main UI centered on the current EAST workflow.
        # Legacy FACT batch tooling remains available internally for compatibility.
        self.on_click_east_batch()

    def _run_east_refinement(self, features_dir: str):
        if not features_dir or not os.path.isfile(os.path.join(features_dir, "features.npy")):
            QMessageBox.warning(
                self, "EAST", "EAST refinement requires features.npy for the current video."
            )
            return
        if not self._ensure_east_model_paths():
            return
        label_txt = self._prepare_east_labels_file(features_dir)
        if not label_txt:
            return
        meta = self._load_feature_meta(features_dir)
        feature_backbone_version = str(
            meta.get("backbone")
            or meta.get("source")
            or meta.get("feature_backbone_version")
            or "east_backbone"
        ).strip()
        video_id = (
            str(getattr(self, "current_video_id", "") or "").strip()
            or os.path.splitext(os.path.basename(str(getattr(self, "video_path", "") or "")))[0]
            or os.path.basename(os.path.abspath(features_dir))
        )
        log_path = os.path.join(features_dir, "pred_east_infer.log")
        self._last_autolabel_log_path = log_path
        self.btn_auto_label.setEnabled(False)
        self._set_status(self._east_refine_launch_summary())
        self._infer_thread_east = QThread(self)
        self._infer_worker_east = EASTInferWorker(
            features_dir=features_dir,
            video_id=video_id,
            video_path=str(getattr(self, "video_path", "") or ""),
            labels_txt=label_txt,
            ckpt=self._east_ckpt,
            cfg=self._east_cfg,
            text_bank_version=self._east_text_bank_version,
            feature_backbone_version=feature_backbone_version,
            category_adapter=self._effective_east_shared_adapter_path(),
            out_prefix="pred_east",
            log_path=log_path,
        )
        self._infer_worker_east.moveToThread(self._infer_thread_east)
        self._infer_thread_east.started.connect(self._infer_worker_east.run)
        self._infer_worker_east.progress.connect(self._set_status)
        self._infer_worker_east.done.connect(self._on_east_infer_done)
        self._infer_worker_east.done.connect(self._infer_thread_east.quit)
        self._infer_thread_east.finished.connect(self._infer_worker_east.deleteLater)
        self._infer_thread_east.start()

    def _on_east_infer_done(self, txt_path: str, json_path: str):
        try:
            self.btn_auto_label.setEnabled(True)
        except Exception:
            pass
        if txt_path:
            self.currentEastFeatureDir = os.path.dirname(txt_path)
        self._on_infer_done(txt_path, json_path, "EAST")

    def on_click_auto_label(self):
        if not getattr(self, "video_path", None):
            QMessageBox.information(
                self, "No video", "Load a video before running EAST refinement."
            )
            return
        if self._east_masked_refresh_running:
            self._set_status("EAST is finishing a background refresh for unlocked regions. Please wait.")
            return
        if not self._ensure_east_runtime_ready(
            feature_name="Batch pre-labeling (EAST)",
            unavailable_status=(
                "Batch pre-labeling is unavailable until a full EAST-main checkout and the EAST runtime dependencies are installed."
            ),
            show_dialog=True,
        ):
            return
        if not self._ensure_east_model_paths():
            return
        features_dir = getattr(self, "currentEastFeatureDir", None)
        need_extract = (
            (not features_dir)
            or (not os.path.isdir(features_dir))
            or (not os.path.isfile(os.path.join(features_dir, "features.npy")))
        )
        if need_extract:
            feat_dir = self._ensure_east_features_for_current_video()
            if not feat_dir:
                return
            self.currentEastFeatureDir = feat_dir
            self._start_feature_extraction_and_east(feat_dir)
            return
        if not self._east_features_compatible(features_dir):
            backbone_name = self._east_feature_meta_backbone(features_dir) or "unknown"
            ret = QMessageBox.question(
                self,
                "Re-extract EAST features",
                "Current refinement features are not EAST-compatible.\n\n"
                f"Detected backbone: {backbone_name}\n\n"
                "Re-extract EAST backbone features for the current video now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if ret != QMessageBox.Yes:
                self._set_status("EAST refinement cancelled.")
                return
            self._start_feature_extraction_and_east(features_dir)
            return
        self._run_east_refinement(features_dir)

    def _build_asot_label_remap(self):
        root_dir = QFileDialog.getExistingDirectory(
            self,
            "Select calibration dataset root (groundTruth + videos)",
            os.getcwd(),
        )
        if not root_dir:
            return
        if not os.path.isdir(os.path.join(root_dir, "groundTruth")):
            QMessageBox.warning(
                self,
                "ASOT label remap",
                "Selected folder does not contain a groundTruth/ directory.",
            )
            return
        default_output = os.path.join(root_dir, "asot_label_remap.json")
        output_json, _ = QFileDialog.getSaveFileName(
            self,
            "Save ASOT Label Remap",
            default_output,
            "JSON Files (*.json)",
        )
        if not output_json:
            return
        class_txt = self._current_label_bank_source_path()
        if not class_txt:
            class_txt = self._write_label_bank_txt(
                os.path.dirname(output_json),
                file_name="asot_semantic_labels.txt",
            )
        if not class_txt:
            class_txt = resolve_label_source(
                extra_dirs=[root_dir],
                repo_root=self._root_dir,
            )
        if not class_txt:
            class_txt, _ = QFileDialog.getOpenFileName(
                self,
                "Choose semantic label bank (TXT)",
                "",
                "Text Files (*.txt)",
            )
        if not class_txt:
            QMessageBox.warning(
                self,
                "Missing labels",
                "ASOT label remap builder requires a semantic label bank.",
            )
            return
        tool_path = os.path.join(self._root_dir, "tools", "build_asot_label_remap.py")
        if not os.path.isfile(tool_path):
            QMessageBox.warning(
                self,
                "Missing script",
                f"ASOT remap builder not found: {tool_path}",
            )
            return
        feature_search_roots: List[str] = []
        for cand in (
            getattr(self, "currentFeatureDir", ""),
            getattr(self, "currentEastFeatureDir", ""),
            os.path.join(root_dir, "videos"),
        ):
            path = os.path.abspath(os.path.expanduser(str(cand or "").strip()))
            if path and os.path.exists(path) and path not in feature_search_roots:
                feature_search_roots.append(path)
        log_path = os.path.join(
            os.path.dirname(output_json),
            "asot_label_remap_build.log",
        )
        self._last_autolabel_log_path = log_path
        self._set_status("Building ASOT label remap in background...")
        self._close_progress_dialog(getattr(self, "_asot_remap_progress", None))
        self._asot_remap_progress = self._open_progress_dialog(
            "ASOT Label Remap",
            "Building ASOT label remap...",
            None,
        )
        self._asot_remap_thread = QThread(self)
        self._asot_remap_worker = ASOTRemapBuildWorker(
            [root_dir],
            output_json=output_json,
            class_names=class_txt,
            feature_search_roots=feature_search_roots,
            pred_prefix="pred_asot",
            tool_path=tool_path,
            log_path=log_path,
        )
        self._asot_remap_worker.moveToThread(self._asot_remap_thread)
        self._asot_remap_thread.started.connect(self._asot_remap_worker.run)
        self._asot_remap_worker.progress.connect(self._on_asot_remap_build_progress)
        self._asot_remap_worker.done.connect(self._on_asot_remap_build_done)
        self._asot_remap_worker.done.connect(self._asot_remap_thread.quit)
        self._asot_remap_thread.finished.connect(self._asot_remap_worker.deleteLater)
        self._asot_remap_thread.start()

    def on_click_east_batch(self):
        if not self._ensure_east_runtime_ready(
            feature_name="EAST refinement",
            unavailable_status=(
                "EAST refinement is unavailable until a full EAST-main checkout and the EAST runtime dependencies are installed."
            ),
            show_dialog=True,
        ):
            return
        if not self._ensure_east_model_paths():
            return
        video_dir = QFileDialog.getExistingDirectory(
            self, "Select unlabeled video directory"
        )
        if not video_dir:
            return
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select output directory for EAST predictions"
        )
        if not output_dir:
            return
        label_txt = self._resolve_action_label_bank_path(
            output_dir,
            generated_file_name="east_batch_labels.txt",
            dialog_title="Choose label bank (TXT)",
            missing_title="Missing labels",
            missing_text="Batch pre-labeling requires a label bank or current action labels.",
        )
        if not label_txt:
            return
        tool_path = os.path.join(self._root_dir, "tools", "east_batch_infer_adapter.py")
        if not os.path.isfile(tool_path):
            QMessageBox.warning(
                self, "Missing script", f"EAST batch script not found: {tool_path}"
            )
            return

        self._set_status("Starting batch pre-labeling with EAST...")
        total_videos = self._count_videos_in_dir(video_dir)
        self._close_progress_dialog(getattr(self, "_east_batch_progress", None))
        label = "Batch pre-labeling (EAST)..."
        if total_videos > 0:
            label = f"Batch pre-labeling (EAST)... 0/{total_videos}"
        self._east_batch_progress = self._open_progress_dialog(
            "Batch Pre-labeling",
            label,
            total_videos if total_videos > 0 else None,
        )
        self._east_batch_done = 0
        self._east_batch_current = 0
        self._east_batch_total = total_videos if total_videos > 0 else None
        log_path = os.path.join(output_dir, "pred_east_batch.log")
        self._last_autolabel_log_path = log_path
        self._east_batch_thread = QThread(self)
        self._east_batch_worker = EASTBatchWorker(
            video_dir=video_dir,
            output_dir=output_dir,
            ckpt=self._east_ckpt,
            cfg=self._east_cfg,
            labels_txt=label_txt,
            category_adapter=self._effective_east_shared_adapter_path(),
            text_bank_version=self._east_text_bank_version,
            feature_backbone_version="east_backbone",
            tool_path=tool_path,
            log_path=log_path,
        )
        self._east_batch_worker.moveToThread(self._east_batch_thread)
        self._east_batch_thread.started.connect(self._east_batch_worker.run)
        self._east_batch_worker.progress.connect(self._on_east_batch_progress)
        self._east_batch_worker.done.connect(self._on_east_batch_done)
        self._east_batch_worker.done.connect(self._east_batch_thread.quit)
        self._east_batch_thread.finished.connect(self._east_batch_worker.deleteLater)
        self._east_batch_thread.start()

    def on_click_fact_batch(self):
        if not self._ensure_python_modules_available(
            ("torch",),
            feature_name="FACT batch labeling",
            install_hint=(
                "Install the optional FACT dependencies only if you need this workflow, "
                "for example: pip install torch torchvision"
            ),
            unavailable_status=(
                "FACT batch labeling is unavailable until the optional dependencies are installed."
            ),
        ):
            return
        video_dir = QFileDialog.getExistingDirectory(
            self, "Select unlabeled video directory"
        )
        if not video_dir:
            return
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select output directory for FACT predictions"
        )
        if not output_dir:
            return

        default_fact_repo = os.path.abspath(
            os.path.join(
                self._root_dir,
                "..",
                "Action-Segmentation-Tool_yinqian",
                "Action-Segmentation-Tool_v1.2",
                "CVPR2024-FACT-main",
            )
        )
        fact_repo = default_fact_repo if os.path.isdir(default_fact_repo) else ""
        if not fact_repo:
            fact_repo = QFileDialog.getExistingDirectory(self, "Select FACT repository")
            if not fact_repo:
                return

        ckpt = os.path.join(fact_repo, "runs/learningcell_front/split1-weight.pth")
        if not os.path.isfile(ckpt):
            ckpt, _ = QFileDialog.getOpenFileName(
                self, "Choose FACT checkpoint", fact_repo, "PyTorch Models (*.pth *.pt)"
            )
            if not ckpt:
                return
        fact_cfg = os.path.join(fact_repo, "fact/configs/learningcell_front.yaml")
        if not os.path.isfile(fact_cfg):
            fact_cfg, _ = QFileDialog.getOpenFileName(
                self, "Choose FACT config", fact_repo, "YAML Files (*.yaml *.yml)"
            )
            if not fact_cfg:
                return

        class_txt = None
        default_classes = os.path.join(
            self._root_dir, "external", "learningcell_front", "mapping", "mapping.txt"
        )
        if os.path.isfile(default_classes):
            class_txt = default_classes
        else:
            fallback = os.path.join(
                self._root_dir, "external", "action_seg_ot", "class_names.txt"
            )
            if os.path.isfile(fallback):
                class_txt = fallback
        if not class_txt:
            class_txt, _ = QFileDialog.getOpenFileName(
                self, "Choose class mapping (TXT)", "", "Text Files (*.txt)"
            )
        if not class_txt:
            QMessageBox.warning(
                self,
                "Missing mapping",
                "Class mapping file is required for FACT batch inference.",
            )
            return

        tool_path = os.path.join(self._root_dir, "tools", "fact_batch_infer.py")
        if not os.path.isfile(tool_path):
            QMessageBox.warning(
                self, "Missing script", f"FACT batch script not found: {tool_path}"
            )
            return

        self._set_status("Starting FACT batch labeling...")
        total_videos = self._count_videos_in_dir(video_dir)
        self._close_progress_dialog(getattr(self, "_fact_batch_progress", None))
        label = "FACT batch labeling..."
        if total_videos > 0:
            label = f"FACT batch labeling... 0/{total_videos}"
        self._fact_batch_progress = self._open_progress_dialog(
            "FACT Batch Labeling",
            label,
            total_videos if total_videos > 0 else None,
        )
        self._fact_batch_done = 0
        self._fact_batch_current = 0
        self._fact_batch_total = total_videos if total_videos > 0 else None
        log_path = os.path.join(output_dir, "pred_fact_batch.log")
        self._last_autolabel_log_path = log_path
        self._fact_batch_thread = QThread(self)
        self._fact_batch_worker = FactBatchWorker(
            video_dir,
            output_dir,
            fact_repo,
            ckpt,
            fact_cfg,
            tool_path=tool_path,
            class_names=class_txt,
            log_path=log_path,
        )
        self._fact_batch_worker.moveToThread(self._fact_batch_thread)
        self._fact_batch_thread.started.connect(self._fact_batch_worker.run)
        self._fact_batch_worker.progress.connect(self._on_fact_batch_progress)
        self._fact_batch_worker.done.connect(self._on_fact_batch_done)
        self._fact_batch_worker.done.connect(self._fact_batch_thread.quit)
        self._fact_batch_thread.finished.connect(self._fact_batch_worker.deleteLater)
        self._fact_batch_thread.start()

    def on_click_auto_label_asot(self):
        if not self._ensure_python_modules_available(
            ("torch",),
            feature_name="ASOT pre-labeling",
            install_hint=(
                "Install the optional ASOT dependencies only if you need this workflow, "
                "for example: pip install torch torchvision"
            ),
            unavailable_status=(
                "ASOT pre-labeling is unavailable until the optional dependencies are installed."
            ),
        ):
            return
        features_dir = getattr(self, "currentFeatureDir", None)
        need_extract = (
            (not features_dir)
            or (not os.path.isdir(features_dir))
            or (not os.path.isfile(os.path.join(features_dir, "features.npy")))
        )
        if need_extract:
            feat_dir = self._ensure_features_for_current_video()
            if feat_dir and not os.path.isfile(os.path.join(feat_dir, "features.npy")):
                # kick off background extraction then resume ASOT when done
                self._start_feature_extraction_and_asot(feat_dir)
                return
            if feat_dir:
                features_dir = feat_dir
                self.currentFeatureDir = feat_dir
            else:
                sel = QFileDialog.getExistingDirectory(
                    self, "Select feature directory (containing features.npy)"
                )
                if not sel:
                    self._set_status("ASOT pre-labeling cancelled.")
                    return
                features_dir = sel
                self.currentFeatureDir = sel

        default_ckpt = self._default_asot_ckpt()
        # let user pick/confirm ckpt; preselect default if available
        init_dir = (
            os.path.dirname(default_ckpt)
            if default_ckpt
            else os.path.join(self._root_dir, "external", "action_seg_ot")
        )
        ckpt, _ = QFileDialog.getOpenFileName(
            self,
            f"Choose ASOT checkpoint (default: {os.path.basename(default_ckpt) if default_ckpt else 'none'})",
            init_dir,
            "PyTorch Models (*.pth *.pt *.ckpt)",
        )
        if not ckpt:
            ckpt = default_ckpt
        if not ckpt or not os.path.isfile(ckpt):
            QMessageBox.warning(
                self, "Missing checkpoint", "No ASOT checkpoint selected or found."
            )
            return

        class_txt = self._resolve_action_label_bank_path(
            features_dir,
            generated_file_name="asot_labels.txt",
            dialog_title="Choose semantic label bank (TXT)",
            missing_title="Missing labels",
            missing_text="ASOT pre-labeling requires a semantic label bank or current action labels.",
        )
        if not class_txt:
            return

        if not os.path.isfile(os.path.join(features_dir, "features.npy")):
            QMessageBox.warning(
                self,
                "Missing file",
                "features.npy not found in the selected directory.",
            )
            return

        # If the cached features are much shorter than the current video span, offer re-extraction.
        try:
            view = self.views[self.active_view_idx] if self.views else {}
            span = int(view.get("end", 0)) - int(view.get("start", 0)) + 1
        except Exception:
            span = None
        feat_len = self._feature_frame_count(features_dir)
        if feat_len is not None and span and feat_len < span * 0.9:
            ret = QMessageBox.question(
                self,
                "ASOT features look too short",
                (
                    f"Current features.npy contains only {feat_len} frames, while the "
                    f"video span is about {span} frames.\nRe-extract features and overwrite?"
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if ret == QMessageBox.Yes:
                self._start_feature_extraction_and_asot(features_dir)
                return
            else:
                self._set_status(
                    f"Continuing with the existing features ({feat_len} frames) for ASOT pre-labeling."
                )

        input_layout = self._feature_layout_hint(features_dir)

        tool_path = os.path.join(self._root_dir, "tools", "asot_full_infer_adapter.py")
        if not os.path.isfile(tool_path):
            QMessageBox.warning(
                self, "Missing script", f"ASOT adapter script not found: {tool_path}"
            )
            return

        log_path = os.path.join(features_dir, "pred_asot_infer.log")
        self._last_autolabel_log_path = log_path
        self._set_status("Starting ASOT pre-labeling...")
        self._infer_thread_asot = QThread(self)
        self._infer_worker_asot = ASOTInferWorker(
            features_dir,
            ckpt,
            class_names=class_txt,
            smooth_k=3,
            standardize=False,
            allow_standardize=False,
            out_prefix="pred_asot",
            tool_path=tool_path,
            log_path=log_path,
            extra_args=["--input_layout", input_layout, "--min_seg_len", "30"],
        )
        self._infer_worker_asot.moveToThread(self._infer_thread_asot)
        self._infer_thread_asot.started.connect(self._infer_worker_asot.run)
        self._infer_worker_asot.progress.connect(self._set_status)
        self._infer_worker_asot.done.connect(
            lambda t, j: self._on_infer_done(t, j, "ASOT")
        )
        self._infer_worker_asot.done.connect(self._infer_thread_asot.quit)
        self._infer_thread_asot.finished.connect(self._infer_worker_asot.deleteLater)
        self._infer_thread_asot.start()

    def _start_feature_extraction(self, feat_dir: str, next_task: str, backbone: str = ""):
        if not getattr(self, "video_path", None):
            QMessageBox.warning(
                self, "No video", "Load a video before extracting features."
            )
            return
        if not self._ensure_feature_extractor_available(show_dialog=True):
            return
        self._feature_followup_task = str(next_task or "").strip().lower()
        self._feature_followup_backbone = str(backbone or "").strip()
        self._last_feature_error_message = ""
        try:
            self.btn_auto_label_asot.setEnabled(False)
        except Exception:
            pass
        try:
            if self._feature_followup_task == "east":
                self.btn_auto_label.setEnabled(False)
        except Exception:
            pass
        status = "Extracting features in background..."
        if self._feature_followup_task == "east":
            status = "Extracting EAST backbone features in background..."
        self._set_status(status)
        self._close_progress_dialog(getattr(self, "_feat_progress", None))
        self._feat_progress = self._open_progress_dialog(
            "Feature Extraction", status, None
        )
        self._feat_thread = QThread(self)
        self._feat_worker = FeatureExtractWorker(
            video_path=self.video_path,
            features_dir=feat_dir,
            batch_size=128,
            frame_stride=1,
            use_fp16=True,
            backbone=self._feature_followup_backbone or None,
            east_ckpt=(
                self._east_ckpt
                if self._feature_followup_task in {"east", "east_bootstrap"}
                else ""
            ),
            east_cfg=(
                self._east_cfg
                if self._feature_followup_task in {"east", "east_bootstrap"}
                else ""
            ),
        )
        self._feat_worker.moveToThread(self._feat_thread)
        self._feat_thread.started.connect(self._feat_worker.run)
        self._feat_worker.progress.connect(self._on_feat_progress_message)
        self._feat_worker.progress_value.connect(self._on_feat_progress)
        self._feat_worker.done.connect(self._on_feat_done)
        self._feat_worker.done.connect(self._feat_thread.quit)
        self._feat_thread.finished.connect(self._feat_worker.deleteLater)
        self._feat_thread.start()

    def _start_feature_extraction_and_asot(self, feat_dir: str):
        self._start_feature_extraction(feat_dir, next_task="asot")

    def _start_feature_extraction_and_east(self, feat_dir: str):
        self._start_feature_extraction(
            feat_dir, next_task="east", backbone="east_backbone"
        )

    def _on_feat_done(self, feat_dir, ok: bool):
        try:
            self.btn_auto_label_asot.setEnabled(True)
        except Exception:
            pass
        try:
            self.btn_auto_label.setEnabled(True)
        except Exception:
            pass
        self._close_progress_dialog(getattr(self, "_feat_progress", None))
        self._feat_progress = None
        task = str(getattr(self, "_feature_followup_task", "") or "").strip().lower()
        self._feature_followup_task = ""
        self._feature_followup_backbone = ""
        if not ok or not feat_dir:
            detail = str(getattr(self, "_last_feature_error_message", "") or "").strip()
            if detail:
                self._set_status(f"Feature extraction failed: {detail}")
                QMessageBox.warning(self, "Feature extraction failed", detail)
            else:
                self._set_status("Feature extraction failed.")
            return
        self._last_feature_error_message = ""
        self._boundary_snap_cache = {}
        if task == "east":
            self.currentEastFeatureDir = feat_dir
            self._set_status("Feature extraction finished. Starting EAST refinement...")
            self._run_east_refinement(feat_dir)
            return
        if task == "east_bootstrap":
            self.currentEastFeatureDir = feat_dir
            records = self._rebuild_confirmed_correction_records_for_active_view()
            persisted = self._write_correction_record_buffer(records, force=True)
            self._update_east_runtime_meta(
                feat_dir,
                runtime_origin="bootstrapped",
                bootstrap_source=str(
                    self._active_label_baseline_source() or "CONFIRMED_SUPERVISION"
                ).strip().upper(),
                bootstrap_pending=False,
                bootstrap_completed=bool(persisted),
                bootstrap_confirmed_count=int(len(records)),
                bootstrap_refresh_deferred=True,
            )
            if persisted:
                self._set_status(
                    "EAST runtime initialized from confirmed supervision. Starting online update..."
                )
                self._schedule_east_online_update(feat_dir)
            else:
                self._set_status(
                    "EAST features are ready, but confirmed supervision could not be persisted."
                )
            return
        if task == "east_finalize":
            self.currentEastFeatureDir = feat_dir
            records = list(getattr(self, "_pending_finalized_records", []) or [])
            self._pending_finalized_records = []
            persisted = self._write_finalized_record_buffer(records, force=True)
            self._update_east_runtime_meta(
                feat_dir,
                video_finalized=bool(persisted),
                finalized_pending=False,
                finalized_record_count=int(len(records)),
                finalized_at=(
                    datetime.utcnow().isoformat(timespec="seconds") + "Z"
                    if persisted
                    else ""
                ),
            )
            if persisted:
                self._set_status(
                    "Current video finalized for EAST offline supervision. Starting online update..."
                )
                self._schedule_east_online_update(feat_dir)
            else:
                self._set_status(
                    "EAST features are ready, but finalized supervision could not be persisted."
                )
            return
        self.currentFeatureDir = feat_dir
        self._set_status("Feature extraction finished. Starting ASOT pre-labeling...")
        self.on_click_auto_label_asot()

    # ----- video events -----
    def _on_player_frame_advanced(self, frame: int):
        # update controls for active view
        try:
            pv = self.views[self.active_view_idx]["player"]
            if pv is self.player:
                self._set_frame_controls(frame)
        except Exception:
            pass
        if self.interaction_mode == "assisted" and self._assisted_loop_range:
            start, end = self._assisted_loop_range
            if frame >= end:
                try:
                    self._sync_views_to_frame(start, preview_only=False)
                    self._sync_other_views(start)
                    self.timeline.set_current_frame(start, follow=True)
                    self._update_overlay_for_frame(start)
                except Exception:
                    pass
                return
        if self.extra_mode:
            self._append_extra_progress(frame)
        self._sync_other_views(frame)
        follow = self._timeline_auto_follow or self._extra_force_follow
        self.timeline.set_current_frame(frame, follow=follow)
        self._update_transcript_workspace_for_frame(frame)
        self._update_overlay_for_frame(frame)
        self._psr_update_component_panel(frame)
        self._log("frame_advanced", frame=frame)

    def _on_player_clicked(self, view_idx: int, frame: int):
        """User clicked on a video view: make it primary, re-enable auto-follow, and align timeline."""
        if view_idx != self.active_view_idx:
            self._set_primary_view(view_idx)
        self._timeline_auto_follow = True
        if self.interaction_mode == "assisted":
            self._activate_assisted_at_frame(frame)
        if self.extra_mode:
            self._split_extra_at(frame)
        self._sync_views_to_frame(frame, preview_only=False)
        self.timeline.set_current_frame(frame, follow=True)
        self._update_overlay_for_frame(frame)
        self._psr_update_component_panel(frame)
        self._log(
            "click_video",
            view=(
                self.views[view_idx].get("name", "")
                if 0 <= view_idx < len(self.views)
                else ""
            ),
            frame=frame,
        )

    def _on_slider_changed(self, pos: int):
        if self.player.cap:
            self._sync_views_to_frame(pos, preview_only=False)
            if self.extra_mode:
                self._append_extra_progress(pos)
            self._update_overlay_for_frame(pos)
            self.timeline.set_current_frame(pos, follow=self._timeline_auto_follow)
            self.timeline.set_current_hits(self._hit_names_for_frame(pos))
            self._update_transcript_workspace_for_frame(pos)
            self._psr_update_component_panel(pos)
            self._log("seek_slider", target=pos)

    def _seek_relative(self, secs: int):
        if not self.player.cap:
            return
        target = max(
            self.player.crop_start,
            min(
                self.player.current_frame + int(round(secs * self.player.frame_rate)),
                self.player.crop_end,
            ),
        )
        self._sync_views_to_frame(target, preview_only=False)
        self.timeline.set_current_frame(target, follow=self._timeline_auto_follow)
        self.timeline.set_current_hits(self._hit_names_for_frame(target))
        self._log("seek_relative", seconds=secs, target=target)

    def _on_speed_changed(self, text: str):
        try:
            v = float(str(text).lower().replace("x", ""))
        except Exception:
            return
        try:
            self.player.set_playback_speed(v)
        except Exception:
            pass
        self._log("set_speed", speed=v)

    def _jump_to_spin(self):
        if not self.player.cap:
            return
        t = self.spin_jump.value()
        if t < self.player.crop_start or t > self.player.crop_end:
            QMessageBox.information(
                self,
                "Info",
                f"Jump within [{self.player.crop_start}, {self.player.crop_end}] only.",
            )
            t = max(self.player.crop_start, min(t, self.player.crop_end))
            self.spin_jump.setValue(t)
        self._sync_views_to_frame(t, preview_only=False)
        self._update_overlay_for_frame(t)
        self.timeline.set_current_frame(t, follow=self._timeline_auto_follow)
        self.timeline.set_current_hits(self._hit_names_for_frame(t))
        self._log("jump_to_frame", target=t)

    # ----- actions -----
    def _on_action_selected(self, idx: int):
        text = self.combo_actions.itemText(idx)
        self.combo_actions.setCurrentIndex(0)
        if not text or text == "Choose action...":
            return
        if text in set(getattr(self, "_action_section_headers", set())):
            return

        if text.startswith("Open Session"):
            self._open_session_wizard()

        elif text.startswith("Load Video"):
            # handle unsaved changes first
            if not self._prompt_save_if_dirty(context="loading a new video"):
                return
            fd = QFileDialog(self)
            fd.setFileMode(QFileDialog.ExistingFile)
            fd.setNameFilter("Video Files (*.mp4 *.avi *.mov *.mkv)")
            if fd.exec_():
                path = fd.selectedFiles()[0]
                self._load_primary_video(path)

        elif text.startswith("Crop"):
            if not self.player.cap:
                QMessageBox.information(self, "Info", "Load a video first.")
                return
            s, ok1 = QInputDialog.getInt(
                self,
                "Crop start",
                f"Start (0..{self.player.frame_count - 1}):",
                value=self.player.current_frame,
                min=0,
                max=max(0, self.player.frame_count - 1),
            )
            if not ok1:
                return
            e, ok2 = QInputDialog.getInt(
                self,
                "Crop end",
                f"End (>=start..{self.player.frame_count - 1}):",
                value=min(
                    s + int(self.player.frame_rate * 10), self.player.frame_count - 1
                ),
                min=s,
                max=max(0, self.player.frame_count - 1),
            )
            if not ok2:
                return
            self.player.set_crop(s, e)
            self.slider.setMinimum(self.player.crop_start)
            self.slider.setMaximum(self.player.crop_end)
            self.slider.setValue(self.player.crop_start)
            self.spin_jump.setMinimum(self.player.crop_start)
            self.spin_jump.setMaximum(self.player.crop_end)
            self.spin_jump.setValue(self.player.crop_start)
            self._log("crop_set", start=s, end=e)

        elif text.startswith("Import JSON (selected views)"):
            self._import_json_for_selected_views()

        elif text.startswith("Export JSON (selected views to folders)"):
            self._export_all_views_json_to_folders()

        elif text.startswith("Export JSON"):
            self._export_annotations_with_adapter_auto()

        elif text.startswith("Export to Seed Dataset"):
            self._export_seed_dataset()

        elif text.startswith("Import label map"):
            self._import_label_map_txt()

        elif text.startswith("Export label map"):
            self._export_label_map_txt()

        elif text.startswith("ASOT: Build Label Remap"):
            self._build_asot_label_remap()

        elif text.startswith("Batch Pre-label"):
            self.on_click_batch_prelabel()

        elif text.startswith("EAST Setup"):
            self._open_east_setup_dialog()

        elif text.startswith("EAST: Finalize Current Video"):
            self._finalize_current_video_for_east()

        elif text.startswith("EAST: Inspect Runtime Assets"):
            self._inspect_east_runtime_assets()

        elif text.startswith("EAST: Export Runtime Report"):
            self._export_east_runtime_report()

        elif text.startswith("EAST: Export Shared Adapter"):
            self._export_current_east_shared_adapter()

        elif text.startswith("EAST: Select Shared Adapter"):
            self._select_east_shared_adapter()

        elif text.startswith("EAST: Clear Shared Adapter"):
            self._clear_east_shared_adapter()

        elif text.startswith("EAST: Consolidate Shared Adapter"):
            self._consolidate_east_shared_adapter()

        elif text.startswith("Assembly State: Load Components"):
            self._load_psr_components()

        elif text.startswith("Assembly State: Save Components"):
            self._save_psr_components()

        elif text.startswith("Assembly State: Load Rules"):
            self._load_psr_rules()

        elif text.startswith("Assembly State: Load State JSON"):
            self._load_psr_asr_json()

        elif text.startswith("Assembly State: Export State JSON"):
            self._export_psr_asr_asd("ASR")

        elif text.startswith("Import Review Log"):
            self._import_validation_log()

        elif text.startswith("Open Review Panel"):
            self._open_review_panel()

        elif text.startswith("Transcript: Open Workspace"):
            self._open_transcript_workspace()

        elif text.startswith("Transcript Audio: Attach External Audio"):

            if not getattr(self, "video_path", None):
                QMessageBox.information(self, "Info", "Load a video first.")
                return

            path, _ = QFileDialog.getOpenFileName(
                self,
                "Choose Audio",
                "",
                "Audio Files (*.wav *.mp3 *.m4a *.aac *.flac);;All Files (*)",
            )

            if not path:
                return

            self.player.attach_audio_file(path)

            self.player.set_audio_enabled(True)

            self.player.set_audio_offset_ms(0)

            self._set_status(f"External audio attached: {os.path.basename(path)}")
            self._log("attach_audio", path=path)

        elif text.startswith("Transcript Audio: Set Audio Offset"):

            if not self.player._audio_enabled:
                QMessageBox.information(self, "Info", "Attach audio first.")
                return

            cur = getattr(self.player, "_audio_offset_ms", 0)

            val, ok = QInputDialog.getInt(
                self,
                "Audio Offset (ms)",
                "Positive = delay audio; Negative = advance audio",
                value=int(cur),
                min=-600000,
                max=600000,
                step=10,
            )

            if ok:
                self.player.set_audio_offset_ms(val)

                self._set_status(f"Audio offset = {val} ms")
                self._log("audio_offset", value_ms=val)

        elif text.startswith("Transcript: Quick Generate / Import"):
            if not self._open_transcript_workspace():
                return
            self._run_external_asr(prompt_for_label=True, reveal_panel=True)

    def _on_interaction_selected(self, idx: int):
        text = self.combo_interaction.itemText(idx)
        self.combo_interaction.setCurrentIndex(0)

        if text.startswith("Manual Segmentation"):
            self.on_extra_clicked()
            return

        if text.startswith("Assisted Review"):
            try:
                self.btn_assisted.setChecked(not self.btn_assisted.isChecked())
            except Exception:
                pass
            self.on_assisted_clicked()
            return

        if text.startswith("Exit Interaction"):
            if self.extra_mode:
                self.exit_extra_mode()
            if self.interaction_mode == "assisted":
                self._exit_assisted_mode()

    def _after_video_loaded(self):
        self._set_controls_enabled(True)
        self.slider.setMinimum(self.player.crop_start)
        self.slider.setMaximum(self.player.crop_end)
        self.slider.setValue(self.player.current_frame)
        self.spin_jump.setMinimum(self.player.crop_start)
        self.spin_jump.setMaximum(self.player.crop_end)
        self.spin_jump.setValue(self.player.current_frame)

        self.timeline.view_start = 0
        self._rebuild_timeline_sources()
        self._timeline_auto_follow = True
        self.timeline.set_current_frame(self.player.current_frame, follow=True)
        self.timeline.set_current_hits(
            self._hit_names_for_frame(self.player.current_frame)
        )
        self._update_gap_indicator()

        if getattr(self, "video_path", None):
            base = os.path.basename(self.video_path)
            self.current_video_name = base
            self.current_video_id = os.path.splitext(base)[0]

        self._set_status(
            f"Loaded: {self.player.frame_count} frames @ {self.player.frame_rate} FPS"
        )
        self._refresh_asr_panels()
        self._update_overlay_for_frame(self.player.current_frame)
        self._psr_mark_dirty()
        self._psr_update_component_panel(self.player.current_frame)

    def closeEvent(self, event):
        ok_gaps, had_gaps = self._check_unlabeled_gaps(context="close")
        if not ok_gaps:
            event.ignore()
            return
        if not self._prompt_save_if_dirty(context="quitting", force_prompt=had_gaps):
            event.ignore()
            return
        event.accept()

    # ----- labels -----
    def _on_label_added(self, lb: LabelDef):
        self._refresh_fine_label_decomposition(refresh_panel=False)
        self._rebuild_timeline_sources()
        self._dirty = True
        self._psr_mark_dirty()
        self._log("label_add", label=lb.name, id=getattr(lb, "id", None))

    def _on_label_removed(self, idx: int):
        if 0 <= idx < len(self.labels):
            name = self.labels[idx].name
            self.store.remove_all_of_label(name)
            if getattr(self, "prelabel_store", None):
                try:
                    self.prelabel_store.remove_all_of_label(name)
                except Exception:
                    pass
            self.label_entity_map.pop(name, None)
            self._label_prototypes.pop(name, None)
            self._label_proto_counts.pop(name, None)
            self._knn_memory = [(g, y) for (g, y) in self._knn_memory if y != name]
            self._log("label_remove", label=name)
        self._refresh_fine_label_decomposition(refresh_panel=False)
        QTimer.singleShot(0, self._rebuild_timeline_sources)
        self._on_store_changed()
        self._dirty = True
        self._psr_mark_dirty()

    def _on_label_renamed(self, old_name: str, new_name: str):
        if old_name == new_name:
            return
        self.store.rename_label(old_name, new_name)
        if getattr(self, "prelabel_store", None):
            try:
                self.prelabel_store.rename_label(old_name, new_name)
            except Exception:
                pass
        if getattr(self, "extra_store", None):
            try:
                self.extra_store.rename_label(old_name, new_name)
            except Exception:
                pass
        for st in self.entity_stores.values():
            try:
                st.rename_label(old_name, new_name)
            except Exception:
                pass
        # migrate mapping key
        if old_name in self.label_entity_map:
            self.label_entity_map[new_name] = self.label_entity_map.pop(old_name)
        if old_name in self._label_prototypes:
            self._label_prototypes[new_name] = self._label_prototypes.pop(old_name)
        if old_name in self._label_proto_counts:
            self._label_proto_counts[new_name] = self._label_proto_counts.pop(old_name)
        if self._knn_memory:
            self._knn_memory = [
                (g, (new_name if y == old_name else y)) for (g, y) in self._knn_memory
            ]
        self._refresh_fine_label_decomposition(refresh_panel=False)
        self.timeline.rebuild_rows()
        self._on_store_changed()
        # refresh entity panel checks
        if 0 <= self.current_label_idx < len(self.labels):
            lb = self.labels[self.current_label_idx]
            self.entities_panel.set_current_label(
                lb.name, self.label_entity_map.get(lb.name, set())
            )
        self._dirty = True
        self._psr_mark_dirty()
        self._log("label_rename", old=old_name, new=new_name)

    # ----- store/timeline sync -----
    def _store_descriptor_from_store(
        self, st: Optional[AnnotationStore]
    ) -> Optional[Dict[str, Any]]:
        if st is None:
            return None
        if st is self.store:
            return {"kind": "store", "store": st}
        if st is getattr(self, "extra_store", None):
            return {"kind": "extra", "store": st}
        for ename, estore in (self.entity_stores or {}).items():
            if estore is st:
                return {"kind": "entity", "entity": ename, "store": st}
        for ename, pstore in (self.phase_stores or {}).items():
            if pstore is st:
                return {"kind": "phase", "entity": ename, "store": st}
        for ename, type_map in (self.anomaly_type_stores or {}).items():
            for tname, astore in (type_map or {}).items():
                if astore is st:
                    return {
                        "kind": "anomaly",
                        "entity": ename,
                        "anomaly_type": tname,
                        "store": st,
                    }
        return None

    def _store_descriptor_from_row(
        self, row, label: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        st = None
        try:
            resolver = getattr(row, "_store_for_label", None)
            if callable(resolver) and label:
                st = resolver(label)
        except Exception:
            st = None
        if st is None:
            st = getattr(row, "store", None)
        if st is None:
            return None
        desc = self._store_descriptor_from_store(st)
        if desc is not None:
            return desc
        return {"kind": "unknown", "store": st}

    @staticmethod
    def _descriptor_key(descriptor: Dict[str, Any]) -> Tuple[str, str, str]:
        return (
            str(descriptor.get("kind") or ""),
            str(descriptor.get("entity") or ""),
            str(descriptor.get("anomaly_type") or ""),
        )

    @staticmethod
    def _syncable_descriptor(descriptor: Optional[Dict[str, Any]]) -> bool:
        if not descriptor:
            return False
        return str(descriptor.get("kind") or "") in (
            "store",
            "entity",
            "phase",
            "anomaly",
        )

    def _phase_frame_signature(
        self, view: Dict[str, Any], entity: str, frame: int
    ) -> Tuple[Optional[str], int]:
        phase_store = (view.get("phase_stores") or {}).get(entity)
        phase_label = None
        if phase_store is not None:
            try:
                phase_label = phase_store.label_at(int(frame))
            except Exception:
                phase_label = None
        if str(phase_label or "").lower() != "anomaly":
            return phase_label, 0
        bitmask = 0
        ent_map = (view.get("anomaly_type_stores") or {}).get(entity) or {}
        for idx, name in enumerate(self._anomaly_type_names()):
            st = ent_map.get(name)
            if st is None:
                continue
            try:
                if st.label_at(int(frame)) == name:
                    bitmask |= 1 << idx
            except Exception:
                continue
        return phase_label, int(bitmask)

    def _mask_descriptor_from_row(self, row) -> Optional[Dict[str, Any]]:
        meta = getattr(row, "_group_meta", None)
        if isinstance(meta, dict):
            row_type = str(meta.get("row_type") or "")
            entity = meta.get("entity")
            if row_type == "phase" and entity:
                return {"kind": "phase", "entity": entity}
            if row_type == "action":
                if entity:
                    return {"kind": "entity", "entity": entity}
                return {"kind": "store"}
        st = getattr(row, "store", None)
        desc = self._store_descriptor_from_store(st)
        if self._syncable_descriptor(desc):
            return desc
        row_sources = getattr(row, "row_sources", None) or []
        for lb, source_store, _prefix in row_sources:
            if is_extra_label(getattr(lb, "name", "")):
                continue
            desc = self._store_descriptor_from_store(source_store)
            if self._syncable_descriptor(desc):
                return desc
        return None

    def _compute_editable_mask_spans(
        self, descriptor: Dict[str, Any], view_indices: List[int]
    ) -> List[Tuple[int, int]]:
        if not self._syncable_descriptor(descriptor):
            return []
        kind = str(descriptor.get("kind") or "")
        entity = str(descriptor.get("entity") or "")
        stores = []
        for idx in view_indices:
            if not (0 <= int(idx) < len(self.views)):
                return []
            st = self._store_for_view_descriptor(self.views[int(idx)], descriptor)
            if st is None:
                return []
            stores.append(st)
        if len(stores) <= 1:
            return []
        fc = max(1, self._get_frame_count())
        spans = []
        run_start = None
        for f in range(fc):
            if kind == "phase" and entity:
                base_label = self._phase_frame_signature(
                    self.views[int(view_indices[0])], entity, f
                )
            else:
                base_label = stores[0].label_at(f)
            same = True
            for i, st in enumerate(stores[1:], start=1):
                if kind == "phase" and entity:
                    cur_label = self._phase_frame_signature(
                        self.views[int(view_indices[i])], entity, f
                    )
                else:
                    cur_label = st.label_at(f)
                if cur_label != base_label:
                    same = False
                    break
            if same:
                if run_start is None:
                    run_start = f
            elif run_start is not None:
                spans.append((run_start, f - 1))
                run_start = None
        if run_start is not None:
            spans.append((run_start, fc - 1))
        return spans

    def _apply_sync_edit_masks(self) -> None:
        if not getattr(self, "timeline", None):
            return
        if not self._multiview_sync_active() or self._is_psr_task():
            try:
                self.timeline.set_row_edit_mask_provider(None)
            except Exception:
                pass
            return
        indices = self._effective_sync_edit_indices()
        cache: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}

        def provider(row):
            descriptor = self._mask_descriptor_from_row(row)
            if not self._syncable_descriptor(descriptor):
                return None
            key = self._descriptor_key(descriptor)
            if key not in cache:
                cache[key] = self._compute_editable_mask_spans(descriptor, indices)
            return cache.get(key, [])

        try:
            self.timeline.set_row_edit_mask_provider(provider)
        except Exception:
            pass

    @staticmethod
    def _store_matches_delta_old_labels(
        st: AnnotationStore, deltas: List[Tuple]
    ) -> bool:
        if st is None or not deltas:
            return False
        # Validate against a virtual state that applies deltas in order.
        # This handles relabel transactions that include:
        #   old -> None, then None -> new (same frames).
        virtual_updates: Dict[int, Optional[str]] = {}
        for s, e, old_label, new_label in ActionWindow._iter_delta_spans(deltas):
            s_i = int(s)
            e_i = int(e)
            if e_i < s_i:
                s_i, e_i = e_i, s_i
            for frame in range(s_i, e_i + 1):
                cur = (
                    virtual_updates[frame]
                    if frame in virtual_updates
                    else st.label_at(frame)
                )
                if old_label is None:
                    if cur is not None:
                        return False
                elif cur != old_label:
                    return False
            for frame in range(s_i, e_i + 1):
                virtual_updates[frame] = new_label
        return True

    @classmethod
    def _project_deltas_for_store(
        cls, st: AnnotationStore, deltas: List[Tuple]
    ) -> List[Tuple]:
        if st is None or not deltas:
            return []
        projected: List[Tuple] = []
        for s, e, old_label, new_label in cls._iter_delta_spans(deltas):
            s_i = int(s)
            e_i = int(e)
            if e_i < s_i:
                s_i, e_i = e_i, s_i
            run_start = None
            run_end = None
            for frame in range(s_i, e_i + 1):
                cur = st.label_at(frame)
                can_apply = False
                if old_label is None:
                    if cur is None:
                        can_apply = True
                else:
                    if cur == old_label:
                        can_apply = True
                if can_apply:
                    if run_start is None:
                        run_start = run_end = frame
                    elif frame == (run_end + 1):
                        run_end = frame
                    else:
                        if run_start == run_end:
                            projected.append((run_start, old_label, new_label))
                        else:
                            projected.append(
                                (run_start, run_end, old_label, new_label)
                            )
                        run_start = run_end = frame
            if run_start is not None:
                if run_start == run_end:
                    projected.append((run_start, old_label, new_label))
                else:
                    projected.append((run_start, run_end, old_label, new_label))
        return projected

    @staticmethod
    def _store_has_exact_segment(
        st: AnnotationStore, label: Optional[str], start: int, end: int
    ) -> bool:
        if st is None or not label:
            return False
        if end < start:
            start, end = end, start
        try:
            frames = list((st.label_to_frames or {}).get(label, []))
        except Exception:
            frames = []
        if not frames:
            return False
        i0 = bisect.bisect_left(frames, int(start))
        if i0 >= len(frames) or int(frames[i0]) != int(start):
            return False
        i1 = bisect.bisect_right(frames, int(end)) - 1
        if i1 < i0 or int(frames[i1]) != int(end):
            return False
        if (i1 - i0) != (int(end) - int(start)):
            return False
        if i0 > 0 and int(frames[i0 - 1]) == int(start) - 1:
            return False
        if (i1 + 1) < len(frames) and int(frames[i1 + 1]) == int(end) + 1:
            return False
        return True

    @staticmethod
    def _store_range_matches_label(
        st: AnnotationStore, label: Optional[str], start: int, end: int
    ) -> bool:
        if st is None or not label:
            return False
        if end < start:
            start, end = end, start
        for f in range(int(start), int(end) + 1):
            if st.label_at(f) != label:
                return False
        return True

    def _store_for_view_descriptor(
        self, view: Dict[str, Any], descriptor: Dict[str, Any]
    ) -> Optional[AnnotationStore]:
        kind = str(descriptor.get("kind") or "")
        if kind == "store":
            return view.get("store")
        if kind == "entity":
            entity = descriptor.get("entity")
            return (view.get("entity_stores") or {}).get(entity)
        if kind == "phase":
            entity = descriptor.get("entity")
            return (view.get("phase_stores") or {}).get(entity)
        if kind == "anomaly":
            entity = descriptor.get("entity")
            tname = descriptor.get("anomaly_type")
            return ((view.get("anomaly_type_stores") or {}).get(entity) or {}).get(
                tname
            )
        if kind == "extra":
            return getattr(self, "extra_store", None)
        return None

    def _store_name_for_view_descriptor(
        self, view_name: str, descriptor: Dict[str, Any]
    ) -> str:
        kind = str(descriptor.get("kind") or "")
        if kind == "store":
            return f"VIEW:{view_name}"
        if kind == "entity":
            return f"VIEW:{view_name}:{descriptor.get('entity')}"
        if kind == "phase":
            return f"VIEW:{view_name}:PHASE:{descriptor.get('entity')}"
        if kind == "anomaly":
            return (
                f"VIEW:{view_name}:ANOMALY:{descriptor.get('entity')}:"
                f"{descriptor.get('anomaly_type')}"
            )
        if kind == "extra":
            return "EXTRA"
        return f"VIEW:{view_name}"

    def _matching_view_targets_for_segment(
        self, descriptor: Dict[str, Any], label: Optional[str], start: int, end: int
    ) -> List[Dict[str, Any]]:
        kind = str(descriptor.get("kind") or "")
        if kind in ("unknown", "extra"):
            return []
        matches = []
        for idx, vw in enumerate(self.views or []):
            st = self._store_for_view_descriptor(vw, descriptor)
            if st is None:
                continue
            if not self._store_has_exact_segment(st, label, start, end):
                continue
            view_name = self._effective_view_name(vw, idx=idx)
            matches.append(
                {
                    "view_idx": idx,
                    "view_name": view_name,
                    "store": st,
                    "store_name": self._store_name_for_view_descriptor(
                        view_name, descriptor
                    ),
                }
            )
        return matches

    def _remove_segment_with_deltas(
        self, st: AnnotationStore, label: Optional[str], start: int, end: int
    ) -> List[Tuple]:
        if st is None or not label:
            return []
        if end < start:
            start, end = end, start
        try:
            st.begin_txn()
        except Exception:
            pass
        try:
            bulk_remove = getattr(st, "remove_range", None)
            if callable(bulk_remove):
                bulk_remove(label, int(start), int(end))
            else:
                for fr in range(int(start), int(end) + 1):
                    if st.label_at(fr) == label:
                        st.remove_at(fr)
        finally:
            try:
                st.end_txn()
            except Exception:
                pass
        consume = getattr(st, "consume_last_deltas", None)
        if callable(consume):
            return consume() or []
        return []

    def _drain_pending_store_deltas(
        self, views: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        seen: Set[int] = set()

        def _drain(st: Optional[AnnotationStore]) -> None:
            if st is None:
                return
            sid = id(st)
            if sid in seen:
                return
            seen.add(sid)
            consume = getattr(st, "consume_last_deltas", None)
            if callable(consume):
                try:
                    consume()
                except Exception:
                    pass

        _drain(getattr(self, "store", None))
        _drain(getattr(self, "extra_store", None))
        for st in (self.entity_stores or {}).values():
            _drain(st)
        for st in (self.phase_stores or {}).values():
            _drain(st)
        for ent_map in (self.anomaly_type_stores or {}).values():
            for st in (ent_map or {}).values():
                _drain(st)

        target_views = list(self.views or []) if views is None else list(views or [])
        for view in target_views:
            if not isinstance(view, dict):
                continue
            _drain(view.get("store"))
            _drain(view.get("prelabel_store"))
            for st in (view.get("entity_stores") or {}).values():
                _drain(st)
            for st in (view.get("phase_stores") or {}).values():
                _drain(st)
            for ent_map in (view.get("anomaly_type_stores") or {}).values():
                for st in (ent_map or {}).values():
                    _drain(st)

    def _refresh_after_manual_store_change(self) -> None:
        self.timeline.refresh_all_rows()
        self._update_gap_indicator()
        if self.phase_mode_enabled and self.mode == "Fine":
            rows = getattr(self.timeline, "_combined_rows", []) or []
            for row in rows:
                meta = getattr(row, "_group_meta", {}) if row is not None else {}
                if meta.get("row_type") != "phase":
                    continue
                ename = meta.get("entity")
                st = self.entity_stores.get(ename)
                if st is None:
                    continue
                try:
                    segs = [(s, e) for (s, e, _lb) in self._segments_from_store(st)]
                    row.set_snap_segments(segs)
                except Exception:
                    pass
        self._apply_sync_edit_masks()

    def _append_validation_entries_with_optional_comment(
        self, entries: List[Dict[str, Any]]
    ) -> None:
        if not self.validation_enabled or not entries:
            return
        if not bool(getattr(self, "_validation_comment_prompt_enabled", True)):
            self._append_validation_modifications(entries)
            return
        comment = ""
        ok = False
        try:
            comment, ok = QInputDialog.getText(
                self,
                "Optional comment",
                "Add a comment for these changes (leave blank to skip):",
                text="",
            )
        except Exception:
            ok = False
        if ok and comment.strip():
            for item in entries:
                item["comment"] = comment.strip()
        self._append_validation_modifications(entries)

    def _on_action_segment_trim(self, frame: int, row) -> bool:
        if self._is_psr_task() or not self.views:
            return False
        try:
            frame = int(frame)
        except Exception:
            return False
        descriptor = self._mask_descriptor_from_row(row)
        if not self._syncable_descriptor(descriptor):
            return False
        self._begin_correction_session(
            "direct_boundary",
            frame=int(frame),
            mode="trim_cut",
        )
        if self._multiview_sync_active():
            target_indices = self._effective_sync_edit_indices()
        else:
            target_indices = [int(self.active_view_idx)]
        ops = []
        added_views = []
        removed_views = []
        for idx in target_indices:
            if not (0 <= idx < len(self.views)):
                continue
            view = self.views[idx]
            st = self._store_for_view_descriptor(view, descriptor)
            if st is None:
                continue
            cuts = self._trim_cut_set_for_view(view, descriptor, create=True)
            if frame in cuts:
                cuts.discard(frame)
                ops.append(
                    {
                        "view_idx": int(idx),
                        "descriptor": dict(descriptor),
                        "frame": int(frame),
                        "op": "remove",
                    }
                )
                removed_views.append(self._effective_view_name(view, idx=idx))
                continue
            seg_s, seg_e, seg_label = self._segment_bounds_with_cuts(st, frame, cuts)
            if seg_label is None:
                continue
            if frame <= int(seg_s) or frame > int(seg_e):
                continue
            cuts.add(int(frame))
            ops.append(
                {
                    "view_idx": int(idx),
                    "descriptor": dict(descriptor),
                    "frame": int(frame),
                    "op": "add",
                }
            )
            added_views.append(self._effective_view_name(view, idx=idx))
        if not ops:
            self._discard_correction_session("trim_no_ops")
            return False
        self._push_undo_item({"kind": "trim_cuts", "ops": ops})
        self._redo_stack.clear()
        for op in ops:
            try:
                vi = int(op.get("view_idx", -1))
            except Exception:
                vi = -1
            if 0 <= vi < len(self.views):
                self.views[vi]["dirty"] = True
        self._dirty = True
        self._rebuild_timeline_sources()
        mode = "toggle"
        if added_views and not removed_views:
            mode = "split"
        elif removed_views and not added_views:
            mode = "merge"
        touched = sorted(set(added_views + removed_views))
        self._log(
            "segment_trim",
            mode=mode,
            frame=int(frame),
            views=",".join(touched[:6])
            + (f",+{len(touched)-6}" if len(touched) > 6 else ""),
            count=len(touched),
        )
        self._note_correction_step()
        self._commit_correction_session(
            point_type="boundary",
            boundary_frame=int(frame),
            mode=str(mode),
        )
        return True

    def _on_action_segment_delete(self, start: int, end: int, label, row) -> bool:
        if self._is_psr_task():
            return False
        try:
            start = int(start)
            end = int(end)
        except Exception:
            return False
        if end < start:
            start, end = end, start
        descriptor = self._store_descriptor_from_row(row, label)
        if not descriptor:
            return False
        self._begin_correction_session(
            "direct_delete",
            start=int(start),
            end=int(end),
            label=str(label or ""),
        )

        # Multi-view sync delete: when views are ctrl-selected, apply deletion
        # directly to all selected views whose span currently matches the label.
        sync_indices = self._effective_sync_edit_indices()
        if len(sync_indices) > 1:
            targets = []
            for idx in sync_indices:
                if not (0 <= idx < len(self.views)):
                    continue
                vw = self.views[idx]
                st = self._store_for_view_descriptor(vw, descriptor)
                if st is None:
                    continue
                if not self._store_range_matches_label(st, label, start, end):
                    continue
                view_name = self._effective_view_name(vw, idx=idx)
                targets.append(
                    {
                        "view_idx": int(idx),
                        "view_name": view_name,
                        "store": st,
                        "store_name": self._store_name_for_view_descriptor(
                            view_name, descriptor
                        ),
                    }
                )
            if len(targets) <= 1:
                self._set_status(
                    "No multi-view delete applied: selected views differ on this segment."
                )
                self._discard_correction_session("delete_no_sync_targets")
                return True

            batch_changes = []
            mod_entries = []
            touched_views = set()
            touched_names = []
            for item in targets:
                st = item.get("store")
                ds = self._remove_segment_with_deltas(st, label, start, end)
                if not ds:
                    continue
                batch_changes.append((st, ds))
                if self.validation_enabled:
                    mod_entries.extend(
                        self._record_modifications(item.get("store_name", ""), ds)
                    )
                v_idx = int(item.get("view_idx", -1))
                if 0 <= v_idx < len(self.views):
                    self.views[v_idx]["dirty"] = True
                    touched_views.add(v_idx)
                v_name = str(item.get("view_name", "")).strip()
                if v_name:
                    touched_names.append(v_name)
            if not batch_changes:
                self._discard_correction_session("delete_no_batch_changes")
                return True

            self._push_undo_batch(
                batch_changes,
                meta={
                    "action": "segment_delete_sync_selected",
                    "label": label,
                    "start": int(start),
                    "end": int(end),
                    "views": sorted(set(touched_names)),
                },
            )
            self._redo_stack.clear()
            self._dirty = True
            self._refresh_after_manual_store_change()
            if self.validation_enabled:
                self._append_validation_entries_with_optional_comment(mod_entries)

            unique_names = sorted(set(touched_names))
            preview = ",".join(unique_names[:4])
            if len(unique_names) > 4:
                preview = f"{preview},+{len(unique_names)-4}"
            self._log(
                "segment_delete_sync_selected",
                label=label,
                start=int(start),
                end=int(end),
                views=preview,
                count=len(touched_views),
            )
            self._note_correction_step()
            self._commit_correction_session(
                point_type="label",
                start=int(start),
                end=int(end),
                label=str(label or ""),
                mode="delete_sync_selected",
            )
            return True

        if not self.validation_enabled or len(self.views or []) <= 1:
            return False

        matched = self._matching_view_targets_for_segment(descriptor, label, start, end)
        if len(matched) <= 1:
            return False
        active_name = "current"
        if self.views and 0 <= self.active_view_idx < len(self.views):
            active_name = self._effective_view_name(
                self.views[self.active_view_idx], idx=self.active_view_idx
            )
        menu = QMenu(self)
        action_current = menu.addAction(f"Delete in current view ({active_name})")
        action_all = menu.addAction(f"Delete in {len(matched)} matched views")
        chosen = menu.exec_(QCursor.pos())
        if chosen is None:
            self._discard_correction_session("delete_menu_cancelled")
            return True
        if chosen is action_current:
            return False
        if chosen is not action_all:
            self._discard_correction_session("delete_menu_other")
            return True

        batch_changes = []
        mod_entries = []
        touched_views = set()
        touched_names = []
        for item in matched:
            st = item.get("store")
            ds = self._remove_segment_with_deltas(st, label, start, end)
            if not ds:
                continue
            batch_changes.append((st, ds))
            mod_entries.extend(
                self._record_modifications(item.get("store_name", ""), ds)
            )
            v_idx = int(item.get("view_idx", -1))
            if 0 <= v_idx < len(self.views):
                self.views[v_idx]["dirty"] = True
                touched_views.add(v_idx)
            v_name = str(item.get("view_name", "")).strip()
            if v_name:
                touched_names.append(v_name)
        if not batch_changes:
            self._discard_correction_session("delete_no_batch_changes")
            return True

        self._push_undo_batch(
            batch_changes,
            meta={
                "action": "segment_delete_multi_view",
                "label": label,
                "start": int(start),
                "end": int(end),
                "views": sorted(set(touched_names)),
            },
        )
        self._redo_stack.clear()
        self._dirty = True
        self._refresh_after_manual_store_change()
        self._append_validation_entries_with_optional_comment(mod_entries)

        unique_names = sorted(set(touched_names))
        preview = ",".join(unique_names[:4])
        if len(unique_names) > 4:
            preview = f"{preview},+{len(unique_names)-4}"
        self._log(
            "segment_delete_multi_view",
            label=label,
            start=int(start),
            end=int(end),
            views=preview,
            count=len(touched_views),
        )
        self._note_correction_step()
        self._commit_correction_session(
            point_type="label",
            start=int(start),
            end=int(end),
            label=str(label or ""),
            mode="delete_multi_view",
        )
        return True

    def _on_store_changed(self, prompt_validation_comment: bool = True):
        if bool(getattr(self, "_suspend_store_changed", False)):
            return
        stores = [
            ("GLOBAL", self.store),
            ("EXTRA", getattr(self, "extra_store", None)),
        ] + list(self.entity_stores.items())
        for ename, pstore in (self.phase_stores or {}).items():
            stores.append((f"PHASE:{ename}", pstore))
        for ename, type_map in (self.anomaly_type_stores or {}).items():
            for tname, st in (type_map or {}).items():
                stores.append((f"ANOMALY:{ename}:{tname}", st))
        if self.views and 0 <= self.active_view_idx < len(self.views):
            vw = self.views[self.active_view_idx]
            vname = vw.get("name", f"view{self.active_view_idx}")
            stores.append((f"VIEW:{vname}", vw.get("store")))
            for ename, st in (vw.get("entity_stores", {}) or {}).items():
                stores.append((f"VIEW:{vname}:{ename}", st))
            for ename, pstore in (vw.get("phase_stores", {}) or {}).items():
                stores.append((f"VIEW:{vname}:PHASE:{ename}", pstore))
            for ename, type_map in (vw.get("anomaly_type_stores", {}) or {}).items():
                for tname, st in (type_map or {}).items():
                    stores.append((f"VIEW:{vname}:ANOMALY:{ename}:{tname}", st))
        seen = set()
        changed_records = []
        for store_name, st in stores:
            if st is None:
                continue
            sid = id(st)
            if sid in seen:
                continue
            seen.add(sid)
            consume = getattr(st, "consume_last_deltas", None)
            if callable(consume):
                ds = consume()
                if ds:
                    changed_frames = self._delta_frame_count(ds)
                    summary = self._summarize_deltas(ds, max_chars=240)
                    descriptor = self._store_descriptor_from_store(st) or {
                        "kind": "unknown",
                        "store": st,
                    }
                    changed_records.append(
                        {
                            "store_name": store_name,
                            "store": st,
                            "deltas": ds,
                            "frames": changed_frames,
                            "summary": summary,
                            "descriptor": descriptor,
                        }
                    )

        if not changed_records:
            self._refresh_after_manual_store_change()
            if self._is_psr_task():
                self._psr_mark_dirty()
                self._psr_update_component_panel()
            active_session = getattr(self._correction_buffer, "active", None)
            if active_session is not None and str(getattr(active_session, "kind", "") or "") == "direct_delete":
                self._discard_correction_session("delete_no_net_change")
            return

        all_history_entries = []
        all_mod_entries = []
        log_summary = []
        log_count = 0
        trim_cleanup_ops: List[Dict[str, Any]] = []
        touched_view_idxs = set()
        if 0 <= self.active_view_idx < len(self.views):
            touched_view_idxs.add(int(self.active_view_idx))
        for rec in changed_records:
            all_history_entries.append((rec["store"], rec["deltas"]))
            rec_spans = [(s, e) for s, e, _old, _new in self._iter_delta_spans(rec["deltas"])]
            rec["spans"] = rec_spans
            log_summary.append(
                f"{rec['store_name']}[{rec['frames']}]: {rec['summary']}"
            )
            log_count += int(rec["frames"])
            if self.validation_enabled:
                all_mod_entries.extend(
                    self._record_modifications(rec["store_name"], rec["deltas"])
                )
            trim_cleanup_ops.extend(
                self._cleanup_trim_cuts_for_spans(
                    self.active_view_idx, rec.get("descriptor"), rec_spans
                )
            )

        sync_views = []
        sync_indices = self._effective_sync_edit_indices()
        if len(sync_indices) > 1:
            for rec in changed_records:
                descriptor = rec.get("descriptor") or {}
                if not self._syncable_descriptor(descriptor):
                    continue
                ds = rec["deltas"]
                rec_spans = rec.get("spans") or [
                    (s, e) for s, e, _old, _new in self._iter_delta_spans(ds)
                ]
                for idx in sync_indices:
                    if idx == self.active_view_idx:
                        continue
                    vw = self.views[idx]
                    target_store = self._store_for_view_descriptor(vw, descriptor)
                    if target_store is None or target_store is rec["store"]:
                        continue
                    apply_ds = ds
                    if not self._store_matches_delta_old_labels(target_store, ds):
                        apply_ds = self._project_deltas_for_store(target_store, ds)
                        if not apply_ds:
                            continue
                    apply_fw = getattr(target_store, "apply_deltas", None)
                    if not callable(apply_fw):
                        continue
                    apply_fw(apply_ds)
                    all_history_entries.append((target_store, apply_ds))
                    touched_view_idxs.add(int(idx))
                    view_name = self._effective_view_name(vw, idx=idx)
                    sync_views.append(view_name)
                    target_store_name = self._store_name_for_view_descriptor(
                        view_name, descriptor
                    )
                    if self.validation_enabled:
                        all_mod_entries.extend(
                            self._record_modifications(target_store_name, apply_ds)
                        )
                    log_count += self._delta_frame_count(apply_ds)
                    trim_cleanup_ops.extend(
                        self._cleanup_trim_cuts_for_spans(
                            idx, descriptor, rec_spans
                        )
                    )

        history_meta = {
            "action": "sync_edit" if len(sync_indices) > 1 else "single_edit",
            "views": sorted(set(sync_views)),
        }
        if all_history_entries or trim_cleanup_ops:
            if trim_cleanup_ops:
                self._push_undo_composite(
                    all_history_entries,
                    trim_cleanup_ops,
                    meta=history_meta,
                )
            elif len(all_history_entries) == 1:
                st, ds = all_history_entries[0]
                self._push_undo_entry(st, ds)
            else:
                self._push_undo_batch(all_history_entries, meta=history_meta)
            self._redo_stack.clear()
            self._dirty = True
        for idx in touched_view_idxs:
            if 0 <= idx < len(self.views):
                self.views[idx]["dirty"] = True
        if trim_cleanup_ops:
            self._rebuild_timeline_sources()
        else:
            self._refresh_after_manual_store_change()
        if self._is_psr_task():
            self._psr_mark_dirty()
            self._psr_update_component_panel()

        if sync_views:
            joined = ",".join(sorted(set(sync_views)))
            log_summary.append(f"SYNC[{len(set(sync_views))}]: {joined}")
        if log_summary:
            merged = " | ".join(log_summary)
            if len(merged) > 320:
                merged = merged[:317] + "..."
            self._log(
                "store_change",
                store=(
                    "SYNC_VIEWS"
                    if len(sync_indices) > 1 and len(touched_view_idxs) > 1
                    else "ACTIVE_VIEW"
                ),
                count=log_count,
                changes=merged,
                trim_cleanup=len(trim_cleanup_ops),
            )
        if self.validation_enabled and all_mod_entries:
            if prompt_validation_comment and bool(
                getattr(self, "_validation_comment_prompt_enabled", True)
            ):
                self._append_validation_entries_with_optional_comment(all_mod_entries)
            else:
                self._append_validation_modifications(all_mod_entries)
        active_session = getattr(self._correction_buffer, "active", None)
        if active_session is not None and str(getattr(active_session, "kind", "") or "") == "direct_delete":
            self._note_correction_step()
            self._commit_correction_session(point_type="label", mode="delete_default")

    def _undo(self):
        if self._is_psr_task():
            self._psr_undo()
            return
        if not getattr(self, "_undo_stack", None):
            return
        item = self._undo_stack.pop()
        if self._apply_history_item(item, reverse=True):
            self._push_redo_item(item)
            self._dirty = True
            if isinstance(item, dict) and item.get("kind") in ("trim_cuts", "composite"):
                self._rebuild_timeline_sources()
            else:
                # self.timeline.update()
                self.timeline.refresh_all_rows()
            self._apply_sync_edit_masks()
            self._log("undo", count=self._history_item_frame_count(item))

    def _redo(self):
        if self._is_psr_task():
            self._psr_redo()
            return
        if not getattr(self, "_redo_stack", None):
            return
        item = self._redo_stack.pop()
        if self._apply_history_item(item, reverse=False):
            self._push_undo_item(item)
            self._dirty = True
            if isinstance(item, dict) and item.get("kind") in ("trim_cuts", "composite"):
                self._rebuild_timeline_sources()
            else:
                # self.timeline.update()
                self.timeline.refresh_all_rows()
            self._apply_sync_edit_masks()
            self._log("redo", count=self._history_item_frame_count(item))

    def _prompt_save_if_dirty(
        self,
        context: str = "this action",
        force_prompt: bool = False,
        extra_warning: str = "",
    ) -> bool:
        """
        Prompt if there are unsaved changes. Return True to continue; False to cancel.
        """
        if not getattr(self, "_dirty", False) and not force_prompt:
            return True
        m = QMessageBox(self)
        m.setIcon(QMessageBox.Warning)
        m.setWindowTitle("Unsaved changes")
        base = f"You have unsaved annotations.\n\nDo you want to save before {context}?"
        if extra_warning:
            base = base + "\n\n" + extra_warning
        m.setText(base)
        m.setStandardButtons(
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
        )
        m.setDefaultButton(QMessageBox.Save)
        ret = m.exec_()
        if ret == QMessageBox.Save:
            ok = self._save_json_annotations(skip_gap_check=force_prompt)
            return bool(ok)
        elif ret == QMessageBox.Discard:
            return True
        else:
            return False

    def _reset_all_state_for_new_video(self, keep_views: bool = False):
        # remember current mode so loading a video does not force a mode switch
        prev_mode = self.mode if self.mode in ("Coarse", "Fine") else "Coarse"
        # clear annotations (coarse + fine)
        self.store.frame_to_label.clear()
        self.store.label_to_frames.clear()
        if getattr(self, "prelabel_store", None):
            self.prelabel_store.frame_to_label.clear()
            self.prelabel_store.label_to_frames.clear()
        self.extra_store = AnnotationStore()
        self.extra_cuts = []
        for st in self.entity_stores.values():
            st.frame_to_label.clear()
            st.label_to_frames.clear()
        for st in self.phase_stores.values():
            st.frame_to_label.clear()
            st.label_to_frames.clear()
        for ent_map in self.anomaly_type_stores.values():
            for st in ent_map.values():
                st.frame_to_label.clear()
                st.label_to_frames.clear()

        # clear labels / entities / display
        self.labels.clear()
        self.label_entity_map.clear()
        self.entities.clear()
        self.entity_stores.clear()
        self.phase_stores.clear()
        self.anomaly_type_stores.clear()
        for v in getattr(self, "views", []):
            v["entity_stores"] = {}
            v["phase_stores"] = {}
            v["anomaly_type_stores"] = {}
            v["prelabel_source"] = ""
            v["confirmed_accept_records"] = []
            v["psr_state"] = self._psr_empty_view_state()
            v["trim_cuts"] = self._empty_trim_state()
            v["baseline_trim_cuts"] = self._empty_trim_state()
            if "prelabel_store" in v and v["prelabel_store"] is not None:
                try:
                    v["prelabel_store"].frame_to_label.clear()
                    v["prelabel_store"].label_to_frames.clear()
                except Exception:
                    v["prelabel_store"] = AnnotationStore()
        if self.views:
            self.entity_stores = self.views[self.active_view_idx].get(
                "entity_stores", {}
            )
            self.phase_stores = self.views[self.active_view_idx].get("phase_stores", {})
            self.anomaly_type_stores = self.views[self.active_view_idx].get(
                "anomaly_type_stores", {}
            )
        self.visible_entities.clear()
        self.current_label_idx = -1
        self.extra_mode = False
        self.extra_label = None
        self.extra_last_frame = None
        self._extra_force_follow = False
        self._extra_txn_stores = []
        self.interaction_mode = None
        self.assisted_points = []
        self.assisted_active_idx = -1
        self._assisted_review_done = False
        self._assisted_loop_range = None
        self._assisted_candidates = {}
        self._assisted_source_segments = []
        self._has_auto_segments = False
        self._auto_boundary_candidates = []
        self._auto_boundary_source = ""
        self._boundary_snap_cache = {}
        self._segment_embedding_cache = {}
        self._label_prototypes = {}
        self._label_proto_counts = {}
        self._knn_memory = []
        self._correction_buffer = CorrectionBuffer()
        self._confirmed_correction_records = []
        self._forced_segment = None
        self.phase_mode_enabled = False
        self._phase_selected = None
        self.fine_verbs = []
        self.fine_nouns = []
        self._prelabel_source = ""
        self.anomaly_types = []
        self._ensure_anomaly_types()
        self._rebuild_anomaly_type_panel()
        try:
            if self._assisted_loop_timer:
                self._assisted_loop_timer.stop()
        except Exception:
            pass
        for sc in (
            getattr(self, "sc_assist_left", None),
            getattr(self, "sc_assist_right", None),
            getattr(self, "sc_assist_confirm", None),
            getattr(self, "sc_assist_down", None),
            getattr(self, "sc_assist_next", None),
            getattr(self, "sc_assist_prev", None),
            getattr(self, "sc_assist_skip", None),
            getattr(self, "sc_assist_merge", None),
            getattr(self, "sc_assist_merge_del", None),
        ):
            if sc:
                try:
                    sc.setEnabled(False)
                except Exception:
                    pass
        self.current_annotation_path = None
        self.currentFeatureDir = None
        self.currentEastFeatureDir = None
        self._feature_followup_task = ""
        self._feature_followup_backbone = ""
        self._psr_clear_undo()
        # New video should always start with a clean transcript/PSR timeline state.
        # Keep rules/components/model settings, but clear per-video state edits.
        self.asr_segments = []
        self.asr_lang = "auto"
        try:
            self.timeline.set_subtitle_tracks([])
        except Exception:
            pass
        self._psr_manual_events = []
        self._psr_segment_cuts = []
        self._psr_gap_spans_combined = []
        self._psr_gap_spans_by_comp = {}
        self._psr_selected_segment = None
        self._psr_state_store_combined = None
        self._psr_state_stores = {}
        self._psr_combined_label_states = {}
        self._psr_state_color_cache = {}
        self._psr_events_cache = []
        self._psr_state_seq = []
        self._psr_state_frames = []
        self._psr_diag = {"events": 0, "unmapped": 0, "rule_mismatch": 0}
        self._psr_action_segment_starts = []
        self._psr_action_segment_ends = []
        self._psr_action_segments = []
        self._psr_detected_flow = "assemble"
        self._psr_detected_initial_state = 0
        self._psr_cache_dirty = True
        self._psr_state_dirty = True
        self._psr_timeline_change_deferred = False
        if self.views and 0 <= self.active_view_idx < len(self.views):
            self._psr_load_view_state(self.views[self.active_view_idx])
        try:
            self.btn_extra.setChecked(False)
            self.btn_assisted.setChecked(False)
        except Exception:
            pass
        try:
            self.lbl_interaction_status.setText("Interaction: idle")
        except Exception:
            pass

        # restore prior mode selection (keep user's choice)
        self.mode = prev_mode
        try:
            self.combo_mode.setCurrentText(prev_mode)
        except Exception:
            pass
        self._apply_default_label_template(prev_mode, force=True, reason="new_video")

        # refresh UI
        self.panel.refresh()
        self.entities_panel.set_current_label(None, set())
        self.entities_panel.setVisible(False)
        self._update_phase_panel_visibility()
        self._rebuild_timeline_sources()
        try:
            self.timeline.set_extra_cuts(self.extra_cuts)
        except Exception:
            pass
        self.timeline.update()

        self._dirty = False
        # reset validation state
        self.validation_modifications.clear()
        self.validation_errors.clear()
        self.validation_enabled = False
        self.validator_name = ""
        self.review_items.clear()
        self.review_idx = -1
        try:
            self.btn_validation.setChecked(False)
            self._set_validation_button_state(False)
            self.player.set_overlay_enabled(False)
            self.player.set_overlay_labels([])
            self._set_review_panel_visible(False)
        except Exception:
            pass
        if not keep_views:
            extras = self.views[1:] if self.views else []
            for extra in extras:
                player = extra.get("player")
                if player is not None:
                    try:
                        player.release_media()
                    except Exception:
                        pass
            self.views = self.views[:1] if self.views else []
            if self.views:
                self.views[0].update(
                    {
                        "start": 0,
                        "end": 0,
                        "path": None,
                        "store": AnnotationStore(),
                        "prelabel_store": AnnotationStore(),
                        "prelabel_source": "",
                        "confirmed_accept_records": [],
                        "entity_stores": {},
                        "phase_stores": {},
                        "anomaly_type_stores": {},
                        "trim_cuts": self._empty_trim_state(),
                        "dirty": False,
                    }
                )
                self.active_view_idx = 0
                self.player = self.views[0]["player"]
                self.prelabel_store = self.views[0]["prelabel_store"]
                self._prelabel_source = str(self.views[0].get("prelabel_source", "") or "")
                self.entity_stores = self.views[0]["entity_stores"]
                self.phase_stores = self.views[0].get("phase_stores", {})
                self.anomaly_type_stores = self.views[0].get("anomaly_type_stores", {})
            self._rebuild_view_widgets()

    # ----- entities / mode callbacks -----
    def _on_entity_added(self, ent):
        for v in self.views:
            stores = v.setdefault("entity_stores", {})
            stores.setdefault(ent.name, AnnotationStore())
            phase_stores = v.setdefault("phase_stores", {})
            phase_stores.setdefault(ent.name, AnnotationStore())
            anom_stores = v.setdefault("anomaly_type_stores", {})
            ent_anom = anom_stores.setdefault(ent.name, {})
            for tname in self._anomaly_type_names():
                ent_anom.setdefault(tname, AnnotationStore())
        if self.views and 0 <= self.active_view_idx < len(self.views):
            self.entity_stores = self.views[self.active_view_idx].get(
                "entity_stores", {}
            )
            self.phase_stores = self.views[self.active_view_idx].get("phase_stores", {})
            self.anomaly_type_stores = self.views[self.active_view_idx].get(
                "anomaly_type_stores", {}
            )
        if self.validation_enabled and self.mode == "Fine":
            if ent.name not in self.visible_entities:
                self.visible_entities.append(ent.name)
        self._rebuild_timeline_sources()
        self._sync_entity_panel_mode()
        self._dirty = True
        self._log("entity_add", entity=ent.name)

    def _on_entity_removed(self, idx: int):
        if 0 <= idx < len(self.entities):
            name = self.entities[idx].name
            for v in self.views:
                stores = v.get("entity_stores")
                if stores is not None:
                    stores.pop(name, None)
                pstores = v.get("phase_stores")
                if pstores is not None:
                    pstores.pop(name, None)
                astores = v.get("anomaly_type_stores")
                if astores is not None:
                    astores.pop(name, None)
                trim_cuts = v.get("trim_cuts")
                if isinstance(trim_cuts, dict):
                    for kind in ("entity", "phase"):
                        bucket = trim_cuts.get(kind)
                        if isinstance(bucket, dict):
                            bucket.pop(name, None)
            if self.views and 0 <= self.active_view_idx < len(self.views):
                self.entity_stores = self.views[self.active_view_idx].get(
                    "entity_stores", {}
                )
                self.phase_stores = self.views[self.active_view_idx].get(
                    "phase_stores", {}
                )
                self.anomaly_type_stores = self.views[self.active_view_idx].get(
                    "anomaly_type_stores", {}
                )
            # remove entity from all label mappings
            for k in list(self.label_entity_map.keys()):
                self.label_entity_map[k].discard(name)
            if name in self.visible_entities:
                self.visible_entities = [n for n in self.visible_entities if n != name]
            if self._active_entity_name == name:
                self._active_entity_name = None
            self._log("entity_remove", entity=name)
        QTimer.singleShot(0, self._rebuild_timeline_sources)
        QTimer.singleShot(0, self._sync_entity_panel_mode)
        self._dirty = True

    def _record_modifications(self, store_name: str, deltas):
        if not deltas:
            return []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        view_name = ""
        if store_name and str(store_name).startswith("VIEW:"):
            tail = str(store_name)[5:]
            view_name = tail.split(":", 1)[0]
        if 0 <= self.active_view_idx < len(self.views):
            view_name = view_name or self.views[self.active_view_idx].get("name", "")
        buckets: Dict[Tuple[Optional[str], Optional[str]], List[Tuple[int, int]]] = {}
        for s, e, old_label, new_label in self._iter_delta_spans(deltas):
            key = (old_label, new_label)
            buckets.setdefault(key, []).append((int(s), int(e)))
        batch_entries = []
        for (old_label, new_label), spans in buckets.items():
            for s, e in self._merge_spans(spans):
                batch_entries.append(
                    {
                        "ts": ts,
                        "editor": self.validator_name or "unknown",
                        "store": store_name,
                        "view": view_name,
                        "start": int(s),
                        "end": int(e),
                        "old": old_label,
                        "new": new_label,
                    }
                )
        return batch_entries

    def _on_label_selected(self, idx: int):
        self.current_label_idx = idx
        handled_forced = False
        forced = getattr(self, "_forced_segment", None)
        if forced and 0 <= idx < len(self.labels):
            lb = self.labels[idx]
            if lb and lb.name:
                if self._apply_label_to_segment(
                    forced.get("start"), forced.get("end"), lb.name
                ):
                    handled_forced = True
                    self._log(
                        "segment_relabel",
                        start=int(forced.get("start", 0)),
                        end=int(forced.get("end", 0)),
                        old=forced.get("label"),
                        new=lb.name,
                    )
                self._forced_segment = None
                try:
                    self.panel.clear_candidate_priority()
                except Exception:
                    pass
                if self.interaction_mode == "assisted":
                    try:
                        self._build_assisted_points_from_store(
                            preserve_status=True,
                            active_hint=("label", int(forced.get("start", 0))),
                        )
                        self._update_assisted_visuals()
                    except Exception:
                        pass
        if self.interaction_mode == "assisted" and not handled_forced:
            pt = self._active_assisted_point()
            if pt and pt.get("type") == "label" and 0 <= idx < len(self.labels):
                lb = self.labels[idx]
                if lb and lb.name:
                    self._apply_assisted_label_choice(lb.name)
        if 0 <= idx < len(self.labels):
            lb = self.labels[idx]
            checked = self.label_entity_map.get(lb.name, set())
            if getattr(self.entities_panel, "mode", "applicability") == "applicability":
                self.entities_panel.set_current_label(lb.name, checked)
            if getattr(self, "timeline", None):
                try:
                    row = getattr(self.timeline, "_active_combined_row", None)
                    meta = getattr(row, "_group_meta", {}) if row is not None else {}
                    if meta.get("row_type") != "phase":
                        self.timeline.apply_combined_label(lb.name)
                except Exception:
                    pass
            if getattr(self, "timeline", None):
                try:
                    self.timeline.flash_label(lb.name)
                except Exception:
                    pass
            self._log("label_select", label=lb.name, id=getattr(lb, "id", None))
        else:
            if getattr(self.entities_panel, "mode", "applicability") == "applicability":
                self.entities_panel.set_current_label(None, set())
            self._log("label_select", label=None, id=None)
        if self.mode == "Fine" and self._active_entity_name:
            self._sync_entity_panel_mode()

    def _on_timeline_label_clicked(self, name: str, frame: int):
        if not name:
            return
        if any(lb.name == name for lb in self.phase_labels):
            btn = self._phase_buttons.get(name)
            if btn is not None:
                try:
                    btn.setChecked(True)
                except Exception:
                    pass
            self._phase_active_label = name
            return
        for idx, lb in enumerate(self.labels):
            if lb.name == name:
                try:
                    self.panel.select_label_by_name(name)
                except Exception:
                    pass
                if self.interaction_mode == "assisted":
                    self._activate_assisted_at_frame(frame)
                self._log("timeline_label_click", label=name, frame=frame)
                break

    def _on_timeline_segment_selected(self, start: int, end: int, label):
        if self._is_psr_task():
            row = getattr(self.timeline, "_active_combined_row", None)
            scope = (
                getattr(row, "_selection_scope", "segment")
                if row is not None
                else "segment"
            )
            self._psr_set_selected_segment(start, end, label, row=row, scope=scope)
            self._psr_update_component_panel(int(start))
        elif self.mode == "Fine":
            row = getattr(self.timeline, "_active_combined_row", None)
            meta = getattr(row, "_group_meta", {}) if row is not None else {}
            title = getattr(row, "title", None) if row is not None else None
            ename = meta.get("entity") or title
            if ename and ename in self.entity_stores:
                if ename != self._active_entity_name:
                    self._active_entity_name = ename
                    self._sync_entity_panel_mode()
            if meta.get("row_type") == "phase":
                self._phase_selected = {
                    "entity": ename,
                    "start": int(start),
                    "end": int(end),
                    "label": label if label is not None else "",
                }
                self._sync_anomaly_panel()
            else:
                if self._phase_selected is not None:
                    self._phase_selected = None
                    self._sync_anomaly_panel()
        if not self._topk_enabled():
            return
        try:
            s = int(start)
            e = int(end)
        except Exception:
            return
        seg = {"start": s, "end": e, "label": label}
        candidates = self._label_candidates_for_segment(seg)
        if not candidates:
            return
        self._forced_segment = {"start": s, "end": e, "label": label}
        try:
            self.panel.set_candidate_priority(candidates)
        except Exception:
            pass
        self._set_interaction_status("Top-K: select a label to apply")

    def _on_label_search_matches(self, names: List[str]):
        if getattr(self, "timeline", None):
            try:
                self.timeline.set_highlight_labels(names)
                if names:
                    self.timeline.flash_label(names[0])
            except Exception:
                pass
        self._log("label_search", matches=",".join(names) if names else "")

    def _on_entity_renamed(self, old: str, new: str):
        if old == new:
            return
        for v in self.views:
            stores = v.get("entity_stores")
            if stores is None:
                continue
            st = stores.pop(old, None)
            if st is not None:
                stores[new] = st
            pstores = v.get("phase_stores")
            if pstores is not None:
                pst = pstores.pop(old, None)
                if pst is not None:
                    pstores[new] = pst
            astores = v.get("anomaly_type_stores")
            if astores is not None:
                amap = astores.pop(old, None)
                if amap is not None:
                    astores[new] = amap
            trim_cuts = v.get("trim_cuts")
            if isinstance(trim_cuts, dict):
                for kind in ("entity", "phase"):
                    bucket = trim_cuts.get(kind)
                    if not isinstance(bucket, dict):
                        continue
                    cuts = bucket.pop(old, None)
                    if cuts is not None:
                        bucket[new] = cuts
        if self.views and 0 <= self.active_view_idx < len(self.views):
            self.entity_stores = self.views[self.active_view_idx].get(
                "entity_stores", {}
            )
            self.phase_stores = self.views[self.active_view_idx].get("phase_stores", {})
            self.anomaly_type_stores = self.views[self.active_view_idx].get(
                "anomaly_type_stores", {}
            )
        if self._active_entity_name == old:
            self._active_entity_name = new
        if old in self.visible_entities:
            self.visible_entities = [
                new if n == old else n for n in self.visible_entities
            ]
        for k in list(self.label_entity_map.keys()):
            s = self.label_entity_map[k]
            if old in s:
                s.discard(old)
                s.add(new)
        if getattr(self.entities_panel, "mode", "applicability") == "applicability":
            if 0 <= self.current_label_idx < len(self.labels):
                lb = self.labels[self.current_label_idx]
                self.entities_panel.set_current_label(
                    lb.name, self.label_entity_map.get(lb.name, set())
                )
        self._rebuild_timeline_sources()
        self._dirty = True
        self._log("entity_rename", old=old, new=new)

    def _on_entity_row_reordered(self, src: Optional[str], dst: Optional[str]) -> None:
        if not src or not dst or src == dst:
            return
        if self._is_psr_task() or self.mode != "Fine":
            return
        names = [e.name for e in self.entities]
        if src not in names or dst not in names:
            return
        i = names.index(src)
        j = names.index(dst)
        ent = self.entities.pop(i)
        if i < j:
            j -= 1
        self.entities.insert(j, ent)
        self._active_entity_name = ent.name
        self._sync_entity_panel_mode()
        self._rebuild_timeline_sources()
        self._log("entity_reorder", src=src, dst=dst)

    def _on_entity_applicability_changed(self, label_name: str, names: List[str]):
        self.label_entity_map[label_name] = set(names)
        self._rebuild_timeline_sources()
        self._dirty = True
        self._log(
            "entity_apply_change", label=label_name, entities=",".join(sorted(names))
        )

    def _on_entity_visibility_changed(self, names: List[str]):
        self.visible_entities = list(names or [])
        try:
            frame = self.views[self.active_view_idx]["player"].current_frame
        except Exception:
            frame = 0
        self._update_overlay_for_frame(frame)
        self._log(
            "entity_visibility_change", entities=",".join(sorted(self.visible_entities))
        )

    def _label_mapping_signature(
        self, labels: Optional[List[LabelDef]] = None
    ) -> List[Tuple[str, int]]:
        rows: List[Tuple[str, int]] = []
        for lb in labels if labels is not None else self.labels:
            try:
                rows.append((str(lb.name), int(lb.id)))
            except Exception:
                continue
        return sorted(rows, key=lambda x: (int(x[1]), str(x[0])))

    def _labels_match_default_template(
        self, mode: str, labels: Optional[List[LabelDef]] = None
    ) -> bool:
        target = sorted(
            get_default_action_label_template(mode),
            key=lambda x: (int(x[1]), str(x[0])),
        )
        return self._label_mapping_signature(labels) == target

    def _labels_match_any_default_template(
        self, labels: Optional[List[LabelDef]] = None
    ) -> bool:
        return self._labels_match_default_template(
            "Coarse", labels
        ) or self._labels_match_default_template("Fine", labels)

    def _has_any_action_annotations(self) -> bool:
        stores: List[Any] = [
            getattr(self, "store", None),
            getattr(self, "prelabel_store", None),
            getattr(self, "extra_store", None),
        ]
        for view in getattr(self, "views", []) or []:
            stores.extend(
                [
                    view.get("store"),
                    view.get("prelabel_store"),
                ]
            )
            stores.extend((view.get("entity_stores") or {}).values())
            stores.extend((view.get("phase_stores") or {}).values())
            for ent_map in (view.get("anomaly_type_stores") or {}).values():
                stores.extend((ent_map or {}).values())
        for st in stores:
            if getattr(st, "frame_to_label", None):
                return True
        return False

    def _replace_action_labels(
        self,
        mapping: List[Tuple[str, int]],
        *,
        refresh_panel: bool = True,
        rebuild_timeline: bool = True,
        preserve_selection: bool = True,
        label_source_path: str = "",
    ) -> int:
        rows: List[Tuple[str, int]] = []
        seen: Set[Tuple[str, int]] = set()
        for raw_name, raw_id in mapping or []:
            name = str(raw_name or "").strip()
            if not name:
                continue
            try:
                lid = int(raw_id)
            except Exception:
                continue
            key = (name, lid)
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
        if not rows:
            return 0

        previous_name = ""
        if preserve_selection and 0 <= self.current_label_idx < len(self.labels):
            try:
                previous_name = str(self.labels[self.current_label_idx].name or "")
            except Exception:
                previous_name = ""

        self.labels.clear()
        for name, lid in rows:
            col = self._auto_color_key_for_id(lid)
            self.labels.append(LabelDef(name=name, color_name=col, id=lid))
        self._action_label_bank_source = str(label_source_path or "").strip()
        if "psr/asr/asd" in (self.combo_task.currentText() or "").lower():
            self._ensure_psr_asr_asd_invisible_label(refresh=False)
        self._refresh_fine_label_decomposition(refresh_panel=False)

        next_idx = -1
        if previous_name:
            for i, lb in enumerate(self.labels):
                if str(lb.name) == previous_name:
                    next_idx = i
                    break
        self.current_label_idx = next_idx

        if refresh_panel and getattr(self, "panel", None):
            self.panel.refresh()
        if rebuild_timeline and getattr(self, "timeline", None):
            self._rebuild_timeline_sources()
            try:
                self.timeline.update()
            except Exception:
                pass
        self._psr_mark_dirty()
        self._psr_update_component_panel()
        return len(rows)

    def _apply_default_label_template(
        self, mode: str, *, force: bool = False, reason: str = ""
    ) -> bool:
        target_mode = "Fine" if str(mode or "").strip().lower() == "fine" else "Coarse"
        if not force:
            if self.labels:
                if self._labels_match_default_template(target_mode):
                    return False
                if not self._labels_match_any_default_template():
                    return False
                if self._has_any_action_annotations():
                    return False
        count = self._replace_action_labels(
            get_default_action_label_template(target_mode),
            refresh_panel=bool(getattr(self, "panel", None)),
            rebuild_timeline=bool(getattr(self, "timeline", None)),
            label_source_path="",
        )
        if count:
            self._log(
                "default_label_template",
                mode=target_mode.lower(),
                count=count,
                reason=reason or ("force" if force else "auto"),
            )
        return bool(count)

    def _sync_entity_panel_mode(self):
        if not getattr(self, "entities_panel", None):
            return
        if self.mode == "Fine" and self.validation_enabled:
            all_names = [e.name for e in self.entities]
            if not self.visible_entities:
                self.visible_entities = list(all_names)
            else:
                self.visible_entities = [
                    n for n in self.visible_entities if n in all_names
                ]
            self.entities_panel.set_mode("visibility", set(self.visible_entities))
            return
        self.entities_panel.set_mode("applicability")
        label_name = ""
        try:
            label_name = self.panel.current_label_name()
        except Exception:
            label_name = ""
        if self.mode == "Fine" and self._active_entity_name:
            checked = {self._active_entity_name}
        else:
            checked = (
                self.label_entity_map.get(label_name, set()) if label_name else set()
            )
        self.entities_panel.set_current_label(label_name or None, checked)

    def _on_mode_changed(self, text: str):
        self.mode = text
        self._apply_default_label_template(text, reason="mode_switch")
        self._update_phase_panel_visibility()
        if self.mode != "Fine":
            self._phase_selected = None
            self._sync_anomaly_panel()
        self._rebuild_timeline_sources()
        self._apply_validation_timeline(self.validation_enabled)
        self._sync_entity_panel_mode()
        self._update_overlay_for_frame(getattr(self.player, "current_frame", 0))
        self._log("mode_change", mode=text)

    def _set_validation_button_state(self, on: bool):
        if not getattr(self, "btn_validation", None):
            return
        if getattr(self, "lbl_validation", None):
            if on:
                self.lbl_validation.setStyleSheet("color: #12b76a; font-weight: 600;")
            else:
                self.lbl_validation.setStyleSheet("")
        if on:
            self.btn_validation.setToolTip("Return to annotation mode")
        else:
            self.btn_validation.setToolTip("Toggle validation on/off")

    def _update_validation_overlay_controls(self) -> None:
        lbl = getattr(self, "lbl_overlay", None)
        combo = getattr(self, "combo_overlay", None)
        if lbl is None or combo is None:
            return
        show = self._is_action_task() and not self._is_psr_task()
        try:
            lbl.setVisible(show)
            combo.setVisible(show)
        except Exception:
            pass
        if not show:
            return
        phase_available = bool(self.mode == "Fine" and self.phase_mode_enabled)
        items = [("Action", "action")]
        if phase_available:
            items.append(("Phase", "phase"))
            items.append(("Action + Phase", "both"))
        combo.blockSignals(True)
        combo.clear()
        for text, val in items:
            combo.addItem(text, val)
        allowed = {val for _, val in items}
        if self._validation_overlay_mode not in allowed:
            self._validation_overlay_mode = "action"
        for i in range(combo.count()):
            if combo.itemData(i) == self._validation_overlay_mode:
                combo.setCurrentIndex(i)
                break
        combo.blockSignals(False)
        review_visible = False
        try:
            review_visible = bool(
                getattr(self, "review_panel", None) and self.review_panel.isVisible()
            )
        except Exception:
            review_visible = False
        enabled = bool(self.validation_enabled or review_visible)
        try:
            combo.setEnabled(enabled)
        except Exception:
            pass

    def _on_validation_overlay_changed(self, _idx: int) -> None:
        combo = getattr(self, "combo_overlay", None)
        if combo is None:
            return
        data = combo.currentData()
        if data:
            self._validation_overlay_mode = str(data)
        self._update_overlay_for_frame(getattr(self.player, "current_frame", 0))

    def _on_validation_toggled(self, on: bool):
        if on:
            name, ok = QInputDialog.getText(self, "Validation", "Editor name:")
            if not ok or not name.strip():
                self.btn_validation.blockSignals(True)
                self.btn_validation.setChecked(False)
                self.btn_validation.blockSignals(False)
                self._set_validation_button_state(False)
                self._log("validation_cancel")
                return
            self.validator_name = name.strip()
            self.validation_enabled = True
            self._normalize_sync_edit_selection()
            self.validation_modifications.clear()
            self.validation_errors.clear()
            self._rebuild_timeline_sources()
            self._update_view_highlight()
            self._sync_entity_panel_mode()
            self._update_overlay_for_frame(getattr(self.player, "current_frame", 0))
            self._apply_validation_timeline(True)
            self._set_validation_button_state(True)
            self._update_validation_overlay_controls()
            self._set_status(f"Validation ON ({self.validator_name})")
            self._log("validation_on", validator=self.validator_name)
        else:
            self.validation_enabled = False
            self._normalize_sync_edit_selection()
            self._rebuild_timeline_sources()
            self._update_view_highlight()
            self._sync_entity_panel_mode()
            self._update_overlay_for_frame(getattr(self.player, "current_frame", 0))
            self._apply_validation_timeline(False)
            self._set_validation_button_state(False)
            self._update_validation_overlay_controls()
            try:
                self._set_review_panel_visible(False)
            except Exception:
                pass
            self._set_status("Validation OFF")
            self._log("validation_off")

    def _apply_validation_timeline(self, on: bool):
        if not getattr(self, "timeline", None):
            return
        try:
            self.timeline.set_combined_editable(True)
        except Exception:
            pass
        if on and self.mode == "Coarse":
            if not self._validation_forced_layout:
                self._validation_prev_combined_text = getattr(
                    self.timeline, "_combined_show_text", True
                )
                self._validation_prev_center_single = getattr(
                    self.timeline, "_center_single_row", False
                )
            self._validation_forced_layout = True
            try:
                self.timeline.set_center_single_row(True)
            except Exception:
                pass
        else:
            if self._validation_forced_layout:
                try:
                    self.timeline.set_combined_label_text(
                        self._validation_prev_combined_text
                    )
                except Exception:
                    pass
                try:
                    self.timeline.set_center_single_row(
                        self._validation_prev_center_single
                    )
                except Exception:
                    pass
                self._validation_forced_layout = False
                self._validation_prev_combined_text = True
                self._validation_prev_center_single = False
            else:
                try:
                    self.timeline.set_center_single_row(False)
                except Exception:
                    pass

    def _rebuild_timeline_sources(self):
        if not getattr(self, "timeline", None):
            return
        self._apply_timeline_snap_settings(refresh=False)
        if self._is_psr_task():
            try:
                self.timeline.set_row_delete_handler(None)
            except Exception:
                pass
            try:
                self.timeline.set_row_split_handler(None)
            except Exception:
                pass
            try:
                self.timeline.set_row_segment_cuts_provider(None)
            except Exception:
                pass
            try:
                self.timeline.set_row_edit_mask_provider(None)
            except Exception:
                pass
            self._psr_mark_dirty()
            self._psr_refresh_state_timeline(force=True)
            return
        snap_cfg = self._timeline_snap_cfg()
        row_sources = []
        extra_sources = []
        has_extra = False
        if self.mode == "Coarse":
            for lb in self.labels:
                if is_extra_label(lb.name):
                    row_sources.append((lb, self.extra_store, ""))
                    extra_sources.append((lb, self.extra_store, ""))
                else:
                    row_sources.append((lb, self.store, ""))
            self.entities_panel.setVisible(False)
            has_extra = bool(getattr(self.extra_store, "frame_to_label", None))
        else:
            self.entities_panel.setVisible(True)
            for lb in self.labels:
                selected = self.label_entity_map.get(lb.name, set())
                if is_extra_label(lb.name):
                    row_sources.append((lb, self.extra_store, ""))
                    extra_sources.append((lb, self.extra_store, ""))
                    continue
                for ename in selected:
                    st = self.entity_stores.setdefault(ename, AnnotationStore())
                    row_sources.append((lb, st, f"[{ename}] "))
        self.timeline.set_row_sources(row_sources)
        combined_groups = None
        tail_groups = None
        if self.mode == "Fine":
            entity_names = [e.name for e in self.entities]
            for ename in sorted(self.entity_stores.keys()):
                if ename not in entity_names:
                    entity_names.append(ename)
            groups = []
            phase_groups = []
            for ename in entity_names:
                st = self.entity_stores.setdefault(ename, AnnotationStore())
                group_sources = []
                for lb in self.labels:
                    if is_extra_label(lb.name):
                        group_sources.append((lb, self.extra_store, ""))
                    else:
                        group_sources.append((lb, st, ""))
                groups.append(
                    (
                        ename,
                        group_sources,
                        {
                            "row_type": "action",
                            "entity": ename,
                            "labels": self.labels,
                            "segment_cuts": self._active_trim_cuts_for_descriptor(
                                {"kind": "entity", "entity": ename}
                            ),
                            "show_segment_cuts": True,
                            "row_height": FINE_ACTION_ROW_HEIGHT,
                        },
                    )
                )
                if self.phase_mode_enabled:
                    pstore = self._ensure_phase_store_for_entity(ename)
                    phase_sources = [(lb, pstore, "") for lb in self.phase_labels]
                    segs = [(s, e) for (s, e, _lb) in self._segments_from_store(st)]
                    phase_cuts = self._phase_anomaly_segment_cuts(ename)
                    trim_phase_cuts = self._active_trim_cuts_for_descriptor(
                        {"kind": "phase", "entity": ename}
                    )
                    phase_group = (
                        f"{ename} (phase)",
                        phase_sources,
                        {
                            "row_type": "phase",
                            "entity": ename,
                            "labels": self.phase_labels,
                            "snap_segments": segs,
                            "snap_mode": "soft",
                            "snap_radius": int(snap_cfg.get("phase_soft_radius", 8)),
                            "segment_cuts": sorted(
                                {
                                    int(c)
                                    for c in (phase_cuts or [])
                                    + (trim_phase_cuts or [])
                                }
                            ),
                            "show_segment_cuts": True,
                            "row_height": FINE_PHASE_ROW_HEIGHT,
                        },
                    )
                    groups.append(phase_group)
                    phase_groups.append(phase_group)
            combined_groups = groups if groups else None
            tail_groups = phase_groups if phase_groups else None
        elif has_extra and extra_sources:
            main_sources = [
                (lb, st, prefix)
                for (lb, st, prefix) in row_sources
                if not is_extra_label(lb.name)
            ]
            coarse_cuts = self._active_trim_cuts_for_descriptor({"kind": "store"})
            combined_groups = [
                (
                    "Timeline",
                    main_sources,
                    {
                        "show_extra_overlay": False,
                        "segment_cuts": coarse_cuts,
                        "show_segment_cuts": True,
                    },
                ),
                (
                    "Manual Segmentation",
                    extra_sources,
                    {
                        "show_extra_overlay": True,
                        "editable": False,
                        "show_segment_cuts": False,
                    },
                ),
            ]
        elif self.mode == "Coarse":
            coarse_cuts = self._active_trim_cuts_for_descriptor({"kind": "store"})
            combined_groups = [
                (
                    "Timeline",
                    row_sources,
                    {
                        "segment_cuts": coarse_cuts,
                        "show_segment_cuts": True,
                    },
                )
            ]
        try:
            self.timeline.set_combined_groups(combined_groups)
        except Exception:
            pass
        try:
            self.timeline.set_tail_combined_groups(tail_groups)
        except Exception:
            pass
        delete_handler = (
            self._on_action_segment_delete if self._is_action_task() else None
        )
        split_handler = self._on_action_segment_trim if self._is_action_task() else None

        def _row_cuts_provider(row):
            descriptor = self._mask_descriptor_from_row(row)
            if not self._syncable_descriptor(descriptor):
                return []
            return self._active_trim_cuts_for_descriptor(descriptor)

        try:
            self.timeline.set_combined_delete_handler(delete_handler)
        except Exception:
            pass
        try:
            self.timeline.set_row_delete_handler(delete_handler)
        except Exception:
            pass
        try:
            self.timeline.set_combined_split_handler(split_handler)
        except Exception:
            pass
        try:
            self.timeline.set_row_split_handler(split_handler)
        except Exception:
            pass
        try:
            self.timeline.set_row_segment_cuts_provider(_row_cuts_provider)
        except Exception:
            pass
        self._apply_sync_edit_masks()
        try:
            if (
                self.mode == "Fine"
                and getattr(self.timeline, "layout_mode", "") == "combined"
            ):
                group_count = len(combined_groups or [])
                self.timeline.set_center_single_row(group_count <= 1)
            elif not (self.validation_enabled and self.mode == "Coarse"):
                self.timeline.set_center_single_row(False)
        except Exception:
            pass
        try:
            self._update_left_splitter_sizes()
        except Exception:
            pass

    def _update_left_splitter_sizes(self):
        try:
            self._update_left_panel_min_height()
        except Exception:
            pass
        sp = getattr(self, "left_splitter", None)
        if sp is None:
            return
        sizes = sp.sizes()
        total = sum(sizes) if sizes else 0
        if total <= 0:
            total = sp.height() or 480
        count = len(sizes) if sizes else 0
        phase_visible = bool(
            getattr(self, "phase_panel", None) and self.phase_panel.isVisible()
        )
        if self.mode == "Fine":
            min_top = 90
            min_mid = 120
            # only seed a default if the top areas are collapsed
            if sizes:
                if phase_visible:
                    if sizes[0] >= min_top and (count < 3 or sizes[1] >= min_mid):
                        return
                else:
                    if sizes[0] >= min_top:
                        return
            if count >= 3:
                top = max(min_top, int(total * 0.24))
                if phase_visible:
                    mid = max(min_mid, int(total * 0.20))
                else:
                    mid = 0
                sp.setSizes([top, mid, max(80, total - top - mid)])
            else:
                top = max(min_top, int(total * 0.30))
                sp.setSizes([top, max(80, total - top)])
        else:
            if count >= 3:
                sp.setSizes([0, 0, max(1, total)])
            else:
                sp.setSizes([0, max(1, total)])

    def _update_left_panel_min_height(self) -> None:
        left = getattr(self, "left_splitter", None)
        if left is None:
            return
        total = 0
        baselines = {
            "entities_panel": 90,
            "phase_panel": 110,
            "panel": 380,
        }
        visible_widgets = []
        for name, w in (
            ("entities_panel", getattr(self, "entities_panel", None)),
            ("phase_panel", getattr(self, "phase_panel", None)),
            ("panel", getattr(self, "panel", None)),
        ):
            if w is None or not w.isVisible():
                continue
            visible_widgets.append(w)
            h = 0
            try:
                h = w.minimumSizeHint().height()
            except Exception:
                h = 0
            if h <= 0:
                try:
                    h = w.sizeHint().height()
                except Exception:
                    h = 0
            if h <= 0:
                try:
                    h = w.minimumHeight()
                except Exception:
                    h = 0
            baseline = baselines.get(name, 0)
            total += max(int(h), int(baseline))
        if visible_widgets:
            try:
                handle = left.handleWidth() or 4
            except Exception:
                handle = 4
            total += max(0, (len(visible_widgets) - 1) * int(handle))
        try:
            left.setMinimumHeight(max(0, int(total)))
        except Exception:
            pass

    def _init_main_splitter_sizes(self):
        sp = getattr(self, "splitter_main", None)
        if sp is None or getattr(self, "_splitter_init_done", False):
            return
        total = sp.height()
        if total <= 0:
            total = max(400, self.height() - 40)
        top = int(total * 0.6)
        sp.setSizes([top, max(1, total - top)])
        self._splitter_init_done = True

    def _hover_preview_indices(self) -> List[int]:
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return []
        hover_cfg = self._timeline_hover_preview_cfg()
        if hover_cfg.get("enabled_multi", True) and self._multiview_sync_active():
            indices = self._effective_sync_edit_indices()
            if indices:
                return [int(i) for i in indices if 0 <= int(i) < len(self.views)]
        return [int(self.active_view_idx)]

    def _flush_timeline_hover_preview(self):
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return
        frame = self._hover_preview_pending_frame
        self._hover_preview_pending_frame = None
        if frame is None:
            return
        active_idx = int(self.active_view_idx)
        active_view = self.views[active_idx]
        active_player = active_view.get("player")
        if not active_player or not getattr(active_player, "cap", None):
            return
        if getattr(active_player, "is_playing", False):
            return

        if int(frame) < 0:
            self._hover_preview_last_frame = None
            reset_indices = sorted(set(self._hover_preview_last_targets.keys()))
            if active_idx not in reset_indices:
                reset_indices.append(active_idx)
            self._hover_preview_last_targets = {}
            for idx in reset_indices:
                if not (0 <= idx < len(self.views)):
                    continue
                player = self.views[idx].get("player")
                if not player or not getattr(player, "cap", None):
                    continue
                try:
                    target = int(getattr(player, "current_frame", 0))
                    target = max(player.crop_start, min(target, player.crop_end))
                except Exception:
                    target = int(getattr(player, "current_frame", 0))
                try:
                    player.seek(target, preview_only=True)
                    self._update_overlay_for_view(idx, target)
                except Exception:
                    continue
            return

        try:
            active_target = int(frame)
        except Exception:
            return
        self._hover_preview_last_frame = int(active_target)
        try:
            active_target = max(
                int(active_player.crop_start),
                min(active_target, int(active_player.crop_end)),
            )
        except Exception:
            pass

        indices = self._hover_preview_indices()
        if active_idx not in indices:
            indices = [active_idx] + [i for i in indices if i != active_idx]
        if not indices:
            indices = [active_idx]
        hover_cfg = self._timeline_hover_preview_cfg()
        align_mode = str(hover_cfg.get("align", "absolute") or "absolute").lower()
        base_start = int(active_view.get("start", 0))
        keep = set(indices)
        dropped = [k for k in self._hover_preview_last_targets.keys() if k not in keep]
        for idx in dropped:
            if not (0 <= idx < len(self.views)):
                continue
            player = self.views[idx].get("player")
            if not player or not getattr(player, "cap", None):
                continue
            try:
                fallback = int(getattr(player, "current_frame", 0))
                fallback = max(player.crop_start, min(fallback, player.crop_end))
                player.seek(fallback, preview_only=True)
                self._update_overlay_for_view(idx, fallback)
            except Exception:
                continue
        self._hover_preview_last_targets = {
            k: v for k, v in self._hover_preview_last_targets.items() if k in keep
        }

        for idx in indices:
            if not (0 <= idx < len(self.views)):
                continue
            vw = self.views[idx]
            player = vw.get("player")
            if not player or not getattr(player, "cap", None):
                continue
            if idx == active_idx:
                target = int(active_target)
            else:
                if align_mode == "offset":
                    offset = int(active_target) - base_start
                    target = int(vw.get("start", 0)) + offset
                    target = max(
                        int(vw.get("start", 0)),
                        min(target, int(vw.get("end", vw.get("start", 0)))),
                    )
                else:
                    target = int(active_target)
                    try:
                        target = max(
                            int(player.crop_start), min(target, int(player.crop_end))
                        )
                    except Exception:
                        pass
            if self._hover_preview_last_targets.get(idx) == target:
                continue
            try:
                player.seek(target, preview_only=True)
                self._update_overlay_for_view(idx, target)
                self._hover_preview_last_targets[idx] = target
            except Exception:
                continue

    def _on_timeline_hover_frame(self, frame: int):
        if not self.views:
            return
        if not (0 <= self.active_view_idx < len(self.views)):
            return
        player = self.views[self.active_view_idx].get("player")
        if not player or not getattr(player, "cap", None):
            return
        if getattr(player, "is_playing", False):
            return
        target = int(frame) if frame is not None else -1
        if self._hover_preview_timer.isActive():
            if self._hover_preview_pending_frame == target:
                return
        else:
            if (
                target >= 0
                and self._hover_preview_last_frame is not None
                and int(self._hover_preview_last_frame) == target
            ):
                return
        self._hover_preview_pending_frame = target
        if not self._hover_preview_timer.isActive():
            self._hover_preview_timer.start()

    # ---------- Transcript / ThinkAloud helper ----------
    def _open_transcript_workspace(self):
        if not getattr(self, "video_path", None):
            QMessageBox.information(self, "Transcript", "Load a video first.")
            return False
        try:
            self._toggle_asr_panel(True)
        except Exception:
            pass
        seg_count = len(self.asr_segments or [])
        if seg_count:
            self._set_status(
                f"Transcript workspace ready. {seg_count} transcript segment(s) are loaded."
            )
        else:
            self._set_status(
                "Transcript workspace ready. Generate or import a transcript to start."
            )
        return True

    def _on_transcript_generate_clicked(self):
        if not self._open_transcript_workspace():
            return
        self._run_external_asr(prompt_for_label=False, reveal_panel=True)

    def _apply_loaded_transcript_to_label(self):
        segs = list(self.asr_segments or [])
        if not segs:
            QMessageBox.information(
                self,
                "Transcript",
                "Generate or import a transcript before applying it to a label.",
            )
            return
        label_name = self._ask_label_for_asr()
        if not label_name:
            self._set_status("Transcript-to-label step skipped.")
            return
        self._apply_segments_to_label(segs, label_name)
        self._set_status(
            f"Created intervals for label '{label_name}' from transcript."
        )

    def _clear_transcript_workspace(self):
        had_segments = bool(self.asr_segments)
        self.asr_segments = []
        self.asr_lang = "auto"
        self.timeline.set_subtitle_tracks([])
        self._refresh_asr_panels()
        if had_segments:
            self._set_status("Transcript workspace cleared.")
            self._log("asr_clear")
        else:
            self._set_status("Transcript workspace is already empty.")

    def _transcript_source_fingerprint(self, src_media: str) -> str:
        norm = os.path.abspath(os.path.expanduser(str(src_media or "")))
        try:
            st = os.stat(norm)
            payload = f"{norm}|{int(st.st_mtime_ns)}|{int(st.st_size)}"
        except Exception:
            payload = norm
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _transcript_output_cache_key(self, src_media: str, lang_code: str, cmd_tmpl: str) -> str:
        payload = "||".join(
            [
                self._transcript_source_fingerprint(src_media),
                str(lang_code or "auto"),
                str(cmd_tmpl or "").strip(),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def _transcript_output_cache_path(
        self, src_media: str, lang_code: str, cmd_tmpl: str
    ) -> str:
        token = self._transcript_output_cache_key(src_media, lang_code, cmd_tmpl)
        return os.path.join(tempfile.gettempdir(), f"_asr_out_{token}.json")

    def _transcript_segments_cache_key(
        self, src_media: str, lang_code: str, cmd_tmpl: str, fps: float
    ) -> str:
        return (
            f"{self._transcript_output_cache_key(src_media, lang_code, cmd_tmpl)}"
            f"|fps={float(fps):.6f}"
        )

    def _run_external_asr(self, prompt_for_label: bool = True, reveal_panel: bool = True):
        """Run speech transcription via external command defined in env ASR_CMD, or load transcript JSON manually."""
        if not getattr(self, "video_path", None):
            QMessageBox.information(self, "Transcript", "Load a video first.")
            return
        if reveal_panel:
            try:
                self._toggle_asr_panel(True)
            except Exception:
                pass
        # language selection
        langs = ["Auto (multi)", "English (en)", "Deutsch (de)", "Chinese (zh)"]
        choice, ok = QInputDialog.getItem(
            self,
            "Transcript Language",
            "Choose language (or Auto):",
            langs,
            0,
            False,
        )
        if not ok:
            self._set_status("Transcript cancelled.")
            return
        lang_map = {
            "Auto (multi)": "auto",
            "English (en)": "en",
            "Deutsch (de)": "de",
            "Chinese (zh)": "zh",
        }
        lang_code = lang_map.get(choice, "auto")

        cmd_tmpl = os.environ.get("ASR_CMD", "").strip()
        fps = max(1, self.player.frame_rate)
        segments = []
        mode = "cmd" if cmd_tmpl else "manual"
        self._log("asr_request", lang=lang_code, mode=mode)

        if not cmd_tmpl:
            # fallback: let user load a transcript JSON
            fp, _ = QFileDialog.getOpenFileName(
                self, "Choose transcript JSON", "", "JSON Files (*.json);;All Files (*)"
            )
            if not fp:
                QMessageBox.information(
                    self,
                    "Transcript",
                    "Please set environment variable ASR_CMD or load a transcript JSON.",
                )
                self._log("asr_fail", reason="no_cmd_no_file")
                return
            try:
                segments = self._load_transcript_json(fp, fps)
            except Exception as ex:
                QMessageBox.warning(
                    self, "Transcript", f"Failed to parse transcript:\n{ex}"
                )
                self._log("asr_fail", reason="parse_manual", detail=str(ex)[:180])
                return
        else:
            # ensure we have a wav input
            src_media = getattr(self.player, "_audio_path", None) or self.video_path
            ok, wav_tmp, elog, reused_audio = ensure_cached_wav_16k_mono_verbose(
                src_media
            )
            if not ok:
                QMessageBox.warning(
                    self, "Transcript", f"Audio extraction failed:\n{elog}"
                )
                self._log("asr_fail", reason="extract_fail", detail=str(elog)[:180])
                return
            cache_key = self._transcript_segments_cache_key(
                src_media, lang_code, cmd_tmpl, fps
            )
            cached_segments = self._transcript_result_cache.get(cache_key)
            if cached_segments:
                segments = list(cached_segments)
                mode = "cmd-cache"
                self._set_status("Transcript ready from in-memory cache.")
            else:
                out_json = self._transcript_output_cache_path(
                    src_media, lang_code, cmd_tmpl
                )
                if os.path.isfile(out_json):
                    try:
                        segments = self._load_transcript_json(out_json, fps)
                        mode = "cmd-cache"
                        self._set_status("Transcript ready from cached output.")
                    except Exception:
                        segments = []
                if not segments:
                    cmd = cmd_tmpl.format(audio=wav_tmp, lang=lang_code, out=out_json)
                    if reused_audio:
                        self._set_status("Reusing cached audio. Running transcript command...")
                    try:
                        proc = subprocess.run(
                            cmd, shell=True, capture_output=True, text=True, timeout=3600
                        )
                    except Exception as ex:
                        QMessageBox.warning(
                            self,
                            "Transcript",
                            f"Failed to run transcription command:\n{ex}",
                        )
                        self._log("asr_fail", reason="run_error", detail=str(ex)[:180])
                        return
                    if proc.returncode != 0:
                        QMessageBox.warning(
                            self,
                            "Transcript",
                            f"Transcription command failed:\n{proc.stderr or proc.stdout}",
                        )
                        self._log("asr_fail", reason="cmd_nonzero", code=proc.returncode)
                        return
                    try:
                        segments = self._load_transcript_json(
                            out_json, fps, fallback_stdout=proc.stdout
                        )
                    except Exception as ex:
                        QMessageBox.warning(
                            self, "Transcript", f"Failed to parse transcript output:\n{ex}"
                        )
                        self._log("asr_fail", reason="parse_cmd", detail=str(ex)[:180])
                        return
                if segments:
                    self._transcript_result_cache[cache_key] = list(segments)

        if not segments:
            QMessageBox.information(self, "Transcript", "No speech segments found.")
            self._log("asr_no_segments", lang=lang_code)
            return

        self.asr_segments = segments
        self.asr_lang = lang_code
        track = TranscriptTrack(
            name=f"Transcript: {lang_code.upper() if lang_code != 'auto' else 'AUTO'}",
            segments=segments,
        )
        self.timeline.set_subtitle_tracks([track])
        self._refresh_asr_panels()
        self._set_status(f"Transcript track added: {track.name}")
        self._log("asr_ready", lang=lang_code, segments=len(segments), mode=mode)

        # optional: create label intervals
        if prompt_for_label:
            label_name = self._ask_label_for_asr()
            if label_name:
                self._apply_segments_to_label(segments, label_name)
                self._set_status(
                    f"Created intervals for label '{label_name}' from transcript."
                )

    def _load_transcript_json(self, path: str, fps: int, fallback_stdout: str = ""):
        """Parse transcript JSON -> List[TranscriptSegment]. Expected list of {start,end,text,lang} in seconds."""
        import json

        raw = None
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except Exception as ex:
                raise ValueError(f"Failed to parse transcript JSON file: {ex}") from ex
        elif fallback_stdout:
            try:
                raw = json.loads(fallback_stdout)
            except Exception:
                raw = None
        if not raw:
            raise ValueError("No JSON content to parse.")
        segs = []
        for item in raw:
            try:
                st = float(item.get("start", 0.0))
                ed = float(item.get("end", st))
                txt = str(item.get("text", "")).strip()
                lang = (item.get("lang") or item.get("language") or "auto").lower()
                ss = max(0, int(round(st * fps)))
                ee = max(ss, int(round(ed * fps)) - 1)
                segs.append(
                    TranscriptSegment(start_frame=ss, end_frame=ee, text=txt, lang=lang)
                )
            except Exception:
                continue
        return segs

    def _ask_label_for_asr(self) -> str:
        if not self.labels:
            return ""
        choices = ["Skip (do not create labels)"] + [lb.name for lb in self.labels]
        choice, ok = QInputDialog.getItem(
            self,
            "Apply label to transcript segments",
            "Choose label (or Skip):",
            choices,
            0,
            False,
        )
        if not ok or choice.startswith("Skip"):
            return ""
        return choice

    def _apply_segments_to_label(self, segs: list, label_name: str):
        st = self.views[self.active_view_idx]["store"]
        for seg in segs:
            try:
                s = int(seg.start_frame)
                e = int(seg.end_frame)
            except Exception:
                continue
            if s > e:
                s, e = e, s
            for f in range(s, e + 1):
                st.add(label_name, f)
        self._on_store_changed()
        self.timeline.refresh_all_rows()
        self._log("asr_apply_label", label=label_name, segments=len(segs))

    # ---------- Transcript UI helpers ----------
    def _fmt_time(self, frames: int) -> str:
        fps = max(1, self._get_fps())
        seconds = frames / float(fps)
        m = int(seconds // 60)
        s = seconds - m * 60
        return f"{m:02d}:{s:05.2f}"

    def _fmt_transcript_delta(self, frames: int) -> str:
        fps = max(1, self._get_fps())
        seconds = max(0.0, frames / float(fps))
        if seconds >= 60.0:
            minutes = int(seconds // 60)
            remain = seconds - (minutes * 60)
            return f"{minutes}m {remain:04.1f}s"
        return f"{seconds:.2f}s"

    def _transcript_excerpt(self, text: str, limit: int = 72) -> str:
        txt = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(txt) <= limit:
            return txt
        return txt[: max(0, limit - 1)].rstrip() + "..."

    def _apply_transcript_item_style(self, frame: QWidget, active: bool = False):
        if frame is None:
            return
        if active:
            frame.setStyleSheet(
                """
                QFrame#transcriptCueCard {
                    background: #e0f2fe;
                    border: 1px solid #38bdf8;
                    border-left: 5px solid #0284c7;
                    border-radius: 10px;
                }
                QLabel#transcriptCueIndex {
                    color: #075985;
                    background: #bae6fd;
                    border-radius: 8px;
                    font-weight: 700;
                    padding: 2px 6px;
                }
                QLabel#transcriptCueSpan {
                    color: #0f172a;
                    font-weight: 600;
                }
                QLabel#transcriptCueMeta {
                    color: #0c4a6e;
                }
                QLabel#transcriptCueText {
                    color: #082f49;
                    font-weight: 600;
                }
                """
            )
        else:
            frame.setStyleSheet(
                """
                QFrame#transcriptCueCard {
                    background: #f8fafc;
                    border: 1px solid #e4e7ec;
                    border-left: 5px solid #d0d5dd;
                    border-radius: 10px;
                }
                QLabel#transcriptCueIndex {
                    color: #475467;
                    background: #eaecf0;
                    border-radius: 8px;
                    font-weight: 700;
                    padding: 2px 6px;
                }
                QLabel#transcriptCueSpan {
                    color: #344054;
                    font-weight: 600;
                }
                QLabel#transcriptCueMeta {
                    color: #667085;
                }
                QLabel#transcriptCueText {
                    color: #101828;
                    font-weight: 600;
                }
                """
            )

    def _build_transcript_list_item_widget(
        self,
        idx: int,
        start: int,
        end: int,
        text: str,
        lang: str,
        gap_before: int = 0,
        active: bool = False,
    ) -> QWidget:
        frame = QFrame(self.list_asr)
        frame.setObjectName("transcriptCueCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)

        lbl_idx = QLabel(f"{idx + 1:02d}")
        lbl_idx.setObjectName("transcriptCueIndex")
        top_row.addWidget(lbl_idx)

        lbl_span = QLabel(f"{self._fmt_time(start)} - {self._fmt_time(end)}")
        lbl_span.setObjectName("transcriptCueSpan")
        top_row.addWidget(lbl_span)

        top_row.addStretch(1)

        duration = max(0, end - start)
        lbl_meta = QLabel(f"{self._fmt_transcript_delta(duration)} | {str(lang).upper()}")
        lbl_meta.setObjectName("transcriptCueMeta")
        top_row.addWidget(lbl_meta)

        layout.addLayout(top_row)

        lbl_text = QLabel(self._transcript_excerpt(text))
        lbl_text.setObjectName("transcriptCueText")
        lbl_text.setWordWrap(True)
        lbl_text.setToolTip(str(text or "").strip())
        layout.addWidget(lbl_text)

        if gap_before > 0:
            lbl_gap = QLabel(f"Gap before: {self._fmt_transcript_delta(gap_before)}")
            lbl_gap.setObjectName("transcriptCueMeta")
            layout.addWidget(lbl_gap)

        self._apply_transcript_item_style(frame, active=active)
        return frame

    def _transcript_find_segment_index(self, frame: int) -> int:
        for idx, seg in enumerate(self.asr_segments or []):
            try:
                start = int(seg.start_frame)
                end = int(seg.end_frame)
            except Exception:
                continue
            if start <= frame <= end:
                return idx
        return -1

    def _transcript_neighbor_indices(self, frame: int) -> Tuple[int, int]:
        prev_idx = -1
        next_idx = -1
        for idx, seg in enumerate(self.asr_segments or []):
            try:
                start = int(seg.start_frame)
                end = int(seg.end_frame)
            except Exception:
                continue
            if end < frame:
                prev_idx = idx
                continue
            if start > frame:
                next_idx = idx
                break
        return prev_idx, next_idx

    def _jump_to_transcript_index(self, idx: int):
        segs = list(self.asr_segments or [])
        if idx < 0 or idx >= len(segs):
            return
        try:
            frame = int(segs[idx].start_frame)
        except Exception:
            return
        self._jump_to_transcript_frame(frame, idx=idx)

    def _jump_to_transcript_frame(self, frame: int, idx: Optional[int] = None):
        self._sync_views_to_frame(frame, preview_only=False)
        self._timeline_auto_follow = True
        self.timeline.set_current_frame(frame, follow=True)
        self.timeline.set_current_hits(self._hit_names_for_frame(frame))
        self._update_transcript_workspace_for_frame(frame, ensure_visible=True)
        log_payload = {"frame": frame}
        if idx is not None:
            log_payload["index"] = idx
        self._log("asr_jump", **log_payload)

    def _jump_to_prev_transcript(self):
        frame = int(getattr(self.player, "current_frame", 0))
        active_idx = self._transcript_find_segment_index(frame)
        if active_idx > 0:
            self._jump_to_transcript_index(active_idx - 1)
            return
        prev_idx, _ = self._transcript_neighbor_indices(frame)
        if prev_idx >= 0:
            self._jump_to_transcript_index(prev_idx)
            return
        self._set_status("Already at the first transcript cue.")

    def _jump_to_playhead_transcript(self):
        frame = int(getattr(self.player, "current_frame", 0))
        active_idx = self._transcript_find_segment_index(frame)
        if active_idx >= 0:
            self._jump_to_transcript_index(active_idx)
            return
        _, next_idx = self._transcript_neighbor_indices(frame)
        if next_idx >= 0:
            self._jump_to_transcript_index(next_idx)
            return
        self._set_status("No transcript cue is available around the current playhead.")

    def _jump_to_next_transcript(self):
        frame = int(getattr(self.player, "current_frame", 0))
        active_idx = self._transcript_find_segment_index(frame)
        if active_idx >= 0 and active_idx + 1 < len(self.asr_segments or []):
            self._jump_to_transcript_index(active_idx + 1)
            return
        _, next_idx = self._transcript_neighbor_indices(frame)
        if next_idx >= 0:
            self._jump_to_transcript_index(next_idx)
            return
        self._set_status("Already at the last transcript cue.")

    def _update_transcript_workspace_for_frame(
        self, frame: int, ensure_visible: bool = False
    ):
        segs = list(self.asr_segments or [])
        prev_active_idx = int(getattr(self, "_transcript_current_index", -1))
        active_idx = self._transcript_find_segment_index(frame)
        prev_idx, next_idx = self._transcript_neighbor_indices(frame)
        self._transcript_current_index = active_idx
        list_widget = getattr(self, "list_asr", None)
        if list_widget is not None:
            list_widget.blockSignals(True)
            if active_idx >= 0:
                list_widget.setCurrentRow(active_idx)
            else:
                list_widget.clearSelection()
                list_widget.setCurrentItem(None)
            list_widget.blockSignals(False)
            if active_idx >= 0 and (ensure_visible or active_idx != prev_active_idx):
                item = list_widget.item(active_idx)
                if item is not None:
                    list_widget.scrollToItem(item, QAbstractItemView.PositionAtCenter)
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                widget = list_widget.itemWidget(item)
                if widget is not None:
                    self._apply_transcript_item_style(widget, active=(i == active_idx))

        if not segs:
            self.lbl_transcript_focus_title.setText("Current Cue")
            self.lbl_transcript_focus_time.setText("--")
            self.lbl_transcript_focus_text.setText("Generate or import a transcript.")
            self.lbl_transcript_focus_meta.setText("Follows the playhead.")
            if getattr(self, "btn_transcript_prev", None):
                self.btn_transcript_prev.setEnabled(False)
            if getattr(self, "btn_transcript_focus", None):
                self.btn_transcript_focus.setEnabled(False)
            if getattr(self, "btn_transcript_next", None):
                self.btn_transcript_next.setEnabled(False)
            return

        if active_idx >= 0:
            seg = segs[active_idx]
            start = int(seg.start_frame)
            end = int(seg.end_frame)
            text = str(getattr(seg, "text", "")).strip() or "(empty transcript line)"
            cue_pos = max(0, frame - start)
            cue_left = max(0, end - frame)
            self.lbl_transcript_focus_title.setText(
                f"Cue {active_idx + 1} of {len(segs)}"
            )
            self.lbl_transcript_focus_time.setText(
                f"{self._fmt_time(start)} - {self._fmt_time(end)}"
            )
            self.lbl_transcript_focus_text.setText(text)
            self.lbl_transcript_focus_meta.setText(
                f"Elapsed {self._fmt_transcript_delta(cue_pos)} | Left {self._fmt_transcript_delta(cue_left)}"
            )
        else:
            self.lbl_transcript_focus_title.setText("No Active Cue")
            self.lbl_transcript_focus_time.setText(self._fmt_time(frame))
            if next_idx >= 0:
                seg = segs[next_idx]
                start = int(seg.start_frame)
                text = str(getattr(seg, "text", "")).strip() or "(empty transcript line)"
                self.lbl_transcript_focus_text.setText(text)
                self.lbl_transcript_focus_meta.setText(
                    f"Next in {self._fmt_transcript_delta(start - frame)} at {self._fmt_time(start)}"
                )
            elif prev_idx >= 0:
                seg = segs[prev_idx]
                end = int(seg.end_frame)
                text = str(getattr(seg, "text", "")).strip() or "(empty transcript line)"
                self.lbl_transcript_focus_text.setText(text)
                self.lbl_transcript_focus_meta.setText(
                    f"Last ended {self._fmt_transcript_delta(frame - end)} ago at {self._fmt_time(end)}"
                )
            else:
                self.lbl_transcript_focus_text.setText("No transcript cue around the playhead.")
                self.lbl_transcript_focus_meta.setText("Load a transcript to start.")
        if getattr(self, "btn_transcript_prev", None):
            self.btn_transcript_prev.setEnabled(prev_idx >= 0 or active_idx > 0)
        if getattr(self, "btn_transcript_focus", None):
            self.btn_transcript_focus.setEnabled(bool(segs))
        if getattr(self, "btn_transcript_next", None):
            self.btn_transcript_next.setEnabled(
                next_idx >= 0 or (active_idx >= 0 and active_idx + 1 < len(segs))
            )

    def _on_asr_item_activated(self, item: QListWidgetItem):
        data = item.data(Qt.UserRole)
        try:
            frame = int(data.get("start_frame", 0))
            idx = int(self.list_asr.row(item))
        except Exception:
            frame = 0
            idx = -1
        if idx >= 0:
            self._jump_to_transcript_index(idx)
            return
        self._jump_to_transcript_frame(frame)

    def _refresh_asr_panels(self):
        # list panel
        try:
            current_frame = int(getattr(self.player, "current_frame", 0))
        except Exception:
            current_frame = 0
        active_idx = self._transcript_find_segment_index(current_frame)
        self.list_asr.blockSignals(True)
        self.list_asr.clear()
        total_span_start = None
        total_span_end = None
        largest_gap = 0
        prev_end = None
        for idx, seg in enumerate(self.asr_segments or []):
            try:
                s = int(seg.start_frame)
                e = int(seg.end_frame)
                txt = seg.text
                lang = getattr(seg, "lang", "auto")
            except Exception:
                continue
            if total_span_start is None:
                total_span_start = s
            total_span_end = e
            gap_before = 0
            if prev_end is not None:
                gap_before = max(0, s - prev_end - 1)
                largest_gap = max(largest_gap, gap_before)
            prev_end = e
            item = QListWidgetItem()
            item.setData(
                Qt.UserRole,
                {"start_frame": s, "end_frame": e, "index": idx},
            )
            widget = self._build_transcript_list_item_widget(
                idx=idx,
                start=s,
                end=e,
                text=txt,
                lang=lang,
                gap_before=gap_before,
                active=(idx == active_idx),
            )
            item.setSizeHint(widget.sizeHint())
            self.list_asr.addItem(item)
            self.list_asr.setItemWidget(item, widget)
        self.list_asr.blockSignals(False)
        seg_count = len(self.asr_segments or [])
        if getattr(self, "lbl_asr_summary", None):
            if seg_count:
                lang = (self.asr_lang or "auto").upper()
                self.lbl_asr_summary.setText(
                    f"{seg_count} cues | {lang} | {self._fmt_time(total_span_start or 0)} - {self._fmt_time(total_span_end or 0)} | max gap {self._fmt_transcript_delta(largest_gap)}"
                )
            else:
                self.lbl_asr_summary.setText("No transcript loaded.")
        if getattr(self, "btn_transcript_apply", None):
            self.btn_transcript_apply.setEnabled(seg_count > 0)
        if getattr(self, "btn_transcript_clear", None):
            self.btn_transcript_clear.setEnabled(seg_count > 0)
        if getattr(self, "lbl_asr_list_title", None):
            self.lbl_asr_list_title.setText(
                "Cues" if not seg_count else f"Cues ({seg_count})"
            )
        self._update_transcript_workspace_for_frame(
            current_frame, ensure_visible=seg_count > 0
        )

    def _toggle_asr_panel(self, on: bool):
        """Hide/show the ASR panel inside the splitter. When hidden, the splitter sets the ASR width to 0; when shown, restore the last remembered width."""
        if not getattr(self, "video_split", None):
            return
        sizes = self.video_split.sizes()
        if not on:
            # store current ASR width to restore later
            if len(sizes) >= 3:
                self._asr_last_size = max(200, sizes[2] or self._asr_last_size)
            total = sum(sizes) if sum(sizes) > 0 else max(1, self.video_split.width())
            left0 = sizes[0] if len(sizes) >= 1 else int(total * 0.6)
            left1 = sizes[1] if len(sizes) >= 2 else int(total * 0.25)
            stub_w = 28
            new_sizes = [left0, left1, 0, stub_w]
            self.asr_panel.setMinimumWidth(0)
            self.asr_panel.setVisible(False)
            if getattr(self, "asr_stub", None):
                self.asr_stub.setVisible(True)
            self.video_split.setSizes(new_sizes)
            # keep header toggle state in sync
            try:
                self.btn_toggle_asr_panel.blockSignals(True)
                self.btn_toggle_asr_panel.setChecked(False)
                self.btn_toggle_asr_panel.setText("Show")
                self.btn_toggle_asr_panel.blockSignals(False)
            except Exception:
                pass
        else:
            # restore last size
            total = sum(sizes) if sum(sizes) > 0 else max(1, self.video_split.width())
            target = max(200, getattr(self, "_asr_last_size", 320))
            min_other = 120
            if total > min_other and target > total - min_other:
                target = max(120, total - min_other)
            elif total <= min_other:
                target = max(120, int(total * 0.3))
            left0 = sizes[0] if len(sizes) >= 1 else int(total * 0.6)
            left1 = sizes[1] if len(sizes) >= 2 else int(total * 0.25)
            rem = max(0, total - target)
            if left0 + left1 > 0:
                new_left0 = int(rem * (left0 / float(left0 + left1)))
            else:
                new_left0 = int(rem * 0.7)
            new_left1 = rem - new_left0
            new_sizes = [new_left0, new_left1, target, 0]
            try:
                self.asr_panel.setMinimumWidth(120)
            except Exception:
                pass
            self.asr_panel.setVisible(True)
            if getattr(self, "asr_stub", None):
                self.asr_stub.setVisible(False)
            self.video_split.setSizes(new_sizes)
            try:
                self.btn_toggle_asr_panel.blockSignals(True)
                self.btn_toggle_asr_panel.setChecked(True)
                self.btn_toggle_asr_panel.setText("Hide")
                self.btn_toggle_asr_panel.blockSignals(False)
            except Exception:
                pass

    # ===== JSON I/O =====
    def _save_validation_log(
        self,
        annotations_path: str,
        video_id: str,
        saved_paths: Optional[List[str]] = None,
    ):
        if not bool(getattr(self, "_validation_summary_enabled", True)):
            return
        if not (
            self.validation_enabled
            or self.validation_modifications
            or self.validation_errors
        ):
            return
        date_str = datetime.now().strftime("%Y%m%d")
        editor = self.validator_name or "unknown"
        paths = [p for p in (saved_paths or []) if p] or (
            [annotations_path] if annotations_path else []
        )
        base_name = (
            os.path.splitext(os.path.basename(paths[0]))[0] if paths else "annotations"
        )
        subject = f"{base_name}_{date_str}_{editor}"

        lines = ["Validation log"]
        if len(paths) > 1:
            lines.append("Annotations files:")
            for p in paths:
                lines.append(f"- {p}")
        elif paths:
            lines.append(f"Annotations file: {paths[0]}")
        else:
            lines.append("Annotations file: (none)")
        lines.extend(
            [
                f"Video ID: {video_id}",
                f"Editor: {editor}",
                f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "Modifications:",
            ]
        )
        if self.validation_modifications:
            for m in self.validation_modifications:
                if m.get("kind") == "psr":
                    start = m.get("start")
                    end = m.get("end")
                    if start is None and end is None:
                        span_txt = "n/a"
                    else:
                        _s, _e, span_txt = self._norm_span(start, end)
                    vname = m.get("view") or self.views[self.active_view_idx].get(
                        "name", "view"
                    )
                    comp = m.get("component") or m.get("component_id") or "combined"
                    old = m.get("old") or "-"
                    new = m.get("new") or "-"
                    action = m.get("action") or "change"
                    note = m.get("note")
                    extra = f" note: {note}" if note else ""
                    lines.append(
                        f"- [{m.get('ts')}] view={vname} PSR {comp} frames {span_txt}: {old} -> {new} action={action} (by {m.get('editor')}){extra}"
                    )
                else:
                    s, e, span_txt = self._norm_span(m.get("start"), m.get("end"))
                    comment = m.get("comment")
                    extra = f" comment: {comment}" if comment else ""
                    vname = m.get("view") or self.views[self.active_view_idx].get(
                        "name", "view"
                    )
                    lines.append(
                        f"- [{m.get('ts')}] view={vname} {m.get('store')} frames {span_txt}: {m.get('old')} -> {m.get('new')} (by {m.get('editor')}){extra}"
                    )
        else:
            lines.append("- None")

        lines.append("")
        lines.append("Errors:")
        if self.validation_errors:
            for e in self.validation_errors:
                vname = e.get("view") or self.views[self.active_view_idx].get(
                    "name", "view"
                )
                lines.append(
                    f"- [{e.get('ts')}] view={vname} frame {e.get('frame')}: {e.get('desc')} (by {e.get('editor')})"
                )
        else:
            lines.append("- None")

        if self.review_items:
            lines.append("")
            lines.append("Review decisions:")
            for item in self.review_items:
                decision = item.get("decision")
                s, e, span_txt = self._norm_span(
                    item.get("start", item.get("frame", 0)), item.get("end")
                )
                vname = item.get("view") or self.views[self.active_view_idx].get(
                    "name", "view"
                )
                lines.append(
                    f"- view={vname} frames {span_txt} {item.get('old')} -> {item.get('new')} decision={decision}"
                )

        log_path = (
            annotations_path or paths[0] if paths else "annotations"
        ) + ".validation.log.txt"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        except Exception as ex:
            print(f"[LOG][ERROR] validation summary write failed: {log_path} ({ex})")
            QMessageBox.warning(
                self,
                "Validation log",
                f"Failed to write review log:\n{log_path}\n\n{ex}",
            )
            return
        print(f"[LOG] {log_path}")

        email_text = f"To: smartage5hp@gmail.com\nSubject: {subject}\n\n" + "\n".join(
            lines
        )
        try:
            QApplication.clipboard().setText(email_text)
        except Exception:
            pass
        QMessageBox.information(
            self,
            "Validation log saved",
            f"Subject: {subject}\nLog file: {os.path.basename(log_path)}\nEmail content copied to clipboard.",
        )

    def _save_validation_log_safely(
        self,
        annotations_path: str,
        video_id: str,
        saved_paths: Optional[List[str]] = None,
    ) -> None:
        try:
            self._save_validation_log(
                annotations_path=annotations_path,
                video_id=video_id,
                saved_paths=saved_paths,
            )
        except Exception as ex:
            anchor = annotations_path or "(unknown)"
            print(f"[LOG][ERROR] validation summary failed: {anchor} ({ex})")
            QMessageBox.warning(
                self,
                "Validation log",
                f"Validation summary logging failed, but annotation save/export already succeeded.\n\nTarget: {anchor}\n\n{ex}",
            )

    def _video_meta_from_player(self) -> Dict[str, Any]:
        fps = float(max(1, self._get_fps()))
        view = self.views[self.active_view_idx] if self.views else {}
        view_start = int(view.get("start", 0))
        view_end = int(view.get("end", view_start))
        num_frames = (
            max(0, view_end - view_start + 1)
            if view_end >= view_start
            else int(self._get_frame_count())
        )
        name = (
            os.path.basename(self.video_path)
            if getattr(self, "video_path", None)
            else "unknown.mp4"
        )
        vid = self.current_video_id or (
            os.path.splitext(name)[0] if name else "unknown"
        )
        return {
            "id": vid,
            "name": name,
            "fps": fps,
            "num_frames": num_frames,
            "duration_sec": num_frames / fps if fps else 0.0,
            "view_start": view_start,
            "view_end": view_end,
        }

    def _meta_data_for_view(self, view: dict) -> Dict[str, Any]:
        player = view.get("player") if isinstance(view, dict) else None
        fps = float(getattr(player, "frame_rate", 0.0) or self._get_fps())
        frame_count = int(getattr(player, "frame_count", 0) or 0)
        width = int(getattr(player, "_frame_w", 0) or 0)
        height = int(getattr(player, "_frame_h", 0) or 0)
        if (width <= 0 or height <= 0) and getattr(player, "cap", None) is not None:
            try:
                import cv2

                cap = player.cap
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            except Exception:
                pass
        view_start = int(view.get("start", 0)) if isinstance(view, dict) else 0
        view_end = (
            int(view.get("end", view_start)) if isinstance(view, dict) else view_start
        )
        num_frames = (
            max(0, view_end - view_start + 1) if view_end >= view_start else frame_count
        )
        return {
            "fps": fps,
            "resolution": {"width": width, "height": height},
            "num_frames": num_frames,
            "view_start": view_start,
            "view_end": view_end,
        }

    def _apply_canonical_to_store(self, canonical: Dict[str, Any]):
        video_info = canonical.get("video", {}) or {}
        view = self.views[self.active_view_idx] if self.views else {}
        view_start = int(
            video_info.get("view_start", view.get("start", 0) if view else 0)
        )
        view_end_meta = video_info.get("view_end", None)
        view_end = (
            int(view_end_meta)
            if view_end_meta is not None
            else (int(view.get("end", view_start)) if view else None)
        )

        self.labels.clear()
        id2label = {}
        for item in sorted(
            canonical.get("categories", []), key=lambda x: int(x.get("id", 0))
        ):
            lid = int(item.get("id"))
            name = item.get("name", f"Label_{lid}")
            col = self._auto_color_key_for_id(lid)
            self.labels.append(LabelDef(name=name, color_name=col, id=lid))
            id2label[lid] = name
        self.panel.refresh()
        self._rebuild_timeline_sources()

        self.store.frame_to_label.clear()
        self.store.label_to_frames.clear()
        self.extra_store = AnnotationStore()
        self.extra_cuts = []
        if self.views and 0 <= self.active_view_idx < len(self.views):
            self.views[self.active_view_idx]["psr_state"] = self._psr_empty_view_state()
            self._psr_load_view_state(self.views[self.active_view_idx])
        for st in self.entity_stores.values():
            st.frame_to_label.clear()
            st.label_to_frames.clear()
        for st in self.phase_stores.values():
            st.frame_to_label.clear()
            st.label_to_frames.clear()
        for ent_map in self.anomaly_type_stores.values():
            for st in ent_map.values():
                st.frame_to_label.clear()
                st.label_to_frames.clear()
        self.entity_stores.clear()
        self.phase_stores.clear()
        self.anomaly_type_stores.clear()
        self.label_entity_map.clear()

        self.mode = "Coarse"
        self.phase_mode_enabled = False
        self._phase_selected = None
        try:
            self.combo_mode.setCurrentText("Coarse")
        except Exception:
            pass
        # ensure interaction row exists (keeps live painting when importing legacy JSON)
        self._ensure_extra_label()

        for ann in canonical.get("annotations", []):
            try:
                lid = int(ann.get("category_id"))
                name = id2label.get(lid, f"Label_{lid}")
                s = int(ann.get("start", {}).get("value"))
                e = int(ann.get("end", {}).get("value"))
            except Exception:
                continue
            s_abs = s + view_start
            e_abs = e + view_start
            if view_end is not None:
                if e_abs < view_start or s_abs > view_end:
                    continue
                e_abs = min(e_abs, view_end)
            target = self.extra_store if is_extra_label(name) else self.store
            for f in range(s_abs, e_abs + 1):
                if (not target.is_occupied(f)) or (target.label_at(f) == name):
                    target.add(name, f)

        self._sync_extra_cuts_from_store()
        self._rebuild_timeline_sources()
        try:
            self.timeline.refresh_all_rows()
        except Exception:
            self.timeline.update()
        self._dirty = True

    def _build_canonical_from_store(self, include_extra: bool = True) -> Dict[str, Any]:
        meta = self._video_meta_from_player()
        canonical = {
            "version": "v1",
            "video": meta,
            "categories": [],
            "annotations": [],
            "extras": {},
        }
        for lb in self.labels:
            if (not include_extra) and is_extra_label(lb.name):
                continue
            canonical["categories"].append({"id": int(lb.id), "name": lb.name})

        view_start = int(meta.get("view_start", 0))
        view_end = meta.get("view_end", None)
        max_rel = None
        if view_end is not None:
            try:
                max_rel = max(0, int(view_end) - view_start)
            except Exception:
                max_rel = None

        def _shift_and_clip(fr_s: int, fr_e: int):
            rs = max(0, fr_s - view_start)
            re = max(0, fr_e - view_start)
            if max_rel is not None:
                if rs > max_rel:
                    return None
                re = min(re, max_rel)
            return rs, re

        if self.mode == "Coarse":
            for lb in self.labels:
                if is_extra_label(lb.name):
                    if not include_extra:
                        continue
                    runs = self._extra_runs()
                else:
                    frames = self.store.frames_of(lb.name)
                    runs = AnnotationStore.frames_to_runs(frames)
                for s, e in runs:
                    shifted = _shift_and_clip(int(s), int(e))
                    if shifted:
                        rs, re = shifted
                        canonical["annotations"].append(
                            {
                                "id": f"ann_{lb.id}_{rs}",
                                "category_id": int(lb.id),
                                "start": {"value": int(rs), "unit": "frame"},
                                "end": {"value": int(re), "unit": "frame"},
                                "attributes": {"source": "export"},
                            }
                        )
        else:
            for ename, st in self.entity_stores.items():
                for lb in self.labels:
                    frames = st.frames_of(lb.name)
                    for s, e in AnnotationStore.frames_to_runs(frames):
                        shifted = _shift_and_clip(int(s), int(e))
                        if shifted:
                            rs, re = shifted
                            canonical["annotations"].append(
                                {
                                    "id": f"ann_{lb.id}_{ename}_{rs}",
                                    "category_id": int(lb.id),
                                    "start": {"value": int(rs), "unit": "frame"},
                                    "end": {"value": int(re), "unit": "frame"},
                                    "attributes": {"entity": ename, "source": "export"},
                                }
                            )
        if not include_extra:
            extra_runs = []
            for s, e in self._extra_runs():
                shifted = (max(0, s - view_start), max(0, e - view_start))
                extra_runs.append({"start": shifted[0], "end": shifted[1]})
            if extra_runs:
                canonical["extras"]["extra_spans"] = extra_runs
        return canonical

    def _selected_view_indices_for_json_io(self) -> List[int]:
        if not self.views or not (0 <= self.active_view_idx < len(self.views)):
            return []
        selected = self._normalize_sync_edit_selection()
        if not selected:
            return [int(self.active_view_idx)]
        return sorted(int(i) for i in selected if 0 <= int(i) < len(self.views))

    def _import_json_for_selected_views(self) -> None:
        if not self.views:
            QMessageBox.information(self, "Info", "Load a video first.")
            return
        indices = self._selected_view_indices_for_json_io()
        if not indices:
            QMessageBox.information(self, "Info", "No view selected.")
            return

        default_dir = ""
        if self.current_annotation_path:
            default_dir = os.path.dirname(self.current_annotation_path)
        elif getattr(self, "video_path", None):
            default_dir = os.path.dirname(self.video_path)
        if not default_dir:
            default_dir = os.getcwd()

        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Import JSON (apply to selected views)",
            default_dir,
            "JSON Files (*.json)",
        )
        if not fp:
            return

        loaded = []
        for idx in indices:
            if not (0 <= idx < len(self.views)):
                continue
            vw = self.views[idx]
            vname = self._effective_view_name(vw, idx=idx)
            ok = self._load_json_annotations(
                path=fp, current_view_only=True, target_view_idx=int(idx)
            )
            if ok:
                loaded.append(vname)
        if not loaded:
            return
        shown = ",".join(loaded[:4])
        if len(loaded) > 4:
            shown = f"{shown},+{len(loaded)-4}"
        self._set_status(f"Imported JSON for {len(loaded)} selected view(s)")
        self._log("import_json_selected_views", count=len(loaded), views=shown)

    def _import_annotations_from_file(self, fp: str) -> bool:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read JSON: {e}")
            return False

        if isinstance(payload, dict):
            segs = payload.get("segments")
            if isinstance(segs, list) and any(
                isinstance(s, dict)
                and (
                    (
                        "action_label" in s
                        and ("start_frame" in s or "end_frame" in s)
                    )
                    or (
                        ("f_start" in s or "f_end" in s)
                        and ("label" in s or "id" in s or "label_id" in s)
                    )
                )
                for s in segs
            ):
                return bool(self._load_json_annotations(fp))

        meta = self._video_meta_from_player()
        for name, adapter in ADAPTERS.items():
            try:
                if hasattr(adapter, "detect") and not adapter.detect(payload):
                    continue
                canonical = adapter.import_to_canonical(payload, meta)
                self._apply_canonical_to_store(canonical)
                self.current_annotation_path = fp
                self._load_extra_sidecar(fp)
                self._set_status(
                    f"Imported as {name}: {len(canonical.get('annotations', []))} segments"
                )
                return True
            except Exception:
                continue

        formats = ", ".join(ADAPTERS.keys())
        QMessageBox.critical(
            self,
            "Error",
            f"Could not auto-detect this JSON format.\nSupported: {formats}",
        )
        return False

    def _export_annotations_with_adapter_auto(self):
        if not self.labels:
            QMessageBox.information(self, "Info", "No annotations to export.")
            return

        names = list(ADAPTERS.keys())
        name, ok = QInputDialog.getItem(
            self, "Export format", "Choose format:", names, 0, False
        )
        if not ok:
            return

        adapter = ADAPTERS.get(name)
        if adapter is None:
            QMessageBox.warning(self, "Warning", "Invalid export format.")
            return
        if name == "Native" and self._is_action_task():
            self._export_action_fine_json()
            return

        include_extra = name not in ("FACT", "FrameTXT")
        canonical = self._build_canonical_from_store(include_extra=include_extra)
        try:
            out_obj = adapter.export_from_canonical(canonical, options=None)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {e}")
            return

        suggested = f"annotations_{name}.json"
        if getattr(self, "video_path", None):
            base = (
                os.path.splitext(os.path.basename(self.video_path))[0] or "annotations"
            )
            suggested = os.path.join(os.path.dirname(self.video_path), base + ".json")
        fp, _ = QFileDialog.getSaveFileName(
            self,
            "Export annotations",
            suggested,
            "JSON Files (*.json)",
        )
        if not fp:
            return

        try:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(out_obj, f, ensure_ascii=False, indent=2)
            self._set_status(f"Exported as {name}: {os.path.basename(fp)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to write file: {e}")

    def _export_action_fine_json(self):
        if not self.views:
            QMessageBox.information(self, "Info", "Load a video first.")
            return
        vid = self.current_video_id or ""
        if not vid and getattr(self, "video_path", None):
            vid = os.path.splitext(os.path.basename(self.video_path))[0] or "video"
        if not vid:
            vid = "video"
        payload = self._build_fine_payload_for_view(
            self.views[self.active_view_idx], vid
        )
        suggested = ""
        if getattr(self, "video_path", None):
            base = (
                os.path.splitext(os.path.basename(self.video_path))[0] or "annotations"
            )
            suggested = os.path.join(os.path.dirname(self.video_path), base + ".json")
        fp, _ = QFileDialog.getSaveFileName(
            self, "Export annotations", suggested, "JSON Files (*.json)"
        )
        if not fp:
            return
        try:
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._set_status(f"Exported Native (fine): {os.path.basename(fp)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to write file: {e}")

    def _default_annotation_filename_for_view(self, view: dict) -> str:
        src_path = ""
        try:
            src_path = str(view.get("path") or "")
        except Exception:
            src_path = ""
        if not src_path and getattr(self, "video_path", None):
            src_path = str(self.video_path)
        if src_path:
            base = os.path.splitext(os.path.basename(src_path))[0]
            if base:
                return f"{base}.json"
        fallback = (
            self.current_video_id
            or self._safe_view_tag(self._effective_view_name(view))
            or "annotations"
        )
        return f"{fallback}.json"

    def _video_id_for_view_export(
        self, view: dict, idx: int, fallback_video_id: str = ""
    ) -> str:
        # Prefer per-view source filename stem so multi-view exports keep
        # distinct video_id values for each view.
        src_path = ""
        try:
            src_path = str(view.get("path") or "")
        except Exception:
            src_path = ""
        if src_path:
            base = os.path.splitext(os.path.basename(src_path))[0]
            if base:
                return base

        view_name = self._safe_view_tag(self._effective_view_name(view, idx=idx))
        base_vid = str(fallback_video_id or "").strip()
        if view_name and base_vid:
            low_base = base_vid.lower()
            low_view = view_name.lower()
            if low_base.endswith("_" + low_view) or low_base == low_view:
                return base_vid
            return f"{base_vid}_{view_name}"
        if view_name:
            return view_name
        if base_vid:
            return base_vid
        if self.current_video_id:
            return str(self.current_video_id)
        return "video"

    def _payload_for_view(self, view: dict, vid: str) -> dict:
        if self._is_action_task():
            return self._build_fine_payload_for_view(view, vid)
        return self._build_payload_for_store(view, vid)

    def _build_multiview_payload_items(
        self, out_dir: str, vid: str, view_indices: Optional[List[int]] = None
    ) -> Tuple[List[Tuple[str, dict]], str]:
        payload_items: List[Tuple[str, dict]] = []
        active_path = ""
        if view_indices is None:
            indices = list(range(len(self.views)))
        else:
            indices = [
                int(i)
                for i in (view_indices or [])
                if 0 <= int(i) < len(self.views)
            ]
        for idx in indices:
            vw = self.views[idx]
            view_name = self._effective_view_name(vw, idx=idx)
            view_dir = os.path.join(out_dir, self._safe_view_tag(view_name))
            file_name = self._default_annotation_filename_for_view(vw)
            out_path = os.path.join(view_dir, file_name)
            per_view_vid = self._video_id_for_view_export(vw, idx, vid)
            payload = self._payload_for_view(vw, per_view_vid)
            payload_items.append((out_path, payload))
            if idx == self.active_view_idx:
                active_path = out_path
        return payload_items, active_path

    def _confirm_overwrite(self, payload_items: List[Tuple[str, dict]]) -> bool:
        exists = [p for p, _ in payload_items if os.path.exists(p)]
        if not exists:
            return True
        preview = ", ".join(os.path.basename(p) for p in exists[:4])
        if len(exists) > 4:
            preview += f", ... (+{len(exists) - 4})"
        ret = QMessageBox.question(
            self,
            "Overwrite files",
            f"{len(exists)} file(s) already exist:\n{preview}\n\nOverwrite?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return ret == QMessageBox.Yes

    def _default_save_json_path(self) -> str:
        if getattr(self, "video_path", None):
            base = (
                os.path.splitext(os.path.basename(self.video_path))[0] or "annotations"
            )
            return os.path.join(os.path.dirname(self.video_path), base + ".json")
        return ""

    def _infer_multiview_export_root(self, fp: str) -> str:
        base_dir = os.path.dirname(fp) if fp else ""
        if not base_dir:
            return os.getcwd()
        parent = os.path.dirname(base_dir)
        try:
            view_tags = {
                self._safe_view_tag(self._effective_view_name(vw, idx=i))
                for i, vw in enumerate(self.views)
            }
        except Exception:
            view_tags = set()
        if parent and os.path.basename(base_dir) in view_tags:
            return parent
        return base_dir

    def _print_saved_locations(self, saved_paths: List[str]) -> None:
        if not saved_paths:
            return
        if len(saved_paths) == 1:
            print(f"[SAVE] {saved_paths[0]}")
            return
        root = os.path.commonpath(saved_paths)
        print(f"[SAVE] {len(saved_paths)} files under {root}")
        for p in saved_paths:
            print(f"[SAVE] - {p}")

    def _flush_ops_log_safely(self, log_path: str, context: str = "save") -> None:
        logger = getattr(self, "op_logger", None)
        if not log_path or logger is None:
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
                f"Operation log write failed, but annotation save/export already succeeded.\n\nTarget: {log_path}\n\n{ex}",
            )

    def _flush_ops_logs_for_paths(
        self,
        primary_base: str,
        saved_paths: Optional[List[str]] = None,
        context: str = "save",
    ) -> None:
        targets: List[str] = []
        if primary_base:
            targets.append(primary_base + ".ops.log.csv")
        for p in saved_paths or []:
            if p:
                targets.append(p + ".ops.log.csv")
        seen = set()
        for path in targets:
            if path in seen:
                continue
            seen.add(path)
            self._flush_ops_log_safely(path, context=context)

    def _export_all_views_json_to_folders(self):
        if not self.views:
            QMessageBox.information(self, "Info", "Load a video first.")
            return
        view_indices = self._selected_view_indices_for_json_io()
        if not view_indices:
            QMessageBox.information(self, "Info", "No view selected.")
            return
        default_vid = self.current_video_id or ""
        if not default_vid and getattr(self, "video_path", None):
            default_vid = os.path.splitext(os.path.basename(self.video_path))[0]
        vid, ok = QInputDialog.getText(self, "Video ID", "video_id:", text=default_vid)
        if not ok or not vid.strip():
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Choose export directory")
        if not out_dir:
            return

        payload_items: List[Tuple[str, dict]] = []
        saved_paths: List[str] = []
        try:
            payload_items, active_out = self._build_multiview_payload_items(
                out_dir, vid.strip(), view_indices=view_indices
            )
            if not self._confirm_overwrite(payload_items):
                return

            for out_path, payload in payload_items:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                saved_paths.append(out_path)

            if active_out:
                self.current_annotation_path = active_out
            self._save_extra_sidecar_if_possible()
            # Mark only exported views as saved; keep others dirty.
            for idx in view_indices:
                if 0 <= idx < len(self.views):
                    self.views[idx]["dirty"] = False
            self._dirty = any(bool(vw.get("dirty")) for vw in self.views)
            log_anchor = os.path.join(out_dir, vid.strip() or "annotations")
            self._save_validation_log_safely(log_anchor, vid.strip(), saved_paths)
            self._set_status(
                f"Exported {len(saved_paths)} selected view file(s) to {out_dir}"
            )
            self._print_saved_locations(saved_paths)
            self._log(
                "export_all_views",
                dir=out_dir,
                count=len(saved_paths),
                video_id=vid.strip(),
                selected=len(view_indices),
            )
            self._flush_ops_logs_for_paths(
                primary_base=log_anchor,
                saved_paths=saved_paths,
                context="export all views",
            )
        except Exception as ex:
            recovery_anchor = os.path.join(
                out_dir, f"{vid.strip() or 'annotations'}.json"
            )
            self._write_recovery_payloads(payload_items, recovery_anchor)
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to export files:\n{ex}\nRecovery files were written.",
            )

    def _build_fine_payload_for_view(self, view: dict, vid: str) -> dict:
        view_start = int(view.get("start", 0))
        view_end = int(view.get("end", view_start))
        max_rel = max(0, view_end - view_start)

        def _shift_and_clip(fr_s: int, fr_e: int):
            rs = max(0, fr_s - view_start)
            re = max(0, fr_e - view_start)
            if rs > max_rel:
                return None
            re = min(re, max_rel)
            return rs, re

        # vocab lists
        verb_map, noun_map = self._ensure_fine_vocab()
        anomaly_types = [
            {"id": int(t.get("id", 0)), "name": str(t.get("name", ""))}
            for t in self.anomaly_types
            if t.get("name")
        ]
        action_labels = [
            {"id": int(lb.id), "name": lb.name}
            for lb in sorted(self.labels, key=lambda x: x.id)
        ]
        label_id_by_name = {lb.name: int(lb.id) for lb in self.labels}

        segments = []
        view_entities = view.get("entity_stores", {})
        has_entity_data = any(st and st.frame_to_label for st in view_entities.values())
        is_fine = (self.mode == "Fine") or has_entity_data

        if not is_fine:
            # coarse: use global store
            for lb in self.labels:
                if is_extra_label(lb.name):
                    frames = self._extra_frames()
                else:
                    st = view.get("store") or self.store
                    frames = st.frames_of(lb.name)
                for s_abs, e_abs in AnnotationStore.frames_to_runs(frames):
                    shifted = _shift_and_clip(int(s_abs), int(e_abs))
                    if not shifted:
                        continue
                    rs, re = shifted
                    vname, nname = self._infer_verb_noun(lb.name, list(verb_map.keys()))
                    segments.append(
                        {
                            "action_label": int(lb.id),
                            "verb": int(verb_map.get(vname, -1)) if vname else -1,
                            "noun": int(noun_map.get(nname, -1)) if nname else -1,
                            "start_frame": int(rs),
                            "end_frame": int(re),
                            "phase": "",
                            "anomaly_type": [0] * len(anomaly_types),
                        }
                    )
        else:
            for ename, st in view_entities.items():
                for s_abs, e_abs, lname in self._segments_from_store(st):
                    if not lname:
                        continue
                    shifted = _shift_and_clip(int(s_abs), int(e_abs))
                    if not shifted:
                        continue
                    rs, re = shifted
                    lid = int(label_id_by_name.get(lname, 0))
                    vname, nname = self._infer_verb_noun(lname, list(verb_map.keys()))
                    phase_label = ""
                    anom_vec = [0] * len(anomaly_types)
                    if self.phase_mode_enabled and ename:
                        phase_map = view.get("phase_stores", {}) or {}
                        anom_map = view.get("anomaly_type_stores", {}) or {}
                        pstore = phase_map.get(ename)
                        phase_label = self._phase_label_for_span(
                            pstore, int(s_abs), int(e_abs)
                        )
                        if phase_label == "anomaly":
                            anom_vec = self._anomaly_vector_for_span(
                                ename, int(s_abs), int(e_abs), anom_map
                            )
                    segments.append(
                        {
                            "action_label": lid,
                            "verb": int(verb_map.get(vname, -1)) if vname else -1,
                            "noun": int(noun_map.get(nname, -1)) if nname else -1,
                            "start_frame": int(rs),
                            "end_frame": int(re),
                            "phase": phase_label or "",
                            "anomaly_type": anom_vec,
                            "entity": ename,
                        }
                    )

        payload = {
            "video_id": vid,
            "view": self._effective_view_name(view),
            "meta_data": self._meta_data_for_view(view),
            "view_start": view_start,
            "view_end": view_end,
            "anomaly_types": anomaly_types,
            "verbs": list(self.fine_verbs or []),
            "nouns": list(self.fine_nouns or []),
            "action_labels": action_labels,
            "segments": segments,
        }
        return payload

    def _seed_segments_for_active_view(self) -> List[dict]:
        if not self.views:
            return []
        view = self.views[self.active_view_idx]
        view_start = int(view.get("start", 0))
        view_end = int(view.get("end", view_start))
        frames = sorted(
            f for f in self.store.frame_to_label.keys() if view_start <= f <= view_end
        )
        if not frames:
            return []
        segments = []
        s = frames[0]
        cur = self.store.label_at(s)
        prev = s
        for f in frames[1:]:
            lb = self.store.label_at(f)
            if lb != cur or f != prev + 1:
                if cur and not is_extra_label(cur):
                    segments.append({"start_frame": s, "end_frame": prev, "label": cur})
                s, cur = f, lb
            prev = f
        if cur and not is_extra_label(cur):
            segments.append({"start_frame": s, "end_frame": prev, "label": cur})
        return segments

    def _update_seed_meta(
        self, seed_dir: str, video_name: str, json_name: str, video_id: str
    ):
        meta_path = os.path.join(seed_dir, "meta.json")
        meta = {"videos": []}
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {"videos": []}
            except Exception:
                meta = {"videos": []}
        if not isinstance(meta, dict):
            meta = {"videos": []}
        entries = meta.get("videos")
        if not isinstance(entries, list):
            entries = []
        entry = {
            "video": video_name,
            "annotation": json_name,
            "video_id": video_id,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
        }
        replaced = False
        for idx, it in enumerate(entries):
            if isinstance(it, dict) and it.get("video_id") == video_id:
                entries[idx] = entry
                replaced = True
                break
        if not replaced:
            entries.append(entry)
        meta["videos"] = entries
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _export_seed_dataset(self):
        if not getattr(self, "video_path", None):
            QMessageBox.information(self, "Info", "Load a video first.")
            return
        segments = self._seed_segments_for_active_view()
        if not segments:
            QMessageBox.information(self, "Info", "No labeled segments to export.")
            return
        seed_dir = QFileDialog.getExistingDirectory(
            self, "Select Seed Dataset directory"
        )
        if not seed_dir:
            return
        os.makedirs(seed_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(self.video_path))[0] or "video"
        video_name = os.path.basename(self.video_path)
        json_name = f"{base}.json"
        video_dst = os.path.join(seed_dir, video_name)
        json_dst = os.path.join(seed_dir, json_name)

        if os.path.exists(video_dst) or os.path.exists(json_dst):
            ret = QMessageBox.question(
                self,
                "Overwrite",
                f"{video_name} or {json_name} already exists in the seed dataset.\nOverwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return

        payload = {
            "video_id": base,
            "segments": segments,
        }
        try:
            shutil.copy2(self.video_path, video_dst)
            with open(json_dst, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._update_seed_meta(seed_dir, video_name, json_name, base)
            self._set_status(
                f"Seed export saved: {os.path.basename(video_dst)}, {json_name}"
            )
            self._log("seed_export", video=video_name, json=json_name, dir=seed_dir)
        except Exception as ex:
            QMessageBox.warning(self, "Error", f"Seed export failed:\n{ex}")

    def _save_json_annotations(self, skip_gap_check: bool = False) -> bool:
        # allow saving even if there are no labels
        default_vid = ""
        if getattr(self, "video_path", None):
            default_vid = os.path.splitext(os.path.basename(self.video_path))[0]
        vid, ok = QInputDialog.getText(self, "Video ID", "video_id:", text=default_vid)
        if not ok or not vid.strip():
            return False
        if not skip_gap_check:
            ok_gaps, _ = self._check_unlabeled_gaps(context="save")
            if not ok_gaps:
                return False

        suggested = self._default_save_json_path()
        saved_paths = []
        payload_items = []
        fp = ""
        active_out = ""
        out_dir = ""
        try:
            if len(self.views) > 1:
                default_dir = ""
                if self.current_annotation_path:
                    default_dir = self._infer_multiview_export_root(
                        self.current_annotation_path
                    )
                elif suggested:
                    default_dir = os.path.dirname(suggested)
                out_dir = QFileDialog.getExistingDirectory(
                    self, "Save JSON (all views to folders)", default_dir
                )
                if not out_dir:
                    return False
                payload_items, active_out = self._build_multiview_payload_items(
                    out_dir, vid.strip()
                )
                if not self._confirm_overwrite(payload_items):
                    return False
            else:
                fp, _ = QFileDialog.getSaveFileName(
                    self, "Save JSON", suggested, "JSON Files (*.json)"
                )
                if not fp:
                    return False
                payload = self._payload_for_view(
                    self.views[self.active_view_idx], vid.strip()
                )
                payload_items.append((fp, payload))
                active_out = fp

            for out_path, payload in payload_items:
                out_parent = os.path.dirname(out_path)
                if out_parent:
                    os.makedirs(out_parent, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                saved_paths.append(out_path)

            if active_out:
                self.current_annotation_path = active_out
            self._save_extra_sidecar_if_possible()
            self._dirty = False
            for vw in self.views:
                vw["dirty"] = False
            paths_txt = ", ".join(os.path.basename(p) for p in saved_paths)
            self._set_status(f"Saved annotations to {paths_txt}")
            self._print_saved_locations(saved_paths)
            log_anchor = (
                os.path.join(out_dir, vid.strip() or "annotations")
                if len(saved_paths) > 1 and out_dir
                else (fp or suggested)
            )
            self._save_validation_log_safely(log_anchor, vid.strip(), saved_paths)
            self._log("save_annotations", paths=paths_txt, video_id=vid.strip())
            log_base = (
                os.path.join(out_dir, vid.strip() or "annotations")
                if len(saved_paths) > 1 and out_dir
                else (fp if fp else suggested)
            )
            if log_base:
                self._flush_ops_logs_for_paths(
                    primary_base=log_base,
                    saved_paths=saved_paths,
                    context="save annotations",
                )
            return True
        except Exception as ex:
            recovery_anchor = fp or (
                os.path.join(out_dir, f"{vid.strip() or 'annotations'}.json")
                if out_dir
                else suggested
            )
            self._write_recovery_payloads(payload_items, recovery_anchor)
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to save file:\n{ex}\nRecovery files were written.",
            )
            return False

    def _save_annotations_to_path(self, fp: str, vid: str) -> bool:
        """
        Save current annotations (and interaction sidecar) to a fixed path without prompting.
        """
        ok_gaps, _ = self._check_unlabeled_gaps(context="save")
        if not ok_gaps:
            return False

        saved_paths = []
        payload_items = []
        active_out = ""
        out_dir = ""
        try:
            if len(self.views) > 1:
                out_dir = self._infer_multiview_export_root(fp)
                payload_items, active_out = self._build_multiview_payload_items(
                    out_dir, vid.strip()
                )
            else:
                payload = self._payload_for_view(
                    self.views[self.active_view_idx], vid.strip()
                )
                payload_items.append((fp, payload))
                active_out = fp
            for out_path, payload in payload_items:
                out_parent = os.path.dirname(out_path)
                if out_parent:
                    os.makedirs(out_parent, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                saved_paths.append(out_path)

            if active_out:
                self.current_annotation_path = active_out
            self._save_extra_sidecar_if_possible()
            self._dirty = False
            for vw in self.views:
                vw["dirty"] = False
            log_base = (
                os.path.join(out_dir, vid.strip() or "annotations")
                if len(saved_paths) > 1 and out_dir
                else fp
            )
            self._log(
                "save_annotations",
                paths=",".join(os.path.basename(p) for p in saved_paths),
                video_id=vid.strip(),
                save_mode="direct_path",
            )
            if log_base:
                self._flush_ops_logs_for_paths(
                    primary_base=log_base,
                    saved_paths=saved_paths,
                    context="auto save annotations",
                )
            self._set_status(
                f"Saved annotations to {', '.join(os.path.basename(p) for p in saved_paths)}"
            )
            self._print_saved_locations(saved_paths)
            log_anchor = (
                os.path.join(out_dir, vid.strip() or "annotations")
                if len(saved_paths) > 1 and out_dir
                else fp
            )
            self._save_validation_log_safely(log_anchor, vid.strip(), saved_paths)
            return True
        except Exception as ex:
            recovery_anchor = fp or (
                os.path.join(out_dir, f"{vid.strip() or 'annotations'}.json")
                if out_dir
                else ""
            )
            self._write_recovery_payloads(payload_items, recovery_anchor)
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to save file:\n{ex}\nRecovery files were written.",
            )
            return False

    def _write_recovery_payloads(
        self, payload_items: List[Tuple[str, dict]], base_path: str
    ) -> None:
        if not payload_items or not base_path:
            return
        try:
            base_dir = os.path.dirname(base_path) or os.getcwd()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recovery_dir = os.path.join(base_dir, f"_recovery_{stamp}")
            os.makedirs(recovery_dir, exist_ok=True)
            for out_path, payload in payload_items:
                name = os.path.basename(out_path) or "annotations.json"
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
                rec_path = os.path.join(recovery_dir, safe_name)
                with open(rec_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _build_payload_for_store(self, view: dict, vid: str) -> dict:
        """Serialize annotations for a given view, shifting frames to the view's crop start."""
        store = view["store"]
        view_start = int(view.get("start", 0))
        view_end = int(view.get("end", view_start))
        max_rel = max(0, view_end - view_start)

        def _shift_and_clip(fr_s, fr_e):
            rs = max(0, fr_s - view_start)
            re = max(0, fr_e - view_start)
            if rs > max_rel:
                return None
            re = min(re, max_rel)
            return rs, re

        segments = []
        view_stores = view.get("entity_stores", {})
        has_entity_data = any(st and st.frame_to_label for st in view_stores.values())
        is_fine = (self.mode == "Fine") or has_entity_data

        if not is_fine:
            for lb in self.labels:
                if is_extra_label(lb.name):
                    runs = self._extra_runs()
                else:
                    frames = store.frames_of(lb.name)
                    runs = AnnotationStore.frames_to_runs(frames)
                for s, e in runs:
                    shifted = _shift_and_clip(int(s), int(e))
                    if shifted:
                        rs, re = shifted
                        segments.append(
                            {
                                "action_label": int(lb.id),
                                "start_frame": rs,
                                "end_frame": re,
                            }
                        )
        else:
            for ename, st in view_stores.items():
                for lb in self.labels:
                    frames = st.frames_of(lb.name)
                    for s, e in AnnotationStore.frames_to_runs(frames):
                        shifted = _shift_and_clip(int(s), int(e))
                        if shifted:
                            rs, re = shifted
                            segments.append(
                                {
                                    "action_label": int(lb.id),
                                    "start_frame": rs,
                                    "end_frame": re,
                                    "entity": ename,
                                }
                            )

        labels_payload = []
        for lb in self.labels:
            colkey = getattr(lb, "color_name", None) or "Gray"
            labels_payload.append({"id": int(lb.id), "name": lb.name, "color": colkey})
        payload = {
            "video_id": vid,
            "view": self._effective_view_name(view),
            "view_start": view_start,
            "view_end": view_end,
            "segments": segments,
            "labels": labels_payload,
        }
        return payload

    @staticmethod
    def _safe_view_tag(name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9]+", "_", name or "view").strip("_")
        return safe or "view"

    def _warn_if_crop_conflicts(self, store: AnnotationStore, start: int, end: int):
        """Warn if existing annotations fall outside the [start, end] crop."""
        frames = sorted(store.frame_to_label.keys())
        if not frames:
            self._update_gap_indicator()
            return
        if frames[0] < start or frames[-1] > end:
            QMessageBox.information(
                self,
                "Crop/Annotation mismatch",
                "Annotations exist outside the selected crop range. Saving will shift/clamp to the cropped span.",
            )
        self._update_gap_indicator()

    # ===== Validation import/review =====
    def _parse_validation_log(self, log_path: str) -> List[dict]:
        entries = []
        if not os.path.exists(log_path):
            return entries
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines()]
        except Exception:
            return entries
        in_mod = False
        psr_pattern = re.compile(
            r"- \[(?P<ts>[^\]]+)\]\s*(?:view=(?P<view>\S+)\s+)?PSR\s+(?P<component>.*?)\s+frames\s+(?P<span>n/a|\d+(?:-\d+)?):\s*(?P<old>.*?)\s*->\s*(?P<new>.*?)\s+action=(?P<action>[^\s]+)\s+\(by\s*(?P<editor>[^)]+)\)(?:\s*(?:note|comment):\s*(?P<desc>.*))?$"
        )
        mod_pattern = re.compile(
            r"- \[(?P<ts>[^\]]+)\]\s*(?:view=(?P<view>\S+)\s+)?(?P<store>.*?)\s+frames\s+(?P<span>\d+(?:-\d+)?):\s*(?P<old>.*?)\s*->\s*(?P<new>.*?)\s+\(by\s*(?P<editor>[^)]+)\)(?:\s*(?:comment|note):\s*(?P<desc>.*))?$"
        )
        for ln in lines:
            if ln.lower().startswith("modifications"):
                in_mod = True
                continue
            if ln.lower().startswith("errors"):
                in_mod = False
                continue
            if in_mod and ln.startswith("-"):
                m = psr_pattern.match(ln)
                if m:
                    d = m.groupdict()
                    span_txt = d.get("span", "0")
                    entries.append(
                        {
                            "ts": d.get("ts"),
                            "kind": "psr",
                            "store": "PSR",
                            "view": d.get("view") or None,
                            "start": None,
                            "end": None,
                            "frame": None,
                            "old": (d.get("old") or "").strip() or None,
                            "new": (d.get("new") or "").strip() or None,
                            "editor": d.get("editor"),
                            "component": (d.get("component") or "").strip() or "combined",
                            "component_id": None,
                            "action": (d.get("action") or "").strip() or "change",
                            "desc": (d.get("desc") or "").strip(),
                            "decision": None,
                        }
                    )
                    if span_txt.lower() != "n/a":
                        if "-" in span_txt:
                            s_txt, e_txt = span_txt.split("-", 1)
                        else:
                            s_txt = e_txt = span_txt
                        try:
                            s_frame = int(s_txt)
                            e_frame = int(e_txt)
                        except Exception:
                            s_frame = e_frame = 0
                        entries[-1]["start"] = s_frame
                        entries[-1]["end"] = e_frame
                        entries[-1]["frame"] = s_frame
                    continue
                m = mod_pattern.match(ln)
                if m:
                    d = m.groupdict()
                    span_txt = d.get("span", "0")
                    if "-" in span_txt:
                        s_txt, e_txt = span_txt.split("-", 1)
                    else:
                        s_txt = e_txt = span_txt
                    try:
                        s_frame = int(s_txt)
                        e_frame = int(e_txt)
                    except Exception:
                        s_frame = e_frame = 0
                    entries.append(
                        {
                            "ts": d.get("ts"),
                            "kind": "store",
                            "store": (d.get("store") or "GLOBAL").strip() or "GLOBAL",
                            "view": d.get("view") or None,
                            "start": s_frame,
                            "end": e_frame,
                            "frame": s_frame,
                            "old": (d.get("old") or "").strip() or None,
                            "new": (d.get("new") or "").strip() or None,
                            "editor": d.get("editor"),
                            "desc": (d.get("desc") or "").strip(),
                            "decision": None,
                        }
                    )
        entries.sort(
            key=lambda x: (
                str(x.get("view") or ""),
                int(x.get("start", x.get("frame", 0) or 0)),
                str(x.get("component") or x.get("store") or ""),
            )
        )
        return entries

    def _psr_component_id_from_value(self, value: Any) -> Optional[Any]:
        if value is None:
            return None
        txt = str(value).strip()
        if not txt:
            return None
        if txt.lower() == "combined":
            return None
        for comp in self.psr_components or []:
            if str(comp.get("id")) == txt:
                return comp.get("id")
            if str(comp.get("name", "")).strip().lower() == txt.lower():
                return comp.get("id")
        return value

    def _psr_build_review_stores(self, include_manual: bool = True) -> Dict[str, Any]:
        saved_manual_events = copy.deepcopy(self._psr_manual_events)
        saved_gap_spans_combined = copy.deepcopy(self._psr_gap_spans_combined)
        saved_gap_spans_by_comp = copy.deepcopy(self._psr_gap_spans_by_comp)
        saved_combined_label_states = copy.deepcopy(self._psr_combined_label_states)
        saved_state_color_cache = copy.deepcopy(self._psr_state_color_cache)
        try:
            if not include_manual:
                self._psr_manual_events = []
                self._psr_gap_spans_combined = []
                self._psr_gap_spans_by_comp = {}
            self._psr_mark_dirty()
            self._psr_recompute_cache()
            runs = self._psr_build_state_runs()
            component_stores = self._psr_build_component_stores(runs)
            _label_defs, combined_store = self._psr_build_combined_store(runs)
            return {
                "component_stores": component_stores,
                "combined_store": combined_store,
                "combined_label_states": copy.deepcopy(
                    self._psr_combined_label_states or {}
                ),
            }
        finally:
            self._psr_manual_events = saved_manual_events
            self._psr_gap_spans_combined = saved_gap_spans_combined
            self._psr_gap_spans_by_comp = saved_gap_spans_by_comp
            self._psr_combined_label_states = saved_combined_label_states
            self._psr_state_color_cache = saved_state_color_cache
            self._psr_mark_dirty()
            self._psr_recompute_cache()

    def _review_apply_store_range(
        self,
        store: Optional[AnnotationStore],
        start: int,
        end: int,
        target_label: Optional[str] = None,
        template_store: Optional[AnnotationStore] = None,
    ) -> None:
        if store is None:
            return
        try:
            start = int(start)
            end = int(end)
        except Exception:
            return
        if end < start:
            start, end = end, start
        if hasattr(store, "begin_txn"):
            store.begin_txn()
        try:
            for frame in range(start, end + 1):
                desired = (
                    template_store.label_at(frame)
                    if template_store is not None
                    else target_label
                )
                cur = store.label_at(frame)
                if cur is not None and cur != desired:
                    store.remove_at(frame)
                if desired and store.label_at(frame) != desired:
                    store.add(desired, frame)
                if not desired:
                    store.remove_at(frame)
        finally:
            if hasattr(store, "end_txn"):
                store.end_txn()

    def _psr_focus_review_item(self, item: dict) -> None:
        if not self._is_psr_task():
            return
        frame = item.get("start", item.get("frame", 0))
        if frame is None:
            self._psr_update_component_panel(force=True)
            return
        try:
            frame = int(frame)
        except Exception:
            frame = 0
        raw_component = (
            item.get("component_id")
            if item.get("component_id") is not None
            else item.get("component")
        )
        comp_id = self._psr_component_id_from_value(
            raw_component
        )
        start = int(item.get("start", frame) or frame)
        end = int(item.get("end", start) or start)
        self._psr_refresh_state_timeline(force=True)
        row = self._psr_row_for_component(comp_id)
        label = None
        if row is not None:
            try:
                _rs, _re, label = row._segment_at(int(frame))
            except Exception:
                label = None
            try:
                row._selected_interval = (int(start), int(end))
                row._selected_label = label
                row._selection_scope = "segment"
                row.update()
            except Exception:
                pass
            try:
                self.timeline._active_combined_row = row
            except Exception:
                pass
        self._psr_set_selected_segment(
            start,
            end,
            label,
            row=row,
            scope="segment",
            component_id=comp_id,
        )
        self._psr_update_component_panel(frame, force=True)

    def _psr_apply_review_change(self, item: dict) -> bool:
        if not self._is_psr_task():
            return False
        start = item.get("start", item.get("frame"))
        end = item.get("end", start)
        if start is None or end is None:
            QMessageBox.information(
                self,
                "Review",
                "This PSR review item has no frame span and cannot be applied automatically.",
            )
            return False
        try:
            start = int(start)
            end = int(end)
        except Exception:
            QMessageBox.information(
                self,
                "Review",
                "This PSR review item has an invalid frame span and cannot be applied.",
            )
            return False
        if end < start:
            start, end = end, start
        new_raw = (item.get("new") or "").strip()
        new_key = new_raw.lower()
        raw_component = (
            item.get("component_id")
            if item.get("component_id") is not None
            else item.get("component")
        )
        comp_id = self._psr_component_id_from_value(
            raw_component
        )
        stores = self._psr_build_review_stores(include_manual=True)
        baseline = None
        template_store = None
        target_label = None
        if new_key == "derived":
            baseline = self._psr_build_review_stores(include_manual=False)
        elif new_key in {"", "-", "none", "unlabeled", "gap"}:
            target_label = None
        if comp_id is None:
            target_store = stores.get("combined_store")
            if target_store is None:
                return False
            if baseline is not None:
                template_store = baseline.get("combined_store")
            elif target_label is None:
                template_store = None
            else:
                label_states = stores.get("combined_label_states", {}) or {}
                if new_raw not in label_states:
                    QMessageBox.information(
                        self,
                        "Review",
                        f"Combined PSR state '{new_raw}' is not available in the current base timeline.",
                    )
                    return False
                target_label = new_raw
            self._review_apply_store_range(
                target_store,
                start,
                end,
                target_label=target_label,
                template_store=template_store,
            )
            self._psr_push_undo("review_apply")
            self._psr_manual_events = self._psr_manual_events_from_combined_store(
                target_store
            )
            if self._psr_no_gap_timeline_enabled():
                self._psr_gap_spans_combined = []
            else:
                self._psr_gap_spans_combined = self._psr_gap_spans_from_store(
                    target_store
                )
            self._psr_gap_spans_by_comp = {}
        else:
            target_store = (stores.get("component_stores") or {}).get(comp_id)
            if target_store is None:
                QMessageBox.information(
                    self,
                    "Review",
                    "Unable to resolve the PSR component referenced by this review item.",
                )
                return False
            if baseline is not None:
                template_store = (baseline.get("component_stores") or {}).get(comp_id)
            elif target_label is None:
                template_store = None
            else:
                state_val = self._psr_label_to_state(new_raw)
                if state_val is None:
                    QMessageBox.information(
                        self,
                        "Review",
                        f"Unsupported PSR state label '{new_raw}' in the review log.",
                    )
                    return False
                target_label = self._psr_state_label_name(state_val)
            self._review_apply_store_range(
                target_store,
                start,
                end,
                target_label=target_label,
                template_store=template_store,
            )
            self._psr_push_undo("review_apply")
            comp_key = str(comp_id)
            self._psr_manual_events = [
                ev
                for ev in self._psr_manual_events
                if str(ev.get("component_id")) != comp_key
            ]
            self._psr_manual_events.extend(
                self._psr_manual_events_from_store(comp_id, target_store)
            )
            if self._psr_no_gap_timeline_enabled():
                self._psr_gap_spans_by_comp.pop(comp_key, None)
            else:
                gaps = self._psr_gap_spans_from_store(target_store)
                if gaps:
                    self._psr_gap_spans_by_comp[comp_key] = gaps
                else:
                    self._psr_gap_spans_by_comp.pop(comp_key, None)
        self._psr_snap_manual_events_to_action_segments()
        self._psr_mark_dirty()
        self._psr_refresh_state_timeline(force=True)
        self._psr_focus_review_item(
            {
                **item,
                "component_id": comp_id,
                "start": start,
                "end": end,
                "frame": start,
            }
        )
        self._dirty = True
        return True

    def _set_review_queue_ui(
        self,
        *,
        info: str,
        progress: str,
        can_decide: bool,
    ) -> None:
        self.lbl_review_title.setText("Review Queue")
        self.lbl_review_info.setText(str(info or ""))
        self.lbl_review_progress.setText(str(progress or "0 / 0"))
        self.btn_accept.setEnabled(bool(can_decide))
        self.btn_reject.setEnabled(bool(can_decide))

    def _open_review_panel(self):
        self._set_review_panel_visible(True)
        if not self.review_items:
            self._set_review_queue_ui(
                info="Import a review log to load review items.",
                progress="0 / 0",
                can_decide=False,
            )
            self._set_status("Review panel opened. Import a review log to load changes.")
            return
        self._update_review_panel()

    def _set_review_panel_visible(self, on: bool):
        self.review_panel.setVisible(on)
        if on and getattr(self, "player", None):
            try:
                self.player.set_overlay_enabled(True)
            except Exception:
                pass
        try:
            self._update_validation_overlay_controls()
        except Exception:
            pass
        try:
            self._update_overlay_for_frame(getattr(self.player, "current_frame", 0))
        except Exception:
            pass

    def _update_review_panel(self):
        if (
            not self.review_items
            or self.review_idx < 0
            or self.review_idx >= len(self.review_items)
        ):
            self._set_review_queue_ui(
                info="Import a review log to load review items.",
                progress="0 / 0",
                can_decide=False,
            )
            return
        item = self.review_items[self.review_idx]
        start = item.get("start", item.get("frame", 0))
        end = item.get("end", start)
        if start is None:
            span_txt = "n/a"
        else:
            _s, _e, span_txt = self._norm_span(start, end)
        if item.get("kind") == "psr":
            comp = item.get("component") or item.get("component_id") or "combined"
            info_lines = [
                f"Frames: {span_txt}",
                "Track: PSR",
                f"Component: {comp}",
                f"Action: {item.get('action') or 'change'}",
                f"View: {item.get('view')}",
                f"Old: {item.get('old')}",
                f"New: {item.get('new')}",
                f"By: {item.get('editor')}",
            ]
        else:
            info_lines = [
                f"Frames: {span_txt}",
                f"Track: {item.get('store')}",
                f"View: {item.get('view')}",
                f"Old: {item.get('old')}",
                f"New: {item.get('new')}",
                f"By: {item.get('editor')}",
            ]
        desc = item.get("desc") or ""
        if desc:
            info_lines.append(f"Note: {desc}")
        decision = item.get("decision")
        if decision is not None:
            info_lines.append(f"Decision: {'accept' if decision else 'reject'}")
        self._set_review_queue_ui(
            info="\n".join(info_lines),
            progress=f"{self.review_idx + 1} / {len(self.review_items)}",
            can_decide=True,
        )

    def _goto_review_idx(self, idx: int):
        if not self.review_items:
            return
        idx = max(0, min(idx, len(self.review_items) - 1))
        self.review_idx = idx
        item = self.review_items[self.review_idx]
        vname = item.get("view")
        if vname:
            vi = self._view_idx_by_name(vname)
            if vi >= 0 and vi != self.active_view_idx:
                self._set_primary_view(vi)
        frame = int(
            self._norm_span(item.get("start", item.get("frame", 0)), item.get("end"))[0]
        )
        self.player.seek(frame)
        self.slider.setValue(frame)
        self.spin_jump.setValue(frame)
        self._update_overlay_for_frame(frame)
        if item.get("kind") == "psr":
            self._psr_focus_review_item(item)
        self._update_review_panel()

    def _goto_next_review(self):
        if self.review_items and self.review_idx + 1 < len(self.review_items):
            self._goto_review_idx(self.review_idx + 1)

    def _goto_prev_review(self):
        if self.review_items and self.review_idx - 1 >= 0:
            self._goto_review_idx(self.review_idx - 1)

    def _apply_change(self, item: dict) -> bool:
        vname = item.get("view")
        if vname:
            vi = self._view_idx_by_name(vname)
            if vi >= 0 and vi != self.active_view_idx:
                self._set_primary_view(vi)
        if item.get("kind") == "psr":
            return bool(self._psr_apply_review_change(item))
        store_name = item.get("store") or "GLOBAL"
        st = (
            self.store
            if store_name == "GLOBAL"
            else self.entity_stores.setdefault(store_name, AnnotationStore())
        )
        start = int(item.get("start", item.get("frame", 0)))
        end = int(item.get("end", start))
        if end < start:
            start, end = end, start
        new_label = item.get("new")
        if hasattr(st, "begin_txn"):
            st.begin_txn()
        try:
            for f in range(start, end + 1):
                cur = st.label_at(f)
                if cur is not None:
                    st.remove_at(f)
                if new_label:
                    st.add(new_label, f)
        finally:
            if hasattr(st, "end_txn"):
                st.end_txn()
        self._dirty = True
        self.timeline.refresh_all_rows()
        return True

    def _decide_current_change(self, accept: bool):
        if (
            not self.review_items
            or self.review_idx < 0
            or self.review_idx >= len(self.review_items)
        ):
            return
        item = self.review_items[self.review_idx]
        if accept:
            if not self._apply_change(item):
                return
            item["decision"] = True
        else:
            item["decision"] = False
        self._update_review_panel()
        self._goto_next_review()

    def _import_validation_log(self):
        log_fp, _ = QFileDialog.getOpenFileName(
            self, "Import Review Log", "", "Text Files (*.txt);;All Files (*)"
        )
        if not log_fp:
            return
        ann_fp, _ = QFileDialog.getOpenFileName(
            self, "Load original annotations", "", "JSON Files (*.json)"
        )
        if not ann_fp:
            return
        # load base annotations
        if not self._load_json_annotations(ann_fp):
            return
        entries = self._parse_validation_log(log_fp)
        if not entries:
            QMessageBox.information(
                self, "Info", "No valid modification entries found in the log."
            )
            return

        # assign defaults for legacy logs
        active_name = self.views[self.active_view_idx]["name"] if self.views else "view"
        for it in entries:
            if not it.get("view"):
                it["view"] = active_name

        missing_views = {
            it["view"] for it in entries if self._view_idx_by_name(it["view"]) < 0
        }
        if missing_views:
            QMessageBox.warning(
                self,
                "Missing views",
                f"Log references views that are not loaded: {', '.join(sorted(missing_views))}.\nLoad the matching videos/views first.",
            )
            return

        self.review_items = entries
        self.review_idx = 0
        self._set_review_panel_visible(True)
        self._goto_review_idx(0)
        self._set_status(
            f"Loaded {len(entries)} changes for review. Review Queue opened."
        )

    def _normalize_native_segments_for_load(
        self, segs: Any, labels_data: Any = None
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], Dict[int, str]]:
        if not isinstance(segs, list):
            return None, "Invalid JSON: 'segments' must be a list.", {}
        if not segs:
            return [], None, {}

        normalized: List[Dict[str, Any]] = []
        issues: List[str] = []
        legacy_like = False
        id_name_hints: Dict[int, str] = {}
        label_name_to_id: Dict[str, int] = {}

        if isinstance(labels_data, list):
            for item in labels_data:
                if not isinstance(item, dict):
                    continue
                try:
                    lid = int(item.get("id"))
                except Exception:
                    continue
                if lid < 0:
                    continue
                name = str(item.get("name", f"Label_{lid}")).strip() or f"Label_{lid}"
                id_name_hints.setdefault(lid, name)
                label_name_to_id.setdefault(name, lid)
        next_auto_lid = max(id_name_hints.keys(), default=-1) + 1

        def _parse_span(seg: Dict[str, Any]) -> Optional[Tuple[int, int]]:
            aliases = (
                ("start_frame", "end_frame"),
                ("f_start", "f_end"),
                ("start", "end"),
                ("frame_start", "frame_end"),
            )
            for sk, ek in aliases:
                raw_s = seg.get(sk)
                raw_e = seg.get(ek)
                if raw_s is None and raw_e is None:
                    continue
                try:
                    s = int(raw_s if raw_s is not None else raw_e)
                    e = int(raw_e if raw_e is not None else s)
                except Exception:
                    continue
                if e < s:
                    s, e = e, s
                return s, e
            return None

        for idx, seg in enumerate(segs):
            if not isinstance(seg, dict):
                issues.append(f"segment[{idx}] is not an object")
                continue

            if "action_label" not in seg and (
                "f_start" in seg
                or "f_end" in seg
                or "label" in seg
                or "label_id" in seg
                or "id" in seg
            ):
                legacy_like = True

            span = _parse_span(seg)
            if span is None:
                issues.append(f"segment[{idx}] missing valid start/end fields")
                continue
            s, e = span

            label_name = None
            for key in ("label", "label_name", "action_name", "name"):
                raw_name = seg.get(key)
                if raw_name is None:
                    continue
                name = str(raw_name).strip()
                if name:
                    label_name = name
                    break

            lid: Optional[int] = label_name_to_id.get(label_name) if label_name else None
            if lid is None:
                for key in ("action_label", "label_id", "id"):
                    raw = seg.get(key)
                    if raw in (None, ""):
                        continue
                    try:
                        lid = int(raw)
                        break
                    except Exception:
                        pass

            if lid is None and label_name:
                lid = next_auto_lid
                next_auto_lid += 1
                label_name_to_id[label_name] = lid
                id_name_hints.setdefault(lid, label_name)

            if lid is None:
                issues.append(
                    f"segment[{idx}] missing action label (action_label/label_id/id/label)"
                )
                continue

            if lid < 0:
                issues.append(f"segment[{idx}] has invalid action_label={lid}")
                continue

            if label_name:
                id_name_hints.setdefault(lid, label_name)
                label_name_to_id.setdefault(label_name, lid)
            else:
                id_name_hints.setdefault(lid, f"Label_{lid}")

            seg_norm = dict(seg)
            seg_norm["action_label"] = lid
            seg_norm["start_frame"] = s
            seg_norm["end_frame"] = e
            normalized.append(seg_norm)

        if issues:
            if legacy_like:
                preview = "\n".join(f"- {x}" for x in issues[:5])
                more = ""
                if len(issues) > 5:
                    more = f"\n... and {len(issues) - 5} more issue(s)."
                msg = (
                    "Legacy segment format detected, but some segments are invalid:\n"
                    f"{preview}{more}"
                )
            else:
                preview = "\n".join(f"- {x}" for x in issues[:5])
                more = ""
                if len(issues) > 5:
                    more = f"\n... and {len(issues) - 5} more issue(s)."
                msg = f"Invalid native JSON segments:\n{preview}{more}"
            return None, msg, {}

        return normalized, None, id_name_hints

    @staticmethod
    def _convert_half_open_segments_if_needed(
        segs: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], bool, Dict[str, int]]:
        if not isinstance(segs, list) or not segs:
            return segs or [], False, {}

        buckets: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {}
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            try:
                s = int(seg.get("start_frame", 0))
                e = int(seg.get("end_frame", s))
            except Exception:
                continue
            if e < s:
                s, e = e, s
            ent = str(seg.get("entity") or "__global__")
            buckets.setdefault(ent, []).append((s, e, seg))

        half_open_entities = set()
        for ent, rows in buckets.items():
            rows.sort(key=lambda x: (x[0], x[1]))
            touch_equal = 0
            touch_plus = 0
            for (s1, e1, _a), (s2, _e2, _b) in zip(rows, rows[1:]):
                if s2 == e1:
                    touch_equal += 1
                elif s2 == e1 + 1:
                    touch_plus += 1
            # Strong signal: boundaries mostly use [start, end) style (next.start == prev.end).
            if touch_equal >= 5 and touch_equal >= (touch_plus * 3):
                half_open_entities.add(ent)

        if not half_open_entities:
            return segs, False, {}

        out: List[Dict[str, Any]] = []
        changed = 0
        for seg in segs:
            if not isinstance(seg, dict):
                out.append(seg)
                continue
            ent = str(seg.get("entity") or "__global__")
            if ent not in half_open_entities:
                out.append(seg)
                continue
            try:
                s = int(seg.get("start_frame", 0))
                e = int(seg.get("end_frame", s))
            except Exception:
                out.append(seg)
                continue
            if e < s:
                s, e = e, s
            if e > s:
                seg2 = dict(seg)
                seg2["end_frame"] = int(e - 1)
                out.append(seg2)
                changed += 1
            else:
                out.append(seg)
        stats = {"entities": len(half_open_entities), "segments_adjusted": int(changed)}
        return out, bool(changed), stats

    def _load_json_annotations(
        self,
        path: str = None,
        current_view_only: bool = False,
        target_view_idx: Optional[int] = None,
    ):
        fp = path
        if not fp:
            fp, _ = QFileDialog.getOpenFileName(
                self, "Load JSON", "", "JSON Files (*.json)"
            )
        if not fp:
            return False
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as ex:
            QMessageBox.warning(self, "Load JSON", f"Failed to read JSON:\n{ex}")
            return False
        if not isinstance(data, dict):
            QMessageBox.warning(self, "Load JSON", "Invalid JSON: root must be an object.")
            return False

        labels_data = data.get("labels") or data.get("action_labels")
        if labels_data is not None and not isinstance(labels_data, list):
            QMessageBox.warning(
                self,
                "Load JSON",
                "Invalid JSON: 'labels'/'action_labels' must be a list. Falling back to ids from segments.",
            )
            labels_data = []
        segs_raw = data.get("segments", [])
        segs, seg_err, label_name_hints = self._normalize_native_segments_for_load(
            segs_raw, labels_data
        )
        if seg_err:
            QMessageBox.warning(self, "Load JSON", seg_err)
            return False
        segs = segs or []
        segs, half_open_fixed, half_open_stats = self._convert_half_open_segments_if_needed(
            segs
        )
        if half_open_fixed:
            self._set_status(
                "Detected legacy [start,end) segments; converted end_frame for load."
            )
            self._log(
                "load_json_half_open_adjust",
                entities=half_open_stats.get("entities", 0),
                adjusted=half_open_stats.get("segments_adjusted", 0),
            )
        # fine vocab (optional)
        verbs_data = data.get("verbs")
        nouns_data = data.get("nouns")
        anomaly_types_data = data.get("anomaly_types")
        self.fine_verbs = verbs_data if isinstance(verbs_data, list) else []
        self.fine_nouns = nouns_data if isinstance(nouns_data, list) else []
        if isinstance(anomaly_types_data, list):
            self._ensure_anomaly_types(anomaly_types_data)
        else:
            self._ensure_anomaly_types()
        self._refresh_anomaly_type_panel(force=True)
        if (
            getattr(self, "anomaly_list", None) is not None
            and self.anomaly_list.count() == 0
        ):
            self._rebuild_anomaly_type_panel()

        def _has_anomaly_mark(value: Any) -> bool:
            if isinstance(value, list):
                for item in value:
                    try:
                        if int(item) != 0:
                            return True
                    except Exception:
                        if bool(item):
                            return True
                return False
            if isinstance(value, dict):
                for item in value.values():
                    try:
                        if int(item) != 0:
                            return True
                    except Exception:
                        if bool(item):
                            return True
                return False
            if value is None:
                return False
            try:
                return int(value) != 0
            except Exception:
                return bool(str(value).strip())

        has_phase_data = any(
            isinstance(seg, dict)
            and (
                str(seg.get("phase") or "").strip() != ""
                or _has_anomaly_mark(seg.get("anomaly_type"))
            )
            for seg in segs
        )
        self.phase_mode_enabled = bool(has_phase_data)
        self._phase_selected = None

        def _to_int_safe(value: Any, default: int) -> int:
            try:
                return int(value)
            except Exception:
                return int(default)

        meta_data = data.get("meta_data") if isinstance(data, dict) else None
        meta_view_start = None
        meta_view_end = None
        if isinstance(meta_data, dict):
            if meta_data.get("view_start") is not None:
                meta_view_start = _to_int_safe(meta_data.get("view_start", 0), 0)
            if meta_data.get("view_end") is not None:
                meta_view_end = _to_int_safe(meta_data.get("view_end"), 0)
        view_start_meta = _to_int_safe(data.get("view_start", meta_view_start or 0), 0)
        view_end_meta = data.get("view_end", meta_view_end)
        # use current view's crop start as default offset
        if target_view_idx is None:
            current_idx = int(self.active_view_idx)
        else:
            try:
                current_idx = int(target_view_idx)
            except Exception:
                QMessageBox.warning(self, "Load JSON", "Invalid target view index.")
                return False
            if not (0 <= current_idx < len(self.views)):
                QMessageBox.warning(self, "Load JSON", "Target view index out of range.")
                return False
        current_view = self.views[current_idx] if self.views else {}
        primary_view_start = (
            view_start_meta
            if view_start_meta is not None
            else _to_int_safe(current_view.get("start", 0), 0)
        )
        primary_view_end = (
            _to_int_safe(view_end_meta, primary_view_start)
            if view_end_meta is not None
            else _to_int_safe(
                current_view.get(
                    "end",
                    self.player.crop_end if self.player.cap else primary_view_start,
                ),
                primary_view_start,
            )
        )
        # warn if annotations exceed the currently loaded video/crop
        if segs:
            try:
                rel_bounds = [
                    (int(seg.get("start_frame")), int(seg.get("end_frame"))) for seg in segs
                ]
                abs_starts = [primary_view_start + s_rel for s_rel, _ in rel_bounds]
                abs_ends = [primary_view_start + e_rel for _, e_rel in rel_bounds]
                min_ann = min(abs_starts + abs_ends)
                max_ann = max(abs_starts + abs_ends)
                crop_start = current_view.get(
                    "start",
                    self.player.crop_start if getattr(self.player, "cap", None) else 0,
                )
                crop_end = current_view.get(
                    "end",
                    (
                        self.player.crop_end
                        if getattr(self.player, "cap", None)
                        else max_ann
                    ),
                )
                if min_ann < crop_start or max_ann > crop_end:
                    QMessageBox.information(
                        self,
                        "Annotation range warning",
                        f"Annotations [{min_ann}, {max_ann}] extend beyond the current crop/video range [{crop_start}, {crop_end}].\n"
                        "They will be clamped to the loaded video range.",
                    )
            except Exception:
                pass

        # --- rebuild labels (with color) ---
        self.labels.clear()
        id2label = {}

        if labels_data:
            # current format: restore directly from JSON
            label_items = [it for it in labels_data if isinstance(it, dict)]
            bad_label_items = 0
            def _label_sort_key(item: Dict[str, Any]) -> int:
                try:
                    return int(item.get("id", 0))
                except Exception:
                    return 10**9

            for item in sorted(label_items, key=_label_sort_key):
                try:
                    lid = int(item.get("id"))
                except Exception:
                    bad_label_items += 1
                    continue
                name = str(item.get("name", f"Label_{lid}")).strip() or f"Label_{lid}"
                col = item.get("color")
                # missing/invalid color key -> auto assign by id
                if not col or (
                    col not in PRESET_COLORS and not str(col).startswith("custom:")
                ):
                    col = self._auto_color_key_for_id(lid)
                self.labels.append(LabelDef(name=name, color_name=col, id=lid))
                id2label[lid] = name
            if bad_label_items:
                QMessageBox.warning(
                    self,
                    "Load JSON",
                    f"Ignored {bad_label_items} invalid label item(s) (missing/invalid id).",
                )

        # If label table is missing or incomplete, synthesize labels from action ids.
        lids = sorted({int(seg.get("action_label")) for seg in segs})
        for lid in lids:
            if lid in id2label:
                continue
            name = str(label_name_hints.get(lid, f"Label_{lid}"))
            col = self._auto_color_key_for_id(lid)
            self.labels.append(LabelDef(name=name, color_name=col, id=lid))
            id2label[lid] = name

        if "psr/asr/asd" in (self.combo_task.currentText() or "").lower():
            self._ensure_psr_asr_asd_invisible_label(refresh=False)
        self._refresh_fine_label_decomposition(refresh_panel=False)
        # refresh UI to show new labels
        self.panel.refresh()
        self._rebuild_timeline_sources()

        # --- clear old annotations for target views ---
        if target_view_idx is not None:
            targets = [self.views[current_idx]] if self.views else []
        elif current_view_only:
            targets = [self.views[self.active_view_idx]] if self.views else []
        else:
            targets = self._target_views_for_annotation_load()
        target_ids = {id(view) for view in targets if isinstance(view, dict)}
        for view in targets:
            st = view.get("store")
            if st:
                st.frame_to_label.clear()
                st.label_to_frames.clear()
            stores = view.setdefault("entity_stores", {})
            for ent_store in stores.values():
                ent_store.frame_to_label.clear()
                ent_store.label_to_frames.clear()
            pstores = view.setdefault("phase_stores", {})
            for pstore in pstores.values():
                pstore.frame_to_label.clear()
                pstore.label_to_frames.clear()
            astores = view.setdefault("anomaly_type_stores", {})
            for ent_map in astores.values():
                for astore in ent_map.values():
                    astore.frame_to_label.clear()
                    astore.label_to_frames.clear()
            # Imported annotations should start from a clean split state; otherwise
            # old manual trim cuts can make phase/action boundaries look mismatched.
            view["psr_state"] = self._psr_empty_view_state()
            view["trim_cuts"] = self._empty_trim_state()

        # --- detect if entity field exists ---
        entities_meta = data.get("entities")
        entity_lookup = {}
        if isinstance(entities_meta, dict):
            for k, v in entities_meta.items():
                key = str(k).strip()
                name = str(v).strip()
                if key and name:
                    entity_lookup[key] = name
                    if key.isdigit():
                        entity_lookup[int(key)] = name
        elif isinstance(entities_meta, list):
            for idx, item in enumerate(entities_meta):
                if isinstance(item, dict):
                    key = item.get("id", idx)
                    name = item.get("name")
                else:
                    key = idx
                    name = item
                key_str = str(key).strip()
                name_str = str(name).strip()
                if key_str and name_str:
                    entity_lookup[key_str] = name_str
                    if key_str.isdigit():
                        entity_lookup[int(key_str)] = name_str

        def _normalize_entity(val):
            if val is None:
                return None
            if isinstance(val, (int, float)):
                key = int(val)
                return entity_lookup.get(key, str(key))
            val_str = str(val).strip()
            if not val_str:
                return None
            if val_str.isdigit():
                key = int(val_str)
                return entity_lookup.get(key, entity_lookup.get(val_str, val_str))
            return entity_lookup.get(val_str, val_str)

        has_entity = any(
            isinstance(seg, dict) and "entity" in seg for seg in segs
        ) or bool(entity_lookup)
        target_mode = "Fine" if (has_entity or has_phase_data) else "Coarse"
        try:
            self.combo_mode.setCurrentText(target_mode)
        except Exception:
            self.mode = target_mode

        # --- assign ids to new entities ---
        if entity_lookup:
            entity_names = sorted({v for v in entity_lookup.values() if v})
        else:
            entity_names = sorted(
                {
                    _normalize_entity(seg.get("entity"))
                    for seg in segs
                    if isinstance(seg, dict)
                    and "entity" in seg
                    and _normalize_entity(seg.get("entity"))
                }
            )
        next_eid = max([e.id for e in self.entities], default=-1) + 1
        for ename in entity_names:
            if not any(x.name == ename for x in self.entities):
                self.entities.append(EntityDef(name=ename, id=next_eid))
                next_eid += 1

        labels_added = False

        def _label_name_from_segment(seg: Dict[str, Any]) -> str:
            nonlocal labels_added
            lid = int(seg.get("action_label"))
            name = id2label.get(lid)
            if name:
                return name
            name = str(label_name_hints.get(lid, f"Label_{lid}"))
            id2label[lid] = name
            if not any(lb.id == lid for lb in self.labels):
                col = self._auto_color_key_for_id(lid)
                self.labels.append(LabelDef(name=name, color_name=col, id=lid))
                labels_added = True
            return name

        # --- build label-to-entity mapping ---
        for seg in segs:
            ename = (
                _normalize_entity(seg.get("entity")) if isinstance(seg, dict) else None
            )
            if ename is None:
                continue
            name = _label_name_from_segment(seg)
            self.label_entity_map.setdefault(name, set()).add(ename)

        # --- apply segments to target views ---
        for view in targets:
            view_store = view.get("store")
            view_entities = view.setdefault("entity_stores", {})
            for ename in entity_names:
                view_entities.setdefault(ename, AnnotationStore())
            view_phase_stores = view.setdefault("phase_stores", {})
            view_anom_stores = view.setdefault("anomaly_type_stores", {})
            for ename in entity_names:
                view_phase_stores.setdefault(ename, AnnotationStore())
                ent_anom = view_anom_stores.setdefault(ename, {})
                for tname in self._anomaly_type_names():
                    ent_anom.setdefault(tname, AnnotationStore())
            if view is current_view:
                view_start = primary_view_start
                view_end = primary_view_end
            else:
                view_start = int(view.get("start", 0))
                view_end = int(view.get("end", view_start))

            for seg in segs:
                s_rel = int(seg.get("start_frame", 0))
                e_rel = int(seg.get("end_frame", s_rel))
                s = s_rel + view_start
                e = e_rel + view_start
                if e < view_start or (view_end and s > view_end):
                    continue
                if view_end:
                    e = min(e, view_end)
                name = _label_name_from_segment(seg)
                ename = _normalize_entity(seg.get("entity")) if has_entity else None

                target = self.extra_store if is_extra_label(name) else view_store
                if ename and self.mode == "Fine":
                    target = view_entities.setdefault(ename, AnnotationStore())

                for f in range(s, e + 1):
                    if (not target.is_occupied(f)) or (target.label_at(f) == name):
                        target.add(name, f)

                # phase + anomaly types (fine JSON)
                if self.phase_mode_enabled and ename:
                    phase_val = str(seg.get("phase") or "").strip().lower()
                    if phase_val in {lb.name for lb in self.phase_labels}:
                        pstore = view_phase_stores.setdefault(ename, AnnotationStore())
                        self._apply_label_range(pstore, s, e, phase_val)
                    anom_vec = seg.get("anomaly_type")
                    if isinstance(anom_vec, list):
                        names = self._anomaly_type_names()
                        ent_anom = view_anom_stores.setdefault(ename, {})
                        for idx, val in enumerate(anom_vec):
                            if idx >= len(names):
                                break
                            try:
                                has_anomaly = int(val) != 0
                            except Exception:
                                has_anomaly = bool(val)
                            if has_anomaly:
                                astore = ent_anom.setdefault(
                                    names[idx], AnnotationStore()
                                )
                                self._apply_label_range(astore, s, e, names[idx])

        self._sync_extra_cuts_from_store()
        # refresh UI
        if self.views and 0 <= self.active_view_idx < len(self.views):
            self.entity_stores = self.views[self.active_view_idx].get(
                "entity_stores", {}
            )
            self.phase_stores = self.views[self.active_view_idx].get("phase_stores", {})
            self.anomaly_type_stores = self.views[self.active_view_idx].get(
                "anomaly_type_stores", {}
            )
            if id(self.views[self.active_view_idx]) in target_ids:
                self._psr_load_view_state(self.views[self.active_view_idx])
        self._refresh_fine_label_decomposition(refresh_panel=False)
        self.panel.refresh()
        self.entities_panel.refresh(
            self.label_entity_map.get(
                (
                    self.labels[self.current_label_idx].name
                    if 0 <= self.current_label_idx < len(self.labels)
                    else None
                ),
                set(),
            )
        )
        self._update_phase_panel_visibility()
        self._refresh_anomaly_type_panel(force=True)
        self._rebuild_timeline_sources()
        self.timeline.update()
        # Loading/import is baseline data, not an undoable user edit.
        # Drain any transaction deltas produced during import to avoid first Ctrl+Z
        # rolling back large phase/anomaly ranges from the load step.
        self._drain_pending_store_deltas()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.current_annotation_path = fp
        self._set_status(f"Loaded annotations from {os.path.basename(fp)}")
        self._log(
            "load_annotations", path=fp, segments=len(segs), labels=len(self.labels)
        )
        self._load_extra_sidecar(fp)
        self._rebuild_timeline_sources()
        try:
            self.timeline.refresh_all_rows()
        except Exception:
            self.timeline.update()
        self._update_gap_indicator()
        self._psr_mark_dirty()
        self._psr_update_component_panel()
        return True

    # ===== Label map TXT I/O =====
    def _import_label_map_txt(self, path: str = ""):
        fp = str(path or "").strip()
        if not fp:
            fp, _ = QFileDialog.getOpenFileName(
                self, "Import label map", "", "Text Files (*.txt)"
            )
        if not fp:
            return
        mapping = []  # (name, id)
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                name = parts[0]
                try:
                    lid = int(parts[1])
                except Exception:
                    continue
                mapping.append((name, lid))
        if not mapping:
            QMessageBox.information(self, "Info", "No valid entries found.")
            return

        self._replace_action_labels(mapping, label_source_path=fp)
        self._set_status(f"Imported {len(mapping)} labels")
        self._log("import_label_map", path=fp, count=len(mapping))

    def _export_label_map_txt(self):
        if not self.labels:
            QMessageBox.information(self, "Info", "No labels to export.")
            return
        fp, _ = QFileDialog.getSaveFileName(
            self, "Export label map", "", "Text Files (*.txt)"
        )
        if not fp:
            return
        with open(fp, "w", encoding="utf-8") as f:
            for lb in sorted(self.labels, key=lambda x: x.id):
                f.write(f"{lb.name} {lb.id}\n")
        self._set_status(f"Exported label map to {os.path.basename(fp)}")
        self._log("export_label_map", path=fp, count=len(self.labels))
