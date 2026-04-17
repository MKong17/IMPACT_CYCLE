from __future__ import annotations

import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import uuid
import zipfile
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
from PyQt5.QtCore import QObject, QSize, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QFileDialog,
    QGridLayout,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QComboBox,
    QScrollArea,
    QShortcut,
    QSlider,
    QSpinBox,
    QSplitter,
    QSizePolicy,
    QStackedLayout,
    QStyle,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.validation_log import (
    apply_decision,
    export_ndjson,
    filter_by_task,
    import_ndjson,
    merge_changes,
    new_change,
    now_iso,
    summarize,
)
from core.ui_feature_service import UIFeatureService
from core.impact_sg.correction_memory import (
    default_correction_memory,
    load_correction_memory,
    merge_correction_memories,
    save_correction_memory,
    summarize_correction_memory,
    update_memory_from_human_decision,
)
from core.impact_sg.cycle_pipeline import rerun_cycle_refine_for_claims, run_cycle_refine
from core.impact_sg.geometry_review import rebuild_spatial_edges
from core.impact_sg.mllm_adapters.defaults import (
    DEFAULT_API_MAX_OUTPUT_TOKENS,
    DEFAULT_API_TIMEOUT_SEC,
    DEFAULT_CYCLE_PROVIDER,
    DEFAULT_GEMINI_API_KEY_ENV,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_ONLINE_TIMEOUT_SEC,
    DEFAULT_OPENAI_API_KEY_ENV,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    LOW_QUOTA_API_MAX_OUTPUT_TOKENS,
    cycle_provider_display_name,
    normalize_cycle_provider,
)
from core.impact_sg.mllm_adapters.factory import build_vision_verifier
from core.impact_sg.ontology import load_ontology, ontology_from_payload
from core.impact_sg.pvsg_reference import load_pvsg_video_reference
from core.impact_sg.stage_validator import StageValidator
from core.impact_sg.pipeline import release_backend_pool, run_build_scene_graph
from core.impact_sg.scene_graph_builder import build_scene_graph
from core.impact_sg.video_sampling import sample_frame_indices
from core.impact_sg.tracking import (
    clip_bbox,
    crop_patch,
    greedy_track_match,
    smooth_bbox,
    template_track_bbox,
    update_template,
)
from core.impact_sg.vqa import generate_multi_turn_vqa, generate_single_turn_vqa
from ui.video_player import VideoPlayer
from ui.ontology_editor import OntologyEditor

# Person filtering / prioritization defaults.
PERSON_MIN_BBOX_AREA = 256
PERSON_MIN_BBOX_WIDTH = 8
PERSON_MIN_BBOX_HEIGHT = 8
PERSON_MIN_AREA_RATIO = 0.0004
PERSON_PRIORITY_TOPK = 4
PERSON_CENTER_BIAS_WEIGHT = 0.10

# Additional defaults for behavior control.
PERSON_FILTERED_MIN_SCORE = 0.15
PERSON_HIGH_MIN_AREA_RATIO = 0.003
PERSON_HIGH_CENTER_MIN_AREA_RATIO = 0.0015
PERSON_HIGH_CENTER_MAX_DISTANCE_NORM = 0.45
PERSON_HIGH_MAX_PER_FRAME = 6
PERSON_LOW_PRIORITY_TRACKING_MODE = "none"
PERSON_LOW_PRIORITY_MAX_LOST_FRAMES = 2
PERSON_FILTERED_DEBUG_KEEP = False
PERSON_TRACK_DEMOTE_POLICY = "terminate"

MASK_EXPORT_MODE = "stats_only"  # none | stats_only | rle | external_png
LOW_CONF_THRESHOLD = 0.5
SCENE_GRAPH_PROGRESS_DEFAULT_WIDTH = 560
QWEN_DEFAULT_MODEL_PATH = "/cvhci/temp/wkong/models/Qwen2.5-VL-3B-Instruct" if os.name != "nt" else r"D:\_My\KIT\Qwen2.5-VL-3B-Instruct"
EDGE_RELATION_UI_CORE = [
    "left_of",
    "right_of",
    "above",
    "below",
    "in_front_of",
    "behind",
]
NODE_LOW_CONF_THRESHOLD = 0.40
EDGE_LOW_CONF_THRESHOLD = 0.55


def _now_iso_utc() -> str:
    t = time.time()
    base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t))
    ms = int((t - int(t)) * 1000.0)
    return f"{base}.{ms:03d}Z"


def _abs_time_tag_local() -> str:
    return time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())


def _run_log_folder_tag_local() -> str:
    return time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())


def _sanitize_name_token(value: object, *, default: str = "unknown") -> str:
    token = str(value or "").strip().lower()
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("._-")
    return token or str(default)


def _build_session_stem(participant_id: str, video_path: str) -> str:
    pid = _sanitize_name_token(participant_id, default="p_unknown")
    video_stem = os.path.splitext(os.path.basename(str(video_path or "").strip()))[0]
    video = _sanitize_name_token(video_stem, default="video")
    ts = _run_log_folder_tag_local()
    return f"{pid}_{video}_{ts}"


def _append_timing_jsonl(path: str, event: str, **fields: object) -> None:
    log_path = str(path or "").strip()
    if not log_path:
        return
    payload: Dict[str, object] = {"ts": _now_iso_utc(), "event": str(event or "").strip()}
    for k, v in fields.items():
        payload[str(k)] = v
    try:
        folder = os.path.dirname(log_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _write_json(path: str, payload: Dict[str, object]) -> None:
    out_path = str(path or "").strip()
    if not out_path:
        return
    folder = os.path.dirname(out_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, out_path)


def _read_json(path: str) -> Dict[str, object]:
    src = str(path or "").strip()
    if not src or (not os.path.isfile(src)):
        return {}
    try:
        with open(src, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return dict(payload)
    except Exception:
        return {}
    return {}


def _run_artifact_manifest(run_dir: str) -> Dict[str, object]:
    folder = str(run_dir or "").strip()
    if not folder or not os.path.isdir(folder):
        return {"file_count": 0, "types": {}}
    counts: Dict[str, int] = {}
    total = 0
    for root, _, files in os.walk(folder):
        for name in files:
            total += 1
            ext = os.path.splitext(name)[1].lower() or "<noext>"
            counts[ext] = int(counts.get(ext, 0) + 1)
    return {"file_count": int(total), "types": counts}

class ClickSeekSlider(QSlider):
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            minimum = int(self.minimum())
            maximum = int(self.maximum())
            if maximum > minimum:
                span = max(1, self.width())
                x_pos = max(0, min(int(event.pos().x()), span))
                value = QStyle.sliderValueFromPosition(minimum, maximum, x_pos, span)
                self.setValue(int(value))
                self.sliderMoved.emit(int(value))
        super().mousePressEvent(event)


def _extract_video_frame_to_cache(
    video_path: str,
    frame_idx: int,
    frame_cache_dir: str,
    *,
    cap=None,
) -> Tuple[str, int, int, Any]:
    if not video_path or not os.path.isfile(video_path):
        raise RuntimeError("Video file is missing. Please open a video first.")

    owns_cap = cap is None
    if owns_cap:
        cap = cv2.VideoCapture(video_path)
    if cap is None or not cap.isOpened():
        raise RuntimeError("Unable to open video.")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
    finally:
        if owns_cap:
            cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Unable to read frame {frame_idx}.")

    os.makedirs(frame_cache_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_name = f"{stem}_f{int(frame_idx):06d}.jpg"
    out_path = os.path.join(frame_cache_dir, out_name)
    if not cv2.imwrite(out_path, frame):
        raise RuntimeError("Unable to save extracted frame image.")

    h, w = frame.shape[:2]
    return out_path, int(w), int(h), frame


class SceneGraphBuildWorker(QObject):
    progress = pyqtSignal(str)
    frame_ready = pyqtSignal(object)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        video_path: str,
        frame_cache_dir: str,
        frame_indices: List[int],
        ontology_path: str,
        pipeline_cfg_path: str,
        enable_sentence_refine: bool,
        custom_ontology_dict: Optional[Dict[str, object]],
        backend_override: Optional[Dict[str, object]],
        source_fps: float,
        batch_size: int = 8,
        timing_log_path: str = "",
    ):
        super().__init__()
        self.video_path = str(video_path or "")
        self.frame_cache_dir = str(frame_cache_dir or "")
        self.frame_indices = [int(x) for x in (frame_indices or [])]
        self.ontology_path = str(ontology_path or "")
        self.pipeline_cfg_path = str(pipeline_cfg_path or "")
        self.enable_sentence_refine = bool(enable_sentence_refine)
        self.custom_ontology_dict = json.loads(json.dumps(custom_ontology_dict)) if isinstance(custom_ontology_dict, dict) else None
        self.backend_override = json.loads(json.dumps(backend_override)) if isinstance(backend_override, dict) else None
        self.source_fps = max(0.1, float(source_fps or 1.0))
        self.batch_size = max(1, int(batch_size or 8))
        self.timing_log_path = str(timing_log_path or "").strip()
        self._cancel_requested = False

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        job_t0 = time.perf_counter()
        _append_timing_jsonl(
            self.timing_log_path,
            "sg_worker_start",
            video_path=self.video_path,
            total_frames=int(len(self.frame_indices)),
            batch_size=int(self.batch_size),
        )
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.failed.emit("Unable to open the current video for scene graph generation.")
            return

        processed = 0
        total = len(self.frame_indices)
        stem = os.path.splitext(os.path.basename(self.video_path or "video"))[0] or "video"
        try:
            batch_size = max(1, int(self.batch_size or 1))
            for batch_start in range(0, total, batch_size):
                if self._cancel_requested:
                    break
                batch_end = min(total, batch_start + batch_size)
                self.progress.emit(
                    f"Preparing frame batch {batch_start + 1}-{batch_end}/{total}"
                )
                batch_t0 = time.perf_counter()
                batch_frames: List[Tuple[int, str, int, int]] = []
                for idx in range(batch_start, batch_end):
                    frame_idx = int(self.frame_indices[idx])
                    extract_t0 = time.perf_counter()
                    img_path, w, h, _ = _extract_video_frame_to_cache(
                        self.video_path,
                        frame_idx,
                        self.frame_cache_dir,
                        cap=cap,
                    )
                    _append_timing_jsonl(
                        self.timing_log_path,
                        "frame_extract_done",
                        frame_idx=int(frame_idx),
                        sec=float(time.perf_counter() - extract_t0),
                    )
                    batch_frames.append((frame_idx, img_path, int(w), int(h)))
                _append_timing_jsonl(
                    self.timing_log_path,
                    "batch_prepare_done",
                    batch_start=int(batch_start + 1),
                    batch_end=int(batch_end),
                    sec=float(time.perf_counter() - batch_t0),
                )
                for local_idx, (frame_idx, img_path, w, h) in enumerate(batch_frames, start=1):
                    if self._cancel_requested:
                        break
                    index = batch_start + local_idx
                    self.progress.emit(f"Running scene graph model on frame {index}/{total} (frame={frame_idx})")
                    frame_t0 = time.perf_counter()
                    graph = run_build_scene_graph(
                        image_id=f"{stem}_f{frame_idx:06d}",
                        image_path=img_path,
                        ontology_path=self.ontology_path,
                        pipeline_cfg_path=self.pipeline_cfg_path,
                        image_size=(w, h),
                        enable_sentence_refine=bool(self.enable_sentence_refine),
                        custom_ontology_dict=self.custom_ontology_dict,
                        pipeline_cfg_override={"backend": self.backend_override} if self.backend_override else None,
                        progress_cb=self.progress.emit,
                        cancel_cb=lambda: bool(self._cancel_requested),
                    )
                    frame_sec = float(time.perf_counter() - frame_t0)
                    _append_timing_jsonl(
                        self.timing_log_path,
                        "frame_sg_done",
                        frame_idx=int(frame_idx),
                        frame_pos=int(index),
                        total_frames=int(total),
                        sec=frame_sec,
                    )
                    self.progress.emit(f"[SG-TIMING] frame={frame_idx} sg_total={frame_sec:.3f}s ({index}/{total})")
                    processed += 1
                    self.frame_ready.emit(
                        {
                            "index": int(index),
                            "total": int(total),
                            "frame_idx": int(frame_idx),
                            "image_path": img_path,
                            "image_size": [int(w), int(h)],
                            "graph": graph,
                        }
                    )
            self.done.emit(
                {
                    "cancelled": bool(self._cancel_requested),
                    "processed_frames": int(processed),
                    "total_frames": int(total),
                    "timing_log_path": self.timing_log_path,
                }
            )
        except Exception as exc:
            if self._cancel_requested:
                self.done.emit(
                    {
                        "cancelled": True,
                        "processed_frames": int(processed),
                        "total_frames": int(total),
                        "timing_log_path": self.timing_log_path,
                    }
                )
            else:
                self.failed.emit(str(exc))
        finally:
            _append_timing_jsonl(
                self.timing_log_path,
                "sg_worker_end",
                cancelled=bool(self._cancel_requested),
                processed_frames=int(processed),
                total_frames=int(total),
                sec=float(time.perf_counter() - job_t0),
            )
            cap.release()
            release_backend_pool()


class LLMBatchSummaryWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        repo_root: str,
        bundle_json: str,
        model_path: str,
        batch_size: int = 5,
        cuda_device: str = "",
        timing_log_path: str = "",
    ):
        super().__init__()
        self.repo_root = str(repo_root or "")
        self.bundle_json = str(bundle_json or "")
        self.model_path = str(model_path or "")
        self.batch_size = max(1, int(batch_size or 5))
        self.cuda_device = str(cuda_device or "").strip()
        self.timing_log_path = str(timing_log_path or "").strip()
        self._cancel_requested = False
        self._proc: Optional[subprocess.Popen] = None

    def _sam_repo_root_from_runtime(self) -> str:
        try:
            runtime_cfg = os.path.join(self.repo_root, "configs", "sam3_runtime.linux.json")
            with open(runtime_cfg, "r", encoding="utf-8") as f:
                payload = json.load(f)
            root = str((payload or {}).get("repo_root", "") or "").strip()
            return os.path.abspath(root) if root else ""
        except Exception:
            return ""

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
            return True
        except Exception:
            return False

    def _force_stop_sam_processes(self) -> int:
        """
        Best-effort cleanup for lingering SAM3 server/infer processes before Qwen starts.
        This avoids GPU OOM from concurrent SAM+Qwen residency on single-GPU machines.
        """
        repo_root = self._sam_repo_root_from_runtime()
        keywords = ["/tools/sam3_infer.py", "sam3_runtime.linux.json", "--serve_jsonl"]
        if repo_root:
            keywords.append(repo_root)
        keywords = [k for k in keywords if str(k or "").strip()]
        if not keywords:
            return 0

        try:
            ps_out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True, encoding="utf-8", errors="replace")
        except Exception:
            return 0

        me = int(os.getpid())
        victims: List[int] = []
        for raw in str(ps_out or "").splitlines()[1:]:
            line = str(raw or "").strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except Exception:
                continue
            cmd = str(parts[1] or "")
            if pid == me:
                continue
            if "qwen_batch_summary.py" in cmd:
                continue
            if any(k in cmd for k in keywords):
                victims.append(pid)

        victims = sorted(set(victims))
        if not victims:
            return 0

        # Graceful terminate first.
        for pid in victims:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        deadline = time.time() + 3.0
        while time.time() < deadline:
            alive = [pid for pid in victims if self._pid_alive(pid)]
            if not alive:
                break
            time.sleep(0.1)

        # Force kill leftovers.
        alive = [pid for pid in victims if self._pid_alive(pid)]
        for pid in alive:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
        return int(len(victims))

    def cancel(self) -> None:
        self._cancel_requested = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def run(self) -> None:
        llm_t0 = time.perf_counter()
        runner_path = os.path.join(self.repo_root, "tools", "runners", "run_in_env.py")
        script_path = os.path.join(self.repo_root, "tools", "qwen_batch_summary.py")
        if not os.path.isfile(runner_path):
            self.failed.emit(f"Missing runner: {runner_path}")
            return
        if not os.path.isfile(script_path):
            self.failed.emit(f"Missing script: {script_path}")
            return
        if not os.path.isfile(self.bundle_json):
            self.failed.emit(f"Bundle JSON not found: {self.bundle_json}")
            return

        cmd = [
            sys.executable,
            runner_path,
            "--profile",
            "qwen",
            "--",
            script_path,
            "--bundle_json",
            self.bundle_json,
            "--model_path",
            self.model_path,
            "--batch_size",
            str(self.batch_size),
            "--out",
            self.bundle_json,
        ]
        device_hint = str(self.cuda_device or "").strip().lower()
        if device_hint:
            cmd.extend(["--device_hint", device_hint])
        if device_hint.startswith("cuda") or device_hint.isdigit():
            cmd.extend(["--require_cuda", "1"])
        try:
            killed = self._force_stop_sam_processes()
            if killed > 0:
                self.progress.emit(f"[LLM-SUMMARY] forced stop of {killed} SAM process(es) before Qwen start.")
            self.progress.emit(f"Launching LLM summary worker: batch_size={self.batch_size}")
            _append_timing_jsonl(
                self.timing_log_path,
                "llm_worker_start",
                bundle_json=self.bundle_json,
                batch_size=int(self.batch_size),
                cuda_device=self.cuda_device,
            )
            child_env = os.environ.copy()
            if device_hint.startswith("cuda:"):
                child_env["CUDA_VISIBLE_DEVICES"] = str(device_hint.split(":", 1)[1]).strip()
            elif device_hint.isdigit():
                child_env["CUDA_VISIBLE_DEVICES"] = device_hint
            elif device_hint:
                child_env["CUDA_VISIBLE_DEVICES"] = device_hint
            if not str(child_env.get("PYTORCH_CUDA_ALLOC_CONF", "") or "").strip():
                child_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            self._proc = subprocess.Popen(
                cmd,
                cwd=self.repo_root,
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert self._proc.stdout is not None
            for raw in self._proc.stdout:
                if self._cancel_requested:
                    break
                line = str(raw or "").rstrip("\r\n")
                if line:
                    if "[LLM-ATTR]" in line or "[LLM-SUMMARY]" in line:
                        _append_timing_jsonl(self.timing_log_path, "llm_progress", message=line)
                    self.progress.emit(line)
            if self._cancel_requested:
                self.done.emit({"cancelled": True, "bundle_json": self.bundle_json, "timing_log_path": self.timing_log_path})
                return
            code = int(self._proc.wait())
            if code != 0:
                self.failed.emit(f"LLM batch summary failed with exit code {code}.")
                return
            self.done.emit({"cancelled": False, "bundle_json": self.bundle_json, "timing_log_path": self.timing_log_path})
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            _append_timing_jsonl(
                self.timing_log_path,
                "llm_worker_end",
                cancelled=bool(self._cancel_requested),
                sec=float(time.perf_counter() - llm_t0),
            )
            self._proc = None


class CycleRefineWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        graph: Dict[str, object],
        image_path: str,
        ontology_path: str,
        cycle_cfg_path: str,
        custom_ontology_dict: Optional[Dict[str, object]] = None,
        correction_memory: Optional[Dict[str, object]] = None,
        cycle_cfg_override: Optional[Dict[str, object]] = None,
        target_claim_ids: Optional[List[str]] = None,
        base_result: Optional[Dict[str, object]] = None,
    ):
        super().__init__()
        self.graph = json.loads(json.dumps(graph or {}))
        self.image_path = str(image_path or "")
        self.ontology_path = str(ontology_path or "")
        self.cycle_cfg_path = str(cycle_cfg_path or "")
        self.custom_ontology_dict = json.loads(json.dumps(custom_ontology_dict)) if isinstance(custom_ontology_dict, dict) else None
        self.correction_memory = json.loads(json.dumps(correction_memory)) if isinstance(correction_memory, dict) else None
        self.cycle_cfg_override = json.loads(json.dumps(cycle_cfg_override)) if isinstance(cycle_cfg_override, dict) else None
        self.target_claim_ids = [str(x).strip() for x in list(target_claim_ids or []) if str(x).strip()]
        self.base_result = json.loads(json.dumps(base_result)) if isinstance(base_result, dict) else None

    @staticmethod
    def _merge_cfg(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
        out = dict(base or {})
        for key, value in dict(override or {}).items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = CycleRefineWorker._merge_cfg(dict(out.get(key) or {}), value)
            else:
                out[key] = value
        return out

    def _load_cycle_cfg(self) -> Dict[str, object]:
        path = os.path.abspath(os.path.expanduser(self.cycle_cfg_path))
        payload: Dict[str, object]
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8-sig") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Cycle config must be a JSON object: {path}")
        else:
            payload = {}
        if isinstance(self.cycle_cfg_override, dict):
            payload = self._merge_cfg(payload, self.cycle_cfg_override)
        return payload

    def _load_ontology(self):
        if isinstance(self.custom_ontology_dict, dict):
            return ontology_from_payload(self.custom_ontology_dict)
        return load_ontology(self.ontology_path)

    def _build_verifier(self, cycle_cfg: Dict[str, object]):
        runtime_cfg = dict(cycle_cfg.get("runtime") or {})
        allow_mock_fallback = bool(runtime_cfg.get("allow_mock_fallback", False))
        preferred_provider = str(runtime_cfg.get("preferred_provider", "") or "").strip().lower()
        if preferred_provider:
            preferred_provider = normalize_cycle_provider(preferred_provider, default=preferred_provider)
            if preferred_provider == "manual":
                preferred_provider = "mock"
        else:
            preferred_provider = "auto"
        return build_vision_verifier(
            cycle_cfg,
            preferred_provider=preferred_provider,
            api_key="",
            progress_cb=self.progress.emit,
            allow_mock_fallback=allow_mock_fallback,
        )

    def run(self) -> None:
        try:
            self.progress.emit("Loading ontology and cycle configuration...")
            cycle_cfg = self._load_cycle_cfg()
            ontology = self._load_ontology()
            self.progress.emit("Preparing cycle verifier...")
            verifier, runtime_meta = self._build_verifier(cycle_cfg)
            warning = str(runtime_meta.get("verifier_warning", "") or "").strip()
            if warning:
                self.progress.emit(warning)
            if self.target_claim_ids:
                self.progress.emit(
                    f"Running targeted cycle refinement for {len(self.target_claim_ids)} affected claim(s)..."
                )
                result = rerun_cycle_refine_for_claims(
                    graph=self.graph,
                    image_path=self.image_path,
                    verifier=verifier,
                    ontology=ontology,
                    cfg=cycle_cfg,
                    target_claim_ids=self.target_claim_ids,
                    base_result=self.base_result,
                    correction_memory=self.correction_memory,
                    progress_cb=self.progress.emit,
                )
            else:
                self.progress.emit("Running cross-task cycle refinement...")
                result = run_cycle_refine(
                    graph=self.graph,
                    image_path=self.image_path,
                    verifier=verifier,
                    ontology=ontology,
                    cfg=cycle_cfg,
                    base_result=self.base_result,
                    correction_memory=self.correction_memory,
                    progress_cb=self.progress.emit,
                )
            if not isinstance(result, dict):
                raise RuntimeError("Cycle refine did not return a JSON object.")
            runtime = dict(result.get("runtime") or {})
            runtime.update(runtime_meta)
            runtime["cycle_cfg_path"] = self.cycle_cfg_path
            runtime["image_path"] = self.image_path
            if self.target_claim_ids:
                runtime["targeted_reverify"] = True
                runtime["requested_target_claim_count"] = len(self.target_claim_ids)
            result["runtime"] = runtime
            self.done.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class TaskSettingsDialog(QDialog):
    """Settings dialog with common and current-task specific tabs."""

    def __init__(
        self,
        *,
        task_name: str,
        ontology_path: str,
        ontology_changed_cb,
        common_settings: Dict[str, object],
        default_common_settings: Dict[str, object],
        task_settings: Dict[str, object],
        default_task_settings: Dict[str, object],
        ontology_status_text: str,
        parent=None,
    ):
        super().__init__(parent)
        self._task_name = task_name
        self._common_settings = dict(common_settings)
        self._default_common_settings = dict(default_common_settings)
        self._task_settings = dict(task_settings)
        self._default_task_settings = dict(default_task_settings)
        self._api_key_input: Optional[QLineEdit] = None
        self._fps_mode_combo: Optional[QComboBox] = None
        self._fps_value_spin: Optional[QDoubleSpinBox] = None
        self._fps_min_spin: Optional[QDoubleSpinBox] = None
        self._fps_max_spin: Optional[QDoubleSpinBox] = None
        self._validator_id_input: Optional[QLineEdit] = None
        self._validation_round_spin: Optional[QSpinBox] = None
        self._max_items_spin: Optional[QSpinBox] = None
        self._sentence_refine_check: Optional[QComboBox] = None
        self._sg_backend_combo: Optional[QComboBox] = None
        self._sg_runner_profile_combo: Optional[QComboBox] = None
        self._sg_runtime_config_input: Optional[QLineEdit] = None
        self._sg_external_args_file_input: Optional[QLineEdit] = None
        self._sg_external_template_input: Optional[QPlainTextEdit] = None
        self._sg_backend_timeout_spin: Optional[QSpinBox] = None
        self._sg_disable_cache_check: Optional[QCheckBox] = None
        self._sg_tracking_combo: Optional[QComboBox] = None
        self._sg_tracking_search_radius_spin: Optional[QSpinBox] = None
        self._sg_tracking_min_response_spin: Optional[QDoubleSpinBox] = None
        self._sg_tracking_alpha_spin: Optional[QDoubleSpinBox] = None
        self._sg_tracking_max_lost_spin: Optional[QSpinBox] = None
        self._sg_video_generation_fps_spin: Optional[QDoubleSpinBox] = None
        self._sg_sampling_every_n_spin: Optional[QSpinBox] = None
        self._sg_cycle_provider_combo: Optional[QComboBox] = None
        self._caption_style_combo: Optional[QComboBox] = None

        self.setWindowTitle(f"Settings - {task_name}")
        self.resize(880, 640)

        root = QVBoxLayout(self)
        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        common_tab = QWidget()
        common_layout = QVBoxLayout(common_tab)
        common_splitter = QSplitter(Qt.Vertical)

        self.ontology_editor = OntologyEditor(ontology_path, parent=self)
        self.ontology_editor.ontology_changed.connect(ontology_changed_cb)
        common_splitter.addWidget(self.ontology_editor)

        common_info = QGroupBox("Common Settings Status")
        common_form = QFormLayout(common_info)
        self.ontology_status_label = QLabel(ontology_status_text)
        common_form.addRow("Ontology:", self.ontology_status_label)
        common_splitter.addWidget(common_info)

        common_splitter.setSizes([460, 110])
        common_splitter.setStretchFactor(0, 1)
        common_splitter.setStretchFactor(1, 0)
        common_layout.addWidget(common_splitter, 1)

        runtime_group = QGroupBox("Runtime Settings")
        runtime_form = QFormLayout(runtime_group)
        self._api_key_input = QLineEdit(str(self._common_settings.get("api_key", "")))
        self._api_key_input.setEchoMode(QLineEdit.Password)
        self._fps_mode_combo = QComboBox()
        self._fps_mode_combo.addItems(["Auto", "Custom"])
        self._fps_mode_combo.setCurrentIndex(1 if bool(self._common_settings.get("fps_override_enabled", False)) else 0)
        self._fps_value_spin = QDoubleSpinBox()
        self._fps_value_spin.setDecimals(2)
        self._fps_value_spin.setSingleStep(0.5)
        self._fps_value_spin.setValue(float(self._common_settings.get("fps_override", 30.0)))
        self._fps_min_spin = QDoubleSpinBox()
        self._fps_min_spin.setDecimals(2)
        self._fps_min_spin.setSingleStep(1.0)
        self._fps_min_spin.setRange(0.1, 1000.0)
        self._fps_min_spin.setValue(float(self._common_settings.get("fps_min", 1.0)))
        self._fps_max_spin = QDoubleSpinBox()
        self._fps_max_spin.setDecimals(2)
        self._fps_max_spin.setSingleStep(1.0)
        self._fps_max_spin.setRange(0.1, 1000.0)
        self._fps_max_spin.setValue(float(self._common_settings.get("fps_max", 120.0)))
        self._validator_id_input = QLineEdit(str(self._common_settings.get("validator_id", "")))
        self._validation_round_spin = QSpinBox()
        self._validation_round_spin.setRange(1, 20)
        self._validation_round_spin.setValue(int(self._common_settings.get("validation_round", 1)))

        runtime_form.addRow("API Key", self._api_key_input)
        runtime_form.addRow("FPS Mode", self._fps_mode_combo)
        runtime_form.addRow("FPS Value", self._fps_value_spin)
        runtime_form.addRow("FPS Min", self._fps_min_spin)
        runtime_form.addRow("FPS Max", self._fps_max_spin)
        runtime_form.addRow("Validator ID", self._validator_id_input)
        runtime_form.addRow("Validation Round", self._validation_round_spin)
        common_layout.addWidget(runtime_group)

        self.btn_reset_common_defaults = QPushButton("Restore Common Defaults")
        self.btn_reset_common_defaults.clicked.connect(self._reset_common_defaults)
        common_layout.addWidget(self.btn_reset_common_defaults)
        tabs.addTab(common_tab, "Common")

        task_tab = QWidget()
        task_layout = QVBoxLayout(task_tab)
        self._task_form = QFormLayout()
        task_layout.addLayout(self._task_form)
        self._build_task_specific_settings()
        self.btn_reset_task_defaults = QPushButton("Restore Task Defaults")
        self.btn_reset_task_defaults.clicked.connect(self._reset_task_defaults)
        task_layout.addWidget(self.btn_reset_task_defaults)
        task_layout.addStretch(1)
        tabs.addTab(task_tab, f"{task_name} Only")

        actions = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        actions.accepted.connect(self.accept)
        actions.rejected.connect(self.reject)
        root.addWidget(actions)

    @staticmethod
    def _sam_runtime_config_from_settings(settings: Dict[str, object]) -> str:
        return str(
            settings.get("sam_runtime_config", "")
            or ""
        ).strip()

    @staticmethod
    def _sam_runner_profile_from_settings(settings: Dict[str, object]) -> str:
        return str(
            settings.get("sam_runner_profile", "")
            or ""
        ).strip()

    def _build_task_specific_settings(self) -> None:
        form = self._task_form
        if self._task_name == "Video Scene Graph":
            self._sentence_refine_check = QComboBox()
            self._sentence_refine_check.addItems(["Disabled", "Enabled"])
            enabled = bool(self._task_settings.get("enable_sentence_refine", False))
            self._sentence_refine_check.setCurrentIndex(1 if enabled else 0)
            form.addRow("Sentence Refine", self._sentence_refine_check)
            self._sg_backend_combo = QComboBox()
            self._sg_backend_combo.addItems(["Use Pipeline Config", "Mock", "External Command"])
            backend_provider = str(self._task_settings.get("backend_provider", "") or "").strip().lower()
            backend_idx = {"": 0, "mock": 1, "external_command": 2}.get(backend_provider, 0)
            self._sg_backend_combo.setCurrentIndex(backend_idx)
            self._sg_runner_profile_combo = QComboBox()
            self._sg_runner_profile_combo.addItems(
                [
                    "Auto",
                    "SAM3 Linux (Native)",
                    "SAM3 Windows",
                    "WSL Ubuntu",
                ]
            )
            runner_profile = self._sam_runner_profile_from_settings(self._task_settings).lower()
            runner_idx = {
                "": 0,
                "sam3": 1,
                "sam3_windows": 2,
                "sam3_wsl": 3,
            }.get(runner_profile, 0)
            self._sg_runner_profile_combo.setCurrentIndex(runner_idx)
            self._sg_runtime_config_input = QLineEdit(self._sam_runtime_config_from_settings(self._task_settings))
            self._sg_runtime_config_input.setPlaceholderText("Built-in SAM3 runtime config JSON")
            self._sg_external_args_file_input = QLineEdit(str(self._task_settings.get("external_command_args_file", "") or ""))
            self._sg_external_args_file_input.setPlaceholderText("Optional JSON argv file for external_command")
            self._sg_external_template_input = QPlainTextEdit(str(self._task_settings.get("external_command_template", "") or ""))
            self._sg_external_template_input.setPlaceholderText("Optional shell template fallback for external_command")
            self._sg_external_template_input.setFixedHeight(90)
            self._sg_backend_timeout_spin = QSpinBox()
            self._sg_backend_timeout_spin.setRange(30, 7200)
            self._sg_backend_timeout_spin.setSingleStep(30)
            self._sg_backend_timeout_spin.setValue(int(self._task_settings.get("backend_timeout_sec", 1800)))
            self._sg_backend_timeout_spin.setSuffix(" s")
            self._sg_disable_cache_check = QCheckBox("Disable backend cache")
            self._sg_disable_cache_check.setChecked(bool(self._task_settings.get("disable_backend_cache", False)))
            self._sg_tracking_combo = QComboBox()
            self._sg_tracking_combo.addItems(["Disabled (Deprecated)"])
            self._sg_tracking_combo.setCurrentIndex(0)
            self._sg_tracking_search_radius_spin = QSpinBox()
            self._sg_tracking_search_radius_spin.setRange(8, 256)
            self._sg_tracking_search_radius_spin.setValue(int(self._task_settings.get("tracking_search_radius", 72)))
            self._sg_tracking_min_response_spin = QDoubleSpinBox()
            self._sg_tracking_min_response_spin.setDecimals(2)
            self._sg_tracking_min_response_spin.setSingleStep(0.05)
            self._sg_tracking_min_response_spin.setRange(0.05, 0.99)
            self._sg_tracking_min_response_spin.setValue(float(self._task_settings.get("tracking_min_response", 0.35)))
            self._sg_tracking_alpha_spin = QDoubleSpinBox()
            self._sg_tracking_alpha_spin.setDecimals(2)
            self._sg_tracking_alpha_spin.setSingleStep(0.05)
            self._sg_tracking_alpha_spin.setRange(0.0, 1.0)
            self._sg_tracking_alpha_spin.setValue(float(self._task_settings.get("tracking_template_update_alpha", 0.25)))
            self._sg_tracking_max_lost_spin = QSpinBox()
            self._sg_tracking_max_lost_spin.setRange(0, 240)
            self._sg_tracking_max_lost_spin.setValue(int(self._task_settings.get("tracking_max_lost_frames", 12)))
            self._sg_video_generation_fps_spin = QDoubleSpinBox()
            self._sg_video_generation_fps_spin.setDecimals(2)
            self._sg_video_generation_fps_spin.setSingleStep(0.25)
            self._sg_video_generation_fps_spin.setRange(0.1, 60.0)
            self._sg_video_generation_fps_spin.setValue(float(self._task_settings.get("video_generation_fps", 1.0)))
            self._sg_cycle_provider_combo = QComboBox()
            self._sg_cycle_provider_combo.addItem("Gemini API", "gemini_api")
            self._sg_cycle_provider_combo.addItem("ChatGPT API", "chatgpt_api")
            self._sg_cycle_provider_combo.addItem("Qwen", "qwen25_vl")
            self._sg_cycle_provider_combo.addItem("Manual", "manual")
            cycle_provider = normalize_cycle_provider(
                self._task_settings.get("cycle_verifier_provider", DEFAULT_CYCLE_PROVIDER),
                default=DEFAULT_CYCLE_PROVIDER,
            )
            cycle_idx = {
                "gemini_api": 0,
                "chatgpt_api": 1,
                "qwen25_vl": 2,
                "manual": 3,
            }.get(cycle_provider, 0)
            self._sg_cycle_provider_combo.setCurrentIndex(cycle_idx)
            form.addRow("Backend", self._sg_backend_combo)
            form.addRow("Runner Profile", self._sg_runner_profile_combo)
            form.addRow("SAM Runtime Config", self._sg_runtime_config_input)
            form.addRow("External Args File", self._sg_external_args_file_input)
            form.addRow("External Template", self._sg_external_template_input)
            form.addRow("Backend Timeout", self._sg_backend_timeout_spin)
            form.addRow("Cache", self._sg_disable_cache_check)
            form.addRow("Tracking", self._sg_tracking_combo)
            form.addRow("Track Search Radius", self._sg_tracking_search_radius_spin)
            form.addRow("Track Min Response", self._sg_tracking_min_response_spin)
            form.addRow("Track Template Alpha", self._sg_tracking_alpha_spin)
            form.addRow("Track Max Lost", self._sg_tracking_max_lost_spin)
            form.addRow("Batch Sampling FPS", self._sg_video_generation_fps_spin)
            form.addRow("Cycle Verifier", self._sg_cycle_provider_combo)
            return

        if self._task_name in ("Single-turn VQA", "Multi-turn VQA"):
            self._max_items_spin = QSpinBox()
            self._max_items_spin.setRange(0, 10000)
            self._max_items_spin.setValue(int(self._task_settings.get("max_items", 0)))
            self._max_items_spin.setToolTip("0 means no limit")
            form.addRow("Max Items", self._max_items_spin)
            return

        if self._task_name == "Video Captioning":
            self._caption_style_combo = QComboBox()
            self._caption_style_combo.addItems(["Concise", "Detailed", "Technical"])
            style = str(self._task_settings.get("default_style", "Concise"))
            idx = self._caption_style_combo.findText(style)
            self._caption_style_combo.setCurrentIndex(max(0, idx))
            form.addRow("Default Style", self._caption_style_combo)
            return

        placeholder = QLabel("No task-specific settings available.")
        form.addRow(placeholder)

    def _reset_task_defaults(self) -> None:
        if self._task_name == "Video Scene Graph":
            enabled = bool(self._default_task_settings.get("enable_sentence_refine", False))
            if self._sentence_refine_check is not None:
                self._sentence_refine_check.setCurrentIndex(1 if enabled else 0)
            backend_provider = str(self._default_task_settings.get("backend_provider", "") or "").strip().lower()
            if self._sg_backend_combo is not None:
                self._sg_backend_combo.setCurrentIndex({"": 0, "mock": 1, "external_command": 2}.get(backend_provider, 0))
            runner_profile = self._sam_runner_profile_from_settings(self._default_task_settings).lower()
            if self._sg_runner_profile_combo is not None:
                self._sg_runner_profile_combo.setCurrentIndex(
                    {
                        "": 0,
                        "sam3": 1,
                        "sam3_windows": 2,
                        "sam3_wsl": 3,
                    }.get(runner_profile, 0)
                )
            if self._sg_runtime_config_input is not None:
                self._sg_runtime_config_input.setText(self._sam_runtime_config_from_settings(self._default_task_settings))
            if self._sg_external_args_file_input is not None:
                self._sg_external_args_file_input.setText(str(self._default_task_settings.get("external_command_args_file", "") or ""))
            if self._sg_external_template_input is not None:
                self._sg_external_template_input.setPlainText(str(self._default_task_settings.get("external_command_template", "") or ""))
            if self._sg_backend_timeout_spin is not None:
                self._sg_backend_timeout_spin.setValue(int(self._default_task_settings.get("backend_timeout_sec", 1800)))
            if self._sg_disable_cache_check is not None:
                self._sg_disable_cache_check.setChecked(bool(self._default_task_settings.get("disable_backend_cache", False)))
            if self._sg_tracking_combo is not None:
                self._sg_tracking_combo.setCurrentIndex(0)
            if self._sg_tracking_search_radius_spin is not None:
                self._sg_tracking_search_radius_spin.setValue(int(self._default_task_settings.get("tracking_search_radius", 72)))
            if self._sg_tracking_min_response_spin is not None:
                self._sg_tracking_min_response_spin.setValue(float(self._default_task_settings.get("tracking_min_response", 0.35)))
            if self._sg_tracking_alpha_spin is not None:
                self._sg_tracking_alpha_spin.setValue(float(self._default_task_settings.get("tracking_template_update_alpha", 0.25)))
            if self._sg_tracking_max_lost_spin is not None:
                self._sg_tracking_max_lost_spin.setValue(int(self._default_task_settings.get("tracking_max_lost_frames", 12)))
            if self._sg_video_generation_fps_spin is not None:
                self._sg_video_generation_fps_spin.setValue(float(self._default_task_settings.get("video_generation_fps", 1.0)))
            if self._sg_cycle_provider_combo is not None:
                cycle_provider = normalize_cycle_provider(
                    self._default_task_settings.get("cycle_verifier_provider", DEFAULT_CYCLE_PROVIDER),
                    default=DEFAULT_CYCLE_PROVIDER,
                )
                self._sg_cycle_provider_combo.setCurrentIndex(
                    {
                        "gemini_api": 0,
                        "chatgpt_api": 1,
                        "qwen25_vl": 2,
                        "manual": 3,
                    }.get(cycle_provider, 0)
                )
            return

        if self._task_name in ("Single-turn VQA", "Multi-turn VQA"):
            max_items = int(self._default_task_settings.get("max_items", 0))
            if self._max_items_spin is not None:
                self._max_items_spin.setValue(max_items)
            return

        if self._task_name == "Video Captioning":
            style = str(self._default_task_settings.get("default_style", "Concise"))
            if self._caption_style_combo is not None:
                idx = self._caption_style_combo.findText(style)
                self._caption_style_combo.setCurrentIndex(max(0, idx))

    def _reset_common_defaults(self) -> None:
        if self._api_key_input is not None:
            self._api_key_input.setText(str(self._default_common_settings.get("api_key", "")))
        if self._fps_mode_combo is not None:
            enabled = bool(self._default_common_settings.get("fps_override_enabled", False))
            self._fps_mode_combo.setCurrentIndex(1 if enabled else 0)
        if self._fps_value_spin is not None:
            self._fps_value_spin.setValue(float(self._default_common_settings.get("fps_override", 30.0)))
        if self._fps_min_spin is not None:
            self._fps_min_spin.setValue(float(self._default_common_settings.get("fps_min", 1.0)))
        if self._fps_max_spin is not None:
            self._fps_max_spin.setValue(float(self._default_common_settings.get("fps_max", 120.0)))
        if self._validator_id_input is not None:
            self._validator_id_input.setText(str(self._default_common_settings.get("validator_id", "")))
        if self._validation_round_spin is not None:
            self._validation_round_spin.setValue(int(self._default_common_settings.get("validation_round", 1)))

    def get_common_settings(self) -> Dict[str, object]:
        fps_min = float(self._fps_min_spin.value()) if self._fps_min_spin is not None else 1.0
        fps_max = float(self._fps_max_spin.value()) if self._fps_max_spin is not None else 120.0
        if fps_max < fps_min:
            fps_min, fps_max = fps_max, fps_min
        fps_value = float(self._fps_value_spin.value()) if self._fps_value_spin is not None else 30.0
        fps_value = max(fps_min, min(fps_value, fps_max))
        enabled = bool(self._fps_mode_combo and self._fps_mode_combo.currentIndex() == 1)

        return {
            "api_key": str(self._api_key_input.text()).strip() if self._api_key_input is not None else "",
            "fps_override_enabled": enabled,
            "fps_override": fps_value,
            "fps_min": fps_min,
            "fps_max": fps_max,
            "validator_id": str(self._validator_id_input.text()).strip() if self._validator_id_input is not None else "",
            "validation_round": int(self._validation_round_spin.value()) if self._validation_round_spin is not None else 1,
        }

    def get_task_settings(self) -> Dict[str, object]:
        if self._task_name == "Video Scene Graph":
            out = dict(self._task_settings)
            enabled = bool(self._sentence_refine_check and self._sentence_refine_check.currentIndex() == 1)
            backend_map = {0: "", 1: "mock", 2: "external_command"}
            runner_map = {
                0: "",
                1: "sam3",
                2: "sam3_windows",
                3: "sam3_wsl",
            }
            out.update({
                "enable_sentence_refine": enabled,
                "backend_provider": backend_map.get(
                    int(self._sg_backend_combo.currentIndex()) if self._sg_backend_combo is not None else 0,
                    "",
                ),
                "sam_runner_profile": runner_map.get(
                    int(self._sg_runner_profile_combo.currentIndex()) if self._sg_runner_profile_combo is not None else 0,
                    "",
                ),
                "sam_runtime_config": (
                    str(self._sg_runtime_config_input.text()).strip() if self._sg_runtime_config_input is not None else ""
                ),
                "external_command_args_file": (
                    str(self._sg_external_args_file_input.text()).strip() if self._sg_external_args_file_input is not None else ""
                ),
                "external_command_template": (
                    str(self._sg_external_template_input.toPlainText()).strip() if self._sg_external_template_input is not None else ""
                ),
                "backend_timeout_sec": (
                    int(self._sg_backend_timeout_spin.value()) if self._sg_backend_timeout_spin is not None else 1800
                ),
                "disable_backend_cache": bool(
                    self._sg_disable_cache_check.isChecked() if self._sg_disable_cache_check is not None else False
                ),
                "tracking_mode": (
                    "disabled"
                ),
                "tracking_search_radius": (
                    int(self._sg_tracking_search_radius_spin.value()) if self._sg_tracking_search_radius_spin is not None else 72
                ),
                "tracking_min_response": (
                    float(self._sg_tracking_min_response_spin.value()) if self._sg_tracking_min_response_spin is not None else 0.35
                ),
                "tracking_template_update_alpha": (
                    float(self._sg_tracking_alpha_spin.value()) if self._sg_tracking_alpha_spin is not None else 0.25
                ),
                "tracking_max_lost_frames": (
                    int(self._sg_tracking_max_lost_spin.value()) if self._sg_tracking_max_lost_spin is not None else 12
                ),
                "video_generation_fps": (
                    float(self._sg_video_generation_fps_spin.value()) if self._sg_video_generation_fps_spin is not None else 1.0
                ),
                "cycle_verifier_provider": normalize_cycle_provider(
                    self._sg_cycle_provider_combo.currentData() if self._sg_cycle_provider_combo is not None else DEFAULT_CYCLE_PROVIDER,
                    default=DEFAULT_CYCLE_PROVIDER,
                ),
            })
            return out

        if self._task_name in ("Single-turn VQA", "Multi-turn VQA"):
            max_items = int(self._max_items_spin.value()) if self._max_items_spin else 0
            out = dict(self._task_settings)
            out["max_items"] = max_items
            return out

        if self._task_name == "Video Captioning":
            style = "Concise"
            if self._caption_style_combo is not None:
                style = str(self._caption_style_combo.currentText())
            out = dict(self._task_settings)
            out["default_style"] = style
            return out

        return dict(self._task_settings)


class ObjectProbeDrawer(QGroupBox):
    """Object-centric evidence drawer for single/multi probes and corrections."""

    def __init__(self, parent=None):
        super().__init__("Object Probe Drawer", parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.detail = QPlainTextEdit(self)
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText(
            "Select an object to inspect Basic Info, Votes, Single-turn, Multi-turn chains, and Corrections."
        )
        layout.addWidget(self.detail)

    def set_content(self, text: str) -> None:
        self.detail.setPlainText(str(text or ""))


class VideoTaskStudio(QWidget):
    """Video-centric UI with task-specific right-side workspaces."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self._ontology_path = os.path.join(self._repo_root, "configs", "impact_sg_ontology.json")
        self._pipeline_cfg = os.path.join(self._repo_root, "configs", "impact_sg_pipeline.json")
        self._cycle_cfg = os.path.join(self._repo_root, "configs", "impact_cycle.json")
        self._cycle_memory_path = os.path.join(self._repo_root, "configs", "impact_cycle_memory.json")
        self._frame_cache_dir = os.path.join(self._repo_root, ".cache", "impact_sg", "frames")
        self._ui_settings_path = os.path.join(self._repo_root, "configs", "video_task_studio_settings.json")

        self.video_path: str = ""
        self._block_seek_signals = False
        self._dragging_slider = False

        self.current_graph: Optional[Dict[str, object]] = None
        self.current_graph_bundle: Optional[Dict[str, object]] = None
        self.current_cycle_result: Optional[Dict[str, object]] = None
        self._cycle_result_frame_idx: int = -1
        self.single_turn_items: List[Dict[str, object]] = []
        self.multi_turn_items: List[Dict[str, object]] = []
        self._selected_probe_id: str = ""
        self._selected_node_ids: List[str] = []
        self._selected_claim_id: str = ""
        self._current_frame_id: int = -1
        self._single_claim_table_rows: List[Dict[str, object]] = []
        self._multi_claim_table_rows: List[Dict[str, object]] = []

        self.caption_batch: List[Dict[str, object]] = []
        self._cycle_stat_widgets: Dict[str, Dict[str, object]] = {}
        self._node_row_by_id: Dict[str, int] = {}
        self._node_display_by_id: Dict[str, str] = {}
        self._entity_id_by_display: Dict[str, str] = {}
        self._edge_rows: List[Dict[str, str]] = []
        self._sync_graph_selection = False
        self._sg_table_rendering = False
        self._sg_bbox_mode = "pixel"  # pixel | norm01 | percent100
        self._sg_bbox_ref_size: Tuple[float, float] = (1.0, 1.0)
        self._sg_ctrl_edge_pick_first: str = ""
        self._scene_graph_undo_stack: List[Dict[str, object]] = []
        self._scene_graph_undo_limit = 80
        self._sg_tracks: Dict[str, Dict[str, Any]] = {}
        self._sg_manual_keyframes: set[int] = set()
        self._sg_next_track_id = 1
        self._sg_last_tracking_frame = -1
        self._sg_last_detection_frame = -1
        self._graph_frame_manual = False
        self._block_graph_frame_sync = False
        self._sg_worker_thread: Optional[QThread] = None
        self._sg_worker: Optional[SceneGraphBuildWorker] = None
        self._llm_summary_thread: Optional[QThread] = None
        self._llm_summary_worker: Optional[LLMBatchSummaryWorker] = None
        self._cycle_worker_thread: Optional[QThread] = None
        self._cycle_worker: Optional[CycleRefineWorker] = None
        self._cycle_progress_dialog: Optional[QProgressDialog] = None
        self._cycle_progress_value: int = 0
        self._sg_progress_dialog: Optional[QProgressDialog] = None
        self._sg_progress_pin_timer = QTimer(self)
        self._sg_progress_pin_timer.setInterval(300)
        self._sg_progress_pin_timer.timeout.connect(self._pin_scene_graph_progress_dialog)
        self._sg_job_mode = ""
        self._sg_job_lightweight = False
        self._sg_job_show_error_dialog = True
        self._sg_job_output_path = ""
        self._sg_job_sampling_fps = 1.0
        self._sg_job_source_fps = 1.0
        self._sg_job_sampling_plan: Dict[str, object] = {}
        self._sg_job_frame_indices: List[int] = []
        self._sg_job_graphs: List[Dict[str, object]] = []
        self._sg_job_started_at = 0.0
        self._sg_timing_log_path = ""
        self._sg_run_dir = ""
        self._sg_runtime_log_path = ""
        self._sg_checkpoint_path = ""
        self._sg_checkpoint_state: Dict[str, object] = {}
        self._sg_run_status = ""
        self._sg_metadata_path = ""
        self._sg_summary_path = ""
        self._sg_stage_compact_path = ""
        self._sg_oplog_path = ""
        self._sg_progress_phase = "Idle"
        self._sg_progress_index = 0
        self._sg_progress_total = 0
        self._sg_progress_frame_idx = -1
        self._sg_llm_binary_progress = False
        self._sg_force_nonpersistent_backend = False
        self._sg_pending_run_mode = ""
        self._sg_pending_resume_dir = ""
        self._pending_llm_bundle_json = ""
        self._pending_llm_num_graphs = 0
        self._pending_llm_timing_log_path = ""
        self._correction_memory: Dict[str, object] = load_correction_memory(self._cycle_memory_path)
        self._project_oplog_path = ""
        self._mode_change_guard = False
        self._last_mode_index = 0
        self._pvsg_video_reference: Dict[str, object] = {}

        self._validation_changes: List[Dict[str, object]] = []
        self._ui_feature_service = UIFeatureService()
        self._stage_validator = StageValidator()
        self._current_stage_validation: Dict[str, object] = {}
        self._score_references: Dict[str, str] = {
            "Video Scene Graph": "",
            "Single-turn VQA": "",
            "Multi-turn VQA": "",
            "Video Captioning": "",
        }
        self._score_results: Dict[str, Dict[str, object]] = {}
        
        # Custom ontology support
        self._custom_ontology: Optional[Dict[str, object]] = None
        self._ontology_status_text = "Default ontology loaded"
        self._common_settings_defaults: Dict[str, object] = {
            "api_key": "",
            "participant_id": "",
            "fps_override_enabled": False,
            "fps_override": 30.0,
            "fps_min": 1.0,
            "fps_max": 120.0,
            "validator_id": "",
            "validation_round": 1,
        }
        self._common_settings: Dict[str, object] = dict(self._common_settings_defaults)
        self._task_settings_defaults: Dict[str, Dict[str, object]] = self._default_task_settings()
        self._task_settings: Dict[str, Dict[str, object]] = {
            k: dict(v) for k, v in self._task_settings_defaults.items()
        }

        self._build_ui()
        self._enable_text_selection_recursive(self)
        self._load_persisted_settings()
        self._apply_common_settings_runtime()
        self._apply_task_settings_to_widgets()
        self._on_mode_changed(self.mode_combo.currentIndex())
        self._last_mode_index = int(self.mode_combo.currentIndex())
        self._update_workspace_header()
        self._refresh_cycle_summary()
        self._refresh_all_score_panels()
        self._save_persisted_settings()

    def _apply_commercial_theme(self) -> None:
        self.setObjectName("studioRoot")
        self.setStyleSheet(
            """
            QWidget#studioRoot {
                background: #f3f6fb;
                color: #122033;
                font-size: 13px;
            }
            QFrame#heroPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f172a, stop:0.55 #16233a, stop:1 #1e3352);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 20px;
            }
            QFrame#topControlBar, QFrame#metricCard, QFrame#leftWorkspace, QFrame#rightWorkspace, QFrame#transportBar, QFrame#runtimeLogPanel {
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid #d9e2ef;
                border-radius: 18px;
            }
            QFrame#metricCard {
                min-height: 66px;
            }
            QFrame#runtimeLogPanel {
                background: #fbfdff;
            }
            QLabel#heroEyebrow {
                color: rgba(226, 232, 240, 0.88);
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.14em;
                text-transform: uppercase;
            }
            QLabel#heroTitle {
                color: #f8fafc;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#heroSubtitle {
                color: rgba(226, 232, 240, 0.82);
                font-size: 12px;
            }
            QLabel#heroPill {
                color: #dbeafe;
                background: rgba(59, 130, 246, 0.16);
                border: 1px solid rgba(147, 197, 253, 0.24);
                border-radius: 11px;
                padding: 4px 9px;
                font-weight: 600;
            }
            QLabel#metricTitle {
                color: #5b6b82;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }
            QLabel#metricValue {
                color: #0f172a;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#metricCaption {
                color: #64748b;
                font-size: 11px;
            }
            QLabel#runtimeLogTitle {
                color: #0f172a;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }
            QLabel#runtimeLogHint {
                color: #64748b;
                font-size: 11px;
            }
            QGroupBox {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid #d9e2ef;
                border-radius: 16px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: 700;
                color: #1e293b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
            }
            QPushButton {
                background: #0f5bd8;
                color: white;
                border: 0;
                border-radius: 12px;
                padding: 8px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #0b4fc3;
            }
            QPushButton:pressed {
                background: #0a43a5;
            }
            QToolButton {
                background: #ffffff;
                color: #163047;
                border: 1px solid #d6deea;
                border-radius: 12px;
                padding: 7px 10px;
                font-weight: 600;
            }
            QToolButton:hover {
                background: #eff5ff;
                border-color: #aab9ce;
            }
            QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit, QListWidget, QTableWidget {
                background: #ffffff;
                color: #122033;
                border: 1px solid #d6deea;
                border-radius: 12px;
                padding: 6px 8px;
                selection-background-color: #dbeafe;
                selection-color: #0f172a;
            }
            QCheckBox {
                color: #1f2a3d;
                font-weight: 600;
            }
            QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QPlainTextEdit:hover, QTextEdit:hover {
                border-color: #b8c5d8;
            }
            QHeaderView::section {
                background: #edf3fb;
                color: #304257;
                border: none;
                border-right: 1px solid #d9e2ef;
                border-bottom: 1px solid #d9e2ef;
                padding: 8px;
                font-weight: 700;
            }
            QListWidget, QTableWidget, QPlainTextEdit, QTextEdit {
                alternate-background-color: #f8fbff;
            }
            QLabel {
                color: #1f2a3d;
            }
            QMenu {
                background: #ffffff;
                border: 1px solid #d6deea;
                border-radius: 12px;
                padding: 8px;
            }
            QMenu::item {
                padding: 8px 14px;
                border-radius: 8px;
            }
            QMenu::item:selected {
                background: #e8f1ff;
            }
            QLabel#playerEmptyHint {
                color: #7c8ca3;
                font-size: 13px;
                font-weight: 600;
            }
            """
        )

    def _make_metric_card(self, title: str, value: str, caption: str) -> tuple[QFrame, QLabel]:
        card = QFrame()
        card.setObjectName("metricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        value_label.setWordWrap(True)
        caption_label = QLabel(caption)
        caption_label.setObjectName("metricCaption")
        caption_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addWidget(caption_label)
        return card, value_label

    def _append_runtime_log(self, text: str, *, level: str = "info") -> None:
        message = str(text or "").strip()
        if not message or not hasattr(self, "runtime_log"):
            return
        prefix_map = {
            "info": "[INFO]",
            "success": "[OK]",
            "warning": "[WARN]",
            "error": "[ERROR]",
        }
        prefix = prefix_map.get(str(level or "info").lower(), "[INFO]")
        self.runtime_log.appendPlainText(f"{prefix} {message}")
        scrollbar = self.runtime_log.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())
        runtime_log_path = str(getattr(self, "_sg_runtime_log_path", "") or "").strip()
        if runtime_log_path:
            try:
                folder = os.path.dirname(runtime_log_path)
                if folder:
                    os.makedirs(folder, exist_ok=True)
                with open(runtime_log_path, "a", encoding="utf-8") as f:
                    f.write(f"{_now_iso_utc()} {prefix} {message}\n")
            except Exception:
                pass

    def _append_oplog(self, event: str, **fields: object) -> None:
        payload: Dict[str, object] = {
            "ts": _now_iso_utc(),
            "event": str(event or "").strip(),
            "participant_id": str(self._common_settings.get("participant_id", "") or ""),
            "video_path": str(self.video_path or ""),
            "run_dir": str(self._sg_run_dir or ""),
        }
        for k, v in fields.items():
            payload[str(k)] = v
        targets: List[str] = []
        run_path = str(getattr(self, "_sg_oplog_path", "") or "").strip()
        if run_path:
            targets.append(run_path)
        seen: set[str] = set()
        for path in targets:
            ap = os.path.abspath(path)
            if ap.endswith(".jsonl") or ap.endswith(".txt"):
                ap = os.path.join(os.path.dirname(ap), "oplog")
            if ap in seen:
                continue
            seen.add(ap)
            try:
                os.makedirs(ap, exist_ok=True)
                ts_token = re.sub(r"[^0-9A-Za-z]+", "-", str(payload.get("ts", ""))).strip("-")
                event_token = re.sub(r"[^0-9A-Za-z_]+", "_", str(payload.get("event", "event") or "event")).strip("_")
                filename = f"{ts_token}_{event_token}_{uuid.uuid4().hex[:8]}.txt"
                with open(os.path.join(ap, filename), "w", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
            except Exception:
                pass

    def _persist_cycle_result_snapshot(self, payload: Dict[str, object]) -> str:
        run_dir = str(self._sg_run_dir or "").strip()
        if not run_dir:
            return ""
        try:
            folder = os.path.join(run_dir, "cycle_runs")
            os.makedirs(folder, exist_ok=True)
            ts_token = re.sub(r"[^0-9A-Za-z]+", "-", _now_iso_utc()).strip("-")
            filename = f"{ts_token}_cycle_result_{uuid.uuid4().hex[:8]}.json"
            out_path = os.path.join(folder, filename)
            probe_rows = [dict(x) for x in list((payload or {}).get("probe_results") or []) if isinstance(x, dict)]
            snapshot = {
                "saved_at": _now_iso_utc(),
                "run_dir": run_dir,
                "summary": dict((payload or {}).get("summary") or {}),
                "runtime": dict((payload or {}).get("runtime") or {}),
                "resolved_claims": [dict(x) for x in list((payload or {}).get("resolved_claims") or []) if isinstance(x, dict)],
                "suppressed_questions": [dict(x) for x in list((payload or {}).get("suppressed_questions") or []) if isinstance(x, dict)],
                "probe_questions": [
                    {
                        "probe_id": str(row.get("probe_id", "") or ""),
                        "view_type": str(row.get("view_type", "") or ""),
                        "claim_id": str(row.get("target_claim_id", row.get("claim_id", "")) or ""),
                        "question": str(row.get("question", "") or ""),
                        "score": (
                            dict(row.get("parsed_response") or {}).get("score")
                            if isinstance(row.get("parsed_response"), dict)
                            else row.get("score", None)
                        ),
                    }
                    for row in probe_rows
                ],
                "payload": dict(payload or {}),
            }
            _write_json(out_path, snapshot)
            return out_path
        except Exception:
            return ""

    def _write_run_metadata(self, *, status: str) -> None:
        # Unified run state: run_metadata.json is legacy; new runs write run_info.json only.
        self._sg_run_status = str(status or "").strip()
        self._write_run_info()

    def _write_run_checkpoint(self, *, stage: str, interrupted: bool = False) -> None:
        processed_indices: List[int] = []
        for g in list(self._sg_job_graphs or []):
            if not isinstance(g, dict):
                continue
            gi = int(self._extract_graph_frame_idx(g) or -1)
            if gi >= 0:
                processed_indices.append(gi)
        processed_indices = sorted(set(processed_indices))
        payload: Dict[str, object] = {
            "updated_at": _now_iso_utc(),
            "stage": str(stage or ""),
            "interrupted": bool(interrupted),
            "participant_id": str(self._common_settings.get("participant_id", "") or ""),
            "video_path": str(self.video_path or ""),
            "output_bundle_json": str(self._sg_job_output_path or ""),
            "timing_log_path": str(self._sg_timing_log_path or ""),
            "processed_graph_count": int(len(processed_indices)),
            "processed_frame_indices": processed_indices,
            "total_frame_indices": [int(x) for x in list(self._sg_job_frame_indices or [])],
            "last_completed_frame_idx": int(processed_indices[-1]) if processed_indices else -1,
            "llm_progress_phase": str(self._sg_progress_phase or ""),
            "llm_progress_index": int(self._sg_progress_index or 0),
            "llm_progress_total": int(self._sg_progress_total or 0),
            "llm_progress_frame_idx": int(self._sg_progress_frame_idx or -1),
        }
        self._sg_checkpoint_state = payload
        self._write_run_info()

    def _write_run_summary(self, *, status: str) -> None:
        # Summary data is consolidated into run_info.json.
        self._write_run_info()

    def _write_run_info(self) -> None:
        """Write consolidated run_info.json: validation scores, pvsg, timing summary, oplog events."""
        run_dir = str(self._sg_run_dir or "").strip()
        if not run_dir:
            return
        run_info_path = os.path.join(run_dir, "run_info.json")
        self._sg_stage_compact_path = run_info_path

        bundle_path = str(self._sg_job_output_path or "").strip() or os.path.join(run_dir, "scene_graph_bundle.json")

        existing_info = _read_json(run_info_path)
        bundle = _read_json(bundle_path)
        ckpt = dict(getattr(self, "_sg_checkpoint_state", {}) or {})
        if not ckpt:
            ckpt = dict(existing_info.get("checkpoint") or {})
        if not ckpt:
            checkpoint_path = str(self._sg_checkpoint_path or "").strip() or os.path.join(run_dir, "checkpoint.json")
            ckpt = _read_json(checkpoint_path)
        graphs = [g for g in list(bundle.get("graphs") or []) if isinstance(g, dict)]

        per_frame_scores: List[Dict[str, object]] = []
        for g in graphs:
            frame_idx = int(self._extract_graph_frame_idx(g) or -1)
            v = dict((g.get("validation") or {}).get("module_scores") or g.get("stage_scores") or {})
            if v:
                per_frame_scores.append({"frame_idx": int(frame_idx), "module_scores": dict(v)})
        per_frame_scores.sort(key=lambda x: int(x.get("frame_idx", -1) or -1))

        # Timing summary (aggregate from timing.jsonl)
        timing_summary: Dict[str, object] = {}
        timing_path = str(self._sg_timing_log_path or "").strip() or os.path.join(run_dir, "timing.jsonl")
        if os.path.isfile(timing_path):
            try:
                timing_events: List[Dict[str, object]] = []
                with open(timing_path, "r", encoding="utf-8") as _tf:
                    for _line in _tf:
                        _line = _line.strip()
                        if _line:
                            timing_events.append(json.loads(_line))
                total_sec = sum(float(e.get("elapsed_sec", 0) or 0) for e in timing_events)
                timing_summary = {
                    "total_events": len(timing_events),
                    "total_elapsed_sec": round(total_sec, 2),
                }
            except Exception:
                pass

        # Oplog event previews. Full event payloads live as separate files under oplog/.
        oplog_events: List[Dict[str, object]] = []
        oplog_path = str(self._sg_oplog_path or "").strip()
        oplog_dir = ""
        if oplog_path:
            oplog_dir = os.path.join(os.path.dirname(oplog_path), "oplog") if oplog_path.endswith(".jsonl") else oplog_path
        if oplog_dir and os.path.isdir(oplog_dir):
            try:
                for _name in sorted(os.listdir(oplog_dir))[-50:]:
                    if not _name.endswith(".txt"):
                        continue
                    with open(os.path.join(oplog_dir, _name), "r", encoding="utf-8") as _of:
                        oplog_events.append(json.load(_of))
            except Exception:
                pass
        elif oplog_path and os.path.isfile(oplog_path):
            try:
                with open(oplog_path, "r", encoding="utf-8") as _of:
                    for _line in _of:
                        _line = _line.strip()
                        if _line:
                            oplog_events.append(json.loads(_line))
            except Exception:
                pass

        elapsed_sec = max(0.0, time.monotonic() - float(self._sg_job_started_at or time.monotonic()))
        run_info: Dict[str, object] = {
            "schema_version": 1,
            "updated_at": _now_iso_utc(),
            "participant_id": str(self._common_settings.get("participant_id", "") or ""),
            "session_name": os.path.basename(str(self._sg_run_dir or "").rstrip(os.sep)),
            "run_dir": run_dir,
            "video_path": str(self.video_path or ""),
            "status": str(self._sg_run_status or ckpt.get("stage", "") or existing_info.get("status", "") or ""),
            "elapsed_sec": round(elapsed_sec, 2),
            "sampling_fps": float(self._sg_job_sampling_fps or 1.0),
            "source_fps": float(self._sg_job_source_fps or 1.0),
            "sampled_frame_indices": [int(x) for x in list(self._sg_job_frame_indices or [])],
            "sampling_plan": dict(self._sg_job_sampling_plan or {}),
            "processed_graphs": int(len(graphs)),
            "stage_validation": {
                "summary": dict(bundle.get("validation") or {}),
                "per_frame_module_scores": per_frame_scores,
            },
            "pvsg_reference": dict(self._pvsg_video_reference or {}),
            "timing": timing_summary,
            "events": oplog_events,
            "checkpoint": ckpt,
        }
        try:
            _write_json(run_info_path, run_info)
        except Exception:
            pass

    def _enable_text_selection_recursive(self, root: QWidget) -> None:
        if root is None:
            return
        flags = Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
        for label in root.findChildren(QLabel):
            try:
                label.setTextInteractionFlags(flags)
            except Exception:
                continue

    @staticmethod
    def _sam_runtime_config_from_settings(settings: Dict[str, object]) -> str:
        return str(
            settings.get("sam_runtime_config", "")
            or ""
        ).strip()

    @staticmethod
    def _sam_runner_profile_from_settings(settings: Dict[str, object]) -> str:
        return str(
            settings.get("sam_runner_profile", "")
            or ""
        ).strip()

    def _remap_repo_local_path(self, path_value: object) -> str:
        raw = str(path_value or "").strip()
        if not raw:
            return ""
        normalized = raw.replace("\\", "/").strip()
        for marker in ("/configs/", "/tools/", "/core/", "/ui/"):
            idx = normalized.lower().find(marker)
            if idx >= 0:
                relative = normalized[idx + 1 :]
                candidate = os.path.join(self._repo_root, *relative.split("/"))
                return os.path.abspath(candidate)
        if os.path.isabs(raw):
            return os.path.abspath(os.path.expanduser(raw))
        return os.path.abspath(os.path.join(self._repo_root, raw))

    def _normalize_scene_graph_settings(self, settings: Dict[str, object]) -> Dict[str, object]:
        normalized = dict(settings or {})
        runtime_keys = ("sam_runtime_config",)
        default_runtime = self._preferred_sam_runtime_config_path()
        runtime_path = ""
        for key in runtime_keys:
            remapped = self._remap_repo_local_path(normalized.get(key, ""))
            if remapped and os.path.isfile(remapped):
                normalized[key] = remapped
                if not runtime_path:
                    runtime_path = remapped
        if not runtime_path:
            runtime_path = default_runtime
        for key in runtime_keys:
            normalized[key] = runtime_path

        args_file = self._remap_repo_local_path(normalized.get("external_command_args_file", ""))
        normalized["external_command_args_file"] = args_file if args_file and os.path.isfile(args_file) else ""

        runner_profile = str(
            normalized.get("sam_runner_profile", "")
            or ""
        ).strip()
        if not runner_profile:
            runner_profile = self._preferred_sam_runner_profile(runtime_path)
        normalized["sam_runner_profile"] = runner_profile
        if int(normalized.get("video_sampling_every_n_frames", 0) or 0) <= 0:
            fps_legacy = float(normalized.get("video_generation_fps", 1.0) or 1.0)
            approx_n = int(round(30.0 / max(0.1, fps_legacy)))
            normalized["video_sampling_every_n_frames"] = max(1, approx_n)
        normalized["video_generation_batch_size"] = max(1, int(normalized.get("video_generation_batch_size", 8) or 8))
        cuda_hint = str(normalized.get("llm_cuda_device", "cuda:0") or "cuda:0").strip().lower()
        if not cuda_hint:
            cuda_hint = "cuda:0"
        if cuda_hint.startswith("cuda:"):
            try:
                idx = int(cuda_hint.split(":", 1)[1].strip())
            except Exception:
                idx = 0
            cuda_hint = f"cuda:{max(0, idx)}"
        elif cuda_hint.isdigit():
            cuda_hint = f"cuda:{max(0, int(cuda_hint))}"
        normalized["llm_cuda_device"] = cuda_hint
        return normalized

    def _detected_gpu_count(self) -> int:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                stderr=subprocess.STDOUT,
                text=True,
            )
            rows = [line.strip() for line in str(out or "").splitlines() if line.strip()]
            return max(0, int(len(rows)))
        except Exception:
            return -1

    def _resolve_llm_cuda_device(self, requested: str) -> str:
        hint = str(requested or "").strip().lower()
        if not hint:
            return "cuda:0"
        if hint.isdigit():
            hint = f"cuda:{hint}"
        if not hint.startswith("cuda:"):
            return hint
        try:
            idx = int(hint.split(":", 1)[1].strip())
        except Exception:
            idx = 0
        idx = max(0, idx)
        gpu_count = self._detected_gpu_count()
        if gpu_count >= 0 and idx >= gpu_count:
            fallback = "cuda:0"
            self._append_runtime_log(
                f"Requested LLM device {hint} is out of range for detected GPUs={gpu_count}; fallback to {fallback}.",
                level="warning",
            )
            return fallback
        return f"cuda:{idx}"

    def _pvsg_settings(self) -> Dict[str, object]:
        sg = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        return {
            "enabled": bool(sg.get("enable_pvsg_gt_reference", True)),
            "pvsg_json_path": str(sg.get("pvsg_json_path", "/cvhci/temp/wkong/sample_videos/pvsg.json") or "").strip(),
            "pvsg_masks_root": str(sg.get("pvsg_masks_root", "/cvhci/temp/wkong/sample_videos/VidOR/masks") or "").strip(),
        }

    def _summarize_pvsg_reference(self, payload: Dict[str, object]) -> str:
        if not isinstance(payload, dict) or not bool(payload.get("reference_available", False)):
            reason = str(payload.get("reason", "not_available") if isinstance(payload, dict) else "not_available")
            return f"PVSG GT unavailable ({reason})."
        video_id = str(payload.get("video_id", "") or "")
        objects_total = int(payload.get("objects_total", 0) or 0)
        relations_total = int(payload.get("relations_total", 0) or 0)
        ranges = [dict(x) for x in list(payload.get("annotated_frame_ranges") or []) if isinstance(x, dict)]
        if ranges:
            preview = ", ".join([f"{int(r.get('start', 0))}-{int(r.get('end', 0))}" for r in ranges[:3]])
            if len(ranges) > 3:
                preview += ", ..."
        else:
            preview = "none"
        return (
            f"PVSG GT loaded: video_id={video_id} objects={objects_total} relations={relations_total} "
            f"annotated_ranges={preview}"
        )

    def _refresh_pvsg_reference_for_video(
        self,
        *,
        frame_indices: Optional[List[int]] = None,
        save_path: str = "",
    ) -> Dict[str, object]:
        cfg = self._pvsg_settings()
        if not bool(cfg.get("enabled", True)):
            self._pvsg_video_reference = {
                "reference_available": False,
                "video_path": str(self.video_path or ""),
                "reason": "disabled_by_setting",
            }
            return dict(self._pvsg_video_reference)
        if not str(self.video_path or "").strip():
            self._pvsg_video_reference = {
                "reference_available": False,
                "video_path": "",
                "reason": "video_not_loaded",
            }
            return dict(self._pvsg_video_reference)

        payload = load_pvsg_video_reference(
            video_path=str(self.video_path or ""),
            frame_indices=[int(x) for x in list(frame_indices or [])] if frame_indices else None,
            pvsg_json_path=str(cfg.get("pvsg_json_path", "") or ""),
            masks_root=str(cfg.get("pvsg_masks_root", "") or ""),
        )
        self._pvsg_video_reference = dict(payload or {})
        # pvsg_reference data is consolidated into run_info.json; no separate file.
        return dict(self._pvsg_video_reference)

    def _pvsg_gt_for_frame(self, frame_idx: int) -> Dict[str, object]:
        payload = dict(self._pvsg_video_reference or {})
        per_frame = dict(payload.get("per_frame") or {})
        row = dict(per_frame.get(str(int(frame_idx)), {}) or {})
        if row:
            return row
        if str(self.video_path or "").strip():
            # Lazy fill for ad-hoc frame queries.
            cfg = self._pvsg_settings()
            fetched = load_pvsg_video_reference(
                video_path=str(self.video_path or ""),
                frame_indices=[int(frame_idx)],
                pvsg_json_path=str(cfg.get("pvsg_json_path", "") or ""),
                masks_root=str(cfg.get("pvsg_masks_root", "") or ""),
            )
            if isinstance(fetched, dict):
                merged = dict(self._pvsg_video_reference or {})
                merged_per = dict(merged.get("per_frame") or {})
                new_per = dict((fetched.get("per_frame") or {}))
                merged_per.update(new_per)
                merged["per_frame"] = merged_per
                if not merged.get("video_id"):
                    merged["video_id"] = str(fetched.get("video_id", "") or "")
                if not merged.get("reference_available"):
                    merged["reference_available"] = bool(fetched.get("reference_available", False))
                self._pvsg_video_reference = merged
                return dict(merged_per.get(str(int(frame_idx)), {}) or {})
        return {}

    def _on_cycle_provider_changed(self) -> None:
        data = ""
        if hasattr(self, "cycle_provider_combo") and self.cycle_provider_combo is not None:
            data = str(self.cycle_provider_combo.currentData() or "").strip().lower()
        if not data:
            data = DEFAULT_CYCLE_PROVIDER
        data = normalize_cycle_provider(data)
        sg = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        sg["cycle_verifier_provider"] = data
        # Paid-mode defaults for Gemini API: disable low-quota constraints.
        if data == DEFAULT_CYCLE_PROVIDER:
            sg["cycle_low_quota_mode"] = False
            if hasattr(self, "cycle_low_quota_check") and self.cycle_low_quota_check is not None:
                self.cycle_low_quota_check.blockSignals(True)
                self.cycle_low_quota_check.setChecked(False)
                self.cycle_low_quota_check.blockSignals(False)
        self._task_settings["Video Scene Graph"] = sg
        self._save_persisted_settings()

    def _on_cycle_model_path_changed(self) -> None:
        path = ""
        if hasattr(self, "cycle_model_path_input") and self.cycle_model_path_input is not None:
            path = str(self.cycle_model_path_input.text() or "").strip()
        if not path:
            path = QWEN_DEFAULT_MODEL_PATH
        sg = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        sg["cycle_local_model_path"] = path
        self._task_settings["Video Scene Graph"] = sg
        self._save_persisted_settings()

    def _on_cycle_max_rounds_changed(self, value: int) -> None:
        sg = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        sg["cycle_max_revision_rounds"] = max(1, min(10, int(value or 1)))
        self._task_settings["Video Scene Graph"] = sg
        self._save_persisted_settings()

    def _on_cycle_low_quota_changed(self, state: int) -> None:
        sg = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        sg["cycle_low_quota_mode"] = bool(int(state) != 0)
        self._task_settings["Video Scene Graph"] = sg
        self._save_persisted_settings()

    def _on_cycle_enable_single_changed(self, state: int) -> None:
        sg = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        sg["cycle_enable_single_turn_probes"] = bool(int(state) != 0)
        self._task_settings["Video Scene Graph"] = sg
        self._save_persisted_settings()

    def _on_cycle_enable_multi_changed(self, state: int) -> None:
        sg = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        sg["cycle_enable_multi_turn_probes"] = bool(int(state) != 0)
        self._task_settings["Video Scene Graph"] = sg
        self._save_persisted_settings()

    def _on_cycle_enable_caption_changed(self, state: int) -> None:
        sg = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        sg["cycle_enable_caption_probe"] = bool(int(state) != 0)
        self._task_settings["Video Scene Graph"] = sg
        self._save_persisted_settings()

    def _on_cycle_debug_mode_changed(self, state: int) -> None:
        sg = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        sg["cycle_debug_mode"] = bool(int(state) != 0)
        self._task_settings["Video Scene Graph"] = sg
        self._save_persisted_settings()
        self._render_single_detail(self.single_list.currentRow())
        self._render_multi_detail(self.multi_list.currentRow())
        self._render_cycle_caption_feedback()

    def _pick_cycle_model_path(self) -> None:
        start = QWEN_DEFAULT_MODEL_PATH
        if hasattr(self, "cycle_model_path_input") and self.cycle_model_path_input is not None:
            start = str(self.cycle_model_path_input.text() or "").strip() or start
        picked = QFileDialog.getExistingDirectory(self, "Select Cycle Local Model Folder", start)
        if not picked:
            return
        if hasattr(self, "cycle_model_path_input") and self.cycle_model_path_input is not None:
            self.cycle_model_path_input.setText(str(picked))
        self._on_cycle_model_path_changed()

    def _default_cycle_cfg_payload(self, model_path: str) -> Dict[str, object]:
        return {
            "local_verifier": {
                "provider": "qwen25_vl",
                "model_id": str(model_path or QWEN_DEFAULT_MODEL_PATH),
                "device": "cuda",
                "max_new_tokens": 192,
                "use_flash_attention": False,
            },
            "api_verifier": {
                "enabled": True,
                "provider": "gemini",
                "model": DEFAULT_GEMINI_MODEL,
                "base_url": "",
                "answer_url": "",
                "caption_url": "",
                "timeout_sec": DEFAULT_API_TIMEOUT_SEC,
                "api_key_env": DEFAULT_GEMINI_API_KEY_ENV,
                "api_key_header": "Authorization",
                "api_key_prefix": "Bearer ",
                "include_image_base64": True,
                "max_output_tokens": DEFAULT_API_MAX_OUTPUT_TOKENS,
                "extra_headers": {},
                "max_calls_per_frame": 4,
            },
            "cycle": {
                "enable_single_turn_probes": True,
                "enable_multi_turn_probes": True,
                "enable_temporal_multi_turn": True,
                "multi_turn_max_chains": 12,
                "max_single_turn_probes": 0,
                "verify_all_claims": True,
                "max_multi_turn_probes": 0,
                "enable_caption_probe": True,
                "enable_geometry_review": True,
                "enable_person_focus": False,
                "focus_subject_label": "person",
                "focus_max_hops": 1,
                "focus_direct_relations_only": False,
                "focus_max_subjects": 0,
                "auto_accept_threshold": 0.85,
                "auto_reject_threshold": 0.80,
                "auto_drop_existence_threshold": 0.92,
                "auto_drop_support_ceiling": 0.20,
                "human_escalation_threshold": 0.45,
                "geometry_conflict_threshold": 0.60,
                "max_geometry_queries_per_frame": 2,
                "max_human_queries_per_frame": 3,
                "max_revision_rounds": 2,
            },
            "memory": {
                "enable_label_confusion_memory": True,
                "enable_relation_confusion_memory": True,
                "enable_prompt_alias_memory": True,
                "finalized_weight_boost": 1.25,
            },
            "caption": {
                "style": "technical",
                "require_relation_mentions": True,
                "max_sentences": 4,
                "structured_feedback": True,
                "emit_conflict_votes": True,
            },
            "role_policy": {
                "caption_label_enabled": False,
            },
            "runtime": {
                "allow_mock_fallback": False,
                "preferred_provider": DEFAULT_CYCLE_PROVIDER,
                "experimental_providers": [],
            },
        }

    def _ensure_cycle_cfg_file(self) -> str:
        path = os.path.abspath(os.path.expanduser(str(self._cycle_cfg or "").strip()))
        if not path:
            path = os.path.join(self._repo_root, "configs", "impact_cycle.json")
        if os.path.isfile(path):
            return path
        sg_settings = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        model_path = str(sg_settings.get("cycle_local_model_path", QWEN_DEFAULT_MODEL_PATH) or QWEN_DEFAULT_MODEL_PATH)
        payload = self._default_cycle_cfg_payload(model_path=model_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        return path

    def _scene_graph_cycle_cfg_override(self) -> Dict[str, object]:
        sg_settings = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        provider = normalize_cycle_provider(
            sg_settings.get("cycle_verifier_provider", DEFAULT_CYCLE_PROVIDER),
            default=DEFAULT_CYCLE_PROVIDER,
        )
        local_model_path = str(sg_settings.get("cycle_local_model_path", QWEN_DEFAULT_MODEL_PATH) or QWEN_DEFAULT_MODEL_PATH).strip()
        gemini_model = str(sg_settings.get("cycle_gemini_model", DEFAULT_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL).strip()
        gemini_key_env = str(sg_settings.get("cycle_gemini_api_key_env", DEFAULT_GEMINI_API_KEY_ENV) or DEFAULT_GEMINI_API_KEY_ENV).strip()
        chatgpt_model = str(sg_settings.get("cycle_chatgpt_model", DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL).strip()
        chatgpt_key_env = str(sg_settings.get("cycle_chatgpt_api_key_env", DEFAULT_OPENAI_API_KEY_ENV) or DEFAULT_OPENAI_API_KEY_ENV).strip()
        chatgpt_base_url = str(sg_settings.get("cycle_chatgpt_base_url", DEFAULT_OPENAI_BASE_URL) or DEFAULT_OPENAI_BASE_URL).strip()
        allow_mock = bool(sg_settings.get("cycle_allow_mock_fallback", False))
        low_quota = bool(sg_settings.get("cycle_low_quota_mode", False))
        use_gemini = provider == "gemini_api"
        use_chatgpt = provider == "chatgpt_api"
        use_qwen = provider == "qwen25_vl"
        use_manual = provider == "manual"
        api_enabled = use_gemini or use_chatgpt
        api_provider = "gemini" if use_gemini else ("openai" if use_chatgpt else "generic_api")
        local_provider = "qwen25_vl" if use_qwen else "mock"
        cycle_payload: Dict[str, object] = {
            "max_revision_rounds": max(1, min(10, int(sg_settings.get("cycle_max_revision_rounds", 2) or 2))),
            "verify_all_claims": bool(sg_settings.get("cycle_verify_all_claims", True)),
            "max_single_turn_probes": int(sg_settings.get("cycle_max_single_turn_probes", 0) or 0),
            "enable_single_turn_probes": bool(sg_settings.get("cycle_enable_single_turn_probes", True)),
            "enable_multi_turn_probes": bool(sg_settings.get("cycle_enable_multi_turn_probes", True)),
            "enable_temporal_multi_turn": bool(sg_settings.get("cycle_enable_temporal_multi_turn", True)),
            "max_multi_turn_probes": int(sg_settings.get("cycle_max_multi_turn_probes", 0) or 0),
            "enable_caption_probe": bool(sg_settings.get("cycle_enable_caption_probe", True)),
            "enable_geometry_review": bool(sg_settings.get("cycle_enable_geometry_review", True)),
            "enable_person_focus": bool(sg_settings.get("cycle_enable_person_focus", False)),
            "focus_subject_label": str(sg_settings.get("cycle_focus_subject_label", "person") or "person"),
            "focus_max_hops": int(sg_settings.get("cycle_focus_max_hops", 1) or 1),
            "focus_direct_relations_only": bool(sg_settings.get("cycle_focus_direct_relations_only", False)),
            "focus_max_subjects": int(sg_settings.get("cycle_focus_max_subjects", 0) or 0),
            "multi_turn_max_chains": int(sg_settings.get("cycle_multi_turn_max_chains", 12) or 12),
            "max_human_queries_per_frame": int(sg_settings.get("cycle_max_human_queries_per_frame", 3) or 3),
            "auto_accept_threshold": float(sg_settings.get("cycle_auto_accept_threshold", 0.85) or 0.85),
            "auto_reject_threshold": float(sg_settings.get("cycle_auto_reject_threshold", 0.80) or 0.80),
            "auto_drop_existence_threshold": float(sg_settings.get("cycle_auto_drop_existence_threshold", 0.92) or 0.92),
            "auto_drop_support_ceiling": float(sg_settings.get("cycle_auto_drop_support_ceiling", 0.20) or 0.20),
            "human_escalation_threshold": float(sg_settings.get("cycle_human_escalation_threshold", 0.45) or 0.45),
            "geometry_conflict_threshold": float(sg_settings.get("cycle_geometry_conflict_threshold", 0.60) or 0.60),
            "max_geometry_queries_per_frame": int(sg_settings.get("cycle_max_geometry_queries_per_frame", 2) or 2),
        }
        if use_manual:
            cycle_payload.update(
                {
                    "enable_single_turn_probes": False,
                    "enable_multi_turn_probes": False,
                    "enable_temporal_multi_turn": False,
                    "enable_caption_probe": False,
                    "enable_geometry_review": False,
                    "max_single_turn_probes": 0,
                    "max_multi_turn_probes": 0,
                    "max_human_queries_per_frame": 0,
                }
            )
        if low_quota and not use_manual:
            cycle_payload.update(
                {
                    "enable_single_turn_probes": True,
                    "enable_multi_turn_probes": False,
                    "enable_temporal_multi_turn": False,
                    "multi_turn_max_chains": 0,
                    "max_single_turn_probes": 3,
                    "max_multi_turn_probes": 0,
                    "enable_caption_probe": False,
                    "max_human_queries_per_frame": 2,
                    "max_geometry_queries_per_frame": 1,
                }
            )
        effective_allow_mock = bool(allow_mock)
        return {
            "local_verifier": {
                "provider": local_provider,
                "model_id": local_model_path,
            },
            "api_verifier": {
                "enabled": bool(api_enabled),
                "provider": api_provider,
                "model": gemini_model if use_gemini else (chatgpt_model if use_chatgpt else ""),
                "api_key_env": gemini_key_env if use_gemini else (chatgpt_key_env if use_chatgpt else "IMPACT_API_KEY"),
                "base_url": "" if use_gemini else (chatgpt_base_url if use_chatgpt else ""),
                "timeout_sec": DEFAULT_API_TIMEOUT_SEC,
                "max_output_tokens": LOW_QUOTA_API_MAX_OUTPUT_TOKENS if low_quota else DEFAULT_API_MAX_OUTPUT_TOKENS,
            },
            "web_verifier": {
                "provider": "",
                "model": gemini_model,
                "timeout_sec": int(
                    sg_settings.get("cycle_gemini_online_timeout_sec", DEFAULT_GEMINI_ONLINE_TIMEOUT_SEC)
                    or DEFAULT_GEMINI_ONLINE_TIMEOUT_SEC
                ),
                "headless": bool(sg_settings.get("cycle_gemini_online_headless", False)),
                "user_data_dir": str(sg_settings.get("cycle_gemini_online_user_data_dir", "") or "").strip(),
                "profile_directory": str(sg_settings.get("cycle_gemini_online_profile_directory", "") or "").strip(),
                "chrome_binary": str(sg_settings.get("cycle_gemini_online_chrome_binary", "") or "").strip(),
                "gemini_url": "https://gemini.google.com/app",
            },
            "runtime": {
                "allow_mock_fallback": bool(effective_allow_mock or use_manual),
                "preferred_provider": (
                    "mock"
                    if use_manual
                    else ("qwen25_vl" if use_qwen else ("chatgpt_api" if use_chatgpt else "gemini_api"))
                ),
                "experimental_providers": [],
            },
            "cycle": cycle_payload,
            "role_policy": {
                "caption_label_enabled": bool(sg_settings.get("cycle_caption_enable_label_vote", False)),
            },
        }

    def _resolve_cycle_image_path(self) -> str:
        graph = self.current_graph if isinstance(self.current_graph, dict) else {}
        meta = dict(graph.get("metadata") or {})
        image_path = str(meta.get("image_path", "") or graph.get("image_path", "") or "").strip()
        if image_path and os.path.isfile(image_path):
            return image_path
        frame_idx = self._extract_graph_frame_idx(graph)
        if frame_idx is None:
            return ""
        try:
            out_path, _w, _h, _frame = _extract_video_frame_to_cache(
                self.video_path,
                int(frame_idx),
                self._frame_cache_dir,
                cap=getattr(self.player, "cap", None),
            )
            return str(out_path or "")
        except Exception:
            return ""

    def _bootstrap_cycle_verifier_auth(self) -> Tuple[bool, str]:
        """Best-effort credential bootstrap before Cycle Verify starts."""
        sg_settings = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        provider = normalize_cycle_provider(
            sg_settings.get("cycle_verifier_provider", DEFAULT_CYCLE_PROVIDER),
            default=DEFAULT_CYCLE_PROVIDER,
        )
        if provider not in {"gemini_api", "chatgpt_api"}:
            return True, ""

        key_env = (
            str(sg_settings.get("cycle_gemini_api_key_env", DEFAULT_GEMINI_API_KEY_ENV) or DEFAULT_GEMINI_API_KEY_ENV).strip()
            if provider == "gemini_api"
            else str(sg_settings.get("cycle_chatgpt_api_key_env", DEFAULT_OPENAI_API_KEY_ENV) or DEFAULT_OPENAI_API_KEY_ENV).strip()
        )
        if not key_env:
            key_env = DEFAULT_GEMINI_API_KEY_ENV if provider == "gemini_api" else DEFAULT_OPENAI_API_KEY_ENV

        # If env already exists, keep it.
        existing = str(os.environ.get(key_env, "") or "").strip()
        if existing:
            return True, f"[CYCLE-AUTH] Using existing {key_env} from environment."

        # Fallback to saved UI key (Runtime API Key field).
        saved_common_key = str((self._common_settings or {}).get("api_key", "") or "").strip()
        if saved_common_key:
            os.environ[key_env] = saved_common_key
            # Keep compatibility env for other integrations.
            os.environ.setdefault("IMPACT_API_KEY", saved_common_key)
            if provider == "gemini_api":
                os.environ.setdefault(DEFAULT_GEMINI_API_KEY_ENV, saved_common_key)
            return True, f"[CYCLE-AUTH] Injected {key_env} from saved Runtime API Key."

        # Last fallback: generic compatibility env.
        fallback_env = str(os.environ.get("IMPACT_API_KEY", "") or "").strip()
        if fallback_env:
            os.environ[key_env] = fallback_env
            return True, f"[CYCLE-AUTH] Reused IMPACT_API_KEY as {key_env}."

        provider_name = cycle_provider_display_name(provider)
        return (
            False,
            f"{provider_name} key missing. Set env `{key_env}` or fill Runtime API Key, then retry Cycle Verify.",
        )

    @staticmethod
    def _graph_with_cycle_temporal_context(graph: Dict[str, object]) -> Dict[str, object]:
        return dict(graph or {})

    @staticmethod
    def _json_safe_clone(value, _stack=None, _depth: int = 0):
        """Clone JSON-like data without recursion using json round-trip."""
        try:
            return json.loads(json.dumps(value, ensure_ascii=True, default=str))
        except Exception:
            # Fallback: stringify the whole object
            try:
                return json.loads(json.dumps(value, ensure_ascii=True, default=lambda x: str(x)))
            except Exception:
                return {}

    def _replace_current_graph_in_bundle(self, graph: Dict[str, object]) -> None:
        bundle = self.current_graph_bundle if isinstance(self.current_graph_bundle, dict) else {}
        graph_clean = self._json_safe_clone(graph) if isinstance(graph, dict) else {}
        graphs = [dict(g) for g in list(bundle.get("graphs") or []) if isinstance(g, dict)]
        if not graphs:
            return
        target_frame = self._extract_graph_frame_idx(graph_clean)
        replaced = False
        for i, row in enumerate(graphs):
            if self._extract_graph_frame_idx(row) == target_frame:
                graphs[i] = dict(graph_clean)
                replaced = True
                break
        if not replaced:
            graphs.append(dict(graph_clean))
        bundle["graphs"] = graphs
        self.current_graph_bundle = bundle

    def _merged_bundle_for_save(self, output_path: str) -> Dict[str, object]:
        bundle = self.current_graph_bundle if isinstance(self.current_graph_bundle, dict) else {}
        disk_bundle: Dict[str, object] = {}
        if output_path and os.path.isfile(output_path):
            try:
                loaded = _read_json(output_path)
                if isinstance(loaded, dict) and isinstance(loaded.get("graphs"), list):
                    disk_bundle = loaded
            except Exception:
                disk_bundle = {}

        base = disk_bundle if disk_bundle else bundle
        if not isinstance(base, dict):
            base = {}
        base_graphs = [dict(g) for g in list(base.get("graphs") or []) if isinstance(g, dict)]
        if isinstance(self.current_graph, dict):
            merge_graphs = [self._json_safe_clone(self.current_graph)]
        else:
            merge_graphs = [dict(g) for g in list(bundle.get("graphs") or []) if isinstance(g, dict)]

        for graph in merge_graphs:
            target_frame = self._extract_graph_frame_idx(graph)
            replaced = False
            for i, row in enumerate(base_graphs):
                if self._extract_graph_frame_idx(row) == target_frame:
                    base_graphs[i] = dict(graph)
                    replaced = True
                    break
            if not replaced:
                base_graphs.append(dict(graph))
        if base_graphs:
            base["graphs"] = base_graphs
        return base

    @staticmethod
    def _compact_node_for_bundle(node: Dict[str, object]) -> Dict[str, object]:
        attrs = []
        for attr in list(node.get("attributes") or []):
            if not isinstance(attr, dict):
                continue
            attrs.append(
                {
                    "slot": str(attr.get("slot", "") or ""),
                    "value": attr.get("value", ""),
                    "confidence": float(attr.get("confidence", 0.0) or 0.0),
                    "provenance": str(attr.get("provenance", "") or ""),
                    "verified": bool(attr.get("verified", False)),
                }
            )
        # Keep detector score as primary confidence in compact rows.
        # verify_confidence is preserved separately for debugging/comparison.
        sam_conf = float(node.get("score", node.get("confidence", 0.0)) or 0.0)
        verify_conf_raw = node.get("verify_confidence", None)
        if verify_conf_raw is None:
            verify_conf = sam_conf
        else:
            try:
                verify_conf = float(verify_conf_raw or 0.0)
            except Exception:
                verify_conf = sam_conf
        out = {
            "entity_id": str(node.get("entity_id", "") or ""),
            "display_name": str(node.get("display_name", "") or ""),
            "label": str(node.get("canonical_label", node.get("label", "")) or ""),
            "bbox": list(node.get("bbox") or [0, 0, 0, 0])[:4],
            "confidence": float(sam_conf),
            "score": float(sam_conf),
            "verify_confidence": float(verify_conf),
            "stage_confidence": {
                "sam": float(sam_conf),
                "verify": float(verify_conf),
            },
            "attributes": attrs,
        }
        track_id = str(node.get("track_id", "") or "")
        if track_id:
            out["track_id"] = track_id
        flags = [str(x) for x in list(node.get("validator_flags") or []) if str(x)]
        if flags:
            out["flags"] = flags
        return out

    @staticmethod
    def _node_label(node: Dict[str, object], default: str = "object") -> str:
        return str(node.get("canonical_label", node.get("label", default)) or default)

    @staticmethod
    def _node_attributes_to_text(node: Dict[str, object]) -> str:
        parts: List[str] = []
        for attr in list(node.get("attributes") or []):
            if not isinstance(attr, dict):
                continue
            slot = str(attr.get("slot", "") or "").strip()
            value = str(attr.get("value", "") or "").strip()
            if not slot:
                continue
            parts.append(f"{slot}={value}")
        return "\n".join(parts)

    @staticmethod
    def _parse_node_attributes_text(text: str) -> Tuple[List[Dict[str, object]], bool]:
        raw = str(text or "").strip()
        if not raw:
            return [], True
        # Accept JSON list payload directly.
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    return [], False
                out: List[Dict[str, object]] = []
                for row in parsed:
                    if not isinstance(row, dict):
                        continue
                    slot = str(row.get("slot", "") or "").strip()
                    if not slot:
                        continue
                    value = str(row.get("value", "") or "").strip()
                    out.append(
                        {
                            "slot": slot,
                            "value": value,
                            "confidence": float(row.get("confidence", 1.0) or 1.0),
                            "provenance": str(row.get("provenance", "human_edit") or "human_edit"),
                            "verified": bool(row.get("verified", True)),
                        }
                    )
                return out, True
            except Exception:
                return [], False
        # Friendly shorthand: slot=value; slot2=value2
        out: List[Dict[str, object]] = []
        normalized = raw.replace("\n", ";")
        for part in [x.strip() for x in normalized.split(";") if str(x).strip()]:
            if "=" not in part:
                return [], False
            slot, value = part.split("=", 1)
            slot = str(slot or "").strip()
            if not slot:
                return [], False
            out.append(
                {
                    "slot": slot,
                    "value": str(value or "").strip(),
                    "confidence": 1.0,
                    "provenance": "human_edit",
                    "verified": True,
                }
            )
        return out, True

    @staticmethod
    def _node_confidence(node: Dict[str, object]) -> float:
        # UI policy: prefer initial scene-graph confidence over post-verify confidence.
        zero_fallback: Optional[float] = None
        for key in ("score", "confidence", "verify_confidence", "conf", "det_confidence", "det_score", "track_score", "quality"):
            try:
                val = float(node.get(key, 0.0) or 0.0)
            except Exception:
                continue
            if val > 1.0 and val <= 100.0:
                val = val / 100.0
            val = max(0.0, min(1.0, val))
            if val > 0.0:
                return val
            if zero_fallback is None:
                zero_fallback = val
        # Compact-bundle fallback for legacy/partial rows.
        stage_conf = node.get("stage_confidence")
        if isinstance(stage_conf, dict):
            for key in ("sam", "verify"):
                try:
                    val = float(stage_conf.get(key, 0.0) or 0.0)
                except Exception:
                    continue
                if val > 1.0 and val <= 100.0:
                    val = val / 100.0
                if val > 0.0:
                    return max(0.0, min(1.0, val))
        if zero_fallback is not None:
            return float(zero_fallback)
        return 0.0

    @staticmethod
    def _node_bbox_confidence(node: Dict[str, object]) -> float:
        # Keep UI confidence source consistent between right table and left overlay.
        return VideoTaskStudio._node_confidence(node)

    @staticmethod
    def _edge_confidence(edge: Dict[str, object]) -> float:
        for key in ("verify_confidence", "confidence", "score", "conf", "quality"):
            try:
                val = float(edge.get(key, 0.0) or 0.0)
            except Exception:
                continue
            if val > 1.0 and val <= 100.0:
                val = val / 100.0
            return max(0.0, min(1.0, val))
        return 0.0

    @staticmethod
    def _compact_edge_for_bundle(edge: Dict[str, object]) -> Dict[str, object]:
        out = {
            "edge_id": str(edge.get("edge_id", "") or ""),
            "src_id": str(edge.get("src_id", "") or ""),
            "dst_id": str(edge.get("dst_id", "") or ""),
            "relation": str(edge.get("relation", "") or ""),
            "confidence": float(edge.get("confidence", edge.get("score", 0.0)) or 0.0),
        }
        flags = [str(x) for x in list(edge.get("validator_flags") or []) if str(x)]
        if flags:
            out["flags"] = flags
        return out

    def _compact_graph_for_bundle(self, graph: Dict[str, object]) -> Dict[str, object]:
        meta = dict(graph.get("metadata") or {})
        validation = dict(graph.get("validation") or {})
        compact: Dict[str, object] = {
            "image_id": str(graph.get("image_id", "") or ""),
            "frame_idx": int(meta.get("graph_frame_idx", graph.get("frame_idx", 0)) or 0),
            "time_sec": float(meta.get("graph_time_sec", graph.get("time_sec", 0.0)) or 0.0),
            "image_path": str(meta.get("image_path", graph.get("image_path", "")) or ""),
            "summary": str(
                meta.get("global_semantic_summary", meta.get("global_summary", graph.get("summary", ""))) or ""
            ),
            "stage_scores": dict(validation.get("module_scores") or graph.get("stage_scores") or {}),
            "entity_display_map": dict(meta.get("entity_display_map") or graph.get("entity_display_map") or {}),
            "nodes": [
                self._compact_node_for_bundle(dict(node))
                for node in list(graph.get("nodes") or [])
                if isinstance(node, dict)
            ],
            "edges": [
                self._compact_edge_for_bundle(dict(edge))
                for edge in list(graph.get("edges") or [])
                if isinstance(edge, dict)
            ],
        }
        if meta.get("cycle_verification") or graph.get("cycle"):
            cv = dict(meta.get("cycle_verification") or graph.get("cycle") or {})
            claims_payload = cv.get("claims")
            if isinstance(claims_payload, dict):
                claims_value: object = dict(claims_payload)
            elif isinstance(claims_payload, list):
                claims_value = [dict(x) if isinstance(x, dict) else x for x in claims_payload]
            else:
                claims_value = {}
            compact["cycle"] = {
                "claims": claims_value,
                "votes": list(cv.get("votes") or []),
                "probe_results": list(cv.get("probe_results") or []),
                "resolved_claims": list(cv.get("resolved_claims") or []),
                "suppressed_questions": list(cv.get("suppressed_questions") or []),
                "correction_candidates": dict(cv.get("correction_candidates") or {}),
                "summary": dict(cv.get("summary") or {}),
                "human_queue": list(cv.get("human_queue") or []),
                "runtime": dict(cv.get("runtime") or {}),
                "caption": dict(cv.get("caption") or {}),
                "policy": dict(cv.get("policy") or {}),
                "debug": dict(cv.get("debug") or {}),
            }
        return compact

    def _compact_scene_graph_bundle(self, bundle: Dict[str, object]) -> Dict[str, object]:
        graphs = [g for g in list(bundle.get("graphs") or []) if isinstance(g, dict)]
        out = {
            "type": "scene_graph_sequence",
            "version": 2,
            "format": "compact",
            "video_path": str(bundle.get("video_path", "") or self.video_path or ""),
            "video_name": str(bundle.get("video_name", "") or os.path.basename(str(self.video_path or ""))),
            "frame_count": int(bundle.get("frame_count", int(self.player.frame_count or 0)) or 0),
            "source_fps": float(bundle.get("source_fps", self._sg_job_source_fps or 1.0) or 1.0),
            "sampling_fps": float(bundle.get("sampling_fps", self._sg_job_sampling_fps or 1.0) or 1.0),
            "sampled_frame_indices": [int(x) for x in list(bundle.get("sampled_frame_indices") or [])],
            "graphs": [self._compact_graph_for_bundle(dict(g)) for g in graphs],
        }
        validation = dict(bundle.get("validation") or {})
        if validation:
            out["validation"] = validation
        llm_summaries = [dict(x) for x in list(bundle.get("llm_batch_summaries") or []) if isinstance(x, dict)]
        if llm_summaries:
            out["llm_batch_summaries"] = llm_summaries
        llm_attrs = [dict(x) for x in list(bundle.get("llm_person_attributes") or []) if isinstance(x, dict)]
        if llm_attrs:
            out["llm_person_attributes"] = llm_attrs
        if str(bundle.get("video_level_caption", "") or "").strip():
            out["video_level_caption"] = str(bundle.get("video_level_caption", "") or "").strip()
        video_level_multi = [dict(x) for x in list(bundle.get("video_level_multi_turn_vqa") or []) if isinstance(x, dict)]
        if video_level_multi:
            out["video_level_multi_turn_vqa"] = video_level_multi
        video_level_keyframes = [int(x) for x in list(bundle.get("video_level_keyframes") or [])]
        if video_level_keyframes:
            out["video_level_keyframes"] = video_level_keyframes
        video_level_verif = dict(bundle.get("video_level_verification") or {})
        if video_level_verif:
            out["video_level_verification"] = video_level_verif
        viz = dict(bundle.get("visualization") or {})
        if viz:
            out["visualization"] = {
                "exported": int(viz.get("exported", 0) or 0),
                "dir": str(viz.get("dir", "") or ""),
                "items": [dict(x) for x in list(viz.get("items") or []) if isinstance(x, dict)],
            }
        return out

    def _resume_summary_rows_from_bundle_graphs(
        self,
        graphs: List[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        sorted_graphs = [dict(g) for g in list(graphs or []) if isinstance(g, dict)]
        sorted_graphs.sort(key=lambda g: int(self._extract_graph_frame_idx(g) or 0))
        for g in sorted_graphs:
            frame_idx = int(self._extract_graph_frame_idx(g) or 0)
            text = str(
                g.get("summary")
                or ((g.get("metadata") or {}).get("global_semantic_summary"))
                or ((g.get("metadata") or {}).get("global_summary"))
                or ""
            ).strip()
            if not text:
                continue
            out.append(
                {
                    "start_frame": int(frame_idx),
                    "end_frame": int(frame_idx),
                    "summary": text,
                }
            )
        return out

    @staticmethod
    def _graph_cycle_payload(graph: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(graph, dict):
            return {}
        meta = dict(graph.get("metadata") or {})
        return dict(meta.get("cycle_verification") or graph.get("cycle") or {})

    def _sync_cycle_result_with_current_graph(self, *, force: bool = False) -> None:
        graph = self.current_graph or {}
        if not isinstance(graph, dict):
            self.current_cycle_result = None
            self._cycle_result_frame_idx = -1
            return
        graph_frame_idx = int(self._extract_graph_frame_idx(graph) or -1)
        if bool(force) or int(getattr(self, "_cycle_result_frame_idx", -1)) != graph_frame_idx:
            self.current_cycle_result = None
            self._cycle_result_frame_idx = graph_frame_idx
        if isinstance(self.current_cycle_result, dict) and not bool(force):
            return
        cv = self._graph_cycle_payload(dict(graph or {}))
        if cv:
            claims_payload = cv.get("claims")
            if isinstance(claims_payload, dict):
                claims_value: object = dict(claims_payload)
            elif isinstance(claims_payload, list):
                claims_value = [dict(x) if isinstance(x, dict) else x for x in claims_payload]
            else:
                claims_value = {}
            self.current_cycle_result = {
                "claims": claims_value,
                "votes": list(cv.get("votes") or []),
                "probe_results": list(cv.get("probe_results") or []),
                "resolved_claims": list(cv.get("resolved_claims") or []),
                "suppressed_questions": list(cv.get("suppressed_questions") or []),
                "correction_candidates": dict(cv.get("correction_candidates") or {}),
                "human_queue": list(cv.get("human_queue") or []),
                "summary": dict(cv.get("summary") or {}),
                "runtime": dict(cv.get("runtime") or {}),
                "caption": dict(cv.get("caption") or {}),
                "policy": dict(cv.get("policy") or {}),
                "debug": dict(cv.get("debug") or {}),
                "graph_after": dict(graph or {}),
            }
        else:
            self.current_cycle_result = None

    def _normalize_graph_for_resume(self, graph: Dict[str, object]) -> Dict[str, object]:
        out = dict(graph or {})
        meta = dict(out.get("metadata") or {})
        if "graph_frame_idx" not in meta:
            try:
                meta["graph_frame_idx"] = int(out.get("frame_idx", 0) or 0)
            except Exception:
                pass
        if "graph_time_sec" not in meta:
            try:
                meta["graph_time_sec"] = float(out.get("time_sec", 0.0) or 0.0)
            except Exception:
                pass
        if "image_path" not in meta and str(out.get("image_path", "") or "").strip():
            meta["image_path"] = str(out.get("image_path", "") or "").strip()
        if "global_semantic_summary" not in meta and str(out.get("summary", "") or "").strip():
            text = str(out.get("summary", "") or "").strip()
            meta["global_semantic_summary"] = text
            meta.setdefault("global_summary", text)
        cycle_payload = self._graph_cycle_payload(out)
        if cycle_payload:
            claims_payload = cycle_payload.get("claims")
            if isinstance(claims_payload, dict):
                claims_value: object = dict(claims_payload)
            elif isinstance(claims_payload, list):
                claims_value = [dict(x) if isinstance(x, dict) else x for x in claims_payload]
            else:
                claims_value = {}
            meta["cycle_verification"] = {
                "claims": claims_value,
                "votes": list(cycle_payload.get("votes") or []),
                "probe_results": list(cycle_payload.get("probe_results") or []),
                "resolved_claims": list(cycle_payload.get("resolved_claims") or []),
                "suppressed_questions": list(cycle_payload.get("suppressed_questions") or []),
                "correction_candidates": dict(cycle_payload.get("correction_candidates") or {}),
                "summary": dict(cycle_payload.get("summary") or {}),
                "human_queue": list(cycle_payload.get("human_queue") or []),
                "runtime": dict(cycle_payload.get("runtime") or {}),
                "caption": dict(cycle_payload.get("caption") or {}),
                "policy": dict(cycle_payload.get("policy") or {}),
                "debug": dict(cycle_payload.get("debug") or {}),
            }
        out["metadata"] = meta
        return out

    def _prepare_bundle_for_resume(
        self,
        *,
        bundle: Dict[str, object],
        run_info_path: str = "",
    ) -> Tuple[Dict[str, object], float, int]:
        out = dict(bundle or {})
        graphs = [self._normalize_graph_for_resume(dict(g)) for g in list(out.get("graphs") or []) if isinstance(g, dict)]
        out["graphs"] = [dict(g) for g in graphs]
        summary_rows = [dict(x) for x in list(out.get("llm_batch_summaries") or []) if isinstance(x, dict)]
        if not summary_rows:
            summary_rows = self._resume_summary_rows_from_bundle_graphs(graphs)
            if summary_rows:
                out["llm_batch_summaries"] = summary_rows

        summary_score = 0.0
        info_path = str(run_info_path or "").strip()
        if info_path and os.path.isfile(info_path):
            run_info = _read_json(info_path)
            stage_validation = dict(run_info.get("stage_validation") or {})
            stage_summary = dict(stage_validation.get("summary") or {})
            if stage_summary:
                validation = dict(out.get("validation") or {})
                merged_validation = dict(stage_summary)
                merged_validation.update(validation)
                out["validation"] = merged_validation
                module_scores = dict(merged_validation.get("module_scores") or merged_validation.get("module_scores_avg") or {})
                try:
                    summary_score = float(module_scores.get("S_summary", 0.0) or 0.0)
                except Exception:
                    summary_score = 0.0

        return out, float(max(0.0, min(1.0, summary_score))), int(len(summary_rows))

    def _export_current_graph_visualization(self, output_path: str) -> None:
        if not isinstance(self.current_graph, dict):
            return
        image_path = self._graph_image_path(self.current_graph)
        if not image_path or not os.path.isfile(image_path):
            return
        frame = cv2.imread(image_path)
        if frame is None:
            return
        frame_idx = int(self._extract_graph_frame_idx(self.current_graph) or 0)
        node_centers: Dict[str, Tuple[int, int]] = {}
        for node in list(self.current_graph.get("nodes") or []):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("entity_id", "") or "").strip()
            bbox = list(node.get("bbox") or [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            x, y, w, h = [int(v) for v in bbox[:4]]
            if w <= 0 or h <= 0:
                continue
            color = (60, 190, 255) if str(self._node_label(node, "") or "").strip().lower() == "person" else (120, 220, 120)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if node_id:
                node_centers[node_id] = (int(x + (w / 2.0)), int(y + (h / 2.0)))
            tid = str(node.get("track_id", node.get("entity_id", "")) or "")
            lbl = str(self._node_label(node, "") or "")
            score = float(self._node_bbox_confidence(node))
            text = f"{lbl} {tid} {score:.2f}".strip()
            cv2.putText(
                frame,
                text,
                (x, max(16, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        for edge in list(self.current_graph.get("edges") or []):
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("src_id", "") or "").strip()
            dst = str(edge.get("dst_id", "") or "").strip()
            if src not in node_centers or dst not in node_centers:
                continue
            p1 = node_centers[src]
            p2 = node_centers[dst]
            rel = str(edge.get("relation", "") or "").strip()
            cv2.line(frame, p1, p2, (0, 160, 80), 1, cv2.LINE_AA)
            mx = int((p1[0] + p2[0]) / 2.0)
            my = int((p1[1] + p2[1]) / 2.0)
            if rel:
                cv2.putText(
                    frame,
                    rel,
                    (mx, max(12, my - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 120, 60),
                    1,
                    cv2.LINE_AA,
                )
        stem = os.path.splitext(os.path.basename(str(output_path or "scene_graph_bundle.json")))[0]
        out_dir = os.path.join(os.path.dirname(str(output_path or self._repo_root)), f"{stem}_viz")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"frame_{frame_idx:06d}.jpg")
        if not cv2.imwrite(out_path, frame):
            return
        bundle = self.current_graph_bundle if isinstance(self.current_graph_bundle, dict) else {}
        viz = dict(bundle.get("visualization") or {})
        items = [dict(x) for x in list(viz.get("items") or []) if isinstance(x, dict)]
        updated = False
        for i, row in enumerate(items):
            if int(row.get("frame_idx", -1) or -1) == frame_idx:
                items[i] = {"frame_idx": frame_idx, "image_path": out_path}
                updated = True
                break
        if not updated:
            items.append({"frame_idx": frame_idx, "image_path": out_path})
        items.sort(key=lambda x: int(x.get("frame_idx", 0) or 0))
        viz["dir"] = out_dir
        viz["items"] = items
        viz["exported"] = int(len(items))
        bundle["visualization"] = viz
        self.current_graph_bundle = bundle

    def _resolve_scene_graph_bundle_output_path(self) -> str:
        path = str(self._sg_job_output_path or "").strip()
        if path:
            return path
        run_dir = str(self._sg_run_dir or "").strip()
        if run_dir:
            return os.path.join(run_dir, "scene_graph_bundle.json")
        video_path = str(self.video_path or "").strip()
        if video_path:
            base_dir = os.path.dirname(video_path) or self._repo_root
            return os.path.join(base_dir, "scene_graph_bundle.json")
        return os.path.join(self._repo_root, "scene_graph_bundle.json")

    def _persist_current_scene_graph_bundle(self, reason: str = "scene_graph_edit") -> None:
        """Overwrite the active run bundle after UI edits."""
        if not isinstance(self.current_graph, dict):
            return
        output_path = self._resolve_scene_graph_bundle_output_path()
        if not output_path:
            return
        try:
            self._export_current_graph_visualization(output_path)
            bundle_to_write = self._merged_bundle_for_save(output_path)
            bundle_to_write = self._compact_scene_graph_bundle(bundle_to_write)
            _write_json(output_path, bundle_to_write)
            self.current_graph_bundle = bundle_to_write
            self._append_oplog(str(reason or "scene_graph_edit_saved"), path=output_path)
            self._set_status(f"Saved scene graph edits: {os.path.basename(output_path)}", status_type="success")
        except Exception as exc:
            self._append_runtime_log(f"Failed to save scene graph edits: {exc}", level="warning")
            self._set_status(f"Failed to save scene graph edits: {exc}", status_type="warning")

    def _render_cycle_probe_outputs(self, probe_results: List[Dict[str, object]]) -> None:
        claim_frame_map = self._cycle_claim_frame_index()
        active_frame_idx = self._active_graph_frame_idx()
        single_all = [
            dict(item)
            for item in list(probe_results or [])
            if self._normalize_probe_view_type(str(item.get("view_type", "") or "")) == "single_turn_vqa"
        ]
        multi_all = [
            dict(item)
            for item in list(probe_results or [])
            if self._normalize_probe_view_type(str(item.get("view_type", "") or "")) == "multi_turn_vqa"
        ]
        single_all = [dict(item) for item in list(single_all or []) if not self._is_probe_manually_resolved(dict(item or {}))]
        multi_all = [dict(item) for item in list(multi_all or []) if not self._is_probe_manually_resolved(dict(item or {}))]
        single_items = [
            dict(item)
            for item in list(single_all or [])
            if self._probe_row_matches_frame(item, active_frame_idx, claim_frame_map=claim_frame_map)
        ]
        multi_items = [
            dict(item)
            for item in list(multi_all or [])
            if self._probe_row_matches_frame(item, active_frame_idx, claim_frame_map=claim_frame_map)
        ]
        # Fallback: if frame filtering hides everything, keep data visible for auditability.
        if not single_items and single_all:
            single_items = [dict(x) for x in single_all]
        if not multi_items and multi_all:
            multi_items = [dict(x) for x in multi_all]
        self.single_turn_items = single_items
        self.multi_turn_items = multi_items
        single_prev = bool(self.single_list.blockSignals(True))
        multi_prev = bool(self.multi_list.blockSignals(True))
        try:
            self.single_list.clear()
            for item in self.single_turn_items:
                self.single_list.addItem(self._build_probe_list_item(item, is_multi=False))
            if self.single_turn_items:
                self.single_list.setCurrentRow(0)
            self.multi_list.clear()
            for item in self.multi_turn_items:
                self.multi_list.addItem(self._build_probe_list_item(item, is_multi=True))
            if self.multi_turn_items:
                self.multi_list.setCurrentRow(0)
        finally:
            self.single_list.blockSignals(single_prev)
            self.multi_list.blockSignals(multi_prev)
        self._render_single_detail(self.single_list.currentRow())
        self._render_multi_detail(self.multi_list.currentRow())
        selected_row = self._selected_row(self.sg_nodes_table) if hasattr(self, "sg_nodes_table") else -1
        node_id = ""
        if selected_row >= 0 and hasattr(self, "sg_nodes_table"):
            node_id = self._entity_id_from_table_item(self.sg_nodes_table.item(selected_row, 0))
        self._render_object_probe_drawer(node_id)

    def _on_single_probe_row_changed(self, row: int) -> None:
        self._render_single_detail(row)
        if row < 0 or row >= len(self.single_turn_items):
            return
        self._focus_probe_result_row(dict(self.single_turn_items[row] or {}))

    def _on_multi_probe_row_changed(self, row: int) -> None:
        self._render_multi_detail(row)
        if row < 0 or row >= len(self.multi_turn_items):
            return
        self._focus_probe_result_row(dict(self.multi_turn_items[row] or {}))

    def _refresh_claim_verification_tables(self) -> None:
        """Populate the per-tab claim verification tables from current_cycle_result.probe_results."""
        result = self.current_cycle_result if isinstance(self.current_cycle_result, dict) else {}
        probe_results = list(result.get("probe_results") or [])
        claim_frame_map = self._cycle_claim_frame_index()
        active_frame_idx = self._active_graph_frame_idx()
        by_type_all: Dict[str, List[Dict[str, object]]] = {
            "single_turn_vqa": [],
            "multi_turn_vqa": [],
            "caption": [],
        }
        by_type: Dict[str, List[Dict[str, object]]] = {
            "single_turn_vqa": [],
            "multi_turn_vqa": [],
            "caption": [],
        }
        for row in probe_results:
            vt = self._normalize_probe_view_type(str(row.get("view_type", "") or ""))
            if vt not in by_type_all:
                continue
            row_norm = dict(row)
            row_norm["view_type"] = vt
            if vt in {"single_turn_vqa", "multi_turn_vqa"} and self._is_probe_manually_resolved(row_norm):
                continue
            by_type_all[vt].append(row_norm)
            if vt in {"single_turn_vqa", "multi_turn_vqa"} and not self._probe_row_matches_frame(
                row_norm,
                active_frame_idx,
                claim_frame_map=claim_frame_map,
            ):
                continue
            by_type[vt].append(row_norm)
        # Fallback: when frame filtering removes all rows, keep unfiltered rows visible.
        for key in ("single_turn_vqa", "multi_turn_vqa"):
            if not by_type[key] and by_type_all[key]:
                by_type[key] = [dict(x) for x in by_type_all[key]]
                try:
                    self._append_runtime_log(
                        f"[CYCLE-UI] frame filter fallback for {key}: active_frame={active_frame_idx} rows={len(by_type_all[key])}",
                        level="info",
                    )
                except Exception:
                    pass
        for attr, vtype in [
            ("single_claims_table", "single_turn_vqa"),
            ("multi_claims_table", "multi_turn_vqa"),
            ("caption_claims_table", "caption"),
        ]:
            tbl = getattr(self, attr, None)
            if not isinstance(tbl, QTableWidget):
                continue
            rows = by_type[vtype]
            prev_block = bool(tbl.blockSignals(True))
            try:
                if vtype == "caption":
                    rows = []
                if vtype == "single_turn_vqa":
                    self._single_claim_table_rows = [dict(x) for x in rows]
                elif vtype == "multi_turn_vqa":
                    self._multi_claim_table_rows = [dict(x) for x in rows]
                tbl.setRowCount(len(rows))
                for i, row in enumerate(rows):
                    resp = self._probe_response_payload(row)
                    if not isinstance(resp, dict) or not resp:
                        legacy_resp = row.get("response")
                        resp = dict(legacy_resp or {}) if isinstance(legacy_resp, dict) else {}
                    answer = str(resp.get("answer") or resp.get("selection") or "")
                    score_val = -1.0
                    try:
                        raw_score = resp.get("score", None)
                        if raw_score is not None:
                            score_val = float(raw_score)
                    except Exception:
                        score_val = -1.0
                    schema_valid = bool(resp.get("schema_valid", row.get("schema_valid", True)))
                    invalid_resp = self._probe_is_invalid(row, resp)
                    stale_resp = self._probe_is_stale(row, resp)
                    resolved_resp = self._is_probe_manually_resolved(row)
                    if resolved_resp:
                        answer = "Resolved"
                        score_val = -1.0
                    elif invalid_resp:
                        answer = "⚠ Invalid Response"
                    elif (not stale_resp) and str(answer).strip().lower() == "uncertain" and score_val >= 0.0 and score_val <= 0.05:
                        answer = "uncertain (low confidence)"
                    provider = str(
                        row.get("response_provider")
                        or (dict(resp.get("raw_response") or {}).get("provider"))
                        or resp.get("provider")
                        or ""
                    ).strip()
                    raw = str(resp.get("reason") or resp.get("raw_text") or "").strip()
                    if resolved_resp:
                        raw = f"[Resolved] {raw}".strip()
                    if provider:
                        raw = f"[{provider}] {raw}".strip()
                    claim_id = str(row.get("target_claim_id") or row.get("claim_id") or "")
                    question = self._humanize_question_text(str(row.get("question") or "").strip())
                    if vtype in {"single_turn_vqa", "multi_turn_vqa"}:
                        frame_idx = self._probe_row_source_frame_idx(row, claim_frame_map=claim_frame_map)
                        if frame_idx is None:
                            frame_idx = self._probe_row_frame_idx(row)
                        values = (
                            str(frame_idx if frame_idx is not None else "-"),
                            claim_id,
                            question,
                            answer,
                            f"{score_val:.2f}" if score_val >= 0.0 else "N/A",
                            "TRUE" if schema_valid else "FALSE",
                            raw,
                        )
                    else:
                        values = (
                            claim_id,
                            question,
                            answer,
                            f"{score_val:.2f}" if score_val >= 0.0 else "N/A",
                            "TRUE" if schema_valid else "FALSE",
                            raw,
                        )
                    for col, text in enumerate(values):
                        cell = QTableWidgetItem(str(text))
                        cell.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                        if invalid_resp and (not resolved_resp):
                            cell.setBackground(QColor(255, 230, 230))
                        elif (not resolved_resp) and (not stale_resp) and score_val >= 0.0 and self._is_low_confidence_probe(answer, score_val):
                            cell.setBackground(QColor(255, 245, 204))
                        tbl.setItem(i, col, cell)
                tbl.resizeRowsToContents()
                if tbl.rowCount() > 0:
                    tbl.selectRow(0)
                else:
                    tbl.clearSelection()
            finally:
                tbl.blockSignals(prev_block)
        if not self.single_turn_items:
            self.single_detail.clear()
        if not self.multi_turn_items:
            self.multi_detail.clear()

    def _probe_row_frame_idx(self, row: Dict[str, object]) -> Optional[int]:
        if not isinstance(row, dict):
            return None
        for key in ("frame_idx", "graph_frame_idx"):
            try:
                if key in row and int(row.get(key)) >= 0:
                    return int(row.get(key))
            except Exception:
                pass
        try:
            if "temporal_anchor_frame_idx" in row and int(row.get("temporal_anchor_frame_idx")) >= 0:
                return int(row.get("temporal_anchor_frame_idx"))
        except Exception:
            pass
        if isinstance(self.current_graph, dict):
            return self._extract_graph_frame_idx(self.current_graph)
        return None

    def _active_graph_frame_idx(self) -> Optional[int]:
        # Prefer the frame bound to the current cycle result to avoid accidental empty UI
        # when playback frame drifts during/after async cycle execution.
        try:
            cycle_frame = int(getattr(self, "_cycle_result_frame_idx", -1))
            if cycle_frame >= 0:
                return cycle_frame
        except Exception:
            pass
        graph = self.current_graph if isinstance(self.current_graph, dict) else {}
        frame_idx = self._extract_graph_frame_idx(graph)
        if frame_idx is not None:
            return int(frame_idx)
        if hasattr(self, "spin_frame_for_graph"):
            try:
                value = int(self.spin_frame_for_graph.value())
                if value >= 0:
                    return value
            except Exception:
                pass
        return None

    def _cycle_claim_frame_index(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        result = self.current_cycle_result if isinstance(self.current_cycle_result, dict) else {}
        claims = result.get("claims")
        claim_rows: List[Dict[str, object]] = []
        if isinstance(claims, dict):
            claim_rows = [dict(v) for v in claims.values() if isinstance(v, dict)]
        elif isinstance(claims, list):
            claim_rows = [dict(v) for v in claims if isinstance(v, dict)]
        for claim in claim_rows:
            claim_id = str(claim.get("claim_id") or "").strip()
            if not claim_id:
                continue
            frame_idx = self._claim_source_frame_idx(claim)
            if frame_idx is not None and frame_idx >= 0:
                out[claim_id] = int(frame_idx)
        return out

    def _claim_source_frame_idx(self, claim: Dict[str, object]) -> Optional[int]:
        if not isinstance(claim, dict):
            return None
        for key in ("frame_idx", "graph_frame_idx", "temporal_anchor_frame_idx"):
            try:
                if key in claim and int(claim.get(key)) >= 0:
                    return int(claim.get(key))
            except Exception:
                pass
        snapshot_id = str(claim.get("source_graph_snapshot_id") or "").strip()
        if snapshot_id:
            m = re.search(r"_f(\d+)$", snapshot_id)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    pass
        provenance = list(claim.get("provenance") or [])
        for row in provenance:
            if not isinstance(row, dict):
                continue
            for key in ("frame_idx", "graph_frame_idx"):
                try:
                    if key in row and int(row.get(key)) >= 0:
                        return int(row.get(key))
                except Exception:
                    pass
        return None

    def _probe_row_source_frame_idx(
        self,
        row: Dict[str, object],
        *,
        claim_frame_map: Optional[Dict[str, int]] = None,
    ) -> Optional[int]:
        if not isinstance(row, dict):
            return None
        for key in ("frame_idx", "graph_frame_idx", "temporal_anchor_frame_idx"):
            try:
                if key in row and int(row.get(key)) >= 0:
                    return int(row.get(key))
            except Exception:
                pass
        span = list(row.get("summary_span") or [])
        if span:
            try:
                start_idx = int(span[0])
                if start_idx >= 0:
                    return start_idx
            except Exception:
                pass
        claim_id = str(row.get("target_claim_id") or row.get("claim_id") or "").strip()
        if claim_id:
            mapping = claim_frame_map if isinstance(claim_frame_map, dict) else self._cycle_claim_frame_index()
            if claim_id in mapping:
                try:
                    return int(mapping.get(claim_id))
                except Exception:
                    pass
        return None

    def _probe_row_matches_frame(
        self,
        row: Dict[str, object],
        active_frame_idx: Optional[int],
        *,
        claim_frame_map: Optional[Dict[str, int]] = None,
    ) -> bool:
        if active_frame_idx is None or int(active_frame_idx) < 0:
            return True
        frame_idx = self._probe_row_source_frame_idx(row, claim_frame_map=claim_frame_map)
        if frame_idx is None:
            # Keep unlabeled rows visible rather than silently hiding potentially useful probes.
            return True
        return int(frame_idx) == int(active_frame_idx)

    def _focus_probe_result_row(self, row: Dict[str, object]) -> None:
        if not isinstance(row, dict):
            return
        self._selected_probe_id = str(row.get("probe_id", "") or "").strip()
        self._selected_claim_id = str(row.get("target_claim_id") or row.get("claim_id") or "").strip()
        self._selected_node_ids = [str(x or "").strip() for x in list(row.get("evidence_node_ids") or []) if str(x or "").strip()]
        frame_idx = self._probe_row_frame_idx(row)
        if frame_idx is not None and frame_idx >= 0:
            self._current_frame_id = int(frame_idx)
            self._seek_frame(int(frame_idx))
            self._set_graph_frame_selector(int(frame_idx), manual=True)
        tmp_row = {
            "claim_id": str(row.get("target_claim_id") or row.get("claim_id") or ""),
            "subject_id": "",
            "object_id": "",
            "predicate": "",
            "evidence_node_ids": list(row.get("evidence_node_ids") or []),
            "evidence_edge_ids": list(row.get("evidence_edge_ids") or []),
        }
        self._focus_human_queue_row(tmp_row)

    def _on_single_probe_item_hovered(self, item: QListWidgetItem) -> None:
        if item is None:
            return
        idx = int(self.single_list.row(item))
        if idx < 0 or idx >= len(self.single_turn_items):
            return
        self._focus_probe_result_row(dict(self.single_turn_items[idx] or {}))

    def _on_multi_probe_item_hovered(self, item: QListWidgetItem) -> None:
        if item is None:
            return
        idx = int(self.multi_list.row(item))
        if idx < 0 or idx >= len(self.multi_turn_items):
            return
        self._focus_probe_result_row(dict(self.multi_turn_items[idx] or {}))

    def _on_single_claim_selection_changed(self) -> None:
        table = getattr(self, "single_claims_table", None)
        if not isinstance(table, QTableWidget):
            return
        row_idx = int(table.currentRow())
        rows = list(getattr(self, "_single_claim_table_rows", []) or [])
        if row_idx < 0 or row_idx >= len(rows):
            return
        self._focus_probe_result_row(dict(rows[row_idx] or {}))

    def _on_multi_claim_selection_changed(self) -> None:
        table = getattr(self, "multi_claims_table", None)
        if not isinstance(table, QTableWidget):
            return
        row_idx = int(table.currentRow())
        rows = list(getattr(self, "_multi_claim_table_rows", []) or [])
        if row_idx < 0 or row_idx >= len(rows):
            return
        self._focus_probe_result_row(dict(rows[row_idx] or {}))

    def _friendly_entity_display_map(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        # Prefer the current frame display map first.
        for eid, display in dict(self._node_display_by_id or {}).items():
            token = str(display or "").strip().replace("_", " ")
            if eid and token:
                out[str(eid)] = token
        # Backfill from the bundle if needed.
        bundle = self.current_graph_bundle if isinstance(self.current_graph_bundle, dict) else {}
        graphs = [dict(g) for g in list(bundle.get("graphs") or []) if isinstance(g, dict)]
        for graph in graphs:
            counts: Dict[str, int] = {}
            for node in list(graph.get("nodes") or []):
                if not isinstance(node, dict):
                    continue
                eid = str(node.get("entity_id", "") or "").strip()
                if not eid or eid in out:
                    continue
                label = self._display_label_token(self._node_label(node, "object")).replace("_", " ")
                counts[label] = int(counts.get(label, 0) or 0) + 1
                out[eid] = f"{label} {counts[label]}"
        return out

    def _humanize_question_text(self, text: str) -> str:
        out = str(text or "").strip()
        if not out:
            return out
        mapping = self._friendly_entity_display_map()
        # Replace entity ids with readable labels, longest first to avoid partial replacements.
        for entity_id in sorted(mapping.keys(), key=len, reverse=True):
            if not entity_id:
                continue
            out = re.sub(rf"\b{re.escape(entity_id)}\b", mapping[entity_id], out)
        # Make slot=value questions easier to read.
        out = re.sub(
            r"Is\s+([^?']+)\s+'([A-Za-z0-9_ -]+)=([A-Za-z0-9_ -]+)'\?",
            r"Is \1 with \2 = \3?",
            out,
            flags=re.IGNORECASE,
        )
        return out

    def _probe_target_summary(self, row: Dict[str, object]) -> str:
        if not isinstance(row, dict):
            return "[unknown]"
        mapping = self._friendly_entity_display_map()
        node_ids = [str(x or "").strip() for x in list(row.get("evidence_node_ids") or []) if str(x or "").strip()]
        edge_ids = [str(x or "").strip() for x in list(row.get("evidence_edge_ids") or []) if str(x or "").strip()]
        if len(node_ids) >= 2:
            left = str(mapping.get(node_ids[0], node_ids[0]))
            right = str(mapping.get(node_ids[1], node_ids[1]))
            relation = ""
            if edge_ids:
                edge_lookup = {
                    str(edge.get("edge_id", "") or ""): dict(edge)
                    for edge in list((self.current_graph or {}).get("edges") or [])
                    if isinstance(edge, dict)
                }
                relation = str(dict(edge_lookup.get(edge_ids[0]) or {}).get("relation", "") or "").strip()
            if relation:
                return f"[{left} -> {relation} -> {right}]"
            return f"[{left} <-> {right}]"
        if len(node_ids) == 1:
            one = str(mapping.get(node_ids[0], node_ids[0]))
            return f"[{one}]"
        if edge_ids:
            return f"[edge:{edge_ids[0]}]"
        claim_id = str(row.get("target_claim_id") or row.get("claim_id") or "").strip()
        if claim_id:
            return f"[{claim_id}]"
        return "[unlinked]"

    def _build_probe_list_item(self, row: Dict[str, object], *, is_multi: bool) -> QListWidgetItem:
        payload = dict(row or {})
        target = self._probe_target_summary(payload)
        question = self._humanize_question_text(str(payload.get("question", "") or ""))
        resp = self._probe_response_payload(payload)
        invalid = self._probe_is_invalid(payload, resp)
        stale = self._probe_is_stale(payload, resp)
        resolved = self._is_probe_manually_resolved(payload)
        answer = str(resp.get("answer") or resp.get("selection") or "uncertain").strip() or "uncertain"
        score = -1.0
        try:
            raw_score = resp.get("score", payload.get("score", None))
            if raw_score is not None:
                score = float(raw_score)
        except Exception:
            score = -1.0
        schema_ok = bool(resp.get("schema_valid", payload.get("schema_valid", True)))
        low = (not invalid) and (not stale) and (not resolved) and score >= 0.0 and self._is_low_confidence_probe(answer, score)
        if invalid:
            answer = "⚠ Invalid Response"
        if resolved:
            answer = "Resolved"
        status_tags = []
        if resolved:
            status_tags.append("resolved")
        if invalid and not resolved:
            status_tags.append("Invalid")
        if low:
            status_tags.append("Low")
        schema_tag = "schema:OK" if schema_ok else "schema:BAD"
        score_tag = f"{score:.2f}" if score >= 0.0 else "N/A"
        header = target
        if is_multi:
            chain = str(payload.get("chain_id", "chain") or "chain")
            turn = int(payload.get("turn", 0) or 0)
            header = f"{target}  {chain} T{turn}"
        if status_tags:
            if resolved:
                leading = "[resolved]"
                tail_tags = [tag for tag in status_tags if tag != "resolved"]
                if tail_tags:
                    header = f"{leading} {header}  [{' | '.join(tail_tags)}]"
                else:
                    header = f"{leading} {header}"
            else:
                header = f"{header}  [{' | '.join(status_tags)}]"
        text = "\n".join(
            [
                header,
                f"Q: {question or '--'}",
                f"A: {answer}",
                f"Score: {score_tag}   {schema_tag}",
            ]
        )
        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, str(payload.get("probe_id", "") or ""))
        item.setSizeHint(item.sizeHint() + QSize(0, 18))
        if invalid and not resolved:
            item.setBackground(QColor(255, 230, 230))
        elif low:
            item.setBackground(QColor(255, 245, 204))
        return item

    @staticmethod
    def _probe_response_payload(row: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(row, dict):
            return {}
        parsed = row.get("parsed_response")
        if isinstance(parsed, dict) and parsed:
            return dict(parsed)
        legacy = row.get("response")
        if isinstance(legacy, dict):
            return dict(legacy)
        return {}

    @staticmethod
    def _normalize_probe_view_type(view_type: str) -> str:
        token = str(view_type or "").strip().lower()
        aliases = {
            "single_turn_vqa": "single_turn_vqa",
            "single_turn": "single_turn_vqa",
            "single_vqa": "single_turn_vqa",
            "single": "single_turn_vqa",
            "multi_turn_vqa": "multi_turn_vqa",
            "multi_turn": "multi_turn_vqa",
            "multi_vqa": "multi_turn_vqa",
            "multi": "multi_turn_vqa",
            "caption": "caption",
            "caption_vqa": "caption",
            "caption_probe": "caption",
            "video_captioning": "caption",
        }
        return aliases.get(token, token)

    def _probe_detection_confidence(self, row: Dict[str, object]) -> float:
        if not isinstance(row, dict):
            return -1.0
        graph = self.current_graph if isinstance(self.current_graph, dict) else {}
        node_map = {
            str(node.get("entity_id", "") or "").strip(): dict(node)
            for node in list(graph.get("nodes") or [])
            if isinstance(node, dict)
        }
        best = 0.0
        for node_id in [str(x or "").strip() for x in list(row.get("evidence_node_ids") or []) if str(x or "").strip()]:
            node = node_map.get(node_id)
            if not isinstance(node, dict):
                continue
            best = max(best, float(self._node_confidence(node)))
        if best <= 0.0:
            return -1.0
        return max(0.0, min(1.0, float(best)))

    @staticmethod
    def _probe_is_invalid(row: Dict[str, object], resp: Dict[str, object]) -> bool:
        schema_valid = bool(resp.get("schema_valid", row.get("schema_valid", True)))
        is_truncated = bool(resp.get("is_truncated", row.get("is_truncated", False)))
        is_valid_flag = resp.get("is_valid", row.get("is_valid", None))
        explicitly_invalid = bool(is_valid_flag is False)
        schema_invalid_and_not_salvaged = (not schema_valid) and (is_valid_flag is not True)
        return schema_invalid_and_not_salvaged or is_truncated or explicitly_invalid

    @staticmethod
    def _probe_is_stale(row: Dict[str, object], resp: Dict[str, object]) -> bool:
        return bool(resp.get("stale", row.get("stale", False)))

    @staticmethod
    def _probe_resolution_identity(row: Dict[str, object]) -> Dict[str, object]:
        if not isinstance(row, dict):
            return {
                "claim_id": "",
                "probe_id": "",
                "chain_id": "",
                "turn": 0,
                "question": "",
                "evidence_node_ids": [],
                "evidence_edge_ids": [],
            }
        claim_id = str(row.get("target_claim_id") or row.get("claim_id") or "").strip()
        probe_id = str(row.get("probe_id") or row.get("qid") or "").strip()
        chain_id = str(row.get("chain_id") or "").strip()
        question = " ".join(str(row.get("question", "") or "").strip().lower().split())
        try:
            turn = int(row.get("turn", 0) or 0)
        except Exception:
            turn = 0
        node_ids = sorted({str(x).strip() for x in list(row.get("evidence_node_ids") or []) if str(x).strip()})
        edge_ids = sorted({str(x).strip() for x in list(row.get("evidence_edge_ids") or []) if str(x).strip()})
        if claim_id.startswith("claim_rel_"):
            eid = claim_id[len("claim_rel_") :].strip()
            if eid and eid not in edge_ids:
                edge_ids.append(eid)
        resolved_key = str(row.get("resolved_key", "") or "").strip()
        if (not resolved_key) and (question or node_ids or edge_ids or (chain_id and int(turn) > 0)):
            parts = [
                claim_id,
                question,
                "|".join(node_ids),
                "|".join(edge_ids),
            ]
            if chain_id and int(turn) > 0:
                parts.append(chain_id)
                parts.append(str(int(turn)))
            resolved_key = "||".join(parts)
        return {
            "claim_id": claim_id,
            "probe_id": probe_id,
            "chain_id": chain_id,
            "turn": int(turn),
            "question": question,
            "evidence_node_ids": node_ids,
            "evidence_edge_ids": edge_ids,
            "resolved_key": resolved_key,
        }

    @classmethod
    def _probe_resolution_match(cls, left: Dict[str, object], right: Dict[str, object]) -> bool:
        a = cls._probe_resolution_identity(dict(left or {}))
        b = cls._probe_resolution_identity(dict(right or {}))
        a_probe = str(a.get("probe_id", "") or "").strip()
        b_probe = str(b.get("probe_id", "") or "").strip()
        if a_probe and b_probe:
            return a_probe == b_probe
        a_chain = str(a.get("chain_id", "") or "").strip()
        b_chain = str(b.get("chain_id", "") or "").strip()
        a_turn = int(a.get("turn", 0) or 0)
        b_turn = int(b.get("turn", 0) or 0)
        if a_chain and b_chain and a_turn > 0 and b_turn > 0:
            return a_chain == b_chain and a_turn == b_turn
        # Legacy fallback: only use claim-level matching when both sides
        # lack probe-level identity (probe_id / chain+turn).
        a_claim = str(a.get("claim_id", "") or "").strip()
        b_claim = str(b.get("claim_id", "") or "").strip()
        a_has_probe_level = bool(a_probe) or bool(a_chain and a_turn > 0)
        b_has_probe_level = bool(b_probe) or bool(b_chain and b_turn > 0)
        a_question = str(a.get("question", "") or "").strip()
        b_question = str(b.get("question", "") or "").strip()
        a_nodes = list(a.get("evidence_node_ids") or [])
        b_nodes = list(b.get("evidence_node_ids") or [])
        a_edges = list(a.get("evidence_edge_ids") or [])
        b_edges = list(b.get("evidence_edge_ids") or [])
        if a_claim and b_claim and a_question and b_question and a_question == b_question:
            if a_nodes == b_nodes and a_edges == b_edges:
                return a_claim == b_claim
        if a_claim and b_claim and (not a_has_probe_level) and (not b_has_probe_level):
            return a_claim == b_claim
        return False

    def _patch_probe_rows_resolved_state(
        self,
        rows: List[Dict[str, object]],
        *,
        target: Dict[str, object],
        ts: str,
        source: str,
    ) -> Tuple[List[Dict[str, object]], int]:
        patched: List[Dict[str, object]] = []
        changed = 0
        for raw in list(rows or []):
            item = dict(raw or {})
            if self._probe_resolution_match(target, item):
                item["resolved"] = True
                item["manually_resolved"] = True
                item["resolved_at"] = ts
                item["resolved_source"] = str(source or "manual_resolve")
                parsed = dict(item.get("parsed_response") or {}) if isinstance(item.get("parsed_response"), dict) else {}
                parsed["resolved"] = True
                parsed["manually_resolved"] = True
                parsed["resolved_at"] = ts
                parsed["resolved_source"] = str(source or "manual_resolve")
                parsed["ui_keep_visible_until_next_cycle"] = True
                item["parsed_response"] = parsed
                if isinstance(item.get("response"), dict):
                    legacy = dict(item.get("response") or {})
                    legacy["resolved"] = True
                    legacy["manually_resolved"] = True
                    legacy["resolved_at"] = ts
                    legacy["resolved_source"] = str(source or "manual_resolve")
                    legacy["ui_keep_visible_until_next_cycle"] = True
                    item["response"] = legacy
                item["ui_keep_visible_until_next_cycle"] = True
                changed += 1
            patched.append(item)
        return patched, changed

    def _resolved_claim_records(self) -> List[Dict[str, object]]:
        result = self.current_cycle_result if isinstance(self.current_cycle_result, dict) else {}
        rows = [dict(x) for x in list(result.get("resolved_claims") or []) if isinstance(x, dict)]
        out: List[Dict[str, object]] = []
        seen = set()
        for row in rows:
            ident = self._probe_resolution_identity(row)
            key = json.dumps(ident, sort_keys=True, ensure_ascii=True)
            if key in seen:
                continue
            seen.add(key)
            out.append(ident)
        return out

    @staticmethod
    def _probe_question_key(row: Dict[str, object]) -> str:
        if not isinstance(row, dict):
            return ""
        question = " ".join(str(row.get("question", "") or "").strip().lower().split())
        if not question:
            return ""
        claim_id = str(row.get("target_claim_id", row.get("claim_id", "")) or "").strip().lower()
        node_ids = ",".join(sorted(str(x or "").strip().lower() for x in list(row.get("evidence_node_ids") or []) if str(x or "").strip()))
        edge_ids = ",".join(sorted(str(x or "").strip().lower() for x in list(row.get("evidence_edge_ids") or []) if str(x or "").strip()))
        view_type = str(row.get("view_type", "") or "").strip().lower()
        return f"q::{view_type}::{claim_id}::{node_ids}::{edge_ids}::{question}"

    @staticmethod
    def _normalize_question_key(value: object) -> str:
        token = str(value or "").strip().lower()
        if not token:
            return ""
        if token.startswith("q::"):
            return " ".join(token.split())
        if "::" in token:
            _prefix, suffix = token.split("::", 1)
            token = suffix.strip().lower()
        return " ".join(token.split())

    def _suppressed_question_records(self) -> List[Dict[str, object]]:
        result = self.current_cycle_result if isinstance(self.current_cycle_result, dict) else {}
        rows = [dict(x) for x in list(result.get("suppressed_questions") or []) if isinstance(x, dict)]
        out: List[Dict[str, object]] = []
        seen = set()
        for row in rows:
            key = self._normalize_question_key(row.get("question_key", ""))
            if not key or key in seen:
                continue
            seen.add(key)
            normalized = dict(row)
            normalized["question_key"] = key
            out.append(normalized)
        return out

    def _is_probe_manually_resolved(self, row: Dict[str, object]) -> bool:
        if not isinstance(row, dict):
            return False
        resp = self._probe_response_payload(row)
        if (
            bool(row.get("ui_keep_visible_until_next_cycle", False))
            or bool(resp.get("ui_keep_visible_until_next_cycle", False))
            or bool(dict(row.get("parsed_response") or {}).get("ui_keep_visible_until_next_cycle", False))
        ):
            return False
        if (
            bool(row.get("resolved", False))
            or bool(resp.get("resolved", False))
            or bool(row.get("manually_resolved", False))
            or bool(resp.get("manually_resolved", False))
        ):
            return True
        qkey = self._probe_question_key(row)
        if qkey and any(str(rec.get("question_key", "") or "").strip() == qkey for rec in self._suppressed_question_records()):
            return True
        ident = self._probe_resolution_identity(row)
        for rec in self._resolved_claim_records():
            if str(ident.get("resolved_key", "") or "").strip() and str(rec.get("resolved_key", "") or "").strip():
                if str(ident.get("resolved_key", "") or "").strip() == str(rec.get("resolved_key", "") or "").strip():
                    return True
            if self._probe_resolution_match(ident, rec):
                return True
        return False

    def _persist_cycle_verification_overrides(self) -> None:
        if not isinstance(self.current_cycle_result, dict):
            return
        if not isinstance(self.current_graph, dict):
            return
        meta = dict(self.current_graph.get("metadata") or {})
        cycle_verification = dict(meta.get("cycle_verification") or {})
        claims_payload = self.current_cycle_result.get("claims")
        if isinstance(claims_payload, dict):
            cycle_verification["claims"] = dict(claims_payload)
        elif isinstance(claims_payload, list):
            cycle_verification["claims"] = [
                dict(x) if isinstance(x, dict) else x
                for x in list(claims_payload or [])
            ]
        cycle_verification["probe_results"] = [
            dict(x) for x in list(self.current_cycle_result.get("probe_results") or []) if isinstance(x, dict)
        ]
        cycle_verification["resolved_claims"] = [
            dict(x) for x in list(self.current_cycle_result.get("resolved_claims") or []) if isinstance(x, dict)
        ]
        cycle_verification["suppressed_questions"] = [
            dict(x) for x in list(self.current_cycle_result.get("suppressed_questions") or []) if isinstance(x, dict)
        ]
        meta["cycle_verification"] = cycle_verification
        self.current_graph["metadata"] = meta
        self._replace_current_graph_in_bundle(self.current_graph)
        self._persist_current_scene_graph_bundle("scene_graph_cycle_mark_resolved_saved")

    @classmethod
    def _probe_exact_identity_match(cls, left: Dict[str, object], right: Dict[str, object]) -> bool:
        a = cls._probe_resolution_identity(dict(left or {}))
        b = cls._probe_resolution_identity(dict(right or {}))
        a_probe = str(a.get("probe_id", "") or "").strip()
        b_probe = str(b.get("probe_id", "") or "").strip()
        if a_probe and b_probe:
            return a_probe == b_probe
        a_chain = str(a.get("chain_id", "") or "").strip()
        b_chain = str(b.get("chain_id", "") or "").strip()
        a_turn = int(a.get("turn", 0) or 0)
        b_turn = int(b.get("turn", 0) or 0)
        if a_chain and b_chain and a_turn > 0 and b_turn > 0:
            return a_chain == b_chain and a_turn == b_turn
        return False

    def _focus_probe_in_views(
        self,
        target: Dict[str, object],
        *,
        is_multi: bool,
        preferred_table_row: Optional[int] = None,
        preferred_list_row: Optional[int] = None,
    ) -> None:
        if is_multi:
            rows = list(getattr(self, "_multi_claim_table_rows", []) or [])
            table = getattr(self, "multi_claims_table", None)
            list_widget = getattr(self, "multi_list", None)
            list_rows = list(self.multi_turn_items or [])
            render = self._render_multi_detail
        else:
            rows = list(getattr(self, "_single_claim_table_rows", []) or [])
            table = getattr(self, "single_claims_table", None)
            list_widget = getattr(self, "single_list", None)
            list_rows = list(self.single_turn_items or [])
            render = self._render_single_detail
        table_idx = -1
        if preferred_table_row is not None and 0 <= int(preferred_table_row) < len(rows):
            preferred_row = dict(rows[int(preferred_table_row)] or {})
            if self._probe_exact_identity_match(target, preferred_row):
                table_idx = int(preferred_table_row)
        if table_idx < 0:
            for idx, row in enumerate(rows):
                if self._probe_exact_identity_match(target, row):
                    table_idx = idx
                    break
        if table_idx < 0:
            for idx, row in enumerate(rows):
                if self._probe_resolution_match(target, row):
                    table_idx = idx
                    break
        if table_idx >= 0 and isinstance(table, QTableWidget):
            table.setCurrentCell(table_idx, 0)
            table.selectRow(table_idx)
        list_idx = -1
        if preferred_list_row is not None and 0 <= int(preferred_list_row) < len(list_rows):
            preferred_row = dict(list_rows[int(preferred_list_row)] or {})
            if self._probe_exact_identity_match(target, preferred_row):
                list_idx = int(preferred_list_row)
        if list_idx < 0:
            for idx, row in enumerate(list_rows):
                if self._probe_exact_identity_match(target, row):
                    list_idx = idx
                    break
        if list_idx < 0:
            for idx, row in enumerate(list_rows):
                if self._probe_resolution_match(target, row):
                    list_idx = idx
                    break
        if isinstance(list_widget, QListWidget) and list_idx >= 0:
            list_widget.setCurrentRow(list_idx)
            render(list_idx)
        else:
            render(0 if list_rows else -1)

    def _mark_probe_as_resolved(self, row: Dict[str, object], *, source: str = "manual_resolve") -> int:
        if not isinstance(self.current_cycle_result, dict):
            return 0
        target = self._probe_resolution_identity(row)
        preferred_single_table_row = int(self.single_claims_table.currentRow()) if isinstance(getattr(self, "single_claims_table", None), QTableWidget) else -1
        preferred_multi_table_row = int(self.multi_claims_table.currentRow()) if isinstance(getattr(self, "multi_claims_table", None), QTableWidget) else -1
        preferred_single_list_row = int(self.single_list.currentRow()) if isinstance(getattr(self, "single_list", None), QListWidget) else -1
        preferred_multi_list_row = int(self.multi_list.currentRow()) if isinstance(getattr(self, "multi_list", None), QListWidget) else -1
        if not any([target.get("claim_id"), target.get("probe_id"), target.get("chain_id"), target.get("evidence_node_ids")]):
            return 0
        rows = [dict(x) for x in list(self.current_cycle_result.get("probe_results") or []) if isinstance(x, dict)]
        changed = 0
        ts = _now_iso_utc()
        rows, changed = self._patch_probe_rows_resolved_state(
            rows,
            target=target,
            ts=ts,
            source=str(source or "manual_resolve"),
        )
        self.single_turn_items, _single_changed = self._patch_probe_rows_resolved_state(
            [dict(x) for x in list(self.single_turn_items or []) if isinstance(x, dict)],
            target=target,
            ts=ts,
            source=str(source or "manual_resolve"),
        )
        self.multi_turn_items, _multi_changed = self._patch_probe_rows_resolved_state(
            [dict(x) for x in list(self.multi_turn_items or []) if isinstance(x, dict)],
            target=target,
            ts=ts,
            source=str(source or "manual_resolve"),
        )
        target_claim_id = str(target.get("claim_id", "") or "").strip()
        if target_claim_id:
            claims_payload = self.current_cycle_result.get("claims")
            if isinstance(claims_payload, dict) and isinstance(claims_payload.get(target_claim_id), dict):
                claim_row = dict(claims_payload.get(target_claim_id) or {})
                claim_row["resolved"] = True
                claim_row["resolved_at"] = ts
                claim_row["resolved_source"] = str(source or "manual_resolve")
                claims_payload[target_claim_id] = claim_row
                self.current_cycle_result["claims"] = claims_payload
            elif isinstance(claims_payload, list):
                patched_claims = []
                for raw in list(claims_payload or []):
                    row_map = dict(raw or {}) if isinstance(raw, dict) else {}
                    claim_id = str(row_map.get("claim_id", "") or "").strip()
                    if claim_id == target_claim_id:
                        row_map["resolved"] = True
                        row_map["resolved_at"] = ts
                        row_map["resolved_source"] = str(source or "manual_resolve")
                    patched_claims.append(row_map)
                self.current_cycle_result["claims"] = patched_claims
        if changed <= 0:
            return 0
        resolved_records = self._resolved_claim_records()
        target_record = dict(target)
        target_record["resolved"] = True
        target_record["resolved_at"] = ts
        target_record["resolved_source"] = str(source or "manual_resolve")
        if not any(self._probe_resolution_match(target_record, rec) for rec in resolved_records):
            resolved_records.append(target_record)
        suppressed_questions = self._suppressed_question_records()
        qkey = self._probe_question_key(row)
        if qkey and not any(str(rec.get("question_key", "") or "").strip() == qkey for rec in suppressed_questions):
            suppressed_questions.append(
                {
                    "question_key": qkey,
                    "question": str(row.get("question", "") or "").strip(),
                    "view_type": str(row.get("view_type", "") or "").strip(),
                    "target_claim_id": str(row.get("target_claim_id", row.get("claim_id", "")) or "").strip(),
                    "claim_id": str(row.get("claim_id", "") or "").strip(),
                    "evidence_node_ids": [str(x or "").strip() for x in list(row.get("evidence_node_ids") or []) if str(x or "").strip()],
                    "evidence_edge_ids": [str(x or "").strip() for x in list(row.get("evidence_edge_ids") or []) if str(x or "").strip()],
                    "resolved_at": ts,
                    "resolved_source": str(source or "manual_resolve"),
                }
            )
        self.current_cycle_result["probe_results"] = rows
        self.current_cycle_result["resolved_claims"] = resolved_records
        self.current_cycle_result["suppressed_questions"] = suppressed_questions
        self._persist_cycle_verification_overrides()
        self._render_cycle_probe_outputs(list(rows))
        self._refresh_claim_verification_tables()
        self._refresh_human_arbitration_view()
        self._focus_probe_in_views(
            target,
            is_multi=bool(target.get("chain_id")),
            preferred_table_row=preferred_multi_table_row if bool(target.get("chain_id")) else preferred_single_table_row,
            preferred_list_row=preferred_multi_list_row if bool(target.get("chain_id")) else preferred_single_list_row,
        )
        return changed

    def _selected_probe_rows_from_panel(self, *, is_multi: bool) -> List[Dict[str, object]]:
        if is_multi:
            table = getattr(self, "multi_claims_table", None)
            rows = list(getattr(self, "_multi_claim_table_rows", []) or [])
        else:
            table = getattr(self, "single_claims_table", None)
            rows = list(getattr(self, "_single_claim_table_rows", []) or [])
        out: List[Dict[str, object]] = []
        if isinstance(table, QTableWidget):
            sel = table.selectionModel()
            if sel is not None:
                for model_idx in list(sel.selectedRows() or []):
                    try:
                        row_idx = int(model_idx.row())
                    except Exception:
                        continue
                    if 0 <= row_idx < len(rows):
                        out.append(dict(rows[row_idx] or {}))
        if not out:
            one = self._selected_probe_row_from_panel(is_multi=is_multi)
            if one:
                out.append(dict(one))
        deduped: List[Dict[str, object]] = []
        seen = set()
        for row in out:
            ident = self._probe_resolution_identity(row)
            key = json.dumps(ident, sort_keys=True, ensure_ascii=True)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(dict(row))
        return deduped

    def _selected_probe_row_from_panel(self, *, is_multi: bool) -> Dict[str, object]:
        if is_multi:
            table = getattr(self, "multi_claims_table", None)
            rows = list(getattr(self, "_multi_claim_table_rows", []) or [])
            list_widget = getattr(self, "multi_list", None)
            list_rows = list(self.multi_turn_items or [])
        else:
            table = getattr(self, "single_claims_table", None)
            rows = list(getattr(self, "_single_claim_table_rows", []) or [])
            list_widget = getattr(self, "single_list", None)
            list_rows = list(self.single_turn_items or [])
        if isinstance(table, QTableWidget):
            idx = int(table.currentRow())
            if 0 <= idx < len(rows):
                return dict(rows[idx] or {})
        if isinstance(list_widget, QListWidget):
            idx = int(list_widget.currentRow())
            if 0 <= idx < len(list_rows):
                return dict(list_rows[idx] or {})
        return {}

    def _mark_probe_rows_resolved(self, rows: List[Dict[str, object]], *, is_multi: bool) -> None:
        if not rows:
            self._set_status(
                "Select one or more multi-turn QA rows first." if is_multi else "Select one or more single-turn QA rows first.",
                status_type="warning",
            )
            return
        changed = 0
        for row in rows:
            changed += int(
                self._mark_probe_as_resolved(
                    row,
                    source="multi_turn_ui" if is_multi else "single_turn_ui",
                ) or 0
            )
        if changed <= 0:
            self._set_status(
                "Selected multi-turn QA item(s) could not be marked resolved."
                if is_multi
                else "Selected single-turn QA item(s) could not be marked resolved.",
                status_type="warning",
            )
            return
        self._set_status(
            f"Marked {changed} multi-turn QA item(s) as resolved."
            if is_multi
            else f"Marked {changed} single-turn QA item(s) as resolved.",
            status_type="success",
        )

    def _show_probe_context_menu(self, pos, *, is_multi: bool, source: str = "list") -> None:
        if is_multi:
            list_widget = getattr(self, "multi_list", None)
            table = getattr(self, "multi_claims_table", None)
            table_rows = list(getattr(self, "_multi_claim_table_rows", []) or [])
            list_rows = list(self.multi_turn_items or [])
        else:
            list_widget = getattr(self, "single_list", None)
            table = getattr(self, "single_claims_table", None)
            table_rows = list(getattr(self, "_single_claim_table_rows", []) or [])
            list_rows = list(self.single_turn_items or [])
        anchor = None
        context_rows: List[Dict[str, object]] = []
        if source == "table" and isinstance(table, QTableWidget):
            item = table.itemAt(pos)
            if item is not None:
                row = int(item.row())
                if row >= 0:
                    selected_rows = set()
                    sel = table.selectionModel()
                    if sel is not None:
                        for idx in list(sel.selectedRows() or []):
                            try:
                                selected_rows.add(int(idx.row()))
                            except Exception:
                                continue
                    if row not in selected_rows:
                        table.setCurrentCell(row, 0)
                        table.selectRow(row)
                        selected_rows = {row}
                    for row_idx in sorted(selected_rows):
                        if 0 <= row_idx < len(table_rows):
                            context_rows.append(dict(table_rows[row_idx] or {}))
            anchor = table.viewport().mapToGlobal(pos)
        elif isinstance(list_widget, QListWidget):
            item = list_widget.itemAt(pos)
            if item is not None:
                row = int(list_widget.row(item))
                if row >= 0:
                    selected_rows = {int(idx.row()) for idx in list_widget.selectedIndexes() if idx.isValid()}
                    if row not in selected_rows:
                        list_widget.setCurrentRow(row)
                        selected_rows = {row}
                    for row_idx in sorted(selected_rows):
                        if 0 <= row_idx < len(list_rows):
                            context_rows.append(dict(list_rows[row_idx] or {}))
            anchor = list_widget.viewport().mapToGlobal(pos)
        if not context_rows:
            fallback_rows = self._selected_probe_rows_from_panel(is_multi=is_multi)
            context_rows = [dict(row) for row in list(fallback_rows or []) if isinstance(row, dict)]
        if not context_rows or anchor is None:
            return
        menu = QMenu(self)
        action = menu.addAction("Mark as Resolved")
        unresolved_rows = [row for row in context_rows if not self._is_probe_manually_resolved(row)]
        if not unresolved_rows:
            action.setEnabled(False)
        else:
            action.triggered.connect(lambda: self._mark_probe_rows_resolved(unresolved_rows, is_multi=is_multi))
        menu.exec_(anchor)

    def _mark_selected_single_probe_resolved(self) -> None:
        self._mark_probe_rows_resolved(self._selected_probe_rows_from_panel(is_multi=False), is_multi=False)

    def _mark_selected_multi_probe_resolved(self) -> None:
        self._mark_probe_rows_resolved(self._selected_probe_rows_from_panel(is_multi=True), is_multi=True)

    @staticmethod
    def _is_low_confidence_probe(_answer: str, score: float) -> bool:
        return float(score) < float(LOW_CONF_THRESHOLD)

    def _extract_vqa_answer_fields(self, item: Dict[str, object]) -> Tuple[str, float, str]:
        row = dict(item or {})
        resp = self._probe_response_payload(row)
        if self._is_probe_manually_resolved(row):
            return "Resolved", -1.0, "manual_resolved"
        if self._probe_is_invalid(row, resp):
            reason = str(row.get("reason") or resp.get("reason") or "invalid_response").strip() or "invalid_response"
            return "⚠ Invalid Response", -1.0, reason
        is_stale = self._probe_is_stale(row, resp)
        answer = str(
            row.get("answer")
            or row.get("selection")
            or resp.get("answer")
            or resp.get("selection")
            or "uncertain"
        ).strip()
        try:
            score = float(row.get("score", resp.get("score", 0.0)) or 0.0)
        except Exception:
            score = 0.0
        reason = str(row.get("reason") or resp.get("reason") or "").strip()
        if not reason:
            reason = str(resp.get("raw_text") or row.get("raw_text") or "").strip()
        if (not is_stale) and answer.strip().lower() == "uncertain" and score <= 0.05:
            answer = "uncertain (low confidence)"
        return answer, max(0.0, min(1.0, score)), reason

    def _cycle_debug_enabled(self) -> bool:
        sg = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        return bool(sg.get("cycle_debug_mode", False))

    def _cycle_vote_summary_by_claim(self) -> Dict[str, Dict[str, float]]:
        result = self.current_cycle_result if isinstance(self.current_cycle_result, dict) else {}
        votes = [dict(x) for x in list(result.get("votes") or []) if isinstance(x, dict)]
        out: Dict[str, Dict[str, float]] = {}
        for row in votes:
            claim_id = str(row.get("claim_id", "") or "").strip()
            if not claim_id:
                continue
            bucket = out.setdefault(claim_id, {"support": 0.0, "conflict": 0.0, "uncertain": 0.0, "weight": 0.0})
            vote_type = str(row.get("vote", "uncertain") or "uncertain").strip().lower()
            try:
                score = float(row.get("score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            try:
                weight = float(row.get("weight", 1.0) or 1.0)
            except Exception:
                weight = 1.0
            if vote_type not in {"support", "conflict", "uncertain"}:
                vote_type = "uncertain"
            bucket[vote_type] = float(bucket.get(vote_type, 0.0) or 0.0) + max(0.0, score)
            bucket["weight"] = float(bucket.get("weight", 0.0) or 0.0) + max(0.0, weight)
        return out

    def _node_vote_hint(self, node: Dict[str, object], vote_map: Dict[str, Dict[str, float]]) -> str:
        nid = str(node.get("entity_id", "") or "").strip()
        if not nid:
            return ""
        claim_ids = [f"claim_exists_{nid}", f"claim_label_{nid}"]
        for att in list(node.get("attributes") or []):
            if not isinstance(att, dict):
                continue
            slot = str(att.get("slot", "") or "").strip().lower()
            if slot:
                claim_ids.append(f"claim_attr_{nid}_{re.sub(r'[^a-z0-9]+', '_', slot).strip('_')}")
        support = 0.0
        conflict = 0.0
        uncertain = 0.0
        total_weight = 0.0
        for claim_id in claim_ids:
            row = dict(vote_map.get(claim_id) or {})
            support += float(row.get("support", 0.0) or 0.0)
            conflict += float(row.get("conflict", 0.0) or 0.0)
            uncertain += float(row.get("uncertain", 0.0) or 0.0)
            total_weight += float(row.get("weight", 0.0) or 0.0)
        correction = ""
        result = self.current_cycle_result if isinstance(self.current_cycle_result, dict) else {}
        corr = dict(result.get("correction_candidates") or {})
        label_claim = f"claim_label_{nid}"
        if isinstance(corr.get(label_claim), dict):
            correction = str(dict(corr.get(label_claim) or {}).get("best_value", "") or "").strip()
        vote_text = f"S:{support:.2f} C:{conflict:.2f} U:{uncertain:.2f} W:{total_weight:.2f}"
        if correction:
            vote_text += f" | fix:{correction}"
        return vote_text

    def _edge_vote_hint(self, edge: Dict[str, object], vote_map: Dict[str, Dict[str, float]]) -> str:
        edge_id = str(edge.get("edge_id", "") or "").strip()
        if not edge_id:
            return ""
        claim_id = f"claim_rel_{edge_id}"
        row = dict(vote_map.get(claim_id) or {})
        support = float(row.get("support", 0.0) or 0.0)
        conflict = float(row.get("conflict", 0.0) or 0.0)
        uncertain = float(row.get("uncertain", 0.0) or 0.0)
        weight = float(row.get("weight", 0.0) or 0.0)
        correction = ""
        result = self.current_cycle_result if isinstance(self.current_cycle_result, dict) else {}
        corr = dict(result.get("correction_candidates") or {})
        if isinstance(corr.get(claim_id), dict):
            correction = str(dict(corr.get(claim_id) or {}).get("best_value", "") or "").strip()
        vote_text = f"S:{support:.2f} C:{conflict:.2f} U:{uncertain:.2f} W:{weight:.2f}"
        if correction:
            vote_text += f" | fix:{correction}"
        return vote_text

    def _render_cycle_caption_feedback(self) -> None:
        if not hasattr(self, "caption_output") or self.caption_output is None:
            return
        result = self.current_cycle_result if isinstance(self.current_cycle_result, dict) else {}
        caption_payload = dict(result.get("caption") or {})
        caption_text = str(
            caption_payload.get("caption_text", "")
            or caption_payload.get("caption", "")
            or ""
        ).strip()
        raw_text = str(caption_payload.get("raw_text", "") or "").strip()
        raw_response = caption_payload.get("raw_response")
        is_truncated = bool(caption_payload.get("is_truncated"))
        schema_valid = bool(caption_payload.get("schema_valid"))
        parse_mode = str(caption_payload.get("parse_mode", "") or "").strip()
        invalid_reason = str(caption_payload.get("invalid_reason", "") or "").strip()
        if is_truncated:
            parts = [
                "[Caption Status]",
                f"truncated={int(is_truncated)} schema_valid={int(schema_valid)} parse_mode={parse_mode or 'unknown'} invalid_reason={invalid_reason or '-'}",
            ]
            if caption_text:
                parts.extend(["", "[Parsed Caption]", caption_text])
            if raw_text and raw_text != caption_text:
                parts.extend(["", "[Raw Text]", raw_text])
            elif raw_text and not caption_text:
                parts.extend(["", "[Raw Text]", raw_text])
            display_text = "\n".join(parts).strip()
        elif raw_text:
            display_text = raw_text
        elif isinstance(raw_response, dict) and raw_response:
            display_text = json.dumps(raw_response, ensure_ascii=False, indent=2)
        else:
            display_text = caption_text
        self.caption_output.setPlainText(display_text)

    def _queue_node_confidence(self, node_id: str) -> float:
        nid = str(node_id or "").strip()
        if not nid:
            return 0.0
        graph = self.current_graph if isinstance(self.current_graph, dict) else {}
        for node in list(graph.get("nodes") or []):
            if not isinstance(node, dict):
                continue
            if str(node.get("entity_id", "") or "").strip() != nid:
                continue
            return float(self._node_confidence(node))
        return 0.0

    def _refresh_human_arbitration_view(self) -> None:
        table = getattr(self, "sg_human_queue_table", None)
        detail = getattr(self, "sg_human_queue_detail", None)
        if not isinstance(table, QTableWidget):
            return
        display_map = self._friendly_entity_display_map()
        result = self.current_cycle_result if isinstance(self.current_cycle_result, dict) else {}
        rows_all = [dict(x) for x in list(result.get("human_queue") or []) if isinstance(x, dict)]
        # UI policy: only show unresolved items in Human Edit Queue.
        rows = [
            row
            for row in rows_all
            if str(row.get("status", "pending") or "pending").strip().lower() not in {"edited", "skipped"}
        ]
        rows.sort(key=lambda row: float(row.get("priority", 0.0) or 0.0), reverse=True)
        self._human_queue_rows = rows
        table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            suggestion = str(row.get("suggested_value", "") or row.get("value", "") or "").strip()
            question = str(row.get("question", "") or "").strip()
            status = str(row.get("status", "pending") or "pending").strip()
            frame_idx = self._human_queue_row_frame_idx(row)
            node_ids = self._human_queue_row_node_ids(row)
            node_tokens = [
                f"{str(display_map.get(nid, nid) or nid)}({self._queue_node_confidence(nid):.2f})"
                for nid in node_ids
            ]
            values = (
                f"{float(row.get('priority', 0.0) or 0.0):.2f}",
                str(frame_idx if frame_idx is not None else "-"),
                str(row.get("claim_id", "") or ""),
                str(row.get("claim_type", row.get("item_type", "")) or ""),
                ", ".join(node_tokens[:4]) + (" ..." if len(node_tokens) > 4 else ""),
                self._humanize_question_text(question),
                suggestion,
                status,
            )
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                table.setItem(i, col, item)
        table.resizeRowsToContents()
        if rows:
            table.setCurrentCell(0, 0)
        elif isinstance(detail, QPlainTextEdit):
            detail.setPlainText("No pending arbitration items.")

    def _human_queue_row_frame_idx(self, row: Dict[str, object]) -> Optional[int]:
        if not isinstance(row, dict):
            return None
        for key in ("frame_idx", "graph_frame_idx"):
            try:
                if key in row and int(row.get(key)) >= 0:
                    return int(row.get(key))
            except Exception:
                pass
        claim_row = dict(row.get("claim_row") or {})
        for key in ("frame_idx", "graph_frame_idx"):
            try:
                if key in claim_row and int(claim_row.get(key)) >= 0:
                    return int(claim_row.get(key))
            except Exception:
                pass
        if isinstance(self.current_graph, dict):
            return self._extract_graph_frame_idx(self.current_graph)
        return None

    @staticmethod
    def _human_queue_row_node_ids(row: Dict[str, object]) -> List[str]:
        if not isinstance(row, dict):
            return []
        out: List[str] = []
        for key in ("evidence_node_ids",):
            for item in list(row.get(key) or []):
                nid = str(item or "").strip()
                if nid and nid not in out:
                    out.append(nid)
        claim_row = dict(row.get("claim_row") or {})
        for item in list(claim_row.get("evidence_node_ids") or []):
            nid = str(item or "").strip()
            if nid and nid not in out:
                out.append(nid)
        for key in ("subject_id", "object_id", "target_node_id"):
            nid = str(claim_row.get(key, "") or "").strip()
            if nid and nid not in out:
                out.append(nid)
        for key in ("subject_id", "object_id", "target_node_id"):
            nid = str(row.get(key, "") or "").strip()
            if nid and nid not in out:
                out.append(nid)
        return out

    @staticmethod
    def _human_queue_row_edge_ids(row: Dict[str, object]) -> List[str]:
        if not isinstance(row, dict):
            return []
        out: List[str] = []
        for item in list(row.get("evidence_edge_ids") or []):
            eid = str(item or "").strip()
            if eid and eid not in out:
                out.append(eid)
        claim_row = dict(row.get("claim_row") or {})
        for item in list(claim_row.get("evidence_edge_ids") or []):
            eid = str(item or "").strip()
            if eid and eid not in out:
                out.append(eid)
        claim_id = str(row.get("claim_id", "") or "").strip()
        if claim_id.startswith("claim_rel_"):
            eid = claim_id[len("claim_rel_") :].strip()
            if eid and eid not in out:
                out.append(eid)
        source_edge = str(row.get("source_relation_edge_id", "") or "").strip()
        if source_edge and source_edge not in out:
            out.append(source_edge)
        return out

    def _focus_human_queue_row(self, row: Dict[str, object]) -> None:
        if not isinstance(row, dict):
            return
        frame_idx = self._human_queue_row_frame_idx(row)
        if frame_idx is not None and frame_idx >= 0:
            self._seek_frame(int(frame_idx))
            self._set_graph_frame_selector(int(frame_idx), manual=True)

        node_ids = self._human_queue_row_node_ids(row)
        subject_id = str(row.get("subject_id", "") or "").strip()
        object_id = str(row.get("object_id", "") or "").strip()
        predicate = str(row.get("predicate", "") or "").strip()
        edge_ids = self._human_queue_row_edge_ids(row)

        self._sync_graph_selection = True
        try:
            self._reset_table_bg(self.sg_nodes_table)
            self._reset_table_bg(self.sg_edges_table)
            self._apply_stage_warning_badges_to_tables()

            selected_node_rows: List[int] = []
            for nid in node_ids:
                row_idx = int(self._node_row_by_id.get(nid, -1))
                if row_idx >= 0 and row_idx not in selected_node_rows:
                    selected_node_rows.append(row_idx)
            if not selected_node_rows:
                subj_row = int(self._node_row_by_id.get(subject_id, -1))
                obj_row = int(self._node_row_by_id.get(object_id, -1))
                if subj_row >= 0:
                    selected_node_rows.append(subj_row)
                if obj_row >= 0 and obj_row not in selected_node_rows:
                    selected_node_rows.append(obj_row)
            if selected_node_rows:
                self.sg_nodes_table.selectRow(int(selected_node_rows[0]))
                for i, row_idx in enumerate(selected_node_rows):
                    self._set_row_bg(
                        self.sg_nodes_table,
                        row_idx,
                        QColor(233, 245, 255) if i == 0 else QColor(245, 250, 255),
                    )

            edge_row_idx = -1
            for i, edge in enumerate(list(self._edge_rows or [])):
                if not isinstance(edge, dict):
                    continue
                edge_id = str(edge.get("edge_id", "") or "").strip()
                src = str(edge.get("src", "") or "").strip()
                dst = str(edge.get("dst", "") or "").strip()
                if edge_ids and edge_id in edge_ids:
                    edge_row_idx = i
                    break
                if subject_id and object_id and src == subject_id and dst == object_id:
                    if predicate:
                        try:
                            graph_edges = [dict(e) for e in list((self.current_graph or {}).get("edges") or []) if isinstance(e, dict)]
                        except Exception:
                            graph_edges = []
                        relation = ""
                        for ge in graph_edges:
                            if str(ge.get("edge_id", "") or "").strip() == edge_id:
                                relation = str(ge.get("relation", "") or "").strip()
                                break
                        if relation == predicate:
                            edge_row_idx = i
                            break
                    else:
                        edge_row_idx = i
                        break
            if edge_row_idx >= 0:
                self.sg_edges_table.selectRow(edge_row_idx)
                self._set_row_bg(self.sg_edges_table, edge_row_idx, QColor(233, 245, 255))

            self._apply_scene_graph_overlay_to_player()
            focus_node_id = ""
            if node_ids:
                focus_node_id = str(node_ids[0] or "").strip()
            if not focus_node_id:
                focus_node_id = str(subject_id or object_id or "").strip()
            self._render_object_probe_drawer(focus_node_id)
        finally:
            self._sync_graph_selection = False

    def _on_human_queue_selection_changed(self) -> None:
        table = getattr(self, "sg_human_queue_table", None)
        detail = getattr(self, "sg_human_queue_detail", None)
        if not isinstance(table, QTableWidget) or not isinstance(detail, QPlainTextEdit):
            return
        row_idx = int(table.currentRow())
        rows = list(getattr(self, "_human_queue_rows", []) or [])
        if row_idx < 0 or row_idx >= len(rows):
            detail.setPlainText("")
            return
        row = dict(rows[row_idx] or {})
        frame_idx = self._human_queue_row_frame_idx(row)
        node_ids = self._human_queue_row_node_ids(row)
        edge_ids = self._human_queue_row_edge_ids(row)
        display_map = self._friendly_entity_display_map()
        node_tokens = [str(display_map.get(nid, nid) or nid) for nid in node_ids]
        lines = [
            f"claim_id: {str(row.get('claim_id', '') or '-')}",
            f"type: {str(row.get('claim_type', row.get('item_type', '')) or '-')}",
            f"frame: {str(frame_idx) if frame_idx is not None else '-'}",
            f"nodes: {', '.join(node_tokens) if node_tokens else '-'}",
            f"edges: {', '.join(edge_ids) if edge_ids else '-'}",
            f"question: {self._humanize_question_text(str(row.get('question', '') or '-'))}",
            f"suggestion: {str(row.get('suggested_value', row.get('value', '')) or '-')}",
        ]
        detail.setPlainText("\n".join(lines))
        self._focus_human_queue_row(row)

    def _selected_human_queue_row(self) -> Tuple[int, List[Dict[str, object]]]:
        table = getattr(self, "sg_human_queue_table", None)
        if not isinstance(table, QTableWidget):
            return (-1, [])
        queue = [dict(x) for x in list((self.current_cycle_result or {}).get("human_queue") or []) if isinstance(x, dict)]
        queue.sort(key=lambda row: float(row.get("priority", 0.0) or 0.0), reverse=True)
        row_idx = int(table.currentRow())
        if row_idx < 0 or row_idx >= len(queue):
            return (-1, queue)
        return (row_idx, queue)

    def _human_queue_edge_id(self, row: Dict[str, object]) -> str:
        claim_id = str((row or {}).get("claim_id", "") or "").strip()
        if claim_id.startswith("claim_rel_"):
            return claim_id[len("claim_rel_") :].strip()
        return str((row or {}).get("source_relation_edge_id", "") or "").strip()

    def _apply_human_queue_suggestion(self) -> None:
        if not isinstance(self.current_cycle_result, dict):
            self._set_status("No cycle result loaded — run Cycle Verify first.", status_type="warning")
            return
        if not isinstance(self.current_graph, dict):
            self._set_status("No scene graph loaded.", status_type="warning")
            return
        row_idx, queue = self._selected_human_queue_row()
        if row_idx < 0:
            self._set_status("Select a row in the edit queue first.", status_type="warning")
            return
        target = dict(queue[row_idx] or {})
        suggestion = str(target.get("suggested_value", target.get("value", "")) or "").strip()
        claim_type = str(target.get("claim_type", target.get("item_type", "")) or "").strip().lower()
        claim_id = str(target.get("claim_id", "") or "").strip()

        # Queue item is guidance only: do not auto-edit graph fields from claim suggestions.
        # Human should fix nodes/edges/masks with the low-level editing tools, then mark edited.
        self._focus_human_queue_row(target)
        if claim_type == "bbox":
            msg = (
                f"Focused queue item {claim_id or '(unknown)'}."
                " Please use low-level bbox/mask tools to correct geometry, then click 'Mark Edited'."
            )
        elif claim_type in {"label", "relation"}:
            hint = f" suggested='{suggestion}'" if suggestion else ""
            msg = (
                f"Focused queue item {claim_id or '(unknown)'} ({claim_type})."
                f" Please edit node/edge directly with graph tools{hint}, then click 'Mark Edited'."
            )
        else:
            msg = (
                f"Focused queue item {claim_id or '(unknown)'}."
                " Please correct with graph editing tools, then click 'Mark Edited'."
            )
        self._set_status(msg, status_type="info")

    def _set_human_queue_status(
        self,
        status: str,
        *,
        row_override: Optional[int] = None,
        action: str = "manual_edit",
        skip_render: bool = False,
    ) -> None:
        status_norm = str(status or "").strip().lower()
        if status_norm not in {"edited", "skipped"}:
            return
        if not isinstance(self.current_cycle_result, dict):
            self._set_status("No cycle result loaded — run Cycle Verify first.", status_type="warning")
            return

        row_idx, queue = self._selected_human_queue_row()
        if row_override is not None:
            row_idx = int(row_override)
            queue = [dict(x) for x in list((self.current_cycle_result or {}).get("human_queue") or []) if isinstance(x, dict)]
            queue.sort(key=lambda row: float(row.get("priority", 0.0) or 0.0), reverse=True)
        if row_idx < 0:
            self._set_status("Select a row in the edit queue first.", status_type="warning")
            return
        if row_idx >= len(queue):
            return

        target = dict(queue[row_idx] or {})
        claim_id = str(target.get("claim_id", "") or "")
        target["status"] = status_norm
        target["human_decision"] = {
            "decision": status_norm,
            "action": str(action or "manual_edit"),
            "ts": _now_iso_utc(),
            "participant_id": str(self._common_settings.get("participant_id", "") or ""),
        }
        queue[row_idx] = target
        self.current_cycle_result["human_queue"] = queue

        if isinstance(self.current_graph, dict):
            meta = dict(self.current_graph.get("metadata") or {})
            cycle_update = dict(meta.get("cycle_update") or {})
            scene_graph_edits = [dict(x) for x in list(cycle_update.get("human_scene_graph_edits") or []) if isinstance(x, dict)]
            scene_graph_edits.append(
                {
                    "claim_id": claim_id,
                    "status": status_norm,
                    "action": str(action or "manual_edit"),
                    "ts": _now_iso_utc(),
                    "participant_id": str(self._common_settings.get("participant_id", "") or ""),
                }
            )
            cycle_update["human_scene_graph_edits"] = scene_graph_edits
            meta["cycle_update"] = cycle_update
            cycle_verification = dict(meta.get("cycle_verification") or {})
            cycle_verification["human_queue"] = [dict(x) for x in queue]
            meta["cycle_verification"] = cycle_verification
            self.current_graph["metadata"] = meta
            self._replace_current_graph_in_bundle(self.current_graph)

        self._append_oplog(
            "human_scene_graph_queue_status",
            claim_id=claim_id,
            status=status_norm,
            action=str(action or "manual_edit"),
            claim_type=str(target.get("claim_type", target.get("item_type", "")) or ""),
            priority=float(target.get("priority", 0.0) or 0.0),
            frame_idx=int(self.current_graph.get("frame_idx", -1)) if isinstance(self.current_graph, dict) else -1,
        )

        self._refresh_human_arbitration_view()
        if not skip_render:
            self._render_graph()
        self._set_status(f"Queue item {claim_id!r} marked as {status_norm}.", status_type="success")

    def _set_human_arbitration_decision(self, decision: str) -> None:
        # Backward compatibility with older button/action hooks.
        decision_norm = str(decision or "").strip().lower()
        if decision_norm in {"accepted", "edited", "resolved"}:
            self._set_human_queue_status("edited", action="legacy_decision")
        elif decision_norm in {"rejected", "deferred", "skipped"}:
            self._set_human_queue_status("skipped", action="legacy_decision")

    def _apply_cycle_result(self, payload: Dict[str, object]) -> None:
        result = dict(payload or {})
        if not list(result.get("probe_results") or []):
            rounds = [dict(x) for x in list(result.get("rounds") or []) if isinstance(x, dict)]
            last_round = rounds[-1] if rounds else {}
            if last_round:
                if "probe_results" in last_round:
                    result["probe_results"] = list(last_round.get("probe_results") or [])
                if "votes" in last_round and not list(result.get("votes") or []):
                    result["votes"] = list(last_round.get("votes") or [])
                if "claims" in last_round and (result.get("claims") is None or result.get("claims") == {}):
                    result["claims"] = last_round.get("claims")
                if "caption" in last_round and not dict(result.get("caption") or {}):
                    result["caption"] = dict(last_round.get("caption") or {})
        result["resolved_claims"] = [
            dict(x) for x in list(result.get("resolved_claims") or []) if isinstance(x, dict)
        ]
        result["suppressed_questions"] = [
            dict(x) for x in list(result.get("suppressed_questions") or []) if isinstance(x, dict)
        ]
        try:
            probe_rows = [dict(x) for x in list(result.get("probe_results") or []) if isinstance(x, dict)]
            single_cnt = len([1 for x in probe_rows if self._normalize_probe_view_type(str(x.get("view_type", "") or "")) == "single_turn_vqa"])
            multi_cnt = len([1 for x in probe_rows if self._normalize_probe_view_type(str(x.get("view_type", "") or "")) == "multi_turn_vqa"])
            caption_cnt = len([1 for x in probe_rows if self._normalize_probe_view_type(str(x.get("view_type", "") or "")) == "caption"])
            self._append_runtime_log(
                f"[CYCLE-UI] apply result probes={len(probe_rows)} single={single_cnt} multi={multi_cnt} caption_rows={caption_cnt}",
                level="info",
            )
        except Exception:
            pass
        before_graph = self._json_safe_clone(self.current_graph) if isinstance(self.current_graph, dict) else {}
        graph_after = dict(result.get("graph_after") or {})
        if graph_after:
            meta = dict(graph_after.get("metadata") or {})
            caption_payload = dict(result.get("caption") or {})
            debug_payload = dict(result.get("debug") or {})
            if not debug_payload:
                debug_payload = {
                    "request_prompt": caption_payload.get("request_prompt"),
                    "request_schema": caption_payload.get("request_schema"),
                    "raw_response": caption_payload.get("raw_response"),
                    "raw_text": caption_payload.get("raw_text"),
                }
            claims_payload = result.get("claims")
            if isinstance(claims_payload, dict):
                claims_value: object = dict(claims_payload)
            elif isinstance(claims_payload, list):
                claims_value = [dict(x) if isinstance(x, dict) else x for x in claims_payload]
            else:
                claims_value = {}
            meta["cycle_verification"] = {
                "claims": claims_value,
                "votes": list(result.get("votes") or []),
                "probe_results": list(result.get("probe_results") or []),
                "resolved_claims": list(result.get("resolved_claims") or []),
                "suppressed_questions": list(result.get("suppressed_questions") or []),
                "correction_candidates": dict(result.get("correction_candidates") or {}),
                "human_queue": list(result.get("human_queue") or []),
                "summary": dict(result.get("summary") or {}),
                "runtime": dict(result.get("runtime") or {}),
                "caption": caption_payload,
                "policy": dict(result.get("policy") or {}),
                "debug": debug_payload,
            }
            graph_after["metadata"] = meta
        if graph_after:
            self.current_graph = graph_after
            self._replace_current_graph_in_bundle(self.current_graph)
        self.current_cycle_result = result
        self._cycle_result_frame_idx = int(self._extract_graph_frame_idx(graph_after) or -1) if graph_after else int(self._extract_graph_frame_idx(self.current_graph or {}) or -1)
        self._render_cycle_probe_outputs(list(result.get("probe_results") or []))
        self._refresh_claim_verification_tables()
        self._refresh_human_arbitration_view()
        queue = list(result.get("human_queue") or [])
        runtime = dict(result.get("runtime") or {})
        verifier_provider = str(runtime.get("verifier_provider", "unknown") or "unknown")
        queued_count = self._record_cycle_human_queue(
            queue=queue,
            claims=dict(result.get("claims") or {}),
        )
        self._refresh_cycle_review_panel()
        self._refresh_cycle_memory_summary()
        self._refresh_cycle_summary()
        self._render_cycle_caption_feedback()
        if graph_after:
            self._render_graph()
            self._persist_current_scene_graph_bundle("scene_graph_cycle_verify_saved")
        if graph_after and graph_after != before_graph:
            self._record_change(
                task_type="scene_graph",
                item_id=str(graph_after.get("image_id", "scene_graph")),
                op="cycle_refine",
                field_path="graph",
                before=before_graph,
                after=graph_after,
                reason="cross-task cycle refinement",
            )
        if hasattr(self, "sg_cycle_summary") and isinstance(self.sg_cycle_summary, QLabel):
            summary = dict(result.get("summary") or {})
            rounds_run = int(summary.get("rounds_run", len(list(result.get("rounds") or []))) or 0)
            probe_count = int(summary.get("probe_count", len(list(result.get("probe_results") or []))) or 0)
            self.sg_cycle_summary.setText(
                f"Cycle verifier={verifier_provider} | rounds={rounds_run} | probes={probe_count} | queue={len(queue)}"
            )
        self._set_status(
            f"Cycle refine finished with verifier {verifier_provider}; queue={len(queue)}; logged={queued_count}.",
            status_type="success",
        )

    def _close_cycle_progress_dialog(self) -> None:
        dlg = self._cycle_progress_dialog
        self._cycle_progress_dialog = None
        self._cycle_progress_value = 0
        if dlg is not None:
            dlg.close()
            dlg.deleteLater()

    def _on_cycle_worker_thread_finished(self) -> None:
        self._cycle_worker = None
        self._cycle_worker_thread = None

    @staticmethod
    def _parse_cycle_stage_progress(msg: str) -> Dict[str, object]:
        text = str(msg or "").strip()
        out: Dict[str, object] = {"ok": False}
        if not text.startswith("[CYCLE-STAGE]"):
            return out
        m = re.match(
            r"^\[CYCLE-STAGE\]\s+(single_vqa|multi_vqa|caption)\s+(begin|progress|done)(?:\s+done=(\d+))?(?:\s+total=(\d+))?(?:\s+pct=(\d+))?",
            text,
            flags=re.IGNORECASE,
        )
        if not m:
            return out
        stage = str(m.group(1) or "").strip().lower()
        event = str(m.group(2) or "").strip().lower()
        try:
            done = int(m.group(3) or 0)
        except Exception:
            done = 0
        try:
            total = int(m.group(4) or 0)
        except Exception:
            total = 0
        try:
            pct = int(m.group(5) or 0)
        except Exception:
            pct = 0
        if event == "begin":
            pct = 0
        elif event == "done":
            pct = 100
        elif total > 0:
            pct = int(round(100.0 * float(done) / float(max(1, total))))
        pct = max(0, min(100, int(pct)))
        out.update(
            {
                "ok": True,
                "stage": stage,
                "event": event,
                "done": int(done),
                "total": int(total),
                "pct": int(pct),
            }
        )
        return out

    @staticmethod
    def _cycle_stage_display_name(stage: str) -> str:
        key = str(stage or "").strip().lower()
        if key == "single_vqa":
            return "Single VQA"
        if key == "multi_vqa":
            return "Multi VQA"
        if key == "caption":
            return "Video Caption"
        return "Cycle"

    def _estimate_cycle_progress_target(self, msg: str, current: int) -> int:
        text = str(msg or "").strip().lower()
        if not text:
            return current
        if "[cycle-progress]" in text:
            if "single_vqa start" in text:
                return max(current, 40)
            if "single_vqa done" in text or "single_vqa skip" in text:
                return max(current, 58)
            if "multi_vqa start" in text:
                return max(current, 62)
            if "multi_vqa done" in text or "multi_vqa skip" in text:
                return max(current, 78)
            if "caption start" in text:
                return max(current, 82)
            if "caption done" in text or "caption skip" in text:
                return max(current, 94)
            if "cycle done" in text:
                return max(current, 98)
            return current
        if "loading ontology" in text or "cycle configuration" in text:
            return max(current, 10)
        if "preparing cycle verifier" in text:
            return max(current, 25)
        if "running cross-task cycle refinement" in text:
            return max(current, 35)
        # Stage-local granular progress is handled by [CYCLE-STAGE] parser.
        if "[cycle-api][start]" in text or "[cycle-api][done]" in text:
            return current
        if "[cycle-api][http]" in text or "[cycle-api][urlerr]" in text or "[cycle-api][error]" in text:
            return current
        if "[cycle-check]" in text:
            return max(current, 98)
        return current

    def _on_cycle_worker_progress(self, text: str) -> None:
        msg = str(text or "").strip()
        if not msg:
            return
        stage_evt = self._parse_cycle_stage_progress(msg)
        if self._cycle_progress_dialog is not None:
            if bool(stage_evt.get("ok", False)):
                stage_name = self._cycle_stage_display_name(str(stage_evt.get("stage", "") or ""))
                pct = int(stage_evt.get("pct", 0) or 0)
                done = int(stage_evt.get("done", 0) or 0)
                total = int(stage_evt.get("total", 0) or 0)
                event = str(stage_evt.get("event", "") or "").strip().lower()
                if total > 0 and event == "progress":
                    label = f"{stage_name}: {pct}% ({done}/{total})"
                elif total > 0 and event == "begin":
                    label = f"{stage_name}: 0% (0/{total})"
                elif total <= 0 and event == "done":
                    label = f"{stage_name}: skipped"
                else:
                    label = f"{stage_name}: {pct}%"
                self._cycle_progress_dialog.setLabelText(label)
                self._cycle_progress_value = int(pct)
                self._cycle_progress_dialog.setValue(int(pct))
            else:
                self._cycle_progress_dialog.setLabelText(msg)
                current_val = int(self._cycle_progress_value or 0)
                target_val = self._estimate_cycle_progress_target(msg, current=current_val)
                if target_val != current_val:
                    self._cycle_progress_value = int(target_val)
                    self._cycle_progress_dialog.setValue(int(target_val))
        self._set_status(msg, status_type="info")
        self._append_runtime_log(msg, level="info")

    @staticmethod
    def _cycle_claim_vote_health(payload: Dict[str, object]) -> Dict[str, int]:
        result = dict(payload or {})
        claims_raw = result.get("claims") or {}
        if isinstance(claims_raw, dict):
            claim_count = len([1 for _k, v in claims_raw.items() if isinstance(v, dict)])
        elif isinstance(claims_raw, list):
            claim_count = len([1 for v in claims_raw if isinstance(v, dict)])
        else:
            claim_count = 0
        votes = [dict(v) for v in list(result.get("votes") or []) if isinstance(v, dict)]
        voted_claim_ids = {
            str(v.get("claim_id", "") or "").strip()
            for v in votes
            if str(v.get("claim_id", "") or "").strip()
        }
        probe_results = [dict(v) for v in list(result.get("probe_results") or []) if isinstance(v, dict)]
        human_queue = [dict(v) for v in list(result.get("human_queue") or []) if isinstance(v, dict)]
        online_probe_count = 0
        verifier_error_count = 0
        for row in probe_results:
            resp = VideoTaskStudio._probe_response_payload(row)
            provider = str(
                row.get("response_provider")
                or (dict(resp.get("raw_response") or {}).get("provider"))
                or resp.get("provider")
                or ""
            ).strip().lower()
            if provider == "gemini_online":
                online_probe_count += 1
            reason = str(resp.get("reason", "") or "").strip().lower()
            if reason.startswith("verifier_error") or bool(resp.get("error")):
                verifier_error_count += 1
        caption_votes = len([1 for row in votes if str(row.get("view_type", "") or "").strip() == "caption"])
        single_votes = len([1 for row in votes if str(row.get("view_type", "") or "").strip() == "single_turn_vqa"])
        multi_votes = len([1 for row in votes if str(row.get("view_type", "") or "").strip() == "multi_turn_vqa"])
        voted_claim_count = min(int(claim_count), len(voted_claim_ids)) if claim_count else len(voted_claim_ids)
        return {
            "claims": int(claim_count),
            "votes": int(len(votes)),
            "voted_claims": int(voted_claim_count),
            "unvoted_claims": max(0, int(claim_count) - int(voted_claim_count)),
            "single_votes": int(single_votes),
            "multi_votes": int(multi_votes),
            "caption_votes": int(caption_votes),
            "probes": int(len(probe_results)),
            "human_queue": int(len(human_queue)),
            "online_probes": int(online_probe_count),
            "verifier_errors": int(verifier_error_count),
        }

    def _on_cycle_worker_done(self, payload: Dict[str, object]) -> None:
        if self._cycle_progress_dialog is not None:
            self._cycle_progress_value = 100
            self._cycle_progress_dialog.setValue(100)
        self._close_cycle_progress_dialog()
        self._set_scene_graph_busy(False)
        snapshot_path = self._persist_cycle_result_snapshot(payload)
        self._apply_cycle_result(payload)
        health = self._cycle_claim_vote_health(payload)
        summary = dict((payload or {}).get("summary") or {})
        if snapshot_path:
            self._append_runtime_log(
                f"[CYCLE-SNAPSHOT] saved={snapshot_path}",
                level="info",
            )
        self._append_runtime_log(
            (
                "[CYCLE-CHECK] "
                f"claims={health['claims']} voted_claims={health['voted_claims']} "
                f"unvoted_claims={health['unvoted_claims']} votes={health['votes']} probes={health['probes']} "
                f"(single={health['single_votes']} multi={health['multi_votes']} caption={health['caption_votes']}) "
                f"queue={health['human_queue']} online_probes={health['online_probes']} "
                f"verifier_errors={health['verifier_errors']}"
            ),
            level="info",
        )
        if int(health["claims"]) <= 0:
            self._set_status(
                "Cycle verify produced empty claims; please check scene graph generation and frame image.",
                status_type="warning",
            )
        elif int(health["votes"]) <= 0:
            if int(health["human_queue"]) > 0:
                self._set_status(
                    "Cycle verify generated no auto probes under current confidence gating; using Human Edit Queue for this frame.",
                    status_type="info",
                )
            else:
                self._set_status(
                    "Cycle verify produced empty votes/probes; please check verifier provider, API key, and frame image.",
                    status_type="warning",
                )
        self._append_oplog(
            "cycle_refine_done",
            rounds_run=int(summary.get("rounds_run", 0) or 0),
            probe_count=int(summary.get("probe_count", 0) or 0),
            queue_count=int(summary.get("queue_count", 0) or 0),
            claim_count=int(health["claims"]),
            vote_count=int(health["votes"]),
            voted_claim_count=int(health["voted_claims"]),
            unvoted_claim_count=int(health["unvoted_claims"]),
            snapshot_path=snapshot_path,
        )
        if snapshot_path:
            self._append_oplog(
                "cycle_refine_snapshot_saved",
                path=snapshot_path,
                probe_count=int(health["probes"]),
            )

    def _on_cycle_worker_failed(self, error_text: str) -> None:
        msg = str(error_text or "Unknown cycle refine error").strip()
        self._close_cycle_progress_dialog()
        self._set_scene_graph_busy(False)
        if hasattr(self, "sg_cycle_summary") and isinstance(self.sg_cycle_summary, QLabel):
            self.sg_cycle_summary.setText(f"Cycle refine failed: {msg}")
        QMessageBox.critical(self, "Cycle Refine Failed", f"Failed to run cycle refine:\n{msg}")
        self._set_status(f"Cycle refine failed: {msg}", status_type="error")
        self._append_runtime_log(f"Cycle refine failed: {msg}", level="error")
        self._append_oplog("cycle_refine_failed", error=msg)

    def _run_cycle_refine_for_current_graph(
        self,
        *,
        target_claim_ids: Optional[List[str]] = None,
        base_result: Optional[Dict[str, object]] = None,
        run_reason: str = "",
    ) -> None:
        if self._scene_graph_job_active() or self._cycle_job_active():
            self._set_status("Cycle Verify waits until SAM and Qwen are fully finished.", status_type="warning")
            return
        ok_auth, auth_msg = self._bootstrap_cycle_verifier_auth()
        if auth_msg:
            self._append_runtime_log(auth_msg, level="info" if ok_auth else "error")
        if not ok_auth:
            QMessageBox.warning(self, "Cycle Verify Auth", auth_msg)
            self._set_status("Cycle Verify blocked: missing API key.", status_type="warning")
            return
        if not isinstance(self.current_graph, dict):
            QMessageBox.information(self, "No Graph", "Build or import a scene graph first.")
            self._set_status("No graph available for cycle refine", status_type="warning")
            return
        image_path = self._resolve_cycle_image_path()
        self._append_runtime_log(
            f"[CYCLE-IMAGE] resolved={image_path!r} exists={os.path.isfile(image_path) if image_path else False} "
            f"video_path={self.video_path!r}",
            level="info",
        )
        if not image_path or not os.path.isfile(image_path):
            QMessageBox.information(
                self,
                "Missing Frame Image",
                "Cycle refine needs the source frame image. Build the graph from a video frame or keep cached frame images.",
            )
            self._set_status("No source frame image available for cycle refine", status_type="warning")
            return
        try:
            cycle_cfg_path = self._ensure_cycle_cfg_file()
        except Exception as exc:
            QMessageBox.critical(self, "Cycle Config Error", f"Failed to prepare cycle config:\n{exc}")
            self._set_status("Failed to prepare cycle config", status_type="error")
            return

        target_ids = [str(x).strip() for x in list(target_claim_ids or []) if str(x).strip()]
        worker = CycleRefineWorker(
            graph=self._graph_with_cycle_temporal_context(self.current_graph),
            image_path=image_path,
            ontology_path=self._ontology_path,
            cycle_cfg_path=cycle_cfg_path,
            custom_ontology_dict=self._custom_ontology,
            correction_memory=self._correction_memory if isinstance(self._correction_memory, dict) else default_correction_memory(),
            cycle_cfg_override=self._scene_graph_cycle_cfg_override(),
            target_claim_ids=target_ids,
            base_result=base_result if isinstance(base_result, dict) else self.current_cycle_result,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_cycle_worker_progress)
        worker.done.connect(self._on_cycle_worker_done)
        worker.done.connect(thread.quit)
        worker.failed.connect(self._on_cycle_worker_failed)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_cycle_worker_thread_finished)

        self._cycle_worker = worker
        self._cycle_worker_thread = thread
        self._set_scene_graph_busy(True)
        if hasattr(self, "sg_cycle_summary") and isinstance(self.sg_cycle_summary, QLabel):
            self.sg_cycle_summary.setText(
                "Targeted cycle refine is running..."
                if target_ids
                else "Cycle refine is running..."
            )
        progress = QProgressDialog("Running cross-task cycle refine...", "", 0, 100, self)
        progress.setWindowTitle("Cycle Refine")
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setWindowModality(Qt.WindowModal)
        progress.setValue(5)
        self._cycle_progress_value = 5
        self._cycle_progress_dialog = progress
        progress.show()
        if target_ids:
            self._set_status(
                f"Starting targeted cycle refine for {len(target_ids)} affected claim(s)...",
                status_type="info",
            )
        else:
            self._set_status("Starting cycle refine...", status_type="info")
        sg_settings = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        low_quota = bool(sg_settings.get("cycle_low_quota_mode", False))
        self._append_runtime_log(
            f"[CYCLE-CONFIG] rounds={int(sg_settings.get('cycle_max_revision_rounds', 2) or 2)} low_quota={int(low_quota)}",
            level="info",
        )
        if target_ids:
            self._append_runtime_log(
                f"[CYCLE-TARGETED] claims={len(target_ids)} reason={str(run_reason or 'targeted_reverify').strip()}",
                level="info",
            )
        self._append_oplog(
            "cycle_refine_start",
            image_path=image_path,
            cycle_cfg_path=cycle_cfg_path,
            low_quota=bool(low_quota),
            rounds=int(sg_settings.get("cycle_max_revision_rounds", 2) or 2),
            target_claim_count=len(target_ids),
            run_reason=str(run_reason or "").strip(),
        )
        thread.start()

    def _sync_scene_graph_toolbar_visibility(self) -> None:
        if not hasattr(self, "scene_graph_quick_actions_wrap"):
            return
        is_scene_graph = self._current_task_name() == "Video Scene Graph"
        self.scene_graph_quick_actions_wrap.setVisible(is_scene_graph)

    def _build_runtime_output_panel(self) -> QFrame:
        runtime_shell = QFrame()
        runtime_shell.setObjectName("runtimeLogPanel")
        runtime_shell.setMinimumHeight(240)
        runtime_shell.setMaximumHeight(240)
        runtime_layout = QVBoxLayout(runtime_shell)
        runtime_layout.setContentsMargins(12, 10, 12, 10)
        runtime_layout.setSpacing(6)

        runtime_header = QHBoxLayout()
        runtime_header.setContentsMargins(0, 0, 0, 0)
        runtime_header.setSpacing(8)
        runtime_title = QLabel("Runtime Log")
        runtime_title.setObjectName("runtimeLogTitle")
        runtime_header.addWidget(runtime_title)
        runtime_header.addStretch(1)
        runtime_layout.addLayout(runtime_header)

        self.runtime_log = QPlainTextEdit()
        self.runtime_log.setReadOnly(True)
        self.runtime_log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.runtime_log.setPlaceholderText("Backend progress, command output, and runtime errors will appear here.")
        self.runtime_log.setMinimumHeight(120)
        self.runtime_log.setTextInteractionFlags(Qt.TextSelectableByKeyboard | Qt.TextSelectableByMouse)
        runtime_layout.addWidget(self.runtime_log, 1)
        return runtime_shell

    def _update_workspace_header(self) -> None:
        if hasattr(self, "workspace_pill_task"):
            self.workspace_pill_task.setText("")
        if hasattr(self, "workspace_pill_mode"):
            self.workspace_pill_mode.setText("")
        if hasattr(self, "metric_task_value"):
            self.metric_task_value.setText("")
        if hasattr(self, "metric_mode_value"):
            self.metric_mode_value.setText("")
        if hasattr(self, "metric_video_value"):
            self.metric_video_value.setText("")
        if hasattr(self, "metric_backend_value"):
            self.metric_backend_value.setText("")

    def _preferred_sam_runtime_config_path(self) -> str:
        candidates: List[str] = []
        if os.name == "nt":
            candidates.append(os.path.join(self._repo_root, "configs", "sam3_runtime.windows.json"))
        else:
            candidates.append(os.path.join(self._repo_root, "configs", "sam3_runtime.linux.json"))
        candidates.append(os.path.join(self._repo_root, "configs", "sam3_runtime.wsl.json"))
        candidates.append(os.path.join(self._repo_root, "configs", "sam3_runtime.example.json"))
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[0] if candidates else ""

    def _preferred_sam_runner_profile(self, runtime_config: str = "") -> str:
        base = os.path.basename(str(runtime_config or "")).strip().lower()
        if "sam3" in base and (".windows." in base or base.endswith(".windows.json")):
            return "sam3_windows"
        if "sam3" in base and (".linux." in base or base.endswith(".linux.json")):
            return "sam3"
        if "sam3" in base and (".wsl." in base or base.endswith(".wsl.json")):
            return "sam3_wsl"
        if os.name == "nt":
            return "sam3_windows"
        return "sam3"

    def _scene_graph_job_active(self) -> bool:
        thread = self._sg_worker_thread
        if bool(thread is not None and thread.isRunning()):
            return True
        llm_thread = self._llm_summary_thread
        return bool(llm_thread is not None and llm_thread.isRunning())

    def _cycle_job_active(self) -> bool:
        thread = self._cycle_worker_thread
        return bool(thread is not None and thread.isRunning())

    def _set_scene_graph_busy(self, busy: bool) -> None:
        widgets = [
            getattr(self, "btn_build_graph", None),
            getattr(self, "btn_build_graph_video", None),
            getattr(self, "btn_new_run_panel", None),
            getattr(self, "btn_overlay_current", None),
            getattr(self, "btn_cycle_refine", None),
            getattr(self, "btn_run_cycle_verify", None),
            getattr(self, "btn_use_current_frame", None),
            getattr(self, "spin_frame_for_graph", None),
            getattr(self, "spin_sg_sampling_every_n_frames", None),
            getattr(self, "btn_next_sg", None),
            getattr(self, "btn_pick_llm", None),
            getattr(self, "cycle_provider_combo", None),
            getattr(self, "cycle_model_path_input", None),
            getattr(self, "btn_cycle_model_browse", None),
            getattr(self, "cycle_max_rounds_spin", None),
            getattr(self, "cycle_low_quota_check", None),
            getattr(self, "cycle_enable_single_check", None),
            getattr(self, "cycle_enable_multi_check", None),
            getattr(self, "cycle_enable_caption_check", None),
            getattr(self, "cycle_debug_mode_check", None),
        ]
        for widget in widgets:
            if widget is not None:
                widget.setEnabled(not busy)
        if getattr(self, "btn_stop_graph", None) is not None:
            self.btn_stop_graph.setEnabled(bool(busy))

    def _load_runtime_config_payload(self, runtime_config_path: str) -> Dict[str, object]:
        with open(runtime_config_path, "r", encoding="utf-8-sig") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Runtime config must be a JSON object: {runtime_config_path}")
        payload["_config_dir"] = os.path.dirname(os.path.abspath(runtime_config_path))
        return payload

    def _resolve_runtime_repo_root(self, runtime_config_path: str) -> str:
        payload = self._load_runtime_config_payload(runtime_config_path)
        repo_root = str(payload.get("repo_root", "") or "").strip()
        if not repo_root:
            raise RuntimeError(f"Missing repo_root in runtime config: {runtime_config_path}")
        if not os.path.isabs(repo_root):
            repo_root = os.path.abspath(os.path.join(str(payload.get("_config_dir", "") or self._repo_root), repo_root))
        return os.path.abspath(os.path.expanduser(repo_root))

    def _scene_graph_backend_preflight(self, sg_settings: Dict[str, object]) -> str:
        backend_provider = str(sg_settings.get("backend_provider", "") or "").strip().lower()
        if backend_provider != "external_command":
            return ""
        runtime_config = self._sam_runtime_config_from_settings(sg_settings)
        if not runtime_config:
            return ""
        runner_profile = self._sam_runner_profile_from_settings(sg_settings)
        if not runner_profile:
            runner_profile = self._preferred_sam_runner_profile(runtime_config)
        try:
            payload = self._load_runtime_config_payload(runtime_config)
        except Exception as exc:
            return f"SAM runtime config is invalid: {exc}"
        configured_device = str(payload.get("device", "") or "").strip().lower()
        if not configured_device:
            configured_device = str(((payload.get("category_discovery") or {}) if isinstance(payload.get("category_discovery"), dict) else {}).get("device", "") or "").strip().lower()
        requires_cuda = configured_device.startswith("cuda")
        if requires_cuda and runner_profile in {"sam3", "sam3_wsl", "sam3_windows"}:
            try:
                probe = subprocess.run(
                    ["nvidia-smi", "-L"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=8,
                    check=False,
                )
                if int(probe.returncode) != 0:
                    detail = str((probe.stderr or probe.stdout or "")).strip()
                    detail = detail.splitlines()[0] if detail else "nvidia-smi failed"
                    return (
                        "CUDA device is not ready for SAM3 (runtime is configured with device=cuda). "
                        f"GPU probe failed: {detail}"
                    )
            except Exception as exc:
                return (
                    "CUDA device is not ready for SAM3 (runtime is configured with device=cuda). "
                    f"GPU probe failed: {exc}"
                )
        if runner_profile != "sam3_windows":
            return ""
        try:
            repo_root = self._resolve_runtime_repo_root(runtime_config)
        except Exception as exc:
            return f"Windows SAM runtime config is invalid: {exc}"
        if not os.path.isdir(repo_root):
            return f"Windows SAM repo_root does not exist: {repo_root}"
        model_family = str(payload.get("model_family", "") or payload.get("backend_family", "") or "").strip().lower()
        if not model_family and os.path.isdir(os.path.join(repo_root, "sam3")):
            model_family = "sam3"
        sam3_pkg_dir = os.path.join(repo_root, "sam3")
        if model_family in {"", "sam3"} and os.path.isdir(sam3_pkg_dir):
            return ""
        return f"Windows SAM3 repo looks incomplete: missing {sam3_pkg_dir}"

    def _default_task_settings(self) -> Dict[str, Dict[str, object]]:
        default_sam_runtime = self._preferred_sam_runtime_config_path()
        default_runner_profile = self._preferred_sam_runner_profile(default_sam_runtime)
        return {
            "Video Scene Graph": {
                "enable_sentence_refine": False,
                "backend_provider": "external_command",
                "sam_runner_profile": default_runner_profile,
                "sam_runtime_config": default_sam_runtime,
                "external_command_args_file": "",
                "external_command_template": "",
                "backend_timeout_sec": 1800,
                "disable_backend_cache": False,
                "tracking_mode": "disabled",
                "tracking_search_radius": 72,
                "tracking_min_response": 0.35,
                "tracking_template_update_alpha": 0.25,
                "tracking_max_lost_frames": 12,
                "tracking_short_gap_frames": 24,
                "tracking_long_gap_frames": 120,
                "tracking_max_gap_frames": 240,
                "tracking_min_match_score_short": 0.2,
                "tracking_min_match_score_long": 0.35,
                "video_generation_fps": 1.0,
                "video_generation_batch_size": 8,
                "video_sampling_every_n_frames": 30,
                "llm_mode": "local",
                "llm_local_provider": "qwen",
                "llm_local_model_path": QWEN_DEFAULT_MODEL_PATH,
                "qwen_model_path": QWEN_DEFAULT_MODEL_PATH,
                "qwen_batch_size": 3,
                "llm_cuda_device": "cuda:0",
                "cycle_verifier_provider": DEFAULT_CYCLE_PROVIDER,
                "cycle_local_model_path": QWEN_DEFAULT_MODEL_PATH,
                "cycle_allow_mock_fallback": False,
                "cycle_gemini_model": DEFAULT_GEMINI_MODEL,
                "cycle_gemini_api_key_env": DEFAULT_GEMINI_API_KEY_ENV,
                "cycle_gemini_online_timeout_sec": DEFAULT_GEMINI_ONLINE_TIMEOUT_SEC,
                "cycle_gemini_online_headless": False,
                "cycle_gemini_online_user_data_dir": "",
                "cycle_gemini_online_profile_directory": "",
                "cycle_gemini_online_chrome_binary": "",
                "cycle_chatgpt_model": DEFAULT_OPENAI_MODEL,
                "cycle_chatgpt_api_key_env": DEFAULT_OPENAI_API_KEY_ENV,
                "cycle_chatgpt_base_url": DEFAULT_OPENAI_BASE_URL,
                "cycle_max_revision_rounds": 1,
                "cycle_low_quota_mode": False,
                "cycle_enable_single_turn_probes": True,
                "cycle_enable_multi_turn_probes": True,
                "cycle_enable_caption_probe": True,
                "cycle_debug_mode": False,
                "enable_pvsg_gt_reference": True,
                "pvsg_json_path": "/cvhci/temp/wkong/sample_videos/pvsg.json",
                "pvsg_masks_root": "/cvhci/temp/wkong/sample_videos/VidOR/masks",
                "video_sampling_mode": "uniform_fps",
                "video_sampling_seed": 42,
                "video_sampling_jitter_ratio": 0.35,
                "video_sampling_max_frames": 3,
                "person_min_bbox_area": PERSON_MIN_BBOX_AREA,
                "person_min_bbox_width": PERSON_MIN_BBOX_WIDTH,
                "person_min_bbox_height": PERSON_MIN_BBOX_HEIGHT,
                "person_min_area_ratio": PERSON_MIN_AREA_RATIO,
                "person_max_tracks_per_frame": PERSON_HIGH_MAX_PER_FRAME,
                "person_priority_topk": PERSON_PRIORITY_TOPK,
                "person_center_bias_weight": PERSON_CENTER_BIAS_WEIGHT,
                "person_filtered_min_area_ratio": PERSON_MIN_AREA_RATIO,
                "person_filtered_min_box_side_px": min(PERSON_MIN_BBOX_WIDTH, PERSON_MIN_BBOX_HEIGHT),
                "person_filtered_min_score": PERSON_FILTERED_MIN_SCORE,
                "person_high_min_area_ratio": PERSON_HIGH_MIN_AREA_RATIO,
                "person_high_center_min_area_ratio": PERSON_HIGH_CENTER_MIN_AREA_RATIO,
                "person_high_center_max_distance_norm": PERSON_HIGH_CENTER_MAX_DISTANCE_NORM,
                "person_high_top_k": PERSON_PRIORITY_TOPK,
                "person_high_max_per_frame": PERSON_HIGH_MAX_PER_FRAME,
                "low_priority_tracking_mode": PERSON_LOW_PRIORITY_TRACKING_MODE,
                "low_priority_max_lost_frames": PERSON_LOW_PRIORITY_MAX_LOST_FRAMES,
                "person_filtered_debug_keep": PERSON_FILTERED_DEBUG_KEEP,
                "person_track_demote_policy": PERSON_TRACK_DEMOTE_POLICY,
                "mask_export_mode": MASK_EXPORT_MODE,
                "mask_external_dir": "masks",
            },
            "Single-turn VQA": {"max_items": 0},
            "Multi-turn VQA": {"max_items": 0},
            "Video Captioning": {"default_style": "Detailed"},
        }

    def _save_persisted_settings(self) -> None:
        payload = {
            "common_settings": self._common_settings,
            "task_settings": self._task_settings,
            "custom_ontology": self._custom_ontology,
            "ontology_status_text": self._ontology_status_text,
            "validation_changes": self._validation_changes,
        }
        try:
            os.makedirs(os.path.dirname(self._ui_settings_path), exist_ok=True)
            with open(self._ui_settings_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
        except Exception:
        # Persistence should not block normal workflow.
            pass

    def _load_persisted_settings(self) -> None:
        if not os.path.isfile(self._ui_settings_path):
            return
        try:
            with open(self._ui_settings_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            loaded_common_settings = payload.get("common_settings", {})
            if isinstance(loaded_common_settings, dict):
                merged_common = dict(self._common_settings_defaults)
                merged_common.update(loaded_common_settings)
                self._common_settings = merged_common

            loaded_task_settings = payload.get("task_settings", {})
            if isinstance(loaded_task_settings, dict):
                for task, defaults in self._task_settings_defaults.items():
                    incoming = loaded_task_settings.get(task, {})
                    merged = dict(defaults)
                    if isinstance(incoming, dict):
                        merged.update(incoming)
                    if task == "Video Scene Graph":
                        merged = self._normalize_scene_graph_settings(merged)
                    self._task_settings[task] = merged

            custom_ontology = payload.get("custom_ontology")
            if isinstance(custom_ontology, dict):
                self._custom_ontology = custom_ontology

            validation_changes = payload.get("validation_changes", [])
            if isinstance(validation_changes, list):
                self._validation_changes = [x for x in validation_changes if isinstance(x, dict)]

            ontology_status_text = payload.get("ontology_status_text")
            if isinstance(ontology_status_text, str) and ontology_status_text.strip():
                self._ontology_status_text = ontology_status_text.strip()
        except Exception:
        # Invalid settings file should not crash startup.
            return

    def _apply_task_settings_to_widgets(self) -> None:
        sg_settings = self._task_settings.get("Video Scene Graph", {})
        if hasattr(self, "spin_sg_sampling_every_n_frames"):
            self.spin_sg_sampling_every_n_frames.setValue(int(sg_settings.get("video_sampling_every_n_frames", 30) or 30))
        if hasattr(self, "spin_sg_sampling_max_frames"):
            self.spin_sg_sampling_max_frames.setValue(int(sg_settings.get("video_sampling_max_frames", 3) or 3))
        if hasattr(self, "lbl_llm_selection"):
            # UI policy: always display/use local Qwen.
            sg_settings["llm_mode"] = "local"
            sg_settings["llm_local_provider"] = "qwen"
            qbs = max(1, int(sg_settings.get("qwen_batch_size", 3) or 3))
            self.lbl_llm_selection.setText(f"Local - Qwen ({qbs}f summary)")
        if hasattr(self, "cycle_provider_combo"):
            mapping = {"gemini_api": 0, "chatgpt_api": 1, "qwen25_vl": 2, "manual": 3}
            provider = normalize_cycle_provider(
                sg_settings.get("cycle_verifier_provider", DEFAULT_CYCLE_PROVIDER),
                default=DEFAULT_CYCLE_PROVIDER,
            )
            idx = int(mapping.get(provider, 0))
            self.cycle_provider_combo.blockSignals(True)
            self.cycle_provider_combo.setCurrentIndex(idx)
            self.cycle_provider_combo.blockSignals(False)
        if hasattr(self, "cycle_max_rounds_spin"):
            rounds = max(1, min(10, int(sg_settings.get("cycle_max_revision_rounds", 2) or 2)))
            self.cycle_max_rounds_spin.blockSignals(True)
            self.cycle_max_rounds_spin.setValue(rounds)
            self.cycle_max_rounds_spin.blockSignals(False)
        if hasattr(self, "cycle_low_quota_check"):
            low_quota = bool(sg_settings.get("cycle_low_quota_mode", False))
            self.cycle_low_quota_check.blockSignals(True)
            self.cycle_low_quota_check.setChecked(low_quota)
            self.cycle_low_quota_check.blockSignals(False)
        if hasattr(self, "cycle_enable_single_check"):
            val = bool(sg_settings.get("cycle_enable_single_turn_probes", True))
            self.cycle_enable_single_check.blockSignals(True)
            self.cycle_enable_single_check.setChecked(val)
            self.cycle_enable_single_check.blockSignals(False)
        if hasattr(self, "cycle_enable_multi_check"):
            val = bool(sg_settings.get("cycle_enable_multi_turn_probes", True))
            self.cycle_enable_multi_check.blockSignals(True)
            self.cycle_enable_multi_check.setChecked(val)
            self.cycle_enable_multi_check.blockSignals(False)
        if hasattr(self, "cycle_enable_caption_check"):
            val = bool(sg_settings.get("cycle_enable_caption_probe", True))
            self.cycle_enable_caption_check.blockSignals(True)
            self.cycle_enable_caption_check.setChecked(val)
            self.cycle_enable_caption_check.blockSignals(False)
        if hasattr(self, "cycle_debug_mode_check"):
            val = bool(sg_settings.get("cycle_debug_mode", False))
            self.cycle_debug_mode_check.blockSignals(True)
            self.cycle_debug_mode_check.setChecked(val)
            self.cycle_debug_mode_check.blockSignals(False)

        cap_settings = self._task_settings.get("Video Captioning", {})
        style = str(cap_settings.get("default_style", "Concise"))
        idx = self.cap_style.findText(style)
        self.cap_style.setCurrentIndex(max(0, idx))

    def _apply_common_settings_runtime(self) -> None:
        # Bootstrap API key at startup:
        # if UI setting is empty but shell env has a key, inject it into the current session.
        api_key = str(self._common_settings.get("api_key", "")).strip()
        if not api_key:
            env_gemini = str(os.environ.get(DEFAULT_GEMINI_API_KEY_ENV, "") or "").strip()
            env_generic = str(os.environ.get("IMPACT_API_KEY", "") or "").strip()
            injected = env_gemini or env_generic
            if injected:
                api_key = injected
                self._common_settings["api_key"] = api_key
                if hasattr(self, "_api_key_input") and self._api_key_input is not None:
                    self._api_key_input.setText(api_key)

        # Expose API key for downstream integrations that read environment variables.
        if api_key:
            os.environ["IMPACT_API_KEY"] = api_key
            if not str(os.environ.get(DEFAULT_GEMINI_API_KEY_ENV, "")).strip():
                os.environ[DEFAULT_GEMINI_API_KEY_ENV] = api_key

        fps_min = float(self._common_settings.get("fps_min", 1.0))
        fps_max = float(self._common_settings.get("fps_max", 120.0))
        if fps_max < fps_min:
            fps_min, fps_max = fps_max, fps_min
            self._common_settings["fps_min"] = fps_min
            self._common_settings["fps_max"] = fps_max

        self.fps_spin.setRange(fps_min, fps_max)
        self.fps_spin.setValue(float(self._common_settings.get("fps_override", 30.0)))
        enabled = bool(self._common_settings.get("fps_override_enabled", False))
        self.fps_mode_combo.setCurrentIndex(1 if enabled else 0)
        self.fps_spin.setEnabled(enabled)

        if hasattr(self, "participant_id_input"):
            self.participant_id_input.setText(str(self._common_settings.get("participant_id", "")))
        self.validator_id_input.setText(str(self._common_settings.get("validator_id", "")))
        self.validation_round_spin.setValue(int(self._common_settings.get("validation_round", 1)))

    def _build_ui(self) -> None:
        self._apply_commercial_theme()
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        impact_title = QLabel("IMPACT")
        impact_title.setObjectName("heroTitle")
        impact_title.setStyleSheet("color: #122033; font-size: 22px; font-weight: 800;")
        root.addWidget(impact_title)

        top_shell = QFrame()
        top_shell.setObjectName("topControlBar")
        top_shell_layout = QVBoxLayout(top_shell)
        top_shell_layout.setContentsMargins(14, 10, 14, 10)
        top_shell_layout.setSpacing(8)
        controls_shell = QWidget()
        controls_layout = QVBoxLayout(controls_shell)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        self.task_combo = QComboBox()
        self.task_combo.setMaximumWidth(170)
        self.task_combo.addItems(
            [
                "Video Scene Graph",
                "Single-turn VQA",
                "Multi-turn VQA",
                "Video Captioning",
            ]
        )
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        top.addWidget(QLabel("Task"))
        top.addWidget(self.task_combo)
        top.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Annotate", "Validate"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        top.addWidget(self.mode_combo)

        top.addWidget(QLabel("Participant"))
        self.participant_id_input = QLineEdit()
        self.participant_id_input.setPlaceholderText("participant_id")
        self.participant_id_input.setMaximumWidth(180)
        self.participant_id_input.editingFinished.connect(self._on_participant_changed)
        top.addWidget(self.participant_id_input)

        top.addWidget(QLabel("Validator"))
        self.validator_id_input = QLineEdit()
        self.validator_id_input.setPlaceholderText("validator_id")
        self.validator_id_input.setMaximumWidth(180)
        self.validator_id_input.editingFinished.connect(self._on_validator_changed)
        top.addWidget(self.validator_id_input)

        top.addWidget(QLabel("Round"))
        self.validation_round_spin = QSpinBox()
        self.validation_round_spin.setRange(1, 20)
        self.validation_round_spin.setValue(1)
        self.validation_round_spin.valueChanged.connect(self._on_validation_round_changed)
        top.addWidget(self.validation_round_spin)

        # Global Data menu (import/export)
        data_btn, data_menu = self._make_actions_menu_button("Data")
        
        action = data_menu.addAction("New Run (Select Video)")
        action.triggered.connect(self._run_new_video_scene_graph_ui)
        action = data_menu.addAction("Resume Run")
        action.triggered.connect(self._resume_video_scene_graph_from_folder_ui)
        action = data_menu.addAction("Open Run Result")
        action.triggered.connect(self._open_video_scene_graph_result_ui)
        
        data_menu.addSeparator()
        
        # Scene Graph section
        action = data_menu.addAction("Import Scene Graph")
        action.triggered.connect(self._import_scene_graph_annotation)
        action = data_menu.addAction("Export Scene Graph")
        action.triggered.connect(self._save_graph)
        
        data_menu.addSeparator()
        
        # Single-turn VQA section
        action = data_menu.addAction("Import Single-turn VQA")
        action.triggered.connect(self._import_single_turn_annotation)
        action = data_menu.addAction("Export Single-turn VQA")
        action.triggered.connect(self._save_single_turn)
        
        data_menu.addSeparator()
        
        # Multi-turn VQA section
        action = data_menu.addAction("Import Multi-turn VQA")
        action.triggered.connect(self._import_multi_turn_annotation)
        action = data_menu.addAction("Export Multi-turn VQA")
        action.triggered.connect(self._save_multi_turn)
        
        data_menu.addSeparator()
        
        # Validation logs section
        action = data_menu.addAction("Import Validation Log (Current Task)")
        action.triggered.connect(lambda: self._import_validation_log_for(self._current_task_name()))
        action = data_menu.addAction("Export Validation Log (Current Task)")
        action.triggered.connect(lambda: self._export_validation_log_for(self._current_task_name()))
        
        top.addWidget(data_btn)
        
        self.btn_task_settings = QToolButton()
        self.btn_task_settings.setText("Prefs")
        self.btn_task_settings.setStyleSheet("font-size: 13px; font-weight: 600;")
        self.btn_task_settings.setFixedWidth(58)
        self.btn_task_settings.setToolTip("Open settings for current task")
        self.btn_task_settings.clicked.connect(self._open_task_settings_dialog)
        top.addWidget(self.btn_task_settings)
        top.addStretch(1)
        controls_layout.addLayout(top)

        self.scene_graph_quick_actions_wrap = QWidget()
        scene_graph_toolbar = QHBoxLayout(self.scene_graph_quick_actions_wrap)
        scene_graph_toolbar.setContentsMargins(0, 0, 0, 0)
        scene_graph_toolbar.setSpacing(6)

        self.btn_build_graph = QPushButton("Build Frame")
        self.btn_build_graph.clicked.connect(self._build_scene_graph_for_selected_frame)
        scene_graph_toolbar.addWidget(self.btn_build_graph)
        self.btn_cycle_refine = QPushButton("Cycle Verify")
        self.btn_cycle_refine.clicked.connect(self._run_cycle_refine_for_current_graph)
        scene_graph_toolbar.addWidget(self.btn_cycle_refine)

        self.btn_stop_graph = QPushButton("Stop Run")
        self.btn_stop_graph.clicked.connect(self._cancel_scene_graph_job)
        self.btn_stop_graph.setEnabled(False)
        scene_graph_toolbar.addWidget(self.btn_stop_graph)

        self.btn_save_graph = self._make_panel_toolbar_button(
            QStyle.SP_DialogSaveButton,
            "Save Scene Graph JSON",
            self._save_graph,
        )
        scene_graph_toolbar.addWidget(self.btn_save_graph)

        self.btn_refresh_graph = self._make_panel_toolbar_button(
            QStyle.SP_BrowserReload,
            "Refresh Display",
            self._render_graph,
        )
        scene_graph_toolbar.addWidget(self.btn_refresh_graph)

        self.btn_overlay_current = self._make_panel_toolbar_button(
            QStyle.SP_DialogApplyButton,
            "Generate Full Video Abstract + Track",
            self._generate_scene_graph_for_video_with_current_keyframe,
        )
        scene_graph_toolbar.addWidget(self.btn_overlay_current)

        self.sg_actions_btn, sg_actions_menu = self._make_actions_menu_button("Actions")
        action = sg_actions_menu.addAction("Import Graph")
        action.triggered.connect(self._import_scene_graph_annotation)
        action = sg_actions_menu.addAction("Apply JSON Edit")
        action.triggered.connect(self._apply_scene_graph_json_edit)
        sg_actions_menu.addSeparator()
        action = sg_actions_menu.addAction("Import Validation Log")
        action.triggered.connect(lambda: self._import_validation_log_for("Video Scene Graph"))
        action = sg_actions_menu.addAction("Export Validation Log")
        action.triggered.connect(lambda: self._export_validation_log_for("Video Scene Graph"))
        action = sg_actions_menu.addAction("Export Final")
        action.triggered.connect(self._export_scene_graph_final_confirmed)
        action = sg_actions_menu.addAction("Export Bundle")
        action.triggered.connect(lambda: self._export_final_bundle_for("Video Scene Graph"))
        sg_actions_menu.addSeparator()
        action = sg_actions_menu.addAction("Confirm Selected Change")
        action.triggered.connect(lambda: self._set_change_decision(task_name="Video Scene Graph", list_widget=self.sg_change_list, approved=True))
        action = sg_actions_menu.addAction("Reject Selected Change")
        action.triggered.connect(lambda: self._set_change_decision(task_name="Video Scene Graph", list_widget=self.sg_change_list, approved=False))
        scene_graph_toolbar.addWidget(self.sg_actions_btn)
        top_shell_layout.addWidget(controls_shell, 0)
        top_shell_layout.addWidget(self.scene_graph_quick_actions_wrap, 0)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        left = QWidget()
        left.setObjectName("leftWorkspace")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(8)
        left_layout.addWidget(top_shell)

        self.player = VideoPlayer(status_cb=self._set_status)
        self.player.setStyleSheet("background: #0b1220; border: 1px solid #22344d; border-radius: 18px;")
        self.player.setText("")
        self.player.on_frame_advanced = self._on_frame_advanced
        self.player.on_playback_state_changed = self._on_play_state_changed
        self.player.setMinimumHeight(320)
        self.player.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.player, 1)

        # Unified runtime output: place log right under video, avoid duplicated status echoes.
        self.sg_runtime_panel = self._build_runtime_output_panel()
        self.sg_runtime_panel.setMinimumHeight(130)
        self.sg_runtime_panel.setMaximumHeight(180)
        self.sg_runtime_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        left_layout.addWidget(self.sg_runtime_panel, 0)

        transport_shell = QFrame()
        transport_shell.setObjectName("transportBar")
        transport_layout = QVBoxLayout(transport_shell)
        transport_layout.setContentsMargins(12, 10, 12, 10)
        transport_layout.setSpacing(8)
        transport_shell.setMinimumHeight(104)
        transport_shell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        transport = QHBoxLayout()
        transport.setContentsMargins(0, 0, 0, 0)
        transport.setSpacing(8)
        self.btn_prev = self._make_transport_button(
            QStyle.SP_MediaSkipBackward,
            "Previous frame",
            lambda: self._step_frame(-1),
        )
        self.btn_play_pause = self._make_transport_button(
            QStyle.SP_MediaPlay,
            "Play / Pause",
            self._toggle_play,
        )
        self.btn_next = self._make_transport_button(
            QStyle.SP_MediaSkipForward,
            "Next frame",
            lambda: self._step_frame(1),
        )
        self.btn_next_sg = self._make_transport_button(
            QStyle.SP_MediaSeekForward,
            "Next scene-graph sampled frame",
            self._jump_to_next_scene_graph_sampled_frame,
        )
        self.btn_stop = self._make_transport_button(
            QStyle.SP_MediaStop,
            "Stop",
            self.player.stop,
        )

        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 0)
        self.frame_spin.valueChanged.connect(self._on_frame_spin_changed)

        self.fps_mode_combo = QComboBox()
        self.fps_mode_combo.addItems(["Auto FPS", "Custom FPS"])
        self.fps_mode_combo.currentIndexChanged.connect(self._on_fps_mode_changed)

        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setDecimals(2)
        self.fps_spin.setSingleStep(0.5)
        self.fps_spin.setRange(1.0, 120.0)
        self.fps_spin.setValue(30.0)
        self.fps_spin.valueChanged.connect(self._on_fps_value_changed)

        self.playback_speed_combo = QComboBox()
        for speed in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
            self.playback_speed_combo.addItem(f"{speed:g}x", float(speed))
        self.playback_speed_combo.setCurrentIndex(3)
        self.playback_speed_combo.currentIndexChanged.connect(self._on_playback_speed_changed)

        self.seek_slider = ClickSeekSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        self.seek_slider.sliderMoved.connect(self._on_slider_moved)
        self.seek_slider.valueChanged.connect(self._on_slider_value_changed)

        self.time_label = QLabel("f=0 / 0  t=00:00.00")
        self.video_name_label = QLabel("")
        self.video_name_label.setStyleSheet("color: #667085;")

        for w in [
            self.btn_prev,
            self.btn_play_pause,
            self.btn_next,
            self.btn_next_sg,
            self.btn_stop,
            QLabel(""),
            self.frame_spin,
            QLabel(""),
            self.playback_speed_combo,
            self.time_label,
            self.video_name_label,
        ]:
            transport.addWidget(w)
        transport.addStretch(1)

        transport_layout.addLayout(transport)
        transport_layout.addWidget(self.seek_slider)
        left_layout.addWidget(transport_shell)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(18)
        self.status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.status_label.setStyleSheet("color: #6B7280; font-weight: 400;")
        left_layout.addWidget(self.status_label)
        left.setMinimumWidth(420)

        splitter.addWidget(left)

        right = QWidget()
        right.setObjectName("rightWorkspace")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)
        self.task_stack = QStackedLayout()
        right_layout.addLayout(self.task_stack, 1)
        right.setMinimumWidth(520)
        splitter.addWidget(right)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([760, 820])

        self._build_scene_graph_panel()
        self._build_single_turn_panel()
        self._build_multi_turn_panel()
        self._build_caption_panel()
        self._sync_scene_graph_toolbar_visibility()

        for attr in ("sg_score_group", "single_score_group", "multi_score_group", "caption_score_group"):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.hide()
        self._space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self._space_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._space_shortcut.activated.connect(self._toggle_play_shortcut)
        self._undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self._undo_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._undo_shortcut.activated.connect(self._undo_scene_graph_last_edit_shortcut)

    def _build_scene_graph_panel(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        cfg_box = QFrame()
        cfg_box.setObjectName("runtimeLogPanel")
        cfg_box_layout = QVBoxLayout(cfg_box)
        cfg_box_layout.setContentsMargins(12, 10, 12, 12)
        cfg_box_layout.setSpacing(8)
        cfg_header = QHBoxLayout()
        cfg_header.setContentsMargins(0, 0, 0, 0)
        cfg_header.setSpacing(8)
        cfg_title = QLabel("Run Configuration")
        cfg_title.setObjectName("runtimeLogTitle")
        cfg_header.addWidget(cfg_title)
        cfg_header.addStretch(1)
        cfg_box_layout.addLayout(cfg_header)
        top_panel_height = 175
        cfg_box.setMinimumHeight(top_panel_height)
        cfg_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)

        _lbl_s = "color: #374151; font-size: 12px;"

        self.spin_frame_for_graph = QSpinBox()
        self.spin_frame_for_graph.setRange(0, 0)
        self.spin_frame_for_graph.setValue(0)
        self.spin_frame_for_graph.setFixedHeight(28)
        self.spin_frame_for_graph.setFixedWidth(80)
        self.spin_frame_for_graph.valueChanged.connect(self._on_graph_frame_selector_changed)
        self.btn_use_current_frame = QPushButton("Add")
        self.btn_use_current_frame.clicked.connect(self._add_current_frame_as_video_keyframe)
        self.btn_use_current_frame.setFixedHeight(28)
        self.btn_use_current_frame.setFixedWidth(80)

        self.spin_sg_sampling_every_n_frames = QSpinBox()
        self.spin_sg_sampling_every_n_frames.setRange(1, 5000)
        self.spin_sg_sampling_every_n_frames.setValue(30)
        self.spin_sg_sampling_every_n_frames.setFixedHeight(28)
        self.spin_sg_sampling_every_n_frames.setFixedWidth(72)
        self.spin_sg_sampling_every_n_frames.setToolTip("Sample one frame every N frames when generating video scene graph")
        self.spin_sg_sampling_every_n_frames.valueChanged.connect(self._on_sg_sampling_every_n_frames_changed)

        self.spin_sg_sampling_max_frames = QSpinBox()
        self.spin_sg_sampling_max_frames.setRange(0, 50000)
        self.spin_sg_sampling_max_frames.setValue(3)
        self.spin_sg_sampling_max_frames.setFixedHeight(28)
        self.spin_sg_sampling_max_frames.setFixedWidth(72)
        self.spin_sg_sampling_max_frames.setToolTip("0 = no cap, otherwise keep at most this many sampled frames (default 3)")
        self.spin_sg_sampling_max_frames.valueChanged.connect(self._on_sg_video_sampling_max_frames_changed)

        self.btn_pick_llm = QPushButton("Qwen")
        self.btn_pick_llm.setFixedHeight(28)
        self.btn_pick_llm.setFixedWidth(80)
        self.btn_pick_llm.setEnabled(False)
        self.btn_pick_llm.setToolTip("LLM Engine is fixed to local Qwen in current UI.")
        self.btn_pick_llm.clicked.connect(self._pick_llm_provider_from_dropdown)
        self.lbl_llm_selection = QLabel("Local - Qwen (3f summary)")
        self.lbl_llm_selection.setMinimumWidth(0)
        self.lbl_llm_selection.setMaximumWidth(260)
        self.lbl_llm_selection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self.cycle_provider_combo = QComboBox()
        self.cycle_provider_combo.addItem("Gemini API", "gemini_api")
        self.cycle_provider_combo.addItem("ChatGPT API", "chatgpt_api")
        self.cycle_provider_combo.addItem("Qwen", "qwen25_vl")
        self.cycle_provider_combo.addItem("Manual", "manual")
        self.cycle_provider_combo.setFixedHeight(28)
        self.cycle_provider_combo.setFixedWidth(130)
        self.cycle_provider_combo.currentIndexChanged.connect(self._on_cycle_provider_changed)

        self.cycle_max_rounds_spin = QSpinBox()
        self.cycle_max_rounds_spin.setRange(1, 10)
        self.cycle_max_rounds_spin.setValue(1)
        self.cycle_max_rounds_spin.setFixedHeight(28)
        self.cycle_max_rounds_spin.setFixedWidth(60)
        self.cycle_max_rounds_spin.setToolTip("How many verification/refinement rounds to run in Cycle Verify.")
        self.cycle_max_rounds_spin.valueChanged.connect(self._on_cycle_max_rounds_changed)

        self.cycle_low_quota_check = QCheckBox("Low Quota Mode")
        self.cycle_low_quota_check.setToolTip("Reduce API request load: disable multi-turn and caption verification.")
        self.cycle_low_quota_check.stateChanged.connect(self._on_cycle_low_quota_changed)
        self.cycle_enable_single_check = QCheckBox("Enable Single-turn")
        self.cycle_enable_single_check.setChecked(True)
        self.cycle_enable_single_check.stateChanged.connect(self._on_cycle_enable_single_changed)
        self.cycle_enable_multi_check = QCheckBox("Enable Multi-turn")
        self.cycle_enable_multi_check.setChecked(True)
        self.cycle_enable_multi_check.stateChanged.connect(self._on_cycle_enable_multi_changed)
        self.cycle_enable_caption_check = QCheckBox("Enable Caption")
        self.cycle_enable_caption_check.setChecked(True)
        self.cycle_enable_caption_check.stateChanged.connect(self._on_cycle_enable_caption_changed)
        self.cycle_debug_mode_check = QCheckBox("Debug Mode")
        self.cycle_debug_mode_check.setChecked(False)
        self.cycle_debug_mode_check.stateChanged.connect(self._on_cycle_debug_mode_changed)

        # Row 0: Frame [spin][Add]  |  LLM Engine [btn][label]
        _cfg_r0 = QHBoxLayout()
        _cfg_r0.setContentsMargins(2, 0, 2, 0)
        _cfg_r0.setSpacing(6)
        _l = QLabel("Frame"); _l.setStyleSheet(_lbl_s); _cfg_r0.addWidget(_l)
        _cfg_r0.addWidget(self.spin_frame_for_graph)
        _cfg_r0.addWidget(self.btn_use_current_frame)
        _cfg_r0.addSpacing(16)
        _l = QLabel("LLM Engine"); _l.setStyleSheet(_lbl_s); _cfg_r0.addWidget(_l)
        _cfg_r0.addWidget(self.btn_pick_llm)
        _cfg_r0.addWidget(self.lbl_llm_selection)
        _cfg_r0.addStretch(1)
        cfg_box_layout.addLayout(_cfg_r0)

        # Row 1: Every N Frames [spin]  |  Max Frames [spin]  |  Cycle Verifier [combo]
        _cfg_r1 = QHBoxLayout()
        _cfg_r1.setContentsMargins(2, 0, 2, 0)
        _cfg_r1.setSpacing(6)
        _l = QLabel("Every N Frames"); _l.setStyleSheet(_lbl_s); _cfg_r1.addWidget(_l)
        _cfg_r1.addWidget(self.spin_sg_sampling_every_n_frames)
        _cfg_r1.addSpacing(16)
        _l = QLabel("Max Frames"); _l.setStyleSheet(_lbl_s); _cfg_r1.addWidget(_l)
        _cfg_r1.addWidget(self.spin_sg_sampling_max_frames)
        _cfg_r1.addSpacing(16)
        _l = QLabel("Cycle Verifier"); _l.setStyleSheet(_lbl_s); _cfg_r1.addWidget(_l)
        _cfg_r1.addWidget(self.cycle_provider_combo)
        _cfg_r1.addStretch(1)
        cfg_box_layout.addLayout(_cfg_r1)

        self.btn_run_cycle_verify = QPushButton("▶ Run Cycle Verify")
        self.btn_run_cycle_verify.setFixedHeight(28)
        self.btn_run_cycle_verify.setToolTip("Run Cycle Verify on the current scene graph frame (uses the Cycle Verifier selected above)")
        self.btn_run_cycle_verify.clicked.connect(self._run_cycle_refine_for_current_graph)

        # Row 2: Cycle toggles + run button.
        _cfg_r2 = QHBoxLayout()
        _cfg_r2.setContentsMargins(2, 0, 2, 0)
        _cfg_r2.setSpacing(6)
        _l = QLabel("Cycle Rounds"); _l.setStyleSheet(_lbl_s); _cfg_r2.addWidget(_l)
        _cfg_r2.addWidget(self.cycle_max_rounds_spin)
        _cfg_r2.addSpacing(16)
        _cfg_r2.addWidget(self.cycle_enable_single_check)
        _cfg_r2.addWidget(self.cycle_enable_multi_check)
        _cfg_r2.addWidget(self.cycle_enable_caption_check)
        _cfg_r2.addWidget(self.cycle_low_quota_check)
        _cfg_r2.addWidget(self.cycle_debug_mode_check)
        _cfg_r2.addStretch(1)
        _cfg_r2.addWidget(self.btn_run_cycle_verify)
        cfg_box_layout.addLayout(_cfg_r2)

        cfg_scroll = QScrollArea()
        cfg_scroll.setWidgetResizable(True)
        cfg_scroll.setFrameShape(QFrame.NoFrame)
        cfg_scroll.setWidget(cfg_box)
        cfg_scroll.setMinimumHeight(top_panel_height)
        cfg_scroll.setMaximumHeight(210)
        cfg_scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)

        layout.addWidget(cfg_scroll)

        # Main vertical splitter: data tables + json preview
        main_splitter = QSplitter(Qt.Vertical)

        # Keep these labels for status updates, but remove them from the visible layout.
        self.sg_summary = QLabel("No graph yet.")
        self.sg_summary.setWordWrap(True)
        self.sg_summary.hide()

        self.sg_overlay_hint = QLabel("Overlay is displayed directly on the left video player. Edit boxes on video (drag / Ctrl+drag add / right-click delete / double-click rename / Ctrl+click two objects add edge / Ctrl+Z undo last edit).")
        self.sg_overlay_hint.setWordWrap(True)
        self.sg_overlay_hint.hide()

        # Hide cycle score cards in UI; keep backend cycle logic available.
        self.sg_cycle_summary = QLabel("Cycle refine not run yet.")
        self.sg_cycle_summary.setWordWrap(True)
        self.sg_cycle_summary.hide()
        self._cycle_stat_widgets = {}

        self.sg_stage_score_strip = self._make_stage_score_strip()
        self.sg_stage_score_strip.setMinimumHeight(92)
        self.sg_stage_score_strip.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        main_splitter.addWidget(self.sg_stage_score_strip)

        # Combined panel: nodes/edges tables + human arbitration queue
        combined_host = QWidget()
        combined_layout = QVBoxLayout(combined_host)
        combined_layout.setContentsMargins(0, 0, 0, 0)
        combined_layout.setSpacing(4)

        # Top: nodes + edges in a horizontal splitter
        ne_splitter = QSplitter(Qt.Horizontal)
        self.sg_nodes_table = QTableWidget(0, 4)
        self.sg_nodes_table.setHorizontalHeaderLabels(["node", "label", "Detection Confidence", "attributes"])
        self.sg_nodes_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sg_nodes_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sg_nodes_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sg_nodes_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.sg_nodes_table.horizontalHeader().setFixedHeight(44)
        self.sg_nodes_table.verticalHeader().setDefaultSectionSize(42)
        self.sg_nodes_table.setWordWrap(True)
        self.sg_nodes_table.setAlternatingRowColors(True)
        self.sg_nodes_table.itemSelectionChanged.connect(self._on_node_selection_changed)
        self.sg_nodes_table.itemChanged.connect(self._on_node_table_item_changed)
        self.sg_nodes_table.itemDoubleClicked.connect(self._on_node_table_item_double_clicked)
        ne_splitter.addWidget(self.sg_nodes_table)

        self.sg_edges_table = QTableWidget(0, 4)
        self.sg_edges_table.setHorizontalHeaderLabels(["edge_id", "relation", "src_id", "dst_id"])
        self.sg_edges_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sg_edges_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sg_edges_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sg_edges_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.sg_edges_table.horizontalHeader().setFixedHeight(44)
        self.sg_edges_table.verticalHeader().setDefaultSectionSize(42)
        self.sg_edges_table.setWordWrap(True)
        self.sg_edges_table.setAlternatingRowColors(True)
        self.sg_edges_table.itemSelectionChanged.connect(self._on_edge_selection_changed)
        self.sg_edges_table.itemChanged.connect(self._on_edge_table_item_changed)
        ne_splitter.addWidget(self.sg_edges_table)
        sg_table_style = (
            "QTableWidget {"
            " font-size: 13px;"
            " color: #152238;"
            " background: #ffffff;"
            " alternate-background-color: #f6f9ff;"
            " gridline-color: #dbe5f4;"
            " selection-background-color: #eaf3ff;"
            " selection-color: #10233f;"
            "}"
            "QHeaderView::section {"
            " background: #eef4ff;"
            " color: #20324d;"
            " border: 1px solid #d8e4f6;"
            " padding: 4px 6px;"
            " font-weight: 700;"
            "}"
        )
        self.sg_nodes_table.setStyleSheet(sg_table_style)
        self.sg_edges_table.setStyleSheet(sg_table_style)
        ne_splitter.setSizes([1, 1])
        combined_layout.addWidget(ne_splitter, 2)

        edge_btn_row = QHBoxLayout()
        edge_btn_row.setContentsMargins(0, 0, 0, 0)
        edge_btn_row.setSpacing(6)
        self.btn_add_sg_edge = QPushButton("Add Edge")
        self.btn_add_sg_edge.setToolTip("Add a relation between two displayed nodes and save it into scene_graph_bundle.json.")
        self.btn_add_sg_edge.clicked.connect(self._add_scene_graph_edge)
        self.btn_rename_sg_edge = QPushButton("Rename Edge")
        self.btn_rename_sg_edge.setToolTip("Rename relation text of the selected edge.")
        self.btn_rename_sg_edge.clicked.connect(self._rename_selected_scene_graph_edge_relation)
        self.btn_delete_sg_edge = QPushButton("Delete Edge")
        self.btn_delete_sg_edge.setToolTip("Delete the selected edge and overwrite scene_graph_bundle.json.")
        self.btn_delete_sg_edge.clicked.connect(self._delete_selected_scene_graph_edge)
        edge_btn_row.addWidget(self.btn_add_sg_edge)
        edge_btn_row.addWidget(self.btn_rename_sg_edge)
        edge_btn_row.addWidget(self.btn_delete_sg_edge)
        edge_btn_row.addStretch(1)
        combined_layout.addLayout(edge_btn_row)

        self.object_probe_drawer = ObjectProbeDrawer(self)
        self.object_probe_detail = self.object_probe_drawer.detail
        combined_layout.addWidget(self.object_probe_drawer, 1)

        # Human Scene-Graph Edit Queue: keep it visible as a lightweight
        # staging area above the fuller Cycle Review panel.
        self.sg_human_queue_section = QWidget()
        arb_section_layout = QVBoxLayout(self.sg_human_queue_section)
        arb_section_layout.setContentsMargins(0, 0, 0, 0)
        arb_section_layout.setSpacing(6)
        arb_title = QLabel("Human Scene-Graph Edit Queue")
        arb_title.setStyleSheet("font-weight: 700;")
        arb_section_layout.addWidget(arb_title, 0)
        arb_hint = QLabel("Pending manual geometry/claim fixes from Cycle Verify. Select a row to focus the graph, then edit or skip.")
        arb_hint.setWordWrap(True)
        arb_hint.setStyleSheet("color: #667085; font-size: 12px;")
        arb_section_layout.addWidget(arb_hint, 0)
        self.sg_human_queue_table = QTableWidget(0, 8)
        self.sg_human_queue_table.setHorizontalHeaderLabels(["priority", "frame", "claim_id", "type", "nodes", "question", "suggestion", "status"])
        self.sg_human_queue_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sg_human_queue_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sg_human_queue_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sg_human_queue_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.sg_human_queue_table.horizontalHeader().setFixedHeight(34)
        self.sg_human_queue_table.verticalHeader().setDefaultSectionSize(32)
        self.sg_human_queue_table.setAlternatingRowColors(True)
        self.sg_human_queue_table.setStyleSheet(sg_table_style)
        self.sg_human_queue_table.setMinimumHeight(128)
        self.sg_human_queue_table.setMaximumHeight(220)
        self.sg_human_queue_table.itemSelectionChanged.connect(self._on_human_queue_selection_changed)
        arb_section_layout.addWidget(self.sg_human_queue_table, 1)
        self.sg_human_queue_detail = QPlainTextEdit(self.sg_human_queue_section)
        self.sg_human_queue_detail.setReadOnly(True)
        self.sg_human_queue_detail.setPlaceholderText("Queue item details appear here.")
        self.sg_human_queue_detail.setMinimumHeight(88)
        self.sg_human_queue_detail.setMaximumHeight(132)
        self.sg_human_queue_detail.setStyleSheet(sg_table_style)
        arb_section_layout.addWidget(self.sg_human_queue_detail, 0)
        arb_btn_row = QHBoxLayout()
        arb_btn_row.setContentsMargins(0, 0, 0, 0)
        arb_btn_row.setSpacing(6)
        self.btn_human_apply_suggest = QPushButton("Focus For Manual Edit")
        self.btn_human_apply_suggest.clicked.connect(self._apply_human_queue_suggestion)
        self.btn_human_mark_edited = QPushButton("Mark Edited")
        self.btn_human_mark_edited.clicked.connect(lambda: self._set_human_queue_status("edited"))
        self.btn_human_skip = QPushButton("Skip")
        self.btn_human_skip.clicked.connect(lambda: self._set_human_queue_status("skipped"))
        arb_btn_row.addWidget(self.btn_human_apply_suggest)
        arb_btn_row.addWidget(self.btn_human_mark_edited)
        arb_btn_row.addWidget(self.btn_human_skip)
        arb_btn_row.addStretch(1)
        arb_section_layout.addLayout(arb_btn_row)
        self.sg_human_queue_section.setVisible(True)
        combined_layout.addWidget(self.sg_human_queue_section, 0)

        main_splitter.addWidget(combined_host)

        # Keep JSON preview available for internal/debug text updates, but do not
        # spend visible workspace on a mostly-empty raw JSON box.
        self.sg_json_preview = QPlainTextEdit(panel)
        self.sg_json_preview.setReadOnly(True)
        self.sg_json_preview.hide()

        main_splitter.setChildrenCollapsible(False)
        main_splitter.setSizes([96, 820])
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 9)

        layout.addWidget(main_splitter, 1)

        self.sg_cycle_review_group = QGroupBox("Cycle Review")
        sg_cycle_review_layout = QVBoxLayout(self.sg_cycle_review_group)
        self.sg_cycle_tabs = QTabWidget()
        sg_cycle_review_layout.addWidget(self.sg_cycle_tabs)

        cycle_queue_tab = QWidget()
        cycle_queue_layout = QVBoxLayout(cycle_queue_tab)
        cycle_queue_layout.setContentsMargins(0, 0, 0, 0)
        cycle_queue_layout.setSpacing(8)
        cycle_filter_row = QHBoxLayout()
        cycle_filter_row.addWidget(QLabel("Queue Filter"))
        self.sg_cycle_review_filter = QComboBox()
        self.sg_cycle_review_filter.addItems(
            [
                "All Pending",
                "High Priority",
                "Locked",
                "Memory Adjusted",
                "Labels",
                "Relations",
                "Geometry",
                "Attributes",
                "Existence",
            ]
        )
        self.sg_cycle_review_filter.currentIndexChanged.connect(self._refresh_cycle_review_panel)
        self.sg_cycle_review_filter.currentIndexChanged.connect(self._refresh_cycle_summary)
        cycle_filter_row.addWidget(self.sg_cycle_review_filter)
        cycle_filter_row.addWidget(QLabel("Search"))
        self.sg_cycle_review_search = QLineEdit()
        self.sg_cycle_review_search.setPlaceholderText("Search question, subject, predicate, or value")
        self.sg_cycle_review_search.setClearButtonEnabled(True)
        self.sg_cycle_review_search.textChanged.connect(self._refresh_cycle_review_panel)
        self.sg_cycle_review_search.textChanged.connect(self._refresh_cycle_summary)
        cycle_filter_row.addWidget(self.sg_cycle_review_search, 1)
        cycle_queue_layout.addLayout(cycle_filter_row)

        cycle_queue_splitter = QSplitter(Qt.Horizontal)
        self.sg_cycle_review_list = QListWidget()
        self.sg_cycle_review_list.setAlternatingRowColors(True)
        self.sg_cycle_review_list.currentRowChanged.connect(self._render_cycle_review_detail)
        cycle_queue_splitter.addWidget(self.sg_cycle_review_list)
        self.sg_cycle_review_detail = QPlainTextEdit()
        self.sg_cycle_review_detail.setReadOnly(True)
        cycle_queue_splitter.addWidget(self.sg_cycle_review_detail)
        cycle_queue_splitter.setSizes([220, 320])
        cycle_queue_layout.addWidget(cycle_queue_splitter, 1)

        self.sg_cycle_review_choice_hint = QLabel("Select a structured resolution when options are available.")
        self.sg_cycle_review_choice_hint.setWordWrap(True)
        cycle_queue_layout.addWidget(self.sg_cycle_review_choice_hint)

        cycle_review_controls = QHBoxLayout()
        cycle_review_controls.addWidget(QLabel("Suggested Choice"))
        self.sg_cycle_review_choice_combo = QComboBox()
        self.sg_cycle_review_choice_combo.currentIndexChanged.connect(self._refresh_cycle_review_action_controls)
        self.sg_cycle_review_choice_combo.currentIndexChanged.connect(self._refresh_cycle_review_visual_preview)
        cycle_review_controls.addWidget(self.sg_cycle_review_choice_combo, 1)
        self.btn_cycle_review_use_suggested = QPushButton("Use Suggested")
        self.btn_cycle_review_use_suggested.clicked.connect(self._use_cycle_review_suggested_choice)
        self.btn_cycle_review_clear_choice = QPushButton("Clear")
        self.btn_cycle_review_clear_choice.clicked.connect(self._clear_cycle_review_resolution)
        cycle_review_controls.addWidget(self.btn_cycle_review_use_suggested)
        cycle_review_controls.addWidget(self.btn_cycle_review_clear_choice)
        cycle_queue_layout.addLayout(cycle_review_controls)

        cycle_review_controls = QHBoxLayout()
        cycle_review_controls.addWidget(QLabel("Manual Override"))
        self.sg_cycle_review_corrected_value = QLineEdit()
        self.sg_cycle_review_corrected_value.setPlaceholderText("Optional corrected label / relation / value")
        self.sg_cycle_review_corrected_value.textChanged.connect(self._refresh_cycle_review_action_controls)
        cycle_review_controls.addWidget(self.sg_cycle_review_corrected_value, 1)
        self.btn_cycle_review_confirm = QPushButton("Confirm")
        self.btn_cycle_review_confirm.clicked.connect(lambda: self._set_cycle_review_decision(True))
        self.btn_cycle_review_reject = QPushButton("Reject")
        self.btn_cycle_review_reject.clicked.connect(lambda: self._set_cycle_review_decision(False))
        self.btn_cycle_review_refresh = QPushButton("Queue")
        self.btn_cycle_review_refresh.clicked.connect(self._refresh_cycle_review_panel)
        self.btn_cycle_memory_import = QPushButton("Import Memory")
        self.btn_cycle_memory_import.clicked.connect(self._import_cycle_memory)
        self.btn_cycle_memory_export = QPushButton("Export Memory")
        self.btn_cycle_memory_export.clicked.connect(self._export_cycle_memory)
        cycle_review_controls.addWidget(self.btn_cycle_review_confirm)
        cycle_review_controls.addWidget(self.btn_cycle_review_reject)
        cycle_review_controls.addWidget(self.btn_cycle_review_refresh)
        cycle_review_controls.addWidget(self.btn_cycle_memory_import)
        cycle_review_controls.addWidget(self.btn_cycle_memory_export)
        self.btn_cycle_session_export = QPushButton("Export Session")
        self.btn_cycle_session_export.clicked.connect(self._export_cycle_session)
        cycle_review_controls.addWidget(self.btn_cycle_session_export)
        cycle_queue_layout.addLayout(cycle_review_controls)
        self.sg_cycle_memory_summary = QLabel("")
        self.sg_cycle_memory_summary.setWordWrap(True)
        cycle_queue_layout.addWidget(self.sg_cycle_memory_summary)
        self.sg_cycle_tabs.addTab(cycle_queue_tab, "Queue")

        cycle_analytics_tab = QWidget()
        cycle_analytics_layout = QVBoxLayout(cycle_analytics_tab)
        cycle_analytics_layout.setContentsMargins(0, 0, 0, 0)
        self.sg_cycle_analytics_text = QPlainTextEdit()
        self.sg_cycle_analytics_text.setReadOnly(True)
        cycle_analytics_layout.addWidget(self.sg_cycle_analytics_text)
        self.sg_cycle_tabs.addTab(cycle_analytics_tab, "Analytics")

        cycle_session_tab = QWidget()
        cycle_session_layout = QVBoxLayout(cycle_session_tab)
        cycle_session_layout.setContentsMargins(0, 0, 0, 0)
        self.sg_cycle_session_text = QPlainTextEdit()
        self.sg_cycle_session_text.setReadOnly(True)
        cycle_session_layout.addWidget(self.sg_cycle_session_text)
        self.sg_cycle_tabs.addTab(cycle_session_tab, "Session")
        layout.addWidget(self.sg_cycle_review_group)

        self.sg_val_group = QGroupBox("Validation Changes")
        sg_val_layout = QVBoxLayout(self.sg_val_group)
        self.sg_change_list = QListWidget()
        sg_val_layout.addWidget(self.sg_change_list)
        self.sg_val_group.hide()

        self.task_stack.addWidget(panel)

    def _build_single_turn_panel(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Fixed toolbar at top
        toolbar = QHBoxLayout()
        self.btn_gen_single = QPushButton("Generate Single-turn VQA")
        self.btn_gen_single.clicked.connect(self._generate_single_turn)
        self.btn_save_single = self._make_panel_toolbar_button(
            QStyle.SP_DialogSaveButton,
            "Save Single-turn JSON",
            self._save_single_turn,
        )
        self.btn_refresh_single = self._make_panel_toolbar_button(
            QStyle.SP_BrowserReload,
            "Refresh Display",
            lambda: self._render_single_detail(self.single_list.currentRow()),
        )
        self.btn_single_mark_resolved = self._make_panel_toolbar_button(
            QStyle.SP_DialogApplyButton,
            "Mark selected single-turn claim as resolved",
            self._mark_selected_single_probe_resolved,
        )
        toolbar.addWidget(self.btn_gen_single)
        toolbar.addWidget(self.btn_save_single)
        toolbar.addWidget(self.btn_refresh_single)
        toolbar.addWidget(self.btn_single_mark_resolved)
        single_actions_btn, single_actions_menu = self._make_actions_menu_button("Actions")
        action = single_actions_menu.addAction("Import Single-turn")
        action.triggered.connect(self._import_single_turn_annotation)
        # Hide raw-JSON edit entry for end users in the simplified UI flow.
        single_actions_menu.addSeparator()
        action = single_actions_menu.addAction("Import Validation Log")
        action.triggered.connect(lambda: self._import_validation_log_for("Single-turn VQA"))
        action = single_actions_menu.addAction("Export Validation Log")
        action.triggered.connect(lambda: self._export_validation_log_for("Single-turn VQA"))
        action = single_actions_menu.addAction("Export Final")
        action.triggered.connect(self._export_single_turn_final_confirmed)
        action = single_actions_menu.addAction("Export Bundle")
        action.triggered.connect(lambda: self._export_final_bundle_for("Single-turn VQA"))
        single_actions_menu.addSeparator()
        action = single_actions_menu.addAction("Confirm Selected Change")
        action.triggered.connect(lambda: self._set_change_decision(task_name="Single-turn VQA", list_widget=self.single_change_list, approved=True))
        action = single_actions_menu.addAction("Reject Selected Change")
        action.triggered.connect(lambda: self._set_change_decision(task_name="Single-turn VQA", list_widget=self.single_change_list, approved=False))
        single_actions_menu.addSeparator()
        action = single_actions_menu.addAction("Mark as Resolved")
        action.triggered.connect(self._mark_selected_single_probe_resolved)
        toolbar.addWidget(single_actions_btn)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        content_splitter = QSplitter(Qt.Vertical)
        
        self.single_list = QListWidget()
        self.single_list.setSpacing(6)
        self.single_list.setStyleSheet("QListWidget::item { padding: 6px 8px; }")
        self.single_list.setMouseTracking(True)
        self.single_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.single_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.single_list.currentRowChanged.connect(self._on_single_probe_row_changed)
        self.single_list.itemEntered.connect(self._on_single_probe_item_hovered)
        self.single_list.customContextMenuRequested.connect(
            lambda pos: self._show_probe_context_menu(pos, is_multi=False, source="list")
        )
        content_splitter.addWidget(self.single_list)

        self.single_detail = QPlainTextEdit()
        self.single_detail.setReadOnly(True)
        self.single_detail.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.single_detail.setStyleSheet(
            "QPlainTextEdit { font-size: 14px; line-height: 1.5; padding: 10px; }"
        )
        content_splitter.addWidget(self.single_detail)
        # User request: hide metric color blocks in Single-turn panel.
        self.single_score_strip = self._make_score_stat_strip("Single-turn VQA")
        self.single_score_strip.setVisible(False)

        self.single_claims_table = QTableWidget(0, 7)
        self.single_claims_table.setHorizontalHeaderLabels(["frame", "claim", "question", "answer", "Verification Score", "schema_valid", "note"])
        self.single_claims_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.single_claims_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.single_claims_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.single_claims_table.horizontalHeader().setFixedHeight(28)
        self.single_claims_table.verticalHeader().setDefaultSectionSize(34)
        self.single_claims_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.single_claims_table.itemSelectionChanged.connect(self._on_single_claim_selection_changed)
        self.single_claims_table.customContextMenuRequested.connect(
            lambda pos: self._show_probe_context_menu(pos, is_multi=False, source="table")
        )
        content_splitter.addWidget(self.single_claims_table)

        content_splitter.setChildrenCollapsible(False)
        content_splitter.setSizes([220, 220, 180])
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 3)
        content_splitter.setStretchFactor(2, 3)

        layout.addWidget(content_splitter, 1)

        self.single_val_group = QGroupBox("")
        single_val_layout = QVBoxLayout(self.single_val_group)
        self.single_change_list = QListWidget()
        single_val_layout.addWidget(self.single_change_list)
        layout.addWidget(self.single_val_group)

        self.task_stack.addWidget(panel)

    def _build_multi_turn_panel(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Fixed toolbar at top
        toolbar = QHBoxLayout()
        self.btn_gen_multi = QPushButton("Generate Multi-turn VQA")
        self.btn_gen_multi.clicked.connect(self._generate_multi_turn)
        self.btn_save_multi = self._make_panel_toolbar_button(
            QStyle.SP_DialogSaveButton,
            "Save Multi-turn JSON",
            self._save_multi_turn,
        )
        self.btn_refresh_multi = self._make_panel_toolbar_button(
            QStyle.SP_BrowserReload,
            "Refresh Display",
            lambda: self._render_multi_detail(self.multi_list.currentRow()),
        )
        self.btn_multi_mark_resolved = self._make_panel_toolbar_button(
            QStyle.SP_DialogApplyButton,
            "Mark selected multi-turn claim as resolved",
            self._mark_selected_multi_probe_resolved,
        )
        toolbar.addWidget(self.btn_gen_multi)
        toolbar.addWidget(self.btn_save_multi)
        toolbar.addWidget(self.btn_refresh_multi)
        toolbar.addWidget(self.btn_multi_mark_resolved)
        multi_actions_btn, multi_actions_menu = self._make_actions_menu_button("Actions")
        action = multi_actions_menu.addAction("Import Multi-turn")
        action.triggered.connect(self._import_multi_turn_annotation)
        # Hide raw-JSON edit entry for end users in the simplified UI flow.
        multi_actions_menu.addSeparator()
        action = multi_actions_menu.addAction("Import Validation Log")
        action.triggered.connect(lambda: self._import_validation_log_for("Multi-turn VQA"))
        action = multi_actions_menu.addAction("Export Validation Log")
        action.triggered.connect(lambda: self._export_validation_log_for("Multi-turn VQA"))
        action = multi_actions_menu.addAction("Export Final")
        action.triggered.connect(self._export_multi_turn_final_confirmed)
        action = multi_actions_menu.addAction("Export Bundle")
        action.triggered.connect(lambda: self._export_final_bundle_for("Multi-turn VQA"))
        multi_actions_menu.addSeparator()
        action = multi_actions_menu.addAction("Confirm Selected Change")
        action.triggered.connect(lambda: self._set_change_decision(task_name="Multi-turn VQA", list_widget=self.multi_change_list, approved=True))
        action = multi_actions_menu.addAction("Reject Selected Change")
        action.triggered.connect(lambda: self._set_change_decision(task_name="Multi-turn VQA", list_widget=self.multi_change_list, approved=False))
        multi_actions_menu.addSeparator()
        action = multi_actions_menu.addAction("Mark as Resolved")
        action.triggered.connect(self._mark_selected_multi_probe_resolved)
        toolbar.addWidget(multi_actions_btn)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        content_splitter = QSplitter(Qt.Vertical)
        
        self.multi_list = QListWidget()
        self.multi_list.setSpacing(6)
        self.multi_list.setStyleSheet("QListWidget::item { padding: 6px 8px; }")
        self.multi_list.setMouseTracking(True)
        self.multi_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.multi_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.multi_list.currentRowChanged.connect(self._on_multi_probe_row_changed)
        self.multi_list.itemEntered.connect(self._on_multi_probe_item_hovered)
        self.multi_list.customContextMenuRequested.connect(
            lambda pos: self._show_probe_context_menu(pos, is_multi=True, source="list")
        )
        content_splitter.addWidget(self.multi_list)

        self.multi_detail = QPlainTextEdit()
        self.multi_detail.setReadOnly(True)
        self.multi_detail.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.multi_detail.setStyleSheet(
            "QPlainTextEdit { font-size: 14px; line-height: 1.5; padding: 10px; }"
        )
        content_splitter.addWidget(self.multi_detail)
        # User request: hide metric color blocks in Multi-turn panel.
        self.multi_score_strip = self._make_score_stat_strip("Multi-turn VQA")
        self.multi_score_strip.setVisible(False)

        self.multi_claims_table = QTableWidget(0, 7)
        self.multi_claims_table.setHorizontalHeaderLabels(["frame", "claim", "question", "answer", "Verification Score", "schema_valid", "note"])
        self.multi_claims_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.multi_claims_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.multi_claims_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.multi_claims_table.horizontalHeader().setFixedHeight(28)
        self.multi_claims_table.verticalHeader().setDefaultSectionSize(34)
        self.multi_claims_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.multi_claims_table.itemSelectionChanged.connect(self._on_multi_claim_selection_changed)
        self.multi_claims_table.customContextMenuRequested.connect(
            lambda pos: self._show_probe_context_menu(pos, is_multi=True, source="table")
        )
        content_splitter.addWidget(self.multi_claims_table)

        content_splitter.setChildrenCollapsible(False)
        content_splitter.setSizes([220, 220, 180])
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 3)
        content_splitter.setStretchFactor(2, 3)

        layout.addWidget(content_splitter, 1)

        self.multi_val_group = QGroupBox("")
        multi_val_layout = QVBoxLayout(self.multi_val_group)
        self.multi_change_list = QListWidget()
        multi_val_layout.addWidget(self.multi_change_list)
        layout.addWidget(self.multi_val_group)

        self.task_stack.addWidget(panel)

    def _build_caption_panel(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        # Fixed toolbar at top
        toolbar = QHBoxLayout()
        self.btn_generate_caption = QPushButton("Generate Caption")
        self.btn_generate_caption.clicked.connect(self._generate_caption)
        self.btn_save_caption = self._make_panel_toolbar_button(
            QStyle.SP_DialogSaveButton,
            "Save Caption TXT",
            self._save_caption,
        )
        self.btn_batch_generate = self._make_panel_toolbar_button(
            QStyle.SP_FileDialogDetailedView,
            "Generate Batch Captions",
            self._caption_generate_batch,
        )
        self.btn_batch_export_jsonl = self._make_panel_toolbar_button(
            QStyle.SP_DirOpenIcon,
            "Export Batch JSONL",
            self._caption_export_batch_jsonl,
        )
        self.btn_batch_export_txt = self._make_panel_toolbar_button(
            QStyle.SP_FileIcon,
            "Export Batch TXT",
            self._caption_export_batch_txt,
        )
        toolbar.addWidget(self.btn_generate_caption)
        toolbar.addWidget(self.btn_save_caption)
        # Temporary product decision: hide segment-batch caption actions from UI.
        self.btn_batch_generate.setVisible(False)
        self.btn_batch_export_jsonl.setVisible(False)
        self.btn_batch_export_txt.setVisible(False)
        caption_actions_btn, caption_actions_menu = self._make_actions_menu_button("Actions")
        action = caption_actions_menu.addAction("Import Caption")
        action.triggered.connect(self._import_caption_annotation)
        action = caption_actions_menu.addAction("Apply Caption Edit")
        action.triggered.connect(self._apply_caption_edit)
        caption_actions_menu.addSeparator()
        action = caption_actions_menu.addAction("Import Validation Log")
        action.triggered.connect(lambda: self._import_validation_log_for("Video Captioning"))
        action = caption_actions_menu.addAction("Export Validation Log")
        action.triggered.connect(lambda: self._export_validation_log_for("Video Captioning"))
        action = caption_actions_menu.addAction("Export Final")
        action.triggered.connect(self._export_caption_final_confirmed)
        action = caption_actions_menu.addAction("Export Bundle")
        action.triggered.connect(lambda: self._export_final_bundle_for("Video Captioning"))
        caption_actions_menu.addSeparator()
        action = caption_actions_menu.addAction("Confirm Selected Change")
        action.triggered.connect(lambda: self._set_change_decision(task_name="Video Captioning", list_widget=self.caption_change_list, approved=True))
        action = caption_actions_menu.addAction("Reject Selected Change")
        action.triggered.connect(lambda: self._set_change_decision(task_name="Video Captioning", list_widget=self.caption_change_list, approved=False))
        toolbar.addWidget(caption_actions_btn)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        # Fixed configuration area
        cfg = QGroupBox("Caption Controls")
        cfg_form = QFormLayout(cfg)

        self.cap_start = QSpinBox()
        self.cap_end = QSpinBox()
        self.cap_start.setRange(0, 0)
        self.cap_end.setRange(0, 0)
        self.cap_style = QComboBox()
        self.cap_style.addItems(["Concise", "Detailed", "Technical"])

        self.btn_cap_from_current = QPushButton("Use Current Frame As Start/End")
        self.btn_cap_from_current.clicked.connect(self._caption_use_current)
        self.btn_batch_add = QPushButton("Add Segment To Batch")
        self.btn_batch_add.clicked.connect(self._caption_add_segment)
        self.btn_batch_remove = QPushButton("Remove Selected Segment")
        self.btn_batch_remove.clicked.connect(self._caption_remove_selected)

        cfg_form.addRow("Start frame", self.cap_start)
        cfg_form.addRow("End frame", self.cap_end)
        cfg_form.addRow("Style", self.cap_style)
        cfg_form.addRow(self.btn_cap_from_current)
        cfg_form.addRow(self.btn_batch_add)
        cfg_form.addRow(self.btn_batch_remove)
        # Keep controls for backward-compatible state, but hide segmented-caption UX.
        self.cap_start.hide()
        self.cap_end.hide()
        self.btn_cap_from_current.hide()
        self.btn_batch_add.hide()
        self.btn_batch_remove.hide()
        layout.addWidget(cfg)

        content_splitter = QSplitter(Qt.Vertical)
        
        self.caption_batch_table = QTableWidget(0, 4)
        self.caption_batch_table.setHorizontalHeaderLabels(["start", "end", "style", "caption"])
        self.caption_batch_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.caption_batch_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.caption_batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.caption_batch_table.itemSelectionChanged.connect(self._caption_batch_selection_changed)
        self.caption_batch_table.setVisible(False)
        content_splitter.addWidget(self.caption_batch_table)

        self.caption_output = QTextEdit()
        self.caption_output.setStyleSheet(
            "QTextEdit { font-size: 14px; line-height: 1.55; padding: 10px; }"
        )
        content_splitter.addWidget(self.caption_output)
        # User request: remove metric color blocks in caption panel.
        self.caption_score_strip = self._make_score_stat_strip("Video Captioning")
        self.caption_score_strip.setVisible(False)

        self.caption_claims_table = QTableWidget(0, 6)
        self.caption_claims_table.setHorizontalHeaderLabels(["claim", "question", "answer", "Verification Score", "schema_valid", "note"])
        self.caption_claims_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.caption_claims_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.caption_claims_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.caption_claims_table.horizontalHeader().setFixedHeight(28)
        self.caption_claims_table.verticalHeader().setDefaultSectionSize(34)
        self.caption_claims_table.setVisible(False)
        content_splitter.addWidget(self.caption_claims_table)

        content_splitter.setChildrenCollapsible(False)
        content_splitter.setSizes([0, 360, 160])
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 3)
        content_splitter.setStretchFactor(2, 2)

        layout.addWidget(content_splitter, 1)

        self.caption_val_group = QGroupBox("")
        caption_val_layout = QVBoxLayout(self.caption_val_group)
        self.caption_change_list = QListWidget()
        caption_val_layout.addWidget(self.caption_change_list)
        layout.addWidget(self.caption_val_group)

        self.task_stack.addWidget(panel)
        
        # Initialize validation groups visibility based on default mode (Annotate)
        self._on_mode_changed(0)  # idx=0 means Annotate mode

    def _set_status(self, text: str, status_type: str = "info") -> None:
        """Set compact status message (non-intrusive; runtime details go to log panel)."""
        message = str(text or "").strip() or "Ready."
        self.status_label.setText(message)

        color_map = {
            "info": "#6B7280",
            "success": "#4B5563",
            "warning": "#9A3412",
            "error": "#B91C1C",
        }
        color = color_map.get(status_type.lower(), "#6B7280")
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 400;")
        if status_type.lower() in {"success", "warning", "error"}:
            self._append_runtime_log(message, level=status_type)

    def _toggle_play_shortcut(self) -> None:
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox)):
            return
        if isinstance(focus, QComboBox) and focus.isEditable():
            return
        self._toggle_play()

    def _make_transport_button(self, icon_enum, tooltip: str, callback) -> QToolButton:
        btn = QToolButton()
        btn.setIcon(self.style().standardIcon(icon_enum))
        btn.setToolTip(tooltip)
        btn.setAutoRaise(False)
        btn.clicked.connect(callback)
        return btn

    def _make_panel_toolbar_button(self, icon_enum, tooltip: str, callback) -> QPushButton:
        """Create standardized panel action button with icon and tooltip."""
        btn = QPushButton()
        btn.setIcon(self.style().standardIcon(icon_enum))
        btn.setToolTip(tooltip)
        btn.setMaximumWidth(40)
        btn.setMinimumWidth(40)
        btn.clicked.connect(callback)
        return btn

    def _make_actions_menu_button(self, title: str = "Actions") -> tuple[QToolButton, QMenu]:
        btn = QToolButton()
        btn.setText(title)
        btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(btn)
        btn.setMenu(menu)
        return btn, menu

    def _make_scoring_group(self, task_name: str) -> QGroupBox:
        group = QGroupBox("Scoring")
        group.setProperty("task_name", task_name)
        group.setMinimumHeight(230)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        row = QHBoxLayout()
        row.addWidget(QLabel("Reference"))
        path_edit = QLineEdit()
        path_edit.setReadOnly(True)
        path_edit.setPlaceholderText("Load a reference JSON/JSONL file to compute visible scores")
        row.addWidget(path_edit, 1)
        btn_load = QPushButton("Load Reference")
        btn_load.clicked.connect(lambda _=False, task=task_name: self._load_score_reference(task))
        row.addWidget(btn_load)
        btn_eval = QPushButton("Compute Score")
        btn_eval.clicked.connect(lambda _=False, task=task_name: self._compute_score_for_task(task))
        row.addWidget(btn_eval)
        layout.addLayout(row)

        summary = QLabel("No score computed yet.")
        summary.setWordWrap(True)
        layout.addWidget(summary)

        cards = QTableWidget(0, 3)
        cards.setHorizontalHeaderLabels(["Metric", "Value", "Score"])
        cards.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        cards.verticalHeader().setVisible(False)
        cards.setEditTriggers(QAbstractItemView.NoEditTriggers)
        cards.setSelectionMode(QAbstractItemView.NoSelection)
        cards.setFocusPolicy(Qt.NoFocus)
        cards.setMinimumHeight(180)
        cards.setMaximumHeight(240)
        layout.addWidget(cards)

        details = QPlainTextEdit()
        details.setReadOnly(True)
        details.setMinimumHeight(110)
        details.setMaximumHeight(160)
        layout.addWidget(details)

        setattr(self, self._score_attr_name(task_name, "path_edit"), path_edit)
        setattr(self, self._score_attr_name(task_name, "summary"), summary)
        setattr(self, self._score_attr_name(task_name, "cards"), cards)
        setattr(self, self._score_attr_name(task_name, "details"), details)
        return group

    def _score_attr_name(self, task_name: str, part: str) -> str:
        mapping = {
            "Video Scene Graph": "sg",
            "Single-turn VQA": "single",
            "Multi-turn VQA": "multi",
            "Video Captioning": "caption",
        }
        prefix = mapping.get(task_name, "score")
        return f"_{prefix}_score_{part}"

    def _score_widgets(self, task_name: str):
        return (
            getattr(self, self._score_attr_name(task_name, "path_edit"), None),
            getattr(self, self._score_attr_name(task_name, "summary"), None),
            getattr(self, self._score_attr_name(task_name, "cards"), None),
            getattr(self, self._score_attr_name(task_name, "details"), None),
        )

    def _load_score_reference(self, task_name: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Load Reference for {task_name}",
            self._repo_root,
            "JSON/JSONL Files (*.json *.jsonl *.ndjson)",
        )
        if not path:
            return
        self._score_references[task_name] = path
        path_edit, summary, _cards, details = self._score_widgets(task_name)
        if path_edit is not None:
            path_edit.setText(path)
        if summary is not None:
            summary.setText(f"Reference loaded: {os.path.basename(path)}")
        if details is not None:
            details.setPlainText("Reference file loaded. Click 'Compute Score' to refresh metrics.")
        self._auto_refresh_score_for_task(task_name)

    def _load_json_like_payload(self, path: str):
        if not path:
            return None
        if path.lower().endswith((".jsonl", ".ndjson")):
            return import_ndjson(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _compute_score_for_task(self, task_name: str) -> None:
        ref_path = str(self._score_references.get(task_name, "")).strip()
        if not ref_path or not os.path.isfile(ref_path):
            QMessageBox.information(self, "Missing Reference", "Load a reference file first.")
            return
        try:
            reference = self._load_json_like_payload(ref_path)
            if task_name == "Video Scene Graph":
                if not isinstance(self.current_graph, dict):
                    raise RuntimeError("No current scene graph is available to score.")
                result = self._ui_feature_service.evaluate_scene_graph_bundle(
                    pred_graph=self.current_graph,
                    gt_graph=reference if isinstance(reference, dict) else {},
                )
            elif task_name == "Single-turn VQA":
                if not self.single_turn_items:
                    raise RuntimeError("No single-turn VQA items are available to score.")
                result = self._ui_feature_service.evaluate_vqa_bundle(
                    pred_items=self.single_turn_items,
                    gt_items=reference if isinstance(reference, list) else [],
                    task_type=task_name,
                )
            elif task_name == "Multi-turn VQA":
                if not self.multi_turn_items:
                    raise RuntimeError("No multi-turn VQA items are available to score.")
                result = self._ui_feature_service.evaluate_vqa_bundle(
                    pred_items=self.multi_turn_items,
                    gt_items=reference if isinstance(reference, list) else [],
                    task_type=task_name,
                )
            else:
                task_changes = self._changes_for_task(task_name)
                result = {
                    "task_type": task_name,
                    "metrics": self._ui_feature_service.summarize_changes(task_changes),
                    "score_summary": {
                        "overall_score": 0.0,
                        "cards": [],
                    },
                }
            self._score_results[task_name] = result
            self._render_score_result(task_name)
            self._set_status(f"Computed score for {task_name}", status_type="success")
        except Exception as exc:
            QMessageBox.critical(self, "Score Failed", f"Failed to compute score:\n{exc}")

    def _refresh_all_score_panels(self) -> None:
        for task_name in self._score_references.keys():
            self._render_score_result(task_name)

    def _auto_refresh_score_for_task(self, task_name: str) -> None:
        ref_path = str(self._score_references.get(task_name, "")).strip()
        if ref_path and os.path.isfile(ref_path):
            self._compute_score_for_task(task_name)

    def _render_score_result(self, task_name: str) -> None:
        path_edit, summary, cards_widget, details = self._score_widgets(task_name)
        if path_edit is not None:
            path_edit.setText(str(self._score_references.get(task_name, "")))
        self._render_score_cards_only(task_name)
        result = self._score_results.get(task_name)
        if not isinstance(result, dict):
            if summary is not None:
                summary.setText("No score computed yet.")
                summary.setStyleSheet("")
            if cards_widget is not None:
                cards_widget.setRowCount(0)
            if details is not None:
                details.setPlainText(self._default_score_hint(task_name))
            return
        score_summary = result.get("score_summary") or {}
        overall = float(score_summary.get("overall_score", 0.0) or 0.0)
        if summary is not None:
            summary.setText(f"Overall score: {overall:.3f}")
            color = "#16A34A" if overall >= 0.8 else "#EA580C" if overall >= 0.5 else "#DC2626"
            summary.setStyleSheet(f"color: {color}; font-weight: 700;")
        metrics = result.get("metrics")
        lines = [f"Task: {task_name}", f"Overall score: {overall:.3f}"]
        if isinstance(metrics, dict):
            lines.append("")
            lines.append("Metrics:")
            for key, value in metrics.items():
                lines.append(f"- {key}: {value}")
        cards = score_summary.get("cards") or []
        if cards_widget is not None:
            cards_widget.setRowCount(len(cards))
            for row_idx, card in enumerate(cards):
                title_item = QTableWidgetItem(str(card.get("title", card.get("metric_id", "metric"))))
                value_item = QTableWidgetItem(str(card.get("value", "")))
                score_val = card.get("score")
                score_item = QTableWidgetItem("" if score_val is None else f"{float(score_val):.3f}")
                if score_val is not None:
                    score_float = float(score_val)
                    color = QColor("#D1FADF") if score_float >= 0.8 else QColor("#FEF0C7") if score_float >= 0.5 else QColor("#FEE4E2")
                    for item in (title_item, value_item, score_item):
                        item.setBackground(color)
                cards_widget.setItem(row_idx, 0, title_item)
                cards_widget.setItem(row_idx, 1, value_item)
                cards_widget.setItem(row_idx, 2, score_item)
            cards_widget.resizeRowsToContents()
        if cards:
            lines.append("")
            lines.append("Score cards:")
            for card in cards:
                lines.append(
                    f"- {card.get('title', card.get('metric_id', 'metric'))}: "
                    f"value={card.get('value')} score={card.get('score')}"
                )
        if details is not None:
            details.setPlainText("\n".join(lines))

    def _default_score_hint(self, task_name: str) -> str:
        if task_name == "Video Scene Graph":
            return "Load a reference scene-graph JSON, then compute entity/relation scores here."
        if task_name in {"Single-turn VQA", "Multi-turn VQA"}:
            return "Load a reference VQA JSON/JSONL file, then compute answer and grounding scores here."
        return "This panel keeps visible validation/scoring status for the current task."

    def _score_card_title_defaults(self, task_name: str) -> List[str]:
        if task_name == "Video Scene Graph":
            return ["Session", "Queue", "Accepted", "Flagged", "Memory Adjusted", "Memory"]
        if task_name == "Single-turn VQA":
            return ["Overall", "Answer", "Grounding", "Coverage", "Consistency", "Robustness"]
        if task_name == "Multi-turn VQA":
            return ["Overall", "Turn Answer", "Dialogue", "Grounding", "Temporal", "Faithfulness"]
        if task_name == "Video Captioning":
            return ["Overall", "CIDEr", "BLEU-4", "METEOR", "ROUGE-L", "Semantic"]
        return ["Overall", "Metric 1", "Metric 2", "Metric 3", "Metric 4", "Metric 5"]

    def _make_score_stat_strip(self, task_name: str) -> QWidget:
        shell = QWidget()
        shell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QGridLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)
        card_specs = [
            ("overall", "#033495", "Overall score for the current task"),
            ("metric_1", "#16233A", "Primary evaluation metric"),
            ("metric_2", "#245B93", "Primary evaluation metric"),
            ("metric_3", "#1E3A8A", "Primary evaluation metric"),
            ("metric_4", "#9A3412", "Primary evaluation metric"),
            ("metric_5", "#B45309", "Primary evaluation metric"),
        ]
        task_widgets = {}
        for index, (key, accent, tooltip) in enumerate(card_specs):
            card, value_label, detail_label = self._make_cycle_stat_card(title="--", accent=accent, tooltip=tooltip)
            layout.addWidget(card, index // 3, index % 3)
            title_label = card.findChildren(QLabel)[0] if card.findChildren(QLabel) else None
            task_widgets[key] = {"frame": card, "title": title_label, "value": value_label, "detail": detail_label}
        for col in range(3):
            layout.setColumnStretch(col, 1)
        for row in range(2):
            layout.setRowStretch(row, 1)
        if not hasattr(self, '_task_score_stat_widgets'):
            self._task_score_stat_widgets = {}
        self._task_score_stat_widgets[task_name] = task_widgets
        return shell

    def _make_stage_score_strip(self) -> QWidget:
        shell = QWidget()
        shell.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout = QGridLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        specs = [
            ("state", "S State", "#B45309", "Validation accuracy of node state attributes"),
            ("target", "T Target", "#0F766E", "Validation accuracy of detected targets"),
            ("attributes", "A Attributes", "#1D4ED8", "Attribute reliability"),
            ("summary", "G Summary", "#BE123C", "Global summary reliability"),
            ("edges", "E Edges", "#7C3AED", "Spatial relation reliability"),
        ]
        widgets: Dict[str, Dict[str, object]] = {}
        for idx, (key, title, accent, tooltip) in enumerate(specs):
            card, value_label, detail_label = self._make_stage_score_card(
                title=title,
                accent=accent,
                tooltip=tooltip,
            )
            layout.addWidget(card, 0, idx)
            widgets[key] = {"value": value_label, "detail": detail_label}
            layout.setColumnStretch(idx, 1)
        self._sg_stage_score_widgets = widgets
        self._render_stage_score_cards({})
        return shell

    def _make_stage_score_card(self, *, title: str, accent: str, tooltip: str = "") -> tuple[QFrame, QLabel, QLabel]:
        card = QFrame()
        card.setMinimumHeight(72)
        card.setMaximumHeight(84)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet(
            "QFrame {"
            f"background: {accent};"
            "border: 1px solid rgba(255, 255, 255, 0.18);"
            "border-radius: 12px;"
            "}"
        )
        if tooltip:
            card.setToolTip(tooltip)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #F8FAFC; font-size: 12px; font-weight: 700;")

        value_label = QLabel("--")
        value_label.setAlignment(Qt.AlignCenter)
        value_label.setStyleSheet("color: #FFFFFF; font-size: 18px; font-weight: 800;")

        detail_label = QLabel("")
        detail_label.hide()

        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return card, value_label, detail_label

    def _set_stage_score_card(self, key: str, value: str, detail: str = "") -> None:
        widgets = dict(getattr(self, "_sg_stage_score_widgets", {}) or {})
        row = dict(widgets.get(key) or {})
        value_label = row.get("value")
        detail_label = row.get("detail")
        if isinstance(value_label, QLabel):
            value_label.setText(str(value))
        if isinstance(detail_label, QLabel):
            detail_label.setText(str(detail))

    def _render_stage_score_cards(self, module_scores: Dict[str, object]) -> None:
        target_acc = float(module_scores.get("target_accuracy", module_scores.get("S_T", 0.0)) or 0.0)
        state_acc = float(module_scores.get("state_accuracy", module_scores.get("S_S", 0.0)) or 0.0)
        score_map = {
            "target": target_acc,
            "attributes": float(module_scores.get("S_A", 0.0) or 0.0),
            "edges": float(module_scores.get("S_E", 0.0) or 0.0),
            "state": state_acc,
            "summary": float(module_scores.get("S_G", 0.0) or 0.0),
        }
        for key, val in score_map.items():
            detail = "Acc" if key in {"state", "target"} else "Auto"
            self._set_stage_score_card(key, f"{max(0.0, min(1.0, float(val))):.3f}", detail)

    def _set_task_score_card(self, task_name: str, key: str, title: str, value: str, detail: str = "") -> None:
        task_widgets = dict(getattr(self, '_task_score_stat_widgets', {}).get(task_name) or {})
        widget = dict(task_widgets.get(key) or {})
        title_label = widget.get('title')
        value_label = widget.get('value')
        detail_label = widget.get('detail')
        if isinstance(title_label, QLabel):
            title_label.setText(str(title))
        if isinstance(value_label, QLabel):
            value_label.setText(str(value))
        if isinstance(detail_label, QLabel):
            detail_label.setText(str(detail))

    def _render_score_cards_only(self, task_name: str) -> None:
        task_widgets = getattr(self, '_task_score_stat_widgets', {}).get(task_name) or {}
        if not task_widgets:
            return
        result = self._score_results.get(task_name)
        if not isinstance(result, dict):
            titles = self._score_card_title_defaults(task_name)
            defaults = [("overall", titles[0], "--", "Load reference")]
            defaults.extend((f"metric_{idx}", titles[idx], "--", "Await score") for idx in range(1, 6))
            for key, title, value, detail in defaults:
                self._set_task_score_card(task_name, key, title, value, detail)
            return
        score_summary = result.get('score_summary') or {}
        overall = float(score_summary.get('overall_score', 0.0) or 0.0)
        cards = list(score_summary.get('cards') or [])
        titles = self._score_card_title_defaults(task_name)
        rows = [("overall", titles[0], f"{overall:.3f}", f"Task: {task_name}")]
        for idx in range(5):
            default_title = titles[idx + 1]
            if idx < len(cards):
                card = dict(cards[idx] or {})
                raw_title = str(card.get('title', card.get('metric_id', '')) or '').strip()
                title = default_title if (not raw_title or raw_title.lower().startswith('metric')) else raw_title
                score_val = card.get('score')
                value = "--" if score_val is None else f"{float(score_val):.3f}"
                detail = f"value {card.get('value', '')}"[:34]
            else:
                title = default_title
                value = "--"
                detail = "No metric available"
            rows.append((f"metric_{idx+1}", title, value, detail))
        for key, title, value, detail in rows:
            self._set_task_score_card(task_name, key, title, value, detail)

    def _on_task_changed(self, idx: int) -> None:
        self.task_stack.setCurrentIndex(max(0, idx))
        self._update_workspace_header()
        self._sync_scene_graph_toolbar_visibility()
        self._apply_scene_graph_overlay_to_player()
        task_name = self._current_task_name()
        if task_name == "Single-turn VQA":
            self._refresh_single_turn_items_for_current_graph_frame(silent=True)
        elif task_name == "Multi-turn VQA":
            self._refresh_multi_turn_items_for_current_context(silent=True)
        elif task_name == "Video Captioning":
            self._render_cycle_caption_feedback()
            if hasattr(self, "caption_output") and self.caption_output is not None:
                if not str(self.caption_output.toPlainText() or "").strip():
                    text = ""
                    if isinstance(self.current_graph_bundle, dict):
                        text = str(self.current_graph_bundle.get("video_level_caption", "") or "").strip()
                    if not text and isinstance(self.current_graph, dict):
                        text = str(
                            self.current_graph.get("summary")
                            or ((self.current_graph.get("metadata") or {}).get("global_semantic_summary"))
                            or ((self.current_graph.get("metadata") or {}).get("global_summary"))
                            or ""
                        ).strip()
                    if text:
                        self.caption_output.setPlainText(text)
        self._refresh_validation_views()

    def _load_video_path(self, path: str) -> bool:
        path = str(path or "").strip()
        if not path:
            return False
        ok = self.player.load(path)
        if not ok:
            QMessageBox.critical(self, "Load Failed", "Unable to open selected video.")
            self._set_status("Video load failed", status_type="error")
            return False

        self.video_path = path
        self._reset_scene_graph_tracking()
        self._sg_manual_keyframes = set()
        self.current_graph = None
        self.current_graph_bundle = None
        self.video_name_label.setText(os.path.basename(path))
        max_frame = max(0, int(self.player.frame_count) - 1)
        self._block_seek_signals = True
        self.seek_slider.setRange(0, max_frame)
        self.seek_slider.setValue(int(self.player.current_frame))
        self.frame_spin.setRange(0, max_frame)
        self.frame_spin.setValue(int(self.player.current_frame))
        self.spin_frame_for_graph.setRange(0, max_frame)
        self._set_graph_frame_selector(int(self.player.current_frame), manual=False)
        self.cap_start.setRange(0, max_frame)
        self.cap_end.setRange(0, max_frame)
        self.cap_start.setValue(0)
        self.cap_end.setValue(max_frame)
        self._block_seek_signals = False

        self._refresh_time_label(int(self.player.current_frame))
        if self._current_task_name() == "Video Scene Graph":
            self._apply_scene_graph_overlay_to_player()
        try:
            pvsg_ref = self._refresh_pvsg_reference_for_video(frame_indices=None)
            self._append_runtime_log(self._summarize_pvsg_reference(pvsg_ref), level="info")
        except Exception as exc:
            self._append_runtime_log(f"PVSG GT preload failed: {exc}", level="warning")
        self._update_workspace_header()
        self._set_status(f"Loaded video: {os.path.basename(path)}", status_type="success")
        return True

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            self._repo_root,
            "Videos (*.mp4 *.avi *.mov *.mkv *.m4v)",
        )
        if not path:
            return
        self._load_video_path(path)

    def _run_new_video_scene_graph_ui(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video For New Run",
            self._repo_root,
            "Videos (*.mp4 *.avi *.mov *.mkv *.m4v)",
        )
        if not path:
            return
        if not self._load_video_path(path):
            return
        self._sg_force_nonpersistent_backend = False
        self._sg_pending_run_mode = "new"
        self._sg_pending_resume_dir = ""
        self._append_runtime_log("New run prepared. Click the check button to start.", level="info")
        self._set_status("New run prepared. Click the check button to start.", status_type="info")

    def _validate_resume_run_folder(self, run_dir: str) -> Tuple[bool, str, Dict[str, str]]:
        folder = os.path.abspath(str(run_dir or "").strip())
        # Accept both new-style run_info.json and legacy run_metadata.json.
        metadata_file = (
            "run_info.json"
            if os.path.isfile(os.path.join(folder, "run_info.json"))
            else "run_metadata.json"
        )
        paths = {
            "run_dir": folder,
            "metadata": os.path.join(folder, metadata_file),
            "bundle": os.path.join(folder, "scene_graph_bundle.json"),
            "checkpoint": os.path.join(folder, "checkpoint.json"),
            "summary": os.path.join(folder, "run_info.json"),
            "stage_compact": os.path.join(folder, "run_info.json"),
            "timing": os.path.join(folder, "timing.jsonl"),
            "runtime": os.path.join(folder, "runtime.log"),
        }
        if not os.path.isdir(folder):
            return False, f"Run folder does not exist: {folder}", paths
        if not os.path.isfile(paths["metadata"]):
            return False, f"Missing {metadata_file}", paths
        if not os.path.isfile(paths["bundle"]):
            return False, "Missing scene_graph_bundle.json", paths
        try:
            with open(paths["metadata"], "r", encoding="utf-8") as f:
                meta = json.load(f)
            with open(paths["bundle"], "r", encoding="utf-8") as f:
                bundle = json.load(f)
        except Exception as exc:
            return False, f"Invalid JSON in run folder: {exc}", paths
        if not isinstance(meta, dict) or not isinstance(bundle, dict):
            return False, "Run logs must be JSON objects", paths
        required_meta = ["video_path", "sampling_plan", "sampled_frame_indices"]
        missing_meta = [k for k in required_meta if k not in meta]
        if missing_meta:
            return False, f"{metadata_file} missing keys: {', '.join(missing_meta)}", paths
        if int(meta.get("schema_version", 0) or 0) < 1:
            return False, f"{metadata_file} missing/invalid schema_version", paths
        if not str(meta.get("video_path", "") or "").strip():
            return False, f"{metadata_file} has empty video_path", paths
        if not isinstance(bundle.get("graphs"), list):
            return False, "scene_graph_bundle.json missing list field: graphs", paths
        if os.path.isfile(paths["checkpoint"]):
            try:
                with open(paths["checkpoint"], "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                if not isinstance(ckpt, dict):
                    return False, "checkpoint.json must be a JSON object", paths
                if "processed_frame_indices" not in ckpt:
                    return False, "checkpoint.json missing key: processed_frame_indices", paths
            except Exception as exc:
                return False, f"Invalid checkpoint.json: {exc}", paths
        return True, "ok", paths

    def _resume_video_scene_graph_from_folder_ui(self) -> None:
        run_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Existing Run Folder",
            os.path.join(self._repo_root, "log"),
        )
        if not run_dir:
            return
        selected = os.path.abspath(str(run_dir))
        ok, message, paths = self._validate_resume_run_folder(selected)
        if not ok and os.path.isdir(selected):
            candidates: List[str] = []
            try:
                for name in os.listdir(selected):
                    child = os.path.join(selected, name)
                    if os.path.isdir(child):
                        candidates.append(os.path.abspath(child))
            except Exception:
                candidates = []
            candidates.sort(reverse=True)
            for cand in candidates:
                ok_c, _, paths_c = self._validate_resume_run_folder(cand)
                if ok_c:
                    selected = cand
                    ok, paths = True, paths_c
                    break
        if not ok:
            QMessageBox.warning(self, "Invalid Run Folder", f"{message}\n\nFolder: {run_dir}")
            self._set_status("Selected run folder is not compliant.", status_type="warning")
            return
        meta: Dict[str, object] = {}
        try:
            with open(paths["metadata"], "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                meta = dict(payload)
        except Exception:
            meta = {}
        participant_id = _sanitize_name_token(meta.get("participant_id", ""), default="")
        if participant_id:
            self._common_settings["participant_id"] = participant_id
            if hasattr(self, "participant_id_input"):
                self.participant_id_input.setText(participant_id)
            self._save_persisted_settings()
        video_path = str(meta.get("video_path", "") or "").strip()
        if not video_path or not os.path.isfile(video_path):
            QMessageBox.warning(self, "Video Missing", f"Video from run metadata not found:\n{video_path}")
            self._set_status("Run folder valid but source video is missing.", status_type="warning")
            return
        if not self._load_video_path(video_path):
            return
        # Preload existing scene graph bundle immediately on resume prepare,
        # so verify/cycle logic can access graph context before rerun starts.
        try:
            bundle = _read_json(paths.get("bundle", ""))
            prepared_bundle, summary_score, summary_count = self._prepare_bundle_for_resume(
                bundle=dict(bundle or {}),
                run_info_path=str(paths.get("summary", "") or ""),
            )
            graphs = [g for g in list((prepared_bundle or {}).get("graphs") or []) if isinstance(g, dict)]
            if graphs:
                self._reset_scene_graph_tracking()
                self.current_graph_bundle = dict(prepared_bundle)
                self._sg_run_dir = str(selected)
                self._sg_job_output_path = str(paths.get("bundle", "") or "")
                self._sg_timing_log_path = str(paths.get("timing", "") or "")
                self._sg_runtime_log_path = str(paths.get("runtime", "") or "")
                self._sg_checkpoint_path = str(paths.get("checkpoint", "") or "")
                self._sg_metadata_path = str(paths.get("metadata", "") or "")
                self._sg_summary_path = str(paths.get("summary", "") or os.path.join(selected, "run_summary.json"))
                self._sg_stage_compact_path = str(paths.get("stage_compact", "") or os.path.join(selected, "run_summary.json"))
                self._sg_oplog_path = os.path.join(selected, "oplog")
                current_frame = int(self.player.current_frame or 0)
                cycle_graphs = [g for g in graphs if self._graph_cycle_payload(g)]
                pick_pool = cycle_graphs if cycle_graphs else graphs
                selected_graph = min(
                    pick_pool,
                    key=lambda graph_obj: abs(int(self._extract_graph_frame_idx(graph_obj) or 0) - current_frame),
                )
                selected_frame = int(self._extract_graph_frame_idx(selected_graph) or 0)
                self.current_graph = dict(selected_graph)
                self._sync_cycle_result_with_current_graph(force=True)
                self._set_graph_frame_selector(selected_frame, manual=(selected_frame != current_frame))
                try:
                    self._seek_frame(int(selected_frame))
                except Exception:
                    pass
                self._render_graph()
                self._append_runtime_log(
                    f"Resume preload: loaded {len(graphs)} graphs from scene_graph_bundle.json",
                    level="info",
                )
                if summary_count > 0:
                    self._append_runtime_log(
                        f"Resume preload: loaded {summary_count} summaries; S_summary={summary_score:.3f}",
                        level="info",
                    )
        except Exception as exc:
            self._append_runtime_log(f"Resume preload skipped: {exc}", level="warning")
        # Resume after cancellation can leave stale persistent SAM server state.
        # Force non-persistent mode for this resumed run to avoid IPC deadlock.
        self._sg_force_nonpersistent_backend = True
        self._append_runtime_log(f"Resume from run session: {selected}", level="info")
        if os.path.isfile(paths["checkpoint"]):
            self._append_runtime_log(f"Using checkpoint: {paths['checkpoint']}", level="info")
        else:
            self._append_runtime_log("No legacy checkpoint.json found; using run_info.json checkpoint state.", level="info")
        self._sg_pending_run_mode = "resume"
        self._sg_pending_resume_dir = str(selected)
        self._sg_oplog_path = os.path.join(selected, "oplog")
        pvsg_ref_path = os.path.join(selected, "pvsg_reference.json")
        if os.path.isfile(pvsg_ref_path):
            self._pvsg_video_reference = _read_json(pvsg_ref_path)
            self._append_runtime_log(self._summarize_pvsg_reference(dict(self._pvsg_video_reference or {})), level="info")
        self._append_oplog("resume_run_prepared", run_dir=selected)
        self._append_runtime_log("Resume prepared. Click the check button to start.", level="info")
        self._set_status("Resume prepared. Click the check button to start.", status_type="info")

    def _open_video_scene_graph_result_ui(self) -> None:
        run_dir = QFileDialog.getExistingDirectory(
            self,
            "Open Existing Run Result",
            os.path.join(self._repo_root, "log"),
        )
        if not run_dir:
            return
        selected = os.path.abspath(str(run_dir))
        ok, message, paths = self._validate_resume_run_folder(selected)
        if not ok and os.path.isdir(selected):
            candidates: List[str] = []
            try:
                for name in os.listdir(selected):
                    child = os.path.join(selected, name)
                    if os.path.isdir(child):
                        candidates.append(os.path.abspath(child))
            except Exception:
                candidates = []
            candidates.sort(reverse=True)
            for cand in candidates:
                ok_c, _, paths_c = self._validate_resume_run_folder(cand)
                if ok_c:
                    selected = cand
                    ok, paths = True, paths_c
                    break
        if not ok:
            QMessageBox.warning(self, "Invalid Run Folder", f"{message}\n\nFolder: {run_dir}")
            self._set_status("Selected run folder is not compliant.", status_type="warning")
            return

        meta = _read_json(paths.get("metadata", ""))
        bundle = _read_json(paths.get("bundle", ""))
        prepared_bundle, summary_score, summary_count = self._prepare_bundle_for_resume(
            bundle=dict(bundle or {}),
            run_info_path=str(paths.get("summary", "") or ""),
        )
        graphs = [g for g in list(prepared_bundle.get("graphs") or []) if isinstance(g, dict)]
        if not graphs:
            QMessageBox.warning(self, "No Graphs", "scene_graph_bundle.json has no graphs.")
            self._set_status("Run folder loaded but no graphs found.", status_type="warning")
            return

        video_path = str(meta.get("video_path", "") or bundle.get("video_path", "") or "").strip()
        if video_path and os.path.isfile(video_path):
            if not self._load_video_path(video_path):
                return
        else:
            QMessageBox.warning(
                self,
                "Video Missing",
                f"Source video from run metadata is missing:\n{video_path}\n\n"
                "Please open the source video first, then open run result again.",
            )
            self._set_status("Run result found, but source video file is missing.", status_type="warning")
            return

        self._reset_scene_graph_tracking()
        self.current_graph_bundle = dict(prepared_bundle)
        self._sg_run_dir = str(selected)
        self._sg_job_output_path = str(paths.get("bundle", "") or "")
        self._sg_timing_log_path = str(paths.get("timing", "") or "")
        self._sg_runtime_log_path = str(paths.get("runtime", "") or "")
        self._sg_checkpoint_path = str(paths.get("checkpoint", "") or "")
        self._sg_metadata_path = str(paths.get("metadata", "") or "")
        self._sg_summary_path = str(paths.get("summary", "") or os.path.join(selected, "run_summary.json"))
        self._sg_stage_compact_path = str(paths.get("stage_compact", "") or os.path.join(selected, "stage_compact.json"))
        self._sg_oplog_path = os.path.join(selected, "oplog")
        pvsg_ref_path = os.path.join(selected, "pvsg_reference.json")
        if os.path.isfile(pvsg_ref_path):
            self._pvsg_video_reference = _read_json(pvsg_ref_path)
            self._append_runtime_log(self._summarize_pvsg_reference(dict(self._pvsg_video_reference or {})), level="info")
        participant_id = _sanitize_name_token(meta.get("participant_id", ""), default="")
        if participant_id:
            self._common_settings["participant_id"] = participant_id
            if hasattr(self, "participant_id_input"):
                self.participant_id_input.setText(participant_id)
            self._save_persisted_settings()

        current_frame = int(self.player.current_frame or 0)
        cycle_graphs = [g for g in graphs if self._graph_cycle_payload(g)]
        pick_pool = cycle_graphs if cycle_graphs else graphs
        selected_graph = min(
            pick_pool,
            key=lambda graph_obj: abs(int(self._extract_graph_frame_idx(graph_obj) or 0) - current_frame),
        )
        selected_frame = int(self._extract_graph_frame_idx(selected_graph) or 0)
        self.current_graph = dict(selected_graph)
        self._sync_cycle_result_with_current_graph(force=True)
        self._set_graph_frame_selector(selected_frame, manual=(selected_frame != current_frame))
        try:
            self._seek_frame(int(selected_frame))
        except Exception:
            pass
        self._render_graph()
        self._append_oplog("open_run_result", run_dir=selected)
        self._append_runtime_log(f"Opened run result: {selected}", level="info")
        if summary_count > 0:
            self._append_runtime_log(
                f"Opened run summary payload: summaries={summary_count}, S_summary={summary_score:.3f}",
                level="info",
            )
        self._set_status(f"Opened run result: {os.path.basename(selected)}", status_type="success")

    def _on_fps_mode_changed(self, idx: int) -> None:
        enabled = int(idx) == 1
        self._common_settings["fps_override_enabled"] = enabled
        self.fps_spin.setEnabled(enabled)
        self._refresh_time_label(int(self.player.current_frame or 0))
        self._save_persisted_settings()

    def _on_fps_value_changed(self, value: float) -> None:
        self._common_settings["fps_override"] = float(value)
        if bool(self._common_settings.get("fps_override_enabled", False)):
            self._refresh_time_label(int(self.player.current_frame or 0))
        self._save_persisted_settings()

    def _on_playback_speed_changed(self, idx: int) -> None:
        if idx < 0:
            return
        try:
            target_speed = float(self.playback_speed_combo.itemData(idx) or 1.0)
        except Exception:
            target_speed = 1.0
        self.player.set_playback_speed(target_speed)
        actual_speed = float(getattr(self.player, "playback_speed", target_speed) or target_speed)
        if abs(actual_speed - target_speed) > 1e-6:
            for i in range(self.playback_speed_combo.count()):
                try:
                    speed = float(self.playback_speed_combo.itemData(i) or 1.0)
                except Exception:
                    continue
                if abs(speed - actual_speed) <= 1e-6:
                    self.playback_speed_combo.blockSignals(True)
                    self.playback_speed_combo.setCurrentIndex(i)
                    self.playback_speed_combo.blockSignals(False)
                    break
        self._set_status(f"Playback speed set to {actual_speed:g}x", status_type="info")

    def _on_mode_changed(self, idx: int) -> None:
        if bool(getattr(self, "_mode_change_guard", False)):
            return
        try:
            current_idx = int(idx)
        except Exception:
            current_idx = 0
        last_idx = int(getattr(self, "_last_mode_index", current_idx))
        if current_idx != last_idx:
            has_outputs = bool(
                isinstance(self.current_graph, dict)
                or bool(self.single_turn_items)
                or bool(self.multi_turn_items)
                or bool(self.caption_batch)
            )
            if has_outputs:
                btn = QMessageBox.question(
                    self,
                    "Save Reminder",
                    "Switching mode may discard unsaved edits.\nPlease save current results first.\n\nContinue switching mode?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if btn != QMessageBox.Yes:
                    self._mode_change_guard = True
                    try:
                        self.mode_combo.setCurrentIndex(last_idx)
                    finally:
                        self._mode_change_guard = False
                    self._append_oplog(
                        "mode_switch_cancelled",
                        from_mode=("Validate" if last_idx == 1 else "Annotate"),
                        to_mode=("Validate" if current_idx == 1 else "Annotate"),
                    )
                    return
            self._append_oplog(
                "mode_switched",
                from_mode=("Validate" if last_idx == 1 else "Annotate"),
                to_mode=("Validate" if current_idx == 1 else "Annotate"),
            )
            self._last_mode_index = current_idx
        is_validate = int(idx) == 1
        self._update_workspace_header()
        if hasattr(self, "single_detail"):
            self.single_detail.setReadOnly(True)
        if hasattr(self, "multi_detail"):
            self.multi_detail.setReadOnly(True)
        if hasattr(self, "caption_output"):
            self.caption_output.setReadOnly(not is_validate)
        
        # Show/hide validation groups based on mode
        if hasattr(self, "sg_val_group"):
            self.sg_val_group.setVisible(False)
        if hasattr(self, "sg_cycle_review_group"):
            self.sg_cycle_review_group.setVisible(is_validate)
        if hasattr(self, "single_val_group"):
            self.single_val_group.setVisible(is_validate)
        if hasattr(self, "multi_val_group"):
            self.multi_val_group.setVisible(is_validate)
        if hasattr(self, "caption_val_group"):
            self.caption_val_group.setVisible(is_validate)
        
        if hasattr(self, "sg_change_list"):
            self._refresh_validation_views()

    def _on_validator_changed(self) -> None:
        self._common_settings["validator_id"] = self.validator_id_input.text().strip()
        self._save_persisted_settings()

    def _on_participant_changed(self) -> None:
        token = _sanitize_name_token(self.participant_id_input.text(), default="")
        self.participant_id_input.setText(token)
        self._common_settings["participant_id"] = token
        self._save_persisted_settings()

    def _require_participant_id(self) -> Optional[str]:
        token = _sanitize_name_token(self._common_settings.get("participant_id", ""), default="")
        if not token and hasattr(self, "participant_id_input"):
            token = _sanitize_name_token(self.participant_id_input.text(), default="")
        if not token:
            QMessageBox.warning(self, "Missing Participant", "Please input Participant ID before starting a run.")
            self._set_status("Participant ID is required before run.", status_type="warning")
            if hasattr(self, "participant_id_input"):
                self.participant_id_input.setFocus()
            return None
        self._common_settings["participant_id"] = token
        if hasattr(self, "participant_id_input"):
            self.participant_id_input.setText(token)
        self._save_persisted_settings()
        return token

    def _on_validation_round_changed(self, value: int) -> None:
        self._common_settings["validation_round"] = int(value)
        self._save_persisted_settings()

    def _is_validate_mode(self) -> bool:
        return int(self.mode_combo.currentIndex()) == 1

    def _current_task_name(self) -> str:
        return str(self.task_combo.currentText()).strip()

    def _task_key(self, task_name: str) -> str:
        mapping = {
            "Video Scene Graph": "scene_graph",
            "Single-turn VQA": "single_turn_vqa",
            "Multi-turn VQA": "multi_turn_vqa",
            "Video Captioning": "video_captioning",
        }
        return mapping.get(task_name, "unknown")

    def _require_validator(self) -> Optional[str]:
        validator = str(self.validator_id_input.text()).strip()
        if not validator:
            QMessageBox.warning(self, "Missing Validator", "Please set Validator ID before validation operations.")
            return None
        return validator

    def _record_change(
        self,
        *,
        task_type: str,
        item_id: str,
        op: str,
        field_path: str,
        before,
        after,
        reason: str = "",
    ) -> None:
        if not self._is_validate_mode():
            return
        validator = self._require_validator()
        if validator is None:
            return
        row = new_change(
            task_type=task_type,
            item_id=item_id,
            op=op,
            field_path=field_path,
            before=before,
            after=after,
            validator_id=validator,
            round_idx=int(self.validation_round_spin.value()),
            reason=reason,
        )
        self._validation_changes.append(row)
        if str(task_type).strip() == "scene_graph" and str(op).strip() != "cycle_refine":
            try:
                before_graph = dict(before or {}) if isinstance(before, dict) else {}
                after_graph = dict(after or {}) if isinstance(after, dict) else {}
                self._mark_cycle_probes_stale_from_graph_delta(
                    before_graph=before_graph,
                    after_graph=after_graph,
                    stale_reason="manual_correction",
                )
            except Exception:
                pass
        self._refresh_validation_views()
        self._save_persisted_settings()

    def _changes_for_task(self, task_name: str) -> List[Dict[str, object]]:
        return filter_by_task(self._validation_changes, self._task_key(task_name))

    def _refresh_validation_views(self) -> None:
        self._refresh_validation_list_for("Video Scene Graph", self.sg_change_list)
        self._refresh_validation_list_for("Single-turn VQA", self.single_change_list)
        self._refresh_validation_list_for("Multi-turn VQA", self.multi_change_list)
        self._refresh_validation_list_for("Video Captioning", self.caption_change_list)
        self._refresh_cycle_review_panel()
        self._refresh_cycle_summary()

    def _refresh_validation_list_for(self, task_name: str, list_widget: QListWidget) -> None:
        list_widget.clear()
        for row in self._changes_for_task(task_name):
            item = QListWidgetItem(summarize(row))
            item.setData(Qt.UserRole, str(row.get("change_id", "")))
            list_widget.addItem(item)

    def _selected_change_id(self, list_widget: QListWidget) -> str:
        row = list_widget.currentRow()
        if row < 0:
            return ""
        item = list_widget.item(row)
        if item is None:
            return ""
        return str(item.data(Qt.UserRole) or "")

    def _set_change_decision(self, *, task_name: str, list_widget: QListWidget, approved: bool) -> None:
        cid = self._selected_change_id(list_widget)
        if not cid:
            QMessageBox.information(self, "No Selection", "Select a change first.")
            return
        self._apply_change_decision(
            task_name=task_name,
            change_id=cid,
            approved=approved,
        )

    def _import_validation_log_for(self, task_name: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Validation Log", self._repo_root, "NDJSON (*.ndjson *.jsonl);;JSON (*.json)")
        if not path:
            return
        try:
            if path.lower().endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    incoming = json.load(f)
                if isinstance(incoming, dict):
                    incoming = [incoming]
            else:
                incoming = import_ndjson(path)
            self._validation_changes = merge_changes(self._validation_changes, list(incoming))
            self._refresh_validation_views()
            self._save_persisted_settings()
            self._set_status(f"Imported validation log: {os.path.basename(path)}", status_type="success")
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", f"Failed to import validation log:\n{exc}")

    def _export_validation_log_for(self, task_name: str) -> None:
        task_changes = self._changes_for_task(task_name)
        if not task_changes:
            QMessageBox.information(self, "No Changes", "No validation changes for this task.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Validation Log", self._repo_root, "NDJSON (*.ndjson)")
        if not path:
            return
        try:
            export_ndjson(task_changes, path)
            self._set_status(f"Exported validation log: {os.path.basename(path)}", status_type="success")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to export validation log:\n{exc}")

    def _confirmed_changes_for(self, task_name: str) -> List[Dict[str, object]]:
        return [
            c
            for c in self._changes_for_task(task_name)
            if str(c.get("status", "")).strip() == "confirmed"
        ]

    def _build_scene_graph_final_confirmed_payload(self) -> Dict[str, object]:
        scene_changes = self._changes_for_task("Video Scene Graph")
        final_graph = json.loads(json.dumps(self.current_graph if isinstance(self.current_graph, dict) else {}))

        confirmed_update_graphs = [
            row
            for row in scene_changes
            if str(row.get("op", "")).strip() == "update_graph"
            and str(row.get("status", "")).strip() == "confirmed"
        ]
        if confirmed_update_graphs:
            after = confirmed_update_graphs[-1].get("after")
            if isinstance(after, dict):
                final_graph = json.loads(json.dumps(after))

        for row in scene_changes:
            status = str(row.get("status", "")).strip()
            if str(row.get("op", "")) == "update_graph":
                continue
            elif str(row.get("op", "")) == "cycle_arbitration" and status in {"confirmed", "rejected"}:
                final_graph = self._apply_cycle_arbitration_change_to_graph(final_graph, row)
        metadata = dict(final_graph.get("metadata") or {})
        if isinstance(self.current_cycle_result, dict):
            runtime = dict((self.current_cycle_result or {}).get("runtime") or {})
            if runtime:
                metadata["cycle_runtime"] = runtime
            memory_summary = dict((self.current_cycle_result or {}).get("memory") or {})
            if memory_summary:
                metadata["cycle_memory_summary"] = memory_summary
        metadata["cycle_pending_review_count"] = len(self._cycle_review_changes())
        metadata["correction_memory_summary"] = self._cycle_memory_stats()
        final_graph["metadata"] = metadata
        return final_graph

    def _build_single_turn_final_confirmed_payload(self) -> List[Dict[str, object]]:
        final_rows = json.loads(json.dumps(self.single_turn_items))
        confirmed = self._confirmed_changes_for("Single-turn VQA")
        for row in confirmed:
            item_id = str(row.get("item_id", ""))
            if not item_id.startswith("single:"):
                continue
            try:
                idx = int(item_id.split(":", 1)[1])
            except Exception:
                continue
            after = row.get("after")
            if 0 <= idx < len(final_rows) and isinstance(after, dict):
                final_rows[idx] = json.loads(json.dumps(after))
        return final_rows

    def _build_multi_turn_final_confirmed_payload(self) -> List[Dict[str, object]]:
        final_rows = json.loads(json.dumps(self.multi_turn_items))
        confirmed = self._confirmed_changes_for("Multi-turn VQA")
        for row in confirmed:
            item_id = str(row.get("item_id", ""))
            if not item_id.startswith("multi:"):
                continue
            try:
                idx = int(item_id.split(":", 1)[1])
            except Exception:
                continue
            after = row.get("after")
            if 0 <= idx < len(final_rows) and isinstance(after, dict):
                final_rows[idx] = json.loads(json.dumps(after))
        return final_rows

    def _build_caption_final_confirmed_payload(self) -> Dict[str, object]:
        final_batch = json.loads(json.dumps(self.caption_batch))
        final_caption_output = str(self.caption_output.toPlainText())
        confirmed = self._confirmed_changes_for("Video Captioning")

        for row in confirmed:
            op = str(row.get("op", ""))
            item_id = str(row.get("item_id", ""))
            if op == "update_caption" and item_id.startswith("segment:"):
                try:
                    idx = int(item_id.split(":", 1)[1])
                except Exception:
                    continue
                if 0 <= idx < len(final_batch):
                    final_batch[idx]["caption"] = row.get("after", final_batch[idx].get("caption", ""))
            elif op == "add_segment":
                after = row.get("after")
                if isinstance(after, dict):
                    final_batch.append(json.loads(json.dumps(after)))
            elif op == "remove_segment" and item_id.startswith("segment:"):
                try:
                    idx = int(item_id.split(":", 1)[1])
                except Exception:
                    continue
                if 0 <= idx < len(final_batch):
                    final_batch.pop(idx)
            elif op == "update_caption" and item_id == "caption_output":
                final_caption_output = str(row.get("after", final_caption_output))

        return {
            "caption_output": final_caption_output,
            "caption_batch": final_batch,
        }

    def _final_payload_for_task(self, task_name: str):
        if task_name == "Video Scene Graph":
            return self._build_scene_graph_final_confirmed_payload()
        if task_name == "Single-turn VQA":
            return self._build_single_turn_final_confirmed_payload()
        if task_name == "Multi-turn VQA":
            return self._build_multi_turn_final_confirmed_payload()
        if task_name == "Video Captioning":
            return self._build_caption_final_confirmed_payload()
        return {}

    def _export_final_bundle_for(self, task_name: str) -> None:
        task_changes = self._changes_for_task(task_name)
        confirmed = self._confirmed_changes_for(task_name)
        final_payload = self._final_payload_for_task(task_name)

        default_name = f"{self._task_key(task_name)}_bundle_round{int(self.validation_round_spin.value())}.zip"
        path, _ = QFileDialog.getSaveFileName(self, "Export Final Bundle", os.path.join(self._repo_root, default_name), "ZIP Files (*.zip)")
        if not path:
            return
        try:
            ndjson_text = "\n".join(json.dumps(x, ensure_ascii=True) for x in task_changes)
            if ndjson_text:
                ndjson_text += "\n"

            manifest = {
                "task_name": task_name,
                "task_key": self._task_key(task_name),
                "exported_at": now_iso(),
                "validator_id": str(self.validator_id_input.text()).strip(),
                "round": int(self.validation_round_spin.value()),
                "total_changes": len(task_changes),
                "confirmed_changes": len(confirmed),
                "rejected_changes": len([x for x in task_changes if str(x.get("status", "")) == "rejected"]),
                "proposed_changes": len([x for x in task_changes if str(x.get("status", "")) == "proposed"]),
                "files": ["final_output.json", "decision_log.ndjson", "manifest.json"],
            }

            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("final_output.json", json.dumps(final_payload, ensure_ascii=True, indent=2))
                zf.writestr("decision_log.ndjson", ndjson_text)
                zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=True, indent=2))

            self._set_status(f"Exported bundle: {os.path.basename(path)}", status_type="success")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to export bundle:\n{exc}")

    def _export_scene_graph_final_confirmed(self) -> None:
        if not isinstance(self.current_graph, dict):
            QMessageBox.information(self, "No Graph", "No scene graph available.")
            return
        final_graph = self._build_scene_graph_final_confirmed_payload()
        path, _ = QFileDialog.getSaveFileName(self, "Export Final Scene Graph", self._repo_root, "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(final_graph, f, ensure_ascii=True, indent=2)
            self._set_status(f"Exported final scene graph: {os.path.basename(path)}", status_type="success")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to export final graph:\n{exc}")

    def _export_single_turn_final_confirmed(self) -> None:
        final_rows = self._build_single_turn_final_confirmed_payload()
        path, _ = QFileDialog.getSaveFileName(self, "Export Final Single-turn VQA", self._repo_root, "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(final_rows, f, ensure_ascii=True, indent=2)
            self._set_status(f"Exported final single-turn VQA: {os.path.basename(path)}", status_type="success")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to export final single-turn VQA:\n{exc}")

    def _export_multi_turn_final_confirmed(self) -> None:
        final_rows = self._build_multi_turn_final_confirmed_payload()
        path, _ = QFileDialog.getSaveFileName(self, "Export Final Multi-turn VQA", self._repo_root, "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(final_rows, f, ensure_ascii=True, indent=2)
            self._set_status(f"Exported final multi-turn VQA: {os.path.basename(path)}", status_type="success")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to export final multi-turn VQA:\n{exc}")

    def _export_caption_final_confirmed(self) -> None:
        payload = self._build_caption_final_confirmed_payload()
        path, _ = QFileDialog.getSaveFileName(self, "Export Final Caption", self._repo_root, "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, indent=2)
            self._set_status(f"Exported final caption package: {os.path.basename(path)}", status_type="success")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to export final caption:\n{exc}")

    def _toggle_play(self) -> None:
        if not self.player.cap:
            QMessageBox.information(self, "No Video", "Open a video first.")
            self._set_status("No video loaded", status_type="warning")
            return
        if self.player.is_playing:
            self.player.pause()
        else:
            self.player.play()

    def _on_play_state_changed(self, is_playing: bool) -> None:
        icon = QStyle.SP_MediaPause if is_playing else QStyle.SP_MediaPlay
        self.btn_play_pause.setIcon(self.style().standardIcon(icon))

    def _step_frame(self, delta: int) -> None:
        if not self.player.cap:
            return
        self.player.pause()
        nxt = int(self.player.current_frame) + int(delta)
        self._seek_frame(nxt)

    def _scene_graph_sampling_step_for_jump(self) -> int:
        # Prefer actual run sampling plan if available.
        try:
            if isinstance(self.current_graph_bundle, dict):
                sampling = dict(self.current_graph_bundle.get("sampling") or {})
                step = int(sampling.get("sampling_every_n_frames", 0) or 0)
                if step > 0:
                    return max(1, int(step))
        except Exception:
            pass
        # Fallback to current UI sampling setting.
        try:
            if hasattr(self, "spin_sg_sampling_every_n_frames"):
                return max(1, int(self.spin_sg_sampling_every_n_frames.value()))
        except Exception:
            pass
        return 20

    def _jump_to_next_scene_graph_sampled_frame(self) -> None:
        if not self.player.cap:
            self._set_status("No video loaded", status_type="warning")
            return
        self.player.pause()
        current = int(self.player.current_frame or 0)
        max_frame = max(0, int(self.player.frame_count) - 1)
        target = max_frame
        sampled_indices: List[int] = []
        try:
            if isinstance(self.current_graph_bundle, dict):
                sampled_indices = [int(x) for x in list(self.current_graph_bundle.get("sampled_frame_indices") or [])]
                if not sampled_indices:
                    graphs = [g for g in list(self.current_graph_bundle.get("graphs") or []) if isinstance(g, dict)]
                    sampled_indices = sorted(
                        {
                            int(self._extract_graph_frame_idx(g) or -1)
                            for g in graphs
                            if int(self._extract_graph_frame_idx(g) or -1) >= 0
                        }
                    )
        except Exception:
            sampled_indices = []
        if sampled_indices:
            nxt = [idx for idx in sampled_indices if int(idx) > current]
            target = int(nxt[0]) if nxt else int(sampled_indices[0])
            target = max(0, min(int(target), max_frame))
        else:
            step = max(1, int(self._scene_graph_sampling_step_for_jump()))
            target = ((current // step) + 1) * step
            if target > max_frame:
                target = max_frame
        self._seek_frame(int(target))
        self._set_graph_frame_selector(int(target), manual=True)

    def _sync_graph_tables_for_current_frame(self) -> None:
        """Keep node/edge tables in sync with current playback frame when exact sampled graph exists."""
        if self._current_task_name() != "Video Scene Graph":
            return
        if not isinstance(self.current_graph_bundle, dict):
            return
        current_frame = int(getattr(self.player, "current_frame", 0) or 0)
        graphs = [g for g in list(self.current_graph_bundle.get("graphs") or []) if isinstance(g, dict)]
        if not graphs:
            return
        matched: Optional[Dict[str, object]] = None
        for g in graphs:
            if int(self._extract_graph_frame_idx(g) or -1) == current_frame:
                matched = g
                break
        if not isinstance(matched, dict):
            return
        current_graph_idx = int(self._extract_graph_frame_idx(self.current_graph) or -1) if isinstance(self.current_graph, dict) else -1
        if current_graph_idx == current_frame:
            return
        self.current_graph = dict(matched)
        self._render_graph()

    def _seek_frame(self, frame_idx: int) -> None:
        if not self.player.cap:
            return
        max_frame = max(0, int(self.player.frame_count) - 1)
        frame_idx = max(0, min(int(frame_idx), max_frame))
        if self._tracking_enabled() and self._sg_last_tracking_frame >= 0 and abs(int(frame_idx) - int(self._sg_last_tracking_frame)) > 180:
            self._reset_scene_graph_tracking()
        self.player.seek(frame_idx)
        self._sync_seek_controls(frame_idx)
        self._maybe_sync_graph_frame_selector(frame_idx)
        self._advance_scene_graph_tracking(int(frame_idx))
        self._apply_scene_graph_overlay_to_player()
        self._sync_graph_tables_for_current_frame()
        if self._current_task_name() == "Single-turn VQA":
            self._refresh_single_turn_items_for_current_graph_frame(silent=True)
        elif self._current_task_name() == "Multi-turn VQA":
            self._refresh_multi_turn_items_for_current_context(silent=True)
        self._refresh_claim_verification_tables()
        self._render_object_probe_drawer("")

    def _on_slider_pressed(self) -> None:
        self._dragging_slider = True

    def _on_slider_released(self) -> None:
        self._dragging_slider = False
        self._seek_frame(self.seek_slider.value())

    def _on_slider_moved(self, value: int) -> None:
        if self._block_seek_signals:
            return
        # Scrub immediately so users can drag forward and backward with direct feedback.
        self._seek_frame(int(value))

    def _on_slider_value_changed(self, value: int) -> None:
        if self._block_seek_signals:
            return
        if self._dragging_slider:
            self._refresh_time_label(int(value))

    def _on_frame_spin_changed(self, value: int) -> None:
        if self._block_seek_signals:
            return
        self._seek_frame(int(value))

    def _on_frame_advanced(self, frame: int) -> None:
        self._sync_seek_controls(int(frame))
        self._maybe_sync_graph_frame_selector(int(frame))
        self._advance_scene_graph_tracking(int(frame))
        self._apply_scene_graph_overlay_to_player()
        self._sync_graph_tables_for_current_frame()
        self._refresh_claim_verification_tables()

    def _sync_seek_controls(self, frame: int) -> None:
        self._block_seek_signals = True
        self.seek_slider.setValue(int(frame))
        self.frame_spin.setValue(int(frame))
        self._block_seek_signals = False
        self._refresh_time_label(int(frame))

    def _refresh_time_label(self, frame: int) -> None:
        total = max(0, int(self.player.frame_count) - 1)
        fps = self._effective_fps()
        sec = float(frame) / fps
        mm = int(sec // 60)
        ss = sec - mm * 60
        self.time_label.setText(f"f={frame} / {total}  t={mm:02d}:{ss:05.2f}")

    @staticmethod
    def _format_status_duration(seconds: float) -> str:
        total = max(0, int(round(float(seconds or 0.0))))
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        if hh > 0:
            return f"{hh:02d}:{mm:02d}:{ss:02d}"
        return f"{mm:02d}:{ss:02d}"

    def _effective_fps(self) -> float:
        if bool(self._common_settings.get("fps_override_enabled", False)):
            fps_override = float(self._common_settings.get("fps_override", 30.0))
            fps_min = float(self._common_settings.get("fps_min", 1.0))
            fps_max = float(self._common_settings.get("fps_max", 120.0))
            if fps_max < fps_min:
                fps_min, fps_max = fps_max, fps_min
            return max(0.1, min(max(fps_override, fps_min), fps_max))
        return max(1.0, float(self.player.frame_rate or 30.0))

    def _scene_graph_sampling_fps_from_settings(self, *, source_fps: float, settings: Dict[str, object]) -> float:
        every_n = int(settings.get("video_sampling_every_n_frames", 0) or 0)
        if every_n <= 0:
            # Backward compatibility with old FPS-based setting.
            fallback = float(settings.get("video_generation_fps", 1.0) or 1.0)
            return max(0.1, fallback)
        return max(0.1, float(source_fps) / float(max(1, every_n)))

    def _set_graph_frame_to_current(self) -> None:
        self._set_graph_frame_selector(int(self.player.current_frame or 0), manual=False)

    def _add_current_frame_as_video_keyframe(self) -> None:
        frame_idx = int(self.player.current_frame or 0)
        self._sg_manual_keyframes.add(frame_idx)
        self._set_graph_frame_selector(frame_idx, manual=False)
        self._set_status(
            f"Added keyframe {frame_idx} for full-video scene graph generation.",
            status_type="success",
        )

    def _build_video_sampling_indices(self, sampling_fps: float) -> Tuple[List[int], Dict[str, object]]:
        source_fps = self._effective_fps()
        settings = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        every_n = max(1, int(settings.get("video_sampling_every_n_frames", 30) or 30))
        sampling_mode = str(settings.get("video_sampling_mode", "uniform_fps") or "uniform_fps").strip().lower()
        sampled = sample_frame_indices(
            int(self.player.frame_count or 0),
            source_fps,
            max(0.1, float(sampling_fps or 1.0)),
            include_last_frame=True,
        )
        if not sampled:
            return [], {
                "mode": sampling_mode,
                "source_fps": float(source_fps),
                "sampling_fps": float(sampling_fps),
                "sampling_every_n_frames": int(every_n),
                "interval_frames": 0.0,
                "sampled_count": 0,
            }

        interval_frames = float(source_fps) / float(max(0.1, float(sampling_fps or 1.0)))
        merged = set(int(x) for x in sampled)
        max_frame = max(merged)

        if sampling_mode in {"stratified_random", "random_stratified"} and len(sampled) > 2:
            seed = int(settings.get("video_sampling_seed", 42) or 42)
            jitter_ratio = max(0.0, min(0.49, float(settings.get("video_sampling_jitter_ratio", 0.35) or 0.35)))
            rng = random.Random(seed)
            base = sorted(merged)
            jittered = {base[0], base[-1]}
            jitter_half = max(0, int(round(interval_frames * jitter_ratio)))
            for idx in base[1:-1]:
                delta = rng.randint(-jitter_half, jitter_half) if jitter_half > 0 else 0
                j = max(0, min(max_frame, int(idx) + int(delta)))
                jittered.add(j)
            merged = jittered

        for frame_idx in list(self._sg_manual_keyframes):
            if 0 <= int(frame_idx) <= int(max_frame):
                merged.add(int(frame_idx))
        out = sorted(merged)

        max_frames = int(settings.get("video_sampling_max_frames", 3) or 3)
        if max_frames > 0 and len(out) > max_frames:
            keep = [out[0], out[-1]]
            interior = out[1:-1]
            if interior and max_frames > 2:
                stride = max(1, int(round(float(len(interior)) / float(max_frames - 2))))
                keep.extend(interior[::stride])
            out = sorted(set(keep))[:max_frames]
            if out[-1] != max(merged):
                out[-1] = max(merged)
                out = sorted(set(out))

        plan = {
            "mode": sampling_mode,
            "source_fps": float(source_fps),
            "sampling_fps": float(sampling_fps),
            "sampling_every_n_frames": int(every_n),
            "interval_frames": round(interval_frames, 4),
            "manual_keyframes": sorted(int(x) for x in self._sg_manual_keyframes),
            "sampled_count": int(len(out)),
        }
        return out, plan

    def _set_graph_frame_selector(self, frame: int, *, manual: bool) -> None:
        self._block_graph_frame_sync = True
        try:
            self.spin_frame_for_graph.setValue(int(frame))
        finally:
            self._block_graph_frame_sync = False
        self._graph_frame_manual = bool(manual)

    def _maybe_sync_graph_frame_selector(self, frame: int) -> None:
        if self._graph_frame_manual:
            return
        self._set_graph_frame_selector(int(frame), manual=False)

    def _on_graph_frame_selector_changed(self, value: int) -> None:
        if self._block_graph_frame_sync:
            return
        current_frame = int(self.player.current_frame or 0)
        self._graph_frame_manual = int(value) != current_frame
        if isinstance(self.current_cycle_result, dict):
            self._render_cycle_probe_outputs(list(self.current_cycle_result.get("probe_results") or []))
        else:
            self._render_cycle_probe_outputs([])
        self._refresh_claim_verification_tables()
        self._render_cycle_caption_feedback()

    def _reset_scene_graph_tracking(self) -> None:
        self._sg_tracks = {}
        self._sg_next_track_id = 1
        self._sg_last_tracking_frame = -1
        self._sg_last_detection_frame = -1

    def _scene_graph_runtime_ontology(self) -> Optional[Dict[str, object]]:
        return self._custom_ontology

    def _tracking_settings(self) -> Dict[str, object]:
        return dict(self._task_settings.get("Video Scene Graph", {}) or {})

    def _tracking_enabled(self) -> bool:
        # Legacy bbox/template tracking is removed.
        return False

    def _relation_vocab_for_tracking(self) -> Dict[str, List[str]]:
        payload: Dict[str, object] = {}
        if isinstance(self._custom_ontology, dict):
            payload = self._custom_ontology
        else:
            try:
                with open(self._ontology_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    payload = obj
            except Exception:
                payload = {}
        relation_vocab = payload.get("relation_vocabulary") or {}
        if not isinstance(relation_vocab, dict):
            return {"spatial": [], "interaction": []}
        out: Dict[str, List[str]] = {}
        for key in ("spatial", "interaction"):
            value = relation_vocab.get(key) or []
            out[key] = [str(x) for x in value if str(x).strip()] if isinstance(value, list) else []
        return out

    def _relation_cfg_for_tracking(self) -> Dict[str, object]:
        try:
            with open(self._pipeline_cfg, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                rel = payload.get("relations") or {}
                if isinstance(rel, dict):
                    return rel
        except Exception:
            pass
        return {"touching_iou_epsilon": 0.02, "pairwise_max": 200}

    def _sam_adapter_path_for_runtime(self, runtime_config: str) -> str:
        _ = runtime_config
        return os.path.join(self._repo_root, "tools", "sam3_infer.py")

    def _auto_external_backend_override(self, sg_settings: Dict[str, object]) -> Dict[str, object]:
        runtime_config = self._sam_runtime_config_from_settings(sg_settings)
        if not runtime_config:
            return {}
        runner_profile = self._sam_runner_profile_from_settings(sg_settings)
        if not runner_profile:
            runner_profile = self._preferred_sam_runner_profile(runtime_config)
        adapter_path = self._sam_adapter_path_for_runtime(runtime_config)
        runner_path = os.path.join(self._repo_root, "tools", "runners", "run_in_env.py")
        if not (os.path.isfile(adapter_path) and os.path.isfile(runner_path)):
            return {}
        runtime_config = self._remap_repo_local_path(runtime_config)
        if not os.path.isfile(runtime_config):
            runtime_config = self._preferred_sam_runtime_config_path()
        return {
            "external_command_args": [
                sys.executable,
                runner_path,
                "--profile",
                runner_profile,
                "--",
                adapter_path,
                "--runtime-config",
                runtime_config,
                "--image_path",
                "{image_path}",
                "--prompt",
                "{prompt}",
                "--stage",
                "{stage}",
                "--max_instances",
                "{max_instances}",
            ],
            "external_batch_command_args": [
                sys.executable,
                runner_path,
                "--profile",
                runner_profile,
                "--",
                adapter_path,
                "--runtime-config",
                runtime_config,
                "--request_json",
                "{request_json_path}",
            ],
        }

    def _scene_graph_backend_override(self, sg_settings: Dict[str, object]) -> Dict[str, object]:
        backend_override: Dict[str, object] = {}
        backend_provider = str(sg_settings.get("backend_provider", "") or "").strip().lower()
        if backend_provider in {"mock", "external_command"}:
            backend_override["provider"] = backend_provider
        args_file = str(sg_settings.get("external_command_args_file", "") or "").strip()
        if args_file:
            backend_override["external_command_args_file"] = args_file
        template = str(sg_settings.get("external_command_template", "") or "").strip()
        if template:
            backend_override["external_command_template"] = template
        if bool(sg_settings.get("disable_backend_cache", False)):
            backend_override["disable_cache"] = True
        timeout_sec = int(sg_settings.get("backend_timeout_sec", 1800) or 1800)
        if timeout_sec > 0:
            backend_override["external_timeout_sec"] = timeout_sec
        if bool(getattr(self, "_sg_force_nonpersistent_backend", False)):
            backend_override["external_use_persistent_process"] = False
            # Resume mode should fail fast on problematic prompts instead of hanging for long time.
            if timeout_sec <= 0 or timeout_sec > 120:
                backend_override["external_timeout_sec"] = 120
        if backend_provider == "external_command" and not args_file and not template:
            backend_override.update(self._auto_external_backend_override(sg_settings))
        return backend_override

    def _track_node_from_graph_node(
        self,
        node: Dict[str, object],
        *,
        entity_id: str,
        bbox: List[int],
        use_mask: bool,
        tracking_score: Optional[float] = None,
    ) -> Dict[str, object]:
        row = dict(node)
        row["entity_id"] = entity_id
        row["track_id"] = entity_id
        row["bbox"] = list(bbox[:4])
        if not use_mask:
            row["mask"] = {"pixels": []}
        if tracking_score is not None:
            row["tracking_score"] = float(tracking_score)
        priority = str(row.get("person_priority_level", row.get("priority", "")) or "").strip()
        if priority:
            row["priority"] = priority
        person_priority = dict(row.get("person_priority") or {})
        if person_priority:
            row["priority_score"] = float(person_priority.get("priority_score", row.get("priority_score", 0.0)) or 0.0)
            row["bbox_area_ratio"] = float(person_priority.get("bbox_area_ratio", row.get("bbox_area_ratio", 0.0)) or 0.0)
        return self._ensure_mask_payload(row)

    @staticmethod
    def _mask_has_pixels(mask: object) -> bool:
        if not isinstance(mask, dict):
            return False
        pixels = mask.get("pixels")
        return isinstance(pixels, list) and len(pixels) > 0

    def _ensure_mask_payload(self, node: Dict[str, object]) -> Dict[str, object]:
        row = dict(node or {})
        if self._mask_has_pixels(row.get("mask")):
            row.setdefault("mask_origin", "segmentation")
            return row
        bbox = list(row.get("bbox") or [0, 0, 0, 0])
        if len(bbox) < 4:
            row["mask"] = {"pixels": []}
            row.setdefault("mask_origin", "none")
            return row
        x, y, w, h = [int(v) for v in bbox[:4]]
        if w <= 0 or h <= 0:
            row["mask"] = {"pixels": []}
            row.setdefault("mask_origin", "none")
            return row
        step = max(3, int(round(max(w, h) / 24.0)))
        pixels: List[List[int]] = []
        for px in range(x, x + w, step):
            for py in range(y, y + h, step):
                pixels.append([int(px), int(py)])
        row["mask"] = {"pixels": pixels}
        row["mask_origin"] = "bbox_fallback"
        return row

    @staticmethod
    def _mask_pixel_set(mask: object) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        if not isinstance(mask, dict):
            return out
        pixels = mask.get("pixels")
        if not isinstance(pixels, list):
            return out
        for item in pixels:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                out.add((int(item[0]), int(item[1])))
            except Exception:
                continue
        return out

    @staticmethod
    def _node_bbox_xywh(node: Dict[str, object]) -> List[int]:
        bbox = list(node.get("bbox") or [0, 0, 0, 0])
        if len(bbox) < 4:
            return [0, 0, 0, 0]
        return [int(bbox[0]), int(bbox[1]), max(0, int(bbox[2])), max(0, int(bbox[3]))]

    def _mask_stats_payload(
        self,
        *,
        node: Dict[str, object],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[Optional[Dict[str, object]], bool]:
        bbox = self._node_bbox_xywh(node)
        px = self._mask_pixel_set(node.get("mask"))
        if px:
            area_pixels = int(len(px))
            sx = 0.0
            sy = 0.0
            min_x = 10**9
            min_y = 10**9
            max_x = -1
            max_y = -1
            for x, y in px:
                sx += float(x)
                sy += float(y)
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            centroid = [round(sx / float(area_pixels), 3), round(sy / float(area_pixels), 3)]
            if bbox[2] <= 0 or bbox[3] <= 0:
                bbox = [int(min_x), int(min_y), int(max(0, max_x - min_x + 1)), int(max(0, max_y - min_y + 1))]
            area_ratio = float(area_pixels) / float(max(1, int(frame_w) * int(frame_h)))
            return {
                "area_pixels": int(area_pixels),
                "area_ratio": round(float(area_ratio), 8),
                "centroid": centroid,
                "bbox": [int(v) for v in bbox[:4]],
                "mask_size": [int(frame_w), int(frame_h)],
            }, True
        if bbox[2] > 0 and bbox[3] > 0:
            area_pixels = int(bbox[2] * bbox[3])
            area_ratio = float(area_pixels) / float(max(1, int(frame_w) * int(frame_h)))
            centroid = [round(float(bbox[0]) + float(bbox[2]) / 2.0, 3), round(float(bbox[1]) + float(bbox[3]) / 2.0, 3)]
            return {
                "area_pixels": int(area_pixels),
                "area_ratio": round(float(area_ratio), 8),
                "centroid": centroid,
                "bbox": [int(v) for v in bbox[:4]],
                "mask_size": [int(frame_w), int(frame_h)],
            }, False
        return None, False

    @staticmethod
    def _mask_shape_descriptors(
        *,
        mask_pixels: set[tuple[int, int]],
        bbox: List[int],
    ) -> Dict[str, Optional[float]]:
        bw = max(0, int(bbox[2])) if len(bbox) >= 3 else 0
        bh = max(0, int(bbox[3])) if len(bbox) >= 4 else 0
        bbox_area = max(1, bw * bh)
        if not mask_pixels:
            return {
                "shape_fill_ratio": None,
                "shape_boundary_density": None,
            }
        area = float(len(mask_pixels))
        fill_ratio = area / float(bbox_area)
        boundary = 0
        for x, y in mask_pixels:
            if (x - 1, y) not in mask_pixels or (x + 1, y) not in mask_pixels or (x, y - 1) not in mask_pixels or (x, y + 1) not in mask_pixels:
                boundary += 1
        boundary_density = float(boundary) / float(max(1.0, area))
        return {
            "shape_fill_ratio": round(float(fill_ratio), 6),
            "shape_boundary_density": round(float(boundary_density), 6),
        }

    @staticmethod
    def _encode_mask_rle_from_pixels(
        pixels: set[tuple[int, int]],
        *,
        bbox: List[int],
    ) -> Dict[str, object]:
        if not pixels:
            return {"size": [0, 0], "counts": []}
        x0, y0, w, h = [int(v) for v in bbox[:4]]
        if w <= 0 or h <= 0:
            xs = [p[0] for p in pixels]
            ys = [p[1] for p in pixels]
            x0 = min(xs)
            y0 = min(ys)
            w = max(xs) - x0 + 1
            h = max(ys) - y0 + 1
        counts: List[int] = []
        run = 0
        cur = 0
        for yy in range(y0, y0 + h):
            for xx in range(x0, x0 + w):
                bit = 1 if (xx, yy) in pixels else 0
                if bit == cur:
                    run += 1
                else:
                    counts.append(int(run))
                    run = 1
                    cur = bit
        counts.append(int(run))
        return {"origin": [int(x0), int(y0)], "size": [int(w), int(h)], "counts": counts}

    def _finalize_graph_masks_for_export(
        self,
        *,
        graphs: List[Dict[str, object]],
        output_path: str,
    ) -> Dict[str, object]:
        settings = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        mode = str(settings.get("mask_export_mode", MASK_EXPORT_MODE) or MASK_EXPORT_MODE).strip().lower()
        if mode not in {"none", "stats_only", "rle", "external_png"}:
            mode = "stats_only"
        ext_dir_name = str(settings.get("mask_external_dir", "masks") or "masks").strip() or "masks"
        base_dir = os.path.dirname(str(output_path or self._repo_root))
        mask_dir = os.path.join(base_dir, ext_dir_name)
        if mode == "external_png":
            try:
                os.makedirs(mask_dir, exist_ok=True)
            except Exception:
                mode = "stats_only"
        total = 0
        exported_files = 0
        for graph in graphs:
            if not isinstance(graph, dict):
                continue
            metadata = dict(graph.get("metadata") or {})
            image_path = str(metadata.get("image_path", "") or "")
            frame_idx = int(metadata.get("graph_frame_idx", 0) or 0)
            frame_w, frame_h = 1, 1
            if image_path and os.path.isfile(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    frame_h, frame_w = img.shape[:2]
            nodes = list(graph.get("nodes") or [])
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                total += 1
                stats, has_real_mask = self._mask_stats_payload(node=node, frame_w=int(frame_w), frame_h=int(frame_h))
                bbox = self._node_bbox_xywh(node)
                bw = max(0, int(bbox[2]))
                bh = max(0, int(bbox[3]))
                aspect_ratio = float(bw) / float(max(1, bh)) if bw > 0 and bh > 0 else 0.0
                px = self._mask_pixel_set(node.get("mask"))
                shape_desc = self._mask_shape_descriptors(mask_pixels=px, bbox=bbox)
                if stats is not None:
                    node["mask_features"] = {
                        "mask_area": int(stats["area_pixels"]),
                        "mask_area_ratio": float(stats["area_ratio"]),
                        "centroid": list(stats["centroid"]),
                        "aspect_ratio": round(float(aspect_ratio), 6),
                        **shape_desc,
                    }
                    node["mask_area"] = int(stats["area_pixels"])
                    node["mask_area_ratio"] = float(stats["area_ratio"])
                    node["centroid"] = list(stats["centroid"])
                    node["aspect_ratio"] = round(float(aspect_ratio), 6)
                else:
                    node["mask_features"] = {
                        "mask_area": 0,
                        "mask_area_ratio": 0.0,
                        "centroid": None,
                        "aspect_ratio": round(float(aspect_ratio), 6),
                        **shape_desc,
                    }
                    node["mask_area"] = 0
                    node["mask_area_ratio"] = 0.0
                    node["centroid"] = None
                    node["aspect_ratio"] = round(float(aspect_ratio), 6)
                if mode == "none":
                    node["mask"] = None
                    node["mask_available"] = bool(has_real_mask)
                    continue
                if stats is None:
                    node["mask"] = None
                    node["mask_available"] = False
                    continue
                if mode == "stats_only":
                    node["mask"] = {
                        "format": "stats_only",
                        "area_pixels": int(stats["area_pixels"]),
                        "area_ratio": float(stats["area_ratio"]),
                        "centroid": list(stats["centroid"]),
                        "bbox": list(stats["bbox"]),
                        "mask_size": list(stats["mask_size"]),
                    }
                    node["mask_available"] = bool(has_real_mask)
                    continue
                if mode == "rle":
                    if not px:
                        node["mask"] = None
                        node["mask_available"] = False
                    else:
                        rle = self._encode_mask_rle_from_pixels(px, bbox=self._node_bbox_xywh(node))
                        node["mask"] = {
                            "format": "rle",
                            "rle": rle,
                            "area_pixels": int(stats["area_pixels"]),
                            "area_ratio": float(stats["area_ratio"]),
                            "centroid": list(stats["centroid"]),
                            "bbox": list(stats["bbox"]),
                            "mask_size": list(stats["mask_size"]),
                        }
                        node["mask_available"] = True
                    continue
                if mode == "external_png":
                    if not px:
                        node["mask"] = None
                        node["mask_available"] = False
                        continue
                    canvas = np.zeros((int(frame_h), int(frame_w)), dtype=np.uint8)
                    for x, y in px:
                        if 0 <= int(x) < int(frame_w) and 0 <= int(y) < int(frame_h):
                            canvas[int(y), int(x)] = 255
                    nid = str(node.get("track_id", node.get("entity_id", "obj")) or "obj").replace(":", "_")
                    fname = f"frame_{frame_idx:06d}_{nid}.png"
                    abs_mask_path = os.path.join(mask_dir, fname)
                    if cv2.imwrite(abs_mask_path, canvas):
                        exported_files += 1
                        rel_mask_path = os.path.relpath(abs_mask_path, base_dir).replace("\\", "/")
                        node["mask"] = {
                            "format": "external_png",
                            "path": rel_mask_path,
                            "area_pixels": int(stats["area_pixels"]),
                            "area_ratio": float(stats["area_ratio"]),
                            "centroid": list(stats["centroid"]),
                            "bbox": list(stats["bbox"]),
                            "mask_size": list(stats["mask_size"]),
                        }
                        node["mask_available"] = True
                    else:
                        node["mask"] = None
                        node["mask_available"] = False
        return {
            "mode": mode,
            "nodes_processed": int(total),
            "external_mask_dir": mask_dir if mode == "external_png" else "",
            "external_files": int(exported_files),
        }

    def _filter_person_detections(
        self,
        detections: List[Dict[str, object]],
        *,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
        settings = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        min_area_ratio = max(0.0, float(settings.get("person_min_area_ratio", 0.0012) or 0.0012))
        min_side = max(1, int(settings.get("person_min_box_side_px", 16) or 16))
        max_person_keep = max(1, int(settings.get("person_max_tracks_per_frame", 6) or 6))
        frame_area = max(1, int(frame_w) * int(frame_h))
        kept: List[Dict[str, object]] = []
        person_pool: List[Tuple[float, Dict[str, object]]] = []
        dropped_tiny = 0
        for row in detections:
            label = str(row.get("canonical_label", row.get("label", "")) or "").strip().lower()
            bbox = list(row.get("bbox") or [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            w = max(0, int(bbox[2]))
            h = max(0, int(bbox[3]))
            area_ratio = float(w * h) / float(frame_area)
            if label == "person":
                person_priority = self._person_priority_features(
                    bbox=[int(v) for v in bbox[:4]],
                    frame_w=int(frame_w),
                    frame_h=int(frame_h),
                    confidence=float(row.get("score", 0.0) or 0.0),
                )
                row["person_priority"] = dict(person_priority)
                if area_ratio < min_area_ratio or w < min_side or h < min_side:
                    dropped_tiny += 1
                    continue
                score = float(row.get("score", 0.0) or 0.0)
                rank = float(person_priority.get("priority_score", 0.0))
                person_pool.append((rank, row))
            else:
                kept.append(row)
        if person_pool:
            person_pool.sort(key=lambda item: item[0], reverse=True)
            kept.extend([item[1] for item in person_pool[:max_person_keep]])
        return kept, {
            "dropped_tiny_person": int(dropped_tiny),
            "person_candidates": int(len(person_pool)),
            "person_kept": int(min(len(person_pool), max_person_keep)),
        }

    def _split_person_priority_groups(
        self,
        detections: List[Dict[str, object]],
        *,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
        settings = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        filtered_min_area_ratio = max(
            0.0,
            float(settings.get("person_filtered_min_area_ratio", settings.get("person_min_area_ratio", PERSON_MIN_AREA_RATIO)) or PERSON_MIN_AREA_RATIO),
        )
        filtered_min_area_abs = max(1, int(settings.get("person_min_bbox_area", PERSON_MIN_BBOX_AREA) or PERSON_MIN_BBOX_AREA))
        filtered_min_w = max(1, int(settings.get("person_min_bbox_width", PERSON_MIN_BBOX_WIDTH) or PERSON_MIN_BBOX_WIDTH))
        filtered_min_h = max(1, int(settings.get("person_min_bbox_height", PERSON_MIN_BBOX_HEIGHT) or PERSON_MIN_BBOX_HEIGHT))
        filtered_min_side = max(
            1,
            int(settings.get("person_filtered_min_box_side_px", min(filtered_min_w, filtered_min_h)) or min(filtered_min_w, filtered_min_h)),
        )
        filtered_min_score = max(0.0, float(settings.get("person_filtered_min_score", PERSON_FILTERED_MIN_SCORE) or PERSON_FILTERED_MIN_SCORE))
        high_min_area_ratio = max(0.0, float(settings.get("person_high_min_area_ratio", PERSON_HIGH_MIN_AREA_RATIO) or PERSON_HIGH_MIN_AREA_RATIO))
        high_center_min_area_ratio = max(
            0.0,
            float(settings.get("person_high_center_min_area_ratio", PERSON_HIGH_CENTER_MIN_AREA_RATIO) or PERSON_HIGH_CENTER_MIN_AREA_RATIO),
        )
        high_center_max_dist_norm = max(
            0.0,
            min(1.0, float(settings.get("person_high_center_max_distance_norm", PERSON_HIGH_CENTER_MAX_DISTANCE_NORM) or PERSON_HIGH_CENTER_MAX_DISTANCE_NORM)),
        )
        high_top_k = max(
            0,
            int(settings.get("person_high_top_k", settings.get("person_priority_topk", PERSON_PRIORITY_TOPK)) or PERSON_PRIORITY_TOPK),
        )
        high_max_per_frame = max(
            1,
            int(settings.get("person_high_max_per_frame", settings.get("person_max_tracks_per_frame", PERSON_HIGH_MAX_PER_FRAME)) or PERSON_HIGH_MAX_PER_FRAME),
        )
        keep_filtered_debug = bool(settings.get("person_filtered_debug_keep", PERSON_FILTERED_DEBUG_KEEP))
        center_bias_weight = float(
            settings.get("person_center_bias_weight", PERSON_CENTER_BIAS_WEIGHT) or PERSON_CENTER_BIAS_WEIGHT
        )

        non_person: List[Dict[str, object]] = []
        candidates: List[Dict[str, object]] = []
        filtered: List[Dict[str, object]] = []
        for row in detections:
            label = str(row.get("canonical_label", row.get("label", "")) or "").strip().lower()
            if label != "person":
                non_person.append(row)
                continue
            bbox = list(row.get("bbox") or [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            w = max(0, int(bbox[2]))
            h = max(0, int(bbox[3]))
            score = float(row.get("score", 0.0) or 0.0)
            person_priority = self._person_priority_features(
                bbox=[int(v) for v in bbox[:4]],
                frame_w=int(frame_w),
                frame_h=int(frame_h),
                confidence=score,
                center_bias_weight=center_bias_weight,
            )
            row["person_priority"] = dict(person_priority)
            if (
                float(person_priority.get("bbox_area_ratio", 0.0)) < filtered_min_area_ratio
                or float(person_priority.get("bbox_area", 0.0)) < float(filtered_min_area_abs)
                or w < filtered_min_w
                or h < filtered_min_h
                or w < filtered_min_side
                or h < filtered_min_side
                or score < filtered_min_score
            ):
                row["person_priority_level"] = "filtered_person"
                row["priority"] = "filtered_person"
                row["priority_score"] = float(person_priority.get("priority_score", 0.0) or 0.0)
                row["bbox_area_ratio"] = float(person_priority.get("bbox_area_ratio", 0.0) or 0.0)
                row["tracking_policy"] = "filtered"
                row["tracking_excluded"] = True
                filtered.append(row)
                continue
            candidates.append(row)

        if not candidates:
            kept_rows = list(non_person)
            if keep_filtered_debug:
                kept_rows.extend(filtered)
            return kept_rows, {
                "person_high_priority": 0,
                "person_low_priority": 0,
                "person_filtered": int(len(filtered)),
            }

        sorted_by_area_idx = sorted(
            range(len(candidates)),
            key=lambda i: float((candidates[i].get("person_priority") or {}).get("bbox_area", 0.0)),
            reverse=True,
        )
        top_k_idx = set(sorted_by_area_idx[:high_top_k]) if high_top_k > 0 else set()
        high_rows: List[Dict[str, object]] = []
        low_rows: List[Dict[str, object]] = []
        for idx, row in enumerate(candidates):
            p = dict(row.get("person_priority") or {})
            area_ratio = float(p.get("bbox_area_ratio", 0.0) or 0.0)
            dist_norm = float(p.get("center_distance_norm", 1.0) or 1.0)
            is_high = (
                idx in top_k_idx
                or area_ratio >= high_min_area_ratio
                or (area_ratio >= high_center_min_area_ratio and dist_norm <= high_center_max_dist_norm)
            )
            if is_high:
                row["person_priority_level"] = "high_priority_person"
                row["priority"] = "high_priority_person"
                row["priority_score"] = float(p.get("priority_score", 0.0) or 0.0)
                row["bbox_area_ratio"] = float(p.get("bbox_area_ratio", 0.0) or 0.0)
                row["tracking_policy"] = "long_term"
                row["tracking_excluded"] = False
                high_rows.append(row)
            else:
                row["person_priority_level"] = "low_priority_person"
                row["priority"] = "low_priority_person"
                row["priority_score"] = float(p.get("priority_score", 0.0) or 0.0)
                row["bbox_area_ratio"] = float(p.get("bbox_area_ratio", 0.0) or 0.0)
                row["tracking_policy"] = "frame_level"
                row["tracking_excluded"] = True
                low_rows.append(row)

        if len(high_rows) > high_max_per_frame:
            high_rows.sort(
                key=lambda r: float((r.get("person_priority") or {}).get("priority_score", 0.0)),
                reverse=True,
            )
            demoted = high_rows[high_max_per_frame:]
            high_rows = high_rows[:high_max_per_frame]
            for row in demoted:
                row["person_priority_level"] = "low_priority_person"
                row["priority"] = "low_priority_person"
                row["tracking_policy"] = "frame_level"
                row["tracking_excluded"] = True
                low_rows.append(row)

        kept_rows = list(non_person) + list(high_rows) + list(low_rows)
        if keep_filtered_debug:
            kept_rows.extend(filtered)
        return kept_rows, {
            "person_high_priority": int(len(high_rows)),
            "person_low_priority": int(len(low_rows)),
            "person_filtered": int(len(filtered)),
        }

    def _frame_only_node_from_detection(self, det: Dict[str, object]) -> Dict[str, object]:
        row = self._ensure_mask_payload(dict(det or {}))
        row["tracking_policy"] = str(row.get("tracking_policy", "frame_level") or "frame_level")
        row["tracking_excluded"] = True
        priority = str(row.get("person_priority_level", row.get("priority", "")) or "").strip()
        if priority:
            row["priority"] = priority
        person_priority = dict(row.get("person_priority") or {})
        if person_priority:
            row["priority_score"] = float(person_priority.get("priority_score", row.get("priority_score", 0.0)) or 0.0)
            row["bbox_area_ratio"] = float(person_priority.get("bbox_area_ratio", row.get("bbox_area_ratio", 0.0)) or 0.0)
        row.pop("track_id", None)
        return row

    @staticmethod
    def _person_priority_features(
        *,
        bbox: List[int],
        frame_w: int,
        frame_h: int,
        confidence: float,
        center_bias_weight: float = PERSON_CENTER_BIAS_WEIGHT,
    ) -> Dict[str, float]:
        x, y, w, h = [int(v) for v in list(bbox[:4])]
        frame_w = max(1, int(frame_w))
        frame_h = max(1, int(frame_h))
        frame_area = max(1, frame_w * frame_h)
        bbox_area = max(0, w) * max(0, h)
        bbox_area_ratio = float(bbox_area) / float(frame_area)
        bbox_center_x = float(x) + (float(w) / 2.0)
        bbox_center_y = float(y) + (float(h) / 2.0)
        img_center_x = float(frame_w) / 2.0
        img_center_y = float(frame_h) / 2.0
        dx = bbox_center_x - img_center_x
        dy = bbox_center_y - img_center_y
        center_distance = float((dx * dx + dy * dy) ** 0.5)
        center_distance_norm = center_distance / float(max(1.0, (img_center_x * img_center_x + img_center_y * img_center_y) ** 0.5))
        center_score = max(0.0, 1.0 - min(1.0, center_distance_norm))
        center_w = max(0.0, min(0.4, float(center_bias_weight)))
        conf_w = 0.05
        size_w = max(0.0, 1.0 - center_w - conf_w)
        # Priority mainly follows size; center gives a configurable small bonus.
        priority_score = (size_w * bbox_area_ratio) + (center_w * center_score) + (conf_w * max(0.0, min(1.0, float(confidence))))
        return {
            "bbox_area": float(bbox_area),
            "bbox_area_ratio": float(bbox_area_ratio),
            "bbox_center_x": float(bbox_center_x),
            "bbox_center_y": float(bbox_center_y),
            "center_distance_to_image_center": float(center_distance),
            "center_distance_norm": float(center_distance_norm),
            "priority_score": float(priority_score),
        }

    def _tracked_graph_from_nodes(
        self,
        nodes: List[Dict[str, object]],
        *,
        frame_idx: int,
        base_graph: Dict[str, object],
    ) -> Dict[str, object]:
        rel_vocab = self._relation_vocab_for_tracking()
        rel_cfg = self._relation_cfg_for_tracking()
        proposals: List[Dict[str, object]] = []
        for node in nodes:
            proposals.append(
                {
                    "entity_id": str(node.get("entity_id", "")),
                    "canonical_label": str(self._node_label(node, "")),
                    "prompt_used": str(node.get("prompt_used", self._node_label(node, "")) or ""),
                    "mask": dict(node.get("mask") or {"pixels": []}),
                    "bbox": list(node.get("bbox") or [0, 0, 0, 0]),
                    "score": float(node.get("score", 0.0) or 0.0),
                    "attributes": list(node.get("attributes") or []),
                    "provenance": list(node.get("provenance") or []),
                    "risk": float(node.get("risk", 0.0) or 0.0),
                    "verified": bool(node.get("verified", False)),
                    "validator_flags": list(node.get("validator_flags") or []),
                }
            )
        graph = build_scene_graph(
            image_id=str(base_graph.get("image_id", "") or "tracked_frame"),
            proposals=proposals,
            relation_vocab=rel_vocab,
            touching_iou_epsilon=float(rel_cfg.get("touching_iou_epsilon", 0.02)),
            pairwise_max=int(rel_cfg.get("pairwise_max", 200)),
            enable_interaction_relations=False,
        )
        node_meta_by_id: Dict[str, Dict[str, object]] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            nid = str(node.get("entity_id", "") or "").strip()
            if not nid:
                continue
            node_meta_by_id[nid] = {
                "track_id": node.get("track_id"),
                "tracking_score": node.get("tracking_score"),
                "tracking_policy": node.get("tracking_policy"),
                "tracking_excluded": node.get("tracking_excluded"),
                "person_priority_level": node.get("person_priority_level"),
                "person_priority": node.get("person_priority"),
                "priority": node.get("priority"),
                "priority_score": node.get("priority_score"),
                "bbox_area_ratio": node.get("bbox_area_ratio"),
                "mask_features": node.get("mask_features"),
                "mask_area": node.get("mask_area"),
                "mask_area_ratio": node.get("mask_area_ratio"),
                "centroid": node.get("centroid"),
                "aspect_ratio": node.get("aspect_ratio"),
                "mask_origin": node.get("mask_origin"),
            }
        for out_node in list(graph.get("nodes") or []):
            if not isinstance(out_node, dict):
                continue
            nid = str(out_node.get("entity_id", "") or "").strip()
            meta = dict(node_meta_by_id.get(nid) or {})
            for key, value in meta.items():
                if value is not None:
                    out_node[key] = value
        metadata = dict(base_graph.get("metadata") or {})
        metadata["graph_frame_idx"] = int(frame_idx)
        metadata["tracking_mode"] = str(self._tracking_settings().get("tracking_mode", "disabled") or "disabled")
        metadata["tracking_track_count"] = int(len(nodes))
        metadata["tracking_last_detection_frame"] = int(self._sg_last_detection_frame)
        graph["metadata"] = metadata
        return graph

    def _apply_tracking_to_built_graph(
        self,
        graph: Dict[str, object],
        *,
        frame_idx: int,
        frame_bgr,
    ) -> Dict[str, object]:
        if not self._tracking_enabled() or frame_bgr is None:
            self._reset_scene_graph_tracking()
            return graph
        frame_h, frame_w = frame_bgr.shape[:2]
        detections: List[Dict[str, object]] = []
        for node in list(graph.get("nodes") or []):
            bbox = clip_bbox(list(node.get("bbox") or [0, 0, 0, 0]), frame_w, frame_h)
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            row = self._ensure_mask_payload(dict(node))
            row["bbox"] = bbox
            det_patch = crop_patch(frame_bgr, bbox)
            appearance_similarity_by_track: Dict[str, float] = {}
            for track_id, track in self._sg_tracks.items():
                if str(track.get("canonical_label", track.get("label", "")) or "").strip().lower() != str(row.get("canonical_label", row.get("label", "")) or "").strip().lower():
                    continue
                track_patch = track.get("template_gray")
                if det_patch is None or track_patch is None:
                    continue
                cur_patch = det_patch
                if getattr(track_patch, "shape", None) != getattr(det_patch, "shape", None):
                    try:
                        cur_patch = cv2.resize(
                            det_patch,
                            (int(track_patch.shape[1]), int(track_patch.shape[0])),
                            interpolation=cv2.INTER_AREA,
                        )
                    except Exception:
                        continue
                try:
                    sim = float(cv2.matchTemplate(cur_patch, track_patch, cv2.TM_CCOEFF_NORMED)[0][0])
                except Exception:
                    continue
                appearance_similarity_by_track[str(track_id)] = max(0.0, min(1.0, (sim + 1.0) * 0.5))
            if appearance_similarity_by_track:
                row["appearance_similarity_by_track"] = appearance_similarity_by_track
            detections.append(row)
        detections, person_filter_stats = self._split_person_priority_groups(
            detections,
            frame_w=frame_w,
            frame_h=frame_h,
        )
        low_mode = str(
            self._tracking_settings().get("low_priority_tracking_mode", PERSON_LOW_PRIORITY_TRACKING_MODE)
            or PERSON_LOW_PRIORITY_TRACKING_MODE
        ).strip().lower()
        low_short_enabled = low_mode in {"short", "short_term", "temporary"}
        low_max_lost = max(
            1,
            int(
                self._tracking_settings().get("low_priority_max_lost_frames", PERSON_LOW_PRIORITY_MAX_LOST_FRAMES)
                or PERSON_LOW_PRIORITY_MAX_LOST_FRAMES
            ),
        )
        if not low_short_enabled:
            for track_id in list(self._sg_tracks.keys()):
                if str(self._sg_tracks.get(track_id, {}).get("person_priority_level", "")).strip().lower() == "low_priority_person":
                    self._sg_tracks.pop(track_id, None)
        demote_policy = str(
            self._tracking_settings().get("person_track_demote_policy", PERSON_TRACK_DEMOTE_POLICY)
            or PERSON_TRACK_DEMOTE_POLICY
        ).strip().lower()

        trackable_det_indices: List[int] = []
        trackable_detections: List[Dict[str, object]] = []
        for det_idx, det in enumerate(detections):
            level = str(det.get("person_priority_level", "") or "").strip().lower()
            if level == "filtered_person":
                continue
            # Include low-priority person detections for possible demotion/termination checks.
            trackable_det_indices.append(det_idx)
            trackable_detections.append(det)

        matches_sub = greedy_track_match(
            self._sg_tracks,
            trackable_detections,
            current_frame_idx=int(frame_idx),
            short_gap_frames=int(self._tracking_settings().get("tracking_short_gap_frames", 24) or 24),
            long_gap_frames=int(self._tracking_settings().get("tracking_long_gap_frames", 120) or 120),
            max_track_gap_frames=int(self._tracking_settings().get("tracking_max_gap_frames", 240) or 240),
            min_match_score_short=float(self._tracking_settings().get("tracking_min_match_score_short", 0.2) or 0.2),
            min_match_score_long=float(self._tracking_settings().get("tracking_min_match_score_long", 0.35) or 0.35),
        )
        matches = [(tid, trackable_det_indices[sub_idx], score) for tid, sub_idx, score in matches_sub]
        matched_track_ids = set()
        matched_det_idx = set()
        visible_nodes: List[Dict[str, object]] = []
        template_alpha = float(self._tracking_settings().get("tracking_template_update_alpha", 0.25) or 0.25)
        max_lost = int(self._tracking_settings().get("tracking_max_lost_frames", 12) or 12)

        for track_id, det_idx, match_score in matches:
            det = detections[det_idx]
            matched_track_ids.add(track_id)
            matched_det_idx.add(det_idx)
            track = dict(self._sg_tracks.get(track_id, {}))
            det_level = str(det.get("person_priority_level", "") or "").strip().lower()
            track_label = str(track.get("canonical_label", track.get("label", "")) or "").strip().lower()
            if track_label == "person" and det_level in {"low_priority_person", "filtered_person"}:
                if demote_policy in {"terminate", "drop", "remove"}:
                    self._sg_tracks.pop(track_id, None)
                    visible_nodes.append(self._frame_only_node_from_detection(det))
                    continue
                track["person_priority_level"] = "low_priority_person"
                track["tracking_policy"] = "frame_level"
                track["max_lost_override"] = int(low_max_lost)
            bbox = clip_bbox(det.get("bbox") or [0, 0, 0, 0], frame_w, frame_h)
            track["bbox"] = smooth_bbox(track.get("bbox") or bbox, bbox, alpha=0.85)
            track["canonical_label"] = str(det.get("canonical_label", det.get("label", "")) or "")
            track["mask"] = dict(det.get("mask") or {"pixels": []})
            track["score"] = float(det.get("score", 0.0) or 0.0)
            track["last_frame"] = int(frame_idx)
            track["misses"] = 0
            track["match_score"] = float(match_score)
            track["person_priority_level"] = str(det.get("person_priority_level", "") or "")
            track["tracking_policy"] = str(det.get("tracking_policy", "long_term") or "long_term")
            track["person_priority"] = dict(det.get("person_priority") or {})
            track["priority"] = str(track.get("person_priority_level", "") or "")
            track["priority_score"] = float((track.get("person_priority") or {}).get("priority_score", 0.0) or 0.0)
            track["bbox_area_ratio"] = float((track.get("person_priority") or {}).get("bbox_area_ratio", 0.0) or 0.0)
            track["appearance_similarity"] = float(
                (det.get("appearance_similarity_by_track") or {}).get(track_id, det.get("appearance_similarity", 0.0)) or 0.0
            )
            track["template_gray"] = update_template(
                track.get("template_gray"),
                crop_patch(frame_bgr, track["bbox"]),
                alpha=template_alpha,
            )
            track["last_node"] = self._track_node_from_graph_node(
                det,
                entity_id=track_id,
                bbox=track["bbox"],
                use_mask=True,
                tracking_score=match_score,
            )
            self._sg_tracks[track_id] = track
            visible_nodes.append(dict(track["last_node"]))

        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_idx:
                continue
            level = str(det.get("person_priority_level", "") or "").strip().lower()
            if level == "filtered_person":
                visible_nodes.append(self._frame_only_node_from_detection(det))
                continue
            if level == "low_priority_person":
                visible_nodes.append(self._frame_only_node_from_detection(det))
                continue
            track_id = f"track_{self._sg_next_track_id:04d}"
            self._sg_next_track_id += 1
            bbox = clip_bbox(det.get("bbox") or [0, 0, 0, 0], frame_w, frame_h)
            track_node = self._track_node_from_graph_node(det, entity_id=track_id, bbox=bbox, use_mask=True, tracking_score=1.0)
            self._sg_tracks[track_id] = {
                "bbox": bbox,
                "canonical_label": str(det.get("canonical_label", det.get("label", "")) or ""),
                "mask": dict(det.get("mask") or {"pixels": []}),
                "score": float(det.get("score", 0.0) or 0.0),
                "last_frame": int(frame_idx),
                "misses": 0,
                "appearance_similarity": 1.0,
                "person_priority_level": str(det.get("person_priority_level", "") or ""),
                "tracking_policy": str(det.get("tracking_policy", "long_term") or "long_term"),
                "person_priority": dict(det.get("person_priority") or {}),
                "priority": str(det.get("person_priority_level", "") or ""),
                "priority_score": float((det.get("person_priority") or {}).get("priority_score", 0.0) or 0.0),
                "bbox_area_ratio": float((det.get("person_priority") or {}).get("bbox_area_ratio", 0.0) or 0.0),
                "max_lost_override": int(low_max_lost) if level == "low_priority_person" else None,
                "template_gray": crop_patch(frame_bgr, bbox),
                "last_node": track_node,
            }
            visible_nodes.append(dict(track_node))

        for track_id in list(self._sg_tracks.keys()):
            if track_id in matched_track_ids:
                continue
            track = self._sg_tracks[track_id]
            track["misses"] = int(track.get("misses", 0) or 0) + 1
            track_max_lost = int(track.get("max_lost_override", max_lost) or max_lost)
            if int(track["misses"]) > track_max_lost:
                self._sg_tracks.pop(track_id, None)

        self._sg_last_detection_frame = int(frame_idx)
        self._sg_last_tracking_frame = int(frame_idx)
        tracked = self._tracked_graph_from_nodes(visible_nodes, frame_idx=frame_idx, base_graph=graph)
        tracked_meta = dict(tracked.get("metadata") or {})
        tracked_meta["person_filter"] = person_filter_stats
        tracked_meta["low_priority_tracking_mode"] = low_mode
        tracked["metadata"] = tracked_meta
        return tracked

    def _advance_scene_graph_tracking(self, frame_idx: int) -> None:
        if not self._tracking_enabled():
            return
        if not self._sg_tracks or not isinstance(self.current_graph, dict):
            return
        if self._extract_graph_frame_idx(self.current_graph) == int(frame_idx):
            return
        if self._sg_last_tracking_frame >= 0 and int(frame_idx) < int(self._sg_last_tracking_frame):
            self._reset_scene_graph_tracking()
            return
        frame_bgr = self.player.get_current_frame_bgr()
        if frame_bgr is None:
            return
        frame_h, frame_w = frame_bgr.shape[:2]
        search_radius = int(self._tracking_settings().get("tracking_search_radius", 72) or 72)
        min_response = float(self._tracking_settings().get("tracking_min_response", 0.35) or 0.35)
        template_alpha = float(self._tracking_settings().get("tracking_template_update_alpha", 0.25) or 0.25)
        max_lost = int(self._tracking_settings().get("tracking_max_lost_frames", 12) or 12)
        visible_nodes: List[Dict[str, object]] = []
        for track_id in list(self._sg_tracks.keys()):
            track = self._sg_tracks.get(track_id, {})
            base_node = dict(track.get("last_node") or {})
            if not base_node:
                continue
            new_bbox, response = template_track_bbox(
                frame_bgr,
                track.get("template_gray"),
                track.get("bbox") or [0, 0, 0, 0],
                search_radius=search_radius,
                min_response=min_response,
            )
            if new_bbox is None:
                track["misses"] = int(track.get("misses", 0) or 0) + 1
                if int(track["misses"]) > max_lost:
                    self._sg_tracks.pop(track_id, None)
                    continue
                base_node = self._track_node_from_graph_node(
                    base_node,
                    entity_id=track_id,
                    bbox=clip_bbox(track.get("bbox") or [0, 0, 0, 0], frame_w, frame_h),
                    use_mask=False,
                )
                visible_nodes.append(base_node)
                continue
            smoothed_bbox = clip_bbox(smooth_bbox(track.get("bbox") or new_bbox, new_bbox, alpha=0.65), frame_w, frame_h)
            track["bbox"] = smoothed_bbox
            track["last_frame"] = int(frame_idx)
            track["misses"] = 0
            track["template_gray"] = update_template(
                track.get("template_gray"),
                crop_patch(frame_bgr, smoothed_bbox),
                alpha=template_alpha,
            )
            track_node = self._track_node_from_graph_node(
                base_node,
                entity_id=track_id,
                bbox=smoothed_bbox,
                use_mask=False,
                tracking_score=response,
            )
            track["last_node"] = track_node
            self._sg_tracks[track_id] = track
            visible_nodes.append(track_node)
        if not visible_nodes:
            return
        self.current_graph = self._tracked_graph_from_nodes(
            visible_nodes,
            frame_idx=int(frame_idx),
            base_graph=self.current_graph,
        )
        self._sg_last_tracking_frame = int(frame_idx)

    def _extract_frame(self, frame_idx: int, *, cap=None) -> Tuple[str, int, int, Any]:
        return _extract_video_frame_to_cache(
            self.video_path,
            frame_idx,
            self._frame_cache_dir,
            cap=cap,
        )

    def _infer_scene_graph_for_frame(
        self,
        *,
        frame_idx: int,
        img_path: str,
        image_size: Tuple[int, int],
        frame_bgr,
        enable_sentence_refine: bool,
    ) -> Dict[str, object]:
        stem = os.path.splitext(os.path.basename(self.video_path))[0]
        image_id = f"{stem}_f{frame_idx:06d}"
        sg_settings = self._task_settings.get("Video Scene Graph", {})
        backend_override = self._scene_graph_backend_override(sg_settings)
        graph = run_build_scene_graph(
            image_id=image_id,
            image_path=img_path,
            ontology_path=self._ontology_path,
            pipeline_cfg_path=self._pipeline_cfg,
            image_size=image_size,
            enable_sentence_refine=bool(enable_sentence_refine),
            custom_ontology_dict=self._scene_graph_runtime_ontology(),
            pipeline_cfg_override={"backend": backend_override} if backend_override else None,
        )
        graph = self._apply_tracking_to_built_graph(
            graph,
            frame_idx=frame_idx,
            frame_bgr=frame_bgr,
        )
        metadata = dict(graph.get("metadata") or {})
        metadata["graph_frame_idx"] = int(frame_idx)
        metadata["graph_time_sec"] = round(float(frame_idx) / float(self._effective_fps()), 6)
        graph["metadata"] = metadata
        return graph

    def _finalize_built_scene_graph(
        self,
        *,
        graph: Dict[str, object],
        frame_idx: int,
        image_path: str,
    ) -> Dict[str, object]:
        frame_bgr = cv2.imread(image_path) if image_path and os.path.isfile(image_path) else None
        graph = self._apply_tracking_to_built_graph(
            graph,
            frame_idx=int(frame_idx),
            frame_bgr=frame_bgr,
        )
        metadata = dict(graph.get("metadata") or {})
        metadata["graph_frame_idx"] = int(frame_idx)
        metadata["graph_time_sec"] = round(float(frame_idx) / float(self._effective_fps()), 6)
        graph["metadata"] = metadata
        return graph

    def _cancel_scene_graph_job(self) -> None:
        sg_worker = self._sg_worker
        llm_worker = self._llm_summary_worker
        if sg_worker is None and llm_worker is None:
            return
        if sg_worker is not None:
            sg_worker.cancel()
            if self._sg_progress_dialog is not None:
                self._sg_progress_phase = "Cancelling"
                label = str(self._sg_progress_dialog.labelText() or "").strip()
                if "Cancelling" not in label:
                    self._update_scene_graph_progress_dialog_label()
        if llm_worker is not None:
            llm_worker.cancel()
            self._sg_progress_phase = "Cancelling"
            self._update_scene_graph_progress_dialog_label()
        self._write_run_checkpoint(stage="cancelling", interrupted=True)
        self._write_run_metadata(status="cancelling")
        self._append_runtime_log("Stop requested for current running job.", level="warning")
        self._set_status("Cancellation requested; stopping the current run.", status_type="warning")

    def _close_scene_graph_progress_dialog(self) -> None:
        self._sg_progress_pin_timer.stop()
        dlg = self._sg_progress_dialog
        self._sg_progress_dialog = None
        if dlg is not None:
            dlg.close()
            dlg.deleteLater()

    def _scene_graph_progress_label(self) -> str:
        phase = str(self._sg_progress_phase or "Running")
        if bool(self._sg_llm_binary_progress):
            return phase
        idx = max(0, int(self._sg_progress_index or 0))
        total = max(0, int(self._sg_progress_total or 0))
        frame_idx = int(self._sg_progress_frame_idx or -1)
        if total > 0:
            if frame_idx >= 0:
                return f"{phase} | {idx}/{total} | frame={frame_idx}"
            return f"{phase} | {idx}/{total}"
        if frame_idx >= 0:
            return f"{phase} | frame={frame_idx}"
        return phase

    def _update_scene_graph_progress_dialog_label(self) -> None:
        dlg = self._sg_progress_dialog
        if dlg is None:
            return
        dlg.setLabelText(self._scene_graph_progress_label())
        self._position_scene_graph_progress_dialog(dlg)

    def _pin_scene_graph_progress_dialog(self) -> None:
        dlg = self._sg_progress_dialog
        if dlg is None or not dlg.isVisible():
            return
        dlg.setLabelText(self._scene_graph_progress_label())
        self._position_scene_graph_progress_dialog(dlg)

    def _position_scene_graph_progress_dialog(self, dialog: QProgressDialog) -> None:
        if dialog is None:
            return
        dialog.adjustSize()
        try:
            dialog.setSizeGripEnabled(False)
        except Exception:
            pass
        # Keep progress bar length and dialog width stable across runs.
        dialog_width = int(SCENE_GRAPH_PROGRESS_DEFAULT_WIDTH)
        if int(dialog.width() or 0) != dialog_width:
            dialog.setFixedWidth(dialog_width)
        dialog_height = max(1, int(dialog.height() or 160))
        rect = self.rect()
        top_left = self.mapToGlobal(rect.topLeft())
        target_x = int(top_left.x() + max(0, (rect.width() - dialog_width) / 2.0))
        target_y = int(top_left.y() + max(0, (rect.height() - dialog_height) / 2.0))
        dialog.move(target_x, target_y)

    def _reset_scene_graph_job_state(self) -> None:
        self._sg_job_mode = ""
        self._sg_job_lightweight = False
        self._sg_job_show_error_dialog = True
        self._sg_job_output_path = ""
        self._sg_job_sampling_fps = 1.0
        self._sg_job_source_fps = 1.0
        self._sg_job_sampling_plan = {}
        self._sg_job_frame_indices = []
        self._sg_job_graphs = []
        self._sg_job_started_at = 0.0
        self._sg_timing_log_path = ""
        self._sg_run_dir = ""
        self._sg_runtime_log_path = ""
        self._sg_checkpoint_path = ""
        self._sg_checkpoint_state = {}
        self._sg_run_status = ""
        self._sg_metadata_path = ""
        self._sg_summary_path = ""
        self._sg_stage_compact_path = ""
        self._sg_oplog_path = ""
        self._sg_force_nonpersistent_backend = False

    def _on_scene_graph_worker_thread_finished(self) -> None:
        self._sg_worker = None
        self._sg_worker_thread = None
        bundle_json = str(getattr(self, "_pending_llm_bundle_json", "") or "").strip()
        num_graphs = int(getattr(self, "_pending_llm_num_graphs", 0) or 0)
        timing_log = str(getattr(self, "_pending_llm_timing_log_path", "") or "").strip()
        if bundle_json and num_graphs > 0:
            self._pending_llm_bundle_json = ""
            self._pending_llm_num_graphs = 0
            self._pending_llm_timing_log_path = ""
            if self._start_llm_batch_summary_worker(
                bundle_json=bundle_json,
                num_graphs=num_graphs,
                timing_log_path=timing_log,
            ):
                self._append_runtime_log(
                    "Auto-starting LLM global summary generation (after SAM teardown).",
                    level="info",
                )

    def _on_llm_summary_worker_thread_finished(self) -> None:
        self._llm_summary_worker = None
        self._llm_summary_thread = None

    def _on_llm_summary_worker_progress(self, text: str) -> None:
        msg = str(text or "").strip()
        if not msg:
            return
        dlg = self._sg_progress_dialog
        current_val = int(dlg.value()) if dlg is not None else 0
        target_val = current_val
        m_attr = re.search(r"LLM-ATTR[^\n]*frame\s+(\d+)\s*/\s*(\d+)", msg, flags=re.IGNORECASE)
        m_sum = re.search(r"LLM-SUMMARY[^\n]*batch\s+(\d+)\s*/\s*(\d+)", msg, flags=re.IGNORECASE)
        if "[LLM-ATTR]" in msg:
            self._sg_progress_phase = "LLM attributes"
            if m_attr:
                idx = max(0, int(m_attr.group(1)))
                total = max(1, int(m_attr.group(2)))
                self._sg_progress_index = idx
                self._sg_progress_total = total
                # Reserve first 70% for per-frame attribute extraction.
                target_val = max(target_val, min(70, int(round((idx / float(total)) * 70.0))))
        elif "[LLM-SUMMARY]" in msg:
            self._sg_progress_phase = "LLM summary"
            if m_sum:
                idx = max(0, int(m_sum.group(1)))
                total = max(1, int(m_sum.group(2)))
                self._sg_progress_index = idx
                self._sg_progress_total = total
                # Reserve last 30% for summary batches.
                target_val = max(target_val, min(99, 70 + int(round((idx / float(total)) * 30.0))))
        if "[LLM-SUMMARY][OK]" in msg:
            target_val = 100
        if dlg is not None and target_val != current_val:
            dlg.setValue(target_val)
        self._update_scene_graph_progress_dialog_label()
        self._write_run_checkpoint(stage="llm_running", interrupted=False)
        self._set_status(msg, status_type="info")
        self._append_runtime_log(msg, level="info")

    def _on_llm_summary_worker_done(self, payload: Dict[str, object]) -> None:
        info = dict(payload or {})
        cancelled = bool(info.get("cancelled", False))
        bundle_json = str(info.get("bundle_json", "") or "")
        timing_log_path = str(info.get("timing_log_path", self._sg_timing_log_path) or self._sg_timing_log_path or "").strip()
        if self._sg_progress_dialog is not None:
            self._sg_progress_dialog.setValue(100)
        self._close_scene_graph_progress_dialog()
        self._set_scene_graph_busy(False)
        self._sg_llm_binary_progress = False
        self._pending_llm_bundle_json = ""
        self._pending_llm_num_graphs = 0
        self._pending_llm_timing_log_path = ""
        if cancelled:
            self._write_run_checkpoint(stage="llm_cancelled", interrupted=True)
            self._write_run_metadata(status="cancelled")
            self._write_run_summary(status="cancelled")
            self._set_status("LLM summary cancelled.", status_type="warning")
            self._append_oplog("llm_summary_cancelled", bundle_json=bundle_json)
            return
        try:
            if bundle_json and os.path.isfile(bundle_json):
                with open(bundle_json, "r", encoding="utf-8") as f:
                    bundle = json.load(f)
                if isinstance(bundle, dict):
                    sg_settings = dict(self._task_settings.get("Video Scene Graph", {}) or {})
                    auto_video_verify = bool(sg_settings.get("post_llm_auto_video_verify", False))
                    if auto_video_verify:
                        bundle = self._augment_bundle_video_vqa_outputs(bundle)
                    else:
                        self._append_runtime_log(
                            "Skip post-LLM Gemini video verification to keep UI responsive (post_llm_auto_video_verify=false).",
                            level="info",
                        )
                    bundle = self._recompute_bundle_stage_validation(bundle)
                    _write_json(bundle_json, bundle)
                    self.current_graph_bundle = dict(bundle)
                    graphs = [g for g in list(bundle.get("graphs") or []) if isinstance(g, dict)]
                    if graphs:
                        current_frame = int(self.player.current_frame or 0)
                        selected_graph = min(
                            graphs,
                            key=lambda graph_obj: abs(int(self._extract_graph_frame_idx(graph_obj) or 0) - current_frame),
                        )
                        self.current_graph = selected_graph
                        self._render_graph()
            self._set_status("LLM global summaries completed.", status_type="success")
            self._append_runtime_log("LLM summary wrote back to scene graph JSON.", level="success")
            if timing_log_path:
                self._append_runtime_log(f"Timing log saved: {timing_log_path}", level="info")
            self._write_run_checkpoint(stage="completed", interrupted=False)
            self._write_run_metadata(status="completed")
            self._write_run_summary(status="completed")
            self._append_oplog("llm_summary_completed", bundle_json=bundle_json)
        except Exception as exc:
            self._append_runtime_log(str(exc), level="error")
            self._set_status("LLM summary finished but bundle reload failed.", status_type="warning")
            self._write_run_checkpoint(stage="llm_done_reload_failed", interrupted=False)
            self._write_run_metadata(status="warning")
            self._write_run_summary(status="warning")
            self._append_oplog("llm_summary_reload_failed", error=str(exc))

    def _on_llm_summary_worker_failed(self, error_text: str) -> None:
        msg = str(error_text or "Unknown LLM summary error").strip()
        if self._sg_progress_dialog is not None:
            self._sg_progress_dialog.setValue(100)
        self._close_scene_graph_progress_dialog()
        self._set_scene_graph_busy(False)
        self._sg_llm_binary_progress = False
        self._append_runtime_log(msg, level="error")
        self._set_status("LLM summary failed. Check Runtime Output for details.", status_type="error")
        self._write_run_checkpoint(stage="llm_failed", interrupted=True)
        self._write_run_metadata(status="failed")
        self._write_run_summary(status="failed")
        self._append_oplog("llm_summary_failed", error=msg)

    def _on_scene_graph_worker_progress(self, text: str) -> None:
        msg = str(text or "").strip()
        if not msg:
            return
        m_prof = re.search(r"\[SG-PROFILE\]\s+([A-Za-z0-9_]+):\s*([0-9.]+)s(.*)$", msg)
        if m_prof and self._sg_timing_log_path:
            _append_timing_jsonl(
                self._sg_timing_log_path,
                "sg_profile",
                stage=str(m_prof.group(1)),
                sec=float(m_prof.group(2)),
                extra=str(m_prof.group(3) or "").strip(),
            )
        m = re.search(r"Running scene graph model on frame\s+(\d+)/(\d+)\s+\(frame=(\d+)\)", msg)
        if m:
            self._sg_progress_phase = "SAM extraction"
            self._sg_progress_index = int(m.group(1))
            self._sg_progress_total = int(m.group(2))
            self._sg_progress_frame_idx = int(m.group(3))
        elif "[SG-PROFILE] extract_attributes" in msg:
            self._sg_progress_phase = "LLM attributes"
        elif "[SG-PROFILE] validate_scene_graph" in msg:
            self._sg_progress_phase = "STAGE validation"
        elif msg.startswith("Preparing frame batch"):
            self._sg_progress_phase = "Preparing frames"
        self._update_scene_graph_progress_dialog_label()
        self._set_status(msg, status_type="info")
        self._append_runtime_log(msg, level="info")

    def _on_scene_graph_worker_frame_ready(self, payload: Dict[str, object]) -> None:
        if not isinstance(payload, dict):
            return
        frame_idx = int(payload.get("frame_idx", 0) or 0)
        image_path = str(payload.get("image_path", "") or "")
        raw_graph = dict(payload.get("graph") or {})
        graph = self._finalize_built_scene_graph(
            graph=raw_graph,
            frame_idx=frame_idx,
            image_path=image_path,
        )
        index = int(payload.get("index", 0) or 0)
        total = max(1, int(payload.get("total", 1) or 1))

        if self._sg_job_mode == "video":
            self._sg_job_graphs.append(graph)
            elapsed_sec = max(0.0, time.monotonic() - float(self._sg_job_started_at or time.monotonic()))
            done_now = len(self._sg_job_graphs)
            total_planned = max(1, int(len(self._sg_job_frame_indices) or total))
            eta_sec = 0.0
            if done_now > 0:
                eta_sec = (elapsed_sec / float(done_now)) * float(max(0, total_planned - done_now))
            if self._sg_progress_dialog is not None:
                self._sg_progress_dialog.setValue(min(done_now, total_planned))
            self._sg_progress_index = int(index)
            self._sg_progress_total = int(total_planned)
            self._sg_progress_frame_idx = int(frame_idx)
            if str(self._sg_progress_phase or "").strip().lower() in {"", "idle", "preparing frames"}:
                self._sg_progress_phase = "SAM extraction"
            self._update_scene_graph_progress_dialog_label()
            self._set_status(
                f"Generating video scene graph {done_now}/{total_planned} | "
                f"frame={frame_idx} | elapsed={self._format_status_duration(elapsed_sec)} | "
                f"eta={self._format_status_duration(eta_sec)}",
                status_type="info",
            )
            self._write_run_checkpoint(stage="sam_running", interrupted=False)
            return

        self.current_graph = graph
        self.current_graph_bundle = None
        self._set_graph_frame_selector(frame_idx, manual=(frame_idx != int(self.player.current_frame or 0)))
        if self._sg_job_lightweight:
            self._render_graph_overlay_only()
        else:
            self.single_turn_items = []
            self.multi_turn_items = []
            self._render_graph()

    def _on_scene_graph_worker_failed(self, error_text: str) -> None:
        msg = str(error_text or "Unknown scene graph error").strip()
        self._close_scene_graph_progress_dialog()
        self._set_scene_graph_busy(False)
        self._append_runtime_log(msg, level="error")
        if self._sg_job_show_error_dialog and not self._sg_job_lightweight:
            QMessageBox.critical(self, "Build Failed", "Scene graph build failed. Please check Runtime Output for details.")
        self._set_status("Scene graph build failed. Check Runtime Output for details.", status_type="error")
        self._write_run_checkpoint(stage="sam_failed", interrupted=True)
        self._write_run_metadata(status="failed")
        self._write_run_summary(status="failed")
        self._reset_scene_graph_job_state()

    def _export_scene_graph_visualizations(
        self,
        *,
        graphs: List[Dict[str, object]],
        output_path: str,
    ) -> Dict[str, object]:
        out: Dict[str, object] = {"exported": 0, "dir": "", "items": []}
        if not graphs:
            return out
        stem = os.path.splitext(os.path.basename(str(output_path or "scene_graph_bundle.json")))[0]
        out_dir = os.path.join(os.path.dirname(str(output_path or self._repo_root)), f"{stem}_viz")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            return out
        exported_items: List[Dict[str, object]] = []
        for graph in graphs:
            if not isinstance(graph, dict):
                continue
            metadata = dict(graph.get("metadata") or {})
            image_path = str(metadata.get("image_path", "") or "")
            frame_idx = int(metadata.get("graph_frame_idx", 0) or 0)
            if not image_path or not os.path.isfile(image_path):
                continue
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            node_centers: Dict[str, Tuple[int, int]] = {}
            for node in list(graph.get("nodes") or []):
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("entity_id", "") or "").strip()
                bbox = list(node.get("bbox") or [0, 0, 0, 0])
                if len(bbox) < 4:
                    continue
                x, y, w, h = [int(v) for v in bbox[:4]]
                if w <= 0 or h <= 0:
                    continue
                color = (60, 190, 255) if str(self._node_label(node, "") or "").strip().lower() == "person" else (120, 220, 120)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                if node_id:
                    node_centers[node_id] = (int(x + (w / 2.0)), int(y + (h / 2.0)))
                tid = str(node.get("track_id", node.get("entity_id", "")) or "")
                lbl = str(self._node_label(node, "") or "")
                score = float(self._node_bbox_confidence(node))
                text = f"{lbl} {tid} {score:.2f}".strip()
                cv2.putText(
                    frame,
                    text,
                    (x, max(16, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            for edge in list(graph.get("edges") or []):
                if not isinstance(edge, dict):
                    continue
                src = str(edge.get("src_id", "") or "").strip()
                dst = str(edge.get("dst_id", "") or "").strip()
                if src not in node_centers or dst not in node_centers:
                    continue
                p1 = node_centers[src]
                p2 = node_centers[dst]
                rel = str(edge.get("relation", "") or "").strip()
                cv2.line(frame, p1, p2, (0, 160, 80), 1, cv2.LINE_AA)
                mx = int((p1[0] + p2[0]) / 2.0)
                my = int((p1[1] + p2[1]) / 2.0)
                if rel:
                    cv2.putText(
                        frame,
                        rel,
                        (mx, max(12, my - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 120, 60),
                        1,
                        cv2.LINE_AA,
                    )
            out_name = f"frame_{frame_idx:06d}.jpg"
            out_path = os.path.join(out_dir, out_name)
            if cv2.imwrite(out_path, frame):
                exported_items.append(
                    {
                        "frame_idx": int(frame_idx),
                        "image_path": out_path,
                    }
                )
        out["exported"] = int(len(exported_items))
        out["dir"] = out_dir
        out["items"] = exported_items
        return out

    def _on_scene_graph_worker_done(self, payload: Dict[str, object]) -> None:
        info = dict(payload or {})
        cancelled = bool(info.get("cancelled", False))
        processed_frames = int(info.get("processed_frames", 0) or 0)
        total_frames = int(info.get("total_frames", 0) or 0)
        timing_log_path = str(info.get("timing_log_path", self._sg_timing_log_path) or self._sg_timing_log_path or "").strip()
        mode = str(self._sg_job_mode or "")
        lightweight = bool(self._sg_job_lightweight)
        output_path = str(self._sg_job_output_path or "")
        sampling_fps = float(self._sg_job_sampling_fps or 1.0)
        source_fps = float(self._sg_job_source_fps or self._effective_fps())
        sampling_plan = dict(self._sg_job_sampling_plan or {})
        graphs = list(self._sg_job_graphs)

        self._close_scene_graph_progress_dialog()
        self._set_scene_graph_busy(False)

        if mode == "video":
            if not graphs and cancelled:
                self.current_graph_bundle = None
                self._set_status(
                    "Video scene graph generation cancelled before any result was produced",
                    status_type="warning",
                )
                self._write_run_checkpoint(stage="cancelled_no_result", interrupted=True)
                self._write_run_metadata(status="cancelled")
                self._write_run_summary(status="cancelled")
                self._append_oplog("scene_graph_run_cancelled_no_result", processed_frames=int(processed_frames), total_frames=int(total_frames))
                self._reset_scene_graph_job_state()
                return

            # Merge/dedupe by frame index so graph updates stay stable.
            graph_by_frame: Dict[int, Dict[str, object]] = {}
            unknown_graphs: List[Dict[str, object]] = []
            for g in graphs:
                if not isinstance(g, dict):
                    continue
                gidx = int(self._extract_graph_frame_idx(g) or -1)
                if gidx >= 0:
                    graph_by_frame[gidx] = g
                else:
                    unknown_graphs.append(g)
            ordered_graphs: List[Dict[str, object]] = []
            for fidx in self._sg_job_frame_indices:
                if int(fidx) in graph_by_frame:
                    ordered_graphs.append(dict(graph_by_frame[int(fidx)]))
            extra_indices = sorted([k for k in graph_by_frame.keys() if int(k) not in {int(x) for x in self._sg_job_frame_indices}])
            for k in extra_indices:
                ordered_graphs.append(dict(graph_by_frame[k]))
            for g in unknown_graphs:
                ordered_graphs.append(dict(g))
            graphs = ordered_graphs
            sampled_indices = [int(self._extract_graph_frame_idx(g) or 0) for g in graphs if int(self._extract_graph_frame_idx(g) or -1) >= 0]

            bundle = {
                "type": "scene_graph_sequence",
                "version": 1,
                "video_path": self.video_path,
                "video_name": os.path.basename(self.video_path or ""),
                "frame_count": int(self.player.frame_count or 0),
                "source_fps": float(source_fps),
                "sampling_fps": float(sampling_fps),
                "sampling": sampling_plan,
                "sampled_frame_indices": sampled_indices,
                "manual_keyframes": sorted(int(x) for x in self._sg_manual_keyframes),
                "cancelled": bool(cancelled),
                "tracking_mode": str((self._task_settings.get("Video Scene Graph", {}) or {}).get("tracking_mode", "disabled") or "disabled"),
                "graphs": graphs,
                "run_artifacts": {
                    "run_dir": str(self._sg_run_dir or ""),
                    "bundle_json": str(output_path or ""),
                    "timing_log": str(self._sg_timing_log_path or ""),
                    "runtime_log": str(self._sg_runtime_log_path or ""),
                    "oplog": str(self._sg_oplog_path or ""),
                    "pvsg_reference": os.path.join(str(self._sg_run_dir or ""), "pvsg_reference.json") if str(self._sg_run_dir or "").strip() else "",
                    "checkpoint": str(self._sg_checkpoint_path or ""),
                    "metadata": str(self._sg_metadata_path or ""),
                    "summary": str(self._sg_summary_path or ""),
                },
                "pvsg_reference": {
                    "video_id": str((self._pvsg_video_reference or {}).get("video_id", "") or ""),
                    "reference_available": bool((self._pvsg_video_reference or {}).get("reference_available", False)),
                },
            }
            mask_export = self._finalize_graph_masks_for_export(graphs=graphs, output_path=output_path)
            bundle["mask_export"] = mask_export
            viz_info = self._export_scene_graph_visualizations(graphs=graphs, output_path=output_path)
            bundle["visualization"] = viz_info
            bundle = self._recompute_bundle_stage_validation(bundle)
            bundle = self._compact_scene_graph_bundle(bundle)
            _write_json(output_path, bundle)
            try:
                if str(self._sg_summary_path or "").strip():
                    _write_json(
                        self._sg_summary_path,
                        {
                            "updated_at": _now_iso_utc(),
                            "status": "partial" if bool(cancelled) else "sam_completed",
                            "video_path": str(self.video_path or ""),
                            "run_dir": str(self._sg_run_dir or ""),
                            "bundle_json": str(output_path or ""),
                            "timing_log": str(self._sg_timing_log_path or ""),
                            "runtime_log": str(self._sg_runtime_log_path or ""),
                            "checkpoint": str(self._sg_checkpoint_path or ""),
                            "processed_graphs": int(len(graphs)),
                            "planned_frames": int(len(self._sg_job_frame_indices)),
                            "last_completed_frame_idx": int(self._extract_graph_frame_idx(graphs[-1]) or -1) if graphs else -1,
                            "mask_export": mask_export,
                            "visualization": viz_info,
                        },
                    )
            except Exception:
                pass

            self.current_graph_bundle = bundle
            if graphs:
                current_frame = int(self.player.current_frame or 0)
                selected_graph = min(
                    graphs,
                    key=lambda graph_obj: abs(int(self._extract_graph_frame_idx(graph_obj) or 0) - current_frame),
                )
                selected_frame = int(self._extract_graph_frame_idx(selected_graph) or 0)
                self.current_graph = selected_graph
                self._set_graph_frame_selector(selected_frame, manual=(selected_frame != current_frame))
                self._render_graph()

            if cancelled:
                self._set_status(
                    f"Saved partial video scene graph bundle with {len(graphs)} frames to {os.path.basename(output_path)}",
                    status_type="warning",
                )
                self._write_run_checkpoint(stage="sam_partial_saved", interrupted=True)
                self._write_run_metadata(status="partial")
                self._write_run_summary(status="partial")
                self._append_oplog("scene_graph_run_partial", saved_graphs=int(len(graphs)), output_path=str(output_path or ""))
            else:
                self._set_status(
                    f"Saved video scene graph bundle with {len(graphs)} sampled frames to {os.path.basename(output_path)}",
                    status_type="success",
                )
                self._write_run_checkpoint(stage="sam_completed", interrupted=False)
                self._write_run_metadata(status="sam_completed")
                self._write_run_summary(status="sam_completed")
                self._append_oplog("scene_graph_run_sam_completed", saved_graphs=int(len(graphs)), output_path=str(output_path or ""))
            if timing_log_path:
                self._append_runtime_log(f"Timing log saved: {timing_log_path}", level="info")
            if graphs and not cancelled:
                self._append_runtime_log(
                    "Video graph generation finished. Temporal tracking is disabled; using scene-graph node attributes only.",
                    level="info",
                )
                # Delay LLM start until SAM worker thread has fully finished and backend pool is released.
                self._pending_llm_bundle_json = str(output_path or "")
                self._pending_llm_num_graphs = int(len(graphs))
                self._pending_llm_timing_log_path = str(timing_log_path or "")
                self._append_runtime_log(
                    "LLM start deferred until SAM teardown completes to avoid CUDA OOM.",
                    level="info",
                )
            self._reset_scene_graph_job_state()
            return

        if cancelled and processed_frames <= 0:
            self._set_status("Scene graph build cancelled", status_type="warning")
            self._reset_scene_graph_job_state()
            return
        if isinstance(self.current_graph, dict) and not lightweight:
            provider = str((self.current_graph.get("metadata") or {}).get("backend_provider", "unknown"))
            frame_idx = int(((self.current_graph.get("metadata") or {}).get("graph_frame_idx", 0)) or 0)
            self._set_status(
                f"Scene graph built for frame {frame_idx} with backend {provider}.",
                status_type="success",
            )
        elif isinstance(self.current_graph, dict):
            frame_idx = int(((self.current_graph.get("metadata") or {}).get("graph_frame_idx", 0)) or 0)
            self._set_status(
                f"Overlay refreshed for frame {frame_idx}.",
                status_type="success",
            )
        self._reset_scene_graph_job_state()

    def _start_scene_graph_worker(
        self,
        *,
        frame_indices: List[int],
        mode: str,
        lightweight: bool = False,
        show_error_dialog: bool = True,
        output_path: str = "",
        sampling_fps: float = 1.0,
        sampling_plan: Optional[Dict[str, object]] = None,
        timing_log_path: str = "",
        preloaded_graphs: Optional[List[Dict[str, object]]] = None,
        all_frame_indices: Optional[List[int]] = None,
        run_dir: str = "",
        runtime_log_path: str = "",
        checkpoint_path: str = "",
        metadata_path: str = "",
        summary_path: str = "",
    ) -> bool:
        if self._scene_graph_job_active():
            self._set_status("Scene graph generation is already running.", status_type="warning")
            return False
        if not self.video_path or not os.path.isfile(self.video_path):
            if show_error_dialog:
                QMessageBox.information(self, "No Video", "Open a video first.")
            self._set_status("No video loaded", status_type="warning")
            return False

        sg_settings = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        preflight_error = self._scene_graph_backend_preflight(sg_settings)
        if preflight_error:
            self._append_runtime_log(preflight_error, level="error")
            if show_error_dialog and not lightweight:
                QMessageBox.critical(self, "SAM Backend Not Ready", "Scene graph backend is not ready. Please check Runtime Output for details.")
            self._set_status("Scene graph backend is not ready. Check Runtime Output for details.", status_type="error")
            return False

        frame_indices = [int(x) for x in (frame_indices or [])]
        if not frame_indices:
            self._set_status("No frames selected for scene graph generation", status_type="warning")
            return False

        enable_sentence_refine = (not lightweight) and bool(sg_settings.get("enable_sentence_refine", False))
        backend_override = self._scene_graph_backend_override(sg_settings)
        source_fps = self._effective_fps()
        if mode == "video":
            self._reset_scene_graph_tracking()

        self._sg_job_mode = str(mode or "")
        self._sg_job_lightweight = bool(lightweight)
        self._sg_job_show_error_dialog = bool(show_error_dialog)
        self._sg_job_output_path = str(output_path or "")
        self._sg_job_sampling_fps = float(sampling_fps or 1.0)
        self._sg_job_source_fps = float(source_fps)
        self._sg_job_sampling_plan = dict(sampling_plan or {})
        self._sg_job_frame_indices = [int(x) for x in list(all_frame_indices or frame_indices)]
        self._sg_job_graphs = [dict(g) for g in list(preloaded_graphs or []) if isinstance(g, dict)]
        self._sg_job_started_at = time.monotonic()
        self._sg_timing_log_path = str(timing_log_path or self._sg_timing_log_path or "").strip()
        self._sg_run_dir = str(run_dir or self._sg_run_dir or "").strip()
        self._sg_runtime_log_path = str(runtime_log_path or self._sg_runtime_log_path or "").strip()
        self._sg_checkpoint_path = str(checkpoint_path or self._sg_checkpoint_path or "").strip()
        self._sg_metadata_path = str(metadata_path or self._sg_metadata_path or "").strip()
        self._sg_summary_path = str(summary_path or self._sg_summary_path or "").strip()
        self._sg_stage_compact_path = os.path.join(self._sg_run_dir, "stage_compact.json") if self._sg_run_dir else ""
        self._sg_oplog_path = os.path.join(self._sg_run_dir, "oplog") if self._sg_run_dir else ""
        self._sg_progress_phase = "Preparing frames"
        self._sg_progress_index = int(len(self._sg_job_graphs))
        self._sg_progress_total = int(len(self._sg_job_frame_indices))
        self._sg_progress_frame_idx = int(frame_indices[0]) if frame_indices else -1
        self._write_run_metadata(status="running")
        self._write_run_checkpoint(stage="sam_start", interrupted=False)
        self._write_run_summary(status="running")
        self._append_oplog(
            "scene_graph_worker_start",
            mode=str(mode or ""),
            total_frames=int(len(self._sg_job_frame_indices)),
            run_output=str(self._sg_job_output_path or ""),
        )

        worker = SceneGraphBuildWorker(
            video_path=self.video_path,
            frame_cache_dir=self._frame_cache_dir,
            frame_indices=frame_indices,
            ontology_path=self._ontology_path,
            pipeline_cfg_path=self._pipeline_cfg,
            enable_sentence_refine=enable_sentence_refine,
            custom_ontology_dict=self._scene_graph_runtime_ontology(),
            backend_override=backend_override,
            source_fps=source_fps,
            batch_size=int(sg_settings.get("video_generation_batch_size", 8) or 8),
            timing_log_path=self._sg_timing_log_path,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_scene_graph_worker_progress)
        worker.frame_ready.connect(self._on_scene_graph_worker_frame_ready)
        worker.done.connect(self._on_scene_graph_worker_done)
        worker.done.connect(thread.quit)
        worker.failed.connect(self._on_scene_graph_worker_failed)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_scene_graph_worker_thread_finished)

        self._sg_worker = worker
        self._sg_worker_thread = thread
        self._set_scene_graph_busy(True)
        self._sg_llm_binary_progress = False

        if mode == "video":
            every_n = int((sampling_plan or {}).get("sampling_every_n_frames", 0) or 0)
            if every_n > 0:
                sampling_desc = f"every {every_n} frames"
            else:
                sampling_desc = f"{sampling_fps:g} FPS"
            progress = QProgressDialog(
                "Generating full-video scene graph bundle...",
                "Cancel",
                0,
                len(self._sg_job_frame_indices),
                self,
            )
            progress.setValue(min(len(self._sg_job_graphs), len(self._sg_job_frame_indices)))
            progress.canceled.connect(self._cancel_scene_graph_job)
            self._set_status(
                f"Starting video scene graph generation: {len(self._sg_job_graphs)}/{len(self._sg_job_frame_indices)} frames ({sampling_desc})",
                status_type="info",
            )
            self._append_runtime_log(
                f"Starting video scene graph generation: {len(self._sg_job_graphs)}/{len(self._sg_job_frame_indices)} frames ({sampling_desc})",
                level="info",
            )
            if self._sg_timing_log_path:
                self._append_runtime_log(f"Timing log file: {self._sg_timing_log_path}", level="info")
        else:
            progress = QProgressDialog("Building scene graph...", "", 0, 0, self)
            progress.setCancelButton(None)
            progress.setValue(0)
            self._set_status("Starting scene graph generation...", status_type="info")
            self._append_runtime_log("Starting scene graph generation...", level="info")
        progress.setWindowTitle("Video Scene Graph")
        progress.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        progress.setWindowModality(Qt.NonModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        self._position_scene_graph_progress_dialog(progress)
        self._sg_progress_dialog = progress
        self._update_scene_graph_progress_dialog_label()
        self._sg_progress_pin_timer.start()

        thread.start()
        return True

    def _start_llm_batch_summary_worker(self, *, bundle_json: str, num_graphs: int, timing_log_path: str = "") -> bool:
        sg_settings = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        llm_mode = str(sg_settings.get("llm_mode", "local") or "local").strip().lower()
        local_provider = str(sg_settings.get("llm_local_provider", "qwen") or "qwen").strip().lower()
        if llm_mode != "local" or local_provider != "qwen":
            return False
        if not bundle_json or not os.path.isfile(bundle_json):
            return False

        model_path = str(
            sg_settings.get("llm_local_model_path", sg_settings.get("qwen_model_path", QWEN_DEFAULT_MODEL_PATH))
            or QWEN_DEFAULT_MODEL_PATH
        ).strip()
        # Match Qwen batch window to max_frames setting (default 3).
        max_frames_setting = int(sg_settings.get("video_sampling_max_frames", 3) or 3)
        batch_size = max(1, max_frames_setting) if max_frames_setting > 0 else 3
        llm_cuda_device_raw = str(sg_settings.get("llm_cuda_device", "cuda:0") or "cuda:0").strip()
        llm_cuda_device = self._resolve_llm_cuda_device(llm_cuda_device_raw)

        worker = LLMBatchSummaryWorker(
            repo_root=self._repo_root,
            bundle_json=bundle_json,
            model_path=model_path,
            batch_size=batch_size,
            cuda_device=llm_cuda_device,
            timing_log_path=str(timing_log_path or self._sg_timing_log_path or "").strip(),
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_llm_summary_worker_progress)
        worker.done.connect(self._on_llm_summary_worker_done)
        worker.done.connect(thread.quit)
        worker.failed.connect(self._on_llm_summary_worker_failed)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_llm_summary_worker_thread_finished)
        self._llm_summary_worker = worker
        self._llm_summary_thread = thread
        self._set_scene_graph_busy(True)

        progress = QProgressDialog("Running LLM summary...", "Cancel", 0, 100, self)
        progress.setValue(0)
        progress.canceled.connect(self._cancel_scene_graph_job)
        progress.setWindowTitle("Video Scene Graph")
        progress.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        progress.setWindowModality(Qt.NonModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()
        self._sg_progress_dialog = progress
        self._sg_llm_binary_progress = False
        self._sg_progress_phase = "LLM attributes"
        self._sg_progress_index = 0
        self._sg_progress_total = max(1, int(num_graphs or 1))
        self._sg_progress_frame_idx = -1
        self._position_scene_graph_progress_dialog(progress)
        self._update_scene_graph_progress_dialog_label()
        self._sg_progress_pin_timer.start()
        thread.start()
        return True

    def _build_scene_graph_for_selected_frame(self, *, lightweight: bool = False, show_error_dialog: bool = True) -> bool:
        if not self.player.cap:
            if show_error_dialog:
                QMessageBox.information(self, "No Video", "Open a video first.")
            self._set_status("No video loaded", status_type="warning")
            return False
        frame_idx = int(self.spin_frame_for_graph.value())
        return self._start_scene_graph_worker(
            frame_indices=[frame_idx],
            mode="single",
            lightweight=lightweight,
            show_error_dialog=show_error_dialog,
        )

    def _generate_scene_graph_for_video_with_current_keyframe(self) -> None:
        if not self.player.cap:
            QMessageBox.information(self, "No Video", "Open a video first.")
            self._set_status("No video loaded", status_type="warning")
            return
        pending_mode = str(getattr(self, "_sg_pending_run_mode", "") or "").strip().lower()
        if pending_mode == "resume":
            pending_dir = str(getattr(self, "_sg_pending_resume_dir", "") or "").strip()
            self._sg_pending_run_mode = ""
            self._sg_pending_resume_dir = ""
            self._generate_scene_graph_for_video(preferred_run_dir=pending_dir)
            return
        if pending_mode == "new":
            self._sg_pending_run_mode = ""
            self._sg_pending_resume_dir = ""
            self._generate_scene_graph_for_video_new()
            return
        self._add_current_frame_as_video_keyframe()
        self._generate_scene_graph_for_video()

    def _render_graph_overlay_only(self) -> None:
        """Lightweight refresh path: update only summary + player overlay."""
        graph = self.current_graph or {}
        nodes = list(graph.get("nodes") or [])
        edges = list(graph.get("edges") or [])
        self.sg_summary.setText(f"Graph ready: nodes={len(nodes)} edges={len(edges)}")
        self._apply_scene_graph_overlay_to_player()

    def _on_sg_sampling_every_n_frames_changed(self, value: int) -> None:
        sg_settings = self._task_settings.setdefault("Video Scene Graph", {})
        step = max(1, int(value))
        sg_settings["video_sampling_every_n_frames"] = step
        self._save_persisted_settings()

    def _on_sg_video_sampling_max_frames_changed(self, value: int) -> None:
        sg_settings = self._task_settings.setdefault("Video Scene Graph", {})
        sg_settings["video_sampling_max_frames"] = max(0, int(value))
        self._save_persisted_settings()

    def _pick_llm_provider_from_dropdown(self) -> None:
        sg_settings = self._task_settings.setdefault("Video Scene Graph", {})
        # UI policy: LLM engine is fixed to local Qwen (other providers remain in code for future use).
        sg_settings["llm_mode"] = "local"
        sg_settings["llm_local_provider"] = "qwen"
        sg_settings["llm_local_model_path"] = str(sg_settings.get("llm_local_model_path", QWEN_DEFAULT_MODEL_PATH) or QWEN_DEFAULT_MODEL_PATH)
        sg_settings["qwen_model_path"] = str(sg_settings.get("qwen_model_path", QWEN_DEFAULT_MODEL_PATH) or QWEN_DEFAULT_MODEL_PATH)
        sg_settings["qwen_batch_size"] = max(1, int(sg_settings.get("qwen_batch_size", 3) or 3))
        label = f"Local - Qwen ({int(sg_settings['qwen_batch_size'])}f summary)"
        if hasattr(self, "lbl_llm_selection"):
            self.lbl_llm_selection.setText(label)
        self._set_status("LLM Engine is fixed to Qwen in current UI.", status_type="info")
        self._save_persisted_settings()

    def _collect_scene_graph_video_run_inputs(self) -> Optional[Tuple[Dict[str, object], float, float, List[int], Dict[str, object]]]:
        if not self.player.cap:
            QMessageBox.information(self, "No Video", "Open a video first.")
            self._set_status("No video loaded", status_type="warning")
            return None
        if bool(getattr(self.player, "is_playing", False)):
            self.player.pause()

        sg_settings = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        if str(sg_settings.get("llm_mode", "local") or "local").strip().lower() != "local":
            sg_settings["llm_mode"] = "local"
            sg_settings["llm_local_provider"] = "qwen"
            sg_settings["llm_local_model_path"] = str(sg_settings.get("llm_local_model_path", QWEN_DEFAULT_MODEL_PATH) or QWEN_DEFAULT_MODEL_PATH)
            sg_settings["qwen_model_path"] = str(sg_settings.get("qwen_model_path", QWEN_DEFAULT_MODEL_PATH) or QWEN_DEFAULT_MODEL_PATH)
            sg_settings["qwen_batch_size"] = max(1, int(sg_settings.get("qwen_batch_size", 3) or 3))
            self._task_settings["Video Scene Graph"] = dict(sg_settings)
            self._save_persisted_settings()
        source_fps = self._effective_fps()
        sampling_fps = self._scene_graph_sampling_fps_from_settings(source_fps=source_fps, settings=sg_settings)
        frame_indices, sampling_plan = self._build_video_sampling_indices(sampling_fps)
        if not frame_indices:
            QMessageBox.information(self, "No Frames", "The loaded video has no readable frames.")
            self._set_status("Video contains no readable frames", status_type="warning")
            return None
        return sg_settings, float(source_fps), float(sampling_fps), [int(x) for x in frame_indices], dict(sampling_plan or {})

    def _generate_scene_graph_for_video_new(self) -> None:
        collected = self._collect_scene_graph_video_run_inputs()
        if collected is None:
            return
        _sg_settings, source_fps, sampling_fps, frame_indices, sampling_plan = collected
        participant_id = self._require_participant_id()
        if participant_id is None:
            return
        session_stem = _build_session_stem(participant_id, self.video_path)

        log_root = os.path.join(self._repo_root, "log")
        try:
            os.makedirs(log_root, exist_ok=True)
        except Exception as exc:
            self._set_status(f"Cannot create log root: {exc}", status_type="error")
            return
        run_dir = os.path.join(log_root, session_stem)
        try:
            os.makedirs(run_dir, exist_ok=True)
        except Exception as exc:
            self._set_status(f"Cannot create run folder: {exc}", status_type="error")
            return

        output_path = os.path.join(run_dir, "scene_graph_bundle.json")
        timing_log_path = os.path.join(run_dir, "timing.jsonl")
        runtime_log_path = os.path.join(run_dir, "runtime.log")
        run_info_path = os.path.join(run_dir, "run_info.json")
        checkpoint_path = ""
        metadata_path = run_info_path
        summary_path = run_info_path
        oplog_path = os.path.join(run_dir, "oplog")
        pvsg_reference_path = os.path.join(run_dir, "pvsg_reference.json")
        try:
            pvsg_ref = self._refresh_pvsg_reference_for_video(frame_indices=frame_indices, save_path=pvsg_reference_path)
            self._append_runtime_log(self._summarize_pvsg_reference(pvsg_ref), level="info")
        except Exception as exc:
            self._append_runtime_log(f"PVSG GT prefetch failed: {exc}", level="warning")
        try:
            _write_json(
                run_info_path,
                {
                    "schema_version": 1,
                    "created_at": _now_iso_utc(),
                    "status": "created",
                    "participant_id": participant_id,
                    "session_name": session_stem,
                    "run_dir": run_dir,
                    "video_path": str(self.video_path or ""),
                    "sampling_fps": float(sampling_fps),
                    "source_fps": float(source_fps),
                    "sampling_plan": dict(sampling_plan or {}),
                    "sampled_frame_indices": [int(x) for x in frame_indices],
                    "bundle_json": output_path,
                    "timing_log": timing_log_path,
                    "runtime_log": runtime_log_path,
                    "summary": run_info_path,
                    "stage_compact": run_info_path,
                    "expected_files": [
                        "run_info.json",
                        "scene_graph_bundle.json",
                        "runtime.log",
                    ],
                },
            )
        except Exception:
            pass
        self._sg_run_dir = str(run_dir)
        self._sg_oplog_path = str(oplog_path)
        self._append_oplog(
            "new_run_created",
            participant_id=participant_id,
            session_name=session_stem,
            run_dir=run_dir,
        )

        self._start_scene_graph_worker(
            frame_indices=frame_indices,
            mode="video",
            lightweight=False,
            show_error_dialog=True,
            output_path=output_path,
            sampling_fps=sampling_fps,
            sampling_plan=sampling_plan,
            timing_log_path=timing_log_path,
            preloaded_graphs=[],
            all_frame_indices=frame_indices,
            run_dir=run_dir,
            runtime_log_path=runtime_log_path,
            checkpoint_path=checkpoint_path,
            metadata_path=metadata_path,
            summary_path=summary_path,
        )

    def _generate_scene_graph_for_video(self, preferred_run_dir: str = "") -> None:
        if not str(preferred_run_dir or "").strip():
            self._generate_scene_graph_for_video_new()
            return
        collected = self._collect_scene_graph_video_run_inputs()
        if collected is None:
            return
        _sg_settings, source_fps, sampling_fps, frame_indices, sampling_plan = collected
        participant_id = self._require_participant_id()
        if participant_id is None:
            return
        session_stem = _build_session_stem(participant_id, self.video_path)

        selected_dir = os.path.abspath(str(preferred_run_dir or "").strip())
        is_existing_run_dir = False
        if selected_dir:
            ok_run, reason, _ = self._validate_resume_run_folder(selected_dir)
            if not ok_run:
                QMessageBox.warning(self, "Invalid Run Folder", f"{reason}\n\nFolder: {selected_dir}")
                self._set_status("Resume failed: run folder is not compliant.", status_type="warning")
                return
            run_dir = selected_dir
            is_existing_run_dir = True
        else:
            log_root = os.path.join(self._repo_root, "log")
            try:
                os.makedirs(log_root, exist_ok=True)
            except Exception as exc:
                self._set_status(f"Cannot create log root: {exc}", status_type="error")
                return
            run_dir = os.path.join(log_root, session_stem)
            try:
                os.makedirs(run_dir, exist_ok=True)
            except Exception as exc:
                self._set_status(f"Cannot create run folder: {exc}", status_type="error")
                return

        output_path = os.path.join(run_dir, "scene_graph_bundle.json")
        timing_log_path = os.path.join(run_dir, "timing.jsonl")
        runtime_log_path = os.path.join(run_dir, "runtime.log")
        run_info_path = os.path.join(run_dir, "run_info.json")
        checkpoint_path = os.path.join(run_dir, "checkpoint.json") if is_existing_run_dir else ""
        metadata_path = os.path.join(run_dir, "run_metadata.json") if is_existing_run_dir else run_info_path
        summary_path = run_info_path
        oplog_path = os.path.join(run_dir, "oplog")
        pvsg_reference_path = os.path.join(run_dir, "pvsg_reference.json")
        if not is_existing_run_dir:
            try:
                pvsg_ref = self._refresh_pvsg_reference_for_video(frame_indices=frame_indices, save_path=pvsg_reference_path)
                self._append_runtime_log(self._summarize_pvsg_reference(pvsg_ref), level="info")
            except Exception as exc:
                self._append_runtime_log(f"PVSG GT prefetch failed: {exc}", level="warning")
            try:
                _write_json(
                    run_info_path,
                    {
                        "schema_version": 1,
                        "created_at": _now_iso_utc(),
                        "status": "created",
                        "participant_id": participant_id,
                        "session_name": session_stem,
                        "run_dir": run_dir,
                        "video_path": str(self.video_path or ""),
                        "sampling_fps": float(sampling_fps),
                        "source_fps": float(source_fps),
                        "sampling_plan": dict(sampling_plan or {}),
                        "sampled_frame_indices": [int(x) for x in frame_indices],
                        "bundle_json": output_path,
                        "timing_log": timing_log_path,
                        "runtime_log": runtime_log_path,
                        "summary": run_info_path,
                        "stage_compact": run_info_path,
                        "expected_files": [
                            "run_info.json",
                            "scene_graph_bundle.json",
                            "runtime.log",
                        ],
                    },
                )
            except Exception:
                pass
            self._sg_run_dir = str(run_dir)
            self._sg_oplog_path = str(oplog_path)
            self._append_oplog(
                "new_run_created",
                participant_id=participant_id,
                session_name=session_stem,
                run_dir=run_dir,
            )
        else:
            self._sg_run_dir = str(run_dir)
            self._sg_oplog_path = str(oplog_path)
            if not os.path.isfile(pvsg_reference_path):
                try:
                    pvsg_ref = self._refresh_pvsg_reference_for_video(frame_indices=frame_indices, save_path=pvsg_reference_path)
                    self._append_runtime_log(self._summarize_pvsg_reference(pvsg_ref), level="info")
                except Exception as exc:
                    self._append_runtime_log(f"PVSG GT prefetch failed: {exc}", level="warning")
            elif not self._pvsg_video_reference:
                self._pvsg_video_reference = _read_json(pvsg_reference_path)
            self._append_oplog("resume_run_selected", run_dir=run_dir, preferred_run_dir=str(preferred_run_dir or ""))

        preloaded_graphs: List[Dict[str, object]] = []
        pending_indices: List[int] = list(frame_indices)
        if os.path.isfile(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    existing_bundle = json.load(f)
                existing_graphs = [dict(g) for g in list((existing_bundle or {}).get("graphs") or []) if isinstance(g, dict)]
                existing_video = str((existing_bundle or {}).get("video_path", "") or "")
                same_video = bool(existing_video) and os.path.abspath(existing_video) == os.path.abspath(str(self.video_path or ""))
                if same_video and existing_graphs:
                    by_idx: Dict[int, Dict[str, object]] = {}
                    for g in existing_graphs:
                        gi = int(self._extract_graph_frame_idx(g) or -1)
                        if gi >= 0:
                            by_idx[gi] = g
                    if by_idx:
                        preloaded_graphs = [dict(by_idx[i]) for i in frame_indices if int(i) in by_idx]
                        pending_indices = [int(i) for i in frame_indices if int(i) not in by_idx]
                        self._append_runtime_log(
                            f"Resume detected: existing={len(preloaded_graphs)} pending={len(pending_indices)}",
                            level="info",
                        )
            except Exception as exc:
                self._append_runtime_log(f"Resume scan skipped: {exc}", level="warning")

        if not pending_indices:
            self._sg_run_dir = str(run_dir)
            self._sg_job_output_path = str(output_path)
            self._sg_timing_log_path = str(timing_log_path)
            self._sg_runtime_log_path = str(runtime_log_path)
            self._sg_checkpoint_path = str(checkpoint_path)
            self._sg_metadata_path = str(metadata_path)
            self._sg_summary_path = str(summary_path)
            self._sg_stage_compact_path = os.path.join(str(run_dir), "stage_compact.json")
            self._sg_job_frame_indices = [int(x) for x in frame_indices]
            self._sg_job_graphs = [dict(g) for g in preloaded_graphs]
            self._write_run_metadata(status="resume_llm_only")
            self._write_run_checkpoint(stage="llm_resume_start", interrupted=False)
            self._write_run_summary(status="resume_llm_only")
            self._set_status("All sampled frames are already completed in the selected JSON.", status_type="success")
            if self._start_llm_batch_summary_worker(bundle_json=output_path, num_graphs=len(frame_indices), timing_log_path=timing_log_path):
                self._append_runtime_log("Auto-starting LLM resume from existing bundle.", level="info")
            return

        # Practical resume fallback: if only frame 0 is missing while almost all sampled frames exist,
        # skip SAM retry and continue with LLM on the existing bundle to avoid repeated resume stalls.
        if len(pending_indices) == 1 and int(pending_indices[0]) == 0 and len(preloaded_graphs) >= max(1, len(frame_indices) - 1):
            self._sg_run_dir = str(run_dir)
            self._sg_job_output_path = str(output_path)
            self._sg_timing_log_path = str(timing_log_path)
            self._sg_runtime_log_path = str(runtime_log_path)
            self._sg_checkpoint_path = str(checkpoint_path)
            self._sg_metadata_path = str(metadata_path)
            self._sg_summary_path = str(summary_path)
            self._sg_stage_compact_path = os.path.join(str(run_dir), "stage_compact.json")
            self._sg_job_frame_indices = [int(x) for x in frame_indices]
            self._sg_job_graphs = [dict(g) for g in preloaded_graphs]
            self._write_run_metadata(status="resume_llm_only_missing_frame0")
            self._write_run_checkpoint(stage="llm_resume_start", interrupted=False)
            self._write_run_summary(status="resume_llm_only_missing_frame0")
            self._append_runtime_log(
                "Resume fallback: only frame=0 is missing; skip SAM retry and continue to LLM.",
                level="warning",
            )
            self._set_status("Resume fallback: skip SAM for missing frame 0 and continue to LLM.", status_type="warning")
            if self._start_llm_batch_summary_worker(bundle_json=output_path, num_graphs=len(frame_indices), timing_log_path=timing_log_path):
                self._append_runtime_log("Auto-starting LLM resume from existing bundle.", level="info")
            return

        self._start_scene_graph_worker(
            frame_indices=pending_indices,
            mode="video",
            lightweight=False,
            show_error_dialog=True,
            output_path=output_path,
            sampling_fps=sampling_fps,
            sampling_plan=sampling_plan,
            timing_log_path=timing_log_path,
            preloaded_graphs=preloaded_graphs,
            all_frame_indices=frame_indices,
            run_dir=run_dir,
            runtime_log_path=runtime_log_path,
            checkpoint_path=checkpoint_path,
            metadata_path=metadata_path,
            summary_path=summary_path,
        )

    def _resolve_overlay_frame_size(self, graph: Dict[str, object]) -> Tuple[float, float]:
        """Best-effort frame size for converting SG coordinates to player pixel space."""
        w = float(getattr(self.player, "_frame_w", 0) or 0)
        h = float(getattr(self.player, "_frame_h", 0) or 0)
        if w > 1 and h > 1:
            return w, h

        # Fallback to decoder metadata even if current frame cache is empty.
        cap = getattr(self.player, "cap", None)
        if cap is not None:
            try:
                cw = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                ch = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if cw > 1 and ch > 1:
                    return cw, ch
            except Exception:
                pass

        image_path = str((graph.get("metadata") or {}).get("image_path", "") or "")
        if image_path and os.path.isfile(image_path):
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    ih, iw = img.shape[:2]
                    if iw > 1 and ih > 1:
                        return float(iw), float(ih)
            except Exception:
                pass

        return 1.0, 1.0

    @staticmethod
    def _detect_bbox_mode(nodes: List[Dict[str, object]], frame_w: float, frame_h: float) -> str:
        """Heuristic bbox mode detection: pixel / norm01 / percent100."""
        valid = 0
        norm01_like = 0
        percent_like = 0
        frame_like = 0
        frame_max = max(1.0, float(frame_w), float(frame_h))
        for node in nodes[:64]:
            bbox = list(node.get("bbox") or [])
            if len(bbox) < 4:
                continue
            try:
                x, y, w, h = (abs(float(bbox[0])), abs(float(bbox[1])), abs(float(bbox[2])), abs(float(bbox[3])))
            except Exception:
                continue
            valid += 1
            m = max(x, y, w, h)
            if m <= 1.5 and w <= 1.2 and h <= 1.2:
                norm01_like += 1
            if m <= 100.0:
                percent_like += 1
            if m <= frame_max * 1.25:
                frame_like += 1
        if valid == 0:
            return "pixel"

        if norm01_like >= max(1, int(valid * 0.6)):
            return "norm01"

        # Percentage coordinates are usually <=100 while frame coordinates are much larger.
        if percent_like >= max(1, int(valid * 0.8)) and frame_max >= 200:
            return "percent100"

        # If most values are within frame bounds, treat as pixels.
        if frame_like >= max(1, int(valid * 0.6)):
            return "pixel"

        # Fallback: small-range values on large frames are usually percentages.
        if percent_like >= max(1, int(valid * 0.6)) and frame_max >= 400:
            return "percent100"

        return "pixel"

    @staticmethod
    def _extract_graph_frame_idx(graph: Dict[str, object]) -> Optional[int]:
        """Try to recover frame index associated with a scene graph."""
        metadata = graph.get("metadata") or {}
        meta_idx: Optional[int] = None
        try:
            if "frame_idx" in graph:
                return int(graph.get("frame_idx"))
        except Exception:
            pass
        try:
            if "graph_frame_idx" in metadata:
                meta_idx = int(metadata.get("graph_frame_idx"))
        except Exception:
            meta_idx = None

        # Try image_id patterns like "xxx_f000123"
        image_id = str(graph.get("image_id", "") or "")
        parsed_image_idx: Optional[int] = None
        if image_id:
            marker = image_id.rfind("_f")
            if marker >= 0:
                tail = image_id[marker + 2 :]
                digits = ""
                for ch in tail:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    try:
                        parsed_image_idx = int(digits)
                    except Exception:
                        pass
        if meta_idx is not None:
            if int(meta_idx) == 0 and parsed_image_idx is not None and int(parsed_image_idx) > 0:
                return int(parsed_image_idx)
            return int(meta_idx)
        if parsed_image_idx is not None:
            return int(parsed_image_idx)

        # Try image path basename like "video_f000123.jpg"
        image_path = str(metadata.get("image_path", "") or "")
        if image_path:
            base = os.path.basename(image_path)
            marker = base.rfind("_f")
            if marker >= 0:
                tail = base[marker + 2 :]
                digits = ""
                for ch in tail:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    try:
                        return int(digits)
                    except Exception:
                        pass

        return None

    def _select_graph_for_current_frame(self) -> Dict[str, object]:
        current_frame_idx = int(getattr(self.player, "current_frame", 0) or 0)
        bundle = self.current_graph_bundle if isinstance(self.current_graph_bundle, dict) else {}
        graphs = [g for g in list(bundle.get("graphs") or []) if isinstance(g, dict)]
        if graphs:
            for g in graphs:
                if self._extract_graph_frame_idx(g) == current_frame_idx:
                    return g
        return self.current_graph if isinstance(self.current_graph, dict) else {}

    def _apply_scene_graph_overlay_to_player(self) -> None:
        """Render current scene graph as overlays directly on the left video player."""
        current_frame_idx = int(getattr(self.player, "current_frame", 0) or 0)
        graph = self._select_graph_for_current_frame()
        if graph and graph is not self.current_graph:
            self.current_graph = graph
        nodes = list(graph.get("nodes") or [])
        edges = list(graph.get("edges") or [])

        frame_w, frame_h = self._resolve_overlay_frame_size(graph)
        self._sg_bbox_mode = self._detect_bbox_mode(nodes, frame_w, frame_h)
        self._sg_bbox_ref_size = (frame_w, frame_h)
        if not nodes:
            self.player.set_overlay_boxes([])
            self.player.set_overlay_relations([])
            self.player.set_overlay_enabled(False)
            allow_edit = self._current_task_name() == "Video Scene Graph"
            self.player.set_edit_context(
                [],
                on_change=self._on_player_scene_graph_box_changed if allow_edit else None,
                on_ctrl_object_pick=self._on_player_ctrl_object_pick_for_edge if allow_edit else None,
                on_relation_pick=self._on_player_relation_pick if allow_edit else None,
                on_relation_edit=self._on_player_relation_edit if allow_edit else None,
                label_suggestions=[],
            )
            self.sg_summary.setText(
                f"Current frame graph is empty: current={current_frame_idx} | nodes=0 edges={len(edges)}"
            )
            return

        selected_node = ""
        row = self._selected_row(self.sg_nodes_table) if hasattr(self, "sg_nodes_table") else -1
        if row >= 0:
            node_item = self.sg_nodes_table.item(row, 0)
            selected_node = self._entity_id_from_table_item(node_item)

        selected_edge = ""
        edge_row = self._selected_row(self.sg_edges_table) if hasattr(self, "sg_edges_table") else -1
        if 0 <= edge_row < len(self._edge_rows):
            selected_edge = str(self._edge_rows[edge_row].get("edge_id", ""))

        node_center = {}
        boxes: List[Dict[str, object]] = []
        edit_boxes: List[Dict[str, object]] = []
        selected_box_payload: Optional[Dict[str, object]] = None
        selected_edit_payload: Optional[Dict[str, object]] = None
        edge_lookup = {str(e.get("edge_id", "") or ""): dict(e) for e in edges if isinstance(e, dict)}
        selected_edge_row = edge_lookup.get(selected_edge, {})
        highlighted_nodes = {
            str(selected_edge_row.get("src_id", "") or ""),
            str(selected_edge_row.get("dst_id", "") or ""),
        } if selected_edge_row else set()
        node_conf_threshold = NODE_LOW_CONF_THRESHOLD
        edge_conf_threshold = EDGE_LOW_CONF_THRESHOLD
        for node in nodes:
            entity_id = str(node.get("entity_id", ""))
            bbox = list(node.get("bbox") or [0, 0, 0, 0])
            if len(bbox) < 4:
                continue
            x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            if self._sg_bbox_mode == "norm01":
                x *= frame_w
                y *= frame_h
                w *= frame_w
                h *= frame_h
            elif self._sg_bbox_mode == "percent100":
                x = (x / 100.0) * frame_w
                y = (y / 100.0) * frame_h
                w = (w / 100.0) * frame_w
                h = (h / 100.0) * frame_h
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = max(x1 + 1.0, x + max(1.0, w))
            y2 = max(y1 + 1.0, y + max(1.0, h))
            label = str(self._node_label(node, "unknown"))
            display_name = str(node.get("display_name", "") or self._node_display_by_id.get(entity_id, "") or label)
            score = self._node_bbox_confidence(node)
            uncertain_node = float(score) < node_conf_threshold
            is_selected = (entity_id == selected_node)
            is_endpoint = entity_id in highlighted_nodes
            # selected: dark blue, high confidence: light blue, low confidence: yellow
            box_color = "#0078FF" if is_selected else ("#FFB020" if uncertain_node else "#5E9BC9")
            box_label = f"{display_name} ({score:.2f})"

            box_payload = {
                "id": entity_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": box_label,
                "color": box_color,
                "thick": bool(is_selected),
            }
            edit_payload = {
                "id": entity_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": label,
            }
            # Keep selected object at the top layer (drawn last + hit-tested first).
            if is_selected:
                selected_box_payload = box_payload
                selected_edit_payload = edit_payload
            else:
                boxes.append(box_payload)
                edit_boxes.append(edit_payload)
            node_center[entity_id] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

        if selected_box_payload:
            boxes.append(selected_box_payload)
        if selected_edit_payload:
            edit_boxes.append(selected_edit_payload)

        rels = []
        for edge in edges:
            src_id = str(edge.get("src_id", ""))
            dst_id = str(edge.get("dst_id", ""))
            edge_id = str(edge.get("edge_id", ""))
            if src_id not in node_center or dst_id not in node_center:
                continue
            if selected_node and not selected_edge and (src_id != selected_node and dst_id != selected_node):
                # Focus mode: only keep edges connected to the selected object.
                continue
            sx, sy = node_center[src_id]
            dx, dy = node_center[dst_id]
            is_selected = False
            if selected_edge:
                is_selected = edge_id == selected_edge
            elif selected_node:
                is_selected = (src_id == selected_node or dst_id == selected_node)
            edge_score = self._edge_confidence(edge)
            edge_flags = {str(x).strip().lower() for x in list(edge.get("validator_flags") or []) if str(x).strip()}
            uncertain_edge = (
                edge_score < edge_conf_threshold
                or bool(
                    edge_flags.intersection(
                        {
                            "cycle_relation_conflict",
                            "human_relation_rejected",
                            "low_confidence_edge",
                        }
                    )
                )
            )
            rels.append(
                {
                    "edge_id": edge_id,
                    "x1": sx,
                    "y1": sy,
                    "x2": dx,
                    "y2": dy,
                    "label": str(edge.get("relation", "")),
                    "color": "#FFB020" if uncertain_edge else ("#22C55E" if is_selected else "#16A34A"),
                    "alpha": 175 if is_selected else 110,
                    "line_width": 1.8 if is_selected else 1.2,
                    "font_size": 8,
                    "font_bold": False,
                    "label_alpha": 145 if is_selected else 95,
                    "text_color": "#14532D",
                }
            )

        self.player.set_overlay_boxes(boxes)
        self.player.set_overlay_relations(rels)
        self.player.set_overlay_enabled(True)
        allow_edit = self._current_task_name() == "Video Scene Graph"
        self.player.set_edit_context(
            edit_boxes,
            on_change=self._on_player_scene_graph_box_changed if allow_edit else None,
            on_ctrl_object_pick=self._on_player_ctrl_object_pick_for_edge if allow_edit else None,
            on_relation_pick=self._on_player_relation_pick if allow_edit else None,
            on_relation_edit=self._on_player_relation_edit if allow_edit else None,
            label_suggestions=[str(self._node_label(n, "")) for n in nodes if str(self._node_label(n, ""))],
        )
        self.sg_summary.setText(
            ""
        )

    def _on_player_scene_graph_box_changed(self, box_id, payload: Dict[str, object]) -> None:
        """Apply box edit operations from player overlay back into scene graph nodes."""
        if not isinstance(self.current_graph, dict):
            return
        before_graph = self._json_safe_clone(self.current_graph)
        graph = self._json_safe_clone(self.current_graph)
        nodes = list(graph.get("nodes") or [])
        edges = list(graph.get("edges") or [])
        action = str(payload.get("_action", "update"))
        selected_entity_after = ""

        def _to_bbox(p: Dict[str, object]) -> List[float]:
            x1 = float(p.get("x1", 0.0) or 0.0)
            y1 = float(p.get("y1", 0.0) or 0.0)
            x2 = float(p.get("x2", x1 + 1.0) or (x1 + 1.0))
            y2 = float(p.get("y2", y1 + 1.0) or (y1 + 1.0))
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            if self._sg_bbox_mode == "norm01":
                fw, fh = self._sg_bbox_ref_size
                fw = max(1.0, float(fw))
                fh = max(1.0, float(fh))
                return [x1 / fw, y1 / fh, bw / fw, bh / fh]
            if self._sg_bbox_mode == "percent100":
                fw, fh = self._sg_bbox_ref_size
                fw = max(1.0, float(fw))
                fh = max(1.0, float(fh))
                return [100.0 * x1 / fw, 100.0 * y1 / fh, 100.0 * bw / fw, 100.0 * bh / fh]
            return [x1, y1, bw, bh]

        changed = False
        if action == "delete":
            target = str(box_id or payload.get("id", ""))
            if target:
                nodes = [n for n in nodes if str(n.get("entity_id", "")) != target]
                edges = [
                    e
                    for e in edges
                    if str(e.get("src_id", "")) != target and str(e.get("dst_id", "")) != target
                ]
                changed = True
        elif action == "add":
            new_id = f"ent_{uuid.uuid4().hex[:10]}"
            new_label = str(payload.get("label", "object") or "object")
            new_bbox = _to_bbox(payload)
            nodes.append(
                {
                    "entity_id": new_id,
                    "canonical_label": new_label,
                    "label": new_label,
                    "bbox": new_bbox,
                    "score": 0.0,
                    "confidence": 0.0,
                    "verify_confidence": 0.0,
                    "validator_flags": [],
                    "attributes": [],
                }
            )
            selected_entity_after = new_id
            changed = True
        else:
            target = str(box_id or payload.get("id", ""))
            new_bbox = _to_bbox(payload)
            for node in nodes:
                if str(node.get("entity_id", "")) == target:
                    selected_entity_after = target
                    old_bbox = list(node.get("bbox") or [])
                    if old_bbox != new_bbox:
                        node["bbox"] = new_bbox
                        changed = True
                    if "label" in payload and str(payload.get("label", "")).strip():
                        new_label = str(payload.get("label", "")).strip()
                        if new_label != str(node.get("canonical_label", "")).strip():
                            node["canonical_label"] = new_label
                            node["label"] = new_label
                            changed = True
                    break

        if not changed and selected_entity_after:
            self._on_sg_visualizer_node_selected(selected_entity_after)
            return
        if not changed:
            return

        graph["nodes"] = nodes
        graph["edges"] = edges
        self._push_scene_graph_undo(before_graph, reason=f"box_{action}")
        self.current_graph = graph
        self._reset_scene_graph_tracking()
        self._replace_current_graph_in_bundle(graph)
        self._persist_current_scene_graph_bundle("scene_graph_bbox_edit_saved")
        self._render_graph()
        if selected_entity_after:
            self._on_sg_visualizer_node_selected(selected_entity_after)

    def _on_player_relation_pick(self, edge_id: str) -> None:
        """Select edge row when clicking relation label on image."""
        self._on_sg_visualizer_edge_selected(str(edge_id or "").strip())

    def _on_player_relation_edit(self, edge_id: str) -> None:
        """Edit edge relation when double-clicking relation label on image."""
        eid = str(edge_id or "").strip()
        if not eid:
            return
        self._on_sg_visualizer_edge_selected(eid)
        self._rename_selected_scene_graph_edge_relation()

    def _render_graph(self) -> None:
        graph = self.current_graph or {}
        prev_selected_node_id = ""
        prev_selected_edge_id = ""
        try:
            node_row = self._selected_row(self.sg_nodes_table) if hasattr(self, "sg_nodes_table") else -1
            if node_row >= 0 and hasattr(self, "sg_nodes_table"):
                prev_selected_node_id = self._entity_id_from_table_item(self.sg_nodes_table.item(node_row, 0))
        except Exception:
            prev_selected_node_id = ""
        try:
            edge_row = self._selected_row(self.sg_edges_table) if hasattr(self, "sg_edges_table") else -1
            if 0 <= edge_row < len(self._edge_rows):
                prev_selected_edge_id = str(self._edge_rows[edge_row].get("edge_id", "") or "")
        except Exception:
            prev_selected_edge_id = ""

        graph_frame_idx = int(self._extract_graph_frame_idx(graph or {}) or -1) if isinstance(graph, dict) else -1
        self._sync_cycle_result_with_current_graph()
        if isinstance(self.current_cycle_result, dict):
            self._render_cycle_probe_outputs(list(self.current_cycle_result.get("probe_results") or []))
        else:
            self._render_cycle_probe_outputs([])
        self._refresh_claim_verification_tables()
        self._render_cycle_caption_feedback()
        nodes = list(graph.get("nodes") or [])
        edges = list(graph.get("edges") or [])
        mode = getattr(self, "_sg_bbox_mode", "pixel")
        fw, fh = getattr(self, "_sg_bbox_ref_size", (1.0, 1.0))
        provider = str((graph.get("metadata") or {}).get("backend_provider", "unknown"))
        gframe = self._extract_graph_frame_idx(graph)
        cur = int(getattr(self.player, "current_frame", 0) or 0)
        frame_note = f"graph_frame={gframe} current={cur}" if gframe is not None else f"current={cur}"
        self.sg_summary.setText(
            f"Graph ready: nodes={len(nodes)} edges={len(edges)} | backend={provider} | "
            f"bbox={mode} ref={int(fw)}x{int(fh)} | {frame_note}"
        )

        self._node_row_by_id = {}
        self._edge_rows = []
        self._refresh_entity_display_map(graph)
        self.current_graph = graph
        self._replace_current_graph_in_bundle(graph)
        vote_map = self._cycle_vote_summary_by_claim()

        self._sg_table_rendering = True
        old_node_block = self.sg_nodes_table.blockSignals(True)
        old_edge_block = self.sg_edges_table.blockSignals(True)
        try:
            self.sg_nodes_table.setRowCount(len(nodes))
            for i, n in enumerate(nodes):
                eid = str(n.get("entity_id", ""))
                display = str(n.get("display_name", "") or self._node_display_by_id.get(eid, eid))
                label = str(self._node_label(n, ""))
                conf = self._node_bbox_confidence(n)
                attrs_text = self._node_attributes_to_text(n)
                vote_hint = self._node_vote_hint(n, vote_map)
                label_text = label if not vote_hint else f"{label} | {vote_hint}"
                self._node_row_by_id[eid] = i
                for col, text in enumerate((display, label_text, f"{conf:.3f}", attrs_text)):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.UserRole, eid)
                    if col in {0}:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    if col == 3:
                        item.setToolTip("Editable. One per line: slot=value (or JSON list). Double-click for large editor.")
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    self.sg_nodes_table.setItem(i, col, item)

            self.sg_edges_table.setRowCount(len(edges))
            for i, e in enumerate(edges):
                edge_id = str(e.get("edge_id", ""))
                rel = str(e.get("relation", ""))
                rel_hint = self._edge_vote_hint(e, vote_map)
                rel_text = rel if not rel_hint else f"{rel} | {rel_hint}"
                src = str(e.get("src_id", ""))
                dst = str(e.get("dst_id", ""))
                src_display = str(self._node_display_by_id.get(src, src))
                dst_display = str(self._node_display_by_id.get(dst, dst))
                self._edge_rows.append({"edge_id": edge_id, "src": src, "dst": dst})
                for col, text in enumerate((edge_id, rel_text, src_display, dst_display)):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.UserRole, {"edge_id": edge_id, "src": src, "dst": dst})
                    if col in {0}:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    if col == 1:
                        item.setToolTip("Double-click to edit relation name, or use 'Rename Edge' button.")
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    self.sg_edges_table.setItem(i, col, item)
        finally:
            self.sg_nodes_table.blockSignals(old_node_block)
            self.sg_edges_table.blockSignals(old_edge_block)
            self._sg_table_rendering = False

        self.sg_nodes_table.resizeRowsToContents()
        self.sg_edges_table.resizeRowsToContents()

        # Restore selection so list/image stay visually synced after edits.
        if prev_selected_node_id:
            row = self._node_row_by_id.get(prev_selected_node_id, -1)
            if row >= 0:
                self.sg_nodes_table.selectRow(int(row))
        if prev_selected_edge_id:
            edge_row = -1
            for i, row in enumerate(list(self._edge_rows or [])):
                if str(row.get("edge_id", "") or "") == prev_selected_edge_id:
                    edge_row = i
                    break
            if edge_row >= 0:
                self.sg_edges_table.selectRow(int(edge_row))

        self._reset_table_bg(self.sg_nodes_table)
        self._reset_table_bg(self.sg_edges_table)
        self.sg_json_preview.setPlainText(json.dumps(self._json_safe_clone(graph), ensure_ascii=True, indent=2))

        self._apply_scene_graph_overlay_to_player()
        self._refresh_stage_validation_view()
        self._refresh_cycle_summary()
        self._refresh_human_arbitration_view()
        selected_row = self._selected_row(self.sg_nodes_table) if hasattr(self, "sg_nodes_table") else -1
        selected_node_id = ""
        if selected_row >= 0 and hasattr(self, "sg_nodes_table"):
            selected_node_id = self._entity_id_from_table_item(self.sg_nodes_table.item(selected_row, 0))
        self._render_object_probe_drawer(selected_node_id)

    def _summary_statements_from_stage(self, graph: Dict[str, object]) -> List[str]:
        metadata = dict(graph.get("metadata") or {})
        current = list(metadata.get("stage_summary_statements") or [])
        out = [str(x).strip() for x in current if str(x).strip()]
        if out:
            return out
        text = str(metadata.get("global_summary", "") or "").strip()
        if text:
            return [x.strip() for x in text.replace("\n", " ").split(".") if x.strip()]
        return []

    def _recompute_bundle_stage_validation(self, bundle: Dict[str, object]) -> Dict[str, object]:
        payload = dict(bundle or {})
        graphs = [dict(g) for g in list(payload.get("graphs") or []) if isinstance(g, dict)]
        if not graphs:
            payload["validation"] = {
                "generated_at": _now_iso_utc(),
                "num_graphs": 0,
                "module_scores": {},
            }
            return payload

        keys = [
            "S_T",
            "S_A",
            "S_E",
            "S_S",
            "S_G",
            "target_accuracy",
            "state_accuracy",
            "S_struct",
            "S_summary",
            "S_mqa",
            "S_verify",
            "S_workflow",
        ]
        agg: Dict[str, List[float]] = {k: [] for k in keys}
        per_frame_scores: List[Dict[str, object]] = []
        validated_graphs: List[Dict[str, object]] = []
        for graph in graphs:
            g = dict(graph)
            try:
                result = self._stage_validator.validate(
                    graph=g,
                    scene_graph_bundle=payload,
                )
            except Exception as exc:
                self._append_runtime_log(f"STAGE validation failed during bundle writeback: {exc}", level="error")
                validated_graphs.append(g)
                continue
            if isinstance(result, dict):
                g["validation"] = dict(result)
                module_scores = dict(result.get("module_scores") or {})
                frame_idx = int(self._extract_graph_frame_idx(g) or -1)
                per_frame_scores.append(
                    {
                        "frame_idx": int(frame_idx),
                        "module_scores": {k: float(module_scores.get(k, 0.0) or 0.0) for k in keys},
                    }
                )
                for key in keys:
                    if key in module_scores:
                        try:
                            agg[key].append(float(module_scores.get(key, 0.0) or 0.0))
                        except Exception:
                            pass
            validated_graphs.append(g)

        payload["graphs"] = validated_graphs
        avg_scores = {
            key: (float(sum(vals) / len(vals)) if vals else 0.0)
            for key, vals in agg.items()
        }
        payload["validation"] = {
            "generated_at": _now_iso_utc(),
            "num_graphs": int(len(validated_graphs)),
            "module_scores": dict(avg_scores),  # backward-compatible alias (average over all sampled frames)
            "module_scores_avg": dict(avg_scores),
            "per_frame_module_scores": per_frame_scores,
        }
        return payload

    def _refresh_stage_validation_view(self) -> None:
        if not hasattr(self, "_sg_stage_score_widgets"):
            return
        graph = self.current_graph if isinstance(self.current_graph, dict) else {}
        if not graph:
            self._current_stage_validation = {}
            self._render_stage_score_cards({})
            return
        try:
            result = self._stage_validator.validate(
                graph=graph,
                scene_graph_bundle=self.current_graph_bundle if isinstance(self.current_graph_bundle, dict) else None,
            )
        except Exception as exc:
            self._append_runtime_log(f"STAGE validation failed: {exc}", level="error")
            fallback_scores = dict(graph.get("stage_scores") or {})
            if fallback_scores:
                self._current_stage_validation = {"module_scores": fallback_scores}
                self._render_stage_score_cards(fallback_scores)
            return
        self._current_stage_validation = dict(result or {})
        module_scores = dict(result.get("module_scores") or graph.get("stage_scores") or {})
        self._render_stage_score_cards(module_scores)
        self._apply_stage_warning_badges_to_tables()

    @staticmethod
    def _warning_rank(warning: str) -> int:
        key = str(warning or "").strip().lower()
        if key == "red":
            return 3
        if key == "purple":
            return 2
        if key == "yellow":
            return 1
        return 0

    @staticmethod
    def _warning_color(warning: str) -> Optional[QColor]:
        key = str(warning or "").strip().lower()
        if key == "red":
            return QColor(254, 226, 226)
        if key == "purple":
            return QColor(243, 232, 255)
        if key == "yellow":
            return QColor(254, 249, 195)
        return None

    def _apply_stage_warning_badges_to_tables(self) -> None:
        stage_items = dict((self._current_stage_validation or {}).get("stage_items") or {})
        node_warn: Dict[str, str] = {}
        edge_warn: Dict[str, str] = {}

        for stage_rows in stage_items.values():
            for row in list(stage_rows or []):
                if not isinstance(row, dict):
                    continue
                warning = str(row.get("warning", "") or "").strip().lower()
                if not warning:
                    continue
                ttype = str(row.get("target_type", "") or "").strip().lower()
                tid = str(row.get("target_id", "") or "").strip()
                payload = dict(row.get("payload") or {})
                if ttype == "edge":
                    key = tid or str(payload.get("edge_id", "") or "").strip()
                    prev = edge_warn.get(key, "")
                    if self._warning_rank(warning) > self._warning_rank(prev):
                        edge_warn[key] = warning
                elif ttype == "track":
                    # map track warning to node rows by track_id/entity_id when possible
                    track_key = tid
                    for n in list(self.current_graph.get("nodes") or []) if isinstance(self.current_graph, dict) else []:
                        if not isinstance(n, dict):
                            continue
                        if str(n.get("track_id", n.get("entity_id", "")) or "") != track_key:
                            continue
                        eid = str(n.get("entity_id", "") or "").strip()
                        prev = node_warn.get(eid, "")
                        if self._warning_rank(warning) > self._warning_rank(prev):
                            node_warn[eid] = warning
                elif ttype == "attribute":
                    eid = str(payload.get("entity_id", "") or "").strip()
                    if eid:
                        prev = node_warn.get(eid, "")
                        if self._warning_rank(warning) > self._warning_rank(prev):
                            node_warn[eid] = warning

        for eid, row_idx in dict(self._node_row_by_id or {}).items():
            color = self._warning_color(node_warn.get(str(eid), ""))
            if color is None:
                continue
            self._set_row_bg(self.sg_nodes_table, int(row_idx), color)

        for idx, edge in enumerate(list(self._edge_rows or [])):
            edge_id = str((edge or {}).get("edge_id", "") or "").strip()
            color = self._warning_color(edge_warn.get(edge_id, ""))
            if color is None:
                continue
            self._set_row_bg(self.sg_edges_table, int(idx), color)

    def _on_stage_action_requested(self, item: Dict[str, object], action: str) -> None:
        if not isinstance(self.current_graph, dict):
            return
        before_graph = self._json_safe_clone(self.current_graph)
        graph = self._json_safe_clone(self.current_graph)
        target_type = str(item.get("target_type", "") or "").strip().lower()
        target_id = str(item.get("target_id", "") or "").strip()
        suggested = str(item.get("suggested_action", "review") or "review").strip().lower()
        selected_action = suggested if action == "suggested" else str(action or "").strip().lower()
        changed = False

        if target_type == "track":
            nodes = [dict(x) for x in list(graph.get("nodes") or []) if isinstance(x, dict)]
            edges = [dict(x) for x in list(graph.get("edges") or []) if isinstance(x, dict)]
            if selected_action in {"remove", "delete"}:
                keep_ids = {
                    str(n.get("entity_id", "") or "")
                    for n in nodes
                    if str(n.get("track_id", n.get("entity_id", "")) or "") != target_id
                }
                nodes = [n for n in nodes if str(n.get("entity_id", "") or "") in keep_ids]
                edges = [
                    e for e in edges
                    if str(e.get("src_id", "") or "") in keep_ids and str(e.get("dst_id", "") or "") in keep_ids
                ]
                changed = True
            elif selected_action in {"relabel", "review"}:
                text, ok = QInputDialog.getText(self, "Relabel Track", "New category label:")
                if ok and str(text).strip():
                    for n in nodes:
                        if str(n.get("track_id", n.get("entity_id", "")) or "") == target_id:
                            n["canonical_label"] = str(text).strip()
                            changed = True
            elif selected_action in {"merge"}:
                dst, ok = QInputDialog.getText(self, "Merge Track", "Merge into track_id:")
                dst = str(dst).strip()
                if ok and dst:
                    for n in nodes:
                        tid = str(n.get("track_id", n.get("entity_id", "")) or "")
                        if tid == target_id:
                            n["track_id"] = dst
                            changed = True
            elif selected_action in {"split", "reassign"}:
                new_tid, ok = QInputDialog.getText(self, "Reassign Track", "New track_id:")
                new_tid = str(new_tid).strip()
                if ok and new_tid:
                    for n in nodes:
                        if str(n.get("track_id", n.get("entity_id", "")) or "") == target_id:
                            n["track_id"] = new_tid
                            changed = True
            graph["nodes"] = nodes
            graph["edges"] = edges

        elif target_type == "attribute":
            payload = dict(item.get("payload") or {})
            entity_id = str(payload.get("entity_id", "") or "")
            slot = str(payload.get("slot", "") or "")
            nodes = [dict(x) for x in list(graph.get("nodes") or []) if isinstance(x, dict)]
            for node in nodes:
                if str(node.get("entity_id", "") or "") != entity_id:
                    continue
                attrs = [dict(x) for x in list(node.get("attributes") or []) if isinstance(x, dict)]
                new_attrs: List[Dict[str, object]] = []
                for attr in attrs:
                    if str(attr.get("slot", "") or "") != slot:
                        new_attrs.append(attr)
                        continue
                    if selected_action in {"remove", "delete"}:
                        changed = True
                        continue
                    if selected_action in {"unknown"}:
                        attr["value"] = "unknown"
                        changed = True
                    elif selected_action in {"relabel", "review"}:
                        text, ok = QInputDialog.getText(self, "Edit Attribute", f"{slot} =")
                        if ok:
                            attr["value"] = str(text).strip() or "unknown"
                            changed = True
                    new_attrs.append(attr)
                node["attributes"] = new_attrs

            graph["nodes"] = nodes

        elif target_type == "edge":
            edges = [dict(x) for x in list(graph.get("edges") or []) if isinstance(x, dict)]
            new_edges: List[Dict[str, object]] = []
            for edge in edges:
                eid = str(edge.get("edge_id", "") or "")
                if eid != target_id:
                    new_edges.append(edge)
                    continue
                if selected_action in {"remove", "delete"}:
                    changed = True
                    continue
                if selected_action in {"relabel", "review"}:
                    text, ok = QInputDialog.getText(self, "Relabel Edge", "New relation:")
                    if ok and str(text).strip():
                        edge["relation"] = str(text).strip()
                        changed = True
                if selected_action in {"inferred"}:
                    edge["evidence_mode"] = "inferred"
                    changed = True
                new_edges.append(edge)
            graph["edges"] = new_edges

        elif target_type == "dynamic":
            metadata = dict(graph.get("metadata") or {})
            dynamics = [dict(x) for x in list(metadata.get("semantic_dynamics") or []) if isinstance(x, dict)]
            if selected_action in {"remove", "delete"}:
                dynamics = [x for x in dynamics if str(x.get("id", "") or "") != target_id]
                changed = True
            else:
                text, ok = QInputDialog.getText(self, "Edit Dynamic", "State / transition text:")
                if ok:
                    found = False
                    for row in dynamics:
                        if str(row.get("id", "") or "") == target_id:
                            row["text"] = str(text).strip()
                            found = True
                            changed = True
                            break
                    if not found and str(text).strip():
                        dynamics.append({"id": target_id, "text": str(text).strip()})
                        changed = True
            metadata["semantic_dynamics"] = dynamics
            graph["metadata"] = metadata

        elif target_type == "summary":
            metadata = dict(graph.get("metadata") or {})
            statements = self._summary_statements_from_stage(graph)
            payload = dict(item.get("payload") or {})
            idx = int(payload.get("index", -1) or -1)
            if 0 <= idx < len(statements):
                if selected_action in {"remove", "delete"}:
                    statements.pop(idx)
                    changed = True
                else:
                    text, ok = QInputDialog.getText(self, "Rewrite Summary", "Summary statement:", text=str(statements[idx]))
                    if ok:
                        statements[idx] = str(text).strip()
                        changed = True
            metadata["stage_summary_statements"] = statements
            metadata["global_summary"] = ". ".join([s for s in statements if s])
            graph["metadata"] = metadata

        if not changed:
            return

        self.current_graph = graph
        self.current_graph_bundle = None
        self._reset_scene_graph_tracking()
        self._replace_current_graph_in_bundle(self.current_graph)
        self._persist_current_scene_graph_bundle("scene_graph_stage_action_saved")
        self._render_graph()
        self._record_change(
            task_type="scene_graph",
            item_id=target_id or target_type,
            op=f"stage_{selected_action}",
            field_path=f"stage/{target_type}",
            before=before_graph,
            after=graph,
            reason=f"stage_panel:{selected_action}",
        )
        self._set_status(f"Applied STAGE action: {selected_action} on {target_type}", status_type="success")

    def _set_row_bg(self, table: QTableWidget, row: int, color: Optional[QColor]) -> None:
        if row < 0:
            return
        col_count = int(table.columnCount())
        # Block signals so setBackground() does not emit itemChanged (which triggers saves).
        old_block = table.blockSignals(True)
        try:
            for c in range(col_count):
                item = table.item(row, c)
                if item is None:
                    item = QTableWidgetItem("")
                    table.setItem(row, c, item)
                if color is None:
                    item.setBackground(QColor(255, 255, 255))
                else:
                    item.setBackground(color)
        finally:
            table.blockSignals(old_block)

    def _reset_table_bg(self, table: QTableWidget) -> None:
        for r in range(int(table.rowCount())):
            self._set_row_bg(table, r, None)

    def _selected_row(self, table: QTableWidget) -> int:
        sel = table.selectionModel()
        if sel is None:
            return -1
        rows = sel.selectedRows()
        if not rows:
            return -1
        return int(rows[0].row())

    def _push_scene_graph_undo(self, graph_before: Dict[str, object], *, reason: str = "") -> None:
        if not isinstance(graph_before, dict):
            return
        snapshot = self._json_safe_clone(graph_before)
        if not snapshot:
            return
        if self._scene_graph_undo_stack and self._scene_graph_undo_stack[-1] == snapshot:
            return
        self._scene_graph_undo_stack.append(snapshot)
        if len(self._scene_graph_undo_stack) > int(self._scene_graph_undo_limit):
            self._scene_graph_undo_stack = self._scene_graph_undo_stack[-int(self._scene_graph_undo_limit):]

    def _undo_scene_graph_last_edit_shortcut(self) -> None:
        if self._current_task_name() != "Video Scene Graph":
            return
        focus = QApplication.focusWidget()
        if isinstance(focus, (QLineEdit, QTextEdit, QPlainTextEdit)):
            return
        self._undo_scene_graph_last_edit()

    def _undo_scene_graph_last_edit(self) -> None:
        if str(self._sg_ctrl_edge_pick_first or "").strip():
            self._sg_ctrl_edge_pick_first = ""
            self._set_status("Cancelled pending Ctrl+click edge pick.", status_type="info")
            return
        if not self._scene_graph_undo_stack:
            self._set_status("No scene graph edit to undo.", status_type="info")
            return
        prev_graph = self._scene_graph_undo_stack.pop()
        if not isinstance(prev_graph, dict) or not prev_graph:
            self._set_status("Undo skipped: previous graph snapshot is invalid.", status_type="warning")
            return
        self.current_graph = self._json_safe_clone(prev_graph)
        self._reset_scene_graph_tracking()
        self._replace_current_graph_in_bundle(self.current_graph)
        self._persist_current_scene_graph_bundle("scene_graph_ctrl_z_undo_saved")
        self._render_graph()
        self._set_status("Undo applied (Ctrl+Z): restored previous scene graph edit.", status_type="success")

    @staticmethod
    def _display_label_token(label: object) -> str:
        token = str(label or "object").strip() or "object"
        token = re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_").lower()
        return token or "object"

    def _refresh_entity_display_map(self, graph: Dict[str, object]) -> Dict[str, str]:
        counts: Dict[str, int] = {}
        display_by_id: Dict[str, str] = {}
        entity_by_display: Dict[str, str] = {}
        for node in list(graph.get("nodes") or []):
            if not isinstance(node, dict):
                continue
            eid = str(node.get("entity_id", "") or "").strip()
            if not eid:
                continue
            label_token = self._display_label_token(self._node_label(node, "object"))
            counts[label_token] = int(counts.get(label_token, 0)) + 1
            display = f"{label_token}_{counts[label_token]}"
            node["display_name"] = display
            display_by_id[eid] = display
            entity_by_display[display] = eid
        meta = dict(graph.get("metadata") or {})
        meta["entity_display_map"] = {
            eid: {"display_name": display, "entity_id": eid}
            for eid, display in display_by_id.items()
        }
        graph["metadata"] = meta
        self._node_display_by_id = display_by_id
        self._entity_id_by_display = entity_by_display
        return display_by_id

    def _entity_id_from_table_item(self, item: Optional[QTableWidgetItem]) -> str:
        if item is None:
            return ""
        raw = item.data(Qt.UserRole)
        if raw:
            return str(raw)
        text = str(item.text() or "").strip()
        return str(self._entity_id_by_display.get(text, text))

    def _on_node_selection_changed(self) -> None:
        if self._sync_graph_selection:
            return
        row = self._selected_row(self.sg_nodes_table)
        self._sync_graph_selection = True
        try:
            self._reset_table_bg(self.sg_nodes_table)
            self._reset_table_bg(self.sg_edges_table)
            self._apply_stage_warning_badges_to_tables()
            if row < 0:
                self._highlight_related_probe_rows(node_id="", edge_id="")
                self._render_object_probe_drawer("")
                self._apply_scene_graph_overlay_to_player()
                return
            self._set_row_bg(self.sg_nodes_table, row, QColor(233, 245, 255))
            node_item = self.sg_nodes_table.item(row, 0)
            node_id = self._entity_id_from_table_item(node_item)
            self._selected_node_ids = [str(node_id or "").strip()] if str(node_id or "").strip() else []
            for i, edge in enumerate(self._edge_rows):
                if edge.get("src") == node_id or edge.get("dst") == node_id:
                    self._set_row_bg(self.sg_edges_table, i, QColor(245, 250, 255))
            self._focus_probe_lists_for_node(node_id)
            self._render_object_probe_drawer(node_id)
            self._apply_scene_graph_overlay_to_player()
        finally:
            self._sync_graph_selection = False
    def _on_edge_selection_changed(self) -> None:
        if self._sync_graph_selection:
            return
        row = self._selected_row(self.sg_edges_table)
        self._sync_graph_selection = True
        try:
            self._reset_table_bg(self.sg_nodes_table)
            self._reset_table_bg(self.sg_edges_table)
            self._apply_stage_warning_badges_to_tables()
            if row < 0 or row >= len(self._edge_rows):
                self._highlight_related_probe_rows(node_id="", edge_id="")
                self._apply_scene_graph_overlay_to_player()
                return
            self._set_row_bg(self.sg_edges_table, row, QColor(233, 245, 255))
            edge = self._edge_rows[row]
            self._selected_node_ids = [str(edge.get("src", "") or "").strip(), str(edge.get("dst", "") or "").strip()]
            src_row = self._node_row_by_id.get(str(edge.get("src", "")), -1)
            dst_row = self._node_row_by_id.get(str(edge.get("dst", "")), -1)
            self._set_row_bg(self.sg_nodes_table, src_row, QColor(245, 250, 255))
            self._set_row_bg(self.sg_nodes_table, dst_row, QColor(245, 250, 255))
            self._focus_probe_lists_for_edge(str(edge.get("edge_id", "") or ""))
            self._apply_scene_graph_overlay_to_player()
        finally:
            self._sync_graph_selection = False

    def _focus_probe_lists_for_node(self, node_id: str) -> None:
        token = str(node_id or "").strip()
        if not token:
            return
        self._highlight_related_probe_rows(node_id=token, edge_id="")
        for idx, row in enumerate(list(self.single_turn_items or [])):
            ids = [str(x or "").strip() for x in list(dict(row or {}).get("evidence_node_ids") or [])]
            if token in ids:
                self.single_list.setCurrentRow(int(idx))
                self.single_list.scrollToItem(self.single_list.item(int(idx)))
                break
        for idx, row in enumerate(list(self.multi_turn_items or [])):
            ids = [str(x or "").strip() for x in list(dict(row or {}).get("evidence_node_ids") or [])]
            if token in ids:
                self.multi_list.setCurrentRow(int(idx))
                self.multi_list.scrollToItem(self.multi_list.item(int(idx)))
                break

    def _focus_probe_lists_for_edge(self, edge_id: str) -> None:
        token = str(edge_id or "").strip()
        if not token:
            return
        self._highlight_related_probe_rows(node_id="", edge_id=token)
        for idx, row in enumerate(list(self.single_turn_items or [])):
            ids = [str(x or "").strip() for x in list(dict(row or {}).get("evidence_edge_ids") or [])]
            if token in ids:
                self.single_list.setCurrentRow(int(idx))
                self.single_list.scrollToItem(self.single_list.item(int(idx)))
                break
        for idx, row in enumerate(list(self.multi_turn_items or [])):
            ids = [str(x or "").strip() for x in list(dict(row or {}).get("evidence_edge_ids") or [])]
            if token in ids:
                self.multi_list.setCurrentRow(int(idx))
                self.multi_list.scrollToItem(self.multi_list.item(int(idx)))
                break

    def _highlight_related_probe_rows(self, *, node_id: str = "", edge_id: str = "") -> None:
        node_token = str(node_id or "").strip()
        edge_token = str(edge_id or "").strip()
        base = QColor(255, 255, 255)
        related = QColor(233, 245, 255)

        def _apply(list_widget: QListWidget, rows: List[Dict[str, object]]) -> None:
            for idx, row in enumerate(list(rows or [])):
                item = list_widget.item(int(idx))
                if item is None:
                    continue
                row_map = dict(row or {})
                node_ids = [str(x or "").strip() for x in list(row_map.get("evidence_node_ids") or [])]
                edge_ids = [str(x or "").strip() for x in list(row_map.get("evidence_edge_ids") or [])]
                matched = False
                if node_token and node_token in node_ids:
                    matched = True
                if edge_token and edge_token in edge_ids:
                    matched = True
                if matched:
                    item.setBackground(related)
                else:
                    # keep low-confidence warning tint if already flagged.
                    resp = self._probe_response_payload(row_map)
                    ans = str(resp.get("answer") or resp.get("selection") or "uncertain").strip() or "uncertain"
                    sc = -1.0
                    try:
                        raw_score = resp.get("score", row_map.get("score", None))
                        if raw_score is not None:
                            sc = float(raw_score)
                    except Exception:
                        sc = -1.0
                    if (
                        (not self._is_probe_manually_resolved(row_map))
                        and (not self._probe_is_invalid(row_map, resp))
                        and sc >= 0.0
                        and self._is_low_confidence_probe(ans, sc)
                    ):
                        item.setBackground(QColor(255, 245, 204))
                    else:
                        item.setBackground(base)

        if hasattr(self, "single_list") and isinstance(self.single_list, QListWidget):
            _apply(self.single_list, [dict(x) for x in list(self.single_turn_items or [])])
        if hasattr(self, "multi_list") and isinstance(self.multi_list, QListWidget):
            _apply(self.multi_list, [dict(x) for x in list(self.multi_turn_items or [])])

    def _render_object_probe_drawer(self, node_id: str) -> None:
        panel = getattr(self, "object_probe_detail", None)
        if not isinstance(panel, QPlainTextEdit):
            return
        token = str(node_id or "").strip()
        if not token:
            panel.setPlainText("Select an object to inspect evidence chains.")
            return
        graph = self.current_graph if isinstance(self.current_graph, dict) else {}
        node = {}
        for row in list(graph.get("nodes") or []):
            if not isinstance(row, dict):
                continue
            if str(row.get("entity_id", "") or "").strip() == token:
                node = dict(row)
                break
        if not node:
            panel.setPlainText(f"Object not found: {token}")
            return

        vote_map = self._cycle_vote_summary_by_claim()
        vote_hint = self._node_vote_hint(node, vote_map)
        claim_label = f"claim_label_{token}"
        corr = dict((self.current_cycle_result or {}).get("correction_candidates") or {}) if isinstance(self.current_cycle_result, dict) else {}
        correction = dict(corr.get(claim_label) or {})
        correction_lines: List[str] = []
        for row in list(correction.get("ranked") or [])[:5]:
            if not isinstance(row, dict):
                continue
            correction_lines.append(f"- {str(row.get('value', '') or '').strip()} ({float(row.get('score', 0.0) or 0.0):.2f})")

        single_rows = [
            dict(item)
            for item in list(self.single_turn_items or [])
            if token in [str(x or "").strip() for x in list(dict(item or {}).get("evidence_node_ids") or [])]
        ]
        single_rows.sort(key=lambda row: str(row.get("probe_id", "") or ""))

        multi_rows = [
            dict(item)
            for item in list(self.multi_turn_items or [])
            if token in [str(x or "").strip() for x in list(dict(item or {}).get("evidence_node_ids") or [])]
        ]
        chains: Dict[str, List[Dict[str, object]]] = {}
        for row in multi_rows:
            cid = str(row.get("chain_id", "chain") or "chain")
            chains.setdefault(cid, []).append(row)
        for cid in list(chains.keys()):
            chains[cid] = sorted(chains[cid], key=lambda r: int(r.get("turn", 0) or 0))

        lines: List[str] = []
        det_conf = float(self._node_bbox_confidence(node))
        lines.extend(
            [
                "Object Panel",
                f"- entity_id: {token}",
                f"- label: {str(self._node_label(node, 'object') or 'object')}",
                f"- Detection Confidence: {det_conf:.3f}" if det_conf > 0.0 else "- Detection Confidence: N/A",
            ]
        )
        lines.extend(["", "Votes Summary", f"- {vote_hint or 'no votes'}"])
        lines.extend(["", "Single-turn Probes"])
        if not single_rows:
            lines.append("- none")
        else:
            for row in single_rows:
                resp = self._probe_response_payload(row)
                invalid_resp = self._probe_is_invalid(row, resp)
                resolved_resp = self._is_probe_manually_resolved(row)
                answer = str(resp.get("answer") or resp.get("selection") or "uncertain").strip() or "uncertain"
                score = -1.0
                try:
                    raw_score = resp.get("score", None)
                    if raw_score is not None:
                        score = float(raw_score)
                except Exception:
                    score = -1.0
                reason = str(resp.get("reason") or "").strip()
                if resolved_resp:
                    answer = "Resolved"
                elif invalid_resp:
                    answer = "⚠ Invalid Response"
                elif answer.lower() == "uncertain" and score >= 0.0 and score <= 0.05:
                    answer = "uncertain (low confidence)"
                lines.append(f"- [{str(row.get('probe_id', '') or '')}] {self._humanize_question_text(str(row.get('question', '') or ''))}")
                lines.append(
                    f"  answer={answer} score={(f'{score:.2f}' if score >= 0.0 and (not invalid_resp) and (not resolved_resp) else 'N/A')} "
                    f"schema_valid={bool(resp.get('schema_valid', row.get('schema_valid', True)))}"
                )
                if reason:
                    lines.append(f"  reason={reason}")

        lines.extend(["", "Multi-turn Chain"])
        if not chains:
            lines.append("- none")
        else:
            for chain_id in sorted(chains.keys()):
                lines.append(f"- {chain_id}")
                for row in list(chains.get(chain_id) or []):
                    resp = self._probe_response_payload(row)
                    invalid_resp = self._probe_is_invalid(row, resp)
                    resolved_resp = self._is_probe_manually_resolved(row)
                    answer = str(resp.get("answer") or resp.get("selection") or "uncertain").strip() or "uncertain"
                    score = -1.0
                    try:
                        raw_score = resp.get("score", None)
                        if raw_score is not None:
                            score = float(raw_score)
                    except Exception:
                        score = -1.0
                    reason = str(resp.get("reason") or "").strip()
                    if resolved_resp:
                        answer = "Resolved"
                    elif invalid_resp:
                        answer = "⚠ Invalid Response"
                    elif answer.lower() == "uncertain" and score >= 0.0 and score <= 0.05:
                        answer = "uncertain (low confidence)"
                    lines.append(
                        f"  T{int(row.get('turn', 0) or 0)} [{str(row.get('probe_family', '') or '')}] "
                        f"{self._humanize_question_text(str(row.get('question', '') or ''))}"
                    )
                    lines.append(
                        f"    answer={answer} score={(f'{score:.2f}' if score >= 0.0 and (not invalid_resp) and (not resolved_resp) else 'N/A')} "
                        f"schema_valid={bool(resp.get('schema_valid', row.get('schema_valid', True)))}"
                    )
                    if reason:
                        lines.append(f"    reason={reason}")

        lines.extend(["", "Corrections"])
        if correction_lines:
            lines.extend(correction_lines)
        else:
            lines.append("- none")
        panel.setPlainText("\n".join(lines))

    def _on_node_table_item_double_clicked(self, item: QTableWidgetItem) -> None:
        """Open a large editor for node attributes."""
        if item is None or int(item.column()) != 3:
            return
        row = int(item.row())
        id_item = self.sg_nodes_table.item(row, 0)
        entity_id = self._entity_id_from_table_item(id_item)
        title = f"Edit Attributes - {entity_id}" if entity_id else "Edit Attributes"

        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        # Keep attribute editor as a small, draggable floating dialog by default
        # so it does not block the image area behind it.
        dialog.setWindowFlag(Qt.WindowMaximizeButtonHint, False)
        dialog.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        dialog.setSizeGripEnabled(True)
        dialog.setMinimumSize(460, 240)
        dialog.resize(540, 320)
        layout = QVBoxLayout(dialog)
        hint = QLabel("One attribute per line: slot=value\nOr paste JSON list.")
        layout.addWidget(hint)
        editor = QTextEdit(dialog)
        editor.setAcceptRichText(False)
        editor.setLineWrapMode(QTextEdit.WidgetWidth)
        editor.setPlainText(str(item.text() or ""))
        layout.addWidget(editor, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        item.setText(editor.toPlainText())

    def _on_node_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._sg_table_rendering or self._sync_graph_selection or not isinstance(self.current_graph, dict):
            return
        row = int(item.row())
        col = int(item.column())
        if col not in {1, 2, 3}:
            return
        id_item = self.sg_nodes_table.item(row, 0)
        entity_id = self._entity_id_from_table_item(id_item)
        if not entity_id:
            return
        graph = self._json_safe_clone(self.current_graph)
        changed = False
        for node in list(graph.get("nodes") or []):
            if str(node.get("entity_id", "") or "") != entity_id:
                continue
            if col == 1:
                value = str(item.text() or "").strip()
                if value and value != str(node.get("canonical_label", "") or "").strip():
                    node["canonical_label"] = value
                    node["label"] = value
                    changed = True
            elif col == 2:
                try:
                    value = max(0.0, min(1.0, float(str(item.text() or "0").strip())))
                except Exception:
                    return
                existing_conf = float(node.get("confidence", node.get("score", 0.0)) or 0.0)
                if abs(value - existing_conf) > 1e-9:
                    node["confidence"] = value
                    node["score"] = max(float(node.get("score", 0.0) or 0.0), value)
                    changed = True
            elif col == 3:
                parsed_attrs, ok = self._parse_node_attributes_text(str(item.text() or ""))
                if not ok:
                    QMessageBox.warning(
                        self,
                        "Invalid Attributes",
                        "Use one per line: slot=value\nor a JSON list of attribute objects.",
                    )
                    self._render_graph()
                    return
                old_attrs = [dict(x) for x in list(node.get("attributes") or []) if isinstance(x, dict)]
                if old_attrs != parsed_attrs:
                    node["attributes"] = parsed_attrs
                    changed = True
            break
        if not changed:
            return
        before = self._json_safe_clone(self.current_graph)
        self.current_graph = graph
        self._replace_current_graph_in_bundle(graph)
        self._persist_current_scene_graph_bundle("scene_graph_node_edit_saved")
        self._record_change(
            task_type="scene_graph",
            item_id=entity_id,
            op="edit_node_table",
            field_path="nodes",
            before=before,
            after=graph,
            reason="node table edit",
        )
        self._render_graph()

    def _on_edge_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._sg_table_rendering or self._sync_graph_selection or not isinstance(self.current_graph, dict):
            return
        row = int(item.row())
        col = int(item.column())
        if col not in {1, 2, 3}:
            return
        if row < 0 or row >= len(self._edge_rows):
            return
        edge_id = str(self._edge_rows[row].get("edge_id", "") or "")
        if not edge_id:
            return
        graph = self._json_safe_clone(self.current_graph)
        changed = False
        proposed_src = ""
        proposed_dst = ""
        for edge in list(graph.get("edges") or []):
            if str(edge.get("edge_id", "") or "") != edge_id:
                continue
            value = str(item.text() or "").strip()
            proposed_src = str(edge.get("src_id", "") or "").strip()
            proposed_dst = str(edge.get("dst_id", "") or "").strip()
            if col == 1 and value and value != str(edge.get("relation", "") or "").strip():
                edge["relation"] = value
                changed = True
            elif col == 2 and value:
                resolved = str(self._entity_id_by_display.get(value, value))
                if resolved != str(edge.get("src_id", "") or "").strip():
                    edge["src_id"] = resolved
                    proposed_src = resolved
                    changed = True
            elif col == 3 and value:
                resolved = str(self._entity_id_by_display.get(value, value))
                if resolved != str(edge.get("dst_id", "") or "").strip():
                    edge["dst_id"] = resolved
                    proposed_dst = resolved
                    changed = True
            break
        if not changed:
            return
        if proposed_src and proposed_dst:
            if proposed_src == proposed_dst:
                self._set_status("Edge source and target must be two different nodes.", status_type="warning")
                self._render_graph()
                return
            conflict = self._find_pair_conflicting_edge(
                graph,
                src_id=proposed_src,
                dst_id=proposed_dst,
                exclude_edge_id=edge_id,
            )
            if conflict:
                existing_rel = str(conflict.get("relation", "") or "").strip() or "related_to"
                self._set_status(
                    f"Duplicate pair blocked: {proposed_src} <-> {proposed_dst} already exists ({existing_rel}).",
                    status_type="warning",
                )
                self._render_graph()
                return
        before = self._json_safe_clone(self.current_graph)
        self._push_scene_graph_undo(before, reason="edit_edge_table")
        self.current_graph = graph
        self._replace_current_graph_in_bundle(graph)
        self._persist_current_scene_graph_bundle("scene_graph_edge_edit_saved")
        self._record_change(
            task_type="scene_graph",
            item_id=edge_id,
            op="edit_edge_table",
            field_path="edges",
            before=before,
            after=graph,
            reason="edge table edit",
        )
        self._render_graph()

    def _find_pair_conflicting_edge(
        self,
        graph: Dict[str, object],
        *,
        src_id: str,
        dst_id: str,
        exclude_edge_id: str = "",
    ) -> Optional[Dict[str, object]]:
        """Return an existing edge for the same unordered node pair (if any)."""
        src = str(src_id or "").strip()
        dst = str(dst_id or "").strip()
        if not src or not dst:
            return None
        pair = tuple(sorted([src, dst]))
        excluded = str(exclude_edge_id or "").strip()
        for edge in list(graph.get("edges") or []):
            if not isinstance(edge, dict):
                continue
            edge_id = str(edge.get("edge_id", "") or "").strip()
            if excluded and edge_id == excluded:
                continue
            a = str(edge.get("src_id", "") or "").strip()
            b = str(edge.get("dst_id", "") or "").strip()
            if not a or not b:
                continue
            if tuple(sorted([a, b])) == pair:
                return edge
        return None

    def _add_scene_graph_edge(self) -> None:
        if not isinstance(self.current_graph, dict):
            self._set_status("No scene graph loaded.", status_type="warning")
            return
        nodes = [n for n in list(self.current_graph.get("nodes") or []) if isinstance(n, dict)]
        if len(nodes) < 2:
            self._set_status("Need at least two nodes to add an edge.", status_type="warning")
            return
        self._refresh_entity_display_map(self.current_graph)
        display_names = [
            str(n.get("display_name", "") or self._node_display_by_id.get(str(n.get("entity_id", "")), ""))
            for n in nodes
            if str(n.get("entity_id", "") or "").strip()
        ]
        display_names = [x for x in display_names if x]
        if len(display_names) < 2:
            self._set_status("Node display mapping is empty; cannot add edge.", status_type="warning")
            return
        src_display, ok = QInputDialog.getItem(self, "Add Edge", "Source node:", display_names, 0, False)
        if not ok or not str(src_display or "").strip():
            return
        dst_display, ok = QInputDialog.getItem(self, "Add Edge", "Target node:", display_names, 0, False)
        if not ok or not str(dst_display or "").strip():
            return
        relation = self._pick_edge_relation(
            title="Add Edge",
            prompt="Relation:",
            current="left_of",
        )
        if not relation:
            return
        src_id = str(self._entity_id_by_display.get(str(src_display), str(src_display)))
        dst_id = str(self._entity_id_by_display.get(str(dst_display), str(dst_display)))
        if not src_id or not dst_id:
            self._set_status("Invalid source/target node.", status_type="warning")
            return
        self._add_scene_graph_edge_between(
            src_id=src_id,
            dst_id=dst_id,
            relation=relation,
            reason="manual edge add",
        )

    def _edge_relation_options(self) -> List[str]:
        """Relation dropdown options (click-only, no free typing)."""
        options: List[str] = list(EDGE_RELATION_UI_CORE)
        try:
            vocab = self._relation_vocab_for_tracking()
            for key in ("spatial", "interaction"):
                for rel in list(vocab.get(key) or []):
                    text = str(rel or "").strip()
                    if text:
                        options.append(text)
        except Exception:
            pass
        if isinstance(self.current_graph, dict):
            for edge in list(self.current_graph.get("edges") or []):
                if not isinstance(edge, dict):
                    continue
                text = str(edge.get("relation", "") or "").strip()
                if text:
                    options.append(text)
        dedup: List[str] = []
        seen = set()
        for rel in options:
            key = str(rel).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            dedup.append(str(rel).strip())
        return dedup or list(EDGE_RELATION_UI_CORE)

    def _pick_edge_relation(self, *, title: str, prompt: str, current: str = "") -> str:
        options = self._edge_relation_options()
        cur = str(current or "").strip()
        idx = 0
        if cur:
            for i, rel in enumerate(options):
                if str(rel).strip().lower() == cur.lower():
                    idx = i
                    break
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)
        dialog.setStyleSheet(
            "QDialog { background: #ffffff; color: #111111; }"
            "QLabel { color: #111111; }"
            "QComboBox { background: #ffffff; color: #111111; border: 1px solid #c9ced6; padding: 4px; }"
            "QListView { background: #ffffff; color: #111111; }"
            "QPushButton { background: #ffffff; color: #111111; border: 1px solid #c9ced6; padding: 5px 10px; }"
            "QPushButton:hover { background: #f5f7fa; }"
        )
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(str(prompt or "Relation:"), dialog))
        combo = QComboBox(dialog)
        combo.setEditable(False)
        combo.addItems(options)
        combo.setCurrentIndex(max(0, min(idx, max(0, combo.count() - 1))))
        layout.addWidget(combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return ""
        return str(combo.currentText() or "").strip()

    def _add_scene_graph_edge_between(
        self,
        *,
        src_id: str,
        dst_id: str,
        relation: str,
        reason: str,
    ) -> bool:
        if not isinstance(self.current_graph, dict):
            self._set_status("No scene graph loaded.", status_type="warning")
            return False
        src = str(src_id or "").strip()
        dst = str(dst_id or "").strip()
        rel = str(relation or "").strip()
        if not src or not dst or src == dst:
            self._set_status("Edge source and target must be two different nodes.", status_type="warning")
            return False
        if not rel:
            self._set_status("Relation cannot be empty.", status_type="warning")
            return False
        graph = self._json_safe_clone(self.current_graph)
        conflict = self._find_pair_conflicting_edge(graph, src_id=src, dst_id=dst)
        if conflict:
            existing_rel = str(conflict.get("relation", "") or "").strip() or "related_to"
            self._set_status(
                f"Duplicate pair blocked: {src} <-> {dst} already exists ({existing_rel}).",
                status_type="warning",
            )
            return False
        edges = [dict(e) for e in list(graph.get("edges") or []) if isinstance(e, dict)]
        edge_id = f"edge_{uuid.uuid4().hex[:10]}"
        before = self._json_safe_clone(self.current_graph)
        edges.append(
            {
                "edge_id": edge_id,
                "src_id": src,
                "dst_id": dst,
                "relation": rel,
                "confidence": 0.0,
                "validator_flags": [],
                "provenance": "human_edit",
            }
        )
        graph["edges"] = edges
        self._push_scene_graph_undo(before, reason="add_edge")
        self.current_graph = graph
        self._replace_current_graph_in_bundle(graph)
        self._persist_current_scene_graph_bundle("scene_graph_edge_add_saved")
        self._record_change(
            task_type="scene_graph",
            item_id=edge_id,
            op="add_edge",
            field_path="edges",
            before=before,
            after=graph,
            reason=reason,
        )
        self._render_graph()
        return True

    def _on_player_ctrl_object_pick_for_edge(self, entity_id: str) -> None:
        """Ctrl+click two boxes on player: create an edge between them."""
        node_id = str(entity_id or "").strip()
        if not node_id:
            return
        if not isinstance(self.current_graph, dict):
            self._set_status("No scene graph loaded.", status_type="warning")
            return
        node_ids = {
            str(n.get("entity_id", "") or "").strip()
            for n in list(self.current_graph.get("nodes") or [])
            if isinstance(n, dict)
        }
        if node_id not in node_ids:
            self._set_status(f"Node not found: {node_id}", status_type="warning")
            return

        # Keep table/overlay focus synced with the clicked object.
        self._on_sg_visualizer_node_selected(node_id)

        first = str(self._sg_ctrl_edge_pick_first or "").strip()
        if not first:
            self._sg_ctrl_edge_pick_first = node_id
            self._set_status(
                f"Edge pick step 1/2: selected {node_id}. Ctrl+click another object to add edge.",
                status_type="info",
            )
            return

        if first == node_id:
            self._set_status("Edge pick cancelled: choose a different second object.", status_type="warning")
            self._sg_ctrl_edge_pick_first = ""
            return

        relation = self._pick_edge_relation(
            title="Add Edge",
            prompt=f"Relation ({first} -> {node_id}):",
            current="left_of",
        )
        self._sg_ctrl_edge_pick_first = ""
        if not relation:
            return
        self._add_scene_graph_edge_between(
            src_id=first,
            dst_id=node_id,
            relation=relation,
            reason="ctrl_click_pair_add_edge",
        )

    def _delete_selected_scene_graph_edge(self) -> None:
        if not isinstance(self.current_graph, dict):
            self._set_status("No scene graph loaded.", status_type="warning")
            return
        row = self._selected_row(self.sg_edges_table) if hasattr(self, "sg_edges_table") else -1
        if row < 0 or row >= len(self._edge_rows):
            self._set_status("Select an edge first.", status_type="warning")
            return
        edge_id = str(self._edge_rows[row].get("edge_id", "") or "")
        if not edge_id:
            return
        graph = self._json_safe_clone(self.current_graph)
        before = self._json_safe_clone(self.current_graph)
        edges = [dict(e) for e in list(graph.get("edges") or []) if isinstance(e, dict)]
        new_edges = [e for e in edges if str(e.get("edge_id", "") or "") != edge_id]
        if len(new_edges) == len(edges):
            self._set_status("Selected edge was not found in graph.", status_type="warning")
            return
        graph["edges"] = new_edges
        self._push_scene_graph_undo(before, reason="delete_edge")
        self.current_graph = graph
        self._replace_current_graph_in_bundle(graph)
        self._persist_current_scene_graph_bundle("scene_graph_edge_delete_saved")
        self._record_change(
            task_type="scene_graph",
            item_id=edge_id,
            op="delete_edge",
            field_path="edges",
            before=before,
            after=graph,
            reason="manual edge delete",
        )
        self._render_graph()

    def _rename_selected_scene_graph_edge_relation(self) -> None:
        if not isinstance(self.current_graph, dict):
            self._set_status("No scene graph loaded.", status_type="warning")
            return
        row = self._selected_row(self.sg_edges_table) if hasattr(self, "sg_edges_table") else -1
        if row < 0 or row >= len(self._edge_rows):
            self._set_status("Select an edge first.", status_type="warning")
            return
        edge_id = str(self._edge_rows[row].get("edge_id", "") or "")
        if not edge_id:
            self._set_status("Selected edge id is empty.", status_type="warning")
            return
        current_relation = ""
        for edge in list(self.current_graph.get("edges") or []):
            if str(edge.get("edge_id", "") or "") == edge_id:
                current_relation = str(edge.get("relation", "") or "").strip()
                break
        new_relation = self._pick_edge_relation(
            title="Rename Edge Relation",
            prompt="New relation:",
            current=current_relation or "left_of",
        )
        if not new_relation:
            return
        if new_relation == current_relation:
            self._set_status("Relation unchanged.", status_type="info")
            return

        graph = self._json_safe_clone(self.current_graph)
        changed = False
        for edge in list(graph.get("edges") or []):
            if str(edge.get("edge_id", "") or "") != edge_id:
                continue
            edge["relation"] = new_relation
            changed = True
            break
        if not changed:
            self._set_status("Selected edge was not found in graph.", status_type="warning")
            return
        before = self._json_safe_clone(self.current_graph)
        self._push_scene_graph_undo(before, reason="rename_edge_relation")
        self.current_graph = graph
        self._replace_current_graph_in_bundle(graph)
        self._persist_current_scene_graph_bundle("scene_graph_edge_relation_renamed")
        self._record_change(
            task_type="scene_graph",
            item_id=edge_id,
            op="rename_edge_relation",
            field_path="edges",
            before=before,
            after=graph,
            reason="manual edge relation rename",
        )
        self._render_graph()

    def _on_sg_visualizer_node_selected(self, entity_id: str) -> None:
        """Handle node selection from the visualizer."""
        if self._sync_graph_selection:
            return
        
        self._sync_graph_selection = True
        try:
            # Find and select the corresponding row in the nodes table
            if entity_id in self._node_row_by_id:
                row = self._node_row_by_id[entity_id]
                self.sg_nodes_table.selectRow(row)
                self._reset_table_bg(self.sg_nodes_table)
                self._reset_table_bg(self.sg_edges_table)
                self._set_row_bg(self.sg_nodes_table, row, QColor(233, 245, 255))
                self._apply_scene_graph_overlay_to_player()
                
                # Also highlight connected edges
                for i, edge in enumerate(self._edge_rows):
                    if edge.get("src") == entity_id or edge.get("dst") == entity_id:
                        self._set_row_bg(self.sg_edges_table, i, QColor(245, 250, 255))
        finally:
            self._sync_graph_selection = False

    def _on_sg_visualizer_edge_selected(self, edge_id: str) -> None:
        """Handle edge selection from the visualizer."""
        if self._sync_graph_selection:
            return
        
        self._sync_graph_selection = True
        try:
            # Find and select the corresponding row in the edges table
            edge_row = -1
            for i, edge in enumerate(self._edge_rows):
                if edge.get("edge_id") == edge_id:
                    edge_row = i
                    break
            
            if edge_row >= 0:
                self.sg_edges_table.selectRow(edge_row)
                self._reset_table_bg(self.sg_nodes_table)
                self._reset_table_bg(self.sg_edges_table)
                self._set_row_bg(self.sg_edges_table, edge_row, QColor(233, 245, 255))
                self._apply_scene_graph_overlay_to_player()
                
                # Also highlight the source and destination nodes
                edge = self._edge_rows[edge_row]
                src_row = self._node_row_by_id.get(str(edge.get("src", "")), -1)
                dst_row = self._node_row_by_id.get(str(edge.get("dst", "")), -1)
                self._set_row_bg(self.sg_nodes_table, src_row, QColor(245, 250, 255))
                self._set_row_bg(self.sg_nodes_table, dst_row, QColor(245, 250, 255))
        finally:
            self._sync_graph_selection = False


    def _make_cycle_stat_card(self, *, title: str, accent: str, tooltip: str = "") -> tuple[QFrame, QLabel, QLabel]:
        card = QFrame()
        card.setMinimumHeight(82)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        card.setFrameShape(QFrame.StyledPanel)
        card.setStyleSheet(
            "QFrame {"
            f"background: {accent};"
            "border: 1px solid rgba(255, 255, 255, 0.18);"
            "border-radius: 12px;"
            "}"
        )
        if tooltip:
            card.setToolTip(tooltip)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(0)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #F8FAFC; font-size: 14px; font-weight: 800;")

        value_label = QLabel("--")
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setStyleSheet("color: #FFFFFF; font-size: 24px; font-weight: 800;")

        row.addWidget(title_label, 1)
        row.addWidget(value_label, 0)
        layout.addLayout(row)

        detail_label = QLabel("")
        detail_label.hide()

        return card, value_label, detail_label

    def _set_cycle_stat_card(self, key: str, value: str, detail: str = "") -> None:
        widget = dict(getattr(self, "_cycle_stat_widgets", {}).get(key) or {})
        value_label = widget.get("value")
        detail_label = widget.get("detail")
        if isinstance(value_label, QLabel):
            value_label.setText(str(value))
        if isinstance(detail_label, QLabel):
            detail_label.setText(str(detail))

    def _current_cycle_update(self) -> Dict[str, object]:
        result = self.current_cycle_result if isinstance(self.current_cycle_result, dict) else {}
        graph_after = dict(result.get("graph_after") or {}) if isinstance(result, dict) else {}
        if graph_after:
            return dict(((graph_after.get("metadata") or {}).get("cycle_update")) or {})
        graph = self.current_graph if isinstance(self.current_graph, dict) else {}
        return dict(((graph.get("metadata") or {}).get("cycle_update")) or {})

    def _cycle_memory_stats(self, memory: Optional[Dict[str, object]] = None) -> Dict[str, int]:
        return summarize_correction_memory(memory or self._correction_memory or default_correction_memory())

    def _find_cycle_memory_adjustment(self, claim_id: str) -> Dict[str, object]:
        target = str(claim_id or "").strip()
        if not target:
            return {}
        for row in list(self._current_cycle_update().get("memory_adjustments") or []):
            if str((row or {}).get("claim_id", "") or "").strip() == target:
                return dict(row or {})
        return {}

    @staticmethod
    def _cycle_priority_band(priority: float) -> str:
        if priority >= 0.85:
            return "Critical"
        if priority >= 0.70:
            return "High"
        if priority >= 0.50:
            return "Medium"
        return "Low"

    @staticmethod
    def _cycle_bool_text(value: object) -> str:
        return "Yes" if bool(value) else "No"

    def _cycle_review_changes(self) -> List[Dict[str, object]]:
        return [
            row
            for row in self._changes_for_task("Video Scene Graph")
            if str(row.get("op", "")).strip() == "cycle_arbitration"
            and str(row.get("status", "")).strip() == "proposed"
        ]

    def _filtered_cycle_review_changes(self) -> List[Dict[str, object]]:
        rows = list(self._cycle_review_changes())
        filter_mode = "All Pending"
        if hasattr(self, "sg_cycle_review_filter"):
            filter_mode = str(self.sg_cycle_review_filter.currentText() or "All Pending")
        term = ""
        if hasattr(self, "sg_cycle_review_search"):
            term = str(self.sg_cycle_review_search.text() or "").strip().lower()

        filtered: List[Dict[str, object]] = []
        for row in rows:
            claim = dict(row.get("before") or {})
            review = dict(row.get("after") or {})
            claim_type = str(claim.get("claim_type", "") or "").strip().lower()
            try:
                priority = float(review.get("priority", 0.0) or 0.0)
            except Exception:
                priority = 0.0
            locked = bool(review.get("locked", False))
            try:
                memory_bonus = float(review.get("memory_bonus", 0.0) or 0.0)
            except Exception:
                memory_bonus = 0.0

            if filter_mode == "High Priority" and priority < 0.70:
                continue
            if filter_mode == "Locked" and not locked:
                continue
            if filter_mode == "Memory Adjusted" and memory_bonus <= 0.0 and not self._find_cycle_memory_adjustment(str(claim.get("claim_id", "") or row.get("item_id", ""))):
                continue
            if filter_mode == "Labels" and claim_type != "label":
                continue
            if filter_mode == "Relations" and claim_type != "relation":
                continue
            if filter_mode == "Geometry" and claim_type != "bbox":
                continue
            if filter_mode == "Attributes" and claim_type != "attribute":
                continue
            if filter_mode == "Existence" and claim_type != "existence":
                continue

            if term:
                haystack = " ".join(
                    [
                        str(row.get("item_id", "") or ""),
                        str(review.get("question", "") or ""),
                        str(claim.get("subject_id", "") or ""),
                        str(claim.get("object_id", "") or ""),
                        str(claim.get("predicate", "") or ""),
                        str(claim.get("value", "") or ""),
                        str(review.get("corrected_value", "") or ""),
                        str(review.get("suggested_value", "") or ""),
                        " ".join([str(x) for x in list(review.get("question_options") or [])]),
                        " ".join(
                            [
                                str(x.get("label", "") or "") + " " + str(x.get("description", "") or "")
                                for x in self._cycle_review_resolution_options(review)
                                if isinstance(x, dict)
                            ]
                        ),
                    ]
                ).lower()
                if term not in haystack:
                    continue
            filtered.append(row)
        return filtered

    def _build_cycle_review_item_label(self, row: Dict[str, object]) -> str:
        review = dict(row.get("after") or {})
        claim = dict(row.get("before") or {})
        question = str(review.get("question", "") or row.get("reason", "") or "").strip()
        claim_type = str(claim.get("claim_type", "") or "claim").strip().title()
        if claim_type.lower() == "bbox":
            claim_type = "Geometry"
        try:
            priority = float(review.get("priority", 0.0) or 0.0)
        except Exception:
            priority = 0.0
        locked = bool(review.get("locked", False))
        try:
            memory_bonus = float(review.get("memory_bonus", 0.0) or 0.0)
        except Exception:
            memory_bonus = 0.0
        band = self._cycle_priority_band(priority)
        modifiers: List[str] = [claim_type, f"p={priority:.2f}"]
        if list(review.get("question_options") or []):
            modifiers.append("choices")
        if memory_bonus > 0.0:
            modifiers.append(f"m={memory_bonus:.2f}")
        if locked:
            modifiers.append("locked")
        prefix = " | ".join([band] + modifiers)
        return f"{prefix} | {question}" if question else prefix

    def _refresh_cycle_review_panel(self) -> None:
        if not hasattr(self, "sg_cycle_review_list"):
            return
        current_change_id = self._selected_cycle_review_change_id()
        self.sg_cycle_review_list.clear()
        rows = self._filtered_cycle_review_changes()
        for row in rows:
            review = dict(row.get("after") or {})
            try:
                priority = float(review.get("priority", 0.0) or 0.0)
            except Exception:
                priority = 0.0
            label = self._build_cycle_review_item_label(row)
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, str(row.get("change_id", "")))
            if priority >= 0.85:
                item.setForeground(QColor("#B42318"))
            elif priority >= 0.70:
                item.setForeground(QColor("#B54708"))
            else:
                item.setForeground(QColor("#344054"))
            self.sg_cycle_review_list.addItem(item)
        if rows:
            target_row = 0
            if current_change_id:
                for i in range(self.sg_cycle_review_list.count()):
                    item = self.sg_cycle_review_list.item(i)
                    if item is not None and str(item.data(Qt.UserRole) or "") == current_change_id:
                        target_row = i
                        break
            self.sg_cycle_review_list.setCurrentRow(target_row)
            if hasattr(self, "btn_cycle_review_confirm"):
                self.btn_cycle_review_confirm.setEnabled(True)
            if hasattr(self, "btn_cycle_review_reject"):
                self.btn_cycle_review_reject.setEnabled(True)
        else:
            if hasattr(self, "sg_cycle_review_detail"):
                if self._cycle_review_changes():
                    self.sg_cycle_review_detail.setPlainText("No pending items match the current queue filter.")
                else:
                    self.sg_cycle_review_detail.setPlainText("No pending cycle review items.")
            if hasattr(self, "sg_cycle_review_choice_hint"):
                self.sg_cycle_review_choice_hint.setText("Select a structured resolution when options are available.")
            if hasattr(self, "sg_cycle_review_choice_combo"):
                self.sg_cycle_review_choice_combo.blockSignals(True)
                self.sg_cycle_review_choice_combo.clear()
                self.sg_cycle_review_choice_combo.blockSignals(False)
                self.sg_cycle_review_choice_combo.setEnabled(False)
            if hasattr(self, "sg_cycle_review_corrected_value"):
                self.sg_cycle_review_corrected_value.clear()
            if hasattr(self, "btn_cycle_review_use_suggested"):
                self.btn_cycle_review_use_suggested.setEnabled(False)
            if hasattr(self, "btn_cycle_review_clear_choice"):
                self.btn_cycle_review_clear_choice.setEnabled(False)
            if hasattr(self, "btn_cycle_review_confirm"):
                self.btn_cycle_review_confirm.setEnabled(False)
                self.btn_cycle_review_confirm.setText("Confirm")
            if hasattr(self, "btn_cycle_review_reject"):
                self.btn_cycle_review_reject.setEnabled(False)
                self.btn_cycle_review_reject.setText("Reject")
            self._refresh_cycle_review_visual_preview()
        self._refresh_cycle_memory_summary()

    def _selected_cycle_review_change_id(self) -> str:
        if not hasattr(self, "sg_cycle_review_list"):
            return ""
        row = self.sg_cycle_review_list.currentRow()
        if row < 0:
            return ""
        item = self.sg_cycle_review_list.item(row)
        if item is None:
            return ""
        return str(item.data(Qt.UserRole) or "")

    @staticmethod
    def _cycle_review_resolution_options(review: Dict[str, object]) -> List[Dict[str, object]]:
        raw = list(review.get("resolution_options") or [])
        out: List[Dict[str, object]] = []
        for row in raw:
            if not isinstance(row, dict):
                continue
            value = str(row.get("value", "") or "").strip()
            if not value:
                continue
            try:
                score = float(row.get("score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            out.append(
                {
                    "value": value,
                    "label": str(row.get("label", value) or value),
                    "description": str(row.get("description", "") or ""),
                    "bbox": list(row.get("bbox") or []),
                    "clear_mask": bool(row.get("clear_mask", False)),
                    "score": score,
                    "target_node_id": str(row.get("target_node_id", "") or ""),
                }
            )
        if out:
            return out
        for token in [str(x).strip() for x in list(review.get("question_options") or []) if str(x).strip()]:
            out.append(
                {
                    "value": token,
                    "label": token,
                    "description": "",
                    "bbox": [],
                    "clear_mask": False,
                    "score": 0.0,
                    "target_node_id": "",
                }
            )
        return out

    def _selected_cycle_review_row(self) -> Dict[str, object]:
        change_id = self._selected_cycle_review_change_id()
        if not change_id:
            return {}
        idx = self._find_validation_change_index(change_id)
        if idx < 0:
            return {}
        return dict(self._validation_changes[idx] or {})

    def _set_cycle_review_combo_value(self, value: str) -> None:
        if not hasattr(self, "sg_cycle_review_choice_combo"):
            return
        target = str(value or "").strip()
        combo = self.sg_cycle_review_choice_combo
        for i in range(combo.count()):
            data = str(combo.itemData(i) or "").strip()
            if data == target:
                combo.setCurrentIndex(i)
                return
        combo.setCurrentIndex(0 if combo.count() > 0 else -1)

    def _selected_cycle_review_resolution_value(self) -> str:
        combo_value = ""
        if hasattr(self, "sg_cycle_review_choice_combo") and self.sg_cycle_review_choice_combo.count() > 0:
            combo_value = str(self.sg_cycle_review_choice_combo.currentData() or "").strip()
        typed_value = str(self.sg_cycle_review_corrected_value.text() or "").strip() if hasattr(self, "sg_cycle_review_corrected_value") else ""
        return combo_value or typed_value

    def _clear_cycle_review_resolution(self) -> None:
        if hasattr(self, "sg_cycle_review_choice_combo") and self.sg_cycle_review_choice_combo.count() > 0:
            self.sg_cycle_review_choice_combo.setCurrentIndex(0)
        if hasattr(self, "sg_cycle_review_corrected_value"):
            self.sg_cycle_review_corrected_value.clear()
        self._refresh_cycle_review_action_controls()
        self._refresh_cycle_review_visual_preview()

    def _use_cycle_review_suggested_choice(self) -> None:
        row = self._selected_cycle_review_row()
        review = dict(row.get("after") or {})
        suggested = str(review.get("suggested_value", "") or "").strip()
        if not suggested:
            return
        options = [str(x.get("value", "") or "").strip() for x in self._cycle_review_resolution_options(review) if str(x.get("value", "") or "").strip()]
        if options and hasattr(self, "sg_cycle_review_choice_combo"):
            self._set_cycle_review_combo_value(suggested)
        elif hasattr(self, "sg_cycle_review_corrected_value"):
            self.sg_cycle_review_corrected_value.setText(suggested)
        self._refresh_cycle_review_action_controls()
        self._refresh_cycle_review_visual_preview()

    def _refresh_cycle_review_action_controls(self) -> None:
        row = self._selected_cycle_review_row()
        review = dict(row.get("after") or {})
        claim = dict(row.get("before") or {})
        proposed = self._proposed_value_from_claim_row(claim)
        claim_type = str(claim.get("claim_type", "") or "").strip().lower()
        combo_value = ""
        if hasattr(self, "sg_cycle_review_choice_combo") and self.sg_cycle_review_choice_combo.count() > 0:
            combo_value = str(self.sg_cycle_review_choice_combo.currentData() or "").strip()
        typed_value = str(self.sg_cycle_review_corrected_value.text() or "").strip() if hasattr(self, "sg_cycle_review_corrected_value") else ""
        resolved_value = combo_value or typed_value
        has_resolution = bool(resolved_value and resolved_value != proposed)
        if hasattr(self, "btn_cycle_review_confirm"):
            if claim_type == "bbox":
                self.btn_cycle_review_confirm.setText("Apply Selected Box" if has_resolution else "Keep Current Box")
            else:
                self.btn_cycle_review_confirm.setText("Confirm Selection" if has_resolution else "Confirm Current")
        if hasattr(self, "btn_cycle_review_reject"):
            if claim_type == "bbox":
                self.btn_cycle_review_reject.setText("Flag Geometry Issue")
            else:
                self.btn_cycle_review_reject.setText("Reject + Correct" if has_resolution else "Reject Claim")
        if hasattr(self, "btn_cycle_review_use_suggested"):
            suggested = str(review.get("suggested_value", "") or "").strip()
            self.btn_cycle_review_use_suggested.setEnabled(bool(suggested and suggested != proposed))

    def _sync_cycle_review_resolution_controls(self, row: Dict[str, object]) -> None:
        review = dict(row.get("after") or {})
        claim = dict(row.get("before") or {})
        proposed = self._proposed_value_from_claim_row(claim)
        corrected = str(review.get("corrected_value", "") or "").strip()
        suggested = str(review.get("suggested_value", "") or "").strip()
        try:
            suggested_score = float(review.get("suggested_score", 0.0) or 0.0)
        except Exception:
            suggested_score = 0.0
        resolution_options = self._cycle_review_resolution_options(review)
        options = [str(x.get("value", "") or "").strip() for x in resolution_options if str(x.get("value", "") or "").strip()]
        suggested_label = suggested
        for option_row in resolution_options:
            if str(option_row.get("value", "") or "").strip() == suggested:
                suggested_label = str(option_row.get("label", suggested) or suggested)
                break

        if hasattr(self, "sg_cycle_review_choice_combo"):
            combo = self.sg_cycle_review_choice_combo
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(f"Keep current ({proposed or 'no change'})", "")
            for option_row in resolution_options:
                option = str(option_row.get("value", "") or "").strip()
                label = str(option_row.get("label", option) or option)
                if option == proposed:
                    label += "  [Current]"
                if option == suggested and option != proposed:
                    label += "  [Suggested]"
                combo.addItem(label, option)
            combo.blockSignals(False)
            preferred = corrected or (suggested if suggested and suggested != proposed else "")
            self._set_cycle_review_combo_value(preferred)
            combo.setEnabled(bool(options))

        if hasattr(self, "sg_cycle_review_corrected_value"):
            self.sg_cycle_review_corrected_value.setText(corrected if corrected and corrected not in options else "")
            if options:
                self.sg_cycle_review_corrected_value.setPlaceholderText("Optional manual override outside the suggested options")
            else:
                self.sg_cycle_review_corrected_value.setPlaceholderText("Optional corrected label / relation / value")

        if hasattr(self, "sg_cycle_review_choice_hint"):
            if options:
                suggestion_text = (
                    f"Suggested: {suggested_label} ({suggested_score:.2f})"
                    if suggested and suggested != proposed
                    else f"Current machine proposal: {proposed or '--'}"
                )
                self.sg_cycle_review_choice_hint.setText(
                    f"Structured review available. Choose one canonical option for this claim. {suggestion_text}"
                )
            else:
                self.sg_cycle_review_choice_hint.setText(
                    "No structured choices were generated for this claim. Use manual override only when needed."
                )

        if hasattr(self, "btn_cycle_review_clear_choice"):
            self.btn_cycle_review_clear_choice.setEnabled(bool(options or corrected))
        self._refresh_cycle_review_action_controls()

    def _find_validation_change_index(self, change_id: str) -> int:
        target = str(change_id or "").strip()
        for i, row in enumerate(self._validation_changes):
            if str(row.get("change_id", "")).strip() == target:
                return i
        return -1

    @staticmethod
    def _proposed_value_from_claim_row(claim_row: Dict[str, object]) -> str:
        claim_type = str(claim_row.get("claim_type", "") or "").strip()
        if claim_type == "relation":
            return str(claim_row.get("predicate", "") or "").strip()
        return str(claim_row.get("value", "") or "").strip()

    @staticmethod
    def _resolution_option_label(
        resolution_options: List[Dict[str, object]],
        value: str,
    ) -> str:
        target = str(value or "").strip()
        for option_row in resolution_options:
            if str(option_row.get("value", "") or "").strip() == target:
                return str(option_row.get("label", target) or target)
        return target

    def _refresh_cycle_review_visual_preview(self) -> None:
        if self._current_task_name() != "Video Scene Graph":
            return
        if not isinstance(self.current_graph, dict):
            return
        self._apply_scene_graph_overlay_to_player()

    def _render_cycle_review_detail(self, row: int) -> None:
        if not hasattr(self, "sg_cycle_review_list"):
            return
        if row < 0:
            if hasattr(self, "sg_cycle_review_detail"):
                self.sg_cycle_review_detail.clear()
            self._refresh_cycle_review_visual_preview()
            return
        item = self.sg_cycle_review_list.item(row)
        if item is None:
            return
        change_id = str(item.data(Qt.UserRole) or "")
        idx = self._find_validation_change_index(change_id)
        if idx < 0:
            return
        row_obj = dict(self._validation_changes[idx] or {})
        claim = dict(row_obj.get("before") or {})
        review = dict(row_obj.get("after") or {})
        proposed = self._proposed_value_from_claim_row(claim)
        corrected = str(review.get("corrected_value", "") or "").strip()
        memory_adjustment = self._find_cycle_memory_adjustment(str(claim.get("claim_id", "") or row_obj.get("item_id", "")))
        self._sync_cycle_review_resolution_controls(row_obj)
        try:
            priority = float(review.get("priority", 0.0) or 0.0)
        except Exception:
            priority = 0.0
        try:
            memory_bonus = float(review.get("memory_bonus", 0.0) or 0.0)
        except Exception:
            memory_bonus = 0.0
        suggested = str(review.get("suggested_value", "") or "").strip()
        try:
            suggested_score = float(review.get("suggested_score", 0.0) or 0.0)
        except Exception:
            suggested_score = 0.0
        resolution_options = self._cycle_review_resolution_options(review)
        question_options = [str(x.get("label", x.get("value", "")) or "").strip() for x in resolution_options if str(x.get("value", "") or "").strip()]
        suggested_label = suggested
        for option_row in resolution_options:
            if str(option_row.get("value", "") or "").strip() == suggested:
                suggested_label = str(option_row.get("label", suggested) or suggested)
                break
        runtime = dict((self.current_cycle_result or {}).get("runtime") or {})
        lines = [
            "Cycle Review Item",
            "",
            "Decision Context:",
            f"- Priority: {priority:.2f} ({self._cycle_priority_band(priority)})",
            f"- Question: {str(review.get('question', '') or row_obj.get('reason', '') or '--')}",
            f"- Locked by memory: {self._cycle_bool_text(review.get('locked', False))}",
            f"- Memory bonus: {memory_bonus:.2f}",
            "",
            "Claim:",
            f"- Claim ID: {str(claim.get('claim_id', '') or row_obj.get('item_id', '') or '--')}",
            f"- Type: {str(claim.get('claim_type', '') or '--')}",
            f"- Subject: {str(claim.get('subject_id', '') or '--')}",
            f"- Predicate: {str(claim.get('predicate', '') or '--')}",
            f"- Object: {str(claim.get('object_id', '') or '--')}",
            f"- Proposed value: {proposed or '--'}",
            f"- Corrected value: {corrected or '--'}",
        ]
        if question_options:
            lines.append(f"- Structured options: {', '.join(question_options)}")
        if suggested:
            lines.append(f"- Suggested choice: {suggested_label} ({suggested_score:.2f})")
        lines.extend(["", "Memory Influence:"])
        if memory_adjustment:
            for key in ("claim_id", "claim_type", "support_threshold", "reject_threshold", "lock_applied", "confusion_count"):
                if key in memory_adjustment:
                    lines.append(f"- {key}: {memory_adjustment.get(key)}")
        else:
            lines.append("- No explicit memory threshold adjustment recorded for this claim.")
        lines.extend(
            [
                "",
                "Runtime:",
                f"- Verifier provider: {str(runtime.get('verifier_provider', '--') or '--')}",
                f"- Verifier model: {str(runtime.get('verifier_model_id', '--') or '--')}",
                f"- Cached verifier: {self._cycle_bool_text(runtime.get('verifier_cached', False))}",
                f"- Image path: {str(runtime.get('image_path', '--') or '--')}",
            ]
        )
        if hasattr(self, "sg_cycle_review_detail"):
            self.sg_cycle_review_detail.setPlainText("\n".join(lines))
        self._refresh_cycle_review_visual_preview()

    def _build_cycle_analytics_payload(self) -> Dict[str, object]:
        result = dict(self.current_cycle_result or {})
        summary = dict(result.get("summary") or {})
        runtime = dict(result.get("runtime") or {})
        cycle_update = self._current_cycle_update()
        queue = list(result.get("human_queue") or [])
        pending_rows = self._cycle_review_changes()
        visible_rows = self._filtered_cycle_review_changes()
        claim_type_counts = Counter(
            str((row.get("before") or {}).get("claim_type", "") or "unknown").strip()
            for row in pending_rows
        )
        visible_type_counts = Counter(
            str((row.get("before") or {}).get("claim_type", "") or "unknown").strip()
            for row in visible_rows
        )
        high_priority_count = 0
        locked_count = 0
        memory_adjusted_count = 0
        for row in pending_rows:
            review = dict(row.get("after") or {})
            try:
                priority = float(review.get("priority", 0.0) or 0.0)
            except Exception:
                priority = 0.0
            try:
                memory_bonus = float(review.get("memory_bonus", 0.0) or 0.0)
            except Exception:
                memory_bonus = 0.0
            if priority >= 0.70:
                high_priority_count += 1
            if bool(review.get("locked", False)):
                locked_count += 1
            if memory_bonus > 0.0:
                memory_adjusted_count += 1

        action_hint = "Queue is clear. No residual claims need arbitration."
        if pending_rows:
            action_hint = "Review the highest-priority items first."
            if high_priority_count > 0:
                action_hint = f"Start with the {high_priority_count} high-priority items."
            elif locked_count > 0:
                action_hint = "Locked items exist; verify them carefully before overriding."
            elif len(visible_rows) != len(pending_rows):
                action_hint = "Current filter is hiding part of the queue."
        return {
            "summary": summary,
            "runtime": runtime,
            "memory": self._cycle_memory_stats(),
            "cycle_update": cycle_update,
            "queue_counts": {
                "total": len(queue),
                "pending": len(pending_rows),
                "visible_after_filter": len(visible_rows),
            },
            "pending_by_type": dict(claim_type_counts),
            "visible_by_type": dict(visible_type_counts),
            "high_priority_count": int(high_priority_count),
            "locked_count": int(locked_count),
            "memory_bonus_count": int(memory_adjusted_count),
            "top_review_questions": list(summary.get("top_review_questions") or []),
            "action_hint": action_hint,
        }

    def _refresh_cycle_analytics_views(self) -> None:
        analytics_payload = self._build_cycle_analytics_payload()
        if hasattr(self, "sg_cycle_analytics_text"):
            if not isinstance(self.current_cycle_result, dict):
                self.sg_cycle_analytics_text.setPlainText(
                    "Cycle analytics will appear here after the first cycle refine run.\n\n"
                    f"Current memory footprint: {json.dumps(self._cycle_memory_stats(), ensure_ascii=True, indent=2)}"
                )
            else:
                self.sg_cycle_analytics_text.setPlainText(json.dumps(analytics_payload, ensure_ascii=True, indent=2))
        if hasattr(self, "sg_cycle_session_text"):
            payload = self._build_cycle_result_export_payload() if isinstance(self.current_cycle_result, dict) else {}
            self.sg_cycle_session_text.setPlainText(json.dumps(payload, ensure_ascii=True, indent=2) if payload else "")

    def _refresh_cycle_summary(self) -> None:
        if not getattr(self, "_cycle_stat_widgets", None):
            return
        memory = self._cycle_memory_stats()
        aliases = int(memory.get("prompt_aliases", 0) or 0)
        locks = int(memory.get("verified_locks", 0) or 0)
        if not isinstance(self.current_cycle_result, dict):
            self._set_cycle_stat_card("session", "--", f"{0} probes | {'unknown'}")
            self._set_cycle_stat_card("queue", "0", "0 pending | 0 visible | 0 total")
            self._set_cycle_stat_card("accepted", "0", "Claims automatically accepted in the current cycle")
            self._set_cycle_stat_card("flagged", "0", "Claims escalated or marked risky in the current cycle")
            self._set_cycle_stat_card("adjusted", "0", "Claims whose thresholds were adjusted by correction memory")
            self._set_cycle_stat_card("memory", str(aliases + locks), f"aliases {aliases} | locks {locks}")
            self._refresh_cycle_analytics_views()
            return
        result = dict(self.current_cycle_result or {})
        runtime = dict(result.get("runtime") or {})
        summary = dict(result.get("summary") or {})
        verifier_provider = str(runtime.get("verifier_provider", "unknown") or "unknown")
        verifier_model = str(runtime.get("verifier_model_id", "") or "").strip()
        verifier_warning = str(runtime.get("verifier_warning", "") or "").strip()
        verifier_cached = bool(runtime.get("verifier_cached", False))
        verifier_label = verifier_provider if not verifier_model else f"{verifier_provider} ({verifier_model})"
        if verifier_cached:
            verifier_label += " cached"
        pending_queue = int(summary.get("pending_queue", len(self._filtered_cycle_review_changes())) or 0)
        visible_queue = len(self._filtered_cycle_review_changes())
        total_queue = int(summary.get("queue_count", len(list(result.get("human_queue") or result.get("review_queue") or []))) or 0)
        accepted = int(summary.get("accepted_claim_count", len(list(self._current_cycle_update().get("accepted_claim_ids") or []))) or 0)
        flagged = int(summary.get("flagged_claim_count", len(list(self._current_cycle_update().get("flagged_claim_ids") or []))) or 0)
        adjusted = int(summary.get("memory_adjusted_count", len(list(self._current_cycle_update().get("memory_adjustments") or []))) or 0)
        rounds_run = int(summary.get("rounds_run", len(list(result.get("rounds") or []))) or 0)
        probe_count = int(summary.get("probe_count", len(list(result.get("probe_results") or []))) or 0)
        memory_footprint = aliases + locks
        self._set_cycle_stat_card("session", str(rounds_run), f"{probe_count} probes | {verifier_provider}")
        self._set_cycle_stat_card("queue", str(pending_queue), f"{pending_queue} pending | {visible_queue} visible | {total_queue} total")
        self._set_cycle_stat_card("accepted", str(accepted), "Claims automatically accepted in the current cycle")
        self._set_cycle_stat_card("flagged", str(flagged), "Claims escalated or marked risky in the current cycle")
        self._set_cycle_stat_card("adjusted", str(adjusted), "Claims whose thresholds were adjusted by correction memory")
        self._set_cycle_stat_card("memory", str(memory_footprint), f"aliases {aliases} | locks {locks}")
        warning_preview = f" | warning={verifier_warning}" if verifier_warning else ""
        self._set_cycle_stat_card("session", str(rounds_run), f"{probe_count} probes | {verifier_provider}")
        self._set_cycle_stat_card("queue", str(pending_queue), f"{pending_queue} pending | {visible_queue} visible | {total_queue} total")
        self._set_cycle_stat_card("accepted", str(accepted), f"single {len(self.single_turn_items)} | multi {len(self.multi_turn_items)}")
        self._set_cycle_stat_card("flagged", str(flagged), verifier_label + warning_preview.replace(" | ", " ").strip())
        self._set_cycle_stat_card("adjusted", str(adjusted), f"adjusted by memory | probes {probe_count}")
        self._set_cycle_stat_card("memory", str(memory_footprint), f"aliases {aliases} | locks {locks}")
        self._refresh_cycle_analytics_views()

    def _save_correction_memory_state(self) -> None:
        try:
            save_correction_memory(self._cycle_memory_path, self._correction_memory)
        except Exception:
            return

    def _refresh_cycle_memory_summary(self) -> None:
        if not hasattr(self, "sg_cycle_memory_summary"):
            return
        memory = self._cycle_memory_stats()
        self.sg_cycle_memory_summary.setText(
            f"Memory: label_confusions={memory['label_confusions']} | "
            f"relation_confusions={memory['relation_confusions']} | "
            f"prompt_aliases={memory['prompt_aliases']} | "
            f"verified_locks={memory['verified_locks']} | "
            f"file={os.path.basename(self._cycle_memory_path)}"
        )

    def _export_cycle_memory(self) -> None:
        default_name = os.path.basename(self._cycle_memory_path) or "impact_cycle_memory.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Correction Memory",
            os.path.join(self._repo_root, default_name),
            "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            save_correction_memory(path, self._correction_memory)
            self._set_status(f"Exported correction memory: {os.path.basename(path)}", status_type="success")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to export correction memory:\n{exc}")

    def _import_cycle_memory(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Correction Memory",
            self._repo_root,
            "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            imported = load_correction_memory(path)
            self._correction_memory = merge_correction_memories([self._correction_memory, imported])
            self._save_correction_memory_state()
            self._sync_current_cycle_result_after_graph_change()
            self._refresh_cycle_memory_summary()
            self._refresh_cycle_summary()
            self._set_status(
                f"Imported correction memory and merged with current state: {os.path.basename(path)}",
                status_type="success",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", f"Failed to import correction memory:\n{exc}")

    def _record_cycle_human_queue(
        self,
        *,
        queue: List[Dict[str, object]],
        claims: Dict[str, object],
    ) -> int:
        if not self._is_validate_mode():
            return 0
        validator = self._require_validator()
        if validator is None:
            return 0

        existing_ids = {
            str(row.get("item_id", "")).strip(): idx
            for idx, row in enumerate(self._validation_changes)
            if str(row.get("task_type", "")).strip() == "scene_graph"
            and str(row.get("op", "")).strip() == "cycle_arbitration"
        }
        incoming_ids = {
            str(item.get("claim_id", "") or "").strip()
            for item in list(queue or [])
            if str(item.get("claim_id", "") or "").strip()
        }
        if incoming_ids:
            retained: List[Dict[str, object]] = []
            pruned = 0
            for row in self._validation_changes:
                if (
                    str(row.get("task_type", "")).strip() == "scene_graph"
                    and str(row.get("op", "")).strip() == "cycle_arbitration"
                    and str(row.get("status", "")).strip() == "proposed"
                    and str(row.get("item_id", "")).strip() not in incoming_ids
                ):
                    pruned += 1
                    continue
                retained.append(row)
            if pruned > 0:
                self._validation_changes = retained
                existing_ids = {
                    str(row.get("item_id", "")).strip(): idx
                    for idx, row in enumerate(self._validation_changes)
                    if str(row.get("task_type", "")).strip() == "scene_graph"
                    and str(row.get("op", "")).strip() == "cycle_arbitration"
                }

        added = 0
        updated_count = 0
        for item in queue:
            claim_id = str(item.get("claim_id", "") or "").strip()
            if not claim_id:
                continue
            question = str(item.get("question", "") or "").strip()
            existing_idx = existing_ids.get(claim_id)
            if existing_idx is not None:
                existing = dict(self._validation_changes[existing_idx] or {})
                if str(existing.get("status", "")).strip() == "proposed":
                    existing["after"] = dict(item or {})
                    existing["reason"] = question
                    existing["timestamp"] = now_iso()
                    self._validation_changes[existing_idx] = existing
                    updated_count += 1
                continue
            row = new_change(
                task_type="scene_graph",
                item_id=claim_id,
                op="cycle_arbitration",
                field_path="human_queue",
                before=dict(item.get("claim_row") or claims.get(claim_id) or {}),
                after=dict(item or {}),
                validator_id=validator,
                round_idx=int(self.validation_round_spin.value()),
                reason=question,
            )
            self._validation_changes.append(row)
            existing_ids[claim_id] = len(self._validation_changes) - 1
            added += 1

        if added > 0 or updated_count > 0:
            self._refresh_validation_views()
            self._save_persisted_settings()
        return added + updated_count

    def _inject_cycle_review_correction(self, row: Dict[str, object]) -> Dict[str, object]:
        out = dict(row or {})
        if str(out.get("op", "")).strip() != "cycle_arbitration":
            return out
        selected_id = self._selected_cycle_review_change_id()
        if str(out.get("change_id", "")).strip() != selected_id:
            return out
        claim = dict(out.get("before") or {})
        proposed = self._proposed_value_from_claim_row(claim)
        selected_option = ""
        if hasattr(self, "sg_cycle_review_choice_combo") and self.sg_cycle_review_choice_combo.count() > 0:
            selected_option = str(self.sg_cycle_review_choice_combo.currentData() or "").strip()
        corrected_value = ""
        if hasattr(self, "sg_cycle_review_corrected_value"):
            corrected_value = str(self.sg_cycle_review_corrected_value.text() or "").strip()
        if selected_option and selected_option != proposed:
            corrected_value = selected_option
        after = dict(out.get("after") or {})
        if corrected_value:
            after["corrected_value"] = corrected_value
        else:
            after.pop("corrected_value", None)
        out["after"] = after
        return out

    def _update_correction_memory_from_cycle_change(self, change: Dict[str, object]) -> None:
        row = dict(change or {})
        if str(row.get("op", "")).strip() != "cycle_arbitration":
            return
        claim = dict(row.get("before") or {})
        review = dict(row.get("after") or {})
        status = str(row.get("status", "")).strip().lower()
        claim_type = str(claim.get("claim_type", "") or "").strip().lower()
        subject_id = str(claim.get("subject_id", "") or "").strip()
        proposed = self._proposed_value_from_claim_row(claim)
        corrected = str(review.get("corrected_value", "") or "").strip()
        if not corrected and status == "confirmed":
            corrected = proposed
        frame_idx = self._extract_graph_frame_idx(self.current_graph or {}) if isinstance(self.current_graph, dict) else None
        frame_start = int(frame_idx) if frame_idx is not None and (status == "confirmed" or corrected) else -1
        frame_end = frame_start
        self._correction_memory = update_memory_from_human_decision(
            self._correction_memory,
            claim_type=claim_type,
            proposed=proposed,
            corrected=corrected,
            subject_id=subject_id,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        self._save_correction_memory_state()
        self._refresh_cycle_memory_summary()

    @staticmethod
    def _append_unique_list_flag(item: Dict[str, object], field_name: str, flag: str) -> None:
        flags = item.get(field_name)
        if not isinstance(flags, list):
            flags = []
            item[field_name] = flags
        if flag not in flags:
            flags.append(flag)

    @staticmethod
    def _discard_list_flag(item: Dict[str, object], field_name: str, flag: str) -> None:
        flags = item.get(field_name)
        if not isinstance(flags, list):
            return
        item[field_name] = [x for x in flags if str(x) != flag]

    @staticmethod
    def _append_unique_record(item: Dict[str, object], field_name: str, record: Dict[str, object]) -> None:
        rows = item.get(field_name)
        if not isinstance(rows, list):
            rows = []
            item[field_name] = rows
        normalized = dict(record)
        if normalized not in rows:
            rows.append(normalized)

    @classmethod
    def _append_provenance_record(cls, item: Dict[str, object], record: Dict[str, object]) -> None:
        cls._append_unique_record(item, "provenance", record)

    @staticmethod
    def _find_scene_graph_attribute(node: Dict[str, object], slot: str):
        attrs = node.get("attributes")
        if not isinstance(attrs, list):
            return None
        target = str(slot or "").strip()
        for att in attrs:
            if not isinstance(att, dict):
                continue
            if str(att.get("slot", "") or "").strip() == target:
                return att
        return None

    @staticmethod
    def _resolve_edge_for_claim(
        graph: Dict[str, object],
        claim_row: Dict[str, object],
    ) -> Optional[Dict[str, object]]:
        edges = list(graph.get("edges") or [])
        edge_ids = list(claim_row.get("evidence_edge_ids") or [])
        if edge_ids:
            target_id = str(edge_ids[0] or "").strip()
            for edge in edges:
                if str(edge.get("edge_id", "") or "").strip() == target_id:
                    return edge
        subject_id = str(claim_row.get("subject_id", "") or "").strip()
        object_id = str(claim_row.get("object_id", "") or "").strip()
        predicate = str(claim_row.get("predicate", "") or "").strip()
        for edge in edges:
            if (
                str(edge.get("src_id", edge.get("src", "")) or "").strip() == subject_id
                and str(edge.get("dst_id", edge.get("dst", "")) or "").strip() == object_id
                and str(edge.get("relation", "") or "").strip() == predicate
            ):
                return edge
        return None

    def _rebuild_scene_graph_after_geometry_change(
        self,
        graph: Dict[str, object],
        *,
        reason: str,
        target_node_id: str = "",
    ) -> Dict[str, object]:
        rel_vocab = self._relation_vocab_for_tracking()
        rel_cfg = self._relation_cfg_for_tracking()
        rebuilt = rebuild_spatial_edges(
            graph,
            relation_vocab=rel_vocab,
            touching_iou_epsilon=float(rel_cfg.get("touching_iou_epsilon", 0.02)),
            pairwise_max=int(rel_cfg.get("pairwise_max", 200)),
        )
        metadata = dict(rebuilt.get("metadata") or {})
        history = list(metadata.get("geometry_edit_history") or [])
        history.append(
            {
                "reason": str(reason or "").strip() or "geometry_change",
                "target_node_id": str(target_node_id or "").strip(),
                "frame_idx": self._extract_graph_frame_idx(rebuilt),
                "timestamp": now_iso(),
            }
        )
        metadata["geometry_edit_history"] = history
        rebuilt["metadata"] = metadata
        return rebuilt

    def _apply_cycle_arbitration_change_to_graph(
        self,
        graph: Dict[str, object],
        change: Dict[str, object],
    ) -> Dict[str, object]:
        out = json.loads(json.dumps(graph or {}))
        claim_row = dict(change.get("before") or {})
        review_row = dict(change.get("after") or {})
        status = str(change.get("status", "") or "").strip().lower()
        claim_type = str(claim_row.get("claim_type", "") or "").strip()
        subject_id = str(claim_row.get("subject_id", "") or "").strip()
        predicate = str(claim_row.get("predicate", "") or "").strip()
        value = str(claim_row.get("value", "") or "").strip()
        claim_id = str(change.get("item_id", "") or "").strip()
        corrected_value = str(review_row.get("corrected_value", "") or "").strip()
        geometry_applied = False

        node_by_id = {
            str(node.get("entity_id", "") or "").strip(): node
            for node in out.get("nodes") or []
            if str(node.get("entity_id", "") or "").strip()
        }
        node = node_by_id.get(subject_id)
        edge = self._resolve_edge_for_claim(out, claim_row)
        geometry_options = {
            str(row.get("value", "") or "").strip(): dict(row)
            for row in self._cycle_review_resolution_options(review_row)
            if isinstance(row, dict) and str(row.get("value", "") or "").strip()
        }

        if status == "confirmed":
            if claim_type == "label" and node is not None and (corrected_value or value):
                final_value = corrected_value or value
                node["canonical_label"] = final_value
                node["verified"] = True
                node["score"] = max(float(node.get("score", 0.0) or 0.0), 1.0)
                self._discard_list_flag(node, "validator_flags", "cycle_label_conflict")
            elif claim_type == "attribute" and node is not None and predicate and (corrected_value or value):
                final_value = corrected_value or value
                att = self._find_scene_graph_attribute(node, predicate)
                if att is None:
                    attrs = node.get("attributes")
                    if not isinstance(attrs, list):
                        attrs = []
                        node["attributes"] = attrs
                    att = {"slot": predicate, "value": final_value, "confidence": 1.0, "verified": True, "provenance": []}
                    attrs.append(att)
                att["value"] = final_value
                att["confidence"] = max(float(att.get("confidence", 0.0) or 0.0), 1.0)
                att["verified"] = True
                node["verified"] = True
                self._discard_list_flag(node, "validator_flags", "cycle_attribute_conflict")
            elif claim_type == "relation" and edge is not None:
                if corrected_value:
                    edge["relation"] = corrected_value
                edge["verified"] = True
                edge["score"] = max(float(edge.get("score", 0.0) or 0.0), 1.0)
                self._discard_list_flag(edge, "validator_flags", "cycle_relation_conflict")
            elif claim_type == "existence" and node is not None:
                node["verified"] = True
                self._discard_list_flag(node, "validator_flags", "cycle_existence_conflict")
            elif claim_type == "bbox":
                target_node_id = str(review_row.get("target_node_id", "") or subject_id or "").strip()
                target_node = node_by_id.get(target_node_id)
                selected = geometry_options.get(corrected_value)
                if target_node is not None and selected is not None:
                    target_node["bbox"] = list(selected.get("bbox") or target_node.get("bbox") or [0, 0, 0, 0])
                    if bool(selected.get("clear_mask", True)):
                        target_node["mask"] = {"pixels": []}
                    target_node["verified"] = True
                    target_node["score"] = max(float(target_node.get("score", 0.0) or 0.0), float(selected.get("score", 0.0) or 0.0))
                    self._discard_list_flag(target_node, "validator_flags", "cycle_bbox_conflict")
                    self._discard_list_flag(target_node, "validator_flags", "human_bbox_rejected")
                    geometry_applied = True
                elif target_node is not None:
                    target_node["verified"] = True
                    self._discard_list_flag(target_node, "validator_flags", "cycle_bbox_conflict")
        elif status == "rejected":
            if claim_type == "label" and node is not None and corrected_value:
                node["canonical_label"] = corrected_value
                node["verified"] = True
                node["score"] = max(float(node.get("score", 0.0) or 0.0), 1.0)
                self._discard_list_flag(node, "validator_flags", "cycle_label_conflict")
                self._discard_list_flag(node, "validator_flags", "human_label_rejected")
            elif claim_type == "relation" and edge is not None and corrected_value:
                edge["relation"] = corrected_value
                edge["verified"] = True
                edge["score"] = max(float(edge.get("score", 0.0) or 0.0), 1.0)
                self._discard_list_flag(edge, "validator_flags", "cycle_relation_conflict")
                self._discard_list_flag(edge, "validator_flags", "human_relation_rejected")
            elif claim_type == "bbox":
                target_node_id = str(review_row.get("target_node_id", "") or subject_id or "").strip()
                target_node = node_by_id.get(target_node_id)
                if target_node is not None:
                    target_node["verified"] = False
                    target_node["risk"] = max(float(target_node.get("risk", 0.0) or 0.0), 1.0)
                    self._append_unique_list_flag(target_node, "validator_flags", "human_bbox_rejected")

        if geometry_applied:
            out = self._rebuild_scene_graph_after_geometry_change(
                out,
                reason="cycle_geometry_arbitration",
                target_node_id=str(review_row.get("target_node_id", "") or subject_id or ""),
            )
        metadata = dict(out.get("metadata") or {})
        history = metadata.get("human_arbitration_history")
        if not isinstance(history, list):
            history = []
        record = {
            "claim_id": claim_id,
            "status": status,
            "claim_type": claim_type,
            "subject_id": subject_id,
            "predicate": predicate,
            "value": value,
            "corrected_value": corrected_value,
            "timestamp": str(change.get("decision_timestamp", "") or ""),
        }
        if record not in history:
            history.append(record)
        metadata["human_arbitration_history"] = history
        out["metadata"] = metadata
        return out

    def _apply_change_decision(self, *, task_name: str, change_id: str, approved: bool) -> bool:
        validator = self._require_validator()
        if validator is None:
            return False
        idx = self._find_validation_change_index(change_id)
        if idx < 0:
            return False
        row = self._inject_cycle_review_correction(dict(self._validation_changes[idx] or {}))
        updated = apply_decision(row, approved=approved, decision_by=validator)
        self._validation_changes[idx] = updated
        rerun_started = False
        if (
            task_name == "Video Scene Graph"
            and str(updated.get("op", "")).strip() == "cycle_arbitration"
            and isinstance(self.current_graph, dict)
        ):
            self.current_graph = self._apply_cycle_arbitration_change_to_graph(self.current_graph, updated)
            self._replace_current_graph_in_bundle(self.current_graph)
            self._persist_current_scene_graph_bundle("scene_graph_cycle_arbitration_saved")
            self._update_correction_memory_from_cycle_change(updated)
            claim_row = dict(updated.get("before") or {})
            review_row = dict(updated.get("after") or {})
            claim_type = str(claim_row.get("claim_type", "") or "").strip().lower()
            corrected_value = str(review_row.get("corrected_value", "") or "").strip()
            if claim_type == "bbox" and corrected_value:
                self._drop_pending_cycle_review_rows(keep_change_id=str(updated.get("change_id", "") or ""))
                rerun_started = self._rerun_cycle_refine_after_graph_change(
                    stale_reason="geometry_correction",
                    reason=(
                        "bbox arbitration applied; rerunning cycle verify on the updated graph "
                        f"for node={str(review_row.get('target_node_id', '') or claim_row.get('subject_id', '') or '').strip()}"
                    ),
                )
            else:
                self._sync_current_cycle_result_after_graph_change()
            self._render_graph()
        self._refresh_validation_views()
        self._save_persisted_settings()
        if rerun_started:
            self._set_status(
                "BBox correction applied; rerunning cycle verify on the updated graph.",
                status_type="info",
            )
        else:
            self._set_status(
                f"Change {change_id} {'confirmed' if approved else 'rejected'} in {task_name}",
                status_type="success" if approved else "warning",
            )
        return True

    def _set_cycle_review_decision(self, approved: bool) -> None:
        change_id = self._selected_cycle_review_change_id()
        if not change_id:
            QMessageBox.information(self, "No Selection", "Select a cycle review item first.")
            return
        self._apply_change_decision(
            task_name="Video Scene Graph",
            change_id=change_id,
            approved=approved,
        )

    def _drop_pending_cycle_review_rows(self, *, keep_change_id: str = "") -> None:
        keep = str(keep_change_id or "").strip()
        retained: List[Dict[str, object]] = []
        for row in self._validation_changes:
            if (
                str(row.get("task_type", "")).strip() == "scene_graph"
                and str(row.get("op", "")).strip() == "cycle_arbitration"
                and str(row.get("status", "")).strip() == "proposed"
                and str(row.get("change_id", "")).strip() != keep
            ):
                continue
            retained.append(row)
        self._validation_changes = retained

    @staticmethod
    def _slot_token(slot: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(slot or "").strip().lower()).strip("_")

    def _mark_related_cycle_probes_stale(
        self,
        *,
        claim_ids: Optional[set[str]] = None,
        node_ids: Optional[set[str]] = None,
        edge_ids: Optional[set[str]] = None,
        mark_all: bool = False,
        stale_reason: str = "manual_correction",
    ) -> int:
        if not isinstance(self.current_cycle_result, dict):
            return 0
        rows = [dict(x) for x in list(self.current_cycle_result.get("probe_results") or []) if isinstance(x, dict)]
        if not rows:
            return 0
        claims = {str(x).strip() for x in list(claim_ids or set()) if str(x).strip()}
        nodes = {str(x).strip() for x in list(node_ids or set()) if str(x).strip()}
        edges = {str(x).strip() for x in list(edge_ids or set()) if str(x).strip()}
        changed = 0
        for row in rows:
            match = bool(mark_all)
            row_claim = str(row.get("target_claim_id") or row.get("claim_id") or "").strip()
            row_nodes = {str(x).strip() for x in list(row.get("evidence_node_ids") or []) if str(x).strip()}
            row_edges = {str(x).strip() for x in list(row.get("evidence_edge_ids") or []) if str(x).strip()}
            if row_claim.startswith("claim_rel_"):
                row_edges.add(row_claim[len("claim_rel_") :].strip())
            if (not match) and claims and row_claim in claims:
                match = True
            if (not match) and nodes and bool(row_nodes.intersection(nodes)):
                match = True
            if (not match) and edges and bool(row_edges.intersection(edges)):
                match = True
            if not match:
                continue
            row["stale"] = True
            row["stale_reason"] = str(stale_reason or "manual_correction")
            parsed = dict(row.get("parsed_response") or {}) if isinstance(row.get("parsed_response"), dict) else {}
            parsed["stale"] = True
            parsed["stale_reason"] = str(stale_reason or "manual_correction")
            row["parsed_response"] = parsed
            if isinstance(row.get("response"), dict):
                response = dict(row.get("response") or {})
                response["stale"] = True
                response["stale_reason"] = str(stale_reason or "manual_correction")
                row["response"] = response
            changed += 1
        if changed <= 0:
            return 0
        self.current_cycle_result["probe_results"] = rows
        if isinstance(self.current_graph, dict):
            meta = dict(self.current_graph.get("metadata") or {})
            cycle_verification = dict(meta.get("cycle_verification") or {})
            if cycle_verification:
                cycle_verification["probe_results"] = [dict(x) for x in rows]
                meta["cycle_verification"] = cycle_verification
                self.current_graph["metadata"] = meta
                self._replace_current_graph_in_bundle(self.current_graph)
        self._render_cycle_probe_outputs(list(rows))
        self._refresh_claim_verification_tables()
        return changed

    def _mark_cycle_probes_stale_from_graph_delta(
        self,
        *,
        before_graph: Dict[str, object],
        after_graph: Dict[str, object],
        stale_reason: str = "manual_correction",
    ) -> int:
        scope = self._graph_delta_target_scope(before_graph=before_graph, after_graph=after_graph)
        changed_nodes = set(scope.get("node_ids") or set())
        changed_edges = set(scope.get("edge_ids") or set())
        claim_ids = set(scope.get("claim_ids") or set())
        if (not changed_nodes) and (not changed_edges):
            return 0
        return self._mark_related_cycle_probes_stale(
            claim_ids=claim_ids,
            node_ids=changed_nodes,
            edge_ids=changed_edges,
            stale_reason=stale_reason,
        )

    def _graph_delta_target_scope(
        self,
        *,
        before_graph: Dict[str, object],
        after_graph: Dict[str, object],
    ) -> Dict[str, object]:
        before_nodes = {
            str(node.get("entity_id", "") or "").strip(): dict(node)
            for node in list(before_graph.get("nodes") or [])
            if isinstance(node, dict) and str(node.get("entity_id", "") or "").strip()
        }
        after_nodes = {
            str(node.get("entity_id", "") or "").strip(): dict(node)
            for node in list(after_graph.get("nodes") or [])
            if isinstance(node, dict) and str(node.get("entity_id", "") or "").strip()
        }
        before_edges = {
            str(edge.get("edge_id", "") or "").strip(): dict(edge)
            for edge in list(before_graph.get("edges") or [])
            if isinstance(edge, dict) and str(edge.get("edge_id", "") or "").strip()
        }
        after_edges = {
            str(edge.get("edge_id", "") or "").strip(): dict(edge)
            for edge in list(after_graph.get("edges") or [])
            if isinstance(edge, dict) and str(edge.get("edge_id", "") or "").strip()
        }
        changed_nodes = {
            nid
            for nid in set(before_nodes.keys()).union(set(after_nodes.keys()))
            if before_nodes.get(nid) != after_nodes.get(nid)
        }
        changed_edges = {
            eid
            for eid in set(before_edges.keys()).union(set(after_edges.keys()))
            if before_edges.get(eid) != after_edges.get(eid)
        }
        connected_edge_ids = {
            eid
            for eid, edge in {**before_edges, **after_edges}.items()
            if (
                str((edge or {}).get("src_id", "") or "").strip() in changed_nodes
                or str((edge or {}).get("dst_id", "") or "").strip() in changed_nodes
            )
        }
        changed_edges.update(connected_edge_ids)
        claim_ids: set[str] = set()
        for nid in changed_nodes:
            claim_ids.add(f"claim_exists_{nid}")
            claim_ids.add(f"claim_label_{nid}")
            node = after_nodes.get(nid) or before_nodes.get(nid) or {}
            for att in list(node.get("attributes") or []):
                if not isinstance(att, dict):
                    continue
                slot = self._slot_token(str(att.get("slot", "") or ""))
                if slot:
                    claim_ids.add(f"claim_attr_{nid}_{slot}")
        for eid in changed_edges:
            claim_ids.add(f"claim_rel_{eid}")
        return {
            "claim_ids": claim_ids,
            "node_ids": changed_nodes,
            "edge_ids": changed_edges,
        }

    def _rerun_cycle_refine_after_graph_change(
        self,
        *,
        stale_reason: str = "manual_correction",
        reason: str = "graph_change",
    ) -> bool:
        if not isinstance(self.current_graph, dict):
            return False
        base_result = dict(self.current_cycle_result or {}) if isinstance(self.current_cycle_result, dict) else {}
        target_claim_ids: List[str] = []
        if isinstance(self.current_cycle_result, dict):
            before_graph = dict(self.current_cycle_result.get("graph_after") or {})
            scope = self._graph_delta_target_scope(
                before_graph=before_graph,
                after_graph=dict(self.current_graph or {}),
            )
            target_claim_ids = sorted({str(x).strip() for x in list(scope.get("claim_ids") or set()) if str(x).strip()})
            self._sync_current_cycle_result_after_graph_change(stale_reason=stale_reason)
        self._append_runtime_log(f"[CYCLE-RERUN] {str(reason or 'graph_change').strip()}", level="info")
        self._run_cycle_refine_for_current_graph(
            target_claim_ids=target_claim_ids,
            base_result=base_result,
            run_reason=str(reason or "graph_change").strip(),
        )
        thread = self._cycle_worker_thread
        return bool(thread is not None and thread.isRunning())

    def _sync_current_cycle_result_after_graph_change(self, *, stale_reason: str = "manual_correction") -> None:
        if not isinstance(self.current_cycle_result, dict):
            return
        if not isinstance(self.current_graph, dict):
            return
        before_graph = dict(self.current_cycle_result.get("graph_after") or {})
        self.current_cycle_result["graph_after"] = json.loads(json.dumps(self.current_graph))
        self.current_cycle_result["human_queue"] = [
            dict(row.get("after") or {})
            for row in self._cycle_review_changes()
        ]
        self.current_cycle_result["correction_memory_summary"] = self._cycle_memory_stats()
        summary = dict(self.current_cycle_result.get("summary") or {})
        if summary:
            summary["queue_count"] = len(self.current_cycle_result["human_queue"])
            summary["accepted_claim_count"] = len(list(self._current_cycle_update().get("accepted_claim_ids") or []))
            summary["flagged_claim_count"] = len(list(self._current_cycle_update().get("flagged_claim_ids") or []))
            summary["memory_adjusted_count"] = len(list(self._current_cycle_update().get("memory_adjustments") or []))
            self.current_cycle_result["summary"] = summary
        if before_graph:
            self._mark_cycle_probes_stale_from_graph_delta(
                before_graph=before_graph,
                after_graph=dict(self.current_graph or {}),
                stale_reason=str(stale_reason or "manual_correction"),
            )

    def _build_cycle_result_export_payload(self) -> Dict[str, object]:
        if not isinstance(self.current_cycle_result, dict):
            return {}
        payload = json.loads(json.dumps(self.current_cycle_result))
        if isinstance(self.current_graph, dict):
            payload["graph_after"] = self._build_scene_graph_final_confirmed_payload()
        payload["pending_human_queue"] = [
            dict(row.get("after") or {})
            for row in self._cycle_review_changes()
        ]
        payload["filtered_pending_human_queue"] = [
            dict(row.get("after") or {})
            for row in self._filtered_cycle_review_changes()
        ]
        payload["correction_memory_summary"] = self._cycle_memory_stats()
        payload["correction_memory_path"] = self._cycle_memory_path
        payload["session_export"] = {
            "exported_at": now_iso(),
            "validator_id": str(self.validator_id_input.text() or "").strip(),
            "validation_round": int(self.validation_round_spin.value()),
            "ui_filter": str(self.sg_cycle_review_filter.currentText() if hasattr(self, "sg_cycle_review_filter") else "All Pending"),
            "ui_search": str(self.sg_cycle_review_search.text() if hasattr(self, "sg_cycle_review_search") else ""),
            "pending_queue_count": len(self._cycle_review_changes()),
            "visible_queue_count": len(self._filtered_cycle_review_changes()),
        }
        return payload

    def _export_cycle_session(self) -> None:
        if not isinstance(self.current_cycle_result, dict):
            QMessageBox.information(self, "No Cycle Session", "Run cycle refine before exporting a cycle session.")
            return
        image_id = str((self.current_graph or {}).get("image_id", "") or "scene_graph").strip() or "scene_graph"
        default_name = f"{image_id}_cycle_session_round{int(self.validation_round_spin.value())}.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Cycle Session",
            os.path.join(self._repo_root, default_name),
            "JSON Files (*.json)",
        )
        if not path:
            return
        payload = self._build_cycle_result_export_payload()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._set_status(f"Exported cycle session: {os.path.basename(path)}", status_type="success")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Failed to export cycle session:\n{exc}")

    def _reset_cycle_summary(self) -> None:
        self.current_cycle_result = None
        self._refresh_cycle_summary()

    def _save_graph(self) -> None:
        if not self.current_graph:
            QMessageBox.information(self, "No Graph", "No scene graph to save.")
            self._set_status("No graph to save", status_type="warning")
            return
        path = self._resolve_scene_graph_bundle_output_path()
        try:
            self._export_current_graph_visualization(path)
            bundle_to_write = self._merged_bundle_for_save(path)
            bundle_to_write = self._compact_scene_graph_bundle(bundle_to_write)
            _write_json(path, bundle_to_write)
            self.current_graph_bundle = bundle_to_write
            self._append_oplog("save_scene_graph", path=path, mode="direct_current_folder")
            self._set_status(f"Saved graph directly to current folder: {os.path.basename(path)}", status_type="success")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Failed to save scene graph:\n{exc}")
            self._set_status(f"Save failed: {exc}", status_type="error")

    def _import_scene_graph_annotation(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Scene Graph", self._repo_root, "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                graph = json.load(f)
            if not isinstance(graph, dict):
                raise ValueError("Graph JSON must be an object.")
            self.current_graph = graph
            self.current_graph_bundle = None
            self._reset_scene_graph_tracking()
            self._render_graph()
            self._auto_refresh_score_for_task("Video Scene Graph")
            self._set_status(f"Imported graph: {os.path.basename(path)}", status_type="success")
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", f"Failed to import graph:\n{exc}")

    def _apply_scene_graph_json_edit(self) -> None:
        if not self.current_graph:
            QMessageBox.information(self, "No Graph", "No graph to edit.")
            return
        before = self._json_safe_clone(self.current_graph)
        try:
            edited = json.loads(self.sg_json_preview.toPlainText())
            if not isinstance(edited, dict):
                raise ValueError("Edited graph must be a JSON object")
            self.current_graph = edited
            self.current_graph_bundle = None
            self._reset_scene_graph_tracking()
            self._replace_current_graph_in_bundle(self.current_graph)
            self._persist_current_scene_graph_bundle("scene_graph_json_edit_saved")
            self._render_graph()
            self._record_change(
                task_type="scene_graph",
                item_id=str(edited.get("image_id", "scene_graph")),
                op="update_graph",
                field_path="graph",
                before=before,
                after=edited,
                reason="manual json edit",
            )
            self._auto_refresh_score_for_task("Video Scene Graph")
            self._set_status("Applied scene graph JSON edit", status_type="success")
        except Exception as exc:
            QMessageBox.critical(self, "Invalid JSON", f"Failed to apply JSON edit:\n{exc}")

    def _refresh_single_turn_items_for_current_graph_frame(self, *, silent: bool = False) -> bool:
        if not isinstance(self.current_graph_bundle, dict):
            return False
        graph = self._select_graph_for_current_frame()
        if not isinstance(graph, dict):
            return False
        rows = [dict(x) for x in list(graph.get("single_turn_vqa") or []) if isinstance(x, dict)]
        if not rows:
            return False
        frame_idx = int(self._extract_graph_frame_idx(graph) or 0)
        items: List[Dict[str, object]] = []
        for row in rows:
            item = dict(row)
            item.setdefault("frame_idx", int(frame_idx))
            if self._is_probe_manually_resolved(item):
                continue
            items.append(item)
        self.single_turn_items = items
        self.single_list.clear()
        for item in self.single_turn_items:
            self.single_list.addItem(self._build_probe_list_item(item, is_multi=False))
        if self.single_turn_items:
            self.single_list.setCurrentRow(0)
        if not silent:
            self._set_status(
                f"Loaded single-turn VQA for frame {frame_idx}: {len(self.single_turn_items)} items",
                status_type="success",
            )
        return True

    def _refresh_multi_turn_items_for_current_context(self, *, silent: bool = False) -> bool:
        items: List[Dict[str, object]] = []
        source = ""
        if isinstance(self.current_graph_bundle, dict):
            video_level = [
                dict(x)
                for x in list(self.current_graph_bundle.get("video_level_multi_turn_vqa") or [])
                if isinstance(x, dict)
            ]
            if video_level:
                items = video_level
                source = "video"
        if not items:
            graph = self._select_graph_for_current_frame()
            if isinstance(graph, dict):
                frame_rows = [dict(x) for x in list(graph.get("multi_turn_vqa") or []) if isinstance(x, dict)]
                if frame_rows:
                    frame_idx = int(self._extract_graph_frame_idx(graph) or 0)
                    for row in frame_rows:
                        row.setdefault("frame_idx", int(frame_idx))
                    frame_rows = [dict(row) for row in list(frame_rows or []) if not self._is_probe_manually_resolved(dict(row or {}))]
                    items = frame_rows
                    source = f"frame {frame_idx}"
        if not items:
            return False
        self.multi_turn_items = items
        self.multi_list.clear()
        for item in self.multi_turn_items:
            self.multi_list.addItem(self._build_probe_list_item(item, is_multi=True))
        if self.multi_turn_items:
            self.multi_list.setCurrentRow(0)
        if not silent:
            label = source or "current context"
            self._set_status(
                f"Loaded multi-turn VQA ({label}): {len(self.multi_turn_items)} turns",
                status_type="success",
            )
        return True

    def _generate_single_turn(self) -> None:
        if self._refresh_single_turn_items_for_current_graph_frame(silent=True):
            self._refresh_claim_verification_tables()
            self._auto_refresh_score_for_task("Single-turn VQA")
            frame_idx = int(self._extract_graph_frame_idx(self._select_graph_for_current_frame()) or 0)
            self._set_status(
                f"Generated single-turn VQA for frame {frame_idx}: {len(self.single_turn_items)} items",
                status_type="success",
            )
            return
        if not self.current_graph:
            QMessageBox.information(self, "No Graph", "Build scene graph first.")
            self._set_status("No graph available", status_type="warning")
            return
        items = generate_single_turn_vqa(self.current_graph)
        single_settings = self._task_settings.get("Single-turn VQA", {})
        max_items = int(single_settings.get("max_items", 0))
        if max_items > 0:
            items = items[:max_items]
        self.single_turn_items = items
        self.single_list.clear()
        for item in self.single_turn_items:
            self.single_list.addItem(self._build_probe_list_item(item, is_multi=False))
        if self.single_turn_items:
            self.single_list.setCurrentRow(0)
        self._refresh_claim_verification_tables()
        self._set_status(f"Generated single-turn VQA: {len(self.single_turn_items)} items", status_type="success")
        self._auto_refresh_score_for_task("Single-turn VQA")

    def _render_single_detail(self, row: int) -> None:
        if row < 0 or row >= len(self.single_turn_items):
            self.single_detail.clear()
            return
        item = dict(self.single_turn_items[row] or {})
        question = self._humanize_question_text(str(item.get("question", "") or ""))
        answer, score, reason = self._extract_vqa_answer_fields(item)
        resp = self._probe_response_payload(item)
        invalid_resp = self._probe_is_invalid(item, resp)
        resolved_resp = self._is_probe_manually_resolved(item)
        raw_text = str(resp.get("raw_text", "") or item.get("raw_text", "") or "").strip()
        provider = str(
            item.get("response_provider")
            or (dict(resp.get("raw_response") or {}).get("provider"))
            or resp.get("provider")
            or ""
        ).strip()
        frame_idx = item.get("frame_idx", "")
        detection_conf = self._probe_detection_confidence(item)
        lines = [
            "Target",
            self._probe_target_summary(item),
            "",
            "Target Node IDs",
            ", ".join([str(x or "").strip() for x in list(item.get("evidence_node_ids") or []) if str(x or "").strip()]) or "--",
            "",
            "Target Edge IDs",
            ", ".join([str(x or "").strip() for x in list(item.get("evidence_edge_ids") or []) if str(x or "").strip()]) or "--",
            "",
            "Question",
            question or "--",
            "",
            "Answer",
            ("Resolved" if resolved_resp else ("⚠ Invalid Response" if invalid_resp else str(answer or "uncertain"))),
            "",
            "Verification Score",
            (f"{float(score):.2f}" if ((not invalid_resp) and (not resolved_resp) and float(score) >= 0.0) else "N/A"),
            "",
            "Detection Confidence",
            (f"{float(detection_conf):.2f}" if float(detection_conf) >= 0.0 else "N/A"),
            "",
            "Schema Valid",
            "TRUE" if bool(resp.get("schema_valid", item.get("schema_valid", True))) else "FALSE",
        ]
        if invalid_resp:
            lines.extend(["", "Invalidity", "truncated or schema failure"])
        if resolved_resp:
            lines.extend(["", "Resolution", "Resolved manually"])
        if str(frame_idx).strip() != "":
            lines.extend(["", "Frame", str(frame_idx)])
        if provider:
            lines.extend(["", "Verifier", provider])
        if reason:
            lines.extend(["", "Reason", str(reason)])
        if raw_text:
            lines.extend(["", "[Raw Text ▼]", raw_text])
        raw_resp = resp.get("raw_response")
        if isinstance(raw_resp, dict) and raw_resp:
            lines.extend(["", "[Raw Response ▼]", json.dumps(raw_resp, ensure_ascii=True, indent=2)])
        parsed_resp = self._probe_response_payload(item)
        if parsed_resp:
            lines.extend(["", "[Parsed Response ▼]", json.dumps(parsed_resp, ensure_ascii=True, indent=2)])
        if self._cycle_debug_enabled():
            lines.extend(["", "Probe Family", str(item.get("probe_family", "") or "")])
            req_prompt = str(resp.get("request_prompt", "") or item.get("request_prompt", "") or "").strip()
            req_schema = resp.get("request_schema", item.get("response_schema"))
            if req_prompt:
                lines.extend(["", "Debug Prompt", req_prompt])
            if isinstance(req_schema, dict) and req_schema:
                lines.extend(["", "Debug Schema", json.dumps(req_schema, ensure_ascii=True, indent=2)])
            if isinstance(raw_resp, dict) and raw_resp:
                lines.extend(["", "Debug Raw Response", json.dumps(raw_resp, ensure_ascii=True, indent=2)])
        self.single_detail.setPlainText("\n".join(lines))

    def _save_single_turn(self) -> None:
        if not self.single_turn_items:
            QMessageBox.information(self, "No Data", "Generate single-turn VQA first.")
            self._set_status("No VQA data to save", status_type="warning")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Single-turn VQA", self._repo_root, "JSON Files (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.single_turn_items, f, ensure_ascii=True, indent=2)
        self._append_oplog("save_single_turn_vqa", path=path, items=int(len(self.single_turn_items)))
        self._set_status(f"Saved single-turn VQA: {os.path.basename(path)}", status_type="success")

    def _import_single_turn_annotation(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Single-turn VQA", self._repo_root, "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if not isinstance(rows, list):
                raise ValueError("Single-turn data must be a JSON array")
            self.single_turn_items = rows
            self.single_list.clear()
            for item in self.single_turn_items:
                self.single_list.addItem(self._build_probe_list_item(item, is_multi=False))
            if self.single_turn_items:
                self.single_list.setCurrentRow(0)
            self._set_status(f"Imported single-turn VQA: {len(rows)}", status_type="success")
            self._auto_refresh_score_for_task("Single-turn VQA")
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", f"Failed to import single-turn VQA:\n{exc}")

    def _apply_single_turn_edit(self) -> None:
        row = self.single_list.currentRow()
        if row < 0 or row >= len(self.single_turn_items):
            QMessageBox.information(self, "No Selection", "Select a single-turn item first.")
            return
        before = json.loads(json.dumps(self.single_turn_items[row]))
        try:
            after = json.loads(self.single_detail.toPlainText())
            if not isinstance(after, dict):
                raise ValueError("Edited item must be a JSON object")
            self.single_turn_items[row] = after
            rebuilt = self._build_probe_list_item(after, is_multi=False)
            _old = self.single_list.takeItem(row)
            del _old
            self.single_list.insertItem(row, rebuilt)
            self.single_list.setCurrentRow(row)
            self._record_change(
                task_type="single_turn_vqa",
                item_id=f"single:{row}",
                op="update_item",
                field_path=f"single_turn[{row}]",
                before=before,
                after=after,
                reason="manual detail edit",
            )
            self._set_status("Applied single-turn edit", status_type="success")
            self._auto_refresh_score_for_task("Single-turn VQA")
        except Exception as exc:
            QMessageBox.critical(self, "Invalid JSON", f"Failed to apply single-turn edit:\n{exc}")

    def _generate_multi_turn(self) -> None:
        if self._refresh_multi_turn_items_for_current_context(silent=True):
            self._refresh_claim_verification_tables()
            self._set_status(f"Generated multi-turn VQA: {len(self.multi_turn_items)} turns", status_type="success")
            self._auto_refresh_score_for_task("Multi-turn VQA")
            return
        if not self.current_graph:
            QMessageBox.information(self, "No Graph", "Build scene graph first.")
            self._set_status("No graph available", status_type="warning")
            return
        items = generate_multi_turn_vqa(self.current_graph)
        multi_settings = self._task_settings.get("Multi-turn VQA", {})
        max_items = int(multi_settings.get("max_items", 0))
        if max_items > 0:
            items = items[:max_items]
        self.multi_turn_items = items
        self.multi_list.clear()
        for item in self.multi_turn_items:
            self.multi_list.addItem(self._build_probe_list_item(item, is_multi=True))
        if self.multi_turn_items:
            self.multi_list.setCurrentRow(0)
        self._refresh_claim_verification_tables()
        self._set_status(f"Generated multi-turn VQA: {len(self.multi_turn_items)} turns", status_type="success")
        self._auto_refresh_score_for_task("Multi-turn VQA")

    def _render_multi_detail(self, row: int) -> None:
        if row < 0 or row >= len(self.multi_turn_items):
            self.multi_detail.clear()
            return
        item = dict(self.multi_turn_items[row] or {})
        chain = str(item.get("chain_id", "chain") or "chain")
        turn = int(item.get("turn", 0) or 0)
        question = self._humanize_question_text(str(item.get("question", "") or ""))
        answer, score, reason = self._extract_vqa_answer_fields(item)
        resp = self._probe_response_payload(item)
        invalid_resp = self._probe_is_invalid(item, resp)
        resolved_resp = self._is_probe_manually_resolved(item)
        raw_text = str(resp.get("raw_text", "") or item.get("raw_text", "") or "").strip()
        provider = str(
            item.get("response_provider")
            or (dict(resp.get("raw_response") or {}).get("provider"))
            or resp.get("provider")
            or ""
        ).strip()
        span = list(item.get("summary_span") or [])
        detection_conf = self._probe_detection_confidence(item)
        lines = [
            "Turn",
            f"{chain} | T{turn}",
            "",
            "Target",
            self._probe_target_summary(item),
            "",
            "Target Node IDs",
            ", ".join([str(x or "").strip() for x in list(item.get("evidence_node_ids") or []) if str(x or "").strip()]) or "--",
            "",
            "Target Edge IDs",
            ", ".join([str(x or "").strip() for x in list(item.get("evidence_edge_ids") or []) if str(x or "").strip()]) or "--",
            "",
            "Question",
            question or "--",
            "",
            "Answer",
            ("Resolved" if resolved_resp else ("⚠ Invalid Response" if invalid_resp else str(answer or "uncertain"))),
            "",
            "Verification Score",
            (f"{float(score):.2f}" if ((not invalid_resp) and (not resolved_resp) and float(score) >= 0.0) else "N/A"),
            "",
            "Detection Confidence",
            (f"{float(detection_conf):.2f}" if float(detection_conf) >= 0.0 else "N/A"),
            "",
            "Schema Valid",
            "TRUE" if bool(resp.get("schema_valid", item.get("schema_valid", True))) else "FALSE",
        ]
        if invalid_resp:
            lines.extend(["", "Invalidity", "truncated or schema failure"])
        if resolved_resp:
            lines.extend(["", "Resolution", "Resolved manually"])
        if len(span) >= 2:
            lines.extend(["", "Summary Span", f"frames {int(span[0])}-{int(span[1])}"])
        if provider:
            lines.extend(["", "Verifier", provider])
        if reason:
            lines.extend(["", "Reason", str(reason)])
        if raw_text:
            lines.extend(["", "[Raw Text ▼]", raw_text])
        raw_resp = resp.get("raw_response")
        if isinstance(raw_resp, dict) and raw_resp:
            lines.extend(["", "[Raw Response ▼]", json.dumps(raw_resp, ensure_ascii=True, indent=2)])
        parsed_resp = self._probe_response_payload(item)
        if parsed_resp:
            lines.extend(["", "[Parsed Response ▼]", json.dumps(parsed_resp, ensure_ascii=True, indent=2)])
        if self._cycle_debug_enabled():
            lines.extend(["", "Probe Family", str(item.get("probe_family", "") or "")])
            req_prompt = str(resp.get("request_prompt", "") or item.get("request_prompt", "") or "").strip()
            req_schema = resp.get("request_schema", item.get("response_schema"))
            if req_prompt:
                lines.extend(["", "Debug Prompt", req_prompt])
            if isinstance(req_schema, dict) and req_schema:
                lines.extend(["", "Debug Schema", json.dumps(req_schema, ensure_ascii=True, indent=2)])
            if isinstance(raw_resp, dict) and raw_resp:
                lines.extend(["", "Debug Raw Response", json.dumps(raw_resp, ensure_ascii=True, indent=2)])
        self.multi_detail.setPlainText("\n".join(lines))

    def _save_multi_turn(self) -> None:
        if not self.multi_turn_items:
            QMessageBox.information(self, "No Data", "Generate multi-turn VQA first.")
            self._set_status("No VQA data to save", status_type="warning")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Multi-turn VQA", self._repo_root, "JSON Files (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.multi_turn_items, f, ensure_ascii=True, indent=2)
        self._append_oplog("save_multi_turn_vqa", path=path, items=int(len(self.multi_turn_items)))
        self._set_status(f"Saved multi-turn VQA: {os.path.basename(path)}", status_type="success")

    def _import_multi_turn_annotation(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Multi-turn VQA", self._repo_root, "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            if not isinstance(rows, list):
                raise ValueError("Multi-turn data must be a JSON array")
            self.multi_turn_items = rows
            self.multi_list.clear()
            for item in self.multi_turn_items:
                self.multi_list.addItem(self._build_probe_list_item(item, is_multi=True))
            if self.multi_turn_items:
                self.multi_list.setCurrentRow(0)
            self._set_status(f"Imported multi-turn VQA: {len(rows)}", status_type="success")
            self._auto_refresh_score_for_task("Multi-turn VQA")
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", f"Failed to import multi-turn VQA:\n{exc}")

    def _apply_multi_turn_edit(self) -> None:
        row = self.multi_list.currentRow()
        if row < 0 or row >= len(self.multi_turn_items):
            QMessageBox.information(self, "No Selection", "Select a multi-turn item first.")
            return
        before = json.loads(json.dumps(self.multi_turn_items[row]))
        try:
            after = json.loads(self.multi_detail.toPlainText())
            if not isinstance(after, dict):
                raise ValueError("Edited item must be a JSON object")
            self.multi_turn_items[row] = after
            rebuilt = self._build_probe_list_item(after, is_multi=True)
            _old = self.multi_list.takeItem(row)
            del _old
            self.multi_list.insertItem(row, rebuilt)
            self.multi_list.setCurrentRow(row)
            self._record_change(
                task_type="multi_turn_vqa",
                item_id=f"multi:{row}",
                op="update_item",
                field_path=f"multi_turn[{row}]",
                before=before,
                after=after,
                reason="manual detail edit",
            )
            self._set_status("Applied multi-turn edit", status_type="success")
            self._auto_refresh_score_for_task("Multi-turn VQA")
        except Exception as exc:
            QMessageBox.critical(self, "Invalid JSON", f"Failed to apply multi-turn edit:\n{exc}")

    def _caption_use_current(self) -> None:
        cur = int(self.player.current_frame or 0)
        self.cap_start.setValue(cur)
        self.cap_end.setValue(cur)

    def _build_caption_text(self, start: int, end: int, style: str) -> str:
        graph = self.current_graph or {}
        nodes = list(graph.get("nodes") or [])
        edges = list(graph.get("edges") or [])

        labels = [str(self._node_label(n, "object")) for n in nodes]
        label_counts = Counter(labels)
        top_labels = ", ".join([f"{k}x{v}" for k, v in label_counts.most_common(4)]) if label_counts else "no confirmed entities"
        rel_preview = ", ".join([str(e.get("relation", "rel")) for e in edges[:4]]) if edges else "no explicit relations"

        if style.lower().startswith("concise"):
            return (
                f"Frames {start}-{end}: scene contains {top_labels}; "
                f"dominant relations: {rel_preview}."
            )
        if style.lower().startswith("technical"):
            return (
                f"Temporal slice [{start}, {end}] summary. "
                f"Entity distribution: {top_labels}. "
                f"Detected relation samples: {rel_preview}. "
                f"Graph size: {len(nodes)} nodes, {len(edges)} edges."
            )
        attrs_preview: List[str] = []
        for node in nodes[:6]:
            node_name = str(node.get("display_name", "") or self._node_label(node, "object"))
            attrs = [dict(a) for a in list(node.get("attributes") or []) if isinstance(a, dict)]
            if not attrs:
                continue
            parts: List[str] = []
            for att in attrs[:3]:
                slot = str(att.get("slot", "") or "").strip()
                value = str(att.get("value", "") or "").strip()
                if slot and value:
                    parts.append(f"{slot}={value}")
            if parts:
                attrs_preview.append(f"{node_name}({'; '.join(parts)})")
        return (
            f"From frame {start} to {end}, the scene graph indicates a dense scene with entities {top_labels}. "
            f"Dominant spatial/interaction relations include {rel_preview}. "
            f"Attribute evidence suggests: {'; '.join(attrs_preview[:4]) if attrs_preview else 'limited explicit attributes'}. "
            f"Overall structure contains {len(nodes)} nodes and {len(edges)} edges, so the clip likely includes multiple overlapping interactions rather than a single isolated action."
        )

    def _select_video_keyframe_graphs(
        self,
        bundle: Dict[str, object],
        *,
        stride: int = 5,
        max_keyframes: int = 8,
    ) -> List[Dict[str, object]]:
        graphs = [dict(g) for g in list(bundle.get("graphs") or []) if isinstance(g, dict)]
        if not graphs:
            return []
        graphs.sort(key=lambda g: int(self._extract_graph_frame_idx(g) or 0))
        candidates: List[Dict[str, object]] = []
        for g in graphs:
            summary_text = str(
                g.get("summary")
                or ((g.get("metadata") or {}).get("global_semantic_summary"))
                or ((g.get("metadata") or {}).get("global_summary"))
                or ""
            ).strip()
            if summary_text:
                candidates.append(g)
        if not candidates:
            candidates = graphs
        step = max(1, int(stride))
        picked = [dict(g) for i, g in enumerate(candidates) if i % step == 0]
        if candidates and (not picked or picked[-1] is not candidates[-1]):
            picked.append(dict(candidates[-1]))
        if len(picked) <= max_keyframes:
            return picked
        out: List[Dict[str, object]] = []
        n = len(picked)
        for i in range(max_keyframes):
            idx = int(round(i * (n - 1) / float(max_keyframes - 1)))
            out.append(dict(picked[idx]))
        return out

    def _graph_digest_line(self, graph: Dict[str, object]) -> str:
        nodes = [dict(n) for n in list(graph.get("nodes") or []) if isinstance(n, dict)]
        edges = [dict(e) for e in list(graph.get("edges") or []) if isinstance(e, dict)]
        labels = [str(self._node_label(n, "object")) for n in nodes]
        label_counts = Counter(labels)
        top_labels = ", ".join([f"{k}x{v}" for k, v in label_counts.most_common(3)]) if label_counts else "none"
        rels = Counter([str(e.get("relation", "") or "").strip() for e in edges if str(e.get("relation", "") or "").strip()])
        top_rels = ", ".join([f"{k}x{v}" for k, v in rels.most_common(3)]) if rels else "none"
        return f"entities[{top_labels}] relations[{top_rels}]"

    @staticmethod
    def _graph_image_path(graph: Dict[str, object]) -> str:
        meta = dict(graph.get("metadata") or {})
        image_path = str(meta.get("image_path", "") or "").strip()
        if not image_path:
            image_path = str(graph.get("image_path", "") or "").strip()
        if image_path and os.path.isfile(image_path):
            return image_path
        return ""

    @staticmethod
    def _relation_counterfactual(relation: str) -> str:
        rel = str(relation or "").strip().lower()
        opposite = {
            "left_of": "right_of",
            "right_of": "left_of",
            "above": "below",
            "below": "above",
            "in_front_of": "behind",
            "behind": "in_front_of",
            "inside": "surrounding",
            "surrounding": "inside",
        }
        return str(opposite.get(rel, "") or "")

    @staticmethod
    def _safe_probe_answer(resp: Dict[str, object], *, allow_uncertain: bool = True, fallback_answer: str = "no") -> Dict[str, object]:
        row = dict(resp or {})
        schema_valid = bool(row.get("schema_valid", True))
        is_truncated = bool(row.get("is_truncated", False))
        is_valid_flag = row.get("is_valid", None)
        explicitly_invalid = bool(is_valid_flag is False)
        schema_invalid_and_not_salvaged = (not schema_valid) and (is_valid_flag is not True)
        if schema_invalid_and_not_salvaged or is_truncated or explicitly_invalid:
            return {
                "answer": "uncertain",
                "score": None,
                "reason": "invalid_response",
                "raw_text": str(row.get("raw_text", "") or "").strip(),
                "raw_response": dict(row.get("raw_response") or {}) if isinstance(row.get("raw_response"), dict) else {},
                "finish_reason": str(row.get("finish_reason", "") or "").strip(),
                "schema_errors": list(row.get("schema_errors") or []) if isinstance(row.get("schema_errors"), list) else [],
                "parse_stage": str(row.get("parse_stage", "") or "").strip(),
                "schema_valid": False,
                "is_truncated": bool(is_truncated),
                "is_valid": False,
            }
        answer = str(row.get("answer", "uncertain") or "uncertain").strip().lower()
        if answer not in {"yes", "no", "uncertain"}:
            answer = "uncertain"
        if (not allow_uncertain) and answer == "uncertain":
            answer = str(fallback_answer or "no").strip().lower()
            if answer not in {"yes", "no"}:
                answer = "no"
        try:
            score = float(row.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        if score <= 0.0:
            score = 0.55 if answer in {"yes", "no"} else 0.5
        if (not allow_uncertain) and answer in {"yes", "no"}:
            score = max(0.55, float(score))
        return {
            "answer": answer,
            "score": max(0.0, min(1.0, float(score))),
            "reason": str(row.get("reason", "") or "").strip(),
            "raw_text": str(row.get("raw_text", "") or "").strip(),
            "raw_response": dict(row.get("raw_response") or {}) if isinstance(row.get("raw_response"), dict) else {},
            "finish_reason": str(row.get("finish_reason", "") or "").strip(),
            "schema_errors": list(row.get("schema_errors") or []) if isinstance(row.get("schema_errors"), list) else [],
            "parse_stage": str(row.get("parse_stage", "") or "").strip(),
            "schema_valid": bool(schema_valid),
            "is_truncated": bool(is_truncated),
            "is_valid": bool(is_valid_flag is not False),
        }

    @staticmethod
    def _single_frame_regions(graph: Dict[str, object], node_ids: List[str]) -> List[Dict[str, object]]:
        node_by_id = {
            str(node.get("entity_id", "") or "").strip(): dict(node)
            for node in list(graph.get("nodes") or [])
            if isinstance(node, dict) and str(node.get("entity_id", "") or "").strip()
        }
        out: List[Dict[str, object]] = []
        for node_id in list(node_ids or []):
            node = node_by_id.get(str(node_id or "").strip())
            if not node:
                continue
            out.append(
                {
                    "entity_id": str(node.get("entity_id", "") or ""),
                    "label": str(node.get("canonical_label", "") or ""),
                    "bbox": list(node.get("bbox") or [0, 0, 0, 0]),
                }
            )
        return out

    @staticmethod
    def _frame_node_aliases(graph: Dict[str, object]) -> Dict[str, str]:
        rows = [dict(n) for n in list(graph.get("nodes") or []) if isinstance(n, dict)]
        bucket: Dict[str, int] = {}
        alias: Dict[str, str] = {}
        for node in rows:
            entity_id = str(node.get("entity_id", "") or "").strip()
            label = str(node.get("canonical_label", "object") or "object").strip().lower() or "object"
            if not entity_id:
                continue
            bucket[label] = int(bucket.get(label, 0) or 0) + 1
            alias[entity_id] = f"{label} {bucket[label]}"
        return alias

    def _build_single_frame_questions(self, graph: Dict[str, object]) -> List[Dict[str, object]]:
        nodes = [dict(n) for n in list(graph.get("nodes") or []) if isinstance(n, dict)]
        edges = [dict(e) for e in list(graph.get("edges") or []) if isinstance(e, dict)]
        aliases = self._frame_node_aliases(graph)
        out: List[Dict[str, object]] = []
        if not nodes and not edges:
            return [
                {
                    "question": "Is there enough visual evidence to verify any scene-graph claim in this frame? Answer yes or no.",
                    "expected_answer": "no",
                    "answer_type": "empty_graph_check",
                    "evidence_node_ids": [],
                    "evidence_edge_ids": [],
                }
            ]

        # 1) Node existence checks — persons first, then other objects, sorted by confidence desc.
        #    For nodes with confidence > 0.5 use a direct existence question so Gemini gives
        #    a definitive yes/no rather than hedging on "uncertain".
        def _node_sort_key(n: Dict[str, object]):
            label = str(n.get("canonical_label", "") or "").strip().lower()
            conf = float(n.get("score", n.get("confidence", 0.0)) or 0.0)
            is_person = 1 if label == "person" else 0
            return (-is_person, -conf)

        sorted_nodes = sorted(nodes, key=_node_sort_key)

        for node in sorted_nodes:
            node_id = str(node.get("entity_id", "") or "").strip()
            if not node_id:
                continue
            label = str(node.get("canonical_label", "object") or "object").strip()
            if not label:
                continue
            conf = float(node.get("score", node.get("confidence", 0.0)) or 0.0)
            is_person = label.lower() == "person"
            high_conf = conf > 0.5

            if is_person:
                # Person existence: clearest possible question
                question = (
                    "Look carefully at the entire image. "
                    "Is there a person (human being) visibly present in this frame? "
                    "Answer yes or no. Do not say uncertain."
                )
            elif high_conf:
                # High-confidence non-person object: direct existence question
                question = (
                    f"Look carefully at the image. "
                    f"Is there a '{label}' visibly present in this frame? "
                    "Answer yes or no. Do not say uncertain."
                )
            else:
                # Low-confidence: still ask but via claim framing
                node_name = str(aliases.get(node_id, node_id) or node_id)
                question = (
                    f"Scene graph claims there is a '{label}' in this frame. "
                    "Does the image actually contain this object? Answer yes or no."
                )
            out.append(
                {
                    "question": question,
                    "expected_answer": "yes",
                    "answer_type": "node_claim_check",
                    "claim_node_id": node_id,
                    "evidence_node_ids": [node_id],
                    "evidence_edge_ids": [],
                }
            )

        # 2) Relation checks for one-to-one relation pairs only.
        src_deg: Dict[Tuple[str, str], int] = {}
        dst_deg: Dict[Tuple[str, str], int] = {}
        for edge in edges:
            src = str(edge.get("src_id", "") or "").strip()
            dst = str(edge.get("dst_id", "") or "").strip()
            rel = str(edge.get("relation", "") or "").strip()
            if not src or not dst or not rel:
                continue
            src_deg[(rel, src)] = int(src_deg.get((rel, src), 0) + 1)
            dst_deg[(rel, dst)] = int(dst_deg.get((rel, dst), 0) + 1)
        for edge in edges:
            src = str(edge.get("src_id", "") or "").strip()
            dst = str(edge.get("dst_id", "") or "").strip()
            rel = str(edge.get("relation", "") or "").strip()
            edge_id = str(edge.get("edge_id", "") or "").strip()
            if not src or not dst or not rel:
                continue
            if src_deg.get((rel, src), 0) > 1 or dst_deg.get((rel, dst), 0) > 1:
                continue
            src_name = str(aliases.get(src, src) or src)
            dst_name = str(aliases.get(dst, dst) or dst)
            alt = self._relation_counterfactual(rel)
            if alt:
                question = (
                    f"Scene graph claim says {src_name} is '{rel}' relative to {dst_name}. "
                    f"Counterfactual check: is {src_name} actually '{alt}' relative to {dst_name}? "
                    "Answer yes or no."
                )
                expected_answer = "no"
                prompt_type = "relation_counterfactual"
            else:
                question = (
                    f"Scene graph claim says {src_name} is '{rel}' relative to {dst_name}. "
                    "Does this claim match the image? Answer yes or no."
                )
                expected_answer = "yes"
                prompt_type = "relation_claim_check"
            out.append(
                {
                    "question": question,
                    "expected_answer": expected_answer,
                    "answer_type": prompt_type,
                    "claim_edge_id": edge_id,
                    "claim_relation": str(rel),
                    "evidence_node_ids": [src, dst],
                    "evidence_edge_ids": [edge_id] if edge_id else [],
                }
            )

        return out[:48]

    def _build_video_level_caption_prompt(self, summaries: List[Dict[str, object]]) -> str:
        lines: List[str] = []
        for row in list(summaries or [])[:12]:
            start_frame = int(row.get("start_frame", 0) or 0)
            end_frame = int(row.get("end_frame", start_frame) or start_frame)
            text = str(row.get("summary", "") or "").strip()
            if not text:
                continue
            lines.append(f"- frames {start_frame}-{end_frame}: {text}")
        if not lines:
            return (
                "You are a video captioning verifier. "
                "Return one detailed multi-sentence paragraph (6-10 sentences) describing the whole video based on verified scene-graph evidence only."
            )
        return (
            "You are a video captioning verifier.\n"
            "Given the temporally ordered batch summaries below, write one detailed coherent video-level caption.\n"
            "Requirements:\n"
            "1) Keep factual and avoid hallucination.\n"
            "2) Mention major temporal evolution and transitions across the whole video.\n"
            "3) Include objects, actions, relations, and any stable attributes when available.\n"
            "4) Keep to 6-10 sentences with enough detail for downstream annotation.\n"
            "5) Do NOT enumerate every frame; summarize globally while keeping temporal logic.\n\n"
            "Batch summaries:\n"
            + "\n".join(lines)
        )

    # ── Gemini object-count / auto-detect helpers ────────────────────────────

    @staticmethod
    def _count_graph_nodes_by_label(graph: Dict[str, object], label: str) -> int:
        """Return how many nodes in *graph* have canonical_label == *label*."""
        return sum(
            1 for n in (graph.get("nodes") or [])
            if isinstance(n, dict) and str(n.get("canonical_label", "") or "").strip().lower() == label.lower()
        )

    @staticmethod
    def _gemini_ask_count(verifier, image_path: str, label: str) -> int:
        """Ask Gemini how many *label* objects are visible. Returns -1 on parse failure."""
        import re as _re
        prompt = (
            f"Count the number of '{label}' objects (human beings if label is 'person') "
            f"that are CLEARLY visible in this image. "
            f"Reply with ONLY a single integer (0, 1, 2, 3 …). No other text."
        )
        try:
            resp = verifier.answer_probe(image_path=image_path, question=prompt, regions=[], response_format=None)
            raw = str(resp.get("raw_text", "") or resp.get("answer", "") or "").strip()
            m = _re.search(r"\b(\d+)\b", raw)
            return int(m.group(1)) if m else -1
        except Exception:
            return -1

    def _gemini_detect_and_add_objects(
        self,
        verifier,
        image_path: str,
        label: str,
        expected_count: int,
        graph: Dict[str, object],
        frame_idx: int,
    ) -> int:
        """Ask Gemini to locate *label* objects and append missing nodes to *graph*.

        Returns the number of new nodes added.
        """
        import re as _re
        detect_prompt = (
            f"There should be {expected_count} '{label}' object(s) visible in this image "
            f"but the scene graph is incomplete. "
            f"Detect ALL visible '{label}' objects and return their bounding boxes.\n"
            f"Return JSON ONLY in this exact format:\n"
            f'  {{"objects": [{{"label": "{label}", "bbox_xywh": [x, y, width, height]}}]}}\n'
            f"where (x, y) is the top-left corner in pixel coordinates and width/height are in pixels.\n"
            f"If none are visible, return {{\"objects\": []}}."
        )
        try:
            cap = verifier.generate_caption(image_path=image_path, prompt=detect_prompt, regions=[])
            raw_text = str(cap.get("raw_text", cap.get("caption", "")) or "").strip()
            # Try to extract JSON from the response
            parsed = None
            try:
                import json as _json
                # Try to find { ... } block
                m = _re.search(r"\{.*\}", raw_text, _re.DOTALL)
                if m:
                    parsed = _json.loads(m.group(0))
            except Exception:
                pass
            if not isinstance(parsed, dict):
                self._append_runtime_log(
                    f"[VIDEO-DETECT] Could not parse bbox JSON for label={label!r}: {raw_text[:200]!r}",
                    level="warning",
                )
                return 0

            objects = list(parsed.get("objects") or [])
            nodes = list(graph.get("nodes") or [])
            added = 0
            for idx, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    continue
                bbox_raw = list(obj.get("bbox_xywh") or obj.get("bbox") or [])
                if len(bbox_raw) < 4:
                    continue
                try:
                    bbox = [int(float(v)) for v in bbox_raw[:4]]
                except Exception:
                    continue
                eid = f"gemini_auto_{label}_{frame_idx}_{idx}"
                new_node: Dict[str, object] = {
                    "entity_id": eid,
                    "canonical_label": label,
                    "label": label,
                    "bbox": bbox,
                    "score": 0.85,
                    "confidence": 0.85,
                    "validator_flags": ["gemini_auto_detected"],
                    "attributes": [],
                }
                nodes.append(new_node)
                added += 1
                self._append_runtime_log(
                    f"[VIDEO-DETECT] Auto-added node eid={eid} label={label!r} bbox={bbox} frame={frame_idx}",
                    level="info",
                )
            graph["nodes"] = nodes
            return added
        except Exception as exc:
            self._append_runtime_log(
                f"[VIDEO-DETECT] Detection call failed for label={label!r}: {str(exc)[:200]}",
                level="warning",
            )
            return 0

    def _build_video_gemini_verifier(self):
        sg_settings = dict(self._task_settings.get("Video Scene Graph", {}) or {})
        key_env = str(sg_settings.get("cycle_gemini_api_key_env", DEFAULT_GEMINI_API_KEY_ENV) or DEFAULT_GEMINI_API_KEY_ENV).strip()
        if not str(os.environ.get(key_env, "") or "").strip():
            saved_common_key = str((self._common_settings or {}).get("api_key", "") or "").strip()
            if saved_common_key:
                os.environ[key_env] = saved_common_key
                os.environ.setdefault(DEFAULT_GEMINI_API_KEY_ENV, saved_common_key)
                os.environ.setdefault("IMPACT_API_KEY", saved_common_key)

        cfg_override = dict(self._scene_graph_cycle_cfg_override() or {})
        api_cfg = dict(cfg_override.get("api_verifier") or {})
        api_cfg["enabled"] = True
        api_cfg["provider"] = "gemini"
        if not str(api_cfg.get("model", "") or "").strip():
            api_cfg["model"] = str(sg_settings.get("cycle_gemini_model", DEFAULT_GEMINI_MODEL) or DEFAULT_GEMINI_MODEL).strip()
        cfg_override["api_verifier"] = api_cfg
        cfg_override["runtime"] = {
            "allow_mock_fallback": False,
            "preferred_provider": DEFAULT_CYCLE_PROVIDER,
            "experimental_providers": [],
        }
        verifier, runtime_meta = build_vision_verifier(
            cfg_override,
            preferred_provider=DEFAULT_CYCLE_PROVIDER,
            api_key="",
            progress_cb=lambda msg: self._append_runtime_log(str(msg or ""), level="info"),
            allow_mock_fallback=False,
        )
        return verifier, dict(runtime_meta or {})

    def _augment_bundle_video_vqa_outputs(self, bundle: Dict[str, object]) -> Dict[str, object]:
        out = dict(bundle or {})
        graphs = [dict(g) for g in list(out.get("graphs") or []) if isinstance(g, dict)]
        if not graphs:
            return out
        graphs.sort(key=lambda g: int(self._extract_graph_frame_idx(g) or 0))
        verifier = None
        verifier_meta: Dict[str, object] = {}
        verifier_error = ""
        try:
            verifier, verifier_meta = self._build_video_gemini_verifier()
            _vkey_env = str(
                (dict(self._task_settings.get("Video Scene Graph", {}) or {})).get(
                    "cycle_gemini_api_key_env",
                    DEFAULT_GEMINI_API_KEY_ENV,
                )
                or DEFAULT_GEMINI_API_KEY_ENV
            ).strip()
            _vkey_set = bool(str(os.environ.get(_vkey_env, "") or "").strip())
            self._append_runtime_log(
                f"[VIDEO-VERIFY] Gemini API verifier ready: model={str(verifier_meta.get('verifier_model_id', '') or '').strip()} "
                f"api_key_env={_vkey_env} key_set={_vkey_set}",
                level="info",
            )
        except Exception as exc:
            verifier_error = str(exc or "").strip() or "Gemini API verifier unavailable."
            self._append_runtime_log(f"[VIDEO-VERIFY] Gemini API verifier unavailable: {verifier_error}", level="warning")

        frame_to_graph: Dict[int, Dict[str, object]] = {
            int(self._extract_graph_frame_idx(g) or -1): dict(g)
            for g in graphs
            if int(self._extract_graph_frame_idx(g) or -1) >= 0
        }

        for g in graphs:
            frame_idx = int(self._extract_graph_frame_idx(g) or 0)
            image_path = self._graph_image_path(g)
            if not image_path and self.video_path and frame_idx >= 0:
                try:
                    image_path, _w, _h, _frame = _extract_video_frame_to_cache(
                        self.video_path, frame_idx, self._frame_cache_dir
                    )
                except Exception:
                    image_path = ""

            # ── Step 0: count-check for persons (and any dominant label) ──────
            # Ask Gemini "how many persons?". If it finds more than the graph contains,
            # auto-detect and add their bboxes before generating VQA questions.
            if verifier is not None and image_path:
                # Always check for persons; also check the most common non-person label
                labels_to_check = ["person"]
                node_labels = [
                    str(n.get("canonical_label", "") or "").strip().lower()
                    for n in (g.get("nodes") or [])
                    if isinstance(n, dict)
                ]
                # Add the most common non-person label (if any) with >1 occurrence
                from collections import Counter as _Counter
                label_freq = _Counter(lb for lb in node_labels if lb and lb != "person")
                if label_freq:
                    top_label, top_count = label_freq.most_common(1)[0]
                    if top_count >= 1 and top_label not in labels_to_check:
                        labels_to_check.append(top_label)

                for chk_label in labels_to_check:
                    graph_count = self._count_graph_nodes_by_label(g, chk_label)
                    gemini_count = self._gemini_ask_count(verifier, image_path, chk_label)
                    self._append_runtime_log(
                        f"[VIDEO-COUNT] frame={frame_idx} label={chk_label!r} "
                        f"graph_count={graph_count} gemini_count={gemini_count}",
                        level="info",
                    )
                    if gemini_count > 0 and gemini_count > graph_count:
                        self._append_runtime_log(
                            f"[VIDEO-COUNT] Mismatch detected for {chk_label!r}: "
                            f"Gemini API says {gemini_count}, graph has {graph_count}. "
                            f"Triggering auto-detection.",
                            level="info",
                        )
                        self._gemini_detect_and_add_objects(
                            verifier, image_path, chk_label, gemini_count, g, frame_idx
                        )

            single_questions = [dict(x) for x in self._build_single_frame_questions(g)]
            single_rows: List[Dict[str, object]] = []
            for single_q in single_questions:
                expected_answer = str(single_q.get("expected_answer", "yes") or "yes").strip().lower()
                response = {
                    "answer": expected_answer,
                    "score": 0.55,
                    "reason": "",
                    "raw_text": "",
                    "schema_valid": True,
                    "is_truncated": False,
                    "is_valid": True,
                }
                if verifier is not None and image_path:
                    try:
                        raw_resp = verifier.answer_probe(
                            image_path=image_path,
                            question=str(single_q.get("question", "") or "").strip(),
                            regions=self._single_frame_regions(g, [str(x) for x in list(single_q.get("evidence_node_ids") or [])]),
                            response_format=None,
                        )
                        response = self._safe_probe_answer(
                            dict(raw_resp or {}),
                            allow_uncertain=False,
                            fallback_answer=expected_answer if expected_answer in {"yes", "no"} else "no",
                        )
                    except Exception as exc:
                        response = {
                            "answer": expected_answer if expected_answer in {"yes", "no"} else "no",
                            "score": None,
                            "reason": f"gemini_error:{str(exc)[:160]}",
                            "raw_text": "",
                            "raw_response": {},
                            "schema_valid": False,
                            "is_truncated": False,
                            "is_valid": False,
                        }
                elif verifier is None:
                    response["reason"] = f"gemini_unavailable:{verifier_error[:160]}"
                    response["schema_valid"] = False
                    response["is_valid"] = False
                else:
                    response["reason"] = "frame_image_missing"
                    response["schema_valid"] = False
                    response["is_valid"] = False
                single_score = None
                try:
                    raw_single_score = response.get("score", None)
                    if raw_single_score is not None:
                        single_score = float(raw_single_score)
                except Exception:
                    single_score = None
                single_rows.append(
                    {
                        "qid": f"video_sq_{frame_idx}_{uuid.uuid4().hex[:8]}",
                        "question": str(single_q.get("question", "") or "").strip(),
                        "answer": str(response.get("answer", expected_answer) or expected_answer),
                        "score": single_score,
                        "reason": str(response.get("reason", "") or ""),
                        "raw_text": str(response.get("raw_text", "") or ""),
                        "raw_response": dict(response.get("raw_response") or {}) if isinstance(response.get("raw_response"), dict) else {},
                        "schema_valid": bool(response.get("schema_valid", True)),
                        "is_truncated": bool(response.get("is_truncated", False)),
                        "is_valid": bool(response.get("is_valid", True)),
                        "parsed_response": dict(response),
                        "answer_type": str(single_q.get("answer_type", "frame_claim_check") or "frame_claim_check"),
                        "expected_answer": expected_answer,
                        "frame_idx": int(frame_idx),
                        "graph_snapshot_id": str(g.get("image_id", "") or ""),
                        "evidence_node_ids": [str(x) for x in list(single_q.get("evidence_node_ids") or []) if str(x)],
                        "evidence_edge_ids": [str(x) for x in list(single_q.get("evidence_edge_ids") or []) if str(x)],
                        "verified_by": str(verifier_meta.get("verifier_provider", "gemini") or "gemini"),
                        "validator_flags": [],
                    }
                )
            g["single_turn_vqa"] = single_rows
        out["graphs"] = graphs

        summary_rows = [dict(x) for x in list(out.get("llm_batch_summaries") or []) if isinstance(x, dict)]
        summary_rows.sort(key=lambda r: int(r.get("start_frame", 0) or 0))
        if not summary_rows:
            key_graphs = self._select_video_keyframe_graphs(out, stride=5, max_keyframes=8)
            for g in key_graphs:
                frame_idx = int(self._extract_graph_frame_idx(g) or 0)
                summary_text = str(
                    g.get("summary")
                    or ((g.get("metadata") or {}).get("global_semantic_summary"))
                    or ((g.get("metadata") or {}).get("global_summary"))
                    or self._graph_digest_line(g)
                ).strip()
                summary_rows.append(
                    {
                        "start_frame": int(frame_idx),
                        "end_frame": int(frame_idx),
                        "summary": summary_text,
                    }
                )

        multi_items: List[Dict[str, object]] = []
        chain_id = f"video_chain_{uuid.uuid4().hex[:8]}"
        prev_summary = ""
        for turn, row in enumerate(summary_rows, start=1):
            start_frame = int(row.get("start_frame", 0) or 0)
            end_frame = int(row.get("end_frame", start_frame) or start_frame)
            summary_text = str(row.get("summary", "") or "").strip()
            if not summary_text:
                continue
            if turn == 1:
                question = (
                    f"Summary for frames {start_frame}-{end_frame}: {summary_text}\n"
                    "Is this summary well grounded in what is visibly shown in the image? "
                    "Answer yes or no. Focus especially on whether mentioned objects and persons are actually present."
                )
            else:
                question = (
                    f"Previous batch summary: {prev_summary}\n"
                    f"Current batch summary (frames {start_frame}-{end_frame}): {summary_text}\n"
                    "Do these summaries stay temporally consistent without obvious contradiction? "
                    "Answer yes or no."
                )
            response = {
                "answer": "uncertain",
                "score": None,
                "reason": "",
                "raw_text": "",
                "schema_valid": True,
                "is_truncated": False,
                "is_valid": True,
            }
            evidence_graph = frame_to_graph.get(start_frame) or frame_to_graph.get(end_frame) or {}
            image_path = self._graph_image_path(evidence_graph) if isinstance(evidence_graph, dict) else ""
            if not image_path and self.video_path and start_frame >= 0:
                try:
                    image_path, _w, _h, _frame = _extract_video_frame_to_cache(
                        self.video_path, start_frame, self._frame_cache_dir
                    )
                except Exception:
                    image_path = ""
            if verifier is not None and image_path:
                try:
                    raw_resp = verifier.answer_probe(
                        image_path=image_path,
                        question=question,
                        regions=[],
                        response_format=None,
                    )
                    response = self._safe_probe_answer(dict(raw_resp or {}), allow_uncertain=False, fallback_answer="yes")
                except Exception as exc:
                    response = {
                        "answer": "uncertain",
                        "score": None,
                        "reason": f"gemini_error:{str(exc)[:160]}",
                        "raw_text": "",
                        "raw_response": {},
                        "schema_valid": False,
                        "is_truncated": False,
                        "is_valid": False,
                    }
            elif verifier is None:
                response["reason"] = f"gemini_unavailable:{verifier_error[:160]}"
                response["schema_valid"] = False
                response["is_valid"] = False
            else:
                response["reason"] = "frame_image_missing"
                response["schema_valid"] = False
                response["is_valid"] = False
            multi_score = None
            try:
                raw_multi_score = response.get("score", None)
                if raw_multi_score is not None:
                    multi_score = float(raw_multi_score)
            except Exception:
                multi_score = None
            multi_items.append(
                {
                    "qid": f"video_mq_{turn}_{uuid.uuid4().hex[:8]}",
                    "chain_id": chain_id,
                    "turn": int(turn),
                    "question": question,
                    "answer": str(response.get("answer", "uncertain") or "uncertain"),
                    "score": multi_score,
                    "reason": str(response.get("reason", "") or ""),
                    "raw_text": str(response.get("raw_text", "") or ""),
                    "raw_response": dict(response.get("raw_response") or {}) if isinstance(response.get("raw_response"), dict) else {},
                    "schema_valid": bool(response.get("schema_valid", True)),
                    "is_truncated": bool(response.get("is_truncated", False)),
                    "is_valid": bool(response.get("is_valid", True)),
                    "parsed_response": dict(response),
                    "answer_type": "summary_temporal_consistency",
                    "summary_span": [int(start_frame), int(end_frame)],
                    "evidence_frame_indices": [int(start_frame), int(end_frame)],
                    "graph_snapshot_id": str((evidence_graph or {}).get("image_id", "") or ""),
                    "verified_by": str(verifier_meta.get("verifier_provider", "gemini") or "gemini"),
                    "verified": False,
                    "validator_flags": [],
                }
            )
            prev_summary = summary_text

        caption_text = ""
        caption_prompt = self._build_video_level_caption_prompt(summary_rows)
        caption_image = ""
        if summary_rows:
            mid_idx = int(len(summary_rows) / 2)
            s_frame = int(summary_rows[mid_idx].get("start_frame", 0) or 0)
            caption_image = self._graph_image_path(frame_to_graph.get(s_frame) or {})
        if verifier is not None and caption_image:
            try:
                cap_payload = dict(
                    verifier.generate_caption(
                        image_path=caption_image,
                        prompt=caption_prompt,
                        regions=[],
                    )
                    or {}
                )
                caption_text = str(cap_payload.get("caption", "") or "").strip()
                if not caption_text:
                    caption_text = str(cap_payload.get("raw_text", "") or "").strip()
            except Exception as exc:
                self._append_runtime_log(f"[VIDEO-CAPTION] Gemini API caption failed: {str(exc)[:200]}", level="warning")
        if not caption_text:
            caption_text = " ".join(
                [
                    str(row.get("summary", "") or "").strip()
                    for row in summary_rows[:12]
                    if str(row.get("summary", "") or "").strip()
                ]
            ).strip()

        out["video_level_keyframes"] = [int(row.get("start_frame", 0) or 0) for row in summary_rows]
        out["video_level_multi_turn_vqa"] = multi_items
        out["video_level_caption"] = caption_text
        out["video_level_verification"] = {
            "provider": str(verifier_meta.get("verifier_provider", "gemini") or "gemini"),
            "model_id": str(verifier_meta.get("verifier_model_id", "") or ""),
            "single_frame_count": int(len(graphs)),
            "summary_turn_count": int(len(multi_items)),
            "gemini_ready": bool(verifier is not None),
            "gemini_error": verifier_error,
        }
        return out

    def _generate_caption(self) -> None:
        # Temporary product mode: only one caption for the current video, no segment slicing.
        style = str(self.cap_style.currentText())
        text = ""
        if isinstance(self.current_graph_bundle, dict):
            text = str(self.current_graph_bundle.get("video_level_caption", "") or "").strip()
        if not text:
            # Fallback to current-frame graph digest when video-level caption is unavailable.
            cur = int(self.player.current_frame or 0)
            text = self._build_caption_text(cur, cur, style)
        self.caption_output.setPlainText(text)
        self._refresh_claim_verification_tables()
        self._set_status("Caption generated.", status_type="success")
        self._score_results["Video Captioning"] = {
            "task_type": "Video Captioning",
            "metrics": self._ui_feature_service.summarize_changes(self._changes_for_task("Video Captioning")),
            "score_summary": {"overall_score": 0.0, "cards": []},
        }
        self._render_score_result("Video Captioning")

    def _caption_add_segment(self) -> None:
        start = int(self.cap_start.value())
        end = int(self.cap_end.value())
        if end < start:
            start, end = end, start
        style = str(self.cap_style.currentText())
        new_item = {"start": start, "end": end, "style": style, "caption": ""}
        self.caption_batch.append(new_item)
        self._render_caption_batch()
        self._record_change(
            task_type="video_captioning",
            item_id=f"segment:{len(self.caption_batch)-1}",
            op="add_segment",
            field_path="caption_batch[]",
            before=None,
            after=new_item,
            reason="segment added",
        )
        self._set_status(f"Added caption segment [{start}, {end}]", status_type="success")
        self._render_score_result("Video Captioning")

    def _caption_remove_selected(self) -> None:
        row = self._selected_row(self.caption_batch_table)
        if row < 0 or row >= len(self.caption_batch):
            return
        removed = self.caption_batch.pop(row)
        self._render_caption_batch()
        self._record_change(
            task_type="video_captioning",
            item_id=f"segment:{row}",
            op="remove_segment",
            field_path=f"caption_batch[{row}]",
            before=removed,
            after=None,
            reason="segment removed",
        )
        self._render_score_result("Video Captioning")

    def _caption_generate_batch(self) -> None:
        if not self.caption_batch:
            QMessageBox.information(self, "No Segments", "Add caption segments first.")
            self._set_status("No caption segments added", status_type="warning")
            return
        for item in self.caption_batch:
            s = int(item.get("start", 0))
            e = int(item.get("end", 0))
            style = str(item.get("style", "Concise"))
            item["caption"] = self._build_caption_text(s, e, style)
        self._render_caption_batch()
        if self.caption_batch:
            self.caption_output.setPlainText(str(self.caption_batch[0].get("caption", "")))
        self._set_status(f"Generated captions for {len(self.caption_batch)} segments", status_type="success")
        self._render_score_result("Video Captioning")

    def _render_caption_batch(self) -> None:
        self.caption_batch_table.setRowCount(len(self.caption_batch))
        for i, item in enumerate(self.caption_batch):
            self.caption_batch_table.setItem(i, 0, QTableWidgetItem(str(item.get("start", 0))))
            self.caption_batch_table.setItem(i, 1, QTableWidgetItem(str(item.get("end", 0))))
            self.caption_batch_table.setItem(i, 2, QTableWidgetItem(str(item.get("style", ""))))
            self.caption_batch_table.setItem(i, 3, QTableWidgetItem(str(item.get("caption", ""))))

    def _caption_batch_selection_changed(self) -> None:
        row = self._selected_row(self.caption_batch_table)
        if row < 0 or row >= len(self.caption_batch):
            return
        self.caption_output.setPlainText(str(self.caption_batch[row].get("caption", "")))

    def _caption_export_batch_jsonl(self) -> None:
        if not self.caption_batch:
            QMessageBox.information(self, "No Data", "No caption batch to export.")
            self._set_status("No caption batch to export", status_type="warning")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Caption Batch JSONL", self._repo_root, "JSONL Files (*.jsonl)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            for item in self.caption_batch:
                f.write(json.dumps(item, ensure_ascii=True) + "\n")
        self._append_oplog("save_caption_batch_jsonl", path=path, items=int(len(self.caption_batch)))
        self._set_status(f"Saved caption batch JSONL: {os.path.basename(path)}", status_type="success")

    def _caption_export_batch_txt(self) -> None:
        if not self.caption_batch:
            QMessageBox.information(self, "No Data", "No caption batch to export.")
            self._set_status("No caption batch to export", status_type="warning")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Caption Batch TXT", self._repo_root, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            for idx, item in enumerate(self.caption_batch, start=1):
                s = int(item.get("start", 0))
                e = int(item.get("end", 0))
                style = str(item.get("style", ""))
                caption = str(item.get("caption", ""))
                f.write(f"[{idx}] frames {s}-{e} ({style})\n")
                f.write(caption + "\n\n")
        self._append_oplog("save_caption_batch_txt", path=path, items=int(len(self.caption_batch)))
        self._set_status(f"Saved caption batch TXT: {os.path.basename(path)}", status_type="success")

    def _save_caption(self) -> None:
        text = self.caption_output.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "No Caption", "Generate caption first.")
            self._set_status("No caption to save", status_type="warning")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Caption", self._repo_root, "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self._append_oplog("save_caption", path=path)
        self._set_status(f"Saved caption: {os.path.basename(path)}", status_type="success")

    def _import_caption_annotation(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Caption", self._repo_root, "JSON Files (*.json);;Text Files (*.txt)")
        if not path:
            return
        try:
            if path.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    self.caption_output.setPlainText(f.read())
            else:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    self.caption_batch = payload
                    self._render_caption_batch()
                    if self.caption_batch:
                        self.caption_output.setPlainText(str(self.caption_batch[0].get("caption", "")))
                elif isinstance(payload, dict):
                    self.caption_output.setPlainText(str(payload.get("caption", "")))
                else:
                    raise ValueError("Caption file must be JSON object/array or text")
            self._set_status(f"Imported caption: {os.path.basename(path)}", status_type="success")
            self._render_score_result("Video Captioning")
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", f"Failed to import caption annotation:\n{exc}")

    def _apply_caption_edit(self) -> None:
        text = self.caption_output.toPlainText()
        row = self.caption_batch_table.currentRow()
        if row >= 0 and row < len(self.caption_batch):
            before = str(self.caption_batch[row].get("caption", ""))
            self.caption_batch[row]["caption"] = text
            self._render_caption_batch()
            self.caption_batch_table.selectRow(row)
            self._record_change(
                task_type="video_captioning",
                item_id=f"segment:{row}",
                op="update_caption",
                field_path=f"caption_batch[{row}].caption",
                before=before,
                after=text,
                reason="manual caption edit",
            )
        else:
            before = ""
            self._record_change(
                task_type="video_captioning",
                item_id="caption_output",
                op="update_caption",
                field_path="caption_output",
                before=before,
                after=text,
                reason="manual caption edit",
            )
        self._set_status("Applied caption edit", status_type="success")
        self._render_score_result("Video Captioning")

    def _open_task_settings_dialog(self) -> None:
        task_name = str(self.task_combo.currentText()).strip()
        task_settings = self._task_settings.get(task_name, {})
        dialog = TaskSettingsDialog(
            task_name=task_name,
            ontology_path=self._ontology_path,
            ontology_changed_cb=self._on_ontology_changed,
            common_settings=self._common_settings,
            default_common_settings=self._common_settings_defaults,
            task_settings=task_settings,
            default_task_settings=self._task_settings_defaults.get(task_name, {}),
            ontology_status_text=self._ontology_status_text,
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted:
            self._common_settings = dialog.get_common_settings()
            self._task_settings[task_name] = dialog.get_task_settings()
            self._apply_common_settings_runtime()
            if task_name == "Video Captioning":
                style = str(self._task_settings[task_name].get("default_style", "Concise"))
                idx = self.cap_style.findText(style)
                self.cap_style.setCurrentIndex(max(0, idx))
            self._save_persisted_settings()
            self._set_status(f"Settings applied for: {task_name}", status_type="success")

    def _on_ontology_changed(self, ontology_dict: Dict[str, object]) -> None:
        """Handle ontology changes from editor."""
        self._custom_ontology = ontology_dict
        entities = len(ontology_dict.get("canonical_entities", []))
        relations = sum(
            len(v) for v in ontology_dict.get("relation_vocabulary", {}).values()
        )
        self._ontology_status_text = f"Custom ontology loaded: {entities} entities, {relations} relations"
        self._save_persisted_settings()
        self._set_status(
            f"Ontology updated: {entities} entities, {relations} relations",
            status_type="success",
        )
