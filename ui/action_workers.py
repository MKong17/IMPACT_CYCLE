import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from bisect import bisect_left, bisect_right
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from core.east_online_update import rebuild_runtime_model_delta
from core.east_online_adapter_train import rebuild_runtime_online_adapter
from utils.feature_env import build_runner_cmd, load_feature_env_defaults
from utils.optional_deps import (
    MissingOptionalDependency,
    format_missing_dependency_message,
    import_optional_module,
)


ACTION_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_feature_env_defaults(repo_root=ACTION_REPO_ROOT)


def load_feature_extractor_module():
    return import_optional_module(
        "tools.feature_extractors",
        feature_name="Feature extraction / ASOT pre-labeling",
        install_hint=(
            "Install the optional feature-extraction dependencies first, "
            "for example: pip install torch torchvision"
        ),
    )


def _run_streaming_command(cmd, progress_cb):
    log_lines = []
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line.rstrip()
                log_lines.append(line + "\n")
                progress_cb(line)
    finally:
        proc.wait()
    return proc.returncode, log_lines


def _write_worker_log(log_path: str, log_lines, progress_cb) -> None:
    if not log_path:
        return
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(log_lines)
        progress_cb(f"[LOG] Saved to {log_path}")
    except Exception as ex:
        progress_cb(f"[WARN] failed to write log: {ex}")


class ASOTInferWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(str, str)  # (txt_path, json_path)

    def __init__(
        self,
        features_dir,
        ckpt,
        class_names=None,
        smooth_k=1,
        standardize=False,
        out_prefix="pred_asot",
        tool_path="asot_infer_adapter.py",
        allow_standardize=True,
        extra_args=None,
        log_path=None,
    ):
        super().__init__()
        self.features_dir = features_dir
        self.ckpt = ckpt
        self.class_names = class_names
        self.smooth_k = smooth_k
        self.standardize = standardize
        self.allow_standardize = allow_standardize
        self.out_prefix = out_prefix
        self.tool_path = tool_path
        self.extra_args = extra_args or []
        self.log_path = log_path or os.path.join(
            features_dir, f"{out_prefix}_infer.log"
        )

    def run(self):
        log_lines = []
        try:
            cmd = build_runner_cmd(
                repo_root=ACTION_REPO_ROOT,
                profile="asot",
                script_path=self.tool_path,
                script_args=[
                    "--features_dir",
                    self.features_dir,
                    "--ckpt",
                    self.ckpt,
                    "--out_prefix",
                    self.out_prefix,
                ],
                python_executable=sys.executable,
            )
            if self.class_names:
                cmd += ["--class_names", self.class_names]
            if self.standardize and self.allow_standardize:
                cmd.append("--standardize")
            if self.smooth_k and int(self.smooth_k) > 1:
                cmd += ["--smooth_k", str(int(self.smooth_k))]
            if self.extra_args:
                cmd += list(self.extra_args)

            returncode, log_lines = _run_streaming_command(cmd, self.progress.emit)
            txt_path = os.path.join(
                self.features_dir, f"{self.out_prefix}_segments.txt"
            )
            json_path = os.path.join(
                self.features_dir, f"{self.out_prefix}_segments.json"
            )
            ok = (returncode == 0) and os.path.isfile(txt_path)
            self.done.emit(txt_path if ok else "", json_path if ok else "")
        except Exception as ex:
            self.progress.emit(f"[ERROR] {ex}")
            self.done.emit("", "")
        finally:
            _write_worker_log(self.log_path, log_lines, self.progress.emit)


class ASOTRemapBuildWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(bool, str)  # (ok, output_json)

    def __init__(
        self,
        roots: Sequence[str],
        *,
        output_json: str,
        class_names: str = "",
        feature_search_roots: Optional[Sequence[str]] = None,
        pred_prefix: str = "pred_asot",
        tool_path: str = "",
        log_path: Optional[str] = None,
    ):
        super().__init__()
        self.roots = [str(x) for x in (roots or []) if str(x or "").strip()]
        self.output_json = str(output_json or "")
        self.class_names = str(class_names or "")
        self.feature_search_roots = [
            str(x) for x in (feature_search_roots or []) if str(x or "").strip()
        ]
        self.pred_prefix = str(pred_prefix or "pred_asot")
        self.tool_path = str(tool_path or "")
        self.log_path = log_path or (
            os.path.join(os.path.dirname(self.output_json), "asot_label_remap_build.log")
            if self.output_json
            else ""
        )

    def run(self):
        log_lines = []
        try:
            cmd = build_runner_cmd(
                repo_root=ACTION_REPO_ROOT,
                profile="current",
                script_path=self.tool_path,
                script_args=[
                    *self.roots,
                    "--pred-prefix",
                    self.pred_prefix,
                    "--output-json",
                    self.output_json,
                ],
                python_executable=sys.executable,
            )
            if self.class_names:
                cmd += ["--class-names", self.class_names]
            for path in self.feature_search_roots:
                cmd += ["--feature-search-root", path]
            returncode, log_lines = _run_streaming_command(cmd, self.progress.emit)
            ok = (returncode == 0) and os.path.isfile(self.output_json)
            self.done.emit(ok, self.output_json)
        except Exception as ex:
            self.progress.emit(f"[ERROR] {ex}")
            self.done.emit(False, self.output_json)
        finally:
            _write_worker_log(self.log_path, log_lines, self.progress.emit)


class EASTInferWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(str, str)  # (txt_path, json_path)

    def __init__(
        self,
        features_dir,
        video_id,
        video_path="",
        labels_txt="",
        ckpt="",
        cfg="",
        text_bank_version="disabled",
        feature_backbone_version="",
        category_adapter="",
        out_prefix="pred_east",
        log_path=None,
    ):
        super().__init__()
        self.features_dir = features_dir
        self.video_id = video_id
        self.video_path = str(video_path or "")
        self.labels_txt = str(labels_txt or "")
        self.ckpt = str(ckpt or "")
        self.cfg = str(cfg or "")
        self.text_bank_version = str(text_bank_version or "disabled")
        self.feature_backbone_version = str(feature_backbone_version or "")
        self.category_adapter = str(category_adapter or "")
        self.out_prefix = out_prefix
        self.log_path = log_path or os.path.join(features_dir, f"{out_prefix}_infer.log")

    def run(self):
        log_lines = []
        try:
            tool_path = os.path.join(
                ACTION_REPO_ROOT, "tools", "east", "east_infer_adapter.py"
            )
            if not os.path.isfile(tool_path):
                raise FileNotFoundError(f"EAST adapter script not found: {tool_path}")
            cmd = build_runner_cmd(
                repo_root=ACTION_REPO_ROOT,
                profile="east",
                script_path=tool_path,
                script_args=[
                    "--features_dir",
                    self.features_dir,
                    "--video_id",
                    self.video_id,
                    "--video_path",
                    self.video_path,
                    "--ckpt",
                    self.ckpt,
                    "--cfg",
                    self.cfg,
                    "--text_bank_version",
                    self.text_bank_version,
                    "--out_prefix",
                    self.out_prefix,
                ],
                python_executable=sys.executable,
            )
            if self.labels_txt:
                cmd += ["--labels_txt", self.labels_txt]
            if self.feature_backbone_version:
                cmd += [
                    "--feature_backbone_version",
                    self.feature_backbone_version,
                ]
            if self.category_adapter:
                cmd += ["--category_adapter", self.category_adapter]

            returncode, log_lines = _run_streaming_command(cmd, self.progress.emit)
            txt_path = os.path.join(
                self.features_dir, f"{self.out_prefix}_segments.txt"
            )
            json_path = os.path.join(
                self.features_dir, f"{self.out_prefix}_segments.json"
            )
            ok = (returncode == 0) and os.path.isfile(txt_path)
            self.done.emit(txt_path if ok else "", json_path if ok else "")
        except Exception as ex:
            self.progress.emit(f"[EAST][ERROR] {ex}")
            self.done.emit("", "")
        finally:
            _write_worker_log(self.log_path, log_lines, self.progress.emit)


class EASTMaskedRefreshWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(object)

    def __init__(
        self,
        features_dir: str,
        video_id: str,
        windows: Sequence[Dict[str, Any]],
        video_path: str = "",
        labels_txt: str = "",
        ckpt: str = "",
        cfg: str = "",
        text_bank_version: str = "disabled",
        feature_backbone_version: str = "",
        category_adapter: str = "",
        out_prefix: str = "pred_east_masked_refresh",
        log_path: Optional[str] = None,
    ):
        super().__init__()
        self.features_dir = str(features_dir or "")
        self.video_id = str(video_id or "")
        self.windows = [dict(item) for item in (windows or []) if isinstance(item, dict)]
        self.video_path = str(video_path or "")
        self.labels_txt = str(labels_txt or "")
        self.ckpt = str(ckpt or "")
        self.cfg = str(cfg or "")
        self.text_bank_version = str(text_bank_version or "disabled")
        self.feature_backbone_version = str(feature_backbone_version or "")
        self.category_adapter = str(category_adapter or "")
        self.out_prefix = str(out_prefix or "pred_east_masked_refresh")
        self.log_path = log_path or os.path.join(
            self.features_dir, f"{self.out_prefix}_infer.log"
        )

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        if not path or not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _load_record_buffer(path: str) -> List[Dict[str, Any]]:
        if not path or not os.path.isfile(path):
            return []
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, list):
                return [dict(x) for x in obj if isinstance(x, dict)]
        except Exception:
            pass
        return []

    @staticmethod
    def _dump_json(path: str, obj: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _copy_if_exists(src: str, dst: str) -> None:
        if src and os.path.isfile(src):
            shutil.copy2(src, dst)

    @staticmethod
    def _span_key(rec: Dict[str, Any]) -> Tuple[int, int]:
        try:
            s = int(rec.get("feedback_start", rec.get("start", 0)))
            e = int(rec.get("feedback_end", rec.get("end", s)))
        except Exception:
            return (0, -1)
        if e < s:
            s, e = e, s
        return int(s), int(e)

    @staticmethod
    def _nearest_local_index(frame_ids: Sequence[int], abs_frame: int) -> Optional[int]:
        if not frame_ids:
            return None
        idx = bisect_left(frame_ids, int(abs_frame))
        if idx <= 0:
            return 0
        if idx >= len(frame_ids):
            return int(len(frame_ids) - 1)
        left = int(frame_ids[idx - 1])
        right = int(frame_ids[idx])
        if abs(int(abs_frame) - left) <= abs(right - int(abs_frame)):
            return int(idx - 1)
        return int(idx)

    @classmethod
    def _local_span_from_abs(
        cls,
        frame_ids: Sequence[int],
        abs_start: int,
        abs_end: int,
    ) -> Optional[Tuple[int, int]]:
        if not frame_ids:
            return None
        s = int(abs_start)
        e = int(abs_end)
        if e < s:
            s, e = e, s
        left = bisect_left(frame_ids, s)
        right = bisect_right(frame_ids, e) - 1
        if left <= right and 0 <= left < len(frame_ids) and 0 <= right < len(frame_ids):
            return int(left), int(right)
        near_s = cls._nearest_local_index(frame_ids, s)
        near_e = cls._nearest_local_index(frame_ids, e)
        if near_s is None or near_e is None:
            return None
        left = min(int(near_s), int(near_e))
        right = max(int(near_s), int(near_e))
        if not (0 <= left < len(frame_ids) and 0 <= right < len(frame_ids)):
            return None
        return int(left), int(right)

    @classmethod
    def _localize_record_buffer(
        cls,
        records: Sequence[Dict[str, Any]],
        frame_ids: Sequence[int],
    ) -> List[Dict[str, Any]]:
        if not records or not frame_ids:
            return []
        window_start = int(frame_ids[0])
        window_end = int(frame_ids[-1])
        out: List[Dict[str, Any]] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            abs_s, abs_e = cls._span_key(rec)
            if abs_e < window_start or abs_s > window_end:
                continue
            local_span = cls._local_span_from_abs(
                frame_ids,
                max(abs_s, window_start),
                min(abs_e, window_end),
            )
            if local_span is None:
                continue
            new_rec = dict(rec)
            new_rec["feedback_start"] = int(local_span[0])
            new_rec["feedback_end"] = int(local_span[1])
            if "boundary_frame" in rec:
                try:
                    mapped = cls._nearest_local_index(frame_ids, int(rec.get("boundary_frame")))
                except Exception:
                    mapped = None
                if mapped is not None:
                    new_rec["boundary_frame"] = int(mapped)
            if "old_boundary_frame" in rec:
                try:
                    mapped = cls._nearest_local_index(frame_ids, int(rec.get("old_boundary_frame")))
                except Exception:
                    mapped = None
                if mapped is not None:
                    new_rec["old_boundary_frame"] = int(mapped)
            out.append(new_rec)
        return out

    @classmethod
    def _localize_seed_payload(
        cls,
        payload: Dict[str, Any],
        frame_ids: Sequence[int],
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict) or not frame_ids:
            return {}
        classes = [str(x).strip() for x in (payload.get("classes") or []) if str(x).strip()]
        raw_segments = payload.get("segments") or []
        if not isinstance(raw_segments, list):
            return {}
        window_start = int(frame_ids[0])
        window_end = int(frame_ids[-1])
        local_segments: List[Dict[str, Any]] = []
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            try:
                abs_s = int(seg.get("start_frame", 0))
                abs_e = int(seg.get("end_frame", abs_s))
            except Exception:
                continue
            if abs_e < window_start or abs_s > window_end:
                continue
            local_span = cls._local_span_from_abs(
                frame_ids,
                max(abs_s, window_start),
                min(abs_e, window_end),
            )
            if local_span is None:
                continue
            item = dict(seg)
            item["start_frame"] = int(local_span[0])
            item["end_frame"] = int(local_span[1])
            local_segments.append(item)
        if not local_segments:
            return {}
        return {
            "segments": local_segments,
            "classes": list(classes),
            "source": str(payload.get("source") or "EAST"),
        }

    @classmethod
    def _map_segment_to_global(
        cls,
        seg: Dict[str, Any],
        frame_ids: Sequence[int],
        classes: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(seg, dict) or not frame_ids:
            return None
        try:
            local_s = int(seg.get("start_frame", 0))
            local_e = int(seg.get("end_frame", local_s))
        except Exception:
            return None
        if local_e < local_s:
            local_s, local_e = local_e, local_s
        if not (0 <= local_s < len(frame_ids) and 0 <= local_e < len(frame_ids)):
            return None
        label = str(seg.get("class_name", "") or "").strip()
        if not label:
            try:
                cid = int(seg.get("class_id", -1))
            except Exception:
                cid = -1
            if 0 <= cid < len(classes):
                label = str(classes[cid])
        if not label:
            return None
        item = {
            "start": int(frame_ids[local_s]),
            "end": int(frame_ids[local_e]),
            "label": label,
            "local_start": int(local_s),
            "local_end": int(local_e),
        }
        topk_rows: List[Tuple[str, Optional[float]]] = []
        for tk in seg.get("topk", []):
            if not isinstance(tk, dict):
                continue
            name = str(tk.get("name", "") or "").strip()
            if not name:
                try:
                    cid = int(tk.get("id", -1))
                except Exception:
                    cid = -1
                if 0 <= cid < len(classes):
                    name = str(classes[cid])
            if not name:
                continue
            score = tk.get("score")
            try:
                score = float(score) if score is not None else None
            except Exception:
                score = None
            topk_rows.append((name, score))
        if topk_rows:
            item["topk"] = list(topk_rows)
        return item

    @classmethod
    def _crop_segments_to_focus_spans(
        cls,
        segments: Sequence[Dict[str, Any]],
        focus_spans: Sequence[Tuple[int, int]],
    ) -> Tuple[List[Dict[str, Any]], List[int], Dict[Tuple[int, int], List[Tuple[str, Optional[float]]]]]:
        out_segments: List[Dict[str, Any]] = []
        boundary_candidates: List[int] = []
        topk_map: Dict[Tuple[int, int], List[Tuple[str, Optional[float]]]] = {}
        for seg in segments or []:
            if not isinstance(seg, dict):
                continue
            seg_s = int(seg.get("start", 0))
            seg_e = int(seg.get("end", seg_s))
            label = str(seg.get("label", "") or "").strip()
            topk = list(seg.get("topk") or [])
            if not label or seg_e < seg_s:
                continue
            for focus_s, focus_e in focus_spans or []:
                fs = int(focus_s)
                fe = int(focus_e)
                if seg_e < fs or seg_s > fe:
                    continue
                frag_s = max(seg_s, fs)
                frag_e = min(seg_e, fe)
                if frag_e < frag_s:
                    continue
                out_segments.append(
                    {
                        "start": int(frag_s),
                        "end": int(frag_e),
                        "label": label,
                    }
                )
                if seg_s == frag_s and frag_s > fs:
                    boundary_candidates.append(int(frag_s))
                if topk:
                    topk_map[(int(frag_s), int(frag_e))] = list(topk)
        out_segments.sort(key=lambda row: (int(row["start"]), int(row["end"]), str(row["label"])))
        boundary_candidates = sorted(set(int(x) for x in boundary_candidates))
        return out_segments, boundary_candidates, topk_map

    def _run_window(
        self,
        tool_path: str,
        window: Dict[str, Any],
        slice_dir: str,
        features: np.ndarray,
        base_meta: Dict[str, Any],
        base_records: Sequence[Dict[str, Any]],
        base_seed_payload: Dict[str, Any],
        window_idx: int,
    ) -> Dict[str, Any]:
        feat_start = int(window.get("feature_start", 0) or 0)
        feat_end = int(window.get("feature_end", feat_start) or feat_start)
        frame_ids = [int(x) for x in (window.get("frame_indices") or [])]
        focus_spans = [
            (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))
            for a, b in (window.get("focus_spans") or [])
        ]
        os.makedirs(slice_dir, exist_ok=True)
        runtime_dir = os.path.join(slice_dir, "east_runtime")
        os.makedirs(runtime_dir, exist_ok=True)

        local_features = np.asarray(features[feat_start : feat_end + 1], dtype=np.float32)
        np.save(os.path.join(slice_dir, "features.npy"), local_features)

        meta = dict(base_meta or {})
        meta["picked_indices"] = list(frame_ids)
        meta["num_frames"] = int(max(frame_ids) + 1) if frame_ids else int(local_features.shape[0])
        meta["masked_refresh_window"] = {
            "feature_start": int(feat_start),
            "feature_end": int(feat_end),
            "focus_spans": [[int(a), int(b)] for a, b in focus_spans],
        }
        self._dump_json(os.path.join(slice_dir, "meta.json"), meta)

        localized_records = self._localize_record_buffer(base_records, frame_ids)
        with open(os.path.join(runtime_dir, "record_buffer.pkl"), "wb") as f:
            pickle.dump(localized_records, f)

        for name in ("online_adapter.pt", "model_delta.pt", "label_text_bank.pt"):
            self._copy_if_exists(
                os.path.join(self.features_dir, "east_runtime", name),
                os.path.join(runtime_dir, name),
            )

        seed_payload = self._localize_seed_payload(base_seed_payload, frame_ids)
        if seed_payload:
            self._dump_json(
                os.path.join(slice_dir, "pred_east_segments.json"),
                seed_payload,
            )

        slice_prefix = f"{self.out_prefix}_slice_{int(window_idx):03d}"
        cmd = build_runner_cmd(
            repo_root=ACTION_REPO_ROOT,
            profile="east",
            script_path=tool_path,
            script_args=[
                "--features_dir",
                slice_dir,
                "--video_id",
                f"{self.video_id}_slice_{int(window_idx):03d}",
                "--video_path",
                self.video_path,
                "--ckpt",
                self.ckpt,
                "--cfg",
                self.cfg,
                "--text_bank_version",
                self.text_bank_version,
                "--out_prefix",
                slice_prefix,
            ],
            python_executable=sys.executable,
        )
        if self.labels_txt:
            cmd += ["--labels_txt", self.labels_txt]
        if self.feature_backbone_version:
            cmd += ["--feature_backbone_version", self.feature_backbone_version]
        if self.category_adapter:
            cmd += ["--category_adapter", self.category_adapter]

        returncode, log_lines = _run_streaming_command(cmd, self.progress.emit)
        json_path = os.path.join(slice_dir, f"{slice_prefix}_segments.json")
        txt_path = os.path.join(slice_dir, f"{slice_prefix}_segments.txt")
        if returncode != 0 or not os.path.isfile(json_path):
            return {
                "ok": False,
                "window_index": int(window_idx),
                "feature_start": int(feat_start),
                "feature_end": int(feat_end),
                "focus_spans": [[int(a), int(b)] for a, b in focus_spans],
                "log_lines": list(log_lines),
                "error": "slice_inference_failed",
                "txt_path": txt_path if os.path.isfile(txt_path) else "",
                "json_path": json_path if os.path.isfile(json_path) else "",
            }

        payload = self._load_json(json_path)
        classes = [str(x).strip() for x in (payload.get("classes") or []) if str(x).strip()]
        raw_segments = payload.get("segments") or []
        mapped_segments: List[Dict[str, Any]] = []
        for seg in raw_segments:
            item = self._map_segment_to_global(seg, frame_ids, classes)
            if item is None:
                continue
            mapped_segments.append(item)

        cropped_segments, boundary_candidates, topk_map = self._crop_segments_to_focus_spans(
            mapped_segments,
            focus_spans,
        )
        return {
            "ok": True,
            "window_index": int(window_idx),
            "feature_start": int(feat_start),
            "feature_end": int(feat_end),
            "slice_frame_start": int(frame_ids[0]) if frame_ids else 0,
            "slice_frame_end": int(frame_ids[-1]) if frame_ids else 0,
            "focus_spans": [[int(a), int(b)] for a, b in focus_spans],
            "classes": list(classes),
            "segments": list(cropped_segments),
            "boundary_candidates": list(boundary_candidates),
            "topk_map": [
                {
                    "start": int(s),
                    "end": int(e),
                    "items": [
                        {"name": str(name), "score": score}
                        for name, score in items
                    ],
                }
                for (s, e), items in topk_map.items()
            ],
            "txt_path": txt_path,
            "json_path": json_path,
            "log_lines": list(log_lines),
        }

    def run(self):
        log_lines: List[str] = []
        temp_root = ""
        try:
            tool_path = os.path.join(
                ACTION_REPO_ROOT, "tools", "east", "east_infer_adapter.py"
            )
            if not os.path.isfile(tool_path):
                raise FileNotFoundError(f"EAST adapter script not found: {tool_path}")
            if not self.windows:
                self.done.emit(
                    {
                        "ok": False,
                        "features_dir": self.features_dir,
                        "error": "no_windows",
                        "windows": [],
                    }
                )
                return
            feat_path = os.path.join(self.features_dir, "features.npy")
            if not os.path.isfile(feat_path):
                raise FileNotFoundError(f"features.npy not found: {feat_path}")
            features = np.load(feat_path, mmap_mode="r")
            base_meta = self._load_json(os.path.join(self.features_dir, "meta.json"))
            runtime_dir = os.path.join(self.features_dir, "east_runtime")
            base_records = self._load_record_buffer(os.path.join(runtime_dir, "record_buffer.pkl"))
            base_seed_payload = self._load_json(os.path.join(runtime_dir, "segments.json"))
            if not base_seed_payload:
                base_seed_payload = self._load_json(
                    os.path.join(self.features_dir, "pred_east_segments.json")
                )
            if not os.path.isdir(runtime_dir):
                os.makedirs(runtime_dir, exist_ok=True)
            temp_root = tempfile.mkdtemp(
                prefix="masked_refresh_",
                dir=runtime_dir,
            )
            results: List[Dict[str, Any]] = []
            ok_count = 0
            for idx, window in enumerate(self.windows):
                frame_ids = [int(x) for x in (window.get("frame_indices") or [])]
                if not frame_ids:
                    continue
                self.progress.emit(
                    f"[EAST][REFRESH] Running local window {int(idx + 1)}/{int(len(self.windows))} "
                    f"({int(frame_ids[0])}-{int(frame_ids[-1])})"
                )
                slice_dir = os.path.join(temp_root, f"slice_{int(idx):03d}")
                result = self._run_window(
                    tool_path=tool_path,
                    window=window,
                    slice_dir=slice_dir,
                    features=features,
                    base_meta=base_meta,
                    base_records=base_records,
                    base_seed_payload=base_seed_payload,
                    window_idx=idx,
                )
                results.append(result)
                log_lines.extend(list(result.get("log_lines") or []))
                if bool(result.get("ok", False)):
                    ok_count += 1
            self.done.emit(
                {
                    "ok": bool(ok_count > 0),
                    "features_dir": self.features_dir,
                    "window_count": int(len(self.windows)),
                    "applied_window_count": int(ok_count),
                    "windows": results,
                    "partial": bool(0 < ok_count < len(self.windows)),
                    "error": "" if ok_count > 0 else "all_windows_failed",
                }
            )
        except Exception as ex:
            self.progress.emit(f"[EAST][ERROR] {ex}")
            self.done.emit(
                {
                    "ok": False,
                    "features_dir": self.features_dir,
                    "window_count": int(len(self.windows)),
                    "applied_window_count": 0,
                    "windows": [],
                    "error": str(ex),
                }
            )
        finally:
            if temp_root and os.path.isdir(temp_root):
                try:
                    shutil.rmtree(temp_root, ignore_errors=True)
                except Exception:
                    pass
            _write_worker_log(self.log_path, log_lines, self.progress.emit)


class FactBatchWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(bool, str)  # (ok, output_dir)

    def __init__(
        self,
        video_dir,
        output_dir,
        fact_repo,
        ckpt,
        fact_cfg,
        tool_path,
        class_names=None,
        log_path=None,
    ):
        super().__init__()
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.fact_repo = fact_repo
        self.ckpt = ckpt
        self.fact_cfg = fact_cfg
        self.tool_path = tool_path
        self.class_names = class_names
        self.log_path = log_path or os.path.join(output_dir, "pred_fact_batch.log")

    def run(self):
        log_lines = []
        try:
            cmd = build_runner_cmd(
                repo_root=ACTION_REPO_ROOT,
                profile="current",
                script_path=self.tool_path,
                script_args=[
                    "--video_dir",
                    self.video_dir,
                    "--output_dir",
                    self.output_dir,
                    "--fact_repo",
                    self.fact_repo,
                    "--fact_cfg",
                    self.fact_cfg,
                    "--ckpt",
                    self.ckpt,
                ],
                python_executable=sys.executable,
            )
            if self.class_names:
                cmd += ["--class_names", self.class_names]

            returncode, log_lines = _run_streaming_command(cmd, self.progress.emit)
            self.done.emit(returncode == 0, self.output_dir)
        except Exception as ex:
            self.progress.emit(f"[ERROR] {ex}")
            self.done.emit(False, self.output_dir)
        finally:
            _write_worker_log(self.log_path, log_lines, self.progress.emit)


class EASTBatchWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(bool, str)  # (ok, output_dir)

    def __init__(
        self,
        video_dir: str,
        output_dir: str,
        ckpt: str,
        cfg: str,
        *,
        labels_txt: str = "",
        category_adapter: str = "",
        text_bank_version: str = "auto",
        feature_backbone_version: str = "east_backbone",
        tool_path: str = "",
        log_path: Optional[str] = None,
    ):
        super().__init__()
        self.video_dir = str(video_dir or "")
        self.output_dir = str(output_dir or "")
        self.ckpt = str(ckpt or "")
        self.cfg = str(cfg or "")
        self.labels_txt = str(labels_txt or "")
        self.category_adapter = str(category_adapter or "")
        self.text_bank_version = str(text_bank_version or "auto")
        self.feature_backbone_version = str(
            feature_backbone_version or "east_backbone"
        )
        self.tool_path = str(tool_path or "")
        self.log_path = log_path or os.path.join(output_dir, "pred_east_batch.log")

    def run(self):
        log_lines = []
        try:
            cmd = build_runner_cmd(
                repo_root=ACTION_REPO_ROOT,
                profile="east",
                script_path=self.tool_path,
                script_args=[
                    "--video_dir",
                    self.video_dir,
                    "--output_dir",
                    self.output_dir,
                    "--ckpt",
                    self.ckpt,
                    "--cfg",
                    self.cfg,
                    "--text_bank_version",
                    self.text_bank_version,
                    "--feature_backbone_version",
                    self.feature_backbone_version,
                ],
                python_executable=sys.executable,
            )
            if self.labels_txt:
                cmd += ["--labels_txt", self.labels_txt]
            if self.category_adapter:
                cmd += ["--category_adapter", self.category_adapter]

            returncode, log_lines = _run_streaming_command(cmd, self.progress.emit)
            self.done.emit(returncode == 0, self.output_dir)
        except Exception as ex:
            self.progress.emit(f"[ERROR] {ex}")
            self.done.emit(False, self.output_dir)
        finally:
            _write_worker_log(self.log_path, log_lines, self.progress.emit)


class FeatureExtractWorker(QObject):
    progress = pyqtSignal(str)
    progress_value = pyqtSignal(int, int)
    done = pyqtSignal(object, bool)  # (features_dir, ok)

    def __init__(
        self,
        video_path: str,
        features_dir: str,
        batch_size: int = 128,
        frame_stride: int = 1,
        use_fp16: bool = True,
        backbone: Optional[str] = None,
        east_ckpt: str = "",
        east_cfg: str = "",
    ):
        super().__init__()
        self.video_path = video_path
        self.features_dir = features_dir
        self.batch_size = batch_size
        self.frame_stride = frame_stride
        self.use_fp16 = use_fp16
        self.backbone = str(backbone or "").strip()
        self.east_ckpt = str(east_ckpt or "").strip()
        self.east_cfg = str(east_cfg or "").strip()

    def run(self):
        try:
            feat_path = os.path.join(self.features_dir, "features.npy")
            os.makedirs(self.features_dir, exist_ok=True)
            backbone = self.backbone or os.environ.get(
                "FEATURE_BACKBONE", "dinov2_vitb14"
            )
            self.progress.emit(f"[FEATS] Extracting {backbone} features to {feat_path}")
            feature_api = load_feature_extractor_module()

            backbone_key = str(backbone or "").strip().lower()
            if backbone_key in {
                "east",
                "east_backbone",
                "east-backbone",
                "videomae",
                "videomaev2",
                "video_mae",
                "videomae2",
            }:
                east_ckpt = self.east_ckpt or str(os.environ.get("EAST_CKPT", "") or "").strip()
                east_cfg = self.east_cfg or str(os.environ.get("EAST_CFG", "") or "").strip()
                if not east_ckpt or not east_cfg:
                    raise RuntimeError(
                        "EAST feature extraction requires both EAST_CKPT and EAST_CFG."
                    )
                from tools.east.east_model_infer import extract_east_backbone_features

                east_out = extract_east_backbone_features(
                    video_path=self.video_path,
                    cfg_path=east_cfg,
                    ckpt_path=east_ckpt,
                    frame_stride=max(1, self.frame_stride),
                    device=str(os.environ.get("EAST_FEAT_DEVICE", "") or "").strip()
                    or None,
                    progress_cb=lambda line: self.progress.emit(str(line)),
                )
                feats = np.asarray((east_out or {}).get("features"), dtype=np.float32)
                meta = dict((east_out or {}).get("meta") or {})
                if feats.ndim != 2:
                    raise RuntimeError(
                        f"EAST feature extractor returned invalid shape: {feats.shape}"
                    )
                feature_api.save_features(self.features_dir, feats, meta=meta)
                self.progress.emit(
                    f"[FEATS] Saved EAST features {feats.shape} to {feat_path}"
                )
                self.done.emit(self.features_dir, True)
                return

            def _emit_progress(done: int, total: int):
                self.progress_value.emit(done, total)

            feats, meta = feature_api.extract_video_features(
                self.video_path,
                backbone=backbone,
                batch_size=self.batch_size,
                frame_stride=max(1, self.frame_stride),
                use_fp16=self.use_fp16,
                progress_cb=_emit_progress,
            )
            if bool(meta.get("model_cached", False)):
                self.progress.emit(f"[FEATS] Reused cached {backbone} model.")
            else:
                self.progress.emit(f"[FEATS] Loaded {backbone} model.")
            feature_api.save_features(self.features_dir, feats, meta=meta)
            self.progress.emit(f"[FEATS] Saved features {feats.shape} to {feat_path}")
            self.done.emit(self.features_dir, True)
        except MissingOptionalDependency as ex:
            self.progress.emit(f"[FEATS][ERROR] {format_missing_dependency_message(ex)}")
            self.done.emit(None, False)
        except Exception as ex:
            self.progress.emit(f"[FEATS][ERROR] {ex}")
            self.done.emit(None, False)


class EASTOnlineUpdateWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(object)

    def __init__(self, features_dir: str, trim_ratio: float = 0.1):
        super().__init__()
        self.features_dir = str(features_dir or "")
        self.trim_ratio = float(trim_ratio)

    def run(self):
        try:
            stats = rebuild_runtime_model_delta(
                self.features_dir,
                trim_ratio=self.trim_ratio,
                progress_cb=self.progress.emit,
            )
        except Exception as ex:
            stats = {
                "ok": False,
                "changed": False,
                "features_dir": self.features_dir,
                "error": str(ex),
            }
            self.progress.emit(f"[EAST][ONLINE][ERROR] {ex}")
        self.done.emit(stats)


class EASTOnlineAdapterTrainWorker(QObject):
    progress = pyqtSignal(str)
    done = pyqtSignal(object)

    def __init__(self, features_dir: str):
        super().__init__()
        self.features_dir = str(features_dir or "")

    def run(self):
        try:
            stats = rebuild_runtime_online_adapter(
                self.features_dir,
                progress_cb=self.progress.emit,
            )
        except Exception as ex:
            stats = {
                "ok": False,
                "changed": False,
                "features_dir": self.features_dir,
                "error": str(ex),
            }
            self.progress.emit(f"[EAST][ADAPTER][ERROR] {ex}")
        self.done.emit(stats)
