from __future__ import annotations

import hashlib
import io
import json
import os
import random
import select
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .mask_ops import bbox_from_mask_rle, bbox_is_valid


@dataclass
class SAMBackendConfig:
    provider: str
    max_instances_per_prompt: int
    enable_two_stage_refinement: bool
    cache_dir: str
    external_command_template: str = ""
    external_command_args: Tuple[str, ...] = field(default_factory=tuple)
    external_command_args_file: str = ""
    external_batch_command_template: str = ""
    external_batch_command_args: Tuple[str, ...] = field(default_factory=tuple)
    external_batch_command_args_file: str = ""
    external_command_cwd: str = ""
    external_timeout_sec: int = 1800
    external_use_persistent_process: bool = True
    mock_results_path: str = ""
    disable_cache: bool = False
    progress_cb: Optional[Callable[[str], None]] = None
    cancel_cb: Optional[Callable[[], bool]] = None


class SAMBackend:
    """
    Grounding backend wrapper used by SAM3-based proposal/refinement.
    """

    def __init__(self, cfg: SAMBackendConfig):
        self.cfg = cfg
        os.makedirs(self.cfg.cache_dir, exist_ok=True)
        self._feature_cache: Dict[str, Dict[str, object]] = {}
        self._active_proc: Optional[subprocess.Popen] = None
        self._persistent_proc: Optional[subprocess.Popen] = None
        self._persistent_lock = threading.Lock()
        self._persistent_stderr_thread: Optional[threading.Thread] = None
        self._last_command_debug: Dict[str, object] = {}

    @staticmethod
    def _path_signature(path: str) -> str:
        norm = os.path.abspath(str(path or ""))
        if not norm:
            return ""
        try:
            st = os.stat(norm)
            mtime_ns = int(getattr(st, "st_mtime_ns", int(float(st.st_mtime) * 1_000_000_000)))
            return f"{norm}|{int(st.st_size)}|{mtime_ns}"
        except Exception:
            return norm

    def _image_fingerprint(self, image_path: str) -> str:
        return self._path_signature(image_path)

    def _emit_progress(self, message: str) -> None:
        cb = self.cfg.progress_cb
        if not callable(cb):
            return
        msg = self._summarize_progress_line(message)
        if not msg:
            return
        try:
            cb(msg)
        except Exception:
            pass

    def _cancel_requested(self) -> bool:
        cb = self.cfg.cancel_cb
        if not callable(cb):
            return False
        try:
            return bool(cb())
        except Exception:
            return False

    def cancel_active_process(self) -> None:
        proc = self._active_proc
        if proc is None:
            return
        try:
            proc.kill()
        except Exception:
            pass
        with self._persistent_lock:
            persistent = self._persistent_proc
        if persistent is not None:
            try:
                persistent.kill()
            except Exception:
                pass

    def close(self) -> None:
        with self._persistent_lock:
            proc = self._persistent_proc
            self._persistent_proc = None
            self._persistent_stderr_thread = None
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    @staticmethod
    def _summarize_progress_line(message: str) -> str:
        text = str(message or "").strip()
        if not text:
            return ""
        benign_warnings = (
            "some model parameters or buffers are not found in the checkpoint",
            "the checkpoint state_dict contains keys that are not used by the model",
        )
        lower = text.lower()
        if any(pattern in lower for pattern in benign_warnings):
            return ""
        marker = "[OWSAM]"
        idx = text.find(marker)
        if idx >= 0:
            return text[idx:].strip()
        tokens = (
            "traceback",
            "exception",
            "error",
            "failed",
            "timed out",
            "no module named",
            "not found",
            "missing",
        )
        if not any(token in lower for token in tokens):
            return ""
        if len(text) <= 240:
            return text
        return text[:237].rstrip() + "..."

    def _backend_signature(self) -> str:
        payload = {
            "provider": str(self.cfg.provider or "mock").strip().lower(),
            "max_instances_per_prompt": int(self.cfg.max_instances_per_prompt),
            "enable_two_stage_refinement": bool(self.cfg.enable_two_stage_refinement),
            "external_command_template": str(self.cfg.external_command_template or ""),
            "external_command_args": list(self.cfg.external_command_args or ()),
            "external_command_args_file": self._path_signature(self.cfg.external_command_args_file),
            "external_batch_command_template": str(self.cfg.external_batch_command_template or ""),
            "external_batch_command_args": list(self.cfg.external_batch_command_args or ()),
            "external_batch_command_args_file": self._path_signature(self.cfg.external_batch_command_args_file),
            "external_command_cwd": os.path.abspath(str(self.cfg.external_command_cwd or "")) if self.cfg.external_command_cwd else "",
            "external_timeout_sec": int(max(1, int(self.cfg.external_timeout_sec or 1800))),
            "external_use_persistent_process": bool(self.cfg.external_use_persistent_process),
            "mock_results_path": self._path_signature(self.cfg.mock_results_path),
        }
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)

    def _cache_key(self, image_path: str, prompt: str, stage: str) -> str:
        raw = f"{self._image_fingerprint(image_path)}|{prompt}|{stage}|{self._backend_signature()}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cfg.cache_dir, f"ows_{key}.json")

    def _read_cache(self, key: str) -> Optional[List[Dict[str, object]]]:
        if bool(self.cfg.disable_cache):
            return None
        path = self._cache_path(key)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_cache(self, key: str, items: List[Dict[str, object]]) -> None:
        if bool(self.cfg.disable_cache):
            return
        path = self._cache_path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=True, indent=2)

    def _mock_results(self, image_path: str, prompt: str, stage: str) -> List[Dict[str, object]]:
        if self.cfg.mock_results_path and os.path.isfile(self.cfg.mock_results_path):
            with open(self.cfg.mock_results_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            records = payload.get("records") if isinstance(payload, dict) else payload
            if isinstance(records, list):
                out = []
                for item in records[: max(1, int(self.cfg.max_instances_per_prompt))]:
                    if not isinstance(item, dict):
                        continue
                    row = self._normalize_result_item(
                        item,
                        image_path=image_path,
                        prompt=prompt,
                        stage=stage,
                        provider_name="mock_file",
                    )
                    row["backend_metadata"]["source"] = self.cfg.mock_results_path
                    if "score" not in item and "confidence" not in item:
                        row["score"] = 0.5
                    out.append(row)
                return out

        # Deterministic pseudo proposals for smoke testing.
        seed = int(hashlib.sha1(f"{image_path}|{prompt}|{stage}".encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)
        n = min(max(1, int(self.cfg.max_instances_per_prompt)), 3)
        out = []
        for idx in range(n):
            x0 = rng.randint(5, 100)
            y0 = rng.randint(5, 100)
            w = rng.randint(20, 80)
            h = rng.randint(20, 80)
            pixels = []
            for x in range(x0, x0 + w):
                for y in range(y0, y0 + h):
                    if (x + y) % 3 == 0:
                        pixels.append([x, y])
            mask = {"pixels": pixels}
            out.append(
                {
                    "mask": mask,
                    "bbox": bbox_from_mask_rle(mask),
                    "score": round(0.45 + 0.15 * rng.random(), 4),
                    "prompt_used": prompt,
                    "stage": stage,
                    "backend_metadata": {"provider": "mock", "image": image_path, "index": idx},
                }
            )
        return out

    @staticmethod
    def _coerce_pixels(value: Any) -> List[List[int]]:
        if not isinstance(value, list):
            return []
        out: List[List[int]] = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                out.append([int(float(item[0])), int(float(item[1]))])
            except Exception:
                continue
        return out

    @staticmethod
    def _coerce_bbox_value(value: Any) -> List[int]:
        if isinstance(value, dict):
            if all(k in value for k in ("x", "y", "w", "h")):
                value = [value.get("x"), value.get("y"), value.get("w"), value.get("h")]
            elif all(k in value for k in ("x1", "y1", "x2", "y2")):
                x1 = float(value.get("x1", 0) or 0)
                y1 = float(value.get("y1", 0) or 0)
                x2 = float(value.get("x2", x1) or x1)
                y2 = float(value.get("y2", y1) or y1)
                value = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
            else:
                return [0, 0, 0, 0]
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return [0, 0, 0, 0]
        try:
            x = int(round(float(value[0] or 0)))
            y = int(round(float(value[1] or 0)))
            w = int(round(float(value[2] or 0)))
            h = int(round(float(value[3] or 0)))
            return [x, y, max(0, w), max(0, h)]
        except Exception:
            return [0, 0, 0, 0]

    def _coerce_bbox(self, item: Dict[str, object]) -> List[int]:
        for key in ("bbox", "xywh", "box"):
            bbox = self._coerce_bbox_value(item.get(key))
            if bbox_is_valid(bbox):
                return bbox
        xyxy = item.get("xyxy")
        if isinstance(xyxy, (list, tuple)) and len(xyxy) >= 4:
            try:
                x1 = float(xyxy[0] or 0)
                y1 = float(xyxy[1] or 0)
                x2 = float(xyxy[2] or x1)
                y2 = float(xyxy[3] or y1)
                bbox = [
                    int(round(x1)),
                    int(round(y1)),
                    int(round(max(0.0, x2 - x1))),
                    int(round(max(0.0, y2 - y1))),
                ]
                if bbox_is_valid(bbox):
                    return bbox
            except Exception:
                pass
        bbox = self._coerce_bbox_value(item)
        if bbox_is_valid(bbox):
            return bbox
        return [0, 0, 0, 0]

    def _normalize_mask(self, item: Dict[str, object]) -> Dict[str, object]:
        raw_mask = item.get("mask")
        if isinstance(raw_mask, dict):
            pixels = self._coerce_pixels(raw_mask.get("pixels"))
            if pixels:
                return {"pixels": pixels}
        pixels = self._coerce_pixels(item.get("pixels"))
        if pixels:
            return {"pixels": pixels}
        return {"pixels": []}

    def _normalize_result_item(
        self,
        item: Dict[str, object],
        *,
        image_path: str,
        prompt: str,
        stage: str,
        provider_name: str,
        raw_stdout: str = "",
    ) -> Dict[str, object]:
        mask = self._normalize_mask(item)
        bbox = bbox_from_mask_rle(mask)
        raw_bbox = self._coerce_bbox(item)
        if not bbox_is_valid(bbox) and bbox_is_valid(raw_bbox):
            bbox = raw_bbox
        metadata = dict(item.get("backend_metadata") or {}) if isinstance(item.get("backend_metadata"), dict) else {}
        metadata.setdefault("provider", provider_name)
        metadata.setdefault("image", image_path)
        return {
            "mask": mask,
            "bbox": bbox if bbox_is_valid(bbox) else [0, 0, 0, 0],
            "score": float(item.get("score", item.get("confidence", 0.0)) or 0.0),
            "prompt_used": prompt,
            "stage": stage,
            "backend_metadata": {
                **metadata,
                "stdout_len": len(raw_stdout),
            },
        }

    @staticmethod
    def _parse_json_payload(text: str) -> Any:
        raw = str(text or "").strip()
        if not raw:
            return []
        try:
            return json.loads(raw)
        except Exception:
            pass
        starts = [idx for idx, ch in enumerate(raw) if ch in "[{"]
        for idx in reversed(starts):
            try:
                return json.loads(raw[idx:])
            except Exception:
                continue
        raise RuntimeError("External SAM command did not return parseable JSON.")

    @staticmethod
    def _extract_payload_records(payload: Any) -> List[Dict[str, object]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            for key in ("records", "results", "instances", "predictions", "detections", "objects", "data"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            if any(key in payload for key in ("mask", "pixels", "bbox", "xyxy", "score", "confidence")):
                return [payload]
            if len(payload) == 1:
                only_val = next(iter(payload.values()))
                if isinstance(only_val, list):
                    return [item for item in only_val if isinstance(item, dict)]
        raise RuntimeError("External SAM command must return a JSON list or an object containing a record list.")

    def _template_context(
        self,
        image_path: str,
        prompt: str,
        stage: str,
        *,
        request_json_path: str = "",
    ) -> Dict[str, object]:
        max_instances = max(1, int(self.cfg.max_instances_per_prompt))
        return {
            "image_path": image_path,
            "image_path_quoted": subprocess.list2cmdline([image_path]),
            "prompt": prompt,
            "prompt_quoted": subprocess.list2cmdline([prompt]),
            "stage": stage,
            "stage_quoted": subprocess.list2cmdline([stage]),
            "max_instances": max_instances,
            "request_json_path": request_json_path,
            "request_json_path_quoted": subprocess.list2cmdline([request_json_path]) if request_json_path else "",
        }

    def _external_argv(
        self,
        image_path: str,
        prompt: str,
        stage: str,
        *,
        raw_args: Sequence[str],
        args_file: str,
        request_json_path: str = "",
    ) -> List[str]:
        context = self._template_context(image_path, prompt, stage, request_json_path=request_json_path)
        raw_args = tuple(raw_args or ())
        if not raw_args and args_file:
            try:
                with open(args_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                raise RuntimeError(f"Failed to load external_command_args_file: {args_file}. {exc}") from exc
            if not isinstance(payload, list):
                raise RuntimeError("external_command_args_file must contain a JSON array of command argv tokens.")
            raw_args = [str(x) for x in payload]
        return [str(token).format(**context) for token in raw_args]

    @staticmethod
    def _decode_output_bytes(data: bytes) -> str:
        if not data:
            return ""
        try:
            return data.decode("utf-8")
        except Exception:
            return data.decode("utf-8", errors="replace")

    def _execute_external_command(
        self,
        *,
        image_path: str,
        prompt: str,
        stage: str,
        raw_args: Sequence[str],
        args_file: str,
        template: str,
        request_json_path: str = "",
    ) -> str:
        argv = self._external_argv(
            image_path,
            prompt,
            stage,
            raw_args=raw_args,
            args_file=args_file,
            request_json_path=request_json_path,
        )
        timeout_sec = max(1, int(self.cfg.external_timeout_sec or 1800))
        run_kwargs: Dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }
        cwd = str(self.cfg.external_command_cwd or "").strip()
        if cwd:
            run_kwargs["cwd"] = cwd
        child_env = os.environ.copy()
        child_env.setdefault("PYTHONIOENCODING", "utf-8")
        child_env.setdefault("PYTHONUTF8", "1")
        run_kwargs["env"] = child_env
        if template:
            run_kwargs["shell"] = True
        else:
            run_kwargs["shell"] = False
        if argv:
            proc = subprocess.Popen(argv, **run_kwargs)
        else:
            tmpl = str(template or "").strip()
            if not tmpl:
                raise RuntimeError(
                    "external_command provider requires external_command_args, external_command_args_file, or external_command_template."
                )
            cmd = tmpl.format(**self._template_context(image_path, prompt, stage, request_json_path=request_json_path))
            proc = subprocess.Popen(cmd, **run_kwargs)
        self._active_proc = proc
        argv_debug = [str(x) for x in list(argv or [])] if argv else [str(template or "").strip()]
        self._last_command_debug = {
            "argv": argv_debug,
            "image_path": image_path,
            "prompt": prompt,
            "stage": stage,
            "request_json_path": request_json_path,
            "return_code": None,
            "stdout_len": 0,
            "stderr_tail": "",
        }

        stdout_chunks: List[bytes] = []
        stderr_chunks: List[bytes] = []

        def _pipe_reader(stream, chunks: List[bytes], *, emit_progress: bool) -> None:
            if stream is None:
                return
            try:
                for raw_line in iter(stream.readline, b""):
                    if not raw_line:
                        break
                    chunks.append(raw_line)
                    if emit_progress:
                        line = self._decode_output_bytes(raw_line).strip()
                        if line:
                            self._emit_progress(line)
                tail = stream.read()
                if tail:
                    chunks.append(tail)
                    if emit_progress:
                        text = self._decode_output_bytes(tail)
                        for line in text.splitlines():
                            line2 = str(line or "").strip()
                            if line2:
                                self._emit_progress(line2)
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        stdout_thread = threading.Thread(
            target=_pipe_reader,
            args=(proc.stdout, stdout_chunks),
            kwargs={"emit_progress": False},
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_pipe_reader,
            args=(proc.stderr, stderr_chunks),
            kwargs={"emit_progress": True},
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        try:
            elapsed = 0.0
            poll_step = min(0.2, max(0.05, float(timeout_sec) / 20.0))
            while True:
                if self._cancel_requested():
                    proc.kill()
                    raise RuntimeError("Scene graph run cancelled by user.")
                code = proc.poll()
                if code is not None:
                    break
                if elapsed >= float(timeout_sec):
                    raise subprocess.TimeoutExpired(argv or template, timeout_sec)
                threading.Event().wait(poll_step)
                elapsed += poll_step
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_thread.join(timeout=2.0)
            stderr_thread.join(timeout=2.0)
            stdout_bytes = b"".join(stdout_chunks)
            stderr_text = self._decode_output_bytes(b"".join(stderr_chunks)).strip()
            raise RuntimeError(
                f"SAM external command timed out after {timeout_sec}s. "
                f"{stderr_text or self._decode_output_bytes(stdout_bytes).strip()}"
            )
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            raise
        finally:
            self._active_proc = None

        stdout_thread.join(timeout=2.0)
        stderr_thread.join(timeout=2.0)
        stdout_bytes = b"".join(stdout_chunks)
        stdout_text = self._decode_output_bytes(stdout_bytes).strip()
        stderr_text = self._decode_output_bytes(b"".join(stderr_chunks)).strip()
        self._last_command_debug = {
            **dict(self._last_command_debug or {}),
            "return_code": int(proc.returncode),
            "stdout_len": len(stdout_text),
            "stderr_tail": stderr_text[-800:],
        }
        if proc.returncode != 0:
            err = str(stderr_text or stdout_text or "").strip()
            raise RuntimeError(f"SAM external command failed: {err}")
        return stdout_text

    def _external_use_persistent_process(self) -> bool:
        return bool(self.cfg.external_use_persistent_process)

    def _server_argv_from_batch_command(self) -> List[str]:
        argv = self._external_argv(
            image_path="",
            prompt="",
            stage="category_discovery",
            raw_args=tuple(self.cfg.external_batch_command_args or ()),
            args_file=str(self.cfg.external_batch_command_args_file or "").strip(),
            request_json_path="",
        )
        if not argv:
            return []
        out: List[str] = []
        skip_next = False
        for idx, token in enumerate(argv):
            if skip_next:
                skip_next = False
                continue
            low = str(token or "").strip().lower()
            if low in {"--request_json", "--request-json"}:
                skip_next = idx + 1 < len(argv)
                continue
            if "{request_json_path}" in str(token):
                continue
            if not str(token or "").strip():
                continue
            out.append(str(token))
        if "--serve_jsonl" not in out:
            out.append("--serve_jsonl")
        return out

    def _start_persistent_server_locked(self) -> subprocess.Popen:
        proc = self._persistent_proc
        if proc is not None and proc.poll() is None:
            return proc

        argv = self._server_argv_from_batch_command()
        if not argv:
            raise RuntimeError(
                "Persistent external process requires external_batch_command_args/_file."
            )

        run_kwargs: Dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        cwd = str(self.cfg.external_command_cwd or "").strip()
        if cwd:
            run_kwargs["cwd"] = cwd
        child_env = os.environ.copy()
        child_env.setdefault("PYTHONIOENCODING", "utf-8")
        child_env.setdefault("PYTHONUTF8", "1")
        run_kwargs["env"] = child_env
        proc = subprocess.Popen(argv, **run_kwargs)
        self._persistent_proc = proc
        self._active_proc = proc

        def _stderr_reader(stream: io.TextIOBase) -> None:
            try:
                for raw in stream:
                    line = str(raw or "").strip()
                    if line:
                        self._emit_progress(line)
            except Exception:
                return

        if proc.stderr is not None:
            thread = threading.Thread(target=_stderr_reader, args=(proc.stderr,), daemon=True)
            thread.start()
            self._persistent_stderr_thread = thread
        return proc

    def _send_persistent_request(self, payload: Dict[str, object]) -> Dict[str, object]:
        timeout_sec = max(1, int(self.cfg.external_timeout_sec or 1800))
        with self._persistent_lock:
            proc = self._start_persistent_server_locked()
            if proc.stdin is None or proc.stdout is None:
                raise RuntimeError("Persistent external server does not expose stdin/stdout.")
            line = json.dumps(payload, ensure_ascii=True)
            try:
                proc.stdin.write(line + "\n")
                proc.stdin.flush()
            except Exception as exc:
                raise RuntimeError(f"Failed to write request to persistent SAM process: {exc}") from exc

            elapsed = 0.0
            poll_step = min(0.2, max(0.05, float(timeout_sec) / 20.0))
            while True:
                if self._cancel_requested():
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    raise RuntimeError("Scene graph run cancelled by user.")
                if proc.poll() is not None:
                    stderr_text = ""
                    if proc.stderr is not None:
                        try:
                            stderr_text = proc.stderr.read().strip()
                        except Exception:
                            stderr_text = ""
                    raise RuntimeError(
                        f"Persistent SAM process exited unexpectedly: {stderr_text or 'no error output'}"
                    )
                try:
                    ready, _, _ = select.select([proc.stdout], [], [], poll_step)
                except Exception:
                    ready = [proc.stdout]
                if ready:
                    break
                if elapsed >= float(timeout_sec):
                    raise RuntimeError(f"Persistent SAM request timed out after {timeout_sec}s.")
                elapsed += poll_step

            response_line = proc.stdout.readline()
            if not response_line:
                raise RuntimeError("Persistent SAM process returned empty response.")
        try:
            payload_obj = json.loads(str(response_line or "").strip())
        except Exception as exc:
            raise RuntimeError(f"Persistent SAM response is not valid JSON: {response_line}") from exc
        if not isinstance(payload_obj, dict):
            raise RuntimeError("Persistent SAM response must be a JSON object.")
        if not bool(payload_obj.get("ok", False)):
            raise RuntimeError(str(payload_obj.get("error", "Persistent SAM request failed.")))
        return payload_obj

    def _has_batch_command(self) -> bool:
        return bool(
            str(self.cfg.external_batch_command_template or "").strip()
            or tuple(self.cfg.external_batch_command_args or ())
            or str(self.cfg.external_batch_command_args_file or "").strip()
        )

    def _external_results(self, image_path: str, prompt: str, stage: str) -> List[Dict[str, object]]:
        text = self._execute_external_command(
            image_path=image_path,
            prompt=prompt,
            stage=stage,
            raw_args=tuple(self.cfg.external_command_args or ()),
            args_file=str(self.cfg.external_command_args_file or "").strip(),
            template=str(self.cfg.external_command_template or "").strip(),
        )
        if not text:
            return []
        payload = self._parse_json_payload(text)
        records = self._extract_payload_records(payload)
        out: List[Dict[str, object]] = []
        for item in records[: max(1, int(self.cfg.max_instances_per_prompt))]:
            out.append(
                self._normalize_result_item(
                    item,
                    image_path=image_path,
                    prompt=prompt,
                    stage=stage,
                    provider_name="external_command",
                    raw_stdout=text,
                )
            )
        return out

    @staticmethod
    def _extract_prompt_record_map(payload: Any) -> Dict[str, List[Dict[str, object]]]:
        out: Dict[str, List[Dict[str, object]]] = {}
        if isinstance(payload, dict):
            results = payload.get("results")
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, list):
                        out[str(key)] = [item for item in value if isinstance(item, dict)]
                return out
            if isinstance(results, list):
                payload = results
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                prompt = str(item.get("prompt", "") or "").strip()
                records = item.get("records")
                if prompt and isinstance(records, list):
                    out[prompt] = [rec for rec in records if isinstance(rec, dict)]
            if out:
                return out
        raise RuntimeError("External SAM batch command must return prompt-keyed results.")

    def _external_batch_results(
        self,
        image_path: str,
        prompt_items: List[Dict[str, str]],
        stage: str,
    ) -> Dict[str, List[Dict[str, object]]]:
        if self._external_use_persistent_process():
            req_payload: Dict[str, object] = {
                "image_path": image_path,
                "stage": stage,
                "max_instances": max(1, int(self.cfg.max_instances_per_prompt)),
                "prompts": list(prompt_items or []),
            }
            resp = self._send_persistent_request(req_payload)
            payload = list(resp.get("results") or [])
            prompt_map = self._extract_prompt_record_map(payload)
            out2: Dict[str, List[Dict[str, object]]] = {}
            for item in prompt_items:
                prompt = str(item.get("prompt", "") or "").strip()
                records = prompt_map.get(prompt, [])
                normalized: List[Dict[str, object]] = []
                for record in records[: max(1, int(self.cfg.max_instances_per_prompt))]:
                    normalized.append(
                        self._normalize_result_item(
                            record,
                            image_path=image_path,
                            prompt=prompt,
                            stage=stage,
                            provider_name="external_command_persistent",
                            raw_stdout="",
                        )
                    )
                out2[prompt] = normalized
            return out2

        request_path = os.path.join(
            self.cfg.cache_dir,
            f"ows_batch_{hashlib.sha1(f'{image_path}|{stage}|{json.dumps(prompt_items, ensure_ascii=True, sort_keys=True)}'.encode('utf-8')).hexdigest()[:12]}.json",
        )
        try:
            with open(request_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "image_path": image_path,
                        "stage": stage,
                        "max_instances": max(1, int(self.cfg.max_instances_per_prompt)),
                        "prompts": list(prompt_items or []),
                    },
                    f,
                    ensure_ascii=True,
                    indent=2,
                )
            text = self._execute_external_command(
                image_path=image_path,
                prompt="",
                stage=stage,
                raw_args=tuple(self.cfg.external_batch_command_args or ()),
                args_file=str(self.cfg.external_batch_command_args_file or "").strip(),
                template=str(self.cfg.external_batch_command_template or "").strip(),
                request_json_path=request_path,
            )
        finally:
            try:
                if os.path.isfile(request_path):
                    os.remove(request_path)
            except Exception:
                pass
        if not text:
            return {str(item.get("prompt", "") or ""): [] for item in prompt_items}
        payload = self._parse_json_payload(text)
        prompt_map = self._extract_prompt_record_map(payload)
        out: Dict[str, List[Dict[str, object]]] = {}
        for item in prompt_items:
            prompt = str(item.get("prompt", "") or "").strip()
            records = prompt_map.get(prompt, [])
            normalized: List[Dict[str, object]] = []
            for record in records[: max(1, int(self.cfg.max_instances_per_prompt))]:
                normalized.append(
                    self._normalize_result_item(
                        record,
                        image_path=image_path,
                        prompt=prompt,
                        stage=stage,
                        provider_name="external_command",
                        raw_stdout=text,
                    )
                )
            out[prompt] = normalized
        return out

    def _run_prompt(self, image_path: str, prompt: str, stage: str) -> List[Dict[str, object]]:
        key = self._cache_key(image_path, prompt, stage)
        cached = self._read_cache(key)
        if cached is not None:
            return cached
        provider = str(self.cfg.provider or "mock").strip().lower()
        if provider == "external_command":
            out = self._external_results(image_path, prompt, stage)
        else:
            out = self._mock_results(image_path, prompt, stage)
        self._write_cache(key, out)
        return out

    def _run_prompt_batch(
        self,
        image_path: str,
        prompt_items: List[Dict[str, str]],
        stage: str,
    ) -> Dict[str, List[Dict[str, object]]]:
        out: Dict[str, List[Dict[str, object]]] = {}
        missing: List[Dict[str, str]] = []
        for item in prompt_items:
            prompt = str(item.get("prompt", "") or "").strip()
            if not prompt:
                continue
            key = self._cache_key(image_path, prompt, stage)
            cached = self._read_cache(key)
            if cached is not None:
                out[prompt] = cached
            else:
                missing.append(item)
        if not missing:
            return out

        provider = str(self.cfg.provider or "mock").strip().lower()
        if provider == "external_command" and self._has_batch_command():
            fresh = self._external_batch_results(image_path, missing, stage)
            for item in missing:
                prompt = str(item.get("prompt", "") or "").strip()
                rows = list(fresh.get(prompt, []))
                self._write_cache(self._cache_key(image_path, prompt, stage), rows)
                out[prompt] = rows
            return out

        for item in missing:
            prompt = str(item.get("prompt", "") or "").strip()
            rows = self._run_prompt(image_path, prompt, stage)
            out[prompt] = rows
        return out

    @staticmethod
    def _filter_by_score(rows: List[Dict[str, object]], score_threshold: Optional[float]) -> List[Dict[str, object]]:
        if score_threshold is None:
            return [dict(x) for x in list(rows or []) if isinstance(x, dict)]
        out: List[Dict[str, object]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            try:
                score = float(item.get("score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            if score >= float(score_threshold):
                out.append(dict(item))
        return out

    @staticmethod
    def _score_stats(rows: List[Dict[str, object]]) -> Dict[str, float]:
        scores: List[float] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            try:
                scores.append(float(item.get("score", 0.0) or 0.0))
            except Exception:
                continue
        if not scores:
            return {"min": 0.0, "max": 0.0}
        return {"min": min(scores), "max": max(scores)}

    def discover_entities_by_category_detailed(
        self,
        image_path: str,
        prompts: List[Dict[str, str]],
        *,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, object]:
        provider = str(self.cfg.provider or "mock").strip().lower()
        unique_items: List[Dict[str, str]] = []
        seen_prompt_keys = set()
        for item in prompts:
            canonical_label = str(item.get("canonical_label", "")).strip().lower()
            prompt = str(item.get("prompt", "")).strip()
            key = (canonical_label, prompt.lower())
            if not canonical_label or not prompt or key in seen_prompt_keys:
                continue
            seen_prompt_keys.add(key)
            unique_items.append({"prompt": prompt, "canonical_label": canonical_label})

        prompt_map: Dict[str, List[Dict[str, object]]] = {}
        prompt_debug: Dict[str, Dict[str, object]] = {}
        missing: List[Dict[str, str]] = []
        for item in unique_items:
            prompt = str(item.get("prompt", "") or "").strip()
            key = self._cache_key(image_path, prompt, "category_discovery")
            cached = self._read_cache(key)
            if cached is not None:
                prompt_map[prompt] = [dict(x) for x in list(cached or []) if isinstance(x, dict)]
                prompt_debug[prompt] = {
                    "cache_hit": True,
                    "command": None,
                }
            else:
                missing.append(item)

        if missing:
            if provider == "external_command" and self._has_batch_command():
                fresh = self._external_batch_results(image_path, missing, "category_discovery")
                shared_command = dict(self._last_command_debug or {})
                for item in missing:
                    prompt = str(item.get("prompt", "") or "").strip()
                    rows = [dict(x) for x in list(fresh.get(prompt, []) or []) if isinstance(x, dict)]
                    self._write_cache(self._cache_key(image_path, prompt, "category_discovery"), rows)
                    prompt_map[prompt] = rows
                    prompt_debug[prompt] = {
                        "cache_hit": False,
                        "command": dict(shared_command),
                    }
            else:
                for item in missing:
                    prompt = str(item.get("prompt", "") or "").strip()
                    rows = [dict(x) for x in list(self._run_prompt(image_path, prompt, "category_discovery") or []) if isinstance(x, dict)]
                    prompt_map[prompt] = rows
                    prompt_debug[prompt] = {
                        "cache_hit": False,
                        "command": dict(self._last_command_debug or {}),
                    }

        prompt_results: List[Dict[str, object]] = []
        flat_post_threshold: List[Dict[str, object]] = []
        for item in unique_items:
            prompt = str(item.get("prompt", "") or "").strip()
            canonical_label = str(item.get("canonical_label", "") or "").strip().lower()
            raw_rows = [dict(x) for x in list(prompt_map.get(prompt, []) or []) if isinstance(x, dict)]
            post_rows = self._filter_by_score(raw_rows, score_threshold)
            raw_stats = self._score_stats(raw_rows)
            prompt_item = {
                "prompt": prompt,
                "canonical_label": canonical_label,
                "raw_records": raw_rows,
                "post_threshold_records": post_rows,
                "raw_count": len(raw_rows),
                "post_threshold_count": len(post_rows),
                "score_min": float(raw_stats.get("min", 0.0)),
                "score_max": float(raw_stats.get("max", 0.0)),
                "cache_hit": bool((prompt_debug.get(prompt) or {}).get("cache_hit", False)),
                "command": dict((prompt_debug.get(prompt) or {}).get("command") or {}),
            }
            prompt_results.append(prompt_item)
            for rec in post_rows:
                row = dict(rec)
                row["canonical_label"] = canonical_label
                row["prompt_used"] = str(row.get("prompt_used", "") or prompt)
                flat_post_threshold.append(row)
        return {
            "image_path": image_path,
            "provider": provider,
            "score_threshold": score_threshold,
            "prompt_results": prompt_results,
            "post_threshold_records": flat_post_threshold,
            "backend_config": {
                "provider": self.cfg.provider,
                "cache_dir": self.cfg.cache_dir,
                "disable_cache": bool(self.cfg.disable_cache),
                "external_command_args_file": self.cfg.external_command_args_file,
                "external_batch_command_args_file": self.cfg.external_batch_command_args_file,
                "external_timeout_sec": int(self.cfg.external_timeout_sec),
                "external_use_persistent_process": bool(self.cfg.external_use_persistent_process),
            },
        }

    def discover_entities_by_category(self, image_path: str, prompts: List[Dict[str, str]]) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        provider = str(self.cfg.provider or "mock").strip().lower()
        if provider == "external_command" and self._has_batch_command():
            unique_items: List[Dict[str, str]] = []
            seen_prompt_keys = set()
            for item in prompts:
                canonical_label = str(item.get("canonical_label", "")).strip().lower()
                prompt = str(item.get("prompt", "")).strip()
                key = (canonical_label, prompt.lower())
                if not canonical_label or not prompt or key in seen_prompt_keys:
                    continue
                seen_prompt_keys.add(key)
                unique_items.append({"prompt": prompt, "canonical_label": canonical_label})
            prompt_map = self._run_prompt_batch(image_path, unique_items, stage="category_discovery")
            for item in unique_items:
                prompt = str(item.get("prompt", "")).strip()
                canonical_label = str(item.get("canonical_label", "")).strip().lower()
                for rec in prompt_map.get(prompt, []):
                    row = dict(rec)
                    row["canonical_label"] = canonical_label
                    row["prompt_used"] = str(row.get("prompt_used", "") or prompt)
                    out.append(row)
            return out

        for item in prompts:
            prompt = str(item.get("prompt", "")).strip()
            canonical_label = str(item.get("canonical_label", "")).strip().lower()
            if not prompt or not canonical_label:
                continue
            res = self._run_prompt(image_path, prompt, stage="category_discovery")
            for rec in res:
                row = dict(rec)
                row["canonical_label"] = canonical_label
                out.append(row)
        return out

    def refine_with_sentence_prompts(
        self,
        image_path: str,
        sentence_prompts: List[str],
        enable_two_stage_refinement: Optional[bool] = None,
    ) -> List[Dict[str, object]]:
        enabled = self.cfg.enable_two_stage_refinement if enable_two_stage_refinement is None else bool(enable_two_stage_refinement)
        if not enabled:
            return []
        out: List[Dict[str, object]] = []
        provider = str(self.cfg.provider or "mock").strip().lower()
        if provider == "external_command" and self._has_batch_command():
            unique_prompts: List[Dict[str, str]] = []
            seen_prompts = set()
            for sent in sentence_prompts:
                prompt = str(sent or "").strip()
                if not prompt or prompt in seen_prompts:
                    continue
                seen_prompts.add(prompt)
                unique_prompts.append({"prompt": prompt})
            prompt_map = self._run_prompt_batch(image_path, unique_prompts, stage="sentence_refine")
            for item in unique_prompts:
                out.extend(prompt_map.get(str(item.get("prompt", "")).strip(), []))
            return out

        for sent in sentence_prompts:
            prompt = str(sent or "").strip()
            if not prompt:
                continue
            out.extend(self._run_prompt(image_path, prompt, stage="sentence_refine"))
        return out
