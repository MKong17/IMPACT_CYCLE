from __future__ import annotations

import copy
import contextlib
import glob
import importlib.util
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from tools.east.online_ieast_adapter import OnlineInteractiveAdapter
except Exception:  # pragma: no cover
    OnlineInteractiveAdapter = None


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")


def _log(progress_cb: Optional[Callable[[str], None]], msg: str) -> None:
    if callable(progress_cb):
        try:
            progress_cb(str(msg))
            return
        except Exception:
            pass
    print(msg)


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "out of memory" not in msg and "cudaerrormemoryallocation" not in msg:
        return False
    if "cuda" in msg:
        return True
    return "acceleratorerror" in exc.__class__.__name__.lower()


def _cleanup_cuda_memory() -> None:
    if torch is None or (not torch.cuda.is_available()):
        return
    with contextlib.suppress(Exception):
        torch.cuda.synchronize()
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        torch.cuda.ipc_collect()


def resolve_east_repo_root() -> str:
    env_root = os.environ.get("EAST_REPO_ROOT", "").strip()
    if env_root:
        root = os.path.abspath(os.path.expanduser(env_root))
        if os.path.isdir(root):
            return root

    this_dir = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    vendored = os.path.abspath(os.path.join(repo_root, "external", "EAST-main"))
    if os.path.isdir(vendored):
        return vendored
    sibling = os.path.abspath(os.path.join(repo_root, "..", "EAST-main"))
    if os.path.isdir(sibling):
        return sibling
    return ""


def _ensure_east_importable(east_root: str) -> None:
    if not east_root or not os.path.isdir(east_root):
        raise FileNotFoundError("EAST repo root not found. Set EAST_REPO_ROOT.")
    opentad_dir = os.path.join(east_root, "opentad")
    if not os.path.isdir(opentad_dir):
        raise FileNotFoundError(f"Invalid EAST repo root: missing {opentad_dir}")
    if east_root not in sys.path:
        sys.path.insert(0, east_root)
    east_root_abs = os.path.abspath(east_root)
    # Purge stale environment installs before importing vendored EAST/OpenTAD.
    for mod_name, mod in list(sys.modules.items()):
        if not (mod_name == "opentad" or mod_name.startswith("opentad.") or mod_name == "nms_1d_cpu"):
            continue
        mod_file = str(getattr(mod, "__file__", "") or "")
        if mod_file and os.path.abspath(mod_file).startswith(east_root_abs):
            continue
        sys.modules.pop(mod_name, None)
    # Force the vendored EAST/OpenTAD nms module so we never hit a stale
    # site-packages installation.
    shim_candidates = [
        os.path.join(east_root_abs, "nms_1d_cpu.py"),
        os.path.join(
            east_root_abs,
            "opentad",
            "models",
            "utils",
            "post_processing",
            "nms",
            "nms_1d_cpu.py",
        ),
    ]
    shim_candidates.extend(
        sorted(
            glob.glob(
                os.path.join(
                    east_root_abs,
                    "opentad",
                    "models",
                    "utils",
                    "post_processing",
                    "nms",
                    "build",
                    "**",
                    "nms_1d_cpu*.so",
                ),
                recursive=True,
            )
        )
    )
    shim_candidates.extend(
        sorted(
            glob.glob(
                os.path.join(
                    east_root_abs,
                    "opentad",
                    "models",
                    "utils",
                    "post_processing",
                    "nms",
                    "build",
                    "**",
                    "nms_1d_cpu*.pyd",
                ),
                recursive=True,
            )
        )
    )
    existing_candidates = [
        os.path.abspath(candidate)
        for candidate in shim_candidates
        if os.path.isfile(os.path.abspath(candidate))
    ]
    if not existing_candidates:
        raise FileNotFoundError(
            "Vendored nms shim not found in any known location: "
            + ", ".join(shim_candidates)
        )
    last_error = None
    for shim_path in existing_candidates:
        try:
            spec = importlib.util.spec_from_file_location("nms_1d_cpu", shim_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Failed to create import spec for {shim_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules["nms_1d_cpu"] = module
            spec.loader.exec_module(module)
            return
        except Exception as exc:
            last_error = exc
            sys.modules.pop("nms_1d_cpu", None)
    raise RuntimeError(
        "Failed to load vendored nms shim from known locations: "
        + ", ".join(existing_candidates)
        + (f" (last error: {last_error})" if last_error else "")
    )


def _probe_video_meta(video_path: str) -> Dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0.0:
        fps = 30.0
    duration = float(frames) / float(fps) if frames > 0 else 0.0
    return {"fps": fps, "total_frames": float(frames), "duration": duration}


def _sanitize_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for k, v in (state_dict or {}).items():
        if not hasattr(v, "shape"):
            continue
        nk = str(k)
        if nk.startswith("module."):
            nk = nk[7:]
        clean[nk] = v
    return clean


def _find_interactive_adapter_prefixes(state_dict: Dict[str, Any]) -> List[str]:
    if OnlineInteractiveAdapter is not None and hasattr(OnlineInteractiveAdapter, "find_vit_interactive_prefixes"):
        try:
            return [str(x) for x in OnlineInteractiveAdapter.find_vit_interactive_prefixes(state_dict) if str(x)]
        except Exception:
            pass
    out: List[str] = []
    marker = ".interactive_adapter."
    for k in (state_dict or {}).keys():
        key = str(k)
        if marker not in key:
            continue
        out.append(key.split(marker, 1)[0] + ".interactive_adapter")
    return sorted(set(out))


def _find_backbone_adapter_prefixes(state_dict: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    marker = ".adapter."
    for k in (state_dict or {}).keys():
        key = str(k)
        if marker not in key:
            continue
        out.append(key.split(marker, 1)[0] + ".adapter")
    return sorted(set(out))


def _load_state_dict(ckpt_path: str) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for EAST inference.")
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        if isinstance(obj.get("state_dict_ema"), dict):
            return _sanitize_state_dict(obj.get("state_dict_ema") or {})
        if isinstance(obj.get("state_dict"), dict):
            return _sanitize_state_dict(obj.get("state_dict") or {})
        if isinstance(obj.get("model"), dict):
            return _sanitize_state_dict(obj.get("model") or {})
        return _sanitize_state_dict(obj)
    raise RuntimeError(f"Unsupported checkpoint format: {type(obj)}")


def _load_vit_interactive_delta_payload(
    delta_source: Any,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    if isinstance(delta_source, dict):
        return delta_source
    path = str(delta_source or "").strip()
    if not path:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        return None
    try:
        delta_obj = torch.load(path, map_location="cpu")
    except Exception as exc:
        _log(progress_cb, f"[EAST][WARN] failed to load vit delta: {exc}")
        return None
    if not isinstance(delta_obj, dict):
        return None
    delta_obj.setdefault("_source_path", path)
    return delta_obj


def _merge_vit_interactive_delta(
    base_state: Dict[str, Any],
    delta_source: Any,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Overlay interactive-adapter weights onto EAST checkpoint state."""
    if torch is None:
        return base_state
    delta_obj = _load_vit_interactive_delta_payload(delta_source, progress_cb=progress_cb)
    if not isinstance(delta_obj, dict):
        return base_state
    state = dict(delta_obj.get("state") or {})
    if not state:
        return base_state
    requested_prefixes = [str(x) for x in (delta_obj.get("target_prefixes") or []) if str(x)]
    available_prefixes = _find_interactive_adapter_prefixes(base_state)
    available_set = set(available_prefixes)
    matched_requested = [p for p in requested_prefixes if p in available_set]
    fallback_used = False
    if matched_requested:
        prefixes = matched_requested
    else:
        prefixes = list(available_prefixes)
        fallback_used = bool(requested_prefixes) or (not requested_prefixes)
    if not prefixes:
        backbone_prefixes = _find_backbone_adapter_prefixes(base_state)
        if backbone_prefixes:
            _log(
                progress_cb,
                "[EAST][WARN] current EAST checkpoint exposes backbone `.adapter` modules "
                f"({len(backbone_prefixes)} prefix(es)) but no `.interactive_adapter` modules; "
                "vit delta will not be applied to official backbone adapters.",
            )
            return base_state
        _log(
            progress_cb,
            "[EAST][WARN] vit delta has no target interactive-adapter prefixes "
            f"(requested={len(requested_prefixes)}, available={len(available_prefixes)}).",
        )
        return base_state

    merged = dict(base_state)
    updated = 0
    matched_prefixes: List[str] = []
    shape_mismatch = 0
    for prefix in prefixes:
        prefix = str(prefix).strip().rstrip(".")
        if not prefix:
            continue
        local_updates = 0
        for k, v in state.items():
            full_key = f"{prefix}.{str(k)}"
            if full_key not in merged:
                continue
            old = merged[full_key]
            if hasattr(old, "shape") and hasattr(v, "shape") and tuple(old.shape) != tuple(v.shape):
                shape_mismatch += 1
                continue
            merged[full_key] = v
            updated += 1
            local_updates += 1
        if local_updates > 0:
            matched_prefixes.append(prefix)
    source_name = str(delta_obj.get("name") or "")
    source_path = str(delta_obj.get("_source_path") or "")
    if not source_name:
        source_name = os.path.basename(source_path) if source_path else "inline_payload"
    unmatched_requested = [p for p in requested_prefixes if p not in available_set]
    if fallback_used and requested_prefixes and prefixes:
        _log(
            progress_cb,
            "[EAST][WARN] vit delta requested prefixes were stale; "
            f"falling back to {len(prefixes)} available prefix(es).",
        )
    if unmatched_requested:
        preview = ", ".join(unmatched_requested[:4])
        if len(unmatched_requested) > 4:
            preview += ", ..."
        _log(
            progress_cb,
            f"[EAST][WARN] vit delta unmatched requested prefixes: {preview}",
        )
    if shape_mismatch > 0:
        _log(progress_cb, f"[EAST][WARN] vit delta skipped {shape_mismatch} tensor(s) due to shape mismatch.")
    if updated <= 0 and shape_mismatch > 0:
        _log(
            progress_cb,
            "[EAST][WARN] vit delta did not update detector adapters; "
            f"delta_input_dim={int(delta_obj.get('input_dim', 0) or 0)}, "
            f"state_keys={int(delta_obj.get('state_key_count', len(state)) or len(state))}.",
        )
    _log(
        progress_cb,
        "[EAST] Applied vit interactive delta: "
        f"tensors={updated}, matched_prefixes={len(matched_prefixes)}/{len(prefixes)}, source={source_name}",
    )
    return merged


def _merge_vit_interactive_deltas(
    base_state: Dict[str, Any],
    delta_sources: Optional[Sequence[Any]],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    merged = dict(base_state)
    for delta_source in list(delta_sources or []):
        merged = _merge_vit_interactive_delta(merged, delta_source, progress_cb=progress_cb)
    return merged


def _patch_prepare_video_info(pipeline: List[Dict[str, Any]], video_ext: str) -> List[Dict[str, Any]]:
    required_meta_keys = (
        "video_name",
        "data_path",
        "fps",
        "duration",
        "snippet_stride",
        "window_start_frame",
        "resize_length",
        "window_size",
        "offset_frames",
        "total_frames",
        "avg_fps",
        "frame_inds",
        "sample_stride",
        "feature_stride",
        "feature_start_idx",
        "feature_end_idx",
    )
    patched: List[Dict[str, Any]] = []
    for step in pipeline:
        if not isinstance(step, dict):
            patched.append(step)
            continue
        item = copy.deepcopy(step)
        t = str(item.get("type", ""))
        if t.endswith("PrepareVideoInfo") or t == "PrepareVideoInfo":
            item["prefix"] = ""
            item["format"] = str(video_ext).lstrip(".")
        if t.endswith("Collect") or t == "Collect":
            # Preserve temporal metadata even if the cfg overrides Collect.meta_keys.
            raw_meta_keys = item.get("meta_keys")
            if isinstance(raw_meta_keys, (list, tuple)):
                meta_keys = [str(key) for key in raw_meta_keys if str(key)]
            else:
                meta_keys = []
            for key in required_meta_keys:
                if key not in meta_keys:
                    meta_keys.append(key)
            item["meta_keys"] = meta_keys
        patched.append(item)
    return patched


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _sample_meta(sample: Dict[str, Any]) -> Dict[str, Any]:
    metas = sample.get("metas")
    if isinstance(metas, dict):
        return metas
    return {}


def _resolve_total_frames(sample: Dict[str, Any], vmeta_total_frames: int) -> int:
    metas = _sample_meta(sample)
    total = _as_int(metas.get("total_frames"), 0)
    if total > 0:
        return total
    return max(0, int(vmeta_total_frames))


def _resolve_avg_fps(sample: Dict[str, Any], vmeta_fps: float) -> float:
    metas = _sample_meta(sample)
    fps = _as_float(metas.get("avg_fps"), 0.0)
    if fps > 1e-6:
        return fps
    return max(0.0, float(vmeta_fps))


def _extract_frame_inds(sample: Dict[str, Any]) -> List[int]:
    metas = _sample_meta(sample)
    raw = metas.get("frame_inds")
    if raw is None:
        return []
    try:
        arr = np.asarray(raw).reshape(-1)
        out = [int(x) for x in arr.tolist()]
        return out
    except Exception:
        return []


def _as_cfg_int(cfg_obj: Any, key: str, default: int) -> int:
    try:
        if hasattr(cfg_obj, key):
            return int(getattr(cfg_obj, key))
    except Exception:
        pass
    try:
        if isinstance(cfg_obj, dict) and key in cfg_obj:
            return int(cfg_obj.get(key))
    except Exception:
        pass
    return int(default)


def _as_cfg_float(cfg_obj: Any, key: str, default: float) -> float:
    try:
        if hasattr(cfg_obj, key):
            return float(getattr(cfg_obj, key))
    except Exception:
        pass
    try:
        if isinstance(cfg_obj, dict) and key in cfg_obj:
            return float(cfg_obj.get(key))
    except Exception:
        pass
    return float(default)


def _detect_load_frames_method(pipeline_cfg: Sequence[Dict[str, Any]]) -> str:
    for step in list(pipeline_cfg or []):
        if not isinstance(step, dict):
            continue
        t = str(step.get("type", ""))
        if t.endswith("LoadFrames") or t == "LoadFrames":
            return str(step.get("method", "")).strip().lower()
    return ""


def _build_sliding_window_spans(test_cfg: Any, total_frames: int) -> Dict[str, Any]:
    sample_stride = max(1, _as_cfg_int(test_cfg, "sample_stride", 1))
    feature_stride = max(1, _as_cfg_int(test_cfg, "feature_stride", 1))
    snippet_stride = max(1, int(sample_stride * feature_stride))
    window_size = max(1, _as_cfg_int(test_cfg, "window_size", 1))
    overlap_ratio = _as_cfg_float(test_cfg, "window_overlap_ratio", 0.0)
    env_overlap = str(os.environ.get("EAST_FEAT_WINDOW_OVERLAP", "") or "").strip()
    if env_overlap:
        try:
            overlap_ratio = float(env_overlap)
        except Exception:
            pass
    overlap_ratio = max(0.0, min(0.95, float(overlap_ratio)))
    window_stride = max(1, int(window_size * (1.0 - overlap_ratio)))

    centers = np.arange(0, max(0, int(total_frames)), int(snippet_stride))
    snippet_num = int(centers.size)

    spans: List[tuple[int, int]] = []
    if snippet_num <= 0:
        spans = [(0, 0)]
    else:
        last_window = False
        for idx in range(max(1, snippet_num // window_stride)):
            start_idx = int(idx * window_stride)
            end_idx = int(start_idx + window_size)
            if end_idx > snippet_num:
                end_idx = int(snippet_num)
                start_idx = int(max(0, end_idx - window_size))
                last_window = True
            if not spans or spans[-1] != (start_idx, end_idx):
                spans.append((start_idx, end_idx))
            if last_window:
                break

    return {
        "sample_stride": int(sample_stride),
        "feature_stride": int(feature_stride),
        "snippet_stride": int(snippet_stride),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "window_overlap_ratio": float(overlap_ratio),
        "snippet_centers": centers,
        "spans": spans,
    }


def _inject_window_sampling_hints(item: Dict[str, Any], test_cfg: Any, total_frames: int) -> None:
    """
    Fill dataset-style temporal keys required by `LoadFrames(method=sliding_window/random_trunc)`
    when running a single-video inference path (outside dataset __getitem__).
    """
    sample_stride = max(1, _as_cfg_int(test_cfg, "sample_stride", _as_int(item.get("sample_stride"), 1)))
    feature_stride = max(1, _as_cfg_int(test_cfg, "feature_stride", _as_int(item.get("feature_stride"), 1)))
    snippet_stride = max(1, int(sample_stride * feature_stride))
    item["sample_stride"] = int(sample_stride)
    item["feature_stride"] = int(feature_stride)
    item["snippet_stride"] = int(snippet_stride)

    # Emulate one valid window for sliding/padding style pipelines.
    # NOTE: single-video adapter currently executes one pipeline/sample only.
    centers = np.arange(0, max(0, int(total_frames)), int(snippet_stride))
    window_size = max(1, _as_cfg_int(test_cfg, "window_size", _as_int(item.get("window_size"), 1)))
    if centers.size > 0:
        start_idx = 0
        end_idx = min(int(centers.size - 1), int(start_idx + window_size - 1))
        item["feature_start_idx"] = int(start_idx)
        item["feature_end_idx"] = int(end_idx)
        item["window_start_frame"] = int(centers[start_idx])
    else:
        item["feature_start_idx"] = 0
        item["feature_end_idx"] = 0
        item["window_start_frame"] = 0


def _build_infer_metas(
    sample: Dict[str, Any],
    item: Dict[str, Any],
    test_cfg: Any,
    vmeta: Dict[str, float],
) -> Dict[str, Any]:
    metas = dict(_sample_meta(sample))

    sampled_total_frames = _resolve_total_frames(sample, int(vmeta["total_frames"]))
    sampled_fps = _resolve_avg_fps(sample, float(vmeta["fps"]))

    sample_stride = max(
        1,
        _as_int(
            metas.get("sample_stride"),
            _as_cfg_int(test_cfg, "sample_stride", _as_int(item.get("sample_stride"), 1)),
        ),
    )
    feature_stride = max(
        1,
        _as_int(
            metas.get("feature_stride"),
            _as_cfg_int(test_cfg, "feature_stride", _as_int(item.get("feature_stride"), 1)),
        ),
    )
    snippet_stride = max(
        1,
        _as_int(
            metas.get("snippet_stride"),
            _as_int(item.get("snippet_stride"), sample_stride * feature_stride),
        ),
    )

    metas["video_name"] = str(metas.get("video_name") or item.get("video_name") or "")
    metas["duration"] = float(
        _as_float(metas.get("duration"), _as_float(item.get("duration"), float(vmeta["duration"])))
    )
    metas["sample_stride"] = int(sample_stride)
    metas["feature_stride"] = int(feature_stride)
    metas["snippet_stride"] = int(snippet_stride)
    metas["window_start_frame"] = int(_as_int(metas.get("window_start_frame"), _as_int(item.get("window_start_frame"), 0)))
    metas["offset_frames"] = int(
        _as_int(
            metas.get("offset_frames"),
            _as_cfg_int(test_cfg, "offset_frames", _as_int(item.get("offset_frames"), 0)),
        )
    )

    window_size = _as_int(metas.get("window_size"), _as_int(item.get("window_size"), 0))
    if window_size > 0:
        metas["window_size"] = int(window_size)

    resize_length = _as_int(metas.get("resize_length"), _as_int(item.get("resize_length"), 0))
    if resize_length > 0:
        metas["resize_length"] = int(resize_length)
        metas["fps"] = -1
    else:
        fps_value = _as_float(metas.get("fps"), 0.0)
        if fps_value <= 0.0:
            fps_value = _as_float(item.get("fps"), sampled_fps if sampled_fps > 1e-6 else float(vmeta["fps"]))
        metas["fps"] = float(fps_value)

    feature_start_idx = _as_int(metas.get("feature_start_idx"), _as_int(item.get("feature_start_idx"), -1))
    if feature_start_idx >= 0:
        metas["feature_start_idx"] = int(feature_start_idx)
    feature_end_idx = _as_int(metas.get("feature_end_idx"), _as_int(item.get("feature_end_idx"), -1))
    if feature_end_idx >= 0:
        metas["feature_end_idx"] = int(feature_end_idx)

    if sampled_total_frames > 0:
        metas["total_frames"] = int(_as_int(metas.get("total_frames"), sampled_total_frames))
    if sampled_fps > 1e-6:
        metas["avg_fps"] = float(_as_float(metas.get("avg_fps"), sampled_fps))

    if "frame_inds" not in metas:
        frame_inds = _extract_frame_inds(sample)
        if frame_inds:
            metas["frame_inds"] = frame_inds

    return metas


def _prepare_cfg_pretrain_path(
    cfg: Any,
    cfg_path: str,
    east_root: str,
    ckpt_path: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Normalize `cfg.model.backbone.custom.pretrain` path.
    - Resolve relative pretrain paths against EAST repo/config dir.
    - If unresolved but a runtime checkpoint is provided, disable pretrain to avoid hard failure.
    """
    try:
        pretrain_raw = cfg.model.backbone.custom.pretrain
    except Exception:
        return

    # In inference/extraction, a full EAST checkpoint is provided separately.
    # Default to ckpt-only loading to avoid noisy/partial backbone-pretrain mismatch logs.
    keep_cfg_pretrain = str(os.environ.get("EAST_KEEP_CFG_PRETRAIN", "") or "").strip().lower()
    if ckpt_path and os.path.isfile(ckpt_path) and keep_cfg_pretrain not in ("1", "true", "yes", "on"):
        try:
            cfg.model.backbone.custom.pretrain = None
        except Exception:
            pass
        # Suppress misleading "randomly initialized" warning from backbone wrapper in ckpt-only inference path.
        os.environ["EAST_SUPPRESS_PRETRAIN_WARNING"] = "1"
        _log(progress_cb, "[EAST] Skip cfg backbone pretrain; EAST_CKPT will be loaded right after model build.")
        return

    pretrain = str(pretrain_raw or "").strip()
    if not pretrain:
        return

    cand_abs = os.path.abspath(os.path.expanduser(pretrain))
    if os.path.isfile(cand_abs):
        try:
            cfg.model.backbone.custom.pretrain = cand_abs
        except Exception:
            pass
        return

    candidates: List[str] = []
    if east_root:
        candidates.append(os.path.abspath(os.path.join(east_root, pretrain)))
    cfg_dir = os.path.dirname(os.path.abspath(cfg_path))
    candidates.append(os.path.abspath(os.path.join(cfg_dir, pretrain)))
    for cand in candidates:
        if os.path.isfile(cand):
            try:
                cfg.model.backbone.custom.pretrain = cand
            except Exception:
                pass
            _log(progress_cb, f"[EAST] Resolved backbone pretrain: {cand}")
            return

    # For inference with an explicit checkpoint, missing backbone pretrain should not be fatal.
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            cfg.model.backbone.custom.pretrain = None
        except Exception:
            pass
        _log(progress_cb, f"[EAST][WARN] Missing cfg pretrain '{pretrain}', fallback to EAST_CKPT only.")


def _validate_ckpt_load_result(
    missing: Sequence[str],
    unexpected: Sequence[str],
    required_prefixes: Sequence[str],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> None:
    miss = [str(x) for x in (missing or [])]
    unexp = [str(x) for x in (unexpected or [])]
    _log(progress_cb, f"[EAST] EAST_CKPT load summary: missing={len(miss)}, unexpected={len(unexp)}")
    if not required_prefixes:
        return

    prefixes = tuple(str(p) for p in required_prefixes if str(p))
    critical = [k for k in miss if k.startswith(prefixes)]
    if critical:
        preview = ", ".join(critical[:6])
        if len(critical) > 6:
            preview += ", ..."
        raise RuntimeError(
            "EAST_CKPT did not load critical modules (backbone/projection/neck). "
            f"missing={len(critical)} keys: {preview}"
        )


def _picked_indices_from_sampling(sample: Dict[str, Any], T: int, total_frames: int) -> List[int]:
    if T <= 0:
        return []

    frame_inds = _extract_frame_inds(sample)
    if frame_inds:
        if len(frame_inds) == T:
            picked = frame_inds
        elif len(frame_inds) == 1:
            picked = [int(frame_inds[0])] * T
        else:
            picked = [frame_inds[int(round(i * (len(frame_inds) - 1) / float(max(1, T - 1))))] for i in range(T)]
        if total_frames > 0:
            picked = [max(0, min(total_frames - 1, int(p))) for p in picked]
        else:
            picked = [max(0, int(p)) for p in picked]
        return picked

    if total_frames <= 0:
        return list(range(T))
    if T == 1 or total_frames == 1:
        return [0] * T
    picked = [int(round(i * (total_frames - 1) / float(T - 1))) for i in range(T)]
    return [max(0, min(total_frames - 1, p)) for p in picked]


def infer_east_detections(
    video_path: str,
    cfg_path: str,
    ckpt_path: str,
    label_bank: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    max_segments: int = 400,
    adapter_delta_path: Optional[str] = None,
    adapter_delta_sources: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for EAST inference.")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"EAST cfg not found: {cfg_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"EAST checkpoint not found: {ckpt_path}")

    east_root = resolve_east_repo_root()
    _ensure_east_importable(east_root)

    # Imported after path injection.
    from mmengine.config import Config  # type: ignore
    from mmengine.dataset import Compose  # type: ignore
    import opentad.datasets  # noqa: F401
    import opentad.models.detectors  # noqa: F401
    import opentad.models.backbones  # noqa: F401
    import opentad.models.projections  # noqa: F401
    import opentad.models.necks  # noqa: F401
    import opentad.models.dense_heads  # noqa: F401
    import opentad.models.losses  # noqa: F401
    from opentad.models.builder import build_detector  # type: ignore

    cfg = Config.fromfile(cfg_path)
    _prepare_cfg_pretrain_path(
        cfg=cfg,
        cfg_path=cfg_path,
        east_root=east_root,
        ckpt_path=ckpt_path,
        progress_cb=progress_cb,
    )
    test_cfg = cfg.dataset.test
    pipeline_cfg = _patch_prepare_video_info(
        copy.deepcopy(list(test_cfg.pipeline)),
        os.path.splitext(video_path)[1].lstrip("."),
    )
    pipeline = Compose(pipeline_cfg)

    base = os.path.splitext(os.path.basename(video_path))[0]
    vmeta = _probe_video_meta(video_path)

    item: Dict[str, Any] = {
        "video_name": base,
        "data_path": os.path.dirname(video_path),
        "duration": float(vmeta["duration"]),
    }
    if "resize_length" in test_cfg:
        item["resize_length"] = int(test_cfg.resize_length)
        item["fps"] = -1
    else:
        item["fps"] = float(vmeta["fps"])
    if "window_size" in test_cfg:
        item["window_size"] = int(test_cfg.window_size)
    if "sample_stride" in test_cfg:
        item["sample_stride"] = int(test_cfg.sample_stride)
    if "feature_stride" in test_cfg:
        item["feature_stride"] = int(test_cfg.feature_stride)
    if "offset_frames" in test_cfg:
        item["offset_frames"] = int(test_cfg.offset_frames)
    _inject_window_sampling_hints(item, test_cfg, total_frames=int(vmeta["total_frames"]))

    _log(progress_cb, f"[EAST] Compose test pipeline for {base}")
    sample = pipeline(item)
    inputs = sample.get("inputs")
    if inputs is None:
        raise RuntimeError("EAST pipeline did not return 'inputs'.")
    if inputs.ndim == 5:
        inputs = inputs.unsqueeze(0)  # [1, N, C, T, H, W]
    elif inputs.ndim != 6:
        raise RuntimeError(f"Unexpected EAST input shape: {tuple(inputs.shape)}")

    masks = sample.get("masks")
    if masks is None:
        masks = torch.ones((inputs.shape[0], int(inputs.shape[3])), dtype=torch.bool)
    elif masks.ndim == 1:
        masks = masks.unsqueeze(0)

    metas = _build_infer_metas(sample=sample, item=item, test_cfg=test_cfg, vmeta=vmeta)

    sampled_total_frames = _resolve_total_frames(sample, int(vmeta["total_frames"]))
    sampled_fps = _resolve_avg_fps(sample, float(vmeta["fps"]))
    if sampled_total_frames > 0 and int(vmeta["total_frames"]) > 0:
        if abs(int(sampled_total_frames) - int(vmeta["total_frames"])) > 5:
            _log(
                progress_cb,
                f"[EAST][WARN] frame count mismatch: pipeline={sampled_total_frames}, cv2={int(vmeta['total_frames'])}",
            )

    state_dict = _load_state_dict(ckpt_path)
    delta_sources: List[Any] = []
    if adapter_delta_path:
        delta_sources.append(adapter_delta_path)
    delta_sources.extend(list(adapter_delta_sources or []))
    state_dict = _merge_vit_interactive_deltas(state_dict, delta_sources, progress_cb=progress_cb)
    probe_model = build_detector(copy.deepcopy(cfg.model))
    missing, unexpected = probe_model.load_state_dict(state_dict, strict=False)
    _validate_ckpt_load_result(
        missing=missing,
        unexpected=unexpected,
        required_prefixes=("backbone.", "projection.", "neck.", "rpn_head."),
        progress_cb=progress_cb,
    )

    infer_cfg = copy.deepcopy(cfg.inference) if hasattr(cfg, "inference") else dict()
    infer_cfg.load_from_raw_predictions = False
    infer_cfg.save_raw_prediction = False

    post_cfg = copy.deepcopy(cfg.post_processing) if hasattr(cfg, "post_processing") else dict()
    post_cfg.sliding_window = False
    post_cfg.nms = None  # Avoid dependency on external nms extension.

    labels = [str(x).strip() for x in (label_bank or []) if str(x).strip()]
    num_classes = int(getattr(getattr(probe_model, "rpn_head", None), "num_classes", 1))
    if num_classes <= 1:
        ext_cls = [labels[0] if labels else "action"]
    else:
        if len(labels) >= num_classes:
            ext_cls = labels[:num_classes]
        else:
            ext_cls = [f"cls_{i}" for i in range(num_classes)]
    del probe_model

    def _collect_rows(outputs_obj: Any) -> List[Dict[str, Any]]:
        rows_local: List[Dict[str, Any]] = []
        if isinstance(outputs_obj, dict):
            preds = outputs_obj.get(base)
            if preds is None and len(outputs_obj) == 1:
                preds = next(iter(outputs_obj.values()))
            if isinstance(preds, list):
                for r in preds:
                    if not isinstance(r, dict):
                        continue
                    seg = r.get("segment")
                    if not isinstance(seg, (list, tuple)) or len(seg) != 2:
                        continue
                    try:
                        s = float(seg[0])
                        e = float(seg[1])
                        sc = float(r.get("score", 0.0))
                    except Exception:
                        continue
                    if e <= s:
                        continue
                    rows_local.append(
                        {
                            "t_start": s,
                            "t_end": e,
                            "score": sc,
                            "label": str(r.get("label", "action")),
                        }
                    )
        rows_local.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        if max_segments > 0:
            rows_local = rows_local[: int(max_segments)]
        return rows_local

    def _run_forward(target_device: torch.device) -> List[Dict[str, Any]]:
        model = build_detector(copy.deepcopy(cfg.model))
        missing_local, unexpected_local = model.load_state_dict(state_dict, strict=False)
        _validate_ckpt_load_result(
            missing=missing_local,
            unexpected=unexpected_local,
            required_prefixes=("backbone.", "projection.", "neck.", "rpn_head."),
            progress_cb=progress_cb,
        )
        model = model.to(target_device)
        model.eval()
        inputs_dev = inputs.to(
            target_device,
            non_blocking=(target_device.type == "cuda"),
        )
        masks_dev = masks.to(
            target_device,
            non_blocking=(target_device.type == "cuda"),
        )
        fp16_enabled = bool(
            target_device.type == "cuda"
            and _env_flag("EAST_DETECTOR_FP16", True)
        )
        mode_txt = f" on {str(target_device).upper()}"
        if fp16_enabled:
            mode_txt += " with fp16 autocast"
        _log(progress_cb, f"[EAST] Running detector forward{mode_txt}...")
        try:
            with torch.no_grad():
                ctx = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if fp16_enabled
                    else contextlib.nullcontext()
                )
                with ctx:
                    outputs = model(
                        inputs=inputs_dev,
                        masks=masks_dev,
                        metas=[metas],
                        return_loss=False,
                        infer_cfg=infer_cfg,
                        post_cfg=post_cfg,
                        ext_cls=ext_cls,
                    )
            return _collect_rows(outputs)
        finally:
            del model
            del inputs_dev
            del masks_dev
            if target_device.type == "cuda":
                _cleanup_cuda_memory()

    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    allow_cpu_fallback = _env_flag("EAST_DETECTOR_ALLOW_CPU_FALLBACK", True)
    try:
        rows = _run_forward(run_device)
    except Exception as exc:
        if run_device.type == "cuda" and allow_cpu_fallback and _is_cuda_oom(exc):
            _log(progress_cb, f"[EAST][WARN] Detector forward OOM on CUDA: {exc}")
            _log(progress_cb, "[EAST] Retrying detector inference on CPU...")
            _cleanup_cuda_memory()
            rows = _run_forward(torch.device("cpu"))
            _log(progress_cb, "[EAST] Detector CPU fallback succeeded.")
        else:
            raise

    return {
        "video_name": base,
        "duration": float(vmeta["duration"]),
        "fps": float(sampled_fps if sampled_fps > 1e-6 else vmeta["fps"]),
        "total_frames": int(sampled_total_frames if sampled_total_frames > 0 else vmeta["total_frames"]),
        "detections": rows,
        "backend": "east_detector",
    }


def extract_east_backbone_features(
    video_path: str,
    cfg_path: str,
    ckpt_path: str,
    frame_stride: Optional[int] = None,
    device: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    adapter_delta_path: Optional[str] = None,
    adapter_delta_sources: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """
    Extract temporal features directly from EAST model path (backbone/projection/neck).

    Returns:
      {
        "features": np.ndarray [T, D],
        "meta": {
          "backbone": str,
          "feature_dim": int,
          "frame_stride": int,
          "picked_indices": List[int],
          "num_frames": int,
          "input_size": int,
          "fps": float,
          "duration": float,
          "video_path": str,
          "source": "east_backbone",
          "east_cfg": str,
          "east_ckpt": str,
        }
      }
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for EAST feature extraction.")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"EAST cfg not found: {cfg_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"EAST checkpoint not found: {ckpt_path}")

    east_root = resolve_east_repo_root()
    _ensure_east_importable(east_root)

    from mmengine.config import Config  # type: ignore
    from mmengine.dataset import Compose  # type: ignore
    import opentad.datasets  # noqa: F401
    import opentad.models.detectors  # noqa: F401
    import opentad.models.backbones  # noqa: F401
    import opentad.models.projections  # noqa: F401
    import opentad.models.necks  # noqa: F401
    import opentad.models.dense_heads  # noqa: F401
    import opentad.models.losses  # noqa: F401
    from opentad.models.builder import build_detector  # type: ignore

    cfg = Config.fromfile(cfg_path)
    _prepare_cfg_pretrain_path(
        cfg=cfg,
        cfg_path=cfg_path,
        east_root=east_root,
        ckpt_path=ckpt_path,
        progress_cb=progress_cb,
    )
    test_cfg = cfg.dataset.test
    requested_frame_stride = None
    if frame_stride is not None:
        try:
            requested_frame_stride = max(1, int(frame_stride))
        except Exception:
            requested_frame_stride = None
    if requested_frame_stride is not None:
        try:
            test_cfg.sample_stride = 1
        except Exception:
            try:
                test_cfg["sample_stride"] = 1
            except Exception:
                pass
        try:
            test_cfg.feature_stride = int(requested_frame_stride)
        except Exception:
            try:
                test_cfg["feature_stride"] = int(requested_frame_stride)
            except Exception:
                pass
    pipeline_cfg = _patch_prepare_video_info(
        copy.deepcopy(list(test_cfg.pipeline)),
        os.path.splitext(video_path)[1].lstrip("."),
    )
    pipeline = Compose(pipeline_cfg)

    base = os.path.splitext(os.path.basename(video_path))[0]
    vmeta = _probe_video_meta(video_path)

    item_base: Dict[str, Any] = {
        "video_name": base,
        "data_path": os.path.dirname(video_path),
        "duration": float(vmeta["duration"]),
    }
    if "resize_length" in test_cfg:
        item_base["resize_length"] = int(test_cfg.resize_length)
        item_base["fps"] = -1
    else:
        item_base["fps"] = float(vmeta["fps"])
    if "window_size" in test_cfg:
        item_base["window_size"] = int(test_cfg.window_size)
    if "sample_stride" in test_cfg:
        item_base["sample_stride"] = int(test_cfg.sample_stride)
    if "feature_stride" in test_cfg:
        item_base["feature_stride"] = int(test_cfg.feature_stride)
    if "offset_frames" in test_cfg:
        item_base["offset_frames"] = int(test_cfg.offset_frames)

    model = build_detector(copy.deepcopy(cfg.model))
    state_dict = _load_state_dict(ckpt_path)
    delta_sources: List[Any] = []
    if adapter_delta_path:
        delta_sources.append(adapter_delta_path)
    delta_sources.extend(list(adapter_delta_sources or []))
    state_dict = _merge_vit_interactive_deltas(state_dict, delta_sources, progress_cb=progress_cb)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    _validate_ckpt_load_result(
        missing=missing,
        unexpected=unexpected,
        required_prefixes=("backbone.", "projection.", "neck."),
        progress_cb=progress_cb,
    )

    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(run_device)
    model.eval()
    # We use the amp to avoid potential OOM on CUDA. 
    amp_flag = str(os.environ.get("EAST_FEAT_AMP", "1") or "1").strip().lower() 
    use_amp = (run_device.type == "cuda") and (amp_flag not in ("0", "false", "no", "off"))
    if use_amp:
        _log(progress_cb, "[EAST] AMP enabled for feature forward on CUDA.")

    def _forward_one_sample(sample: Dict[str, Any]) -> tuple[np.ndarray, List[int], int, float]:
        inputs = sample.get("inputs")
        if inputs is None:
            raise RuntimeError("EAST pipeline did not return 'inputs'.")
        if inputs.ndim == 5:
            inputs = inputs.unsqueeze(0)  # [1, N, C, T, H, W]
        elif inputs.ndim != 6:
            raise RuntimeError(f"Unexpected EAST input shape: {tuple(inputs.shape)}")

        masks = sample.get("masks")
        if masks is None:
            masks = torch.ones((inputs.shape[0], int(inputs.shape[3])), dtype=torch.bool)
        elif masks.ndim == 1:
            masks = masks.unsqueeze(0)

        inputs = inputs.to(run_device, non_blocking=True)
        masks = masks.to(run_device, non_blocking=True)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else contextlib.nullcontext()
        )
        with torch.no_grad(), amp_ctx:
            x: Any = inputs
            m: Any = masks

            if getattr(model, "with_backbone", False):
                try:
                    x = model.backbone(x, m)
                except TypeError:
                    x = model.backbone(x)

            if hasattr(model, "pad_data") and callable(getattr(model, "pad_data")):
                try:
                    x, m = model.pad_data(x, m)
                except Exception:
                    pass

            if getattr(model, "with_projection", False):
                x, m = model.projection(x, m)

            if getattr(model, "with_neck", False):
                x, m = model.neck(x, m)

        feat_tensor = None
        if torch.is_tensor(x):
            feat_tensor = x
        elif isinstance(x, (list, tuple)):
            cand = [t for t in x if torch.is_tensor(t) and t.ndim >= 3]
            if cand:
                feat_tensor = max(cand, key=lambda t: int(t.shape[-1]))
        if feat_tensor is None:
            raise RuntimeError("Unable to parse EAST temporal feature tensor from model forward.")
        if feat_tensor.ndim == 2:
            feat_tensor = feat_tensor.unsqueeze(0)
        if feat_tensor.ndim != 3:
            raise RuntimeError(f"Unexpected EAST feature shape: {tuple(feat_tensor.shape)}")

        feat_win = feat_tensor[0].detach().to(torch.float32).cpu().numpy().T
        if feat_win.ndim != 2:
            raise RuntimeError(f"Unexpected EAST feature array shape after transpose: {feat_win.shape}")

        sampled_total = _resolve_total_frames(sample, int(vmeta["total_frames"]))
        if sampled_total <= 0:
            sampled_total = int(vmeta["total_frames"])
        sampled_fps_local = _resolve_avg_fps(sample, float(vmeta["fps"]))

        T_win = int(feat_win.shape[0])
        picked_win = _picked_indices_from_sampling(sample, T=T_win, total_frames=int(sampled_total))
        if len(picked_win) != T_win:
            if T_win <= 1:
                picked_win = [0] * T_win
            else:
                picked_win = [int(round(i * (sampled_total - 1) / float(T_win - 1))) for i in range(T_win)]
        return np.asarray(feat_win, dtype=np.float32), picked_win, int(sampled_total), float(sampled_fps_local)

    load_method = _detect_load_frames_method(pipeline_cfg)
    is_sliding = load_method == "sliding_window"

    spans: List[tuple[int, int]] = [(0, 0)]
    centers = np.asarray([], dtype=np.int64)
    plan: Dict[str, Any] = {}
    if is_sliding:
        plan = _build_sliding_window_spans(test_cfg, int(vmeta["total_frames"]))
        spans = list(plan.get("spans") or [])
        centers = np.asarray(plan.get("snippet_centers"), dtype=np.int64).reshape(-1)
        _log(
            progress_cb,
            (
                f"[EAST] Compose feature pipeline for {base} "
                f"(sliding windows={len(spans)}, window_size={plan.get('window_size')}, "
                f"overlap={float(plan.get('window_overlap_ratio', 0.0)):.2f}, "
                f"snippet_stride={plan.get('snippet_stride')})"
            ),
        )
    else:
        _log(progress_cb, f"[EAST] Compose feature pipeline for {base}")

    _log(progress_cb, "[EAST] Forward backbone/projection for temporal features...")
    feat_sum: Dict[int, np.ndarray] = {}
    feat_cnt: Dict[int, int] = {}
    sampled_total_frames = int(vmeta["total_frames"])
    sampled_fps = float(vmeta["fps"])

    for win_i, span in enumerate(spans, start=1):
        item = dict(item_base)
        if is_sliding:
            start_idx = int(span[0])
            end_idx = int(span[1])
            end_idx = max(start_idx, end_idx - 1)
            item["sample_stride"] = int(plan.get("sample_stride", 1))
            item["feature_stride"] = int(plan.get("feature_stride", 1))
            item["snippet_stride"] = int(plan.get("snippet_stride", 1))
            item["window_size"] = int(plan.get("window_size", _as_int(item.get("window_size"), 1)))
            item["feature_start_idx"] = int(start_idx)
            item["feature_end_idx"] = int(end_idx)
            if centers.size > start_idx:
                item["window_start_frame"] = int(centers[start_idx])
            else:
                item["window_start_frame"] = 0

            log_every = max(1, len(spans) // 8)
            if win_i == 1 or win_i == len(spans) or (win_i % log_every == 0):
                _log(
                    progress_cb,
                    f"[EAST] Sliding window {win_i}/{len(spans)}: feature_idx=[{start_idx}, {end_idx}]",
                )
        else:
            _inject_window_sampling_hints(item, test_cfg, total_frames=int(vmeta["total_frames"]))

        sample = pipeline(item)
        feat_win, picked_win, sampled_total, sampled_fps_local = _forward_one_sample(sample)
        sampled_total_frames = max(int(sampled_total_frames), int(sampled_total))
        if sampled_fps_local > 1e-6:
            sampled_fps = float(sampled_fps_local)

        n = min(int(feat_win.shape[0]), len(picked_win))
        for j in range(n):
            frame_idx = int(picked_win[j])
            vec = np.asarray(feat_win[j], dtype=np.float32)
            if frame_idx in feat_sum:
                feat_sum[frame_idx] += vec
                feat_cnt[frame_idx] = int(feat_cnt[frame_idx]) + 1
            else:
                feat_sum[frame_idx] = vec.copy()
                feat_cnt[frame_idx] = 1

    if not feat_sum:
        raise RuntimeError("EAST feature extraction produced no temporal features.")

    picked = sorted(int(k) for k in feat_sum.keys())
    feat = np.stack([feat_sum[k] / float(max(1, feat_cnt[k])) for k in picked], axis=0).astype(np.float32)
    T = int(feat.shape[0])
    total_frames = int(sampled_total_frames if sampled_total_frames > 0 else vmeta["total_frames"])

    if sampled_total_frames > 0 and int(vmeta["total_frames"]) > 0:
        if abs(int(sampled_total_frames) - int(vmeta["total_frames"])) > 5:
            _log(
                progress_cb,
                f"[EAST][WARN] frame count mismatch: pipeline={sampled_total_frames}, cv2={int(vmeta['total_frames'])}",
            )

    if picked:
        _log(
            progress_cb,
            f"[EAST] Temporal sampling: T={T}, total_frames={total_frames}, picked_range=[{picked[0]}, {picked[-1]}]",
        )
    else:
        _log(progress_cb, f"[EAST] Temporal sampling: T={T}, total_frames={total_frames}, picked_range=[]")

    est_stride = 1
    if len(picked) >= 2:
        diffs = np.diff(np.asarray(picked, dtype=np.int64))
        diffs = diffs[diffs > 0]
        if diffs.size > 0:
            est_stride = max(1, int(np.median(diffs)))
        elif T > 1:
            est_stride = max(1, int(round(float(total_frames) / float(T))))
    elif T > 1:
        est_stride = max(1, int(round(float(total_frames) / float(T))))

    duration = float(vmeta["duration"])
    if sampled_fps > 1e-6 and total_frames > 0:
        duration = float(total_frames) / float(sampled_fps)

    meta = {
        "backbone": "east_backbone",
        "feature_dim": int(feat.shape[1]),
        "frame_stride": int(requested_frame_stride if requested_frame_stride is not None else est_stride),
        "picked_indices": picked,
        "num_frames": int(total_frames),
        "input_size": 224,
        "fps": float(sampled_fps if sampled_fps > 1e-6 else vmeta["fps"]),
        "duration": float(duration),
        "video_path": str(video_path),
        "source": "east_backbone",
        "east_cfg": os.path.abspath(os.path.expanduser(cfg_path)),
        "east_ckpt": os.path.abspath(os.path.expanduser(ckpt_path)),
    }
    return {"features": np.asarray(feat, dtype=np.float32), "meta": meta}
