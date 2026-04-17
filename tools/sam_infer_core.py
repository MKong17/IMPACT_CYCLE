from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import re
import sys
import tempfile
import types
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_MODEL_CACHE: Dict[str, Dict[str, object]] = {}
_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass
class StageRuntimeConfig:
    mode: str
    config_file: str
    weights: str
    weights_url: str
    opts: Tuple[str, ...]
    device: str
    score_threshold: float
    nms_threshold: float
    iou_threshold: float
    mask_export: str
    max_mask_pixels: int
    vision_pretrained_url: str


@dataclass
class Sam3StageRuntimeConfig:
    version: str
    checkpoint_path: str
    checkpoint_url: str
    device: str
    score_threshold: float
    mask_export: str
    max_mask_pixels: int
    compile: bool
    preprocess_max_side: int


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Runtime config must be a JSON object: {path}")
    payload["_config_dir"] = os.path.dirname(os.path.abspath(path))
    return payload


def _is_google_drive_url(url: str) -> bool:
    text = str(url or "").strip().lower()
    return "drive.google.com" in text


def _download_to_path(url: str, target_path: str) -> None:
    target = os.path.abspath(os.path.expanduser(str(target_path or "").strip()))
    if not target:
        raise RuntimeError("Missing target path for checkpoint download.")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="owsam_dl_", suffix=".part", dir=os.path.dirname(target))
    os.close(fd)
    try:
        if _is_google_drive_url(url):
            try:
                gdown = importlib.import_module("gdown")
            except Exception as exc:
                raise RuntimeError(
                    "Checkpoint download requires gdown for Google Drive URLs. "
                    "Install it in the OpenWorldSAM environment."
                ) from exc
            ok = bool(gdown.download(url=str(url), output=tmp_path, quiet=True, fuzzy=True))
            if not ok:
                raise RuntimeError(f"gdown failed to download: {url}")
        else:
            with urllib.request.urlopen(str(url)) as src, open(tmp_path, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
        if not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) <= 0:
            raise RuntimeError(f"Downloaded file is empty: {url}")
        os.replace(tmp_path, target)
    finally:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _ensure_downloaded_file(path: str, url: str, *, label: str) -> str:
    target = os.path.abspath(os.path.expanduser(str(path or "").strip()))
    if not target:
        raise RuntimeError(f"Missing path for {label}.")
    if os.path.isfile(target) and os.path.getsize(target) > 0:
        return target
    source_url = str(url or "").strip()
    if not source_url:
        raise RuntimeError(f"Missing {label}: {target}")
    print(f"[OWSAM] downloading {label} -> {target}", file=sys.stderr)
    _download_to_path(source_url, target)
    print(f"[OWSAM] download complete: {label}", file=sys.stderr)
    return target


def _normalize_text(text: str) -> str:
    out = str(text or "").strip().lower()
    out = re.sub(r"[\-_\/]+", " ", out)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def _runtime_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return raw
    if os.name == "nt":
        return raw
    if _WINDOWS_ABS_RE.match(raw):
        drive = raw[0].lower()
        suffix = raw[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{suffix}"
    return raw


def _resize_pil_image_max_side(image, max_side: int):
    limit = max(0, int(max_side or 0))
    if limit <= 0:
        return image, 1.0
    width = int(getattr(image, "width", 0) or 0)
    height = int(getattr(image, "height", 0) or 0)
    longest = max(width, height)
    if longest <= 0 or longest <= limit:
        return image, 1.0
    scale = float(limit) / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resample = getattr(image, "Resampling", None)
    lanczos = getattr(resample, "LANCZOS", None) if resample is not None else None
    if lanczos is None:
        lanczos = getattr(image, "LANCZOS", 1)
    return image.resize((new_width, new_height), lanczos), scale


def _scale_rows_to_original_size(
    rows: List[Dict[str, object]],
    *,
    scale_x: float,
    scale_y: float,
    width: int,
    height: int,
) -> List[Dict[str, object]]:
    if not rows or (abs(float(scale_x) - 1.0) < 1e-6 and abs(float(scale_y) - 1.0) < 1e-6):
        return rows
    out: List[Dict[str, object]] = []
    for row in rows:
        item = dict(row or {})
        bbox = list(item.get("bbox") or [0, 0, 0, 0])
        if len(bbox) >= 4:
            x = float(bbox[0] or 0.0) * float(scale_x)
            y = float(bbox[1] or 0.0) * float(scale_y)
            w_box = float(bbox[2] or 0.0) * float(scale_x)
            h_box = float(bbox[3] or 0.0) * float(scale_y)
            item["bbox"] = _clip_bbox_xywh([x, y, w_box, h_box], width, height)
        mask = item.get("mask")
        if isinstance(mask, dict):
            pixels = mask.get("pixels")
            if isinstance(pixels, list) and pixels:
                scaled_pixels: List[List[int]] = []
                for pt in pixels:
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        px = int(round(float(pt[0] or 0.0) * float(scale_x)))
                        py = int(round(float(pt[1] or 0.0) * float(scale_y)))
                        scaled_pixels.append([px, py])
                item["mask"] = {"pixels": scaled_pixels}
        out.append(item)
    return out


def _category_prompt_variants(prompt: str) -> List[str]:
    text = _normalize_text(prompt)
    if not text:
        return []
    variants = [text]
    stripped = text
    for prefix in ("all ", "the ", "a ", "an "):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].strip()
            variants.append(stripped)
    for suffix in (" instances", " instance", " objects", " object"):
        if stripped.endswith(suffix):
            variants.append(stripped[: -len(suffix)].strip())
    uniq: List[str] = []
    for item in variants:
        item2 = _normalize_text(item)
        if item2 and item2 not in uniq:
            uniq.append(item2)
    return uniq


def _resolve_prompt_alias(prompt: str, aliases: Dict[str, str]) -> str:
    if not aliases:
        aliases = {}
    alias_map = {_normalize_text(k): str(v or "").strip() for k, v in aliases.items()}
    for variant in _category_prompt_variants(prompt):
        if variant in alias_map and alias_map[variant]:
            return alias_map[variant]
    variants = _category_prompt_variants(prompt)
    return variants[-1] if variants else str(prompt or "").strip()


def _runtime_repo_root(runtime_cfg: Dict[str, object]) -> str:
    repo_root = str(runtime_cfg.get("repo_root", "") or "").strip()
    if not repo_root:
        repo_root = str(os.environ.get("OPENWORLDSAM_REPO_ROOT", "") or "").strip()
    if not repo_root:
        raise RuntimeError("Missing OpenWorldSAM repo_root. Set it in the runtime config or OPENWORLDSAM_REPO_ROOT.")
    if not os.path.isabs(repo_root):
        repo_root = os.path.abspath(os.path.join(str(runtime_cfg.get("_config_dir", "") or os.getcwd()), repo_root))
    repo_root = os.path.abspath(os.path.expanduser(repo_root))
    if not os.path.isdir(repo_root):
        raise RuntimeError(f"OpenWorldSAM repo_root does not exist: {repo_root}")
    return repo_root


def _runtime_model_family(runtime_cfg: Dict[str, object], repo_root: str) -> str:
    explicit = str(runtime_cfg.get("model_family", "") or runtime_cfg.get("backend_family", "") or "").strip().lower()
    if explicit in {"sam3", "sam"}:
        return "sam3"
    if explicit in {"openworldsam", "open-world-sam"}:
        return "sam3"
    if os.path.isdir(os.path.join(repo_root, "sam3")):
        return "sam3"
    return "sam3"


def _validate_native_extension(repo_root: str) -> None:
    sam2_dir = os.path.join(repo_root, "model", "segment_anything_2", "sam2")
    if os.name == "nt":
        native_path = os.path.join(sam2_dir, "_C.pyd")
        if os.path.isfile(native_path):
            return
        linux_only_path = os.path.join(sam2_dir, "_C.so")
        tail = ""
        if os.path.isfile(linux_only_path):
            tail = f" Found Linux-only build at {linux_only_path}."
        raise RuntimeError(
            "Windows SAM build is incomplete: missing SAM2 extension "
            f"{native_path}.{tail}"
        )


def _load_openworldsam_api(repo_root: str) -> Dict[str, object]:
    _validate_native_extension(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        inference_utils = importlib.import_module("demo.inference_utils")
    except Exception as exc:
        raise RuntimeError(f"Failed to import OpenWorldSAM demo.inference_utils from {repo_root}: {exc}") from exc
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(f"Failed to import torch in the OpenWorldSAM environment: {exc}") from exc
    return {
        "torch": torch,
        "setup_cfg": getattr(inference_utils, "setup_cfg"),
        "load_model": getattr(inference_utils, "load_model"),
        "prepare_image_inputs": getattr(inference_utils, "prepare_image_inputs"),
        "build_inference_inputs": getattr(inference_utils, "build_inference_inputs"),
        "get_metadata": getattr(inference_utils, "get_metadata"),
        "resolve_category_ids": getattr(inference_utils, "resolve_category_ids"),
    }


def _load_sam3_api(repo_root: str) -> Dict[str, object]:
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(f"Failed to import torch in the SAM3 environment: {exc}") from exc
    try:
        pil_image = importlib.import_module("PIL.Image")
    except Exception as exc:
        raise RuntimeError(f"Failed to import PIL in the SAM3 environment: {exc}") from exc
    if "pkg_resources" not in sys.modules:
        try:
            importlib.import_module("pkg_resources")
        except Exception:
            shim = types.ModuleType("pkg_resources")

            def _resource_filename(package_name: str, resource_name: str) -> str:
                module = importlib.import_module(str(package_name))
                module_file = str(getattr(module, "__file__", "") or "")
                if not module_file:
                    raise RuntimeError(f"Cannot resolve package path for resource lookup: {package_name}")
                base_dir = os.path.dirname(os.path.abspath(module_file))
                return os.path.abspath(os.path.join(base_dir, str(resource_name).replace("/", os.sep)))

            shim.resource_filename = _resource_filename  # type: ignore[attr-defined]
            sys.modules["pkg_resources"] = shim
    try:
        importlib.import_module("triton")
    except Exception:
        if "sam3.model.edt" not in sys.modules:
            edt_shim = types.ModuleType("sam3.model.edt")

            def _edt_triton_fallback(data):
                torch_mod = importlib.import_module("torch")
                tensor = data
                if not isinstance(tensor, torch_mod.Tensor):
                    tensor = torch_mod.as_tensor(tensor)
                if tensor.dim() != 3:
                    raise RuntimeError("edt_triton fallback expects a 3D tensor of shape (B, H, W).")
                device = tensor.device
                tensor_cpu = tensor.detach().to(device="cpu")
                try:
                    cv2_mod = importlib.import_module("cv2")
                except Exception as exc:
                    raise RuntimeError(
                        "SAM3 requires either triton or OpenCV for EDT fallback support."
                    ) from exc
                outputs = []
                for mask in tensor_cpu:
                    arr = mask.to(dtype=torch_mod.uint8).numpy()
                    dist = cv2_mod.distanceTransform(arr, cv2_mod.DIST_L2, 0)
                    outputs.append(torch_mod.from_numpy(dist))
                stacked = torch_mod.stack(outputs, dim=0).to(device=device)
                return stacked

            edt_shim.edt_triton = _edt_triton_fallback  # type: ignore[attr-defined]
            sys.modules["sam3.model.edt"] = edt_shim
    try:
        model_builder = importlib.import_module("sam3.model_builder")
        image_processor = importlib.import_module("sam3.model.sam3_image_processor")
        vitdet_module = importlib.import_module("sam3.model.vitdet")
    except Exception as exc:
        raise RuntimeError(f"Failed to import SAM3 modules from {repo_root}: {exc}") from exc
    if hasattr(vitdet_module, "addmm_act"):
        def _safe_addmm_act(activation, linear, mat1):
            weight = getattr(linear, "weight", None)
            bias = getattr(linear, "bias", None)
            target_dtype = getattr(weight, "dtype", None)
            target_device = getattr(weight, "device", None)
            if target_dtype is not None and hasattr(mat1, "is_floating_point") and mat1.is_floating_point():
                mat1 = mat1.to(dtype=target_dtype)
            if target_device is not None and hasattr(mat1, "to"):
                mat1 = mat1.to(device=target_device)
            out = torch.nn.functional.linear(mat1, weight, bias)
            if activation in (torch.nn.functional.relu, torch.nn.ReLU):
                return torch.nn.functional.relu(out)
            if activation in (torch.nn.functional.gelu, torch.nn.GELU):
                return torch.nn.functional.gelu(out)
            return out

        vitdet_module.addmm_act = _safe_addmm_act
    return {
        "torch": torch,
        "PIL_Image": pil_image,
        "build_sam3_image_model": getattr(model_builder, "build_sam3_image_model"),
        "Sam3Processor": getattr(image_processor, "Sam3Processor"),
    }


def _stage_runtime_config(runtime_cfg: Dict[str, object], stage: str) -> StageRuntimeConfig:
    stage_key = "sentence_refine" if str(stage or "").strip() == "sentence_refine" else "category_discovery"
    global_device = str(runtime_cfg.get("device", "") or os.environ.get("OPENWORLDSAM_DEVICE", "") or "").strip()
    global_mask_export = str(runtime_cfg.get("mask_export", "none") or "none").strip().lower()
    global_max_pixels = int(runtime_cfg.get("max_mask_pixels", 12000) or 12000)
    stage_payload = runtime_cfg.get(stage_key) or {}
    if not isinstance(stage_payload, dict):
        raise RuntimeError(f"Stage config must be an object: {stage_key}")
    mode = str(stage_payload.get("mode", "instance" if stage_key == "category_discovery" else "referring") or "").strip().lower()
    config_file = str(stage_payload.get("config_file", "") or "").strip()
    weights = str(stage_payload.get("weights", "") or "").strip()
    weights_url = str(stage_payload.get("weights_url", "") or "").strip()
    if not config_file:
        raise RuntimeError(f"Missing config_file for stage: {stage_key}")
    if not weights:
        raise RuntimeError(f"Missing weights for stage: {stage_key}")
    config_dir = str(runtime_cfg.get("_config_dir", "") or os.getcwd())
    if not os.path.isabs(config_file):
        config_file = os.path.abspath(os.path.join(config_dir, config_file))
    if not os.path.isabs(weights):
        weights = os.path.abspath(os.path.join(config_dir, weights))
    opts = stage_payload.get("opts") or []
    opts_tuple = tuple(str(x) for x in opts) if isinstance(opts, list) else tuple()
    device = str(stage_payload.get("device", "") or global_device or "").strip()
    score_threshold = float(stage_payload.get("score_threshold", runtime_cfg.get("score_threshold", 0.25)) or 0.25)
    nms_threshold = float(stage_payload.get("nms_threshold", runtime_cfg.get("nms_threshold", 0.5)) or 0.5)
    iou_threshold = float(stage_payload.get("iou_threshold", runtime_cfg.get("iou_threshold", 0.8)) or 0.8)
    mask_export = str(stage_payload.get("mask_export", global_mask_export) or "none").strip().lower()
    max_mask_pixels = int(stage_payload.get("max_mask_pixels", global_max_pixels) or global_max_pixels)
    vision_pretrained_url = str(
        stage_payload.get("vision_pretrained_url", runtime_cfg.get("vision_pretrained_url", "")) or ""
    ).strip()
    return StageRuntimeConfig(
        mode=mode,
        config_file=config_file,
        weights=weights,
        weights_url=weights_url,
        opts=opts_tuple,
        device=device,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        iou_threshold=iou_threshold,
        mask_export=mask_export,
        max_mask_pixels=max_mask_pixels,
        vision_pretrained_url=vision_pretrained_url,
    )


def _sam3_stage_runtime_config(runtime_cfg: Dict[str, object], stage: str) -> Sam3StageRuntimeConfig:
    stage_key = "sentence_refine" if str(stage or "").strip() == "sentence_refine" else "category_discovery"
    global_device = str(runtime_cfg.get("device", "") or os.environ.get("SAM3_DEVICE", "") or "").strip()
    global_mask_export = str(runtime_cfg.get("mask_export", "none") or "none").strip().lower()
    global_max_pixels = int(runtime_cfg.get("max_mask_pixels", 12000) or 12000)
    global_preprocess_max_side = int(runtime_cfg.get("preprocess_max_side", 0) or 0)
    stage_payload = runtime_cfg.get(stage_key) or {}
    if not isinstance(stage_payload, dict):
        raise RuntimeError(f"Stage config must be an object: {stage_key}")
    config_dir = str(runtime_cfg.get("_config_dir", "") or os.getcwd())
    checkpoint_path = str(
        stage_payload.get("checkpoint_path", stage_payload.get("weights", runtime_cfg.get("checkpoint_path", ""))) or ""
    ).strip()
    checkpoint_url = str(
        stage_payload.get("checkpoint_url", stage_payload.get("weights_url", runtime_cfg.get("checkpoint_url", ""))) or ""
    ).strip()
    if checkpoint_path and not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(os.path.join(config_dir, checkpoint_path))
    return Sam3StageRuntimeConfig(
        version=str(stage_payload.get("model_version", runtime_cfg.get("model_version", "sam3")) or "sam3").strip(),
        checkpoint_path=checkpoint_path,
        checkpoint_url=checkpoint_url,
        device=str(stage_payload.get("device", "") or global_device or "").strip(),
        score_threshold=float(stage_payload.get("score_threshold", runtime_cfg.get("score_threshold", 0.25)) or 0.25),
        mask_export=str(stage_payload.get("mask_export", global_mask_export) or "none").strip().lower(),
        max_mask_pixels=int(stage_payload.get("max_mask_pixels", global_max_pixels) or global_max_pixels),
        compile=bool(stage_payload.get("compile", runtime_cfg.get("compile", False))),
        preprocess_max_side=int(stage_payload.get("preprocess_max_side", global_preprocess_max_side) or global_preprocess_max_side),
    )


def _cache_key(repo_root: str, stage_cfg: StageRuntimeConfig) -> str:
    payload = {
        "repo_root": repo_root,
        "config_file": stage_cfg.config_file,
        "weights": stage_cfg.weights,
        "opts": list(stage_cfg.opts),
        "device": stage_cfg.device,
        "mode": stage_cfg.mode,
        "nms_threshold": stage_cfg.nms_threshold,
        "iou_threshold": stage_cfg.iou_threshold,
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _model_uses_meta_tensors(model) -> bool:
    try:
        for tensor in model.parameters(recurse=True):
            if getattr(tensor, "is_meta", False):
                return True
        for tensor in model.buffers(recurse=True):
            if getattr(tensor, "is_meta", False):
                return True
    except Exception:
        return False
    return False


def _apply_stage_thresholds(cfg, stage_cfg: StageRuntimeConfig) -> None:
    try:
        cfg.MODEL.OpenWorldSAM2.TEST.NMS_THRESHOLD = float(stage_cfg.nms_threshold)
    except Exception:
        pass
    try:
        cfg.MODEL.OpenWorldSAM2.TEST.IOU_THRESHOLD = float(stage_cfg.iou_threshold)
    except Exception:
        pass


def _resolve_existing_path(base_dirs: Sequence[str], raw_path: str) -> str:
    value = str(raw_path or "").strip()
    if not value:
        return value
    if os.path.isabs(value):
        return value
    for base_dir in base_dirs:
        base = str(base_dir or "").strip()
        if not base:
            continue
        candidate = os.path.abspath(os.path.join(base, value))
        if os.path.exists(candidate):
            return candidate
    first_base = str(next((x for x in base_dirs if str(x or "").strip()), "") or "").strip()
    if first_base:
        return os.path.abspath(os.path.join(first_base, value))
    return os.path.abspath(value)


def _get_model_bundle(api: Dict[str, object], repo_root: str, stage_cfg: StageRuntimeConfig) -> Dict[str, object]:
    key = _cache_key(repo_root, stage_cfg)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    stage_weights = _ensure_downloaded_file(
        stage_cfg.weights,
        stage_cfg.weights_url,
        label=f"{stage_cfg.mode} checkpoint",
    )
    setup_cfg = api["setup_cfg"]
    load_model = api["load_model"]
    cfg = setup_cfg(
        config_file=stage_cfg.config_file,
        weights=stage_weights,
        device=(stage_cfg.device or None),
        opts=list(stage_cfg.opts),
    )
    try:
        vision_pretrained = str(cfg.MODEL.OpenWorldSAM2.VISION_PRETRAINED or "").strip()
    except Exception:
        vision_pretrained = ""
    if vision_pretrained:
        resolved_backbone = _resolve_existing_path(
            base_dirs=[repo_root, os.path.dirname(stage_cfg.config_file)],
            raw_path=vision_pretrained,
        )
        resolved_backbone = _ensure_downloaded_file(
            resolved_backbone,
            stage_cfg.vision_pretrained_url,
            label="sam2 backbone checkpoint",
        )
        try:
            cfg.MODEL.OpenWorldSAM2.VISION_PRETRAINED = resolved_backbone
        except Exception:
            pass
    _apply_stage_thresholds(cfg, stage_cfg)
    model = load_model(cfg)
    if _model_uses_meta_tensors(model):
        raise RuntimeError(
            "OpenWorldSAM model still contains meta tensors after loading. "
            "The upstream model loader should materialize meta modules with to_empty() before loading weights."
        )
    bundle = {"cfg": cfg, "model": model}
    _MODEL_CACHE[key] = bundle
    return bundle


def _sam3_cache_key(repo_root: str, stage_cfg: Sam3StageRuntimeConfig) -> str:
    payload = {
        "repo_root": repo_root,
        "version": stage_cfg.version,
        "checkpoint_path": stage_cfg.checkpoint_path,
        "device": stage_cfg.device,
        "score_threshold": stage_cfg.score_threshold,
        "compile": stage_cfg.compile,
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _get_sam3_model_bundle(api: Dict[str, object], repo_root: str, stage_cfg: Sam3StageRuntimeConfig) -> Dict[str, object]:
    key = _sam3_cache_key(repo_root, stage_cfg)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    checkpoint_path = str(stage_cfg.checkpoint_path or "").strip()
    load_from_hf = not checkpoint_path
    if checkpoint_path:
        checkpoint_path = _ensure_downloaded_file(
            checkpoint_path,
            stage_cfg.checkpoint_url,
            label=f"{stage_cfg.version} checkpoint",
        )
    model = api["build_sam3_image_model"](
        device=(stage_cfg.device or "cuda"),
        eval_mode=True,
        checkpoint_path=(checkpoint_path or None),
        load_from_HF=load_from_hf,
        enable_inst_interactivity=False,
        compile=bool(stage_cfg.compile),
    )
    torch_mod = api["torch"]
    target_device = str(stage_cfg.device or "cuda").strip() or "cuda"
    # Force the whole model onto one device to avoid CPU/GPU mixed-module states.
    try:
        model = model.to(device=target_device)
    except Exception:
        pass
    device_name = str(stage_cfg.device or "cuda").strip().lower()
    if device_name.startswith("cuda"):
        try:
            model = model.to(dtype=torch_mod.float32)
        except Exception:
            pass
    processor = api["Sam3Processor"](
        model,
        device=(stage_cfg.device or "cuda"),
        confidence_threshold=float(stage_cfg.score_threshold),
    )
    try:
        model_dtype = next(
            param.dtype for param in model.parameters() if getattr(param, "is_floating_point", lambda: False)()
        )
    except Exception:
        model_dtype = None
    if model_dtype is not None and hasattr(model, "backbone") and hasattr(model.backbone, "forward_image"):
        original_forward_image = model.backbone.forward_image

        def _forward_image_cast(samples, *args, **kwargs):
            tensor = samples
            if hasattr(tensor, "is_floating_point") and tensor.is_floating_point():
                tensor = tensor.to(dtype=model_dtype)
            return original_forward_image(tensor, *args, **kwargs)

        model.backbone.forward_image = _forward_image_cast
    if model_dtype is not None and hasattr(model, "_get_dummy_prompt"):
        original_get_dummy_prompt = model._get_dummy_prompt

        def _get_dummy_prompt_cast(*args, **kwargs):
            prompt_obj = original_get_dummy_prompt(*args, **kwargs)
            return _cast_prompt_tensors_to_dtype(prompt_obj, model_dtype)

        model._get_dummy_prompt = _get_dummy_prompt_cast
    if model_dtype is not None and hasattr(processor, "add_geometric_prompt"):
        original_add_geometric_prompt = processor.add_geometric_prompt

        def _add_geometric_prompt_cast(box, label, state):
            out_state = original_add_geometric_prompt(box, label, state)
            if isinstance(out_state, dict) and "geometric_prompt" in out_state:
                out_state["geometric_prompt"] = _cast_prompt_tensors_to_dtype(
                    out_state.get("geometric_prompt"), model_dtype
                )
            return out_state

        processor.add_geometric_prompt = _add_geometric_prompt_cast
    if model_dtype is not None and hasattr(processor, "_forward_grounding"):
        original_forward_grounding = processor._forward_grounding

        def _forward_grounding_cast(state):
            if isinstance(state, dict):
                if "geometric_prompt" in state:
                    state["geometric_prompt"] = _cast_prompt_tensors_to_dtype(
                        state.get("geometric_prompt"), model_dtype
                    )
                if "backbone_out" in state:
                    state["backbone_out"] = _cast_tensor_tree_to_dtype(
                        state.get("backbone_out"), model_dtype
                    )
            return original_forward_grounding(state)

        processor._forward_grounding = _forward_grounding_cast
    if model_dtype is not None and hasattr(model, "geometry_encoder") and hasattr(model.geometry_encoder, "forward"):
        original_geometry_forward = model.geometry_encoder.forward

        def _geometry_forward_cast(geo_prompt, img_feats, img_sizes, img_pos_embeds=None):
            geo_prompt = _cast_prompt_tensors_to_dtype(geo_prompt, model_dtype)
            img_feats = _cast_tensor_tree_to_dtype(img_feats, model_dtype)
            img_pos_embeds = _cast_tensor_tree_to_dtype(img_pos_embeds, model_dtype)
            return original_geometry_forward(geo_prompt, img_feats, img_sizes, img_pos_embeds)

        model.geometry_encoder.forward = _geometry_forward_cast
    bundle = {"model": model, "processor": processor}
    _MODEL_CACHE[key] = bundle
    return bundle


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _cast_prompt_tensors_to_dtype(prompt_obj, dtype) -> object:
    if prompt_obj is None or dtype is None:
        return prompt_obj
    float_fields = (
        "box_embeddings",
        "point_embeddings",
        "mask_embeddings",
    )
    for field_name in float_fields:
        value = getattr(prompt_obj, field_name, None)
        if value is not None and hasattr(value, "is_floating_point") and value.is_floating_point():
            setattr(prompt_obj, field_name, value.to(dtype=dtype))
    return prompt_obj


def _cast_tensor_tree_to_dtype(value, dtype):
    if value is None or dtype is None:
        return value
    if hasattr(value, "is_floating_point") and value.is_floating_point():
        return value.to(dtype=dtype)
    if isinstance(value, list):
        return [_cast_tensor_tree_to_dtype(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_cast_tensor_tree_to_dtype(item, dtype) for item in value)
    if isinstance(value, dict):
        return {key: _cast_tensor_tree_to_dtype(item, dtype) for key, item in value.items()}
    return value


def _field_from_instances(instances, field_name: str):
    if hasattr(instances, field_name):
        return getattr(instances, field_name)
    try:
        fields = instances.get_fields()
        if field_name in fields:
            return fields[field_name]
    except Exception:
        pass
    return None


def _to_numpy(value) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(value, "tensor"):
        value = value.tensor
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except Exception:
            pass
    return np.asarray(value)


def _clip_bbox_xywh(bbox: Sequence[float], width: int, height: int) -> List[int]:
    x = int(round(float(bbox[0] or 0)))
    y = int(round(float(bbox[1] or 0)))
    w = int(round(float(bbox[2] or 0)))
    h = int(round(float(bbox[3] or 0)))
    if width > 0:
        x = max(0, min(x, width - 1))
        w = max(0, min(w, width - x))
    if height > 0:
        y = max(0, min(y, height - 1))
        h = max(0, min(h, height - y))
    return [x, y, max(0, w), max(0, h)]


def _mask_payload(mask: np.ndarray, *, mask_export: str, max_mask_pixels: int) -> Dict[str, object]:
    if mask_export != "pixels":
        return {"pixels": []}
    ys, xs = np.where(mask > 0)
    n = int(len(xs))
    if n <= 0:
        return {"pixels": []}
    step = max(1, int(math.ceil(float(n) / float(max(1, int(max_mask_pixels))))))
    pixels = [[int(xs[i]), int(ys[i])] for i in range(0, n, step)]
    return {"pixels": pixels}


def _extract_instance_rows(
    instances,
    *,
    width: int,
    height: int,
    mask_export: str,
    max_mask_pixels: int,
) -> List[Dict[str, object]]:
    if instances is None:
        return []
    try:
        instances = instances.to("cpu")
    except Exception:
        pass
    boxes_arr = _to_numpy(_field_from_instances(instances, "pred_boxes"))
    scores_arr = _to_numpy(_field_from_instances(instances, "scores"))
    classes_arr = _to_numpy(_field_from_instances(instances, "pred_classes"))
    masks_arr = _to_numpy(_field_from_instances(instances, "pred_masks"))
    num = 0
    if boxes_arr.ndim == 2 and boxes_arr.shape[1] >= 4:
        num = int(boxes_arr.shape[0])
    elif masks_arr.ndim >= 3:
        num = int(masks_arr.shape[0])
    elif scores_arr.ndim >= 1:
        num = int(scores_arr.shape[0])
    out: List[Dict[str, object]] = []
    for idx in range(num):
        if boxes_arr.ndim == 2 and boxes_arr.shape[1] >= 4:
            x1, y1, x2, y2 = [float(v) for v in boxes_arr[idx][:4]]
            bbox = _clip_bbox_xywh([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], width, height)
        else:
            bbox = [0, 0, 0, 0]
        score = float(scores_arr[idx]) if scores_arr.ndim >= 1 and idx < len(scores_arr) else 0.0
        class_id = int(classes_arr[idx]) if classes_arr.ndim >= 1 and idx < len(classes_arr) else -1
        mask_payload = {"pixels": []}
        if masks_arr.ndim >= 3 and idx < int(masks_arr.shape[0]):
            mask_payload = _mask_payload(np.asarray(masks_arr[idx]).astype(np.uint8), mask_export=mask_export, max_mask_pixels=max_mask_pixels)
        out.append(
            {
                "bbox": bbox,
                "mask": mask_payload,
                "score": score,
                "class_id": class_id,
            }
        )
    out.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    return out


def _extract_rows_from_output(
    output: Any,
    *,
    width: int,
    height: int,
    mask_export: str,
    max_mask_pixels: int,
) -> List[Dict[str, object]]:
    if isinstance(output, list) and output:
        output = output[0]
    if isinstance(output, dict) and "instances" in output:
        return _extract_instance_rows(
            output.get("instances"),
            width=width,
            height=height,
            mask_export=mask_export,
            max_mask_pixels=max_mask_pixels,
        )
    if isinstance(output, dict):
        masks = output.get("pred_masks")
        if masks is None:
            masks = output.get("masks")
        boxes = output.get("pred_boxes")
        if boxes is None:
            boxes = output.get("boxes")
        scores = output.get("scores")
        classes = output.get("pred_classes")
        tmp = type("InstancesLike", (), {})()
        setattr(tmp, "pred_masks", masks)
        setattr(tmp, "pred_boxes", boxes)
        setattr(tmp, "scores", scores)
        setattr(tmp, "pred_classes", classes)
        return _extract_instance_rows(
            tmp,
            width=width,
            height=height,
            mask_export=mask_export,
            max_mask_pixels=max_mask_pixels,
        )
    return []


def _prepare_image(api: Dict[str, object], image_path: str, image_format: str = "BGR"):
    prepare_image_inputs = api["prepare_image_inputs"]
    original_image, sam_tensor, beit_tensor, height, width = prepare_image_inputs(
        _runtime_path(image_path),
        image_format=image_format,
    )
    return original_image, sam_tensor, beit_tensor, int(height), int(width)


def _run_category_discovery(
    api: Dict[str, object],
    runtime_cfg: Dict[str, object],
    image_path: str,
    prompts: List[str],
    *,
    max_instances: int,
) -> Dict[str, List[Dict[str, object]]]:
    repo_root = _runtime_repo_root(runtime_cfg)
    stage_cfg = _stage_runtime_config(runtime_cfg, "category_discovery")
    bundle = _get_model_bundle(api, repo_root, stage_cfg)
    cfg = bundle["cfg"]
    model = bundle["model"]
    _, sam_tensor, beit_tensor, height, width = _prepare_image(api, image_path)
    metadata = api["get_metadata"](cfg)
    aliases = runtime_cfg.get("prompt_aliases") or {}
    normalized_prompts: List[str] = []
    original_by_normalized: Dict[str, List[str]] = {}
    category_id_by_prompt: Dict[str, int] = {}
    for prompt in prompts:
        mapped = _resolve_prompt_alias(prompt, aliases if isinstance(aliases, dict) else {})
        if not mapped:
            continue
        try:
            resolved_ids = api["resolve_category_ids"]([mapped], metadata)
        except Exception:
            continue
        if not resolved_ids:
            continue
        category_id = int(resolved_ids[0])
        original_by_normalized.setdefault(mapped, []).append(prompt)
        category_id_by_prompt[mapped] = category_id
        if mapped not in normalized_prompts:
            normalized_prompts.append(mapped)
    if not normalized_prompts:
        return {prompt: [] for prompt in prompts}
    unique_categories: List[int] = []
    normalized_by_category: Dict[int, str] = {}
    for mapped_prompt in normalized_prompts:
        category_id = int(category_id_by_prompt.get(mapped_prompt, -1))
        if category_id < 0:
            continue
        normalized_by_category[category_id] = mapped_prompt
        if category_id not in unique_categories:
            unique_categories.append(category_id)
    inputs = api["build_inference_inputs"](
        sam_tensor,
        beit_tensor,
        height,
        width,
        normalized_prompts,
        unique_categories,
    )
    torch = api["torch"]
    with torch.no_grad():
        output = model(inputs)
    rows = _extract_rows_from_output(
        output[0] if isinstance(output, list) and output else output,
        width=width,
        height=height,
        mask_export=stage_cfg.mask_export,
        max_mask_pixels=stage_cfg.max_mask_pixels,
    )
    by_prompt: Dict[str, List[Dict[str, object]]] = {prompt: [] for prompt in prompts}
    by_category_id: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        category_id = int(row.get("class_id", -1))
        if category_id < 0:
            continue
        row2 = dict(row)
        row2.pop("class_id", None)
        by_category_id.setdefault(category_id, []).append(row2)
    for category_id, mapped_prompt in normalized_by_category.items():
        grouped_rows = list(by_category_id.get(category_id, []))[: max(1, int(max_instances))]
        for original_prompt in original_by_normalized.get(mapped_prompt, []):
            by_prompt[original_prompt] = [copy.deepcopy(item) for item in grouped_rows]
    return by_prompt


def _run_sentence_refine(
    api: Dict[str, object],
    runtime_cfg: Dict[str, object],
    image_path: str,
    prompts: List[str],
    *,
    max_instances: int,
) -> Dict[str, List[Dict[str, object]]]:
    repo_root = _runtime_repo_root(runtime_cfg)
    stage_cfg = _stage_runtime_config(runtime_cfg, "sentence_refine")
    bundle = _get_model_bundle(api, repo_root, stage_cfg)
    model = bundle["model"]
    _, sam_tensor, beit_tensor, height, width = _prepare_image(api, image_path)
    torch = api["torch"]
    out: Dict[str, List[Dict[str, object]]] = {}
    for prompt in prompts:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            continue
        inputs = api["build_inference_inputs"](
            sam_tensor,
            beit_tensor,
            height,
            width,
            [prompt_text],
            [0],
        )
        with torch.no_grad():
            output = model(inputs)
        rows = _extract_rows_from_output(
            output[0] if isinstance(output, list) and output else output,
            width=width,
            height=height,
            mask_export=stage_cfg.mask_export,
            max_mask_pixels=stage_cfg.max_mask_pixels,
        )
        cleaned_rows: List[Dict[str, object]] = []
        for row in rows[: max(1, int(max_instances))]:
            row2 = dict(row)
            row2.pop("class_id", None)
            cleaned_rows.append(row2)
        out[prompt_text] = cleaned_rows
    return out


def _run_prompts(
    runtime_cfg: Dict[str, object],
    *,
    image_path: str,
    stage: str,
    prompts: List[str],
    max_instances: int,
) -> Dict[str, List[Dict[str, object]]]:
    repo_root = _runtime_repo_root(runtime_cfg)
    model_family = _runtime_model_family(runtime_cfg, repo_root)
    if model_family != "sam3":
        raise RuntimeError(f"Unsupported model_family={model_family!r}. This runtime only supports SAM3.")
    api = _load_sam3_api(repo_root)
    stage_cfg = _sam3_stage_runtime_config(runtime_cfg, stage)
    bundle = _get_sam3_model_bundle(api, repo_root, stage_cfg)
    pil_image = api["PIL_Image"]
    torch_mod = api["torch"]
    processor = bundle["processor"]
    image = pil_image.open(_runtime_path(image_path)).convert("RGB")
    original_width = int(getattr(image, "width", 0) or 0)
    original_height = int(getattr(image, "height", 0) or 0)
    sam_image, _ = _resize_pil_image_max_side(image, stage_cfg.preprocess_max_side)
    scale_x = float(original_width) / float(max(1, int(getattr(sam_image, "width", original_width) or original_width)))
    scale_y = float(original_height) / float(max(1, int(getattr(sam_image, "height", original_height) or original_height)))
    results: Dict[str, List[Dict[str, object]]] = {}
    use_cuda = str(stage_cfg.device or "").strip().lower().startswith("cuda")
    autocast_ctx = (
        torch_mod.autocast(device_type="cuda", enabled=False)
        if use_cuda and hasattr(torch_mod, "autocast")
        else _NullContext()
    )
    with autocast_ctx:
        base_state = processor.set_image(sam_image)
    for prompt in prompts:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            results[prompt] = []
            continue
        with autocast_ctx:
            state = dict(base_state)
            backbone_out = base_state.get("backbone_out")
            state["backbone_out"] = dict(backbone_out) if isinstance(backbone_out, dict) else backbone_out
            state = processor.set_text_prompt(prompt_text, state=state)
        rows = _extract_rows_from_output(
            {
                "masks": state.get("masks"),
                "boxes": state.get("boxes"),
                "scores": state.get("scores"),
            },
            width=int(state.get("original_width", 0) or 0),
            height=int(state.get("original_height", 0) or 0),
            mask_export=stage_cfg.mask_export,
            max_mask_pixels=stage_cfg.max_mask_pixels,
        )
        rows = _scale_rows_to_original_size(
            rows,
            scale_x=scale_x,
            scale_y=scale_y,
            width=original_width,
            height=original_height,
        )
        results[prompt] = rows[: max(1, int(max_instances))]
    return results


def _load_request(path: str) -> Dict[str, object]:
    payload = _load_json(path)
    return payload


def _request_prompts_from_payload(raw_prompts: Any) -> List[str]:
    prompts: List[str] = []
    if not isinstance(raw_prompts, list):
        raise RuntimeError("request_json.prompts must be a list.")
    for item in raw_prompts:
        if isinstance(item, dict):
            prompt = str(item.get("prompt", "") or "").strip()
        else:
            prompt = str(item or "").strip()
        if prompt:
            prompts.append(prompt)
    return prompts


def _run_single_request(
    runtime_cfg: Dict[str, object],
    request: Dict[str, object],
    *,
    fallback_stage: str,
    fallback_max_instances: int,
) -> List[Dict[str, object]]:
    image_path = _runtime_path(str(request.get("image_path", "") or "").strip())
    if not image_path:
        raise RuntimeError("Missing request.image_path.")
    stage = str(request.get("stage", fallback_stage) or fallback_stage).strip()
    max_instances = int(request.get("max_instances", fallback_max_instances) or fallback_max_instances)
    prompts = _request_prompts_from_payload(request.get("prompts") or [])
    results = _run_prompts(
        runtime_cfg,
        image_path=image_path,
        stage=stage,
        prompts=prompts,
        max_instances=max_instances,
    )
    return [{"prompt": prompt, "records": results.get(prompt, [])} for prompt in prompts]


def _run_jsonl_server(
    runtime_cfg: Dict[str, object],
    *,
    fallback_stage: str,
    fallback_max_instances: int,
) -> int:
    for raw_line in sys.stdin:
        line = str(raw_line or "").strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise RuntimeError("JSONL request must be a JSON object.")
            # Supports either one request or a list of requests (multi-image batch).
            batch_items = request.get("requests")
            if isinstance(batch_items, list):
                batch_out: List[Dict[str, object]] = []
                for idx, item in enumerate(batch_items):
                    if not isinstance(item, dict):
                        continue
                    req_id = str(item.get("request_id", f"req_{idx}") or f"req_{idx}")
                    rows = _run_single_request(
                        runtime_cfg,
                        item,
                        fallback_stage=fallback_stage,
                        fallback_max_instances=fallback_max_instances,
                    )
                    batch_out.append({"request_id": req_id, "results": rows})
                response = {"ok": True, "batch_results": batch_out}
            else:
                rows = _run_single_request(
                    runtime_cfg,
                    request,
                    fallback_stage=fallback_stage,
                    fallback_max_instances=fallback_max_instances,
                )
                response = {"ok": True, "results": rows}
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
        sys.stdout.flush()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run SAM grounding inference and emit IMPACT-SG-compatible JSON.")
    ap.add_argument("--runtime-config", required=True, help="Path to the SAM runtime config JSON.")
    ap.add_argument("--image_path", default="")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--stage", default="category_discovery")
    ap.add_argument("--max_instances", type=int, default=8)
    ap.add_argument("--request_json", default="", help="Optional JSON request containing image_path/stage/max_instances/prompts.")
    ap.add_argument("--serve_jsonl", action="store_true", help="Serve stdin JSONL requests and return one-line JSON responses.")
    args = ap.parse_args()

    runtime_cfg = _load_json(os.path.abspath(os.path.expanduser(args.runtime_config)))

    if bool(args.serve_jsonl):
        return _run_jsonl_server(
            runtime_cfg,
            fallback_stage=str(args.stage or "category_discovery"),
            fallback_max_instances=max(1, int(args.max_instances)),
        )

    if args.request_json:
        request = _load_request(os.path.abspath(os.path.expanduser(args.request_json)))
        payload = _run_single_request(
            runtime_cfg,
            request,
            fallback_stage=str(args.stage or "category_discovery"),
            fallback_max_instances=max(1, int(args.max_instances)),
        )
        print(json.dumps(payload, ensure_ascii=True))
        return 0

    image_path = _runtime_path(str(args.image_path or "").strip())
    prompt = str(args.prompt or "").strip()
    if not image_path:
        raise RuntimeError("Missing --image_path.")
    if not prompt:
        raise RuntimeError("Missing --prompt.")
    results = _run_prompts(
        runtime_cfg,
        image_path=image_path,
        stage=str(args.stage or "category_discovery"),
        prompts=[prompt],
        max_instances=max(1, int(args.max_instances)),
    )
    print(json.dumps(results.get(prompt, []), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
