# -*- coding: utf-8 -*-
"""
Backbone-agnostic feature extraction helpers for per-frame video features.
"""
from __future__ import annotations

import json
import os
import threading
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".m4v")
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.RLock()

try:
    _autocast = torch.amp.autocast
    _autocast_kwargs = {"device_type": "cuda"}
except AttributeError:
    _autocast = torch.cuda.amp.autocast
    _autocast_kwargs = {}


def _resize_shorter(batch: torch.Tensor, size: int, mode: str) -> torch.Tensor:
    if batch.ndim != 4:
        raise ValueError(f"Expected 4D tensor (B,C,H,W), got {batch.shape}")
    _, _, h, w = batch.shape
    if min(h, w) == size:
        return batch
    if h < w:
        new_h = size
        new_w = int(round(w * size / h))
    else:
        new_w = size
        new_h = int(round(h * size / w))
    return F.interpolate(batch, size=(new_h, new_w), mode=mode, align_corners=False)


def _center_crop(batch: torch.Tensor, size: int) -> torch.Tensor:
    if batch.ndim != 4:
        raise ValueError(f"Expected 4D tensor (B,C,H,W), got {batch.shape}")
    _, _, h, w = batch.shape
    if h == size and w == size:
        return batch
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return batch[:, :, top : top + size, left : left + size]


def _to_tensor(
    frames: Union[np.ndarray, torch.Tensor, List[np.ndarray]]
) -> torch.Tensor:
    if isinstance(frames, torch.Tensor):
        tensor = frames
    elif isinstance(frames, np.ndarray):
        tensor = torch.from_numpy(frames)
    elif isinstance(frames, list):
        if not frames:
            raise ValueError("Empty frame list.")
        tensor = torch.from_numpy(np.stack(frames, axis=0))
    else:
        raise TypeError(f"Unsupported frame type: {type(frames)}")

    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor (T,C,H,W), got {tensor.shape}")

    if tensor.shape[1] != 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 3, 1, 2)
    return tensor


class BaseFeatureExtractor(ABC):
    name = "base"
    feature_dim: int = 0
    resize_shorter: int = 224
    crop_size: int = 224
    interpolate_mode: str = "bilinear"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(
        self, device: Optional[str] = None, batch_size: int = 128, use_fp16: bool = True
    ):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = max(1, int(batch_size))
        self.use_fp16 = bool(use_fp16) and self.device.type == "cuda"
        self.model = self._get_cached_model()
        self.model.eval()
        self.model.requires_grad_(False)

    def _model_cache_key(self) -> Tuple[str, str]:
        return (str(getattr(self, "name", self.__class__.__name__)), str(self.device))

    def _get_cached_model(self) -> torch.nn.Module:
        key = self._model_cache_key()
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                self._reused_model = True
                return cached
            model = self._load_model().to(self.device)
            model.eval()
            model.requires_grad_(False)
            _MODEL_CACHE[key] = model
            self._reused_model = False
            return model

    @abstractmethod
    def _load_model(self) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _preprocess(self, batch: torch.Tensor) -> torch.Tensor:
        batch = _resize_shorter(batch, self.resize_shorter, mode=self.interpolate_mode)
        batch = _center_crop(batch, self.crop_size)
        mean = torch.as_tensor(self.mean, device=batch.device).view(1, 3, 1, 1)
        std = torch.as_tensor(self.std, device=batch.device).view(1, 3, 1, 1)
        return (batch - mean) / std

    def _encode_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.dtype != torch.float32:
            batch = batch.float()
        if batch.max() > 1.5:
            batch = batch / 255.0
        batch = batch.to(self.device, non_blocking=True)
        batch = self._preprocess(batch)
        with torch.no_grad():
            if self.use_fp16 and self.device.type == "cuda":
                with _autocast(**_autocast_kwargs):
                    out = self._forward(batch)
            else:
                out = self._forward(batch)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, dict):
            out = next(iter(out.values()))
        if out.dim() == 3:
            out = out.mean(dim=1)
        return out.detach().cpu()

    def extract_frames(
        self, frames: Union[np.ndarray, torch.Tensor, List[np.ndarray]]
    ) -> np.ndarray:
        """
        Extract per-frame features from input frames (T, C, H, W) in RGB order.
        Returns a float32 array of shape (T, D).
        """
        tensor = _to_tensor(frames)
        if tensor.numel() == 0:
            return np.zeros((0, int(self.feature_dim)), dtype=np.float32)
        feats = []
        for i in range(0, tensor.shape[0], self.batch_size):
            batch = tensor[i : i + self.batch_size]
            feats.append(self._encode_batch(batch))
        if not feats:
            return np.zeros((0, int(self.feature_dim)), dtype=np.float32)
        feat_tensor = torch.cat(feats, dim=0)
        return feat_tensor.numpy().astype(np.float32, copy=False)

    def extract_from_video(
        self,
        video_path: str,
        frame_stride: int = 1,
        max_frames: Optional[int] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[np.ndarray, List[int], int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        use_total = total if (max_frames is None) else min(total, int(max_frames))
        if use_total <= 0:
            use_total = None

        stride = max(1, int(frame_stride))
        expected = 0
        if use_total is not None:
            expected = (use_total + stride - 1) // stride
        feats = []
        picked_indices: List[int] = []
        batch_frames: List[np.ndarray] = []

        if progress_cb:
            try:
                progress_cb(0, expected)
            except Exception:
                pass

        idx = 0
        while True:
            if use_total is not None and idx >= use_total:
                break
            ret, frame = cap.read()
            if not ret:
                break
            if stride > 1 and (idx % stride) != 0:
                idx += 1
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame)
            picked_indices.append(idx)
            if len(batch_frames) >= self.batch_size:
                batch = _to_tensor(batch_frames)
                feats.append(self._encode_batch(batch))
                batch_frames = []
                if progress_cb:
                    try:
                        progress_cb(len(picked_indices), expected)
                    except Exception:
                        pass
            idx += 1

        if batch_frames:
            batch = _to_tensor(batch_frames)
            feats.append(self._encode_batch(batch))
            if progress_cb:
                try:
                    progress_cb(len(picked_indices), expected)
                except Exception:
                    pass

        cap.release()
        total_frames = total if total > 0 else idx

        if not feats:
            empty = np.zeros((0, int(self.feature_dim)), dtype=np.float32)
            return empty, picked_indices, total_frames

        feat_tensor = torch.cat(feats, dim=0)
        if progress_cb:
            try:
                progress_cb(len(picked_indices), expected or len(picked_indices))
            except Exception:
                pass
        return (
            feat_tensor.numpy().astype(np.float32, copy=False),
            picked_indices,
            total_frames,
        )


class DinoV2FeatureExtractor(BaseFeatureExtractor):
    name = "dinov2_vitb14"
    feature_dim = 768
    resize_shorter = 224
    crop_size = 224
    interpolate_mode = "bicubic"

    def _load_model(self) -> torch.nn.Module:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        return model

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.model(batch)
        if isinstance(out, dict):
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            if "x_norm_patchtokens" in out:
                return out["x_norm_patchtokens"].mean(dim=1)
            if "x_norm" in out:
                return out["x_norm"]
            return next(iter(out.values()))
        return out


class ResNet50FeatureExtractor(BaseFeatureExtractor):
    name = "resnet50"
    feature_dim = 2048
    resize_shorter = 256
    crop_size = 224
    interpolate_mode = "bilinear"

    def _load_model(self) -> torch.nn.Module:
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        return model

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)


def build_feature_extractor(
    backbone: Optional[str],
    device: Optional[str] = None,
    batch_size: int = 128,
    use_fp16: bool = True,
) -> BaseFeatureExtractor:
    key = (backbone or "dinov2_vitb14").lower()
    if key in ("dinov2_vitb14", "dinov2", "dino", "dino_v2"):
        return DinoV2FeatureExtractor(
            device=device, batch_size=batch_size, use_fp16=use_fp16
        )
    if key in ("resnet50", "resnet"):
        return ResNet50FeatureExtractor(
            device=device, batch_size=batch_size, use_fp16=use_fp16
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def clear_feature_extractor_cache() -> None:
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def extract_video_features(
    video_path: str,
    backbone: Optional[str] = None,
    batch_size: int = 128,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    device: Optional[str] = None,
    use_fp16: bool = True,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, dict]:
    extractor = build_feature_extractor(
        backbone=backbone,
        device=device,
        batch_size=batch_size,
        use_fp16=use_fp16,
    )
    features, picked, total = extractor.extract_from_video(
        video_path,
        frame_stride=frame_stride,
        max_frames=max_frames,
        progress_cb=progress_cb,
    )
    meta = {
        "backbone": extractor.name,
        "feature_dim": int(extractor.feature_dim),
        "frame_stride": int(frame_stride),
        "picked_indices": picked,
        "num_frames": int(total),
        "input_size": int(extractor.crop_size),
        "model_cached": bool(getattr(extractor, "_reused_model", False)),
    }
    return features, meta


def save_features(
    features_dir: str, features: np.ndarray, meta: Optional[dict] = None
) -> str:
    os.makedirs(features_dir, exist_ok=True)
    feat_path = os.path.join(features_dir, "features.npy")
    np.save(feat_path, features.astype(np.float32, copy=False))
    if isinstance(meta, dict):
        meta_path = os.path.join(features_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)
    return feat_path
