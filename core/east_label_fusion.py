from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


SOURCE_NAMES: Tuple[str, ...] = (
    "base",
    "text_prior",
    "class_table",
    "prototype",
)


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    mat = np.asarray(arr, dtype=np.float32)
    if mat.ndim != 2 or mat.size == 0:
        return mat.astype(np.float32, copy=False)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return (mat / norms).astype(np.float32, copy=False)


def softmax_rows(arr: np.ndarray) -> np.ndarray:
    mat = np.asarray(arr, dtype=np.float32)
    if mat.ndim != 2 or mat.size == 0:
        return mat.astype(np.float32, copy=False)
    mat = mat - np.max(mat, axis=1, keepdims=True)
    exp = np.exp(mat)
    exp_sum = np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-6)
    return (exp / exp_sum).astype(np.float32, copy=False)


def build_gate_features(
    base_scores: np.ndarray,
    segment_lengths: Sequence[int],
    *,
    text_available: bool,
    class_available: bool,
    proto_available: bool,
) -> np.ndarray:
    base = np.asarray(base_scores, dtype=np.float32)
    if base.ndim != 2:
        raise ValueError("base_scores must be 2D")
    n = int(base.shape[0])
    lens = np.asarray([max(1, int(x)) for x in segment_lengths], dtype=np.float32).reshape(-1)
    if lens.size != n:
        raise ValueError("segment_lengths must align with base_scores rows")
    row_sum = np.maximum(base.sum(axis=1, keepdims=True), 1e-6)
    probs = base / row_sum
    conf = np.max(probs, axis=1)
    top2 = np.partition(probs, kth=max(0, probs.shape[1] - 2), axis=1)[:, -2:] if probs.shape[1] >= 2 else np.concatenate([probs, probs], axis=1)
    top1 = np.max(top2, axis=1)
    second = np.min(top2, axis=1)
    margin = top1 - second
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-6)), axis=1)
    if probs.shape[1] > 1:
        entropy = entropy / np.log(float(probs.shape[1]))
    length_norm = np.log1p(lens) / np.log1p(max(float(np.max(lens)), 2.0))
    features = np.stack(
        [
            conf.astype(np.float32),
            margin.astype(np.float32),
            entropy.astype(np.float32),
            length_norm.astype(np.float32),
            np.full((n,), 1.0 if text_available else 0.0, dtype=np.float32),
            np.full((n,), 1.0 if class_available else 0.0, dtype=np.float32),
            np.full((n,), 1.0 if proto_available else 0.0, dtype=np.float32),
        ],
        axis=1,
    )
    return features.astype(np.float32, copy=False)


def apply_fusion_gate(
    source_probs: Dict[str, np.ndarray],
    gate_weight: np.ndarray,
    gate_bias: np.ndarray,
    gate_features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if not source_probs:
        raise ValueError("source_probs is required")
    first = next(iter(source_probs.values()))
    first_arr = np.asarray(first, dtype=np.float32)
    n, k = int(first_arr.shape[0]), int(first_arr.shape[1])
    feat = np.asarray(gate_features, dtype=np.float32)
    if feat.ndim != 2 or feat.shape[0] != n:
        raise ValueError("gate_features must be [N, F]")
    weight = np.asarray(gate_weight, dtype=np.float32)
    bias = np.asarray(gate_bias, dtype=np.float32).reshape(-1)
    if weight.ndim != 2 or weight.shape[0] != len(SOURCE_NAMES) or weight.shape[1] != feat.shape[1]:
        raise ValueError("invalid gate_weight shape")
    if bias.size != len(SOURCE_NAMES):
        raise ValueError("invalid gate_bias shape")
    logits = feat @ weight.T + bias.reshape(1, -1)
    mix = softmax_rows(logits)
    fused = np.zeros((n, k), dtype=np.float32)
    for idx, name in enumerate(SOURCE_NAMES):
        row = np.asarray(source_probs.get(name), dtype=np.float32)
        if row.ndim != 2 or row.shape != (n, k):
            continue
        fused += mix[:, idx : idx + 1] * row
    fused = fused / np.maximum(fused.sum(axis=1, keepdims=True), 1e-6)
    return fused.astype(np.float32, copy=False), mix.astype(np.float32, copy=False)
