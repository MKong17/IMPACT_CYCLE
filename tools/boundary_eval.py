#!/usr/bin/env python3
"""
Boundary evaluation tool.

Inputs:
  - features.npy (T x D or D x T) in a features_dir, OR
  - a single features .npy file (ASRF datasets store per-video features)
  - GT labels (frame-wise labels or segments) in .npy/.json/.txt

Methods:
  - m0: adjacent-frame cosine distance (baseline energy peak)
  - m1: SegSim (segmentation-only, boundary from label changes)
  - m3: ASRF (boundary scores/logits provided by user)

Outputs:
  - Boundary F1@delta for multiple deltas
  - Optional saved scores and predicted boundary frames

Examples (ASRF dataset layout):
  python tools/boundary_eval.py \\
    --features_path external/ASRF/dataset/50salads/features/rgb-01-1.npy \\
    --gt external/ASRF/dataset/50salads/groundTruth/rgb-01-1.txt \\
    --method m0

  python tools/boundary_eval.py \\
    --features_path /home/yinqian/IsaacDrive/YQ_MasterThesis/AS_Tool/cvhci-video-annotation-suite/external/ASRF/dataset/50salads/features/rgb-01-1.npy \\
    --gt /home/yinqian/IsaacDrive/YQ_MasterThesis/AS_Tool/cvhci-video-annotation-suite/external/ASRF/dataset/50salads/groundTruth/rgb-01-1.txt \\
    --method m0

  python tools/boundary_eval.py \\
    --manifest_csv external/ASRF/csv/gtea/train3.csv \\
    --csv_gt_col label \\
    --methods m0,m1,m3

  python tools/boundary_eval.py \\
    --manifest_csv external/ASRF/csv/gtea/train3.csv \\
    --csv_gt_col label \\
    --methods m0,m1,m3 \\
    --peak_mode relmax \\
    --threshold_quantile 0.9

  python tools/boundary_eval.py \\
    --manifest_csv external/ASRF/csv/gtea/train3.csv \\
    --csv_gt_col label \\
    --methods m0,m1,m3 \\
    --peak_mode relmax \\
    --threshold_quantile 0.9 \\
    --plot_dir plots/boundary_debug \\
    --plot_max 5

  python tools/boundary_eval.py \\
    --manifest_csv external/ASRF/csv/gtea/train3.csv \\
    --csv_gt_col label \\
    --methods m0,m1,m3 \\
    --asrf_eval
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_txt_labels(path: str) -> List[str]:
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labels.append(line)
    return labels


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    smin = float(np.nanmin(scores))
    smax = float(np.nanmax(scores))
    if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
        return np.zeros_like(scores)
    return (scores - smin) / (smax - smin)


def _smooth_1d(scores: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or scores.size == 0:
        return scores
    win = int(win)
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(scores, kernel, mode="same")


def _parse_scales(text: str) -> List[int]:
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            items.append(int(part))
        except Exception:
            continue
    return [x for x in items if x > 0] or [1]


def _parse_methods(text: str) -> List[str]:
    if not text:
        return []
    allowed = {"m0", "m1", "m3", "asrf"}
    out: List[str] = []
    for part in text.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key == "asrf":
            key = "m3"
        if key not in allowed:
            raise ValueError(f"Unknown method: {key}")
        if key not in out:
            out.append(key)
    return out


def _infer_feat_layout(feat: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> np.ndarray:
    if feat.ndim != 2:
        raise ValueError(f"features.npy must be 2D, got {feat.shape}")
    if meta and isinstance(meta.get("feature_dim"), (int, float)):
        dim = int(meta.get("feature_dim"))
        if feat.shape[0] == dim and feat.shape[1] != dim:
            return feat.T
        if feat.shape[1] == dim and feat.shape[0] != dim:
            return feat
    typical_dims = {256, 512, 768, 1024, 1536, 2048, 3072, 4096}
    if feat.shape[0] in typical_dims and feat.shape[1] not in typical_dims:
        return feat.T
    if feat.shape[1] in typical_dims and feat.shape[0] not in typical_dims:
        return feat
    return feat


def _load_meta(features_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(features_dir, "meta.json")
    if os.path.isfile(meta_path):
        try:
            return _load_json(meta_path)
        except Exception:
            return {}
    return {}


def _build_frame_map(seq_len: int, meta: Optional[Dict[str, Any]] = None, gt_num_frames: Optional[int] = None) -> List[int]:
    if gt_num_frames is not None and seq_len == gt_num_frames:
        return list(range(seq_len))
    if meta:
        picked = meta.get("picked_indices")
        if isinstance(picked, list) and len(picked) == seq_len:
            try:
                return [int(x) for x in picked]
            except Exception:
                pass
        stride = meta.get("frame_stride")
        if stride is not None:
            try:
                stride = int(stride)
                return [i * stride for i in range(seq_len)]
            except Exception:
                pass
    return list(range(seq_len))


def _boundaries_from_frame_labels(labels: List[Any]) -> List[int]:
    if not labels:
        return []
    b = []
    prev = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != prev:
            b.append(i)
            prev = labels[i]
    return b


def _segments_from_payload(payload: Dict[str, Any], fps: Optional[float], segments_in_seconds: bool) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    view_start = int(payload.get("view_start", 0) or 0)

    if "segments" in payload:
        raw = payload.get("segments") or []
        for item in raw:
            if isinstance(item, dict):
                s = item.get("start_frame", item.get("start"))
                e = item.get("end_frame", item.get("end"))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                s, e = item[0], item[1]
            else:
                continue
            if s is None or e is None:
                continue
            try:
                s_val = float(s)
                e_val = float(e)
            except Exception:
                continue
            if segments_in_seconds:
                if fps is None:
                    raise ValueError("segments_in_seconds=True but fps is not provided.")
                s_val = s_val * fps
                e_val = e_val * fps
            s_fr = int(round(s_val)) + view_start
            e_fr = int(round(e_val)) + view_start
            segs.append((s_fr, e_fr))
        return segs

    if "annotations" in payload:
        if fps is None:
            raise ValueError("ActivityNet-style annotations require fps to convert seconds to frames.")
        for ann in payload.get("annotations", []):
            for seg in ann.get("segments", []) or []:
                if not (isinstance(seg, (list, tuple)) and len(seg) >= 2):
                    continue
                s_val, e_val = float(seg[0]), float(seg[1])
                s_fr = int(round(s_val * fps))
                e_fr = int(round(e_val * fps))
                segs.append((s_fr, e_fr))
        return segs

    return segs


def _boundaries_from_segments(segs: List[Tuple[int, int]]) -> List[int]:
    if not segs:
        return []
    segs_sorted = sorted(segs, key=lambda x: (x[0], x[1]))
    b = []
    for idx, (s, _e) in enumerate(segs_sorted):
        if idx == 0:
            continue
        if s > 0:
            b.append(int(s))
    return sorted(set(b))


def _load_gt_boundaries(
    gt_path: str,
    fps: Optional[float],
    segments_in_seconds: bool,
    gt_is_boundary: bool = False,
) -> Tuple[List[int], int]:
    ext = os.path.splitext(gt_path)[1].lower()
    if ext == ".npy":
        arr = np.load(gt_path, allow_pickle=True)
        if arr.ndim != 1:
            raise ValueError(f"GT npy must be 1D, got shape {arr.shape}")
        if gt_is_boundary:
            idx = np.where(arr.astype(np.int64) > 0)[0]
            return idx.astype(int).tolist(), int(arr.shape[0])
        labels = arr.tolist()
        b = _boundaries_from_frame_labels(labels)
        return b, len(labels)
    if ext in (".txt",):
        labels = _load_txt_labels(gt_path)
        b = _boundaries_from_frame_labels(labels)
        return b, len(labels)
    if ext in (".json",):
        payload = _load_json(gt_path)
        if isinstance(payload, dict) and isinstance(payload.get("frame_labels"), list):
            labels = payload.get("frame_labels") or []
            b = _boundaries_from_frame_labels(labels)
            return b, len(labels)
        segs = _segments_from_payload(payload, fps=fps, segments_in_seconds=segments_in_seconds)
        if segs:
            max_end = max(e for _s, e in segs)
            b = _boundaries_from_segments(segs)
            return b, max_end + 1
        raise ValueError("Unsupported GT json format (no frame_labels / segments / annotations).")
    raise ValueError(f"Unsupported GT file extension: {ext}")


def _cosine_distance_adjacent(feat_td: np.ndarray) -> np.ndarray:
    if feat_td.size == 0:
        return np.zeros((0,), dtype=np.float32)
    v = feat_td.astype(np.float32, copy=False)
    norms = np.linalg.norm(v, axis=1)
    norms[norms == 0] = 1.0
    v = v / norms[:, None]
    dots = np.sum(v[1:] * v[:-1], axis=1)
    scores = 0.5 * (1.0 - dots)
    scores = np.concatenate([[0.0], scores], axis=0)
    scores = np.nan_to_num(scores)
    return scores.astype(np.float32, copy=False)


def _multi_scale_cosine(feat_td: np.ndarray, scales: List[int]) -> np.ndarray:
    if feat_td.size == 0:
        return np.zeros((0,), dtype=np.float32)
    v = feat_td.astype(np.float32, copy=False)
    norms = np.linalg.norm(v, axis=1)
    norms[norms == 0] = 1.0
    v = v / norms[:, None]
    T = v.shape[0]
    accum = np.zeros((T,), dtype=np.float32)
    count = np.zeros((T,), dtype=np.float32)
    for s in scales:
        if s <= 0 or s >= T:
            continue
        dots = np.sum(v[s:] * v[:-s], axis=1)
        scores = 0.5 * (1.0 - dots)
        padded = np.zeros((T,), dtype=np.float32)
        padded[s:] = scores
        accum += padded
        count += (padded != 0).astype(np.float32)
    count[count == 0] = 1.0
    out = accum / count
    out[0] = 0.0
    out = np.nan_to_num(out)
    return out.astype(np.float32, copy=False)


def _load_boundary_logits(path: str, apply_sigmoid: bool) -> np.ndarray:
    logits = np.load(path)
    if logits.ndim == 0:
        logits = logits.reshape(1)
    if logits.ndim > 1:
        # if shape (T, C) or (C, T), take max over channel axis
        if logits.shape[0] < logits.shape[-1]:
            logits = logits.max(axis=0)
        else:
            logits = logits.max(axis=-1)
    logits = logits.astype(np.float32, copy=False)
    if apply_sigmoid:
        logits = 1.0 / (1.0 + np.exp(-logits))
    return logits


def _load_boundary_binary(path: str) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        # reduce to 1D if needed
        if arr.shape[0] < arr.shape[-1]:
            arr = arr.max(axis=0)
        else:
            arr = arr.max(axis=-1)
    return (arr > 0).astype(np.float32)


def _thin_boundaries(boundaries: List[int], min_sep: int) -> List[int]:
    if min_sep <= 0 or len(boundaries) <= 1:
        return sorted(boundaries)
    boundaries = sorted(boundaries)
    kept = [boundaries[0]]
    for b in boundaries[1:]:
        if b - kept[-1] > min_sep:
            kept.append(b)
    return kept


def _pick_peaks(
    scores: np.ndarray,
    frame_map: List[int],
    min_sep: int,
    topk: Optional[int],
    threshold: Optional[float],
    peak_mode: str,
) -> List[int]:
    if scores.size == 0:
        return []
    cand: List[int] = []
    if peak_mode == "relmax":
        for i in range(1, len(scores) - 1):
            if scores[i] <= scores[i - 1] or scores[i] <= scores[i + 1]:
                continue
            if threshold is not None and scores[i] < threshold:
                continue
            cand.append(i)
    else:
        for i in range(1, len(scores)):
            if threshold is not None and scores[i] < threshold:
                continue
            cand.append(i)
    if not cand:
        return []
    cand.sort(key=lambda i: (-float(scores[i]), i))
    if topk is not None:
        cand = cand[: max(0, int(topk))]
    picked_frames: List[int] = []
    for idx in cand:
        frame = int(frame_map[idx]) if idx < len(frame_map) else int(idx)
        if not picked_frames:
            picked_frames.append(frame)
            continue
        if all(abs(frame - pf) > min_sep for pf in picked_frames):
            picked_frames.append(frame)
    return sorted(set(picked_frames))


def _match_boundaries(pred: List[int], gt: List[int], delta: int) -> Tuple[int, int, int, List[int]]:
    pred = sorted(pred)
    gt = sorted(gt)
    used = [False] * len(gt)
    offsets: List[int] = []
    for p in pred:
        lo = p - delta
        hi = p + delta
        start = bisect.bisect_left(gt, lo)
        end = bisect.bisect_right(gt, hi)
        best = None
        for j in range(start, end):
            if used[j]:
                continue
            d = abs(gt[j] - p)
            if best is None or d < best[0]:
                best = (d, j)
        if best is not None:
            used[best[1]] = True
            offsets.append(int(best[0]))
    tp = len(offsets)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn, offsets


def _eval_one(pred: List[int], gt: List[int], deltas: List[int]) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    for d in deltas:
        tp, fp, fn, offsets = _match_boundaries(pred, gt, int(d))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        mean_off = float(np.mean(offsets)) if offsets else None
        results[int(d)] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "mean_offset": mean_off,
        }
    return results


def _accumulate_results(agg: Dict[int, Dict[str, Any]], res: Dict[int, Dict[str, Any]]) -> None:
    for d, r in res.items():
        slot = agg.setdefault(int(d), {"tp": 0, "fp": 0, "fn": 0, "offsets": []})
        slot["tp"] += int(r.get("tp", 0))
        slot["fp"] += int(r.get("fp", 0))
        slot["fn"] += int(r.get("fn", 0))
        offs = r.get("offsets")
        if isinstance(offs, list):
            slot["offsets"].extend(offs)


def _format_metrics(tp: int, fp: int, fn: int, offsets: Optional[List[int]] = None) -> Dict[str, Any]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    mean_off = float(np.mean(offsets)) if offsets else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mean_offset": mean_off,
    }


def _resolve_threshold(
    scores: np.ndarray,
    threshold: Optional[float],
    threshold_std: Optional[float],
    threshold_quantile: Optional[float],
) -> Optional[float]:
    if scores.size == 0:
        return threshold
    if threshold_quantile is not None:
        q = float(threshold_quantile)
        q = min(max(q, 0.0), 1.0)
        return float(np.quantile(scores, q))
    if threshold_std is not None:
        mean = float(np.mean(scores))
        std = float(np.std(scores))
        return mean + float(threshold_std) * std
    return threshold


def _sanitize_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "item"


def _score_at_frame(frame_map: List[int], scores: np.ndarray, frame: int) -> Optional[float]:
    if not frame_map:
        return None
    idx = bisect.bisect_left(frame_map, frame)
    if idx >= len(frame_map):
        idx = len(frame_map) - 1
    if idx > 0:
        prev = idx - 1
        if abs(frame_map[prev] - frame) <= abs(frame_map[idx] - frame):
            idx = prev
    if idx < 0 or idx >= len(scores):
        return None
    return float(scores[idx])


def _save_plots(
    plot_dir: str,
    plots: Dict[int, Dict[str, Any]],
    methods: List[str],
    stride: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib not available; skip plotting.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    for idx, pdata in plots.items():
        name = _sanitize_name(pdata.get("id", f"item_{idx}"))
        gt = pdata.get("gt") or []
        method_data = pdata.get("methods") or {}
        n_rows = max(1, len(methods))
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.2 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]
        for ax, method in zip(axes, methods):
            entry = method_data.get(method)
            if not entry:
                ax.set_title(f"{method}: (no data)")
                continue
            scores = entry.get("scores")
            frame_map = entry.get("frame_map") or []
            pred = entry.get("pred") or []
            thr = entry.get("threshold")
            if scores is None:
                ax.set_title(f"{method}: (missing scores)")
                continue
            if stride < 1:
                stride = 1
            if frame_map and len(frame_map) >= len(scores):
                xs = frame_map[: len(scores): stride]
            else:
                xs = list(range(0, len(scores), stride))
            ys = scores[::stride]
            ax.plot(xs, ys, linewidth=1)
            # GT boundaries
            for b in gt:
                ax.axvline(b, color="#2ca02c", alpha=0.2, linewidth=1)
            # Pred boundaries
            if pred:
                py = []
                px = []
                for p in pred:
                    y = _score_at_frame(frame_map, scores, int(p))
                    if y is None:
                        continue
                    px.append(int(p))
                    py.append(y)
                if px:
                    ax.scatter(px, py, s=18, marker="x", color="#d62728", label="pred")
            if thr is not None:
                ax.axhline(float(thr), color="#ff7f0e", linestyle="--", linewidth=1, label="thr")
            ax.set_title(f"{method}: pred={len(pred)} gt={len(gt)}")
            ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        out_path = os.path.join(plot_dir, f"{idx:03d}_{name}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)


def _load_manifest(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    payload = _load_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("items") or []
    return []


def _load_manifest_csv(
    path: str,
    feat_col: str,
    gt_col: str,
    pred_col: Optional[str] = None,
    logits_col: Optional[str] = None,
) -> List[Dict[str, Any]]:
    import csv

    items = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat = row.get(feat_col)
            gt = row.get(gt_col)
            if not feat or not gt:
                continue
            item = {"features_path": feat, "gt": gt}
            if pred_col:
                pred = row.get(pred_col)
                if pred:
                    item["boundary_pred"] = pred
            if logits_col:
                logits = row.get(logits_col)
                if logits:
                    item["boundary_logits"] = logits
            items.append(item)
    return items


def _auto_find_gt(features_dir: str) -> Optional[str]:
    for name in (
        "gt_frame_labels.npy",
        "gt_frame_labels.json",
        "gt_frame_labels.txt",
        "gt_segments.json",
        "gt.json",
    ):
        path = os.path.join(features_dir, name)
        if os.path.isfile(path):
            return path
    return None


def _compute_scores(
    method: str,
    features_dir: str,
    features_path: Optional[str],
    meta: Dict[str, Any],
    scales: List[int],
    smooth: int,
    normalize: bool,
    boundary_logits: Optional[str],
    apply_sigmoid: bool,
) -> Tuple[np.ndarray, List[int]]:
    if method == "m0":
        feat_path = features_path or os.path.join(features_dir, "features.npy")
        if not os.path.isfile(feat_path):
            raise FileNotFoundError(f"features file not found: {feat_path}")
        feat = np.load(feat_path, mmap_mode="r")
        feat_td = _infer_feat_layout(feat, meta=meta)
        scores = _cosine_distance_adjacent(feat_td)
        if smooth > 1:
            scores = _smooth_1d(scores, smooth)
        if normalize:
            scores = _normalize_scores(scores)
        frame_map = _build_frame_map(int(feat_td.shape[0]), meta=meta, gt_num_frames=None)
        return scores, frame_map
    if method == "m3":
        if not boundary_logits:
            raise ValueError("m3 (asrf) requires --boundary_logits")
        logits = _load_boundary_logits(boundary_logits, apply_sigmoid=apply_sigmoid)
        scores = logits
        if smooth > 1:
            scores = _smooth_1d(scores, smooth)
        if normalize:
            scores = _normalize_scores(scores)
        frame_map = _build_frame_map(int(scores.shape[0]), meta=meta, gt_num_frames=None)
        return scores, frame_map
    raise ValueError(f"Unknown method: {method}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Boundary evaluation for M0-M3 (energy, SegSim, ASRF).")
    default_deltas = "1,2,5,10,20"
    ap.add_argument("--features_dir", type=str, help="Directory containing features.npy and meta.json")
    ap.add_argument("--features_path", type=str, help="Single features .npy file (ASRF datasets)")
    ap.add_argument("--gt", type=str, help="GT labels path (.npy/.json/.txt)")
    ap.add_argument("--manifest", type=str, help="JSON/JSONL list of items with features_dir and gt")
    ap.add_argument("--manifest_csv", type=str, help="CSV list of items (ASRF format: feature,label,boundary)")
    ap.add_argument("--csv_feat_col", type=str, default="feature", help="CSV column for features path")
    ap.add_argument("--csv_gt_col", type=str, default="label", help="CSV column for GT path")
    ap.add_argument("--csv_pred_col", type=str, default=None, help="CSV column for boundary_pred (m1)")
    ap.add_argument(
        "--csv_logits_col",
        type=str,
        default=None,
        help="CSV column for boundary logits/probabilities (m3)",
    )
    ap.add_argument("--method", type=str, default="m0", choices=["m0", "m1", "m3", "asrf"])
    ap.add_argument("--methods", type=str, default=None, help="Comma-separated methods, e.g. m0,m1,m3")
    ap.add_argument("--delta_frames", type=str, default=default_deltas, help="Comma-separated list")
    ap.add_argument("--min_sep", type=int, default=5, help="Min separation (frames) for peak picking")
    ap.add_argument("--topk", type=int, default=None, help="Pick top-K peaks")
    ap.add_argument("--topk_ratio", type=float, default=None, help="Pick K = ratio * |B_gt|")
    ap.add_argument("--threshold", type=float, default=None, help="Score threshold for peaks (absolute)")
    ap.add_argument("--threshold_std", type=float, default=None, help="Threshold = mean + k*std")
    ap.add_argument("--threshold_quantile", type=float, default=None, help="Threshold at quantile (0-1)")
    ap.add_argument("--peak_mode", type=str, default="rank", choices=["rank", "relmax"], help="Peak picking mode")
    ap.add_argument("--asrf_eval", action="store_true", help="ASRF-style eval (m3): threshold=0.7, peak_mode=relmax, delta_frames=5, min_sep=1")
    ap.add_argument("--plot_dir", type=str, default=None, help="Save per-video plots to this directory")
    ap.add_argument("--plot_max", type=int, default=5, help="Max number of videos to plot")
    ap.add_argument("--plot_stride", type=int, default=1, help="Downsample points when plotting")
    ap.add_argument("--smooth", type=int, default=3, help="Smoothing window size (m0/m3)")
    ap.add_argument("--scales", type=str, default="1,2,4", help="Scales for m0 (unused by m1/m3)")
    ap.add_argument("--no_normalize", action="store_true", help="Disable score normalization")
    ap.add_argument("--boundary_pred", type=str, default=None, help="Path to boundary_pred .npy (m1)")
    ap.add_argument("--boundary_logits", type=str, default=None, help="Path to boundary logits/probabilities (m3)")
    ap.add_argument("--apply_sigmoid", action="store_true", help="Apply sigmoid to logits")
    ap.add_argument("--fps", type=float, default=None, help="FPS for converting seconds to frames")
    ap.add_argument("--segments_in_seconds", action="store_true", help="Interpret segments as seconds")
    ap.add_argument("--gt_is_boundary", action="store_true", help="GT npy is boundary (0/1) instead of labels")
    ap.add_argument("--save_scores", type=str, default=None, help="Save scores to .npy")
    ap.add_argument("--save_pred", type=str, default=None, help="Save predicted boundary frames to .json")
    ap.add_argument("--save_report", type=str, default=None, help="Save metrics to .json")
    args = ap.parse_args(argv)

    delta_frames_input = args.delta_frames
    deltas = _parse_scales(args.delta_frames)
    scales = _parse_scales(args.scales)
    normalize = not bool(args.no_normalize)
    try:
        methods = _parse_methods(args.methods) if args.methods else _parse_methods(args.method)
    except ValueError as exc:
        ap.error(str(exc))

    items: List[Dict[str, Any]] = []
    if args.manifest:
        items = _load_manifest(args.manifest)
    elif args.manifest_csv:
        items = _load_manifest_csv(
            args.manifest_csv,
            args.csv_feat_col,
            args.csv_gt_col,
            args.csv_pred_col,
            args.csv_logits_col,
        )
    else:
        if not args.features_dir and not args.features_path:
            ap.error("--features_dir or --features_path is required when not using --manifest")
        if args.features_dir:
            gt_path = args.gt or _auto_find_gt(args.features_dir)
        else:
            gt_path = args.gt
        if not gt_path:
            ap.error("GT path not provided and auto-detection failed.")
        items = [{"features_dir": args.features_dir, "features_path": args.features_path, "gt": gt_path}]

    def _suffix_path(path: str, suffix: str) -> str:
        root, ext = os.path.splitext(path)
        return f"{root}_{suffix}{ext or ''}"

    all_reports: Dict[str, Any] = {}
    plot_cache: Dict[int, Dict[str, Any]] = {}

    for method in methods:
        method_deltas = list(deltas)
        method_threshold = args.threshold
        method_threshold_std = args.threshold_std
        method_threshold_quantile = args.threshold_quantile
        method_peak_mode = args.peak_mode
        method_min_sep = args.min_sep

        if args.asrf_eval and method == "m3":
            method_threshold = 0.7
            method_threshold_std = None
            method_threshold_quantile = None
            method_peak_mode = "relmax"
            method_min_sep = 1
            method_deltas = [5]

        agg: Dict[int, Dict[str, Any]] = {}
        per_item_results: List[Dict[str, Any]] = []
        last_scores = None
        last_pred = None

        for idx, item in enumerate(items):
            features_dir = os.path.expanduser(item.get("features_dir") or "")
            features_path = item.get("features_path")
            if features_path:
                features_path = os.path.expanduser(features_path)
                if not features_dir:
                    features_dir = os.path.dirname(features_path)
            gt_path = os.path.expanduser(item.get("gt") or "")
            if (not features_dir and not features_path) or not gt_path:
                continue

            meta = _load_meta(features_dir) if features_dir else {}
            gt_boundaries, gt_num_frames = _load_gt_boundaries(
                gt_path,
                fps=args.fps,
                segments_in_seconds=args.segments_in_seconds,
                gt_is_boundary=args.gt_is_boundary,
            )

            boundary_pred = item.get("boundary_pred") or args.boundary_pred
            boundary_logits = item.get("boundary_logits") or args.boundary_logits

            if method == "m1":
                pred_path = boundary_pred or boundary_logits
                if not pred_path:
                    raise ValueError("m1 (segsim-seg) requires boundary_pred: use --boundary_pred/--csv_pred_col")
                boundary_binary = _load_boundary_binary(pred_path)
                scores = boundary_binary.astype(np.float32, copy=False)
                frame_map = _build_frame_map(len(scores), meta=meta, gt_num_frames=gt_num_frames)
                pred_frames = [frame_map[i] for i in np.where(boundary_binary > 0)[0]]
                pred_frames = _thin_boundaries(pred_frames, method_min_sep)
                threshold = None
            else:
                scores, frame_map = _compute_scores(
                    method=method,
                    features_dir=features_dir,
                    features_path=features_path,
                    meta=meta,
                    scales=scales,
                    smooth=args.smooth,
                    normalize=normalize,
                    boundary_logits=boundary_logits,
                    apply_sigmoid=args.apply_sigmoid,
                )
                # rebuild frame_map with gt length when possible
                frame_map = _build_frame_map(len(scores), meta=meta, gt_num_frames=gt_num_frames)

                topk = args.topk
                if args.topk_ratio is not None:
                    if len(gt_boundaries) == 0:
                        topk = 0
                    else:
                        topk = max(1, int(round(args.topk_ratio * len(gt_boundaries))))

                threshold = _resolve_threshold(scores, method_threshold, method_threshold_std, method_threshold_quantile)
                pred_frames = _pick_peaks(
                    scores,
                    frame_map,
                    min_sep=method_min_sep,
                    topk=topk,
                    threshold=threshold,
                    peak_mode=method_peak_mode,
                )

            last_scores = scores
            last_pred = pred_frames
            res = _eval_one(pred_frames, gt_boundaries, method_deltas)

            # attach offsets for aggregation
            for d in res:
                tp, fp, fn, offs = _match_boundaries(pred_frames, gt_boundaries, int(d))
                res[d]["offsets"] = offs
                _accumulate_results(agg, {d: {"tp": tp, "fp": fp, "fn": fn, "offsets": offs}})

            per_item_results.append({
                "features_dir": features_dir or features_path,
                "gt": gt_path,
                "gt_boundaries": len(gt_boundaries),
                "pred_boundaries": len(pred_frames),
                "metrics": res,
            })

            if args.plot_dir and idx < int(args.plot_max):
                base_id = os.path.splitext(os.path.basename(gt_path))[0]
                if not base_id and features_path:
                    base_id = os.path.splitext(os.path.basename(features_path))[0]
                entry = plot_cache.setdefault(idx, {"id": base_id, "gt": gt_boundaries, "methods": {}})
                entry["gt"] = gt_boundaries
                entry["methods"][method] = {
                    "scores": scores,
                    "frame_map": frame_map,
                    "pred": pred_frames,
                    "threshold": threshold,
                }

        report = {"method": method, "deltas": method_deltas, "items": per_item_results}
        if agg:
            agg_metrics = {}
            for d, slot in agg.items():
                agg_metrics[int(d)] = _format_metrics(slot["tp"], slot["fp"], slot["fn"], slot.get("offsets"))
            report["aggregate"] = agg_metrics
        all_reports[method] = report

        # output per method (optional)
        if args.save_scores and last_scores is not None:
            out_path = _suffix_path(args.save_scores, method) if len(methods) > 1 else args.save_scores
            np.save(out_path, last_scores.astype(np.float32, copy=False))
        if args.save_pred and last_pred is not None:
            out_path = _suffix_path(args.save_pred, method) if len(methods) > 1 else args.save_pred
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"pred_boundaries": last_pred}, f, ensure_ascii=True, indent=2)

    # save report
    final_report: Dict[str, Any]
    if len(methods) == 1:
        final_report = all_reports[methods[0]]
    else:
        final_report = {"methods": methods, "reports": all_reports}

    if args.save_report:
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=True, indent=2)

    if args.plot_dir:
        _save_plots(args.plot_dir, plot_cache, methods, stride=args.plot_stride)

    # print summary
    for method in methods:
        report = all_reports[method]
        report_deltas = report.get("deltas", deltas)
        if len(methods) > 1:
            print(f"== Method: {method} ==")
        if "aggregate" in report:
            print("Aggregate metrics:")
            for d in report_deltas:
                m = report["aggregate"].get(int(d), {})
                print(f"  F1@{d}: {m.get('f1', 0):.4f}  P={m.get('precision', 0):.4f}  R={m.get('recall', 0):.4f}  mean_off={m.get('mean_offset')}")
        else:
            for it in report["items"]:
                print(f"[{it['features_dir']}] GT={it['gt_boundaries']} Pred={it['pred_boundaries']}")
                for d in report_deltas:
                    m = it["metrics"].get(int(d), {})
                    print(f"  F1@{d}: {m.get('f1', 0):.4f}  P={m.get('precision', 0):.4f}  R={m.get('recall', 0):.4f}  mean_off={m.get('mean_offset')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
