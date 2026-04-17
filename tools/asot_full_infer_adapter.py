# -*- coding: utf-8 -*-
"""
Full ASOT inference adapter (VideoSSL + ASOT solver).

Loads the full Lightning checkpoint from the ASOT repo, runs the model's forward
path (MLP -> cost matrix -> ASOT solver) on features.npy, and exports:
  - {out_prefix}_per_frame.npy
  - {out_prefix}_segments.json
  - {out_prefix}_segments.txt

This mirrors the training/eval pipeline used in run_*.sh, using the checkpoint's
hyperparameters when available.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError as e:
    raise SystemExit(
        "PyTorch is required for ASOT inference. Please install it and retry."
    ) from e

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASOT_ROOT = os.path.join(REPO_ROOT, "external", "action_seg_ot")
ASOT_SRC = os.path.join(ASOT_ROOT, "src")
if ASOT_SRC not in sys.path:
    sys.path.insert(0, ASOT_SRC)

from train import VideoSSL  # type: ignore  # noqa: E402

try:
    from tools.segment_utils import (
        majority_smooth,
        merge_segments_by_label,
        apply_min_seg_len,
    )
    from tools.asot_label_remap_utils import (
        aggregate_cluster_probs_to_semantic_probs,
        load_asot_label_remap,
        remap_cluster_ids_to_semantic_ids,
        resolve_asot_label_remap_path,
    )
except Exception:
    from segment_utils import (
        majority_smooth,
        merge_segments_by_label,
        apply_min_seg_len,
    )
    from asot_label_remap_utils import (
        aggregate_cluster_probs_to_semantic_probs,
        load_asot_label_remap,
        remap_cluster_ids_to_semantic_ids,
        resolve_asot_label_remap_path,
    )


def strip_prefix(key: str) -> str:
    for px in ("module.", "model.", "net."):
        if key.startswith(px):
            return key[len(px) :]
    return key


def extract_hparams(ckpt_obj: Dict) -> Dict:
    if not isinstance(ckpt_obj, dict):
        return {}
    for key in ("hyper_parameters", "hparams", "config", "args"):
        hp = ckpt_obj.get(key)
        if isinstance(hp, dict):
            return hp
    return {}


def infer_layer_sizes_from_state(state_dict: Dict) -> Optional[List[int]]:
    weights = []
    for k, v in state_dict.items():
        if not k.endswith("weight"):
            continue
        key = strip_prefix(k)
        if not key.startswith("mlp."):
            continue
        parts = key.split(".")
        idx = None
        for p in parts[1:]:
            if p.isdigit():
                idx = int(p)
                break
        if idx is None:
            continue
        if hasattr(v, "shape") and len(v.shape) == 2:
            weights.append((idx, int(v.shape[1]), int(v.shape[0])))
    if not weights:
        return None
    weights.sort(key=lambda x: x[0])
    sizes = [weights[0][1]]
    for _, _, out_dim in weights:
        sizes.append(out_dim)
    return sizes


def infer_n_clusters(state_dict: Dict) -> Optional[int]:
    for k, v in state_dict.items():
        key = strip_prefix(k)
        if key.startswith("clusters") and hasattr(v, "shape"):
            return int(v.shape[0])
    return None


def load_classes(path: Optional[str], num_classes: int) -> List[str]:
    names: List[str] = []
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
    if len(names) < num_classes:
        for i in range(len(names), num_classes):
            names.append(f"cls_{i}")
    return names[:num_classes]


def build_model(ckpt_path: str) -> Tuple[VideoSSL, Dict]:
    obj = torch.load(ckpt_path, map_location="cpu")
    hp = extract_hparams(obj)
    state = (
        obj.get("state_dict") if isinstance(obj, dict) and "state_dict" in obj else obj
    )
    if state is None or not isinstance(state, dict):
        state = {}

    # Try Lightning's helper first.
    try:
        model = VideoSSL.load_from_checkpoint(ckpt_path)
        return model, hp
    except Exception:
        pass

    kwargs = {}
    if isinstance(hp, dict):
        kwargs.update(hp)
    if "layer_sizes" not in kwargs:
        sizes = infer_layer_sizes_from_state(state)
        if sizes:
            kwargs["layer_sizes"] = sizes
    if "n_clusters" not in kwargs:
        nc = infer_n_clusters(state)
        if nc:
            kwargs["n_clusters"] = nc

    model = VideoSSL(**kwargs)
    model.load_state_dict(state, strict=False)
    return model, hp


def main():
    ap = argparse.ArgumentParser(
        description="Run full ASOT (VideoSSL + ASOT solver) inference on features.npy."
    )
    ap.add_argument(
        "--features_dir", required=True, help="Directory containing features.npy"
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        help=".ckpt (Lightning) or .pth state dict from ASOT training",
    )
    ap.add_argument(
        "--class_names", default=None, help="TXT with one class name per line"
    )
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--input_layout",
        choices=["BTD", "BDT"],
        default="BTD",
        help="Feature layout for model input",
    )
    ap.add_argument(
        "--smooth_k",
        type=int,
        default=3,
        help="Odd window size for majority smoothing (1=disable)",
    )
    ap.add_argument(
        "--min_seg_len",
        type=int,
        default=30,
        help="Minimum segment length in frames (<=1 disables)",
    )
    ap.add_argument("--device", default=None, help="cpu / cuda / cuda:0 ...")
    ap.add_argument(
        "--label_remap_json",
        default=None,
        help="Optional JSON remap from ASOT cluster IDs to semantic label IDs/names.",
    )
    ap.add_argument(
        "--out_prefix", default="pred_asot_full", help="Prefix for output files"
    )
    args = ap.parse_args()

    features_dir = os.path.expanduser(args.features_dir)
    ckpt_path = os.path.expanduser(args.ckpt)
    class_txt = os.path.expanduser(args.class_names) if args.class_names else None

    feat_path = os.path.join(features_dir, "features.npy")
    if not os.path.isfile(feat_path):
        raise FileNotFoundError(f"features.npy not found in {features_dir}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, hp = build_model(ckpt_path)
    try:
        print(f"[INFO] Using checkpoint: {ckpt_path}")
    except Exception:
        pass
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device).eval()

    feat = np.load(feat_path)
    if feat.ndim != 2:
        raise ValueError(f"features.npy must be 2D (T x D); got shape {feat.shape}")
    # If layout=BDT and file is (2048, T), transpose to (T, 2048) before downstream checks.
    if args.input_layout == "BDT" and feat.shape[0] == 2048 and feat.shape[1] != 2048:
        feat = feat.T
    T, D = int(feat.shape[0]), int(feat.shape[1])

    expected_dim = None
    try:
        expected_dim = int(model.layer_sizes[0])
    except Exception:
        expected_dim = None
    if expected_dim:
        print(f"[INFO] Model expects feature dim = {expected_dim}")
    if expected_dim and D != expected_dim:
        if D > expected_dim:
            print(
                f"[WARN] feature dim {D} > expected {expected_dim}; slicing to first {expected_dim} dims"
            )
            feat = feat[:, :expected_dim]
        else:
            print(f"[WARN] feature dim {D} < expected {expected_dim}; padding zeros")
            pad = np.zeros((T, expected_dim - D), dtype=feat.dtype)
            feat = np.concatenate([feat, pad], axis=1)
        D = expected_dim

    # Prepare tensor with requested layout.
    x = torch.from_numpy(feat).float()
    if args.input_layout == "BTD":
        x = x.unsqueeze(0)  # B=1, T, D
    else:  # BDT
        x = x.t().unsqueeze(0)  # B=1, D, T
    x = x.to(device)

    with torch.no_grad():
        soft = model(x)
        while soft.dim() > 3:
            soft = soft.squeeze(0)
        if soft.dim() == 2:
            soft = soft.unsqueeze(0)
        probs = torch.softmax(soft, dim=-1)
        pred_idx = probs.argmax(dim=-1).squeeze(0).cpu().numpy()

    cluster_pred_idx = majority_smooth(pred_idx, k=args.smooth_k).astype(np.int32, copy=False)
    cluster_probs_np = probs.squeeze(0).cpu().numpy().astype(np.float32, copy=False)  # [T, C]

    # Determine number of classes from model clusters if present.
    num_classes = (
        model.n_clusters
        if hasattr(model, "n_clusters")
        else int(hp.get("n_clusters", cluster_pred_idx.max() + 1))
    )

    # Prefer class_names provided by user; else try mapping.txt relative to features_dir; else default.
    default_classes = os.path.join(ASOT_ROOT, "class_names.txt")
    cand_map = [
        os.path.join(features_dir, "..", "mapping", "mapping.txt"),
        os.path.join(features_dir, "../..", "mapping", "mapping.txt"),
    ]
    if class_txt:
        print(f"[INFO] Using class_names: {class_txt}")
        classes = load_classes(class_txt, num_classes=num_classes)
        class_bank_path = class_txt
    else:
        chosen = ""
        for c in cand_map:
            c = os.path.abspath(c)
            if os.path.isfile(c):
                chosen = c
                break
        if chosen:
            print(f"[INFO] Using class_names: {chosen}")
        else:
            print(f"[INFO] Using default class_names: {default_classes}")
        classes = load_classes(chosen or default_classes, num_classes=num_classes)
        class_bank_path = chosen or default_classes

    pred_idx = np.asarray(cluster_pred_idx, dtype=np.int32)
    probs_np = np.asarray(cluster_probs_np, dtype=np.float32)
    remap_path = resolve_asot_label_remap_path(
        explicit_path=(args.label_remap_json or ""),
        class_names_path=(class_bank_path or ""),
        features_dir=features_dir,
        repo_root=REPO_ROOT,
    )
    remap_meta: Dict[str, object] = {
        "label_remap_applied": False,
        "label_remap_path": "",
        "label_remap_clusters_applied": 0,
        "cluster_count": int(num_classes),
        "semantic_class_count": int(len(classes)),
    }
    if remap_path:
        semantic_classes = load_classes(class_bank_path) if class_bank_path else list(classes)
        remap = load_asot_label_remap(
            remap_path,
            semantic_classes or classes,
            num_clusters=num_classes,
        )
        if remap:
            classes = list(remap.get("semantic_classes") or semantic_classes or classes)
            probs_np = aggregate_cluster_probs_to_semantic_probs(
                cluster_probs_np,
                remap.get("cluster_to_semantic_idx") or [],
                len(classes),
            )
            pred_idx = remap_cluster_ids_to_semantic_ids(
                cluster_pred_idx,
                remap.get("cluster_to_semantic_idx") or [],
                len(classes),
            )
            remap_meta = {
                "label_remap_applied": True,
                "label_remap_path": str(remap.get("path") or remap_path),
                "label_remap_clusters_applied": int(remap.get("clusters_applied", 0) or 0),
                "cluster_count": int(num_classes),
                "semantic_class_count": int(len(classes)),
            }
            print(
                "[INFO] Applied ASOT label remap: "
                f"{os.path.basename(str(remap_meta['label_remap_path']))} "
                f"({int(remap_meta['label_remap_clusters_applied'])} mapped clusters)"
            )
    if args.min_seg_len and int(args.min_seg_len) > 1:
        pred_idx = apply_min_seg_len(pred_idx, probs_np, int(args.min_seg_len))

    segments = merge_segments_by_label(pred_idx, classes, fallback_prefix="cls_")
    # attach per-segment top-k (avg probs in segment)
    for seg in segments:
        s = int(seg["start_frame"])
        e = int(seg["end_frame"])
        s = max(0, min(s, probs_np.shape[0] - 1))
        e = max(s, min(e, probs_np.shape[0] - 1))
        seg_probs = probs_np[s : e + 1]
        mean_probs = seg_probs.mean(axis=0) if seg_probs.size else probs_np[0]
        k = min(5, mean_probs.shape[0])
        topk_idx = mean_probs.argsort()[::-1][:k]
        seg["topk"] = [
            {
                "id": int(i),
                "name": classes[i] if i < len(classes) else f"cls_{i}",
                "score": float(mean_probs[i]),
            }
            for i in topk_idx
        ]

    np.save(
        os.path.join(features_dir, f"{args.out_prefix}_cluster_per_frame.npy"),
        np.asarray(cluster_pred_idx, dtype=np.int32),
    )
    np.save(os.path.join(features_dir, f"{args.out_prefix}_per_frame.npy"), pred_idx)
    with open(
        os.path.join(features_dir, f"{args.out_prefix}_segments.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "segments": segments,
                "classes": classes,
                "meta": remap_meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(
        os.path.join(features_dir, f"{args.out_prefix}_segments.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        for s in segments:
            f.write(f"{s['start_frame']}\n{s['end_frame']}\n{s['class_name']}\n\n")

    print(
        f"[OK] ASOT full-model inference done. Segments written to {args.out_prefix}_segments.* under {features_dir}"
    )


if __name__ == "__main__":
    main()
