# -*- coding: utf-8 -*-
"""
ASOT inference adapter.

Reads features.npy from a feature directory, runs the packaged ASOT MLP checkpoint,
and writes:
  - {out_prefix}_per_frame.npy
  - {out_prefix}_segments.json  (start/end in frames)
  - {out_prefix}_segments.txt   (three-line-per-segment)

Default paths assume the repo layout in this tool; override via CLI if needed.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise SystemExit(
        "PyTorch is required for ASOT inference. Please install it and retry."
    ) from e


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASOT_ROOT = os.path.join(REPO_ROOT, "external", "action_seg_ot")

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


def standardize_features(feat: np.ndarray) -> np.ndarray:
    """Mirror the per-video standardization in the ASOT dataset loader."""
    if feat.ndim != 2:
        return feat
    mask = np.ones(feat.shape[0], dtype=bool)
    for idx, row in enumerate(feat):
        if np.sum(row) == 0:
            mask[idx] = False
    out = np.zeros_like(feat)
    if mask.any():
        z = feat[mask]
        z = z - np.mean(z, axis=0)
        std = np.std(z, axis=0)
        std[std == 0] = 1.0
        z = z / std
        out[mask] = z
    out = np.nan_to_num(out)
    out /= np.sqrt(max(1, out.shape[1]))
    return out


def extract_hparams(ckpt_obj: Dict) -> Dict:
    if not isinstance(ckpt_obj, dict):
        return {}
    for key in ("hyper_parameters", "hparams", "config", "args"):
        hp = ckpt_obj.get(key)
        if isinstance(hp, dict):
            return hp
    return {}


def infer_layer_sizes_from_state_dict(state_dict: Dict) -> Optional[List[int]]:
    """Try to reconstruct layer_sizes from mlp.* weight shapes."""
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


def has_nested_first_layer(state_dict: Dict) -> bool:
    keys = [strip_prefix(k) for k in state_dict.keys()]
    return any(k.startswith("mlp.0.0.") for k in keys)


def build_mlp(layer_sizes: List[int], nested0: bool) -> nn.Module:
    layers = []
    for i in range(len(layer_sizes) - 1):
        lin = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
        if i < len(layer_sizes) - 2:
            block = nn.Sequential(lin, nn.ReLU())
        else:
            block = lin
        layers.append(block)

    seq = nn.Sequential()
    for i, block in enumerate(layers):
        name = str(i)
        seq.add_module(name, block)

    model = nn.Sequential()
    # Keep the "mlp" prefix so load_state_dict with mlp.* keys works.
    model.add_module("mlp", seq)
    # If checkpoint uses nested naming (mlp.0.0.*), wrap the first block in another Sequential.
    if nested0:
        first = seq[0]
        if not isinstance(first, nn.Sequential):
            seq[0] = nn.Sequential(first)
    return model


def load_mlp_weights(model: nn.Module, state_dict: Dict):
    sd = {strip_prefix(k): v for k, v in state_dict.items()}
    mlp_sd = {k: v for k, v in sd.items() if k.startswith("mlp.")}
    if not mlp_sd:
        raise KeyError("No mlp.* keys found in checkpoint.")
    model.load_state_dict(mlp_sd, strict=False)


def load_classes(path: Optional[str], num_classes: int) -> List[str]:
    names: List[str] = []
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
    if len(names) < num_classes:
        # pad fallback names to match the output dimension
        for i in range(len(names), num_classes):
            names.append(f"cls_{i}")
    return names[:num_classes]


def main():
    ap = argparse.ArgumentParser(
        description="Run ASOT MLP inference on features.npy and export segments."
    )
    ap.add_argument(
        "--features_dir", required=True, help="Directory containing features.npy"
    )
    ap.add_argument("--ckpt", required=True, help="ASOT checkpoint (.ckpt or .pth)")
    ap.add_argument(
        "--class_names", default=None, help="TXT with one class name per line"
    )
    ap.add_argument("--device", default=None, help="cpu / cuda / cuda:0 ...")
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
    ap.add_argument(
        "--standardize",
        action="store_true",
        help="Apply the dataset-style per-video standardization",
    )
    ap.add_argument(
        "--label_remap_json",
        default=None,
        help="Optional JSON remap from ASOT cluster IDs to semantic label IDs/names.",
    )
    ap.add_argument("--out_prefix", default="pred_asot", help="Prefix for output files")
    args = ap.parse_args()

    features_dir = os.path.expanduser(args.features_dir)
    ckpt_path = os.path.expanduser(args.ckpt)
    class_names_path = (
        os.path.expanduser(args.class_names) if args.class_names else None
    )

    feat_path = os.path.join(features_dir, "features.npy")
    assert os.path.isfile(feat_path), f"features.npy not found in {features_dir}"
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = extract_hparams(ckpt)

    state_dict = ckpt.get("state_dict") if isinstance(ckpt, dict) else ckpt
    if state_dict is None or not isinstance(state_dict, dict):
        state_dict = ckpt if isinstance(ckpt, dict) else {}

    layer_sizes = hp.get("layer_sizes") if isinstance(hp, dict) else None
    if not layer_sizes:
        layer_sizes = infer_layer_sizes_from_state_dict(state_dict)
    if not layer_sizes:
        raise ValueError(
            "Unable to infer layer_sizes from checkpoint; please provide a Lightning .ckpt with hyper_parameters."
        )

    nested0 = has_nested_first_layer(state_dict)
    model = build_mlp(layer_sizes, nested0)
    load_mlp_weights(model, state_dict)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device).eval()

    feat = np.load(feat_path)
    T, D = int(feat.shape[0]), int(feat.shape[1])
    in_dim = int(layer_sizes[0])
    if D != in_dim:
        if D > in_dim:
            print(
                f"[WARN] feature dim {D} > expected {in_dim}; slicing to first {in_dim} dims"
            )
            feat = feat[:, :in_dim]
        else:
            print(f"[WARN] feature dim {D} < expected {in_dim}; padding zeros")
            pad = np.zeros((T, in_dim - D), dtype=feat.dtype)
            feat = np.concatenate([feat, pad], axis=1)
    if args.standardize:
        feat = standardize_features(feat)

    x = torch.from_numpy(feat).float().to(device)
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1).cpu().numpy()

    cluster_pred_sm = majority_smooth(pred, k=args.smooth_k).astype(np.int32, copy=False)
    cluster_probs_np = probs.detach().cpu().numpy().astype(np.float32, copy=False)
    num_classes = int(layer_sizes[-1])
    default_class_txt = os.path.join(ASOT_ROOT, "class_names.txt")
    class_bank_path = class_names_path or default_class_txt
    classes = load_classes(class_bank_path, num_classes=num_classes)
    pred_sm = np.asarray(cluster_pred_sm, dtype=np.int32)
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
            pred_sm = remap_cluster_ids_to_semantic_ids(
                cluster_pred_sm,
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
        pred_sm = apply_min_seg_len(pred_sm, probs_np, int(args.min_seg_len))
    segments = merge_segments_by_label(pred_sm, classes, fallback_prefix="cls_")

    np.save(
        os.path.join(features_dir, f"{args.out_prefix}_cluster_per_frame.npy"),
        np.asarray(cluster_pred_sm, dtype=np.int32),
    )
    np.save(os.path.join(features_dir, f"{args.out_prefix}_per_frame.npy"), pred_sm)
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
        f"[OK] ASOT inference done. Segments written to {args.out_prefix}_segments.* under {features_dir}"
    )


if __name__ == "__main__":
    main()
