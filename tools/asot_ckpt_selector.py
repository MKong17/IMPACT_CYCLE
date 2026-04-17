# -*- coding: utf-8 -*-
"""
Heuristic checkpoint selector for ASOT.

Given a list/glob of checkpoints and a features directory, run ASOT inference
and compute label/segment diversity metrics to pick a checkpoint that is more
useful for auto-annotation (avoid collapsed late-stage models).

Usage:
  python tools/asot_ckpt_selector.py \
    --features_dir /path/to/features_dir \
    --ckpt_glob "external/action_seg_ot/wandb/video_ssl/**/checkpoints/*.ckpt" \
    --class_names external/learningcell_front/mapping/mapping.txt \
    --target_avg_seg 200

Outputs a ranked list with label count, segment count, boundary density, and
suggests the top checkpoint.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

# Reuse inference helpers from asot_full_infer_adapter
from tools.asot_full_infer_adapter import build_model, merge_segments, load_classes


def run_infer(
    ckpt: str, features_dir: str, class_txt: str, input_layout: str = "BTD"
) -> Tuple[np.ndarray, List[str]]:
    model, hp = build_model(ckpt)
    feat_path = os.path.join(features_dir, "features.npy")
    feat = np.load(feat_path)
    if feat.ndim != 2:
        raise ValueError(f"features.npy must be 2D, got {feat.shape}")

    expected_dim = getattr(model, "layer_sizes", [feat.shape[1]])[0]
    if feat.shape[1] > expected_dim:
        feat = feat[:, :expected_dim]
    elif feat.shape[1] < expected_dim:
        pad = np.zeros((feat.shape[0], expected_dim - feat.shape[1]), dtype=feat.dtype)
        feat = np.concatenate([feat, pad], axis=1)

    x = torch.from_numpy(feat).float()
    if input_layout == "BTD":
        x = x.unsqueeze(0)
    else:
        x = x.t().unsqueeze(0)
    x = x.to(
        next(model.parameters()).device
        if any(p.device.type == "cuda" for p in model.parameters())
        and torch.cuda.is_available()
        else "cpu"
    )
    model.eval()
    with torch.no_grad():
        soft = model(x)
        while soft.dim() > 3:
            soft = soft.squeeze(0)
        if soft.dim() == 2:
            soft = soft.unsqueeze(0)
        pred = soft.argmax(dim=-1).squeeze(0).cpu().numpy()

    num_classes = (
        model.n_clusters
        if hasattr(model, "n_clusters")
        else int(hp.get("n_clusters", pred.max() + 1))
    )
    classes = load_classes(class_txt, num_classes=num_classes)
    return pred, classes


def summarize(
    pred: np.ndarray, classes: List[str], target_avg_seg: float = 200.0
) -> Dict:
    segs = merge_segments(pred, classes)
    uniq = np.unique(pred)
    seg_count = max(1, len(segs))
    avg_len = float(len(pred) / seg_count)
    boundary_density = float(max(0, seg_count - 1)) / float(len(pred))
    # heuristic score: more labels + closeness of avg_len to target
    label_score = len(uniq)
    length_score = -abs(avg_len - target_avg_seg) / target_avg_seg
    score = label_score + length_score
    return {
        "labels_used": len(uniq),
        "segments": seg_count,
        "avg_seg_len": avg_len,
        "boundary_density": boundary_density,
        "score": score,
        "uniq_ids": uniq.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Heuristic selector for ASOT checkpoints (diversity-oriented)."
    )
    ap.add_argument("--features_dir", required=True, help="Dir containing features.npy")
    ap.add_argument(
        "--ckpt_glob",
        required=True,
        help='Glob for ckpts, e.g., "external/action_seg_ot/wandb/video_ssl/**/checkpoints/*.ckpt"',
    )
    ap.add_argument(
        "--class_names", required=True, help="TXT with class names (mapping)"
    )
    ap.add_argument(
        "--target_avg_seg",
        type=float,
        default=200.0,
        help="Target average segment length (frames)",
    )
    ap.add_argument("--input_layout", choices=["BTD", "BDT"], default="BTD")
    args = ap.parse_args()

    ckpts = sorted(glob.glob(args.ckpt_glob, recursive=True))
    if not ckpts:
        raise FileNotFoundError(f"No ckpts matched glob: {args.ckpt_glob}")

    results = []
    for ck in ckpts:
        try:
            pred, classes = run_infer(
                ck, args.features_dir, args.class_names, input_layout=args.input_layout
            )
            summary = summarize(pred, classes, target_avg_seg=args.target_avg_seg)
            summary["ckpt"] = ck
            results.append(summary)
            print(f"[CKPT] {ck}")
            print(
                f"  labels_used={summary['labels_used']} segments={summary['segments']} avg_len={summary['avg_seg_len']:.1f} boundary={summary['boundary_density']:.4f} score={summary['score']:.3f}"
            )
        except Exception as e:
            print(f"[SKIP] {ck} due to {e}")

    if not results:
        print("No valid results.")
        return

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    out = {
        "best": best,
        "all": results,
    }
    rec_path = os.path.join(args.features_dir, "ckpt_selection.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n[RECOMMEND] ckpt =", best["ckpt"])
    print(
        f"labels_used={best['labels_used']} segments={best['segments']} avg_len={best['avg_seg_len']:.1f} boundary={best['boundary_density']:.4f}"
    )
    print(f"Saved report to {rec_path}")


if __name__ == "__main__":
    main()
