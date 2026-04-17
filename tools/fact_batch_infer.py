# -*- coding: utf-8 -*-
"""
Batch FACT inference on raw videos (offline predictions only).

Outputs one JSON per video to the specified output directory:
  {video_id}.json  -> {"video_id": ..., "segments": [...], "classes": [...]}
"""

import argparse
import json
import os
import sys
from typing import List

import numpy as np
import torch

try:
    from tools.segment_utils import majority_smooth, merge_segments_by_label
except Exception:
    from segment_utils import majority_smooth, merge_segments_by_label


VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".m4v")


def list_videos(video_dir: str) -> List[str]:
    items = []
    for name in os.listdir(video_dir):
        if name.lower().endswith(VIDEO_EXTS):
            items.append(os.path.join(video_dir, name))
    return sorted(items)


def load_class_names(path: str) -> List[str]:
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if parts and parts[0].isdigit() and len(parts) > 1:
                name = " ".join(parts[1:])
            else:
                name = ln
            names.append(name)
    return names


def run_fact_inference(model, X: np.ndarray, device: torch.device):
    T, _D = X.shape
    seq = torch.from_numpy(X).float().to(device)
    seq = seq.unsqueeze(1)
    with torch.no_grad():
        model._forward_one_video(seq, transcript=None)
        last_block = model.block_list[-1]
        pred = last_block.eval(transcript=None)
        pred = pred.detach().cpu().numpy()
    return pred


def build_fact_model(
    fact_repo: str,
    fact_cfg: str,
    ckpt_path: str,
    n_classes: int,
    in_dim: int,
    device: torch.device,
):
    if fact_repo not in sys.path:
        sys.path.insert(0, fact_repo)
    from fact.models.blocks import FACT
    from fact.configs.utils import setup_cfg

    if not os.path.isabs(fact_cfg):
        fact_cfg = os.path.join(fact_repo, fact_cfg)
    if not os.path.isfile(fact_cfg):
        raise FileNotFoundError(f"FACT cfg not found: {fact_cfg}")
    cfg = setup_cfg([fact_cfg], None)
    model = FACT(cfg, in_dim=in_dim, n_classes=n_classes).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model_sd = model.state_dict()
    pruned_state = {}
    mismatched = []
    for k, v in state.items():
        if k not in model_sd:
            continue
        if model_sd[k].shape != v.shape:
            mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        pruned_state[k] = v
    if mismatched:
        print("[WARN] pruned mismatched keys:", mismatched)
    print("[INFO] loaded params:", len(pruned_state))
    model.load_state_dict(pruned_state, strict=False)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser(description="Run FACT batch inference on raw videos.")
    ap.add_argument(
        "--video_dir", required=True, help="Directory containing input videos."
    )
    ap.add_argument(
        "--output_dir", required=True, help="Directory to write prediction JSONs."
    )
    ap.add_argument("--fact_repo", required=True, help="Path to local FACT repository.")
    ap.add_argument("--fact_cfg", required=True, help="Path to FACT cfg file.")
    ap.add_argument("--ckpt", required=True, help="Path to FACT checkpoint (.pth/.pt).")
    ap.add_argument(
        "--class_names",
        required=True,
        help="TXT with one class name per line (mapping).",
    )
    ap.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for feature extraction."
    )
    ap.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Frame stride for feature extraction.",
    )
    ap.add_argument(
        "--smooth_k",
        type=int,
        default=1,
        help="Odd window size for majority smoothing.",
    )
    ap.add_argument("--device", type=str, default=None, help="cpu / cuda / cuda:0 ...")
    ap.add_argument(
        "--no_fp16", action="store_true", help="Disable FP16 autocast on CUDA."
    )
    args = ap.parse_args()

    video_dir = os.path.expanduser(args.video_dir)
    output_dir = os.path.expanduser(args.output_dir)
    fact_repo = os.path.expanduser(args.fact_repo)
    ckpt_path = os.path.expanduser(args.ckpt)
    class_names_path = os.path.expanduser(args.class_names)

    if not os.path.isdir(video_dir):
        raise SystemExit(f"Video dir not found: {video_dir}")
    if not os.path.isdir(fact_repo):
        raise SystemExit(f"FACT repo not found: {fact_repo}")
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isfile(class_names_path):
        raise SystemExit(f"class_names not found: {class_names_path}")

    os.makedirs(output_dir, exist_ok=True)
    videos = list_videos(video_dir)
    if not videos:
        raise SystemExit(f"No videos found in {video_dir}")

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tools_dir = os.path.join(repo_root, "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from extract_resnet50_feats import video_to_feats

    classes = load_class_names(class_names_path)
    n_classes = len(classes)
    if n_classes <= 0:
        raise SystemExit("class_names is empty.")

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = build_fact_model(
        fact_repo, args.fact_cfg, ckpt_path, n_classes, in_dim=2048, device=device
    )

    print(f"[INFO] Found {len(videos)} videos.")
    for idx, vpath in enumerate(videos, 1):
        base = os.path.splitext(os.path.basename(vpath))[0] or f"video_{idx:04d}"
        out_json = os.path.join(output_dir, f"{base}.json")
        print(f"[INFO] ({idx}/{len(videos)}) {os.path.basename(vpath)} -> {out_json}")

        feats = video_to_feats(
            vpath,
            batch_size=args.batch_size,
            frame_stride=max(1, args.frame_stride),
            use_fp16=not args.no_fp16,
        )
        if feats.size == 0:
            print(f"[WARN] no features extracted for {vpath}")
            continue

        X = feats
        if X.ndim == 2 and X.shape[0] == 2048 and X.shape[1] != 2048:
            X = X.T
        elif X.ndim == 2 and X.shape[1] == 2048:
            pass
        elif X.ndim == 2 and X.shape[0] < X.shape[1]:
            X = X.T
        else:
            print(f"[WARN] unexpected feature shape {X.shape}, attempting transpose")
            if X.ndim == 2:
                X = X.T

        if X.shape[1] != 2048:
            print(
                f"[WARN] feature dim {X.shape[1]} != 2048; slicing/padding may be needed"
            )
            X = (
                X[:, :2048]
                if X.shape[1] > 2048
                else np.pad(X, ((0, 0), (0, 2048 - X.shape[1])), mode="constant")
            )

        pred = run_fact_inference(model, X, device)
        pred_sm = majority_smooth(pred, k=args.smooth_k)
        segs = merge_segments_by_label(
            pred_sm, classes, fallback_prefix="c", max_len=150
        )

        payload = {"video_id": base, "segments": segs, "classes": classes}
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] FACT batch inference done. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
