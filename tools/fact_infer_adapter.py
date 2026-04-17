# -*- coding: utf-8 -*-
"""
fact_infer_adapter.py  (FACT-only 简化版)

功能：
- 从 features_dir 读取 features.npy
- 在 FACT 官方仓库中加载 FACT 模型 + cfg + ckpt
- 对整段特征做一次无标签推理（假定 cfg.FACT.trans == False）
- 得到每帧的 action id 序列
- 做一次多数平滑（可选）
- 合并成段，输出：
    - pred_fact_per_frame.npy
    - pred_fact_segments.json
    - pred_fact_segments.txt   （三行一段，给你的 GUI 用）
"""

import os
import sys
import json
import argparse
from typing import List

import numpy as np
import torch

try:
    from tools.segment_utils import majority_smooth, merge_segments_by_label
except Exception:
    from segment_utils import majority_smooth, merge_segments_by_label


def load_classes(features_dir: str) -> List[str]:
    """
    Try to read classes.txt. If not present, create dummy c0..c{C-1}
    based on labels.npy if available, else default C=10.
    """
    p = os.path.join(features_dir, "classes.txt")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        return names

    ly = os.path.join(features_dir, "labels.npy")
    if os.path.isfile(ly):
        y = np.load(ly, mmap_mode="r")
        C = int(y.max()) + 1 if y.size > 0 else 1
    else:
        C = 10
    return [f"c{i}" for i in range(C)]


def run_fact_inference(model, X: np.ndarray, device: torch.device):
    """
    Run FACT model in inference-only mode (no labels).

    Assumes cfg.FACT.trans == False, so transcript is not required.
    X: numpy array [T, D]
    Returns: numpy array [T] of per-frame action ids.
    """
    T, D = X.shape
    seq = torch.from_numpy(X).float().to(device)
    seq = seq.unsqueeze(1)

    with torch.no_grad():
        model._forward_one_video(seq, transcript=None)
        last_block = model.block_list[-1]
        pred = last_block.eval(transcript=None)
        pred = pred.detach().cpu().numpy()
    return pred


def main():
    ap = argparse.ArgumentParser(
        description="Run FACT inference on features.npy and export segments (FACT-only)."
    )
    ap.add_argument(
        "--features_dir", type=str, required=True, help="Dir containing features.npy"
    )
    ap.add_argument(
        "--fact_repo",
        type=str,
        required=True,
        help="Path to local FACT repository (added to sys.path)",
    )
    ap.add_argument(
        "--fact_cfg",
        type=str,
        required=True,
        help="Path to FACT cfg file (relative to repo or absolute)",
    )
    ap.add_argument(
        "--ckpt", type=str, required=True, help="Path to FACT checkpoint (.pth/.pt)"
    )
    ap.add_argument(
        "--fact_in_dim",
        type=int,
        default=None,
        help="Feature dimension D; default=features.npy.shape[1]",
    )
    ap.add_argument("--device", type=str, default=None, help="cpu / cuda / cuda:0 ...")
    ap.add_argument(
        "--smooth_k",
        type=int,
        default=1,
        help="Odd window size for majority smoothing (1=disable)",
    )
    ap.add_argument(
        "--out_prefix", type=str, default="pred_fact", help="Prefix for output files"
    )
    ap.add_argument(
        "--class_names",
        type=str,
        default=None,
        help="TXT with one class name per line (mapping).",
    )
    args = ap.parse_args()

    features_dir = os.path.expanduser(args.features_dir)
    fact_repo = os.path.expanduser(args.fact_repo)
    ckpt_path = os.path.expanduser(args.ckpt)
    cfg_path = os.path.expanduser(args.fact_cfg)

    fx = os.path.join(features_dir, "features.npy")
    assert os.path.isfile(fx), f"features.npy not found in {features_dir}"
    assert os.path.isdir(fact_repo), f"FACT repo not found: {fact_repo}"
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    if fact_repo not in sys.path:
        sys.path.insert(0, fact_repo)

    from fact.models.blocks import FACT
    from fact.configs.utils import setup_cfg

    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(fact_repo, cfg_path)
    assert os.path.isfile(cfg_path), f"FACT cfg not found: {cfg_path}"

    print("[DEBUG] Using FACT cfg:", cfg_path)
    cfg = setup_cfg([cfg_path], None)

    X = np.load(fx)
    T, D = X.shape

    if D != 2048:
        print(f"[DEBUG] FACT: input feature dim = {D}, slicing to 2048 to match model")
        X = X[:, :2048]
        D = 2048

    in_dim = D
    class_names_path = (
        os.path.expanduser(args.class_names) if args.class_names else None
    )
    if class_names_path and os.path.isfile(class_names_path):
        with open(class_names_path, "r", encoding="utf-8") as f:
            classes = [ln.strip() for ln in f if ln.strip()]
        print(
            f"[DEBUG] FACT: using class_names {class_names_path} ({len(classes)} classes)"
        )
    else:
        classes = load_classes(features_dir)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    NUM_CLASSES = len(classes)
    model = FACT(cfg, in_dim=in_dim, n_classes=NUM_CLASSES).to(device)

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

    print("[DEBUG] pruned mismatched keys:", mismatched)
    print("[DEBUG] finally loaded params:", len(pruned_state))
    model.load_state_dict(pruned_state, strict=False)

    pred = run_fact_inference(model, X, device)
    pred_sm = majority_smooth(pred, k=args.smooth_k)
    segs = merge_segments_by_label(pred_sm, classes, fallback_prefix="c", max_len=150)

    np.save(os.path.join(features_dir, f"{args.out_prefix}_per_frame.npy"), pred_sm)
    with open(
        os.path.join(features_dir, f"{args.out_prefix}_segments.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {"segments": segs, "classes": classes}, f, ensure_ascii=False, indent=2
        )
    with open(
        os.path.join(features_dir, f"{args.out_prefix}_segments.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        for s in segs:
            f.write(f"{s['start_frame']}\n{s['end_frame']}\n{s['class_name']}\n\n")

    print(
        f"[INFO] FACT inference done. Segments written to {args.out_prefix}_segments.* under {features_dir}"
    )


if __name__ == "__main__":
    main()
