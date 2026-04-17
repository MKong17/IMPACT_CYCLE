"""
Extract per-frame ResNet-50 features and save as (2048, num_frames) float32 arrays,
matching the layout used by ASOT/FACT reference features.
Supports frame stride and optional FP16 (CUDA) for faster throughput.
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image


VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".m4v")

# ---- model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # autotune for fixed input sizes
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.fc = torch.nn.Identity()
model.eval().to(device)

# ---- preprocessing (resize/crop/normalize from the official weights) ----
preprocess = weights.transforms()

try:
    _autocast = torch.amp.autocast
    _autocast_kwargs = {"device_type": "cuda"}
except AttributeError:
    _autocast = torch.cuda.amp.autocast
    _autocast_kwargs = {}


def video_to_feats(
    vpath, batch_size=128, max_frames=None, frame_stride=1, use_fp16=True
):
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {vpath}")

    frames = []
    feats = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    use_total = total if (max_frames is None) else min(total, max_frames)

    for idx in range(use_total):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_stride > 1 and (idx % frame_stride) != 0:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(preprocess(Image.fromarray(frame)))

        if len(frames) == batch_size:
            with torch.no_grad():
                batch = torch.stack(frames, dim=0).to(device)  # [B,3,224,224]
                if use_fp16 and device.type == "cuda":
                    with _autocast(**_autocast_kwargs):
                        f = model(batch)
                else:
                    f = model(batch)
                feats.append(f.detach().cpu())
            frames = []

    if frames:
        with torch.no_grad():
            batch = torch.stack(frames, dim=0).to(device)
            if use_fp16 and device.type == "cuda":
                with _autocast(**_autocast_kwargs):
                    f = model(batch)
            else:
                f = model(batch)
            feats.append(f.detach().cpu())

    cap.release()

    if not feats:
        return np.zeros((0, 2048), dtype=np.float32)

    # Stack to [T, 2048] then transpose to [2048, T] to match reference format.
    F = torch.cat(feats, dim=0).numpy().astype(np.float32)
    return F.T.copy()  # ensure contiguous (2048, num_frames)


def main():
    parser = argparse.ArgumentParser(
        description="Extract ResNet-50 frame features to (2048, T) .npy files."
    )
    parser.add_argument(
        "--src",
        default=os.environ.get("DATA_SRC", "/cvhci/temp/qiany/test_videos_dataset"),
        help="Directory containing source videos.",
    )
    parser.add_argument(
        "--out",
        default=os.environ.get("DATA_OUT", "/cvhci/temp/qiany/custom_set_for_asot_hvq"),
        help="Output root; features will be saved under out/features",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for CNN inference."
    )
    parser.add_argument(
        "--max_frames", type=int, default=None, help="Optional cap on frames per video."
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Sample every Nth frame (1 = use all frames).",
    )
    parser.add_argument(
        "--no_fp16", action="store_true", help="Disable FP16 autocast on CUDA."
    )
    args = parser.parse_args()

    out_dir = os.path.join(args.out, "features")
    os.makedirs(out_dir, exist_ok=True)

    videos = [f for f in os.listdir(args.src) if f.lower().endswith(VIDEO_EXTS)]
    if not videos:
        print(f"[ERR] no videos in {args.src}")
        sys.exit(1)

    for v in sorted(videos):
        base, ext = os.path.splitext(v)
        out_npy = os.path.join(out_dir, base + ".npy")
        if os.path.exists(out_npy):
            print(f"[SKIP] exists: {out_npy}")
            continue

        vpath = os.path.join(args.src, v)
        print(f"[EXTRACT] {v} -> {out_npy}")
        F = video_to_feats(
            vpath,
            batch_size=args.batch_size,
            max_frames=args.max_frames,
            frame_stride=max(1, args.frame_stride),
            use_fp16=not args.no_fp16,
        )
        np.save(out_npy, F.astype(np.float32, copy=False))
        print(f"[OK] {base}: feats {F.shape} dtype={F.dtype}")


if __name__ == "__main__":
    main()
