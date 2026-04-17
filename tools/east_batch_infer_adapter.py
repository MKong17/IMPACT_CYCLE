from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.label_utils import load_label_names, resolve_label_source
from utils.feature_env import load_feature_env_defaults


VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".m4v", ".webm")


def _print(msg: str) -> None:
    print(str(msg), flush=True)


def _list_videos(video_dir: str) -> List[str]:
    out: List[str] = []
    for name in os.listdir(video_dir):
        if name.lower().endswith(VIDEO_EXTS):
            out.append(os.path.join(video_dir, name))
    return sorted(out)


def _load_labels(labels_txt: str = "", labels_json: str = "") -> List[str]:
    labels: List[str] = []
    if labels_json and os.path.isfile(labels_json):
        try:
            with open(labels_json, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                labels = [str(x).strip() for x in obj if str(x).strip()]
            elif isinstance(obj, dict) and isinstance(obj.get("labels"), list):
                labels = [str(x).strip() for x in obj.get("labels") if str(x).strip()]
        except Exception:
            labels = []
    if (not labels) and labels_txt and os.path.isfile(labels_txt):
        labels = load_label_names(labels_txt)
    uniq: List[str] = []
    for name in labels:
        if name and name not in uniq:
            uniq.append(name)
    return uniq


def _extract_features_for_video(
    video_path: str,
    features_dir: str,
    *,
    east_ckpt: str,
    east_cfg: str,
    batch_size: int,
    frame_stride: int,
    use_fp16: bool,
    feature_backbone_version: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    from tools.feature_extractors import extract_video_features, save_features

    def _emit(msg: str) -> None:
        if callable(progress_cb):
            progress_cb(str(msg))
        else:
            _print(msg)

    load_feature_env_defaults(repo_root=_REPO_ROOT)
    feat_path = os.path.join(features_dir, "features.npy")
    if os.path.isfile(feat_path):
        _emit(f"[INFO] Using cached features: {feat_path}")
        return feat_path

    os.makedirs(features_dir, exist_ok=True)
    backbone = str(
        feature_backbone_version
        or os.environ.get("FEATURE_BACKBONE", "east_backbone")
        or "east_backbone"
    ).strip()
    provider = str(os.environ.get("FEATURE_PROVIDER", "internal") or "internal").strip().lower()

    if provider in ("external", "external_cmd", "cmd"):
        cmd_tpl = os.environ.get("FEATURE_CMD", "").strip()
        if not cmd_tpl:
            raise RuntimeError("FEATURE_CMD is not set for external feature provider.")
        cmd = cmd_tpl.format(
            video_path=video_path,
            features_dir=features_dir,
            frame_stride=max(1, int(frame_stride)),
            backbone=backbone,
            feat_path=feat_path,
        )
        _emit(f"[FEATS] External cmd: {cmd}")
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if proc.stdout:
            for ln in proc.stdout.strip().splitlines():
                _emit(ln)
        if proc.stderr:
            for ln in proc.stderr.strip().splitlines():
                _emit(ln)
        if proc.returncode != 0:
            raise RuntimeError(f"External feature command failed (code {proc.returncode}).")
        if not os.path.isfile(feat_path):
            raise RuntimeError("External feature command did not produce features.npy.")
        _emit(f"[OK] features ready: {feat_path}")
        return feat_path

    if provider in ("http", "external_http", "api"):
        url = os.environ.get("FEATURE_API_URL", "").strip()
        if not url:
            raise RuntimeError("FEATURE_API_URL is not set for external feature API.")
        payload = {
            "video_path": video_path,
            "features_dir": features_dir,
            "frame_stride": max(1, int(frame_stride)),
            "backbone": backbone,
        }
        _emit(f"[FEATS] Requesting features from API: {url}")
        import urllib.error
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp_data = resp.read().decode("utf-8")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Feature API request failed: {e}") from e
        try:
            res = json.loads(resp_data) if resp_data else {}
        except Exception:
            res = {}
        if not os.path.isfile(feat_path):
            raise RuntimeError("Feature API did not produce features.npy in features_dir.")
        if isinstance(res, dict) and res.get("meta") and not os.path.isfile(os.path.join(features_dir, "meta.json")):
            with open(os.path.join(features_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(res.get("meta"), f, ensure_ascii=True, indent=2)
        _emit(f"[OK] features ready: {feat_path}")
        return feat_path

    backbone_key = str(backbone or "").strip().lower()
    if backbone_key in {
        "east",
        "east_backbone",
        "east-backbone",
        "videomae",
        "videomae_v1",
        "videomaev2",
        "video_mae",
        "videomae2",
    }:
        if not east_ckpt or not east_cfg:
            raise RuntimeError("EAST feature extraction requires both --ckpt and --cfg.")
        from tools.east.east_model_infer import extract_east_backbone_features

        pref_device = str(os.environ.get("EAST_FEAT_DEVICE", "") or "").strip().lower()
        run_device = pref_device if pref_device else None
        _emit("[FEATS] Extracting features via EAST backbone path...")
        try:
            east_out = extract_east_backbone_features(
                video_path=video_path,
                cfg_path=east_cfg,
                ckpt_path=east_ckpt,
                frame_stride=max(1, int(frame_stride)),
                device=run_device,
                progress_cb=lambda m: _emit(str(m)),
            )
        except RuntimeError as exc:
            msg = str(exc)
            is_cuda_oom = ("out of memory" in msg.lower()) and (
                "cuda" in msg.lower() or bool(torch and torch.cuda.is_available())
            )
            allow_cpu_fallback = str(os.environ.get("EAST_FEAT_ALLOW_CPU_FALLBACK", "1") or "1").strip().lower()
            allow_cpu_fallback = allow_cpu_fallback in ("1", "true", "yes", "on")
            if (run_device in (None, "", "cuda", "cuda:0")) and is_cuda_oom and allow_cpu_fallback:
                try:
                    if torch is not None and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                _emit("[FEATS][WARN] CUDA OOM; retrying EAST feature extraction on CPU (slower).")
                east_out = extract_east_backbone_features(
                    video_path=video_path,
                    cfg_path=east_cfg,
                    ckpt_path=east_ckpt,
                    frame_stride=max(1, int(frame_stride)),
                    device="cpu",
                    progress_cb=lambda m: _emit(str(m)),
                )
            else:
                raise
        feats = np.asarray((east_out or {}).get("features"), dtype=np.float32)
        meta = dict((east_out or {}).get("meta") or {})
        meta["video_path"] = str(video_path)
        if feats.ndim != 2:
            raise RuntimeError(f"EAST feature extractor returned invalid shape: {feats.shape}")
        save_features(features_dir, feats, meta=meta)
        _emit(f"[OK] features ready: {feat_path}")
        return feat_path

    _emit(f"[FEATS] Extracting {backbone} features to {feat_path}")
    feats, meta = extract_video_features(
        video_path,
        backbone=backbone,
        batch_size=max(1, int(batch_size)),
        frame_stride=max(1, int(frame_stride)),
        use_fp16=bool(use_fp16),
    )
    if feats.ndim != 2:
        raise RuntimeError(f"Feature extractor returned invalid shape: {feats.shape}")
    meta = dict(meta or {})
    meta["video_path"] = str(video_path)
    save_features(features_dir, feats, meta=meta)
    _emit(f"[OK] features ready: {feat_path}")
    return feat_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch EAST inference on raw videos with optional shared adapter."
    )
    ap.add_argument("--video_dir", required=True, help="Directory containing input videos.")
    ap.add_argument("--output_dir", required=True, help="Directory to write per-video EAST outputs.")
    ap.add_argument("--labels_txt", default="", help="TXT label bank / mapping.")
    ap.add_argument("--labels_json", default="", help="JSON label bank.")
    ap.add_argument("--ckpt", required=True, help="EAST checkpoint path.")
    ap.add_argument("--cfg", required=True, help="EAST config path.")
    ap.add_argument("--category_adapter", default="", help="Optional shared EAST adapter asset (.pt).")
    ap.add_argument("--text_bank_version", default="auto")
    ap.add_argument("--feature_backbone_version", default="east_backbone")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--no_fp16", action="store_true")
    args = ap.parse_args()

    video_dir = os.path.abspath(os.path.expanduser(args.video_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    ckpt = os.path.abspath(os.path.expanduser(args.ckpt))
    cfg = os.path.abspath(os.path.expanduser(args.cfg))
    category_adapter = str(args.category_adapter or "").strip()

    if not os.path.isdir(video_dir):
        raise SystemExit(f"Video dir not found: {video_dir}")
    if not os.path.isfile(ckpt):
        raise SystemExit(f"EAST checkpoint not found: {ckpt}")
    if not os.path.isfile(cfg):
        raise SystemExit(f"EAST cfg not found: {cfg}")

    os.makedirs(output_dir, exist_ok=True)
    os.environ["EAST_CKPT"] = ckpt
    os.environ["EAST_CFG"] = cfg

    videos = _list_videos(video_dir)
    if not videos:
        raise SystemExit(f"No videos found in {video_dir}")

    label_txt = str(args.labels_txt or "").strip()
    if (not label_txt) or (not os.path.isfile(label_txt)):
        label_txt = resolve_label_source(repo_root=_REPO_ROOT, extra_dirs=[video_dir])
    labels = _load_labels(label_txt, args.labels_json)
    if not labels:
        raise SystemExit("Label bank is empty. Provide --labels_txt or --labels_json.")

    from tools.east.east_infer_adapter import EastRuntime
    from tools.east.east_output_format import write_segments_outputs

    runtime = EastRuntime(
        ckpt_path=ckpt,
        cfg_path=cfg,
        text_bank_version=args.text_bank_version,
        feature_backbone_version=args.feature_backbone_version or None,
        progress_cb=lambda m: _print(str(m)),
        category_adapter_path=category_adapter or None,
    )

    failures = 0
    _print(f"[INFO] Found {len(videos)} videos.")
    for idx, video_path in enumerate(videos, 1):
        base = os.path.splitext(os.path.basename(video_path))[0] or f"video_{idx:04d}"
        features_dir = os.path.join(output_dir, base)
        _print(f"[INFO] ({idx}/{len(videos)}) {os.path.basename(video_path)} -> {features_dir}")
        try:
            _extract_features_for_video(
                video_path,
                features_dir,
                east_ckpt=ckpt,
                east_cfg=cfg,
                batch_size=args.batch_size,
                frame_stride=args.frame_stride,
                use_fp16=not args.no_fp16,
                feature_backbone_version=str(args.feature_backbone_version or "east_backbone"),
                progress_cb=lambda m: _print(str(m)),
            )
            payload, _boundary, _scores, _embeds = runtime.predict(
                video_id=base,
                features_dir=features_dir,
                label_bank=labels,
                video_path=video_path,
            )
            txt_path, json_path = write_segments_outputs(features_dir, "pred_east", payload)
            _print(f"[OK] EAST inference done: {json_path}")
            _print(f"[OK] EAST txt written: {txt_path}")
        except Exception as exc:
            failures += 1
            _print(f"[WARN] failed {os.path.basename(video_path)}: {exc}")

    _print(f"[OK] EAST batch inference done. Outputs written to {output_dir} (failures={failures})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
