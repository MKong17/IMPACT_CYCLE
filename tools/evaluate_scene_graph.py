from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.impact_sg.eval_scene_graph import evaluate_scene_graph


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate scene graph prediction against ground truth.")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--iou_threshold", type=float, default=0.5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(os.path.abspath(os.path.expanduser(args.pred)), "r", encoding="utf-8") as f:
        pred = json.load(f)
    with open(os.path.abspath(os.path.expanduser(args.gt)), "r", encoding="utf-8") as f:
        gt = json.load(f)

    metrics = evaluate_scene_graph(pred, gt, iou_threshold=float(args.iou_threshold))

    if args.out:
        out_path = os.path.abspath(os.path.expanduser(args.out))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=True, indent=2)
        print(f"[OK] scene-graph metrics saved: {out_path}")

    for k, v in metrics.items():
        print(f"{k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
