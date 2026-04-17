from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.impact_sg.eval_vqa import evaluate_vqa


def _flatten(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("all"), list):
            return payload["all"]
        out = []
        for key in ("single_turn", "multi_turn"):
            val = payload.get(key)
            if isinstance(val, list):
                out.extend(val)
        return out
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate graph-grounded VQA outputs.")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    with open(os.path.abspath(os.path.expanduser(args.pred)), "r", encoding="utf-8") as f:
        pred_payload = json.load(f)
    with open(os.path.abspath(os.path.expanduser(args.gt)), "r", encoding="utf-8") as f:
        gt_payload = json.load(f)

    pred_items = _flatten(pred_payload)
    gt_items = _flatten(gt_payload)

    metrics = evaluate_vqa(pred_items, gt_items)

    if args.out:
        out_path = os.path.abspath(os.path.expanduser(args.out))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=True, indent=2)
        print(f"[OK] VQA metrics saved: {out_path}")

    for k, v in metrics.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
