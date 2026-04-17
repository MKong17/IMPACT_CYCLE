from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.impact_sg.pipeline import run_generate_vqa


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate graph-grounded single-turn and multi-turn VQA from scene graph.")
    ap.add_argument("--scene_graph", required=True)
    ap.add_argument("--pipeline_cfg", default=os.path.join(_REPO_ROOT, "configs", "impact_sg_pipeline.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(os.path.abspath(os.path.expanduser(args.scene_graph)), "r", encoding="utf-8") as f:
        graph = json.load(f)

    payload = run_generate_vqa(graph, pipeline_cfg_path=str(args.pipeline_cfg))

    out_path = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    print(f"[OK] VQA saved: {out_path}")
    print(f"[INFO] single={len(payload.get('single_turn', []))} multi={len(payload.get('multi_turn', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
