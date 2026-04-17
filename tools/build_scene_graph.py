from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.impact_sg.pipeline import run_build_scene_graph


def main() -> int:
    ap = argparse.ArgumentParser(description="Build IMPACT-SG scene graph from image using the configured SAM grounding backend.")
    ap.add_argument("--image_id", required=True)
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--ontology", default=os.path.join(_REPO_ROOT, "configs", "impact_sg_ontology.json"))
    ap.add_argument("--pipeline_cfg", default=os.path.join(_REPO_ROOT, "configs", "impact_sg_pipeline.json"))
    ap.add_argument("--image_width", type=int, default=1280)
    ap.add_argument("--image_height", type=int, default=720)
    ap.add_argument("--enable_sentence_refine", action="store_true")
    ap.add_argument("--backend_provider", choices=["mock", "external_command"], default="")
    ap.add_argument("--sam_runtime_config", default="")
    ap.add_argument("--sam_runner_profile", default="")
    ap.add_argument("--sam_adapter", default="")
    ap.add_argument("--external_command_template", default="")
    ap.add_argument("--external_command_args_json", default="", help="JSON array of argv tokens for external_command provider.")
    ap.add_argument("--external_command_args_file", default="")
    ap.add_argument("--external_command_cwd", default="")
    ap.add_argument("--external_timeout_sec", type=int, default=0)
    ap.add_argument("--disable_backend_cache", action="store_true")
    ap.add_argument("--out", required=True, help="Output scene graph JSON path")
    args = ap.parse_args()

    pipeline_cfg_override: Dict[str, object] = {}
    backend_override: Dict[str, object] = {}
    if args.backend_provider:
        backend_override["provider"] = str(args.backend_provider)
    if args.external_command_template:
        backend_override["external_command_template"] = str(args.external_command_template)
    if args.external_command_args_file:
        backend_override["external_command_args_file"] = str(args.external_command_args_file)
    if args.external_command_cwd:
        backend_override["external_command_cwd"] = str(args.external_command_cwd)
    if args.external_timeout_sec > 0:
        backend_override["external_timeout_sec"] = int(args.external_timeout_sec)
    if args.disable_backend_cache:
        backend_override["disable_cache"] = True
    if args.external_command_args_json:
        try:
            parsed = json.loads(args.external_command_args_json)
        except Exception as exc:
            raise SystemExit(f"[ERROR] Invalid --external_command_args_json: {exc}")
        if not isinstance(parsed, list):
            raise SystemExit("[ERROR] --external_command_args_json must be a JSON array.")
        backend_override["external_command_args"] = parsed
    runtime_cfg = str(args.sam_runtime_config or "").strip()
    if runtime_cfg and backend_override.get("provider") == "external_command":
        runner_path = os.path.join(_REPO_ROOT, "tools", "runners", "run_in_env.py")
        adapter_path = str(args.sam_adapter or "").strip() or os.path.join(_REPO_ROOT, "tools", "sam3_infer.py")
        runner_profile = str(args.sam_runner_profile or "").strip() or "sam3"
        backend_override["external_command_args"] = [
            sys.executable,
            runner_path,
            "--profile",
            runner_profile,
            "--",
            adapter_path,
            "--runtime-config",
            runtime_cfg,
            "--image_path",
            "{image_path}",
            "--prompt",
            "{prompt}",
            "--stage",
            "{stage}",
            "--max_instances",
            "{max_instances}",
        ]
        backend_override["external_batch_command_args"] = [
            sys.executable,
            runner_path,
            "--profile",
            runner_profile,
            "--",
            adapter_path,
            "--runtime-config",
            runtime_cfg,
            "--request_json",
            "{request_json_path}",
        ]
    if backend_override:
        pipeline_cfg_override["backend"] = backend_override

    graph = run_build_scene_graph(
        image_id=str(args.image_id),
        image_path=str(args.image_path),
        ontology_path=str(args.ontology),
        pipeline_cfg_path=str(args.pipeline_cfg),
        image_size=(int(args.image_width), int(args.image_height)),
        enable_sentence_refine=bool(args.enable_sentence_refine),
        pipeline_cfg_override=pipeline_cfg_override or None,
    )

    out_path = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=True, indent=2)

    print(f"[OK] scene graph saved: {out_path}")
    print(f"[INFO] nodes={len(graph.get('nodes', []))} edges={len(graph.get('edges', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
