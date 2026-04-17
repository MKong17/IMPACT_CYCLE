from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Set

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.impact_sg.detection_io import save_detection_record, summarize_detection_payload
from core.impact_sg.pipeline import _backend_from_cfg, _condense_category_prompt_items, _load_prompt_pack, _merge_cfg, _merge_ontology_payload, load_json, release_backend_pool
from core.impact_sg.ontology import ontology_from_payload
from run_vidor_gt_frame_eval import (
    _annotated_frames_from_entry,
    _extract_frame_to_jpg,
    _limit_frames_uniform,
    _resolve_video_path,
    _sample_verify_frames_from_entry,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage-1 offline SAM3 detection precompute for the vidor-eval pipeline.")
    ap.add_argument("--videos_dir", default="/cvhci/temp/wkong/sample_videos/VidOR/videos")
    ap.add_argument("--gt_json", default="/cvhci/temp/wkong/sample_videos/pvsg.json")
    ap.add_argument("--pipeline_config", default="configs/impact_sg_pipeline.json")
    ap.add_argument("--ontology", default="configs/impact_sg_ontology.json")
    ap.add_argument("--output_dir", default="outputs/sam3_precompute")
    ap.add_argument("--video_ids", nargs="*", default=None, help="Optional explicit video ids.")
    ap.add_argument("--reverse_entries", action="store_true")
    ap.add_argument("--video_ord_start", type=int, default=0)
    ap.add_argument("--video_ord_end", type=int, default=0)
    ap.add_argument("--max_videos", type=int, default=5)
    ap.add_argument("--max_frames_per_video", type=int, default=5)
    ap.add_argument("--disable_cache", action="store_true", help="Disable backend cache during Stage 1 precompute.")
    ap.add_argument("--score_threshold", type=float, default=-1.0, help="Post-backend score threshold. Negative keeps all backend outputs.")
    ap.add_argument("--debug_video_id", default="", help="Optional one-video debug override.")
    ap.add_argument("--debug_frame_idx", type=int, default=-1, help="Optional one-frame debug override.")
    return ap.parse_args()


def _save_json(path: str, payload: object) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def _build_prompt_items(pipeline_cfg_path: str, ontology_path: str) -> List[Dict[str, str]]:
    cfg = load_json(pipeline_cfg_path)
    ontology_payload = load_json(ontology_path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    prompt_pack_file = str(cfg.get("ontology_prompt_pack_file", "") or "").strip()
    explicit_category_prompts: List[Dict[str, str]] = []
    if prompt_pack_file:
        if not os.path.isabs(prompt_pack_file):
            prompt_pack_file = os.path.abspath(os.path.join(repo_root, prompt_pack_file))
        if os.path.isfile(prompt_pack_file):
            prompt_pack_payload = _load_prompt_pack(prompt_pack_file)
            ontology_payload = _merge_ontology_payload(ontology_payload, prompt_pack_payload)
            explicit_category_prompts = [
                {
                    "canonical_label": str(x.get("canonical_label", "") or "").strip(),
                    "prompt": str(x.get("prompt", "") or "").strip(),
                }
                for x in list(prompt_pack_payload.get("explicit_category_prompts") or [])
                if isinstance(x, dict)
            ]
    ontology = ontology_from_payload(ontology_payload)
    if explicit_category_prompts:
        prompt_items = _condense_category_prompt_items(list(explicit_category_prompts))
    else:
        prompt_items = _condense_category_prompt_items(list(ontology.build_prompt_bank().category_prompts or []))
    proposal_cfg = dict(cfg.get("proposal") or {})
    max_category_prompts = int(proposal_cfg.get("max_category_prompts", 32) or 32)
    if max_category_prompts > 0 and len(prompt_items) > max_category_prompts:
        prompt_items = prompt_items[:max_category_prompts]
    return prompt_items


def main() -> None:
    args = _parse_args()
    gt_json_path = os.path.abspath(os.path.expanduser(str(args.gt_json)))
    videos_dir = os.path.abspath(os.path.expanduser(str(args.videos_dir)))
    pipeline_cfg_path = str(args.pipeline_config)
    if not os.path.isabs(pipeline_cfg_path):
        pipeline_cfg_path = os.path.join(_REPO_ROOT, pipeline_cfg_path)
    pipeline_cfg_path = os.path.abspath(os.path.expanduser(pipeline_cfg_path))
    ontology_path = str(args.ontology)
    if not os.path.isabs(ontology_path):
        ontology_path = os.path.join(_REPO_ROOT, ontology_path)
    ontology_path = os.path.abspath(os.path.expanduser(ontology_path))
    output_root = os.path.abspath(os.path.expanduser(str(args.output_dir)))
    frames_cache_dir = os.path.join(output_root, "_frame_cache")
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(frames_cache_dir, exist_ok=True)

    gt_payload = load_json(gt_json_path)
    entries = [dict(x) for x in list(gt_payload.get("data") or []) if isinstance(x, dict)]
    requested_videos: Set[str] = {str(x).strip() for x in list(args.video_ids or []) if str(x).strip()}
    if requested_videos:
        entries = [row for row in entries if str(row.get("video_id", "") or "").strip() in requested_videos]
    if str(args.debug_video_id or "").strip():
        entries = [row for row in entries if str(row.get("video_id", "") or "").strip() == str(args.debug_video_id).strip()]
    if bool(args.reverse_entries):
        entries = list(reversed(entries))
    if int(args.video_ord_start or 0) > 0:
        video_ord_end = int(args.video_ord_end or 0)
        if video_ord_end <= 0:
            video_ord_end = int(args.video_ord_start)
        start_idx = max(0, int(args.video_ord_start) - 1)
        end_idx = min(len(entries), video_ord_end)
        entries = entries[start_idx:end_idx]
    if int(args.max_videos or 0) > 0:
        entries = entries[: int(args.max_videos)]

    score_threshold = None if float(args.score_threshold) < 0.0 else float(args.score_threshold)
    pipeline_cfg = load_json(pipeline_cfg_path)
    if bool(args.disable_cache):
        pipeline_cfg = _merge_cfg(pipeline_cfg, {"backend": {"disable_cache": True}})
    prompt_items = _build_prompt_items(pipeline_cfg_path, ontology_path)
    backend = _backend_from_cfg(pipeline_cfg, repo_root=_REPO_ROOT)
    run_rows: List[Dict[str, object]] = []

    print(f"[sam3-precompute] Output dir: {output_root}")
    print(f"[sam3-precompute] Prompt count: {len(prompt_items)}")
    print(f"[sam3-precompute] Cache enabled: {int(not bool(args.disable_cache))}")
    print(f"[sam3-precompute] Score threshold: {'backend_only' if score_threshold is None else score_threshold}")

    try:
        for vid_idx, entry in enumerate(entries, start=1):
            video_id = str(entry.get("video_id", "") or "").strip()
            if not video_id:
                continue
            video_path = _resolve_video_path(videos_dir, video_id)
            if not video_path:
                print(f"[sam3-precompute] [{vid_idx}/{len(entries)}] skip {video_id}: video file missing")
                continue
            sampled_frames = _sample_verify_frames_from_entry(entry)
            final_frames = _limit_frames_uniform(sampled_frames, int(args.max_frames_per_video or 0))
            if int(args.debug_frame_idx or -1) >= 0:
                final_frames = [idx for idx in final_frames if int(idx) == int(args.debug_frame_idx)]
            print(
                f"[sam3-precompute] [{vid_idx}/{len(entries)}] video={video_id} "
                f"annotated_frames={len(_annotated_frames_from_entry(entry))} sampled={len(sampled_frames)} final={len(final_frames)}"
            )
            for frame_pos, frame_idx in enumerate(final_frames, start=1):
                t0 = time.time()
                frame_img_path, frame_w, frame_h = _extract_frame_to_jpg(
                    video_path=video_path,
                    frame_idx=int(frame_idx),
                    out_dir=os.path.join(frames_cache_dir, video_id),
                )
                details = backend.discover_entities_by_category_detailed(
                    frame_img_path,
                    prompt_items,
                    score_threshold=score_threshold,
                )
                per_frame_summary = summarize_detection_payload(details)
                command_samples = [
                    dict(item.get("command") or {})
                    for item in list(details.get("prompt_results") or [])
                    if isinstance(item, dict) and isinstance(item.get("command"), dict) and item.get("command")
                ]
                payload = {
                    "stage": "sam3_detection_precompute",
                    "video_id": video_id,
                    "frame_idx": int(frame_idx),
                    "image_id": f"{video_id}_f{int(frame_idx):06d}",
                    "image_path": frame_img_path,
                    "image_size": {"width": int(frame_w), "height": int(frame_h)},
                    "score_threshold": score_threshold,
                    "backend_provider": str((details.get("backend_config") or {}).get("provider", "")),
                    "backend_config": dict(details.get("backend_config") or {}),
                    "runtime_metadata": {
                        "pipeline_config_path": pipeline_cfg_path,
                        "ontology_path": ontology_path,
                        "cache_enabled": int(not bool(args.disable_cache)),
                        "prompt_count": len(prompt_items),
                        "command_debug_samples": command_samples[:3],
                    },
                    "prompt_results": [dict(x) for x in list(details.get("prompt_results") or []) if isinstance(x, dict)],
                    "post_threshold_records": [dict(x) for x in list(details.get("post_threshold_records") or []) if isinstance(x, dict)],
                    "summary": per_frame_summary,
                    "elapsed_sec": round(time.time() - t0, 3),
                }
                out_path = save_detection_record(output_root, video_id, int(frame_idx), payload)
                print(
                    f"[sam3-precompute] [{vid_idx}/{len(entries)}] "
                    f"frame {frame_pos}/{len(final_frames)} video={video_id} frame_idx={int(frame_idx)} "
                    f"raw={int(per_frame_summary.get('raw_detection_count', 0) or 0)} "
                    f"post={int(per_frame_summary.get('post_threshold_count', 0) or 0)} "
                    f"cache_hits={int(per_frame_summary.get('cache_hit_prompts', 0) or 0)}/{int(per_frame_summary.get('prompt_count', 0) or 0)} "
                    f"scores={float(per_frame_summary.get('score_min', 0.0) or 0.0):.3f}-{float(per_frame_summary.get('score_max', 0.0) or 0.0):.3f} "
                    f"file={out_path}"
                )
                run_rows.append(
                    {
                        "video_id": video_id,
                        "frame_idx": int(frame_idx),
                        "image_path": frame_img_path,
                        "raw_detection_count": int(per_frame_summary.get("raw_detection_count", 0) or 0),
                        "post_threshold_count": int(per_frame_summary.get("post_threshold_count", 0) or 0),
                        "cache_hit_prompts": int(per_frame_summary.get("cache_hit_prompts", 0) or 0),
                        "prompt_count": int(per_frame_summary.get("prompt_count", 0) or 0),
                        "score_min": float(per_frame_summary.get("score_min", 0.0) or 0.0),
                        "score_max": float(per_frame_summary.get("score_max", 0.0) or 0.0),
                        "elapsed_sec": round(time.time() - t0, 3),
                        "detection_json": out_path,
                    }
                )
    finally:
        release_backend_pool()

    summary = {
        "stage": "sam3_detection_precompute",
        "output_dir": output_root,
        "frame_count": len(run_rows),
        "video_count": len({str(row.get("video_id", "")) for row in run_rows}),
        "raw_detection_total": sum(int(row.get("raw_detection_count", 0) or 0) for row in run_rows),
        "post_threshold_total": sum(int(row.get("post_threshold_count", 0) or 0) for row in run_rows),
        "cache_hit_prompt_total": sum(int(row.get("cache_hit_prompts", 0) or 0) for row in run_rows),
        "score_threshold": score_threshold,
        "disable_cache": bool(args.disable_cache),
        "pipeline_config_path": pipeline_cfg_path,
        "ontology_path": ontology_path,
        "videos_dir": videos_dir,
        "gt_json_path": gt_json_path,
    }
    _save_json(os.path.join(output_root, "summary.json"), summary)
    _save_json(os.path.join(output_root, "per_frame_debug.json"), run_rows)
    csv_path = os.path.join(output_root, "summary.csv")
    fieldnames = [
        "video_id",
        "frame_idx",
        "image_path",
        "raw_detection_count",
        "post_threshold_count",
        "cache_hit_prompts",
        "prompt_count",
        "score_min",
        "score_max",
        "elapsed_sec",
        "detection_json",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in run_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"[sam3-precompute] Summary JSON: {os.path.join(output_root, 'summary.json')}")
    print(f"[sam3-precompute] Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
