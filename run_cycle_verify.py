"""Standalone script: run Cycle Verify on frames from an existing scene graph bundle.

Usage:
    python run_cycle_verify.py \
        --bundle log/0_1203_8316378691_12-04-2026-22-41-01/scene_graph_bundle.json \
        --frames 0 1 2         # frame indices (0-based) to verify; omit = all
        --provider gemini_api  # or chatgpt_api / qwen25_vl / manual
        --rounds 1
        --output log/0_1203_8316378691_12-04-2026-22-41-01/cycle_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ── resolve project root so local imports work ─────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from core.impact_sg.cycle_pipeline import run_cycle_refine
from core.impact_sg.mllm_adapters.defaults import DEFAULT_CYCLE_PROVIDER, normalize_cycle_provider
from core.impact_sg.mllm_adapters.factory import build_vision_verifier
from core.impact_sg.ontology import load_ontology, ontology_from_payload

# ─────────────────────────────────────────────────────────────────────────────


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=True, indent=2)


def _sanitize_saved_graph(graph: Dict[str, object]) -> Dict[str, object]:
    out = dict(graph or {})
    metadata = dict(out.get("metadata") or {})
    metadata.pop("image_path", None)
    out["metadata"] = metadata
    return out


def _sanitize_saved_probe(row: Dict[str, object]) -> Dict[str, object]:
    out = dict(row or {})
    for payload_key in ("parsed_response", "response"):
        payload = dict(out.get(payload_key) or {}) if isinstance(out.get(payload_key), dict) else None
        if payload is None:
            continue
        for key in ("raw_response", "raw_text", "request_prompt", "request_schema"):
            payload.pop(key, None)
        out[payload_key] = payload
    return out


def _sanitize_saved_result(row: Dict[str, object]) -> Dict[str, object]:
    out = dict(row or {})
    out["probe_results"] = [
        _sanitize_saved_probe(item)
        for item in list(out.get("probe_results") or [])
        if isinstance(item, dict)
    ]
    graph_after = out.get("graph_after")
    if isinstance(graph_after, dict):
        out["graph_after"] = _sanitize_saved_graph(graph_after)
    return out


def _resolve_image_path(graph: Dict[str, object], bundle_dir: str) -> str:
    """Return the frame image path, falling back to cached frames."""
    meta = dict(graph.get("metadata") or {})
    cached = str(meta.get("image_path", "") or "").strip()
    if cached and os.path.isfile(cached):
        return cached
    # Try bundle-local jpg named after image_id
    image_id = str(graph.get("image_id", "") or "").strip()
    if image_id:
        candidate = os.path.join(bundle_dir, image_id + ".jpg")
        if os.path.isfile(candidate):
            return candidate
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cycle Verify on a scene graph bundle")
    parser.add_argument("--bundle", required=True, help="Path to scene_graph_bundle.json")
    parser.add_argument("--frames", nargs="*", type=int, default=None,
                        help="0-based graph indices to verify (default: all)")
    parser.add_argument("--provider", default=DEFAULT_CYCLE_PROVIDER,
                        choices=["qwen25_vl", "gemini_api", "chatgpt_api", "manual", "mock", "gemini", "openai", "chatgpt"],
                        help="Verifier provider (supports legacy aliases too)")
    parser.add_argument("--rounds", type=int, default=1, help="Cycle refine rounds")
    parser.add_argument("--cycle-cfg", default="configs/impact_cycle.json",
                        help="Path to impact_cycle.json config")
    parser.add_argument("--ontology", default="configs/impact_sg_ontology.json",
                        help="Path to ontology JSON")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <bundle_dir>/cycle_results.json)")
    parser.add_argument("--low-quota", action="store_true",
                        help="Disable multi-turn and caption probes (saves API quota)")
    args = parser.parse_args()
    requested_provider = str(args.provider or DEFAULT_CYCLE_PROVIDER).strip().lower() or DEFAULT_CYCLE_PROVIDER
    requested_provider = normalize_cycle_provider(requested_provider, default=requested_provider)
    if requested_provider == "manual":
        requested_provider = "mock"

    bundle_path = os.path.abspath(args.bundle)
    bundle_dir = os.path.dirname(bundle_path)
    output_path = args.output or os.path.join(bundle_dir, "cycle_results.json")

    print(f"[cycle-verify] Loading bundle: {bundle_path}")
    bundle = _load_json(bundle_path)
    graphs: List[Dict[str, object]] = list(bundle.get("graphs") or [])
    if not graphs:
        print("[cycle-verify] ERROR: bundle contains no graphs")
        sys.exit(1)

    # Select frames
    frame_indices: List[int] = args.frames if args.frames is not None else list(range(len(graphs)))
    frame_indices = [i for i in frame_indices if 0 <= i < len(graphs)]
    if not frame_indices:
        print(f"[cycle-verify] ERROR: no valid frame indices (bundle has {len(graphs)} graphs)")
        sys.exit(1)

    print(f"[cycle-verify] Will verify {len(frame_indices)} frame(s): {frame_indices}")

    # Load cycle config
    cycle_cfg_path = os.path.join(_here, args.cycle_cfg) if not os.path.isabs(args.cycle_cfg) else args.cycle_cfg
    if not os.path.isfile(cycle_cfg_path):
        print(f"[cycle-verify] ERROR: cycle config not found: {cycle_cfg_path}")
        sys.exit(1)
    cycle_cfg = _load_json(cycle_cfg_path)

    # Apply overrides
    cycle_cfg.setdefault("cycle", {})["max_revision_rounds"] = args.rounds
    if args.low_quota:
        cycle_cfg["cycle"]["enable_multi_turn_probes"] = False
        cycle_cfg["cycle"]["enable_caption_probe"] = False
        print("[cycle-verify] Low-quota mode: multi-turn and caption probes disabled")

    # Load ontology
    ont_path = os.path.join(_here, args.ontology) if not os.path.isabs(args.ontology) else args.ontology
    if os.path.isfile(ont_path):
        ont_payload = _load_json(ont_path)
        ontology = ontology_from_payload(ont_payload)
        print(f"[cycle-verify] Ontology loaded: {ont_path}")
    else:
        ontology = ontology_from_payload({})
        print(f"[cycle-verify] WARNING: ontology not found at {ont_path}, using empty ontology")

    # Build verifier
    print(f"[cycle-verify] Building verifier: provider={requested_provider}")
    verifier, runtime_meta = build_vision_verifier(
        cycle_cfg,
        preferred_provider=requested_provider,
        allow_mock_fallback=False,
    )
    print(f"[cycle-verify] Verifier ready: {runtime_meta.get('verifier_provider')} "
          f"model={runtime_meta.get('verifier_model_id','?')}")

    # Run verification frame by frame
    results: List[Dict[str, object]] = []
    t_start = time.time()

    for seq_num, graph_idx in enumerate(frame_indices):
        graph = dict(graphs[graph_idx])
        image_path = _resolve_image_path(graph, bundle_dir)
        image_id = str(graph.get("image_id", f"graph_{graph_idx}") or f"graph_{graph_idx}")

        if not image_path:
            print(f"[cycle-verify] [{seq_num+1}/{len(frame_indices)}] SKIP {image_id}: frame image not found")
            results.append({"graph_idx": graph_idx, "image_id": image_id, "error": "image_not_found"})
            continue

        print(f"[cycle-verify] [{seq_num+1}/{len(frame_indices)}] Verifying {image_id} "
              f"(nodes={len(graph.get('nodes',[]))}, edges={len(graph.get('edges',[]))}) ...")

        t0 = time.time()
        try:
            result = run_cycle_refine(
                graph=graph,
                image_path=image_path,
                verifier=verifier,
                ontology=ontology,
                cfg=cycle_cfg,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"[cycle-verify]   ERROR after {elapsed:.1f}s: {exc}")
            results.append({"graph_idx": graph_idx, "image_id": image_id, "error": str(exc)})
            continue

        elapsed = time.time() - t0
        summary = dict(result.get("summary") or {})
        queue = list(result.get("human_queue") or [])
        probes = list(result.get("probe_results") or [])
        print(f"[cycle-verify]   Done in {elapsed:.1f}s | probes={len(probes)} "
              f"accepted={summary.get('accepted_claim_count',0)} "
              f"flagged={summary.get('flagged_claim_count',0)} "
              f"queue={len(queue)}")

        results.append(_sanitize_saved_result({
            "graph_idx": graph_idx,
            "image_id": image_id,
            "elapsed_sec": round(elapsed, 2),
            "summary": summary,
            "human_queue": queue,
            "probe_results": probes,
            "votes": list(result.get("votes") or []),
            "graph_after": result.get("graph_after"),
        }))

    total_elapsed = time.time() - t_start
    output_payload = {
        "bundle": os.path.basename(bundle_path),
        "provider": requested_provider,
        "rounds": args.rounds,
        "total_elapsed_sec": round(total_elapsed, 2),
        "frames_attempted": len(frame_indices),
        "frames_succeeded": sum(1 for r in results if "error" not in r),
        "results": results,
    }
    _save_json(output_path, output_payload)
    print(f"\n[cycle-verify] Finished {len(frame_indices)} frame(s) in {total_elapsed:.1f}s")
    print(f"[cycle-verify] Results saved to: {output_path}")

    # Print queue summary
    all_queue = [item for r in results for item in list(r.get("human_queue") or [])]
    if all_queue:
        print(f"\n[cycle-verify] Human Arbitration Queue: {len(all_queue)} item(s)")
        for item in sorted(all_queue, key=lambda x: float(x.get("priority", 0)), reverse=True)[:10]:
            cid = item.get("claim_id", "?")
            pri = float(item.get("priority", 0))
            q = str(item.get("question", ""))[:80]
            print(f"  [{pri:.2f}] {cid}: {q}")
    else:
        print("\n[cycle-verify] Human Arbitration Queue is empty (all claims auto-resolved).")


if __name__ == "__main__":
    main()
