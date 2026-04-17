from __future__ import annotations

import json
import os
import sys
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.impact_sg.cycle_pipeline import run_cycle_refine
from core.impact_sg.mllm_adapters.base import MockVisionVerifier
from core.impact_sg.ontology import ontology_from_payload
from core.impact_sg.visual_verifier.mock_data import build_mock_cycle_cfg, build_mock_scene_graph


def main() -> None:
    graph = build_mock_scene_graph()
    cfg = build_mock_cycle_cfg()

    ontology_payload = {
        "entities": [{"label": "person"}, {"label": "cup"}],
        "relations": {"spatial": ["left_of", "right_of", "holding"]},
    }
    ontology = ontology_from_payload(ontology_payload)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(b"mock-frame")
        image_path = tmp.name
    try:
        result = run_cycle_refine(
            graph=graph,
            image_path=image_path,
            verifier=MockVisionVerifier(),
            ontology=ontology,
            cfg=cfg,
            correction_memory={},
        )
        output = {
            "summary": result.get("summary"),
            "policy": result.get("policy"),
            "vote_count": len(list(result.get("votes") or [])),
            "probe_count": len(list(result.get("probe_results") or [])),
            "caption_feedback": dict((result.get("caption") or {}).get("feedback") or {}),
        }
        print(json.dumps(output, ensure_ascii=True, indent=2))
    finally:
        try:
            os.unlink(image_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
