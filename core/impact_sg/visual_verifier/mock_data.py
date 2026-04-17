from __future__ import annotations

from typing import Dict


def build_mock_scene_graph() -> Dict[str, object]:
    return {
        "image_id": "demo_frame_000120",
        "nodes": [
            {
                "entity_id": "track_1",
                "canonical_label": "person",
                "bbox": [160, 80, 180, 320],
                "score": 0.86,
                "attributes": [{"slot": "state", "value": "standing", "confidence": 0.78}],
            },
            {
                "entity_id": "track_2",
                "canonical_label": "cup",
                "bbox": [340, 290, 60, 80],
                "score": 0.62,
                "attributes": [{"slot": "color", "value": "red", "confidence": 0.57}],
            },
        ],
        "edges": [
            {
                "edge_id": "edge_1",
                "src_id": "track_1",
                "relation": "holding",
                "dst_id": "track_2",
                "score": 0.68,
            }
        ],
        "metadata": {
            "frame_idx": 120,
            "temporal_context": {
                "track_history": {
                    "track_1": {
                        "previous_frame_idx": 110,
                        "previous_label": "person",
                        "seen_frames": [100, 110],
                    }
                },
                "relation_history": {
                    "track_1|holding|track_2": {
                        "previous_frame_idx": 110,
                    }
                },
            },
        },
    }


def build_mock_cycle_cfg() -> Dict[str, object]:
    return {
        "cycle": {
            "enable_single_turn_probes": True,
            "enable_multi_turn_probes": True,
            "enable_caption_probe": True,
        },
        "caption": {
            "style": "technical",
            "max_sentences": 3,
            "structured_feedback": True,
            "emit_conflict_votes": True,
        },
        "role_policy": {
            "caption_label_enabled": False,
        },
    }
