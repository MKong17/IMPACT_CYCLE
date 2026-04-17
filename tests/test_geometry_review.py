import unittest

from core.impact_sg.cycle_types import Claim
from core.impact_sg.geometry_review import build_geometry_review_queue, rebuild_spatial_edges


def _graph():
    return {
        "image_id": "img_001",
        "nodes": [
            {
                "entity_id": "track_1",
                "canonical_label": "person",
                "prompt_used": "person",
                "mask": {"pixels": []},
                "bbox": [40, 0, 10, 10],
                "score": 0.95,
                "attributes": [],
                "provenance": [],
                "validator_flags": [],
                "risk": 0.0,
                "verified": False,
            },
            {
                "entity_id": "track_2",
                "canonical_label": "cup",
                "prompt_used": "cup",
                "mask": {"pixels": []},
                "bbox": [20, 0, 10, 10],
                "score": 0.55,
                "attributes": [],
                "provenance": [],
                "validator_flags": [],
                "risk": 0.0,
                "verified": False,
            },
        ],
        "edges": [
            {
                "edge_id": "edge_spatial",
                "src_id": "track_1",
                "relation": "left_of",
                "dst_id": "track_2",
                "score": 1.0,
                "evidence": [],
                "validator_flags": [],
                "risk": 0.0,
                "verified": False,
            },
            {
                "edge_id": "edge_interaction",
                "src_id": "track_1",
                "relation": "holding",
                "dst_id": "track_2",
                "score": 0.7,
                "evidence": [],
                "validator_flags": [],
                "risk": 0.0,
                "verified": False,
            },
        ],
        "validator_flags": [],
        "metadata": {},
    }


class GeometryReviewTests(unittest.TestCase):
    def test_geometry_review_queue_returns_bbox_choice_for_spatial_conflict(self) -> None:
        graph = _graph()
        claims = {
            "claim_label_track_1": Claim(
                claim_id="claim_label_track_1",
                claim_type="label",
                subject_id="track_1",
                predicate="label",
                value="person",
                support_score=0.9,
                conflict_score=0.0,
            ),
            "claim_label_track_2": Claim(
                claim_id="claim_label_track_2",
                claim_type="label",
                subject_id="track_2",
                predicate="label",
                value="cup",
                support_score=0.8,
                conflict_score=0.0,
            ),
            "claim_rel_edge_spatial": Claim(
                claim_id="claim_rel_edge_spatial",
                claim_type="relation",
                subject_id="track_1",
                predicate="left_of",
                object_id="track_2",
                evidence_edge_ids=["edge_spatial"],
                support_score=0.1,
                conflict_score=0.9,
            ),
        }
        queue = build_geometry_review_queue(
            graph,
            claims,
            relation_vocab={"spatial": ["left_of", "right_of"], "interaction": ["holding"]},
            preferred_anchor_label="person",
            conflict_threshold=0.6,
            max_items=2,
        )
        self.assertEqual(len(queue), 1)
        row = queue[0]
        self.assertEqual(str(row.get("claim_type", "")), "bbox")
        self.assertEqual(str(row.get("target_node_id", "")), "track_2")
        self.assertTrue(bool(str(row.get("suggested_value", ""))))
        self.assertTrue(any(str(opt.get("value", "")) == "keep_current" for opt in list(row.get("resolution_options") or [])))
        self.assertTrue(any(bool(list(opt.get("bbox") or [])) for opt in list(row.get("resolution_options") or [])))

    def test_rebuild_spatial_edges_preserves_non_spatial_relations(self) -> None:
        graph = _graph()
        graph["nodes"][1]["bbox"] = [70, 0, 10, 10]
        rebuilt = rebuild_spatial_edges(
            graph,
            relation_vocab={"spatial": ["left_of", "right_of"], "interaction": ["holding"]},
            touching_iou_epsilon=0.02,
            pairwise_max=32,
        )
        edge_labels = {(str(edge.get("src_id", "")), str(edge.get("relation", "")), str(edge.get("dst_id", ""))) for edge in rebuilt.get("edges") or []}
        self.assertIn(("track_1", "left_of", "track_2"), edge_labels)
        self.assertIn(("track_1", "holding", "track_2"), edge_labels)


if __name__ == "__main__":
    unittest.main()
