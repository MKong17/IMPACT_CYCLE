import unittest

from core.impact_sg.belief_update import revise_graph_from_claims
from core.impact_sg.cycle_types import Claim


def _graph():
    return {
        "image_id": "img_001",
        "nodes": [
            {
                "entity_id": "track_1",
                "canonical_label": "object",
                "prompt_used": "object",
                "mask": {"pixels": []},
                "bbox": [0, 0, 10, 10],
                "score": 0.3,
                "attributes": [],
                "provenance": [],
                "validator_flags": [],
                "risk": 0.0,
                "verified": False,
            }
        ],
        "edges": [
            {
                "edge_id": "edge_1",
                "src_id": "track_1",
                "relation": "left_of",
                "dst_id": "track_2",
                "score": 0.4,
                "evidence": [],
                "validator_flags": [],
                "risk": 0.0,
                "verified": False,
            }
        ],
        "validator_flags": [],
        "metadata": {},
    }


class BeliefUpdateTests(unittest.TestCase):
    def test_label_and_attribute_support_update_graph(self) -> None:
        graph = _graph()
        claims = {
            "label": Claim(
                claim_id="claim_label_track_1",
                claim_type="label",
                subject_id="track_1",
                predicate="label",
                value="person",
                support_score=0.95,
                conflict_score=0.0,
            ),
            "attr": Claim(
                claim_id="claim_attr_track_1_state",
                claim_type="attribute",
                subject_id="track_1",
                predicate="state",
                value="visible",
                support_score=0.9,
                conflict_score=0.0,
            ),
        }
        out = revise_graph_from_claims(graph, claims)
        node = out["nodes"][0]
        self.assertEqual(node["canonical_label"], "person")
        self.assertTrue(node["verified"])
        self.assertEqual(node["attributes"][0]["value"], "visible")

    def test_relation_conflict_flags_edge(self) -> None:
        graph = _graph()
        claims = {
            "rel": Claim(
                claim_id="claim_rel_edge_1",
                claim_type="relation",
                subject_id="track_1",
                predicate="left_of",
                object_id="track_2",
                evidence_edge_ids=["edge_1"],
                support_score=0.0,
                conflict_score=0.92,
            )
        }
        out = revise_graph_from_claims(graph, claims)
        edge = out["edges"][0]
        self.assertIn("cycle_relation_conflict", edge["validator_flags"])
        self.assertGreaterEqual(edge["risk"], 0.8)

    def test_high_prior_can_push_label_claim_over_accept_threshold(self) -> None:
        graph = _graph()
        claims = {
            "label": Claim(
                claim_id="claim_label_track_1",
                claim_type="label",
                subject_id="track_1",
                predicate="label",
                value="person",
                prior_score=0.9,
                support_score=0.42,
                conflict_score=0.08,
            ),
        }
        out = revise_graph_from_claims(graph, claims)
        node = out["nodes"][0]
        self.assertEqual(node["canonical_label"], "person")
        self.assertTrue(out["metadata"]["cycle_update"]["memory_adjustments"])

    def test_confusion_memory_makes_relation_conflict_easier_to_flag(self) -> None:
        graph = _graph()
        claims = {
            "rel": Claim(
                claim_id="claim_rel_edge_1",
                claim_type="relation",
                subject_id="track_1",
                predicate="left_of",
                object_id="track_2",
                evidence_edge_ids=["edge_1"],
                prior_score=0.5,
                support_score=0.24,
                conflict_score=0.56,
            )
        }
        out = revise_graph_from_claims(
            graph,
            claims,
            correction_memory={"relation_confusions": {"left_of": {"overlapping": 4}}},
        )
        edge = out["edges"][0]
        self.assertIn("cycle_relation_conflict", edge["validator_flags"])

    def test_verified_lock_resists_auto_reject_for_moderate_conflict(self) -> None:
        graph = _graph()
        graph["metadata"]["frame_idx"] = 12
        claims = {
            "label": Claim(
                claim_id="claim_label_track_1",
                claim_type="label",
                subject_id="track_1",
                predicate="label",
                value="object",
                prior_score=0.6,
                support_score=0.18,
                conflict_score=0.82,
            ),
        }
        out = revise_graph_from_claims(
            graph,
            claims,
            correction_memory={
                "verified_locks": {
                    "track_1": {"status": "confirmed", "frame_start": 10, "frame_end": 15}
                }
            },
            frame_idx=12,
            finalized_weight_boost=1.25,
        )
        node = out["nodes"][0]
        self.assertNotIn("cycle_label_conflict", node["validator_flags"])

    def test_constrained_label_correction_can_update_graph(self) -> None:
        graph = _graph()
        claims = {
            "label": Claim(
                claim_id="claim_label_track_1",
                claim_type="label",
                subject_id="track_1",
                predicate="label",
                value="object",
                support_score=0.2,
                conflict_score=0.6,
            ),
        }
        out = revise_graph_from_claims(
            graph,
            claims,
            correction_candidates={
                "claim_label_track_1": {
                    "best_value": "person",
                    "best_score": 0.91,
                    "options": ["object", "person", "mannequin"],
                }
            },
        )
        node = out["nodes"][0]
        self.assertEqual(node["canonical_label"], "person")
        self.assertTrue(out["metadata"]["cycle_update"]["correction_applied"])

    def test_constrained_relation_correction_can_update_graph(self) -> None:
        graph = _graph()
        claims = {
            "rel": Claim(
                claim_id="claim_rel_edge_1",
                claim_type="relation",
                subject_id="track_1",
                predicate="left_of",
                object_id="track_2",
                evidence_edge_ids=["edge_1"],
                support_score=0.18,
                conflict_score=0.62,
            )
        }
        out = revise_graph_from_claims(
            graph,
            claims,
            correction_candidates={
                "claim_rel_edge_1": {
                    "best_value": "touching",
                    "best_score": 0.9,
                    "options": ["left_of", "touching", "overlapping"],
                }
            },
        )
        edge = out["edges"][0]
        self.assertEqual(edge["relation"], "touching")
        self.assertTrue(edge["verified"])

    def test_existence_high_conflict_can_auto_remove_node_and_edges(self) -> None:
        graph = _graph()
        graph["nodes"].append(
            {
                "entity_id": "track_2",
                "canonical_label": "cup",
                "prompt_used": "cup",
                "mask": {"pixels": []},
                "bbox": [20, 0, 8, 8],
                "score": 0.4,
                "attributes": [],
                "provenance": [],
                "validator_flags": [],
                "risk": 0.0,
                "verified": False,
            }
        )
        claims = {
            "exist": Claim(
                claim_id="claim_exists_track_1",
                claim_type="existence",
                subject_id="track_1",
                predicate="exists",
                value="true",
                support_score=0.02,
                conflict_score=0.98,
            )
        }
        out = revise_graph_from_claims(graph, claims)
        node_ids = {str(n.get("entity_id", "")) for n in out.get("nodes") or []}
        self.assertNotIn("track_1", node_ids)
        self.assertTrue(any(str(x) == "track_1" for x in list((out.get("metadata") or {}).get("cycle_update", {}).get("auto_removed_node_ids") or [])))


if __name__ == "__main__":
    unittest.main()
