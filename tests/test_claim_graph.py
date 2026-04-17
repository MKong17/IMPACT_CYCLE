import unittest

from core.impact_sg.claim_graph import (
    attribute_claim_id,
    build_focus_graph,
    build_multi_turn_probes,
    build_single_turn_probes,
    existence_claim_id,
    graph_to_claims,
    label_claim_id,
    relation_claim_id,
)
from core.impact_sg.ontology import ontology_from_payload


def _sample_graph():
    return {
        "image_id": "img_001",
        "nodes": [
            {
                "entity_id": "track_1",
                "canonical_label": "person",
                "prompt_used": "person",
                "mask": {"pixels": []},
                "bbox": [0, 0, 20, 30],
                "score": 0.92,
                "attributes": [
                    {"slot": "state", "value": "visible", "confidence": 0.8, "provenance": [], "verified": False}
                ],
                "provenance": [],
                "risk": 0.1,
                "verified": False,
            },
            {
                "entity_id": "track_2",
                "canonical_label": "cup",
                "prompt_used": "cup",
                "mask": {"pixels": []},
                "bbox": [30, 0, 15, 25],
                "score": 0.88,
                "attributes": [],
                "provenance": [],
                "risk": 0.1,
                "verified": False,
            },
        ],
        "edges": [
            {
                "edge_id": "edge_1",
                "src_id": "track_1",
                "relation": "holding",
                "dst_id": "track_2",
                "score": 0.9,
                "evidence": [],
                "validator_flags": [],
                "risk": 0.1,
                "verified": False,
            }
        ],
        "metadata": {"graph_snapshot_id": "graph_snap_1"},
    }


class ClaimGraphTests(unittest.TestCase):
    def _ontology(self):
        return ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "person", "synonyms": ["human"]},
                    {"label": "cup", "synonyms": ["mug"]},
                    {"label": "bottle", "synonyms": []},
                    {"label": "glass", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": ["left_of"], "interaction": ["holding", "touching"]},
                "question_types": [],
            }
        )

    def test_graph_to_claims_covers_node_edge_and_attribute(self) -> None:
        claims = {row.claim_id: row for row in graph_to_claims(_sample_graph())}
        self.assertIn(existence_claim_id("track_1"), claims)
        self.assertIn(label_claim_id("track_1"), claims)
        self.assertIn(attribute_claim_id("track_1", "state"), claims)
        self.assertIn(relation_claim_id("edge_1"), claims)
        self.assertEqual(claims[label_claim_id("track_1")].value, "person")

    def test_single_turn_and_multi_turn_probes_map_back_to_claims(self) -> None:
        graph = _sample_graph()
        single = build_single_turn_probes(graph, ontology=self._ontology())
        multi = build_multi_turn_probes(graph, max_chains=1, ontology=self._ontology())
        single_ids = {row["target_claim_id"] for row in single}
        multi_ids = {row["target_claim_id"] for row in multi}
        self.assertIn(attribute_claim_id("track_1", "state"), single_ids)
        self.assertIn(relation_claim_id("edge_1"), single_ids)
        self.assertIn(label_claim_id("track_1"), multi_ids)
        self.assertIn(relation_claim_id("edge_1"), multi_ids)
        self.assertTrue(any(str(row.get("probe_family", "")) == "constrained_correction" for row in single))
        self.assertTrue(any(str(row.get("probe_family", "")) == "constrained_correction" for row in multi))

    def test_probe_questions_include_alias_and_confusion_guidance(self) -> None:
        graph = _sample_graph()
        graph["nodes"][1]["canonical_label"] = "laptop"
        memory = {
            "label_confusions": {"laptop": {"tablet": 2}},
            "relation_confusions": {"holding": {"touching": 3}},
            "prompt_aliases": {"laptop": ["computer", "notebook computer"]},
            "verified_locks": {},
        }
        single = build_single_turn_probes(graph, correction_memory=memory, ontology=self._ontology())
        multi = build_multi_turn_probes(graph, max_chains=2, correction_memory=memory, ontology=self._ontology())
        laptop_label_probe = next(
            row for row in single if row["target_claim_id"] == label_claim_id("track_2")
            and str(row.get("probe_family", "")) != "constrained_correction"
        )
        relation_probe = next(
            row for row in single if row["target_claim_id"] == relation_claim_id("edge_1")
            and str(row.get("probe_family", "")) != "constrained_correction"
        )
        multi_label_probe = next(
            row for row in multi if row["target_claim_id"] == label_claim_id("track_2")
            and str(row.get("probe_family", "")) != "constrained_correction"
        )
        correction_probe = next(
            row for row in single if row["target_claim_id"] == label_claim_id("track_2")
            and str(row.get("probe_family", "")) == "constrained_correction"
        )
        self.assertIn("computer", laptop_label_probe["question"])
        self.assertIn("tablet", laptop_label_probe["question"])
        self.assertIn("touching", relation_probe["question"])
        self.assertIn("notebook computer", multi_label_probe["question"])
        self.assertIn("tablet", correction_probe["question"])
        self.assertIn("laptop", list(correction_probe.get("candidate_options") or []))

    def test_build_focus_graph_keeps_person_centered_subgraph(self) -> None:
        graph = _sample_graph()
        graph["nodes"].append(
            {
                "entity_id": "track_3",
                "canonical_label": "table",
                "prompt_used": "table",
                "mask": {"pixels": []},
                "bbox": [60, 0, 20, 20],
                "score": 0.4,
                "attributes": [],
                "provenance": [],
                "risk": 0.0,
                "verified": False,
            }
        )
        focused = build_focus_graph(
            graph,
            enabled=True,
            subject_label="person",
            max_hops=1,
            direct_relations_only=True,
        )
        self.assertEqual([row["entity_id"] for row in focused["nodes"]], ["track_1", "track_2"])
        self.assertEqual([row["edge_id"] for row in focused["edges"]], ["edge_1"])
        focus_meta = dict((focused.get("metadata") or {}).get("focus_filter") or {})
        self.assertTrue(bool(focus_meta.get("applied")))
        self.assertEqual(str(focus_meta.get("subject_label", "")), "person")

    def test_multi_turn_probes_add_temporal_consistency_turns_when_context_exists(self) -> None:
        graph = _sample_graph()
        graph["metadata"]["temporal_context"] = {
            "current_frame_idx": 24,
            "frames_in_session": 3,
            "track_history": {
                "track_1": {
                    "observation_count": 3,
                    "seen_frames": [12, 24, 36],
                    "previous_frame_idx": 12,
                    "previous_label": "person",
                }
            },
            "relation_history": {
                "track_1|holding|track_2": {
                    "observation_count": 2,
                    "seen_frames": [12, 24],
                    "previous_frame_idx": 12,
                }
            },
        }
        probes = build_multi_turn_probes(graph, max_chains=1, ontology=self._ontology())
        temporal = [row for row in probes if str(row.get("probe_family", "")) == "temporal_consistency"]
        self.assertTrue(temporal)
        self.assertTrue(any(row["target_claim_id"] == existence_claim_id("track_1") for row in temporal))
        self.assertTrue(any(row["target_claim_id"] == label_claim_id("track_1") for row in temporal))
        self.assertTrue(any(row["target_claim_id"] == relation_claim_id("edge_1") for row in temporal))
        self.assertTrue(any("sampled frame 12" in str(row.get("question", "")) for row in temporal))


if __name__ == "__main__":
    unittest.main()
