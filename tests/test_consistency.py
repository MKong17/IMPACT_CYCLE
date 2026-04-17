import unittest

from core.impact_sg.captioning import build_graph_caption_preview, caption_to_claim_votes
from core.impact_sg.claim_graph import graph_to_claims, label_claim_id, relation_claim_id
from core.impact_sg.consistency import aggregate_claim_scores
from core.impact_sg.ontology import ontology_from_payload


def _graph():
    return {
        "image_id": "img_001",
        "nodes": [
            {
                "entity_id": "track_1",
                "canonical_label": "person",
                "prompt_used": "person",
                "mask": {"pixels": []},
                "bbox": [0, 0, 10, 10],
                "score": 0.9,
                "attributes": [],
                "provenance": [],
                "risk": 0.0,
                "verified": False,
            },
            {
                "entity_id": "track_2",
                "canonical_label": "cup",
                "prompt_used": "cup",
                "mask": {"pixels": []},
                "bbox": [20, 0, 10, 10],
                "score": 0.8,
                "attributes": [],
                "provenance": [],
                "risk": 0.0,
                "verified": False,
            },
        ],
        "edges": [
            {
                "edge_id": "edge_1",
                "src_id": "track_1",
                "relation": "holding",
                "dst_id": "track_2",
                "score": 0.8,
                "evidence": [],
                "validator_flags": [],
                "risk": 0.0,
                "verified": False,
            }
        ],
    }


class ConsistencyTests(unittest.TestCase):
    def test_aggregate_claim_scores_tracks_support_and_conflict(self) -> None:
        claims = {row.claim_id: row for row in graph_to_claims(_graph())}
        votes = [
            {"claim_id": label_claim_id("track_1"), "vote": "support", "score": 0.7},
            {"claim_id": label_claim_id("track_1"), "vote": "conflict", "score": 0.2},
            {"claim_id": relation_claim_id("edge_1"), "vote": "support", "score": 0.8},
        ]
        updated = aggregate_claim_scores(claims, votes)
        self.assertEqual(updated[label_claim_id("track_1")].status, "supported")
        self.assertGreater(updated[label_claim_id("track_1")].support_ratio, 0.7)
        self.assertEqual(updated[relation_claim_id("edge_1")].status, "supported")

    def test_caption_votes_align_with_graph_entities_and_relations(self) -> None:
        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "person", "synonyms": ["human"]},
                    {"label": "cup", "synonyms": ["mug"]},
                ],
                "relation_vocabulary": {"spatial": [], "interaction": ["holding"]},
                "question_types": [],
            }
        )
        votes = caption_to_claim_votes(
            "A person is holding a cup.",
            _graph(),
            ontology,
        )
        vote_ids = {row["claim_id"] for row in votes}
        self.assertIn(label_claim_id("track_1"), vote_ids)
        self.assertIn(relation_claim_id("edge_1"), vote_ids)

    def test_caption_votes_respect_prompt_alias_memory(self) -> None:
        graph = _graph()
        graph["nodes"][1]["canonical_label"] = "laptop"
        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "person", "synonyms": []},
                    {"label": "laptop", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": [], "interaction": ["holding"]},
                "question_types": [],
            }
        )
        votes = caption_to_claim_votes(
            "A person is holding a computer.",
            graph,
            ontology,
            correction_memory={"prompt_aliases": {"laptop": ["computer"]}},
        )
        vote_ids = {row["claim_id"] for row in votes}
        self.assertIn(label_claim_id("track_2"), vote_ids)

    def test_graph_caption_preview_respects_relation_setting(self) -> None:
        preview = build_graph_caption_preview(
            _graph(),
            style="technical",
            max_sentences=2,
            require_relation_mentions=True,
        )
        self.assertIn("person", preview.lower())
        self.assertIn("holding", preview.lower())
        self.assertLessEqual(preview.count("."), 2)


if __name__ == "__main__":
    unittest.main()
