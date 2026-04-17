import unittest

from core.impact_sg.captioning import build_caption_prompt, caption_to_claim_feedback
from core.impact_sg.claim_graph import attribute_claim_id, existence_claim_id, relation_claim_id
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
                "attributes": [
                    {"slot": "state", "value": "visible", "confidence": 0.8, "provenance": [], "verified": False}
                ],
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


class CaptioningTests(unittest.TestCase):
    def test_build_caption_prompt_requests_structured_feedback(self) -> None:
        prompt = build_caption_prompt(
            _graph(),
            style="technical",
            max_sentences=3,
            require_relation_mentions=True,
            structured_feedback=True,
        )
        self.assertIn("Return JSON only", prompt)
        self.assertIn("supported_entities", prompt)
        self.assertIn("unsupported_relations", prompt)
        self.assertIn("track_1", prompt)
        self.assertIn("edge_1", prompt)

    def test_caption_feedback_converts_structured_support_and_conflict(self) -> None:
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
        feedback = caption_to_claim_feedback(
            {
                "caption": "A person is visible, but the holding relation is unsupported.",
                "supported_entities": ["track_1"],
                "unsupported_entities": ["track_2"],
                "supported_attributes": [{"entity_id": "track_1", "slot": "state", "value": "visible"}],
                "unsupported_relations": ["edge_1"],
                "hallucinated_mentions": ["dog"],
            },
            _graph(),
            ontology,
        )
        vote_map = {(row["claim_id"], row["vote"]) for row in feedback["votes"]}
        self.assertIn((existence_claim_id("track_1"), "support"), vote_map)
        self.assertIn((existence_claim_id("track_2"), "conflict"), vote_map)
        self.assertIn((attribute_claim_id("track_1", "state"), "support"), vote_map)
        self.assertIn((relation_claim_id("edge_1"), "conflict"), vote_map)
        self.assertTrue(bool(feedback["report"]["structured"]))
        self.assertEqual(["dog"], list(feedback["report"]["hallucinated_mentions"] or []))

    def test_caption_feedback_falls_back_to_lexical_matching(self) -> None:
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
        feedback = caption_to_claim_feedback(
            "A human is holding a mug.",
            _graph(),
            ontology,
        )
        vote_ids = {row["claim_id"] for row in feedback["votes"]}
        self.assertIn(existence_claim_id("track_1"), vote_ids)
        self.assertIn(relation_claim_id("edge_1"), vote_ids)
        self.assertTrue(bool(feedback["report"]["fallback_used"]))


if __name__ == "__main__":
    unittest.main()
