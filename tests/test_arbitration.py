import unittest

from core.impact_sg.arbitration import build_human_arbitration_queue
from core.impact_sg.cycle_types import Claim


class ArbitrationTests(unittest.TestCase):
    def test_queue_uses_memory_bonus_and_respects_verified_locks(self) -> None:
        locked_claim = Claim(
            claim_id="claim_label_track_1",
            claim_type="label",
            subject_id="track_1",
            predicate="label",
            value="cup",
            uncertainty_score=0.9,
            support_score=0.1,
            conflict_score=0.2,
        )
        relation_claim = Claim(
            claim_id="claim_rel_edge_1",
            claim_type="relation",
            subject_id="track_2",
            predicate="holding",
            object_id="track_3",
            uncertainty_score=0.6,
            support_score=0.2,
            conflict_score=0.6,
        )
        memory = {
            "label_confusions": {"cup": {"bottle": 1}},
            "relation_confusions": {"holding": {"touching": 4}},
            "prompt_aliases": {},
            "verified_locks": {
                "track_1": {"status": "confirmed", "frame_start": 10, "frame_end": 20}
            },
        }
        queue = build_human_arbitration_queue(
            {
                locked_claim.claim_id: locked_claim,
                relation_claim.claim_id: relation_claim,
            },
            max_items=5,
            threshold=0.45,
            correction_memory=memory,
            correction_candidates={
                relation_claim.claim_id: {
                    "best_value": "touching",
                    "best_score": 0.88,
                    "options": ["holding", "touching", "overlapping"],
                }
            },
            frame_idx=15,
        )
        queued_ids = {row["claim_id"] for row in queue}
        self.assertNotIn(locked_claim.claim_id, queued_ids)
        self.assertIn(relation_claim.claim_id, queued_ids)
        relation_row = next(row for row in queue if row["claim_id"] == relation_claim.claim_id)
        self.assertGreater(float(relation_row.get("memory_bonus", 0.0)), 0.0)
        self.assertIn("touching", str(relation_row.get("question", "")))
        self.assertIn("touching", list(relation_row.get("question_options") or []))
        self.assertEqual("touching", str(relation_row.get("suggested_value", "")))


if __name__ == "__main__":
    unittest.main()
