import unittest

from core.impact_sg.eval_cycle import evaluate_cycle_result, infer_frames_in_session


def _payload():
    return {
        "claims": {
            "claim_1": {
                "claim_id": "claim_1",
                "status": "supported",
                "support_score": 0.9,
                "conflict_score": 0.1,
            },
            "claim_2": {
                "claim_id": "claim_2",
                "status": "conflicted",
                "support_score": 0.2,
                "conflict_score": 0.8,
            },
            "claim_3": {
                "claim_id": "claim_3",
                "status": "uncertain",
                "support_score": 0.5,
                "conflict_score": 0.5,
            },
        },
        "votes": [
            {"claim_id": "claim_1", "view_type": "caption", "vote": "support", "score": 0.6},
            {"claim_id": "claim_2", "view_type": "caption", "vote": "conflict", "score": 0.7},
            {"claim_id": "claim_1", "view_type": "single_turn_vqa", "vote": "support", "score": 0.8},
            {"claim_id": "claim_2", "view_type": "multi_turn_vqa", "vote": "conflict", "score": 0.9},
        ],
        "probe_results": [
            {"view_type": "multi_turn_vqa", "probe_family": "temporal_consistency"},
            {"view_type": "multi_turn_vqa", "probe_family": "binary_verification"},
        ],
        "human_queue": [{"claim_id": "claim_3"}],
        "caption": {
            "feedback": {
                "structured": True,
                "hallucinated_mentions": ["lamp"],
            }
        },
        "graph_after": {
            "metadata": {
                "temporal_context": {"frames_in_session": 2},
                "cycle_update": {"accepted_claim_ids": ["claim_1", "claim_2"]},
            }
        },
        "summary": {"accepted_claim_count": 2},
    }


class EvalCycleTests(unittest.TestCase):
    def test_infer_frames_in_session_uses_temporal_context(self) -> None:
        self.assertEqual(2, infer_frames_in_session(_payload()))

    def test_evaluate_cycle_result_reports_cycle_metrics(self) -> None:
        metrics = evaluate_cycle_result(_payload())
        self.assertAlmostEqual(2.0 / 3.0, float(metrics["claim_agreement_rate"]), places=4)
        self.assertAlmostEqual(0.5, float(metrics["graph_caption_contradiction_rate"]), places=4)
        self.assertAlmostEqual(0.5, float(metrics["graph_vqa_contradiction_rate"]), places=4)
        self.assertAlmostEqual(0.5, float(metrics["human_queries_per_frame"]), places=4)
        self.assertAlmostEqual(2.0 / 3.0, float(metrics["automatic_resolution_rate_before_human_review"]), places=4)
        self.assertAlmostEqual(0.5, float(metrics["temporal_multi_turn_share"]), places=4)
        self.assertEqual(1.0, float(metrics["caption_structured_feedback_rate"]))


if __name__ == "__main__":
    unittest.main()
