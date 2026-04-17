from __future__ import annotations

import unittest

from core.impact_sg.qwen_temporal_tracking import BatchResult, build_carry_context, chunk_sequence


class QwenTemporalTrackingTests(unittest.TestCase):
    def test_chunk_sequence_fixed_batch(self) -> None:
        chunks = chunk_sequence(list(range(12)), chunk_size=5)
        self.assertEqual(chunks, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11]])

    def test_build_carry_context_first_batch(self) -> None:
        text = build_carry_context(None)
        self.assertIn("第一批", text)

    def test_build_carry_context_with_previous(self) -> None:
        prev = BatchResult(
            batch_index=0,
            frame_indices=[0, 1, 2, 3, 4],
            frame_paths=["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"],
            person_attributes=[{"person_id": "p1", "gender": "male"}],
            global_semantic_summary="summary",
            tracking_text="tracking text",
            raw_response="{}",
        )
        text = build_carry_context(prev)
        self.assertIn("previous_batch_index", text)
        self.assertIn("tracking text", text)


if __name__ == "__main__":
    unittest.main()
