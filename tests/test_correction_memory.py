import os
import tempfile
import unittest

from core.impact_sg.correction_memory import (
    common_confusions,
    confusion_frequency,
    default_correction_memory,
    is_verified_locked,
    load_correction_memory,
    merge_correction_memories,
    prompt_alias_candidates,
    save_correction_memory,
    summarize_correction_memory,
    update_memory_from_human_decision,
)


class CorrectionMemoryTests(unittest.TestCase):
    def test_memory_updates_label_and_relation_confusions(self) -> None:
        memory = default_correction_memory()
        memory = update_memory_from_human_decision(
            memory,
            claim_type="label",
            proposed="bottle",
            corrected="cup",
        )
        memory = update_memory_from_human_decision(
            memory,
            claim_type="relation",
            proposed="inside",
            corrected="on",
        )
        self.assertEqual(memory["label_confusions"]["cup"]["bottle"], 1)
        self.assertEqual(memory["relation_confusions"]["on"]["inside"], 1)

    def test_memory_round_trip_persists_verified_lock(self) -> None:
        memory = update_memory_from_human_decision(
            default_correction_memory(),
            claim_type="alias",
            proposed="computer",
            corrected="laptop",
            subject_id="track_4",
            frame_start=10,
            frame_end=18,
        )
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "memory.json")
            save_correction_memory(path, memory)
            restored = load_correction_memory(path)
        self.assertIn("computer", restored["prompt_aliases"]["laptop"])
        self.assertEqual(restored["verified_locks"]["track_4"]["frame_end"], 18)

    def test_memory_helpers_expose_alias_confusion_and_lock_state(self) -> None:
        memory = default_correction_memory()
        memory = update_memory_from_human_decision(
            memory,
            claim_type="alias",
            proposed="computer",
            corrected="laptop",
        )
        memory = update_memory_from_human_decision(
            memory,
            claim_type="label",
            proposed="tablet",
            corrected="laptop",
            subject_id="track_7",
            frame_start=12,
            frame_end=16,
        )
        self.assertIn("computer", prompt_alias_candidates(memory, "laptop"))
        self.assertIn("tablet", common_confusions(memory, claim_type="label", canonical_value="laptop"))
        self.assertEqual(confusion_frequency(memory, claim_type="label", canonical_value="laptop"), 1)
        self.assertTrue(is_verified_locked(memory, subject_id="track_7", frame_idx=13))
        self.assertFalse(is_verified_locked(memory, subject_id="track_7", frame_idx=20))

    def test_merge_and_summary_combine_memories(self) -> None:
        first = {
            "label_confusions": {"cup": {"bottle": 2}},
            "relation_confusions": {"holding": {"touching": 1}},
            "prompt_aliases": {"laptop": ["computer"]},
            "verified_locks": {"track_1": {"status": "confirmed", "frame_start": 2, "frame_end": 6}},
        }
        second = {
            "label_confusions": {"cup": {"glass": 1}},
            "relation_confusions": {"holding": {"touching": 3}},
            "prompt_aliases": {"laptop": ["notebook computer"]},
            "verified_locks": {"track_1": {"status": "confirmed", "frame_start": 4, "frame_end": 9}},
        }
        merged = merge_correction_memories([first, second])
        self.assertEqual(merged["label_confusions"]["cup"]["bottle"], 2)
        self.assertEqual(merged["label_confusions"]["cup"]["glass"], 1)
        self.assertEqual(merged["relation_confusions"]["holding"]["touching"], 4)
        self.assertIn("computer", merged["prompt_aliases"]["laptop"])
        self.assertIn("notebook computer", merged["prompt_aliases"]["laptop"])
        self.assertEqual(merged["verified_locks"]["track_1"]["frame_start"], 2)
        self.assertEqual(merged["verified_locks"]["track_1"]["frame_end"], 9)
        summary = summarize_correction_memory(merged)
        self.assertEqual(summary["label_confusions"], 3)
        self.assertEqual(summary["relation_confusions"], 4)
        self.assertEqual(summary["prompt_aliases"], 2)
        self.assertEqual(summary["verified_locks"], 1)


if __name__ == "__main__":
    unittest.main()
