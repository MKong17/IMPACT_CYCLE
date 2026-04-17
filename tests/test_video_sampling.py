import unittest

from core.impact_sg.video_sampling import sample_frame_indices


class VideoSamplingTests(unittest.TestCase):
    def test_sampling_uses_requested_fps(self) -> None:
        self.assertEqual(
            sample_frame_indices(100, 30.0, 2.0),
            [0, 15, 30, 45, 60, 75, 90, 99],
        )

    def test_sampling_caps_at_every_frame(self) -> None:
        self.assertEqual(
            sample_frame_indices(6, 24.0, 60.0),
            [0, 1, 2, 3, 4, 5],
        )

    def test_sampling_handles_short_videos(self) -> None:
        self.assertEqual(sample_frame_indices(1, 30.0, 1.0), [0])
        self.assertEqual(sample_frame_indices(0, 30.0, 1.0), [])


if __name__ == "__main__":
    unittest.main()
