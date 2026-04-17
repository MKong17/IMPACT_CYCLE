from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.impact_sg.openworldsam_backend import OpenWorldSAMBackend, OpenWorldSAMConfig
from core.impact_sg.proposal_pipeline import build_entity_proposals
from core.impact_sg.scene_graph_builder import build_scene_graph
from tools.openworldsam_infer import _ensure_downloaded_file, _resolve_prompt_alias


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class OpenWorldSAMBackendTests(unittest.TestCase):
    def test_prompt_alias_normalizes_template_prompts(self) -> None:
        aliases = {
            "person": "person",
            "phone": "cell phone",
        }
        self.assertEqual(_resolve_prompt_alias("a person", aliases), "person")
        self.assertEqual(_resolve_prompt_alias("all person instances", aliases), "person")
        self.assertEqual(_resolve_prompt_alias("the phone", aliases), "cell phone")

    def test_cache_key_separates_mock_and_external_provider(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            image_path = os.path.join(td, "frame 001.jpg")
            with open(image_path, "wb") as f:
                f.write(b"frame-bytes")

            cache_dir = os.path.join(td, "cache")
            mock_backend = OpenWorldSAMBackend(
                OpenWorldSAMConfig(
                    provider="mock",
                    max_instances_per_prompt=1,
                    enable_two_stage_refinement=False,
                    cache_dir=cache_dir,
                )
            )
            mock_rows = mock_backend.discover_entities_by_category(
                image_path,
                [{"prompt": "person", "canonical_label": "person"}],
            )
            self.assertTrue(mock_rows)
            self.assertEqual(mock_rows[0]["backend_metadata"]["provider"], "mock")

            script_path = os.path.join(td, "external_provider.py")
            _write_text(
                script_path,
                "\n".join(
                    [
                        "import json",
                        "print(json.dumps([{'bbox': [11, 12, 13, 14], 'score': 0.99}]))",
                    ]
                ),
            )
            external_backend = OpenWorldSAMBackend(
                OpenWorldSAMConfig(
                    provider="external_command",
                    max_instances_per_prompt=1,
                    enable_two_stage_refinement=False,
                    cache_dir=cache_dir,
                    external_command_args=(
                        sys.executable,
                        script_path,
                        "--image_path",
                        "{image_path}",
                        "--prompt",
                        "{prompt}",
                        "--stage",
                        "{stage}",
                    ),
                    external_timeout_sec=10,
                )
            )
            external_rows = external_backend.discover_entities_by_category(
                image_path,
                [{"prompt": "person", "canonical_label": "person"}],
            )
            self.assertTrue(external_rows)
            self.assertEqual(external_rows[0]["backend_metadata"]["provider"], "external_command")
            self.assertEqual(external_rows[0]["bbox"], [11, 12, 13, 14])
            self.assertAlmostEqual(float(external_rows[0]["score"]), 0.99, places=6)

    def test_external_args_file_and_bbox_fallback_survive_scene_graph_build(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            image_path = os.path.join(td, "frame.jpg")
            with open(image_path, "wb") as f:
                f.write(b"frame-b")

            script_path = os.path.join(td, "bbox_provider.py")
            _write_text(
                script_path,
                "\n".join(
                    [
                        "import json",
                        "print('INFO provider warmup')",
                        "print(json.dumps({'predictions': [{'bbox': [10, 20, 30, 40], 'score': 0.9}]}))",
                    ]
                ),
            )
            args_file = os.path.join(td, "openworldsam_args.json")
            with open(args_file, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        sys.executable,
                        script_path,
                        "--image_path",
                        "{image_path}",
                        "--prompt",
                        "{prompt}",
                        "--stage",
                        "{stage}",
                        "--max_instances",
                        "{max_instances}",
                    ],
                    f,
                    ensure_ascii=True,
                    indent=2,
                )

            backend = OpenWorldSAMBackend(
                OpenWorldSAMConfig(
                    provider="external_command",
                    max_instances_per_prompt=1,
                    enable_two_stage_refinement=False,
                    cache_dir=os.path.join(td, "cache"),
                    external_command_args_file=args_file,
                    external_timeout_sec=10,
                    disable_cache=True,
                )
            )
            raw_rows = backend.discover_entities_by_category(
                image_path,
                [{"prompt": "person", "canonical_label": "person"}],
            )
            self.assertEqual(len(raw_rows), 1)
            self.assertEqual(raw_rows[0]["bbox"], [10, 20, 30, 40])
            self.assertEqual(raw_rows[0]["mask"], {"pixels": []})

            proposals = build_entity_proposals(
                raw_rows,
                merge_mask_iou_threshold=0.75,
                image_wh=(100, 100),
                risk_weights={},
                low_confidence_threshold=0.45,
                small_area_ratio=0.001,
                thin_min_dim=8,
            )
            graph = build_scene_graph(
                image_id="img_001",
                proposals=proposals,
                relation_vocab={
                    "spatial": [
                        "left_of",
                        "right_of",
                        "above",
                        "below",
                        "overlap",
                        "inside",
                        "surrounding",
                        "intersect",
                        "touching",
                    ],
                    "interaction": [],
                },
                touching_iou_epsilon=0.02,
            )
            self.assertEqual(len(graph["nodes"]), 1)
            self.assertEqual(graph["nodes"][0]["bbox"], [10, 20, 30, 40])

    def test_external_batch_command_deduplicates_category_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            image_path = os.path.join(td, "frame.jpg")
            with open(image_path, "wb") as f:
                f.write(b"frame-c")

            script_path = os.path.join(td, "batch_provider.py")
            count_path = os.path.join(td, "batch_count.txt")
            _write_text(
                script_path,
                "\n".join(
                    [
                        "import json, sys",
                        "args = sys.argv[1:]",
                        "req_path = args[args.index('--request_json') + 1]",
                        "count_path = args[args.index('--count_path') + 1]",
                        "with open(req_path, 'r', encoding='utf-8') as f:",
                        "    req = json.load(f)",
                        "with open(count_path, 'w', encoding='utf-8') as f:",
                        "    f.write(str(len(req.get('prompts', []))))",
                        "payload = []",
                        "for item in req.get('prompts', []):",
                        "    payload.append({'prompt': item.get('prompt', ''), 'records': [{'bbox': [1, 2, 30, 40], 'score': 0.7}]})",
                        "print(json.dumps(payload))",
                    ]
                ),
            )
            backend = OpenWorldSAMBackend(
                OpenWorldSAMConfig(
                    provider="external_command",
                    max_instances_per_prompt=1,
                    enable_two_stage_refinement=False,
                    cache_dir=os.path.join(td, "cache"),
                    external_batch_command_args=(
                        sys.executable,
                        script_path,
                        "--request_json",
                        "{request_json_path}",
                        "--count_path",
                        count_path,
                    ),
                    external_timeout_sec=10,
                    disable_cache=True,
                )
            )
            rows = backend.discover_entities_by_category(
                image_path,
                [
                    {"prompt": "person", "canonical_label": "person"},
                    {"prompt": "a person", "canonical_label": "person"},
                    {"prompt": "all person instances", "canonical_label": "person"},
                    {"prompt": "chair", "canonical_label": "chair"},
                ],
            )
            self.assertEqual(len(rows), 2)
            self.assertTrue(os.path.isfile(count_path))
            with open(count_path, "r", encoding="utf-8") as f:
                count_value = f.read().strip()
            self.assertEqual(count_value, "2")
            labels = sorted(str(row.get("canonical_label", "")) for row in rows)
            self.assertEqual(labels, ["chair", "person"])

    def test_ensure_downloaded_file_uses_url_only_when_target_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            existing_path = os.path.join(td, "existing.bin")
            with open(existing_path, "wb") as f:
                f.write(b"ready")

            with mock.patch("tools.openworldsam_infer._download_to_path") as download_mock:
                resolved_existing = _ensure_downloaded_file(
                    existing_path,
                    "https://example.invalid/existing.bin",
                    label="existing checkpoint",
                )
            self.assertEqual(resolved_existing, existing_path)
            download_mock.assert_not_called()

            missing_path = os.path.join(td, "missing.bin")

            def _fake_download(url: str, target_path: str) -> None:
                self.assertEqual(url, "https://example.invalid/missing.bin")
                self.assertEqual(target_path, missing_path)
                with open(target_path, "wb") as f:
                    f.write(b"downloaded")

            with mock.patch("tools.openworldsam_infer._download_to_path", side_effect=_fake_download) as download_mock:
                resolved_missing = _ensure_downloaded_file(
                    missing_path,
                    "https://example.invalid/missing.bin",
                    label="missing checkpoint",
                )
            self.assertEqual(resolved_missing, missing_path)
            self.assertTrue(os.path.isfile(missing_path))
            with open(missing_path, "rb") as f:
                self.assertEqual(f.read(), b"downloaded")
            download_mock.assert_called_once()

    def test_external_command_tolerates_non_utf8_stderr_and_emits_only_key_progress(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            image_path = os.path.join(td, "frame.jpg")
            with open(image_path, "wb") as f:
                f.write(b"frame-d")

            script_path = os.path.join(td, "stderr_provider.py")
            _write_text(
                script_path,
                "\n".join(
                    [
                        "import json, sys",
                        "sys.stderr.buffer.write(b'noise:\\x8f raw downloader output\\n')",
                        "sys.stderr.buffer.write(b'prefix:\\x8f[OWSAM] downloading checkpoint\\n')",
                        "sys.stderr.flush()",
                        "print(json.dumps([{'bbox': [3, 4, 50, 60], 'score': 0.88}]))",
                    ]
                ),
            )
            progress_lines = []
            backend = OpenWorldSAMBackend(
                OpenWorldSAMConfig(
                    provider="external_command",
                    max_instances_per_prompt=1,
                    enable_two_stage_refinement=False,
                    cache_dir=os.path.join(td, "cache"),
                    external_command_args=(
                        sys.executable,
                        script_path,
                        "--image_path",
                        "{image_path}",
                        "--prompt",
                        "{prompt}",
                        "--stage",
                        "{stage}",
                    ),
                    external_timeout_sec=10,
                    disable_cache=True,
                    progress_cb=progress_lines.append,
                )
            )
            rows = backend.discover_entities_by_category(
                image_path,
                [{"prompt": "person", "canonical_label": "person"}],
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["bbox"], [3, 4, 50, 60])
            self.assertTrue(progress_lines)
            self.assertTrue(any("downloading checkpoint" in line for line in progress_lines))
            self.assertFalse(any("raw downloader output" in line for line in progress_lines))


if __name__ == "__main__":
    unittest.main()
