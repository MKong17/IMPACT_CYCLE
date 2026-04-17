import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class VideoTaskStudioQtUiTests(unittest.TestCase):
    _qt_ready = None
    _qt_skip_reason = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls._qt_ready, cls._qt_skip_reason = cls._probe_qt_widget_runtime()

    @classmethod
    def _probe_qt_widget_runtime(cls):
        code = textwrap.dedent(
            """
            import sys
            from pathlib import Path

            repo = Path(__REPO__)
            sys.path.insert(0, str(repo))

            from PyQt5.QtWidgets import QApplication
            from ui.video_task_studio import VideoTaskStudio

            app = QApplication.instance() or QApplication([])
            widget = VideoTaskStudio()
            print("QT_WIDGET_READY")
            widget.close()
            app.quit()
            """
        ).replace("__REPO__", repr(str(REPO_ROOT)))
        try:
            proc = subprocess.run(
                [sys.executable, "-u", "-c", code],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=45,
            )
        except subprocess.TimeoutExpired:
            return False, "Qt widget preflight timed out"

        if proc.returncode == 0 and "QT_WIDGET_READY" in proc.stdout:
            return True, ""

        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        detail = stderr or stdout or f"process exited with code {proc.returncode}"
        return False, f"Qt widget runtime unavailable in this environment: {detail}"

    def _ensure_qt_runtime(self) -> None:
        if not self._qt_ready:
            self.skipTest(self._qt_skip_reason)

    def _run_child(self, code: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-u", "-c", textwrap.dedent(code)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_graph_frame_selector_switches_graph_and_player_frame(self) -> None:
        self._ensure_qt_runtime()
        proc = self._run_child(
            """
            import sys
            from pathlib import Path

            repo = Path(__REPO__)
            sys.path.insert(0, str(repo))

            from PyQt5.QtWidgets import QApplication
            from ui.video_task_studio import VideoTaskStudio

            app = QApplication.instance() or QApplication([])
            widget = VideoTaskStudio()
            widget._save_persisted_settings = lambda: None
            widget._set_status = lambda *args, **kwargs: None

            render_calls = {{"count": 0}}

            def fake_render():
                render_calls["count"] += 1

            def fake_seek(frame, preview_only=False):
                widget.player.current_frame = int(frame)

            widget._render_graph = fake_render
            widget._sync_seek_controls = lambda frame: None
            widget._sync_cycle_result_with_current_graph = lambda: None
            widget.player.seek = fake_seek
            widget.player.current_frame = 0
            widget.player.frame_count = 100

            graph0 = {{"image_id": "sample_f000000", "metadata": {{"graph_frame_idx": 0}}, "nodes": [], "edges": []}}
            graph12 = {{"image_id": "sample_f000012", "metadata": {{"graph_frame_idx": 12}}, "nodes": [], "edges": []}}
            graph24 = {{"image_id": "sample_f000024", "metadata": {{"graph_frame_idx": 24}}, "nodes": [], "edges": []}}

            widget.current_graph_bundle = {{"graphs": [graph0, graph12, graph24]}}
            widget.current_graph = graph0
            widget._set_graph_frame_selector(0, manual=False)

            widget.spin_frame_for_graph.setValue(13)
            widget._on_graph_frame_selector_changed(13)

            assert widget.current_graph.get("image_id") == "sample_f000012", widget.current_graph
            assert int(widget.player.current_frame) == 12, widget.player.current_frame
            assert bool(widget._graph_frame_manual) is True
            assert render_calls["count"] == 1, render_calls

            print("GRAPH_SELECTOR_OK")
            widget.close()
            app.quit()
            """
            .replace("__REPO__", repr(str(REPO_ROOT)))
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        self.assertIn("GRAPH_SELECTOR_OK", proc.stdout)

    def test_safe_probe_answer_keeps_salvaged_gemini_response_valid(self) -> None:
        self._ensure_qt_runtime()
        proc = self._run_child(
            """
            import sys
            from pathlib import Path

            repo = Path(__REPO__)
            sys.path.insert(0, str(repo))

            from ui.video_task_studio import VideoTaskStudio

            resp = VideoTaskStudio._safe_probe_answer(
                {
                    "answer": "yes",
                    "score": 0.88,
                    "reason": "fallback_parse",
                    "raw_text": "answer=yes score=0.88",
                    "finish_reason": "STOP",
                    "schema_errors": ["$.score: expected number"],
                    "parse_stage": "fallback_text",
                    "schema_valid": False,
                    "is_truncated": False,
                    "is_valid": True,
                },
                allow_uncertain=False,
                fallback_answer="no",
            )

            assert resp["answer"] == "yes", resp
            assert bool(resp["is_valid"]) is True, resp
            assert bool(resp["schema_valid"]) is False, resp
            assert resp["finish_reason"] == "STOP", resp
            assert resp["parse_stage"] == "fallback_text", resp
            assert len(resp["schema_errors"]) == 1, resp
            print("SAFE_PROBE_SALVAGE_OK")
            """
            .replace("__REPO__", repr(str(REPO_ROOT)))
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        self.assertIn("SAFE_PROBE_SALVAGE_OK", proc.stdout)

    def test_cycle_review_ui_applies_structured_bbox_choice(self) -> None:
        self._ensure_qt_runtime()
        proc = self._run_child(
            """
            import sys
            from pathlib import Path

            repo = Path(__REPO__)
            sys.path.insert(0, str(repo))

            from PyQt5.QtWidgets import QApplication
            from core.validation_log import new_change
            from ui.video_task_studio import VideoTaskStudio

            app = QApplication.instance() or QApplication([])
            widget = VideoTaskStudio()
            widget._save_persisted_settings = lambda: None
            widget._save_correction_memory_state = lambda: None
            widget._set_status = lambda *args, **kwargs: None
            widget._render_graph = lambda: None
            widget._apply_scene_graph_overlay_to_player = lambda: None
            rerun_calls = {{"count": 0}}
            widget._rerun_cycle_refine_after_graph_change = lambda **kwargs: rerun_calls.__setitem__("count", rerun_calls["count"] + 1) or True

            widget.mode_combo.setCurrentIndex(1)
            widget.validator_id_input.setText("tester")

            widget.current_graph = {{
                "image_id": "sample_f000010",
                "metadata": {{"graph_frame_idx": 10}},
                "nodes": [
                    {{
                        "entity_id": "track_1",
                        "canonical_label": "cup",
                        "bbox": [10, 10, 20, 20],
                        "mask": {{"pixels": [1]}},
                        "validator_flags": ["cycle_bbox_conflict"],
                    }},
                    {{
                        "entity_id": "track_2",
                        "canonical_label": "person",
                        "bbox": [60, 10, 20, 40],
                    }},
                ],
                "edges": [],
            }}
            widget.current_graph_bundle = {{"graphs": [widget.current_graph]}}

            before = {{
                "claim_id": "claim_bbox_track_1",
                "claim_type": "bbox",
                "subject_id": "track_1",
                "predicate": "bbox",
                "object_id": "track_2",
                "value": "",
            }}
            after = {{
                "claim_id": "claim_bbox_track_1",
                "claim_type": "bbox",
                "subject_id": "track_1",
                "predicate": "left_of",
                "object_id": "track_2",
                "question": "Which candidate box best resolves the spatial conflict?",
                "priority": 0.92,
                "target_node_id": "track_1",
                "suggested_value": "candidate_a",
                "suggested_score": 0.91,
                "resolution_options": [
                    {{
                        "value": "candidate_a",
                        "label": "Candidate A",
                        "bbox": [12, 10, 22, 20],
                        "clear_mask": True,
                        "score": 0.91,
                        "target_node_id": "track_1",
                    }},
                    {{
                        "value": "candidate_b",
                        "label": "Candidate B",
                        "bbox": [18, 10, 22, 20],
                        "clear_mask": True,
                        "score": 0.52,
                        "target_node_id": "track_1",
                    }},
                ],
            }}
            row = new_change(
                task_type="scene_graph",
                item_id="claim_bbox_track_1",
                op="cycle_arbitration",
                field_path="human_queue",
                before=before,
                after=after,
                validator_id="tester",
                round_idx=1,
                reason=after["question"],
            )
            widget._validation_changes = [row]
            widget._refresh_cycle_review_panel()
            widget.sg_cycle_review_list.setCurrentRow(0)
            widget._render_cycle_review_detail(0)

            assert widget.sg_cycle_review_choice_combo.count() == 3, widget.sg_cycle_review_choice_combo.count()
            assert "Candidate A" in widget.sg_cycle_review_detail.toPlainText()

            widget._use_cycle_review_suggested_choice()
            assert widget.sg_cycle_review_choice_combo.currentData() == "candidate_a", widget.sg_cycle_review_choice_combo.currentData()
            assert "Apply Selected Box" in widget.btn_cycle_review_confirm.text(), widget.btn_cycle_review_confirm.text()

            applied = widget._apply_change_decision(
                task_name="Video Scene Graph",
                change_id=str(row.get("change_id")),
                approved=True,
            )
            assert applied is True
            updated_bbox = list((widget.current_graph.get("nodes") or [])[0].get("bbox") or [])
            updated_mask = dict((widget.current_graph.get("nodes") or [])[0].get("mask") or {})
            assert updated_bbox == [12, 10, 22, 20], updated_bbox
            assert updated_mask.get("pixels") == [], updated_mask
            assert str(widget._validation_changes[0].get("status", "")) == "confirmed", widget._validation_changes[0]
            assert rerun_calls["count"] == 1, rerun_calls

            widget._refresh_cycle_review_panel()
            assert widget.sg_cycle_review_list.count() == 0, widget.sg_cycle_review_list.count()

            print("CYCLE_REVIEW_UI_OK")
            widget.close()
            app.quit()
            """
            .replace("__REPO__", repr(str(REPO_ROOT)))
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        self.assertIn("CYCLE_REVIEW_UI_OK", proc.stdout)

    def test_single_turn_generation_bootstraps_scene_graph_from_current_video_frame(self) -> None:
        self._ensure_qt_runtime()
        proc = self._run_child(
            """
            import sys
            from pathlib import Path

            repo = Path(__REPO__)
            sys.path.insert(0, str(repo))

            from PyQt5.QtWidgets import QApplication
            from ui.video_task_studio import VideoTaskStudio

            app = QApplication.instance() or QApplication([])
            widget = VideoTaskStudio()
            widget._save_persisted_settings = lambda: None
            widget._set_status = lambda *args, **kwargs: None
            widget._render_graph = lambda: None
            widget._sync_cycle_result_with_current_graph = lambda: None
            widget.player.cap = object()
            widget.player.current_frame = 7
            widget.player.is_playing = False
            widget.current_graph = None
            widget.current_graph_bundle = None
            widget._scene_graph_backend_preflight = lambda settings: ""
            widget._extract_frame = lambda frame_idx, cap=None: ("frame.jpg", 1280, 720, object())
            widget._infer_scene_graph_for_frame = (
                lambda *, frame_idx, img_path, image_size, frame_bgr, enable_sentence_refine: {
                    "image_id": f"sample_f{int(frame_idx):06d}",
                    "metadata": {"graph_frame_idx": int(frame_idx), "image_path": img_path},
                    "nodes": [
                        {
                            "entity_id": "track_1",
                            "canonical_label": "person",
                            "bbox": [0, 0, 100, 200],
                            "attributes": [{"slot": "state", "value": "visible"}],
                        }
                    ],
                    "edges": [],
                }
            )

            widget._generate_single_turn()

            assert isinstance(widget.current_graph, dict), widget.current_graph
            assert widget.current_graph.get("image_id") == "sample_f000007", widget.current_graph
            assert len(widget.single_turn_items) >= 2, widget.single_turn_items
            assert int(widget.spin_frame_for_graph.value()) == 7, widget.spin_frame_for_graph.value()

            print("SINGLE_TURN_BOOTSTRAP_OK")
            widget.close()
            app.quit()
            """
            .replace("__REPO__", repr(str(REPO_ROOT)))
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        self.assertIn("SINGLE_TURN_BOOTSTRAP_OK", proc.stdout)

    def test_task_settings_dialog_exposes_chatgpt_cycle_verifier_option(self) -> None:
        self._ensure_qt_runtime()
        proc = self._run_child(
            """
            import sys
            from pathlib import Path

            repo = Path(__REPO__)
            sys.path.insert(0, str(repo))

            from PyQt5.QtWidgets import QApplication
            from core.impact_sg.mllm_adapters.defaults import DEFAULT_CYCLE_PROVIDER
            from ui.video_task_studio import TaskSettingsDialog

            app = QApplication.instance() or QApplication([])
            dialog = TaskSettingsDialog(
                task_name="Video Scene Graph",
                ontology_path=str(repo / "configs" / "impact_sg_ontology.json"),
                ontology_changed_cb=lambda payload: None,
                common_settings={},
                default_common_settings={},
                task_settings={"cycle_verifier_provider": DEFAULT_CYCLE_PROVIDER},
                default_task_settings={"cycle_verifier_provider": DEFAULT_CYCLE_PROVIDER},
                ontology_status_text="ready",
                parent=None,
            )

            combo = dialog._sg_cycle_provider_combo
            assert combo is not None
            labels = [combo.itemText(i) for i in range(combo.count())]
            values = [combo.itemData(i) for i in range(combo.count())]

            assert "ChatGPT API" in labels, labels
            assert "chatgpt_api" in values, values

            combo.setCurrentIndex(values.index("chatgpt_api"))
            saved = dialog.get_task_settings()
            assert saved["cycle_verifier_provider"] == "chatgpt_api", saved

            print("TASK_SETTINGS_CHATGPT_OK")
            dialog.close()
            app.quit()
            """
            .replace("__REPO__", repr(str(REPO_ROOT)))
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        self.assertIn("TASK_SETTINGS_CHATGPT_OK", proc.stdout)

    def test_force_sync_cycle_result_reloads_saved_probe_and_caption_payload(self) -> None:
        self._ensure_qt_runtime()
        proc = self._run_child(
            """
            import sys
            from pathlib import Path

            repo = Path(__REPO__)
            sys.path.insert(0, str(repo))

            from PyQt5.QtWidgets import QApplication
            from ui.video_task_studio import VideoTaskStudio

            app = QApplication.instance() or QApplication([])
            widget = VideoTaskStudio()
            widget._save_persisted_settings = lambda: None
            widget.current_cycle_result = {
                "probe_results": [{"probe_id": "stale_probe", "question": "stale"}],
                "caption": {"caption_text": "stale caption"},
            }
            widget._cycle_result_frame_idx = 0
            widget.current_graph = {
                "image_id": "img_001",
                "nodes": [],
                "edges": [],
                "metadata": {
                    "graph_frame_idx": 0,
                    "cycle_verification": {
                        "claims": {},
                        "votes": [],
                        "probe_results": [
                            {
                                "probe_id": "saved_probe",
                                "view_type": "single_turn_vqa",
                                "question": "Is track_1 a person?",
                                "target_claim_id": "claim_label_track_1",
                                "evidence_node_ids": ["track_1"],
                                "evidence_edge_ids": [],
                                "parsed_response": {"answer": "yes", "score": 1.0},
                            }
                        ],
                        "caption": {"caption_text": "saved caption"},
                    },
                },
            }

            widget._sync_cycle_result_with_current_graph(force=True)

            assert widget.current_cycle_result["probe_results"][0]["probe_id"] == "saved_probe", widget.current_cycle_result
            assert widget.current_cycle_result["caption"]["caption_text"] == "saved caption", widget.current_cycle_result

            print("CYCLE_RESUME_SYNC_OK")
            widget.close()
            app.quit()
            """
            .replace("__REPO__", repr(str(REPO_ROOT)))
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        self.assertIn("CYCLE_RESUME_SYNC_OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
