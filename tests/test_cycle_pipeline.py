import unittest

from core.impact_sg.cycle_pipeline import (
    _probe_prompt_context,
    _probe_is_resolved,
    _resolved_scope,
    rerun_cycle_refine_for_claims,
    run_cycle_refine,
)
from core.impact_sg.claim_graph import attribute_claim_id
from core.impact_sg.mllm_adapters.base import MockVisionVerifier
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
                "validator_flags": [],
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
                "validator_flags": [],
                "risk": 0.0,
                "verified": False,
            },
            {
                "entity_id": "track_3",
                "canonical_label": "chair",
                "prompt_used": "chair",
                "mask": {"pixels": []},
                "bbox": [40, 0, 10, 10],
                "score": 0.5,
                "attributes": [],
                "provenance": [],
                "validator_flags": [],
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
                "score": 0.75,
                "evidence": [],
                "validator_flags": [],
                "risk": 0.0,
                "verified": False,
            }
        ],
        "validator_flags": [],
        "metadata": {"image_path": "frame.jpg", "graph_snapshot_id": "snap_1"},
    }


class CyclePipelineTests(unittest.TestCase):
    def test_resolved_scope_keeps_probe_level_records_out_of_claim_level_scope(self) -> None:
        graph = _graph()
        graph["metadata"] = {
            **dict(graph.get("metadata") or {}),
            "cycle_verification": {
                "resolved_claims": [
                    {"claim_id": "claim_label_track_1", "probe_id": "probe_label_track_1"}
                ]
            },
        }
        scope = _resolved_scope(graph=graph, base_result=None)
        self.assertIn("probe_label_track_1", set(scope.get("probe_ids") or set()))
        self.assertNotIn("claim_label_track_1", set(scope.get("claim_ids") or set()))

    def test_probe_resolution_filter_is_qa_item_specific_when_probe_id_exists(self) -> None:
        graph = _graph()
        graph["metadata"] = {
            **dict(graph.get("metadata") or {}),
            "cycle_verification": {
                "resolved_claims": [
                    {"claim_id": "claim_label_track_1", "probe_id": "probe_label_track_1"}
                ]
            },
        }
        scope = _resolved_scope(graph=graph, base_result=None)
        self.assertTrue(
            _probe_is_resolved(
                {"target_claim_id": "claim_label_track_1", "probe_id": "probe_label_track_1"},
                scope,
            )
        )
        self.assertFalse(
            _probe_is_resolved(
                {"target_claim_id": "claim_label_track_1", "probe_id": "probe_label_track_1_alt"},
                scope,
            )
        )

    def test_probe_resolution_filter_matches_question_and_evidence_when_ids_change(self) -> None:
        graph = _graph()
        graph["metadata"] = {
            **dict(graph.get("metadata") or {}),
            "cycle_verification": {
                "resolved_claims": [
                    {
                        "claim_id": "claim_label_track_1",
                        "probe_id": "old_probe_id",
                        "question": "What is the label of the highlighted object?",
                        "evidence_node_ids": ["track_1"],
                        "evidence_edge_ids": [],
                    }
                ]
            },
        }
        scope = _resolved_scope(graph=graph, base_result=None)
        self.assertTrue(
            _probe_is_resolved(
                {
                    "target_claim_id": "claim_label_track_1",
                    "probe_id": "new_probe_id",
                    "question": "What is the label of the highlighted object?",
                    "evidence_node_ids": ["track_1"],
                    "evidence_edge_ids": [],
                },
                scope,
            )
        )
        self.assertFalse(
            _probe_is_resolved(
                {
                    "target_claim_id": "claim_label_track_1",
                    "probe_id": "new_probe_id",
                    "question": "What is the label of the highlighted object?",
                    "evidence_node_ids": ["track_2"],
                    "evidence_edge_ids": [],
                },
                scope,
            )
        )

    def test_probe_resolution_filter_uses_persisted_resolved_key(self) -> None:
        graph = _graph()
        graph["metadata"] = {
            **dict(graph.get("metadata") or {}),
            "cycle_verification": {
                "resolved_claims": [
                    {
                        "claim_id": "claim_label_track_1",
                        "probe_id": "old_probe_id",
                        "resolved_key": "claim_label_track_1||what is the label of the highlighted object?||track_1||",
                    }
                ]
            },
        }
        scope = _resolved_scope(graph=graph, base_result=None)
        self.assertTrue(
            _probe_is_resolved(
                {
                    "target_claim_id": "claim_label_track_1",
                    "probe_id": "completely_new_probe_id",
                    "question": "What is the label of the highlighted object?",
                    "evidence_node_ids": ["track_1"],
                    "evidence_edge_ids": [],
                },
                scope,
            )
        )

    def test_probe_resolution_filter_skips_high_score_previous_probe(self) -> None:
        graph = _graph()
        graph["metadata"] = {
            **dict(graph.get("metadata") or {}),
            "cycle_verification": {
                "probe_results": [
                    {
                        "target_claim_id": "claim_label_track_1",
                        "probe_id": "probe_label_track_1_old",
                        "question": "What is the label of the highlighted object?",
                        "evidence_node_ids": ["track_1"],
                        "evidence_edge_ids": [],
                        "parsed_response": {"answer": "person", "score": 0.72, "schema_valid": True},
                    }
                ]
            },
        }
        scope = _resolved_scope(graph=graph, base_result=None)
        self.assertTrue(
            _probe_is_resolved(
                {
                    "target_claim_id": "claim_label_track_1",
                    "probe_id": "probe_label_track_1_new",
                    "question": "What is the label of the highlighted object?",
                    "evidence_node_ids": ["track_1"],
                    "evidence_edge_ids": [],
                },
                scope,
            )
        )
        self.assertTrue(
            _probe_is_resolved(
                {
                    "target_claim_id": "claim_label_track_1",
                    "probe_id": "probe_label_track_1_newer",
                    "question": "What is the label of the highlighted object?",
                    "evidence_node_ids": ["track_9"],
                    "evidence_edge_ids": [],
                },
                scope,
            )
        )

    def test_probe_resolution_filter_uses_suppressed_question_key(self) -> None:
        graph = _graph()
        graph["metadata"] = {
            **dict(graph.get("metadata") or {}),
            "cycle_verification": {
                "suppressed_questions": [
                    {
                        "question_key": "single_turn_vqa::what is the label of the highlighted object?",
                        "question": "What is the label of the highlighted object?",
                        "view_type": "single_turn_vqa",
                        "target_claim_id": "claim_label_track_1",
                        "evidence_node_ids": ["track_9"],
                        "evidence_edge_ids": [],
                    }
                ]
            },
        }
        scope = _resolved_scope(graph=graph, base_result=None)
        self.assertTrue(
            _probe_is_resolved(
                {
                    "target_claim_id": "claim_label_track_1",
                    "probe_id": "probe_label_track_1_new",
                    "view_type": "single_turn_vqa",
                    "question": "What is the label of the highlighted object?",
                    "evidence_node_ids": ["track_9"],
                    "evidence_edge_ids": [],
                },
                scope,
            )
        )

    def test_probe_resolution_filter_scopes_suppressed_question_to_claim_and_evidence(self) -> None:
        graph = _graph()
        graph["metadata"] = {
            **dict(graph.get("metadata") or {}),
            "cycle_verification": {
                "suppressed_questions": [
                    {
                        "question_key": (
                            "q::single_turn_vqa::claim_exists_track_1::track_1::::"
                            "look carefully at the entire image. is there a person (human being) visibly present in this frame? answer yes, no, or uncertain."
                        ),
                        "question": "Look carefully at the entire image. Is there a person (human being) visibly present in this frame? Answer yes, no, or uncertain.",
                        "view_type": "single_turn_vqa",
                        "target_claim_id": "claim_exists_track_1",
                        "evidence_node_ids": ["track_1"],
                        "evidence_edge_ids": [],
                    }
                ]
            },
        }
        scope = _resolved_scope(graph=graph, base_result=None)
        self.assertTrue(
            _probe_is_resolved(
                {
                    "target_claim_id": "claim_exists_track_1",
                    "probe_id": "probe_exist_track_1_new",
                    "view_type": "single_turn_vqa",
                    "question": "Look carefully at the entire image. Is there a person (human being) visibly present in this frame? Answer yes, no, or uncertain.",
                    "evidence_node_ids": ["track_1"],
                    "evidence_edge_ids": [],
                },
                scope,
            )
        )
        self.assertFalse(
            _probe_is_resolved(
                {
                    "target_claim_id": "claim_exists_track_9",
                    "probe_id": "probe_exist_track_9_new",
                    "view_type": "single_turn_vqa",
                    "question": "Look carefully at the entire image. Is there a person (human being) visibly present in this frame? Answer yes, no, or uncertain.",
                    "evidence_node_ids": ["track_9"],
                    "evidence_edge_ids": [],
                },
                scope,
            )
        )

    def test_cycle_pipeline_returns_rounds_and_refined_graph(self) -> None:
        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "person", "synonyms": []},
                    {"label": "cup", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": [], "interaction": ["holding"]},
                "question_types": [],
            }
        )
        result = run_cycle_refine(
            graph=_graph(),
            image_path="frame.jpg",
            verifier=MockVisionVerifier(),
            ontology=ontology,
            cfg={
                "cycle": {
                "enable_single_turn_probes": True,
                "enable_multi_turn_probes": True,
                "enable_caption_probe": True,
                "enable_person_focus": True,
                "focus_subject_label": "person",
                "focus_max_hops": 1,
                "focus_direct_relations_only": True,
                "max_human_queries_per_frame": 3,
                "max_revision_rounds": 2,
            },
                "caption": {"style": "technical", "require_relation_mentions": True, "max_sentences": 3},
            },
            correction_memory={
                "label_confusions": {"person": {"mannequin": 2}},
                "relation_confusions": {"holding": {"touching": 3}},
                "prompt_aliases": {"person": ["human"]},
                "verified_locks": {},
            },
        )
        self.assertTrue(result["rounds"])
        self.assertIn("graph_after", result)
        self.assertIn("claims", result)
        self.assertIn("probe_results", result)
        self.assertIsInstance(result["human_queue"], list)
        self.assertTrue(any(str(row.get("view_type")) == "multi_turn_vqa" for row in result["probe_results"]))
        self.assertTrue(any(str(row.get("chain_id", "")).startswith("chain_") for row in result["probe_results"]))
        self.assertTrue(any("human" in str(row.get("question", "")) for row in result["probe_results"]))
        self.assertIn("memory", result)
        self.assertEqual(int(result["memory"].get("prompt_aliases", 0)), 1)
        self.assertIn("summary", result)
        self.assertGreaterEqual(int(result["summary"].get("probe_count", 0)), 1)
        self.assertEqual(int(result["summary"].get("queue_count", -1)), len(result["human_queue"]))
        self.assertTrue(bool(((result.get("caption") or {}).get("feedback"))))
        self.assertTrue(bool(((result.get("caption") or {}).get("feedback") or {}).get("structured")))
        self.assertIn("agents", result)
        self.assertTrue(bool(((result.get("agents") or {}).get("captioning") or {}).get("enabled")))
        self.assertIn("metrics", result)
        self.assertIn("claim_agreement_rate", result["metrics"])
        self.assertIn("graph_caption_contradiction_rate", result["summary"])
        cycle_update = ((result.get("graph_after") or {}).get("metadata") or {}).get("cycle_update") or {}
        self.assertIn("memory_adjustments", cycle_update)
        target_claim_ids = {
            str(row.get("target_claim_id", "") or "")
            for row in result["probe_results"]
        }
        self.assertTrue(all("track_3" not in claim_id for claim_id in target_claim_ids))
        self.assertEqual(len((result.get("graph_after") or {}).get("nodes") or []), 3)
        self.assertTrue(bool((result.get("focus") or {}).get("applied")))

    def test_cycle_pipeline_accumulates_probe_results_across_rounds(self) -> None:
        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "person", "synonyms": []},
                    {"label": "cup", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": [], "interaction": ["holding"]},
                "question_types": [],
            }
        )
        result = run_cycle_refine(
            graph=_graph(),
            image_path="frame.jpg",
            verifier=MockVisionVerifier(),
            ontology=ontology,
            cfg={
                "cycle": {
                    "enable_single_turn_probes": True,
                    "enable_multi_turn_probes": True,
                    "enable_caption_probe": False,
                    "enable_person_focus": True,
                    "focus_subject_label": "person",
                    "focus_max_hops": 1,
                    "focus_direct_relations_only": True,
                    "max_human_queries_per_frame": 3,
                    "max_revision_rounds": 2,
                },
                "caption": {},
            },
            correction_memory={
                "label_confusions": {},
                "relation_confusions": {},
                "prompt_aliases": {},
                "verified_locks": {},
            },
        )
        self.assertEqual(len(result["probe_results"]), sum(len(r.get("probe_results") or []) for r in result["rounds"]))
        self.assertEqual(len(result["votes"]), sum(len(r.get("votes") or []) for r in result["rounds"]))

    def test_cycle_pipeline_uses_caption_feedback_votes(self) -> None:
        class _CaptionOnlyVerifier:
            def generate_caption(self, *, image_path: str, prompt: str, regions, schema=None):
                _ = image_path
                _ = prompt
                _ = regions
                _ = schema
                return {
                    "caption": "A person is holding a cup.",
                    "supported_entities": ["track_1", "track_2"],
                    "unsupported_entities": [],
                    "supported_attributes": [],
                    "unsupported_attributes": [],
                    "supported_relations": ["edge_1"],
                    "unsupported_relations": [],
                    "hallucinated_mentions": [],
                }

        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "person", "synonyms": []},
                    {"label": "cup", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": [], "interaction": ["holding"]},
                "question_types": [],
            }
        )
        result = run_cycle_refine(
            graph=_graph(),
            image_path="frame.jpg",
            verifier=_CaptionOnlyVerifier(),
            ontology=ontology,
            cfg={
                "cycle": {
                    "enable_single_turn_probes": False,
                    "enable_multi_turn_probes": False,
                    "enable_caption_probe": True,
                    "max_human_queries_per_frame": 3,
                    "max_revision_rounds": 1,
                },
                "caption": {
                    "style": "technical",
                    "require_relation_mentions": True,
                    "max_sentences": 3,
                    "structured_feedback": True,
                    "emit_conflict_votes": True,
                },
            },
        )
        caption_votes = [
            row for row in list(result.get("votes") or [])
            if str(row.get("view_type", "") or "").strip() == "caption"
        ]
        self.assertTrue(caption_votes)
        self.assertGreater(int((result.get("summary") or {}).get("caption_vote_count", 0) or 0), 0)
        self.assertTrue(bool(((result.get("caption") or {}).get("feedback") or {}).get("structured")))

    def test_cycle_pipeline_injects_probe_prompt_context(self) -> None:
        class _PromptContextVerifier:
            def __init__(self) -> None:
                self.calls = []

            def answer_probe(self, *, image_path: str, question: str, regions, response_format=None, schema=None):
                _ = image_path
                _ = question
                _ = regions
                _ = schema
                fmt = dict(response_format or {})
                self.calls.append(fmt)
                if str(fmt.get("type", "") or "").strip().lower() == "selection":
                    return {
                        "selection": str(fmt.get("default_selection", "uncertain") or "uncertain"),
                        "score": 0.8,
                        "reason": "default selection",
                    }
                return {"answer": "yes", "score": 0.8, "reason": "supported"}

            def generate_caption(self, *, image_path: str, prompt: str, regions):
                _ = image_path
                _ = prompt
                _ = regions
                return {"caption": ""}

        verifier = _PromptContextVerifier()
        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "person", "synonyms": []},
                    {"label": "cup", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": [], "interaction": ["holding"]},
                "question_types": [],
            }
        )
        run_cycle_refine(
            graph=_graph(),
            image_path="frame.jpg",
            verifier=verifier,
            ontology=ontology,
            cfg={
                "cycle": {
                    "enable_single_turn_probes": True,
                    "enable_multi_turn_probes": False,
                    "enable_caption_probe": False,
                    "max_human_queries_per_frame": 3,
                    "max_revision_rounds": 1,
                },
            },
        )
        prompt_contexts = [
            dict(call.get("_prompt_context") or {})
            for call in verifier.calls
            if isinstance(call, dict) and isinstance(call.get("_prompt_context"), dict)
        ]
        self.assertTrue(
            any(
                str(ctx.get("claim_type", "") or "") == "label"
                and str(ctx.get("subject_label", "") or "") == "person"
                for ctx in prompt_contexts
            )
        )
        self.assertTrue(
            any(
                str(ctx.get("claim_type", "") or "") == "relation"
                and str(ctx.get("subject_label", "") or "") == "person"
                and str(ctx.get("object_label", "") or "") == "cup"
                and str(ctx.get("current_value", "") or "") == "holding"
                for ctx in prompt_contexts
            )
        )

    def test_probe_prompt_context_uses_attribute_claim_subject_not_stale_evidence(self) -> None:
        graph = _graph()
        graph["nodes"].append(
            {
                "entity_id": "track_4",
                "canonical_label": "person",
                "prompt_used": "person",
                "mask": {"pixels": []},
                "bbox": [60, 0, 10, 10],
                "score": 0.7,
                "attributes": [
                    {"slot": "state", "value": "occluded", "confidence": 0.6, "provenance": [], "verified": False}
                ],
                "provenance": [],
                "validator_flags": [],
                "risk": 0.0,
                "verified": False,
            }
        )
        probe = {
            "target_claim_id": attribute_claim_id("track_4", "state"),
            "probe_family": "binary_verification",
            "evidence_node_ids": ["track_1"],
        }

        prompt_context = _probe_prompt_context(graph, probe)

        self.assertEqual(prompt_context.get("claim_type"), "attribute")
        self.assertEqual(prompt_context.get("subject_id"), "track_4")
        self.assertEqual(prompt_context.get("subject_label"), "person")
        self.assertEqual(prompt_context.get("slot"), "state")
        self.assertEqual(prompt_context.get("current_value"), "occluded")

    def test_targeted_cycle_rerun_replaces_only_affected_claim_state(self) -> None:
        class _TargetedVerifier:
            def answer_probe(self, *, image_path: str, question: str, regions, response_format=None, schema=None):
                _ = image_path
                _ = regions
                _ = response_format
                _ = schema
                if "left_of" in str(question):
                    return {"answer": "yes", "score": 0.96, "reason": "geometry now matches left_of"}
                return {"answer": "yes", "score": 0.81, "reason": "supported"}

            def generate_caption(self, *, image_path: str, prompt: str, regions, schema=None):
                _ = image_path
                _ = prompt
                _ = regions
                _ = schema
                return {"caption": ""}

        graph = _graph()
        graph["edges"] = [
            {
                "edge_id": "edge_1",
                "src_id": "track_1",
                "relation": "left_of",
                "dst_id": "track_2",
                "score": 0.9,
                "evidence": [],
                "validator_flags": [],
                "risk": 0.0,
                "verified": False,
            }
        ]
        graph["nodes"][0]["bbox"] = [0, 0, 10, 10]
        graph["nodes"][1]["bbox"] = [20, 0, 10, 10]
        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "person", "synonyms": []},
                    {"label": "cup", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": ["left_of", "right_of"], "interaction": []},
                "question_types": [],
            }
        )
        base_result = {
            "graph_after": {
                **graph,
                "metadata": {
                    **dict(graph.get("metadata") or {}),
                    "cycle_update": {
                        "accepted_claim_ids": ["claim_label_track_1"],
                        "flagged_claim_ids": ["claim_rel_edge_1"],
                        "memory_adjustments": [],
                        "correction_applied": [],
                        "auto_removed_node_ids": [],
                        "auto_removed_claim_ids": [],
                    },
                },
            },
            "claims": {
                "claim_label_track_1": {
                    "claim_id": "claim_label_track_1",
                    "claim_type": "label",
                    "subject_id": "track_1",
                    "predicate": "label",
                    "value": "person",
                },
                "claim_rel_edge_1": {
                    "claim_id": "claim_rel_edge_1",
                    "claim_type": "relation",
                    "subject_id": "track_1",
                    "predicate": "left_of",
                    "object_id": "track_2",
                    "value": "",
                },
            },
            "votes": [
                {"claim_id": "claim_label_track_1", "view_type": "single_turn_vqa", "vote": "support", "score": 0.88},
                {"claim_id": "claim_rel_edge_1", "view_type": "single_turn_vqa", "vote": "conflict", "score": 0.72},
            ],
            "probe_results": [
                {
                    "probe_id": "probe_label_track_1",
                    "view_type": "single_turn_vqa",
                    "target_claim_id": "claim_label_track_1",
                    "question": "Is track_1 a person?",
                    "evidence_node_ids": ["track_1"],
                    "evidence_edge_ids": [],
                    "parsed_response": {"answer": "yes", "score": 0.88, "reason": "person"},
                    "response": {"answer": "yes", "score": 0.88, "reason": "person"},
                },
                {
                    "probe_id": "probe_rel_edge_1",
                    "view_type": "single_turn_vqa",
                    "target_claim_id": "claim_rel_edge_1",
                    "question": "Does track_1 stand in relation 'left_of' to track_2?",
                    "evidence_node_ids": ["track_1", "track_2"],
                    "evidence_edge_ids": ["edge_1"],
                    "parsed_response": {"answer": "no", "score": 0.72, "reason": "old geometry mismatch"},
                    "response": {"answer": "no", "score": 0.72, "reason": "old geometry mismatch"},
                },
            ],
            "human_queue": [
                {
                    "claim_id": "claim_bbox_track_1_claim_rel_edge_1",
                    "claim_type": "bbox",
                    "source_relation_claim_id": "claim_rel_edge_1",
                    "source_relation_edge_id": "edge_1",
                    "subject_id": "track_1",
                    "object_id": "track_2",
                    "target_node_id": "track_1",
                    "evidence_node_ids": ["track_1", "track_2"],
                    "question": "Which bounding box best restores the spatial relation?",
                }
            ],
            "correction_candidates": {
                "claim_rel_edge_1": {
                    "best_value": "right_of",
                    "best_score": 0.61,
                    "options": ["left_of", "right_of"],
                }
            },
            "caption": {"feedback": {"structured": False, "vote_count": 0}},
            "rounds": [],
            "runtime": {"verifier_provider": "mock"},
            "resolved_claims": [],
        }
        result = rerun_cycle_refine_for_claims(
            graph=graph,
            image_path="frame.jpg",
            verifier=_TargetedVerifier(),
            ontology=ontology,
            cfg={
                "cycle": {
                    "enable_single_turn_probes": True,
                    "enable_multi_turn_probes": False,
                    "enable_caption_probe": False,
                    "enable_geometry_review": True,
                    "max_human_queries_per_frame": 3,
                    "max_revision_rounds": 1,
                },
            },
            target_claim_ids=["claim_rel_edge_1"],
            base_result=base_result,
        )
        self.assertTrue(bool((result.get("runtime") or {}).get("targeted_reverify")))
        relation_votes = [
            row
            for row in list(result.get("votes") or [])
            if str(row.get("claim_id", "") or "") == "claim_rel_edge_1"
        ]
        self.assertTrue(relation_votes)
        self.assertTrue(all(str(row.get("vote", "") or "") == "support" for row in relation_votes))
        self.assertTrue(
            any(
                str(row.get("claim_id", "") or "") == "claim_label_track_1"
                for row in list(result.get("votes") or [])
            )
        )
        self.assertFalse(
            any(
                str(row.get("source_relation_claim_id", "") or "") == "claim_rel_edge_1"
                for row in list(result.get("human_queue") or [])
            )
        )
        cycle_update = dict((((result.get("graph_after") or {}).get("metadata") or {}).get("cycle_update")) or {})
        self.assertFalse("claim_rel_edge_1" in list(cycle_update.get("flagged_claim_ids") or []))
        self.assertTrue("claim_label_track_1" in list(cycle_update.get("accepted_claim_ids") or []))

    def test_targeted_cycle_rerun_keeps_resolved_probe_records(self) -> None:
        class _TargetedVerifier:
            def answer_probe(self, *, image_path: str, question: str, regions, response_format=None, schema=None):
                _ = image_path
                _ = question
                _ = regions
                _ = response_format
                _ = schema
                return {"answer": "yes", "score": 0.91, "reason": "supported"}

            def generate_caption(self, *, image_path: str, prompt: str, regions, schema=None):
                _ = image_path
                _ = prompt
                _ = regions
                _ = schema
                return {"caption": ""}

        graph = _graph()
        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "person", "synonyms": []},
                    {"label": "cup", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": ["left_of"], "interaction": ["holding"]},
                "question_types": [],
            }
        )
        resolved_row = {
            "claim_id": "claim_label_track_1",
            "probe_id": "probe_label_track_1",
            "question": "Is track_1 a person?",
            "evidence_node_ids": ["track_1"],
            "evidence_edge_ids": [],
            "resolved_key": "claim_label_track_1||is track_1 a person?||track_1||",
        }
        base_result = {
            "graph_after": dict(graph),
            "claims": {},
            "votes": [],
            "probe_results": [],
            "human_queue": [],
            "correction_candidates": {},
            "caption": {"feedback": {"structured": False, "vote_count": 0}},
            "rounds": [],
            "runtime": {"verifier_provider": "mock"},
            "resolved_claims": [dict(resolved_row)],
        }
        result = rerun_cycle_refine_for_claims(
            graph=graph,
            image_path="frame.jpg",
            verifier=_TargetedVerifier(),
            ontology=ontology,
            cfg={
                "cycle": {
                    "enable_single_turn_probes": True,
                    "enable_multi_turn_probes": False,
                    "enable_caption_probe": False,
                    "enable_geometry_review": False,
                    "max_human_queries_per_frame": 1,
                    "max_revision_rounds": 1,
                },
            },
            target_claim_ids=["claim_label_track_1"],
            base_result=base_result,
        )
        resolved_claims = [dict(row) for row in list(result.get("resolved_claims") or []) if isinstance(row, dict)]
        self.assertTrue(
            any(str(row.get("resolved_key", "") or "") == str(resolved_row.get("resolved_key", "") or "") for row in resolved_claims)
        )

    def test_targeted_cycle_rerun_expands_related_relation_claims_from_node_scope(self) -> None:
        class _NodeScopeVerifier:
            def answer_probe(self, *, image_path: str, question: str, regions, response_format=None, schema=None):
                _ = image_path
                _ = regions
                _ = response_format
                _ = schema
                if "left_of" in str(question):
                    return {"answer": "yes", "score": 0.95, "reason": "updated bbox now supports left_of"}
                return {"answer": "yes", "score": 0.82, "reason": "supported"}

            def generate_caption(self, *, image_path: str, prompt: str, regions, schema=None):
                _ = image_path
                _ = prompt
                _ = regions
                _ = schema
                return {"caption": ""}

        graph = _graph()
        graph["edges"] = [
            {
                "edge_id": "edge_1",
                "src_id": "track_1",
                "relation": "left_of",
                "dst_id": "track_2",
                "score": 0.9,
                "evidence": [],
                "validator_flags": [],
                "risk": 0.0,
                "verified": False,
            }
        ]
        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "person", "synonyms": []},
                    {"label": "cup", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": ["left_of", "right_of"], "interaction": []},
                "question_types": [],
            }
        )
        base_result = {
            "graph_after": {
                **graph,
                "metadata": {
                    **dict(graph.get("metadata") or {}),
                    "cycle_update": {
                        "accepted_claim_ids": ["claim_label_track_1"],
                        "flagged_claim_ids": ["claim_rel_edge_1"],
                        "memory_adjustments": [],
                        "correction_applied": [],
                        "auto_removed_node_ids": [],
                        "auto_removed_claim_ids": [],
                    },
                },
            },
            "claims": {
                "claim_label_track_1": {
                    "claim_id": "claim_label_track_1",
                    "claim_type": "label",
                    "subject_id": "track_1",
                    "predicate": "label",
                    "value": "person",
                },
                "claim_rel_edge_1": {
                    "claim_id": "claim_rel_edge_1",
                    "claim_type": "relation",
                    "subject_id": "track_1",
                    "predicate": "left_of",
                    "object_id": "track_2",
                    "value": "",
                },
            },
            "votes": [
                {"claim_id": "claim_label_track_1", "view_type": "single_turn_vqa", "vote": "support", "score": 0.88},
                {"claim_id": "claim_rel_edge_1", "view_type": "single_turn_vqa", "vote": "conflict", "score": 0.72},
            ],
            "probe_results": [
                {
                    "probe_id": "probe_label_track_1",
                    "view_type": "single_turn_vqa",
                    "target_claim_id": "claim_label_track_1",
                    "question": "Is track_1 a person?",
                    "evidence_node_ids": ["track_1"],
                    "evidence_edge_ids": [],
                    "parsed_response": {"answer": "yes", "score": 0.88, "reason": "person"},
                    "response": {"answer": "yes", "score": 0.88, "reason": "person"},
                },
                {
                    "probe_id": "probe_rel_edge_1",
                    "view_type": "single_turn_vqa",
                    "target_claim_id": "claim_rel_edge_1",
                    "question": "Does track_1 stand in relation 'left_of' to track_2?",
                    "evidence_node_ids": ["track_1", "track_2"],
                    "evidence_edge_ids": ["edge_1"],
                    "parsed_response": {"answer": "no", "score": 0.72, "reason": "old geometry mismatch"},
                    "response": {"answer": "no", "score": 0.72, "reason": "old geometry mismatch"},
                },
            ],
            "human_queue": [
                {
                    "claim_id": "claim_bbox_track_1_claim_rel_edge_1",
                    "claim_type": "bbox",
                    "source_relation_claim_id": "claim_rel_edge_1",
                    "source_relation_edge_id": "edge_1",
                    "subject_id": "track_1",
                    "object_id": "track_2",
                    "target_node_id": "track_1",
                    "evidence_node_ids": ["track_1", "track_2"],
                    "question": "Which bounding box best restores the spatial relation?",
                }
            ],
            "correction_candidates": {},
            "caption": {"feedback": {"structured": False, "vote_count": 0}},
            "rounds": [],
            "runtime": {"verifier_provider": "mock"},
            "resolved_claims": [],
        }
        result = rerun_cycle_refine_for_claims(
            graph=graph,
            image_path="frame.jpg",
            verifier=_NodeScopeVerifier(),
            ontology=ontology,
            cfg={
                "cycle": {
                    "enable_single_turn_probes": True,
                    "enable_multi_turn_probes": False,
                    "enable_caption_probe": False,
                    "enable_geometry_review": True,
                    "max_human_queries_per_frame": 3,
                    "max_revision_rounds": 1,
                },
            },
            target_claim_ids=["claim_label_track_1"],
            base_result=base_result,
        )
        relation_votes = [
            row
            for row in list(result.get("votes") or [])
            if str(row.get("claim_id", "") or "") == "claim_rel_edge_1"
        ]
        self.assertTrue(relation_votes)
        self.assertTrue(all(str(row.get("vote", "") or "") == "support" for row in relation_votes))
        self.assertGreaterEqual(int((result.get("runtime") or {}).get("target_claim_count", 0) or 0), 2)
        cycle_update = dict((((result.get("graph_after") or {}).get("metadata") or {}).get("cycle_update")) or {})
        self.assertFalse("claim_rel_edge_1" in list(cycle_update.get("flagged_claim_ids") or []))

    def test_targeted_cycle_rerun_keeps_current_graph_when_focus_filters_target_claim(self) -> None:
        class _NoopVerifier:
            def answer_probe(self, *, image_path: str, question: str, regions, response_format=None, schema=None):
                _ = image_path
                _ = question
                _ = regions
                _ = response_format
                _ = schema
                return {"answer": "yes", "score": 0.8, "reason": "supported"}

            def generate_caption(self, *, image_path: str, prompt: str, regions, schema=None):
                _ = image_path
                _ = prompt
                _ = regions
                _ = schema
                return {"caption": ""}

        graph = _graph()
        graph["nodes"][2]["bbox"] = [55, 0, 12, 12]
        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "person", "synonyms": []},
                    {"label": "cup", "synonyms": []},
                    {"label": "chair", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": [], "interaction": ["holding"]},
                "question_types": [],
            }
        )
        old_graph = _graph()
        base_result = {
            "graph_after": old_graph,
            "claims": {},
            "votes": [],
            "probe_results": [],
            "human_queue": [],
            "correction_candidates": {},
            "caption": {"feedback": {"structured": False, "vote_count": 0}},
            "rounds": [],
            "runtime": {"verifier_provider": "mock"},
            "resolved_claims": [],
        }
        result = rerun_cycle_refine_for_claims(
            graph=graph,
            image_path="frame.jpg",
            verifier=_NoopVerifier(),
            ontology=ontology,
            cfg={
                "cycle": {
                    "enable_single_turn_probes": True,
                    "enable_multi_turn_probes": False,
                    "enable_caption_probe": False,
                    "enable_person_focus": True,
                    "focus_subject_label": "person",
                    "focus_max_hops": 1,
                    "focus_direct_relations_only": True,
                    "max_human_queries_per_frame": 3,
                    "max_revision_rounds": 1,
                },
            },
            target_claim_ids=["claim_label_track_3"],
            base_result=base_result,
        )
        updated_bbox = list(((result.get("graph_after") or {}).get("nodes") or [])[2].get("bbox") or [])
        self.assertEqual(updated_bbox, [55, 0, 12, 12])

    def test_cycle_pipeline_applies_constrained_label_correction(self) -> None:
        class _CorrectionVerifier:
            def answer_probe(self, *, image_path: str, question: str, regions, response_format=None):
                fmt = dict(response_format or {})
                if str(fmt.get("type", "") or "") == "selection":
                    return {"selection": "person", "score": 0.93, "reason": "selected constrained correction"}
                return {"answer": "uncertain", "score": 0.2, "reason": "skip binary"}

            def generate_caption(self, *, image_path: str, prompt: str, regions):
                return {"caption": ""}

        graph = _graph()
        graph["nodes"][0]["canonical_label"] = "object"
        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "object", "synonyms": []},
                    {"label": "person", "synonyms": ["human"]},
                    {"label": "cup", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": [], "interaction": ["holding"]},
                "question_types": [],
            }
        )
        result = run_cycle_refine(
            graph=graph,
            image_path="frame.jpg",
            verifier=_CorrectionVerifier(),
            ontology=ontology,
            cfg={
                "cycle": {
                    "enable_single_turn_probes": True,
                    "enable_multi_turn_probes": False,
                    "enable_caption_probe": False,
                    "max_human_queries_per_frame": 3,
                    "max_revision_rounds": 1,
                },
                "caption": {"style": "technical", "require_relation_mentions": True, "max_sentences": 3},
            },
        )
        node = (result.get("graph_after") or {}).get("nodes", [])[0]
        self.assertEqual("person", str(node.get("canonical_label", "")))
        self.assertTrue(bool(((result.get("graph_after") or {}).get("metadata") or {}).get("cycle_update", {}).get("correction_applied")))

    def test_cycle_pipeline_enqueues_geometry_review_for_spatial_conflict(self) -> None:
        class _GeometryVerifier:
            def answer_probe(self, *, image_path: str, question: str, regions, response_format=None):
                if "relation 'left_of'" in str(question):
                    return {"answer": "no", "score": 0.92, "reason": "spatial relation does not match geometry"}
                return {"answer": "yes", "score": 0.9, "reason": "entity is visible"}

            def generate_caption(self, *, image_path: str, prompt: str, regions):
                return {"caption": ""}

        graph = _graph()
        graph["edges"] = [
            {
                "edge_id": "edge_1",
                "src_id": "track_1",
                "relation": "left_of",
                "dst_id": "track_2",
                "score": 0.9,
                "evidence": [],
                "validator_flags": [],
                "risk": 0.0,
                "verified": False,
            }
        ]
        graph["nodes"][0]["bbox"] = [40, 0, 10, 10]
        graph["nodes"][1]["bbox"] = [20, 0, 10, 10]
        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "person", "synonyms": []},
                    {"label": "cup", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": ["left_of", "right_of"], "interaction": []},
                "question_types": [],
            }
        )
        result = run_cycle_refine(
            graph=graph,
            image_path="frame.jpg",
            verifier=_GeometryVerifier(),
            ontology=ontology,
            cfg={
                "cycle": {
                    "enable_single_turn_probes": True,
                    "enable_multi_turn_probes": False,
                    "enable_caption_probe": False,
                    "enable_geometry_review": True,
                    "geometry_conflict_threshold": 0.6,
                    "max_geometry_queries_per_frame": 2,
                    "max_human_queries_per_frame": 4,
                    "max_revision_rounds": 1,
                },
                "caption": {"style": "technical", "require_relation_mentions": True, "max_sentences": 3},
            },
        )
        geometry_items = [
            row for row in list(result.get("human_queue") or [])
            if str(row.get("claim_type", "") or "").strip().lower() == "bbox"
        ]
        self.assertTrue(geometry_items)
        self.assertTrue(list(geometry_items[0].get("resolution_options") or []))
        self.assertTrue(bool(str(geometry_items[0].get("suggested_value", "") or "")))

    def test_cycle_pipeline_can_disable_hitl_queue(self) -> None:
        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "person", "synonyms": []},
                    {"label": "cup", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": [], "interaction": ["holding"]},
                "question_types": [],
            }
        )
        result = run_cycle_refine(
            graph=_graph(),
            image_path="frame.jpg",
            verifier=MockVisionVerifier(),
            ontology=ontology,
            cfg={
                "cycle": {
                    "enable_single_turn_probes": True,
                    "enable_multi_turn_probes": True,
                    "enable_caption_probe": True,
                    "enable_geometry_review": False,
                    "max_human_queries_per_frame": 0,
                    "max_revision_rounds": 1,
                },
            },
        )
        self.assertEqual([], list(result.get("human_queue") or []))
        self.assertEqual(0, int((result.get("summary") or {}).get("queue_count", -1)))

    def test_cycle_pipeline_uses_temporal_context_for_multi_turn_metrics(self) -> None:
        graph = _graph()
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
        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "person", "synonyms": []},
                    {"label": "cup", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": [], "interaction": ["holding"]},
                "question_types": [],
            }
        )
        result = run_cycle_refine(
            graph=graph,
            image_path="frame.jpg",
            verifier=MockVisionVerifier(),
            ontology=ontology,
            cfg={
                "cycle": {
                    "enable_single_turn_probes": False,
                    "enable_multi_turn_probes": True,
                    "enable_caption_probe": False,
                    "max_human_queries_per_frame": 3,
                    "max_revision_rounds": 1,
                },
            },
        )
        multi_agent = dict((result.get("agents") or {}).get("multi_turn_vqa") or {})
        self.assertGreater(int(multi_agent.get("temporal_probe_count", 0) or 0), 0)
        self.assertGreater(float((result.get("metrics") or {}).get("temporal_multi_turn_share", 0.0) or 0.0), 0.0)

    def test_cycle_pipeline_uses_graph_frame_idx_for_verified_locks(self) -> None:
        graph = {
            "image_id": "img_lock_001",
            "nodes": [
                {
                    "entity_id": "track_1",
                    "canonical_label": "object",
                    "prompt_used": "object",
                    "mask": {"pixels": []},
                    "bbox": [0, 0, 10, 10],
                    "score": 0.2,
                    "attributes": [],
                    "provenance": [],
                    "validator_flags": [],
                    "risk": 0.0,
                    "verified": False,
                }
            ],
            "edges": [],
            "validator_flags": [],
            "metadata": {
                "image_path": "frame.jpg",
                "graph_snapshot_id": "snap_lock",
                "graph_frame_idx": 12,
            },
        }
        ontology = ontology_from_payload(
            {
                "canonical_entities": [
                    {"label": "object", "synonyms": []},
                ],
                "relation_vocabulary": {"spatial": [], "interaction": []},
                "question_types": [],
            }
        )
        result = run_cycle_refine(
            graph=graph,
            image_path="frame.jpg",
            verifier=MockVisionVerifier(),
            ontology=ontology,
            cfg={
                "cycle": {
                    "enable_single_turn_probes": False,
                    "enable_multi_turn_probes": False,
                    "enable_caption_probe": False,
                    "max_human_queries_per_frame": 3,
                    "max_revision_rounds": 1,
                },
            },
            correction_memory={
                "verified_locks": {
                    "track_1": {
                        "status": "confirmed",
                        "frame_start": 12,
                        "frame_end": 12,
                    }
                }
            },
        )
        self.assertEqual([], list(result.get("human_queue") or []))
        memory_adjustments = list(
            (((result.get("graph_after") or {}).get("metadata") or {}).get("cycle_update") or {}).get(
                "memory_adjustments",
                [],
            )
        )
        self.assertTrue(
            any(
                str(row.get("subject_id", "") or "") == "track_1"
                and bool(row.get("locked", False))
                for row in memory_adjustments
            )
        )


if __name__ == "__main__":
    unittest.main()
