from __future__ import annotations

from typing import Callable, Dict, List, Optional

from core.impact_sg.claim_graph import build_multi_turn_probes, build_single_turn_probes, graph_to_claims

from .policy import apply_role_policy
from .schemas import probe_response_schema


class VisualVerifierPipeline:
    """Unified visual verifier pipeline for single-turn, multi-turn and caption checks."""

    def __init__(
        self,
        *,
        verifier,
        ontology,
        cfg: Dict[str, object],
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.verifier = verifier
        self.ontology = ontology
        self.cfg = dict(cfg or {})
        self.progress_cb = progress_cb

    def _emit(self, text: str) -> None:
        if self.progress_cb is None:
            return
        try:
            self.progress_cb(str(text or "").strip())
        except Exception:
            pass

    @staticmethod
    def _regions(graph: Dict[str, object], node_ids: List[str]) -> List[Dict[str, object]]:
        node_map = {
            str(node.get("entity_id", "") or ""): dict(node)
            for node in list(graph.get("nodes") or [])
            if str(node.get("entity_id", "") or "").strip()
        }
        out: List[Dict[str, object]] = []
        for node_id in list(node_ids or []):
            node = node_map.get(str(node_id or "").strip())
            if not node:
                continue
            out.append(
                {
                    "entity_id": str(node.get("entity_id", "") or "").strip(),
                    "label": str(node.get("canonical_label", node.get("label", "")) or "").strip(),
                    "bbox": list(node.get("bbox") or [0, 0, 0, 0]),
                }
            )
        return out

    def _probe_to_vote(self, probe: Dict[str, object], resp: Dict[str, object], view_type: str) -> Dict[str, object]:
        response_format = dict(probe.get("response_format") or {})
        fmt_type = str(response_format.get("type", "") or "").strip().lower()
        score = float(resp.get("score", 0.0) or 0.0)
        if fmt_type == "selection":
            selected = str(resp.get("selection", "") or "").strip()
            expected = str(probe.get("expected_answer", "") or "").strip()
            if selected and selected.lower() == expected.lower():
                vote = "support"
            elif selected and selected != "uncertain":
                vote = "conflict"
            else:
                vote = "uncertain"
            return {
                "claim_id": str(probe.get("target_claim_id", "") or "").strip(),
                "view_type": view_type,
                "vote": vote,
                "score": score,
                "probe_id": str(probe.get("probe_id", "") or "").strip(),
                "probe_family": str(probe.get("probe_family", "constrained_correction") or "constrained_correction"),
                "correction_value": selected if vote == "conflict" else "",
                "candidate_options": list(probe.get("candidate_options") or []),
            }

        ans = str(resp.get("answer", "uncertain") or "uncertain").strip().lower()
        if ans == "yes":
            vote = "support"
        elif ans == "no":
            vote = "conflict"
        else:
            vote = "uncertain"
        return {
            "claim_id": str(probe.get("target_claim_id", "") or "").strip(),
            "view_type": view_type,
            "vote": vote,
            "score": score,
            "probe_id": str(probe.get("probe_id", "") or "").strip(),
            "probe_family": str(probe.get("probe_family", "binary_verification") or "binary_verification"),
        }

    def _run_probes(
        self,
        *,
        graph: Dict[str, object],
        image_path: str,
        probes: List[Dict[str, object]],
        view_type: str,
    ) -> Dict[str, List[Dict[str, object]]]:
        votes: List[Dict[str, object]] = []
        probe_results: List[Dict[str, object]] = []
        for probe in list(probes or []):
            schema = probe_response_schema(probe)
            response = self.verifier.answer_probe(
                image_path=image_path,
                question=str(probe.get("question", "") or ""),
                regions=self._regions(graph, list(probe.get("evidence_node_ids") or [])),
                response_format=dict(probe.get("response_format") or {}),
                schema=schema,
            )
            vote = self._probe_to_vote(probe, response, view_type)
            votes.append(vote)
            probe_results.append(
                {
                    "probe_id": str(probe.get("probe_id", "") or ""),
                    "view_type": view_type,
                    "question": str(probe.get("question", "") or ""),
                    "target_claim_id": str(probe.get("target_claim_id", "") or ""),
                    "probe_family": str(probe.get("probe_family", "") or ""),
                    "chain_id": str(probe.get("chain_id", "") or ""),
                    "turn": int(probe.get("turn", 0) or 0),
                    "schema_valid": bool(dict(response or {}).get("schema_valid", True)),
                    "response": dict(response or {}),
                }
            )
        return {"votes": votes, "probe_results": probe_results}

    def run(self, *, graph: Dict[str, object], image_path: str, correction_memory: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        cycle_cfg = dict(self.cfg.get("cycle") or {})
        caption_cfg = dict(self.cfg.get("caption") or {})
        memory = dict(correction_memory or {})

        claims = {row.claim_id: row.to_dict() for row in graph_to_claims(graph)}
        all_votes: List[Dict[str, object]] = []
        probe_results: List[Dict[str, object]] = []

        if bool(cycle_cfg.get("enable_single_turn_probes", True)):
            self._emit("[VV] single-turn probes")
            single_probes = build_single_turn_probes(graph, correction_memory=memory, ontology=self.ontology)
            batch = self._run_probes(graph=graph, image_path=image_path, probes=single_probes, view_type="single_turn_vqa")
            all_votes.extend(batch["votes"])
            probe_results.extend(batch["probe_results"])

        if bool(cycle_cfg.get("enable_multi_turn_probes", True)):
            self._emit("[VV] multi-turn probes")
            multi_probes = build_multi_turn_probes(graph, correction_memory=memory, ontology=self.ontology)
            batch = self._run_probes(graph=graph, image_path=image_path, probes=multi_probes, view_type="multi_turn_vqa")
            all_votes.extend(batch["votes"])
            probe_results.extend(batch["probe_results"])

        caption_payload: Dict[str, object] = {}
        if bool(cycle_cfg.get("enable_caption_probe", True)):
            self._emit("[VV] caption verifier")
            prompt = "Describe the image briefly in one sentence."
            cap = self.verifier.generate_caption(
                image_path=image_path,
                prompt=prompt,
                regions=self._regions(
                    graph,
                    [str(node.get("entity_id", "") or "") for node in list(graph.get("nodes") or [])],
                ),
            )
            caption_payload = dict(cap or {})
            caption_text = str(caption_payload.get("caption_text", "") or caption_payload.get("caption", "") or caption_payload.get("raw_text", "") or "").strip()
            caption_payload["caption_text"] = caption_text
            caption_payload["caption"] = caption_text

        weighted_votes, policy_report = apply_role_policy(
            all_votes,
            policy_override=dict(self.cfg.get("role_policy") or {}),
        )
        return {
            "claims": claims,
            "votes": weighted_votes,
            "probe_results": probe_results,
            "caption": caption_payload,
            "policy": policy_report,
        }
