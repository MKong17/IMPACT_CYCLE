from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple

from .llm_verifier import HeuristicVerifier, VerificationTarget
from .pvsg_reference import evaluate_graph_against_pvsg
from .sam_verifier import detect_main_actor_missing, verify_sam_mask
from .scoring import (
    ReliabilitySignals,
    reliability_score,
    review_priority_score,
    stage_module_score,
    workflow_score,
)


STAGE_T = "Tracks"
STAGE_A = "Attributes"
STAGE_E = "Edges"
STAGE_S = "Semantic Dynamics"
STAGE_G = "Global Summary"

STATUS_GROUNDED = "grounded"
STATUS_WEAK = "weak"
STATUS_CONFLICTING = "conflicting"
STATUS_UNSUPPORTED = "unsupported"

WARN_YELLOW = "yellow"
WARN_RED = "red"
WARN_PURPLE = "purple"


@dataclass
class StageValidationItem:
    stage: str
    target_type: str
    target_id: str
    label: str
    reliability: float
    priority: float
    status: str
    warning: str
    reasons: List[str]
    suggested_action: str
    llm_check: Dict[str, Any]
    signals: Dict[str, float]
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": str(self.stage),
            "target_type": str(self.target_type),
            "target_id": str(self.target_id),
            "label": str(self.label),
            "reliability": float(self.reliability),
            "priority": float(self.priority),
            "status": str(self.status),
            "warning": str(self.warning),
            "reasons": [str(x) for x in list(self.reasons or [])],
            "suggested_action": str(self.suggested_action),
            "llm_check": dict(self.llm_check or {}),
            "signals": dict(self.signals or {}),
            "payload": dict(self.payload or {}),
        }


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _status_from_scores(*, reliability: float, conflict_score: float) -> str:
    if float(conflict_score) >= 0.55:
        return STATUS_CONFLICTING
    if float(reliability) < 0.30:
        return STATUS_UNSUPPORTED
    if float(reliability) < 0.60:
        return STATUS_WEAK
    return STATUS_GROUNDED


def _warning_from_status(*, status: str, priority: float) -> str:
    if status in {STATUS_CONFLICTING, STATUS_UNSUPPORTED}:
        return WARN_RED
    if float(priority) >= 0.75:
        return WARN_PURPLE
    return WARN_YELLOW


def _edge_nodes_map(nodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in nodes:
        if not isinstance(row, dict):
            continue
        key = str(row.get("entity_id", "") or "").strip()
        if key:
            out[key] = row
    return out


def _extract_summary_statements(graph: Dict[str, Any]) -> List[str]:
    statements: List[str] = []
    metadata = dict(graph.get("metadata") or {})
    summary_text = str(metadata.get("global_summary", "") or "").strip()
    if summary_text:
        statements.extend([s.strip() for s in summary_text.replace("\n", " ").split(".") if s.strip()])
    if not statements:
        n = len(list(graph.get("nodes") or []))
        e = len(list(graph.get("edges") or []))
        statements.append(f"Scene contains {n} entities and {e} relations.")
    dedup: List[str] = []
    seen: set[str] = set()
    for s in statements:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            dedup.append(s)
    return dedup


def _summary_token_set(text: str) -> set[str]:
    return {
        str(tok).strip().lower()
        for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]+", str(text or "").lower())
        if len(str(tok)) >= 3
    }


def _summary_grounding_metrics(
    *,
    statement: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> Tuple[float, float, List[str]]:
    stmt_tokens = _summary_token_set(statement)
    reasons: List[str] = []
    if not stmt_tokens:
        return 0.15, 0.65, ["summary_statement_empty"]

    label_counts: Dict[str, int] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        label = str(node.get("canonical_label", "") or "").strip().lower()
        if label:
            label_counts[label] = int(label_counts.get(label, 0) + 1)
    top_labels = [k for k, _v in sorted(label_counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:6]]
    label_hits = [x for x in top_labels if x in stmt_tokens]
    label_cov = float(len(label_hits)) / float(max(1, len(top_labels)))

    relation_counts: Dict[str, int] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        rel = str(edge.get("relation", "") or "").strip().lower()
        if rel:
            relation_counts[rel] = int(relation_counts.get(rel, 0) + 1)
    top_rels = [k for k, _v in sorted(relation_counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:6]]
    rel_hits = [x for x in top_rels if x in stmt_tokens]
    rel_cov = float(len(rel_hits)) / float(max(1, len(top_rels))) if top_rels else 0.5

    actor_hint = 1.0 if ("person" in stmt_tokens or "people" in stmt_tokens or "actor" in stmt_tokens) else 0.0
    density = _clamp01(float(len(nodes)) / 8.0)
    support = _clamp01(0.10 + 0.45 * label_cov + 0.25 * rel_cov + 0.10 * actor_hint + 0.10 * density)
    conflict = _clamp01((1.0 - support) * 0.65)
    if label_cov < 0.25:
        reasons.append("summary_low_label_grounding")
    if rel_cov < 0.20 and len(top_rels) > 0:
        reasons.append("summary_low_relation_grounding")
    return support, conflict, reasons


def _importance_for_item(stage: str, row: Dict[str, Any], *, main_actor_missing: bool) -> float:
    if stage == STAGE_T:
        label = str(row.get("canonical_label", "") or "").strip().lower()
        if label == "person":
            return 1.4 if not main_actor_missing else 1.6
        return 1.0
    if stage == STAGE_S:
        return 1.35
    if stage == STAGE_E:
        return 1.2
    if stage == STAGE_G:
        return 1.3
    return 1.0


def _propagation_for_item(stage: str, row: Dict[str, Any]) -> float:
    if stage == STAGE_T:
        return 1.4
    if stage == STAGE_A:
        return 1.1
    if stage == STAGE_E:
        return 1.25
    if stage == STAGE_S:
        return 1.4
    if stage == STAGE_G:
        return 1.35
    return 1.0


class StageValidator:
    def __init__(self) -> None:
        self._verifier = HeuristicVerifier()

    def _verify_with_llm(
        self,
        *,
        target_type: str,
        target_id: str,
        payload: Dict[str, Any],
        image_path: str = "",
    ) -> Dict[str, Any]:
        target = VerificationTarget(
            target_type=str(target_type),
            target_id=str(target_id),
            payload=payload,
            image_path=str(image_path or ""),
        )
        return self._verifier.verify(target).to_dict()

    def _build_track_items(
        self,
        *,
        graph: Dict[str, Any],
    ) -> List[StageValidationItem]:
        _ = graph
        # Tracking stage has been removed from the pipeline.
        return []

    def _build_attribute_items(
        self,
        *,
        graph: Dict[str, Any],
        node_map: Dict[str, Dict[str, Any]],
        llm_attr_available: bool = True,
    ) -> List[StageValidationItem]:
        if not bool(llm_attr_available):
            return []
        out: List[StageValidationItem] = []
        for node in list(graph.get("nodes") or []):
            if not isinstance(node, dict):
                continue
            eid = str(node.get("entity_id", "") or "").strip()
            label = str(node.get("canonical_label", "") or "").strip()
            attrs = list(node.get("attributes") or [])
            node_risk = _clamp01(float(node.get("risk", 0.0) or 0.0))
            for idx, attr in enumerate(attrs):
                if not isinstance(attr, dict):
                    continue
                slot = str(attr.get("slot", "") or "").strip() or f"attr_{idx}"
                value = str(attr.get("value", "") or "").strip() or "unknown"
                conf = _clamp01(float(attr.get("confidence", node.get("score", 0.0)) or 0.0))
                s_temp = _clamp01(1.0 - node_risk)
                s_struct = 0.8 if value != "unknown" else 0.5
                s_cross = 0.6
                s_support = conf
                s_conflict = _clamp01(node_risk * 0.6)
                signals = ReliabilitySignals(conf, s_temp, s_struct, s_cross, s_support, s_conflict)
                r = reliability_score(signals)
                aid = f"{eid}::attr::{slot}"
                llm = self._verify_with_llm(
                    target_type="attribute",
                    target_id=aid,
                    payload={"estimated_support": r, "estimated_conflict": s_conflict, "slot": slot, "value": value},
                    image_path=str((graph.get("metadata") or {}).get("image_path", "") or ""),
                )
                status = _status_from_scores(reliability=r, conflict_score=float(llm.get("conflict_score", 0.0) or 0.0))
                reasons = list(llm.get("reasons") or [])
                if conf < 0.35:
                    reasons.append("low_classifier_confidence")
                if value == "unknown":
                    reasons.append("visually_ambiguous_or_missing")
                p = review_priority_score(
                    reliability=r,
                    importance_weight=_importance_for_item(STAGE_A, attr, main_actor_missing=False),
                    propagation_weight=_propagation_for_item(STAGE_A, attr),
                )
                out.append(
                    StageValidationItem(
                        stage=STAGE_A,
                        target_type="attribute",
                        target_id=aid,
                        label=f"{label}:{slot}={value}",
                        reliability=r,
                        priority=p,
                        status=status,
                        warning=_warning_from_status(status=status, priority=p),
                        reasons=sorted(set([str(x) for x in reasons if str(x).strip()])),
                        suggested_action=str(llm.get("suggested_action", "review") or "review"),
                        llm_check=llm,
                        signals={
                            "s_det": signals.s_det,
                            "s_temp": signals.s_temp,
                            "s_struct": signals.s_struct,
                            "s_cross": signals.s_cross,
                            "s_support": signals.s_support,
                            "s_conflict": signals.s_conflict,
                        },
                        payload={"entity_id": eid, "slot": slot, "value": value, "attribute": dict(attr), "node": dict(node_map.get(eid) or {})},
                    )
                )
        return out

    def _build_edge_items(
        self,
        *,
        graph: Dict[str, Any],
        node_map: Dict[str, Dict[str, Any]],
    ) -> List[StageValidationItem]:
        out: List[StageValidationItem] = []
        for edge in list(graph.get("edges") or []):
            if not isinstance(edge, dict):
                continue
            eid = str(edge.get("edge_id", "") or "").strip()
            src = str(edge.get("src_id", "") or "").strip()
            dst = str(edge.get("dst_id", "") or "").strip()
            rel = str(edge.get("relation", "") or "").strip()
            src_node = dict(node_map.get(src) or {})
            dst_node = dict(node_map.get(dst) or {})
            src_risk = _clamp01(float(src_node.get("risk", 0.0) or 0.0))
            dst_risk = _clamp01(float(dst_node.get("risk", 0.0) or 0.0))
            edge_risk = _clamp01(float(edge.get("risk", 0.0) or 0.0))
            support = _clamp01(1.0 - (0.45 * edge_risk + 0.3 * max(src_risk, dst_risk)))
            conflict = _clamp01(edge_risk + 0.2 * max(src_risk, dst_risk))
            s_temp = _clamp01(1.0 - edge_risk)
            signals = ReliabilitySignals(
                s_det=support,
                s_temp=s_temp,
                s_struct=_clamp01(1.0 - max(src_risk, dst_risk)),
                s_cross=0.55,
                s_support=support,
                s_conflict=conflict,
            )
            r = reliability_score(signals)
            llm = self._verify_with_llm(
                target_type="edge",
                target_id=eid or f"{src}->{rel}->{dst}",
                payload={"estimated_support": r, "estimated_conflict": conflict, "relation": rel},
                image_path=str((graph.get("metadata") or {}).get("image_path", "") or ""),
            )
            status = _status_from_scores(reliability=r, conflict_score=float(llm.get("conflict_score", 0.0) or 0.0))
            reasons = list(llm.get("reasons") or [])
            if edge_risk > 0.55:
                reasons.append("weak_spatial_evidence")
            if max(src_risk, dst_risk) > 0.6:
                reasons.append("depends_on_uncertain_nodes")
            p = review_priority_score(
                reliability=r,
                importance_weight=_importance_for_item(STAGE_E, edge, main_actor_missing=False),
                propagation_weight=_propagation_for_item(STAGE_E, edge),
            )
            out.append(
                StageValidationItem(
                    stage=STAGE_E,
                    target_type="edge",
                    target_id=eid or f"{src}->{rel}->{dst}",
                    label=f"{src} -{rel}-> {dst}",
                    reliability=r,
                    priority=p,
                    status=status,
                    warning=_warning_from_status(status=status, priority=p),
                    reasons=sorted(set([str(x) for x in reasons if str(x).strip()])),
                    suggested_action=str(llm.get("suggested_action", "review") or "review"),
                    llm_check=llm,
                    signals={
                        "s_det": signals.s_det,
                        "s_temp": signals.s_temp,
                        "s_struct": signals.s_struct,
                        "s_cross": signals.s_cross,
                        "s_support": signals.s_support,
                        "s_conflict": signals.s_conflict,
                    },
                    payload=dict(edge),
                )
            )
        return out

    def _build_dynamic_items(
        self,
        *,
        graph: Dict[str, Any],
    ) -> List[StageValidationItem]:
        out: List[StageValidationItem] = []
        dynamic_events = list((graph.get("metadata") or {}).get("dynamic_events") or [])
        for idx, row in enumerate(dynamic_events):
            if not isinstance(row, dict):
                continue
            did = str(row.get("event_id", f"dynamic_event_{idx}") or f"dynamic_event_{idx}")
            support = _clamp01(float(row.get("confidence", 0.5) or 0.5))
            conflict = _clamp01(1.0 - support)
            signals = ReliabilitySignals(support, 0.5, 0.6, 0.55, support, conflict)
            r = reliability_score(signals)
            p = review_priority_score(
                reliability=r,
                importance_weight=_importance_for_item(STAGE_S, row, main_actor_missing=False),
                propagation_weight=_propagation_for_item(STAGE_S, row),
            )
            out.append(
                StageValidationItem(
                    stage=STAGE_S,
                    target_type="dynamic",
                    target_id=did,
                    label=str(row.get("label", did) or did),
                    reliability=r,
                    priority=p,
                    status=_status_from_scores(reliability=r, conflict_score=conflict),
                    warning=_warning_from_status(
                        status=_status_from_scores(reliability=r, conflict_score=conflict),
                        priority=p,
                    ),
                    reasons=[],
                    suggested_action="review",
                    llm_check={},
                    signals={
                        "s_det": signals.s_det,
                        "s_temp": signals.s_temp,
                        "s_struct": signals.s_struct,
                        "s_cross": signals.s_cross,
                        "s_support": signals.s_support,
                        "s_conflict": signals.s_conflict,
                    },
                    payload=dict(row),
                )
            )
        return out

    def _build_summary_items(
        self,
        *,
        graph: Dict[str, Any],
        llm_summary_available: bool = True,
    ) -> List[StageValidationItem]:
        if not bool(llm_summary_available):
            return []
        out: List[StageValidationItem] = []
        stmts = _extract_summary_statements(graph)
        nodes = [dict(x) for x in list(graph.get("nodes") or []) if isinstance(x, dict)]
        edges = [dict(x) for x in list(graph.get("edges") or []) if isinstance(x, dict)]
        for idx, stmt in enumerate(stmts):
            sid = f"summary_stmt_{idx}"
            support, conflict, grounding_reasons = _summary_grounding_metrics(
                statement=stmt,
                nodes=nodes,
                edges=edges,
            )
            signals = ReliabilitySignals(
                s_det=support,
                s_temp=0.72,
                s_struct=support,
                s_cross=0.66,
                s_support=support,
                s_conflict=conflict,
            )
            r = reliability_score(signals)
            llm = self._verify_with_llm(
                target_type="summary",
                target_id=sid,
                payload={"estimated_support": r, "estimated_conflict": conflict, "statement": stmt},
                image_path=str((graph.get("metadata") or {}).get("image_path", "") or ""),
            )
            status = _status_from_scores(reliability=r, conflict_score=float(llm.get("conflict_score", 0.0) or 0.0))
            reasons = list(llm.get("reasons") or []) + list(grounding_reasons or [])
            if support < 0.5:
                reasons.append("weakly_grounded_in_stage_memory")
            p = review_priority_score(
                reliability=r,
                importance_weight=_importance_for_item(STAGE_G, {"statement": stmt}, main_actor_missing=False),
                propagation_weight=_propagation_for_item(STAGE_G, {"statement": stmt}),
            )
            out.append(
                StageValidationItem(
                    stage=STAGE_G,
                    target_type="summary",
                    target_id=sid,
                    label=stmt,
                    reliability=r,
                    priority=p,
                    status=status,
                    warning=_warning_from_status(status=status, priority=p),
                    reasons=sorted(set([str(x) for x in reasons if str(x).strip()])),
                    suggested_action=str(llm.get("suggested_action", "rewrite") or "rewrite"),
                    llm_check=llm,
                    signals={
                        "s_det": signals.s_det,
                        "s_temp": signals.s_temp,
                        "s_struct": signals.s_struct,
                        "s_cross": signals.s_cross,
                        "s_support": signals.s_support,
                        "s_conflict": signals.s_conflict,
                    },
                    payload={"statement": stmt, "index": idx},
                )
            )
        return out

    def validate(
        self,
        *,
        graph: Dict[str, Any],
        scene_graph_bundle: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        graph = dict(graph or {})
        nodes = [dict(x) for x in list(graph.get("nodes") or []) if isinstance(x, dict)]
        node_map = _edge_nodes_map(nodes)
        metadata = dict(graph.get("metadata") or {})

        frame_w = int(metadata.get("frame_width", 0) or metadata.get("image_width", 0) or 0)
        frame_h = int(metadata.get("frame_height", 0) or metadata.get("image_height", 0) or 0)
        if frame_w <= 0 or frame_h <= 0:
            frame_w, frame_h = 1280, 720

        bundle = dict(scene_graph_bundle or {})
        llm_attr_available = bool(list(bundle.get("llm_person_attributes") or []))
        if not llm_attr_available:
            for node in nodes:
                for attr in list(node.get("attributes") or []):
                    if not isinstance(attr, dict):
                        continue
                    provenance = str(attr.get("provenance", "") or "").strip().lower()
                    value = str(attr.get("value", "") or "").strip()
                    if provenance in {"qwen", "llm"} and value:
                        llm_attr_available = True
                        break
                if llm_attr_available:
                    break
        llm_summary_available = bool(list(bundle.get("llm_batch_summaries") or []))

        sam_checks: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            nid = str(node.get("entity_id", "") or "").strip()
            if not nid:
                continue
            check = verify_sam_mask(node, frame_size=(frame_w, frame_h))
            sam_checks[nid] = check
            node["sam_check"] = check
            node["mask_area_ratio"] = float(check.get("mask_area_ratio", node.get("mask_area_ratio", 0.0)) or 0.0)
            node["bbox_area_ratio"] = float(check.get("bbox_area_ratio", node.get("bbox_area_ratio", 0.0)) or 0.0)
        actor_missing = detect_main_actor_missing(nodes)
        main_actor_missing = bool(actor_missing.get("likely_missing_main_actor", False))

        tracks = self._build_track_items(
            graph=graph,
        )
        attrs = self._build_attribute_items(
            graph=graph,
            node_map=node_map,
            llm_attr_available=bool(llm_attr_available),
        )
        edges = self._build_edge_items(graph=graph, node_map=node_map)
        dynamics = self._build_dynamic_items(
            graph=graph,
        )
        summaries = self._build_summary_items(
            graph=graph,
            llm_summary_available=bool(llm_summary_available),
        )

        stage_map = {
            STAGE_T: tracks,
            STAGE_A: attrs,
            STAGE_E: edges,
            STAGE_S: dynamics,
            STAGE_G: summaries,
        }
        module_scores = {
            "S_T": stage_module_score({str(i): it.reliability for i, it in enumerate(tracks)}, {"0": 1.0}) if tracks else 0.0,
            "S_A": stage_module_score({str(i): it.reliability for i, it in enumerate(attrs)}, {"0": 1.0}) if attrs else 0.0,
            "S_E": stage_module_score({str(i): it.reliability for i, it in enumerate(edges)}, {"0": 1.0}) if edges else 0.0,
            "S_S": stage_module_score({str(i): it.reliability for i, it in enumerate(dynamics)}, {"0": 1.0}) if dynamics else 0.0,
            "S_G": stage_module_score({str(i): it.reliability for i, it in enumerate(summaries)}, {"0": 1.0}) if summaries else 0.0,
        }
        # Target accuracy: aggregate SAM plausibility of detected nodes.
        target_vals = [
            _clamp01(float((row or {}).get("plausibility_score", 0.0) or 0.0))
            for row in sam_checks.values()
            if isinstance(row, dict)
        ]
        target_accuracy = float(sum(target_vals) / len(target_vals)) if target_vals else 0.0

        # State accuracy: aggregate reliability of validated "state" attributes.
        state_attr_items = [
            it for it in attrs
            if str((it.payload or {}).get("slot", "") or "").strip().lower() == "state"
        ]
        state_vals = [_clamp01(float(it.reliability)) for it in state_attr_items]
        state_accuracy = float(sum(state_vals) / len(state_vals)) if state_vals else 0.0

        pvsg_ref = evaluate_graph_against_pvsg(graph=graph)
        ref_available = bool(isinstance(pvsg_ref, dict) and pvsg_ref.get("reference_available", False))
        if ref_available:
            # Keep target accuracy as SAM-output-only quality:
            # do not penalize objects that SAM did not output as nodes/bboxes.
            module_scores["S_E"] = _clamp01(float((pvsg_ref or {}).get("edge_accuracy_gt", module_scores.get("S_E", 0.0)) or module_scores.get("S_E", 0.0)))

        if not tracks:
            # No explicit tracking stage in current pipeline.
            # Use detection/segmentation quality as proxy instead of forcing S_T to 0.
            person_plaus = [
                _clamp01(float((sam_checks.get(str(node.get("entity_id", "") or ""), {}) or {}).get("plausibility_score", 0.0) or 0.0))
                for node in nodes
                if str(node.get("canonical_label", "") or "").strip().lower() == "person"
            ]
            person_mean = float(sum(person_plaus) / len(person_plaus)) if person_plaus else float(target_accuracy)
            proxy = _clamp01(0.7 * float(target_accuracy) + 0.3 * float(person_mean))
            if bool(main_actor_missing):
                proxy = _clamp01(proxy * 0.85)
            module_scores["S_T"] = float(proxy)

        s_struct = stage_module_score(
            {"T": module_scores["S_T"], "A": module_scores["S_A"], "E": module_scores["S_E"], "S": module_scores["S_S"]},
            {"T": 0.30, "A": 0.20, "E": 0.25, "S": 0.25},
        )
        s_summary = module_scores["S_G"]
        # Let MQA reflect current run evidence instead of a fixed constant.
        s_mqa = _clamp01(0.5 * float(target_accuracy) + 0.5 * float(state_accuracy))
        conflict_vals = [
            _clamp01(float((it.llm_check or {}).get("conflict_score", 0.0) or 0.0))
            for values in stage_map.values()
            for it in values
            if isinstance(getattr(it, "llm_check", None), dict) and ("conflict_score" in (it.llm_check or {}))
        ]
        # Verify score is valid only when cycle verification actually ran.
        meta = dict(graph.get("metadata") or {})
        cycle_payload = dict(meta.get("cycle_verification") or graph.get("cycle") or {})
        cycle_runtime = dict(cycle_payload.get("runtime") or {})
        cycle_provider = str(cycle_runtime.get("verifier_provider", "") or "").strip()
        cycle_claims = cycle_payload.get("claims") or {}
        cycle_votes = cycle_payload.get("votes") or []
        cycle_probes = cycle_payload.get("probe_results") or []
        cycle_queue = cycle_payload.get("human_queue") or []
        cycle_verify_executed = bool(
            cycle_provider
            or (isinstance(cycle_claims, dict) and len(cycle_claims) > 0)
            or (isinstance(cycle_claims, list) and len(cycle_claims) > 0)
            or (isinstance(cycle_votes, list) and len(cycle_votes) > 0)
            or (isinstance(cycle_probes, list) and len(cycle_probes) > 0)
            or (isinstance(cycle_queue, list) and len(cycle_queue) > 0)
        )
        # Avoid over-optimistic verify score when cycle was not executed.
        s_verify = _clamp01(1.0 - max(conflict_vals)) if (cycle_verify_executed and conflict_vals) else 0.0
        workflow = workflow_score(
            s_struct=s_struct,
            s_summary=s_summary,
            s_mqa=s_mqa,
            s_verify=s_verify,
        )

        flat_items = [it.to_dict() for values in stage_map.values() for it in values]
        review_queue = sorted(flat_items, key=lambda row: float(row.get("priority", 0.0) or 0.0), reverse=True)
        conflict_items = [
            row for row in flat_items if str(row.get("status", "")).strip().lower() in {STATUS_CONFLICTING, STATUS_UNSUPPORTED}
        ]

        return {
            "stage_items": {k: [it.to_dict() for it in v] for k, v in stage_map.items()},
            "module_scores": {
                **module_scores,
                "target_accuracy": float(target_accuracy),
                "target_accuracy_gt": float((pvsg_ref or {}).get("target_accuracy_gt", 0.0) if ref_available else 0.0),
                "edge_accuracy_gt": float((pvsg_ref or {}).get("edge_accuracy_gt", 0.0) if ref_available else 0.0),
                "pvsg_reference_available": bool(ref_available),
                "state_accuracy": float(state_accuracy),
                "llm_attr_available": bool(llm_attr_available),
                "llm_summary_available": bool(llm_summary_available),
                "S_struct": float(s_struct),
                "S_summary": float(s_summary),
                "S_mqa": float(s_mqa),
                "S_verify": float(s_verify),
                "cycle_verify_executed": bool(cycle_verify_executed),
                "cycle_verifier_provider": str(cycle_provider),
                "verify_not_run": bool(not cycle_verify_executed),
                "S_workflow": float(workflow),
            },
            "sam_verification": {
                "per_node": sam_checks,
                "main_actor_check": actor_missing,
            },
            "pvsg_reference": dict(pvsg_ref or {}),
            "review_queue": review_queue,
            "conflicts": conflict_items,
        }
