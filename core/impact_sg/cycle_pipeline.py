from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Set

from .arbitration import build_human_arbitration_queue
from .belief_update import revise_graph_from_claims
from .captioning import build_caption_prompt, caption_to_claim_feedback
from .claim_graph import (
    attribute_claim_id,
    build_focus_graph,
    build_multi_turn_probes,
    build_single_turn_probes,
    existence_claim_id,
    graph_to_claims,
    label_claim_id,
    relation_claim_id,
)
from .consistency import aggregate_claim_scores
from .correction_memory import normalize_correction_memory
from .cycle_types import Claim
from .eval_cycle import evaluate_cycle_result
from .geometry_review import build_geometry_review_queue
from .visual_verifier.policy import apply_role_policy
from .visual_verifier.schemas import CAPTION_FEEDBACK_SCHEMA, probe_response_schema

REPEAT_SKIP_SCORE_THRESHOLD = 0.55
SENSITIVE_RESULT_KEYS = {
    "raw_response",
    "raw_text",
    "request_prompt",
    "request_schema",
}


def _node_map(graph: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {
        str(node.get("entity_id", "") or ""): node
        for node in graph.get("nodes") or []
        if str(node.get("entity_id", "") or "").strip()
    }


def _edge_map(graph: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {
        str(edge.get("edge_id", "") or ""): edge
        for edge in graph.get("edges") or []
        if str(edge.get("edge_id", "") or "").strip()
    }


def _node_label(node: Dict[str, object]) -> str:
    canonical = str(node.get("canonical_label", "") or "").strip()
    if canonical:
        return canonical
    fallback = str(node.get("label", "") or "").strip()
    return fallback or "object"


def _sanitize_result_payload(payload: Dict[str, object]) -> Dict[str, object]:
    out = dict(_copy_payload(payload or {}))
    for key in SENSITIVE_RESULT_KEYS:
        out.pop(key, None)
    provider = str((dict(payload or {}).get("provider", "") or "")).strip()
    if not provider:
        provider = str((dict((payload or {}).get("raw_response") or {}).get("provider") or "")).strip()
    if provider:
        out["provider"] = provider
    return out


def _sanitize_probe_result_row(row: Dict[str, object]) -> Dict[str, object]:
    out = dict(_copy_payload(row or {}))
    if isinstance(out.get("parsed_response"), dict):
        out["parsed_response"] = _sanitize_result_payload(dict(out.get("parsed_response") or {}))
    if isinstance(out.get("response"), dict):
        out["response"] = _sanitize_result_payload(dict(out.get("response") or {}))
    return out


def _sanitize_caption_payload(payload: Dict[str, object]) -> Dict[str, object]:
    out = _sanitize_result_payload(dict(payload or {}))
    if isinstance(out.get("feedback"), dict):
        out["feedback"] = dict(_copy_payload(out.get("feedback") or {}))
    out["votes"] = [
        dict(_copy_payload(row))
        for row in list(out.get("votes") or [])
        if isinstance(row, dict)
    ]
    return out


def _build_regions(graph: Dict[str, object], evidence_node_ids: List[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    node_by_id = _node_map(graph)
    for node_id in evidence_node_ids or []:
        node = node_by_id.get(str(node_id))
        if node is None:
            continue
        out.append(
            {
                "entity_id": node.get("entity_id"),
                "label": node.get("canonical_label"),
                "bbox": list(node.get("bbox") or [0, 0, 0, 0]),
            }
        )
    return out


def _relation_group_map(ontology) -> Dict[str, str]:
    out: Dict[str, str] = {}
    relation_vocab = getattr(ontology, "relation_vocabulary", {}) or {}
    for group_name, values in dict(relation_vocab or {}).items():
        group_token = str(group_name or "").strip().lower()
        for item in list(values or []):
            token = str(item or "").strip().lower()
            if token and token not in out:
                out[token] = group_token
    return out


def _resolve_attribute_claim_binding(
    graph: Dict[str, object],
    claim_id: str,
) -> Dict[str, str]:
    token = str(claim_id or "").strip()
    if not token.startswith("claim_attr_"):
        return {}
    node_by_id = _node_map(graph)
    for node_id, node in node_by_id.items():
        for att in list(node.get("attributes") or []):
            if not isinstance(att, dict):
                continue
            slot = str(att.get("slot", "") or "").strip()
            if not slot or attribute_claim_id(node_id, slot) != token:
                continue
            return {
                "subject_id": str(node_id or "").strip(),
                "slot": slot,
                "current_value": str(att.get("value", "") or "").strip(),
            }
    return {}


def _prune_prompt_context(payload: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in dict(payload or {}).items():
        if isinstance(value, bool):
            out[str(key)] = bool(value)
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[str(key)] = value
            continue
        token = str(value or "").strip()
        if token:
            out[str(key)] = token
    return out


def _probe_prompt_context(
    graph: Dict[str, object],
    probe: Dict[str, object],
    *,
    ontology=None,
) -> Dict[str, object]:
    claim_id = str((probe or {}).get("target_claim_id", "") or "").strip()
    probe_family = str((probe or {}).get("probe_family", "") or "binary_verification").strip().lower()
    evidence_node_ids = [str(x).strip() for x in list((probe or {}).get("evidence_node_ids") or []) if str(x).strip()]
    node_by_id = _node_map(graph)
    edge_by_id = _edge_map(graph)
    relation_groups = _relation_group_map(ontology)
    out: Dict[str, object] = {
        "claim_id": claim_id,
        "probe_family": probe_family or "binary_verification",
    }

    subject_id = evidence_node_ids[0] if evidence_node_ids else ""
    object_id = evidence_node_ids[1] if len(evidence_node_ids) > 1 else ""
    if claim_id.startswith("claim_exists_"):
        out["claim_type"] = "existence"
        subject_id = claim_id[len("claim_exists_") :].strip() or subject_id
    elif claim_id.startswith("claim_label_"):
        out["claim_type"] = "label"
        subject_id = claim_id[len("claim_label_") :].strip() or subject_id
        node = node_by_id.get(subject_id, {})
        out["current_value"] = _node_label(node)
    elif claim_id.startswith("claim_attr_"):
        out["claim_type"] = "attribute"
        binding = _resolve_attribute_claim_binding(graph, claim_id)
        subject_id = str(binding.get("subject_id", "") or "").strip() or subject_id
        if binding:
            out["slot"] = str(binding.get("slot", "") or "").strip()
            out["current_value"] = str(binding.get("current_value", "") or "").strip()
        else:
            node = node_by_id.get(subject_id, {})
            for att in list(node.get("attributes") or []):
                if not isinstance(att, dict):
                    continue
                slot = str(att.get("slot", "") or "").strip()
                if not slot or attribute_claim_id(subject_id, slot) != claim_id:
                    continue
                out["slot"] = slot
                out["current_value"] = str(att.get("value", "") or "").strip()
                break
    elif claim_id.startswith("claim_rel_"):
        out["claim_type"] = "relation"
        edge_id = claim_id[len("claim_rel_") :].strip()
        edge = edge_by_id.get(edge_id, {})
        subject_id = str(edge.get("src_id", "") or "").strip() or subject_id
        object_id = str(edge.get("dst_id", "") or "").strip() or object_id
        relation = str(edge.get("relation", "") or "").strip()
        relation_group = relation_groups.get(relation.lower(), "")
        out["edge_id"] = edge_id
        out["relation"] = relation
        out["current_value"] = relation
        if relation_group:
            out["relation_group"] = relation_group
        out["is_spatial"] = relation.lower() in {
            "left_of",
            "right_of",
            "above",
            "below",
            "in_front_of",
            "behind",
            "overlapping",
            "intersecting",
        } or relation_group == "spatial"
    else:
        out["claim_type"] = "other"

    if subject_id:
        out["subject_id"] = subject_id
        out["subject_label"] = _node_label(node_by_id.get(subject_id, {}))
    if object_id:
        out["object_id"] = object_id
        out["object_label"] = _node_label(node_by_id.get(object_id, {}))
    if "temporal_anchor_frame_idx" in probe:
        try:
            out["temporal_anchor_frame_idx"] = int(probe.get("temporal_anchor_frame_idx"))
        except Exception:
            pass
    return _prune_prompt_context(out)


def _probe_response_format(
    graph: Dict[str, object],
    probe: Dict[str, object],
    *,
    ontology=None,
) -> Optional[Dict[str, object]]:
    base = dict(probe.get("response_format") or {}) if isinstance(probe.get("response_format"), dict) else {}
    prompt_context = _probe_prompt_context(graph, probe, ontology=ontology)
    if prompt_context:
        base["_prompt_context"] = prompt_context
    return base or None


def _empty_caption_feedback(caption_text: str = "") -> Dict[str, object]:
    report = {
        "structured": False,
        "fallback_used": False,
        "caption_text": str(caption_text or "").strip(),
        "supported_entities": [],
        "unsupported_entities": [],
        "supported_attributes": [],
        "unsupported_attributes": [],
        "supported_relations": [],
        "unsupported_relations": [],
        "hallucinated_mentions": [],
        "vote_count": 0,
        "support_vote_count": 0,
        "conflict_vote_count": 0,
    }
    return {"votes": [], "report": report}


def _copy_payload(value):
    if isinstance(value, dict):
        return {str(key): _copy_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_payload(item) for item in value]
    return value


def _normalized_claim_ids(values: Iterable[object]) -> Set[str]:
    return {
        str(item or "").strip()
        for item in list(values or [])
        if str(item or "").strip()
    }


def _resolved_identity_from_row(row: Dict[str, object]) -> Dict[str, object]:
    claim_id = str(row.get("claim_id") or row.get("target_claim_id") or "").strip()
    probe_id = str(row.get("probe_id") or row.get("qid") or "").strip()
    chain_id = str(row.get("chain_id") or "").strip()
    question = " ".join(str(row.get("question", "") or "").strip().lower().split())
    try:
        turn = int(row.get("turn", 0) or 0)
    except Exception:
        turn = 0
    node_ids = sorted({str(x).strip() for x in list(row.get("evidence_node_ids") or []) if str(x).strip()})
    edge_ids = sorted({str(x).strip() for x in list(row.get("evidence_edge_ids") or []) if str(x).strip()})
    if claim_id.startswith("claim_rel_"):
        eid = claim_id[len("claim_rel_") :].strip()
        if eid and eid not in edge_ids:
            edge_ids.append(eid)
    resolved_key = str(row.get("resolved_key", "") or "").strip()
    if (not resolved_key) and (question or node_ids or edge_ids or (chain_id and int(turn) > 0)):
        parts = [
            claim_id,
            question,
            "|".join(node_ids),
            "|".join(edge_ids),
        ]
        if chain_id and int(turn) > 0:
            parts.append(chain_id)
            parts.append(str(int(turn)))
        resolved_key = "||".join(parts)
    return {
        "claim_id": claim_id,
        "probe_id": probe_id,
        "chain_id": chain_id,
        "turn": int(turn),
        "question": question,
        "evidence_node_ids": node_ids,
        "evidence_edge_ids": edge_ids,
        "resolved_key": resolved_key,
    }


def _question_key_from_row(row: Dict[str, object]) -> str:
    if not isinstance(row, dict):
        return ""
    question = " ".join(str(row.get("question", "") or "").strip().lower().split())
    if not question:
        return ""
    claim_id = str(row.get("target_claim_id", row.get("claim_id", "")) or "").strip().lower()
    node_ids = ",".join(sorted(str(x or "").strip().lower() for x in list(row.get("evidence_node_ids") or []) if str(x or "").strip()))
    edge_ids = ",".join(sorted(str(x or "").strip().lower() for x in list(row.get("evidence_edge_ids") or []) if str(x or "").strip()))
    view_type = str(row.get("view_type", "") or "").strip().lower()
    return f"q::{view_type}::{claim_id}::{node_ids}::{edge_ids}::{question}"


def _normalize_question_key(value: object) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    if token.startswith("q::"):
        return " ".join(token.split())
    if "::" in token:
        _prefix, suffix = token.split("::", 1)
        token = suffix.strip().lower()
    return " ".join(token.split())


def _question_key_from_suppressed_row(row: Dict[str, object]) -> str:
    if not isinstance(row, dict):
        return ""
    explicit = _normalize_question_key(row.get("question_key", ""))
    if explicit.startswith("q::"):
        return explicit
    synthetic = _question_key_from_row(
        {
            "question": row.get("question", ""),
            "view_type": row.get("view_type", ""),
            "target_claim_id": row.get("target_claim_id", row.get("claim_id", "")),
            "claim_id": row.get("claim_id", ""),
            "evidence_node_ids": list(row.get("evidence_node_ids") or []),
            "evidence_edge_ids": list(row.get("evidence_edge_ids") or []),
        }
    )
    if synthetic:
        return synthetic
    return explicit


def _probe_result_score(row: Dict[str, object]) -> float:
    if not isinstance(row, dict):
        return -1.0
    parsed = dict(row.get("parsed_response") or {}) if isinstance(row.get("parsed_response"), dict) else {}
    legacy = dict(row.get("response") or {}) if isinstance(row.get("response"), dict) else {}
    for source in (row, parsed, legacy):
        raw = source.get("score", None)
        if raw is None:
            continue
        try:
            return float(raw)
        except Exception:
            continue
    return -1.0


def _probe_result_is_skipworthy(row: Dict[str, object]) -> bool:
    if not isinstance(row, dict):
        return False
    parsed = dict(row.get("parsed_response") or {}) if isinstance(row.get("parsed_response"), dict) else {}
    legacy = dict(row.get("response") or {}) if isinstance(row.get("response"), dict) else {}
    payloads = (row, parsed, legacy)
    for source in payloads:
        if bool(source.get("resolved", False)) or bool(source.get("manually_resolved", False)):
            return True
    if any(bool(source.get("stale", False)) for source in payloads):
        return False
    if any(bool(source.get("is_truncated", False)) for source in payloads):
        return False
    if any(source.get("is_valid", None) is False for source in payloads):
        return False
    for source in payloads:
        if "schema_valid" in source and bool(source.get("schema_valid")) is False:
            return False
    return _probe_result_score(row) > float(REPEAT_SKIP_SCORE_THRESHOLD)


def _resolved_scope(
    *,
    graph: Dict[str, object],
    base_result: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    meta = dict((graph or {}).get("metadata") or {})
    graph_cycle = dict(meta.get("cycle_verification") or (graph or {}).get("cycle") or {})
    records: List[Dict[str, object]] = []
    records.extend([dict(row) for row in list(graph_cycle.get("resolved_claims") or []) if isinstance(row, dict)])
    records.extend([dict(row) for row in list((base_result or {}).get("resolved_claims") or []) if isinstance(row, dict)])
    historical_probe_rows: List[Dict[str, object]] = []
    historical_probe_rows.extend([dict(row) for row in list(graph_cycle.get("probe_results") or []) if isinstance(row, dict)])
    historical_probe_rows.extend([dict(row) for row in list((base_result or {}).get("probe_results") or []) if isinstance(row, dict)])
    out_records: List[Dict[str, object]] = []
    suppressed_question_records: List[Dict[str, object]] = []
    suppressed_seen: Set[str] = set()
    claim_ids: Set[str] = set()
    chain_turns: Set[tuple[str, int]] = set()
    probe_ids: Set[str] = set()
    resolved_keys: Set[str] = set()
    suppressed_question_keys: Set[str] = set()
    seen = set()
    for row in records:
        ident = _resolved_identity_from_row(row)
        key = (
            ident["claim_id"],
            ident["probe_id"],
            ident["chain_id"],
            int(ident["turn"]),
            str(ident.get("resolved_key", "") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        has_probe_level_identity = bool(ident["probe_id"]) or bool(ident["chain_id"] and int(ident["turn"]) > 0)
        # Keep claim-level scope only for legacy records that do not carry
        # probe-level identity; otherwise resolution is QA-item specific.
        if ident["claim_id"] and not has_probe_level_identity:
            claim_ids.add(ident["claim_id"])
        if ident["probe_id"]:
            probe_ids.add(ident["probe_id"])
        if ident["chain_id"] and int(ident["turn"]) > 0:
            chain_turns.add((ident["chain_id"], int(ident["turn"])))
        if str(ident.get("resolved_key", "") or "").strip():
            resolved_keys.add(str(ident.get("resolved_key", "") or "").strip())
        merged = dict(row)
        merged.update(ident)
        out_records.append(merged)
    for row in [dict(x) for x in list(graph_cycle.get("suppressed_questions") or []) if isinstance(x, dict)]:
        qkey = _question_key_from_suppressed_row(row)
        if qkey:
            suppressed_question_keys.add(qkey)
            if qkey not in suppressed_seen:
                suppressed_seen.add(qkey)
                merged = dict(row)
                merged["question_key"] = qkey
                suppressed_question_records.append(merged)
    for row in [dict(x) for x in list((base_result or {}).get("suppressed_questions") or []) if isinstance(x, dict)]:
        qkey = _question_key_from_suppressed_row(row)
        if qkey:
            suppressed_question_keys.add(qkey)
            if qkey not in suppressed_seen:
                suppressed_seen.add(qkey)
                merged = dict(row)
                merged["question_key"] = qkey
                suppressed_question_records.append(merged)
    for row in historical_probe_rows:
        if not _probe_result_is_skipworthy(row):
            continue
        ident = _resolved_identity_from_row(row)
        resolved_key = str(ident.get("resolved_key", "") or "").strip()
        if resolved_key:
            resolved_keys.add(resolved_key)
        qkey = _question_key_from_row(row)
        if qkey:
            suppressed_question_keys.add(qkey)
    return {
        "records": out_records,
        "suppressed_questions": suppressed_question_records,
        "claim_ids": claim_ids,
        "chain_turns": chain_turns,
        "probe_ids": probe_ids,
        "resolved_keys": resolved_keys,
        "suppressed_question_keys": suppressed_question_keys,
    }


def _probe_is_resolved(row: Dict[str, object], scope: Dict[str, object]) -> bool:
    ident = _resolved_identity_from_row(row)
    question_key = _question_key_from_row(row)
    if question_key and question_key in set(scope.get("suppressed_question_keys") or set()):
        return True
    resolved_key = str(ident.get("resolved_key", "") or "").strip()
    if resolved_key and resolved_key in set(scope.get("resolved_keys") or set()):
        return True
    probe_id = ident["probe_id"]
    if probe_id and probe_id in set(scope.get("probe_ids") or set()):
        return True
    chain_id = ident["chain_id"]
    turn = int(ident["turn"])
    if chain_id and turn > 0 and (chain_id, turn) in set(scope.get("chain_turns") or set()):
        return True
    question = str(ident.get("question", "") or "").strip()
    node_ids = list(ident.get("evidence_node_ids") or [])
    edge_ids = list(ident.get("evidence_edge_ids") or [])
    if question:
        for rec in list(scope.get("records") or []):
            other = _resolved_identity_from_row(dict(rec or {}))
            other_question = str(other.get("question", "") or "").strip()
            if not other_question or other_question != question:
                continue
            if list(other.get("evidence_node_ids") or []) != node_ids:
                continue
            if list(other.get("evidence_edge_ids") or []) != edge_ids:
                continue
            other_claim_id = str(other.get("claim_id", "") or "").strip()
            claim_id = str(ident.get("claim_id", "") or "").strip()
            if claim_id and other_claim_id and claim_id == other_claim_id:
                return True
    # Legacy fallback for old claim-only resolved records.
    claim_id = ident["claim_id"]
    if claim_id and claim_id in set(scope.get("claim_ids") or set()):
        return True
    return False


def _filter_out_resolved_probes(
    probes: Iterable[Dict[str, object]],
    scope: Dict[str, object],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for probe in list(probes or []):
        row = dict(probe or {})
        if _probe_is_resolved(row, scope):
            continue
        out.append(row)
    return out


def _claim_payload_to_dict(payload: object) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    if isinstance(payload, dict):
        for key, value in dict(payload or {}).items():
            claim_id = str((value or {}).get("claim_id", "") or key or "").strip() if isinstance(value, dict) else str(key or "").strip()
            if claim_id and isinstance(value, dict):
                out[claim_id] = dict(value)
        return out
    if isinstance(payload, list):
        for row in list(payload or []):
            if not isinstance(row, dict):
                continue
            claim_id = str(row.get("claim_id", "") or "").strip()
            if claim_id:
                out[claim_id] = dict(row)
    return out


def _claim_exists_in_graph(graph: Dict[str, object], claim_id: str) -> bool:
    token = str(claim_id or "").strip()
    if not token:
        return False
    node_by_id = _node_map(graph)
    edge_by_id = _edge_map(graph)
    if token.startswith("claim_exists_") or token.startswith("claim_label_"):
        node_id = token.split("_", 2)[-1].strip()
        return bool(node_id and node_id in node_by_id)
    if token.startswith("claim_attr_"):
        for node_id, node in node_by_id.items():
            for att in list(node.get("attributes") or []):
                if not isinstance(att, dict):
                    continue
                slot = str(att.get("slot", "") or "").strip()
                if slot and attribute_claim_id(node_id, slot) == token:
                    return True
        return False
    if token.startswith("claim_rel_"):
        edge_id = token[len("claim_rel_") :].strip()
        return bool(edge_id and edge_id in edge_by_id)
    return False


def _claim_from_graph(graph: Dict[str, object], claim_id: str) -> Optional[Claim]:
    token = str(claim_id or "").strip()
    if not token:
        return None
    snapshot_id = str((graph.get("metadata") or {}).get("graph_snapshot_id") or graph.get("image_id") or "")
    node_by_id = _node_map(graph)
    edge_by_id = _edge_map(graph)
    if token.startswith("claim_exists_"):
        node_id = token[len("claim_exists_") :].strip()
        node = node_by_id.get(node_id)
        if not isinstance(node, dict):
            return None
        prior = float(node.get("verify_confidence", node.get("confidence", node.get("score", 0.5))) or 0.5)
        return Claim(
            claim_id=token,
            claim_type="existence",
            subject_id=node_id,
            predicate="exists",
            value="true",
            source_graph_snapshot_id=snapshot_id,
            evidence_node_ids=[node_id],
            prior_score=prior,
            provenance=[{"source": "scene_graph", "field": "node"}],
        )
    if token.startswith("claim_label_"):
        node_id = token[len("claim_label_") :].strip()
        node = node_by_id.get(node_id)
        if not isinstance(node, dict):
            return None
        prior = float(node.get("verify_confidence", node.get("confidence", node.get("score", 0.5))) or 0.5)
        return Claim(
            claim_id=token,
            claim_type="label",
            subject_id=node_id,
            predicate="label",
            value=_node_label(node),
            source_graph_snapshot_id=snapshot_id,
            evidence_node_ids=[node_id],
            prior_score=prior,
            provenance=[{"source": "scene_graph", "field": "canonical_label"}],
        )
    if token.startswith("claim_attr_"):
        for node_id, node in node_by_id.items():
            for att in list(node.get("attributes") or []):
                if not isinstance(att, dict):
                    continue
                slot = str(att.get("slot", "") or "").strip()
                value = str(att.get("value", "") or "").strip()
                if not slot or not value or attribute_claim_id(node_id, slot) != token:
                    continue
                prior = float(att.get("verify_confidence", att.get("confidence", node.get("score", 0.5))) or node.get("score", 0.5) or 0.5)
                return Claim(
                    claim_id=token,
                    claim_type="attribute",
                    subject_id=node_id,
                    predicate=slot,
                    value=value,
                    source_graph_snapshot_id=snapshot_id,
                    evidence_node_ids=[node_id],
                    prior_score=prior,
                    provenance=[{"source": "scene_graph", "field": "attribute"}],
                )
        return None
    if token.startswith("claim_rel_"):
        edge_id = token[len("claim_rel_") :].strip()
        edge = edge_by_id.get(edge_id)
        if not isinstance(edge, dict):
            return None
        src_id = str(edge.get("src_id", "") or "").strip()
        dst_id = str(edge.get("dst_id", "") or "").strip()
        relation = str(edge.get("relation", "") or "").strip()
        if not src_id or not dst_id or not relation:
            return None
        prior = float(edge.get("verify_confidence", edge.get("confidence", edge.get("score", 0.5))) or 0.5)
        return Claim(
            claim_id=token,
            claim_type="relation",
            subject_id=src_id,
            predicate=relation,
            object_id=dst_id,
            source_graph_snapshot_id=snapshot_id,
            evidence_node_ids=[src_id, dst_id],
            evidence_edge_ids=[edge_id],
            prior_score=prior,
            provenance=[{"source": "scene_graph", "field": "edge"}],
        )
    return None


def _target_scope_from_claim_ids(
    graph: Dict[str, object],
    claim_ids: Iterable[object],
) -> Dict[str, Set[str]]:
    claims = _normalized_claim_ids(claim_ids)
    node_ids: Set[str] = set()
    edge_ids: Set[str] = set()
    node_by_id = _node_map(graph)
    edge_by_id = _edge_map(graph)
    for claim_id in claims:
        if claim_id.startswith("claim_exists_"):
            node_id = claim_id[len("claim_exists_") :].strip()
            if node_id:
                node_ids.add(node_id)
        elif claim_id.startswith("claim_label_"):
            node_id = claim_id[len("claim_label_") :].strip()
            if node_id:
                node_ids.add(node_id)
        elif claim_id.startswith("claim_attr_"):
            for node_id, node in node_by_id.items():
                for att in list(node.get("attributes") or []):
                    if not isinstance(att, dict):
                        continue
                    slot = str(att.get("slot", "") or "").strip()
                    if slot and attribute_claim_id(node_id, slot) == claim_id:
                        node_ids.add(node_id)
                        break
        elif claim_id.startswith("claim_rel_"):
            edge_id = claim_id[len("claim_rel_") :].strip()
            edge = edge_by_id.get(edge_id)
            if edge_id:
                edge_ids.add(edge_id)
            if isinstance(edge, dict):
                src_id = str(edge.get("src_id", "") or "").strip()
                dst_id = str(edge.get("dst_id", "") or "").strip()
                if src_id:
                    node_ids.add(src_id)
                if dst_id:
                    node_ids.add(dst_id)
    return {"claim_ids": claims, "node_ids": node_ids, "edge_ids": edge_ids}


def _row_matches_target_scope(
    row: Dict[str, object],
    *,
    claim_ids: Set[str],
    node_ids: Set[str],
    edge_ids: Set[str],
) -> bool:
    if not isinstance(row, dict):
        return False
    direct_claim_ids = {
        str(row.get("claim_id", "") or "").strip(),
        str(row.get("target_claim_id", "") or "").strip(),
        str(row.get("source_relation_claim_id", "") or "").strip(),
    }
    if bool(direct_claim_ids.intersection(claim_ids)):
        return True
    claim_row = dict(row.get("claim_row") or {})
    if str(claim_row.get("claim_id", "") or "").strip() in claim_ids:
        return True
    for prov in list(claim_row.get("provenance") or []):
        if not isinstance(prov, dict):
            continue
        if str(prov.get("relation_claim_id", "") or "").strip() in claim_ids:
            return True
    row_nodes = {
        str(item).strip()
        for item in list(row.get("evidence_node_ids") or []) + list(claim_row.get("evidence_node_ids") or [])
        if str(item).strip()
    }
    for key in ("subject_id", "object_id", "target_node_id"):
        token = str(row.get(key, "") or claim_row.get(key, "") or "").strip()
        if token:
            row_nodes.add(token)
    if row_nodes and bool(row_nodes.intersection(node_ids)):
        return True
    row_edges = {
        str(item).strip()
        for item in list(row.get("evidence_edge_ids") or []) + list(claim_row.get("evidence_edge_ids") or [])
        if str(item).strip()
    }
    for key in ("source_relation_edge_id",):
        token = str(row.get(key, "") or "").strip()
        if token:
            row_edges.add(token)
    claim_token = str(row.get("claim_id", "") or row.get("target_claim_id", "") or "").strip()
    if claim_token.startswith("claim_rel_"):
        row_edges.add(claim_token[len("claim_rel_") :].strip())
    if row_edges and bool(row_edges.intersection(edge_ids)):
        return True
    return False


def _filter_rows_outside_scope(
    rows: Iterable[Dict[str, object]],
    *,
    claim_ids: Set[str],
    node_ids: Set[str],
    edge_ids: Set[str],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in list(rows or []):
        item = dict(row or {})
        if _row_matches_target_scope(item, claim_ids=claim_ids, node_ids=node_ids, edge_ids=edge_ids):
            continue
        out.append(item)
    return out


def _dedupe_payload_rows(rows: Iterable[object]) -> List[object]:
    out: List[object] = []
    seen: Set[str] = set()
    for row in list(rows or []):
        item = _copy_payload(row)
        key = repr(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _claim_id_for_caption_entity(entity_id: str) -> Set[str]:
    token = str(entity_id or "").strip()
    if not token:
        return set()
    return {existence_claim_id(token), label_claim_id(token)}


def _filter_caption_report_for_scope(
    report: Dict[str, object],
    *,
    claim_ids: Set[str],
) -> Dict[str, object]:
    out = dict(report or {})
    supported_entities = [
        str(item).strip()
        for item in list(report.get("supported_entities") or [])
        if _claim_id_for_caption_entity(str(item or "").strip()).intersection(claim_ids)
    ]
    unsupported_entities = [
        str(item).strip()
        for item in list(report.get("unsupported_entities") or [])
        if _claim_id_for_caption_entity(str(item or "").strip()).intersection(claim_ids)
    ]
    supported_attributes = [
        dict(row)
        for row in list(report.get("supported_attributes") or [])
        if isinstance(row, dict)
        and attribute_claim_id(str(row.get("entity_id", "") or ""), str(row.get("slot", "") or "")) in claim_ids
    ]
    unsupported_attributes = [
        dict(row)
        for row in list(report.get("unsupported_attributes") or [])
        if isinstance(row, dict)
        and attribute_claim_id(str(row.get("entity_id", "") or ""), str(row.get("slot", "") or "")) in claim_ids
    ]
    supported_relations = [
        str(item).strip()
        for item in list(report.get("supported_relations") or [])
        if relation_claim_id(str(item or "").strip()) in claim_ids
    ]
    unsupported_relations = [
        str(item).strip()
        for item in list(report.get("unsupported_relations") or [])
        if relation_claim_id(str(item or "").strip()) in claim_ids
    ]
    out["supported_entities"] = _dedupe_payload_rows(supported_entities)
    out["unsupported_entities"] = _dedupe_payload_rows(unsupported_entities)
    out["supported_attributes"] = _dedupe_payload_rows(supported_attributes)
    out["unsupported_attributes"] = _dedupe_payload_rows(unsupported_attributes)
    out["supported_relations"] = _dedupe_payload_rows(supported_relations)
    out["unsupported_relations"] = _dedupe_payload_rows(unsupported_relations)
    return out


def _merge_string_claim_lists(
    base_values: Iterable[object],
    new_values: Iterable[object],
    *,
    remove_claim_ids: Set[str],
) -> List[str]:
    out = [
        str(item).strip()
        for item in list(base_values or [])
        if str(item).strip() and str(item).strip() not in remove_claim_ids
    ]
    for item in list(new_values or []):
        token = str(item).strip()
        if token and token not in out:
            out.append(token)
    return out


def _merge_claim_keyed_rows(
    base_rows: Iterable[Dict[str, object]],
    new_rows: Iterable[Dict[str, object]],
    *,
    remove_claim_ids: Set[str],
    key_name: str = "claim_id",
) -> List[Dict[str, object]]:
    out = [
        dict(row)
        for row in list(base_rows or [])
        if str((row or {}).get(key_name, "") or "").strip() not in remove_claim_ids
    ]
    out.extend([dict(row) for row in list(new_rows or []) if isinstance(row, dict)])
    return out


def _merge_cycle_update(
    base_update: Dict[str, object],
    new_update: Dict[str, object],
    *,
    remove_claim_ids: Set[str],
    remove_node_ids: Set[str],
) -> Dict[str, object]:
    merged = dict(_copy_payload(base_update or {}))
    for key in (
        "accepted_claim_ids",
        "accepted_confirm_claim_ids",
        "accepted_correct_claim_ids",
        "flagged_claim_ids",
        "auto_removed_claim_ids",
    ):
        merged[key] = _merge_string_claim_lists(
            merged.get(key) or [],
            list(new_update.get(key) or []),
            remove_claim_ids=remove_claim_ids,
        )
    for key in ("memory_adjustments", "correction_applied"):
        merged[key] = _merge_claim_keyed_rows(
            merged.get(key) or [],
            list(new_update.get(key) or []),
            remove_claim_ids=remove_claim_ids,
        )
    auto_removed_nodes = [
        str(item).strip()
        for item in list(merged.get("auto_removed_node_ids") or [])
        if str(item).strip() and str(item).strip() not in remove_node_ids
    ]
    for item in list(new_update.get("auto_removed_node_ids") or []):
        token = str(item).strip()
        if token and token not in auto_removed_nodes:
            auto_removed_nodes.append(token)
    merged["auto_removed_node_ids"] = auto_removed_nodes
    for key, value in dict(new_update or {}).items():
        if key in {
            "accepted_claim_ids",
            "accepted_confirm_claim_ids",
            "accepted_correct_claim_ids",
            "flagged_claim_ids",
            "memory_adjustments",
            "correction_applied",
            "auto_removed_node_ids",
            "auto_removed_claim_ids",
        }:
            continue
        merged[key] = _copy_payload(value)
    return merged


def _merge_caption_payload(
    base_caption: Dict[str, object],
    new_caption: Dict[str, object],
    *,
    target_claim_ids: Set[str],
    merged_caption_votes: List[Dict[str, object]],
) -> Dict[str, object]:
    base_payload = dict(_copy_payload(base_caption or {}))
    new_payload = dict(_copy_payload(new_caption or {}))
    base_report = dict(base_payload.get("feedback") or {})
    new_report = dict(new_payload.get("feedback") or {})
    keep_report = _filter_caption_report_for_scope(base_report, claim_ids=set())
    remove_old_report = _filter_caption_report_for_scope(base_report, claim_ids=target_claim_ids)
    for key in (
        "supported_entities",
        "unsupported_entities",
        "supported_attributes",
        "unsupported_attributes",
        "supported_relations",
        "unsupported_relations",
    ):
        preserved = [
            item
            for item in list(base_report.get(key) or [])
            if item not in list(remove_old_report.get(key) or [])
        ]
        updated = preserved + list(_filter_caption_report_for_scope(new_report, claim_ids=target_claim_ids).get(key) or [])
        keep_report[key] = _dedupe_payload_rows(updated)
    keep_report["structured"] = bool(new_report.get("structured", base_report.get("structured", False)))
    keep_report["fallback_used"] = bool(new_report.get("fallback_used", base_report.get("fallback_used", False)))
    keep_report["caption_text"] = str(
        new_report.get("caption_text", "")
        or new_payload.get("caption_text", "")
        or new_payload.get("caption", "")
        or base_report.get("caption_text", "")
        or base_payload.get("caption_text", "")
        or base_payload.get("caption", "")
        or ""
    ).strip()
    keep_report["hallucinated_mentions"] = _dedupe_payload_rows(
        list(new_report.get("hallucinated_mentions") or [])
        or list(base_report.get("hallucinated_mentions") or [])
    )
    keep_report["vote_count"] = len(list(merged_caption_votes or []))
    keep_report["support_vote_count"] = len(
        [
            row
            for row in list(merged_caption_votes or [])
            if str((row or {}).get("vote", "") or "").strip().lower() == "support"
        ]
    )
    keep_report["conflict_vote_count"] = len(
        [
            row
            for row in list(merged_caption_votes or [])
            if str((row or {}).get("vote", "") or "").strip().lower() == "conflict"
        ]
    )
    merged = dict(base_payload)
    merged.update(new_payload)
    merged["feedback"] = keep_report
    merged["votes"] = [dict(row) for row in list(merged_caption_votes or []) if isinstance(row, dict)]
    merged["caption_text"] = str(
        new_payload.get("caption_text", "")
        or new_payload.get("caption", "")
        or keep_report.get("caption_text", "")
        or base_payload.get("caption_text", "")
        or base_payload.get("caption", "")
        or ""
    ).strip()
    merged["caption"] = str(
        new_payload.get("caption", "")
        or merged.get("caption_text", "")
        or base_payload.get("caption", "")
        or ""
    ).strip()
    return merged


def _merge_policy_report(
    base_policy: Dict[str, object],
    new_policy: Dict[str, object],
    *,
    merged_votes: List[Dict[str, object]],
) -> Dict[str, object]:
    merged = dict(_copy_payload(base_policy or {}))
    incoming = dict(_copy_payload(new_policy or {}))
    for key, value in incoming.items():
        if key == "policy" and not isinstance(value, dict):
            continue
        merged[key] = _copy_payload(value)
    if not isinstance(merged.get("policy"), dict):
        merged["policy"] = dict(incoming.get("policy") or base_policy.get("policy") or {})
    merged["vote_count"] = len(list(merged_votes or []))
    weighted_vote_count = 0
    for row in list(merged_votes or []):
        if not isinstance(row, dict):
            continue
        try:
            weight = float(row.get("weight", 1.0) or 0.0)
        except Exception:
            weight = 0.0
        if weight > 0.0:
            weighted_vote_count += 1
    merged["weighted_vote_count"] = weighted_vote_count
    return merged


def _normalize_selection(value: object, options: List[str]) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    lowered = token.lower()
    if lowered in {"uncertain", "unknown", "not_sure", "not sure"}:
        return "uncertain"
    option_map = {str(item).strip().lower(): str(item).strip() for item in options if str(item).strip()}
    if lowered in option_map:
        return option_map[lowered]
    normalized = lowered.strip("'\"")
    if normalized in option_map:
        return option_map[normalized]
    for key, canonical in option_map.items():
        if lowered == key or f"\"{key}\"" in lowered or f"'{key}'" in lowered:
            return canonical
    return ""


def _normalize_answer_token(value: object) -> str:
    token = str(value or "").strip().lower()
    if token in {"yes", "y", "true", "1", "supported"}:
        return "yes"
    if token in {"no", "n", "false", "0", "conflict"}:
        return "no"
    if token in {"uncertain", "unknown", "not_sure", "not sure", "unsure", ""}:
        return "uncertain"
    return "uncertain"


def _extract_answer_fields(resp: Dict[str, object], *, options: Optional[List[str]] = None) -> Dict[str, object]:
    if not isinstance(resp, dict):
        return {
            "answer": "uncertain",
            "score": None,
            "reason": "invalid_response",
            "raw_text": str(resp or ""),
            "selection": "uncertain",
            "schema_valid": False,
            "is_valid": False,
            "is_truncated": False,
        }
    schema_valid = bool(resp.get("schema_valid", True))
    is_truncated = bool(resp.get("is_truncated", False))
    is_valid_flag = resp.get("is_valid", True)
    explicitly_invalid = bool(is_valid_flag is False)
    schema_invalid_and_not_salvaged = (not schema_valid) and (is_valid_flag is not True)
    if schema_invalid_and_not_salvaged or is_truncated or explicitly_invalid:
        return {
            "answer": "uncertain",
            "score": None,
            "reason": "invalid_response",
            "raw_text": str(resp.get("raw_text", "") or ""),
            "selection": "uncertain",
            "schema_valid": False,
            "is_valid": False,
            "is_truncated": bool(is_truncated),
        }
    answer = str(resp.get("answer", "") or "").strip().lower()
    selection = ""
    if options:
        for key in ("selection", "choice", "selected_option", "value"):
            selection = _normalize_selection(resp.get(key), options)
            if selection:
                break
    score_raw = resp.get("score", 0.0)
    try:
        score = float(score_raw)
    except Exception:
        score = 0.0
    raw_text = str(resp.get("raw_text", "") or "")
    if answer not in {"yes", "no", "uncertain"}:
        answer = "uncertain"
    return {
        "answer": answer,
        "score": min(1.0, max(0.0, score)),
        "reason": str(resp.get("reason", "") or "").strip(),
        "raw_text": raw_text,
        "selection": selection,
        "schema_valid": True,
        "is_valid": bool(resp.get("is_valid", True)),
        "is_truncated": bool(resp.get("is_truncated", False)),
    }


def _extract_frame_idx(graph: Dict[str, object]) -> Optional[int]:
    meta = dict(graph.get("metadata") or {})
    for container in (dict(graph or {}), meta):
        for key in ("frame_idx", "frame_index", "graph_frame_idx", "current_frame_idx"):
            raw_value = container.get(key)
            try:
                return int(raw_value)
            except Exception:
                continue
    return None


def _limit_probes(
    probes: List[Dict[str, object]],
    *,
    max_items: int,
) -> List[Dict[str, object]]:
    rows = [dict(p) for p in list(probes or []) if isinstance(p, dict)]
    if max_items <= 0:
        return rows
    if len(rows) <= max_items:
        return rows
    # Prefer constrained correction probes first, then deterministic order by probe_id.
    def _rank(row: Dict[str, object]) -> tuple[int, str]:
        family = str(row.get("probe_family", "") or "").strip().lower()
        rank = 0 if family == "constrained_correction" else 1
        return (rank, str(row.get("probe_id", "") or ""))
    rows.sort(key=_rank)
    return rows[:max_items]


def _single_probe_claim_type(probe: Dict[str, object]) -> str:
    claim_id = str((probe or {}).get("target_claim_id", "") or "").strip()
    if claim_id.startswith("claim_rel_"):
        return "edge"
    if claim_id.startswith("claim_exists_") or claim_id.startswith("claim_label_") or claim_id.startswith("claim_attr_"):
        return "node"
    return "other"


def _select_single_turn_probes_balanced(
    probes: List[Dict[str, object]],
    *,
    max_items: int,
) -> List[Dict[str, object]]:
    rows = [dict(p) for p in list(probes or []) if isinstance(p, dict)]
    if max_items <= 0 or not rows:
        return []
    if len(rows) <= max_items:
        return rows

    # Deterministic order so repeated runs are stable.
    rows.sort(key=lambda row: str(row.get("probe_id", "") or ""))
    nodes = [row for row in rows if _single_probe_claim_type(row) == "node"]
    edges = [row for row in rows if _single_probe_claim_type(row) == "edge"]
    others = [row for row in rows if _single_probe_claim_type(row) not in {"node", "edge"}]

    picked: List[Dict[str, object]] = []
    if nodes:
        picked.append(nodes.pop(0))
    if edges and len(picked) < max_items:
        picked.append(edges.pop(0))
    # Prefer one more node, then edge, then fill.
    if nodes and len(picked) < max_items:
        picked.append(nodes.pop(0))
    elif edges and len(picked) < max_items:
        picked.append(edges.pop(0))

    for pool in (nodes, edges, others):
        for row in pool:
            if len(picked) >= max_items:
                break
            picked.append(row)
        if len(picked) >= max_items:
            break
    return picked[:max_items]


def _effective_correction_memory(
    cfg: Dict[str, object],
    correction_memory: Optional[Dict[str, object]],
) -> Dict[str, object]:
    payload = normalize_correction_memory(correction_memory)
    memory_cfg = dict(cfg.get("memory") or {})
    if not bool(memory_cfg.get("enable_label_confusion_memory", True)):
        payload["label_confusions"] = {}
    if not bool(memory_cfg.get("enable_relation_confusion_memory", True)):
        payload["relation_confusions"] = {}
    if not bool(memory_cfg.get("enable_prompt_alias_memory", True)):
        payload["prompt_aliases"] = {}
    return payload


def _memory_summary(memory: Dict[str, object]) -> Dict[str, int]:
    return {
        "label_confusions": sum(
            int(v)
            for bucket in dict(memory.get("label_confusions") or {}).values()
            for v in dict(bucket or {}).values()
        ),
        "relation_confusions": sum(
            int(v)
            for bucket in dict(memory.get("relation_confusions") or {}).values()
            for v in dict(bucket or {}).values()
        ),
        "prompt_aliases": sum(len(list(bucket or [])) for bucket in dict(memory.get("prompt_aliases") or {}).values()),
        "verified_locks": len(dict(memory.get("verified_locks") or {})),
    }


def _cycle_summary(
    *,
    result: Dict[str, object],
    graph_after: Dict[str, object],
    rounds: List[Dict[str, object]],
) -> Dict[str, object]:
    runtime = dict(result.get("runtime") or {})
    cycle_update = dict(((graph_after.get("metadata") or {}).get("cycle_update")) or {})
    human_queue = list(result.get("human_queue") or [])
    probe_results = list(result.get("probe_results") or [])
    focus = dict(result.get("focus") or {})
    caption_feedback = dict(((result.get("caption") or {}).get("feedback")) or {})
    return {
        "rounds_run": len(rounds),
        "probe_count": len(probe_results),
        "single_turn_count": len([x for x in probe_results if str(x.get("view_type", "") or "") == "single_turn_vqa"]),
        "multi_turn_count": len([x for x in probe_results if str(x.get("view_type", "") or "") == "multi_turn_vqa"]),
        "caption_vote_count": int(caption_feedback.get("vote_count", 0) or 0),
        "caption_support_vote_count": int(caption_feedback.get("support_vote_count", 0) or 0),
        "caption_conflict_vote_count": int(caption_feedback.get("conflict_vote_count", 0) or 0),
        "caption_structured_feedback": bool(caption_feedback.get("structured")),
        "vote_count": len(list(result.get("votes") or [])),
        "queue_count": len(human_queue),
        "geometry_queue_count": len(
            [
                row
                for row in human_queue
                if str(row.get("claim_type", "") or "").strip().lower() == "bbox"
            ]
        ),
        "accepted_claim_count": len(list(cycle_update.get("accepted_claim_ids") or [])),
        "accepted_confirm_count": len(list(cycle_update.get("accepted_confirm_claim_ids") or [])),
        "accepted_correct_count": len(list(cycle_update.get("accepted_correct_claim_ids") or [])),
        "flagged_claim_count": len(list(cycle_update.get("flagged_claim_ids") or [])),
        "memory_adjusted_count": len(list(cycle_update.get("memory_adjustments") or [])),
        "top_review_questions": [
            str(row.get("question", "") or "").strip()
            for row in human_queue[:3]
            if str(row.get("question", "") or "").strip()
        ],
        "focus_applied": bool(focus.get("applied")),
        "focus_subject_label": str(focus.get("subject_label", "") or ""),
        "focus_kept_node_count": int(focus.get("kept_node_count", 0) or 0),
        "verifier_provider": str(runtime.get("verifier_provider", "") or ""),
        "verifier_model_id": str(runtime.get("verifier_model_id", "") or ""),
    }


def _build_agent_summary(
    *,
    cycle_cfg: Dict[str, object],
    probe_results: List[Dict[str, object]],
    caption_payload: Dict[str, object],
    human_queue: List[Dict[str, object]],
    graph_after: Dict[str, object],
) -> Dict[str, object]:
    caption_feedback = dict((caption_payload or {}).get("feedback") or {})
    return {
        "scene_graph_backbone": {
            "enabled": True,
            "role": "latent_structured_state",
            "node_count": len(list(graph_after.get("nodes") or [])),
            "edge_count": len(list(graph_after.get("edges") or [])),
        },
        "single_turn_vqa": {
            "enabled": bool(cycle_cfg.get("enable_single_turn_probes", True)),
            "role": "local_binary_and_constrained_verifier",
            "probe_count": len(
                [row for row in probe_results if str(row.get("view_type", "") or "") == "single_turn_vqa"]
            ),
        },
        "multi_turn_vqa": {
            "enabled": bool(cycle_cfg.get("enable_multi_turn_probes", True)),
            "role": "temporal_compositional_chain_verifier",
            "probe_count": len(
                [row for row in probe_results if str(row.get("view_type", "") or "") == "multi_turn_vqa"]
            ),
            "temporal_probe_count": len(
                [
                    row
                    for row in probe_results
                    if (
                        str(row.get("view_type", "") or "") == "multi_turn_vqa"
                        and str(row.get("probe_family", "") or "") == "temporal_consistency"
                    )
                ]
            ),
        },
        "captioning": {
            "enabled": bool(cycle_cfg.get("enable_caption_probe", True)),
            "role": "holistic_structured_verifier",
            "structured_feedback": bool(caption_feedback.get("structured")),
            "vote_count": int(caption_feedback.get("vote_count", 0) or 0),
            "hallucination_count": len(list(caption_feedback.get("hallucinated_mentions") or [])),
        },
        "hitl": {
            "enabled": True,
            "role": "residual_conflict_arbiter",
            "queue_count": len(human_queue),
            "geometry_queue_count": len(
                [
                    row
                    for row in human_queue
                    if str(row.get("claim_type", "") or "").strip().lower() == "bbox"
                ]
            ),
        },
    }


def _probe_to_vote(probe: Dict[str, object], resp: Dict[str, object], view_type: str) -> Dict[str, object]:
    response_format = dict(probe.get("response_format") or {})
    options = [str(x).strip() for x in list(probe.get("candidate_options") or []) if str(x).strip()]
    parsed = _extract_answer_fields(resp, options=options)
    vote_score = 0.0
    try:
        parsed_score = parsed.get("score", 0.0)
        vote_score = float(parsed_score) if parsed_score is not None else 0.0
    except Exception:
        vote_score = 0.0
    if str(response_format.get("type", "") or "").strip().lower() == "selection":
        selected = str(parsed.get("selection", "") or "").strip()
        expected = str(probe.get("expected_answer", "") or "").strip()
        if selected and selected.lower() == expected.lower():
            decision = "support"
        elif selected and selected != "uncertain":
            decision = "conflict"
        else:
            decision = "uncertain"
        return {
            "claim_id": str(probe.get("target_claim_id", "") or ""),
            "view_type": view_type,
            "vote": decision,
            "score": float(vote_score),
            "probe_id": str(probe.get("probe_id", "") or ""),
            "raw_text": str(parsed.get("raw_text", "") or ""),
            "probe_family": str(probe.get("probe_family", "constrained_correction") or "constrained_correction"),
            "selected_value": selected,
            "candidate_options": options,
            "correction_value": (
                selected
                if selected and selected != "uncertain" and selected.lower() != expected.lower()
                else ""
            ),
        }
    answer = str(parsed["answer"])
    expected = str(probe.get("expected_answer", "yes") or "yes").strip().lower()
    if expected not in {"yes", "no"}:
        expected = "yes"
    if answer in {"yes", "no"}:
        decision = "support" if answer == expected else "conflict"
    else:
        decision = "uncertain"
    return {
        "claim_id": str(probe.get("target_claim_id", "") or ""),
        "view_type": view_type,
        "vote": decision,
        "score": float(vote_score),
        "probe_id": str(probe.get("probe_id", "") or ""),
        "raw_text": str(parsed.get("raw_text", "") or ""),
        "probe_family": str(probe.get("probe_family", "binary_verification") or "binary_verification"),
        "expected_answer": expected,
        "observed_answer": answer,
    }


def _collect_correction_candidates(votes: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for vote in votes:
        claim_id = str(vote.get("claim_id", "") or "").strip()
        correction_value = str(vote.get("correction_value", "") or "").strip()
        if not claim_id or not correction_value:
            continue
        bucket = out.setdefault(claim_id, {"scores": {}, "options": []})
        scores = dict(bucket.get("scores") or {})
        options = list(bucket.get("options") or [])
        scores[correction_value] = min(1.0, float(scores.get(correction_value, 0.0) or 0.0) + float(vote.get("score", 0.0) or 0.0))
        for item in list(vote.get("candidate_options") or []):
            token = str(item or "").strip()
            if token and token not in options:
                options.append(token)
        bucket["scores"] = scores
        bucket["options"] = options
    for claim_id, bucket in out.items():
        scores = dict(bucket.get("scores") or {})
        ranked = sorted(scores.items(), key=lambda row: (-float(row[1]), str(row[0])))
        bucket["ranked"] = [{"value": key, "score": float(value)} for key, value in ranked]
        if ranked:
            bucket["best_value"] = str(ranked[0][0])
            bucket["best_score"] = float(ranked[0][1])
    return out


def _select_claim_covering_probes(
    probes: List[Dict[str, object]],
    *,
    max_items: int = 0,
) -> List[Dict[str, object]]:
    """Pick at least one low-cost probe per claim before adding correction probes."""
    limit = int(max_items or 0)
    selected: List[Dict[str, object]] = []
    seen_claims = set()

    def _try_add(probe: Dict[str, object]) -> bool:
        if limit > 0 and len(selected) >= limit:
            return False
        claim_id = str(probe.get("target_claim_id", "") or "").strip()
        if not claim_id or claim_id in seen_claims:
            return True
        selected.append(probe)
        seen_claims.add(claim_id)
        return True

    for probe in probes:
        probe_type = str(probe.get("probe_type", "") or "").strip().lower()
        if "correction" in probe_type:
            continue
        if not _try_add(probe):
            return selected

    for probe in probes:
        if not _try_add(probe):
            return selected
    return selected


def _run_probe_batch(
    *,
    graph: Dict[str, object],
    image_path: str,
    verifier,
    probes: List[Dict[str, object]],
    view_type: str,
    ontology=None,
    progress_cb: Optional[Callable[[str], None]] = None,
    stage_name: str = "",
) -> Dict[str, List[Dict[str, object]]]:
    def _emit(text: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(text or "").strip())
        except Exception:
            pass

    votes: List[Dict[str, object]] = []
    results: List[Dict[str, object]] = []
    total = int(len(list(probes or [])))
    if stage_name:
        _emit(f"[CYCLE-STAGE] {stage_name} begin total={total}")
    batch_items: List[Dict[str, object]] = []
    for probe in probes:
        batch_items.append(
            {
                "probe_id": probe.get("probe_id"),
                "question": str(probe.get("question", "") or ""),
                "regions": _build_regions(graph, list(probe.get("evidence_node_ids") or [])),
                "response_format": _probe_response_format(graph, probe, ontology=ontology),
                "schema": probe_response_schema(probe),
            }
        )

    batch_responses: Optional[List[Dict[str, object]]] = None
    if batch_items:
        try:
            batch_responses = list(
                verifier.answer_probe_batch(
                    image_path=image_path,
                    probes=batch_items,
                )
                or []
            )
            if len(batch_responses) != len(batch_items):
                raise RuntimeError(
                    f"batch verifier returned {len(batch_responses)} responses for {len(batch_items)} probes"
                )
            if stage_name:
                _emit(f"[CYCLE-STAGE] {stage_name} api_mode=batch count={len(batch_items)}")
        except Exception as exc:
            batch_responses = None
            if stage_name:
                _emit(f"[CYCLE-STAGE][WARN] {stage_name} batch fallback to single-call mode: {str(exc)[:180]}")

    for idx, probe in enumerate(probes, start=1):
        response_format = _probe_response_format(graph, probe, ontology=ontology)
        try:
            schema = probe_response_schema(probe)
            if batch_responses is not None:
                resp = dict(batch_responses[idx - 1] or {})
            else:
                try:
                    resp = verifier.answer_probe(
                        image_path=image_path,
                        question=str(probe.get("question", "") or ""),
                        regions=_build_regions(graph, list(probe.get("evidence_node_ids") or [])),
                        response_format=response_format,
                        schema=schema,
                    )
                except TypeError:
                    resp = verifier.answer_probe(
                        image_path=image_path,
                        question=str(probe.get("question", "") or ""),
                        regions=_build_regions(graph, list(probe.get("evidence_node_ids") or [])),
                        response_format=response_format,
                    )
        except Exception as exc:
            # Keep cycle alive under API throttling/transient failures.
            resp = {
                "answer": "uncertain",
                "score": None,
                "reason": "invalid_response",
                "raw_text": str(exc),
                "error": str(exc),
                "raw_response": {},
                "schema_valid": False,
                "is_truncated": False,
                "is_valid": False,
            }
            if stage_name:
                _emit(
                    f"[CYCLE-STAGE][WARN] {stage_name} probe failed: {str(exc)[:180]}"
                )
        vote = _probe_to_vote(probe, resp, view_type)
        votes.append(vote)
        response_provider = str(
            (dict(resp.get("raw_response") or {}).get("provider"))
            or resp.get("provider")
            or ""
        ).strip()
        results.append(
            _sanitize_probe_result_row(
                {
                "probe_id": probe.get("probe_id"),
                "view_type": view_type,
                "probe_type": probe.get("probe_type"),
                "chain_id": probe.get("chain_id"),
                "turn": probe.get("turn"),
                "question": probe.get("question"),
                "target_claim_id": probe.get("target_claim_id"),
                "evidence_node_ids": list(probe.get("evidence_node_ids") or []),
                "evidence_edge_ids": list(probe.get("evidence_edge_ids") or []),
                "candidate_options": list(probe.get("candidate_options") or []),
                "probe_family": probe.get("probe_family"),
                "expected_answer": probe.get("expected_answer"),
                "response_schema": probe_response_schema(probe),
                "schema_valid": bool(resp.get("schema_valid", True)),
                "response_provider": response_provider,
                "parsed_response": dict(resp or {}),
                "response": dict(resp or {}),
                }
            )
        )
        if stage_name:
            pct = int(round(100.0 * float(idx) / float(max(1, total))))
            _emit(f"[CYCLE-STAGE] {stage_name} progress done={idx} total={total} pct={pct}")
    if stage_name:
        _emit(f"[CYCLE-STAGE] {stage_name} done total={total}")
    return {"votes": votes, "results": results}


def _select_probes_for_target_scope(
    probes: Iterable[Dict[str, object]],
    *,
    claim_ids: Set[str],
    node_ids: Set[str],
    edge_ids: Set[str],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for probe in list(probes or []):
        row = dict(probe or {})
        if _row_matches_target_scope(row, claim_ids=claim_ids, node_ids=node_ids, edge_ids=edge_ids):
            out.append(row)
    return out


def rerun_cycle_refine_for_claims(
    *,
    graph: Dict[str, object],
    image_path: str,
    verifier,
    ontology,
    cfg: Dict[str, object],
    target_claim_ids: Iterable[object],
    base_result: Optional[Dict[str, object]] = None,
    correction_memory: Optional[Dict[str, object]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    def _emit_progress(text: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(text or "").strip())
        except Exception:
            pass

    cycle_cfg = dict(cfg.get("cycle") or {})
    caption_cfg = dict(cfg.get("caption") or {})
    effective_memory = _effective_correction_memory(cfg, correction_memory)
    current_graph = {
        "image_id": graph.get("image_id"),
        "nodes": [dict(x) for x in graph.get("nodes") or []],
        "edges": [dict(x) for x in graph.get("edges") or []],
        "validator_flags": list(graph.get("validator_flags") or []),
        "metadata": dict(graph.get("metadata") or {}),
    }
    current_graph["metadata"]["image_path"] = image_path
    resolved_scope = _resolved_scope(graph=current_graph, base_result=base_result)

    focus_cfg = {
        "enabled": bool(cycle_cfg.get("enable_person_focus", False)),
        "subject_label": str(cycle_cfg.get("focus_subject_label", "person") or "person"),
        "max_hops": int(cycle_cfg.get("focus_max_hops", 1) or 1),
        "direct_relations_only": bool(cycle_cfg.get("focus_direct_relations_only", False)),
        "max_subjects": int(cycle_cfg.get("focus_max_subjects", 0) or 0),
    }
    focused_graph = build_focus_graph(
        current_graph,
        enabled=focus_cfg["enabled"],
        subject_label=focus_cfg["subject_label"],
        max_hops=focus_cfg["max_hops"],
        direct_relations_only=focus_cfg["direct_relations_only"],
        max_subjects=focus_cfg["max_subjects"],
    )

    all_claims = {c.claim_id: c for c in graph_to_claims(focused_graph)}
    scope = _target_scope_from_claim_ids(current_graph, target_claim_ids)
    requested_claim_ids = set(scope["claim_ids"])
    target_node_ids = set(scope["node_ids"])
    target_edge_ids = set(scope["edge_ids"])
    if not requested_claim_ids:
        _emit_progress("[CYCLE-TARGETED] empty target scope; falling back to full cycle rerun")
        return run_cycle_refine(
            graph=graph,
            image_path=image_path,
            verifier=verifier,
            ontology=ontology,
            cfg=cfg,
            base_result=base_result,
            correction_memory=correction_memory,
            progress_cb=progress_cb,
        )

    target_ids = set(requested_claim_ids)
    target_ids.update(
        claim_id
        for claim_id, claim in all_claims.items()
        if _row_matches_target_scope(
            claim.to_dict(),
            claim_ids=requested_claim_ids,
            node_ids=target_node_ids,
            edge_ids=target_edge_ids,
        )
    )

    graph_present_claim_ids = {
        claim_id
        for claim_id in target_ids
        if _claim_exists_in_graph(current_graph, claim_id)
    }
    graph_present_claim_ids = {
        claim_id
        for claim_id in graph_present_claim_ids
        if claim_id not in set(resolved_scope.get("claim_ids") or set())
    }
    target_ids = {
        claim_id
        for claim_id in target_ids
        if claim_id not in set(resolved_scope.get("claim_ids") or set())
    }
    removed_claim_ids = set(target_ids).difference(graph_present_claim_ids)

    target_claims: Dict[str, Claim] = {
        claim_id: claim
        for claim_id, claim in all_claims.items()
        if claim_id in target_ids
    }
    for claim_id in graph_present_claim_ids:
        if claim_id in target_claims:
            continue
        synthetic_claim = _claim_from_graph(focused_graph, claim_id)
        if synthetic_claim is not None:
            target_claims[claim_id] = synthetic_claim

    _emit_progress(
        "[CYCLE-TARGETED] "
        f"start requested_claims={len(requested_claim_ids)} affected_claims={len(target_ids)} "
        f"nodes={len(target_node_ids)} edges={len(target_edge_ids)}"
    )

    votes: List[Dict[str, object]] = []
    probe_results: List[Dict[str, object]] = []
    caption_payload: Dict[str, object] = {}

    if bool(cycle_cfg.get("enable_single_turn_probes", True)):
        all_single_probes = build_single_turn_probes(
            focused_graph,
            correction_memory=effective_memory,
            ontology=ontology,
            resolved_claim_ids=set(resolved_scope.get("claim_ids") or set()),
        )
        all_single_probes = _filter_out_resolved_probes(all_single_probes, resolved_scope)
        single_probes = _select_probes_for_target_scope(
            all_single_probes,
            claim_ids=target_ids,
            node_ids=target_node_ids,
            edge_ids=target_edge_ids,
        )
        if single_probes:
            batch = _run_probe_batch(
                graph=focused_graph,
                image_path=image_path,
                verifier=verifier,
                probes=single_probes,
                view_type="single_turn_vqa",
                ontology=ontology,
                progress_cb=_emit_progress,
                stage_name="single_vqa",
            )
            votes.extend(batch["votes"])
            probe_results.extend(batch["results"])
        else:
            _emit_progress("[CYCLE-TARGETED] single_vqa no affected probes")

    if bool(cycle_cfg.get("enable_multi_turn_probes", True)):
        all_multi_probes = build_multi_turn_probes(
            focused_graph,
            max_chains=int(cycle_cfg.get("multi_turn_max_chains", 24) or 24),
            enable_temporal_context=bool(cycle_cfg.get("enable_temporal_multi_turn", True)),
            correction_memory=effective_memory,
            ontology=ontology,
            resolved_claim_ids=set(resolved_scope.get("claim_ids") or set()),
        )
        all_multi_probes = _filter_out_resolved_probes(all_multi_probes, resolved_scope)
        multi_probes = _select_probes_for_target_scope(
            all_multi_probes,
            claim_ids=target_ids,
            node_ids=target_node_ids,
            edge_ids=target_edge_ids,
        )
        if multi_probes:
            batch = _run_probe_batch(
                graph=focused_graph,
                image_path=image_path,
                verifier=verifier,
                probes=multi_probes,
                view_type="multi_turn_vqa",
                ontology=ontology,
                progress_cb=_emit_progress,
                stage_name="multi_vqa",
            )
            votes.extend(batch["votes"])
            probe_results.extend(batch["results"])
        else:
            _emit_progress("[CYCLE-TARGETED] multi_vqa no affected probes")

    if bool(cycle_cfg.get("enable_caption_probe", True)):
        structured_feedback = bool(caption_cfg.get("structured_feedback", True))
        caption_prompt = build_caption_prompt(
            focused_graph,
            style=str(caption_cfg.get("style", "technical") or "technical"),
            max_sentences=int(caption_cfg.get("max_sentences", 4) or 4),
            require_relation_mentions=bool(caption_cfg.get("require_relation_mentions", True)),
            correction_memory=effective_memory,
            structured_feedback=structured_feedback,
        )
        caption_schema = dict(CAPTION_FEEDBACK_SCHEMA) if structured_feedback else None
        caption_regions = _build_regions(
            focused_graph,
            [str(node.get("entity_id", "") or "") for node in focused_graph.get("nodes") or []],
        )
        try:
            try:
                full_caption_payload = verifier.generate_caption(
                    image_path=image_path,
                    prompt=caption_prompt,
                    regions=caption_regions,
                    schema=caption_schema,
                )
            except TypeError:
                full_caption_payload = verifier.generate_caption(
                    image_path=image_path,
                    prompt=caption_prompt,
                    regions=caption_regions,
                )
            full_caption_payload = dict(full_caption_payload or {})
            if bool(full_caption_payload.get("error")):
                caption_feedback = _empty_caption_feedback()
            else:
                caption_feedback = caption_to_claim_feedback(
                    full_caption_payload,
                    focused_graph,
                    ontology,
                    effective_memory,
                    allow_unsupported_conflicts=bool(caption_cfg.get("emit_conflict_votes", True)),
                )
            full_caption_votes = [
                dict(row)
                for row in list(caption_feedback.get("votes") or [])
                if isinstance(row, dict)
            ]
            filtered_caption_votes = [
                dict(row)
                for row in full_caption_votes
                if str(row.get("claim_id", "") or "").strip() in target_ids
            ]
            filtered_caption_report = _filter_caption_report_for_scope(
                dict(caption_feedback.get("report") or {}),
                claim_ids=target_ids,
            )
            filtered_caption_report["caption_text"] = str(
                filtered_caption_report.get("caption_text", "")
                or full_caption_payload.get("caption_text", "")
                or full_caption_payload.get("caption", "")
                or full_caption_payload.get("raw_text", "")
                or ""
            ).strip()
            filtered_caption_report["vote_count"] = len(filtered_caption_votes)
            filtered_caption_report["support_vote_count"] = len(
                [row for row in filtered_caption_votes if str(row.get("vote", "") or "").strip().lower() == "support"]
            )
            filtered_caption_report["conflict_vote_count"] = len(
                [row for row in filtered_caption_votes if str(row.get("vote", "") or "").strip().lower() == "conflict"]
            )
            caption_payload = dict(full_caption_payload)
            caption_payload["caption_text"] = str(
                full_caption_payload.get("caption_text", "")
                or full_caption_payload.get("caption", "")
                or filtered_caption_report.get("caption_text", "")
                or ""
            ).strip()
            caption_payload["caption"] = str(
                full_caption_payload.get("caption", "")
                or caption_payload.get("caption_text", "")
                or ""
            ).strip()
            caption_payload["feedback"] = filtered_caption_report
            caption_payload["votes"] = filtered_caption_votes
            votes.extend(filtered_caption_votes)
            _emit_progress(
                "[CYCLE-CAPTION] "
                f"provider={str((dict(caption_payload.get('raw_response') or {}).get('provider')) or '').strip() or 'unknown'} "
                f"raw_text_len={len(str(caption_payload.get('raw_text', '') or ''))} "
                f"caption_len={len(str(caption_payload.get('caption_text', '') or ''))} "
                f"error={1 if bool(caption_payload.get('error')) else 0}"
            )
            _emit_progress(
                f"[CYCLE-TARGETED] caption refreshed affected_votes={len(filtered_caption_votes)}"
            )
        except Exception as exc:
            caption_payload = {
                "caption_text": "",
                "caption": "",
                "raw_text": str(exc),
                "raw_response": {},
                "schema_valid": False,
                "is_truncated": False,
                "is_valid": False,
                "error": str(exc),
                "feedback": dict(_empty_caption_feedback().get("report") or {}),
                "votes": [],
            }
            _emit_progress(
                "[CYCLE-CAPTION] "
                f"provider=unknown raw_text_len=0 caption_len=0 error=1 detail={str(exc)[:180]}"
            )

    weighted_votes, policy_report = apply_role_policy(
        votes,
        policy_override=dict(cfg.get("role_policy") or {}),
    )
    target_claims = aggregate_claim_scores(target_claims, weighted_votes)
    correction_candidates = _collect_correction_candidates(weighted_votes)
    revised_graph = revise_graph_from_claims(
        current_graph,
        target_claims,
        auto_accept_threshold=float(cycle_cfg.get("auto_accept_threshold", 0.85) or 0.85),
        auto_reject_threshold=float(cycle_cfg.get("auto_reject_threshold", 0.80) or 0.80),
        auto_drop_existence_threshold=float(cycle_cfg.get("auto_drop_existence_threshold", 0.92) or 0.92),
        auto_drop_support_ceiling=float(cycle_cfg.get("auto_drop_support_ceiling", 0.20) or 0.20),
        correction_memory=effective_memory,
        correction_candidates=correction_candidates,
        frame_idx=_extract_frame_idx(current_graph),
        finalized_weight_boost=float((cfg.get("memory") or {}).get("finalized_weight_boost", 1.25) or 1.25),
    )
    queue = build_human_arbitration_queue(
        target_claims,
        max_items=int(cycle_cfg.get("max_human_queries_per_frame", 3) or 3),
        threshold=float(cycle_cfg.get("human_escalation_threshold", 0.45) or 0.45),
        correction_memory=effective_memory,
        correction_candidates=correction_candidates,
        frame_idx=_extract_frame_idx(current_graph),
    )
    if bool(cycle_cfg.get("enable_geometry_review", True)):
        geometry_queue = build_geometry_review_queue(
            current_graph,
            target_claims,
            relation_vocab=getattr(ontology, "relation_vocabulary", {}) or {},
            preferred_anchor_label=str(focus_cfg.get("subject_label", "person") or "person"),
            conflict_threshold=float(cycle_cfg.get("geometry_conflict_threshold", 0.60) or 0.60),
            max_items=int(cycle_cfg.get("max_geometry_queries_per_frame", 2) or 2),
        )
        if geometry_queue:
            queue = list(queue) + list(geometry_queue)
            queue.sort(key=lambda row: float(row.get("priority", 0.0) or 0.0), reverse=True)
            queue = queue[: max(0, int(cycle_cfg.get("max_human_queries_per_frame", 3) or 3))]

    targeted_update = dict((revised_graph.get("metadata") or {}).get("cycle_update") or {})
    revised_graph["metadata"]["cycle_update"] = _merge_cycle_update(
        dict((((base_result or {}).get("graph_after") or {}).get("metadata") or {}).get("cycle_update") or {}),
        targeted_update,
        remove_claim_ids=target_ids,
        remove_node_ids=target_node_ids,
    )
    removed_nodes = {
        str(x or "").strip()
        for x in list(revised_graph["metadata"]["cycle_update"].get("auto_removed_node_ids") or [])
        if str(x or "").strip()
    }
    removed_claims = {
        str(x or "").strip()
        for x in list(revised_graph["metadata"]["cycle_update"].get("auto_removed_claim_ids") or [])
        if str(x or "").strip()
    }
    if removed_nodes or removed_claims:
        queue = [
            dict(row)
            for row in list(queue or [])
            if (
                str(row.get("claim_id", "") or "").strip() not in removed_claims
                and str(row.get("subject_id", "") or "").strip() not in removed_nodes
                and str(row.get("target_node_id", "") or "").strip() not in removed_nodes
            )
        ]

    partial_round = {
        "round_idx": len(list((base_result or {}).get("rounds") or [])),
        "mode": "targeted_claim_rerun",
        "target_claim_ids": sorted(target_ids),
        "focus_graph": focused_graph,
        "claims": {k: v.to_dict() for k, v in target_claims.items()},
        "votes": weighted_votes,
        "correction_candidates": correction_candidates,
        "probe_results": [_sanitize_probe_result_row(row) for row in list(probe_results or []) if isinstance(row, dict)],
        "caption": _sanitize_caption_payload(caption_payload),
        "human_queue": queue,
        "policy": policy_report,
    }

    base_claim_rows = _claim_payload_to_dict((base_result or {}).get("claims"))
    recomputed_claim_ids = set(removed_claim_ids)
    recomputed_claim_ids.update(target_claims.keys())
    recomputed_claim_ids.update(
        str(row.get("claim_id", "") or "").strip()
        for row in list(weighted_votes or [])
        if str(row.get("claim_id", "") or "").strip()
    )
    merged_claims = {
        claim_id: dict(row)
        for claim_id, row in base_claim_rows.items()
        if claim_id not in recomputed_claim_ids
    }
    merged_claims.update({claim_id: claim.to_dict() for claim_id, claim in target_claims.items()})

    base_votes = _filter_rows_outside_scope(
        list((base_result or {}).get("votes") or []),
        claim_ids=target_ids,
        node_ids=target_node_ids,
        edge_ids=target_edge_ids,
    )
    merged_votes = base_votes + [dict(row) for row in list(weighted_votes or []) if isinstance(row, dict)]

    base_probe_results = _filter_rows_outside_scope(
        list((base_result or {}).get("probe_results") or []),
        claim_ids=target_ids,
        node_ids=target_node_ids,
        edge_ids=target_edge_ids,
    )
    merged_probe_results = base_probe_results + [
        _sanitize_probe_result_row(row)
        for row in list(probe_results or [])
        if isinstance(row, dict)
    ]

    base_queue = _filter_rows_outside_scope(
        list((base_result or {}).get("human_queue") or []),
        claim_ids=target_ids,
        node_ids=target_node_ids,
        edge_ids=target_edge_ids,
    )
    merged_queue = base_queue + [dict(row) for row in list(queue or []) if isinstance(row, dict)]

    merged_correction_candidates = {
        str(key): dict(value)
        for key, value in dict((base_result or {}).get("correction_candidates") or {}).items()
        if str(key).strip() not in target_ids and isinstance(value, dict)
    }
    for key, value in dict(correction_candidates or {}).items():
        if str(key).strip() and isinstance(value, dict):
            merged_correction_candidates[str(key).strip()] = dict(value)

    merged_resolved_claims = [
        dict(row)
        for row in list(resolved_scope.get("records") or [])
        if isinstance(row, dict)
    ]

    merged_caption_votes = [
        dict(row)
        for row in list(merged_votes or [])
        if str(row.get("view_type", "") or "").strip() == "caption"
    ]
    merged_caption = _merge_caption_payload(
        dict((base_result or {}).get("caption") or {}),
        _sanitize_caption_payload(caption_payload),
        target_claim_ids=target_ids,
        merged_caption_votes=merged_caption_votes,
    )

    merged_rounds = [dict(row) for row in list((base_result or {}).get("rounds") or []) if isinstance(row, dict)]
    merged_rounds.append(partial_round)
    merged_policy = _merge_policy_report(
        dict((base_result or {}).get("policy") or {}),
        dict(policy_report or {}),
        merged_votes=merged_votes,
    )
    result = {
        "graph_before": dict((base_result or {}).get("graph_after") or graph or {}),
        "graph_after": revised_graph,
        "rounds": merged_rounds,
        "claims": merged_claims,
        "votes": merged_votes,
        "correction_candidates": merged_correction_candidates,
        "probe_results": merged_probe_results,
        "caption": merged_caption,
        "human_queue": merged_queue,
        "policy": merged_policy,
        "resolved_claims": merged_resolved_claims,
        "suppressed_questions": [
            dict(row)
            for row in list((resolved_scope.get("suppressed_questions") or []))
            if isinstance(row, dict)
        ],
        "memory": _memory_summary(effective_memory),
        "focus": dict(((focused_graph.get("metadata") or {}).get("focus_filter")) or {}),
    }
    runtime = dict((base_result or {}).get("runtime") or {})
    runtime["targeted_reverify"] = True
    runtime["target_claim_count"] = len(target_ids)
    runtime["target_node_count"] = len(target_node_ids)
    runtime["target_edge_count"] = len(target_edge_ids)
    result["runtime"] = runtime
    result["agents"] = _build_agent_summary(
        cycle_cfg=cycle_cfg,
        probe_results=list(result.get("probe_results") or []),
        caption_payload=dict(result.get("caption") or {}),
        human_queue=list(result.get("human_queue") or []),
        graph_after=revised_graph,
    )
    result["summary"] = _cycle_summary(result=result, graph_after=revised_graph, rounds=merged_rounds)
    result["metrics"] = evaluate_cycle_result(result)
    result["summary"].update(dict(result.get("metrics") or {}))
    _emit_progress(
        f"[CYCLE-TARGETED] done recomputed_claims={len(target_claims)} votes={len(weighted_votes)} probes={len(probe_results)}"
    )
    return result


def run_cycle_refine(
    *,
    graph: Dict[str, object],
    image_path: str,
    verifier,
    ontology,
    cfg: Dict[str, object],
    base_result: Optional[Dict[str, object]] = None,
    correction_memory: Optional[Dict[str, object]] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    def _emit_progress(text: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(text or "").strip())
        except Exception:
            pass

    cycle_cfg = dict(cfg.get("cycle") or {})
    caption_cfg = dict(cfg.get("caption") or {})
    effective_memory = _effective_correction_memory(cfg, correction_memory)

    current_graph = {
        "image_id": graph.get("image_id"),
        "nodes": [dict(x) for x in graph.get("nodes") or []],
        "edges": [dict(x) for x in graph.get("edges") or []],
        "validator_flags": list(graph.get("validator_flags") or []),
        "metadata": dict(graph.get("metadata") or {}),
    }
    current_graph["metadata"]["image_path"] = image_path
    resolved_scope = _resolved_scope(graph=current_graph, base_result=base_result)

    rounds: List[Dict[str, object]] = []
    max_rounds = max(1, int(cycle_cfg.get("max_revision_rounds", 1) or 1))
    _emit_progress(f"[CYCLE-PROGRESS] cycle start rounds={max_rounds}")
    focus_cfg = {
        "enabled": bool(cycle_cfg.get("enable_person_focus", False)),
        "subject_label": str(cycle_cfg.get("focus_subject_label", "person") or "person"),
        "max_hops": int(cycle_cfg.get("focus_max_hops", 1) or 1),
        "direct_relations_only": bool(cycle_cfg.get("focus_direct_relations_only", False)),
        "max_subjects": int(cycle_cfg.get("focus_max_subjects", 0) or 0),
    }

    for round_idx in range(max_rounds):
        focused_graph = build_focus_graph(
            current_graph,
            enabled=focus_cfg["enabled"],
            subject_label=focus_cfg["subject_label"],
            max_hops=focus_cfg["max_hops"],
            direct_relations_only=focus_cfg["direct_relations_only"],
            max_subjects=focus_cfg["max_subjects"],
        )
        claims = {c.claim_id: c for c in graph_to_claims(focused_graph)}
        votes: List[Dict[str, object]] = []
        probe_results: List[Dict[str, object]] = []
        caption_payload: Dict[str, object] = {}

        if bool(cycle_cfg.get("enable_single_turn_probes", True)):
            _emit_progress(f"[CYCLE-PROGRESS] single_vqa start round={round_idx + 1}/{max_rounds}")
            all_single_probes = build_single_turn_probes(
                focused_graph,
                correction_memory=effective_memory,
                ontology=ontology,
                resolved_claim_ids=set(resolved_scope.get("claim_ids") or set()),
            )
            all_single_probes = _filter_out_resolved_probes(all_single_probes, resolved_scope)
            max_single_raw = cycle_cfg.get("max_single_turn_probes", None)
            max_single: Optional[int]
            if max_single_raw is None:
                max_single = None
            else:
                try:
                    max_single = int(max_single_raw)
                except Exception:
                    max_single = None
            if max_single is None or max_single <= 0:
                single_probes = all_single_probes
            else:
                if bool(cycle_cfg.get("verify_all_claims", False)):
                    single_probes = _select_claim_covering_probes(all_single_probes, max_items=max_single)
                else:
                    if max_single == 3:
                        single_probes = _select_single_turn_probes_balanced(all_single_probes, max_items=max_single)
                    else:
                        single_probes = _limit_probes(all_single_probes, max_items=max_single)
            batch = _run_probe_batch(
                graph=focused_graph,
                image_path=image_path,
                verifier=verifier,
                probes=single_probes,
                view_type="single_turn_vqa",
                ontology=ontology,
                progress_cb=_emit_progress,
                stage_name="single_vqa",
            )
            votes.extend(batch["votes"])
            probe_results.extend(batch["results"])
            _emit_progress(
                f"[CYCLE-PROGRESS] single_vqa done round={round_idx + 1}/{max_rounds} probes={len(single_probes)}"
            )
        else:
            _emit_progress(f"[CYCLE-PROGRESS] single_vqa skip round={round_idx + 1}/{max_rounds}")
            _emit_progress("[CYCLE-STAGE] single_vqa begin total=0")
            _emit_progress("[CYCLE-STAGE] single_vqa done total=0")

        if bool(cycle_cfg.get("enable_multi_turn_probes", True)):
            _emit_progress(f"[CYCLE-PROGRESS] multi_vqa start round={round_idx + 1}/{max_rounds}")
            all_multi_probes = build_multi_turn_probes(
                focused_graph,
                max_chains=int(cycle_cfg.get("multi_turn_max_chains", 24) or 24),
                enable_temporal_context=bool(cycle_cfg.get("enable_temporal_multi_turn", True)),
                correction_memory=effective_memory,
                ontology=ontology,
                resolved_claim_ids=set(resolved_scope.get("claim_ids") or set()),
            )
            all_multi_probes = _filter_out_resolved_probes(all_multi_probes, resolved_scope)
            max_multi_raw = cycle_cfg.get("max_multi_turn_probes", None)
            max_multi: Optional[int]
            if max_multi_raw is None:
                max_multi = None
            else:
                try:
                    max_multi = int(max_multi_raw)
                except Exception:
                    max_multi = None
            if max_multi is None or max_multi <= 0:
                multi_probes = all_multi_probes
            else:
                multi_probes = _limit_probes(all_multi_probes, max_items=max_multi)
            batch = _run_probe_batch(
                graph=focused_graph,
                image_path=image_path,
                verifier=verifier,
                probes=multi_probes,
                view_type="multi_turn_vqa",
                ontology=ontology,
                progress_cb=_emit_progress,
                stage_name="multi_vqa",
            )
            votes.extend(batch["votes"])
            probe_results.extend(batch["results"])
            _emit_progress(
                f"[CYCLE-PROGRESS] multi_vqa done round={round_idx + 1}/{max_rounds} probes={len(multi_probes)}"
            )
        else:
            _emit_progress(f"[CYCLE-PROGRESS] multi_vqa skip round={round_idx + 1}/{max_rounds}")
            _emit_progress("[CYCLE-STAGE] multi_vqa begin total=0")
            _emit_progress("[CYCLE-STAGE] multi_vqa done total=0")

        if bool(cycle_cfg.get("enable_caption_probe", True)):
            _emit_progress(f"[CYCLE-PROGRESS] caption start round={round_idx + 1}/{max_rounds}")
            _emit_progress("[CYCLE-STAGE] caption begin total=1")
            structured_feedback = bool(caption_cfg.get("structured_feedback", True))
            caption_prompt = build_caption_prompt(
                focused_graph,
                style=str(caption_cfg.get("style", "technical") or "technical"),
                max_sentences=int(caption_cfg.get("max_sentences", 4) or 4),
                require_relation_mentions=bool(caption_cfg.get("require_relation_mentions", True)),
                correction_memory=effective_memory,
                structured_feedback=structured_feedback,
            )
            caption_schema = dict(CAPTION_FEEDBACK_SCHEMA) if structured_feedback else None
            caption_regions = _build_regions(
                focused_graph,
                [str(node.get("entity_id", "") or "") for node in focused_graph.get("nodes") or []],
            )
            try:
                try:
                    caption_payload = verifier.generate_caption(
                        image_path=image_path,
                        prompt=caption_prompt,
                        regions=caption_regions,
                        schema=caption_schema,
                    )
                except TypeError:
                    caption_payload = verifier.generate_caption(
                        image_path=image_path,
                        prompt=caption_prompt,
                        regions=caption_regions,
                    )
                caption_payload = dict(caption_payload or {})
                if bool(caption_payload.get("error")):
                    caption_feedback = _empty_caption_feedback()
                else:
                    caption_feedback = caption_to_claim_feedback(
                        caption_payload,
                        focused_graph,
                        ontology,
                        effective_memory,
                        allow_unsupported_conflicts=bool(caption_cfg.get("emit_conflict_votes", True)),
                    )
                caption_votes = [dict(row) for row in list(caption_feedback.get("votes") or []) if isinstance(row, dict)]
                caption_report = dict(caption_feedback.get("report") or {})
                caption_text = str(
                    caption_report.get("caption_text", "")
                    or caption_payload.get("caption_text", "")
                    or caption_payload.get("caption", "")
                    or caption_payload.get("raw_text", "")
                    or ""
                ).strip()
                caption_payload["caption_text"] = caption_text
                caption_payload["caption"] = caption_text
                caption_payload["feedback"] = caption_report
                caption_payload["votes"] = caption_votes
                votes.extend(caption_votes)
                _emit_progress(
                    "[CYCLE-CAPTION] "
                    f"provider={str((dict(caption_payload.get('raw_response') or {}).get('provider')) or '').strip() or 'unknown'} "
                    f"raw_text_len={len(str(caption_payload.get('raw_text', '') or ''))} "
                    f"caption_len={len(str(caption_text or ''))} "
                    f"error={1 if bool(caption_payload.get('error')) else 0}"
                )
                _emit_progress("[CYCLE-STAGE] caption progress done=1 total=1 pct=100")
            except Exception as exc:
                caption_payload = {
                    "caption_text": "",
                    "caption": "",
                    "raw_text": str(exc),
                    "raw_response": {},
                    "schema_valid": False,
                    "is_truncated": False,
                    "is_valid": False,
                    "error": str(exc),
                }
                caption_feedback = _empty_caption_feedback()
                caption_payload["feedback"] = dict(caption_feedback.get("report") or {})
                caption_payload["votes"] = []
                _emit_progress(
                    "[CYCLE-CAPTION] "
                    f"provider=unknown raw_text_len=0 caption_len=0 error=1 detail={str(exc)[:180]}"
                )
                _emit_progress("[CYCLE-STAGE] caption progress done=1 total=1 pct=100")
            _emit_progress(
                f"[CYCLE-PROGRESS] caption done round={round_idx + 1}/{max_rounds} "
                f"votes={len(list(caption_payload.get('votes') or []))}"
            )
            _emit_progress("[CYCLE-STAGE] caption done total=1")
        else:
            _emit_progress(f"[CYCLE-PROGRESS] caption skip round={round_idx + 1}/{max_rounds}")
            _emit_progress("[CYCLE-STAGE] caption begin total=0")
            _emit_progress("[CYCLE-STAGE] caption done total=0")

        weighted_votes, policy_report = apply_role_policy(
            votes,
            policy_override=dict(cfg.get("role_policy") or {}),
        )
        claims = aggregate_claim_scores(claims, weighted_votes)
        correction_candidates = _collect_correction_candidates(weighted_votes)
        revised_graph = revise_graph_from_claims(
            current_graph,
            claims,
            auto_accept_threshold=float(cycle_cfg.get("auto_accept_threshold", 0.85) or 0.85),
            auto_reject_threshold=float(cycle_cfg.get("auto_reject_threshold", 0.80) or 0.80),
            auto_drop_existence_threshold=float(cycle_cfg.get("auto_drop_existence_threshold", 0.92) or 0.92),
            auto_drop_support_ceiling=float(cycle_cfg.get("auto_drop_support_ceiling", 0.20) or 0.20),
            correction_memory=effective_memory,
            correction_candidates=correction_candidates,
            frame_idx=_extract_frame_idx(current_graph),
            finalized_weight_boost=float((cfg.get("memory") or {}).get("finalized_weight_boost", 1.25) or 1.25),
        )
        queue = build_human_arbitration_queue(
            claims,
            max_items=int(cycle_cfg.get("max_human_queries_per_frame", 3) or 3),
            threshold=float(cycle_cfg.get("human_escalation_threshold", 0.45) or 0.45),
            correction_memory=effective_memory,
            correction_candidates=correction_candidates,
            frame_idx=_extract_frame_idx(current_graph),
        )
        if bool(cycle_cfg.get("enable_geometry_review", True)):
            geometry_queue = build_geometry_review_queue(
                current_graph,
                claims,
                relation_vocab=getattr(ontology, "relation_vocabulary", {}) or {},
                preferred_anchor_label=str(focus_cfg.get("subject_label", "person") or "person"),
                conflict_threshold=float(cycle_cfg.get("geometry_conflict_threshold", 0.60) or 0.60),
                max_items=int(cycle_cfg.get("max_geometry_queries_per_frame", 2) or 2),
            )
            if geometry_queue:
                queue = list(queue) + list(geometry_queue)
                queue.sort(key=lambda row: float(row.get("priority", 0.0) or 0.0), reverse=True)
                queue = queue[: max(0, int(cycle_cfg.get("max_human_queries_per_frame", 3) or 3))]
        cycle_update = dict((revised_graph.get("metadata") or {}).get("cycle_update") or {})
        removed_nodes = {str(x or "").strip() for x in list(cycle_update.get("auto_removed_node_ids") or []) if str(x or "").strip()}
        removed_claims = {str(x or "").strip() for x in list(cycle_update.get("auto_removed_claim_ids") or []) if str(x or "").strip()}
        if removed_nodes or removed_claims:
            queue = [
                dict(row)
                for row in list(queue or [])
                if (
                    str(row.get("claim_id", "") or "").strip() not in removed_claims
                    and str(row.get("subject_id", "") or "").strip() not in removed_nodes
                    and str(row.get("target_node_id", "") or "").strip() not in removed_nodes
                )
            ]

        round_payload = {
            "round_idx": round_idx,
            "focus_graph": focused_graph,
            "claims": {k: v.to_dict() for k, v in claims.items()},
            "votes": weighted_votes,
            "correction_candidates": correction_candidates,
            "probe_results": [_sanitize_probe_result_row(row) for row in list(probe_results or []) if isinstance(row, dict)],
            "caption": _sanitize_caption_payload(caption_payload),
            "human_queue": queue,
            "policy": policy_report,
        }
        rounds.append(round_payload)
        _emit_progress(f"[CYCLE-PROGRESS] round done round={round_idx + 1}/{max_rounds}")

        if revised_graph == current_graph:
            current_graph = revised_graph
            break
        current_graph = revised_graph

    final_round = rounds[-1] if rounds else {}
    all_votes: List[Dict[str, object]] = []
    all_probe_results: List[Dict[str, object]] = []
    for round_payload in rounds:
        all_votes.extend([dict(row) for row in list(round_payload.get("votes") or []) if isinstance(row, dict)])
        all_probe_results.extend(
            [_sanitize_probe_result_row(row) for row in list(round_payload.get("probe_results") or []) if isinstance(row, dict)]
        )

    result = {
        "graph_before": graph,
        "graph_after": current_graph,
        "rounds": rounds,
        "claims": dict(final_round.get("claims") or {}),
        "votes": all_votes,
        "correction_candidates": dict(final_round.get("correction_candidates") or {}),
        "probe_results": all_probe_results,
        "caption": _sanitize_caption_payload(dict(final_round.get("caption") or {})),
        "human_queue": list(final_round.get("human_queue") or []),
        "policy": dict(final_round.get("policy") or {}),
        "resolved_claims": [dict(row) for row in list(resolved_scope.get("records") or []) if isinstance(row, dict)],
        "suppressed_questions": [
            dict(row)
            for row in list((resolved_scope.get("suppressed_questions") or []))
            if isinstance(row, dict)
        ],
        "memory": _memory_summary(effective_memory),
        "focus": dict(((final_round.get("focus_graph") or {}).get("metadata") or {}).get("focus_filter") or {}),
    }
    result["agents"] = _build_agent_summary(
        cycle_cfg=cycle_cfg,
        probe_results=list(result.get("probe_results") or []),
        caption_payload=dict(result.get("caption") or {}),
        human_queue=list(result.get("human_queue") or []),
        graph_after=current_graph,
    )
    result["summary"] = _cycle_summary(result=result, graph_after=current_graph, rounds=rounds)
    result["metrics"] = evaluate_cycle_result(result)
    result["summary"].update(dict(result.get("metrics") or {}))
    _emit_progress("[CYCLE-PROGRESS] cycle done")
    return result
