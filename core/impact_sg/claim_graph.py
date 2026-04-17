from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set

from .cycle_types import Claim
from .correction_memory import common_confusions, prompt_alias_candidates


def _tokenize(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def _is_placeholder_value(value: object) -> bool:
    token = str(value or "").strip().lower()
    if not token:
        return True
    return token in {
        "unknown",
        "unk",
        "none",
        "null",
        "n/a",
        "na",
        "unspecified",
        "unlabeled",
    }


def _is_generic_label(label: object) -> bool:
    token = str(label or "").strip().lower()
    if not token:
        return True
    return token in {"object", "thing", "entity", "unknown", "unlabeled", "unspecified"}


def _node_label(node: Dict[str, object]) -> str:
    canonical = str(node.get("canonical_label", "") or "").strip()
    if canonical:
        return canonical
    fallback = str(node.get("label", "") or "").strip()
    return fallback or "object"


def _edge_identity(edge: Dict[str, object]) -> str:
    eid = str(edge.get("edge_id", "") or "").strip()
    if eid:
        return eid
    src = str(edge.get("src_id", "") or "").strip()
    rel = str(edge.get("relation", "") or "").strip()
    dst = str(edge.get("dst_id", "") or "").strip()
    if src and rel and dst:
        return f"edge_{_tokenize(src)}_{_tokenize(rel)}_{_tokenize(dst)}"
    return ""


def _node_alias_map(graph: Dict[str, object]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    buckets: Dict[str, int] = {}
    for node in list(graph.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        nid = str(node.get("entity_id", "") or "").strip()
        if not nid:
            continue
        label = _node_label(node).strip().lower() or "object"
        buckets[label] = int(buckets.get(label, 0) + 1)
        aliases[nid] = f"{label} {buckets[label]}"
    return aliases


def _as_confidence(payload: Dict[str, object], keys: List[str], default: float = 0.0) -> float:
    for key in keys:
        if key not in payload:
            continue
        try:
            val = float(payload.get(key) or default)
        except Exception:
            continue
        if val > 1.0 and val <= 100.0:
            val = val / 100.0
        return max(0.0, min(1.0, val))
    return max(0.0, min(1.0, float(default)))


def existence_claim_id(subject_id: str) -> str:
    return f"claim_exists_{subject_id}"


def label_claim_id(subject_id: str) -> str:
    return f"claim_label_{subject_id}"


def attribute_claim_id(subject_id: str, slot: str) -> str:
    return f"claim_attr_{subject_id}_{_tokenize(slot)}"


def relation_claim_id(edge_id: str) -> str:
    return f"claim_rel_{edge_id}"


def _clone_graph_payload(graph: Dict[str, object]) -> Dict[str, object]:
    return {
        "image_id": graph.get("image_id"),
        "nodes": [dict(x) for x in graph.get("nodes") or []],
        "edges": [dict(x) for x in graph.get("edges") or []],
        "validator_flags": list(graph.get("validator_flags") or []),
        "metadata": dict(graph.get("metadata") or {}),
    }


def _temporal_context(graph: Dict[str, object]) -> Dict[str, object]:
    return dict((graph.get("metadata") or {}).get("temporal_context") or {})


def _track_history(graph: Dict[str, object], subject_id: str) -> Dict[str, object]:
    context = _temporal_context(graph)
    return dict((context.get("track_history") or {}).get(str(subject_id or "").strip()) or {})


def _relation_history(graph: Dict[str, object], src_id: str, relation: str, dst_id: str) -> Dict[str, object]:
    context = _temporal_context(graph)
    signature = "|".join(
        [
            str(src_id or "").strip(),
            str(relation or "").strip(),
            str(dst_id or "").strip(),
        ]
    )
    return dict((context.get("relation_history") or {}).get(signature) or {})


def _frame_phrase(frame_idx: object) -> str:
    try:
        return f"sampled frame {int(frame_idx)}"
    except Exception:
        return "a nearby sampled frame"


def _seen_frames_phrase(values: object, *, limit: int = 3) -> str:
    rendered: List[str] = []
    for item in list(values or []):
        try:
            rendered.append(str(int(item)))
        except Exception:
            continue
        if len(rendered) >= max(1, int(limit)):
            break
    if not rendered:
        return ""
    return ", ".join(rendered)


def build_focus_graph(
    graph: Dict[str, object],
    *,
    enabled: bool = False,
    subject_label: str = "person",
    max_hops: int = 1,
    direct_relations_only: bool = False,
    max_subjects: int = 0,
) -> Dict[str, object]:
    out = _clone_graph_payload(graph)
    focus_label = str(subject_label or "").strip().lower()
    if not enabled or not focus_label:
        return out

    ranked_subject_nodes = []
    for index, node in enumerate(out.get("nodes") or []):
        label = _node_label(node).strip().lower()
        if label != focus_label:
            continue
        try:
            score = float(node.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        ranked_subject_nodes.append((score, -index, str(node.get("entity_id", "") or "").strip()))
    ranked_subject_nodes.sort(reverse=True)
    subject_node_ids = [node_id for _, _, node_id in ranked_subject_nodes if node_id]
    if max_subjects and int(max_subjects) > 0:
        subject_node_ids = subject_node_ids[: max(1, int(max_subjects))]
    if not subject_node_ids:
        out.setdefault("metadata", {})["focus_filter"] = {
            "enabled": True,
            "subject_label": focus_label,
            "applied": False,
            "reason": "no_focus_subject_found",
        }
        return out

    adjacency: Dict[str, Set[str]] = {}
    for edge in out.get("edges") or []:
        src = str(edge.get("src_id", "") or "").strip()
        dst = str(edge.get("dst_id", "") or "").strip()
        if not src or not dst:
            continue
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)

    keep_node_ids: Set[str] = set(subject_node_ids)
    frontier: Set[str] = set(subject_node_ids)
    for _ in range(max(0, int(max_hops or 0))):
        next_frontier: Set[str] = set()
        for node_id in frontier:
            next_frontier.update(adjacency.get(node_id, set()))
        next_frontier.difference_update(keep_node_ids)
        if not next_frontier:
            break
        keep_node_ids.update(next_frontier)
        frontier = next_frontier

    out["nodes"] = [
        node
        for node in out.get("nodes") or []
        if str(node.get("entity_id", "") or "").strip() in keep_node_ids
    ]
    if direct_relations_only:
        out["edges"] = [
            edge
            for edge in out.get("edges") or []
            if (
                str(edge.get("src_id", "") or "").strip() in keep_node_ids
                and str(edge.get("dst_id", "") or "").strip() in keep_node_ids
                and (
                    str(edge.get("src_id", "") or "").strip() in subject_node_ids
                    or str(edge.get("dst_id", "") or "").strip() in subject_node_ids
                )
            )
        ]
    else:
        out["edges"] = [
            edge
            for edge in out.get("edges") or []
            if (
                str(edge.get("src_id", "") or "").strip() in keep_node_ids
                and str(edge.get("dst_id", "") or "").strip() in keep_node_ids
            )
        ]

    out.setdefault("metadata", {})["focus_filter"] = {
        "enabled": True,
        "subject_label": focus_label,
        "applied": True,
        "subject_node_ids": list(subject_node_ids),
        "kept_node_count": len(out.get("nodes") or []),
        "kept_edge_count": len(out.get("edges") or []),
        "max_hops": max(0, int(max_hops or 0)),
        "direct_relations_only": bool(direct_relations_only),
    }
    return out


def _alias_guidance(label: str, correction_memory: Optional[Dict[str, object]]) -> str:
    aliases = prompt_alias_candidates(correction_memory, label)
    if not aliases:
        return ""
    return " Treat aliases like " + ", ".join(aliases) + " as the same canonical label."


def _confusion_guidance(
    claim_type: str,
    canonical_value: str,
    correction_memory: Optional[Dict[str, object]],
) -> str:
    confusions = common_confusions(
        correction_memory,
        claim_type=claim_type,
        canonical_value=canonical_value,
    )
    if not confusions:
        return ""
    return " Check carefully against common confusions such as " + ", ".join(confusions) + "."


def _dedupe_tokens(values: List[str], *, limit: int = 0) -> List[str]:
    out: List[str] = []
    for value in values:
        token = str(value or "").strip()
        if not token or token in out:
            continue
        out.append(token)
        if limit > 0 and len(out) >= limit:
            break
    return out


def _rank_similar_candidates(target: str, candidates: List[str], *, limit: int = 3) -> List[str]:
    scored = []
    token = str(target or "").strip().lower()
    for candidate in candidates:
        item = str(candidate or "").strip()
        if not item:
            continue
        ratio = SequenceMatcher(None, token, item.lower()).ratio() if token else 0.0
        scored.append((ratio, item.lower(), item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return _dedupe_tokens([item for _, _, item in scored], limit=max(0, int(limit)))


def _ontology_entity_labels(ontology) -> List[str]:
    if ontology is None:
        return []
    out: List[str] = []
    for row in getattr(ontology, "canonical_entities", []) or []:
        label = str((row or {}).get("label", "") or "").strip()
        if label and label not in out:
            out.append(label)
    return out


def _ontology_relations(ontology) -> List[str]:
    if ontology is None:
        return []
    out: List[str] = []
    relation_vocab = getattr(ontology, "relation_vocabulary", {}) or {}
    for values in relation_vocab.values():
        for item in list(values or []):
            token = str(item or "").strip()
            if token and token not in out:
                out.append(token)
    return out


def _label_candidate_options(
    label: str,
    *,
    ontology,
    correction_memory: Optional[Dict[str, object]],
) -> List[str]:
    current = str(label or "").strip()
    if not current:
        return []
    out = [current]
    out.extend(common_confusions(correction_memory, claim_type="label", canonical_value=current))
    for alias in prompt_alias_candidates(correction_memory, current):
        if ontology is None:
            continue
        mapped = str(getattr(ontology, "synonym_to_canonical", {}).get(str(alias or "").strip().lower(), "") or "").strip()
        if mapped:
            out.append(mapped)
    ontology_labels = [item for item in _ontology_entity_labels(ontology) if str(item).strip() != current]
    if _is_generic_label(current):
        # For generic labels (e.g., object/thing), expose top ontology labels directly
        # so constrained correction can resolve to a concrete class.
        out.extend(_dedupe_tokens(ontology_labels, limit=4))
    else:
        out.extend(_rank_similar_candidates(current, ontology_labels, limit=3))
    return _dedupe_tokens(out, limit=4)


def _relation_candidate_options(
    relation: str,
    *,
    ontology,
    correction_memory: Optional[Dict[str, object]],
) -> List[str]:
    current = str(relation or "").strip()
    if not current:
        return []
    out = [current]
    out.extend(common_confusions(correction_memory, claim_type="relation", canonical_value=current))
    ontology_relations = [item for item in _ontology_relations(ontology) if str(item).strip() != current]
    out.extend(_rank_similar_candidates(current, ontology_relations, limit=3))
    return _dedupe_tokens(out, limit=4)


def _correction_prompt(options: List[str]) -> str:
    rendered = " | ".join(f"'{item}'" for item in options)
    return f" Options: {rendered} | 'uncertain'."


def _counterfactual_relation(
    relation: str,
    *,
    ontology,
    correction_memory: Optional[Dict[str, object]],
) -> str:
    token = str(relation or "").strip()
    if not token:
        return ""
    opposite = {
        "left_of": "right_of",
        "right_of": "left_of",
        "above": "below",
        "below": "above",
        "in_front_of": "behind",
        "behind": "in_front_of",
        "overlapping": "disjoint",
        "intersecting": "disjoint",
        "holding": "near",
    }
    if token in opposite:
        return opposite[token]
    # Fall back to confusion memory / ontology alternatives.
    options = _relation_candidate_options(
        token,
        ontology=ontology,
        correction_memory=correction_memory,
    )
    for item in options:
        alt = str(item or "").strip()
        if alt and alt.lower() != token.lower():
            return alt
    return ""


def _counterfactual_attribute_value(
    slot: str,
    value: str,
    *,
    correction_memory: Optional[Dict[str, object]],
) -> str:
    slot_token = str(slot or "").strip().lower()
    value_token = str(value or "").strip().lower()
    if not slot_token or not value_token:
        return ""
    antonyms = {
        ("state", "sitting"): "standing",
        ("state", "standing"): "sitting",
        ("state", "open"): "closed",
        ("state", "closed"): "open",
        ("state", "on"): "off",
        ("state", "off"): "on",
        ("visibility", "visible"): "occluded",
        ("visibility", "occluded"): "visible",
    }
    if (slot_token, value_token) in antonyms:
        return antonyms[(slot_token, value_token)]
    confusions = common_confusions(
        correction_memory,
        claim_type="attribute",
        canonical_value=value_token,
    )
    for item in confusions:
        alt = str(item or "").strip()
        if alt and alt.lower() != value_token:
            return alt
    return ""


def graph_to_claims(graph: Dict[str, object]) -> List[Claim]:
    out: List[Claim] = []
    snapshot_id = str((graph.get("metadata") or {}).get("graph_snapshot_id") or graph.get("image_id") or "")
    low_node_conf = 0.55
    low_edge_conf = 0.55
    attr_low_conf = 0.60

    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        nid = str(node.get("entity_id", "") or "").strip()
        if not nid:
            continue
        label = _node_label(node).strip()
        prior = _as_confidence(node, ["gemini_confidence", "verify_confidence", "confidence", "score"], default=0.5)
        flags = {str(x).strip().lower() for x in list(node.get("validator_flags") or []) if str(x).strip()}
        uncertain_node = (
            _is_generic_label(label)
            or
            prior < low_node_conf
            or bool(flags.intersection({"cycle_label_conflict", "cycle_existence_conflict", "human_bbox_rejected", "human_label_rejected", "low_confidence_node"}))
        )
        if uncertain_node:
            out.append(
                Claim(
                    claim_id=existence_claim_id(nid),
                    claim_type="existence",
                    subject_id=nid,
                    predicate="exists",
                    value="true",
                    source_graph_snapshot_id=snapshot_id,
                    evidence_node_ids=[nid],
                    prior_score=prior,
                    provenance=[{"source": "scene_graph", "field": "node"}],
                )
            )
            out.append(
                Claim(
                    claim_id=label_claim_id(nid),
                    claim_type="label",
                    subject_id=nid,
                    predicate="label",
                    value=label,
                    source_graph_snapshot_id=snapshot_id,
                    evidence_node_ids=[nid],
                    prior_score=prior,
                    provenance=[{"source": "scene_graph", "field": "canonical_label"}],
                )
            )
        for att in node.get("attributes") or []:
            if not isinstance(att, dict):
                continue
            slot = str(att.get("slot", "") or "").strip()
            value = str(att.get("value", "") or "").strip()
            if not slot or not value:
                continue
            att_conf = _as_confidence(att, ["gemini_confidence", "verify_confidence", "confidence", "score"], default=prior)
            if att_conf >= attr_low_conf and not uncertain_node:
                continue
            out.append(
                Claim(
                    claim_id=attribute_claim_id(nid, slot),
                    claim_type="attribute",
                    subject_id=nid,
                    predicate=slot,
                    value=value,
                    source_graph_snapshot_id=snapshot_id,
                    evidence_node_ids=[nid],
                    prior_score=float(att.get("confidence", 0.35) or 0.35),
                    provenance=[{"source": "scene_graph", "field": "attribute"}],
                )
            )

    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        eid = str(edge.get("edge_id", "") or "").strip()
        if not eid:
            continue
        src = str(edge.get("src_id", "") or "").strip()
        dst = str(edge.get("dst_id", "") or "").strip()
        rel = str(edge.get("relation", "") or "").strip()
        flags = {str(x).strip().lower() for x in list(edge.get("validator_flags") or []) if str(x).strip()}
        edge_prior = _as_confidence(edge, ["gemini_confidence", "verify_confidence", "confidence", "score"], default=0.5)
        uncertain_edge = (
            edge_prior < low_edge_conf
            or bool(flags.intersection({"cycle_relation_conflict", "human_relation_rejected", "low_confidence_edge"}))
        )
        is_spatial_relation = rel.strip().lower() in {
            "left_of",
            "right_of",
            "above",
            "below",
            "in_front_of",
            "behind",
            "overlapping",
            "intersecting",
        }
        if not uncertain_edge and not is_spatial_relation:
            continue
        out.append(
            Claim(
                claim_id=relation_claim_id(eid),
                claim_type="relation",
                subject_id=src,
                predicate=rel,
                object_id=dst,
                source_graph_snapshot_id=snapshot_id,
                evidence_node_ids=[x for x in [src, dst] if x],
                evidence_edge_ids=[eid],
                prior_score=edge_prior,
                provenance=[{"source": "scene_graph", "field": "edge"}],
            )
        )

    return out


def build_single_turn_probes(
    graph: Dict[str, object],
    *,
    correction_memory: Optional[Dict[str, object]] = None,
    ontology=None,
    resolved_claim_ids: Optional[Set[str]] = None,
) -> List[Dict[str, object]]:
    probes: List[Dict[str, object]] = []
    resolved_claims = {
        str(item or "").strip()
        for item in list(resolved_claim_ids or [])
        if str(item or "").strip()
    }
    aliases = _node_alias_map(graph)
    node_by_id: Dict[str, Dict[str, object]] = {}
    # Ask only when scene-graph evidence is sufficiently valid.
    node_min_conf = 0.20
    label_min_conf = 0.25
    relation_min_conf = 0.30
    attribute_node_min_conf = 0.45
    attribute_min_conf = 0.55
    # Sort nodes: persons first, then by confidence descending, so high-priority existence
    # probes are generated (and processed) before generic low-confidence ones.
    sorted_nodes = sorted(
        [n for n in (graph.get("nodes") or []) if isinstance(n, dict)],
        key=lambda n: (
            0 if _node_label(n).strip().lower() == "person" else 1,
            -_as_confidence(n, ["gemini_confidence", "verify_confidence", "verification_confidence", "confidence", "score"], default=0.5),
        ),
    )
    for node in sorted_nodes:
        nid = str(node.get("entity_id", "") or "")
        label = _node_label(node).strip() or "object"
        node_alias = aliases.get(nid, nid)
        if not nid:
            continue
        node_by_id[nid] = node
        node_conf = _as_confidence(
            node,
            [
                "gemini_confidence",
                "verify_confidence",
                "verification_confidence",
                "confidence",
                "score",
            ],
            default=0.5,
        )
        if node_conf < node_min_conf:
            continue
        alias_hint = _alias_guidance(label, correction_memory)
        label_confusion_hint = _confusion_guidance("label", label, correction_memory)
        is_person = label.lower() == "person"
        high_conf = node_conf > 0.5
        # For person nodes and high-confidence nodes, use a direct, unambiguous existence
        # question that explicitly asks about the specific object category.
        if is_person:
            exist_question = (
                "Look carefully at the entire image. "
                "Is there a person (human being) visibly present in this frame? "
                "Answer yes, no, or uncertain."
            )
        elif high_conf:
            exist_question = (
                f"Look carefully at the image. "
                f"Is there a '{label}' visibly present in this frame? "
                f"Answer yes, no, or uncertain."
            )
        else:
            exist_question = (
                f"Is there a visible object for {node_alias} in this frame?"
                f"{alias_hint} Answer yes, no, or uncertain."
            )
        exist_claim_id = existence_claim_id(nid)
        if exist_claim_id not in resolved_claims:
            probes.append(
                {
                    "probe_id": f"probe_exist_{nid}",
                    "probe_type": "single_turn",
                    "question": exist_question,
                    "target_claim_id": exist_claim_id,
                    "evidence_node_ids": [nid],
                    "expected_answer": "yes",
                }
            )
        label_claim = label_claim_id(nid)
        if (not _is_generic_label(label)) and node_conf >= label_min_conf:
            if label_claim not in resolved_claims:
                probes.append(
                    {
                        "probe_id": f"probe_label_{nid}",
                        "probe_type": "single_turn",
                        "probe_family": "binary_verification",
                        "question": (
                            f"Is {node_alias} best described as the canonical label '{label}'?"
                            f"{alias_hint}{label_confusion_hint} Answer yes, no, or uncertain and briefly explain."
                        ),
                        "target_claim_id": label_claim,
                        "evidence_node_ids": [nid],
                        "expected_answer": "yes",
                    }
                )
            label_options = _label_candidate_options(
                label,
                ontology=ontology,
                correction_memory=correction_memory,
            )
            label_alt = ""
            for item in label_options:
                alt = str(item or "").strip()
                if alt and alt.lower() != label.lower():
                    label_alt = alt
                    break
            if label_alt and label_claim not in resolved_claims:
                probes.append(
                    {
                        "probe_id": f"probe_label_neg_{nid}",
                        "probe_type": "single_turn",
                        "probe_family": "counterfactual_verification",
                        "question": (
                            f"Counterfactual check: is {node_alias} better described as '{label_alt}' "
                            f"instead of '{label}'? Answer yes, no, or uncertain."
                        ),
                        "target_claim_id": label_claim,
                        "evidence_node_ids": [nid],
                        "expected_answer": "no",
                    }
                )
        if node_conf >= label_min_conf:
            label_options = _label_candidate_options(
                label,
                ontology=ontology,
                correction_memory=correction_memory,
            )
            if len(label_options) > 1 and label_claim not in resolved_claims:
                probes.append(
                    {
                        "probe_id": f"probe_label_fix_{nid}",
                        "probe_type": "single_turn_correction",
                        "probe_family": "constrained_correction",
                        "question": (
                            f"If the current canonical label '{label}' for {node_alias} is wrong, which canonical label best fits it?"
                            f"{_correction_prompt(label_options)} Answer with one option only."
                        ),
                        "target_claim_id": label_claim,
                        "evidence_node_ids": [nid],
                        "expected_answer": label,
                        "candidate_options": label_options,
                        "response_format": {
                            "type": "selection",
                            "options": label_options,
                            "default_selection": label,
                        },
                    }
                )
        for att in node.get("attributes") or []:
            if not isinstance(att, dict):
                continue
            if node_conf < attribute_node_min_conf:
                continue
            slot = str(att.get("slot", "") or "").strip()
            value = str(att.get("value", "") or "").strip()
            if not slot or not value:
                continue
            if _is_placeholder_value(slot) or _is_placeholder_value(value):
                continue
            attr_conf = _as_confidence(
                att,
                ["gemini_confidence", "verify_confidence", "confidence", "score"],
                default=node_conf,
            )
            if attr_conf < attribute_min_conf:
                continue
            attr_claim = attribute_claim_id(nid, slot)
            if attr_claim in resolved_claims:
                continue
            probes.append(
                {
                    "probe_id": f"probe_attr_{nid}_{_tokenize(slot)}",
                    "probe_type": "single_turn",
                    "probe_family": "binary_verification",
                    "question": (
                        f"Is {node_alias} characterized by {slot}='{value}'? "
                        "Answer yes, no, or uncertain and briefly explain."
                    ),
                    "target_claim_id": attr_claim,
                    "evidence_node_ids": [nid],
                    "expected_answer": "yes",
                }
            )
            alt_value = _counterfactual_attribute_value(
                slot,
                value,
                correction_memory=correction_memory,
            )
            if alt_value:
                probes.append(
                    {
                        "probe_id": f"probe_attr_neg_{nid}_{_tokenize(slot)}",
                        "probe_type": "single_turn",
                        "probe_family": "counterfactual_verification",
                        "question": (
                            f"Counterfactual check: is {node_alias} characterized by {slot}='{alt_value}' "
                            f"instead of {slot}='{value}'? Answer yes, no, or uncertain."
                        ),
                        "target_claim_id": attr_claim,
                        "evidence_node_ids": [nid],
                        "expected_answer": "no",
                    }
                )

    relation_src_count: Dict[tuple[str, str], int] = {}
    relation_dst_count: Dict[tuple[str, str], int] = {}
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        src = str(edge.get("src_id", "") or "").strip()
        dst = str(edge.get("dst_id", "") or "").strip()
        rel = str(edge.get("relation", "") or "").strip()
        if not src or not dst or not rel:
            continue
        relation_src_count[(rel, src)] = int(relation_src_count.get((rel, src), 0) + 1)
        relation_dst_count[(rel, dst)] = int(relation_dst_count.get((rel, dst), 0) + 1)

    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        eid = _edge_identity(edge)
        src = str(edge.get("src_id", "") or "")
        dst = str(edge.get("dst_id", "") or "")
        rel = str(edge.get("relation", "") or "")
        if not eid or not src or not dst or not rel:
            continue
        if _is_placeholder_value(rel):
            continue
        # Single-turn relation checks are reserved for mostly one-to-one relations.
        if relation_src_count.get((rel, src), 0) > 1 or relation_dst_count.get((rel, dst), 0) > 1:
            continue
        src_node = node_by_id.get(src, {})
        dst_node = node_by_id.get(dst, {})
        src_conf = _as_confidence(
            src_node,
            [
                "gemini_confidence",
                "verify_confidence",
                "verification_confidence",
                "confidence",
                "score",
            ],
            default=0.5,
        )
        dst_conf = _as_confidence(
            dst_node,
            [
                "gemini_confidence",
                "verify_confidence",
                "verification_confidence",
                "confidence",
                "score",
            ],
            default=0.5,
        )
        edge_conf = _as_confidence(
            edge,
            ["gemini_confidence", "verify_confidence", "confidence", "score"],
            default=0.5,
        )
        if edge_conf < relation_min_conf or src_conf < node_min_conf or dst_conf < node_min_conf:
            continue
        relation_claim = relation_claim_id(eid)
        if relation_claim in resolved_claims:
            continue
        src_name = aliases.get(src, src)
        dst_name = aliases.get(dst, dst)
        relation_confusion_hint = _confusion_guidance("relation", rel, correction_memory)
        probes.append(
            {
                "probe_id": f"probe_rel_{eid}",
                "probe_type": "single_turn",
                "probe_family": "binary_verification",
                "question": (
                    f"Does {src_name} stand in relation '{rel}' to {dst_name}? "
                    f"{relation_confusion_hint} Answer yes, no, or uncertain and explain using visible evidence."
                ),
                "target_claim_id": relation_claim,
                "evidence_node_ids": [src, dst],
                "evidence_edge_ids": [eid],
                "expected_answer": "yes",
            }
        )
        neg_rel = _counterfactual_relation(
            rel,
            ontology=ontology,
            correction_memory=correction_memory,
        )
        if neg_rel and neg_rel.lower() != rel.lower():
            probes.append(
                {
                    "probe_id": f"probe_rel_neg_{eid}",
                    "probe_type": "single_turn",
                    "probe_family": "counterfactual_verification",
                    "question": (
                        f"Counterfactual check: between {src_name} and {dst_name}, is relation '{neg_rel}' true "
                        f"instead of '{rel}'? Answer yes, no, or uncertain."
                    ),
                    "target_claim_id": relation_claim,
                    "evidence_node_ids": [src, dst],
                    "evidence_edge_ids": [eid],
                    "expected_answer": "no",
                }
            )
        relation_options = _relation_candidate_options(
            rel,
            ontology=ontology,
            correction_memory=correction_memory,
        )
        if len(relation_options) > 1:
            probes.append(
                {
                    "probe_id": f"probe_rel_fix_{eid}",
                    "probe_type": "single_turn_correction",
                    "probe_family": "constrained_correction",
                    "question": (
                        f"If relation '{rel}' is not the best canonical relation between {src_name} and {dst_name}, "
                        f"which canonical relation should replace it?{_correction_prompt(relation_options)} "
                        "Answer with one option only."
                    ),
                    "target_claim_id": relation_claim,
                    "evidence_node_ids": [src, dst],
                    "evidence_edge_ids": [eid],
                    "expected_answer": rel,
                    "candidate_options": relation_options,
                    "response_format": {
                        "type": "selection",
                        "options": relation_options,
                        "default_selection": rel,
                    },
                }
            )
    return probes


def build_multi_turn_probes(
    graph: Dict[str, object],
    max_chains: int = 24,
    *,
    enable_temporal_context: bool = True,
    correction_memory: Optional[Dict[str, object]] = None,
    ontology=None,
    resolved_claim_ids: Optional[Set[str]] = None,
) -> List[Dict[str, object]]:
    probes: List[Dict[str, object]] = []
    resolved_claims = {
        str(item or "").strip()
        for item in list(resolved_claim_ids or [])
        if str(item or "").strip()
    }
    edges = graph.get("edges") or []
    aliases = _node_alias_map(graph)
    node_by_id: Dict[str, Dict[str, object]] = {
        str(n.get("entity_id", "") or "").strip(): n
        for n in (graph.get("nodes") or [])
        if isinstance(n, dict) and str(n.get("entity_id", "") or "").strip()
    }
    node_min_conf = 0.35
    label_min_conf = 0.45
    relation_min_conf = 0.45
    attribute_node_min_conf = 0.72
    attribute_min_conf = 0.75

    chain_count = 0
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        if chain_count >= max(1, int(max_chains)):
            break
        nid = str(node.get("entity_id", "") or "")
        label = _node_label(node).strip() or "object"
        node_alias = aliases.get(nid, nid)
        if not nid:
            continue
        node_conf = _as_confidence(
            node,
            [
                "gemini_confidence",
                "verify_confidence",
                "verification_confidence",
                "confidence",
                "score",
            ],
            default=0.5,
        )
        if node_conf < node_min_conf:
            continue
        chain_id = f"chain_{nid}"
        alias_hint = _alias_guidance(label, correction_memory)
        label_confusion_hint = _confusion_guidance("label", label, correction_memory)
        history = _track_history(graph, nid) if bool(enable_temporal_context) else {}
        turn_idx = 1
        chain_probe_start = len(probes)
        previous_frame_idx = history.get("previous_frame_idx")
        previous_label = str(history.get("previous_label", "") or "").strip()
        seen_frames = _seen_frames_phrase(history.get("seen_frames"))
        if previous_frame_idx is not None:
            exist_claim_id = existence_claim_id(nid)
            if exist_claim_id not in resolved_claims:
                continuity_suffix = f" This track has also appeared in sampled frames {seen_frames}." if seen_frames else ""
                probes.append(
                    {
                        "probe_id": f"{chain_id}_turn{turn_idx}",
                        "probe_type": "multi_turn",
                        "probe_family": "temporal_consistency",
                        "chain_id": chain_id,
                        "turn": turn_idx,
                        "question": (
                            "We will keep talking about the same tracked object across sampled frames. "
                            f"In {_frame_phrase(previous_frame_idx)}, a matching track was already visible."
                            f"{continuity_suffix} Does the current frame still show the same persistent object track rather than a mismatched box? "
                            "Answer yes, no, or uncertain."
                        ),
                        "target_claim_id": exist_claim_id,
                        "evidence_node_ids": [nid],
                        "expected_answer": "yes",
                        "temporal_anchor_frame_idx": previous_frame_idx,
                    }
                )
                turn_idx += 1
        label_claim = label_claim_id(nid)
        if (
            previous_frame_idx is not None
            and previous_label
            and node_conf >= label_min_conf
            and (not _is_generic_label(label))
            and (not _is_generic_label(previous_label))
            and label_claim not in resolved_claims
        ):
            probes.append(
                {
                    "probe_id": f"{chain_id}_turn{turn_idx}",
                    "probe_type": "multi_turn",
                    "probe_family": "temporal_consistency",
                    "chain_id": chain_id,
                    "turn": turn_idx,
                    "question": (
                        f"Still referring to the same tracked object ({node_alias}), {_frame_phrase(previous_frame_idx)} labeled it as '{previous_label}'. "
                        f"In the current frame, is '{label}' still the best canonical label for that same track?"
                        f"{alias_hint}{label_confusion_hint} Answer yes, no, or uncertain."
                    ),
                    "target_claim_id": label_claim,
                    "evidence_node_ids": [nid],
                    "expected_answer": "yes",
                    "temporal_anchor_frame_idx": previous_frame_idx,
                }
            )
            turn_idx += 1
        if (not _is_generic_label(label)) and node_conf >= label_min_conf and label_claim not in resolved_claims:
            probes.append(
                {
                    "probe_id": f"{chain_id}_turn{turn_idx}",
                    "probe_type": "multi_turn",
                    "probe_family": "binary_verification",
                    "chain_id": chain_id,
                    "turn": turn_idx,
                    "question": (
                        f"We will keep talking about the same highlighted object ({node_alias}). "
                        f"Is it correctly labeled as the canonical label '{label}'?"
                        f"{alias_hint}{label_confusion_hint} Answer yes, no, or uncertain."
                    ),
                    "target_claim_id": label_claim,
                    "evidence_node_ids": [nid],
                    "expected_answer": "yes",
                }
            )
            turn_idx += 1
            label_options = _label_candidate_options(
                label,
                ontology=ontology,
                correction_memory=correction_memory,
            )
            if len(label_options) > 1:
                label_alt = ""
                for item in label_options:
                    alt = str(item or "").strip()
                    if alt and alt.lower() != label.lower():
                        label_alt = alt
                        break
                if label_alt:
                    probes.append(
                        {
                            "probe_id": f"{chain_id}_turn{turn_idx}",
                            "probe_type": "multi_turn",
                            "probe_family": "counterfactual_verification",
                            "chain_id": chain_id,
                            "turn": turn_idx,
                            "question": (
                                f"Counterfactual check for the same tracked object ({node_alias}): is label '{label_alt}' "
                                f"more appropriate than '{label}'? Answer yes, no, or uncertain."
                            ),
                            "target_claim_id": label_claim,
                            "evidence_node_ids": [nid],
                            "expected_answer": "no",
                        }
                    )
                    turn_idx += 1
                probes.append(
                    {
                        "probe_id": f"{chain_id}_turn{turn_idx}",
                        "probe_type": "multi_turn_correction",
                        "probe_family": "constrained_correction",
                        "chain_id": chain_id,
                        "turn": turn_idx,
                        "question": (
                            f"Still referring to {node_alias}, if '{label}' is not correct, "
                            f"which canonical label should replace it?{_correction_prompt(label_options)} "
                            "Answer with one option only."
                        ),
                        "target_claim_id": label_claim,
                        "evidence_node_ids": [nid],
                        "expected_answer": label,
                        "candidate_options": label_options,
                        "response_format": {
                            "type": "selection",
                            "options": label_options,
                            "default_selection": label,
                        },
                    }
                )
                turn_idx += 1
        attrs = [x for x in (node.get("attributes") or []) if isinstance(x, dict)]
        if attrs and node_conf >= attribute_node_min_conf:
            attrs_sorted = sorted(
                attrs,
                key=lambda a: _as_confidence(
                    a,
                    ["gemini_confidence", "verify_confidence", "confidence", "score"],
                    default=0.0,
                ),
                reverse=True,
            )
            for top_attr in attrs_sorted:
                slot = str(top_attr.get("slot", "") or "").strip()
                value = str(top_attr.get("value", "") or "").strip()
                if not slot or not value:
                    continue
                if _is_placeholder_value(slot) or _is_placeholder_value(value):
                    continue
                attr_conf = _as_confidence(
                    top_attr,
                    ["gemini_confidence", "verify_confidence", "confidence", "score"],
                    default=node_conf,
                )
                if attr_conf < attribute_min_conf:
                    continue
                attr_claim = attribute_claim_id(nid, slot)
                if attr_claim in resolved_claims:
                    continue
                probes.append(
                    {
                        "probe_id": f"{chain_id}_turn{turn_idx}",
                        "probe_type": "multi_turn",
                        "probe_family": "binary_verification",
                        "chain_id": chain_id,
                        "turn": turn_idx,
                        "question": (
                            f"Still referring to {node_alias}, is its {slot} equal to '{value}'? "
                            "Answer yes, no, or uncertain."
                        ),
                        "target_claim_id": attr_claim,
                        "evidence_node_ids": [nid],
                        "expected_answer": "yes",
                    }
                )
                turn_idx += 1
                break
        rel_edge = None
        rel_edge_conf = 0.0
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("src_id", "") or "")
            dst = str(edge.get("dst_id", "") or "")
            rel = str(edge.get("relation", "") or "").strip()
            if _is_placeholder_value(rel):
                continue
            if src != nid and dst != nid:
                continue
            src_conf = _as_confidence(
                node_by_id.get(src, {}),
                [
                    "gemini_confidence",
                    "verify_confidence",
                    "verification_confidence",
                    "confidence",
                    "score",
                ],
                default=0.5,
            )
            dst_conf = _as_confidence(
                node_by_id.get(dst, {}),
                [
                    "gemini_confidence",
                    "verify_confidence",
                    "verification_confidence",
                    "confidence",
                    "score",
                ],
                default=0.5,
            )
            edge_conf = _as_confidence(
                edge,
                ["gemini_confidence", "verify_confidence", "confidence", "score"],
                default=0.5,
            )
            if edge_conf < relation_min_conf or src_conf < node_min_conf or dst_conf < node_min_conf:
                continue
            if rel_edge is None or edge_conf > rel_edge_conf:
                rel_edge = edge
                rel_edge_conf = edge_conf
        if rel_edge is not None:
            eid = _edge_identity(rel_edge)
            rel = str(rel_edge.get("relation", "") or "")
            src = str(rel_edge.get("src_id", "") or "")
            dst = str(rel_edge.get("dst_id", "") or "")
            if eid and rel and src and dst:
                relation_claim = relation_claim_id(eid)
                if relation_claim in resolved_claims:
                    continue
                src_name = aliases.get(src, src)
                dst_name = aliases.get(dst, dst)
                relation_confusion_hint = _confusion_guidance("relation", rel, correction_memory)
                probes.append(
                    {
                        "probe_id": f"{chain_id}_turn{turn_idx}",
                        "probe_type": "multi_turn",
                        "probe_family": "binary_verification",
                        "chain_id": chain_id,
                        "turn": turn_idx,
                        "question": (
                            f"Still referring to the same scene context, does relation '{rel}' hold "
                            f"between {src_name} and {dst_name}?{relation_confusion_hint} Answer yes, no, or uncertain."
                        ),
                        "target_claim_id": relation_claim,
                        "evidence_node_ids": [src, dst],
                        "evidence_edge_ids": [eid],
                        "expected_answer": "yes",
                    }
                )
                turn_idx += 1
                neg_rel = _counterfactual_relation(
                    rel,
                    ontology=ontology,
                    correction_memory=correction_memory,
                )
                if neg_rel and neg_rel.lower() != rel.lower():
                    probes.append(
                        {
                            "probe_id": f"{chain_id}_turn{turn_idx}",
                            "probe_type": "multi_turn",
                            "probe_family": "counterfactual_verification",
                            "chain_id": chain_id,
                            "turn": turn_idx,
                            "question": (
                                f"Counterfactual check in the same scene context: is relation '{neg_rel}' "
                                f"between {src_name} and {dst_name} true instead of '{rel}'? Answer yes, no, or uncertain."
                            ),
                            "target_claim_id": relation_claim,
                            "evidence_node_ids": [src, dst],
                            "evidence_edge_ids": [eid],
                            "expected_answer": "no",
                        }
                    )
                    turn_idx += 1
                rel_history = _relation_history(graph, src, rel, dst) if bool(enable_temporal_context) else {}
                rel_previous_frame_idx = rel_history.get("previous_frame_idx")
                if rel_previous_frame_idx is not None:
                    probes.append(
                        {
                            "probe_id": f"{chain_id}_turn{turn_idx}",
                            "probe_type": "multi_turn",
                            "probe_family": "temporal_consistency",
                            "chain_id": chain_id,
                            "turn": turn_idx,
                            "question": (
                                f"Across sampled frames, relation '{rel}' between {src_name} and {dst_name} was also present in {_frame_phrase(rel_previous_frame_idx)}. "
                                "Does that same relation still hold now? Answer yes, no, or uncertain."
                            ),
                            "target_claim_id": relation_claim,
                            "evidence_node_ids": [src, dst],
                            "evidence_edge_ids": [eid],
                            "expected_answer": "yes",
                            "temporal_anchor_frame_idx": rel_previous_frame_idx,
                        }
                    )
                    turn_idx += 1
                relation_options = _relation_candidate_options(
                    rel,
                    ontology=ontology,
                    correction_memory=correction_memory,
                )
                if len(relation_options) > 1:
                    probes.append(
                        {
                            "probe_id": f"{chain_id}_turn{turn_idx}",
                            "probe_type": "multi_turn_correction",
                            "probe_family": "constrained_correction",
                            "chain_id": chain_id,
                            "turn": turn_idx,
                            "question": (
                                f"Still referring to that same relation context, if '{rel}' is not correct, "
                                f"which canonical relation should replace it?{_correction_prompt(relation_options)} "
                                "Answer with one option only."
                            ),
                            "target_claim_id": relation_claim,
                            "evidence_node_ids": [src, dst],
                            "evidence_edge_ids": [eid],
                            "expected_answer": rel,
                            "candidate_options": relation_options,
                            "response_format": {
                                "type": "selection",
                                "options": relation_options,
                                "default_selection": rel,
                            },
                        }
                    )
        if len(probes) > chain_probe_start:
            chain_count += 1

    # Add compact many-to-many relation checks to cover ambiguous relation groups with minimal turns.
    relation_groups: Dict[str, List[Dict[str, object]]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        rel = str(edge.get("relation", "") or "").strip()
        src = str(edge.get("src_id", "") or "").strip()
        dst = str(edge.get("dst_id", "") or "").strip()
        eid = _edge_identity(edge).strip()
        if not rel or not src or not dst or not eid or _is_placeholder_value(rel):
            continue
        src_conf = _as_confidence(node_by_id.get(src, {}), ["gemini_confidence", "verify_confidence", "verification_confidence", "confidence", "score"], default=0.5)
        dst_conf = _as_confidence(node_by_id.get(dst, {}), ["gemini_confidence", "verify_confidence", "verification_confidence", "confidence", "score"], default=0.5)
        edge_conf = _as_confidence(edge, ["gemini_confidence", "verify_confidence", "confidence", "score"], default=0.5)
        if edge_conf < relation_min_conf or src_conf < node_min_conf or dst_conf < node_min_conf:
            continue
        if relation_claim_id(eid) in resolved_claims:
            continue
        relation_groups.setdefault(rel, []).append(edge)

    group_idx = 0
    for rel, group_edges in relation_groups.items():
        src_set = {str(e.get("src_id", "") or "").strip() for e in group_edges}
        dst_set = {str(e.get("dst_id", "") or "").strip() for e in group_edges}
        if len(src_set) <= 1 and len(dst_set) <= 1:
            continue
        group_idx += 1
        samples = group_edges[: min(6, len(group_edges))]
        pair_text = ", ".join(
            [
                f"{aliases.get(str(e.get('src_id', '') or '').strip(), str(e.get('src_id', '') or '').strip())} -> "
                f"{aliases.get(str(e.get('dst_id', '') or '').strip(), str(e.get('dst_id', '') or '').strip())}"
                for e in samples
            ]
        )
        evidence_nodes: List[str] = []
        evidence_edges: List[str] = []
        for e in samples:
            eid = str(e.get("edge_id", "") or "").strip()
            src = str(e.get("src_id", "") or "").strip()
            dst = str(e.get("dst_id", "") or "").strip()
            if eid and eid not in evidence_edges:
                evidence_edges.append(eid)
            for nid in (src, dst):
                if nid and nid not in evidence_nodes:
                    evidence_nodes.append(nid)
        probes.append(
            {
                "probe_id": f"group_rel_{_tokenize(rel)}_{group_idx}",
                "probe_type": "multi_turn",
                "probe_family": "group_relation_verification",
                "chain_id": f"group_chain_{_tokenize(rel)}_{group_idx}",
                "turn": 1,
                "question": (
                    f"For relation '{rel}', check these pairs in one pass: {pair_text}. "
                    "Are these relation claims jointly consistent in the current frame? Answer yes, no, or uncertain."
                ),
                "target_claim_id": relation_claim_id(str(samples[0].get("edge_id", "") or "").strip()),
                "evidence_node_ids": evidence_nodes,
                "evidence_edge_ids": evidence_edges,
                "expected_answer": "yes",
            }
        )

    return probes
