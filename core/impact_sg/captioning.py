from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Optional

from .claim_graph import attribute_claim_id, existence_claim_id, label_claim_id, relation_claim_id
from .correction_memory import common_confusions, prompt_alias_candidates


def _strip_code_fence(text: object) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
    return raw


def _extract_json_object(text: object) -> Optional[Dict[str, object]]:
    raw = _strip_code_fence(text)
    if not raw:
        return None
    candidates = [raw]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_token(value: object) -> str:
    return str(value or "").strip().lower()


def _node_label(node: Dict[str, object]) -> str:
    canonical = str(node.get("canonical_label", "") or "").strip()
    if canonical:
        return canonical
    fallback = str(node.get("label", "") or "").strip()
    return fallback or "object"


def _edge_identity(edge: Dict[str, object]) -> str:
    edge_id = str(edge.get("edge_id", "") or "").strip()
    if edge_id:
        return edge_id
    src_id = str(edge.get("src_id", "") or "").strip()
    relation = str(edge.get("relation", "") or "").strip()
    dst_id = str(edge.get("dst_id", "") or "").strip()
    if src_id and relation and dst_id:
        safe = re.sub(r"[^a-z0-9]+", "_", f"{src_id}_{relation}_{dst_id}".lower()).strip("_")
        return f"edge_{safe}" if safe else ""
    return ""


def _label_candidates(
    label: str,
    ontology,
    correction_memory: Optional[Dict[str, object]] = None,
) -> List[str]:
    out: List[str] = []
    token = _normalize_token(label)
    if token:
        out.append(token)
    if ontology is not None:
        aliases = getattr(ontology, "canonical_to_synonyms", {}).get(token, [])
        for item in aliases:
            alias = _normalize_token(item)
            if alias and alias not in out:
                out.append(alias)
    for item in prompt_alias_candidates(correction_memory, label):
        alias = _normalize_token(item)
        if alias and alias not in out:
            out.append(alias)
    return out


def _canonical_entity_labels(ontology) -> List[str]:
    out: List[str] = []
    if ontology is None:
        return out
    for row in getattr(ontology, "canonical_entities", []) or []:
        label = str((row or {}).get("label", "") or "").strip()
        if label and label not in out:
            out.append(label)
    return out


def _canonical_relation_labels(ontology) -> List[str]:
    out: List[str] = []
    if ontology is None:
        return out
    for values in (getattr(ontology, "relation_vocabulary", {}) or {}).values():
        for value in list(values or []):
            token = str(value or "").strip()
            if token and token not in out:
                out.append(token)
    return out


def _node_lookup(graph: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {
        str(node.get("entity_id", "") or "").strip(): node
        for node in graph.get("nodes") or []
        if str(node.get("entity_id", "") or "").strip()
    }


def _edge_lookup(graph: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return {
        str(edge.get("edge_id", "") or "").strip(): edge
        for edge in graph.get("edges") or []
        if str(edge.get("edge_id", "") or "").strip()
    }


def _unique_label_map(graph: Dict[str, object]) -> Dict[str, str]:
    buckets: Dict[str, List[str]] = {}
    for node in graph.get("nodes") or []:
        entity_id = str(node.get("entity_id", "") or "").strip()
        label = _normalize_token(_node_label(node))
        if not entity_id or not label:
            continue
        buckets.setdefault(label, []).append(entity_id)
    return {
        label: ids[0]
        for label, ids in buckets.items()
        if len(ids) == 1 and ids[0]
    }


def _unique_relation_map(graph: Dict[str, object]) -> Dict[str, str]:
    buckets: Dict[str, List[str]] = {}
    for edge in graph.get("edges") or []:
        edge_id = str(edge.get("edge_id", "") or "").strip()
        relation = _normalize_token(edge.get("relation"))
        if not edge_id or not relation:
            continue
        buckets.setdefault(relation, []).append(edge_id)
        spaced = relation.replace("_", " ")
        if spaced and spaced != relation:
            buckets.setdefault(spaced, []).append(edge_id)
    return {
        relation: ids[0]
        for relation, ids in buckets.items()
        if len(ids) == 1 and ids[0]
    }


def _resolve_entity_id(value: object, graph: Dict[str, object]) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    node_by_id = _node_lookup(graph)
    if token in node_by_id:
        return token
    label_map = _unique_label_map(graph)
    return label_map.get(token.lower(), "")


def _resolve_relation_id(value: object, graph: Dict[str, object]) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    edge_by_id = _edge_lookup(graph)
    if token in edge_by_id:
        return token
    relation_map = _unique_relation_map(graph)
    return relation_map.get(token.lower(), "")


def _normalize_attribute_rows(rows: object, graph: Dict[str, object]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for row in list(rows or []):
        if isinstance(row, dict):
            entity_id = _resolve_entity_id(row.get("entity_id"), graph)
            slot = str(row.get("slot", "") or "").strip()
            value = str(row.get("value", "") or "").strip()
        else:
            entity_id = ""
            slot = ""
            value = ""
            text = str(row or "").strip()
            match = re.match(r"^([^:]+):([^=]+)=(.+)$", text)
            if match:
                entity_id = _resolve_entity_id(match.group(1), graph)
                slot = match.group(2).strip()
                value = match.group(3).strip()
        key = (entity_id, slot.lower(), value.lower())
        if not entity_id or not slot or not value or key in seen:
            continue
        seen.add(key)
        out.append({"entity_id": entity_id, "slot": slot, "value": value})
    return out


def _normalize_entity_rows(rows: object, graph: Dict[str, object]) -> List[str]:
    out: List[str] = []
    for row in list(rows or []):
        if isinstance(row, dict):
            entity_id = _resolve_entity_id(row.get("entity_id") or row.get("id"), graph)
        else:
            entity_id = _resolve_entity_id(row, graph)
        if entity_id and entity_id not in out:
            out.append(entity_id)
    return out


def _normalize_relation_rows(rows: object, graph: Dict[str, object]) -> List[str]:
    out: List[str] = []
    for row in list(rows or []):
        if isinstance(row, dict):
            edge_id = _resolve_relation_id(row.get("edge_id") or row.get("id") or row.get("relation"), graph)
        else:
            edge_id = _resolve_relation_id(row, graph)
        if edge_id and edge_id not in out:
            out.append(edge_id)
    return out


def build_caption_prompt(
    graph: Dict[str, object],
    *,
    style: str = "technical",
    max_sentences: int = 4,
    require_relation_mentions: bool = True,
    correction_memory: Optional[Dict[str, object]] = None,
    structured_feedback: bool = True,
) -> str:
    node_lines: List[str] = []
    attribute_lines: List[str] = []
    relation_lines: List[str] = []
    naming_lines: List[str] = []

    for node in graph.get("nodes") or []:
        entity_id = str(node.get("entity_id", "") or "").strip()
        label = _node_label(node).strip()
        if not entity_id or not label:
            continue
        node_lines.append(
            f"- {entity_id} | label={label} | bbox={list(node.get('bbox') or [0, 0, 0, 0])}"
        )
        aliases = prompt_alias_candidates(correction_memory, label)
        confusions = common_confusions(
            correction_memory,
            claim_type="label",
            canonical_value=label,
        )
        hints: List[str] = []
        if aliases:
            hints.append("aliases: " + ", ".join(aliases))
        if confusions:
            hints.append("common confusions: " + ", ".join(confusions))
        if hints:
            naming_lines.append(f"- {label}: " + "; ".join(hints))
        for att in node.get("attributes") or []:
            if not isinstance(att, dict):
                continue
            slot = str(att.get("slot", "") or "").strip()
            value = str(att.get("value", "") or "").strip()
            if not slot or not value:
                continue
            attribute_lines.append(
                f"- {attribute_claim_id(entity_id, slot)} | entity={entity_id} | slot={slot} | value={value}"
            )

    for edge in graph.get("edges") or []:
        edge_id = _edge_identity(edge).strip()
        src_id = str(edge.get("src_id", "") or "").strip()
        relation = str(edge.get("relation", "") or "").strip()
        dst_id = str(edge.get("dst_id", "") or "").strip()
        if not edge_id or not src_id or not relation or not dst_id:
            continue
        relation_lines.append(
            f"- {edge_id} | src={src_id} | relation={relation} | dst={dst_id} | claim={relation_claim_id(edge_id)}"
        )

    relation_req = "- mention important listed relations in the caption when they are visually supported\n" if require_relation_mentions else ""
    guidance_block = ""
    if naming_lines:
        guidance_block = (
            "Canonical naming hints:\n"
            "- keep wording canonical and prefer the listed labels over free-form paraphrases\n"
            + "\n".join(naming_lines)
            + "\n\n"
        )

    if structured_feedback:
        return (
            "You are the Captioning agent in a multi-agent scene graph verification loop.\n"
            "Your job is not to invent a new scene graph. Your job is to write a constrained caption and report which existing graph claims are visually supported or contradicted.\n"
            "Procedure:\n"
            "1. Read the listed node, attribute, and edge IDs.\n"
            "2. Write a concise caption grounded only in those listed items.\n"
            "3. Mark which listed entities, attributes, and relations are supported by visible evidence.\n"
            "4. Mark listed items that appear unsupported. Do not invent new graph structure.\n"
            "5. If you mention an entity that is not listed, add it under hallucinated_mentions instead of treating it as graph evidence.\n\n"
            f"Write a {style} caption of at most {max(1, int(max_sentences))} sentences.\n"
            "Requirements:\n"
            "- mention only visible entities supported by the listed graph items\n"
            f"{relation_req}"
            "- do not invent unseen objects\n"
            "- use the listed entity IDs, attribute slots, and edge IDs in the JSON fields\n\n"
            + guidance_block
            + "Nodes:\n"
            + ("\n".join(node_lines) if node_lines else "- none")
            + "\n\nAttributes:\n"
            + ("\n".join(attribute_lines) if attribute_lines else "- none")
            + "\n\nEdges:\n"
            + ("\n".join(relation_lines) if relation_lines else "- none")
            + "\n\nAnswer the task according to the provided JSON schema."
        )

    return (
        "You are given a scene graph for one frame.\n"
        f"Write a {style} caption of at most {max(1, int(max_sentences))} sentences.\n"
        "Requirements:\n"
        "- mention only visible entities supported by the graph\n"
        f"{relation_req}"
        "- do not invent unseen objects\n"
        "- keep wording canonical when possible\n\n"
        + guidance_block
        + "Nodes:\n"
        + "\n".join(node_lines)
        + "\n\nEdges:\n"
        + "\n".join(relation_lines)
    )


def build_graph_caption_preview(
    graph: Dict[str, object],
    *,
    style: str = "technical",
    max_sentences: int = 4,
    require_relation_mentions: bool = True,
) -> str:
    nodes = list(graph.get("nodes") or [])
    edges = list(graph.get("edges") or [])
    labels = [_node_label(node).strip() for node in nodes]
    label_counts = Counter([label for label in labels if label])
    entity_summary = ", ".join(f"{label} x{count}" for label, count in label_counts.most_common(4)) or "no confirmed entities"

    attribute_fragments: List[str] = []
    for node in nodes[:4]:
        label = _node_label(node).strip()
        attrs = [x for x in (node.get("attributes") or []) if isinstance(x, dict)]
        if not attrs:
            continue
        pieces: List[str] = []
        for att in attrs[:2]:
            slot = str(att.get("slot", "") or "").strip()
            value = str(att.get("value", "") or "").strip()
            if slot and value:
                pieces.append(f"{slot}={value}")
        if pieces:
            attribute_fragments.append(f"{label} ({', '.join(pieces)})")

    relation_fragments: List[str] = []
    for edge in edges[:4]:
        src = str(edge.get("src_id", "") or "").strip()
        rel = str(edge.get("relation", "") or "").strip().replace("_", " ")
        dst = str(edge.get("dst_id", "") or "").strip()
        if src and rel and dst:
            relation_fragments.append(f"{src} {rel} {dst}")

    sentences: List[str] = []
    tone = str(style or "technical").strip().lower()
    if tone.startswith("concise"):
        sentences.append(f"Visible entities: {entity_summary}.")
        if require_relation_mentions and relation_fragments:
            sentences.append("Key relations: " + "; ".join(relation_fragments[:2]) + ".")
    elif tone.startswith("detailed"):
        sentences.append(f"The scene contains {entity_summary}.")
        if attribute_fragments:
            sentences.append("Observed attributes include " + "; ".join(attribute_fragments[:3]) + ".")
        if require_relation_mentions and relation_fragments:
            sentences.append("Relations visible in the graph include " + "; ".join(relation_fragments[:3]) + ".")
        sentences.append(f"The current graph contains {len(nodes)} nodes and {len(edges)} edges.")
    else:
        sentences.append(f"Frame graph summary: {entity_summary}.")
        if attribute_fragments:
            sentences.append("Attribute coverage: " + "; ".join(attribute_fragments[:3]) + ".")
        if require_relation_mentions and relation_fragments:
            sentences.append("Relation coverage: " + "; ".join(relation_fragments[:3]) + ".")
        sentences.append(f"Graph size: {len(nodes)} nodes, {len(edges)} edges.")

    limit = max(1, int(max_sentences or 1))
    return " ".join(sentences[:limit]).strip()


def _fallback_caption_votes(
    caption_text: str,
    graph: Dict[str, object],
    ontology,
    correction_memory: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    text = str(caption_text or "").strip().lower()
    votes: List[Dict[str, object]] = []
    if not text:
        return votes

    for node in graph.get("nodes") or []:
        nid = str(node.get("entity_id", "") or "")
        label = _node_label(node).strip().lower()
        if not nid or not label:
            continue
        label_hits = _label_candidates(label, ontology, correction_memory)
        if any(token and token in text for token in label_hits):
            votes.append(
                {
                    "claim_id": existence_claim_id(nid),
                    "view_type": "caption",
                    "vote": "support",
                    "score": 0.65,
                }
            )
            votes.append(
                {
                    "claim_id": label_claim_id(nid),
                    "view_type": "caption",
                    "vote": "support",
                    "score": 0.60,
                }
            )
        for att in node.get("attributes") or []:
            if not isinstance(att, dict):
                continue
            slot = str(att.get("slot", "") or "").strip()
            value = str(att.get("value", "") or "").strip().lower()
            if not slot or not value:
                continue
            if value in text and any(token and token in text for token in label_hits):
                votes.append(
                    {
                        "claim_id": attribute_claim_id(nid, slot),
                        "view_type": "caption",
                        "vote": "support",
                        "score": 0.55,
                    }
                )

    for edge in graph.get("edges") or []:
        eid = str(edge.get("edge_id", "") or "")
        rel = str(edge.get("relation", "") or "").strip().lower()
        if not eid or not rel:
            continue
        rel_phrase = rel.replace("_", " ")
        if rel_phrase in text or rel in text:
            votes.append(
                {
                    "claim_id": relation_claim_id(eid),
                    "view_type": "caption",
                    "vote": "support",
                    "score": 0.60,
                }
            )
    return votes


def _structured_feedback_report(
    payload: Dict[str, object],
    graph: Dict[str, object],
) -> Dict[str, object]:
    return {
        "structured": True,
        "fallback_used": False,
        "caption_text": str(payload.get("caption", "") or "").strip(),
        "supported_entities": _normalize_entity_rows(payload.get("supported_entities"), graph),
        "unsupported_entities": _normalize_entity_rows(payload.get("unsupported_entities"), graph),
        "supported_attributes": _normalize_attribute_rows(payload.get("supported_attributes"), graph),
        "unsupported_attributes": _normalize_attribute_rows(payload.get("unsupported_attributes"), graph),
        "supported_relations": _normalize_relation_rows(payload.get("supported_relations"), graph),
        "unsupported_relations": _normalize_relation_rows(payload.get("unsupported_relations"), graph),
        "hallucinated_mentions": [
            str(item or "").strip()
            for item in list(payload.get("hallucinated_mentions") or [])
            if str(item or "").strip()
        ],
    }


def _structured_report_to_votes(
    report: Dict[str, object],
    *,
    allow_unsupported_conflicts: bool = True,
) -> List[Dict[str, object]]:
    votes: List[Dict[str, object]] = []
    for entity_id in list(report.get("supported_entities") or []):
        votes.append(
            {
                "claim_id": existence_claim_id(str(entity_id)),
                "view_type": "caption",
                "vote": "support",
                "score": 0.65,
            }
        )
        votes.append(
            {
                "claim_id": label_claim_id(str(entity_id)),
                "view_type": "caption",
                "vote": "support",
                "score": 0.60,
            }
        )
    if allow_unsupported_conflicts:
        for entity_id in list(report.get("unsupported_entities") or []):
            votes.append(
                {
                    "claim_id": existence_claim_id(str(entity_id)),
                    "view_type": "caption",
                    "vote": "conflict",
                    "score": 0.70,
                }
            )
            votes.append(
                {
                    "claim_id": label_claim_id(str(entity_id)),
                    "view_type": "caption",
                    "vote": "conflict",
                    "score": 0.55,
                }
            )

    for row in list(report.get("supported_attributes") or []):
        votes.append(
            {
                "claim_id": attribute_claim_id(str(row.get("entity_id", "")), str(row.get("slot", ""))),
                "view_type": "caption",
                "vote": "support",
                "score": 0.55,
            }
        )
    if allow_unsupported_conflicts:
        for row in list(report.get("unsupported_attributes") or []):
            votes.append(
                {
                    "claim_id": attribute_claim_id(str(row.get("entity_id", "")), str(row.get("slot", ""))),
                    "view_type": "caption",
                    "vote": "conflict",
                    "score": 0.60,
                }
            )

    for edge_id in list(report.get("supported_relations") or []):
        votes.append(
            {
                "claim_id": relation_claim_id(str(edge_id)),
                "view_type": "caption",
                "vote": "support",
                "score": 0.60,
            }
        )
    if allow_unsupported_conflicts:
        for edge_id in list(report.get("unsupported_relations") or []):
            votes.append(
                {
                    "claim_id": relation_claim_id(str(edge_id)),
                    "view_type": "caption",
                    "vote": "conflict",
                    "score": 0.68,
                }
            )
    return votes


def caption_to_claim_feedback(
    caption_payload: object,
    graph: Dict[str, object],
    ontology,
    correction_memory: Optional[Dict[str, object]] = None,
    *,
    allow_unsupported_conflicts: bool = True,
) -> Dict[str, object]:
    raw_payload = caption_payload
    parsed = None
    caption_text = ""
    if isinstance(raw_payload, dict):
        parsed = raw_payload if any(str(key).startswith("supported_") or str(key).startswith("unsupported_") for key in raw_payload.keys()) else None
        caption_text = str(raw_payload.get("caption", "") or raw_payload.get("raw_text", "") or "").strip()
        if parsed is None and caption_text:
            parsed = _extract_json_object(caption_text)
    else:
        caption_text = str(raw_payload or "").strip()
        parsed = _extract_json_object(caption_text)

    if isinstance(parsed, dict):
        report = _structured_feedback_report(parsed, graph)
        if not report["caption_text"]:
            report["caption_text"] = caption_text
        votes = _structured_report_to_votes(
            report,
            allow_unsupported_conflicts=allow_unsupported_conflicts,
        )
    else:
        text = _strip_code_fence(caption_text)
        votes = _fallback_caption_votes(text, graph, ontology, correction_memory)
        report = {
            "structured": False,
            "fallback_used": True,
            "caption_text": text,
            "supported_entities": [],
            "unsupported_entities": [],
            "supported_attributes": [],
            "unsupported_attributes": [],
            "supported_relations": [],
            "unsupported_relations": [],
            "hallucinated_mentions": [],
        }

    report["vote_count"] = len(votes)
    report["support_vote_count"] = len([row for row in votes if str(row.get("vote", "")) == "support"])
    report["conflict_vote_count"] = len([row for row in votes if str(row.get("vote", "")) == "conflict"])
    return {"votes": votes, "report": report}


def caption_to_claim_votes(
    caption_payload: object,
    graph: Dict[str, object],
    ontology,
    correction_memory: Optional[Dict[str, object]] = None,
    *,
    allow_unsupported_conflicts: bool = True,
) -> List[Dict[str, object]]:
    return caption_to_claim_feedback(
        caption_payload,
        graph,
        ontology,
        correction_memory,
        allow_unsupported_conflicts=allow_unsupported_conflicts,
    )["votes"]
