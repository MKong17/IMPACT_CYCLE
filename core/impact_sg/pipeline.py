from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from typing import Callable, Dict, List, Tuple

from .attribute_extractor import extract_attributes_for_nodes
from .detection_io import flatten_post_threshold_records, summarize_detection_payload
from .ontology import load_ontology, ontology_from_payload
from .sam_backend import SAMBackend, SAMBackendConfig
from .proposal_pipeline import build_entity_proposals, is_human_like, is_static_background_like
from .review_queue import build_review_queue
from .scene_graph_builder import build_scene_graph
from .validators import validate_scene_graph, validate_vqa_evidence
from .vqa import generate_multi_turn_vqa, generate_single_turn_vqa


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_prompt_pack(path: str) -> Dict[str, object]:
    abs_path = os.path.abspath(os.path.expanduser(str(path or "").strip()))
    if abs_path.lower().endswith(".json"):
        return load_json(abs_path)

    with open(abs_path, "r", encoding="utf-8") as f:
        lines = [str(line).rstrip("\n") for line in f]

    mode = ""
    category_rows: List[Dict[str, str]] = []
    sentence_rows: List[str] = []
    seen_cat: set[tuple[str, str]] = set()
    seen_sent: set[str] = set()

    for raw in lines:
        line = str(raw or "").strip()
        if not line:
            continue
        if line.startswith("=== CATEGORY PROMPTS ==="):
            mode = "category"
            continue
        if line.startswith("=== SENTENCE PROMPTS ==="):
            mode = "sentence"
            continue
        if line.startswith("===") or line.startswith("FINAL_"):
            continue
        if mode == "category":
            if "\t" in line:
                left, right = line.split("\t", 1)
            else:
                parts = re.split(r"\s{2,}", line, maxsplit=1)
                left = str(parts[0] if parts else "").strip()
                right = str(parts[1] if len(parts) > 1 else left).strip()
            canonical_label = str(left or "").strip()
            prompt = str(right or canonical_label).strip()
            key = (canonical_label.lower(), prompt.lower())
            if not canonical_label or not prompt or key in seen_cat:
                continue
            seen_cat.add(key)
            category_rows.append({"canonical_label": canonical_label, "prompt": prompt})
        elif mode == "sentence":
            key = line.lower()
            if key in seen_sent:
                continue
            seen_sent.add(key)
            sentence_rows.append(line)

    canonical_entities: List[Dict[str, object]] = []
    seen_label: set[str] = set()
    for row in category_rows:
        label = str(row.get("canonical_label", "") or "").strip()
        prompt = str(row.get("prompt", "") or "").strip()
        key = label.lower()
        if not label or key in seen_label:
            continue
        seen_label.add(key)
        prompt_variants = [prompt] if prompt and prompt.lower() != key else [label]
        canonical_entities.append(
            {
                "label": label,
                "synonyms": [],
                "prompt_variants": prompt_variants,
                "attribute_slots": [],
                "mandatory_attributes": [],
            }
        )

    return {
        "canonical_entities": canonical_entities,
        "category_prompt_templates": ["{label}"],
        "sentence_prompt_templates": [],
        "explicit_category_prompts": category_rows,
        "explicit_sentence_prompts": sentence_rows,
    }


def _merge_cfg(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_cfg(dict(out.get(key) or {}), value)
        else:
            out[key] = value
    return out


def _merge_unique_text_list(base: List[object], extra: List[object]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for source in (base or []), (extra or []):
        for value in source:
            text = str(value or "").strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            out.append(text)
    return out


def _merge_entity_lists(base: List[object], extra: List[object]) -> List[Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = {}
    ordered_keys: List[str] = []
    for source in (base or []), (extra or []):
        for item in source:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "") or "").strip()
            if not label:
                continue
            key = label.lower()
            if key not in merged:
                merged[key] = {
                    "label": label,
                    "synonyms": [],
                    "prompt_variants": [],
                    "attribute_slots": [],
                    "mandatory_attributes": [],
                }
                ordered_keys.append(key)
            row = merged[key]
            row["synonyms"] = _merge_unique_text_list(
                list(row.get("synonyms") or []),
                list(item.get("synonyms") or []),
            )
            row["prompt_variants"] = _merge_unique_text_list(
                list(row.get("prompt_variants") or []),
                list(item.get("prompt_variants") or []),
            )
            row["attribute_slots"] = _merge_unique_text_list(
                list(row.get("attribute_slots") or []),
                list(item.get("attribute_slots") or []),
            )
            row["mandatory_attributes"] = _merge_unique_text_list(
                list(row.get("mandatory_attributes") or []),
                list(item.get("mandatory_attributes") or []),
            )
    return [merged[key] for key in ordered_keys]


def _build_global_semantic_summary(graph: Dict[str, object]) -> str:
    nodes = [dict(x) for x in list(graph.get("nodes") or []) if isinstance(x, dict)]
    edges = [dict(x) for x in list(graph.get("edges") or []) if isinstance(x, dict)]
    if not nodes:
        return "No objects detected in this frame."

    label_counts: Dict[str, int] = {}
    for node in nodes:
        label = str(node.get("canonical_label", "object") or "object").strip().lower() or "object"
        label_counts[label] = int(label_counts.get(label, 0) + 1)
    top_labels = sorted(label_counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:3]
    label_text = ", ".join([f"{name} x{count}" for name, count in top_labels]) if top_labels else "objects"

    relation_counts: Dict[str, int] = {}
    for edge in edges:
        rel = str(edge.get("relation", "") or "").strip().lower()
        if rel:
            relation_counts[rel] = int(relation_counts.get(rel, 0) + 1)
    top_rels = sorted(relation_counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:2]
    rel_text = ", ".join([f"{name} x{count}" for name, count in top_rels]) if top_rels else ""

    attr_fragments: List[str] = []
    for node in nodes[:3]:
        label = str(node.get("canonical_label", "object") or "object").strip().lower() or "object"
        attrs = [dict(a) for a in list(node.get("attributes") or []) if isinstance(a, dict)]
        non_empty = [a for a in attrs if str(a.get("value", "") or "").strip()]
        if not non_empty:
            continue
        a = non_empty[0]
        attr_fragments.append(f"{label}.{str(a.get('slot','')).strip()}={str(a.get('value','')).strip()}")
    attr_text = ", ".join(attr_fragments)

    parts = [f"Detected {len(nodes)} objects ({label_text})"]
    if rel_text:
        parts.append(f"main relations: {rel_text}")
    if attr_text:
        parts.append(f"example attributes: {attr_text}")
    return ". ".join(parts) + "."


def _merge_ontology_payload(base: Dict[str, object], extra: Dict[str, object]) -> Dict[str, object]:
    merged = dict(base or {})
    merged["canonical_entities"] = _merge_entity_lists(
        list((base or {}).get("canonical_entities") or []),
        list((extra or {}).get("canonical_entities") or []),
    )
    merged["category_prompt_templates"] = _merge_unique_text_list(
        list((base or {}).get("category_prompt_templates") or []),
        list((extra or {}).get("category_prompt_templates") or []),
    )
    merged["sentence_prompt_templates"] = _merge_unique_text_list(
        list((base or {}).get("sentence_prompt_templates") or []),
        list((extra or {}).get("sentence_prompt_templates") or []),
    )
    merged["question_types"] = _merge_unique_text_list(
        list((base or {}).get("question_types") or []),
        list((extra or {}).get("question_types") or []),
    )
    base_rel = dict((base or {}).get("relation_vocabulary") or {})
    extra_rel = dict((extra or {}).get("relation_vocabulary") or {})
    rel_keys = list(base_rel.keys())
    for key in extra_rel.keys():
        if key not in rel_keys:
            rel_keys.append(key)
    merged_rel: Dict[str, List[str]] = {}
    for key in rel_keys:
        merged_rel[str(key)] = _merge_unique_text_list(
            list(base_rel.get(key) or []),
            list(extra_rel.get(key) or []),
        )
    merged["relation_vocabulary"] = merged_rel
    return merged


def _provenance_backend_name(cfg: Dict[str, object]) -> str:
    backend = dict(cfg.get("backend") or {})
    args_file = str(backend.get("external_command_args_file", "") or "").strip().lower()
    batch_args_file = str(backend.get("external_batch_command_args_file", "") or "").strip().lower()
    template = str(backend.get("external_command_template", "") or "").strip().lower()
    runtime_hint = " ".join([args_file, batch_args_file, template]).strip()
    if "sam3" in runtime_hint:
        return "SAM3"
    return "SAM3"


HUMAN_PRIORITY_PROMPTS = [
    "person",
    "player",
    "football player",
    "goalkeeper",
    "referee",
    "athlete",
    "man",
]

_BACKEND_POOL_LOCK = threading.Lock()
_BACKEND_POOL: Dict[str, SAMBackend] = {}


def _emit_debug(progress_cb: Callable[[str], None] | None, message: str) -> None:
    if not callable(progress_cb):
        return
    try:
        progress_cb(str(message))
    except Exception:
        pass


def _emit_profile(progress_cb: Callable[[str], None] | None, stage: str, elapsed_sec: float, **extra: object) -> None:
    if not callable(progress_cb):
        return
    parts = [f"[SG-PROFILE] {stage}: {float(elapsed_sec):.3f}s"]
    for key, value in extra.items():
        parts.append(f"{key}={value}")
    _emit_debug(progress_cb, " ".join(parts))


def _bbox_area_ratio(item: Dict[str, object], image_size: Tuple[int, int]) -> float:
    bbox = list(item.get("bbox") or [0, 0, 0, 0])
    if len(bbox) < 4:
        return 0.0
    img_area = max(1, int(image_size[0]) * int(image_size[1]))
    return float(max(0, int(bbox[2])) * max(0, int(bbox[3]))) / float(img_area)


def _bbox_xyxy(item: Dict[str, object]) -> Tuple[float, float, float, float]:
    bbox = list(item.get("bbox") or [0, 0, 0, 0])
    if len(bbox) < 4:
        return 0.0, 0.0, 0.0, 0.0
    x = float(bbox[0] or 0.0)
    y = float(bbox[1] or 0.0)
    w = max(0.0, float(bbox[2] or 0.0))
    h = max(0.0, float(bbox[3] or 0.0))
    return x, y, x + w, y + h


def _bbox_intersection_area(a: Dict[str, object], b: Dict[str, object]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_xyxy(a)
    bx1, by1, bx2, by2 = _bbox_xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return float(iw * ih)


def _bbox_iou(a: Dict[str, object], b: Dict[str, object]) -> float:
    inter = _bbox_intersection_area(a, b)
    aw = max(0.0, float((list(a.get("bbox") or [0, 0, 0, 0]) + [0, 0, 0, 0])[2] or 0.0))
    ah = max(0.0, float((list(a.get("bbox") or [0, 0, 0, 0]) + [0, 0, 0, 0])[3] or 0.0))
    bw = max(0.0, float((list(b.get("bbox") or [0, 0, 0, 0]) + [0, 0, 0, 0])[2] or 0.0))
    bh = max(0.0, float((list(b.get("bbox") or [0, 0, 0, 0]) + [0, 0, 0, 0])[3] or 0.0))
    area_a = aw * ah
    area_b = bw * bh
    union = max(1e-6, area_a + area_b - inter)
    return float(inter / union)


def _bbox_center_distance_norm(a: Dict[str, object], b: Dict[str, object]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_xyxy(a)
    bx1, by1, bx2, by2 = _bbox_xyxy(b)
    acx = (ax1 + ax2) * 0.5
    acy = (ay1 + ay2) * 0.5
    bcx = (bx1 + bx2) * 0.5
    bcy = (by1 + by2) * 0.5
    dx = acx - bcx
    dy = acy - bcy
    dist = (dx * dx + dy * dy) ** 0.5
    aw = max(1.0, ax2 - ax1)
    ah = max(1.0, ay2 - ay1)
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)
    ref = max(1.0, min((aw * ah) ** 0.5, (bw * bh) ** 0.5))
    return float(dist / ref)


def _suppress_nested_human_proposals(
    proposals: List[Dict[str, object]],
    *,
    overlap_ratio_threshold: float,
    area_ratio_threshold: float,
    score_margin: float,
    duplicate_iou_threshold: float,
    duplicate_center_norm_threshold: float,
    large_ambiguous_area_ratio: float,
    large_ambiguous_contained_overlap: float,
    large_ambiguous_min_contained: int,
    large_ambiguous_score_margin: float,
    small_duplicate_area_ratio_max: float,
    small_duplicate_iou_threshold: float,
    small_duplicate_center_norm_threshold: float,
    multi_person_merge_area_ratio_min: float,
    multi_person_merge_contained_overlap: float,
    multi_person_merge_min_contained: int,
    multi_person_merge_score_margin: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rows = [dict(x) for x in list(proposals or []) if isinstance(x, dict)]
    human_indices = [i for i, row in enumerate(rows) if is_human_like(row)]
    if len(human_indices) <= 1:
        return rows, []

    drop_indices: set[int] = set()
    for ai in range(len(human_indices)):
        i = human_indices[ai]
        if i in drop_indices:
            continue
        a = rows[i]
        area_a = max(0.0, float((a.get("bbox") or [0, 0, 0, 0])[2] or 0.0) * float((a.get("bbox") or [0, 0, 0, 0])[3] or 0.0))
        if area_a <= 1.0:
            continue
        for bi in range(ai + 1, len(human_indices)):
            j = human_indices[bi]
            if j in drop_indices:
                continue
            b = rows[j]
            area_b = max(0.0, float((b.get("bbox") or [0, 0, 0, 0])[2] or 0.0) * float((b.get("bbox") or [0, 0, 0, 0])[3] or 0.0))
            if area_b <= 1.0:
                continue
            inter = _bbox_intersection_area(a, b)
            iou = _bbox_iou(a, b)
            center_norm = _bbox_center_distance_norm(a, b)
            small = min(area_a, area_b)
            large = max(area_a, area_b)
            if small <= 1.0:
                continue
            containment = inter / small
            area_ratio = large / small
            area_ratio_a = float(a.get("debug_metrics", {}).get("area_ratio", 0.0) or 0.0)
            area_ratio_b = float(b.get("debug_metrics", {}).get("area_ratio", 0.0) or 0.0)
            both_small = bool(
                area_ratio_a <= float(small_duplicate_area_ratio_max)
                and area_ratio_b <= float(small_duplicate_area_ratio_max)
            )
            is_near_duplicate = bool(
                iou >= float(duplicate_iou_threshold)
                or (containment >= float(overlap_ratio_threshold) and area_ratio >= float(area_ratio_threshold))
                or (iou >= 0.15 and center_norm <= float(duplicate_center_norm_threshold))
                or (both_small and iou >= float(small_duplicate_iou_threshold) and center_norm <= float(small_duplicate_center_norm_threshold))
            )
            if not is_near_duplicate:
                continue

            score_a = float(a.get("score", 0.0) or 0.0)
            score_b = float(b.get("score", 0.0) or 0.0)
            # Keep the better full-body candidate; suppress nested partial duplicate.
            if score_a > score_b + float(score_margin):
                drop_idx = j
            elif score_b > score_a + float(score_margin):
                drop_idx = i
            else:
                drop_idx = i if area_a < area_b else j
            drop_indices.add(drop_idx)

    # Suppress one oversized ambiguous person box when it clearly covers several person boxes.
    for i in human_indices:
        if i in drop_indices:
            continue
        row = rows[i]
        area_ratio_row = float(row.get("debug_metrics", {}).get("area_ratio", 0.0) or 0.0)
        if area_ratio_row < float(large_ambiguous_area_ratio):
            continue
        score_row = float(row.get("score", 0.0) or 0.0)
        contained_scores: List[float] = []
        for j in human_indices:
            if i == j or j in drop_indices:
                continue
            other = rows[j]
            inter = _bbox_intersection_area(row, other)
            ow = max(0.0, float((list(other.get("bbox") or [0, 0, 0, 0]) + [0, 0, 0, 0])[2] or 0.0))
            oh = max(0.0, float((list(other.get("bbox") or [0, 0, 0, 0]) + [0, 0, 0, 0])[3] or 0.0))
            oa = max(1e-6, ow * oh)
            if inter / oa >= float(large_ambiguous_contained_overlap):
                contained_scores.append(float(other.get("score", 0.0) or 0.0))
        if len(contained_scores) >= int(large_ambiguous_min_contained):
            max_contained = max(contained_scores) if contained_scores else 0.0
            if score_row <= max_contained + float(large_ambiguous_score_margin):
                drop_indices.add(i)

    # Hard suppression: a single large person box covering multiple person boxes likely merges >1 person.
    for i in human_indices:
        if i in drop_indices:
            continue
        row = rows[i]
        area_ratio_row = float(row.get("debug_metrics", {}).get("area_ratio", 0.0) or 0.0)
        if area_ratio_row < float(multi_person_merge_area_ratio_min):
            continue
        score_row = float(row.get("score", 0.0) or 0.0)
        contained_scores: List[float] = []
        for j in human_indices:
            if i == j or j in drop_indices:
                continue
            other = rows[j]
            inter = _bbox_intersection_area(row, other)
            ow = max(0.0, float((list(other.get("bbox") or [0, 0, 0, 0]) + [0, 0, 0, 0])[2] or 0.0))
            oh = max(0.0, float((list(other.get("bbox") or [0, 0, 0, 0]) + [0, 0, 0, 0])[3] or 0.0))
            oa = max(1e-6, ow * oh)
            if inter / oa >= float(multi_person_merge_contained_overlap):
                contained_scores.append(float(other.get("score", 0.0) or 0.0))
        if len(contained_scores) >= int(multi_person_merge_min_contained):
            if score_row <= max(contained_scores) + float(multi_person_merge_score_margin):
                drop_indices.add(i)

    kept: List[Dict[str, object]] = []
    dropped: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        if idx in drop_indices:
            row2 = dict(row)
            row2["drop_reason"] = "nested/duplicate human candidate"
            dropped.append(row2)
        else:
            kept.append(row)
    return kept, dropped


def _edge_overlap_ratio(item: Dict[str, object], image_size: Tuple[int, int], edge_width_ratio: float) -> float:
    bbox = list(item.get("bbox") or [0, 0, 0, 0])
    if len(bbox) < 4:
        return 0.0
    x = float(bbox[0] or 0.0)
    w = float(bbox[2] or 0.0)
    if w <= 0.0:
        return 0.0
    w_img = max(1.0, float(image_size[0]))
    edge_w = max(1.0, min(w_img * 0.49, w_img * max(0.01, float(edge_width_ratio))))
    left_ov = max(0.0, min(x + w, edge_w) - max(x, 0.0))
    right_ov = max(0.0, min(x + w, w_img) - max(x, w_img - edge_w))
    return max(left_ov, right_ov) / max(1.0, w)


def _is_edge_human_candidate(
    item: Dict[str, object],
    image_size: Tuple[int, int],
    *,
    edge_width_ratio: float,
    min_overlap_ratio: float,
) -> bool:
    if not is_human_like(item):
        return False
    overlap = _edge_overlap_ratio(item, image_size, edge_width_ratio=edge_width_ratio)
    return overlap >= float(min_overlap_ratio)


def _human_center_bias(item: Dict[str, object], image_size: Tuple[int, int]) -> float:
    bbox = list(item.get("bbox") or [0, 0, 0, 0])
    if len(bbox) < 4:
        return 0.0
    w_img = max(1, int(image_size[0]))
    h_img = max(1, int(image_size[1]))
    cx = float(bbox[0]) + float(bbox[2]) / 2.0
    cy = float(bbox[1]) + float(bbox[3]) / 2.0
    x_norm = cx / float(w_img)
    y_norm = cy / float(h_img)
    center = max(0.0, 1.0 - min(1.0, abs(x_norm - 0.5) / 0.5))
    lower_mid = max(0.0, 1.0 - min(1.0, abs(y_norm - 0.68) / 0.68))
    return (0.45 * center) + (0.55 * lower_mid)


def _proposal_priority(
    item: Dict[str, object],
    image_size: Tuple[int, int],
    *,
    human_priority_bias: bool = True,
) -> float:
    score = float(item.get("score", 0.0))
    risk = float(item.get("risk", 0.0))
    priority = score - (0.35 * risk)
    if bool(human_priority_bias) and is_human_like(item):
        priority += 0.18 + (0.22 * _human_center_bias(item, image_size))
        # Panoramic cameras often distort people near left/right boundaries.
        # Add mild edge bonus so true edge persons are not consistently ranked out.
        edge_ratio = _edge_overlap_ratio(item, image_size, edge_width_ratio=0.22)
        priority += 0.10 * max(0.0, min(1.0, edge_ratio))
    if is_static_background_like(item):
        priority -= 0.12 + (0.08 * min(1.0, _bbox_area_ratio(item, image_size) * 4.0))
    return priority


def _sports_human_prompt_items(ontology) -> List[Dict[str, str]]:
    person_label = ""
    for item in list(getattr(ontology, "canonical_entities", []) or []):
        label = str(item.get("label", "") or "").strip()
        if label.lower() == "person":
            person_label = label
            break
    if not person_label:
        person_label = "person"
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for prompt in HUMAN_PRIORITY_PROMPTS:
        key = prompt.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"canonical_label": person_label, "prompt": prompt})
    return out


def _person_focus_prompt_items(ontology, proposal_cfg: Dict[str, object]) -> List[Dict[str, str]]:
    """
    Build a SAM3-person-first prompt list.
    This mimics the strong baseline in /lsdf/users/wkong/sam3/scripts/demo.py
    where inference is done with prompt="person" before broader scene prompts.
    """
    person_label = "person"
    for item in list(getattr(ontology, "canonical_entities", []) or []):
        label = str(item.get("label", "") or "").strip()
        if label.lower() == "person":
            person_label = label
            break

    raw = proposal_cfg.get("person_focus_prompts")
    prompts: List[str] = []
    if isinstance(raw, list):
        prompts = [str(x or "").strip() for x in raw if str(x or "").strip()]
    if not prompts:
        prompts = ["person"]

    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for p in prompts:
        key = str(p).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"canonical_label": person_label, "prompt": p})
    return out


def _condense_category_prompt_items(prompt_items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen_labels: set[str] = set()
    for item in prompt_items or []:
        canonical_label = str(item.get("canonical_label", "") or "").strip()
        prompt = str(item.get("prompt", "") or "").strip()
        key = canonical_label.lower()
        if not canonical_label or not prompt or key in seen_labels:
            continue
        seen_labels.add(key)
        out.append({"canonical_label": canonical_label, "prompt": prompt})
    return out


def _log_candidate_rows(
    progress_cb: Callable[[str], None] | None,
    *,
    title: str,
    rows: List[Dict[str, object]],
    image_size: Tuple[int, int],
) -> None:
    _emit_debug(progress_cb, f"[SG-DEBUG] {title}: {len(rows)} candidates")
    for idx, row in enumerate(rows):
        prompt = str(row.get("prompt_used", "") or "").strip()
        label = str(row.get("canonical_label", "") or "").strip()
        score = float(row.get("score", 0.0))
        risk = float(row.get("risk", 0.0))
        area_ratio = _bbox_area_ratio(row, image_size)
        dbg = dict(row.get("debug_metrics") or {})
        overlap = float(dbg.get("max_same_label_overlap", 0.0) or 0.0)
        merged_from = list(row.get("merged_from") or [])
        merged_suffix = f" merged={len(merged_from)}" if merged_from else ""
        flags = []
        if is_human_like(row):
            flags.append("human")
        if is_static_background_like(row):
            flags.append("background")
        flag_text = f" flags={','.join(flags)}" if flags else ""
        _emit_debug(
            progress_cb,
            f"[SG-DEBUG]   #{idx+1} label={label or '-'} prompt={prompt or '-'} score={score:.3f} "
            f"risk={risk:.3f} area={area_ratio:.4f} overlap={overlap:.3f}{merged_suffix}{flag_text}",
        )


def _log_prompt_presence(progress_cb: Callable[[str], None] | None, rows: List[Dict[str, object]]) -> None:
    labels = ["person", "player", "football player", "goalkeeper", "referee", "man"]
    raw_text = " || ".join(
        f"{str(item.get('canonical_label', '') or '').lower()} | {str(item.get('prompt_used', '') or '').lower()}"
        for item in rows
    )
    for label in labels:
        present = label.lower() in raw_text
        _emit_debug(progress_cb, f"[SG-DEBUG] prompt presence '{label}': {'YES' if present else 'NO'}")


def _apply_sports_filtering(
    proposals: List[Dict[str, object]],
    *,
    image_size: Tuple[int, int],
    proposal_cfg: Dict[str, object],
    progress_cb: Callable[[str], None] | None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    human_special = bool(proposal_cfg.get("human_special_handling", True))
    min_score = float(proposal_cfg.get("min_score", proposal_cfg.get("low_confidence_threshold", 0.45)))
    max_risk = float(proposal_cfg.get("max_risk", 1.0))
    human_min_score = float(proposal_cfg.get("human_min_score", max(0.18, min_score - 0.20)))
    human_max_risk = float(proposal_cfg.get("human_max_risk", min(1.0, max_risk + 0.13)))
    edge_width_ratio = float(proposal_cfg.get("panorama_edge_width_ratio", 0.22))
    edge_min_overlap = float(proposal_cfg.get("panorama_edge_min_overlap", 0.35))
    human_edge_min_score = float(proposal_cfg.get("human_edge_min_score", max(0.16, human_min_score - 0.10)))
    human_edge_max_risk = float(proposal_cfg.get("human_edge_max_risk", min(1.0, human_max_risk + 0.08)))
    human_large_area_ratio = float(proposal_cfg.get("human_large_area_ratio", proposal_cfg.get("human_max_area_ratio", 0.65)))
    human_edge_large_area_ratio = float(proposal_cfg.get("human_edge_large_area_ratio", proposal_cfg.get("human_edge_max_area_ratio", 0.75)))
    human_large_area_min_score = float(proposal_cfg.get("human_large_area_min_score", 0.82))
    human_edge_large_area_min_score = float(proposal_cfg.get("human_edge_large_area_min_score", 0.88))
    human_hard_max_area_ratio = float(proposal_cfg.get("human_hard_max_area_ratio", 0.90))
    human_nested_overlap_ratio = float(proposal_cfg.get("human_nested_overlap_ratio", 0.72))
    human_nested_area_ratio = float(proposal_cfg.get("human_nested_area_ratio", 1.35))
    human_nested_score_margin = float(proposal_cfg.get("human_nested_score_margin", 0.08))
    human_duplicate_iou_threshold = float(proposal_cfg.get("human_duplicate_iou_threshold", 0.34))
    human_duplicate_center_norm_threshold = float(proposal_cfg.get("human_duplicate_center_norm_threshold", 0.22))
    human_large_ambiguous_area_ratio = float(proposal_cfg.get("human_large_ambiguous_area_ratio", 0.28))
    human_large_ambiguous_contained_overlap = float(proposal_cfg.get("human_large_ambiguous_contained_overlap", 0.60))
    human_large_ambiguous_min_contained = int(proposal_cfg.get("human_large_ambiguous_min_contained", 2))
    human_large_ambiguous_score_margin = float(proposal_cfg.get("human_large_ambiguous_score_margin", 0.06))
    human_small_duplicate_area_ratio_max = float(proposal_cfg.get("human_small_duplicate_area_ratio_max", 0.03))
    human_small_duplicate_iou_threshold = float(proposal_cfg.get("human_small_duplicate_iou_threshold", 0.12))
    human_small_duplicate_center_norm_threshold = float(proposal_cfg.get("human_small_duplicate_center_norm_threshold", 0.35))
    human_multi_person_merge_area_ratio_min = float(proposal_cfg.get("human_multi_person_merge_area_ratio_min", 0.20))
    human_multi_person_merge_contained_overlap = float(proposal_cfg.get("human_multi_person_merge_contained_overlap", 0.55))
    human_multi_person_merge_min_contained = int(proposal_cfg.get("human_multi_person_merge_min_contained", 2))
    human_multi_person_merge_score_margin = float(proposal_cfg.get("human_multi_person_merge_score_margin", 0.04))
    human_edge_priority_bonus = float(proposal_cfg.get("human_edge_priority_bonus", 0.10))
    max_nodes_after_merge = int(proposal_cfg.get("max_nodes_after_merge", 80))
    ensure_min_human = max(0, int(proposal_cfg.get("ensure_min_human_candidates", 4))) if human_special else 0

    kept: List[Dict[str, object]] = []
    dropped: List[Dict[str, object]] = []
    proposals_for_filter = [dict(x) for x in proposals if isinstance(x, dict)]
    if human_special:
        proposals_for_filter, nested_dropped = _suppress_nested_human_proposals(
            proposals_for_filter,
            overlap_ratio_threshold=human_nested_overlap_ratio,
            area_ratio_threshold=human_nested_area_ratio,
            score_margin=human_nested_score_margin,
            duplicate_iou_threshold=human_duplicate_iou_threshold,
            duplicate_center_norm_threshold=human_duplicate_center_norm_threshold,
            large_ambiguous_area_ratio=human_large_ambiguous_area_ratio,
            large_ambiguous_contained_overlap=human_large_ambiguous_contained_overlap,
            large_ambiguous_min_contained=human_large_ambiguous_min_contained,
            large_ambiguous_score_margin=human_large_ambiguous_score_margin,
            small_duplicate_area_ratio_max=human_small_duplicate_area_ratio_max,
            small_duplicate_iou_threshold=human_small_duplicate_iou_threshold,
            small_duplicate_center_norm_threshold=human_small_duplicate_center_norm_threshold,
            multi_person_merge_area_ratio_min=human_multi_person_merge_area_ratio_min,
            multi_person_merge_contained_overlap=human_multi_person_merge_contained_overlap,
            multi_person_merge_min_contained=human_multi_person_merge_min_contained,
            multi_person_merge_score_margin=human_multi_person_merge_score_margin,
        )
        dropped.extend(list(nested_dropped))
    for item in proposals_for_filter:
        row = dict(item)
        row["selection_priority"] = float(
            _proposal_priority(row, image_size, human_priority_bias=human_special)
        )
        score = float(row.get("score", 0.0))
        risk = float(row.get("risk", 0.0))
        human_like = bool(human_special and is_human_like(row))
        edge_human = bool(
            human_special
            and _is_edge_human_candidate(
                row,
                image_size,
                edge_width_ratio=edge_width_ratio,
                min_overlap_ratio=edge_min_overlap,
            )
        )
        if edge_human:
            row["selection_priority"] = float(row.get("selection_priority", 0.0)) + float(human_edge_priority_bonus)
            row["edge_compensation"] = True
        dbg = dict(row.get("debug_metrics") or {})
        area_ratio = float(dbg.get("area_ratio", _bbox_area_ratio(row, image_size)))
        reasons: List[str] = []
        if human_like:
            person_min_score = human_edge_min_score if edge_human else human_min_score
            person_max_risk = human_edge_max_risk if edge_human else human_max_risk
            person_large_area = human_edge_large_area_ratio if edge_human else human_large_area_ratio
            person_large_area_min_score = human_edge_large_area_min_score if edge_human else human_large_area_min_score
            if score < person_min_score:
                reasons.append("below confidence threshold")
            if risk > person_max_risk:
                reasons.append("high risk")
            if float(human_hard_max_area_ratio) > 0.0 and area_ratio >= float(human_hard_max_area_ratio):
                reasons.append("person box too large")
            elif float(person_large_area) > 0.0 and area_ratio >= float(person_large_area) and score < float(person_large_area_min_score):
                reasons.append("large person box needs very high confidence")
        else:
            if score < min_score:
                reasons.append("below confidence threshold")
            if risk > max_risk:
                reasons.append("high risk")
        if area_ratio <= float(proposal_cfg.get("small_object_area_ratio", 0.001)) and not human_like:
            reasons.append("too small")
        if is_static_background_like(row) and not human_like and area_ratio >= float(proposal_cfg.get("background_large_area_ratio", 0.08)):
            reasons.append("too large")
        if reasons:
            row["drop_reason"] = ", ".join(reasons)
            dropped.append(row)
        else:
            kept.append(row)

    kept.sort(key=lambda x: float(x.get("selection_priority", 0.0)), reverse=True)
    if max_nodes_after_merge > 0 and len(kept) > max_nodes_after_merge:
        kept_now = kept[:max_nodes_after_merge]
        overflow = kept[max_nodes_after_merge:]
        for row in overflow:
            row2 = dict(row)
            row2["drop_reason"] = "low rank / top-k truncation"
            dropped.append(row2)
        kept = kept_now

    human_candidates = [row for row in proposals_for_filter if bool(human_special and is_human_like(row))]
    human_kept = [row for row in kept if bool(human_special and is_human_like(row))]
    if bool(human_special) and ensure_min_human > 0 and len(human_kept) < ensure_min_human:
        missing = ensure_min_human - len(human_kept)
        pool = [row for row in human_candidates if str(row.get("entity_id", "")) not in {str(x.get("entity_id", "")) for x in human_kept}]
        # Never rescue oversized full-frame person boxes.
        filtered_pool: List[Dict[str, object]] = []
        for row in pool:
            dbg = dict(row.get("debug_metrics") or {})
            area_ratio = float(dbg.get("area_ratio", _bbox_area_ratio(row, image_size)))
            edge_human = bool(
                _is_edge_human_candidate(
                    row,
                    image_size,
                    edge_width_ratio=edge_width_ratio,
                    min_overlap_ratio=edge_min_overlap,
                )
            )
            large_area = human_edge_large_area_ratio if edge_human else human_large_area_ratio
            large_area_min_score = human_edge_large_area_min_score if edge_human else human_large_area_min_score
            if float(human_hard_max_area_ratio) > 0.0 and area_ratio >= float(human_hard_max_area_ratio):
                continue
            if float(large_area) > 0.0 and area_ratio >= float(large_area) and float(row.get("score", 0.0) or 0.0) < float(large_area_min_score):
                continue
            filtered_pool.append(row)
        pool = filtered_pool
        pool.sort(
            key=lambda x: float(_proposal_priority(x, image_size, human_priority_bias=human_special)),
            reverse=True,
        )
        rescued = 0
        for row in pool:
            row2 = dict(row)
            row2["selection_priority"] = float(
                _proposal_priority(row2, image_size, human_priority_bias=human_special)
            )
            kept.append(row2)
            rescued += 1
            _emit_debug(progress_cb, f"[SG-DEBUG] rescued human candidate label={row2.get('canonical_label','')} prompt={row2.get('prompt_used','')} reason=foreground-human bias")
            if rescued >= missing:
                break
        kept.sort(key=lambda x: float(x.get("selection_priority", 0.0)), reverse=True)
        if max_nodes_after_merge > 0:
            kept = kept[:max_nodes_after_merge]

    return kept, dropped


def _backend_from_cfg(
    cfg: Dict[str, object],
    repo_root: str,
    *,
    progress_cb: Callable[[str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> SAMBackend:
    backend = cfg.get("backend") or {}
    cache_dir = str(backend.get("cache_dir", ".cache/impact_sg"))
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.abspath(os.path.join(repo_root, cache_dir))
    mock_results_path = str(backend.get("mock_results_path", "") or "")
    if mock_results_path and not os.path.isabs(mock_results_path):
        mock_results_path = os.path.abspath(os.path.join(repo_root, mock_results_path))
    external_command_args_file = str(backend.get("external_command_args_file", "") or "")
    if external_command_args_file and not os.path.isabs(external_command_args_file):
        external_command_args_file = os.path.abspath(os.path.join(repo_root, external_command_args_file))
    external_batch_command_args_file = str(backend.get("external_batch_command_args_file", "") or "")
    if external_batch_command_args_file and not os.path.isabs(external_batch_command_args_file):
        external_batch_command_args_file = os.path.abspath(os.path.join(repo_root, external_batch_command_args_file))
    external_command_cwd = str(backend.get("external_command_cwd", "") or "")
    if external_command_cwd and not os.path.isabs(external_command_cwd):
        external_command_cwd = os.path.abspath(os.path.join(repo_root, external_command_cwd))
    raw_args = backend.get("external_command_args") or []
    external_command_args = tuple(str(x) for x in raw_args) if isinstance(raw_args, list) else tuple()
    raw_batch_args = backend.get("external_batch_command_args") or []
    external_batch_command_args = tuple(str(x) for x in raw_batch_args) if isinstance(raw_batch_args, list) else tuple()
    config_obj = SAMBackendConfig(
        provider=str(backend.get("provider", "mock")),
        max_instances_per_prompt=int(backend.get("max_instances_per_prompt", 20)),
        enable_two_stage_refinement=bool(backend.get("enable_two_stage_refinement", False)),
        cache_dir=cache_dir,
        external_command_template=str(backend.get("external_command_template", "")),
        external_command_args=external_command_args,
        external_command_args_file=external_command_args_file,
        external_batch_command_template=str(backend.get("external_batch_command_template", "")),
        external_batch_command_args=external_batch_command_args,
        external_batch_command_args_file=external_batch_command_args_file,
        external_command_cwd=external_command_cwd,
        external_timeout_sec=int(backend.get("external_timeout_sec", 1800)),
        external_use_persistent_process=bool(backend.get("external_use_persistent_process", True)),
        mock_results_path=mock_results_path,
        disable_cache=bool(backend.get("disable_cache", False)),
        progress_cb=progress_cb,
        cancel_cb=cancel_cb,
    )

    reuse_enabled = bool(backend.get("reuse_backend_across_calls", True))
    if not reuse_enabled:
        return SAMBackend(config_obj)

    pool_payload = {
        "provider": config_obj.provider,
        "max_instances_per_prompt": config_obj.max_instances_per_prompt,
        "enable_two_stage_refinement": config_obj.enable_two_stage_refinement,
        "cache_dir": config_obj.cache_dir,
        "external_command_template": config_obj.external_command_template,
        "external_command_args": list(config_obj.external_command_args),
        "external_command_args_file": config_obj.external_command_args_file,
        "external_batch_command_template": config_obj.external_batch_command_template,
        "external_batch_command_args": list(config_obj.external_batch_command_args),
        "external_batch_command_args_file": config_obj.external_batch_command_args_file,
        "external_command_cwd": config_obj.external_command_cwd,
        "external_timeout_sec": config_obj.external_timeout_sec,
        "external_use_persistent_process": config_obj.external_use_persistent_process,
        "mock_results_path": config_obj.mock_results_path,
        "disable_cache": config_obj.disable_cache,
    }
    pool_key = json.dumps(pool_payload, ensure_ascii=True, sort_keys=True)
    with _BACKEND_POOL_LOCK:
        cached = _BACKEND_POOL.get(pool_key)
        if cached is None:
            cached = SAMBackend(config_obj)
            _BACKEND_POOL[pool_key] = cached
        else:
            cached.cfg.progress_cb = progress_cb
            cached.cfg.cancel_cb = cancel_cb
        return cached


def release_backend_pool() -> None:
    with _BACKEND_POOL_LOCK:
        backends = list(_BACKEND_POOL.values())
        _BACKEND_POOL.clear()
    for backend in backends:
        try:
            backend.close()
        except Exception:
            pass


def run_build_scene_graph(
    *,
    image_id: str,
    image_path: str,
    ontology_path: str,
    pipeline_cfg_path: str,
    image_size: Tuple[int, int],
    enable_sentence_refine: bool = False,
    custom_ontology_dict: Dict[str, object] | None = None,
    pipeline_cfg_override: Dict[str, object] | None = None,
    precomputed_detections: Dict[str, object] | None = None,
    progress_cb: Callable[[str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> Dict[str, object]:
    """Build scene graph for an image.
    
    Args:
        image_id: Unique identifier for the image
        image_path: Path to the image file
        ontology_path: Path to ontology JSON file (used as fallback)
        pipeline_cfg_path: Path to pipeline config JSON file
        image_size: (width, height) tuple of image size
        enable_sentence_refine: Enable sentence-level refinement
        custom_ontology_dict: Optional custom ontology dict. If provided, 
                             this overrides loading from ontology_path.
    
    Returns:
        Scene graph dict with nodes, edges, and metadata
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Use custom ontology if provided, otherwise load from file
    if custom_ontology_dict is not None:
        ontology_payload = dict(custom_ontology_dict)
    else:
        ontology_payload = load_json(ontology_path)

    cfg = load_json(pipeline_cfg_path)
    prompt_pack_file = str(cfg.get("ontology_prompt_pack_file", "") or "").strip()
    explicit_category_prompts: List[Dict[str, str]] = []
    explicit_sentence_prompts: List[str] = []
    if prompt_pack_file:
        if not os.path.isabs(prompt_pack_file):
            prompt_pack_file = os.path.abspath(os.path.join(repo_root, prompt_pack_file))
        if os.path.isfile(prompt_pack_file):
            prompt_pack_payload = _load_prompt_pack(prompt_pack_file)
            explicit_category_prompts = [
                {"canonical_label": str(x.get("canonical_label", "") or "").strip(), "prompt": str(x.get("prompt", "") or "").strip()}
                for x in list(prompt_pack_payload.get("explicit_category_prompts") or [])
                if isinstance(x, dict)
            ]
            explicit_sentence_prompts = [
                str(x).strip() for x in list(prompt_pack_payload.get("explicit_sentence_prompts") or []) if str(x).strip()
            ]
            ontology_payload = _merge_ontology_payload(ontology_payload, prompt_pack_payload)
    if isinstance(pipeline_cfg_override, dict) and pipeline_cfg_override:
        cfg = _merge_cfg(cfg, pipeline_cfg_override)
    ontology = ontology_from_payload(ontology_payload)
    ablation = cfg.get("ablation") or {}

    total_t0 = time.perf_counter()
    provenance_backend = _provenance_backend_name(cfg)
    prompts = ontology.build_prompt_bank()
    if explicit_category_prompts:
        prompt_items = _condense_category_prompt_items(list(explicit_category_prompts))
    else:
        prompt_items = _condense_category_prompt_items(list(prompts.category_prompts or []))
    sentence_prompts = list(explicit_sentence_prompts or list(prompts.sentence_prompts or []))
    human_priority_prompts = _sports_human_prompt_items(ontology)
    seen_prompt_keys: set[tuple[str, str]] = set()
    merged_prompt_items: List[Dict[str, str]] = []
    for item in list(prompt_items):
        canonical_label = str(item.get("canonical_label", "") or "").strip()
        prompt = str(item.get("prompt", "") or "").strip()
        key = (canonical_label.lower(), prompt.lower())
        if not canonical_label or not prompt or key in seen_prompt_keys:
            continue
        seen_prompt_keys.add(key)
        merged_prompt_items.append({"canonical_label": canonical_label, "prompt": prompt})
    prompt_items = merged_prompt_items

    proposal_cfg = cfg.get("proposal") or {}
    max_category_prompts = int(proposal_cfg.get("max_category_prompts", 32) or 32)
    if max_category_prompts > 0 and len(prompt_items) > max_category_prompts:
        _emit_debug(
            progress_cb,
            f"[SG-DEBUG] category prompt count capped {len(prompt_items)} -> {max_category_prompts}",
        )
        prompt_items = prompt_items[:max_category_prompts]
    _emit_debug(progress_cb, f"[SG-DEBUG] category prompt count={len(prompt_items)}")
    backend_cfg = dict(cfg.get("backend") or {})
    discover_timeout_sec = int(backend_cfg.get("external_timeout_sec", 1800) or 1800)

    backend = None
    loaded_detection_summary: Dict[str, object] = {}
    if isinstance(precomputed_detections, dict) and precomputed_detections:
        raw_props = flatten_post_threshold_records(precomputed_detections)
        refine_rows: List[Dict[str, object]] = []
        raw_sam_only = bool((cfg.get("ablation") or {}).get("use_raw_sam_results", False))
        loaded_detection_summary = summarize_detection_payload(precomputed_detections)
        _emit_debug(
            progress_cb,
            "[SG-DEBUG] using precomputed detections "
            f"loaded_count={len(raw_props)} prompts={int(loaded_detection_summary.get('prompt_count', 0) or 0)} "
            f"source={str(precomputed_detections.get('_loaded_from', '') or '')}",
        )
        proposal_t0 = time.perf_counter()
        if raw_sam_only:
            proposals = [dict(x) for x in list(raw_props or []) if isinstance(x, dict)]
            merged_count = len(proposals)
            dropped_props = []
            _emit_profile(
                progress_cb,
                "proposal_and_filtering",
                time.perf_counter() - proposal_t0,
                merged=merged_count,
                kept=len(proposals),
                dropped=0,
                mode="precomputed_raw_sam_only",
            )
        else:
            proposals = build_entity_proposals(
                raw_props,
                merge_mask_iou_threshold=float(proposal_cfg.get("merge_mask_iou_threshold", 0.75)),
                image_wh=image_size,
                risk_weights=dict(proposal_cfg.get("risk_weights") or {}),
                low_confidence_threshold=float(proposal_cfg.get("low_confidence_threshold", 0.45)),
                small_area_ratio=float(proposal_cfg.get("small_object_area_ratio", 0.001)),
                thin_min_dim=int(proposal_cfg.get("thin_object_min_dim", 8)),
            )
            merged_count = len(proposals)
            proposals, dropped_props = _apply_sports_filtering(
                proposals,
                image_size=image_size,
                proposal_cfg=dict(proposal_cfg),
                progress_cb=progress_cb,
            )
            _emit_profile(
                progress_cb,
                "proposal_and_filtering",
                time.perf_counter() - proposal_t0,
                merged=merged_count,
                kept=len(proposals),
                dropped=len(dropped_props),
                mode="precomputed",
            )
    else:
        backend = _backend_from_cfg(cfg, repo_root=repo_root, progress_cb=progress_cb, cancel_cb=cancel_cb)
        # Person-first pass: align with strong SAM3 baseline usage (prompt='person').
        person_focus_enabled = bool(proposal_cfg.get("person_focus_pass", True))
        person_focus_rows: List[Dict[str, object]] = []
        if person_focus_enabled:
            person_prompt_items = _person_focus_prompt_items(ontology, proposal_cfg=dict(proposal_cfg))
            if person_prompt_items:
                person_t0 = time.perf_counter()
                _emit_debug(
                    progress_cb,
                    f"[SG-DEBUG] person-focus discover start prompts={len(person_prompt_items)} timeout={discover_timeout_sec}s",
                )
                person_focus_rows = backend.discover_entities_by_category(image_path, person_prompt_items)
                for row in person_focus_rows:
                    if isinstance(row, dict):
                        row["person_focus_pass"] = True
                _emit_profile(
                    progress_cb,
                    "discover_person_focus",
                    time.perf_counter() - person_t0,
                    prompts=len(person_prompt_items),
                    candidates=len(person_focus_rows),
                )

        discover_t0 = time.perf_counter()
        _emit_debug(
            progress_cb,
            f"[SG-DEBUG] main discover start prompts={len(prompt_items)} timeout={discover_timeout_sec}s",
        )
        raw_props_main: List[Dict[str, object]] = []
        main_discover_ok = False
        try:
            raw_props_main = backend.discover_entities_by_category(image_path, prompt_items)
            main_discover_ok = True
        except Exception as exc:
            if callable(cancel_cb) and bool(cancel_cb()):
                raise RuntimeError("Scene graph run cancelled by user.") from exc
            _emit_debug(progress_cb, f"[SG-DEBUG] main discover failed in batch mode: {exc}")
            _emit_debug(progress_cb, "[SG-DEBUG] fallback: discover prompts one-by-one (skip failed prompts)")
            for i, item in enumerate(list(prompt_items), start=1):
                if callable(cancel_cb):
                    try:
                        if bool(cancel_cb()):
                            raise RuntimeError("Scene graph run cancelled by user.")
                    except Exception:
                        raise
                prompt = str(item.get("prompt", "") or "").strip()
                label = str(item.get("canonical_label", "") or "").strip().lower()
                if not prompt or not label:
                    continue
                _emit_debug(progress_cb, f"[SG-DEBUG] prompt fallback {i}/{len(prompt_items)} start label={label} prompt={prompt}")
                try:
                    rows_i = backend.discover_entities_by_category(image_path, [{"canonical_label": label, "prompt": prompt}])
                    raw_props_main.extend([dict(x) for x in list(rows_i or []) if isinstance(x, dict)])
                    _emit_debug(progress_cb, f"[SG-DEBUG] prompt fallback {i}/{len(prompt_items)} done candidates={len(rows_i)}")
                except Exception as item_exc:
                    _emit_debug(progress_cb, f"[SG-DEBUG] prompt fallback {i}/{len(prompt_items)} failed: {item_exc}")
        _emit_profile(
            progress_cb,
            "discover_entities_by_category",
            time.perf_counter() - discover_t0,
            prompts=len(prompt_items),
            candidates=len(raw_props_main),
            mode=("batch" if main_discover_ok else "fallback_per_prompt"),
        )

        prepend_person = bool(proposal_cfg.get("person_focus_prepend", True))
        if prepend_person:
            raw_props = list(person_focus_rows) + list(raw_props_main)
        else:
            raw_props = list(raw_props_main) + list(person_focus_rows)
        raw_sam_only = bool((cfg.get("ablation") or {}).get("use_raw_sam_results", False))
        human_fallback_enabled = bool(proposal_cfg.get("human_priority_fallback_pass", True))
        if human_fallback_enabled and (not raw_sam_only) and (not any(is_human_like(row) for row in raw_props)):
            _emit_debug(progress_cb, "[SG-DEBUG] no human-like raw candidates found; running human-priority fallback prompts")
            fallback_t0 = time.perf_counter()
            extra_raw = backend.discover_entities_by_category(image_path, human_priority_prompts)
            raw_props.extend(extra_raw)
            _emit_profile(progress_cb, "fallback_human_prompts", time.perf_counter() - fallback_t0, prompts=len(human_priority_prompts), rescued=len(extra_raw))

        do_sentence = bool(enable_sentence_refine) or bool(ablation.get("use_sentence_refinement", False))
        refine_t0 = time.perf_counter()
        refine_rows = backend.refine_with_sentence_prompts(
            image_path,
            sentence_prompts,
            enable_two_stage_refinement=do_sentence,
        )
        _emit_profile(progress_cb, "sentence_refine", time.perf_counter() - refine_t0, enabled=bool(do_sentence), prompts=len(sentence_prompts), rows=len(refine_rows))

        # Sentence refine rows can be linked as provenance signals only.
        for row in refine_rows:
            row["provenance"] = [
                {
                    "backend": provenance_backend,
                    "stage": "sentence_refine",
                    "prompt": row.get("prompt_used", ""),
                    "image_path": image_path,
                }
            ]

        proposal_t0 = time.perf_counter()
        if raw_sam_only:
            proposals = [dict(x) for x in raw_props if isinstance(x, dict)]
            merged_count = len(proposals)
            dropped_props = []
            _emit_profile(
                progress_cb,
                "proposal_and_filtering",
                time.perf_counter() - proposal_t0,
                merged=merged_count,
                kept=len(proposals),
                dropped=0,
                mode="raw_sam_only",
            )
            _log_candidate_rows(progress_cb, title="raw sam proposals (no filtering)", rows=proposals, image_size=image_size)
        else:
            proposals = build_entity_proposals(
                raw_props,
                merge_mask_iou_threshold=float(proposal_cfg.get("merge_mask_iou_threshold", 0.75)),
                image_wh=image_size,
                risk_weights=dict(proposal_cfg.get("risk_weights") or {}),
                low_confidence_threshold=float(proposal_cfg.get("low_confidence_threshold", 0.45)),
                small_area_ratio=float(proposal_cfg.get("small_object_area_ratio", 0.001)),
                thin_min_dim=int(proposal_cfg.get("thin_object_min_dim", 8)),
            )
            merged_count = len(proposals)
            _log_candidate_rows(progress_cb, title="merged proposals", rows=proposals, image_size=image_size)

            filtered_props, dropped_props = _apply_sports_filtering(
                proposals,
                image_size=image_size,
                proposal_cfg=dict(proposal_cfg),
                progress_cb=progress_cb,
            )
            _emit_profile(progress_cb, "proposal_and_filtering", time.perf_counter() - proposal_t0, merged=merged_count, kept=len(filtered_props), dropped=len(dropped_props))
            _log_candidate_rows(progress_cb, title="final kept proposals", rows=filtered_props, image_size=image_size)
            if dropped_props:
                _emit_debug(progress_cb, f"[SG-DEBUG] dropped proposals: {len(dropped_props)}")
                for row in dropped_props:
                    _emit_debug(
                        progress_cb,
                        f"[SG-DEBUG]   drop label={row.get('canonical_label','')} prompt={row.get('prompt_used','')} "
                        f"score={float(row.get('score',0.0)):.3f} risk={float(row.get('risk',0.0)):.3f} reason={row.get('drop_reason','unknown')}",
                    )
            proposals = filtered_props

            if not any(is_human_like(item) for item in proposals):
                human_pool = [row for row in build_entity_proposals(
                    raw_props,
                    merge_mask_iou_threshold=float(proposal_cfg.get("human_merge_mask_iou_threshold", proposal_cfg.get("merge_mask_iou_threshold", 0.75))),
                    image_wh=image_size,
                    risk_weights=dict(proposal_cfg.get("risk_weights") or {}),
                    low_confidence_threshold=float(proposal_cfg.get("human_low_confidence_threshold", proposal_cfg.get("low_confidence_threshold", 0.45))),
                    small_area_ratio=float(proposal_cfg.get("small_object_area_ratio", 0.001)),
                    thin_min_dim=int(proposal_cfg.get("thin_object_min_dim", 8)),
                ) if is_human_like(row)]
                if human_pool:
                    human_pool.sort(key=lambda x: float(_proposal_priority(x, image_size)), reverse=True)
                    rescue_count = max(1, int(proposal_cfg.get("fallback_human_keep_count", 3)))
                    proposals.extend(human_pool[:rescue_count])
                    _emit_debug(progress_cb, f"[SG-DEBUG] fallback retained {min(rescue_count, len(human_pool))} human candidates after empty-human result")

    for row in raw_props:
        canonical, meta = ontology.canonicalize_label(str(row.get("canonical_label", "")))
        if canonical:
            row["canonical_label"] = canonical
        row["ontology_ambiguous"] = bool(meta.get("ambiguous", False))
        row["provenance"] = [
            {
                "backend": provenance_backend,
                "stage": row.get("stage", "category_discovery"),
                "prompt": row.get("prompt_used", ""),
                "image_path": image_path,
            }
        ]
    _log_prompt_presence(progress_cb, raw_props)
    _log_candidate_rows(progress_cb, title="raw pre-filter", rows=raw_props, image_size=image_size)
    # Give stable IDs and default flags before graph build.
    for item in proposals:
        item.setdefault("entity_id", f"ent_{uuid.uuid4().hex[:10]}")
        item.setdefault("verified", False)
        item.setdefault("validator_flags", [])

    relation_cfg = cfg.get("relations") or {}
    graph_t0 = time.perf_counter()
    graph = build_scene_graph(
        image_id=image_id,
        proposals=proposals,
        relation_vocab=ontology.relation_vocabulary,
        touching_iou_epsilon=float(relation_cfg.get("touching_iou_epsilon", 0.02)),
        pairwise_max=int(relation_cfg.get("pairwise_max", 200)),
        enable_interaction_relations=bool(relation_cfg.get("enable_interaction_relations", False)),
        interaction_relation_hook=None,
    )
    _emit_profile(progress_cb, "build_scene_graph", time.perf_counter() - graph_t0, nodes=len(list(graph.get("nodes") or [])), edges=len(list(graph.get("edges") or [])))

    # Attribute extraction constrained by ontology slots.
    attr_cfg = cfg.get("attributes") or {}
    attr_t0 = time.perf_counter()
    graph["nodes"] = extract_attributes_for_nodes(
        graph.get("nodes") or [],
        ontology=ontology,
        default_confidence=float(attr_cfg.get("default_confidence", 0.35)),
        allow_affordance=bool(attr_cfg.get("allow_affordance", True)),
    )
    _emit_profile(progress_cb, "extract_attributes", time.perf_counter() - attr_t0, nodes=len(list(graph.get("nodes") or [])))

    use_validator = bool((cfg.get("ablation") or {}).get("use_validator", True))
    if use_validator:
        validator_t0 = time.perf_counter()
        val = validate_scene_graph(graph, ontology=ontology, cfg=dict(cfg.get("validators") or {}))
        for node in graph.get("nodes") or []:
            nid = str(node.get("entity_id"))
            node["validator_flags"] = val.get("node_flags", {}).get(nid, [])
        for edge in graph.get("edges") or []:
            eid = str(edge.get("edge_id"))
            edge["validator_flags"] = val.get("edge_flags", {}).get(eid, [])
        graph["validator_flags"] = val.get("graph_flags", [])
        _emit_profile(
            progress_cb,
            "validate_scene_graph",
            time.perf_counter() - validator_t0,
            node_flags=sum(len(list(x or [])) for x in dict(val.get("node_flags") or {}).values()),
            edge_flags=sum(len(list(x or [])) for x in dict(val.get("edge_flags") or {}).values()),
            graph_flags=len(list(val.get("graph_flags") or [])),
        )
    else:
        graph["validator_flags"] = []

    frame_summary = _build_global_semantic_summary(graph)
    graph["metadata"] = {
        "image_path": image_path,
        "graph_snapshot_id": f"graph_{uuid.uuid4().hex[:12]}",
        "ablation": cfg.get("ablation") or {},
        "backend_provider": (cfg.get("backend") or {}).get("provider", "mock"),
        "global_summary": frame_summary,
        "global_semantic_summary": frame_summary,
        "stage_summary_statements": [frame_summary],
        "candidate_debug": {
            "raw_count": len(raw_props),
            "merged_count": merged_count,
            "kept_count": len(proposals),
            "dropped_count": len(dropped_props),
            "human_kept_count": len([x for x in proposals if is_human_like(x)]),
            "human_raw_count": len([x for x in raw_props if is_human_like(x)]),
            "loaded_from_detection_file": bool(isinstance(precomputed_detections, dict) and precomputed_detections),
            "loaded_detection_path": str((precomputed_detections or {}).get("_loaded_from", "") if isinstance(precomputed_detections, dict) else ""),
            "loaded_detection_prompt_count": int(loaded_detection_summary.get("prompt_count", 0) or 0),
        },
    }
    _emit_profile(progress_cb, "run_build_scene_graph_total", time.perf_counter() - total_t0, image_id=image_id, backend=provenance_backend)
    return graph


def run_generate_vqa(graph: Dict[str, object], pipeline_cfg_path: str) -> Dict[str, object]:
    cfg = load_json(pipeline_cfg_path)
    vqa_cfg = cfg.get("vqa") or {}

    single = generate_single_turn_vqa(graph, max_questions=int(vqa_cfg.get("single_turn_max_questions", 64)))
    multi = generate_multi_turn_vqa(graph, max_chains=int(vqa_cfg.get("multi_turn_max_chains", 24)))

    all_items = list(single) + list(multi)
    all_items = validate_vqa_evidence(all_items, graph)

    queue = build_review_queue(graph, all_items)

    return {
        "graph_snapshot_id": ((graph.get("metadata") or {}).get("graph_snapshot_id") or graph.get("image_id")),
        "single_turn": single,
        "multi_turn": multi,
        "all": all_items,
        "review_queue": queue,
    }
