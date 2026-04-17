from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class PromptBank:
    category_prompts: List[Dict[str, str]]
    sentence_prompts: List[str]


@dataclass
class Ontology:
    canonical_entities: List[Dict[str, object]]
    relation_vocabulary: Dict[str, List[str]]
    question_types: List[str]
    category_prompt_templates: List[str]
    sentence_prompt_templates: List[str]
    canonical_to_synonyms: Dict[str, List[str]]
    synonym_to_canonical: Dict[str, str]

    def canonicalize_label(self, raw_label: str) -> Tuple[Optional[str], Dict[str, object]]:
        text = str(raw_label or "").strip().lower()
        if not text:
            return None, {"ambiguous": False, "candidates": []}
        if text in self.synonym_to_canonical:
            return self.synonym_to_canonical[text], {"ambiguous": False, "candidates": [self.synonym_to_canonical[text]]}

        candidates: List[str] = []
        for canon, synonyms in self.canonical_to_synonyms.items():
            if text == canon:
                candidates.append(canon)
                continue
            for syn in synonyms:
                if text in syn or syn in text:
                    candidates.append(canon)
                    break
        uniq = []
        for item in candidates:
            if item not in uniq:
                uniq.append(item)
        if not uniq:
            return None, {"ambiguous": False, "candidates": []}
        if len(uniq) == 1:
            return uniq[0], {"ambiguous": False, "candidates": uniq}
        return uniq[0], {"ambiguous": True, "candidates": uniq}

    def attribute_slots_for_label(self, canonical_label: str) -> List[str]:
        name = str(canonical_label or "").strip().lower()
        for item in self.canonical_entities:
            if str(item.get("label", "")).strip().lower() == name:
                slots = item.get("attribute_slots") or []
                if isinstance(slots, list):
                    return [str(x) for x in slots if str(x).strip()]
        return []

    def mandatory_attributes_for_label(self, canonical_label: str) -> List[str]:
        name = str(canonical_label or "").strip().lower()
        for item in self.canonical_entities:
            if str(item.get("label", "")).strip().lower() == name:
                slots = item.get("mandatory_attributes") or []
                if isinstance(slots, list):
                    return [str(x) for x in slots if str(x).strip()]
        return []

    def build_prompt_bank(self) -> PromptBank:
        category_prompts: List[Dict[str, str]] = []
        sentence_prompts: List[str] = []

        labels = [str(item.get("label", "")).strip() for item in self.canonical_entities]
        labels = [x for x in labels if x]
        seen_category: set[tuple[str, str]] = set()
        for item in self.canonical_entities:
            label = str(item.get("label", "")).strip()
            if not label:
                continue
            prompt_variants = [label]
            raw_variants = item.get("prompt_variants") or []
            if isinstance(raw_variants, list):
                for value in raw_variants:
                    text = str(value or "").strip()
                    if text and text not in prompt_variants:
                        prompt_variants.append(text)
            for tmpl in self.category_prompt_templates:
                for variant in prompt_variants:
                    prompt = str(tmpl).format(label=variant)
                    key = (label.lower(), prompt.strip().lower())
                    if not prompt.strip() or key in seen_category:
                        continue
                    seen_category.add(key)
                    category_prompts.append({"canonical_label": label, "prompt": prompt})

        # Expand sentence templates with simple pairwise combinations.
        for label in labels:
            for other in labels:
                if label == other:
                    continue
                for tmpl in self.sentence_prompt_templates:
                    sentence_prompts.append(
                        str(tmpl).format(label=label, other_label=other, attribute="distinctive attribute")
                    )

        return PromptBank(category_prompts=category_prompts, sentence_prompts=sentence_prompts)


def _build_maps(canonical_entities: List[Dict[str, object]]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    c2s: Dict[str, List[str]] = {}
    s2c: Dict[str, str] = {}
    for item in canonical_entities:
        canonical = str(item.get("label", "")).strip().lower()
        if not canonical:
            continue
        synonyms = item.get("synonyms") or []
        if not isinstance(synonyms, list):
            synonyms = []
        syn_norm: List[str] = []
        for syn in synonyms:
            token = str(syn).strip().lower()
            if token and token not in syn_norm:
                syn_norm.append(token)
        c2s[canonical] = syn_norm
        s2c[canonical] = canonical
        for syn in syn_norm:
            if syn not in s2c:
                s2c[syn] = canonical
    return c2s, s2c


def load_ontology(path: str) -> Ontology:
    abs_path = os.path.abspath(os.path.expanduser(path))
    with open(abs_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return ontology_from_payload(payload)


def ontology_from_payload(payload: Dict[str, object]) -> Ontology:
    canonical_entities = payload.get("canonical_entities") or []
    relation_vocab = payload.get("relation_vocabulary") or {}
    question_types = payload.get("question_types") or []
    category_templates = payload.get("category_prompt_templates") or ["{label}"]
    sentence_templates = payload.get("sentence_prompt_templates") or []

    c2s, s2c = _build_maps(canonical_entities)

    return Ontology(
        canonical_entities=canonical_entities,
        relation_vocabulary=relation_vocab,
        question_types=[str(x) for x in question_types],
        category_prompt_templates=[str(x) for x in category_templates],
        sentence_prompt_templates=[str(x) for x in sentence_templates],
        canonical_to_synonyms=c2s,
        synonym_to_canonical=s2c,
    )
