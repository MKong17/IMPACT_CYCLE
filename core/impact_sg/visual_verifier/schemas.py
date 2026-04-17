from __future__ import annotations

from typing import Dict, List


def binary_answer_schema(*, allow_uncertain: bool = True) -> Dict[str, object]:
    enum_values = ["yes", "no"] + (["uncertain"] if allow_uncertain else [])
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": enum_values},
            "reason": {"type": "string"},
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["answer", "reason", "score"],
        "additionalProperties": False,
    }


BINARY_ANSWER_SCHEMA: Dict[str, object] = binary_answer_schema(allow_uncertain=True)


def selection_answer_schema(candidates: List[str], *, allow_uncertain: bool = True) -> Dict[str, object]:
    options = [str(x).strip() for x in list(candidates or []) if str(x).strip()]
    if allow_uncertain:
        options.append("uncertain")
    if not options:
        options = ["uncertain"] if allow_uncertain else [""]
    options = [x for x in options if x]
    if not options:
        options = ["uncertain"]
    return {
        "type": "object",
        "properties": {
            "selection": {"type": "string", "enum": options},
            "reason": {"type": "string"},
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["selection", "reason", "score"],
        "additionalProperties": False,
    }


CAPTION_FEEDBACK_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "caption": {"type": "string"},
        "supported_entities": {"type": "array", "items": {"type": "string"}},
        "unsupported_entities": {"type": "array", "items": {"type": "string"}},
        "supported_relations": {"type": "array", "items": {"type": "string"}},
        "unsupported_relations": {"type": "array", "items": {"type": "string"}},
        "supported_attributes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "slot": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["entity_id", "slot", "value"],
                "additionalProperties": False,
            },
        },
        "unsupported_attributes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string"},
                    "slot": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["entity_id", "slot", "value"],
                "additionalProperties": False,
            },
        },
        "hallucinated_mentions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "caption",
        "supported_entities",
        "supported_relations",
    ],
    "additionalProperties": False,
}


def probe_allows_uncertain(probe: Dict[str, object]) -> bool:
    response_format = dict(probe.get("response_format") or {})
    if "allow_uncertain" in response_format:
        return bool(response_format.get("allow_uncertain"))
    if "allow_uncertain" in probe:
        return bool(probe.get("allow_uncertain"))
    question = str(probe.get("question", "") or "").strip().lower()
    if "do not say uncertain" in question:
        return False
    if "answer yes or no" in question:
        return False
    return True


def probe_response_schema(probe: Dict[str, object]) -> Dict[str, object]:
    response_format = dict(probe.get("response_format") or {})
    fmt_type = str(response_format.get("type", "") or "").strip().lower()
    if fmt_type == "selection":
        options = [str(x).strip() for x in list(probe.get("candidate_options") or []) if str(x).strip()]
        if not options:
            options = [str(x).strip() for x in list(response_format.get("options") or []) if str(x).strip()]
        return selection_answer_schema(options, allow_uncertain=bool(response_format.get("allow_uncertain", True)))
    return binary_answer_schema(allow_uncertain=probe_allows_uncertain(probe))
