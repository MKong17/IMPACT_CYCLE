from __future__ import annotations

import json
from typing import Dict, List, Protocol


class VisionVerifier(Protocol):
    def answer_probe(
        self,
        *,
        image_path: str,
        question: str,
        regions: List[Dict[str, object]],
        response_format: Dict[str, object] | None = None,
        schema: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        ...

    def generate_caption(
        self,
        *,
        image_path: str,
        prompt: str,
        regions: List[Dict[str, object]],
        video_or_frames: object = None,
        schema: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        ...


class MockVisionVerifier:
    """Deterministic verifier for smoke tests and pipeline bring-up."""

    def answer_probe(
        self,
        *,
        image_path: str,
        question: str,
        regions: List[Dict[str, object]],
        response_format: Dict[str, object] | None = None,
        schema: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        fmt = dict(response_format or {})
        if str(fmt.get("type", "") or "").strip().lower() == "selection":
            default_selection = str(
                fmt.get("default_selection")
                or fmt.get("expected")
                or "uncertain"
            ).strip()
            payload = {
                "selection": default_selection or "uncertain",
                "reason": "mock verifier defaults to graph-consistent canonical selection",
                "score": 0.7,
            }
            return {
                "selection": payload["selection"],
                "reason": payload["reason"],
                "score": payload["score"],
                "raw_text": json.dumps(payload, ensure_ascii=True),
                "schema_valid": True,
            }
        payload = {
            "answer": "yes",
            "reason": "mock verifier defaults to graph-consistent support",
            "score": 0.7,
        }
        return {
            "answer": payload["answer"],
            "reason": payload["reason"],
            "score": payload["score"],
            "raw_text": json.dumps(payload, ensure_ascii=True),
            "schema_valid": True,
        }

    def generate_caption(
        self,
        *,
        image_path: str,
        prompt: str,
        regions: List[Dict[str, object]],
        video_or_frames: object = None,
        schema: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        labels: List[str] = []
        entities: List[str] = []
        attributes: List[Dict[str, str]] = []
        relations: List[Dict[str, str]] = []
        section = ""
        for line in str(prompt or "").splitlines():
            text = line.strip()
            if text == "Nodes:":
                section = "nodes"
                continue
            if text == "Attributes:":
                section = "attributes"
                continue
            if text == "Edges:":
                section = "edges"
                continue
            if not text.startswith("- "):
                continue
            body = text[2:]
            if section == "nodes":
                fields = [part.strip() for part in body.split("|")]
                entity_id = fields[0] if fields else ""
                label = ""
                for field in fields[1:]:
                    if field.startswith("label="):
                        label = field.split("=", 1)[1].strip()
                        break
                if entity_id and entity_id not in entities:
                    entities.append(entity_id)
                if label and label not in labels:
                    labels.append(label)
            elif section == "attributes":
                fields = [part.strip() for part in body.split("|")]
                row: Dict[str, str] = {}
                for field in fields[1:]:
                    if "=" not in field:
                        continue
                    key, value = field.split("=", 1)
                    row[key.strip()] = value.strip()
                if row:
                    attributes.append(row)
            elif section == "edges":
                fields = [part.strip() for part in body.split("|")]
                row = {"edge_id": fields[0] if fields else ""}
                for field in fields[1:]:
                    if "=" not in field:
                        continue
                    key, value = field.split("=", 1)
                    row[key.strip()] = value.strip()
                if row.get("edge_id"):
                    relations.append(row)

        pieces = []
        if labels:
            pieces.append("Visible entities include " + ", ".join(labels) + ".")
        if relations:
            rendered_relations = [
                f"{row.get('src', '').strip()} {row.get('relation', '').strip()} {row.get('dst', '').strip()}".strip()
                for row in relations[:2]
                if str(row.get("relation", "")).strip()
            ]
            rendered_relations = [item for item in rendered_relations if item]
            if rendered_relations:
                pieces.append("Key relations: " + "; ".join(rendered_relations) + ".")
        if not pieces:
            pieces.append("A scene is visible.")
        caption_text = " ".join(pieces)

        if "Return JSON only" not in str(prompt or ""):
            return {"caption": caption_text}

        payload = {
            "caption": caption_text,
            "supported_entities": entities,
            "unsupported_entities": [],
            "supported_attributes": [
                {
                    "entity_id": str(row.get("entity", "") or "").strip(),
                    "slot": str(row.get("slot", "") or "").strip(),
                    "value": str(row.get("value", "") or "").strip(),
                }
                for row in attributes
                if str(row.get("entity", "") or "").strip()
                and str(row.get("slot", "") or "").strip()
                and str(row.get("value", "") or "").strip()
            ],
            "unsupported_attributes": [],
            "supported_relations": [str(row.get("edge_id", "") or "").strip() for row in relations if str(row.get("edge_id", "") or "").strip()],
            "unsupported_relations": [],
            "hallucinated_mentions": [],
        }
        raw_text = json.dumps(payload, ensure_ascii=True)
        return {"caption": raw_text, "raw_text": raw_text, "schema_valid": True}
