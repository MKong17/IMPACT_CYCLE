from __future__ import annotations

import json
import os
from typing import Dict, List


def _strip_code_fence(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
    return raw


class Qwen25VLVerifier:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        *,
        max_new_tokens: int = 256,
        device_map: str = "auto",
    ) -> None:
        try:
            from PIL import Image  # type: ignore
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Qwen25VLVerifier requires pillow and transformers to be installed."
            ) from exc

        self._image_cls = Image
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.max_new_tokens = int(max_new_tokens)

    def _run(self, messages: List[Dict[str, object]], image_path: str, max_new_tokens: int) -> str:
        abs_image_path = os.path.abspath(os.path.expanduser(image_path))
        if not os.path.isfile(abs_image_path):
            raise FileNotFoundError(abs_image_path)
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image = self._image_cls.open(abs_image_path).convert("RGB")
        inputs = self.processor(
            text=[prompt_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return str(output).strip()

    def answer_probe(
        self,
        *,
        image_path: str,
        question: str,
        regions: List[Dict[str, object]],
        response_format: Dict[str, object] | None = None,
        schema: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        region_text = ""
        if regions:
            region_text = "\nRegions:\n" + "\n".join(
                f"- {row.get('entity_id')}: bbox={row.get('bbox')}" for row in regions
            )
        fmt = dict(response_format or {})
        if str(fmt.get("type", "") or "").strip().lower() == "selection":
            options = [str(x).strip() for x in list(fmt.get("options") or []) if str(x).strip()]
            option_text = ", ".join(options) if options else "the listed options"
            prompt = (
                f"{question}{region_text}\n"
                f"Choose exactly one canonical option from: {option_text}. "
                'If the evidence is insufficient, choose "uncertain". '
                'Return JSON only: {"selection":"<one option or uncertain>","reason":"...","score":0.0}'
            )
        else:
            prompt = (
                f"{question}{region_text}\n"
                'Return JSON only: {"answer":"yes|no|uncertain","reason":"...","score":0.0}'
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": os.path.abspath(os.path.expanduser(image_path))},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._run(messages, image_path, self.max_new_tokens)
        payload = {"raw_text": text}
        try:
            parsed = json.loads(_strip_code_fence(text))
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            payload.update(parsed)
            payload["schema_valid"] = True
        else:
            payload["schema_valid"] = False
        return payload

    def generate_caption(
        self,
        *,
        image_path: str,
        prompt: str,
        regions: List[Dict[str, object]],
        video_or_frames: object = None,
        schema: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        region_text = ""
        if regions:
            region_text = "\nVisible regions:\n" + "\n".join(
                f"- {row.get('entity_id')}: bbox={row.get('bbox')}" for row in regions
            )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": os.path.abspath(os.path.expanduser(image_path))},
                    {"type": "text", "text": str(prompt or "") + region_text},
                ],
            }
        ]
        text = self._run(messages, image_path, self.max_new_tokens)
        cleaned = _strip_code_fence(text)
        payload = {"caption": cleaned, "raw_text": cleaned}
        try:
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            payload.update(parsed)
            if "caption" not in payload:
                payload["caption"] = cleaned
            payload["schema_valid"] = True
        else:
            payload["schema_valid"] = False
        return payload
