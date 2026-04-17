from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_PROMPT = (
    "你是视频时序追踪助手。请仅输出JSON对象，不要输出额外解释。"
    "基于输入图片，重点关注person并输出: "
    "person_attributes(列表，包含person_id, gender, age_range, clothing, appearance_features, confidence), "
    "global_semantic_summary(字符串), "
    "tracking_text(字符串，给下一批继续追踪用，需包含跨帧身份连续性与变化点)。"
)


@dataclass
class BatchResult:
    batch_index: int
    frame_indices: List[int]
    frame_paths: List[str]
    person_attributes: List[Dict[str, Any]]
    global_semantic_summary: str
    tracking_text: str
    raw_response: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_index": int(self.batch_index),
            "frame_indices": list(self.frame_indices),
            "frame_paths": list(self.frame_paths),
            "person_attributes": list(self.person_attributes),
            "global_semantic_summary": str(self.global_semantic_summary),
            "tracking_text": str(self.tracking_text),
            "raw_response": str(self.raw_response),
        }


def chunk_sequence(values: Sequence[Any], chunk_size: int) -> List[List[Any]]:
    size = max(1, int(chunk_size))
    out: List[List[Any]] = []
    for i in range(0, len(values), size):
        out.append(list(values[i : i + size]))
    return out


def build_carry_context(previous: Optional[BatchResult]) -> str:
    if previous is None:
        return (
            "这是第一批帧，请初始化人物追踪。"
            "重点提取person属性: 性别、年龄段、着装、外观特征，并总结全局语义。"
        )
    payload = {
        "previous_batch_index": int(previous.batch_index),
        "previous_frame_indices": list(previous.frame_indices),
        "previous_person_attributes": list(previous.person_attributes),
        "previous_global_semantic_summary": str(previous.global_semantic_summary),
        "previous_tracking_text": str(previous.tracking_text),
    }
    return (
        "以下是上一批追踪上下文(JSON):\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
        "请结合当前批次图片继续保持person跨批次ID连续。"
    )


def _safe_json_extract(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass
    first = raw.find("{")
    last = raw.rfind("}")
    if first >= 0 and last > first:
        clip = raw[first : last + 1]
        try:
            return json.loads(clip)
        except Exception:
            return {}
    return {}


class QwenVLRunner:
    def __init__(
        self,
        *,
        model_path: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> None:
        self.model_path = os.path.abspath(os.path.expanduser(str(model_path)))
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self._processor = None
        self._model = None

    def _lazy_load(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Qwen runner requires torch + transformers with vision-language support."
            ) from exc

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    def infer(self, *, image_paths: Sequence[str], prompt: str) -> str:
        self._lazy_load()
        assert self._processor is not None
        assert self._model is not None
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise RuntimeError("Pillow is required for Qwen image inference.") from exc

        images = [Image.open(path).convert("RGB") for path in image_paths]
        try:
            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            text = self._processor.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._processor(text=[text], images=images, return_tensors="pt")
            model_device = getattr(self._model, "device", None)
            if model_device is not None:
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max(32, int(self.max_new_tokens)),
                do_sample=bool(self.temperature > 0.0),
                temperature=max(1e-5, float(self.temperature)),
            )
            input_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
            trimmed = generated_ids[:, input_len:] if input_len > 0 else generated_ids
            output = self._processor.batch_decode(trimmed, skip_special_tokens=True)
            return str(output[0]).strip() if output else ""
        finally:
            for image in images:
                try:
                    image.close()
                except Exception:
                    pass


def run_temporal_tracking(
    *,
    frame_paths: Sequence[str],
    frame_indices: Sequence[int],
    qwen_runner: QwenVLRunner,
    batch_size: int = 5,
    system_prompt: str = DEFAULT_PROMPT,
) -> Dict[str, Any]:
    if len(frame_paths) != len(frame_indices):
        raise ValueError("frame_paths and frame_indices length mismatch.")
    if not frame_paths:
        return {"batches": [], "metadata": {"num_batches": 0}}

    path_chunks = chunk_sequence(frame_paths, chunk_size=batch_size)
    idx_chunks = chunk_sequence(frame_indices, chunk_size=batch_size)
    if len(path_chunks) != len(idx_chunks):
        raise RuntimeError("internal chunking mismatch")

    results: List[BatchResult] = []
    for batch_idx, (paths, indices) in enumerate(zip(path_chunks, idx_chunks)):
        previous = results[-1] if results else None
        carry = build_carry_context(previous)
        prompt = (
            f"{system_prompt}\n\n"
            f"当前批次编号: {batch_idx}\n"
            f"当前批次帧索引: {list(indices)}\n"
            f"{carry}\n\n"
            "请输出JSON对象，结构示例："
            "{\"person_attributes\": [], \"global_semantic_summary\": \"\", \"tracking_text\": \"\"}"
        )
        raw = qwen_runner.infer(image_paths=paths, prompt=prompt)
        obj = _safe_json_extract(raw)
        attrs = obj.get("person_attributes")
        if not isinstance(attrs, list):
            attrs = []
        summary = str(obj.get("global_semantic_summary", "") or "").strip()
        tracking_text = str(obj.get("tracking_text", "") or "").strip()
        if not tracking_text:
            tracking_text = summary
        results.append(
            BatchResult(
                batch_index=int(batch_idx),
                frame_indices=[int(x) for x in indices],
                frame_paths=[str(x) for x in paths],
                person_attributes=[x for x in attrs if isinstance(x, dict)],
                global_semantic_summary=summary,
                tracking_text=tracking_text,
                raw_response=raw,
            )
        )

    return {
        "batches": [x.to_dict() for x in results],
        "metadata": {
            "num_frames": int(len(frame_paths)),
            "num_batches": int(len(results)),
            "batch_size": int(max(1, batch_size)),
            "frame_start": int(frame_indices[0]),
            "frame_end": int(frame_indices[-1]),
            "carry_strategy": "previous_tracking_text_and_metadata",
            "summary": str(results[-1].global_semantic_summary if results else ""),
            "tracking_text": str(results[-1].tracking_text if results else ""),
        },
    }


def estimate_total_batches(num_frames: int, batch_size: int) -> int:
    n = max(0, int(num_frames))
    b = max(1, int(batch_size))
    return int(math.ceil(float(n) / float(b)))
