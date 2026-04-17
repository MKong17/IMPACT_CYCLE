from __future__ import annotations

import base64
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional

from .defaults import (
    DEFAULT_API_MAX_OUTPUT_TOKENS,
    DEFAULT_API_TIMEOUT_SEC,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
)


class APIVisionVerifier:
    def __init__(
        self,
        *,
        answer_handler: Callable[..., Dict[str, object]] | None = None,
        caption_handler: Callable[[str, str, List[Dict[str, object]]], Dict[str, object]] | None = None,
        provider: str = "generic_api",
        model: str = "",
        base_url: str = "",
        answer_url: str = "",
        caption_url: str = "",
        timeout_sec: int = 120,
        api_key: str = "",
        api_key_env: str = "IMPACT_API_KEY",
        api_key_header: str = "Authorization",
        api_key_prefix: str = "Bearer ",
        extra_headers: Optional[Dict[str, object]] = None,
        include_image_base64: bool = True,
        max_output_tokens: int = DEFAULT_API_MAX_OUTPUT_TOKENS,
        debug_log: Callable[[str], None] | None = None,
    ) -> None:
        self._answer_handler = answer_handler
        self._caption_handler = caption_handler
        self.provider = str(provider or "generic_api").strip() or "generic_api"
        self.model = str(model or "").strip()
        self.base_url = str(base_url or "").strip()
        self.answer_url = str(answer_url or "").strip()
        self.caption_url = str(caption_url or "").strip()
        self.timeout_sec = max(1, int(timeout_sec or DEFAULT_API_TIMEOUT_SEC))
        env_key = str(api_key_env or "IMPACT_API_KEY").strip()
        resolved_api_key = str(api_key or "").strip()
        if not resolved_api_key and env_key:
            resolved_api_key = str(os.environ.get(env_key, "") or "").strip()
        self.api_key = resolved_api_key
        self.api_key_env = env_key
        self.api_key_header = str(api_key_header or "Authorization").strip() or "Authorization"
        self.api_key_prefix = str(api_key_prefix or "")
        self.extra_headers = {
            str(key).strip(): str(value).strip()
            for key, value in dict(extra_headers or {}).items()
            if str(key).strip()
        }
        self.include_image_base64 = bool(include_image_base64)
        self.max_output_tokens = max(16, int(max_output_tokens or DEFAULT_API_MAX_OUTPUT_TOKENS))
        self._is_gemini = self.provider in {"gemini", "google_gemini"}
        self._is_openai = self.provider in {"openai", "chatgpt"}
        self._debug_log = debug_log

    def _log(self, text: str) -> None:
        cb = self._debug_log
        if cb is None:
            return
        try:
            cb(self._sanitize_log_text(str(text or "").strip()))
        except Exception:
            pass

    @staticmethod
    def _sanitize_log_text(text: str) -> str:
        out = str(text or "").strip()
        if not out:
            return out
        # Hide any URL-like tokens entirely.
        out = re.sub(r"https?://\S+", "[REDACTED_URL]", out, flags=re.IGNORECASE)
        # Hide potential API key fragments in free-form text.
        out = re.sub(r"(?i)\b(api[_-]?key|token|access[_-]?token)\s*=\s*\S+", r"\1=[REDACTED]", out)
        # Hide common auth header payloads.
        out = re.sub(r"(?i)\b(authorization)\s*:\s*\S+", r"\1: [REDACTED]", out)
        return out

    def _target_url(self, *, kind: str) -> str:
        if kind == "answer":
            return self.answer_url or self.base_url
        if kind == "caption":
            return self.caption_url or self.base_url
        return self.base_url

    def _request_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        headers.update({key: value for key, value in self.extra_headers.items() if key and value})
        if self.api_key and not self._is_gemini:
            headers[self.api_key_header] = f"{self.api_key_prefix}{self.api_key}"
        return headers

    @staticmethod
    def _encode_image_base64(image_path: str) -> str:
        abs_path = os.path.abspath(os.path.expanduser(image_path))
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(abs_path)
        with open(abs_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def _base_payload(self, *, image_path: str, regions: List[Dict[str, object]]) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "provider": self.provider,
            "model": self.model,
            "image_path": os.path.abspath(os.path.expanduser(image_path)),
            "regions": list(regions or []),
        }
        if self.include_image_base64:
            payload["image_base64"] = self._encode_image_base64(image_path)
        return payload

    def _post_json(self, url: str, payload: Dict[str, object]) -> Dict[str, object]:
        target_url = str(url or "").strip()
        if not target_url:
            raise RuntimeError("API verifier URL is not configured.")
        payload_bytes = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        payload_size = len(payload_bytes)
        payload_keys = list(dict(payload or {}).keys())
        generation_cfg = dict(payload.get("generationConfig") or {})
        last_exc: Exception | None = None
        retries = 3
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            self._log(
                "[CYCLE-API][START] "
                f"provider={self.provider} attempt={attempt + 1}/{retries + 1} timeout={self.timeout_sec}s "
                f"url={self._safe_url_for_log(target_url)} payload_bytes={payload_size} "
                f"payload_keys={payload_keys[:8]} "
                f"max_output_tokens={generation_cfg.get('maxOutputTokens', '')!r} "
                f"response_mime={generation_cfg.get('responseMimeType', '')!r}"
            )
            request = urllib.request.Request(
                target_url,
                data=payload_bytes,
                headers=self._request_headers(),
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    raw = response.read().decode("utf-8")
                parsed = json.loads(raw) if raw else {}
                if not isinstance(parsed, dict):
                    raise RuntimeError("API verifier must return a JSON object.")
                self._log(
                    f"[CYCLE-API][DONE] provider={self.provider} attempt={attempt + 1}/{retries + 1} sec={time.perf_counter() - t0:.3f}"
                )
                return parsed
            except urllib.error.HTTPError as exc:
                last_exc = exc
                error_body = ""
                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    error_body = ""
                self._log(
                    "[CYCLE-API][HTTP] "
                    f"provider={self.provider} code={exc.code} attempt={attempt + 1}/{retries + 1} "
                    f"sec={time.perf_counter() - t0:.3f} "
                    f"reason={str(exc.reason or '')[:120]!r} "
                    f"body={self._sanitize_log_text(str(error_body or '')[:500])!r}"
                )
                if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                    time.sleep(min(8.0, 1.5 * (2**attempt)))
                    continue
                detail = str(exc.reason or "HTTP error").strip()
                body_note = self._sanitize_log_text(str(error_body or "").strip())
                if body_note:
                    detail = f"{detail} | body={body_note[:240]}"
                raise RuntimeError(
                    f"API verifier HTTP error {exc.code} for {self._safe_url_for_log(target_url)}: {detail}"
                ) from exc
            except urllib.error.URLError as exc:
                last_exc = exc
                reason = str(getattr(exc, "reason", "") or "request error").strip()
                is_timeout = isinstance(getattr(exc, "reason", None), socket.timeout) or "timed out" in reason.lower()
                self._log(
                    "[CYCLE-API][URLERR] "
                    f"provider={self.provider} attempt={attempt + 1}/{retries + 1} "
                    f"sec={time.perf_counter() - t0:.3f} timeout={int(bool(is_timeout))} error={reason}"
                )
                if attempt < retries:
                    time.sleep(min(8.0, 1.0 * (2**attempt)))
                    continue
                if is_timeout:
                    raise RuntimeError(
                        f"API verifier request timed out after {self.timeout_sec}s for "
                        f"{self._safe_url_for_log(target_url)}"
                    ) from exc
                raise RuntimeError(
                    f"API verifier request failed for {self._safe_url_for_log(target_url)}: {reason}"
                ) from exc
            except TimeoutError as exc:
                last_exc = exc
                self._log(
                    "[CYCLE-API][TIMEOUT] "
                    f"provider={self.provider} attempt={attempt + 1}/{retries + 1} "
                    f"sec={time.perf_counter() - t0:.3f} timeout={self.timeout_sec}s"
                )
                if attempt < retries:
                    time.sleep(min(8.0, 1.0 * (2**attempt)))
                    continue
                raise RuntimeError(
                    f"API verifier request timed out after {self.timeout_sec}s for "
                    f"{self._safe_url_for_log(target_url)}"
                ) from exc
            except Exception as exc:
                last_exc = exc
                self._log(
                    f"[CYCLE-API][ERROR] provider={self.provider} attempt={attempt + 1}/{retries + 1} sec={time.perf_counter() - t0:.3f} error={str(exc)[:240]}"
                )
                raise RuntimeError(
                    f"API verifier returned invalid JSON from {self._safe_url_for_log(target_url)}: {str(exc)[:200]}"
                ) from exc
        if last_exc is not None:
            raise RuntimeError(
                f"API verifier request failed after retries for {self._safe_url_for_log(target_url)}: "
                f"{type(last_exc).__name__}: {str(last_exc)[:200]}"
            ) from last_exc
        raise RuntimeError("API verifier request failed after retries.")

    @staticmethod
    def _safe_url_for_log(url: str) -> str:
        """Redact sensitive query params (e.g., API key) before logging."""
        raw = str(url or "").strip()
        if not raw:
            return raw
        try:
            parsed = urllib.parse.urlparse(raw)
            q = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
            redacted = []
            for k, v in q:
                key = str(k or "").strip().lower()
                if key in {"key", "api_key", "apikey", "token", "access_token"}:
                    val = str(v or "")
                    masked = (val[:4] + "..." + val[-3:]) if len(val) > 10 else "***"
                    redacted.append((k, masked))
                else:
                    redacted.append((k, v))
            new_query = urllib.parse.urlencode(redacted, doseq=True)
            return urllib.parse.urlunparse(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    parsed.params,
                    new_query,
                    parsed.fragment,
                )
            )
        except Exception:
            return raw

    @staticmethod
    def _extract_json_block(text: str) -> Dict[str, object]:
        raw = str(text or "").strip()
        if not raw:
            return {}
        if raw.startswith("```"):
            lines = [line for line in raw.splitlines() if not line.strip().startswith("```")]
            raw = "\n".join(lines).strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    @staticmethod
    def _extract_json_value(text: str) -> object:
        raw = str(text or "").strip()
        if not raw:
            return {}
        if raw.startswith("```"):
            lines = [line for line in raw.splitlines() if not line.strip().startswith("```")]
            raw = "\n".join(lines).strip()
        try:
            return json.loads(raw)
        except Exception:
            pass
        start_candidates = []
        first_obj = raw.find("{")
        first_arr = raw.find("[")
        if first_obj >= 0:
            start_candidates.append(first_obj)
        if first_arr >= 0:
            start_candidates.append(first_arr)
        if not start_candidates:
            return {}
        start = min(start_candidates)
        obj_end = raw.rfind("}")
        arr_end = raw.rfind("]")
        end = max(obj_end, arr_end)
        if end > start:
            try:
                return json.loads(raw[start : end + 1])
            except Exception:
                return {}
        return {}

    @staticmethod
    def _find_dict_with_keys(payload: object, keys: List[str], *, depth: int = 0) -> Dict[str, object]:
        if depth > 6:
            return {}
        if isinstance(payload, dict):
            lower_keys = {str(k).strip().lower(): k for k in payload.keys()}
            if all(str(k).strip().lower() in lower_keys for k in keys):
                out: Dict[str, object] = {}
                for k in keys:
                    src_k = lower_keys.get(str(k).strip().lower())
                    if src_k is not None:
                        out[str(k)] = payload.get(src_k)
                return out
            for value in payload.values():
                found = APIVisionVerifier._find_dict_with_keys(value, keys, depth=depth + 1)
                if found:
                    return found
        if isinstance(payload, list):
            for item in payload:
                found = APIVisionVerifier._find_dict_with_keys(item, keys, depth=depth + 1)
                if found:
                    return found
        return {}

    @staticmethod
    def _find_first_value(payload: object, keys: List[str], *, depth: int = 0) -> object:
        if depth > 6:
            return None
        wanted = {str(k).strip().lower() for k in list(keys or []) if str(k).strip()}
        if not wanted:
            return None
        if isinstance(payload, dict):
            for k, v in payload.items():
                if str(k).strip().lower() in wanted:
                    return v
            for value in payload.values():
                found = APIVisionVerifier._find_first_value(value, list(wanted), depth=depth + 1)
                if found is not None:
                    return found
        if isinstance(payload, list):
            for item in payload:
                found = APIVisionVerifier._find_first_value(item, list(wanted), depth=depth + 1)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _extract_gemini_text(resp: Dict[str, object]) -> str:
        try:
            candidates = list(resp.get("candidates") or [])
            for cand in candidates:
                content = dict(cand.get("content") or {})
                parts = list(content.get("parts") or [])
                for part in parts:
                    if isinstance(part, str):
                        text = str(part).strip()
                    else:
                        text = str((part or {}).get("text", "") or "").strip()
                    if text:
                        return text
                    # Some gateway adapters may wrap text in alternate keys.
                    alt = str((part or {}).get("output_text", "") or "").strip()
                    if alt:
                        return alt
                    # Some responses carry JSON under non-text keys.
                    json_blob = (part or {}).get("json")
                    if isinstance(json_blob, dict) and json_blob:
                        return json.dumps(json_blob, ensure_ascii=True)
            # Compatibility fallback when Gemini is exposed by an OpenAI-style proxy.
            choices = list(resp.get("choices") or [])
            if choices:
                msg = dict((choices[0] or {}).get("message") or {})
                content = msg.get("content", "")
                if isinstance(content, str) and str(content).strip():
                    return str(content).strip()
                if isinstance(content, list):
                    chunks: List[str] = []
                    for part in content:
                        text = str((part or {}).get("text", "") or "").strip()
                        if text:
                            chunks.append(text)
                    if chunks:
                        return "\n".join(chunks).strip()
            # Last-resort fallback: parse top-level text field if present.
            top_text = str(resp.get("text", "") or "").strip()
            if top_text:
                return top_text
            # Non-standard fallback: recursively search any `text`/`output_text`.
            any_text = APIVisionVerifier._find_first_value(resp, ["text", "output_text", "content"])
            if isinstance(any_text, str) and any_text.strip():
                return any_text.strip()
            if isinstance(any_text, (dict, list)) and any_text:
                return json.dumps(any_text, ensure_ascii=True)
            candidates = list(resp.get("candidates") or [])
            if candidates:
                # Keep debuggable raw text when the gateway returns structured parts without plain text.
                return json.dumps(candidates[0], ensure_ascii=True)
        except Exception:
            return ""
        return ""

    @staticmethod
    def _safe_float(value: object, default: float = 0.5) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _extract_score_from_text(text: str, default: float = 0.5) -> float:
        raw = str(text or "").strip()
        if not raw:
            return float(default)
        patterns = [
            r'"score"\s*:\s*([0-9]*\.?[0-9]+)',
            r"\bscore\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"\bconfidence\s*[:=]\s*([0-9]*\.?[0-9]+)",
        ]
        for pat in patterns:
            m = re.search(pat, raw, flags=re.IGNORECASE)
            if not m:
                continue
            try:
                val = float(m.group(1))
            except Exception:
                continue
            if val > 1.0 and val <= 100.0:
                val = val / 100.0
            return max(0.0, min(1.0, val))
        return float(default)

    @staticmethod
    def _infer_binary_answer_from_text(text: str) -> str:
        raw = str(text or "").strip().lower()
        if not raw:
            return "uncertain"
        if re.search(r"\b(answer|result)\b\s*[:=]\s*\"?(yes|no|uncertain)\"?", raw):
            m = re.search(r"\b(answer|result)\b\s*[:=]\s*\"?(yes|no|uncertain)\"?", raw)
            if m:
                return str(m.group(2) or "uncertain").strip().lower()
        if re.search(r"\bdefinitely\s+no\b|\bnot\s+true\b|\bdoes\s+not\b|\bno\b", raw):
            return "no"
        if re.search(r"\bdefinitely\s+yes\b|\byes\b|\btrue\b|\bcorrect\b", raw):
            return "yes"
        if "uncertain" in raw or "not sure" in raw or "unknown" in raw:
            return "uncertain"
        return "uncertain"

    @staticmethod
    def _extract_reason_from_text(text: str, default: str = "fallback_parse") -> str:
        raw = str(text or "").strip()
        if not raw:
            return str(default)
        parsed = APIVisionVerifier._extract_json_block(raw)
        if isinstance(parsed, dict):
            reason = str(parsed.get("reason", "") or "").strip()
            if reason:
                return reason
        for pat in [
            r'"reason"\s*:\s*"([^"]+)"',
            r"\breason\s*[:=]\s*([^\n\r]+)",
        ]:
            m = re.search(pat, raw, flags=re.IGNORECASE)
            if m:
                reason = str(m.group(1) or "").strip().strip(",")
                if reason:
                    return reason
        compact = re.sub(r"\s+", " ", raw).strip()
        if len(compact) > 180:
            compact = compact[:180].rstrip() + "..."
        return compact or str(default)

    @staticmethod
    def _normalize_binary_answer(value: object) -> str:
        token = str(value or "").strip().lower()
        if token in {"yes", "y", "true", "1", "supported"}:
            return "yes"
        if token in {"no", "n", "false", "0", "conflict"}:
            return "no"
        if token in {"uncertain", "unknown", "not_sure", "not sure", "unsure"}:
            return "uncertain"
        return ""

    @staticmethod
    def _infer_selection_from_text(text: str, options: List[str]) -> str:
        raw = str(text or "").strip().lower()
        if not raw:
            return ""
        normalized = {str(opt or "").strip().lower(): str(opt or "").strip() for opt in list(options or []) if str(opt or "").strip()}
        if not normalized:
            return ""
        m = re.search(r"\b(selection|choice|selected_option|answer)\b\s*[:=]\s*\"?([a-z0-9_ -]+)\"?", raw)
        if m:
            picked = str(m.group(2) or "").strip().lower()
            if picked in normalized:
                return normalized[picked]
        for key, canonical in normalized.items():
            if re.search(rf"\b{re.escape(key)}\b", raw):
                return canonical
        return ""

    @staticmethod
    def _extract_openai_text(resp: Dict[str, object]) -> str:
        try:
            choices = list(resp.get("choices") or [])
            if not choices:
                return ""
            msg = dict((choices[0] or {}).get("message") or {})
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                out: List[str] = []
                for part in content:
                    text = str((part or {}).get("text", "") or "").strip()
                    if text:
                        out.append(text)
                return "\n".join(out).strip()
        except Exception:
            return ""
        return ""

    @staticmethod
    def _regions_hint(regions: List[Dict[str, object]]) -> str:
        rows: List[str] = []
        for idx, region in enumerate(list(regions or [])[:4], start=1):
            if not isinstance(region, dict):
                continue
            label = str(region.get("label", "") or "").strip() or "object"
            entity_id = str(region.get("entity_id", "") or "").strip() or f"obj_{idx}"
            bbox = list(region.get("bbox") or [])
            if len(bbox) >= 4:
                try:
                    x = float(bbox[0] or 0.0)
                    y = float(bbox[1] or 0.0)
                    w = float(bbox[2] or 0.0)
                    h = float(bbox[3] or 0.0)
                    rows.append(f"- {entity_id} ({label}) bbox=[{x:.1f},{y:.1f},{w:.1f},{h:.1f}]")
                    continue
                except Exception:
                    pass
            rows.append(f"- {entity_id} ({label})")
        if not rows:
            return ""
        return "Focus regions in image:\n" + "\n".join(rows) + "\n"

    @staticmethod
    def _normalize_prompt_context(prompt_context: Dict[str, object] | None) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for key, value in dict(prompt_context or {}).items():
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

    @staticmethod
    def _prompt_context_block(prompt_context: Dict[str, object] | None) -> str:
        ctx = APIVisionVerifier._normalize_prompt_context(prompt_context)
        if not ctx:
            return ""
        lines: List[str] = []
        claim_type = str(ctx.get("claim_type", "") or "").strip().lower()
        probe_family = str(ctx.get("probe_family", "") or "").strip().lower()
        if claim_type:
            lines.append(f"- claim type: {claim_type}")
        if probe_family:
            lines.append(f"- probe family: {probe_family}")
        subject_id = str(ctx.get("subject_id", "") or "").strip()
        subject_label = str(ctx.get("subject_label", "") or "").strip()
        object_id = str(ctx.get("object_id", "") or "").strip()
        object_label = str(ctx.get("object_label", "") or "").strip()
        current_value = str(ctx.get("current_value", "") or "").strip()
        slot = str(ctx.get("slot", "") or "").strip()
        relation = str(ctx.get("relation", "") or "").strip()
        if subject_id or subject_label:
            lines.append(f"- subject: {subject_label or 'object'} ({subject_id or 'unknown'})")
        if object_id or object_label:
            lines.append(f"- object: {object_label or 'object'} ({object_id or 'unknown'})")
        if slot:
            lines.append(f"- attribute slot: {slot}")
        if relation:
            lines.append(f"- current relation: {relation}")
        if current_value and current_value != relation:
            lines.append(f"- current graph value: {current_value}")
        if "temporal_anchor_frame_idx" in ctx:
            lines.append(f"- temporal anchor frame: {ctx.get('temporal_anchor_frame_idx')}")
        if not lines:
            return ""
        return "Claim context:\n" + "\n".join(lines) + "\n"

    @staticmethod
    def _claim_guidance_lines(
        prompt_context: Dict[str, object] | None,
        *,
        selection: bool,
    ) -> List[str]:
        ctx = APIVisionVerifier._normalize_prompt_context(prompt_context)
        if not ctx:
            return []
        claim_type = str(ctx.get("claim_type", "") or "").strip().lower()
        probe_family = str(ctx.get("probe_family", "") or "").strip().lower()
        out: List[str] = []
        if claim_type == "label":
            out.append("Judge the highlighted object's canonical category only, not nearby objects or scene priors.")
        elif claim_type == "attribute":
            out.append("Judge only the named attribute slot/value for the target entity, not hidden or inferred state.")
        elif claim_type == "relation":
            out.append("Judge the specific relation between the named subject and object, not general co-occurrence.")
            if bool(ctx.get("is_spatial", False)):
                out.append("This is a spatial relation task. Rely on geometry, relative position, and overlap cues.")
        elif claim_type == "existence":
            out.append("Judge whether the target entity is visibly present at all.")
        if probe_family == "counterfactual_verification":
            out.append("Treat this as a counterfactual alternative. Answer yes only if the alternative is more visually supported.")
        elif probe_family == "temporal_consistency":
            out.append("Use the current image to decide whether the same track or relation still holds now.")
        elif selection and probe_family == "constrained_correction":
            out.append("Choose the best canonical option for the evidence even if it differs from the current graph guess.")
        return out

    @staticmethod
    def _schema_enum_values(schema: Dict[str, object] | None, field_name: str) -> List[str]:
        if not isinstance(schema, dict):
            return []
        props = dict(schema.get("properties") or {})
        field_schema = dict(props.get(str(field_name), {}) or {})
        return [str(x).strip() for x in list(field_schema.get("enum") or []) if str(x).strip()]

    @classmethod
    def _schema_allows_value(
        cls,
        schema: Dict[str, object] | None,
        field_name: str,
        value: str,
        *,
        default: bool,
    ) -> bool:
        enum_values = cls._schema_enum_values(schema, field_name)
        if not enum_values:
            return bool(default)
        return str(value or "").strip() in enum_values

    @staticmethod
    def _transport_response_format(response_format: Dict[str, object] | None) -> Dict[str, object] | None:
        if not isinstance(response_format, dict):
            return None
        cleaned = {
            str(key): value
            for key, value in dict(response_format or {}).items()
            if str(key).strip() and not str(key).startswith("_")
        }
        return cleaned or None

    @staticmethod
    def _build_selection_prompt(
        *,
        question: str,
        regions: List[Dict[str, object]],
        options: List[str],
        default_selection: str,
        prompt_context: Dict[str, object] | None = None,
        allow_uncertain: bool = True,
    ) -> str:
        default_token = str(default_selection or "uncertain").strip() or "uncertain"
        context_block = APIVisionVerifier._prompt_context_block(prompt_context)
        guidance_lines = APIVisionVerifier._claim_guidance_lines(prompt_context, selection=True)
        guidance_block = ""
        if guidance_lines:
            guidance_block = "Extra guidance:\n" + "\n".join(f"- {line}" for line in guidance_lines) + "\n"
        uncertain_line = (
            '- If none of the options is clearly supported, return "uncertain".\n'
            if allow_uncertain
            else ""
        )
        return (
            f"{APIVisionVerifier._regions_hint(regions)}"
            "Use only visible evidence from the image and the claim context.\n"
            f"{context_block}"
            f"{guidance_block}"
            "Task: choose the best canonical option for the highlighted evidence.\n"
            f'- The current graph guess is "{default_token}"; keep it only if visual evidence supports it.\n'
            f"{uncertain_line}"
            "Respond by filling the provided JSON schema.\n"
            "Question:\n"
            f"{str(question or '').strip()}"
        )

    @staticmethod
    def _build_binary_prompt(
        *,
        question: str,
        regions: List[Dict[str, object]],
        prompt_context: Dict[str, object] | None = None,
        allow_uncertain: bool = True,
    ) -> str:
        context_block = APIVisionVerifier._prompt_context_block(prompt_context)
        guidance_lines = APIVisionVerifier._claim_guidance_lines(prompt_context, selection=False)
        guidance_block = ""
        if guidance_lines:
            guidance_block = "Extra guidance:\n" + "\n".join(f"- {line}" for line in guidance_lines) + "\n"
        task_line = (
            "Task: answer the question using only yes, no, or uncertain.\n"
            if allow_uncertain
            else "Task: answer the question using only yes or no.\n"
        )
        uncertain_line = (
            'If evidence is insufficient, use "uncertain".\n'
            if allow_uncertain
            else ""
        )
        return (
            f"{APIVisionVerifier._regions_hint(regions)}"
            "Use only visible evidence from the image and the claim context.\n"
            f"{context_block}"
            f"{guidance_block}"
            f"{task_line}"
            f"{uncertain_line}"
            "Respond by filling the provided JSON schema.\n"
            "Question:\n"
            f"{str(question or '').strip()}"
        )

    @staticmethod
    def _prompt_requests_json(prompt: str, schema: Dict[str, object] | None = None) -> bool:
        if isinstance(schema, dict) and schema:
            return True
        raw = str(prompt or "").strip().lower()
        if not raw:
            return False
        markers = (
            "return json",
            "json only",
            "only json",
            "exact format",
            "one json object",
            "return only a json object",
            "return only one json object",
        )
        return any(marker in raw for marker in markers)

    @staticmethod
    def _build_caption_prompt(
        *,
        prompt: str,
        schema: Dict[str, object] | None = None,
        regions: List[Dict[str, object]] | None = None,
    ) -> tuple[str, bool]:
        base_prompt = str(prompt or "").strip() or "Describe the image briefly in one sentence."
        prefix = APIVisionVerifier._regions_hint(list(regions or []))
        if prefix:
            base_prompt = prefix + "\n" + base_prompt
        expects_json = APIVisionVerifier._prompt_requests_json(base_prompt, schema)
        if not expects_json:
            return base_prompt, False
        return (
            "You are a visual captioning verifier.\n"
            "Follow the task exactly.\n"
            "Answer the task according to the provided JSON schema.\n\n"
            "Task:\n"
            f"{base_prompt}",
            True,
        )

    @staticmethod
    def _json_type_ok(value: object, schema_type: str) -> bool:
        token = str(schema_type or "").strip().lower()
        if token == "object":
            return isinstance(value, dict)
        if token == "array":
            return isinstance(value, list)
        if token == "string":
            return isinstance(value, str)
        if token == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if token == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if token == "boolean":
            return isinstance(value, bool)
        if token == "null":
            return value is None
        return True

    @classmethod
    def _validate_json_schema(cls, value: object, schema: Dict[str, object], path: str = "$") -> List[str]:
        if not isinstance(schema, dict):
            return []
        errors: List[str] = []
        schema_type = str(schema.get("type", "") or "").strip().lower()
        if schema_type and (not cls._json_type_ok(value, schema_type)):
            return [f"{path}: expected {schema_type}"]

        enum_values = schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            if value not in enum_values:
                errors.append(f"{path}: value not in enum")
                return errors

        if isinstance(value, dict):
            props = dict(schema.get("properties") or {})
            required = [str(x).strip() for x in list(schema.get("required") or []) if str(x).strip()]
            for key in required:
                if key not in value:
                    errors.append(f"{path}.{key}: required field missing")
            for key, sub_schema in props.items():
                if key not in value:
                    continue
                errors.extend(cls._validate_json_schema(value.get(key), dict(sub_schema or {}), f"{path}.{key}"))
            if bool(schema.get("additionalProperties") is False):
                allowed = set(props.keys())
                for key in value.keys():
                    if str(key) not in allowed:
                        errors.append(f"{path}.{key}: additional property not allowed")
        elif isinstance(value, list):
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for idx, item in enumerate(value):
                    errors.extend(cls._validate_json_schema(item, item_schema, f"{path}[{idx}]"))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema:
                try:
                    if float(value) < float(schema.get("minimum")):
                        errors.append(f"{path}: below minimum")
                except Exception:
                    pass
            if "maximum" in schema:
                try:
                    if float(value) > float(schema.get("maximum")):
                        errors.append(f"{path}: above maximum")
                except Exception:
                    pass
        return errors

    def _gemini_parse_with_schema(
        self,
        *,
        text: str,
        raw_resp: Dict[str, object],
        schema: Dict[str, object],
        required_keys: List[str],
    ) -> tuple[Dict[str, object], bool]:
        parsed = self._extract_json_block(text)
        if not parsed:
            parsed = self._find_dict_with_keys(raw_resp, required_keys)
        if not isinstance(parsed, dict) or not parsed:
            return {}, False
        schema_errors = self._validate_json_schema(parsed, schema)
        return parsed, len(schema_errors) == 0

    @staticmethod
    def _mime_from_path(path: str) -> str:
        token = str(path or "").strip().lower()
        if token.endswith(".png"):
            return "image/png"
        if token.endswith(".webp"):
            return "image/webp"
        if token.endswith(".gif"):
            return "image/gif"
        if token.endswith(".mp4"):
            return "video/mp4"
        if token.endswith(".mov"):
            return "video/quicktime"
        if token.endswith(".avi"):
            return "video/x-msvideo"
        if token.endswith(".mkv"):
            return "video/x-matroska"
        return "image/jpeg"

    def _gemini_url(self) -> str:
        base = str(self.base_url or "").strip()
        model = str(self.model or DEFAULT_GEMINI_MODEL).strip()
        if base:
            return base
        if not self.api_key:
            raise RuntimeError(f"Gemini API key is missing (set {self.api_key_env} or api_key).")
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"

    def _gemini_post(
        self,
        text_prompt: str,
        image_path: str,
        *,
        response_mime_type: str = "",
        response_json_schema: Dict[str, object] | None = None,
        media_paths: Optional[List[str]] = None,
        max_output_tokens_override: int = 0,
    ) -> Dict[str, object]:
        max_tokens = int(max_output_tokens_override or self.max_output_tokens)
        max_tokens = max(16, int(max_tokens))
        # Structured JSON responses need enough room to finish object generation.
        # Keep this guard centralized so retries share identical safe limits.
        if str(response_mime_type or "").strip().lower() == "application/json" and isinstance(response_json_schema, dict) and response_json_schema:
            max_tokens = max(512, int(max_tokens))
        media_items = [str(x).strip() for x in list(media_paths or []) if str(x).strip()]
        if not media_items and str(image_path or "").strip():
            media_items = [str(image_path).strip()]
        payload: Dict[str, object] = {
            "contents": [
                {
                    "parts": [
                        {"text": str(text_prompt or "")},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": int(max_tokens),
            },
        }
        mime_type = str(response_mime_type or "").strip()
        if mime_type:
            payload["generationConfig"]["responseMimeType"] = mime_type
        if isinstance(response_json_schema, dict) and response_json_schema:
            # v1beta generateContent accepts JSON Schema under `responseJsonSchema`.
            # Sending the same schema via `responseSchema` can be rejected for fields
            # like `additionalProperties`, resulting in HTTP 400.
            payload["generationConfig"]["responseJsonSchema"] = dict(response_json_schema)
        if self.include_image_base64 and media_items:
            appended = 0
            for media_path in media_items[:16]:
                b64_data = self._encode_image_base64(media_path)
                payload["contents"][0]["parts"].append(
                    {
                        "inline_data": {
                            "mime_type": self._mime_from_path(media_path),
                            "data": b64_data,
                        }
                    }
                )
                appended += 1
            self._log(
                f"[GEMINI-SEND] parts={1 + appended} (text+media) media_count={appended}"
            )
        else:
            self._log(
                "[GEMINI-SEND] parts=1 (text only, include_image_base64=False or no media)"
            )
        try:
            schema_keys = list(dict(response_json_schema or {}).keys()) if isinstance(response_json_schema, dict) else []
            self._log(
                "[GEMINI-REQ] "
                f"model={str(self.model or DEFAULT_GEMINI_MODEL).strip()} "
                f"url={self._safe_url_for_log(self._gemini_url())} "
                f"timeout_sec={self.timeout_sec} "
                f"max_output_tokens={int(payload.get('generationConfig', {}).get('maxOutputTokens', 0) or 0)} "
                f"responseMimeType={str(payload.get('generationConfig', {}).get('responseMimeType', '') or '')!r} "
                f"has_responseSchema={int('responseSchema' in dict(payload.get('generationConfig') or {}))} "
                f"has_responseJsonSchema={int('responseJsonSchema' in dict(payload.get('generationConfig') or {}))} "
                f"schema_root_keys={schema_keys[:8]} "
                f"prompt_len={len(str(text_prompt or ''))} "
                f"media_count={len(media_items)}"
            )
        except Exception:
            pass
        return self._post_json(self._gemini_url(), payload)

    @staticmethod
    def _gemini_finish_reason(resp: Dict[str, object]) -> str:
        try:
            candidates = list(resp.get("candidates") or [])
            if not candidates:
                # Some gateway/proxy wrappers nest Gemini payload under a `response` object.
                nested = dict(resp.get("response") or {})
                candidates = list(nested.get("candidates") or [])
            if not candidates:
                return ""
            first = dict(candidates[0] or {})
            reason = str(first.get("finishReason", "") or first.get("finish_reason", "") or "").strip()
            return reason.upper().replace(" ", "_")
        except Exception:
            return ""

    @staticmethod
    def _is_gemini_truncated(resp: Dict[str, object]) -> bool:
        return APIVisionVerifier._gemini_finish_reason(resp) == "MAX_TOKENS"

    def _probe_max_output_tokens_for_attempt(self, attempt_idx: int) -> int:
        # Probe prompts are strict and schema-bound. Escalate budget on retries to
        # avoid repeated MAX_TOKENS loops that return no visible text parts.
        base = max(512, int(self.max_output_tokens or 256))
        return min(2048, int(base + max(0, int(attempt_idx)) * 256))

    def _caption_max_output_tokens_for_attempt(self, attempt_idx: int, *, expects_json: bool) -> int:
        # Caption JSON payloads are larger than probe answers and need a larger base budget.
        if expects_json:
            base = max(1536, int(self.max_output_tokens or 256) * 3)
        else:
            base = max(256, int(self.max_output_tokens or 256))
        return min(4096, int(base + max(0, int(attempt_idx)) * 512))

    @staticmethod
    def _invalid_reason(*, truncated: bool, raw_text: str, parsed: object, schema_ok: bool) -> str:
        if bool(truncated):
            return "truncated_response"
        if not str(raw_text or "").strip():
            return "empty_response"
        if bool(schema_ok):
            return ""
        if isinstance(parsed, dict) and parsed:
            return "schema_mismatch"
        return "unparseable_response"

    @staticmethod
    def _normalize_string_list(value: object) -> List[str]:
        out: List[str] = []
        for item in list(value or []):
            token = str(item or "").strip()
            if token:
                out.append(token)
        return out

    def _caption_relaxed_payload(
        self,
        *,
        parsed: object,
        text: str,
        request_prompt: str,
        request_schema: Dict[str, object],
        raw_response: Dict[str, object],
        truncated: bool,
    ) -> Dict[str, object] | None:
        data = dict(parsed or {}) if isinstance(parsed, dict) else {}
        caption_text = str(data.get("caption", "") or data.get("caption_text", "") or "").strip()
        if not caption_text and str(text or "").strip():
            caption_text = str(text or "").strip()
        if not caption_text:
            return None
        out = dict(data)
        out["caption"] = caption_text
        out["caption_text"] = caption_text
        out.setdefault("supported_entities", self._normalize_string_list(data.get("supported_entities")))
        out.setdefault("supported_relations", self._normalize_string_list(data.get("supported_relations")))
        out.setdefault("unsupported_entities", self._normalize_string_list(data.get("unsupported_entities")))
        out.setdefault("unsupported_relations", self._normalize_string_list(data.get("unsupported_relations")))
        out.setdefault("hallucinated_mentions", self._normalize_string_list(data.get("hallucinated_mentions")))
        out.setdefault("supported_attributes", list(data.get("supported_attributes") or []))
        out.setdefault("unsupported_attributes", list(data.get("unsupported_attributes") or []))
        out.setdefault("raw_text", str(text or ""))
        out.setdefault("request_prompt", request_prompt)
        out.setdefault("request_schema", request_schema)
        out.setdefault("raw_response", dict(raw_response or {}))
        out["schema_valid"] = False
        out["is_truncated"] = bool(truncated)
        out["is_valid"] = True
        out["parse_mode"] = "relaxed" if isinstance(parsed, dict) and parsed else "fallback"
        out["invalid_reason"] = self._invalid_reason(
            truncated=bool(truncated),
            raw_text=str(text or ""),
            parsed=parsed,
            schema_ok=False,
        )
        return out

    @staticmethod
    def _gemini_has_text_part(resp: Dict[str, object]) -> bool:
        try:
            candidates = list(resp.get("candidates") or [])
            for cand in candidates:
                content = dict((cand or {}).get("content") or {})
                parts = list(content.get("parts") or [])
                for part in parts:
                    if isinstance(part, str) and str(part).strip():
                        return True
                    if isinstance(part, dict) and str(part.get("text", "") or "").strip():
                        return True
        except Exception:
            return False
        return False

    @staticmethod
    def _gemini_usage_meta(resp: Dict[str, object]) -> Dict[str, object]:
        usage = dict(resp.get("usageMetadata") or {})
        if not usage and isinstance(resp.get("response"), dict):
            usage = dict((resp.get("response") or {}).get("usageMetadata") or {})
        if not usage:
            return {}
        out: Dict[str, object] = {}
        for key in (
            "promptTokenCount",
            "candidatesTokenCount",
            "totalTokenCount",
            "thoughtsTokenCount",
        ):
            if key in usage:
                out[key] = usage.get(key)
        return out

    def _openai_url(self) -> str:
        base = str(self.base_url or "").strip()
        return base or DEFAULT_OPENAI_BASE_URL

    def _openai_post(self, text_prompt: str, image_path: str, *, max_tokens_override: int = 0) -> Dict[str, object]:
        if not self.api_key:
            raise RuntimeError(f"OpenAI API key is missing (set {self.api_key_env} or api_key).")
        parts: List[Dict[str, object]] = [{"type": "text", "text": str(text_prompt or "")}]
        if self.include_image_base64:
            b64 = self._encode_image_base64(image_path)
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "low",
                    },
                }
            )
        payload: Dict[str, object] = {
            "model": str(self.model or DEFAULT_OPENAI_MODEL).strip(),
            "messages": [{"role": "user", "content": parts}],
            "temperature": 0.0,
            "max_tokens": int(max_tokens_override or self.max_output_tokens),
        }
        return self._post_json(self._openai_url(), payload)

    def _batch_probe_prompt_and_schema(self, probes: List[Dict[str, object]]) -> tuple[str, Dict[str, object]]:
        blocks: List[str] = []
        for idx, probe in enumerate(list(probes or []), start=1):
            probe_id = str(probe.get("probe_id", "") or f"probe_{idx}").strip() or f"probe_{idx}"
            fmt = dict(probe.get("response_format") or {}) if isinstance(probe.get("response_format"), dict) else {}
            fmt_type = str(fmt.get("type", "") or "").strip().lower()
            schema = dict(probe.get("schema") or {}) if isinstance(probe.get("schema"), dict) else {}
            if fmt_type == "selection":
                options = [str(x).strip() for x in list(fmt.get("options") or []) if str(x).strip()]
                prompt_text = self._build_selection_prompt(
                    question=str(probe.get("question", "") or "").strip(),
                    regions=list(probe.get("regions") or []),
                    options=options,
                    default_selection=str(fmt.get("default_selection", "uncertain") or "uncertain").strip(),
                    prompt_context=dict(fmt.get("_prompt_context") or fmt.get("prompt_context") or {}),
                    allow_uncertain=self._schema_allows_value(
                        schema,
                        "selection",
                        "uncertain",
                        default=bool(fmt.get("allow_uncertain", True)),
                    ),
                )
                response_contract = (
                    f'For this probe return: {{"probe_id":"{probe_id}","selection":"one of {options + ["uncertain"]} or empty string if disallowed",'
                    '"answer":"","reason":"...","score":0.0}'
                )
            else:
                prompt_text = self._build_binary_prompt(
                    question=str(probe.get("question", "") or "").strip(),
                    regions=list(probe.get("regions") or []),
                    prompt_context=dict(fmt.get("_prompt_context") or fmt.get("prompt_context") or {}),
                    allow_uncertain=self._schema_allows_value(
                        schema,
                        "answer",
                        "uncertain",
                        default=bool(fmt.get("allow_uncertain", True)),
                    ),
                )
                response_contract = (
                    f'For this probe return: {{"probe_id":"{probe_id}","answer":"yes/no/uncertain",'
                    '"selection":"","reason":"...","score":0.0}'
                )
            blocks.append(
                f"Probe {idx}\n"
                f"probe_id: {probe_id}\n"
                f"kind: {fmt_type or 'binary'}\n"
                f"{prompt_text}\n"
                f"{response_contract}"
            )
        prompt = (
            "You are validating multiple visual probes for the same image.\n"
            "Solve every probe independently using only visible evidence from the image and each probe's own context.\n"
            "Return JSON only.\n"
            'Output format: {"results":[{"probe_id":"...","answer":"yes|no|uncertain|","selection":"candidate|uncertain|","reason":"...","score":0.0}]}\n'
            "Use an empty string for the field that does not apply to that probe kind.\n\n"
            + "\n\n".join(blocks)
        )
        schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "probe_id": {"type": "string"},
                            "answer": {"type": "string"},
                            "selection": {"type": "string"},
                            "reason": {"type": "string"},
                            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                        "required": ["probe_id", "answer", "selection", "reason", "score"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        }
        return prompt, schema

    def answer_probe_batch(self, *, image_path: str = "", probes: List[Dict[str, object]] | None = None) -> List[Dict[str, object]]:
        items = [dict(item or {}) for item in list(probes or []) if isinstance(item, dict)]
        if not items:
            return []
        if self._answer_handler is not None or (not self._is_openai):
            out: List[Dict[str, object]] = []
            for item in items:
                out.append(
                    self.answer_probe(
                        image_path=image_path,
                        question=str(item.get("question", "") or ""),
                        regions=list(item.get("regions") or []),
                        response_format=dict(item.get("response_format") or {}) if isinstance(item.get("response_format"), dict) else None,
                        schema=dict(item.get("schema") or {}) if isinstance(item.get("schema"), dict) else None,
                    )
                )
            return out

        prompt, batch_schema = self._batch_probe_prompt_and_schema(items)
        batch_tokens = min(4096, max(512, int(self.max_output_tokens) * max(2, len(items))))
        raw_resp = self._openai_post(prompt, image_path=image_path, max_tokens_override=batch_tokens)
        text = self._extract_openai_text(raw_resp)
        payload = self._extract_json_value(text)
        rows = list((payload.get("results") or [])) if isinstance(payload, dict) else []
        by_probe_id = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            probe_id = str(row.get("probe_id", "") or "").strip()
            if probe_id:
                by_probe_id[probe_id] = dict(row)

        results: List[Dict[str, object]] = []
        for item in items:
            probe_id = str(item.get("probe_id", "") or "").strip()
            row = dict(by_probe_id.get(probe_id) or {})
            result = {
                "answer": str(row.get("answer", "") or "").strip(),
                "selection": str(row.get("selection", "") or "").strip(),
                "score": max(0.0, min(1.0, self._safe_float(row.get("score"), 0.0))),
                "reason": str(row.get("reason", "") or "").strip() or str(text or "").strip(),
                "raw_text": text,
                "request_prompt": prompt,
                "request_schema": batch_schema,
                "raw_response": dict(raw_resp or {}),
                "schema_valid": bool(probe_id and probe_id in by_probe_id),
                "parse_mode": "strict",
                "invalid_reason": "" if (probe_id and probe_id in by_probe_id) else "missing_probe_result",
                "batch_probe_id": probe_id,
                "batch_response_count": len(by_probe_id),
            }
            results.append(result)
        return results

    def answer_probe(
        self,
        *,
        image_path: str = "",
        question: str = "",
        regions: List[Dict[str, object]] | None = None,
        response_format: Dict[str, object] | None = None,
        image: str = "",
        prompt: str = "",
        schema: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        resolved_image_path = str(image_path or image or "").strip()
        resolved_question = str(question or prompt or "").strip()
        resolved_regions = list(regions or [])
        if self._answer_handler is not None:
            try:
                return dict(
                    self._answer_handler(
                        resolved_image_path,
                        resolved_question,
                        resolved_regions,
                        response_format=response_format,
                        schema=schema,
                    )
                    or {}
                )
            except TypeError:
                return dict(self._answer_handler(resolved_image_path, resolved_question, resolved_regions) or {})

        img_exists = os.path.isfile(str(resolved_image_path or "")) if str(resolved_image_path or "").strip() else False
        img_size = os.path.getsize(str(resolved_image_path)) if img_exists else 0
        self._log(
            f"[PROBE-IMAGE] path={str(resolved_image_path or '')!r} exists={img_exists} "
            f"size={img_size} include_b64={self.include_image_base64}"
        )

        if self._is_gemini or self._is_openai:
            fmt = dict(response_format or {}) if isinstance(response_format, dict) else {}
            fmt_type = str(fmt.get("type", "") or "").strip().lower()
            prompt_context = dict(fmt.get("_prompt_context") or fmt.get("prompt_context") or {})
            target_schema = dict(schema or {})
            if fmt_type == "selection":
                options = [str(x).strip() for x in list(fmt.get("options") or []) if str(x).strip()]
                default_sel = str(fmt.get("default_selection", "uncertain") or "uncertain").strip()
                if not target_schema:
                    allow_uncertain = bool(fmt.get("allow_uncertain", True))
                    target_schema = {
                        "type": "object",
                        "properties": {
                            "selection": {"type": "string", "enum": options + (["uncertain"] if allow_uncertain else [])},
                            "reason": {"type": "string"},
                            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        },
                        "required": ["selection", "reason", "score"],
                        "additionalProperties": False,
                    }
                allow_uncertain = self._schema_allows_value(
                    target_schema,
                    "selection",
                    "uncertain",
                    default=bool(fmt.get("allow_uncertain", True)),
                )
                prompt = self._build_selection_prompt(
                    question=resolved_question,
                    regions=resolved_regions,
                    options=options,
                    default_selection=default_sel,
                    prompt_context=prompt_context,
                    allow_uncertain=allow_uncertain,
                )
                if self._is_gemini:
                    max_attempts = 3
                    last_text = ""
                    last_resp: Dict[str, object] = {}
                    last_truncated = False
                    last_parsed: object = {}
                    for attempt_idx in range(max_attempts):
                        attempt_tokens = self._probe_max_output_tokens_for_attempt(attempt_idx)
                        self._log(
                            f"[PROBE-ATTEMPT] kind=selection attempt={attempt_idx + 1}/{max_attempts} "
                            f"max_output_tokens={attempt_tokens} prompt_len={len(str(prompt or ''))}"
                        )
                        raw_resp = self._gemini_post(
                            prompt,
                            image_path=resolved_image_path,
                            response_mime_type="application/json",
                            response_json_schema=target_schema,
                            max_output_tokens_override=attempt_tokens,
                        )
                        text = self._extract_gemini_text(raw_resp)
                        truncated = self._is_gemini_truncated(raw_resp)
                        finish_reason = self._gemini_finish_reason(raw_resp)
                        self._log(
                            f"[PROBE-RESP] finish={finish_reason or 'UNKNOWN'} "
                            f"truncated={int(truncated)} max_out={attempt_tokens} "
                            f"raw_text={str(text or '')[:300]!r}"
                        )
                        parsed, schema_ok = self._gemini_parse_with_schema(
                            text=text,
                            raw_resp=raw_resp,
                            schema=target_schema,
                            required_keys=["selection", "reason", "score"],
                        )
                        self._log(
                            "[PROBE-RESP-SUMMARY] "
                            f"kind=selection attempt={attempt_idx + 1}/{max_attempts} "
                            f"finish={finish_reason or 'UNKNOWN'} "
                            f"schema_valid={int(bool(schema_ok))} "
                            f"is_truncated={int(bool(truncated))} "
                            f"has_text_part={int(self._gemini_has_text_part(raw_resp))} "
                            f"raw_text_len={len(str(text or ''))} "
                            f"usage={self._gemini_usage_meta(raw_resp)}"
                        )
                        if schema_ok and not truncated:
                            return {
                                "selection": str(parsed.get("selection", "uncertain") or "uncertain").strip() or "uncertain",
                                "score": max(0.0, min(1.0, self._safe_float(parsed.get("score"), 0.0))),
                                "reason": str(parsed.get("reason", "") or "").strip(),
                                "raw_text": str(text or ""),
                                "request_prompt": prompt,
                                "request_schema": target_schema,
                                "raw_response": dict(raw_resp or {}),
                                "schema_valid": True,
                                "is_truncated": False,
                                "is_valid": True,
                                "parse_mode": "strict",
                                "invalid_reason": "",
                            }
                        if (not schema_ok) and str(text or "").strip():
                            selected = self._infer_selection_from_text(text, options)
                            if (not selected) and isinstance(parsed, dict):
                                selected = self._infer_selection_from_text(
                                    json.dumps(parsed, ensure_ascii=True),
                                    options,
                                )
                            fallback_score = self._extract_score_from_text(text, default=0.5)
                            if isinstance(parsed, dict):
                                fallback_score = self._safe_float(parsed.get("score"), fallback_score)
                            fallback_reason = self._extract_reason_from_text(
                                str((parsed or {}).get("reason", "") or text) if isinstance(parsed, dict) else text,
                                default="fallback_parse",
                            )
                            selected = selected or "uncertain"
                            if selected == "uncertain" and not allow_uncertain:
                                last_text = str(text or "")
                                last_resp = dict(raw_resp or {})
                                last_parsed = parsed
                                last_truncated = bool(last_truncated or truncated)
                                if attempt_idx < (max_attempts - 1):
                                    self._log(f"[PROBE-RESP] contract violation (unexpected uncertain), retry={attempt_idx + 1}/{max_attempts - 1}")
                                continue
                            return {
                                "selection": selected,
                                "score": max(0.0, min(1.0, float(fallback_score))),
                                "reason": str(fallback_reason or "fallback_parse"),
                                "raw_text": str(text or ""),
                                "request_prompt": prompt,
                                "request_schema": target_schema,
                                "raw_response": dict(raw_resp or {}),
                                "schema_valid": False,
                                "is_truncated": bool(truncated),
                                "is_valid": True,
                                "parse_mode": "relaxed" if isinstance(parsed, dict) and parsed else "fallback",
                                "invalid_reason": self._invalid_reason(
                                    truncated=bool(truncated),
                                    raw_text=str(text or ""),
                                    parsed=parsed,
                                    schema_ok=False,
                                ),
                            }
                        last_text = str(text or "")
                        last_resp = dict(raw_resp or {})
                        last_parsed = parsed
                        last_truncated = bool(last_truncated or truncated)
                        if attempt_idx < (max_attempts - 1):
                            self._log(f"[PROBE-RESP] schema invalid, retry={attempt_idx + 1}/{max_attempts - 1}")
                    if str(last_text or "").strip():
                        selected = self._infer_selection_from_text(last_text, options) or "uncertain"
                        if selected == "uncertain" and not allow_uncertain:
                            invalid_reason = self._invalid_reason(
                                truncated=bool(last_truncated),
                                raw_text=last_text,
                                parsed=last_parsed,
                                schema_ok=False,
                            )
                            return {
                                "selection": "uncertain",
                                "score": None,
                                "reason": invalid_reason or "invalid_response",
                                "raw_text": last_text,
                                "request_prompt": prompt,
                                "request_schema": target_schema,
                                "raw_response": last_resp,
                                "schema_valid": False,
                                "is_truncated": bool(last_truncated),
                                "is_valid": False,
                                "parse_mode": "strict",
                                "invalid_reason": invalid_reason or "contract_violation_unexpected_uncertain",
                            }
                        fallback_score = self._extract_score_from_text(last_text, default=0.5)
                        if isinstance(last_parsed, dict):
                            fallback_score = self._safe_float(last_parsed.get("score"), fallback_score)
                        fallback_reason = self._extract_reason_from_text(
                            str((last_parsed or {}).get("reason", "") or last_text) if isinstance(last_parsed, dict) else last_text,
                            default="fallback_parse",
                        )
                        return {
                            "selection": selected,
                            "score": max(0.0, min(1.0, float(fallback_score))),
                            "reason": str(fallback_reason or "fallback_parse"),
                            "raw_text": last_text,
                            "request_prompt": prompt,
                            "request_schema": target_schema,
                            "raw_response": last_resp,
                            "schema_valid": False,
                            "is_truncated": bool(last_truncated),
                            "is_valid": True,
                            "parse_mode": "relaxed" if isinstance(last_parsed, dict) and last_parsed else "fallback",
                            "invalid_reason": self._invalid_reason(
                                truncated=bool(last_truncated),
                                raw_text=last_text,
                                parsed=last_parsed,
                                schema_ok=False,
                            ),
                        }
                    invalid_reason = self._invalid_reason(
                        truncated=bool(last_truncated),
                        raw_text=last_text,
                        parsed=last_parsed,
                        schema_ok=False,
                    )
                    return {
                        "selection": "uncertain",
                        "score": None,
                        "reason": invalid_reason or "invalid_response",
                        "raw_text": last_text,
                        "request_prompt": prompt,
                        "request_schema": target_schema,
                        "raw_response": last_resp,
                        "schema_valid": False,
                        "is_truncated": bool(last_truncated),
                        "is_valid": False,
                        "parse_mode": "strict",
                        "invalid_reason": invalid_reason or "unparseable_response",
                    }
                raw_resp = self._openai_post(prompt, image_path=resolved_image_path)
                text = self._extract_openai_text(raw_resp)
                parsed = self._extract_json_block(text)
                selected = str(parsed.get("selection", "") or "").strip()
                score = self._safe_float(parsed.get("score"), 0.0)
                return {
                    "selection": selected,
                    "score": max(0.0, min(1.0, float(score))),
                    "reason": str(parsed.get("reason", "") or "").strip() or str(text or "").strip(),
                    "raw_text": text,
                    "request_prompt": prompt,
                    "request_schema": target_schema,
                    "raw_response": dict(raw_resp or {}),
                    "schema_valid": len(self._validate_json_schema(parsed, target_schema)) == 0 if isinstance(parsed, dict) else False,
                    "parse_mode": "strict",
                    "invalid_reason": "",
                }
            if not target_schema:
                allow_uncertain = bool(fmt.get("allow_uncertain", True))
                target_schema = {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "enum": ["yes", "no"] + (["uncertain"] if allow_uncertain else [])},
                        "reason": {"type": "string"},
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["answer", "reason", "score"],
                    "additionalProperties": False,
                }
            allow_uncertain = self._schema_allows_value(
                target_schema,
                "answer",
                "uncertain",
                default=bool(fmt.get("allow_uncertain", True)),
            )
            prompt = self._build_binary_prompt(
                question=resolved_question,
                regions=resolved_regions,
                prompt_context=prompt_context,
                allow_uncertain=allow_uncertain,
            )
            if self._is_gemini:
                max_attempts = 3
                last_text = ""
                last_resp: Dict[str, object] = {}
                last_truncated = False
                last_parsed: object = {}
                last_relaxed: Dict[str, object] | None = None
                for attempt_idx in range(max_attempts):
                    attempt_tokens = self._probe_max_output_tokens_for_attempt(attempt_idx)
                    self._log(
                        f"[PROBE-ATTEMPT] kind=binary attempt={attempt_idx + 1}/{max_attempts} "
                        f"max_output_tokens={attempt_tokens} prompt_len={len(str(prompt or ''))}"
                    )
                    raw_resp = self._gemini_post(
                        prompt,
                        image_path=resolved_image_path,
                        response_mime_type="application/json",
                        response_json_schema=target_schema,
                        max_output_tokens_override=attempt_tokens,
                    )
                    text = self._extract_gemini_text(raw_resp)
                    truncated = self._is_gemini_truncated(raw_resp)
                    finish_reason = self._gemini_finish_reason(raw_resp)
                    self._log(
                        f"[PROBE-RESP] finish={finish_reason or 'UNKNOWN'} "
                        f"truncated={int(truncated)} max_out={attempt_tokens} "
                        f"raw_text={str(text or '')[:300]!r}"
                    )
                    parsed, schema_ok = self._gemini_parse_with_schema(
                        text=text,
                        raw_resp=raw_resp,
                        schema=target_schema,
                        required_keys=["answer", "reason", "score"],
                    )
                    self._log(
                        "[PROBE-RESP-SUMMARY] "
                        f"kind=binary attempt={attempt_idx + 1}/{max_attempts} "
                        f"finish={finish_reason or 'UNKNOWN'} "
                        f"schema_valid={int(bool(schema_ok))} "
                        f"is_truncated={int(bool(truncated))} "
                        f"has_text_part={int(self._gemini_has_text_part(raw_resp))} "
                        f"raw_text_len={len(str(text or ''))} "
                        f"usage={self._gemini_usage_meta(raw_resp)}"
                    )
                    if schema_ok and not truncated:
                        return {
                            "answer": self._normalize_binary_answer(parsed.get("answer")) or "uncertain",
                            "score": max(0.0, min(1.0, self._safe_float(parsed.get("score"), 0.0))),
                            "reason": str(parsed.get("reason", "") or "").strip(),
                            "raw_text": str(text or ""),
                            "request_prompt": prompt,
                            "request_schema": target_schema,
                            "raw_response": dict(raw_resp or {}),
                            "schema_valid": True,
                            "is_truncated": False,
                            "is_valid": True,
                            "parse_mode": "strict",
                            "invalid_reason": "",
                        }
                    if (not schema_ok) and str(text or "").strip():
                        fallback_answer = self._normalize_binary_answer(
                            ((parsed or {}).get("answer") if isinstance(parsed, dict) else "")
                        ) or self._infer_binary_answer_from_text(text)
                        if fallback_answer == "uncertain" and not allow_uncertain:
                            last_text = str(text or "")
                            last_resp = dict(raw_resp or {})
                            last_parsed = parsed
                            last_truncated = bool(last_truncated or truncated)
                            if attempt_idx < (max_attempts - 1):
                                self._log(f"[PROBE-RESP] contract violation (unexpected uncertain), retry={attempt_idx + 1}/{max_attempts - 1}")
                            continue
                        fallback_score = self._extract_score_from_text(text, default=0.5)
                        fallback_reason = self._extract_reason_from_text(
                            str((parsed or {}).get("reason", "") or text) if isinstance(parsed, dict) else text,
                            default="fallback_parse",
                        )
                        return {
                            "answer": fallback_answer or "uncertain",
                            "score": max(0.0, min(1.0, float(fallback_score))),
                            "reason": str(fallback_reason or "fallback_parse"),
                            "raw_text": str(text or ""),
                            "request_prompt": prompt,
                            "request_schema": target_schema,
                            "raw_response": dict(raw_resp or {}),
                            "schema_valid": False,
                            "is_truncated": bool(truncated),
                            "is_valid": True,
                            "parse_mode": "relaxed" if isinstance(parsed, dict) and parsed else "fallback",
                            "invalid_reason": self._invalid_reason(
                                truncated=bool(truncated),
                                raw_text=str(text or ""),
                                parsed=parsed,
                                schema_ok=False,
                            ),
                        }
                    last_text = str(text or "")
                    last_resp = dict(raw_resp or {})
                    last_parsed = parsed
                    last_truncated = bool(last_truncated or truncated)
                    if attempt_idx < (max_attempts - 1):
                        self._log(f"[PROBE-RESP] schema invalid, retry={attempt_idx + 1}/{max_attempts - 1}")
                if str(last_text or "").strip():
                    fallback_answer = self._normalize_binary_answer(
                        (last_parsed or {}).get("answer") if isinstance(last_parsed, dict) else ""
                    ) or self._infer_binary_answer_from_text(last_text)
                    if fallback_answer == "uncertain" and not allow_uncertain:
                        invalid_reason = self._invalid_reason(
                            truncated=bool(last_truncated),
                            raw_text=last_text,
                            parsed=last_parsed,
                            schema_ok=False,
                        )
                        return {
                            "answer": "uncertain",
                            "score": None,
                            "reason": invalid_reason or "invalid_response",
                            "raw_text": last_text,
                            "request_prompt": prompt,
                            "request_schema": target_schema,
                            "raw_response": last_resp,
                            "schema_valid": False,
                            "is_truncated": bool(last_truncated),
                            "is_valid": False,
                            "parse_mode": "strict",
                            "invalid_reason": invalid_reason or "contract_violation_unexpected_uncertain",
                        }
                    fallback_score = self._extract_score_from_text(last_text, default=0.5)
                    if isinstance(last_parsed, dict):
                        fallback_score = self._safe_float(last_parsed.get("score"), fallback_score)
                    fallback_reason = self._extract_reason_from_text(
                        str((last_parsed or {}).get("reason", "") or last_text) if isinstance(last_parsed, dict) else last_text,
                        default="fallback_parse",
                    )
                    return {
                        "answer": fallback_answer or "uncertain",
                        "score": max(0.0, min(1.0, float(fallback_score))),
                        "reason": str(fallback_reason or "fallback_parse"),
                        "raw_text": last_text,
                        "request_prompt": prompt,
                        "request_schema": target_schema,
                        "raw_response": last_resp,
                        "schema_valid": False,
                        "is_truncated": bool(last_truncated),
                        "is_valid": True,
                        "parse_mode": "relaxed" if isinstance(last_parsed, dict) and last_parsed else "fallback",
                        "invalid_reason": self._invalid_reason(
                            truncated=bool(last_truncated),
                            raw_text=last_text,
                            parsed=last_parsed,
                            schema_ok=False,
                        ),
                    }
                invalid_reason = self._invalid_reason(
                    truncated=bool(last_truncated),
                    raw_text=last_text,
                    parsed=last_parsed,
                    schema_ok=False,
                )
                return {
                    "answer": "uncertain",
                    "score": None,
                    "reason": invalid_reason or "invalid_response",
                    "raw_text": last_text,
                    "request_prompt": prompt,
                    "request_schema": target_schema,
                    "raw_response": last_resp,
                    "schema_valid": False,
                    "is_truncated": bool(last_truncated),
                    "is_valid": False,
                    "parse_mode": "strict",
                    "invalid_reason": invalid_reason or "unparseable_response",
                }
            raw_resp = self._openai_post(prompt, image_path=resolved_image_path)
            text = self._extract_openai_text(raw_resp)
            parsed = self._extract_json_block(text)
            answer = self._normalize_binary_answer(parsed.get("answer")) if isinstance(parsed, dict) else "uncertain"
            return {
                "answer": answer or "uncertain",
                "score": max(0.0, min(1.0, self._safe_float((parsed or {}).get("score"), 0.0))),
                "reason": str((parsed or {}).get("reason", "") or "").strip() or str(text or "").strip(),
                "raw_text": text,
                "request_prompt": prompt,
                "request_schema": target_schema,
                "raw_response": dict(raw_resp or {}),
                "schema_valid": len(self._validate_json_schema(parsed, target_schema)) == 0 if isinstance(parsed, dict) else False,
                "parse_mode": "strict",
                "invalid_reason": "",
            }

        payload = self._base_payload(image_path=resolved_image_path, regions=resolved_regions)
        payload.update(
            {
                "task": "answer_probe",
                "question": str(resolved_question or ""),
                "response_format": self._transport_response_format(response_format),
                "schema": dict(schema or {}) if isinstance(schema, dict) else schema,
            }
        )
        return self._post_json(self._target_url(kind="answer"), payload)

    def generate_caption(
        self,
        *,
        image_path: str = "",
        prompt: str = "",
        regions: List[Dict[str, object]] | None = None,
        video_or_frames: object = None,
        schema: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        media_paths: List[str] = []
        if isinstance(video_or_frames, (list, tuple)):
            media_paths.extend([str(x).strip() for x in list(video_or_frames) if str(x).strip()])
        elif isinstance(video_or_frames, str) and str(video_or_frames).strip():
            media_paths.append(str(video_or_frames).strip())
        if str(image_path or "").strip():
            media_paths.insert(0, str(image_path).strip())
        resolved_image = media_paths[0] if media_paths else ""
        resolved_regions = list(regions or [])
        _ = list(resolved_regions or [])  # keep compatibility for existing call signature
        if self._caption_handler is not None:
            try:
                return dict(self._caption_handler(resolved_image, prompt, resolved_regions, schema=None) or {})
            except TypeError:
                return dict(self._caption_handler(resolved_image, prompt, resolved_regions) or {})

        if self._is_gemini or self._is_openai:
            target_schema = dict(schema or {})
            caption_prompt, expects_json = self._build_caption_prompt(
                prompt=prompt,
                schema=target_schema,
                regions=resolved_regions,
            )
            if self._is_gemini:
                max_attempts = 3
                last_text = ""
                last_resp: Dict[str, object] = {}
                last_truncated = False
                last_parsed: object = {}
                for attempt_idx in range(max_attempts):
                    attempt_tokens = self._caption_max_output_tokens_for_attempt(
                        attempt_idx,
                        expects_json=bool(expects_json),
                    )
                    self._log(
                        f"[CAPTION-ATTEMPT] attempt={attempt_idx + 1}/{max_attempts} "
                        f"max_output_tokens={attempt_tokens} prompt_len={len(str(caption_prompt or ''))} "
                        f"expects_json={int(bool(expects_json))}"
                    )
                    raw_resp = self._gemini_post(
                        caption_prompt,
                        image_path=resolved_image,
                        media_paths=media_paths,
                        response_mime_type="application/json" if expects_json else "",
                        response_json_schema=target_schema if expects_json and target_schema else None,
                        max_output_tokens_override=attempt_tokens,
                    )
                    text = self._extract_gemini_text(raw_resp)
                    truncated = self._is_gemini_truncated(raw_resp)
                    finish_reason = self._gemini_finish_reason(raw_resp)
                    self._log(
                        "[CAPTION-RESP-SUMMARY] "
                        f"attempt={attempt_idx + 1}/{max_attempts} "
                        f"finish={finish_reason or 'UNKNOWN'} "
                        f"is_truncated={int(bool(truncated))} "
                        f"has_text_part={int(self._gemini_has_text_part(raw_resp))} "
                        f"raw_text_len={len(str(text or ''))} "
                        f"usage={self._gemini_usage_meta(raw_resp)}"
                    )
                    if expects_json and str(text or "").strip():
                        parsed = self._extract_json_block(text)
                        schema_ok = True
                        if target_schema:
                            schema_ok = len(self._validate_json_schema(parsed, target_schema)) == 0 if isinstance(parsed, dict) else False
                        if isinstance(parsed, dict) and schema_ok and (not truncated):
                            parsed.setdefault("raw_text", str(text or ""))
                            parsed.setdefault("request_prompt", caption_prompt)
                            parsed.setdefault("request_schema", target_schema)
                            parsed.setdefault("raw_response", dict(raw_resp or {}))
                            parsed["schema_valid"] = bool(target_schema)
                            parsed["is_truncated"] = False
                            parsed["is_valid"] = True
                            parsed["parse_mode"] = "strict"
                            parsed["invalid_reason"] = ""
                            if "caption" in parsed and "caption_text" not in parsed:
                                parsed["caption_text"] = str(parsed.get("caption", "") or "").strip()
                            return parsed
                        relaxed = self._caption_relaxed_payload(
                            parsed=parsed,
                            text=str(text or ""),
                            request_prompt=caption_prompt,
                            request_schema=target_schema,
                            raw_response=dict(raw_resp or {}),
                            truncated=bool(truncated),
                        )
                        if relaxed is not None:
                            last_relaxed = dict(relaxed)
                            if not truncated:
                                return last_relaxed
                            if attempt_idx < (max_attempts - 1):
                                self._log(
                                    f"[CAPTION-RETRY] truncated caption payload, retry={attempt_idx + 1}/{max_attempts - 1}"
                                )
                    if str(text or "").strip() and not truncated and not expects_json:
                        caption_text = str(text or "").strip()
                        return {
                            "caption_text": caption_text,
                            "caption": caption_text,
                            "raw_text": caption_text,
                            "request_prompt": caption_prompt,
                            "raw_response": dict(raw_resp or {}),
                            "schema_valid": True,
                            "is_truncated": False,
                            "is_valid": True,
                            "parse_mode": "strict",
                            "invalid_reason": "",
                        }
                    last_text = str(text or "")
                    last_resp = dict(raw_resp or {})
                    last_parsed = self._extract_json_block(text) if str(text or "").strip() else {}
                    last_truncated = bool(last_truncated or truncated)
                    if attempt_idx < (max_attempts - 1):
                        self._log(f"[CAPTION-RESP] empty/truncated, retry={attempt_idx + 1}/{max_attempts - 1}")
                if expects_json and str(last_text or "").strip():
                    parsed = self._extract_json_block(last_text)
                    if isinstance(parsed, dict):
                        schema_ok = (
                            len(self._validate_json_schema(parsed, target_schema)) == 0
                            if target_schema
                            else True
                        )
                        if schema_ok:
                            parsed.setdefault("raw_text", last_text)
                            parsed.setdefault("request_prompt", caption_prompt)
                            parsed.setdefault("request_schema", target_schema)
                            parsed.setdefault("raw_response", last_resp)
                            parsed["schema_valid"] = bool(target_schema)
                            parsed["is_truncated"] = bool(last_truncated)
                            parsed["is_valid"] = True
                            parsed["parse_mode"] = "strict"
                            parsed["invalid_reason"] = ""
                            if "caption" in parsed and "caption_text" not in parsed:
                                parsed["caption_text"] = str(parsed.get("caption", "") or "").strip()
                            return parsed
                    relaxed = self._caption_relaxed_payload(
                        parsed=parsed,
                        text=last_text,
                        request_prompt=caption_prompt,
                        request_schema=target_schema,
                        raw_response=last_resp,
                        truncated=bool(last_truncated),
                    )
                    if relaxed is not None:
                        return relaxed
                if last_relaxed is not None:
                    return dict(last_relaxed)
                if str(last_text or "").strip():
                    return {
                        "caption_text": str(last_text or "").strip(),
                        "caption": str(last_text or "").strip(),
                        "raw_text": last_text,
                        "request_prompt": caption_prompt,
                        "raw_response": last_resp,
                        "schema_valid": False,
                        "is_truncated": bool(last_truncated),
                        "is_valid": True,
                        "parse_mode": "fallback",
                        "invalid_reason": self._invalid_reason(
                            truncated=bool(last_truncated),
                            raw_text=last_text,
                            parsed=last_parsed,
                            schema_ok=False,
                        ),
                    }
                invalid_reason = self._invalid_reason(
                    truncated=bool(last_truncated),
                    raw_text=last_text,
                    parsed=last_parsed,
                    schema_ok=False,
                )
                return {
                    "caption_text": str(last_text or "").strip(),
                    "caption": str(last_text or "").strip(),
                    "raw_text": last_text,
                    "request_prompt": caption_prompt,
                    "raw_response": last_resp,
                    "schema_valid": False,
                    "is_truncated": bool(last_truncated),
                    "is_valid": False,
                    "parse_mode": "strict",
                    "invalid_reason": invalid_reason or "unparseable_response",
                }
            raw_resp = self._openai_post(caption_prompt, image_path=resolved_image)
            text = str(self._extract_openai_text(raw_resp) or "").strip()
            if expects_json:
                parsed = self._extract_json_block(text)
                schema_ok = len(self._validate_json_schema(parsed, target_schema)) == 0 if target_schema and isinstance(parsed, dict) else False
                if isinstance(parsed, dict):
                    parsed.setdefault("raw_text", text)
                    parsed.setdefault("request_prompt", caption_prompt)
                    parsed.setdefault("request_schema", target_schema)
                    parsed.setdefault("raw_response", dict(raw_resp or {}))
                    parsed["schema_valid"] = bool(schema_ok)
                    parsed["is_truncated"] = False
                    parsed["is_valid"] = bool(schema_ok or not target_schema)
                    parsed["parse_mode"] = "strict" if schema_ok else "relaxed"
                    parsed["invalid_reason"] = "" if schema_ok else "schema_mismatch"
                    if "caption" in parsed and "caption_text" not in parsed:
                        parsed["caption_text"] = str(parsed.get("caption", "") or "").strip()
                    return parsed
            return {
                "caption_text": text,
                "caption": text,
                "raw_text": text,
                "request_prompt": caption_prompt,
                "raw_response": dict(raw_resp or {}),
                "schema_valid": bool(text),
                "is_truncated": False,
                "is_valid": bool(text),
                "parse_mode": "fallback" if bool(text) else "strict",
                "invalid_reason": "" if bool(text) else "empty_response",
            }

        payload = self._base_payload(image_path=resolved_image, regions=resolved_regions)
        payload.update(
            {
                "task": "generate_caption",
                "prompt": str(prompt or "").strip() or "Describe the image briefly in one sentence.",
                "video_or_frames": list(media_paths),
                "schema": dict(schema or {}) if isinstance(schema, dict) else schema,
            }
        )
        return self._post_json(self._target_url(kind="caption"), payload)
