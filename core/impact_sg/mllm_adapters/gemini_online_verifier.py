from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, List, Optional

from .api_verifier import APIVisionVerifier


class GeminiOnlineVisionVerifier:
    """
    Browser-driven Gemini verifier.

    Notes:
    - Requires `selenium` and a working Chrome/Chromium webdriver setup.
    - Uses a persistent Chrome user profile so users can log in with their own account.
    - Intended for assisted automation; Gemini web DOM can change over time.
    """

    def __init__(
        self,
        *,
        model: str = "",
        timeout_sec: int = 180,
        debug_log: Callable[[str], None] | None = None,
        user_data_dir: str = "",
        profile_directory: str = "",
        chrome_binary: str = "",
        headless: bool = False,
        gemini_url: str = "https://gemini.google.com/app",
    ) -> None:
        self.model = str(model or "").strip()
        self.timeout_sec = max(30, int(timeout_sec or 180))
        self._debug_log = debug_log
        self.user_data_dir = str(user_data_dir or "").strip()
        self.profile_directory = str(profile_directory or "").strip()
        self.chrome_binary = str(chrome_binary or "").strip()
        self.headless = bool(headless)
        self.gemini_url = str(gemini_url or "https://gemini.google.com/app").strip()
        self._driver = None
        self._wait = None

    def _log(self, text: str) -> None:
        cb = self._debug_log
        if cb is None:
            return
        try:
            cb(str(text or "").strip())
        except Exception:
            pass

    def _ensure_driver(self) -> None:
        if self._driver is not None:
            return
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.support.ui import WebDriverWait
        except Exception as exc:
            raise RuntimeError(
                "gemini_online requires selenium. Install with: pip install selenium"
            ) from exc

        opts = Options()
        if self.headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--start-maximized")

        if self.user_data_dir:
            opts.add_argument(f"--user-data-dir={os.path.abspath(os.path.expanduser(self.user_data_dir))}")
        if self.profile_directory:
            opts.add_argument(f"--profile-directory={self.profile_directory}")
        if self.chrome_binary:
            opts.binary_location = os.path.abspath(os.path.expanduser(self.chrome_binary))

        self._driver = webdriver.Chrome(options=opts)
        self._wait = WebDriverWait(self._driver, self.timeout_sec)
        self._driver.get(self.gemini_url)
        self._log("[GEMINI-ONLINE] Browser launched. Please complete login if prompted.")
        self._wait_until_composer_ready()

    def _wait_until_composer_ready(self) -> None:
        from selenium.webdriver.common.by import By

        deadline = time.time() + float(self.timeout_sec)
        while time.time() < deadline:
            if self._driver is None:
                break
            for selector in [
                "textarea",
                "div[contenteditable='true']",
                "rich-textarea div[contenteditable='true']",
            ]:
                try:
                    elems = self._driver.find_elements(By.CSS_SELECTOR, selector)
                    if any(e.is_displayed() for e in elems):
                        return
                except Exception:
                    continue
            time.sleep(0.5)
        raise RuntimeError(
            "Gemini web input box not found. Please confirm browser is logged in and Gemini page is open."
        )

    def _find_composer(self):
        from selenium.webdriver.common.by import By

        for selector in [
            "rich-textarea div[contenteditable='true']",
            "div[contenteditable='true']",
            "textarea",
        ]:
            try:
                elems = self._driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elems:
                    if elem.is_displayed():
                        return elem
            except Exception:
                continue
        return None

    def _find_upload_input(self):
        from selenium.webdriver.common.by import By

        for selector in [
            "input[type='file']",
            "input[accept*='image']",
        ]:
            try:
                elems = self._driver.find_elements(By.CSS_SELECTOR, selector)
                for elem in elems:
                    if elem.is_enabled():
                        return elem
            except Exception:
                continue
        return None

    def _response_text_candidates(self) -> List[str]:
        from selenium.webdriver.common.by import By

        candidates: List[str] = []
        selectors = [
            "message-content",
            "model-response",
            "div.markdown",
            "div.response-content",
            "div[data-message-author-role='assistant']",
        ]
        for selector in selectors:
            try:
                elems = self._driver.find_elements(By.CSS_SELECTOR, selector)
            except Exception:
                elems = []
            for elem in elems[-6:]:
                try:
                    txt = str(elem.text or "").strip()
                except Exception:
                    txt = ""
                if txt and txt not in candidates:
                    candidates.append(txt)
        return candidates

    def _best_response_text(self) -> str:
        candidates = self._response_text_candidates()
        if not candidates:
            return ""
        # Prefer the latest visible assistant text rather than the longest one.
        # In long browser sessions, picking the longest response can keep returning
        # an older answer and make new short JSON replies look "missing".
        return candidates[-1]

    def _submit_prompt(self, *, prompt: str, image_path: str = "") -> str:
        from selenium.webdriver.common.keys import Keys

        self._ensure_driver()
        composer = self._find_composer()
        if composer is None:
            self._wait_until_composer_ready()
            composer = self._find_composer()
        if composer is None:
            raise RuntimeError("Cannot find Gemini input composer.")

        img = str(image_path or "").strip()
        if img:
            abs_img = os.path.abspath(os.path.expanduser(img))
            if os.path.isfile(abs_img):
                upload_input = self._find_upload_input()
                if upload_input is None:
                    raise RuntimeError("Cannot find file upload input on Gemini web page.")
                upload_input.send_keys(abs_img)
                time.sleep(0.8)

        baseline_candidates = self._response_text_candidates()
        baseline_set = {str(item).strip() for item in baseline_candidates if str(item).strip()}

        try:
            composer.click()
        except Exception:
            pass
        try:
            composer.clear()
        except Exception:
            pass
        composer.send_keys(str(prompt or "").strip())
        composer.send_keys(Keys.ENTER)
        self._log("[GEMINI-ONLINE] Prompt submitted.")

        deadline = time.time() + float(self.timeout_sec)
        stable_rounds = 0
        last = ""
        while time.time() < deadline:
            current_candidates = self._response_text_candidates()
            fresh_candidates = [
                str(item).strip()
                for item in current_candidates
                if str(item).strip() and str(item).strip() not in baseline_set
            ]
            now = fresh_candidates[-1] if fresh_candidates else ""
            if now:
                if now == last:
                    stable_rounds += 1
                else:
                    stable_rounds = 0
                last = now
                if stable_rounds >= 2:
                    return now
            time.sleep(1.0)
        raise RuntimeError("Timed out waiting for Gemini web response.")

    def answer_probe(
        self,
        *,
        image_path: str,
        question: str,
        regions: List[Dict[str, object]],
        response_format: Dict[str, object] | None = None,
        schema: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        fmt = dict(response_format or {}) if isinstance(response_format, dict) else {}
        fmt_type = str(fmt.get("type", "") or "").strip().lower()
        prompt_context = dict(fmt.get("_prompt_context") or fmt.get("prompt_context") or {})

        if fmt_type == "selection":
            options = [str(x).strip() for x in list(fmt.get("options") or []) if str(x).strip()]
            default_sel = str(fmt.get("default_selection", "uncertain") or "uncertain").strip()
            prompt = APIVisionVerifier._build_selection_prompt(
                question=str(question or "").strip(),
                regions=regions,
                options=options,
                default_selection=default_sel,
                prompt_context=prompt_context,
            )
            target_schema = dict(schema or {})
            if not target_schema:
                target_schema = {
                    "type": "object",
                    "properties": {
                        "selection": {"type": "string", "enum": options + ["uncertain"]},
                        "reason": {"type": "string"},
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["selection", "reason", "score"],
                    "additionalProperties": False,
                }
            last_text = ""
            for _attempt in range(3):
                text = self._submit_prompt(prompt=prompt, image_path=image_path)
                parsed = APIVisionVerifier._extract_json_block(text)
                schema_errors = APIVisionVerifier._validate_json_schema(parsed, target_schema) if isinstance(parsed, dict) else ["invalid_json"]
                if isinstance(parsed, dict) and not schema_errors:
                    return {
                        "selection": str(parsed.get("selection", "uncertain") or "uncertain").strip() or "uncertain",
                        "score": max(0.0, min(1.0, APIVisionVerifier._safe_float(parsed.get("score"), 0.0))),
                        "reason": str(parsed.get("reason", "") or "").strip(),
                        "raw_text": text,
                        "request_prompt": prompt,
                        "request_schema": target_schema,
                        "raw_response": {"provider": "gemini_online"},
                        "schema_valid": True,
                        "is_truncated": False,
                        "is_valid": True,
                    }
                selected = APIVisionVerifier._infer_selection_from_text(text, options)
                if (not selected) and isinstance(parsed, dict):
                    selected = APIVisionVerifier._infer_selection_from_text(
                        json.dumps(parsed, ensure_ascii=True),
                        options,
                    )
                if selected:
                    fallback_score = APIVisionVerifier._extract_score_from_text(text, default=0.5)
                    if isinstance(parsed, dict):
                        fallback_score = APIVisionVerifier._safe_float(parsed.get("score"), fallback_score)
                    fallback_reason = APIVisionVerifier._extract_reason_from_text(
                        str((parsed or {}).get("reason", "") or text) if isinstance(parsed, dict) else text,
                        default="fallback_parse",
                    )
                    return {
                        "selection": selected,
                        "score": max(0.0, min(1.0, float(fallback_score))),
                        "reason": str(fallback_reason or "fallback_parse"),
                        "raw_text": text,
                        "request_prompt": prompt,
                        "request_schema": target_schema,
                        "raw_response": {"provider": "gemini_online"},
                        "schema_valid": False,
                        "is_truncated": False,
                        "is_valid": True,
                    }
                last_text = str(text or "")
            return {
                "selection": "uncertain",
                "score": None,
                "reason": "invalid_response",
                "raw_text": last_text,
                "request_prompt": prompt,
                "request_schema": target_schema,
                "raw_response": {"provider": "gemini_online"},
                "schema_valid": False,
                "is_truncated": False,
                "is_valid": False,
            }

        prompt = APIVisionVerifier._build_binary_prompt(
            question=str(question or "").strip(),
            regions=regions,
            prompt_context=prompt_context,
        )
        target_schema = dict(schema or {})
        if not target_schema:
            target_schema = {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "enum": ["yes", "no", "uncertain"]},
                    "reason": {"type": "string"},
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["answer", "reason", "score"],
                "additionalProperties": False,
            }
        last_text = ""
        for _attempt in range(3):
            text = self._submit_prompt(prompt=prompt, image_path=image_path)
            parsed = APIVisionVerifier._extract_json_block(text)
            schema_errors = APIVisionVerifier._validate_json_schema(parsed, target_schema) if isinstance(parsed, dict) else ["invalid_json"]
            if isinstance(parsed, dict) and not schema_errors:
                answer = APIVisionVerifier._normalize_binary_answer(parsed.get("answer")) or "uncertain"
                return {
                    "answer": answer,
                    "score": max(0.0, min(1.0, APIVisionVerifier._safe_float(parsed.get("score"), 0.0))),
                    "reason": str(parsed.get("reason", "") or "").strip(),
                    "raw_text": text,
                    "request_prompt": prompt,
                    "request_schema": target_schema,
                    "raw_response": {"provider": "gemini_online"},
                    "schema_valid": True,
                    "is_truncated": False,
                    "is_valid": True,
                }
            if str(text or "").strip():
                fallback_answer = APIVisionVerifier._normalize_binary_answer(
                    ((parsed or {}).get("answer") if isinstance(parsed, dict) else "")
                ) or APIVisionVerifier._infer_binary_answer_from_text(text)
                fallback_score = APIVisionVerifier._extract_score_from_text(text, default=0.5)
                if isinstance(parsed, dict):
                    fallback_score = APIVisionVerifier._safe_float(parsed.get("score"), fallback_score)
                fallback_reason = APIVisionVerifier._extract_reason_from_text(
                    str((parsed or {}).get("reason", "") or text) if isinstance(parsed, dict) else text,
                    default="fallback_parse",
                )
                return {
                    "answer": fallback_answer or "uncertain",
                    "score": max(0.0, min(1.0, float(fallback_score))),
                    "reason": str(fallback_reason or "fallback_parse"),
                    "raw_text": text,
                    "request_prompt": prompt,
                    "request_schema": target_schema,
                    "raw_response": {"provider": "gemini_online"},
                    "schema_valid": False,
                    "is_truncated": False,
                    "is_valid": True,
                }
            last_text = str(text or "")
        return {
            "answer": "uncertain",
            "score": None,
            "reason": "invalid_response",
            "raw_text": last_text,
            "request_prompt": prompt,
            "request_schema": target_schema,
            "raw_response": {"provider": "gemini_online"},
            "schema_valid": False,
            "is_truncated": False,
            "is_valid": False,
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
        strict_prompt, expects_json = APIVisionVerifier._build_caption_prompt(
            prompt=str(prompt or "").strip(),
            schema=schema,
            regions=regions,
        )
        target_schema = dict(schema or {})
        last_text = ""
        for _attempt in range(3):
            text = self._submit_prompt(prompt=strict_prompt, image_path=image_path)
            if expects_json:
                parsed = APIVisionVerifier._extract_json_block(text)
                schema_errors = APIVisionVerifier._validate_json_schema(parsed, target_schema) if target_schema and isinstance(parsed, dict) else []
                if isinstance(parsed, dict) and not schema_errors:
                    parsed.setdefault("raw_text", text)
                    parsed.setdefault("request_prompt", strict_prompt)
                    parsed.setdefault("request_schema", target_schema)
                    parsed.setdefault("raw_response", {"provider": "gemini_online"})
                    parsed["schema_valid"] = bool(target_schema)
                    parsed["is_valid"] = True
                    if "caption" in parsed and "caption_text" not in parsed:
                        parsed["caption_text"] = str(parsed.get("caption", "") or "").strip()
                    return parsed
            elif str(text or "").strip():
                caption_text = str(text or "").strip()
                return {
                    "caption_text": caption_text,
                    "caption": caption_text,
                    "raw_text": caption_text,
                    "request_prompt": strict_prompt,
                    "raw_response": {"provider": "gemini_online"},
                    "schema_valid": True,
                    "is_valid": True,
                }
            last_text = str(text or "")
        if expects_json and str(last_text or "").strip():
            parsed = APIVisionVerifier._extract_json_block(last_text)
            if isinstance(parsed, dict):
                schema_ok = (
                    len(APIVisionVerifier._validate_json_schema(parsed, target_schema)) == 0
                    if target_schema
                    else False
                )
                parsed.setdefault("raw_text", last_text)
                parsed.setdefault("request_prompt", strict_prompt)
                parsed.setdefault("request_schema", target_schema)
                parsed.setdefault("raw_response", {"provider": "gemini_online"})
                parsed["schema_valid"] = bool(schema_ok)
                parsed["is_valid"] = bool(schema_ok or not target_schema)
                if "caption" in parsed and "caption_text" not in parsed:
                    parsed["caption_text"] = str(parsed.get("caption", "") or "").strip()
                return parsed
        return {
            "caption_text": str(last_text or "").strip(),
            "caption": str(last_text or "").strip(),
            "raw_text": last_text,
            "request_prompt": strict_prompt,
            "request_schema": target_schema,
            "raw_response": {"provider": "gemini_online"},
            "schema_valid": False,
            "is_valid": False,
        }

    def close(self) -> None:
        driver = self._driver
        self._driver = None
        if driver is None:
            return
        try:
            driver.quit()
        except Exception:
            pass
