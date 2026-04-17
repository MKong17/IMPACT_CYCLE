import base64
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from core.impact_sg.mllm_adapters.api_verifier import APIVisionVerifier
from core.impact_sg.mllm_adapters.factory import build_vision_verifier
from core.impact_sg.mllm_adapters.gemini_online_verifier import GeminiOnlineVisionVerifier
from core.impact_sg.mllm_adapters.service import ThreadSafeVisionVerifier
from core.impact_sg.visual_verifier.schemas import probe_response_schema


class _RequestRecorderHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body or "{}")
        self.__class__.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "payload": payload,
            }
        )
        if self.path.endswith("/caption"):
            response = {
                "caption": "Visible entities include person.",
                "supported_entities": ["track_1"],
                "supported_relations": [],
            }
        else:
            response = {"answer": "yes", "score": 0.91, "reason": "supported"}
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):  # noqa: A003
        return


class MllmAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _RequestRecorderHandler.requests = []
        cls._server = HTTPServer(("127.0.0.1", 0), _RequestRecorderHandler)
        cls._thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._thread.start()
        cls._base_url = f"http://127.0.0.1:{cls._server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.shutdown()
        cls._server.server_close()
        cls._thread.join(timeout=5)

    def setUp(self) -> None:
        _RequestRecorderHandler.requests = []

    def _tmp_image(self) -> str:
        handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        handle.write(b"fake-image-bytes")
        handle.flush()
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_threadsafe_verifier_forwards_response_format(self) -> None:
        class DummyVerifier:
            def answer_probe(self, *, image_path: str, question: str, regions, response_format=None):
                return {"question": question, "response_format": response_format, "regions": list(regions or [])}

            def generate_caption(self, *, image_path: str, prompt: str, regions):
                return {"caption": prompt, "regions": list(regions or [])}

        verifier = ThreadSafeVisionVerifier(DummyVerifier())
        resp = verifier.answer_probe(
            image_path="frame.jpg",
            question="Choose one option",
            regions=[{"entity_id": "track_1"}],
            response_format={"type": "selection", "options": ["cup", "bottle"]},
        )
        self.assertEqual("selection", str((resp.get("response_format") or {}).get("type", "")))
        self.assertEqual(["cup", "bottle"], list((resp.get("response_format") or {}).get("options") or []))

    def test_api_verifier_posts_answer_and_caption_requests(self) -> None:
        image_path = self._tmp_image()
        verifier = APIVisionVerifier(
            provider="generic_api",
            model="demo-model",
            base_url=self._base_url,
            answer_url=self._base_url + "/answer",
            caption_url=self._base_url + "/caption",
            api_key="secret",
        )
        answer = verifier.answer_probe(
            image_path=image_path,
            question="Is there a person?",
            regions=[{"entity_id": "track_1", "bbox": [1, 2, 3, 4]}],
            response_format={"type": "selection", "options": ["yes", "no"]},
        )
        caption = verifier.generate_caption(
            image_path=image_path,
            prompt="Return structured caption JSON.",
            regions=[{"entity_id": "track_1", "bbox": [1, 2, 3, 4]}],
        )
        self.assertEqual("yes", str(answer.get("answer", "")))
        self.assertIn("caption", caption)
        self.assertEqual(2, len(_RequestRecorderHandler.requests))

        answer_req = _RequestRecorderHandler.requests[0]
        self.assertEqual("/answer", answer_req["path"])
        self.assertEqual("Bearer secret", answer_req["headers"].get("Authorization"))
        self.assertEqual("answer_probe", answer_req["payload"].get("task"))
        self.assertEqual("selection", str((answer_req["payload"].get("response_format") or {}).get("type", "")))
        self.assertTrue(str(answer_req["payload"].get("image_base64", "")))
        decoded = base64.b64decode(str(answer_req["payload"].get("image_base64", "")))
        self.assertEqual(b"fake-image-bytes", decoded)

        caption_req = _RequestRecorderHandler.requests[1]
        self.assertEqual("/caption", caption_req["path"])
        self.assertEqual("generate_caption", caption_req["payload"].get("task"))
        self.assertEqual("Return structured caption JSON.", caption_req["payload"].get("prompt"))

    def test_generic_api_strips_internal_prompt_context_from_transport(self) -> None:
        image_path = self._tmp_image()
        verifier = APIVisionVerifier(
            provider="generic_api",
            model="demo-model",
            base_url=self._base_url,
            answer_url=self._base_url + "/answer",
            caption_url=self._base_url + "/caption",
            api_key="secret",
        )
        verifier.answer_probe(
            image_path=image_path,
            question="Is there a person?",
            regions=[{"entity_id": "track_1", "bbox": [1, 2, 3, 4]}],
            response_format={
                "type": "selection",
                "options": ["yes", "no"],
                "_prompt_context": {"claim_type": "relation", "relation": "left_of"},
            },
        )
        answer_req = _RequestRecorderHandler.requests[0]
        self.assertEqual("selection", str((answer_req["payload"].get("response_format") or {}).get("type", "")))
        self.assertNotIn("_prompt_context", dict(answer_req["payload"].get("response_format") or {}))

    def test_factory_uses_api_verifier_when_enabled(self) -> None:
        cycle_cfg = {
            "local_verifier": {
                "provider": "qwen25_vl",
                "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
            },
            "api_verifier": {
                "enabled": True,
                "provider": "generic_api",
                "base_url": self._base_url,
                "answer_url": self._base_url + "/answer",
                "caption_url": self._base_url + "/caption",
                "model": "api-demo",
                "api_key_prefix": "Bearer ",
            },
        }
        verifier, meta = build_vision_verifier(
            cycle_cfg,
            preferred_provider="auto",
            api_key="factory-secret",
            allow_mock_fallback=False,
        )
        self.assertIsInstance(verifier, ThreadSafeVisionVerifier)
        self.assertIsInstance(verifier._verifier, APIVisionVerifier)
        self.assertEqual("generic_api", str(meta.get("verifier_provider", "")))
        self.assertEqual("api-demo", str(meta.get("verifier_model_id", "")))

    def test_factory_reports_gemini_api_runtime_meta(self) -> None:
        cycle_cfg = {
            "api_verifier": {
                "enabled": True,
                "provider": "gemini",
                "model": "gemini-2.5-pro",
            },
        }
        verifier, meta = build_vision_verifier(
            cycle_cfg,
            preferred_provider="gemini_api",
            api_key="factory-secret",
            allow_mock_fallback=False,
        )
        self.assertIsInstance(verifier, ThreadSafeVisionVerifier)
        self.assertIsInstance(verifier._verifier, APIVisionVerifier)
        self.assertEqual("gemini_api", str(meta.get("verifier_provider", "")))
        self.assertEqual("gemini-2.5-pro", str(meta.get("verifier_model_id", "")))

    def test_gemini_api_selection_fallback_accepts_parseable_text(self) -> None:
        image_path = self._tmp_image()
        verifier = APIVisionVerifier(
            provider="gemini",
            model="gemini-2.5-pro",
            api_key="secret",
        )

        def _fake_gemini_post(*args, **kwargs):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "selection: person\nscore: 0.91\nreason: visible human in the region"
                                }
                            ]
                        }
                    }
                ]
            }

        verifier._gemini_post = _fake_gemini_post  # type: ignore[method-assign]
        resp = verifier.answer_probe(
            image_path=image_path,
            question="Which canonical label best fits this object?",
            regions=[{"entity_id": "track_1", "bbox": [1, 2, 3, 4]}],
            response_format={"type": "selection", "options": ["person", "cup"], "default_selection": "cup"},
        )
        self.assertEqual("person", str(resp.get("selection", "")))
        self.assertTrue(bool(resp.get("is_valid", False)))
        self.assertFalse(bool(resp.get("schema_valid", True)))

    def test_gemini_api_caption_uses_caller_json_prompt(self) -> None:
        image_path = self._tmp_image()
        verifier = APIVisionVerifier(
            provider="gemini",
            model="gemini-2.5-pro",
            api_key="secret",
        )
        captured = {}

        def _fake_gemini_post(text_prompt, image_path, **kwargs):
            captured["prompt"] = text_prompt
            captured["mime"] = kwargs.get("response_mime_type")
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"objects":[{"label":"person","bbox_xywh":[1,2,3,4]}]}'
                                }
                            ]
                        }
                    }
                ]
            }

        verifier._gemini_post = _fake_gemini_post  # type: ignore[method-assign]
        resp = verifier.generate_caption(
            image_path=image_path,
            prompt='Return JSON ONLY in this exact format: {"objects":[{"label":"person","bbox_xywh":[x,y,width,height]}]}',
            regions=[],
        )
        self.assertIn("objects", str(captured.get("prompt", "")))
        self.assertEqual("application/json", str(captured.get("mime", "")))
        self.assertEqual("person", str(((resp.get("objects") or [])[0] or {}).get("label", "")))
        self.assertTrue(bool(resp.get("is_valid", False)))

    def test_binary_prompt_includes_spatial_context_guidance(self) -> None:
        prompt = APIVisionVerifier._build_binary_prompt(
            question="Does the relation hold?",
            regions=[{"entity_id": "track_1", "bbox": [1, 2, 3, 4]}],
            prompt_context={
                "claim_type": "relation",
                "probe_family": "binary_verification",
                "subject_id": "track_1",
                "subject_label": "person",
                "object_id": "track_2",
                "object_label": "cup",
                "relation": "left_of",
                "is_spatial": True,
            },
        )
        self.assertIn("Claim context:", prompt)
        self.assertIn("subject: person (track_1)", prompt)
        self.assertIn("spatial relation task", prompt.lower())
        self.assertIn("geometry", prompt.lower())

    def test_binary_prompt_respects_strict_contract(self) -> None:
        prompt = APIVisionVerifier._build_binary_prompt(
            question="Is there a person in the frame?",
            regions=[{"entity_id": "track_1", "bbox": [1, 2, 3, 4]}],
            prompt_context={"claim_type": "existence"},
            allow_uncertain=False,
        )
        self.assertIn("yes or no", prompt.lower())
        self.assertNotIn('use "uncertain"', prompt.lower())

    def test_probe_response_schema_uses_strict_binary_when_uncertain_is_disabled(self) -> None:
        schema = probe_response_schema(
            {
                "question": "Does this claim match the image? Answer yes or no.",
                "response_format": {"type": "binary", "allow_uncertain": False},
            }
        )
        self.assertEqual(["yes", "no"], list((((schema.get("properties") or {}).get("answer") or {}).get("enum") or [])))

    def test_gemini_api_strict_binary_rejects_uncertain_fallback(self) -> None:
        image_path = self._tmp_image()
        verifier = APIVisionVerifier(
            provider="gemini",
            model="gemini-2.5-pro",
            api_key="secret",
        )

        def _fake_gemini_post(*args, **kwargs):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "answer: uncertain\nscore: 0.42\nreason: evidence is ambiguous"
                                }
                            ]
                        }
                    }
                ]
            }

        verifier._gemini_post = _fake_gemini_post  # type: ignore[method-assign]
        resp = verifier.answer_probe(
            image_path=image_path,
            question="Does this claim match the image? Answer yes or no.",
            regions=[{"entity_id": "track_1", "bbox": [1, 2, 3, 4]}],
            response_format={"type": "binary", "allow_uncertain": False},
        )
        self.assertFalse(bool(resp.get("is_valid", True)))
        self.assertFalse(bool(resp.get("schema_valid", True)))

    def test_gemini_online_selection_fallback_accepts_parseable_text(self) -> None:
        class _FakeOnlineVerifier(GeminiOnlineVisionVerifier):
            def _submit_prompt(self, *, prompt: str, image_path: str = "") -> str:
                _ = prompt
                _ = image_path
                return "answer: person\nscore: 0.88\nreason: visible person in the marked area"

        verifier = _FakeOnlineVerifier()
        resp = verifier.answer_probe(
            image_path="frame.jpg",
            question="Which canonical label best fits this object?",
            regions=[{"entity_id": "track_1", "bbox": [1, 2, 3, 4]}],
            response_format={"type": "selection", "options": ["person", "cup"], "default_selection": "cup"},
        )
        self.assertEqual("person", str(resp.get("selection", "")))
        self.assertTrue(bool(resp.get("is_valid", False)))
        self.assertFalse(bool(resp.get("schema_valid", True)))


if __name__ == "__main__":
    unittest.main()
