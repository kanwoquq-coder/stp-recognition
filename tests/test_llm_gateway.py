from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from stp_similarity import (  # noqa: E402
    LLMVisionUnavailableError,
    OpenAICompatibleChatGateway,
    parse_llm_json,
)


def _response(content: str = '{"status":"ok"}'):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _HTTPError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class LLMGatewayTests(unittest.TestCase):
    def _gateway(self, outcomes, **kwargs):
        gateway = OpenAICompatibleChatGateway(
            api_key="test-key",
            base_url="http://127.0.0.1:1/v1",
            max_retries=0,
            **kwargs,
        )
        fake = _FakeCompletions(outcomes)
        gateway.client = SimpleNamespace(
            chat=SimpleNamespace(completions=fake)
        )
        return gateway, fake

    def test_parse_qwen_json_variants(self):
        self.assertEqual(parse_llm_json('{"status":"ok"}')["status"], "ok")
        self.assertEqual(
            parse_llm_json('思考内容\n```json\n{"status":"ok"}\n```')["status"],
            "ok",
        )

    def test_response_format_auto_fallback_and_thinking_off(self):
        gateway, fake = self._gateway(
            [RuntimeError("response_format is not supported"), _response()],
            response_format_mode="auto",
            thinking_mode="off",
        )
        response = gateway.create_json_completion(
            model="Qwen3.5-27B",
            messages=[{"role": "user", "content": "返回JSON"}],
        )
        self.assertEqual(response.choices[0].message.content, '{"status":"ok"}')
        self.assertIn("response_format", fake.calls[0])
        self.assertNotIn("response_format", fake.calls[1])
        self.assertFalse(
            fake.calls[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )

    def test_visual_502_opens_fuse_after_text_probe_succeeds(self):
        gateway, fake = self._gateway(
            [_HTTPError(502, "Bad Gateway"), _response("OK")],
            response_format_mode="none",
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "分析图片"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            ],
        }]
        with self.assertRaises(LLMVisionUnavailableError):
            gateway.create_json_completion("Qwen3.5-27B", messages)
        self.assertFalse(gateway.vision_available)
        self.assertEqual(len(fake.calls), 2)


if __name__ == "__main__":
    unittest.main()
