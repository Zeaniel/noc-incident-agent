import io
import json
import urllib.error

import pytest

from noc_agent import llm as llm_mod
from noc_agent.llm import LLMError, OpenAICompatibleLLM, extract_json, strip_thinking


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    'Here you go:\n```json\n{"a": 1}\n```',
    '<think>{"a": 2} is wrong</think>{"a": 1}',
    'prefix {"a": 1} suffix {"b": 2}',
])
def test_extract_json(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_handles_braces_in_strings():
    assert extract_json('{"s": "a } b {"}') == {"s": "a } b {"}


def test_extract_json_errors():
    with pytest.raises(ValueError):
        extract_json("no json here")
    with pytest.raises(ValueError):
        extract_json('{"a": 1')


def test_strip_thinking_with_open_tag_in_template():
    assert strip_thinking("reasoning...</think>\nanswer") == "answer"


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_client_request_and_response(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data)
        sent["auth"] = req.get_header("Authorization")
        return _Resp(json.dumps({"choices": [{"message": {
            "content": "<think>hmm</think>ok",
            "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "{}"}}]}}]}).encode())

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleLLM("http://x/v1/", "m", api_key="k",
                                 extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    out = client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert sent["url"] == "http://x/v1/chat/completions"
    assert sent["auth"] == "Bearer k"
    assert sent["body"]["tools"] and sent["body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert out["content"] == "ok" and out["tool_calls"][0]["id"] == "1"


def test_client_does_not_retry_client_errors(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, io.BytesIO(b"model not found"))

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(LLMError, match="404"):
        OpenAICompatibleLLM("http://x/v1", "m").chat([])
    assert len(calls) == 1
