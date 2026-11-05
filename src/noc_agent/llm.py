"""Chat model clients.

`OpenAICompatibleLLM` talks to any server that implements the OpenAI chat
completions API: vLLM, llama.cpp, Ollama, LM Studio or OpenAI itself. It uses
only the standard library, so the package has no SDK dependency.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

Message = dict[str, Any]

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ChatModel(Protocol):
    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        """Return the assistant message (with `content` and optional `tool_calls`)."""


class LLMError(RuntimeError):
    pass


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: float = 120,
        extra_body: dict | None = None,
        retries: int = 2,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self.retries = retries

    @classmethod
    def from_env(cls) -> OpenAICompatibleLLM:
        """NOC_LLM_BASE_URL, NOC_LLM_MODEL, NOC_LLM_API_KEY, NOC_LLM_EXTRA_BODY (JSON)."""
        extra = os.environ.get("NOC_LLM_EXTRA_BODY")
        return cls(
            base_url=os.environ.get("NOC_LLM_BASE_URL", "http://localhost:11434/v1"),
            model=os.environ.get("NOC_LLM_MODEL", "qwen3:8b"),
            api_key=os.environ.get("NOC_LLM_API_KEY"),
            extra_body=json.loads(extra) if extra else None,
        )

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode())
                break
            except urllib.error.HTTPError as e:
                # 4xx means the request itself is wrong; retrying will not help
                if e.code < 500:
                    raise LLMError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from e
                last_err = e
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
            time.sleep(1.5 * (attempt + 1))
        else:
            raise LLMError(f"LLM request failed after {self.retries + 1} attempts: {last_err}")

        msg = data["choices"][0]["message"]
        out: Message = {"role": "assistant", "content": strip_thinking(msg.get("content") or "")}
        if msg.get("tool_calls"):
            out["tool_calls"] = msg["tool_calls"]
        return out


class CountingLLM:
    """Wraps a model and counts calls and latency, for the evaluation table."""

    def __init__(self, inner: ChatModel):
        self.inner = inner
        self.calls = 0
        self.seconds = 0.0

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        t = time.perf_counter()
        try:
            return self.inner.chat(messages, tools)
        finally:
            self.calls += 1
            self.seconds += time.perf_counter() - t


def strip_thinking(text: str) -> str:
    """Reasoning models may prepend <think>...</think>; it is never part of the answer."""
    text = _THINK.sub("", text)
    if "</think>" in text:  # opening tag was in the prompt template
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> Any:
    """Parse the first JSON object in a model reply (fenced or bare)."""
    text = strip_thinking(text)
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unterminated JSON object in reply")
