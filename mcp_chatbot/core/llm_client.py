from __future__ import annotations

import json
import logging
from typing import Any

import httpx


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str, model: str, temp: float, frontend: str) -> None:
        self.api_key: str = api_key
        self.model: str = model
        self.temp: float = temp
        self.max_tokens = 20480
        self.frontend: str = frontend
        self.timeout = httpx.Timeout(
            timeout=300.0,
            read=60.0,
            write=30.0,
            connect=10.0,  
        )

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Return (url, headers, payload) for an LLM request."""
        url = (
            "https://api.groq.com/openai/v1/chat/completions"
            if self.api_key != "local"
            else "http://localhost:1234/v1/chat/completions"
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: dict[str, Any] = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temp,
            "max_tokens": self.max_tokens,
            "stream": stream,
            "stream_options": {"include_usage": True},
            "stop": None,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return url, headers, payload

    # Used for context compression
    def get_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Non-streaming call. Returns the full assistant message dict.
        """
        url, headers, payload = self._build_payload(messages, tools, stream=False)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]
        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)
            if isinstance(e, httpx.HTTPStatusError):
                logging.error("Status code: %s", e.response.status_code)
                logging.error("Response details: %s", e.response.text)
            return {"role": "assistant", "content": f"I encountered an error: {error_message}."}

    def stream_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ):
        url, headers, payload = self._build_payload(messages, tools, stream=True)

        tool_calls = {}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if "usage" in data:
                            yield {"type": "usage", "data": data["usage"]}

                        if "choices" not in data or not data["choices"]:
                            continue

                        choice = data["choices"][0]
                        delta = choice.get("delta", {})

                        content = delta.get("content")
                        if content:
                            yield {"type": "content", "data": content}

                        for tc in delta.get("tool_calls", []) or []:
                            idx = tc["index"]
                            if idx not in tool_calls:
                                tool_calls[idx] = {
                                    "id": tc.get("id", ""),
                                    "type": tc.get("type", "function"),
                                    "function": {"name": "", "arguments": ""},
                                }

                            current = tool_calls[idx]

                            if tc.get("id"):
                                current["id"] = tc["id"]

                            fn = tc.get("function", {})
                            if fn.get("name"):
                                current["function"]["name"] += fn["name"]
                                yield {"type": "tool_name", "index": idx, "data": fn["name"]}

                            if fn.get("arguments"):
                                current["function"]["arguments"] += fn["arguments"]
                                yield {"type": "tool_arguments", "index": idx, "data": fn["arguments"]}

                    if tool_calls:
                        yield {
                            "type": "tool_calls_final",
                            "data": [tool_calls[i] for i in sorted(tool_calls)],
                        }

        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)
            if isinstance(e, httpx.HTTPStatusError):
                logging.error("Status code: %s", e.response.status_code)
                logging.error("Response details: %s", e.response.text)
            yield {"type": "error", "data": f"I encountered an error: {error_message}. Please try again or rephrase your request."}
