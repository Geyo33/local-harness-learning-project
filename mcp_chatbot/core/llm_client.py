from __future__ import annotations

import json
import logging
from typing import Any

import httpx


def _read_error_body(e: httpx.HTTPStatusError) -> str:
    """Best-effort extraction of an HTTP error body. Streamed responses raise
    ResponseNotRead on `.text` until read(), so force a read first."""
    try:
        e.response.read()
    except Exception:
        pass
    try:
        return e.response.text.strip() or "(empty body)"
    except Exception:
        return "(unreadable body)"


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temp: float,
        frontend: str,
        max_output_tokens: int = 4096,
    ) -> None:
        self.api_key: str = api_key
        self.model: str = model
        self.temp: float = temp
        # Context window: total prompt+completion budget the model can hold.
        # Drives ContextManager compression threshold and the UI usage bar.
        self.max_tokens = 32000
        # Completion reservation sent as payload `max_tokens`. Distinct concept
        # from the context window: backends (LM Studio / llama.cpp) reject with
        # HTTP 400 when prompt_tokens + this > n_ctx, so it must stay well below
        # max_tokens to leave room for the prompt to grow.
        self.max_output_tokens = max_output_tokens
        self.frontend: str = frontend
        self.timeout = httpx.Timeout(
            timeout=600.0,
            read=60.0,
            write=30.0,
            connect=10.0,  
        )

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
        tool_choice: dict | str | None = None,
        response_format: dict | None = None,
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
            "max_tokens": self.max_output_tokens,
            "stream": stream,
            "stream_options": {"include_usage": True},
            "stop": None,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice if tool_choice is not None else "auto"
        if response_format is not None and self.api_key == "local":
            payload["response_format"] = response_format
        return url, headers, payload

    # Used for context compression
    def get_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict | str | None = None,
        response_format: dict | None = None,
    ) -> dict[str, Any]:
        """
        Non-streaming call. Returns the full assistant message dict.
        """
        url, headers, payload = self._build_payload(messages, tools, stream=False, tool_choice=tool_choice, response_format=response_format)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]
        except httpx.HTTPStatusError as e:
            body = _read_error_body(e)
            logging.error(
                "LLM HTTP %s error: %s", e.response.status_code, body
            )
            return {"role": "assistant", "content": f"I encountered an error: HTTP {e.response.status_code} — {body}"}
        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)
            return {"role": "assistant", "content": f"I encountered an error: {error_message}."}

    def stream_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict | str | None = None,
        response_format: dict | None = None,
    ):
        url, headers, payload = self._build_payload(messages, tools, stream=True, tool_choice=tool_choice, response_format=response_format)

        tool_calls = {}
        finish_reason = None
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
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
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

                    if finish_reason == "length":
                        yield {"type": "truncated", "data": finish_reason}

        except httpx.HTTPStatusError as e:
            body = _read_error_body(e)
            logging.error("LLM HTTP %s error: %s", e.response.status_code, body)
            yield {"type": "error", "data": f"I encountered an error: HTTP {e.response.status_code} — {body}. Please try again or rephrase your request."}
        except httpx.RequestError as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)
            yield {"type": "error", "data": f"I encountered an error: {error_message}. Please try again or rephrase your request."}
